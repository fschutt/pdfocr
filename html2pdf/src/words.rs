//! The word spans of a pdf-ocr-bench page:
//!
//! ```html
//! <span class="word" style="left: 5.96%; top: 10.60%; width: 0.24%; height: 0.71%; font-size: 34.81pt;"
//!       data-confidence="0.390" data-block="1" data-par="0" data-line="0">Rev. </span>
//! ```
//!
//! `left`/`width`/`height` are the OCR box and `top` is where the text is drawn, all in percent of
//! the page. `data-block`/`data-par`/`data-line` are the engine's own layout (ids unique within the
//! page), present only where the engine reports that level.

use std::ops::Range;

const OPEN: &str = "<span class=\"word\"";
const CLOSE: &str = "</span>";

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    /// Page fractions (0..1).
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub font_size_pt: f32,
    pub block: Option<u32>,
    pub par: Option<u32>,
    pub line: Option<u32>,
}

/// Byte ranges of one word span: its attributes and its text.
struct Span {
    attrs: Range<usize>,
    text: Range<usize>,
}

fn spans(html: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = html[pos..].find(OPEN) {
        let start = pos + i;
        // attribute values never contain '>' (numbers and the inline style)
        let Some(gt) = html[start..].find('>') else { break };
        let text_start = start + gt + 1;
        let Some(len) = html[text_start..].find(CLOSE) else { break };
        out.push(Span { attrs: start + OPEN.len()..start + gt, text: text_start..text_start + len });
        pos = text_start + len + CLOSE.len();
    }
    out
}

/// All word spans of a page, in document order.
pub fn parse_words(html: &str) -> Vec<Word> {
    spans(html)
        .into_iter()
        .filter_map(|span| {
            let mut word = Word {
                text: unescape(&html[span.text]).trim().to_string(),
                left: 0.0,
                top: 0.0,
                width: 0.0,
                height: 0.0,
                font_size_pt: 0.0,
                block: None,
                par: None,
                line: None,
            };
            for (name, value) in attributes(&html[span.attrs]) {
                match name {
                    "style" => apply_style(&mut word, value),
                    "data-block" => word.block = value.parse().ok(),
                    "data-par" => word.par = value.parse().ok(),
                    "data-line" => word.line = value.parse().ok(),
                    _ => {}
                }
            }
            (!word.text.is_empty() && word.font_size_pt > 0.0).then_some(word)
        })
        .collect()
}

/// The page with every word's text replaced by `f(text)`; markup and spacing are kept.
pub fn rewrite_texts(html: &str, mut f: impl FnMut(&str) -> String) -> String {
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    for span in spans(html) {
        let raw = unescape(&html[span.text.clone()]);
        let core = raw.trim_end();
        out.push_str(&html[pos..span.text.start]);
        out.push_str(&escape(&f(core.trim_start())));
        out.push_str(&raw[core.len()..]); // the space that separates words of a line
        pos = span.text.end;
    }
    out.push_str(&html[pos..]);
    out
}

/// The page with every word of its text blocks (`<div class="region">`, as `pdf-ocr-bench
/// reconstruct` writes them) replaced by `f(word, italic)`; markup and spacing are kept. A word
/// is italic in an italic block (`font-style: italic`) or in `<i>`, upright in `<span class="up">`.
pub fn rewrite_region_texts(html: &str, mut f: impl FnMut(&str, bool) -> String) -> String {
    const OPEN: &str = "<div class=\"region\"";
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    while let Some(i) = html[pos..].find(OPEN) {
        let start = pos + i;
        let end = html[start..].find("</div>").map_or(html.len(), |e| start + e);
        out.push_str(&html[pos..start]);
        // the text between the block's tags, word by word
        let mut rest = &html[start..end];
        let block_italic = rest[..rest.find('>').unwrap_or(rest.len())].contains("font-style: italic");
        let mut italic = block_italic;
        while let Some(gt) = rest.find('>') {
            let tag = &rest[..=gt];
            if tag.starts_with("<i>") || tag.starts_with("<i ") {
                italic = true;
            } else if tag.starts_with("<span class=\"up\"") {
                italic = false;
            } else if tag.starts_with("</i") || tag.starts_with("</span") {
                italic = block_italic;
            }
            out.push_str(tag);
            rest = &rest[gt + 1..];
            let text_end = rest.find('<').unwrap_or(rest.len());
            let text = unescape(&rest[..text_end]);
            let mut word = String::new();
            for c in text.chars().chain(std::iter::once(' ')) {
                if c.is_whitespace() {
                    if !word.is_empty() {
                        out.push_str(&escape(&f(&word, italic)));
                        word.clear();
                    }
                    out.push(c);
                } else {
                    word.push(c);
                }
            }
            out.pop(); // the space chained on
            rest = &rest[text_end..];
        }
        out.push_str(rest);
        pos = end;
    }
    out.push_str(&html[pos..]);
    out
}

/// The `lang` attribute of the page's `<html>` element.
pub fn html_lang(html: &str) -> Option<&str> {
    let tag = &html[html.find("<html")?..];
    let tag = &tag[..tag.find('>')?];
    attributes(tag).into_iter().find(|(name, _)| *name == "lang").map(|(_, value)| value)
}

fn attributes(s: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(eq) = rest.find("=\"") {
        let name = rest[..eq].trim();
        let after = &rest[eq + 2..];
        let Some(end) = after.find('"') else { break };
        out.push((name.rsplit(char::is_whitespace).next().unwrap_or(name), &after[..end]));
        rest = &after[end + 1..];
    }
    out
}

fn apply_style(word: &mut Word, style: &str) {
    for decl in style.split(';') {
        let Some((key, value)) = decl.split_once(':') else { continue };
        let value = value.trim();
        let percent = || value.strip_suffix('%').and_then(|v| v.trim().parse::<f32>().ok()).map(|v| v / 100.0);
        match key.trim() {
            "left" => word.left = percent().unwrap_or(0.0),
            "top" => word.top = percent().unwrap_or(0.0),
            "width" => word.width = percent().unwrap_or(0.0),
            "height" => word.height = percent().unwrap_or(0.0),
            "font-size" => {
                word.font_size_pt = value.strip_suffix("pt").and_then(|v| v.trim().parse().ok()).unwrap_or(0.0)
            }
            _ => {}
        }
    }
}

pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let decoded = tail.find(';').filter(|&end| end <= 10).and_then(|end| {
            let entity = &tail[1..end];
            let ch = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ => entity
                    .strip_prefix("#x")
                    .or_else(|| entity.strip_prefix("#X"))
                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                    .or_else(|| entity.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                    .and_then(char::from_u32),
            };
            ch.map(|c| (c, end))
        });
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &tail[end + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&#34;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<html lang="en"><body><div class="page">
  <span class="word" style="left: 5.9635%; top: 10.6017%; width: 0.2434%; height: 0.7101%; font-size: 34.81pt;" data-confidence="0.390" data-block="1" data-par="0" data-line="0">ſhould </span>
  <span class="word" style="left: 12.0%; top: 10.6%; width: 3.0%; height: 0.7%; font-size: 33.73pt;" data-confidence="0.940" data-line="0">Moſes&#39;s</span>
</div></body></html>"#;

    #[test]
    fn parses_geometry_text_and_layout_ids() {
        let words = parse_words(PAGE);
        assert_eq!(words.len(), 2);
        let w = &words[0];
        assert_eq!(w.text, "ſhould");
        assert!((w.left - 0.059635).abs() < 1e-6 && (w.top - 0.106017).abs() < 1e-6);
        assert!((w.font_size_pt - 34.81).abs() < 1e-4);
        assert_eq!((w.block, w.par, w.line), (Some(1), Some(0), Some(0)));
        assert_eq!(words[1].text, "Moſes's");
        assert_eq!((words[1].block, words[1].line), (None, Some(0)));
        assert_eq!(html_lang(PAGE), Some("en"));
    }

    #[test]
    fn rewrite_keeps_markup_and_trailing_space() {
        let out = rewrite_texts(PAGE, |t| t.replace('ſ', "s"));
        assert!(out.contains(">should </span>"));
        assert!(out.contains(">Moses&#39;s</span>"));
        assert_eq!(out.len(), PAGE.len() - 2); // 'ſ' is 2 bytes, 's' 1
    }

    #[test]
    fn knows_which_words_are_italic() {
        let page = r#"<div class="region" style="left: 1%;"><p>a <i>b c</i> d</p></div><div class="region" style="font-style: italic;"><p>e <span class="up">f</span> g</p></div>"#;
        let mut seen = Vec::new();
        rewrite_region_texts(page, |w, italic| {
            seen.push(format!("{w}{}", if italic { "/i" } else { "" }));
            w.to_string()
        });
        assert_eq!(seen, ["a", "b/i", "c/i", "d", "e/i", "f", "g/i"]);
    }

    #[test]
    fn rewrites_the_words_of_text_blocks() {
        let page = r#"<div class="page"><div class="region" style="left: 1%;"><p style="text-indent: 9pt;">to addrefs  Thefe &amp; ſo</p></div><img class="pic" src="a.png"></div>"#;
        let out = rewrite_region_texts(page, |w, _| w.replace('f', "s").replace('ſ', "s"));
        assert_eq!(
            out,
            r#"<div class="page"><div class="region" style="left: 1%;"><p style="text-indent: 9pt;">to address  These &amp; so</p></div><img class="pic" src="a.png"></div>"#
        );
    }

    #[test]
    fn unescapes_named_and_numeric_entities() {
        assert_eq!(unescape("a&amp;b &#34;c&#x27; &lt;&gt; & x"), "a&b \"c' <> & x");
        assert_eq!(escape("a&b<\"'"), "a&amp;b&lt;&#34;&#39;");
    }
}
