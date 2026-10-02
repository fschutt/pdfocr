//! `--layout flow`: rebuild text blocks, lines and paragraphs from the positioned words, and set
//! each block as flowing paragraphs in a box where the block was on the page.
//!
//! 1. Regions: a recursive XY-cut on the word boxes. A page is cut where a horizontal band of
//!    white crosses it (a blank line's height or more), else where a vertical band of white runs
//!    through it (a column gutter, the margin beside marginal notes), until no cut is left.
//!    Bands come out top to bottom and columns left to right, which is the reading order.
//! 2. Lines: the engine's own lines (`data-line`) where every word has one, else words whose
//!    centers are within half a font size of each other.
//! 3. Paragraphs: a new one where the engine's paragraph (`data-par`) changes, where a line is
//!    indented and the one before is not, after a vertical gap, or after a short line ending a
//!    sentence (only without `data-par`).
//! 4. A word split by a line-end hyphen is joined, the `--long-s` fix applied, and every region
//!    becomes an absolutely positioned box of `<p>`s at the region's place, its font size and line
//!    height taken from the words.
//!
//! Each region is its own box rather than one `column-count: 2` container: the columns then
//! break exactly where the scan's do, and marginal notes, headers and footnotes stay in place.

use std::fmt::Write as _;

use crate::text::{join_hyphenated, Dictionary, Fixer};
use crate::words::{escape, Word};

/// `top` of a word span is the top of the text drawn at `font-size` with `line-height: 1`; the
/// middle of Helvetica's cap height sits this many em below it (see pdf-ocr-bench's layout.py).
const TEXT_CENTER_EM: f32 = 0.770 - 0.718 / 2.0;
/// A horizontal cut needs a white band at least this high (in median font sizes); the white
/// between lines of a column is ~0.3 em, between a running head or footnotes and the text ~0.8.
const MIN_ROW_GAP_EM: f32 = 0.6;
/// A vertical cut needs a white band through every line at least this wide. The more lines the
/// band crosses, the less likely it is word spacing that happens to line up, so a column of 10+
/// lines is cut at a hairline gutter (a marginal note set close to the text), a short block only
/// at a wide one.
const MIN_GUTTER_EM: [(f32, f32); 3] = [(10.0, 0.12), (3.0, 0.3), (0.0, 1.0)];

#[derive(Clone, Copy, Debug)]
struct Rect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

/// One word in page points.
#[derive(Clone, Debug)]
struct Item {
    text: String,
    rect: Rect,
    center: f32,
    size: f32,
    block: Option<u32>,
    par: Option<u32>,
    line: Option<u32>,
}

struct Line {
    words: Vec<usize>,
    x0: f32,
    x1: f32,
    center: f32,
    size: f32,
    par: Option<u32>,
}

struct Paragraph {
    text: String,
    indent: f32,
}

pub struct FlowStats {
    pub regions: usize,
    pub paragraphs: usize,
    pub hyphens_joined: usize,
}

/// The page as flowing text in positioned boxes, ready for printpdf.
pub fn page_html(
    words: &[Word],
    width_pt: f32,
    height_pt: f32,
    lang: &str,
    fixer: &mut Fixer,
    dict: Option<&Dictionary>,
    scale: f32,
) -> (String, FlowStats) {
    let items: Vec<Item> = words
        .iter()
        .map(|w| {
            let size = w.font_size_pt;
            let center = w.top * height_pt + TEXT_CENTER_EM * size;
            Item {
                text: w.text.clone(),
                rect: Rect {
                    x0: w.left * width_pt,
                    x1: (w.left + w.width) * width_pt,
                    y0: center - 0.45 * size,
                    y1: center + 0.45 * size,
                },
                center,
                size,
                block: w.block,
                par: w.par,
                line: w.line,
            }
        })
        .collect();

    let mut regions = Vec::new();
    xy_cut((0..items.len()).collect(), &items, &mut regions);
    let regions: Vec<Vec<usize>> = regions.into_iter().flat_map(|r| split_blocks(r, &items)).collect();

    let mut stats = FlowStats { regions: 0, paragraphs: 0, hyphens_joined: 0 };
    let mut body = String::new();
    for region in &regions {
        let lines = region_lines(region, &items);
        if lines.is_empty() {
            continue;
        }
        let paragraphs = paragraphs(&lines, &items, fixer, dict, &mut stats.hyphens_joined);
        stats.regions += 1;
        stats.paragraphs += paragraphs.len();
        write_region(&mut body, &lines, &paragraphs, width_pt, height_pt, scale);
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="generator" content="html2pdf --layout flow">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  .page {{ position: relative; width: {width_pt:.2}pt; height: {height_pt:.2}pt; overflow: hidden; }}
  .region {{ position: absolute; color: #000; font-family: Helvetica, Arial, sans-serif; overflow: hidden; }}
  .region p {{ margin: 0; }}
</style>
</head>
<body>
<div class="page">
{body}</div>
</body>
</html>
"#,
        lang = escape(lang)
    );
    (html, stats)
}

fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// White bands along one axis: merged [start, end) intervals and the gaps between them.
fn gaps(mut intervals: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let mut end = f32::NEG_INFINITY;
    for (start, stop) in intervals {
        if end.is_finite() && start > end {
            out.push((end, start));
        }
        end = end.max(stop);
    }
    out
}

fn xy_cut(ids: Vec<usize>, items: &[Item], out: &mut Vec<Vec<usize>>) {
    if ids.len() < 2 {
        if !ids.is_empty() {
            out.push(ids);
        }
        return;
    }
    // a speckle read as a 200pt "word" must not hide every gap
    let em = median(ids.iter().map(|&i| items[i].size).collect());
    let clamp = |r: Rect, c: f32| {
        let half = ((r.y1 - r.y0) / 2.0).min(em);
        (c - half, c + half)
    };

    let rows: Vec<(f32, f32)> = gaps(ids.iter().map(|&i| clamp(items[i].rect, items[i].center)).collect())
        .into_iter()
        .filter(|(a, b)| b - a >= MIN_ROW_GAP_EM * em)
        .collect();
    if !rows.is_empty() {
        let cuts: Vec<f32> = rows.iter().map(|(a, b)| (a + b) / 2.0).collect();
        return split(ids, &cuts, |i| items[i].center, items, out);
    }

    let (top, bottom) = ids.iter().fold((f32::MAX, f32::MIN), |(t, b), &i| (t.min(items[i].rect.y0), b.max(items[i].rect.y1)));
    let lines = (bottom - top) / (1.2 * em);
    let min_gutter = MIN_GUTTER_EM.iter().find(|(min_lines, _)| lines >= *min_lines).map_or(1.0, |(_, g)| *g) * em;
    let columns: Vec<f32> = gaps(ids.iter().map(|&i| (items[i].rect.x0, items[i].rect.x1)).collect())
        .into_iter()
        .filter(|(a, b)| b - a >= min_gutter)
        .map(|(a, b)| (a + b) / 2.0)
        .collect();
    if !columns.is_empty() {
        return split(ids, &columns, |i| (items[i].rect.x0 + items[i].rect.x1) / 2.0, items, out);
    }
    out.push(ids);
}

fn split(ids: Vec<usize>, cuts: &[f32], key: impl Fn(usize) -> f32, items: &[Item], out: &mut Vec<Vec<usize>>) {
    let mut parts: Vec<Vec<usize>> = vec![Vec::new(); cuts.len() + 1];
    for i in ids {
        let k = key(i);
        parts[cuts.iter().take_while(|&&c| k > c).count()].push(i);
    }
    for part in parts.into_iter().filter(|p| !p.is_empty()) {
        xy_cut(part, items, out);
    }
}

/// A region holding several of the engine's text blocks (a column and the marginal notes set
/// close to it) split into one region per block, top to bottom.
fn split_blocks(ids: Vec<usize>, items: &[Item]) -> Vec<Vec<usize>> {
    if !ids.iter().all(|&i| items[i].block.is_some()) {
        return vec![ids];
    }
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for i in ids {
        match blocks.iter_mut().find(|b| items[b[0]].block == items[i].block) {
            Some(b) => b.push(i),
            None => blocks.push(vec![i]),
        }
    }
    let top = |b: &Vec<usize>| b.iter().map(|&i| items[i].rect.y0).fold(f32::MAX, f32::min);
    blocks.sort_by(|a, b| top(a).total_cmp(&top(b)));
    blocks
}

fn region_lines(ids: &[usize], items: &[Item]) -> Vec<Line> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    if ids.iter().all(|&i| items[i].line.is_some()) {
        let mut sorted = ids.to_vec();
        sorted.sort_by_key(|&i| items[i].line);
        for i in sorted {
            match groups.last_mut() {
                Some(g) if items[g[0]].line == items[i].line => g.push(i),
                _ => groups.push(vec![i]),
            }
        }
        // Engines that return line fragments (EasyOCR, Vision observations) give one printed
        // line several ids: fragments whose centers are within half a font size are one line.
        let center = |g: &Vec<usize>| median(g.iter().map(|&i| items[i].center).collect());
        let size = |g: &Vec<usize>| median(g.iter().map(|&i| items[i].size).collect());
        groups.sort_by(|a, b| center(a).total_cmp(&center(b)));
        let mut merged: Vec<Vec<usize>> = Vec::new();
        for g in groups {
            match merged.last_mut() {
                Some(m) if (center(&g) - center(m)).abs() < 0.5 * size(m).max(size(&g)) => m.extend(g),
                _ => merged.push(g),
            }
        }
        groups = merged;
    } else {
        let mut sorted = ids.to_vec();
        sorted.sort_by(|&a, &b| items[a].center.total_cmp(&items[b].center));
        for i in sorted {
            match groups.last_mut() {
                Some(g) if (items[i].center - items[g[0]].center).abs() < 0.5 * items[g[0]].size.max(items[i].size) => g.push(i),
                _ => groups.push(vec![i]),
            }
        }
    }
    let mut lines: Vec<Line> = groups
        .into_iter()
        .map(|mut words| {
            words.sort_by(|&a, &b| items[a].rect.x0.total_cmp(&items[b].rect.x0));
            let par = majority(words.iter().map(|&i| items[i].par));
            Line {
                x0: words.iter().map(|&i| items[i].rect.x0).fold(f32::MAX, f32::min),
                x1: words.iter().map(|&i| items[i].rect.x1).fold(f32::MIN, f32::max),
                center: median(words.iter().map(|&i| items[i].center).collect()),
                size: median(words.iter().map(|&i| items[i].size).collect()),
                par,
                words,
            }
        })
        .collect();
    lines.sort_by(|a, b| a.center.total_cmp(&b.center));
    lines
}

fn majority(values: impl Iterator<Item = Option<u32>>) -> Option<u32> {
    let mut counts: Vec<(Option<u32>, usize)> = Vec::new();
    for v in values {
        match counts.iter_mut().find(|(k, _)| *k == v) {
            Some((_, n)) => *n += 1,
            None => counts.push((v, 1)),
        }
    }
    counts.into_iter().max_by_key(|&(_, n)| n).and_then(|(v, _)| v)
}

fn pitch(lines: &[Line]) -> f32 {
    let steps: Vec<f32> = lines.windows(2).map(|w| w[1].center - w[0].center).filter(|d| *d > 0.0).collect();
    if steps.is_empty() {
        1.2 * lines[0].size
    } else {
        median(steps)
    }
}

fn paragraphs(lines: &[Line], items: &[Item], fixer: &mut Fixer, dict: Option<&Dictionary>, joined: &mut usize) -> Vec<Paragraph> {
    let em = median(lines.iter().map(|l| l.size).collect());
    let left = lines.iter().map(|l| l.x0).fold(f32::MAX, f32::min);
    let right = lines.iter().map(|l| l.x1).fold(f32::MIN, f32::max);
    let step = pitch(lines);
    let has_par = lines.iter().all(|l| l.par.is_some());
    let indent = |l: &Line| l.x0 - left;

    let mut out = Vec::new();
    // the paragraph being built: its first line's indent and its words so far
    let mut current: Option<(f32, Vec<String>)> = None;
    for (n, line) in lines.iter().enumerate() {
        let breaks = n > 0 && {
            let prev = &lines[n - 1];
            let last = items[*prev.words.last().unwrap()].text.as_str();
            (has_par && line.par != prev.par)
                || line.center - prev.center > 1.6 * step
                || (indent(line) > 0.7 * em && indent(prev) < 0.4 * em)
                || (!has_par && prev.x1 < right - 3.0 * em && last.ends_with(['.', '!', '?', ':']))
        };
        if breaks {
            out.extend(current.take().map(|(indent, words)| Paragraph { text: finish(&words, fixer), indent }));
        }
        let (_, words) = current.get_or_insert_with(|| (indent(line).min(4.0 * em), Vec::new()));
        for (k, &i) in line.words.iter().enumerate() {
            let text = &items[i].text;
            if k == 0 {
                if let Some(whole) = words.last().and_then(|prev| join_hyphenated(prev, text, dict)) {
                    *joined += 1;
                    *words.last_mut().unwrap() = whole;
                    continue;
                }
            }
            words.push(text.clone());
        }
    }
    out.extend(current.map(|(indent, words)| Paragraph { text: finish(&words, fixer), indent }));
    out
}

/// The long-s fix runs on whole words, after the halves of a hyphenated word are joined.
fn finish(words: &[String], fixer: &mut Fixer) -> String {
    words.iter().map(|w| fixer.word(w)).collect::<Vec<_>>().join(" ")
}

/// Helvetica advance widths in em, coarse (enough to estimate how many lines a paragraph needs).
fn advance_em(c: char) -> f32 {
    match c {
        ' ' | 'i' | 'j' | 'l' | '.' | ',' | ':' | ';' | '\'' | '!' | '|' => 0.24,
        'f' | 't' | 'r' | 'I' | '(' | ')' | '-' | '[' | ']' => 0.32,
        'm' | 'M' | 'W' => 0.85,
        'w' => 0.72,
        c if c.is_uppercase() => 0.68,
        c if c.is_ascii_digit() => 0.556,
        _ => 0.54,
    }
}

/// `scale` < 1 sets every block smaller than its words, for a page whose reflowed text would
/// otherwise run off the page (the width estimate here is coarse).
fn write_region(out: &mut String, lines: &[Line], paragraphs: &[Paragraph], width_pt: f32, height_pt: f32, scale: f32) {
    let left = lines.iter().map(|l| l.x0).fold(f32::MAX, f32::min);
    let right = lines.iter().map(|l| l.x1).fold(f32::MIN, f32::max);
    let box_w = (right - left).max(1.0);
    let mut size = median(lines.iter().map(|l| l.size).collect());
    let mut line_h = if lines.len() > 1 { pitch(lines).clamp(1.0 * size, 1.8 * size) } else { 1.2 * size };
    let box_h = lines.last().unwrap().center - lines[0].center + line_h;
    // the room the block has: its own lines, and never past the bottom of the page
    let room = (box_h + 0.5 * line_h).min(height_pt - (lines[0].center - line_h / 2.0));
    // where the block's lines were on the page (data-box: left top right bottom, in %)
    let extent = [left, lines[0].center - line_h / 2.0, right, lines[0].center - line_h / 2.0 + box_h];
    size *= scale;
    line_h *= scale;
    let single = lines.len() == 1;

    // shrink until the reflowed text needs no more lines than the scan had room for
    if !single {
        let ratio = line_h / size;
        for _ in 0..20 {
            let needed: f32 = paragraphs
                .iter()
                .map(|p| ((p.indent + p.text.chars().map(advance_em).sum::<f32>() * size) / box_w * 1.04).ceil().max(1.0))
                .sum();
            if needed * line_h <= room || size < 4.0 {
                break;
            }
            size *= 0.95;
            line_h = ratio * size;
        }
    }

    let top = lines[0].center - line_h / 2.0;
    let _ = write!(
        out,
        "<div class=\"region\" data-box=\"{:.3} {:.3} {:.3} {:.3}\" style=\"left: {:.4}%; top: {:.4}%; width: {:.4}%; font-size: {size:.2}pt; line-height: {line_h:.2}pt; text-align: {};{}\">",
        100.0 * extent[0] / width_pt,
        100.0 * extent[1].max(0.0) / height_pt,
        100.0 * extent[2] / width_pt,
        100.0 * extent[3].min(height_pt) / height_pt,
        100.0 * left / width_pt,
        100.0 * top.max(0.0) / height_pt,
        100.0 * (box_w + 0.5 * size) / width_pt,
        if lines.len() >= 3 { "justify" } else { "left" },
        if single { " white-space: nowrap;" } else { "" },
    );
    for p in paragraphs {
        if p.indent > 0.5 * size {
            let _ = write!(out, "<p style=\"text-indent: {:.1}pt;\">{}</p>", p.indent, escape(&p.text));
        } else {
            let _ = write!(out, "<p>{}</p>", escape(&p.text));
        }
    }
    out.push_str("</div>\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::LongS;

    /// A word in page points: x, center line y, width, font size 10pt.
    fn word(text: &str, x: f32, y: f32, w: f32, par: Option<u32>, line: Option<u32>) -> Word {
        let (page_w, page_h, size) = (400.0, 400.0, 10.0);
        Word {
            text: text.into(),
            left: x / page_w,
            width: w / page_w,
            top: (y - TEXT_CENTER_EM * size) / page_h,
            height: size / page_h,
            font_size_pt: size,
            block: None,
            par,
            line,
        }
    }

    /// Two columns (x 20-175 and 200-380) of four lines, 12pt apart, under a full-width header.
    /// Words sit at ragged positions, as in real text, so no white runs through a column.
    fn two_columns(with_ids: bool) -> Vec<Word> {
        let id = |v: u32| with_ids.then_some(v);
        let mut words = vec![word("HEADER", 150.0, 20.0, 60.0, id(0), id(0))];
        let left: [&[&str]; 4] = [&["The", "first", "con-"], &["tinues", "here."], &["New", "para-"], &["graph", "ends."]];
        let right: [&[&str]; 4] = [&["Right", "column", "starts"], &["and", "ends"], &["just", "now", "after"], &["four", "lines."]];
        for (col, rows, x0, x1, first_line, pars) in [(0, left, 20.0, 175.0, 1, [1, 1, 2, 2]), (1, right, 200.0, 380.0, 10, [3, 3, 3, 3])] {
            for (r, row) in rows.iter().enumerate() {
                let y = 50.0 + 12.0 * r as f32;
                let indent = if col == 0 && r == 2 { 15.0 } else { 0.0 }; // "New para-" starts a paragraph
                let step = (x1 - x0 - indent) / row.len() as f32;
                for (k, t) in row.iter().enumerate() {
                    let x = x0 + indent + step * k as f32 + 3.0 * ((r + k) % 3) as f32;
                    words.push(word(t, x, y, step - 12.0, id(pars[r]), id(first_line + r as u32)));
                }
            }
        }
        words
    }

    fn texts(html: &str) -> Vec<String> {
        html.split("<p").skip(1).map(|p| p[p.find('>').unwrap() + 1..p.find("</p>").unwrap()].to_string()).collect()
    }

    #[test]
    fn reads_columns_in_order_and_joins_hyphens() {
        for with_ids in [true, false] {
            let mut fixer = Fixer::new(LongS::Keep, None);
            let (html, stats) = page_html(&two_columns(with_ids), 400.0, 400.0, "en", &mut fixer, None, 1.0);
            assert_eq!(
                texts(&html),
                ["HEADER", "The first continues here.", "New paragraph ends.", "Right column starts and ends just now after four lines."],
                "with_ids={with_ids}"
            );
            assert_eq!(stats.regions, 3);
            assert_eq!(stats.hyphens_joined, 2);
            assert_eq!(html.matches("text-indent:").count(), 1, "only the indented paragraph");
        }
    }

    #[test]
    fn line_fragments_with_their_own_ids_are_one_line() {
        // EasyOCR-style: "The first" and "line here." are two detections on one baseline
        let words = vec![
            word("The", 20.0, 50.0, 30.0, None, Some(0)),
            word("first", 60.0, 50.0, 40.0, None, Some(0)),
            word("line", 120.0, 50.4, 30.0, None, Some(1)),
            word("here.", 160.0, 50.4, 40.0, None, Some(1)),
            word("Second", 20.0, 62.0, 60.0, None, Some(2)),
            word("line.", 90.0, 62.0, 110.0, None, Some(2)),
        ];
        let mut fixer = Fixer::new(LongS::Keep, None);
        let (html, stats) = page_html(&words, 400.0, 400.0, "en", &mut fixer, None, 1.0);
        assert_eq!(stats.paragraphs, 1);
        assert_eq!(texts(&html), ["The first line here. Second line."]);
    }

    #[test]
    fn gaps_between_merged_intervals() {
        assert_eq!(gaps(vec![(0.0, 2.0), (1.0, 3.0), (5.0, 6.0), (6.5, 7.0)]), vec![(3.0, 5.0), (6.0, 6.5)]);
    }
}
