//! `--layout-report`: where printpdf/azul actually put each text block (`.region`) of a page.
//!
//! The page is rendered once more with every block in its own text colour, `rgb(0, g, b)` with
//! (g, b) the block number. In the rendered ops, the fill colour set before each glyph run says
//! which block the run belongs to; the run's text matrix and the glyphs' advance widths (from the
//! fonts printpdf embeds, plus the kerning in the run) give where its glyphs are. That is
//! the rendered extent of every block, to compare with the box its CSS asked for and with the other
//! blocks: text outside its own box points at the layout engine, blocks that overlap while each
//! stays inside its box point at the boxes (the page's HTML).

use printpdf::{Color, Op, PdfDocument, PdfFontHandle};
use serde::Serialize;

use crate::words::unescape;

/// A rectangle in points, origin top left.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Rect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Rect {
    fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }

    fn intersection(&self, other: &Rect) -> f32 {
        let w = self.x1.min(other.x1) - self.x0.max(other.x0);
        let h = self.y1.min(other.y1) - self.y0.max(other.y0);
        w.max(0.0) * h.max(0.0)
    }
}

/// One block as the page's HTML declares it.
#[derive(Clone, Debug, Serialize)]
pub struct Region {
    pub index: usize,
    pub role: String,
    pub text: String,
    /// `left`, `top`, `width` of the block (the height follows from its text)
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub font_size: f32,
    pub nowrap: bool,
}

#[derive(Debug, Serialize)]
pub struct RegionReport {
    #[serde(flatten)]
    pub region: Region,
    /// Where its glyphs landed (None: nothing of it was drawn)
    pub rendered: Option<Rect>,
    /// Text drawn left or right of the block's own box (beyond half an em): the engine did not keep
    /// the text inside `width`
    pub outside_box: bool,
    /// Text drawn beyond the page
    pub off_page: bool,
    /// Other blocks whose rendered text covers more than 10% of this one's or theirs
    pub overlaps: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct PageReport {
    pub page: String,
    pub width_pt: f32,
    pub height_pt: f32,
    pub regions: Vec<RegionReport>,
}

impl PageReport {
    pub fn problems(&self) -> (usize, usize, usize, usize) {
        let count = |f: &dyn Fn(&RegionReport) -> bool| self.regions.iter().filter(|r| f(r)).count();
        (
            count(&|r| r.outside_box),
            count(&|r| r.off_page),
            count(&|r| !r.overlaps.is_empty()),
            count(&|r| r.rendered.is_none() && !r.region.text.trim().is_empty()),
        )
    }
}

/// The page with every block tagged by its text colour, and the blocks it declares.
pub fn tag_regions(html: &str, width_pt: f32, height_pt: f32) -> (String, Vec<Region>) {
    const OPEN: &str = "<div class=\"region\"";
    let mut out = String::with_capacity(html.len() + 64);
    let mut regions = Vec::new();
    let mut pos = 0;
    while let Some(i) = html[pos..].find(OPEN) {
        let start = pos + i;
        let Some(gt) = html[start..].find('>') else { break };
        let tag = &html[start..start + gt];
        let index = regions.len();
        let style = attr(tag, "style").unwrap_or_default();
        let body_end = html[start..].find("</div>").map_or(html.len(), |e| start + e);
        let text: String = strip_tags(&html[start + gt + 1..body_end]);
        regions.push(Region {
            index,
            role: attr(tag, "data-role").unwrap_or("text").to_string(),
            text: text.chars().take(80).collect(),
            left: percent(style, "left") * width_pt,
            top: percent(style, "top") * height_pt,
            width: percent(style, "width") * width_pt,
            font_size: style_value(style, "font-size").and_then(|v| v.strip_suffix("pt")?.trim().parse().ok()).unwrap_or(0.0),
            nowrap: style.contains("nowrap"),
        });
        let k = index + 1; // colour 0 stays black
        let colour = format!(" color: rgb(0, {}, {});", (k / 256) % 256, k % 256);
        out.push_str(&html[pos..start]);
        match tag.find("style=\"") {
            Some(s) => {
                let value_end = start + s + 7 + style.len();
                out.push_str(&html[start..value_end]);
                out.push_str(&colour);
                out.push_str(&html[value_end..start + gt]);
            }
            None => {
                out.push_str(tag);
                out.push_str(&format!(" style=\"{}\"", colour.trim()));
            }
        }
        pos = start + gt;
    }
    out.push_str(&html[pos..]);
    (out, regions)
}

/// Rendered extents of the tagged blocks and the problems they show.
pub fn report(page: &str, doc: &PdfDocument, regions: Vec<Region>, width_pt: f32, height_pt: f32) -> PageReport {
    let mut extents: Vec<Option<Rect>> = vec![None; regions.len()];
    let mut current: Option<usize> = None;
    let mut size = 0.0f32;
    let mut font: Option<&printpdf::PdfFont> = None;
    let mut origin: Option<(f32, f32)> = None;
    for op in doc.pages.first().map(|p| p.ops.as_slice()).unwrap_or_default() {
        match op {
            Op::SetFillColor { col: Color::Rgb(c) } => {
                let (r, g, b) = ((c.r * 255.0).round() as usize, (c.g * 255.0).round() as usize, (c.b * 255.0).round() as usize);
                let k = g * 256 + b;
                current = (r == 0 && k >= 1 && k <= regions.len()).then(|| k - 1);
            }
            Op::SetFillColor { .. } => current = None,
            Op::SetFont { size: s, font: handle } => {
                size = s.0;
                font = match handle {
                    PdfFontHandle::External(id) => doc.resources.fonts.map.get(id),
                    PdfFontHandle::Builtin(_) => None,
                };
            }
            Op::SetTextMatrix { matrix } => {
                let m = matrix.as_array();
                origin = Some((m[4], m[5]));
            }
            Op::ShowText { items } => {
                let (Some(k), Some((x, y))) = (current, origin) else { continue };
                let advance = run_advance(items, font, size);
                origin = Some((x + advance, y)); // a show op without a new Tm continues here
                // the run on its baseline (PDF y up); text ascends ~0.75 em, descends ~0.2 em
                let glyph = Rect {
                    x0: x,
                    x1: x + advance.max(0.0),
                    y0: height_pt - y - 0.75 * size,
                    y1: height_pt - y + 0.2 * size,
                };
                let e = extents[k].get_or_insert(glyph);
                e.x0 = e.x0.min(glyph.x0);
                e.x1 = e.x1.max(glyph.x1);
                e.y0 = e.y0.min(glyph.y0);
                e.y1 = e.y1.max(glyph.y1);
            }
            _ => {}
        }
    }

    let mut reports: Vec<RegionReport> = regions
        .into_iter()
        .zip(&extents)
        .map(|(region, rendered)| {
            let em = region.font_size.max(1.0);
            let outside_box = rendered.is_some_and(|e| {
                !region.nowrap && (e.x0 < region.left - 0.5 * em || e.x1 > region.left + region.width + 0.5 * em)
            });
            let off_page =
                rendered.is_some_and(|e| e.x0 < -1.0 || e.y0 < -1.0 || e.x1 > width_pt + 1.0 || e.y1 > height_pt + 1.0);
            RegionReport { region, rendered: *rendered, outside_box, off_page, overlaps: Vec::new() }
        })
        .collect();
    for i in 0..reports.len() {
        for j in 0..reports.len() {
            let (Some(a), Some(b)) = (reports[i].rendered, reports[j].rendered) else { continue };
            if i != j && a.intersection(&b) > 0.1 * a.area().min(b.area()) {
                reports[i].overlaps.push(j);
            }
        }
    }
    PageReport { page: page.to_string(), width_pt, height_pt, regions: reports }
}

/// How far a text-show op moves the pen, in pt: the glyphs' advance widths from the embedded font,
/// and the kerning numbers between them (thousandths of an em, positive moves left). Without the
/// font, half an em per glyph.
fn run_advance(items: &[printpdf::TextItem], font: Option<&printpdf::PdfFont>, size: f32) -> f32 {
    let per_em = |gid: u16| -> f32 {
        font.map_or(0.5, |f| {
            let upem = f32::from(f.parsed_font.font_metrics.units_per_em.max(1));
            f32::from(f.parsed_font.get_horizontal_advance(gid)) / upem
        })
    };
    items
        .iter()
        .map(|item| match item {
            printpdf::TextItem::GlyphIds(glyphs) => {
                glyphs.iter().map(|g| (per_em(g.gid) - g.offset / 1000.0) * size).sum::<f32>()
            }
            printpdf::TextItem::Offset(o) => -o / 1000.0 * size,
            printpdf::TextItem::Text(t) => 0.5 * size * t.chars().count() as f32,
        })
        .sum()
}

fn attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let start = tag.find(&key)? + key.len();
    Some(&tag[start..start + tag[start..].find('"')?])
}

fn style_value<'a>(style: &'a str, key: &str) -> Option<&'a str> {
    style.split(';').find_map(|d| {
        let (k, v) = d.split_once(':')?;
        (k.trim() == key).then(|| v.trim())
    })
}

fn percent(style: &str, key: &str) -> f32 {
    style_value(style, key).and_then(|v| v.strip_suffix('%')?.trim().parse::<f32>().ok()).map_or(0.0, |v| v / 100.0)
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => {
                in_tag = true;
                out.push(' ');
            }
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    unescape(out.split_whitespace().collect::<Vec<_>>().join(" ").as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_each_region_with_its_own_colour() {
        let html = r#"<div class="page"><div class="region" data-role="note" style="left: 10%; top: 20%; width: 30%; font-size: 9pt;"><p>A &amp; B</p></div><div class="region" style="left: 50%; top: 20%; width: 40%; font-size: 11pt; white-space: nowrap;"><p>C</p></div></div>"#;
        let (tagged, regions) = tag_regions(html, 100.0, 200.0);
        assert!(tagged.contains("font-size: 9pt; color: rgb(0, 0, 1);\">"));
        assert!(tagged.contains("white-space: nowrap; color: rgb(0, 0, 2);\">"));
        assert_eq!(regions.len(), 2);
        assert_eq!((regions[0].role.as_str(), regions[0].text.as_str()), ("note", "A & B"));
        assert!((regions[0].left - 10.0).abs() < 1e-4 && (regions[0].top - 40.0).abs() < 1e-4);
        assert!((regions[1].width - 40.0).abs() < 1e-4 && regions[1].nowrap && regions[1].font_size == 11.0);
    }
}
