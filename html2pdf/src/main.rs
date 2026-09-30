use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use printpdf::html::{build_font_pool, SharedFontPool};
use printpdf::{Base64OrRaw, GeneratePdfOptions, PdfDocument, PdfSaveOptions, PdfWarnMsg};
use serde::Deserialize;

const MM_PER_PT: f32 = 25.4 / 72.0;
const A4_MM: (f32, f32) = (210.0, 297.0);

/// Convert a pdf-ocr-bench `pages.zip` (page_NNN.html + page_NNN.png) into one PDF.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Input zip produced by `pdf-ocr-bench run`
    input: PathBuf,

    /// Output PDF [default: <input stem>.pdf]
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Extra font as NAME=PATH (repeatable). HTML refers to it via `font-family: NAME`.
    #[arg(long = "font", value_name = "NAME=PATH")]
    fonts: Vec<String>,

    /// Render the OCR text visibly (red) instead of transparent, for checking alignment
    #[arg(long)]
    visible_text: bool,

    /// Document title
    #[arg(long, default_value = "OCR Document")]
    title: String,

    /// Print printpdf warnings
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Deserialize, Default)]
struct Metadata {
    engine: Option<String>,
    #[serde(default)]
    pages: Vec<PageMeta>,
}

#[derive(Deserialize)]
struct PageMeta {
    html: String,
    width_pt: f32,
    height_pt: f32,
}

struct Bundle {
    pages: Vec<(String, String)>,
    assets: BTreeMap<String, Vec<u8>>,
    metadata: Metadata,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output = args.output.clone().unwrap_or_else(|| args.input.with_extension("pdf"));
    let start = Instant::now();

    let bundle = read_bundle(&args.input)?;
    if bundle.pages.is_empty() {
        bail!("{}: no page_NNN.html entries found", args.input.display());
    }
    println!(
        "[html2pdf] {}: {} pages (engine: {})",
        args.input.display(),
        bundle.pages.len(),
        bundle.metadata.engine.as_deref().unwrap_or("unknown"),
    );

    let fonts = load_fonts(&args.fonts)?;
    let pool = build_font_pool(&raw_fonts(&fonts), None);

    let mut doc = PdfDocument::new(&args.title);
    let mut warnings = Vec::new();
    for (i, (name, html)) in bundle.pages.iter().enumerate() {
        let page_start = Instant::now();
        let html = if args.visible_text { make_text_visible(html) } else { html.clone() };
        let size = page_size_mm(&html, &bundle.metadata, name);
        let page = render_page(&html, &bundle.assets, &fonts, &pool, size, &mut warnings)
            .with_context(|| format!("rendering {name}"))?;
        if page.pages.is_empty() {
            bail!("{name}: printpdf produced no page ({} warnings)", warnings.len());
        }
        if page.pages.len() > 1 {
            eprintln!("[html2pdf] warning: {name} overflowed into {} pages", page.pages.len());
        }
        doc.append_document(page);
        println!(
            "[html2pdf] Page {}/{}: {name} {:.1}x{:.1}mm, {:.2}s",
            i + 1,
            bundle.pages.len(),
            size.0,
            size.1,
            page_start.elapsed().as_secs_f32()
        );
    }

    let bytes = doc.save(&PdfSaveOptions::default(), &mut warnings);
    std::fs::write(&output, &bytes).with_context(|| format!("writing {}", output.display()))?;
    report_warnings(&warnings, args.verbose);
    println!(
        "[html2pdf] Wrote {} ({} pages, {:.1} KiB) in {:.1}s",
        output.display(),
        doc.page_count(),
        bytes.len() as f32 / 1024.0,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

fn read_bundle(path: &Path) -> Result<Bundle> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("reading zip")?;
    let mut pages = Vec::new();
    let mut assets = BTreeMap::new();
    let mut metadata = Metadata::default();

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().rsplit('/').next().unwrap_or_default().to_string();
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;

        if name == "metadata.json" {
            metadata = serde_json::from_slice(&bytes).context("parsing metadata.json")?;
            continue;
        }
        if page_number(&name).is_some() {
            pages.push((name, String::from_utf8(bytes).context("page HTML is not UTF-8")?));
            continue;
        }
        assets.insert(name, bytes);
    }

    pages.sort_by_key(|(name, _)| page_number(name));
    Ok(Bundle { pages, assets, metadata })
}

/// `page_012.html` -> 12
fn page_number(name: &str) -> Option<u32> {
    name.strip_prefix("page_")?.strip_suffix(".html")?.parse().ok()
}

fn render_page(
    html: &str,
    assets: &BTreeMap<String, Vec<u8>>,
    fonts: &BTreeMap<String, Base64OrRaw>,
    pool: &SharedFontPool,
    (width, height): (f32, f32),
    warnings: &mut Vec<PdfWarnMsg>,
) -> Result<PdfDocument> {
    // Only hand over the images this page references: printpdf decodes every entry.
    let images = assets
        .iter()
        .filter(|(name, _)| html.contains(&format!("src=\"{name}\"")))
        .map(|(name, bytes)| (name.clone(), Base64OrRaw::Raw(bytes.clone())))
        .collect();
    let options = GeneratePdfOptions {
        page_width: Some(width),
        page_height: Some(height),
        margin_top: Some(0.0),
        margin_right: Some(0.0),
        margin_bottom: Some(0.0),
        margin_left: Some(0.0),
        ..Default::default()
    };
    PdfDocument::from_html_with_cache(html, &images, fonts, &options, warnings, Some(pool.clone()))
        .map_err(anyhow::Error::msg)
}

/// Page size in mm: `<meta name="pdf.options.pageWidth">` wins, then metadata.json, then
/// the `.page { width: Xpt; height: Ypt }` rule, then A4.
fn page_size_mm(html: &str, metadata: &Metadata, name: &str) -> (f32, f32) {
    let from_meta_tags = || Some((meta_content(html, "pdf.options.pageWidth")?, meta_content(html, "pdf.options.pageHeight")?));
    let from_metadata = || {
        metadata
            .pages
            .iter()
            .find(|p| p.html == name)
            .map(|p| (p.width_pt * MM_PER_PT, p.height_pt * MM_PER_PT))
    };
    let from_css = || Some((css_pt(html, "width")? * MM_PER_PT, css_pt(html, "height")? * MM_PER_PT));
    from_meta_tags().or_else(from_metadata).or_else(from_css).unwrap_or(A4_MM)
}

fn meta_content(html: &str, name: &str) -> Option<f32> {
    let tag_start = html.find(&format!("name=\"{name}\""))?;
    let tag = &html[tag_start..tag_start + html[tag_start..].find('>')?];
    let value = tag.split("content=\"").nth(1)?.split('"').next()?;
    value.trim().parse().ok()
}

/// First `<prop>: <n>pt` inside the `.page {` rule.
fn css_pt(html: &str, prop: &str) -> Option<f32> {
    let rule_start = html.find(".page {")?;
    let rule = &html[rule_start..rule_start + html[rule_start..].find('}')?];
    rule.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(prop)?.trim_start().strip_prefix(':'))
        .and_then(|v| v.trim().trim_end_matches(';').strip_suffix("pt"))
        .and_then(|v| v.trim().parse().ok())
}

fn make_text_visible(html: &str) -> String {
    html.replacen("color: transparent;", "color: rgba(255, 0, 0, 0.6);", 1)
}

fn load_fonts(specs: &[String]) -> Result<BTreeMap<String, Base64OrRaw>> {
    specs
        .iter()
        .map(|spec| {
            let (name, path) = spec.split_once('=').with_context(|| format!("--font expects NAME=PATH, got '{spec}'"))?;
            let bytes = std::fs::read(path).with_context(|| format!("reading font {path}"))?;
            Ok((name.to_string(), Base64OrRaw::Raw(bytes)))
        })
        .collect()
}

fn raw_fonts(fonts: &BTreeMap<String, Base64OrRaw>) -> BTreeMap<String, Vec<u8>> {
    fonts
        .iter()
        .filter_map(|(name, font)| match font {
            Base64OrRaw::Raw(bytes) => Some((name.clone(), bytes.clone())),
            Base64OrRaw::B64(_) => None,
        })
        .collect()
}

fn report_warnings(warnings: &[PdfWarnMsg], verbose: bool) {
    if warnings.is_empty() {
        return;
    }
    if !verbose {
        eprintln!("[html2pdf] {} printpdf warnings (use -v to show)", warnings.len());
        return;
    }
    warnings.iter().for_each(|w| eprintln!("[html2pdf] warning: {w:?}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    const HTML: &str = r#"<head>
<meta name="pdf.options.pageWidth" content="210.0016">
<meta name="pdf.options.pageHeight" content="297.0001">
<style>
  .page {
    position: relative;
    width: 612.00pt;
    height: 792.00pt;
  }
</style>"#;

    #[test]
    fn page_numbers() {
        assert_eq!(page_number("page_001.html"), Some(1));
        assert_eq!(page_number("page_120.html"), Some(120));
        assert_eq!(page_number("page_001.png"), None);
        assert_eq!(page_number("metadata.json"), None);
    }

    #[test]
    fn size_from_meta_tags() {
        let (w, h) = page_size_mm(HTML, &Metadata::default(), "page_001.html");
        assert!((w - 210.0016).abs() < 1e-3 && (h - 297.0001).abs() < 1e-3);
    }

    #[test]
    fn size_from_css() {
        let html = HTML.replace("pdf.options.page", "x");
        let (w, h) = page_size_mm(&html, &Metadata::default(), "page_001.html");
        assert!((w - 215.9).abs() < 0.01 && (h - 279.4).abs() < 0.01);
    }

    #[test]
    fn size_defaults_to_a4() {
        assert_eq!(page_size_mm("<html></html>", &Metadata::default(), "page_001.html"), A4_MM);
    }
}
