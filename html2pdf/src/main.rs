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
use zip::ZipArchive;

const MM_PER_PT: f32 = 25.4 / 72.0;

/// Convert a pdf-ocr-bench `pages.zip` (page_NNN.html + metadata.json) into one text-only PDF.
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

    /// Document title
    #[arg(long, default_value = "OCR Document")]
    title: String,

    /// Print printpdf warnings
    #[arg(short, long)]
    verbose: bool,
}

/// The zip's manifest, written by pdf_ocr_bench.html_output.zipper.
#[derive(Deserialize)]
struct Metadata {
    engine: String,
    pages: Vec<PageMeta>,
}

#[derive(Deserialize)]
struct PageMeta {
    page_num: u32,
    html: String,
    width_pt: f32,
    height_pt: f32,
}

impl PageMeta {
    fn size_mm(&self) -> (f32, f32) {
        (self.width_pt * MM_PER_PT, self.height_pt * MM_PER_PT)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output = args.output.clone().unwrap_or_else(|| args.input.with_extension("pdf"));
    let start = Instant::now();

    let mut zip = open_zip(&args.input)?;
    let metadata: Metadata = serde_json::from_slice(&read_entry(&mut zip, "metadata.json")?)
        .context("parsing metadata.json")?;
    if metadata.pages.is_empty() {
        bail!("{}: metadata.json lists no pages", args.input.display());
    }
    println!(
        "[html2pdf] {}: {} pages (engine: {})",
        args.input.display(),
        metadata.pages.len(),
        metadata.engine
    );

    let fonts = load_fonts(&args.fonts)?;
    let pool = build_font_pool(&raw_fonts(&fonts), None);

    let mut pages = metadata.pages;
    pages.sort_by_key(|p| p.page_num);

    let mut doc = PdfDocument::new(&args.title);
    let mut warnings = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        let page_start = Instant::now();
        let rendered = render_page(&mut zip, page, &fonts, &pool, &mut warnings)
            .with_context(|| format!("rendering {}", page.html))?;
        doc.append_document(rendered);
        let (w, h) = page.size_mm();
        println!(
            "[html2pdf] Page {}/{}: {} {w:.1}x{h:.1}mm, {:.2}s",
            i + 1,
            pages.len(),
            page.html,
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

fn open_zip(path: &Path) -> Result<ZipArchive<File>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    ZipArchive::new(file).with_context(|| format!("reading zip {}", path.display()))
}

fn read_entry(zip: &mut ZipArchive<File>, name: &str) -> Result<Vec<u8>> {
    let mut entry = zip.by_name(name).with_context(|| format!("zip has no {name}"))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Render one HTML page to a single-page document sized from metadata.json.
fn render_page(
    zip: &mut ZipArchive<File>,
    page: &PageMeta,
    fonts: &BTreeMap<String, Base64OrRaw>,
    pool: &SharedFontPool,
    warnings: &mut Vec<PdfWarnMsg>,
) -> Result<PdfDocument> {
    let html = String::from_utf8(read_entry(zip, &page.html)?).context("page HTML is not UTF-8")?;
    let (width, height) = page.size_mm();
    let options = GeneratePdfOptions {
        page_width: Some(width),
        page_height: Some(height),
        margin_top: Some(0.0),
        margin_right: Some(0.0),
        margin_bottom: Some(0.0),
        margin_left: Some(0.0),
        ..Default::default()
    };
    let doc = PdfDocument::from_html_with_cache(&html, &BTreeMap::new(), fonts, &options, warnings, Some(pool.clone()))
        .map_err(anyhow::Error::msg)?;
    match doc.pages.len() {
        1 => Ok(doc),
        0 => bail!("printpdf produced no page"),
        n => bail!("content overflowed into {n} pages; the page box must fit the page size"),
    }
}

fn load_fonts(specs: &[String]) -> Result<BTreeMap<String, Base64OrRaw>> {
    specs
        .iter()
        .map(|spec| {
            let (name, path) = spec
                .split_once('=')
                .with_context(|| format!("--font expects NAME=PATH, got '{spec}'"))?;
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

    const METADATA: &str = r#"{
        "engine": "tesseract",
        "page_count": 2,
        "pages": [
            {"page_num": 1, "html": "page_002.html", "width_pt": 612.0, "height_pt": 792.0, "words": 3},
            {"page_num": 0, "html": "page_001.html", "width_pt": 595.28, "height_pt": 841.89, "words": 5}
        ]
    }"#;

    #[test]
    fn parses_manifest_and_converts_sizes() {
        let meta: Metadata = serde_json::from_str(METADATA).unwrap();
        assert_eq!(meta.engine, "tesseract");
        let (w, h) = meta.pages[0].size_mm();
        assert!((w - 215.9).abs() < 0.01 && (h - 279.4).abs() < 0.01);
        let (w, h) = meta.pages[1].size_mm();
        assert!((w - 210.0).abs() < 0.01 && (h - 297.0).abs() < 0.01);
    }

    #[test]
    fn font_spec_requires_name_and_path() {
        assert!(load_fonts(&["no-equals-sign".into()]).is_err());
        assert!(load_fonts(&["Name=/definitely/missing.ttf".into()]).is_err());
    }
}
