mod flow;
mod measure;
mod text;
mod words;

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

use text::{Dictionary, Fixer, LongS};

const MM_PER_PT: f32 = 25.4 / 72.0;
const SYSTEM_WORD_LIST: &str = "/usr/share/dict/words";
/// Text sizes a flowed page is tried at until it fits on one page.
const FLOW_SCALES: [f32; 4] = [1.0, 0.9, 0.8, 0.7];

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Layout {
    /// Every word at its OCR position (as in the HTML)
    Positioned,
    /// Rebuild text blocks, lines and paragraphs, and set each block as flowing text in place
    Flow,
}

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

    /// Positioned words, or text blocks rebuilt into columns and paragraphs (joins words split
    /// by a line-end hyphen)
    #[arg(long, value_enum, default_value_t = Layout::Positioned)]
    layout: Layout,

    /// The long s (ſ) of old print: keep it, turn it into s, or also repair f read for it
    /// ("fhould" -> "should") using the word list
    #[arg(long, value_enum, default_value_t = LongS::Keep)]
    long_s: LongS,

    /// Word list for `--long-s repair` and for joining hyphenated words, one word per line
    #[arg(long, value_name = "PATH", default_value = SYSTEM_WORD_LIST)]
    dict: PathBuf,

    /// Also write the HTML each page is rendered from into this directory
    #[arg(long, value_name = "DIR")]
    html_out: Option<PathBuf>,

    /// Write where the layout engine put each text block (`.region`) of every page, and the
    /// problems that shows (text outside its box, off the page, blocks overlapping), as JSON
    #[arg(long, value_name = "FILE")]
    layout_report: Option<PathBuf>,

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
    let images = load_images(&mut zip)?;
    let dict = load_dictionary(&args)?;
    let mut fixer = Fixer::new(args.long_s, dict.as_ref());
    let mut flow_stats = (0, 0, 0);
    let mut page_reports = Vec::new();
    if let Some(dir) = &args.html_out {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    let mut pages = metadata.pages;
    pages.sort_by_key(|p| p.page_num);

    let mut doc = PdfDocument::new(&args.title);
    let mut warnings = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        let page_start = Instant::now();
        let source = String::from_utf8(read_entry(&mut zip, &page.html)?).context("page HTML is not UTF-8")?;
        // A flowed page whose text runs off the page (printpdf starts a second one) is set again
        // smaller; positioned words always fit.
        let scales: &[f32] = if args.layout == Layout::Flow { &FLOW_SCALES } else { &[1.0] };
        let mut rendered = None;
        let mut final_html = String::new();
        for (attempt, &scale) in scales.iter().enumerate() {
            let counts = (fixer.long_s, fixer.repaired, flow_stats);
            let html = transform(&source, page, &args, &mut fixer, dict.as_ref(), &mut flow_stats, scale);
            let doc = render_page(&html, page, &images, &fonts, &pool, &mut warnings)
                .with_context(|| format!("rendering {}", page.html))?;
            let last = attempt + 1 == scales.len();
            if doc.pages.len() == 1 || last {
                if doc.pages.len() > 1 {
                    eprintln!(
                        "[html2pdf] {}: text still runs off the page at {:.0}% size; only the first page is kept",
                        page.html,
                        100.0 * scale
                    );
                } else if scale < 1.0 {
                    eprintln!("[html2pdf] {}: set at {:.0}% size to fit the page", page.html, 100.0 * scale);
                }
                if let Some(dir) = &args.html_out {
                    std::fs::write(dir.join(&page.html), &html).with_context(|| format!("writing {}", page.html))?;
                }
                rendered = Some(doc);
                final_html = html;
                break;
            }
            (fixer.long_s, fixer.repaired, flow_stats) = counts; // the next attempt counts again
        }
        if args.layout_report.is_some() {
            let (tagged, regions) = measure::tag_regions(&final_html, page.width_pt, page.height_pt);
            if !regions.is_empty() {
                let doc = render_page(&tagged, page, &images, &fonts, &pool, &mut Vec::new())
                    .with_context(|| format!("rendering {} for the layout report", page.html))?;
                page_reports.push(measure::report(&page.html, &doc, regions, page.width_pt, page.height_pt));
            }
        }
        let mut rendered = rendered.expect("at least one attempt");
        rendered.pages.truncate(1);
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

    if let Some(path) = &args.layout_report {
        let totals = page_reports.iter().map(|r| r.problems()).fold((0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
        let blocks: usize = page_reports.iter().map(|r| r.regions.len()).sum();
        std::fs::write(path, serde_json::to_string_pretty(&page_reports)?).with_context(|| format!("writing {}", path.display()))?;
        println!(
            "[html2pdf] Layout report {}: {blocks} blocks; {} with text outside their box, {} off the page, {} overlapping another, {} not drawn",
            path.display(),
            totals.0,
            totals.1,
            totals.2,
            totals.3
        );
    }
    let bytes = doc.save(&PdfSaveOptions::default(), &mut warnings);
    std::fs::write(&output, &bytes).with_context(|| format!("writing {}", output.display()))?;
    report_warnings(&warnings, args.verbose);
    if args.layout == Layout::Flow {
        let (regions, paragraphs, joined) = flow_stats;
        println!("[html2pdf] Flow layout: {regions} text blocks, {paragraphs} paragraphs, {joined} hyphenated words joined");
    }
    if args.long_s != LongS::Keep {
        let mut examples: Vec<_> = fixer.examples.iter().map(|(from, to)| format!("{from} -> {to}")).collect();
        examples.sort();
        println!(
            "[html2pdf] Long s: {} words with ſ turned to s, {} words repaired{}",
            fixer.long_s,
            fixer.repaired,
            if examples.is_empty() { String::new() } else { format!(" (e.g. {})", examples.join(", ")) }
        );
    }
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

/// The word list, if `--long-s repair` or `--layout flow` (hyphen joining) uses one. Without it,
/// repair is an error and flow joins every line-end hyphen.
fn load_dictionary(args: &Args) -> Result<Option<Dictionary>> {
    let needed = args.long_s == LongS::Repair || args.layout == Layout::Flow;
    if !needed {
        return Ok(None);
    }
    match Dictionary::load(&args.dict) {
        Ok(dict) => {
            println!("[html2pdf] Word list {}: {} words", args.dict.display(), dict.len());
            Ok(Some(dict))
        }
        Err(err) if args.long_s == LongS::Repair => Err(err.context("--long-s repair needs a word list (--dict PATH)")),
        Err(err) => {
            eprintln!("[html2pdf] {err:#}; line-end hyphens are joined without checking words");
            Ok(None)
        }
    }
}

/// The page HTML as it will be rendered: words fixed for `--long-s`, and rebuilt for `--layout flow`.
fn transform(
    html: &str,
    page: &PageMeta,
    args: &Args,
    fixer: &mut Fixer,
    dict: Option<&Dictionary>,
    stats: &mut (usize, usize, usize),
    scale: f32,
) -> String {
    match args.layout {
        Layout::Positioned if args.long_s == LongS::Keep => html.to_string(),
        Layout::Positioned => {
            words::rewrite_region_texts(&words::rewrite_texts(html, |word| fixer.word(word)), |word| fixer.word(word))
        }
        Layout::Flow => {
            let lang = words::html_lang(html).unwrap_or("en").to_string();
            let (flowed, s) =
                flow::page_html(&words::parse_words(html), page.width_pt, page.height_pt, &lang, fixer, dict, scale);
            *stats = (stats.0 + s.regions, stats.1 + s.paragraphs, stats.2 + s.hyphens_joined);
            flowed
        }
    }
}

/// Render one HTML page to a single-page document sized from metadata.json.
fn render_page(
    html: &str,
    page: &PageMeta,
    images: &BTreeMap<String, Base64OrRaw>,
    fonts: &BTreeMap<String, Base64OrRaw>,
    pool: &SharedFontPool,
    warnings: &mut Vec<PdfWarnMsg>,
) -> Result<PdfDocument> {
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
    let doc = PdfDocument::from_html_with_cache(html, images, fonts, &options, warnings, Some(pool.clone()))
        .map_err(anyhow::Error::msg)?;
    if doc.pages.is_empty() {
        bail!("printpdf produced no page");
    }
    Ok(doc) // more than one page: the caller sets the text smaller or keeps the first

}

/// Every PNG/JPEG in the zip, keyed by its path there (what a page's `<img src>` names).
fn load_images(zip: &mut ZipArchive<File>) -> Result<BTreeMap<String, Base64OrRaw>> {
    let names: Vec<String> = zip
        .file_names()
        .filter(|n| [".png", ".jpg", ".jpeg"].iter().any(|ext| n.to_lowercase().ends_with(ext)))
        .map(str::to_string)
        .collect();
    names.into_iter().map(|name| Ok((name.clone(), Base64OrRaw::Raw(read_entry(zip, &name)?)))).collect()
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
