# pdf-ocr-bench

Run several OCR engines on a scanned (image-only) PDF and compare them. For **each engine** the
result is a `.zip` of HTML pages: the original page as background image with every recognized word
absolutely positioned on top, as invisible but selectable text. A separate Rust tool,
[`html2pdf`](html2pdf/), turns the zip you like best into the final searchable PDF.

```
input.pdf
  │  rendered once with pypdfium2 → results/images/page_NNN.png (shared by all engines)
  │
  ├─► Tesseract         ─► results/tesseract/pages.zip
  ├─► RapidOCR          ─► results/rapidocr/pages.zip
  ├─► PaddleOCR         ─► results/paddleocr/pages.zip
  ├─► EasyOCR           ─► results/easyocr/pages.zip
  ├─► docTR             ─► results/doctr/pages.zip
  ├─► Surya             ─► results/surya/pages.zip
  ├─► ocrmypdf          ─► results/ocrmypdf/pages.zip
  ├─► ocrmypdf+rapid    ─► results/ocrmypdf_rapid/pages.zip
  ├─► olmOCR            ─► results/olmocr/pages.zip     (opt-in: --include-gpu-engines)
  │
  ▼
  results/report.json — CER/WER/agreement/bbox-IoU matrices, engine ranking
  │
  ▼
  html2pdf results/<best>/pages.zip -o final.pdf
```

## Why HTML as the intermediate format

PDF text layers go wrong in ways that are hard to see: broken `ToUnicode` maps, wrong cmaps,
invisible text that copy-pastes as garbage. Positioned HTML avoids that:

* every engine's result can be opened in a browser (press **`d`** to show the OCR text in red),
* the text is plain UTF-8, so there are no encoding problems,
* engines can be compared side by side,
* `html2pdf` controls font embedding when the PDF is finally written.

## Install

```sh
pip install -e ".[ci]"         # all CPU engines
pip install -e ".[tesseract,rapidocr,ocrmypdf]"   # or pick engines
pip install -e ".[all]"        # + olmOCR
```

System packages: `tesseract-ocr` plus language packs (`tesseract-ocr-deu`, …), and `ghostscript`,
`qpdf`, `unpaper` and `pngquant` for ocrmypdf. Surya 2 needs a `llama-server` binary from
[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) on `PATH` for CPU, or vLLM for GPU.

## Usage

```sh
pdf-ocr-bench run INPUT_PDF [OPTIONS]

  -o, --output-dir PATH       Output directory [default: results]
  -e, --engines TEXT          Comma-separated engines [default: all]
  --lang TEXT                 Document language as Tesseract code [default: eng]
                              e.g. eng, deu, fra, deu_frak, frk, chi_sim, jpn, ara, rus, lat
                              Compound: eng+deu (Tesseract tries both)
  --dpi INT                   Render DPI [default: 300]
  --pages TEXT                Page range, e.g. "1-5" or "1,3,7-10" [default: all]
  --timeout-per-page INT      Seconds before skipping a page for an engine (0 = no limit) [default: 300]
  --include-gpu-engines       Also run GPU-requiring engines (olmOCR)
  --report PATH               Report path [default: OUTPUT_DIR/report.json]
  -v, --verbose               Debug logging

pdf-ocr-bench engines         # list engines
pdf-ocr-bench warmup --lang deu   # load every engine once (downloads models)
```

Example:

```sh
pdf-ocr-bench run sample.pdf --lang eng+deu --dpi 200
cargo run --release --manifest-path html2pdf/Cargo.toml -- results/tesseract/pages.zip -o final.pdf
```

Every log line is flushed right away (for live GitHub Actions logs):

```
[Pipeline] Rendering pages at 200 DPI...
[Pipeline] Rendered 2 pages in 0.5s
[Pipeline] === Engine 1/8: Tesseract ===
[Tesseract] Page 1/2: 100 words, 0.9s, avg conf 0.95
[Tesseract] Page 2/2: 60 words, 0.8s, avg conf 0.94
[Pipeline] Tesseract complete: 160 words total, 1.9s
[Pipeline] Creating zip: results/tesseract/pages.zip
...
[Evaluation] Computing cross-engine CER matrix...
```

followed by a ranking table.

If an engine fails to import or load, it is logged, marked `success: false` in the report, and the
run continues. A page that exceeds `--timeout-per-page` is skipped with a warning; its metadata
entry says `"skipped": true`.

## Output

```
results/
├── images/page_001.png …        shared page renders
├── <engine>/pages.zip
└── report.json
```

`pages.zip`:

```
page_001.html   page_001.png   page_002.html   page_002.png   …   metadata.json
```

Each HTML page has fixed `pt` dimensions (the source PDF page size), `<meta name="pdf.options.pageWidth/
pageHeight">` in mm for converters, and one `<span class="word">` per word. The span's
`left/top/width/height` are percentages of the page, and it carries `data-confidence`. Font size
follows `bbox.h × page_height_pt × 0.8`, using the tallest box on the word's line so a line has
one size. It is then capped so the word fits its box width (otherwise PDF extractors merge
neighbouring words). Words are centered vertically on their line, and every word except the last
on a line ends with a space, so copy-paste keeps word boundaries.

`metadata.json` is the zip's manifest. It holds the engine name, page count, total words, elapsed
time and average confidence (`null` for engines that report no confidence). Per page, it holds the
HTML and image file names, the page size in px and pt, words, confidence, time, and any
skip/error. `html2pdf` reads page order and sizes from it.

## Engines

| Engine | Word boxes | Notes |
|---|---|---|
| `tesseract` | native | `deu_frak` falls back to `frk` / the `Fraktur` script model if not installed |
| `rapidocr` | native (`return_word_box`) | non-ch/en scripts use the PP-OCRv5 mobile recognizers |
| `paddleocr` | native (`return_word_box`, fragments re-joined at spaces) | runs with `enable_mkldnn=False`: PaddlePaddle 3.x oneDNN crashes on CPU |
| `easyocr` | line boxes split by character count | |
| `doctr` | native | non-Latin languages try the multilingual PARSeq model from the HF hub |
| `surya` | block boxes, lines spread evenly | Surya 2 is a VLM served by llama.cpp/vLLM; slow on CPU |
| `ocrmypdf` | read back from the PDF text layer (PyMuPDF) | run per page on a PDF built from the shared image; no confidence |
| `ocrmypdf_rapid` | same, with `--plugin ocrmypdf_rapidocr` | single language only |
| `olmocr` | none, lines spread over the ink area | 7B VLM (`allenai/olmOCR-7B-0725`, `OLMOCR_MODEL` to change, `OLMOCR_SERVER` for a running vLLM); minutes per page on CPU, so opt-in |

All boxes are normalized to `BBox(x, y, w, h)` in 0..1 page fractions.

### Languages

`--lang` takes Tesseract codes. `lang_map.py` translates them for the other engines (`deu` →
PaddleOCR `german`, EasyOCR `de`, RapidOCR `latin`, …). For compound codes like `eng+deu`,
non-Tesseract engines use the first language. Most engines have no Fraktur model and read
blackletter with their German/Latin model. That difference is exactly what the comparison is
meant to show.

## Evaluation

There is no ground truth, so engines are compared with each other. For each pair of engines and
each page both processed:

* **CER / WER** via `jiwer`, averaged over both directions and capped at 1.0 per page,
* **agreement**, the normalized Levenshtein similarity (`rapidfuzz`),
* **bbox IoU**, the mean best-match IoU of word boxes in both directions.

The ranking sorts engines by average CER against all other engines. The engine closest to the
consensus comes first and is reported as `best_engine`. Per-engine words, average confidence and
time are listed next to it.

## GitHub Actions

`.github/workflows/ocr.yml` (manual `workflow_dispatch`) downloads a PDF from a URL, runs every CPU
engine, and uploads each engine's zip as its own artifact (`ocr-tesseract`, `ocr-rapidocr`, …),
plus `all-results` and `ocr-report`. It also writes a summary table to the run page. Inputs:
`pdf_url`, `lang`, `engines`, `dpi`, `page_range`, `timeout_per_page`. olmOCR is excluded.

`.github/workflows/tests.yml` runs `pytest` and the `html2pdf` tests on every push.

## Upstream fixes

The text layer is only invisible with two fixes that are not released yet:

* [fschutt/printpdf#287](https://github.com/fschutt/printpdf/pull/287): HTML text honours the
  alpha of its color. `color: transparent` becomes text render mode 3 (invisible but selectable)
  instead of opaque black.
* [fschutt/azul#480](https://github.com/fschutt/azul/pull/480): an invalid CSS selector drops its
  whole rule. Before, `.word::selection { color: #000 }` was applied to every `.word`.

`html2pdf/Cargo.toml` builds against both PR branches: printpdf as a git dependency, and azul-css
through `[patch.crates-io]`. `Cargo.lock` pins the exact commits. Once both are released, switch
back to crates.io versions.

## Tests

```sh
pip install -e ".[test]" && pytest
cd html2pdf && cargo test
```

The pipeline tests use fake engines, so they need no OCR models. The Tesseract test is skipped
when `tesseract` is not installed. `scripts/make_sample_pdf.py` regenerates `sample.pdf`.
