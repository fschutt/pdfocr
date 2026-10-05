# pdf-ocr-bench

Run several OCR engines on a scanned (image-only) PDF and compare them. For **each engine** the
result is a small `.zip` of HTML pages: every recognized word in black, absolutely positioned where
it was on the scanned page. A separate Rust tool, [`html2pdf`](html2pdf/), turns the zip you like
best into the final PDF. That PDF holds only the text, with no scan, so it stays small and fully
searchable. It is meant for archiving and search, not for reproducing the scan.

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
  ├─► macOS Vision      ─► results/macos_vision/pages.zip   (macOS only)
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

* every engine's result can be opened in a browser (press **`d`** to outline the word boxes),
* the text is plain UTF-8, so there are no encoding problems,
* engines can be compared side by side,
* `html2pdf` controls font embedding when the PDF is finally written.

## Run locally

From a fresh clone, one step after another (macOS with Homebrew, or Debian/Ubuntu):

```sh
git clone https://github.com/fschutt/pdfocr && cd pdfocr
make setup OCR_LANG=deu+eng      # system packages, .venv with every engine, html2pdf
make test                        # unit tests; on a Mac also the macOS Vision tests
make check PDF=scan.pdf OCR_LANG=deu+eng                 # validate + show which model each engine uses
make ocr   PDF=scan.pdf OCR_LANG=deu+eng PREPROCESS=grayscale,deskew,denoise \
           OPTIONS="macos_vision.level=accurate tesseract.psm=6"
make pdfs                        # results/<engine>.pdf for every engine
make all   PDF=scan.pdf OCR_LANG=deu+eng                 # all of the above in one go
```

`make ocr` also takes `ENGINES=`, `DPI=`, `PAGES=` and `OUT=`. The variable is `OCR_LANG`
because `LANG` is the locale. `make setup` runs `scripts/install_system_deps.sh`: on macOS,
Homebrew's Python 3.12, Tesseract with all language models, ocrmypdf's tools, llama.cpp (Surya)
and Rust; on Debian/Ubuntu, the apt packages for the requested languages, plus a hint for Rust
>= 1.88 (rustup) and llama.cpp. On a Mac, `macos_vision` runs as part of `ENGINES=all`.

`EXTRAS=` picks the engines the venv gets (default `ci,test`: every CPU engine, several GB of
PyTorch and PaddlePaddle). To try only macOS Vision, with Tesseract to compare against:

```sh
make setup EXTRAS=macos-vision,tesseract,test OCR_LANG=deu+eng
make test
make ocr PDF=scan.pdf OCR_LANG=deu+eng ENGINES=macos_vision,tesseract OPTIONS="macos_vision.level=accurate"
make pdfs
```

Every engine's model and `-O` parameters are listed by `.venv/bin/pdf-ocr-bench engines` and at
the end of `.venv/bin/pdf-ocr-bench run --help`; see [Engine parameters](#engine-parameters).

## Install

```sh
pip install -e ".[ci]"         # all CPU engines (+ macOS Vision on macOS)
pip install -e ".[tesseract,rapidocr,ocrmypdf]"   # or pick engines
pip install -e ".[all]"        # + olmOCR
```

System packages: `tesseract-ocr` plus the model packages for your languages
(`pdf-ocr-bench tesseract-packages --lang deu+eng` prints them), and `ghostscript`, `qpdf`,
`unpaper` and `pngquant` for ocrmypdf. Surya 2 needs a `llama-server` binary from
[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) on `PATH` for CPU, or vLLM for GPU.
The `macos_vision` engine needs macOS.

## Inputs

```sh
pdf-ocr-bench run INPUT_PDF [OPTIONS]
```

| Option | Default | Meaning |
|---|---|---|
| `INPUT_PDF` | | The scanned (image-only) PDF |
| `--lang` | `eng` | Document language(s): Tesseract codes joined with `+`, **main language first** (e.g. `deu+eng`). See [Languages](#languages). |
| `-e, --engines` | `all` | `all`, or a comma list of `tesseract, rapidocr, paddleocr, easyocr, doctr, surya, ocrmypdf, ocrmypdf_rapid, macos_vision, olmocr` |
| `--preprocess` | none | Image filters applied in order before OCR, e.g. `grayscale,deskew,denoise`. See [Preprocessing](#preprocessing). |
| `-O, --engine-option` | none | Engine parameter `ENGINE.KEY=VALUE`, repeatable. See [Engine parameters](#engine-parameters). |
| `--dpi` | `300` | Resolution the pages are rendered at for the engines (72–1200). 150 is a quick draft; 400–600 helps with small print. For 1-bit (black/white) scans, render a bit *below* the scan's own resolution: pdfium then smooths the jagged edges, which macOS Vision in particular reads much better. |
| `--pages` | all | Page range, 1-based and inclusive: `3`, `1-5`, `1,3,7-10` |
| `--timeout-per-page` | `300` | Seconds per page and engine before the page is skipped (`0` = no limit) |
| `--include-gpu-engines` | off | With `--engines all`, also run GPU engines (olmOCR) |
| `-o, --output-dir` | `results` | Where images, zips and the report go |
| `--report` | `OUTPUT_DIR/report.json` | Report path |
| `-v, --verbose` | off | Debug logging |

**All input is validated before anything is rendered**, and bad input is rejected with a message
saying what to change:

* an unknown language code, filter or engine, or a malformed page range;
* a page range outside the document;
* an engine **named in `--engines`** that cannot read the requested languages (for example
  `--engines doctr --lang chi_sim`: docTR only has Latin-script models), or that cannot run on
  this machine: its Python package is not installed (the message names the `pip install` extra),
  a Tesseract language pack is missing, Surya has no `llama-server`, or macOS Vision is not on
  macOS.

With `--engines all`, engines that cannot read the languages or cannot run here are skipped
instead. The log and `report.json` say why. It is still an error if no engine at all can run.

Helper commands:

```sh
pdf-ocr-bench check --lang deu+eng --engines all --preprocess grayscale,denoise   # validate + show routing
pdf-ocr-bench check --lang deu_frak --installed      # ... also against this machine (packages, models)
pdf-ocr-bench languages                               # every --lang code and the model each engine uses
pdf-ocr-bench filters                                 # the --preprocess filters
pdf-ocr-bench tesseract-packages --lang deu_frak+eng  # apt packages for the Tesseract models
pdf-ocr-bench engines                                 # every engine: its model and its -O parameters
pdf-ocr-bench warmup --lang deu                       # load every routed engine once (downloads models)
```

Every run starts by logging how each engine was routed:

```
[Pipeline] Languages: deu (German), eng (English)
[Pipeline] Preprocessing: grayscale -> denoise
[Pipeline] Engine routing:
[Pipeline]   tesseract       deu+eng
[Pipeline]   rapidocr        latin (PP-OCRv5 mobile)
[Pipeline]   paddleocr       de (PP-OCRv6 multilingual)
[Pipeline]   easyocr         de, en
[Pipeline]   doctr           multilingual PARSeq (HF hub)
[Pipeline]   ocrmypdf        deu+eng
[Pipeline]   ocrmypdf_rapid  -l deu: latin (PP-OCRv5 mobile)
[Pipeline]   surya           skipped: llama.cpp's llama-server is not installed: brew install llama.cpp, ...
[Pipeline]   macos_vision    skipped: macOS only (Apple Vision framework)
```

### Languages

`--lang` takes Tesseract codes. Each engine is routed to the model that covers **all** requested
languages:

| Code | Language | Tesseract | RapidOCR | PaddleOCR | EasyOCR | docTR | macOS Vision |
|---|---|---|---|---|---|---|---|
| `eng` | English | eng | PP-OCRv6 small | PP-OCRv6 multilingual | en | built-in | en-US |
| `enm` | English, historical (long s) | enm | PP-OCRv6 small | PP-OCRv6 multilingual | en | built-in | en-US |
| `deu` | German | deu | latin PP-OCRv5 | PP-OCRv6 multilingual | de | multilingual PARSeq | de-DE |
| `deu_frak` | German (Fraktur) | frk → Fraktur | latin PP-OCRv5 | PP-OCRv6 multilingual | de | multilingual PARSeq | de-DE |
| `frk` | Fraktur | frk → Fraktur | latin PP-OCRv5 | PP-OCRv6 multilingual | de | multilingual PARSeq | de-DE |
| `fra` | French | fra | latin PP-OCRv5 | PP-OCRv6 multilingual | fr | built-in | fr-FR |
| `spa` `ita` `por` `nld` `pol` | Spanish, Italian, Portuguese, Dutch, Polish | same code | latin PP-OCRv5 | PP-OCRv6 multilingual | es it pt nl pl | multilingual PARSeq | es-ES it-IT pt-BR nl-NL pl-PL |
| `lat` | Latin | lat | latin PP-OCRv5 | PP-OCRv6 multilingual | la | multilingual PARSeq | – |
| `rus` `ukr` | Russian, Ukrainian | same code | eslav PP-OCRv5 | eslav PP-OCRv5 | ru uk | – | ru-RU uk-UA |
| `ara` | Arabic | ara | arabic PP-OCRv5 | arabic PP-OCRv5 | ar | – | ar-SA |
| `chi_sim` `chi_tra` | Chinese (Simplified, Traditional) | same code | PP-OCRv6 small | PP-OCRv6 multilingual | ch_sim ch_tra | – | zh-Hans zh-Hant |
| `jpn` | Japanese | jpn | PP-OCRv6 small | PP-OCRv6 multilingual | ja | – | ja-JP |
| `kor` | Korean | kor | korean PP-OCRv5 | korean PP-OCRv5 | ko | – | ko-KR |

Surya and olmOCR are vision-language models and take no language setting. ocrmypdf uses the
Tesseract models, and `ocrmypdf_rapid` uses RapidOCR's recognizers through its plugin.

Combining languages (`deu+eng`, `chi_sim+eng`):

* **Tesseract / ocrmypdf** load every model (`-l deu+eng`).
* **RapidOCR, PaddleOCR, ocrmypdf_rapid** have one recognizer per script family, and each also
  reads English. All non-English languages must share one recognizer: `deu+fra+eng` works,
  `rus+deu` does not.
* **EasyOCR** mixes Latin languages freely. A Cyrillic or Arabic language only combines with its
  own script and English; Chinese, Japanese and Korean only combine with English.
* **docTR** reads Latin-script languages only. English and French use the built-in model; other
  languages need the multilingual PARSeq model from the Hugging Face hub, which knows ä, ö, ß.
* **macOS Vision** takes any list of the languages it supports.

Put the main language first: engines that pick a single model use the first non-English language.
For German documents use `deu+eng`, not `eng+deu`. With `eng+deu`, RapidOCR still gets the Latin
model (German is covered), but the page `lang` attribute becomes `en`.

Fraktur: current Tesseract data has no `deu_frak` model. `deu_frak` and `frk` use `frk` (German
Fraktur) or the `Fraktur` script model, whichever is installed (`tesseract-ocr-frk`,
`tesseract-ocr-script-frak`). The other engines have no Fraktur model and read blackletter with
their German/Latin model; comparing them is what this tool is for.

Historical English (16th–18th century print): use `enm`. Tesseract's Middle English model knows
the long s and keeps it (`ſhould`, `Addreſſes`), where `eng` reads it as f (`fhould`,
`Addrefles`). The other engines have no long-s model and read `enm` with their English model.

### Preprocessing

`--preprocess` applies filters to the page renders once, in the given order, before any engine
runs, so every engine reads the same input:

| Filter | Effect |
|---|---|
| `grayscale` | convert to 8-bit gray |
| `autocontrast` | stretch contrast so ink is black and paper white (1% outliers ignored) |
| `denoise` | 3×3 median filter; removes scanner speckle smaller than a stroke |
| `sharpen` | unsharp mask for crisper stroke edges |
| `binarize` | Otsu threshold to pure black and white |
| `deskew` | straighten text lines (projection-profile search up to ±5°) |

A good start for noisy scans is `grayscale,deskew,denoise`. On `sample.pdf` it removes all of
Tesseract's speckle "words" and fixes `(a, 6` to `(ä, ö`. The trade-off: the median filter can erase
the dots over small capitals (`Ä` becomes `A`), so raise `--dpi` for small print. Use `binarize`
with care: Tesseract often likes it, while the neural engines usually do better on gray.

### Engine parameters

Each engine has a few parameters, set with `-O ENGINE.KEY=VALUE` (repeatable) and validated
before the run like everything else. An unknown engine or key, a value outside its range, or an
engine that is not selected is rejected. `pdf-ocr-bench engines` and the end of
`pdf-ocr-bench run --help` list them, generated from the code:

| Engine | Model | Parameters (default) |
|---|---|---|
| `tesseract` | Tesseract 5 LSTM models, one per `--lang` code | `psm=3`: page segmentation mode (1, 3–13; 3 automatic, 4 one column, 6 one block, 11 sparse text) |
| `rapidocr` | PP-OCR on ONNX Runtime (PP-OCRv6 small for en/zh/ja, PP-OCRv5 mobile otherwise) | `min_score=0.5` (0–1): drop lines recognized below it; `text_orientation=true`: turn upside-down lines |
| `paddleocr` | PaddleOCR 3.x (PP-OCRv6 medium multilingual, or the PP-OCRv5 model of the script) | `min_score=0.0` (0–1); `textline_orientation=true`; `det_max_side=0`: shrink the page to this many px for text detection only (0 = full size, up to 4000) |
| `easyocr` | CRAFT detector + one CRNN recognizer per script group | `decoder=greedy` (greedy, beamsearch, wordbeamsearch) |
| `doctr` | detector + CRNN (en/fr) or multilingual PARSeq | `det_arch=fast_base` (fast_*, db_*, linknet_*); `straight_pages=true` |
| `surya` | Surya 2 VLM via llama.cpp / vLLM | none |
| `ocrmypdf` | ocrmypdf with Tesseract | `psm=3` (as Tesseract) |
| `ocrmypdf_rapid` | ocrmypdf with the RapidOCR plugin | none |
| `macos_vision` | Apple Vision `VNRecognizeTextRequest` (Live Text) | `level=accurate` (accurate, fast); `language_correction=true`; `min_text_height=0.0` (0–1 of the page height) |
| `olmocr` | olmOCR 7B VLM | `model=allenai/olmOCR-7B-0725` (HF id or path, e.g. the FP8 variant); `server=` (URL of a running vLLM, empty = spawn one) |

The report records every engine's effective parameters next to its model.

## Rebuilding a page: layout, text and structure

`pdf-ocr-bench reconstruct` rebuilds scanned pages as structured HTML, as close to the printed
page as it can: columns, paragraphs, marginal notes beside their lines, drop capitals, running
heads, footnotes, and the pictures cut out of the scan.

```sh
pdf-ocr-bench reconstruct scan.pdf -o out --pages 1-20 --lang enm \
    --semantic-context "An English translation of Calmet's Dictionary of the Holy Bible, printed 1732"
# -> out/pages.zip, and out/out.pdf (html2pdf, built under html2pdf/target/release)
```

1. **Render** each page at the resolution of its scan, and at 3/4 of it for Vision (scaling a
   black-and-white scan down smooths its edges, which Vision reads much better).
2. **Layout from the pixels** (`page_layout.py`, OpenCV): the running head, text columns,
   marginal-note strips, footnotes, other text blocks (titles, captions), pictures and drop
   capitals. Columns are cut at gutters, the x ranges only a few words cross; the lines that do
   cross one (a running head, a centred title, a footnote) become full-width blocks first.
3. **Text**: macOS Vision reads the lines (on the device); every word goes to its zone. Vision
   sometimes skips a whole line (an italic line under a handwritten mark, two lines between title
   lines); inked words no line covers are read again from a strip cut around them.
4. **Pictures** are cut out of the scan as PNGs and placed as images.
5. **Structure**: an LLM (`claude -p`, `--model sonnet`, `--agents` pages at once) gets the zones
   with their OCR lines and the scan (reduced, and as 8 full-resolution tiles), plus
   `--semantic-context`. It answers which lines make a paragraph, a note or a heading line, the
   corrected text (long s, misread letters), and drop capitals. `--no-llm` uses a heuristic
   from the line geometry instead (nothing is uploaded). `--zoom` lets the model crop the scan.
6. **HTML**: every paragraph at the place of its first line, as wide as its column (beside the
   margin notes in it), justified and hyphenated, in Times at one size per column (the size at
   which it wraps to its original number of lines, at the original line pitch); notes at one
   size beside the line they annotate; a drop capital as a letter with the lines beside it
   narrowed. Then a render-and-measure loop (`html2pdf --layout-report`, only the pages that
   changed) sets smaller whatever still runs into the block below or past its box. The PDF is
   rendered with `--long-s repair` for English.

Steps 1-4 run in `--workers` processes (4: about 1.1 s a page on an M-series Mac); each page goes
to the model as soon as it is read, `--agents` (8) at a time. Sonnet answers a page in about
20 s, so the model sets the pace: about 2.6 s a page, a 1000-page volume in about 45 minutes.
Haiku is no faster here: it thinks some 20k tokens about a page (minutes), and with
`--no-thinking` it modernises spellings and drops words. `--learn-from DIR` adds an earlier
run's most frequent corrections (OCR words that are no words) and one of its pages, worked
through, to the model's instructions.

`out/work/page_NNN/` keeps each page's renders, `layout.png` (the zones drawn on the page),
`layout.json`, `lines.json` and the LLM's prompt and answer; an answer is reused as long as the
page's zones and lines (and the model and instructions) are unchanged, so a run that stopped is
continued by running it again. When the subscription's usage limit is hit, no further pages are
sent; they get the heuristic structure until the next run. A page is one request of about
0.5 MB of images, sent twice (the answer, and the structured-output turn): about 1 MB.

## Correcting with an LLM

`scripts/llm_correct.py` proofreads a result page by page with Claude (`claude -p`, so a Claude
subscription works): spelling (the long s read as f, misread italics), OCR junk from pictures
and specks, and marginal notes run into the text.

```sh
scripts/llm_correct.py results/paddleocr/pages.zip --images results/images \
    -o results/paddleocr_haiku.zip --pdf results/paddleocr_haiku.pdf [--agents 20] [--model haiku] [--pages 1-20]
```

The pages are first rebuilt as text blocks (`html2pdf --layout flow --long-s repair`). Every page
then goes to its own `claude -p`, up to `--agents` at once. The model gets the blocks as JSON, the
page reduced for the layout, and the scan cut into 8 overlapping full-resolution tiles (1-bit
PNG for black-and-white scans). For a word it still cannot read it may crop and zoom with
ImageMagick. It returns every block corrected, with a role: `text`, `note`, `header`,
`footnote` or `noise` (dropped). The result is a new zip that `html2pdf` renders as it is, plus
`<output>_work/page_NNN/` with everything the model saw and answered. A page whose call fails
keeps the uncorrected layout.

The scan goes over the network: about 0.4 MB of images per page, re-sent on every turn of
the conversation (usually 4–15 turns; `--no-zoom` makes it one). A page that runs out of turns
is asked again without tools. Answers are kept in the work directory and reused by the next
run of the same command (`--fresh` asks again).

## Usage

```sh
pdf-ocr-bench run scan.pdf --lang deu+eng --preprocess grayscale,deskew,denoise
cargo run --release --manifest-path html2pdf/Cargo.toml -- results/rapidocr/pages.zip -o final.pdf
```

Every log line is flushed right away (for live GitHub Actions logs):

```
[Pipeline] Rendering pages at 300 DPI...
[Pipeline] Rendered 2 pages in 0.8s
[Pipeline] === Engine 1/7: Tesseract ===
[Tesseract] Page 1/2: 113 words, 1.2s, avg conf 0.91
[Tesseract] Page 2/2: 65 words, 1.1s, avg conf 0.82
[Pipeline] Tesseract complete: 178 words total, 2.4s
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

`pages.zip` holds only text, a few KB per page:

```
page_001.html   page_002.html   …   metadata.json
```

The page renders in `results/images/` are the OCR input. They are shared by all engines and never
go into a zip.

Each HTML page has fixed `pt` dimensions (the source PDF page size), `<meta name="pdf.options.pageWidth/
pageHeight">` in mm for converters, and one `<span class="word">` per word. The span's `left`,
`width` and `height` are the OCR box as percentages of the page, and it carries `data-confidence`
and the engine's layout as page-unique ids: `data-line` (all engines; Surya and olmOCR spread
their lines evenly), `data-par` (Tesseract) and `data-block` (Tesseract, ocrmypdf, docTR, Surya).
Words of one engine line are set as one line, even when an engine's boxes overlap the next line.
The text is black Helvetica (`Helvetica, Arial, sans-serif`); layout lives in
`html_output/layout.py`:

* **Size**: each word is measured with Helvetica's real glyph widths. PyMuPDF's built-in
  Helvetica has the same letter, digit and umlaut widths as the Helvetica printpdf embeds, and as
  Arial. The size at which
  a word exactly spans its OCR box is computed per word, and the line uses the median, so a line
  has one size and one badly boxed word cannot distort it. A word only gets smaller than its line
  if it would otherwise run into the next word (never below half the line size). As a guard,
  the size is capped at 1.5× the line's tallest box.
* **Position**: words keep their OCR x. Vertically, every span is centered on the median center of
  its line's boxes. That is stable across engines (Tesseract returns tight ink boxes,
  RapidOCR/PaddleOCR padded detection boxes) and across words with or without descenders.
* Every word except the last on a line ends with a space, so copy-paste and PDF text extraction
  keep the word boundaries.

`metadata.json` is the zip's manifest. It holds the engine name, page count, total words, elapsed
time and average confidence (`null` for engines that report no confidence). Per page, it holds the
HTML file name, the page size in px and pt, words, confidence, time, and any skip/error.
`html2pdf` reads page order and sizes from it.

## Engines

| Engine | Word boxes | Notes |
|---|---|---|
| `tesseract` | native | languages resolved against the installed models (`tesseract --list-langs`) |
| `rapidocr` | native (`return_word_box`) | PP-OCRv6 for en/zh/ja, PP-OCRv5 mobile for Latin, East Slavic, Arabic, Korean |
| `paddleocr` | native (`return_word_box`); the recognized line text decides word boundaries | runs with `enable_mkldnn=False` (PaddlePaddle 3.x oneDNN crashes on CPU); slow on CPU, about 1.5 min per dense page at 300 DPI; `-O paddleocr.det_max_side=2048` halves that with nearly the same text |
| `easyocr` | line boxes split by character count | |
| `doctr` | native | multilingual PARSeq from the HF hub for languages beyond English/French |
| `surya` | block boxes, lines spread evenly | Surya 2 is a VLM served by llama.cpp/vLLM; slow on CPU |
| `ocrmypdf` | read back from the PDF text layer (PyMuPDF) | run per page on a PDF built from the shared image; no confidence |
| `ocrmypdf_rapid` | same, with `--plugin ocrmypdf_rapidocr` | one language; PP-OCRv5 recognizers are selected with a generated `--rapidocr-config-path` |
| `macos_vision` | per word (`boundingBoxForRange`) | Apple Vision `VNRecognizeTextRequest`: the recognizer behind Live Text in Preview and Photos, called directly (Preview's VisionKit API only exposes plain text). **macOS only**; `pip install ".[macos-vision]"` (pyobjc) |
| `olmocr` | none, lines spread over the ink area | 7B VLM (`-O olmocr.model=…`, `-O olmocr.server=…` for a running vLLM); minutes per page on CPU, so opt-in |

All boxes are normalized to `BBox(x, y, w, h)` in 0..1 page fractions.

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

`.github/workflows/ocr.yml` is started manually (`workflow_dispatch`). Its inputs match the CLI:

| Input | Default | Meaning |
|---|---|---|
| `pdf_url` | (required) | Public URL of the scanned PDF |
| `lang` | `eng` | as `--lang` |
| `engines` | `all` | as `--engines`; `macos_vision` runs in a separate macOS job |
| `engine_options` | empty | space-separated `ENGINE.KEY=VALUE`, as `-O` |
| `preprocess` | empty | as `--preprocess` |
| `dpi` | `300` | as `--dpi` |
| `page_range` | empty (all) | as `--pages` |
| `timeout_per_page` | `300` | as `--timeout-per-page` |

Jobs:

1. **validate** installs only the base package and runs `pdf-ocr-bench check`, so bad input fails
   in about a minute, before any engine is installed. It also computes which Tesseract packages
   `lang` needs.
2. **ocr** (Ubuntu) installs those packages and the engines, runs every selected engine except
   macOS Vision, and uploads each engine's zip as its own artifact (`ocr-tesseract`,
   `ocr-rapidocr`, …), plus `all-results` and `ocr-report`. It also writes a summary table
   (engine, model, words, confidence, time, status) to the run page.
3. **ocr-macos-vision** (macOS runner) runs macOS Vision when `engines` names `macos_vision`, or
   is `all` and Vision can read `lang`, and uploads `ocr-macos-vision`. Note that macOS runner
   minutes are billed at 10× Linux on private repositories.

The page renders in `results/images/` are left out of the artifacts, since they are only OCR input.
olmOCR is not run in CI.

`.github/workflows/tests.yml` runs `pytest` and the `html2pdf` tests on every push. A macOS job runs
the macOS Vision engine on `sample.pdf` (`tests/test_macos_vision.py`, which `make test` also runs
on a Mac).

## Tests

```sh
make test                                   # or:
pip install -e ".[test]" && pytest
cd html2pdf && cargo test
```

The pipeline tests use fake engines, so they need no OCR models. The Tesseract test is skipped
when `tesseract` is not installed, and `tests/test_macos_vision.py` runs only on macOS with
`.[macos-vision]` installed. `scripts/make_sample_pdf.py` regenerates `sample.pdf`.
