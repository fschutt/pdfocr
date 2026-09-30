# html2pdf

Converts a `pages.zip` produced by `pdf-ocr-bench run` into a searchable PDF using
[printpdf](https://github.com/fschutt/printpdf)'s HTML renderer: each `page_NNN.html` becomes one
page with the scan as background image and the OCR words as invisible, selectable text on top.

```sh
cargo run --release -- results/tesseract/pages.zip -o final.pdf
```

| Option | |
|---|---|
| `-o, --output PATH` | output PDF (default: `<input>.pdf`) |
| `--font NAME=PATH` | register a TTF/OTF under `NAME` (repeatable), e.g. a CJK font |
| `--visible-text` | draw the OCR text in red instead of invisible, to check alignment |
| `--title TEXT` | document title |
| `-v, --verbose` | print printpdf warnings |

## How it works

1. Reads every `page_NNN.html` from the zip in page order, plus the images and `metadata.json`.
2. Page size comes from `<meta name="pdf.options.pageWidth/pageHeight">` (mm), falling back to
   `metadata.json`, then the `.page { width/height: …pt }` rule, then A4.
3. Each page is rendered with `PdfDocument::from_html_with_cache` (zero margins, one shared font
   pool), handing over only the images that page references.
4. Single-page documents are merged with `PdfDocument::append_document` and saved once.

## printpdf requirement

Invisible OCR text relies on printpdf honouring `color: transparent`. printpdf ≤ 0.12.8 drops the
alpha channel of text colors, so the text layer renders as **opaque black** over the scan. The fix
(text render mode 3 for alpha 0, an ExtGState for partial alpha) is on printpdf branch
`claude/nice-bardeen-hjit6y`. Until it is released, build against a local checkout:

```toml
# html2pdf/.cargo/config.toml (git-ignored)
[patch.crates-io]
printpdf = { path = "../../printpdf" }
```

## Known azul-css limitation

azul-css ≤ 0.0.16 applies a rule with an unknown pseudo-element to its base selector, so
`.word::selection { color: #000 }` would make every word black. The page template therefore injects
its browser-only styles (`::selection`, debug mode) from `<script>`, which printpdf ignores.
