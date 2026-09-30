# html2pdf

Converts a `pages.zip` produced by `pdf-ocr-bench run` into a small, searchable PDF using
[printpdf](https://github.com/fschutt/printpdf)'s HTML renderer. Each `page_NNN.html` becomes one
page with the OCR text in black at its original position. There are no images, only text and one
subset font, so a page costs a few KB.

```sh
cargo run --release -- results/tesseract/pages.zip -o final.pdf
```

| Option | |
|---|---|
| `-o, --output PATH` | output PDF (default: `<input>.pdf`) |
| `--font NAME=PATH` | register a TTF/OTF under `NAME` (repeatable), e.g. a CJK font |
| `--title TEXT` | document title |
| `-v, --verbose` | print printpdf warnings |

## How it works

1. Read `metadata.json`, the zip's manifest. It lists every page with its HTML file and size in
   pt. A zip without it is rejected.
2. Render each page with `PdfDocument::from_html_with_cache`, using zero margins, the page size from
   the manifest, and one font pool shared by all pages. A page whose content overflows onto a
   second PDF page is an error.
3. Merge the single-page documents with `PdfDocument::append_document` and save once.

The pages are set in Helvetica. printpdf embeds its own Helvetica, and the word sizes computed
by pdf-ocr-bench use the same glyph widths, so words land on their OCR boxes. Characters outside
Windows-1252 (Cyrillic, CJK, …) fall back to a system font that has them.

Open a page in a browser and press `d` to outline the word boxes.
