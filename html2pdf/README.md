# html2pdf

Converts a `pages.zip` produced by `pdf-ocr-bench run` into a searchable PDF using
[printpdf](https://github.com/fschutt/printpdf)'s HTML renderer. Each `page_NNN.html` becomes one
page, with the scan as background image and the OCR words as invisible, selectable text on top.

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

1. Read `metadata.json`, the zip's manifest. It lists every page with its HTML file, image and size
   in pt. A zip without it is rejected.
2. Render each page with `PdfDocument::from_html_with_cache`, using zero margins, the page size from
   the manifest, the page's image, and one font pool shared by all pages. A page whose content
   overflows onto a second PDF page is an error.
3. Merge the single-page documents with `PdfDocument::append_document` and save once.

To check alignment, open a page in a browser and press `d`. That adds the `debug` class, and the
`.page.debug .word` rule shows the text in red. printpdf renders the same rule if the HTML carries
`class="page debug"`.

## printpdf requirement

The text is only invisible with two upstream fixes, both in [`../patches`](../patches/):
- printpdf must honour the alpha of `color: transparent`;
- azul-css must drop invalid rules such as `.word::selection` instead of applying them to `.word`.

With the released printpdf 0.12.8, the text layer renders as black text over the scan.
