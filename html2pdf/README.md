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

## Upstream fixes

Invisible text relies on two unreleased fixes:
- [fschutt/printpdf#287](https://github.com/fschutt/printpdf/pull/287): `color: transparent`
  becomes invisible text instead of black.
- [fschutt/azul#480](https://github.com/fschutt/azul/pull/480): `.word::selection` no longer
  applies to `.word`.

`Cargo.toml` builds against both PR branches: printpdf as a git dependency, and azul-css through
`[patch.crates-io]`. `Cargo.lock` pins the exact commits. azul-core and azul-layout stay on the
0.0.16 release, because azul master's azul-layout does not build with printpdf's feature set.
When both fixes are released, go back to the crates.io versions.
