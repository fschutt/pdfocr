# Upstream patches

`html2pdf` needs two upstream fixes so that the OCR text layer is invisible. Each directory holds a
`git format-patch` file for one repository. Apply it with `git am`; both apply cleanly to the base
commit listed.

| Directory | Repository | Base commit | Patch |
|---|---|---|---|
| `printpdf/` | [fschutt/printpdf](https://github.com/fschutt/printpdf) | `6ff6bb5` (0.12.8, "Merge PR #286") | html: honour text color alpha |
| `azul/` | [fschutt/azul](https://github.com/fschutt/azul) | `626aa43c6` ("Merge pull request #475") | css: an invalid selector invalidates its whole rule |

```sh
cd printpdf && git am /path/to/pdfocr/patches/printpdf/*.patch
cd azul     && git am /path/to/pdfocr/patches/azul/*.patch
```

## printpdf: honour text color alpha

**Bug.** The HTML bridge (`src/html/bridge.rs`) built text fill colors from the RGB channels and
ignored alpha. As a result, `color: transparent` painted opaque black glyphs, and
`color: rgba(…, 0.5)` painted fully opaque ones. For pdf-ocr-bench this puts the OCR text layer
over the scan as black text.

**Fix.** Each text run whose color is not opaque gets its own `q … Q` scope:
- alpha 0 uses text rendering mode 3 (invisible). The text stays selectable and extractable, which
  is how OCR text layers work.
- partial alpha loads a fill/stroke-alpha `ExtGState`, as translucent rects already did.

The glyph-run inline `background-color` pass gets the same `ExtGState`. Opaque text emits the same
ops as before.

**Tests.** `tests/html_text_alpha.rs` has 6 tests; 4 of them fail without the fix. The full
`cargo test`, `cargo test --no-default-features` and `cargo check --examples` pass.

**Pre-existing and not changed:** an inline `<span>` background is painted three times (two
identical azul `Rect` items plus the glyph-run background pass). Opaque colors look fine;
translucent ones stack.

## azul: an invalid selector invalidates its whole rule

**Bug.** When `css/src/parser2.rs` hit a selector part it could not parse (an unknown
pseudo-class/-element, an unknown type selector, a malformed attribute selector), it warned and
skipped only that token. It kept the rest of the selector, so the rule applied to a *wider*
selector than written. `.word::selection { color: #000 }` became `.word { color: #000 }`, which
overrode `color: transparent` and made every OCR word black.

**Fix.** Follow Selectors 4 §3.7: an invalid selector makes the whole rule invalid, meaning every
selector in the list plus all nested rules. The `SkippedRule` warning is kept.

**Tests.** `parser2::invalid_selector_tests` has 8 tests; 6 fail without the fix. The full
`cargo test -p azul-css --features parser --lib` has 2891 passing. The one failure,
`css::autotest_generated::css_path_display_and_debug_agree_and_compose`, already fails on the base
commit.

The patch also applies (`patch -p2`, offsets only) to the published `azul-css` 0.0.16 source, which
is what printpdf 0.12.8 builds against.

## Release order

printpdf pins `azul-css`, `azul-core` and `azul-layout` to one exact version. The two repos
therefore release as a pair:
1. release azul with the css fix;
2. bump printpdf's three `azul-*` pins together and release it with the alpha fix;
3. raise `html2pdf/Cargo.toml`'s `printpdf` requirement to that release.

## Verified end to end

Setup: `html2pdf` built against both patched crates. This used a local, git-ignored
`[patch.crates-io]` in `html2pdf/.cargo/config.toml` pointing at the patched printpdf checkout and
at azul-css 0.0.16 with the azul patch applied.

Results on `sample.pdf` with Tesseract, RapidOCR, docTR and ocrmypdf:
- every OCR word is extractable from the PDF (word counts identical);
- every text run uses `3 Tr`;
- the rendered PDF is pixel-identical to the same pages with no text layer;
- with `class="page debug"`, the words render translucent red.
