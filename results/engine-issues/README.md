# azul / printpdf issues found rendering Calmet vol. 1 (1053 pages)

## Status on azul fix/input-bugs-2026-09-19 02cc380e4 + printpdf perf/optimize-images cd845ff (2026-10-08)

- 1 sup/sub: FIXED. "a" (<sup>) and "b" (vertical-align: super) at baseline 44.2 on a line at 50.3,
  "2" (<sub>) at 54.3. A raised inline now grows its line box, as CSS 2.1 says (+4.6 pt on a 14 pt
  line); pdfocr sets `sup, sub { line-height: 0 }`, print's marks do not move the lines either.
- 5 justify + letter-spacing: still open (-0.02em: lines end at 316-327 of 330).

## Status on azul b515c112e + printpdf fix/html-images-decoded-per-page f5e7444 (2026-10-05)

- 2 relative inline: FIXED (page 2: "a" at y 37.8, baseline 45.0 -> 7.2 pt up).
- 3 hyphens/lang: FIXED, with `lang` on the block and with `lang` only on `<html>` (page 4).
- 4 images: FIXED. 60 pages with all 645 pictures of the volume in the map, no filter: 10.4 s
  (0.17 s a page; was 1.26 s).
- 1 sup/sub: NOT fixed in the PDF. The line box grows (baseline 45.0 -> 50.3) but every run of
  the line is drawn at the same baseline: the content stream has `1 0 0 1 x 249.6672 Tm` for
  "Cock" (16 pt), "a" (<sup>, 13.28 pt), "b" (vertical-align: super, 9.6 pt) and "2" (<sub>,
  13.28 pt). Relative offsets do reach the PDF now, so the bridge draws the positions it gets:
  the glyph positions of sup/sub clusters (not only the line height) need the shift.


azul `fix/input-bugs-2026-09-19` at 0836ecb96, printpdf `azul-codegen-api` at 84dce8c.
Repro: `html2pdf repro.zip -o repro.pdf` (one page per issue 1-3); span positions via PyMuPDF.

## 1. azul: `<sup>`, `<sub>`, `vertical-align: super|sub` shrink the text but do not move it

    <p>Cock<sup>a</sup> and Job<span style="vertical-align: super; font-size: 0.6em">b</span> and x<sub>2</sub></p>

Every span is drawn on the line's baseline (y 45.0 for all of them); only the size changes
(13.3 / 9.6 pt). Expected: "a", "b" raised, "2" lowered.

Likely root cause: `solver3/getters.rs:3747` sets a run's `vertical_align` from
`get_vertical_align_for_node(styled_dom, dom_id, ..)` for the node whose text is laid out, i.e.
the text node inside `<sup>`/`<span>`. `vertical-align` is not inherited, so the text node's
own value is always `baseline`; the value is on the parent inline element. (`font-size` is
inherited, which is why the size does change.) `text3/cache.rs:12655` (the Super arm) is never
reached for text. Check: the run's style should take `vertical-align` from the nearest inline
ancestor up to the IFC root (each nested inline adds its own shift).

## 2. azul: `position: relative; top/left` on an inline span does not move its text

    <p>mentioned<span style="position: relative; top: -0.45em">a</span> in Job</p>

"a" stays on the baseline. CSS 2 §9.4.3: a relatively positioned inline box is offset after
the line is laid out, with its text.

## 3. azul: `hyphens: auto` needs `-azul-hyphenation-language`; the `lang` attribute is ignored

    <div lang="en" style="width: 60pt; hyphens: auto">Advertisement</div>                  -> not hyphenated, overflows
    <div style="width: 60pt; hyphens: auto; -azul-hyphenation-language: en">Advertisement</div> -> "Adver-" / "tisement"

CSS Text 3 §5.4: hyphenation uses the content language, from `lang`/`xml:lang`. The
language comes only from the azul property (`solver3/fc.rs` around 5914,
`DOM_HAS_HYPHENATION_LANGUAGE`). Fix: fall back to the nearest `lang` attribute.

## 4. printpdf: every image in the map is decoded for every page

`PdfDocument::from_html_with_cache(html, images, ..)` copies every entry of `images`
(`lib.rs:463`) and `html::bridge::resolve_html_images` (`bridge.rs:108`) decodes every one of
them, whether the page's HTML uses it or not. Rendering a 1053-page book page by page with the
book's 57 pictures in the map took 22 minutes (1.26 s a page); passing each page only the
images its `<img src>` names: 3.5 minutes (0.24 s a page). Fix: decode lazily, only the
`src` the DOM references (or cache decoded images across calls in the font-pool-like cache).

## Fixed on the branch (verified)

- text-indent narrows the first line (b5728f613); column-count (azul#481); printpdf #289
  text merging. On vol. 1 pp. 1-20 the layout report is clean.


## 5. azul: `text-align: justify` ignores `letter-spacing` when it fills the line (found 2026-10-06)

Repro: `justify_letter_spacing.html` (three copies of one paragraph, 300 pt wide, 14 pt Times,
`text-align: justify`, letter-spacing 0 / -0.02em / +0.05em). Every line but the last should end
at the box's right edge (x 330):

    letter-spacing +0.00em: lines end at [330, 330, 330, 330]
    letter-spacing -0.02em: lines end at [316, 326, 325, 327]        (short by ~letter-spacing x chars)
    letter-spacing +0.05em: lines end at [356, 354, 357, 360, 359]   (past the box)

The line breaks do take the spacing into account (-0.02em puts more words on a line), but the
justification's extra space is computed from the line's width without it, and the spacing is
added when the glyphs are positioned. Look at `text3/cache.rs` `position_one_line`
(`remaining_space = segment.width - effective_segment_width`) and where `letter_spacing` is
added to the pen: the measured width of the line must include it.
(pdfocr does not use letter-spacing on justified text, so no workaround is in place.)
