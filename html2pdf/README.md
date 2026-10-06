# html2pdf

Converts a `pages.zip` produced by `pdf-ocr-bench run` into a small, searchable PDF using
[printpdf](https://github.com/fschutt/printpdf)'s HTML renderer. Each `page_NNN.html` becomes one
page with the OCR text in black at its original position, and the pictures its HTML shows
(`<img>`, PNG or JPEG from the zip). The text is set in one subset font, so a page of text costs
a few KB; the pictures are dithered to black and white, see [Pictures](#pictures).

```sh
cargo run --release -- results/tesseract/pages.zip -o final.pdf
```

| Option | |
|---|---|
| `-o, --output PATH` | output PDF (default: `<input>.pdf`) |
| `--font NAME=PATH` | register a TTF/OTF under `NAME` (repeatable), e.g. a CJK font |
| `--title TEXT` | document title |
| `--layout positioned\|flow` | every word at its OCR position (default), or text blocks rebuilt into columns and paragraphs; see [Flow layout](#flow-layout) |
| `--long-s keep\|s\|repair` | the long s (ſ) of old print: keep it (default), turn it into `s`, or also repair `f` read for it; see [Long s](#long-s) |
| `--dict PATH` | word list for `--long-s repair` and for joining hyphenated words (default `/usr/share/dict/words`) |
| `--grey-pictures` | keep the pictures' greys (default: dithered to black and white, one bit per pixel); see [Pictures](#pictures) |
| `--html-out DIR` | also write the HTML each page is rendered from (open it in a browser) |
| `-v, --verbose` | print printpdf warnings |

## How it works

1. Read `metadata.json`, the zip's manifest. It lists every page with its HTML file and size in
   pt. A zip without it is rejected.
2. Render each page with `PdfDocument::from_html_with_cache`, using zero margins, the page size from
   the manifest, and one font pool shared by all pages. A page whose content overflows onto a
   second PDF page is an error.
3. Merge the single-page documents with `PdfDocument::append_document` and save once.
4. Encode the saved PDF's pictures again with `printpdf::optimize_images`; see
   [Pictures](#pictures).

The pages are set in Helvetica. printpdf embeds its own Helvetica, and the word sizes computed
by pdf-ocr-bench use the same glyph widths, so words land on their OCR boxes. Characters outside
Windows-1252 (Cyrillic, CJK, …) fall back to a system font that has them.

Open a page in a browser and press `d` to outline the word boxes.

## Pictures

The books' pictures are black-and-white prints (copper-plate engravings, woodcuts, initials),
scanned as eight-bit greyscale. After saving, `printpdf::optimize_images` decodes each picture of
the PDF, dithers it to black and white (Floyd–Steinberg) and writes it with one bit per pixel,
under the same object number, so pages, fonts and text stay byte for byte as they were. A picture
that would not come out smaller is kept as it is. For the 1053-page Calmet volume 1 this takes
the 55 engravings from 11.3 MB to 5.6 MB and the PDF from 24.4 MB to 18.7 MB, in about 6 s.

`--grey-pictures` keeps the greys (for photographs): the pictures are then only compressed again,
without loss.

## Long s

Print before ~1800 sets a long s (ſ) everywhere except at the end of a word: "ſhould", "Moſes".
Tesseract's `enm` model reads it as `ſ`; the other engines read it as `f` ("fhould", "Mofes"),
in about 6% of all words of an 18th-century English text.

* `--long-s s` replaces `ſ` (and the `ﬅ` ligature) with `s`.
* `--long-s repair` also changes a word containing `f` that is not in the word list into the first
  variant with one to three of its `f` (never the last letter: a long s never ends a word) turned
  into `s` that is: "hiftorical" → "historical", "Addreffes" → "Addresses", "Mofes's" → "Moses's".
  Regular inflections count (-s, -es, -ed, -d, -ing, -ly, 's, 'd). A few f-forms that are words
  but rare in old prose are always changed: fo, fame, fent, fet, fide(s), fin(s), fon(s), fun,
  fum, fee(s), fays, faying. Pairs where both are common (faith/saith, fold/sold, fight/sight) are left alone,
  and so are capitals (a capital S is never long). British -our spellings count as words though an
  American list has them as -or ("favour" stays, not "savour").

It works on the OCR word spans and on the text blocks (`<div class="region">`) that
`pdf-ocr-bench reconstruct` writes.

* `--long-s careful` is for text a reader already corrected (`reconstruct` uses it): no fixed
  pairs ("fame" may be fame), and an f before a vowel only in a lower-case word of five letters or
  more that is not set in italic. On vol. 1 of the Calmet that keeps "fide", "fuit", "fol.",
  "Rufin." and "fewer", and still repairs "againft", "fhall", "laft", "Hiftory".

The run prints how many words were changed, with examples.

## Flow layout

`--layout flow` turns each page into flowing text instead of positioned words, so that the PDF's
reading order and phrase search work across lines and columns:

1. **Text blocks** by a recursive XY-cut on the word boxes: the page is cut where a white band
   at least 0.6 em high crosses it (between a running head, the text and footnotes), else where a
   white band runs down through every line (a column gutter, the margin beside marginal notes;
   down to 0.12 em wide in blocks of 10+ lines, 1 em in blocks of one or two lines). Bands come out
   top to bottom and columns left to right, which is the reading order. Where the engine reports
   text blocks (`data-block`, Tesseract), a cut region is also split by block.
2. **Lines**: the engine's own lines (`data-line`) where every word has one, else words whose
   centers lie within half a font size.
3. **Paragraphs**: a new one where the engine's paragraph (`data-par`) changes, where a line is
   indented and the one before is not, after a gap of 1.6 line pitches, or (without `data-par`)
   after a short line ending a sentence.
4. A word split by a line-end hyphen is joined ("du-" + "ring" → "during"); the hyphen stays only
   if both halves are words and the joined form is not ("Ear-rings").
5. Each block is set as `<p>` paragraphs in a box at its place on the page, with the font size
   and line pitch of its words (made smaller if the reflowed text would not fit).

Each block is its own box rather than one `column-count: 2` container: the columns then break
where the scan's do, and marginal notes, running heads and footnotes stay in place. (printpdf
also applies `column-count` to each paragraph separately: fschutt/azul#481.)
