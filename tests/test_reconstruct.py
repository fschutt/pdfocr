"""Page layout from pixels, and the structure → HTML steps of `reconstruct`, on synthetic pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pdf_ocr_bench.llm_structure import to_items  # noqa: E402
from pdf_ocr_bench.page_layout import Box, PageLayout, Zone, analyse  # noqa: E402
from pdf_ocr_bench.reconstruct import Item, Line, build_blocks, blocks_html, fit_size, heuristic_structure, wrap_lines  # noqa: E402

LH = 24  # glyph height of the synthetic pages


def _text_lines(page: np.ndarray, x0: int, x1: int, y0: int, n: int, pitch: int = 40, rng=None) -> None:
    """`n` lines of ragged "words" (black boxes a glyph high) between x0 and x1."""
    rng = rng or np.random.default_rng(1)
    for i in range(n):
        y, x = y0 + i * pitch, x0
        while x < x1 - 40:
            w = int(rng.integers(30, 110))
            w = min(w, x1 - x)
            for gx in range(x, x + w - 6, 14):  # glyphs, a few px apart
                page[y:y + LH, gx:gx + 10] = 0
            x += w + int(rng.integers(12, 20))  # word space, narrower than any gutter


def synthetic_page(tmp_path: Path) -> Path:
    page = np.full((2400, 1600), 255, np.uint8)
    rng = np.random.default_rng(7)
    _text_lines(page, 500, 1100, 60, 1, rng=rng)  # running head, across the gutter
    _text_lines(page, 200, 760, 200, 45, rng=rng)  # column 1
    _text_lines(page, 820, 1380, 200, 45, rng=rng)  # column 2 (60 px gutter)
    for y in (300, 900, 1500):  # marginal notes, 15 px from column 2
        _text_lines(page, 1395, 1560, y, 2, rng=rng)
    page[2050:2350, 300:1300] = 0  # a picture: one big dark shape
    page[2080:2320, 330:1270] = 255
    page[2100:2300, 350:1250:7] = 0  # its hatching
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), page)
    return path


def test_layout_finds_head_columns_notes_and_picture(tmp_path):
    layout = analyse(synthetic_page(tmp_path))
    roles = sorted(z.role for z in layout.zones)
    assert roles.count("column") == 2, layout.zones
    assert "header" in roles and "picture" in roles
    assert [z.side for z in layout.of("notes")] == ["right"]
    col1, col2 = sorted(layout.of("column"), key=lambda z: z.box.x0)
    assert col1.box.x1 <= 790 and col2.box.x0 >= 800  # cut in the gutter
    assert abs(layout.line_height - LH) <= 2


def test_wrap_and_fit_with_real_metrics():
    import pymupdf

    font = pymupdf.Font("tiro")
    text = " ".join(["word"] * 60)
    one = wrap_lines(text, 10_000, 12, font)
    assert one == 1
    narrow = wrap_lines(text, 200, 12, font)
    assert narrow > 5
    size = fit_size(text, 200, 4, 12, font)
    assert size < 12 and wrap_lines(text, 200, size, font) <= 4
    assert fit_size("short", 200, 1, 12, font) == 12


def _lines() -> list[Line]:
    rows = [("L1", "First line of the para-", 100), ("L2", "graph goes on here.", 140), ("L3", "Second paragraph.", 220)]
    return [Line(i, t, Box(100, y, 900, y + 30), [], "column0") for i, t, y in rows]


def _layout() -> PageLayout:
    layout = PageLayout(width=1000, height=1000, line_height=30)
    layout.zones = [Zone("column", Box(100, 100, 900, 300), 0)]
    return layout


def test_heuristic_joins_hyphens_and_splits_at_gaps():
    items = heuristic_structure(_lines(), _layout())
    assert [i.text for i in items] == ["First line of the paragraph goes on here.", "Second paragraph."]
    assert items[0].lines == ["L1", "L2"]


def test_blocks_stop_at_the_next_block_and_render_as_regions():
    items = heuristic_structure(_lines(), _layout())
    blocks = build_blocks(items, _lines(), _layout(), {}, 1000, 1000)
    assert len(blocks) == 2
    assert blocks[0].limit == pytest.approx(blocks[1].y)  # the first paragraph may reach the second
    assert blocks[1].limit == 1000  # the last: the page's end
    page = blocks_html(blocks, 1000, 1000)
    assert page.count('<div class="region"') == 2 and "Times" in page


def test_answer_is_checked_against_the_ocr_lines():
    layout, lines = _layout(), _lines()
    answer = {"items": [
        {"zone": "column0", "kind": "paragraph", "lines": ["L1", "L2", "L9"], "text": "First line of the paragraph goes on here."},
        {"zone": "nowhere", "kind": "paragraph", "lines": ["L3"], "text": "Second paragraph."},
        {"zone": "column0", "kind": "noise", "lines": [], "text": "x"},
    ]}
    items = to_items(answer, lines, layout)
    assert [i.lines for i in items] == [["L1", "L2"], ["L3"]]  # unknown L9 dropped
    assert items[1].zone == "column0"  # an unknown zone: the zone its lines are in
    lost = {"items": [{"zone": "column0", "kind": "paragraph", "lines": ["L3"], "text": "Second paragraph."}]}
    assert to_items(lost, lines, layout) is None  # most of the page's text left out


def test_a_line_listed_for_two_items_belongs_to_the_one_whose_text_has_it():
    layout = _layout()
    lines = [Line("L1", "reigned three years eight months, and", Box(100, 100, 900, 130), [], "column0"),
             Line("L2", "Cleopatra reigned from 3957, and killed", Box(100, 140, 900, 170), [], "column0"),
             Line("L3", "herfelf in 3974. The City of Alexandria", Box(100, 180, 900, 210), [], "column0")]
    answer = {"items": [
        {"zone": "column0", "kind": "paragraph", "lines": ["L1", "L2", "L3"], "text": "reigned three years eight months, and died."},
        {"zone": "column0", "kind": "paragraph", "lines": ["L2", "L3"],
         "text": "Cleopatra reigned from 3957, and killed herself in 3974. The City of Alexandria"},
    ]}
    assert [i.lines for i in to_items(answer, lines, layout)] == [["L1"], ["L2", "L3"]]


def test_a_column_part_ends_at_the_word_the_ocr_read_last_in_the_column():
    from pdf_ocr_bench.reconstruct import column_parts

    first = [Line("A1", "Ptolemy Euergetes or Physcon, reigned", Box(100, 800, 480, 830), [], "column0")]
    # the next column's lines as Vision read them, a word short: by characters the first part
    # would take "fifty three" too
    rest = [Line("B1", "fifty three years, part", Box(520, 100, 900, 130), [], "column1"),
            Line("B2", "his Brother", Box(520, 140, 900, 170), [], "column1")]
    text = "Ptolemy Euergetes or Physcon, reigned fifty three years, part with his Brother Philometer."
    parts = column_parts(Item("column0", text, ["A1", "B1", "B2"]), first + rest, 30)
    assert [p.text for p, _ in parts] == ["Ptolemy Euergetes or Physcon, reigned",
                                          "fifty three years, part with his Brother Philometer."]
    # a word the print divided between the columns goes on to the next part
    hyphen = [Line("A1", "the chief City of Judæa, on the fide of Sama-", Box(100, 800, 480, 830), [], "column0")]
    text = "the chief City of Judæa, on the side of Samaria, near the frontiers of Ephraim."
    parts = column_parts(Item("column0", text, ["A1", "B1"]), hyphen + rest[:1], 30)
    assert parts[0][0].text.endswith("side of") and parts[1][0].text.startswith("Samaria,")


def test_a_note_is_found_by_its_words_that_vision_ran_into_the_text():
    from pdf_ocr_bench.reconstruct import locate

    def words(y, *spec):
        return [(t, Box(x0, y, x1, y + 30)) for t, x0, x1 in spec]
    lines = [Line("L1", "accusers would haveLuk. xxiii.", Box(100, 100, 800, 130),
                  words(100, ("accusers", 100, 300), ("would", 320, 450), ("haveLuk.", 470, 700), ("xxiii.", 720, 800)), "column0"),
             Line("L2", "to our Matth.", Box(100, 300, 900, 330),
                  words(300, ("to", 100, 150), ("our", 170, 250), ("Matth.", 800, 900)), "column0")]
    box, _ = locate("Luk. xxiii. 2.", lines, 30)
    assert box.y0 == 100 and 550 < box.x0 < 700  # its share of "haveLuk.", not the text's "have"
    box, _ = locate("Matth. xxi.16,17.", lines, 30)  # its figures unread: the name alone will do
    assert (box.x0, box.y0) == (800, 300)
    assert locate("Gen. xii. 3.", lines, 30) is None


def test_a_line_given_to_an_item_whose_text_lacks_it_goes_to_the_one_that_has_it():
    layout = _layout()
    lines = [Line("L1", "ALE", Box(400, 50, 500, 80), [], "column0"),
             Line("L2", "Adna however went frequently to the", Box(100, 100, 900, 130), [], "column0"),
             Line("L3", "Cave to visit her Son, and give him Milk", Box(100, 140, 900, 170), [], "column0")]
    answer = {"items": [
        {"zone": "column0", "kind": "heading", "lines": ["L1", "L2"], "text": "ALE"},  # ids a line off
        {"zone": "column0", "kind": "paragraph", "lines": ["L3"],
         "text": "Adna however went frequently to the Cave to visit her Son, and give him Milk."},
    ]}
    assert [i.lines for i in to_items(answer, lines, layout)] == [["L1"], ["L2", "L3"]]


def test_a_paragraph_ends_where_another_items_text_it_repeats_begins():
    layout = _layout()
    lines = [Line("A1", "Coele-Syria is distinguished by no particular name", Box(100, 100, 900, 130), [], "column0"),
             Line("A2", "in Scripture, but is comprized under Aram", Box(100, 140, 900, 170), [], "column0"),
             Line("B1", "reached to Coele-Syria; of which notwithstanding I do not know", Box(520, 100, 900, 130), [], "column0")]
    tail = "reached to Coele-Syria; of which notwithstanding I do not know that there are any good proofs."
    answer = {"items": [
        {"zone": "column0", "kind": "paragraph", "lines": ["A1", "A2"],
         "text": "Coele-Syria is distinguished by no particular name in Scripture, but is comprized under Aram " + tail},
        {"zone": "column0", "kind": "paragraph", "lines": ["B1"], "text": tail},
    ]}
    items = to_items(answer, lines, layout)
    assert items[0].text == "Coele-Syria is distinguished by no particular name in Scripture, but is comprized under Aram"
    assert items[1].text == tail


def test_a_note_read_only_from_its_last_line_starts_where_its_first_was_printed():
    from pdf_ocr_bench.reconstruct import _lead_words

    text = "In the Year of the World 3291, before J. C. 709, before the vulgar Æra 705."
    assert _lead_words(text, Line("N1", "fore the", Box(0, 0, 10, 10), [], "notes0")) == 11  # "be-fore the"
    assert _lead_words(text, Line("N1", "In the Year of", Box(0, 0, 10, 10), [], "notes0")) == 0
    assert _lead_words(text, Line("N1", "the", Box(0, 0, 10, 10), [], "notes0")) == 0  # one word: no telling where


def test_pitch_ignores_the_order_lines_are_listed_in_and_notes_keep_their_margin():
    layout = PageLayout(width=1000, height=1000, line_height=30)
    layout.zones = [Zone("column", Box(100, 100, 900, 600), 0)]
    rows = [(f"L{i}", f"line {i} of the paragraph with some words", 100 + 40 * i) for i in range(8)]
    lines = [Line(i, t, Box(220, y, 900, y + 30), [], "column0") for i, t, y in rows]
    lines.append(Line("N1", "A note.", Box(100, 140, 190, 170), [], "column0"))
    order = ["L0", "L7", "L1", "L2", "L3", "L4", "L5", "L6"]  # the last line listed second
    items = [Item("column0", " ".join(l.text for l in lines[:8]), order),
             Item("column0", "A note.", ["N1"], "note")]
    blocks = build_blocks(items, lines, layout, {}, 1000, 1000)
    para = next(b for b in blocks if b.item.kind == "paragraph")
    assert para.line_h == pytest.approx(40, abs=1)  # the printed pitch, not (y7 - y0) / 7 of the listed order
    assert para.x >= 190  # beside the note, not under it


def test_learnings_keep_misreadings_that_are_no_words(tmp_path):
    import json

    from pdf_ocr_bench.llm_structure import learnings

    pw = tmp_path / "work" / "page_001"
    (pw / "llm").mkdir(parents=True)
    (pw / "lines.json").write_text(json.dumps([{"id": "L1", "text": "He muft fee the whole Hiftory"}]))
    (pw / "llm" / "prompt.txt").write_text("sonnet\nSYSTEM\nThe page's zones and OCR lines (boxes X Y W H on page.png):\n\n[]")
    answer = {"items": [{"zone": "column0", "kind": "paragraph", "lines": ["L1"], "text": "He must see the whose History"}]}
    (pw / "llm" / "response.jsonl").write_text(json.dumps({"type": "result", "structured_output": answer}) + "\n")
    guide = learnings(tmp_path)
    assert "muft -> must" in guide and "Hiftory -> History" in guide
    assert "whole -> whose" not in guide and "fee -> see" not in guide  # words: right only on their page
    assert "<example_answer>" in guide


def test_drop_capital_is_the_scans_picture_over_its_letter_and_spaced_titles_are_spaced():
    layout = PageLayout(width=1000, height=1000, line_height=30)
    layout.zones = [Zone("column", Box(100, 100, 900, 600), 0), Zone("dropcap", Box(100, 100, 180, 190), 0)]
    rows = [("L1", "THE first line beside it", 100, 100), ("L2", "and the second one too", 140, 200),
            ("L3", "then the full width lines go on and on here", 180, 100)]
    lines = [Line(i, t, Box(x, y, 900, y + 30), [], "column0") for i, t, y, x in rows]  # L1 takes in the capital
    title = Line("T1", "TITLE", Box(300, 40, 700, 80), [], "column0", glyph=40, spacing=0.6)
    items = [Item("column0", "TITLE", ["T1"], "heading", align="center"),
             Item("column0", "THE first line beside it and the second one too then the full width lines go on and on here",
                  ["L1", "L2", "L3"], drop_cap="T")]
    blocks = build_blocks(items, lines + [title], layout, {"dropcap0": "pictures/p_cap0.png"}, 1000, 1000)
    cap = next(b for b in blocks if b.item and b.item.kind == "dropcap")
    assert cap.picture == "pictures/p_cap0.png"
    heading = next(b for b in blocks if b.item and b.item.kind == "heading")
    assert heading.letter_spacing > 0.1
    page = blocks_html(blocks, 1000, 1000)
    assert '<img class="pic" src="pictures/p_cap0.png"' in page
    assert "color: rgba(0, 0, 0, 0);" in page and ">T</p>" in page  # the letter, invisible: found and copied
    assert "letter-spacing:" in page


def test_italic_marks_survive_splitting_drop_capitals_and_escaping():
    from pdf_ocr_bench.reconstruct import _split_text, clean_marks, drop_first_letter, marked_html, plain

    assert clean_marks("a <i>b <i>c</i> d</i> <i></i>e <i>f") == "a <i>b c</i> d e <i>f</i>"
    assert clean_marks("x<sup><i>1</i></sup> <sup>a<sup>b</sup></sup></sub>") == "x<sup><i>1</i></sup> <sup>ab</sup>"
    assert plain("in <i>Jer.</i> i. 6.") == "in Jer. i. 6."
    assert drop_first_letter("<i>Y</i>OU have", "Y") == "OU have"
    assert drop_first_letter("THE obliging", "T") == "HE obliging"
    lines = [Line("L1", "one two three", Box(0, 0, 1, 1)), Line("L2", "four five", Box(0, 2, 1, 3))]
    first, rest = _split_text("one <i>two three four</i> five", 1, lines)
    assert (first, rest) == ("one <i>two three</i>", "<i>four</i> five")
    first, rest = _split_text("one <i>two <sup>three four</sup></i> five", 1, lines)
    assert (first, rest) == ("one <i>two <sup>three</sup></i>", "<i><sup>four</sup></i> five")
    assert marked_html("A & <i>B</i>", italic=False) == "A &amp; <i>B</i>"
    assert marked_html("Cock<sup>a</sup>", italic=True) == "Cock<sup>a</sup>"
    assert marked_html("<i>Rome</i> & <i>Paris", italic=True) == '<span class="up">Rome</span> &amp; <span class="up">Paris</span>'


def test_a_paragraph_running_on_into_the_next_column_is_a_block_per_column():
    layout = PageLayout(width=1000, height=1000, line_height=30)
    layout.zones = [Zone("column", Box(100, 100, 480, 900), 0), Zone("column", Box(520, 100, 900, 900), 1),
                    Zone("footnotes", Box(300, 950, 305, 955), 0)]
    left = [Line(f"A{i}", "words of the first column here", Box(100, 700 + 40 * i, 480, 730 + 40 * i), [], "column0") for i in range(5)]
    right = [Line(f"B{i}", "and the end of it here", Box(520, 100 + 40 * i, 900, 130 + 40 * i), [], "column1") for i in range(2)]
    notes = [Line("F1", "a Plin. l. 6.", Box(100, 940, 250, 965), [], "footnotes0")]
    text = " ".join(["words of the first column here"] * 5 + ["and the end of it here"] * 2)
    items = [Item("column0", text, [l.id for l in left + right]), Item("footnotes0", "a Plin. l. 6.", ["F1"], "note")]
    blocks = [b for b in build_blocks(items, left + right + notes, layout, {}, 1000, 1000) if b.item]
    parts = [b for b in blocks if b.item.kind == "paragraph" and "foot" not in b.item.label]
    assert [round(b.x) for b in parts] == [100, 520] and [b.item.zone for b in parts] == ["column0", "column1"]
    assert parts[1].y == pytest.approx(100) and " ".join(b.item.text for b in parts) == text
    foot = next(b for b in blocks if b.item.text == "a Plin. l. 6.")
    assert foot.w > 100  # as wide as its line, not the 5 px speck of a zone it was labelled with


def test_other_markup_in_an_answer_is_normalised():
    from pdf_ocr_bench.reconstruct import normalize_markup

    assert normalize_markup("Chester<sup>a</sup>, <em>Job</em> <small>xi.</small> <SUP class=x>q</SUP>") == "Chester<sup>a</sup>, <i>Job</i> xi. <sup>q</sup>"
    assert normalize_markup('<span class="x">Basil</span><br>Rome') == "Basil Rome"


def test_a_note_without_lines_of_its_own_is_placed_by_its_words_in_the_column_margin():
    layout = PageLayout(width=1000, height=1000, line_height=30)
    layout.zones = [Zone("column", Box(50, 100, 900, 900), 0)]
    # Vision ran the margin notes into the lines beside them
    def line(i, y, note, text, x_note=60):
        words = [(w, Box(x_note + 40 * k, y, x_note + 40 * k + 30, y + 30)) for k, w in enumerate(note.split())]
        words += [(w, Box(250 + 60 * k, y, 250 + 60 * k + 50, y + 30)) for k, w in enumerate(text.split())]
        return Line(f"L{i}", " ".join([note, text]).strip(), Box(x_note if note else 250, y, 900, y + 30), words, "column0")
    lines = [line(1, 100, "Gen. i. 2.", "the paragraph text goes on here"),
             line(2, 140, "", "and on in the next line of it"),
             line(3, 300, "Exod. iv.", "another paragraph begins on this line"),
             line(4, 340, "", "and it ends here on this one")]
    items = [Item("column0", "the paragraph text goes on here and on in the next line of it", ["L1", "L2"]),
             Item("column0", "Gen. i. 2.", [], "note"),  # the model read it from the scan, with no line
             Item("column0", "another paragraph begins on this line and it ends here on this one", ["L3", "L4"]),
             Item("column0", "Exod. iv.", [], "note")]
    blocks = [b for b in build_blocks(items, lines, layout, {}, 1000, 1000) if b.item]
    notes = [b for b in blocks if b.item.kind == "note"]
    assert [n.item.text for n in notes] == ["Gen. i. 2.", "Exod. iv."]  # kept, not dropped
    assert notes[0].y == pytest.approx(100) and notes[1].y == pytest.approx(300)
    assert all(n.x < 200 and n.x + n.w <= 250 for n in notes)  # in the margin, left of the text
    paras = [b for b in blocks if b.item.kind == "paragraph"]
    # each paragraph clear of the note beside it (notes placed by their words make no column margin)
    assert paras[0].x >= notes[0].x + notes[0].w and paras[1].x >= notes[1].x + notes[1].w


def test_type_sizes_of_a_page_are_grouped():
    from pdf_ocr_bench.reconstruct import size_levels

    # body text measured 39-41 pt, notes 28-30, one Hebrew note 47
    est = [(40.0, 20), (39.2, 15), (41.0, 18), (28.5, 2), (29.6, 1), (30.1, 2), (47.0, 2)]
    assert size_levels(est) == [29.6, 40.0, 47.0]

