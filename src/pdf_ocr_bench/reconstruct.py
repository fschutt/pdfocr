"""Rebuild scanned pages as structured HTML: OpenCV layout + macOS Vision text (+ an LLM).

    pdf-ocr-bench reconstruct scan.pdf -o out/ --pages 1-20 --lang enm \\
        [--semantic-context "an English dictionary of the Bible, printed 1732"] [--model sonnet] [--no-llm]

Per page:

1. Render the page at the resolution of its scan (the largest image on the page), and at 3/4 of
   it for Vision: scaling a black-and-white scan down smooths its edges, which Vision reads much
   better than the raw pixels.
2. `page_layout.analyse`: running head, text columns, marginal-note strips, footnotes, other text
   blocks, pictures and drop capitals, from the pixels.
3. macOS Vision reads the text lines; every word goes to the zone it stands in.
4. Pictures are cut out of the scan as PNGs.
5. Structure: which lines make a paragraph, which marginal note sits beside which line, which
   paragraph opens with a drop capital, and the corrected text. An LLM does it (`claude -p`, the
   page's tiles attached, `--semantic-context` saying what the book is), or, with `--no-llm`, a
   heuristic from the line geometry.
6. HTML: every paragraph is a block at the place of its first line, the width of its column,
   justified, set in Times at one size per column (the size at which its paragraphs wrap to
   their original number of lines; a paragraph that needs it gets smaller); notes at one size
   beside the line they annotate; a drop capital as a letter block with the lines beside it
   narrowed; pictures as images. A render-and-measure loop (html2pdf --layout-report) then sets
   smaller what still runs into the block below.

Output: `out/pages.zip` (metadata.json, page_NNN.html, pictures/), which `html2pdf` renders, and
`out/work/page_NNN/` (layout.json, layout.png, lines.json, the LLM transcript).
"""

from __future__ import annotations

import html as htmlmod
import json
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .log import get_logger
from .page_layout import Box, PageLayout, Zone, analyse, draw

log = get_logger("Reconstruct")

# old books are set in a serif; Times (one of the PDF base-14 fonts) is close to their widths
FONT_FAMILY = "Times, 'Times New Roman', serif"
TEXT_ROLES = ("header", "column", "notes", "footnotes", "text")


@dataclass
class Line:
    id: str
    text: str
    box: Box  # native px
    words: list[tuple[str, Box]] = field(default_factory=list)
    zone: str = ""
    glyph: float = 0.0  # median height of its glyphs in the scan, native px
    spacing: float = 0.0  # median gap between its glyphs, in glyph heights (letter-spaced: > 0.25)


@dataclass
class Item:
    """One unit of the page's structure: a paragraph, a marginal note, a heading line."""

    zone: str
    text: str
    lines: list[str]  # ids of the OCR lines it was read from (first = where it starts)
    kind: str = "paragraph"  # paragraph | note | heading
    drop_cap: str = ""  # the initial letter when the paragraph opens with a drop capital
    align: str = "justify"  # justify | left | center | right
    italic: bool = False
    label: str = ""  # the zone the model named (footnotes0: a footnote, whatever zone it stands in)


# --- 1. rendering ----------------------------------------------------------------------------


def native_dpi(page) -> float:
    """The resolution of the page's largest image (the scan), 300 if it has none."""
    import pypdfium2.raw as c

    best = None
    for obj in page.get_objects(filter=[c.FPDF_PAGEOBJ_IMAGE]):
        w_px, _ = obj.get_px_size()
        x0, _, x1, _ = obj.get_bounds()
        if x1 > x0 and (best is None or w_px * (x1 - x0) > best[0] * best[1]):
            best = (w_px, x1 - x0)
    return best[0] / (best[1] / 72.0) if best else 300.0


def render(page, dpi: float, path: Path, gray: bool = True) -> tuple[int, int]:
    bitmap = page.render(scale=dpi / 72.0)
    image = bitmap.to_pil().convert("L" if gray else "RGB")
    image.save(path)
    return image.size


# --- 2./3. text lines in zones ---------------------------------------------------------------


def vision_lines(image: Path, lang: tuple[str, ...], native: tuple[int, int], engine=None) -> list[Line]:
    """Vision's text lines (its own line grouping), in native px."""
    from PIL import Image

    from .engines import ENGINES
    from .engines.base import Route
    from .models import PageImage

    if engine is None:
        engine = ENGINES["macos_vision"](Route(lang=lang))
        engine.prepare()
    with Image.open(image) as im:
        page = PageImage(page_num=0, path=image, width_px=im.width, height_px=im.height, width_pt=1, height_pt=1, dpi=72)
    W, H = native
    by_line: dict[int, list] = {}
    for word in engine.run(page).words:
        by_line.setdefault(word.line if word.line is not None else -1, []).append(word)
    lines = []
    for n, words in enumerate(sorted(by_line.values(), key=lambda ws: (min(w.bbox.y for w in ws), min(w.bbox.x for w in ws)))):
        words.sort(key=lambda w: w.bbox.x)
        boxes = [Box(round(w.bbox.x * W), round(w.bbox.y * H), round((w.bbox.x + w.bbox.w) * W), round((w.bbox.y + w.bbox.h) * H))
                 for w in words]
        box = boxes[0]
        for b in boxes[1:]:
            box = box.union(b)
        lines.append(Line(f"L{n + 1}", " ".join(w.text for w in words), box, [(w.text, b) for w, b in zip(words, boxes)]))
    return lines


def ocr_coverage(lines: list[Line], layout: PageLayout) -> float:
    """The share of the inked words in the text zones (OpenCV) that an OCR word covers."""
    lh = layout.line_height
    zones = [z.box for z in layout.zones if z.role in TEXT_ROLES]
    inked = [w for w in layout.words if w.w >= 1.5 * lh
             and any(z.x0 <= (w.x0 + w.x1) / 2 <= z.x1 and z.y0 <= (w.y0 + w.y1) / 2 <= z.y1 for z in zones)]
    if not inked:
        return 1.0
    read = [b for l in lines for _, b in l.words]
    hit = sum(1 for w in inked if any(c.x0 - 0.3 * lh <= (w.x0 + w.x1) / 2 <= c.x1 + 0.3 * lh
                                      and c.y0 - 0.3 * lh <= (w.y0 + w.y1) / 2 <= c.y1 + 0.3 * lh for c in read))
    return hit / len(inked)


def zone_lines(layout: PageLayout, image: Path, engine, work: Path) -> list[Line]:
    """Every text zone read on its own (a crop of `image`): Vision reads some pages whole only in
    part (p. 995 of vol. 1: 54 words of the page; its two columns alone: 480 and 512)."""
    from PIL import Image

    from .models import PageImage

    lh = layout.line_height
    out: list[Line] = []
    with Image.open(image) as im:
        scale = im.width / layout.width
        for z in layout.zones:
            if z.role not in TEXT_ROLES:
                continue
            x0, y0 = max(0, z.box.x0 - 0.5 * lh), max(0, z.box.y0 - 0.5 * lh)
            x1, y1 = min(layout.width, z.box.x1 + 0.5 * lh), min(layout.height, z.box.y1 + 0.5 * lh)
            crop = im.crop((round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale)))
            path = work / "zone.png"
            crop.save(path)
            page = PageImage(page_num=0, path=path, width_px=crop.width, height_px=crop.height, width_pt=1, height_pt=1, dpi=72)
            by_line: dict[int, list] = {}
            for word in engine.run(page).words:
                by_line.setdefault(word.line if word.line is not None else -1, []).append(word)
            for words in sorted(by_line.values(), key=lambda ws: min(w.bbox.y for w in ws)):
                words.sort(key=lambda w: w.bbox.x)
                boxes = [Box(round(x0 + w.bbox.x * (x1 - x0)), round(y0 + w.bbox.y * (y1 - y0)),
                             round(x0 + (w.bbox.x + w.bbox.w) * (x1 - x0)), round(y0 + (w.bbox.y + w.bbox.h) * (y1 - y0)))
                         for w in words]
                box = boxes[0]
                for b in boxes[1:]:
                    box = box.union(b)
                out.append(Line(f"L{len(out) + 1}", " ".join(w.text for w in words), box, [(w.text, b) for w, b in zip(words, boxes)]))
        (work / "zone.png").unlink(missing_ok=True)
    return out


def recover_missed(lines: list[Line], layout: PageLayout, image: Path, engine, work: Path,
                   max_strips: int = 60) -> list[Line]:
    """Lines Vision left out of the whole page, read again from a strip of the page around them.

    Vision sometimes skips a line (an italic line under a handwritten mark, two lines between
    headings of a title page) that it reads from a crop. Inked words (OpenCV) that no Vision
    line covers are grouped into strips; each strip is cropped from `image` and read alone.
    """
    from PIL import Image

    from .models import PageImage

    lh = layout.line_height
    covered = [l.box for l in lines]
    blocked = [z.box for z in layout.zones if z.role in ("picture", "dropcap")]

    def is_covered(b: Box) -> bool:
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        return any(c.x0 - 0.3 * lh <= cx <= c.x1 + 0.3 * lh and c.y0 - 0.3 * lh <= cy <= c.y1 + 0.3 * lh
                   for c in covered + blocked)

    missed = sorted((b for b in layout.words if b.w >= 1.5 * lh and not is_covered(b)), key=lambda b: (b.y0, b.x0))
    strips: list[Box] = []
    for b in missed:  # words that share a line (more than half their height) make one strip
        for i, s in enumerate(strips):
            if min(s.y1, b.y1) - max(s.y0, b.y0) > 0.5 * min(s.h, b.h):
                strips[i] = s.union(b)
                break
        else:
            strips.append(b)
    strips = [s for s in strips if s.w >= 3 * lh][:max_strips]
    if not strips:
        return []
    out: list[Line] = []
    with Image.open(image) as im:
        scale = im.width / layout.width  # the Vision render is smaller than the native one
        for k, s in enumerate(strips):
            x0, y0 = max(0, s.x0 - lh), max(0, s.y0 - 0.6 * lh)
            x1, y1 = min(layout.width, s.x1 + lh), min(layout.height, s.y1 + 0.6 * lh)
            crop = im.crop((round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale)))
            path = work / f"recover_{k + 1}.png"
            crop.save(path)
            page = PageImage(page_num=0, path=path, width_px=crop.width, height_px=crop.height, width_pt=1, height_pt=1, dpi=72)
            by_line: dict[int, list] = {}
            for word in engine.run(page).words:
                by_line.setdefault(word.line if word.line is not None else -1, []).append(word)
            for words in by_line.values():
                words.sort(key=lambda w: w.bbox.x)
                boxes = [Box(round(x0 + w.bbox.x * (x1 - x0)), round(y0 + w.bbox.y * (y1 - y0)),
                             round(x0 + (w.bbox.x + w.bbox.w) * (x1 - x0)), round(y0 + (w.bbox.y + w.bbox.h) * (y1 - y0)))
                         for w in words]
                fresh = [(w, b) for w, b in zip(words, boxes) if not is_covered(b)]  # not read before
                if not fresh:
                    continue
                box = fresh[0][1]
                for _, b in fresh[1:]:
                    box = box.union(b)
                out.append(Line(f"L{len(lines) + len(out) + 1}", " ".join(w.text for w, _ in fresh), box,
                                [(w.text, b) for w, b in fresh]))
    return out


SPACED = 0.25  # gaps between glyphs in glyph heights: 0.04-0.12 in text, 0.3-1.5 letter-spaced
TIMES_CAP, TIMES_X = 0.662, 0.448  # cap height and x-height of Times, em


def measure_glyphs(lines: list[Line], native: Path) -> None:
    """Each line's glyph height and the gaps between its glyphs, from the scan's ink."""
    import cv2
    import numpy as np

    gray = cv2.imread(str(native), cv2.IMREAD_GRAYSCALE)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    for line in lines:
        b = line.box
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink[b.y0:b.y1, b.x0:b.x1], connectivity=8)
        glyphs = sorted((x, x + w, h) for x, y, w, h, a in stats[1:] if a >= 8 and h >= 0.3 * b.h)
        if len(glyphs) < 4:
            continue
        line.glyph = float(np.median([g[2] for g in glyphs]))
        gaps = [max(0, nxt[0] - cur[1]) for cur, nxt in zip(glyphs, glyphs[1:])]
        line.spacing = float(np.median(gaps)) / max(line.glyph, 1.0)


def assign(lines: list[Line], layout: PageLayout) -> list[Line]:
    """Every line in the zone its words stand in; a line Vision ran across two zones (a note
    and the text beside it) is split into one line per zone."""
    text_zones = [z for z in layout.zones if z.role in TEXT_ROLES]
    out: list[Line] = []
    for line in lines:
        parts: dict[str, list[tuple[str, Box]]] = {}
        for text, box in line.words:
            cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
            zone = next((z for z in text_zones if z.box.contains_point(cx, cy)), None)
            if zone is None:  # just outside every zone: the nearest one
                zone = min(text_zones, key=lambda z: _distance(z.box, cx, cy), default=None)
            if zone is None or zone.role == "picture":
                continue
            parts.setdefault(zone.id, []).append((text, box))
        for k, (zone_id, words) in enumerate(parts.items()):
            box = words[0][1]
            for _, b in words[1:]:
                box = box.union(b)
            piece = "abcdefghijklmnopqrstuvwxyz"[k] if k < 26 else f"z{k}"  # a line across a table: many zones
            out.append(Line(line.id if len(parts) == 1 else f"{line.id}{piece}",
                            " ".join(t for t, _ in words), box, words, zone_id))
    return out


def _distance(box: Box, x: float, y: float) -> float:
    dx = max(box.x0 - x, 0, x - box.x1)
    dy = max(box.y0 - y, 0, y - box.y1)
    return (dx * dx + dy * dy) ** 0.5


# --- 4. pictures -----------------------------------------------------------------------------


def clip_pictures(native: Path, layout: PageLayout, out_dir: Path, page_name: str) -> dict[str, str]:
    """{zone id: path in the zip} for every picture, cut from the native render."""
    from PIL import Image

    paths = {}
    with Image.open(native) as im:
        for z in layout.of("picture") + layout.of("dropcap"):  # a drop capital is set as cut from the scan
            name = f"pictures/{page_name}_{'cap' if z.role == 'dropcap' else ''}{z.index}.png"
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            im.crop((z.box.x0, z.box.y0, z.box.x1, z.box.y1)).save(target, optimize=True)
            paths[z.id] = name
    return paths


# --- 5. structure without an LLM -------------------------------------------------------------


def heuristic_structure(lines: list[Line], layout: PageLayout) -> list[Item]:
    """Paragraphs from the line geometry; each note line group is a note; header lines as they are."""
    lh = layout.line_height
    zones = {z.id: z for z in layout.zones}
    items: list[Item] = []
    for zone in layout.zones:
        mine = sorted((l for l in lines if l.zone == zone.id), key=lambda l: l.box.y0)
        if not mine:
            continue
        if zone.role == "notes":
            group: list[Line] = []
            for line in mine:
                if group and line.box.y0 - group[-1].box.y1 > 0.8 * lh:
                    items.append(_item(zone.id, group, "note", "left"))
                    group = []
                group.append(line)
            if group:
                items.append(_item(zone.id, group, "note", "left"))
            continue
        if zone.role in ("header", "text"):
            centred = all(abs((l.box.x0 + l.box.x1) / 2 - (zone.box.x0 + zone.box.x1) / 2) < 2 * lh for l in mine)
            for line in mine:
                items.append(_item(zone.id, [line], "heading", "center" if centred else "left"))
            continue
        left = min(l.box.x0 for l in mine)
        right = max(l.box.x1 for l in mine)
        group = []
        for line in mine:
            if group:
                prev = group[-1]
                indented = line.box.x0 - left > 0.8 * lh and prev.box.x0 - left < 0.5 * lh
                gap = line.box.y0 - prev.box.y1 > 1.0 * lh
                short_end = right - prev.box.x1 > 3 * lh and prev.text.rstrip().endswith((".", ":", "?", "!"))
                if indented or gap or short_end:
                    items.append(_item(zone.id, group))
                    group = []
            group.append(line)
        if group:
            items.append(_item(zone.id, group))
    # a drop capital opens the paragraph whose first line stands beside it
    for cap in layout.of("dropcap"):
        beside = [it for it in items if it.kind == "paragraph" and zones[it.zone].role in ("column", "text")]
        first = next((it for it in beside if _line_by_id(lines, it.lines[0]).box.y0 <= cap.box.y0 + lh
                      and _line_by_id(lines, it.lines[0]).box.y1 >= cap.box.y0 - lh
                      and _line_by_id(lines, it.lines[0]).box.x0 >= cap.box.x1 - lh), None)
        if first and first.text:
            first.drop_cap = first.text[0]
    return items


def _item(zone: str, group: list[Line], kind: str = "paragraph", align: str = "justify") -> Item:
    words: list[str] = []
    for line in group:
        for word in line.text.split():
            if words and words[-1].endswith("-") and word[:1].islower():
                words[-1] = words[-1][:-1] + word
            else:
                words.append(word)
    return Item(zone=zone, text=" ".join(words), lines=[l.id for l in group], kind=kind, align=align)


def _line_by_id(lines: list[Line], line_id: str) -> Line:
    return next(l for l in lines if l.id == line_id)


# --- 6. HTML ---------------------------------------------------------------------------------


# the book's own fonts (`reconstruct --typeface`, built by `pdf-ocr-bench typeface`), else Times:
# families, files, x-height and cap height (em), and a key of the fonts (for the fit cache)
TYPEFACE: dict = {}


def use_typeface(path: Path | None) -> None:
    """Set the text in the fonts of a `pdf-ocr-bench typeface` directory (None: in Times)."""
    import hashlib

    TYPEFACE.clear()
    _FONTS.clear()
    if path is None:
        return
    info = json.loads((path / "typeface.json").read_text(encoding="utf-8"))
    files = {style: (path / name).resolve() for style, name in info["files"].items()}
    roman = info["metrics"]["roman"]["glyphs"]
    TYPEFACE.update(families=info["families"], files=files,
                    x=roman.get("x", {}).get("top", TIMES_X), cap=roman.get("H", {}).get("top", TIMES_CAP),
                    key=hashlib.sha1(b"".join(f.read_bytes() for f in files.values())).hexdigest())


def font_family(italic: bool = False) -> str:
    """The CSS font-family of the text: the book's fonts (Times for what they lack), or Times."""
    if not TYPEFACE:
        return FONT_FAMILY
    return f"'{TYPEFACE['families']['italic' if italic else 'roman']}', {FONT_FAMILY}"


def html2pdf_fonts() -> list[str]:
    """html2pdf's --font arguments for the book's fonts."""
    return [arg for style in ("roman", "italic") if TYPEFACE
            for arg in ("--font", f"{TYPEFACE['families'][style]}={TYPEFACE['files'][style]}")]


_FONTS: dict = {}


def _times(italic: bool = False):
    """The text's font for measuring: the book's (`--typeface`), else Times."""
    import pymupdf

    key = "italic" if italic else "roman"
    if key not in _FONTS:
        _FONTS[key] = (pymupdf.Font(fontfile=str(TYPEFACE["files"][key])) if TYPEFACE
                       else pymupdf.Font("tiit" if italic else "tiro"))  # Times-Italic / Times-Roman
    return _FONTS[key]


def wrap_lines(text: str, width_pt: float, size: float, font, indent: float = 0.0) -> int:
    """Lines `text` takes when set justified-wrapped at `size` in `width_pt` (`font`'s metrics)."""
    space = font.text_length(" ", size)
    lines, x = 1, indent
    for word in text.split():
        w = font.text_length(word, size)
        if x > 0 and x + space + w > width_pt:
            lines, x = lines + 1, w
        else:
            x = x + (space if x > 0 else 0) + w
    return lines


def fit_size(text: str, width_pt: float, n_lines: int, start: float, font, indent: float = 0.0) -> float:
    """The largest size up to `start` at which `text` wraps to at most `n_lines` lines."""
    lo, hi = 0.35 * start, start
    if wrap_lines(text, width_pt, hi, font, indent) <= n_lines:
        return hi
    for _ in range(18):
        mid = (lo + hi) / 2
        if wrap_lines(text, width_pt, mid, font, indent) <= n_lines:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class Block:
    """One absolutely positioned box of the page: a run of text (a `.region`) or a picture."""

    item: Item | None
    x: float  # pt
    y: float
    w: float
    size: float = 0.0  # font size, pt
    line_h: float = 0.0
    align: str = "justify"
    indent: float = 0.0
    nowrap: bool = False
    picture: str = ""  # path in the zip
    h: float = 0.0  # pictures only
    limit: float = 0.0  # the lowest its text may reach (the next block below it in its zone)
    fit: tuple | None = None  # (lines in the scan, size to start from, line pitch) for sizing
    start_size: float = 0.0  # the size before the fit loop
    level: float = 0.0  # the page's type size it was set at (its group in the fit loop); 0: its own
    area: str = ""  # text, notes, footnotes (Hebrew apart): what measured its size
    letter_spacing: float = 0.0  # em
    pitch: float = 0.0  # the scan's line pitch, pt: the line height whenever the text is smaller
    measure: str = ""  # what measured its size: "line width" (its area's printed lines), else its glyphs
    lead: int = 0  # a note: the words of its text before the first line the OCR read of it
    steps: tuple = ()  # the steps from one of its printed rows to the next, pt (what measured `pitch`)


def column_parts(item: Item, group: list[Line], lh: float) -> list[tuple[Item, list[Line]]]:
    """A paragraph that runs on into the next column: one part per column (a new part where its
    lines go back up the page), each with its share of the text (by the OCR's characters), in the
    zone most of its lines stand in."""
    runs: list[list[Line]] = [[group[0]]]
    for line in group[1:]:
        prev = runs[-1][-1]
        # back up the page, and across to another column (lines merely listed out of order stay)
        if line.box.y0 < prev.box.y0 - 2 * lh and (line.zone != prev.zone or abs(line.box.x0 - prev.box.x0) > 4 * lh):
            runs.append([line])
        else:
            runs[-1].append(line)
    if len(runs) == 1:
        return [(replace(item, zone=_majority_zone(group, item.zone)), group)]
    chars = [sum(len(l.text) for l in run) for run in runs]
    words, out, taken = item.text.split(), [], 0
    rest = item.text
    ends = _run_ends(runs, words)
    for k, run in enumerate(runs):
        last = k == len(runs) - 1
        n = len(words) - taken if last else round(len(words) * sum(chars[:k + 1]) / max(sum(chars), 1)) - taken
        if not last and ends[k] is not None:
            # where the words read in this column end in the text: all of the item's OCR words
            # lined up with its text (by the characters a part took 9 lines more than its 4,
            # its text's continuation not all among the item's lines, p. 455)
            n = ends[k] - taken
        elif not last:
            # at the word the OCR read last in the column, when it is near (the characters are a
            # word or so off: "reigned fifty" for the printed "reigned", p. 126)
            row = [l for l in run if l.box.y0 > max(m.box.y0 for m in run) - 0.5 * lh]
            n = _split_near(words, taken + n, max(row, key=lambda l: l.box.x0).text) - taken
        piece, rest = (rest, "") if last else _split_at(rest, max(0, n))
        taken += max(0, n)
        if plain(piece).strip():
            out.append((replace(item, text=piece, lines=[l.id for l in run], zone=_majority_zone(run, item.zone),
                                drop_cap=item.drop_cap if k == 0 else ""), run))
    return out or [(item, group)]


def _run_ends(runs: list[list[Line]], words: list[str]) -> list[int | None]:
    """For each run of a paragraph's lines (a column's), the number of `words` (its text's) up to
    the last one lined up with a word read in that run; None for a run none of whose words is."""
    import difflib

    ocr, owner = [], []
    for k, run in enumerate(runs):
        rows: list[list[Line]] = []  # (its lines top to bottom, the pieces of each left to right)
        for line in sorted(run, key=lambda l: l.box.y0):
            if rows and line.box.y0 - rows[-1][0].box.y0 < 0.5 * max(rows[-1][0].box.h, 1):
                rows[-1].append(line)
            else:
                rows.append([line])
        for line in [l for row in rows for l in sorted(row, key=lambda l: l.box.x0)]:
            for w in line.text.split():
                key = _norm_word(w)
                if key:
                    ocr.append(key)
                    owner.append(k)
    keys = [_norm_word(plain(w)) for w in words]
    match = difflib.SequenceMatcher(None, ocr, keys, autojunk=False)
    ends: list[int | None] = [None] * len(runs)
    for a, b, size in match.get_matching_blocks():
        for j in range(size):
            k = owner[a + j]
            ends[k] = max(ends[k] or 0, b + j + 1)
    # (in order: a later run's end never before an earlier one's)
    for k in range(1, len(ends)):
        if ends[k] is not None and ends[k - 1] is not None and ends[k] < ends[k - 1]:
            ends[k] = None
    return ends


def _lead_words(text: str, top: Line) -> int:
    """How many words of `text` come before the first one the OCR line `top` read: where the
    most of its words follow one another in the text (its first word may be the end of one the
    print divided: "fore the" of "be-fore the"); 0 if fewer than two of them do."""
    words = [_norm_word(plain(w)) for w in text.split()]
    read = [k for k in (_norm_word(w) for w in top.text.split()) if k]
    if not read:
        return 0
    best, at = 0, 0
    for j in range(len(words)):
        n = 0
        while n < len(read) and j + n < len(words) and (
                words[j + n] == read[n] or (n == 0 and len(read[0]) >= 3 and words[j].endswith(read[0]))):
            n += 1
        if n > best:
            best, at = n, j
    return at if best >= 2 else 0


def _split_near(words: list[str], n: int, last: str) -> int:
    """`n` (a split after the first `n` of `words`) moved to just after the word near it that the
    OCR read as `last` (the last of a line); a word the print hyphenated there goes on to the
    next part. `n` if there is none."""
    tokens = last.split()
    target = _norm_word(tokens[-1]) if tokens else ""
    if len(target) < 2:
        return n
    runs_on = tokens[-1].endswith(("-", "\u00ac"))
    best = None
    for i in range(max(0, n - 5), min(len(words), n + 5)):
        w = _norm_word(plain(words[i]))
        if runs_on and w.startswith(target) and w != target:
            at = i
        elif w == target:
            at = i + 1
        else:
            continue
        if best is None or abs(at - n) < abs(best - n):
            best = at
    return n if best is None else best


def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower().replace("ſ", "s").replace("f", "s"))


def locate(text: str, lines: list[Line], lh: float) -> tuple[Box, list] | None:
    """Where the words of `text` stand on the page, for an item that has no OCR lines of its own:
    the line with most of its words, and the lines just below it (a note of several lines),
    only the words near those on that line."""
    want: dict[str, int] = {}
    for w in plain(text).split():
        k = _norm_word(w)
        if k:
            want[k] = want.get(k, 0) + 1
    if not want:
        return None
    def word_hit(token: str, box: Box, left: dict) -> tuple[str, Box] | None:
        # its word, or one Vision ran onto the end of the text's word before it ("haveLuk."):
        # then the box of its share of the letters
        k = _norm_word(token)
        if left.get(k, 0) > 0:
            return k, box
        w = next((w for w, n in left.items() if n > 0 and len(w) >= 3 and w.isalpha() and k.endswith(w)), None)
        if w is None:
            return None
        return w, replace(box, x0=box.x1 - int(box.w * (len(w) + 1) / max(len(k) + 1, 1)))

    def matched(line: Line) -> list:
        left, out = dict(want), []
        for t, b in line.words:
            h = word_hit(t, b, left)
            if h is not None:
                left[h[0]] -= 1
                out.append(h)
        return out
    scored = [(matched(l), l) for l in lines]
    best_m, best = max(scored, key=lambda s: (len(s[0]), -s[1].box.y0), default=([], None))
    best_n, n_words = len(best_m), sum(want.values())
    # (two words: one will do when it is a word, not a figure: "Matth." of "Matth. xxi.16,17.",
    # whose figures Vision did not read, p. 705)
    enough = 1 if n_words == 1 or (n_words == 2 and any(len(k) >= 4 and k.isalpha() for k, _ in best_m)) \
        else max(2, 0.4 * n_words)
    if best is None or best_n < enough:
        return None
    # on that line, the longest run of its words (a gap of one word allowed): "Ch. iv. 14, 15" at
    # the line's start, not every "14" in the text beside it
    keys = dict(want)
    flags = []
    for t, b in best.words:
        h = word_hit(t, b, keys)
        if h is not None:
            keys[h[0]] -= 1
        flags.append((h is not None, h[1] if h else b))
    runs, cur, gap = [], [], 0
    for hit, b in flags:
        if hit:
            cur.append(b)
            gap = 0
        elif cur and gap == 0:
            gap = 1
        else:
            if cur:
                runs.append(cur)
            cur, gap = [], 0
    if cur:
        runs.append(cur)
    first = max(runs, key=len, default=[])
    if len(first) < (1 if enough == 1 else max(2, -(-n_words // 2))):
        return None  # a stray "iv" or "14" in the text beside it is no find
    x0, x1 = min(b.x0 for b in first) - lh, max(b.x1 for b in first) + lh
    found, left = [], dict(want)
    for line in sorted(lines, key=lambda l: l.box.y0):
        if not (best.box.y0 - 0.5 * lh <= line.box.y0 <= best.box.y0 + (n_words / 2 + 1) * 1.5 * lh):
            continue
        for t, b in line.words:
            h = word_hit(t, b, left)
            if h is not None and h[1].x1 >= x0 and h[1].x0 <= x1:
                left[h[0]] -= 1
                found.append((t, h[1]))
    box = found[0][1]
    for _, b in found[1:]:
        box = box.union(b)
    return box, found


def unread_ink(layout: PageLayout, lines: list[Line], prev: list[Line], lh: float) -> tuple[Box, list] | None:
    """Where text no OCR read stands (Hebrew, which Vision does not read): inked words (OpenCV)
    no OCR word covers, run together on a row; the first such run after the item before it."""
    covered = [b for l in lines for _, b in l.words] or [l.box for l in lines]
    blocked = [z.box for z in layout.zones if z.role in ("picture", "dropcap")]
    free = [w for w in layout.words if w.w >= 0.8 * lh and not any(
        c.x0 - 0.2 * lh <= (w.x0 + w.x1) / 2 <= c.x1 + 0.2 * lh and c.y0 - 0.2 * lh <= (w.y0 + w.y1) / 2 <= c.y1 + 0.2 * lh
        for c in covered + blocked)]
    runs: list[list[Box]] = []
    for w in sorted(free, key=lambda w: (w.y0, w.x0)):
        run = next((r for r in runs if abs(r[-1].y0 - w.y0) < 0.6 * lh and w.x0 - r[-1].x1 < 1.5 * lh), None)
        if run is None:
            runs.append([w])
        else:
            run.append(w)
    if not runs:
        return None
    py = min((l.box.y0 for l in prev), default=0)
    px1 = max((l.box.x1 for l in prev if l.box.y0 < py + 0.6 * lh), default=0)
    after = [r for r in runs if (abs(r[0].y0 - py) < 0.6 * lh and r[0].x0 >= px1 - lh) or r[0].y0 >= py + 0.6 * lh]
    if prev and max(l.box.x1 for l in prev) - min(l.box.x0 for l in prev) > 10 * lh:
        # (in a margin, not in the text it follows: a word of that text Vision's box missed by a
        # few px stood for a Hebrew note, which was set in the paragraph's first lines, p. 106)
        tx0, tx1 = min(l.box.x0 for l in prev), max(l.box.x1 for l in prev)
        after = [r for r in after if r[-1].x1 < tx0 + lh or r[0].x0 > tx1 - lh]
    if not after:
        return None
    run = min(after, key=lambda r: (r[0].y0 >= py + 0.6 * lh, r[0].y0, r[0].x0))
    box = run[0]
    for w in run[1:]:
        box = box.union(w)
    return box, []


def place_lineless(items: list[Item], lines: list[Line], layout: PageLayout) -> tuple[list[Line], set[str]]:
    """Lines for the items that have none: where their words are (`locate`), else just below the
    item before them. Each such item gets the new line's id; the new lines are returned, and the
    ids of the notes that were not found (they go to their column's margin, see `build_blocks`)."""
    known = {l.id for l in lines}
    text_zones = [z for z in layout.zones if z.role in TEXT_ROLES]
    lh = layout.line_height
    made: list[Line] = []
    floating: set[str] = set()
    prev: list[Line] = []
    # the words of each line its items' texts do not account for (a margin note Vision ran into
    # the line: "Jor-Aden" of the text is the text's, "Jor-aden." at the line's end the note's)
    owners: dict[str, list[str]] = {}
    for it in items:
        for i in it.lines:
            owners.setdefault(i, []).append(it.text)
    # note strips whose lines are mostly pieces of paragraphs' lines
    kind_of = {i: it.kind for it in items for i in it.lines}
    text_strips = set()
    for z in layout.zones:
        if z.role == "notes":
            mine = [l for l in lines if l.zone == z.id and l.id in kind_of]
            if mine and sum(kind_of[l.id] == "paragraph" for l in mine) > 0.5 * len(mine):
                text_strips.add(z.id)
    loose = []
    for l in lines:
        left: dict[str, int] = {}
        for t in owners.get(l.id, []):
            for w in plain(t).split():
                k = _norm_word(w)
                left[k] = left.get(k, 0) + 1
        free = []
        for t, b in l.words:
            k = _norm_word(t)
            if left.get(k, 0) > 0:
                left[k] -= 1
            else:
                free.append((t, b))
        loose.append(replace(l, words=free))
    for k, item in enumerate(items):
        own = [l for l in lines + made if l.id in item.lines]
        if own:
            prev = own
            continue
        hit = locate(item.text, loose, lh)
        # the note strips beside the item before it (one elsewhere on the page is no margin here:
        # its x at this height is in the text, 174 pt of a paragraph kept free for it, p. 705; nor
        # one whose lines are the ends of a paragraph's, cut off by the layout, p. 106)
        strips = [z for z in layout.zones if z.role == "notes" and prev and z.id not in text_strips
                  and z.box.y0 - 2 * lh <= min(l.box.y0 for l in prev) <= z.box.y1]
        if not hit and _ratio_of(item.text) == HEBREW_LETTER:
            hit = unread_ink(layout, lines + made, prev, lh)
        if hit:
            box, words = hit
        elif item.kind == "note" and prev and not strips:
            # beside the start of the entry it follows; its column's margin is set later
            y0 = min(l.box.y0 for l in prev)
            box, words = Box(min(l.box.x0 for l in prev), y0, max(l.box.x1 for l in prev), int(y0 + 1.2 * lh)), []
            floating.add(f"X{k + 1}")
        elif item.kind == "note" and strips and prev:
            # a note read from the scan alone: in the note strip beside the item it follows, under
            # the notes put there before it
            py0 = min(l.box.y0 for l in prev)
            pcx = (min(l.box.x0 for l in prev) + max(l.box.x1 for l in prev)) / 2
            strip = min(strips, key=lambda z: abs((z.box.x0 + z.box.x1) / 2 - pcx))
            y0 = max([py0] + [l.box.y1 + int(0.3 * lh) for l in made if l.zone == strip.id])
            box, words = Box(strip.box.x0, y0, strip.box.x1, int(y0 + 1.2 * lh)), []
        else:
            # read by the model alone: under the item before it, as wide as that item's lines
            x0 = min((l.box.x0 for l in prev), default=text_zones[0].box.x0 if text_zones else 0)
            x1 = max((l.box.x1 for l in prev), default=text_zones[0].box.x1 if text_zones else layout.width)
            y0 = max((l.box.y1 for l in prev), default=text_zones[0].box.y0 if text_zones else 0) + int(0.3 * lh)
            box, words = Box(int(x0), y0, int(x1), int(y0 + 1.2 * lh)), []
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        zone = next((z for z in text_zones if z.box.contains_point(cx, cy)), None) or \
            min(text_zones, key=lambda z: _distance(z.box, cx, cy), default=None)
        line_id = f"X{k + 1}"
        while line_id in known:
            line_id += "x"
        if f"X{k + 1}" in floating and line_id != f"X{k + 1}":
            floating.discard(f"X{k + 1}")
            floating.add(line_id)
        line = Line(line_id, plain(item.text), box, words, zone.id if zone else item.zone)
        made.append(line)
        item.lines = [line_id]
        if zone is not None and item.zone not in {z.id for z in layout.zones}:
            item.zone = zone.id
        if item.kind != "note" or line_id not in floating:
            prev = [line]  # notes beside one entry stay beside its start
    return made, floating


def own_words(text: str, group: list[Line]) -> list:
    """The OCR words of `group` that are words of `text` (its own, on lines it shares)."""
    want: dict[str, int] = {}
    for w in plain(text).split():
        k = _norm_word(w)
        if k:
            want[k] = want.get(k, 0) + 1
    found = []
    for line in group:
        for t, box in line.words:
            k = _norm_word(t)
            if want.get(k, 0) > 0:
                want[k] -= 1
                found.append((t, box))
    return found


def own_extent(text: str, group: list[Line]) -> Box | None:
    """Where the words of `text` stand on its OCR lines, when those lines also carry other text
    (Vision reads a margin note and the line of text beside it as one line); None if too few of
    its words are found there."""
    norm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower().replace("ſ", "s").replace("f", "s"))
    want: dict[str, int] = {}
    words = [norm(w) for w in plain(text).split()]
    for w in words:
        if w:
            want[w] = want.get(w, 0) + 1
    found = []
    for line in group:
        for t, box in line.words:
            k = norm(t)
            if want.get(k, 0) > 0:
                want[k] -= 1
                found.append(box)
    if not found or len(found) < 0.5 * len([w for w in words if w]):
        return None
    out = found[0]
    for b in found[1:]:
        out = out.union(b)
    return out


def row_extent(group: list[Line], lh: float, ragged: bool = False) -> tuple[float, float, int]:
    """Where `group`'s printed lines (their pieces joined) mostly start and end (where the longest
    ends, for `ragged` text: a margin note), and how many printed lines it has."""
    rows: list[list[float]] = []
    for l in sorted(group, key=lambda l: l.box.y0):
        if rows and l.box.y0 - rows[-1][2] <= 0.5 * lh:
            rows[-1][0] = min(rows[-1][0], l.box.x0)
            rows[-1][1] = max(rows[-1][1], l.box.x1)
        else:
            rows.append([l.box.x0, l.box.x1, l.box.y0])
    # (where most end: justified lines end at the column's edge; a line Vision ran across the gutter
    # into the next column must not carry the measure there, p. 914 and p. 947)
    x0s, x1s = sorted(r[0] for r in rows), sorted(r[1] for r in rows)
    return x0s[round(0.2 * (len(x0s) - 1))], x1s[(len(x1s) - 1) // 2 if len(x1s) >= 3 and not ragged else -1], len(rows)


def text_edges(ink, e0: float, e1: float, y0: float, y1: float, lh: float, rows: int,
               words: list | None = None, text: str = "") -> tuple[float, float]:
    """Where a paragraph's text starts and ends (`e0`..`e1`: where its lines do, its `rows` rows
    `y0`..`y1`, px), when margin notes are printed close against it (6 pt) and Vision ran them
    into its lines: then its lines end where the notes do (vol. 1 p. 245: the column measured
    1696 pt wide to 1610 printed, its paragraphs set over their notes). At a valley of ink down
    its rows near its edge (six rows of justified text have none of their own), with notes
    beyond it: past a white line, or sparser than the text, or in a dozen rows or more. And the
    words of its lines (`words`) beyond the cut must be no words of its `text` (the model set the
    notes apart): a river of spaces down seven rows, or lines that start where they please (an
    indent, a quotation mark), leave its own words there (p. 433, p. 823)."""
    import numpy as np

    if ink is None or rows < 6:
        return e0, e1
    a, b = max(0, int(e0)), min(ink.shape[1], int(e1) + 1)
    band = ink[max(0, int(y0)):min(ink.shape[0], int(y1)), a:b]
    reach, least, beyond = int(8 * lh), max(2, int(0.15 * lh)), int(1.5 * lh)
    if band.shape[1] < 3 * reach:
        return e0, e1
    cols = band.sum(axis=0).astype(float)
    body = float(np.median(cols[reach:-reach]))
    # (in a dozen rows or more, a fifth will do: where few lines have ink, the notes touch it)
    many = rows >= 12
    low = cols <= (0.2 if many else 0.15) * body
    least = 2 if many else least

    def edge(order: list[int]) -> int | None:
        # walking out of the text to its edge: the first valley is where it ends (its last, or
        # first, inked column), if notes stand beyond
        run = 0
        for k, i in enumerate(order):
            if low[i]:
                run += 1
                continue
            if run >= least:
                if k - run < 1:
                    return None
                notes = cols[order[k:]]
                notes = notes[notes > 0.15 * body]
                clean = cols[order[k - run:k]].min() <= 0.01 * body  # a white line in it
                if len(notes) >= beyond and (clean or many or notes.mean() <= 0.6 * body):
                    return order[k - run - 1]
                return None
            run = 0
        return None

    def notes_beyond(cut: float, side: str) -> bool:
        # (words with a letter or figure: a quotation mark opens each line of a quotation)
        if not words or not text:
            return True
        own: dict[str, int] = {}
        for t in plain(text).split():
            k = _norm_word(t)
            own[k] = own.get(k, 0) + 1
        keyed = [(_norm_word(t), (b.x0 + b.x1) / 2) for t, b in words if _norm_word(t)]
        outside = [k for k, c in keyed if (c > cut if side == "right" else c < cut)]
        for k, c in keyed:  # its own words read inside the cut
            if (c <= cut if side == "right" else c >= cut) and own.get(k, 0) > 0:
                own[k] -= 1
        mine = 0
        for k in outside:
            if own.get(k, 0) > 0:
                own[k] -= 1
                mine += 1
        return mine <= 0.3 * len(outside)

    w = len(cols)
    new0, new1 = e0, e1
    right = edge(list(range(w - reach, w)))
    if right is not None and notes_beyond(a + right + 1, "right"):
        new1 = a + right + 1
    left = edge(list(range(reach, -1, -1)))
    if left is not None and notes_beyond(a + left, "left"):
        new0 = a + left
    return new0, new1


def _row_ends(group: list[Line], lh: float) -> tuple[float, float]:
    """Where `group`'s printed lines (their pieces joined) start and end, the middle ones."""
    rows: list[list[float]] = []
    for l in sorted(group, key=lambda l: l.box.y0):
        if rows and l.box.y0 - rows[-1][2] <= 0.5 * lh:
            rows[-1][0], rows[-1][1] = min(rows[-1][0], l.box.x0), max(rows[-1][1], l.box.x1)
        else:
            rows.append([l.box.x0, l.box.x1, l.box.y0])
    x0s, x1s = sorted(r[0] for r in rows), sorted(r[1] for r in rows)
    return x0s[len(x0s) // 2], x1s[len(x1s) // 2]


def _majority_zone(group: list[Line], default: str) -> str:
    """The zone most of the text of `group` stands in (by characters: a line Vision ran across a
    margin note is split into a long piece in the column and a short one in the note strip)."""
    counts: dict[str, int] = {}
    for line in group:
        if line.zone:
            counts[line.zone] = counts.get(line.zone, 0) + max(1, len(line.text))
    return max(counts, key=counts.get) if counts else default


FOOT_MARK = re.compile(r"\s*(?:<sup>[^<]{1,3}</sup>|<i>[a-z]</i>\s|[ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ*†‡§‖¶])")
NOTE_TEXT = 0.79  # margin notes' size to the text's (vol. 1: the median of 754 pages, p10 0.71)
WRAP_QUANTILE = 0.1  # the paragraphs a page's text size is to fit in their printed lines: all but a tenth
HEBREW_LETTER = 0.56  # height of a Hebrew letter in azul's Times, em (x-height 0.448, capitals 0.662)


def _ratio_of(text: str) -> float:
    """Glyph height / em for the main script of `text`."""
    letters = [c for c in plain(text) if c.isalpha()]
    cap, x = TYPEFACE.get("cap", TIMES_CAP), TYPEFACE.get("x", TIMES_X)
    if not letters:
        return cap  # figures: as tall as capitals
    if sum("\u0590" <= c <= "\u05ff" for c in letters) > 0.5 * len(letters):
        return HEBREW_LETTER
    if sum(c.isupper() for c in letters) > 0.6 * len(letters):
        return cap
    return x


def measured_em(item: Item, group: list[Line], ink, px: float) -> tuple[float, int] | None:
    """The size (pt) the scan sets `item` in: the median height of the glyphs of its own words
    (a note Vision ran into the line beside it is measured apart from that line), by the height
    of such glyphs in Times, and how many glyphs that is. None if there is nothing to measure."""
    import numpy as np

    heights: list[int] = []
    if ink is not None:
        import cv2

        words = own_words(item.text, group) if item.kind == "note" else [w for l in group for w in l.words]
        for _, b in words:
            crop = ink[max(0, b.y0):b.y1, max(0, b.x0):b.x1]
            if crop.size == 0:
                continue
            _, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
            heights += [int(h) for x, y, w, h, a in stats[1:] if a >= 8 and h >= 0.3 * b.h]
    if len(heights) < 4:
        heights = [l.glyph for l in group if l.glyph]  # the lines' own measure
    if not heights:
        return None
    return float(np.median(heights)) * px / _ratio_of(item.text), len(heights)


def size_levels(estimates: list[tuple[float, int]], tol: float = 0.1) -> list[float]:
    """The type sizes of a page: size estimates (with their weights, lines) that lie within `tol`
    of the next make one size, the weighted median of its estimates."""
    levels, group = [], []
    for size, weight in sorted(estimates):
        if group and size > group[-1][0] * (1 + tol):
            levels.append(_weighted_median(group))
            group = []
        group.append((size, weight))
    if group:
        levels.append(_weighted_median(group))
    return levels


def _weighted_median(pairs: list[tuple[float, int]]) -> float:
    total, acc = sum(w for _, w in pairs), 0
    for size, weight in pairs:
        acc += weight
        if acc >= total / 2:
            return size
    return pairs[-1][0]


def wrap_size(blocks: list["Block"], cap: float, pitch: float, least: int = 4) -> tuple[float, int] | None:
    """The size (pt, at most `cap`) at which Times sets the paragraphs of `blocks` in as many lines
    as they were printed in, and of how many paragraphs; None for fewer than 4 of 3 lines or more.

    Times is wider than the book's face at the same x-height, and its width, not its height,
    decides where a line breaks: at the size of the glyphs a paragraph of 7 printed lines took 8
    (vol. 1 p. 126: 41.9 pt by the glyphs; "remains of Cleopatra's magnificent Palace," fits its
    printed 700 pt up to 41.0 pt). A paragraph was printed in its rows, or in as many as stand
    down to the next block at the page's `pitch` (the OCR missed one). The low end of the
    paragraphs' own sizes (WRAP_QUANTILE): one whose text the model lengthened asks for any size.
    None for fewer than `least` paragraphs (the footnotes are one block)."""
    sizes = []
    for b in blocks:
        if b.fit is None or b.fit[0] < 2:
            continue
        rows = max(b.fit[0], int((b.limit - b.y) / pitch + 0.15) if pitch else 0)
        # (the em space between run-in footnotes is as wide as an M, not a word space)
        sizes.append(fit_size(plain(b.item.text).replace("\u2003", "M"), b.w, rows, cap, _times(b.item.italic), b.indent))
    if len(sizes) < least:
        return None
    sizes.sort()
    return sizes[int(WRAP_QUANTILE * len(sizes))], len(sizes)


def is_footnote(item: Item, zones: dict) -> bool:
    """A footnote: the model named a footnote zone for it ("footnotes0", "foot"), or it stands in one."""
    zone = zones.get(item.zone)
    return "foot" in item.label.lower() or (zone is not None and zone.role == "footnotes")


def page_zones(layout: PageLayout) -> list[Zone]:
    """The layout's zones, a note strip as wide as a column of its own band taken for a column (the
    layout calls a column a note strip when another band of the page is set full width: it
    compares with the widest column of the page). The model's answer keeps the layout's names."""
    lh = layout.line_height
    out = []
    for z in layout.zones:
        if z.role == "notes":
            band = [o for o in layout.zones if o.role in ("column", "notes") and o is not z
                    and min(o.box.y1, z.box.y1) - max(o.box.y0, z.box.y0) > 0.5 * min(o.box.h, z.box.h)]
            widest = max([o.box.w for o in band] + [z.box.w])
            if z.box.w >= 0.45 * widest and z.box.w >= 12 * lh:
                z = replace(z, role="column")
        out.append(z)
    return out


def set_footnotes(blocks: list[Block], by_id: dict, lh: float, px: float) -> None:
    """The footnotes as one run-in block across the footnote band, in reading order, a wide space
    between them, at the scan's row pitch: print sets them one after another in rows. (Placed
    one by one from the OCR they ran into each other: lines the OCR merged across the band,
    Hebrew it could not read, a word of one footnote found in another.)"""
    foot = [b for b in blocks if b.item is not None and b.item.kind in ("paragraph", "note") and "foot" in b.item.label.lower()]
    if not foot:
        return
    lines = [by_id[i] for b in foot for i in b.item.lines if i in by_id]
    if not lines:
        return
    x0, x1 = min(l.box.x0 for l in lines), max(l.box.x1 for l in lines)
    top = min(l.box.y0 for l in lines)
    tops = sorted({l.box.y0 for l in lines})
    rows: list[float] = []
    for t in tops:
        if not rows or t - rows[-1] > 0.5 * lh:
            rows.append(t)
    steps = sorted(b2 - a for a, b2 in zip(rows, rows[1:]))
    first = foot[0]
    first.item = replace(first.item, text=" \u2003 ".join(b.item.text for b in foot),
                         lines=[i for b in foot for i in b.item.lines], kind="paragraph")
    first.x, first.w, first.y = x0 * px, (x1 - x0) * px * 1.01, top * px
    first.indent, first.align = 0.0, "left"
    first.pitch = steps[len(steps) // 2] * px if steps else 0.0
    first.steps = tuple(s * px for s in steps)
    first.fit = (len(rows), first.size, first.pitch)  # its printed rows: what sizes it (build_blocks)
    for b in foot[1:]:
        blocks.remove(b)


def _follow_size(b: Block) -> None:
    """A wrapped block's line height: the scan's line pitch, when it was printed on two lines or
    more; else its size. The dictionary is set solid, its notes too (rows 27-32 pt apart for
    notes of 30-33 pt, vol. 1): 1.15 of the size set every note a sixth taller than printed, into
    the note below it, and the fit loop set all the page's notes 15% smaller (p. 105)."""
    if not b.nowrap and b.item is not None and b.item.kind != "dropcap":
        b.line_h = b.pitch if b.pitch else b.size


def _ink(native: Path):
    """The scan as 1 where inked (None if it cannot be read)."""
    import cv2

    gray = cv2.imread(str(native), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return ink


def page_fonts(blocks: list[Block], layout: PageLayout, width_pt: float) -> dict:
    """What type sizes a page uses where (work/page_NNN/fonts.json): every size measured in the
    scan, set at (after the fit loop), and the zones and kinds of block in it; and the columns."""
    px = width_pt / layout.width
    sizes: dict[float, dict] = {}
    for b in blocks:
        if b.item is None or not b.level:
            continue
        entry = sizes.setdefault(round(b.level, 2), {"measured_pt": round(b.level, 2), "set_pt": round(b.size, 2),
                                                     "by": b.measure or "glyph height", "areas": {}})
        area = entry["areas"].setdefault(b.area, {"blocks": 0, "zones": []})
        area["blocks"] += 1
        if b.item.zone not in area["zones"]:
            area["zones"].append(b.item.zone)
    columns = [{"zone": z.id, "x0_pt": round(z.box.x0 * px, 1), "x1_pt": round(z.box.x1 * px, 1)}
               for z in layout.zones if z.role == "column"]
    return {"sizes": sorted(sizes.values(), key=lambda e: -e["measured_pt"]), "columns": columns}


def build_blocks(items: list[Item], lines: list[Line], layout: PageLayout, pictures: dict[str, str],
                 width_pt: float, height_pt: float, ink=None) -> list[Block]:
    """The page's blocks. `ink` (the scan, 1 where inked) measures the type sizes."""
    font = _times()
    px = width_pt / layout.width  # pt per native px
    items = [replace(it) for it in items]  # their lines may be set below
    layout = replace(layout, zones=page_zones(layout))
    made, floating = place_lineless(items, lines, layout)
    lines = lines + made
    by_id = {l.id: l for l in lines}
    zones = {z.id: z for z in layout.zones}
    # footnotes the model put in a column (the layout may cut the footnote band into columns too):
    # text that starts below the bottom of the page's main columns
    # (below every column set in the text's type: a column may run on in a short band below the
    # tall ones; the footnote band, cut into columns too, is set smaller)
    main = [z for z in layout.zones if z.role == "column" and z.box.h >= 0.3 * layout.height]
    body_glyph = sorted(z.glyph for z in main)[len(main) // 2] if main else 0.0
    body = [z for z in layout.zones if z.role == "column" and z.glyph >= 0.9 * body_glyph]
    if main:
        foot_top = max(z.box.y1 for z in body) - 0.3 * layout.line_height
        for it in items:
            g = [by_id[i] for i in it.lines if i in by_id]
            if it.kind in ("paragraph", "note") and g and min(l.box.y0 for l in g) >= foot_top and "foot" not in it.label.lower():
                it.label = "footnotes (below the text)"
    # and by their reference marks, in the lower part of the page (a page the layout left as one
    # block has no columns to be below: p. 445)
    marked = [it for it in items if it.kind in ("paragraph", "note") and FOOT_MARK.match(it.text)
              and (g := [by_id[i] for i in it.lines if i in by_id]) and min(l.box.y0 for l in g) > 0.6 * layout.height]
    if len(marked) >= 2:
        for it in marked:
            if "foot" not in it.label.lower():
                it.label = "footnotes (marked)"

    def em_of(group: list[Line]) -> float:
        # a Vision line box is cap height + descender, ~1.05 em for this kind of face
        hs = sorted(l.box.h for l in group)
        return hs[len(hs) // 2] * px / 1.05 if hs else 10.0

    blocks: list[Block] = []
    for zid, rel in pictures.items():
        if zones[zid].role != "picture":
            continue  # a drop capital: with the paragraph it opens
        b = zones[zid].box
        blocks.append(Block(None, b.x0 * px, b.y0 * px, b.w * px, picture=rel, h=b.h * px))

    # margin notes the layout left inside a column: the text beside them starts right of (ends
    # left of) them, so a paragraph does not run under its note
    inner_notes = []
    seen_notes = []  # those with OCR lines of their own (not placed by their words)
    for it in items:
        g = [by_id[i] for i in it.lines if i in by_id]
        z = zones.get(_majority_zone(g, it.zone)) if g else None
        if it.kind != "note" or z is None or z.role == "notes" or not g or g[0].id in floating:
            continue
        own = own_extent(it.text, g)
        nx0, nx1 = (own.x0, own.x1) if own else (min(l.box.x0 for l in g), max(l.box.x1 for l in g))
        ny0, ny1 = (own.y0, own.y1) if own else (min(l.box.y0 for l in g), max(l.box.y1 for l in g))
        # narrow, at an edge of its column (a footnote under the text is neither)
        if nx1 - nx0 < 0.3 * z.box.w and (nx0 < z.box.x0 + 0.1 * z.box.w or nx1 > z.box.x1 - 0.1 * z.box.w):
            inner_notes.append((nx0, ny0, nx1, ny1, z.id, "left" if (nx0 + nx1) / 2 < z.box.x0 + 0.5 * z.box.w else "right"))
            if not g[0].id.startswith("X"):
                seen_notes.append(inner_notes[-1])
    # a column's note margin: where its notes typically start and end (Vision may have read only
    # a fragment of one note, "3. 4" of "Numb. xix. 3, 4, 5, 6.")
    # (from notes the OCR read there: one on the column's outer side, two on its inner side;
    # a note placed by its words alone may stand anywhere in the text)
    margin_of: dict[tuple[str, str], tuple[float, float]] = {}
    # (and for a note's own box, as wide as its longest note: notes are ragged; the paragraph
    # beside them keeps clear of where they typically end, the note may run to the text)
    note_margin: dict[tuple[str, str], tuple[float, float]] = {}
    for key in {(nt[4], nt[5]) for nt in seen_notes}:
        mine = [nt for nt in seen_notes if (nt[4], nt[5]) == key]
        z = zones[key[0]]
        outer = key[1] == ("left" if z.box.x0 + z.box.w / 2 < layout.width / 2 else "right")
        if len(mine) < (1 if outer else 2):
            continue
        x0s, x1s = sorted(nt[0] for nt in mine), sorted(nt[2] for nt in mine)
        margin_of[key] = (x0s[len(x0s) // 2], x1s[len(x1s) // 2])
        note_margin[key] = (x0s[len(x0s) // 2], x1s[-1])
    # the page's columns as printed: the spans (where most lines start and end) of its paragraphs of
    # three printed lines or more, grouped where they overlap; each column's measure is the median
    # of its spans. Zones do not say it: the layout may cut a column in two, leave two columns one
    # band, or take in a note margin; and Vision runs lines across the gutter.
    spans: list[tuple[float, float]] = []
    spans_at: list[tuple[float, float, float, float]] = []  # and how far down the page each runs
    for it in items:
        if it.kind != "paragraph" or "foot" in it.label.lower():
            continue
        whole = [by_id[i] for i in it.lines if i in by_id]
        for part, g in (column_parts(it, whole, layout.line_height) if whole else []):
            if not g:
                continue
            e0, e1, n_rows = row_extent(g, layout.line_height)
            e0, e1 = text_edges(ink, e0, e1, min(l.box.y0 for l in g), max(l.box.y1 for l in g), layout.line_height, n_rows,
                                [w for l in g for w in l.words], part.text)
            if n_rows >= 3 and len(plain(part.text)) <= 2 * sum(len(l.text) for l in g) and e1 - e0 > 4 * layout.line_height:
                spans.append((e0, e1))
                spans_at.append((e0, e1, min(l.box.y0 for l in g), max(l.box.y1 for l in g)))
    clusters: list[list[tuple[float, float]]] = []
    for e0, e1 in sorted(spans, key=lambda sp: (sp[0] + sp[1]) / 2):
        for c in clusters:
            c0 = sorted(x for x, _ in c)[len(c) // 2]
            c1 = sorted(x for _, x in c)[len(c) // 2]
            if min(e1, c1) - max(e0, c0) > 0.6 * min(e1 - e0, c1 - c0):
                c.append((e0, e1))
                break
        else:
            clusters.append([(e0, e1)])
    columns_measured = [(sorted(x for x, _ in c)[len(c) // 2], sorted(x for _, x in c)[len(c) // 2]) for c in clusters]

    def measure_for(e0: float, e1: float) -> tuple[float, float] | None:
        """The printed column a paragraph's lines (spanning e0..e1) belong to."""
        best = min(columns_measured, key=lambda m: abs(e0 - m[0]) + abs(e1 - m[1]), default=None)
        if best is None or abs(e0 - best[0]) + abs(e1 - best[1]) > 0.35 * (best[1] - best[0]):
            return None
        return best[0], best[1] + 0.01 * (best[1] - best[0])

    # a margin is beside the text, not over it: one the notes taken for it put into a column's
    # text (a note in the gutter: column 1's "left margin" over column 0's line ends, p. 120; a
    # note's extent that took in a word of the text, 130 pt into its own column, p. 37) is none
    for key, (mx0, mx1) in list(margin_of.items()):
        if any(min(mx1, m1) - max(mx0, m0) > layout.line_height for m0, m1 in columns_measured):
            del margin_of[key]
            note_margin.pop(key, None)
    # the notes a paragraph keeps clear of: those the OCR read there, and those in a margin (one
    # placed by its words only where the paragraph's other lines stand clear of it: a "Luke" of
    # the text set the paragraph 267 pt narrower, in 37 lines for 22, p. 605)
    keep_clear = [(m[0], nt[1], m[1], nt[3], nt[4], nt[5]) if (m := margin_of.get((nt[4], nt[5]))) else nt
                  for nt in inner_notes if nt in seen_notes or (nt[4], nt[5]) in margin_of]
    inner_notes = [(m[0], nt[1], m[1], nt[3], nt[4], nt[5]) if (m := margin_of.get((nt[4], nt[5]))) else nt
                   for nt in inner_notes]

    def into_margin(left: float, right: float, top: float, bottom: float) -> tuple[float, float]:
        """A margin note's room: the margin beside the nearest column, from the outer edge of the
        notes and note strips there to just before the column's text (never into it). Vision
        may have read only a piece of a note ("17." of "1 Cor. x. 17.", p. 355): its own lines
        are no measure of its width. The column as printed beside it (`top`..`bottom`): one band
        of a page may be set narrower than another, beside a strip of notes."""
        lh_ = layout.line_height
        gap, reach = 0.5 * lh_, 12 * lh_  # (a column's measure runs 1% past its lines)
        spans = [(z.box.x0, z.box.x1) for z in layout.of("notes")] + [(nt[0], nt[2]) for nt in seen_notes]
        columns = [(e0, e1) for e0, e1, a, b in spans_at if a - 3 * lh_ < bottom and top < b + 3 * lh_] or columns_measured
        centre = (left + right) / 2
        beside = [(m0 - centre, "left", m0) for m0, _ in columns if m0 - reach < centre < m0] + \
                 [(centre - m1, "right", m1) for _, m1 in columns if m1 < centre < m1 + reach]
        if not beside:
            # one that starts in a margin and runs into the text (its extent took in a word of
            # the text beside it, 130 pt over the paragraph, p. 37): that margin's
            beside = [(0.0, "left", m0) for m0, m1 in columns if m0 - reach < left < m0 - lh_ and centre < (m0 + m1) / 2] + \
                     [(0.0, "right", m1) for m0, m1 in columns if m1 + lh_ < right < m1 + reach and centre > (m0 + m1) / 2]
        if not beside:
            return left, right
        _, side, edge = min(beside)
        inked = margin_ink(edge, side, top, bottom)
        # (never over the next column's text: between two columns the ink runs on into it, and a
        # note set in the gutter stood in the text, a paragraph kept 210 pt clear of it, p. 120)
        if side == "left":
            outer = min([left] + [a for a, b in spans if edge - reach < (a + b) / 2 < edge] + ([inked] if inked else []))
            outer = max([outer] + [m1 + gap for _, m1 in columns if m1 <= edge - lh_])
            return (outer, edge - gap) if edge - gap - outer >= 2 * lh_ else (left, right)
        outer = max([right] + [b for a, b in spans if edge < (a + b) / 2 < edge + reach] + ([inked] if inked else []))
        outer = min([outer] + [m0 - gap for m0, _ in columns if m0 >= edge + lh_])
        return (edge + gap, outer) if outer - (edge + gap) >= 2 * lh_ else (left, right)

    def margin_ink(edge: float, side: str, top: float, bottom: float) -> float | None:
        """How far the ink of a column's margin reaches beside `top`..`bottom` (and a few lines
        below: a note runs on below where it starts), out from the column's `edge` to the first
        white 1.5 lines wide. The OCR may have read only pieces of the notes there (p. 605)."""
        if ink is None:
            return None
        lh_ = layout.line_height
        y0, y1 = max(0, int(top - lh_)), min(ink.shape[0], int(bottom + 4 * lh_))
        if side == "right":
            a, b = int(edge + 0.3 * lh_), min(ink.shape[1], int(edge + 12 * lh_))
        else:
            a, b = max(0, int(edge - 12 * lh_)), int(edge - 0.3 * lh_)
        if b - a < 2 or y1 <= y0:
            return None
        cols = ink[y0:y1, a:b].sum(axis=0) >= 3
        order = range(len(cols)) if side == "right" else range(len(cols) - 1, -1, -1)
        last, white = None, 0
        for i in order:
            if cols[i]:
                last, white = i, 0
            else:
                white += 1
                if white > 1.5 * lh_:
                    break
        if last is None:
            return None
        return a + last + 1 if side == "right" else a + last

    # notes not found on the page: in their column's margin (its outer side if it has none yet),
    # stacked where several stand beside one entry
    lh = layout.line_height
    for line in (by_id[i] for i in sorted(floating)):
        z = zones.get(line.zone)
        if z is None or z.role not in ("column", "text"):
            continue
        cx = (line.box.x0 + line.box.x1) / 2
        near = [(e0, e1) for e0, e1, a, b in spans_at
                if a - 3 * lh < line.box.y1 and line.box.y0 < b + 3 * lh and e0 < cx < e1]
        e0, e1 = near[0] if near else (line.box.x0, line.box.x1)
        # (the outer side of its column's text as printed: its zone may take in more of the page)
        outside = "right" if (e0 + e1) / 2 > layout.width / 2 else "left"
        sides = [side for (zid, side) in margin_of if zid == z.id]
        side = sides[0] if sides else outside

        def white(side: str) -> tuple[float, float]:
            # the white beside its text, off the next column's (p. 120: in the gutter, over it)
            if side == "right":
                outer = min([layout.width - 0.5 * lh, e1 + 6 * lh] + [m0 - 0.5 * lh for m0, _ in columns_measured if m0 >= e1 + lh])
                return e1 + 0.5 * lh, outer
            outer = max([0.5 * lh, e0 - 6 * lh] + [m1 + 0.5 * lh for _, m1 in columns_measured if m1 <= e0 - lh])
            return outer, e0 - 0.5 * lh

        def room(side: str) -> tuple[float, float]:
            probe = (e1 + 0.6 * lh, e1 + 0.7 * lh) if side == "right" else (e0 - 0.7 * lh, e0 - 0.6 * lh)
            return into_margin(*probe, line.box.y0, line.box.y1)

        if (z.id, side) in margin_of:
            mx0, mx1 = margin_of[(z.id, side)]
        else:
            # beside the column's text as printed at its height (that of the entry it follows, whose
            # lines it was put beside), as wide as the margin's ink: the zone's own edge may be the
            # text's (a note in its last eighth stood in the text, p. 705), or a band across the
            # page's columns (p. 630). A side with no room (the gutter between two columns): the
            # column's outer side
            mx0, mx1 = room(side)
            if mx1 - mx0 < 2 * lh and side != outside:
                side = outside
                mx0, mx1 = room(side)
            if mx1 - mx0 < 2 * lh:  # no ink beside it: the white beside the text all the same
                mx0, mx1 = white(side)
                if mx1 - mx0 < 2 * lh:
                    side = "left" if side == "right" else "right"
                    mx0, mx1 = white(side)
        margin_of.setdefault((z.id, side), (mx0, mx1))
        note_margin.setdefault((z.id, side), (mx0, mx1))
        y0, gap = line.box.y0, int(0.3 * lh) + 1
        while taken := [nt[3] for nt in inner_notes if nt[4] == z.id and nt[5] == side and nt[1] <= y0 < nt[3] + gap]:
            y0 = max(taken) + gap  # below the notes already there (each step moves down)
        line.box = Box(int(mx0), int(y0), int(mx1), int(y0 + 2.4 * lh))
        inner_notes.append((mx0, line.box.y0, mx1, line.box.y1, z.id, side))
        keep_clear.append(inner_notes[-1])

    for whole in items:
      group_all = [by_id[i] for i in whole.lines if i in by_id]
      if not group_all or not plain(whole.text).strip():
          continue
      parts = column_parts(whole, group_all, layout.line_height) if whole.kind in ("paragraph", "note") else [(whole, group_all)]
      for part_no, (item, group) in enumerate(parts):
        zone = zones.get(item.zone)
        if zone is None:
            continue
        first = min(group, key=lambda l: (l.box.y0, l.box.x0)) if part_no else group[0]
        top = first.box.y0 * px
        size0 = em_of(group)
        lead_words = _lead_words(item.text, min(group, key=lambda l: l.box.y0)) if item.kind == "note" and part_no == 0 else 0
        # the printed lines (Vision may read one in two pieces) and their pitch: the usual step
        # from one to the next, whatever order the lines were listed in
        rows: list[int] = []
        for y in sorted(l.box.y0 for l in group):
            if not rows or y - rows[-1] > 0.5 * layout.line_height:
                rows.append(y)
        n = len(rows)
        steps = sorted(b - a for a, b in zip(rows, rows[1:]))
        pitch = steps[len(steps) // 2] * px if steps else 1.2 * size0
        steps_pt = tuple(s * px for s in steps)
        if item.kind == "heading" and (len(group) == 1 or n == 1):
            # one line as printed (in one piece or several Vision read apart: "(10" and ")" of the
            # page number were set in a column-wide box, over the running head beside it, p. 35):
            # the box of its pieces, the size that spans it
            first = min(group, key=lambda l: l.box.x0)
            x0, w = first.box.x0 * px, (max(l.box.x1 for l in group) - first.box.x0) * px
            face = _times(item.italic)
            label = plain(item.text)
            size = min(1.1 * size0, w / max(face.text_length(label, 1.0), 0.1))
            spacing = 0.0
            if first.spacing > SPACED and first.glyph and len(label) > 2:
                # letter-spaced (D I C T I O N A R Y): the size of its glyphs, the rest of the
                # printed width between the letters
                letters = [c for c in label if c.isalpha()]
                caps = sum(c.isupper() for c in letters) >= 0.5 * max(len(letters), 1)
                em = first.glyph * px / (TYPEFACE.get("cap", TIMES_CAP) if caps else TYPEFACE.get("x", TIMES_X))
                natural = face.text_length(label, em)
                if natural < w:
                    size, spacing = em, (w - natural) / em / len(label)
            blocks.append(Block(item, x0, top, max(w, 1.0) * 1.02, size, 1.2 * size, "left", nowrap=True,
                                letter_spacing=spacing))
            continue
        if item.kind == "note" and zone.role != "notes":
            # a note the layout did not set apart (in a column's margin): as wide as its own words
            own = own_extent(item.text, group)
            gx0, gx1 = (own.x0, own.x1) if own else (min(l.box.x0 for l in group), max(l.box.x1 for l in group))
            if own:
                top = own.y0 * px
            side = "left" if (gx0 + gx1) / 2 < zone.box.x0 + 0.5 * zone.box.w else "right"
            if (zone.id, side) in margin_of and (own or gx1 - gx0 > 0.3 * zone.box.w):
                # the column's note margin (its words may be a fragment, or not on its line)
                gx0, gx1 = note_margin[(zone.id, side)]
            elif not own:
                # its words are not on its lines (the model read the note from the scan and gave
                # it a line of the text beside it): the margin the column's other notes stand in
                same = [nt for nt in inner_notes if zone.box.x0 - layout.line_height <= (nt[0] + nt[2]) / 2 <= zone.box.x1]
                if same:
                    gx0, _, gx1, _ = min(same, key=lambda nt: abs(nt[1] - first.box.y0))[:4]
            # its margin, never into a column's text
            gx0, gx1 = into_margin(gx0, gx1, top / px, max(l.box.y1 for l in group))
            x0, w = gx0 * px, (gx1 - gx0) * px
        else:
            left, right = zone.box.x0, zone.box.x1
            # the measure of its own lines: where most start, where the long ones end
            # (the pieces of each printed line joined: the layout may cut a band into narrow zones,
            # and assign() splits a line by them)
            ex0, ex1, n_rows = row_extent(group, layout.line_height, ragged=item.kind == "note")
            if item.kind == "paragraph":  # (the notes Vision ran into its lines are no part of it)
                ex0, ex1 = text_edges(ink, ex0, ex1, min(l.box.y0 for l in group), max(l.box.y1 for l in group),
                                      layout.line_height, n_rows, [w for l in group for w in l.words], item.text)
            measure = measure_for(ex0, ex1) if len(rows) >= 2 or zone.role in ("column", "text") else None
            if measure is not None:
                # the column's text measure, as its paragraphs are printed (its zone may take in a
                # note margin, or the gutter)
                left, right = measure
            # a measure from its lines needs two printed lines (one is often a fragment the OCR read
            # of a paragraph whose text the model read from the scan: a 57 px "in" on p. 995)
            one_line = len(rows) == 1 and zone.role in ("column", "text")
            # nor do lines that hold less than half its text (the model read the rest from the scan)
            read_elsewhere = len(plain(item.text)) > 2 * sum(len(l.text) for l in group) and zone.role in ("column", "text")
            # nor, in a column whose measure is known, lines wider than it (pieces of the next
            # column's line Vision ran into it: p. 165)
            too_wide = measure is not None and ex1 - ex0 > right - left
            if not (one_line or read_elsewhere or too_wide) and not (0.8 * (ex1 - ex0) <= right - left <= 1.6 * (ex1 - ex0)):
                # a zone that is no column of it: a speck labelled footnotes, footnotes set in
                # three columns inside a wide zone, a zone narrower than the text
                # (a verse centred in its column keeps the column's measure)
                left, right = ex0, ex0 + 1.02 * (ex1 - ex0)
            gy0, gy1 = min(l.box.y0 for l in group), max(l.box.y1 for l in group)
            # the column's note margins: kept free for all its text, as in print. (Not where most of
            # its own lines are printed: a note's extent taken into the text set the margin 130 pt
            # into the column, the paragraph beside it 25% narrower, p. 37. Its lines' middle start
            # and end, not where the few that a note ran into start.)
            row_x0, row_x1 = _row_ends(group, layout.line_height)
            # (within the text's own edges and its column's measure: where notes Vision ran into half
            # its lines end, p. 605)
            row_x0, row_x1 = max(row_x0, ex0, left), min(row_x1, ex1, right)
            if (zone.id, "left") in margin_of:
                m = margin_of[(zone.id, "left")][1] + 0.5 * layout.line_height
                if row_x0 >= m - layout.line_height:
                    left = max(left, m)
            if (zone.id, "right") in margin_of:
                m = margin_of[(zone.id, "right")][0] - 0.5 * layout.line_height
                if row_x1 <= m + layout.line_height:
                    right = min(right, m)
            # note strips the layout found inside the column's width count as its margin too (not
            # those its own lines stand in: the layout may cut a band of text into narrow strips)
            own_zones = {l.zone for l in group}
            strips = [(z.box.x0, z.box.y0, z.box.x1, z.box.y1) for z in layout.of("notes") if z.id not in own_zones]
            def clear_of_text(nt: tuple) -> bool:
                rest = [l for l in group if min(l.box.y1, nt[3]) <= max(l.box.y0, nt[1])]
                if not rest:
                    return False
                # (within the text's own edges, as above)
                if (nt[0] + nt[2]) / 2 < (left + right) / 2:
                    return sorted(max(l.box.x0, ex0, left) for l in rest)[len(rest) // 2] >= nt[2] - 0.5 * layout.line_height
                return sorted(min(l.box.x1, ex1, right) for l in rest)[len(rest) // 2] <= nt[0] + 0.5 * layout.line_height

            placed = [nt for nt in inner_notes if nt not in keep_clear and clear_of_text(nt)]
            for nx0, ny0, nx1, ny1, *_ in keep_clear + placed + strips:
                if min(ny1, gy1) - max(ny0, gy0) < 0.5 * layout.line_height or nx1 < left or nx0 > right:
                    continue  # not beside this paragraph
                # (and its lines mostly printed clear of it: a note's extent that took in a word of
                # the text beside it reached 130 pt into it, p. 37)
                if (nx0 + nx1) / 2 < left + 0.3 * (right - left):
                    if row_x0 >= nx1 - layout.line_height:
                        left = nx1 + 0.5 * layout.line_height
                elif (nx0 + nx1) / 2 > right - 0.3 * (right - left):
                    if row_x1 <= nx0 + layout.line_height:
                        right = nx0 - 0.5 * layout.line_height
            if right - left < 0.5 * zone.box.w and right - left < 0.5 * (ex1 - ex0):
                # not a margin: the notes took most of it (a column: its own measure)
                left, right = (zone.box.x0, zone.box.x1) if zone.role in ("column", "text") else (ex0, ex0 + 1.02 * (ex1 - ex0))
            if item.kind == "note":  # a note strip's note: its margin, not into the column's text either
                left, right = into_margin(left, right, gy0, gy1)
            x0, w = left * px, (right - left) * px
        text = item.text
        font = _times(item.italic)
        if part_no == 0 and item.drop_cap and plain(text).lstrip().startswith(item.drop_cap):
            lh = layout.line_height
            # the capital left of the paragraph's first line, or in it (Vision may read it as the
            # line's first letter)
            cap = next((c for c in layout.of("dropcap")
                        if c.box.y0 - lh <= first.box.y0 <= c.box.y1
                        and c.box.x0 - 2 * lh <= first.box.x0 <= c.box.x1 + 3 * lh), None)
            if cap is not None:
                # the printed lines beside the capital, and those below it
                cut = cap.box.y1 - 0.3 * layout.line_height
                k = max(1, len([y for y in rows if y < cut]))
                below = [l for l in group if l.box.y0 >= cut]
                cap_size = cap.box.h * px * 0.95
                # the scan's own capital (plain or ornamented), over the letter as invisible text
                blocks.append(Block(Item(item.zone, item.drop_cap, [], "dropcap"), cap.box.x0 * px, cap.box.y0 * px,
                                    cap.box.w * px * 1.1, cap_size, cap_size, "left", nowrap=True,
                                    picture=pictures.get(cap.id, ""), h=cap.box.h * px))
                if first.box.x0 < cap.box.x1:
                    # Vision's box of the first line takes in the capital: the line's own top is
                    # where the capital's top is (it stands on the first line's cap height)
                    top = cap.box.y0 * px - 0.28 * size0
                # the lines beside the capital, narrower, starting right of it, and the rest below at
                # the same size: the paragraph's size over its printed lines, the words beside the
                # capital as many as its lines hold at that size (Times is not the book's face, so
                # not always the words printed there)
                body = drop_first_letter(text, item.drop_cap)
                bx0 = cap.box.x1 * px + 0.3 * size0
                bw = x0 + w - bx0
                size = fit_size(plain(body), 0.97 * w, n, size0, font)
                count = words_fitting(body.split(), 0.97 * bw, size, font, k) if below else len(body.split())
                beside_text, rest_text = _split_at(body, count)
                part = dict(zone=item.zone, align=item.align, italic=item.italic)
                blocks.append(Block(Item(text=beside_text, lines=[l.id for l in group if l not in below], **part),
                                    bx0, top, bw, size, max(pitch, size), pitch=pitch, steps=steps_pt))
                if plain(rest_text).strip() and below:
                    rest = Block(Item(text=rest_text, lines=[l.id for l in below], **part),
                                 x0, min(l.box.y0 for l in below) * px, w, size, max(pitch, size), pitch=pitch,
                                 steps=steps_pt)
                    rest.fit = (n - k, size, pitch)  # sized with the others, to the room below it
                    blocks.append(rest)
                continue
        # the first-line indent: from its topmost line inside the block (Vision runs a line across
        # both columns; its piece from the other column is no indent: 717 pt in a 691 pt block
        # set the line beyond the block, p. 689), and no more than an indent can be
        inside = [l for l in group if x0 - size0 <= l.box.x0 * px <= x0 + w]
        lead = min(inside, key=lambda l: (l.box.y0, l.box.x0)) if inside else first
        indent = max(0.0, lead.box.x0 * px - x0) if item.kind == "paragraph" and part_no == 0 else 0.0
        indent = indent if 0.5 * size0 < indent <= min(6 * size0, 0.3 * w) else 0.0
        align = item.align if item.kind == "paragraph" else "left"
        if item.kind == "heading":
            align = item.align if item.align != "justify" else "left"
        block = Block(item, x0, top, w, size0, max(pitch, 1.0 * size0), align, indent=indent, pitch=pitch if n >= 2 else 0.0,
                      steps=steps_pt, lead=lead_words)
        block.fit = (n, size0, pitch)  # sized below, once the room down to the next block is known
        blocks.append(block)

    set_footnotes(blocks, by_id, layout.line_height, px)

    # how far down each text block may reach: the top of the next block below it that it would
    # run into (same horizontal span; footnotes and pictures are blocks too), else the page's end.
    # Not its zone's end: a paragraph may run on from one band of columns into the next. Not the
    # next margin note, for a note: that one moves down below it (the fit loop), as in print.
    regions = [b for b in blocks if b.item is not None]

    def margin_note(o: Block) -> bool:
        return o.item is not None and o.item.kind == "note" and "foot" not in o.item.label.lower()

    for b in regions:
        # (beside it, a few pt into its span, is not below it: a margin note set against a column)
        below = [o.y for o in blocks if o is not b and o.y > b.y + 0.5 * b.line_h
                 and min(b.x + b.w, o.x + o.w) - max(b.x, o.x) > 0.3 * max(b.size, 1.0)
                 and not (o.item is not None and o.item.kind == "dropcap")  # a capital stands beside
                 and not (margin_note(b) and margin_note(o))]
        b.limit = min([height_pt, *below])  # (the footnotes: the signature line under them)

    # size every paragraph to the room it has: as many lines (at the original pitch) as fit down
    # to the next block, at least as many as the scan had
    for b in regions:
        if b.fit is None:
            continue
        n, size0, pitch = b.fit
        room_lines = int((b.limit - b.y) / max(pitch, 1.0) + 0.15)
        b.size = fit_size(plain(b.item.text), 0.97 * b.w, max(n, room_lines), size0, _times(b.item.italic), b.indent)
        b.line_h = max(pitch, b.size)

    # the page's type sizes, by area: its text, its margin notes, its footnotes (Hebrew apart), each
    # at the median of its blocks measured in the scan (a single note measures 30 or 48 pt, by its
    # figures and numerals; an area's median is steady); areas within 6% are one size. Headings and
    # capitals keep their own.
    text_blocks = [b for b in blocks if b.item is not None and not b.nowrap and b.item.kind in ("paragraph", "note")]

    def role(b: Block) -> str:
        if is_footnote(b.item, zones):
            return "footnotes"
        return "notes" if b.item.kind == "note" else "text"

    area_of = {id(b): (role(b), _ratio_of(b.item.text) == HEBREW_LETTER) for b in text_blocks}
    measures: dict[tuple, list[tuple[float, int]]] = {}
    own: dict[int, tuple[float, int]] = {}
    for b in text_blocks:
        m = measured_em(b.item, [by_id[i] for i in b.item.lines if i in by_id], ink, px)
        if m:
            own[id(b)] = m
            measures.setdefault(area_of[id(b)], []).append((m[0], max(1, len(b.item.lines))))
    glyph_size = {area: _weighted_median(sorted(m)) for area, m in measures.items()}
    # the text (and the footnotes) at the size Times sets its paragraphs at in as many lines as
    # printed, when it has enough of them (its glyphs' size is a pixel off, 4%, and Times is wider than the book's
    # face); no larger than its glyphs (and a pixel), no smaller than 0.8 of them. (The margin
    # notes are ragged: their printed lines are as wide as Times sets them at their glyphs' size.)
    area_size = dict(glyph_size)
    for area in glyph_size:
        members = [b for b in text_blocks if area_of[id(b)] == area and b.item.kind == "paragraph"]
        steps = sorted(st for b in members for st in b.steps)
        if area == ("text", False):
            # (no larger than its glyphs and a pixel; in the book's own font, than its rows are
            # apart: that font's em is the body the type was cast on, its line pitch)
            cap = 1.04 * glyph_size[area]
            if TYPEFACE and len(steps) >= 3:
                cap = min(cap, steps[len(steps) // 2])
            by_wrap = wrap_size(members, cap, steps[len(steps) // 2] if steps else 0.0)
        elif area == ("footnotes", False):  # in its printed rows (they run on to the page's end)
            by_wrap = wrap_size(members, glyph_size[area], 0.0, least=1)
        else:
            by_wrap = None
        if by_wrap and by_wrap[0] >= 0.8 * glyph_size[area]:
            area_size[area] = by_wrap[0]
            for b in members:
                b.measure = "line breaks"
    # margin notes nothing measured (Vision read none of them: they had the size of a box of two
    # lines, the text's, p. 197): the book's notes are some four fifths of its text
    if ("text", False) in area_size and ("notes", False) not in area_size and ("notes", False) in area_of.values():
        area_size[("notes", False)] = NOTE_TEXT * area_size[("text", False)]
        measures[("notes", False)] = [(area_size[("notes", False)], 1)]
    # and the margin notes in a smaller type than the text, as the book's always are (0.65-0.88 of
    # it: measured on their capitals and figures they came out within 6% of the text and were set
    # in its size, p. 106; the book's median is 0.79)
    if ("text", False) in area_size and ("notes", False) in area_size:
        text_size = area_size[("text", False)]
        area_size[("notes", False)] = min(max(area_size[("notes", False)], 0.65 * text_size), 0.88 * text_size)
    # and no larger than its rows are apart (the book is set solid): footnotes measured 35 pt by
    # their capitals and figures stand in rows 28.5 pt apart (p. 155)
    for area in area_size:
        rows = sorted(st for b in text_blocks if area_of[id(b)] == area and b.pitch for st in b.steps)
        if len(rows) >= 3:
            area_size[area] = min(area_size[area], (1.0 if TYPEFACE else 1.02) * rows[len(rows) // 2])
    levels = size_levels([(size, sum(w for _, w in measures[area])) for area, size in area_size.items()], tol=0.06)
    own_size = set()
    for b in text_blocks:
        area = area_of[id(b)]
        if area in area_size:
            size = area_size[area]
        elif (area[0], False) in area_size:  # a Hebrew note with nothing measured: as the notes
            size = area_size[(area[0], False)]
        else:
            continue
        # a block of text measured on many letters that is set clearly smaller or larger than its
        # area (a verse quoted in smaller type) keeps its own size; a short reference does not
        # (its figures and numerals make its measure unsteady)
        glyphs = glyph_size.get(area) or glyph_size.get((area[0], False)) or size
        if id(b) in own and own[id(b)][1] >= 20 and abs(own[id(b)][0] / glyphs - 1) > 0.09:
            text = plain(b.item.text)
            if sum(c.isalpha() for c in text) >= 0.6 * len(text.replace(" ", "")):
                size = own[id(b)][0] * size / glyphs  # as much smaller as its area is by its lines
                b.measure = ""
                own_size.add(id(b))
                if all(abs(level / size - 1) > 0.06 for level in levels):
                    levels.append(size)
        b.size = b.level = min(levels, key=lambda level: abs(level - size))
        b.area = area[0] + (" (Hebrew)" if area[1] else "")
        if b.item.kind == "note" and not "foot" in b.item.label.lower():
            b.pitch = 0.0  # a margin note's line height follows its size (footnotes keep their rows')
    # and the page's line pitch, by area as its sizes: print sets an area solid at one pitch; a
    # paragraph's own two or three rows say little (one line Vision boxed 10 pt low gave a
    # paragraph 51 pt lines among 42 pt ones, p. 126). A block in a size of its own keeps its own,
    # and so does one measured on many rows that is set wider or closer.
    area_steps: dict[tuple, list[float]] = {}
    for b in text_blocks:
        if b.pitch and id(b) not in own_size:
            area_steps.setdefault(area_of[id(b)], []).extend(b.steps)
    for b in text_blocks:
        steps = sorted(area_steps.get(area_of[id(b)], []))
        if not b.pitch:
            continue
        if len(steps) < 3:
            # one or two steps between rows say nothing (two rows of footnotes 50 pt apart for
            # type of 35, p. 155): set solid, as the book is
            b.pitch = 0.0
            continue
        area_pitch = steps[len(steps) // 2]
        # its own: from its first row to its last, over its steps (a row Vision boxed high, 28 pt
        # from the one above and 54 from the one below, made the median 51 for rows 41 pt apart,
        # p. 30), or over as many of the area's as that is (a row Vision did not read)
        span = sum(b.steps)
        own = span / max(len(b.steps), round(span / area_pitch)) if b.steps else b.pitch
        if id(b) in own_size:
            # a size of its own (or a measure of its glyphs off by its capitals): its own rows
            b.pitch = own if b.steps else area_pitch * b.size / area_size.get(area_of[id(b)], b.size)
            continue
        # (its own, when it has rows enough: a scan is not to scale all over, a column near the
        # spine 40.1 pt a line where the page's median is 41.6, fourteen lines 20 pt longer, p. 51)
        if len(b.steps) >= 4:
            b.pitch = own
            continue
        b.pitch = area_pitch
    for b in blocks:
        _follow_size(b)
        b.start_size = b.size
        if b.lead >= 4 and b.item is not None:
            # a note whose top line the OCR read is not its first (only "fore the" of "In the Year of
            # the World 3291, before the vulgar Æra": it stood five lines low, over the note below,
            # p. 730): as many lines higher as its words before that line take
            before = " ".join(plain(b.item.text).split()[:b.lead])
            b.y = max(0.0, b.y - wrap_lines(before, b.w, b.size, _times(b.item.italic)) * b.line_h)
    return blocks


def blocks_html(blocks: list[Block], width_pt: float, height_pt: float, lang: str = "en") -> str:
    parts = []
    for b in blocks:
        if b.picture:
            w = b.w / 1.1 if b.item is not None else b.w  # a drop capital's box has room to spare
            parts.append(f'<img class="pic" src="{b.picture}" style="{_pos(b.x, b.y, w, height_pt, width_pt)} '
                         f'height: {100 * b.h / height_pt:.4f}%;">')
        if b.item is not None:
            parts.append(_block(b.item, b.x, b.y, b.w, b.size, b.line_h, b.align, width_pt, height_pt, b.indent, b.nowrap,
                                b.letter_spacing, invisible=bool(b.picture)))
    return f"""<!DOCTYPE html>
<html lang="{htmlmod.escape(lang)}">
<head>
<meta charset="utf-8">
<meta name="generator" content="pdf-ocr-bench reconstruct">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  .page {{ position: relative; width: {width_pt:.2f}pt; height: {height_pt:.2f}pt; overflow: hidden; }}
  .region {{ position: absolute; color: #000; font-family: {font_family()}; hyphens: auto; }}
  .region p {{ margin: 0; }}
  .region sup, .region sub {{ line-height: 0; }}
  .region .up {{ font-style: normal;{f" font-family: {font_family()};" if TYPEFACE else ""} }}{italic_css()}
  .pic {{ position: absolute; }}
</style>
</head>
<body>
<div class="page">
{chr(10).join(parts)}
</div>
</body>
</html>
"""


def italic_css() -> str:
    """With the book's fonts: <i> in its italic font (not slanted again)."""
    if not TYPEFACE:
        return ""
    return f"\n  .region i {{ font-family: {font_family(True)}; font-style: normal; }}"


def _split_text(text: str, k: int, group: list[Line]) -> tuple[str, str]:
    """The words of the first `k` OCR lines, and the rest (by the OCR lines' word counts)."""
    return _split_at(text, sum(len(l.text.split()) for l in group[:k]))


def _split_at(text: str, n: int) -> tuple[str, str]:
    """The first `n` words of `text` and the rest; marks open across the split are closed in the
    first part and opened again in the second."""
    words = clean_marks(text).split()
    first, rest = " ".join(words[:n]), " ".join(words[n:])
    still_open = _open_marks(first)
    first += "".join(f"</{t}>" for t in reversed(still_open))
    rest = "".join(f"<{t}>" for t in still_open) + rest
    return clean_marks(first), clean_marks(rest)


def words_fitting(words: list[str], width_pt: float, size: float, font, n_lines: int) -> int:
    """How many of `words` fill `n_lines` lines of `width_pt` at `size` (as `wrap_lines` sets them)."""
    space = font.text_length(" ", size)
    lines, x = 1, 0.0
    for i, word in enumerate(words):
        w = font.text_length(plain(word), size)
        if x > 0 and x + space + w > width_pt:
            lines, x = lines + 1, w
            if lines > n_lines:
                return i
        else:
            x = x + (space if x > 0 else 0) + w
    return len(words)


# --- the other style inside a text: <i>...</i> ------------------------------------------------

MARKS = ("i", "sup", "sub")
MARK = re.compile(r"(</?(?:i|sup|sub)>)")


def plain(text: str) -> str:
    """`text` without its marks."""
    return MARK.sub("", text)


TAG = re.compile(r"<(/?)([a-zA-Z]+)[^<>]*>")


def normalize_markup(text: str) -> str:
    """A model's answer as text with <i>, <sup> and <sub> marks: <em> is <i>, other tags are
    dropped with their text kept."""
    def tag(m: re.Match) -> str:
        name = {"em": "i"}.get(m.group(2).lower(), m.group(2).lower())
        if name in MARKS:
            return f"<{m.group(1)}{name}>"
        return " " if name == "br" else ""
    return TAG.sub(tag, text)


def _open_marks(text: str) -> list[str]:
    """The marks still open at the end of balanced-so-far `text`, outermost first."""
    stack: list[str] = []
    for part in MARK.split(text):
        if part.startswith("</") and part[2:-1] in stack:
            stack = stack[:len(stack) - 1 - stack[::-1].index(part[2:-1])]
        elif part.startswith("<") and part[1:-1] in MARKS:
            stack.append(part[1:-1])
    return stack


def clean_marks(text: str) -> str:
    """Marks balanced (a close without its open dropped; an open never closed closed at the end;
    nesting kept, <sup><i>1</i></sup>), no empty pairs; anything else is text."""
    out, stack = [], []
    for part in MARK.split(text):
        if part.startswith("</") and part[2:-1] in MARKS:
            name = part[2:-1]
            if name not in stack:
                continue
            while stack:  # close the marks opened inside it as well
                top = stack.pop()
                out.append(f"</{top}>")
                if top == name:
                    break
        elif part.startswith("<") and part[1:-1] in MARKS:
            if part[1:-1] in stack:
                continue  # <i> inside <i>: one italic
            stack.append(part[1:-1])
            out.append(part)
        else:
            out.append(part)
    out += [f"</{t}>" for t in reversed(stack)]
    joined = "".join(out)
    empty = re.compile(r"<(i|sup|sub)></\1>")
    while empty.search(joined):
        joined = empty.sub("", joined)
    return joined


def drop_first_letter(text: str, letter: str) -> str:
    """`text` without its first letter `letter` (a drop capital), marks kept."""
    m = re.match(r"((?:\s|</?(?:i|sup|sub)>)*)", text)
    head, rest = m.group(1), text[m.end():]
    return clean_marks(head.strip() + rest[len(letter):].lstrip()) if rest.startswith(letter) else text


def marked_html(text: str, italic: bool) -> str:
    """Escaped text with its marks as markup: <i> in upright text, in italic text the marked
    words are the upright ones; <sup>, <sub> as they are."""
    tags = {"<sup>": "<sup>", "</sup>": "</sup>", "<sub>": "<sub>", "</sub>": "</sub>",
            "<i>": '<span class="up">' if italic else "<i>", "</i>": "</span>" if italic else "</i>"}
    return "".join(tags.get(part, htmlmod.escape(part)) for part in MARK.split(clean_marks(text)))


def _pos(x: float, y: float, w: float, height_pt: float, width_pt: float) -> str:
    return f"left: {100 * x / width_pt:.4f}%; top: {100 * y / height_pt:.4f}%; width: {100 * w / width_pt:.4f}%;"


def _block(item: Item, x: float, y: float, w: float, size: float, line_h: float, align: str,
           width_pt: float, height_pt: float, indent: float = 0.0, nowrap: bool = False,
           letter_spacing: float = 0.0, invisible: bool = False) -> str:
    style = _pos(x, y, w, height_pt, width_pt) + f" font-size: {size:.2f}pt; line-height: {line_h:.2f}pt; text-align: {align};"
    if item.italic:
        # (the book's italic is a font of its own: not slanted again)
        style += f" font-family: {font_family(True)};" if TYPEFACE else " font-style: italic;"
    if nowrap:
        style += " white-space: nowrap;"
    if letter_spacing:
        style += f" letter-spacing: {letter_spacing:.3f}em;"
    if invisible:  # the text of a letter drawn as a picture: there to be found and copied
        style += " color: rgba(0, 0, 0, 0);"
    p_style = f' style="text-indent: {indent:.1f}pt;"' if indent else ""
    attrs = f'data-zone="{item.zone}" data-role="{item.kind}"' + (f' data-lines="{" ".join(item.lines)}"' if item.lines else "")
    if item.italic and TYPEFACE:
        attrs += ' data-italic="1"'  # (html2pdf's long-s repair reads italic words apart)
    return f'<div class="region" {attrs} style="{style}"><p{p_style}>{marked_html(item.text, item.italic)}</p></div>'


# --- the pipeline ----------------------------------------------------------------------------


HTML2PDF = Path(__file__).resolve().parents[2] / "html2pdf" / "target" / "release" / "html2pdf"
WORD_LIST = "/usr/share/dict/words"
MIN_FIT = 0.7  # the fit loop sets a block at most this much smaller than it started


def _write_zip(target: Path, metadata: dict, pages_meta: list[dict], out: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        for p in pages_meta:
            z.write(out / p["html"], p["html"])
        used = set()  # the pictures the pages show (a capital no paragraph opens with is left out)
        for p in pages_meta:
            used.update(re.findall(r'<img class="pic" src="(pictures/[^"]+)"', (out / p["html"]).read_text(encoding="utf-8")))
        for rel in sorted(used):
            if (out / rel).exists():
                z.write(out / rel, rel)


def fit_round(target: Path, page_blocks: dict[int, list[Block]], pages_meta: list[dict], work: Path) -> dict[int, int]:
    """Render with html2pdf, and set smaller every block whose text runs past its limit.

    The layout engine wraps a little differently than the Helvetica estimate; its rendered glyph
    positions (html2pdf --layout-report) say by how much.
    """
    report_path = work / "layout-report.json"
    subprocess.run([str(HTML2PDF), str(target), "-o", str(work / "fit.pdf"), "--layout-report", str(report_path), *html2pdf_fonts()],
                   check=True, capture_output=True)
    reports = {r["page"]: r for r in json.loads(report_path.read_text())}
    shrunk: dict[int, int] = {}  # page number: blocks set smaller
    for p in pages_meta:
        report = reports.get(p["html"])
        if report is None:
            continue
        regions = [b for b in page_blocks[p["page_num"]] if b.item is not None]
        drawn = [m.get("rendered") for m in report["regions"]]
        # (by size and area: margin notes of the text's size that run over do not set the text smaller)
        groups: dict[tuple[float, str], list[tuple[Block, float]]] = {}
        # a paragraph that runs a line or two into the block below it in its column moves that one
        # down (when it still fits above what follows it), as print would: the type stays the
        # page's size, the text a line or two lower than printed
        flowing = [(b, d) for b, d in zip(regions, drawn) if d and b.item.kind in ("paragraph", "heading")
                   and "foot" not in b.item.label.lower()]
        for a, da in sorted(flowing, key=lambda t: t[0].y):
            # the rest of its column below it: those move down together, if the column's last
            # block still ends above what follows it (the footnotes, the page's end)
            chain = sorted([(b, db) for b, db in flowing if b is not a and b.y > a.y + 0.5 * a.line_h
                            and min(a.x + a.w, b.x + b.w) - max(a.x, b.x) > 0.5 * min(a.w, b.w)], key=lambda t: t[0].y)
            if not chain:
                continue
            over = da["y1"] - chain[0][0].y
            if not 0.3 * a.line_h < over < 3 * a.line_h:
                continue
            shift = over + 0.15 * chain[0][0].size
            last, dlast = max(chain, key=lambda t: t[1]["y1"])
            if dlast["y1"] + shift <= min(last.limit, report["height_pt"]) + 0.2 * last.size:
                for b, _ in chain:
                    b.y += shift
                    b.limit = max(b.limit, b.limit + shift) if b is not last else b.limit
                a.limit = max(a.limit, chain[0][0].y)
                shrunk[p["page_num"]] = shrunk.get(p["page_num"], 0) + len(chain)
        for k, (block, measured) in enumerate(zip(regions, report["regions"])):
            rendered = measured.get("rendered")
            if not rendered:
                continue
            if block.item.kind == "note":
                # a margin note that runs into the note above it moves down below it, as in print
                above = [drawn[j]["y1"] for j in measured.get("overlaps", [])
                         if j < len(regions) and regions[j].item.kind == "note" and drawn[j]
                         and (regions[j].y < block.y or (regions[j].y == block.y and j < k))]
                # (no further than its text still fits above what is below it, the footnotes or the
                # page's end; else it is set smaller below)
                lowest = min(block.limit, report["height_pt"]) - (rendered["y1"] - block.y) - 0.2 * block.size
                if above and block.y + 0.5 < max(above) + 0.2 * block.size <= lowest:
                    block.y = max(above) + 0.2 * block.size
                    shrunk[p["page_num"]] = shrunk.get(p["page_num"], 0) + 1
                    if block.level:
                        groups.setdefault((block.level, block.area), []).append((block, 1.0))
                    continue
                room = min(block.limit, report["height_pt"]) - (max(above, default=0.0) + 0.2 * block.size) - 0.3 * block.line_h
                if above and block.y + 0.5 < max(above) + 0.2 * block.size and room > block.line_h:
                    # no room below it for all of it (a margin of notes as long as the page, p. 245):
                    # below the note above all the same, and this one set smaller, to the room left
                    # (one note a step smaller rather than two notes one over the other)
                    took = max(rendered["y1"] - block.y, 1.0)
                    block.y = max(above) + 0.2 * block.size
                    factor = max(0.85, min(0.97, room / took))
                    if block.size * factor >= MIN_FIT * block.start_size:
                        block.size *= factor
                        _follow_size(block)
                    shrunk[p["page_num"]] = shrunk.get(p["page_num"], 0) + 1
                    continue
                end = min(block.limit, report["height_pt"])
                wide = rendered["x1"] > block.x + block.w + 0.25 * block.size
                if not above and (wide or rendered["y1"] > end - 0.1 * block.line_h) and end - block.y - 0.3 * block.line_h > block.line_h:
                    # and one so moved that runs past the page's end (or the footnotes), or one with a
                    # word wider than its margin ("Mechilta." into the paragraph beside it, p. 106),
                    # a step smaller still, by itself (its group's other notes have their room)
                    factor = (end - block.y - 0.3 * block.line_h) / max(rendered["y1"] - block.y, 1.0)
                    if wide:
                        factor = min(factor, block.w / max(rendered["x1"] - block.x, 1.0))
                    factor = max(0.85, min(0.97, factor))
                    if block.size * factor >= MIN_FIT * block.start_size:
                        block.size *= factor
                        _follow_size(block)
                        shrunk[p["page_num"]] = shrunk.get(p["page_num"], 0) + 1
                    continue
            ratio = 1.0
            # text may reach half a line into the next block (its first line's OCR box is a few
            # px higher or lower than where its line box begins), never past the page
            slack = 0.0 if block.limit >= report["height_pt"] - 1 else 0.5 * block.line_h
            if rendered["y1"] > block.limit + slack:
                # fewer, smaller lines (a smaller heading line): the ratio of the room to what it took
                took = max(rendered["y1"] - block.y, 1.0)
                room = max(block.limit - block.y, 0.5 * block.line_h)
                ratio = room / took
            if rendered["x1"] > block.x + block.w + 0.25 * block.size:
                # a line wider than its box: a heading, or a word longer than a note's line
                ratio = min(ratio, block.w / max(rendered["x1"] - block.x, 1.0))
            if "foot" in block.item.label.lower() and rendered["y1"] - rendered["y0"] < 1.6 * block.line_h:
                # a footnote of one line runs into the next one on its row
                start = block.x + block.indent
                after = [o.x + o.indent for o in regions if o is not block and "foot" in o.item.label.lower()
                         and abs(o.y - block.y) < 0.5 * block.line_h and o.x + o.indent > start + block.size]
                if after and rendered["x1"] > min(after) - 0.3 * block.size:
                    ratio = min(ratio, (min(after) - 0.5 * block.size - start) / max(rendered["x1"] - start, 1.0))
            note = block.item.kind == "note" and "foot" not in block.item.label.lower()
            if not "foot" in block.item.label.lower() and any(
                    regions[j].y > block.y and not (note and regions[j].item.kind == "note") for j in measured.get("overlaps", [])
                    if j < len(regions)):
                # its text reaches into the text of a block below (a heading's descenders): a step
                # smaller (a note into the note below: that one moves down)
                ratio = min(ratio, 0.97)
            if block.level:
                groups.setdefault((block.level, block.area), []).append((block, ratio))
                continue
            if ratio >= 1.0:
                continue
            factor = max(0.85, min(0.97, ratio))  # a bit less than the ratio, at most 15% a round
            if block.size * factor < MIN_FIT * block.start_size:
                continue  # it does not fit at any readable size (its room is wrong): leave it
            block.size *= factor
            if block.nowrap:
                block.line_h *= factor  # one line: its line box is the text's
            _follow_size(block)
            shrunk[p["page_num"]] = shrunk.get(p["page_num"], 0) + 1
        # a type size of an area of the page is set smaller as a whole, when a fifth of its blocks run over
        # (one block that does not fit at all keeps its size: its room is wrong, not the size)
        for (level, _), members in groups.items():
            ratios = sorted(r for _, r in members)
            # (two blocks at least: in a group of five the fifth was any one, and one paragraph
            # whose room is wrong set the page's text 28% smaller, p. 280)
            r20 = ratios[min(len(ratios) - 1, max(1, int(0.2 * (len(ratios) - 1))))]
            if r20 >= 1.0:
                continue
            factor = max(0.85, min(0.97, r20))
            if members[0][0].size * factor < MIN_FIT * members[0][0].start_size:
                continue
            for b, _ in members:
                b.size *= factor
                _follow_size(b)
            shrunk[p["page_num"]] = shrunk.get(p["page_num"], 0) + len(members)
    return shrunk


# one per worker process: the open PDF and the Vision engine
_WORKER: dict = {}


def prep_key(pdf_path: Path, lang: tuple[str, ...]) -> str:
    """What steps 1-4 of a page depend on: the PDF, the languages, and the code of those steps
    (not the code that sets the page, which changes more often)."""
    import hashlib
    import inspect

    here = Path(__file__).resolve().parent
    h = hashlib.sha1(f"{pdf_path.resolve()}|{pdf_path.stat().st_size}|{lang}".encode())
    h.update((here / "page_layout.py").read_bytes())
    for step in (prepare_page, native_dpi, render, vision_lines, ocr_coverage, zone_lines, recover_missed,
                 measure_glyphs, assign, clip_pictures, Line):
        h.update(inspect.getsource(step).encode())
    return h.hexdigest()


def prepare_page(pdf_path: Path, n: int, out: Path, lang: tuple[str, ...], key: str = "") -> dict:
    """Steps 1-4 for one page (in a worker process): renders, layout, Vision's lines (and the ones
    it skipped, read again from strips), the pictures. Writes work/page_NNN/, and keeps the result
    (work/page_NNN/prepared.pkl) for the next run with the same `key`."""
    import pickle

    import pypdfium2 as pdfium

    cache = out / "work" / f"page_{n + 1:03d}" / "prepared.pkl"
    if key and cache.exists():
        try:
            saved = pickle.loads(cache.read_bytes())
            if saved.get("key") == key and all((out / rel).exists() for rel in saved["result"]["pictures"].values()):
                return {**saved["result"], "seconds": 0.0, "cached": True}
        except Exception:
            pass  # unreadable: prepared again

    if _WORKER.get("path") != pdf_path:
        from .engines import ENGINES
        from .engines.base import Route

        if "engine" not in _WORKER:
            _WORKER["engine"] = ENGINES["macos_vision"](Route(lang=lang))
            _WORKER["engine"].prepare()
        _WORKER.update(path=pdf_path, pdf=pdfium.PdfDocument(str(pdf_path)))
    engine = _WORKER["engine"]
    t = time.perf_counter()
    name = f"page_{n + 1:03d}"
    pw = out / "work" / name
    pw.mkdir(parents=True, exist_ok=True)
    page = _WORKER["pdf"][n]
    width_pt, height_pt = page.get_size()
    dpi = native_dpi(page)
    native = render(page, dpi, pw / "native.png")
    render(page, 0.75 * dpi, pw / "vision.png")
    page.close()
    layout = analyse(pw / "native.png")
    draw(pw / "native.png", layout, pw / "layout.png", scale=0.25)
    read = vision_lines(pw / "vision.png", lang, native, engine)
    if ocr_coverage(read, layout) < 0.9:
        # Vision read the page whole only in part: its zones one by one, if that reads more
        by_zone = zone_lines(layout, pw / "vision.png", engine, pw)
        if ocr_coverage(by_zone, layout) > ocr_coverage(read, layout):
            read = by_zone
    recovered = recover_missed(read, layout, pw / "vision.png", engine, pw)
    lines = assign(read + recovered, layout)
    measure_glyphs(lines, pw / "native.png")
    for old in (out / "pictures").glob(f"{name}_*.png"):  # from an earlier run's layout
        old.unlink()
    pictures = clip_pictures(pw / "native.png", layout, out, name)
    (pw / "layout.json").write_text(json.dumps(layout.to_dict(), indent=1))
    (pw / "lines.json").write_text(json.dumps([{**asdict(l), "box": asdict(l.box), "words": None} for l in lines], indent=1, ensure_ascii=False))
    result = {"name": name, "layout": layout, "lines": lines, "pictures": pictures, "size": (width_pt, height_pt),
              "dpi": dpi, "recovered": len(recovered), "seconds": time.perf_counter() - t}
    if key:
        cache.write_bytes(pickle.dumps({"key": key, "result": result}))
    return result


def scan_only_page(pdf_path: Path, n: int, out: Path) -> dict:
    """A page the pipeline could not rebuild: the whole scan as one picture."""
    import pypdfium2 as pdfium

    name = f"page_{n + 1:03d}"
    pw = out / "work" / name
    pw.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[n]
        width_pt, height_pt = page.get_size()
        dpi = native_dpi(page)
        width, height = render(page, dpi, pw / "native.png")
    finally:
        pdf.close()
    layout = PageLayout(width=width, height=height, line_height=30.0)
    layout.zones = [Zone("picture", Box(0, 0, width, height), 0)]
    for old in (out / "pictures").glob(f"{name}_*.png"):
        old.unlink()
    pictures = clip_pictures(pw / "native.png", layout, out, name)
    return {"name": name, "layout": layout, "lines": [], "pictures": pictures, "size": (width_pt, height_pt),
            "dpi": dpi, "recovered": 0, "seconds": 0.0}


def reconstruct(pdf_path: Path, out: Path, pages: list[int], lang: tuple[str, ...], html_lang: str,
                semantic_context: str = "", llm: bool = True, model: str = "sonnet", agents: int = 4,
                fit_rounds: int = 10, zoom: bool = False, guide: str = "", thinking: bool = True,
                workers: int = 4, redo: set[int] | None = None, fresh_layout: bool = False,
                typeface: Path | None = None) -> Path:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    use_typeface(typeface)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict] = {}
    start = time.perf_counter()
    structurer = None
    if llm:
        from .llm_structure import Structurer

        # each page goes to the model as soon as it is read, while the workers read the next ones
        structurer = Structurer(work, semantic_context, model, agents, zoom, guide, thinking)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        key = "" if fresh_layout else prep_key(pdf_path, lang)
        jobs = {pool.submit(prepare_page, pdf_path.resolve(), n, out, lang, key): n for n in pages}
        for done, job in enumerate(as_completed(jobs), 1):
            n = jobs[job]
            try:
                r = results[n] = job.result()
            except Exception as exc:  # one page must not stop a volume: it is set as its scan
                log.warning(f"[{done}/{len(pages)}] page_{n + 1:03d}: {type(exc).__name__}: {exc}; set as the scanned image")
                results[n] = scan_only_page(pdf_path, n, out)
                continue
            again = f" ({r['recovered']} read again from a strip)" if r["recovered"] else ""
            took = "as before" if r.get("cached") else f"{r['seconds']:.1f}s"
            log.info(f"[{done}/{len(pages)}] {r['name']}: {r['dpi']:.0f} dpi, {len(r['layout'].zones)} zones, "
                     f"{len(r['lines'])} lines{again}, {len(r['pictures'])} pictures, {took}")
            if structurer is not None:
                structurer.submit(n, r, fresh=n in (redo or ()))

    structures: dict[int, list[Item]] = structurer.collect() if structurer is not None else {}
    for n, r in results.items():
        if n not in structures:
            structures[n] = heuristic_structure(r["lines"], r["layout"])

    pages_meta, page_blocks = [], {}
    for n, r in sorted(results.items()):
        width_pt, height_pt = r["size"]
        page_blocks[n] = build_blocks(structures[n], r["lines"], r["layout"], r["pictures"], width_pt, height_pt,
                                      ink=_ink(work / r["name"] / "native.png"))
        pages_meta.append({"page_num": n, "html": f"{r['name']}.html", "width_pt": width_pt, "height_pt": height_pt,
                           "width_px": r["layout"].width, "height_px": r["layout"].height})
    metadata = {"engine": "reconstruct" + ("+" + model if llm else ""), "page_count": len(pages_meta), "pages": pages_meta}
    target = out / "pages.zip"
    # a page whose blocks are as in the last run gets the sizes the fit loop found then
    import hashlib

    import inspect

    fit_code = inspect.getsource(globals()["fit_round"]) + inspect.getsource(_follow_size) + TYPEFACE.get("key", "")  # sizes found by other code (or fonts) are no use
    start_html = {}
    todo = []  # the pages to (re)write and fit: those not fitted before, then those set smaller
    for p in pages_meta:
        blocks = page_blocks[p["page_num"]]
        start_html[p["page_num"]] = hashlib.sha1((fit_code + blocks_html(blocks, p["width_pt"], p["height_pt"], html_lang)).encode()).hexdigest()
        fitted = work / p["html"].removesuffix(".html") / "fit.json"
        saved = json.loads(fitted.read_text()) if fitted.exists() else {}
        sizes = saved.get("sizes", [])
        if saved.get("key") == start_html[p["page_num"]] and len(sizes) == len(blocks) and (out / p["html"]).exists():
            for b, (size, line_h) in zip(blocks, sizes):
                b.size, b.line_h = size, line_h
        else:
            todo.append(p)
    if len(todo) < len(pages_meta):
        log.info(f"{len(pages_meta) - len(todo)} pages are set as fitted before; {len(todo)} to fit")
    for round_ in range(fit_rounds + 1):
        for p in todo:
            width_pt, height_pt = p["width_pt"], p["height_pt"]
            (out / p["html"]).write_text(blocks_html(page_blocks[p["page_num"]], width_pt, height_pt, html_lang), encoding="utf-8")
        if round_ == fit_rounds or not HTML2PDF.exists() or not todo:
            break
        fit_zip = work / "fit.zip"
        _write_zip(fit_zip, {**metadata, "page_count": len(todo), "pages": todo}, todo, out)
        changed = fit_round(fit_zip, page_blocks, todo, work)
        log.info(f"Fit round {round_ + 1}: {sum(changed.values())} blocks on {len(changed)} of {len(todo)} pages "
                 "ran past the next block and were set smaller")
        todo = [p for p in pages_meta if p["page_num"] in changed]
    for p in pages_meta:
        (work / p["html"].removesuffix(".html") / "fonts.json").write_text(json.dumps(
            page_fonts(page_blocks[p["page_num"]], results[p["page_num"]]["layout"], p["width_pt"]), indent=1))
        (work / p["html"].removesuffix(".html") / "fit.json").write_text(json.dumps(
            {"key": start_html[p["page_num"]], "sizes": [[b.size, b.line_h] for b in page_blocks[p["page_num"]]]}))
    _write_zip(target, metadata, pages_meta, out)
    log.info(f"Wrote {target} ({len(pages_meta)} pages) in {time.perf_counter() - start:.0f}s")
    if HTML2PDF.exists():
        pdf_out, report = out / f"{out.resolve().name}.pdf", out / "layout-report.json"
        cmd = [str(HTML2PDF), str(target), "-o", str(pdf_out), "--layout-report", str(report), *html2pdf_fonts()]
        if html_lang.split("-")[0] == "en" and Path(WORD_LIST).exists():
            # f the model still read for a long s ("addrefs"), from the word list
            cmd += ["--long-s", "careful", "--dict", WORD_LIST]
        done = subprocess.run(cmd, capture_output=True, text=True)
        for line in done.stdout.splitlines() + done.stderr.splitlines():
            if "Layout report" in line or "Long s" in line or "Wrote" in line or "error" in line.lower():
                log.info(line.removeprefix("[html2pdf] "))
    return target
