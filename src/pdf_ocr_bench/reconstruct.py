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
from dataclasses import asdict, dataclass, field
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
            out.append(Line(line.id if len(parts) == 1 else f"{line.id}{'abcdefgh'[k]}",
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
        for z in layout.of("picture"):
            name = f"pictures/{page_name}_{z.index}.png"
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


def _times(italic: bool = False):
    import pymupdf

    return pymupdf.Font("tiit" if italic else "tiro")  # Times-Italic / Times-Roman metrics


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


def build_blocks(items: list[Item], lines: list[Line], layout: PageLayout, pictures: dict[str, str],
                 width_pt: float, height_pt: float) -> list[Block]:
    font = _times()
    px = width_pt / layout.width  # pt per native px
    by_id = {l.id: l for l in lines}
    zones = {z.id: z for z in layout.zones}

    def em_of(group: list[Line]) -> float:
        # a Vision line box is cap height + descender, ~1.05 em for this kind of face
        hs = sorted(l.box.h for l in group)
        return hs[len(hs) // 2] * px / 1.05 if hs else 10.0

    blocks: list[Block] = []
    for zid, rel in pictures.items():
        b = zones[zid].box
        blocks.append(Block(None, b.x0 * px, b.y0 * px, b.w * px, picture=rel, h=b.h * px))

    for item in items:
        zone = zones.get(item.zone)
        group = [by_id[i] for i in item.lines if i in by_id]
        if zone is None or not group or not item.text.strip():
            continue
        first = group[0]
        top = first.box.y0 * px
        size0 = em_of(group)
        n = max(1, len(group))
        pitch = ((group[-1].box.y0 - group[0].box.y0) / (n - 1) * px) if n > 1 else 1.2 * size0
        if item.kind == "heading" and len(group) == 1:
            # one line as printed: its own box, the size that spans it
            x0, w = first.box.x0 * px, first.box.w * px
            size = min(1.1 * size0, w / max(font.text_length(item.text, 1.0), 0.1))
            blocks.append(Block(item, x0, top, max(w, 1.0) * 1.02, size, 1.2 * size, "left", nowrap=True))
            continue
        x0, w = zone.box.x0 * px, zone.box.w * px
        text = item.text
        if item.drop_cap and text.startswith(item.drop_cap):
            lh = layout.line_height
            cap = next((c for c in layout.of("dropcap")
                        if c.box.y0 - lh <= first.box.y0 <= c.box.y1
                        and first.box.x0 - 3 * lh <= c.box.x1 <= first.box.x0 + lh), None)
            if cap is not None:
                k = max(1, len([l for l in group if l.box.y0 < cap.box.y1 - 0.3 * layout.line_height]))
                cap_size = cap.box.h * px * 0.95
                blocks.append(Block(Item(item.zone, item.drop_cap, [], "dropcap"), cap.box.x0 * px, cap.box.y0 * px,
                                    cap.box.w * px * 1.1, cap_size, cap_size, "left", nowrap=True))
                # the lines beside the capital: narrower, starting right of it
                beside_text, rest_text = _split_text(text[len(item.drop_cap):].lstrip(), k, group)
                bx0 = cap.box.x1 * px + 0.3 * size0
                bw = x0 + w - bx0
                blocks.append(Block(Item(item.zone, beside_text, item.lines[:k]), bx0, top, bw,
                                    fit_size(beside_text, 0.97 * bw, k, size0, font), pitch))
                rest = group[k:]
                if rest_text and rest:
                    blocks.append(Block(Item(item.zone, rest_text, item.lines[k:]), x0, rest[0].box.y0 * px, w,
                                        fit_size(rest_text, 0.97 * w, len(rest), size0, font), pitch))
                continue
        indent = max(0.0, (first.box.x0 - zone.box.x0) * px) if item.kind == "paragraph" else 0.0
        indent = indent if indent > 0.5 * size0 else 0.0
        align = item.align if item.kind == "paragraph" else "left"
        if item.kind == "heading":
            align = item.align if item.align != "justify" else "left"
        block = Block(item, x0, top, w, size0, max(pitch, 1.0 * size0), align, indent=indent)
        block.fit = (n, size0, pitch)  # sized below, once the room down to the next block is known
        blocks.append(block)

    # how far down each text block may reach: the top of the next block below it that it would
    # run into (same horizontal span; footnotes and pictures are blocks too), else the page's end.
    # Not its zone's end: a paragraph may run on from one band of columns into the next.
    regions = [b for b in blocks if b.item is not None]
    for b in regions:
        below = [o.y for o in blocks if o is not b and o.y > b.y + 0.5 * b.line_h
                 and o.x < b.x + b.w and b.x < o.x + o.w]
        b.limit = min([height_pt, *below])

    # size every paragraph to the room it has: as many lines (at the original pitch) as fit down
    # to the next block, at least as many as the scan had
    for b in regions:
        if b.fit is None:
            continue
        n, size0, pitch = b.fit
        room_lines = int((b.limit - b.y) / max(pitch, 1.0) + 0.15)
        b.size = fit_size(b.item.text, 0.97 * b.w, max(n, room_lines), size0, _times(b.item.italic), b.indent)
        b.line_h = max(pitch, b.size)

    # one size per zone: the median of its paragraphs' fitted sizes, a paragraph that needs a
    # smaller one keeps it; notes all at the median note size (capped by their own fit)
    for zone_id in {b.item.zone for b in blocks if b.item is not None}:
        mine = [b for b in blocks if b.item is not None and b.item.zone == zone_id and not b.nowrap]
        if len(mine) > 1:
            common = sorted(b.size for b in mine)[len(mine) // 2]
            for b in mine:
                b.size = min(b.size, common)
    notes = [b for b in blocks if b.item is not None and b.item.kind == "note"]
    if notes:
        common = sorted(b.size for b in notes)[len(notes) // 2]
        for b in notes:
            b.size = min(b.size, common)
            b.line_h = 1.15 * b.size
    return blocks


def blocks_html(blocks: list[Block], width_pt: float, height_pt: float, lang: str = "en") -> str:
    parts = []
    for b in blocks:
        if b.picture:
            parts.append(f'<img class="pic" src="{b.picture}" style="{_pos(b.x, b.y, b.w, height_pt, width_pt)} '
                         f'height: {100 * b.h / height_pt:.4f}%;">')
        else:
            parts.append(_block(b.item, b.x, b.y, b.w, b.size, b.line_h, b.align, width_pt, height_pt, b.indent, b.nowrap))
    return f"""<!DOCTYPE html>
<html lang="{htmlmod.escape(lang)}">
<head>
<meta charset="utf-8">
<meta name="generator" content="pdf-ocr-bench reconstruct">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  .page {{ position: relative; width: {width_pt:.2f}pt; height: {height_pt:.2f}pt; overflow: hidden; }}
  .region {{ position: absolute; color: #000; font-family: {FONT_FAMILY}; }}
  .region p {{ margin: 0; }}
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


def _split_text(text: str, k: int, group: list[Line]) -> tuple[str, str]:
    """The words of the first `k` OCR lines, and the rest (by the OCR lines' word counts)."""
    n = sum(len(l.text.split()) for l in group[:k])
    words = text.split()
    return " ".join(words[:n]), " ".join(words[n:])


def _pos(x: float, y: float, w: float, height_pt: float, width_pt: float) -> str:
    return f"left: {100 * x / width_pt:.4f}%; top: {100 * y / height_pt:.4f}%; width: {100 * w / width_pt:.4f}%;"


def _block(item: Item, x: float, y: float, w: float, size: float, line_h: float, align: str,
           width_pt: float, height_pt: float, indent: float = 0.0, nowrap: bool = False) -> str:
    style = _pos(x, y, w, height_pt, width_pt) + f" font-size: {size:.2f}pt; line-height: {line_h:.2f}pt; text-align: {align};"
    if item.italic:
        style += " font-style: italic;"
    if nowrap:
        style += " white-space: nowrap;"
    # azul does not apply text-indent yet: a first-line indent is no-break spaces (~0.25 em each)
    lead = "\u00a0" * round(indent / (0.25 * size)) if indent and size else ""
    p_style = ""
    attrs = f'data-zone="{item.zone}" data-role="{item.kind}"' + (f' data-lines="{" ".join(item.lines)}"' if item.lines else "")
    return f'<div class="region" {attrs} style="{style}"><p{p_style}>{lead}{htmlmod.escape(item.text)}</p></div>'


# --- the pipeline ----------------------------------------------------------------------------


HTML2PDF = Path(__file__).resolve().parents[2] / "html2pdf" / "target" / "release" / "html2pdf"


def _write_zip(target: Path, metadata: dict, pages_meta: list[dict], out: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        for p in pages_meta:
            z.write(out / p["html"], p["html"])
        for picture in sorted((out / "pictures").glob("*.png")) if (out / "pictures").exists() else []:
            z.write(picture, f"pictures/{picture.name}")


def fit_round(target: Path, page_blocks: dict[int, list[Block]], pages_meta: list[dict], work: Path) -> int:
    """Render with html2pdf, and set smaller every block whose text runs past its limit.

    The layout engine wraps a little differently than the Helvetica estimate; its rendered glyph
    positions (html2pdf --layout-report) say by how much.
    """
    report_path = work / "layout-report.json"
    subprocess.run([str(HTML2PDF), str(target), "-o", str(work / "fit.pdf"), "--layout-report", str(report_path)],
                   check=True, capture_output=True)
    reports = {r["page"]: r for r in json.loads(report_path.read_text())}
    shrunk = 0
    for p in pages_meta:
        report = reports.get(p["html"])
        if report is None:
            continue
        regions = [b for b in page_blocks[p["page_num"]] if b.item is not None]
        for block, measured in zip(regions, report["regions"]):
            rendered = measured.get("rendered")
            if block.nowrap or not rendered or rendered["y1"] <= block.limit + 0.25 * block.size:
                continue
            # fewer, smaller lines: the size by the ratio of the room to what it took, a bit less
            took = max(rendered["y1"] - block.y, 1.0)
            room = max(block.limit - block.y, 0.5 * block.line_h)
            block.size *= max(0.85, min(0.97, room / took))
            shrunk += 1
    return shrunk


def reconstruct(pdf_path: Path, out: Path, pages: list[int], lang: tuple[str, ...], html_lang: str,
                semantic_context: str = "", llm: bool = True, model: str = "sonnet", agents: int = 4,
                fit_rounds: int = 6, zoom: bool = False) -> Path:
    import pypdfium2 as pdfium

    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    engine = None
    results: dict[int, dict] = {}
    start = time.perf_counter()
    for n in pages:
        name = f"page_{n + 1:03d}"
        pw = work / name
        pw.mkdir(exist_ok=True)
        page = pdf[n]
        width_pt, height_pt = page.get_size()
        dpi = native_dpi(page)
        t = time.perf_counter()
        native = render(page, dpi, pw / "native.png")
        render(page, 0.75 * dpi, pw / "vision.png")
        layout = analyse(pw / "native.png")
        draw(pw / "native.png", layout, pw / "layout.png", scale=0.25)
        if engine is None:
            from .engines import ENGINES
            from .engines.base import Route

            engine = ENGINES["macos_vision"](Route(lang=lang))
            engine.prepare()
        lines = assign(vision_lines(pw / "vision.png", lang, native, engine), layout)
        pictures = clip_pictures(pw / "native.png", layout, out, name)
        (pw / "layout.json").write_text(json.dumps(layout.to_dict(), indent=1))
        (pw / "lines.json").write_text(json.dumps([{**asdict(l), "box": asdict(l.box), "words": None} for l in lines], indent=1, ensure_ascii=False))
        results[n] = {"name": name, "layout": layout, "lines": lines, "pictures": pictures,
                      "size": (width_pt, height_pt), "dpi": dpi}
        log.info(f"{name}: {dpi:.0f} dpi, {len(layout.zones)} zones, {len(lines)} lines, {len(pictures)} pictures, {time.perf_counter() - t:.1f}s")
    pdf.close()

    structures: dict[int, list[Item]] = {}
    if llm:
        from .llm_structure import structure_pages

        structures = structure_pages(results, work, semantic_context, model, agents, zoom)
    for n, r in results.items():
        if n not in structures:
            structures[n] = heuristic_structure(r["lines"], r["layout"])

    pages_meta, page_blocks = [], {}
    for n, r in sorted(results.items()):
        width_pt, height_pt = r["size"]
        page_blocks[n] = build_blocks(structures[n], r["lines"], r["layout"], r["pictures"], width_pt, height_pt)
        pages_meta.append({"page_num": n, "html": f"{r['name']}.html", "width_pt": width_pt, "height_pt": height_pt,
                           "width_px": r["layout"].width, "height_px": r["layout"].height})
    metadata = {"engine": "reconstruct" + ("+" + model if llm else ""), "page_count": len(pages_meta), "pages": pages_meta}
    target = out / "pages.zip"
    for round_ in range(fit_rounds + 1):
        for p in pages_meta:
            width_pt, height_pt = p["width_pt"], p["height_pt"]
            (out / p["html"]).write_text(blocks_html(page_blocks[p["page_num"]], width_pt, height_pt, html_lang), encoding="utf-8")
        _write_zip(target, metadata, pages_meta, out)
        if round_ == fit_rounds or not HTML2PDF.exists():
            break
        shrunk = fit_round(target, page_blocks, pages_meta, work)
        log.info(f"Fit round {round_ + 1}: {shrunk} blocks ran past the next block and were set smaller")
        if not shrunk:
            break
    log.info(f"Wrote {target} ({len(pages_meta)} pages) in {time.perf_counter() - start:.0f}s")
    return target
