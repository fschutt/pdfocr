"""The layout of a scanned book page from its pixels (OpenCV), before any text is read.

Finds the zones a page is set in: running head, text columns, marginal-note strips, footnotes,
other text blocks (titles, headings, captions), pictures (engravings, ornaments) and drop
capitals. Works on the native-resolution render of a black-and-white scan; all sizes are
relative to the page's text line height `lh` (the median height of its glyphs).

1. Ink: Otsu threshold. Connected components are glyphs (a few px to a few `lh`), specks (dust),
   rules (thin and long) or large shapes.
2. Large shapes, grouped with whatever they touch: a group of several large letters on one line
   is large type (text); a group at least 6 `lh` in both directions is a picture; a single
   large shape at the left edge of a column, with text beside it, is a drop capital.
3. Blocks: the text ink is cut recursively. Horizontally at white bands at least 1.2 `lh` high.
   Vertically at gutters: x ranges that at most a few text lines cross. The lines that do cross
   a gutter (a running head, a footnote, a centred title) are cut out as full-width blocks first,
   and what remains is cut into columns. Wide bands are text columns, narrow ones beside them
   marginal-note strips.
4. Roles: a thin block at the top is the running head; blocks below the columns, set smaller
   than the body or under a rule, are footnotes; other single blocks are text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROLES = ("header", "column", "notes", "footnotes", "text", "picture", "dropcap")
MIN_PICTURE_INK = 0.3  # share of a picture's box that is inked


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def overlaps(self, other: "Box", margin: float = 0) -> bool:
        return not (self.x1 + margin <= other.x0 or other.x1 + margin <= self.x0
                    or self.y1 + margin <= other.y0 or other.y1 + margin <= self.y0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class Zone:
    role: str  # one of ROLES
    box: Box
    index: int = 0  # per role, in reading order
    side: str = ""  # notes: "left" / "right" of the column beside them
    glyph: float = 0.0  # median glyph height in px (the zone's type size)

    @property
    def id(self) -> str:
        return f"{self.role}{self.index}"


@dataclass
class PageLayout:
    width: int
    height: int
    line_height: float  # lh: median glyph height of the page's text, px
    zones: list[Zone] = field(default_factory=list)
    words: list[Box] = field(default_factory=list)  # inked words (outside pictures and initials)

    def of(self, role: str) -> list[Zone]:
        return [z for z in self.zones if z.role == role]

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "line_height": round(self.line_height, 1),
                "zones": [{"id": z.id, "role": z.role, "side": z.side, "glyph": round(z.glyph, 1), **asdict(z.box)}
                          for z in self.zones]}


def analyse(path: Path) -> PageLayout:
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read {path}")
    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    stats, ids = stats[1:], np.arange(1, count)  # [x, y, w, h, area]; background dropped
    x, y, w, h, area = (stats[:, i] for i in range(5))

    specks = area < 6
    lh = _line_height(stats[~specks])
    rules = ~specks & (h < 0.4 * lh) & (w > 6 * lh)
    large = ~specks & ~rules & ((w > 2.5 * lh) | (h > 2.5 * lh))
    pictures, initials, large_type = _large_shapes(stats, large, lh, width)

    text = ink.copy()
    drop = specks | rules | (large & ~large_type)
    if drop.any():
        text[np.isin(labels, ids[drop])] = 0
    for box in pictures + initials:
        text[box.y0:box.y1, box.x0:box.x1] = 0

    # words: letters joined by a smear shorter than a word space (and the gap beside a margin note)
    smear = cv2.dilate(text, np.ones((1, max(2, int(0.3 * lh))), np.uint8))
    _, _, wstats, _ = cv2.connectedComponentsWithStats(smear, connectivity=8)
    wx, wy, ww, wh = (wstats[1:, i] for i in range(4))
    keep = wh >= 0.3 * lh
    words = np.stack([wx, wy, wx + ww, wy + wh], axis=1)[keep]

    blocks: list[tuple[Box, str]] = []
    _split(text, words, _tight(text, Box(0, 0, width, height)), lh, blocks)
    layout = PageLayout(width=width, height=height, line_height=lh,
                        words=[Box(int(a), int(b), int(c), int(d)) for a, b, c, d in words])
    layout.zones = _roles(blocks, stats[~specks & ~large], stats[rules], pictures, initials, lh, height)
    return layout


def _line_height(stats: np.ndarray) -> float:
    """Median height of glyph-sized components."""
    w, h = stats[:, 2], stats[:, 3]
    glyphs = h[(h >= 8) & (h <= 150) & (w <= 150) & (w >= 3)]
    return float(np.median(glyphs)) if len(glyphs) else 30.0


def _large_shapes(stats: np.ndarray, large: np.ndarray, lh: float, width: int) -> tuple[list[Box], list[Box], np.ndarray]:
    """(pictures, drop-capital candidates, mask of components that are large type)."""
    idx = np.flatnonzero(large)
    boxes = {i: Box(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 0] + stats[i, 2]), int(stats[i, 1] + stats[i, 3]))
             for i in idx}
    # group large shapes that touch or nearly touch: the hatching of one engraving is many shapes
    groups: list[tuple[Box, list[int]]] = []
    for i in sorted(idx, key=lambda i: (boxes[i].y0, boxes[i].x0)):
        hits = [g for g in groups if g[0].overlaps(boxes[i], margin=0.5 * lh)]
        box, members = boxes[i], [i]
        for g in hits:
            groups.remove(g)
            box, members = box.union(g[0]), members + g[1]
        groups.append((box, members))

    large_type = np.zeros(len(stats), dtype=bool)
    pictures, initials = [], []
    for box, members in groups:
        letters = [boxes[i] for i in members]
        if box.w >= 6 * lh and box.h >= 6 * lh and len(members) > 2 or box.h >= 16 * lh:
            pictures.append(box)
            continue
        # large letters: several shapes of similar height on one line, or one shape among
        # other large letters of its line (a title in big capitals)
        same_line = [b for g, m in groups for b in [g] if g is not box and abs(g.y0 - box.y0) < 0.5 * box.h
                     and abs(g.h - box.h) < 0.35 * box.h]
        if len(members) > 1 and box.h < 6 * lh or same_line:
            large_type[members] = True
        elif box.w >= 6 * lh and box.h >= 6 * lh:
            pictures.append(box)
        else:
            initials.append(box)
    # an engraving's frame may enclose a few separate groups: merge pictures that overlap
    pictures = _merge_overlapping(pictures, lh)
    # an engraving or ornament is dense (45-60% of its box inked); big title letters with a
    # library mark written across them are not (20%): they stay text
    kept = []
    for box in pictures:
        inside = ((stats[:, 0] >= box.x0) & (stats[:, 0] + stats[:, 2] <= box.x1)
                  & (stats[:, 1] >= box.y0) & (stats[:, 1] + stats[:, 3] <= box.y1))
        if stats[inside, 4].sum() >= MIN_PICTURE_INK * box.w * box.h:
            kept.append(box)
        elif box.w <= 16 * lh and box.h <= 16 * lh:
            initials.append(box)  # a plain capital as tall as eight lines: a drop capital, if a column's
        else:
            large_type[inside & large] = True
    return kept, initials, large_type


def _merge_overlapping(boxes: list[Box], lh: float) -> list[Box]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i].overlaps(merged[j], margin=0.5 * lh):
                    merged[i] = merged[i].union(merged.pop(j))
                    changed = True
                    break
            if changed:
                break
    return merged


def _runs(mask: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """[start, end) runs of True, merging runs separated by at most `min_gap` False."""
    runs: list[tuple[int, int]] = []
    start, last = None, None
    for i in np.flatnonzero(mask):
        if start is None:
            start = i
        elif i - last - 1 > min_gap:
            runs.append((int(start), int(last) + 1))
            start = i
        last = i
    if start is not None:
        runs.append((int(start), int(last) + 1))
    return runs


def _split(text: np.ndarray, words: np.ndarray, box: Box, lh: float, out: list[tuple[Box, str]], depth: int = 0) -> None:
    """Cut `box` into blocks: (box, "columns" | "block") per piece, appended to `out` in reading order."""
    if box.w <= 0 or box.h <= 0:
        return
    region = text[box.y0:box.y1, box.x0:box.x1]
    cx, cy = (words[:, 0] + words[:, 2]) / 2, (words[:, 1] + words[:, 3]) / 2
    mine = words[(cx >= box.x0) & (cx < box.x1) & (cy >= box.y0) & (cy < box.y1)]
    if not len(mine):
        return  # specks only
    # 1. horizontal white bands, from the words (dust in a margin does not bridge a gap)
    covered = np.zeros(box.h + 1, dtype=np.int32)
    np.add.at(covered, np.clip(mine[:, 1] - box.y0, 0, box.h), 1)
    np.add.at(covered, np.clip(mine[:, 3] - box.y0, 0, box.h), -1)
    rows = _runs(np.cumsum(covered)[:box.h] > 0, int(1.2 * lh))
    if len(rows) > 1:
        for a, b in rows:
            _split(text, words, _tight(text, Box(box.x0, box.y0 + a, box.x1, box.y0 + b)), lh, out, depth + 1)
        return
    # 2. gutters: x ranges crossed by at most a couple of words (a running head, a footnote);
    #    a text column is crossed by a word on nearly every line, a note strip by every note
    diff = np.zeros(box.w + 1, dtype=np.int32)
    np.add.at(diff, np.clip(mine[:, 0] - box.x0, 0, box.w), 1)
    np.add.at(diff, np.clip(mine[:, 2] - box.x0, 0, box.w), -1)
    occupancy = np.cumsum(diff)[:box.w]
    tall = box.h >= 8 * lh
    # a gutter is crossed by a tenth of the words that cross a typical x of a text column
    dense = float(np.percentile(occupancy, 75)) if len(occupancy) else 0.0
    free = occupancy <= max(2.0, 0.1 * dense) if tall else occupancy == 0
    min_gutter = max(3, int((0.12 if tall else 1.5) * lh))
    gutters = [(a, b) for a, b in _runs(free, 0) if b - a >= min_gutter and a > 0 and b < box.w]
    if not gutters and tall and box.w >= 0.6 * text.shape[1]:
        # a page of two columns over many lines of footnotes across the page: more than a tenth
        # of the lines cross its gutter (a fifth do not, in a column of text)
        free = occupancy <= max(2.0, 0.2 * dense)
        gutters = [(a, b) for a, b in _runs(free, 0) if b - a >= 2 * min_gutter and a > 0.25 * box.w and b < 0.75 * box.w]
    if not gutters or depth > 12:
        out.append((box, "block"))
        return
    # 3. lines with a word across a gutter are full-width blocks of their own
    crossing_rows = np.zeros(box.h, dtype=bool)
    # only a column gutter: beside a marginal note the white is a few px, and a note that nearly
    # touches the text merges with it into one "word" across the gap
    for a, b in [g for g in gutters if g[1] - g[0] >= 0.6 * lh]:
        # the gutter's core: its emptiest x positions (the ends of a gutter run into ragged edges)
        low = occupancy[a:b].min()
        core = np.flatnonzero(occupancy[a:b] <= low + 1)
        a, b = a + int(core[0]), a + int(core[-1]) + 1
        # a word crosses when it bridges the core or stands in it ("(2)" of a running head), not
        # when the end of a line pokes into it
        x0, x1 = mine[:, 0] - box.x0, mine[:, 2] - box.x0
        across = mine[((x0 <= a + 2) & (x1 >= b - 2)) | (((x0 + x1) / 2 >= a) & ((x0 + x1) / 2 < b))]
        for _, y0, _, y1 in across:
            crossing_rows[max(0, y0 - box.y0):max(0, y1 - box.y0)] = True
    # a band thinner than half a line is a speck or a rule end in the gutter, not a line of text
    spans = [(a, b) for a, b in _runs(crossing_rows, int(0.6 * lh)) if b - a >= 0.5 * lh]
    if spans:
        edges, y = [], 0
        for a, b in spans:
            if a > y:
                edges.append((y, a, False))
            edges.append((a, b, True))
            y = b
        if y < box.h:
            edges.append((y, box.h, False))
        if len(edges) > 1:
            for a, b, full in edges:
                piece = _tight(text, Box(box.x0, box.y0 + a, box.x1, box.y0 + b))
                if full:
                    out.append((piece, "block"))
                else:
                    _split(text, words, piece, lh, out, depth + 1)
            return
    # 4. columns
    bands, x = [], 0
    for a, b in gutters:
        bands.append((x, a))
        x = b
    bands.append((x, box.w))
    # a marginal-note strip is crossed by few words, so a gap between words there can look like a
    # gutter: neighbouring narrow bands close together are one strip
    widest = max(b - a for a, b in bands)
    merged = [bands[0]]
    for a, b in bands[1:]:
        pa, pb = merged[-1]
        if pb - pa < 0.3 * widest and b - a < 0.3 * widest and a - pb < 1.2 * lh:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    bands = merged
    if not tall:
        # a short block is split only into real columns of words, not at the letter spacing of
        # a title in large capitals
        mx = (mine[:, 0] + mine[:, 2]) / 2 - box.x0
        if any(((mx >= a) & (mx < b)).sum() < 3 for a, b in bands):
            out.append((box, "block"))
            return
    for a, b in bands:
        piece = _tight(text, Box(box.x0 + a, box.y0, box.x0 + b, box.y1))
        if piece.w > 0:
            # a "column" as wide as most of the page: two columns, when footnotes across the page
            # cross their gutter (p. 880 of vol. 1)
            out.extend((p, "columns") for p in _two_columns(text, words, piece, lh) or [piece])


def _two_columns(text: np.ndarray, words: np.ndarray, box: Box, lh: float) -> list[Box] | None:
    """`box` cut at a gutter in its middle half that at most a fifth of its lines cross, if it is
    a tall piece across most of the page; else None."""
    if box.w < 0.6 * text.shape[1] or box.h < 8 * lh:
        return None
    cx, cy = (words[:, 0] + words[:, 2]) / 2, (words[:, 1] + words[:, 3]) / 2
    mine = words[(cx >= box.x0) & (cx < box.x1) & (cy >= box.y0) & (cy < box.y1)]
    diff = np.zeros(box.w + 1, dtype=np.int32)
    np.add.at(diff, np.clip(mine[:, 0] - box.x0, 0, box.w), 1)
    np.add.at(diff, np.clip(mine[:, 2] - box.x0, 0, box.w), -1)
    occupancy = np.cumsum(diff)[:box.w]
    dense = float(np.percentile(occupancy, 75)) if len(occupancy) else 0.0
    free = occupancy <= max(2.0, 0.2 * dense)
    gutters = [(a, b) for a, b in _runs(free, 0) if b - a >= max(6, int(0.24 * lh)) and a > 0.25 * box.w and b < 0.75 * box.w]
    if not gutters:
        return None
    a, b = max(gutters, key=lambda g: g[1] - g[0])
    left = _tight(text, Box(box.x0, box.y0, box.x0 + a, box.y1))
    right = _tight(text, Box(box.x0 + b, box.y0, box.x1, box.y1))
    return [left, right] if left.w > 0 and right.w > 0 else None


def _line_start(region: np.ndarray, y: int, lh: float) -> int:
    """Top of the text line containing row y (rows of ink above it, up to a line height)."""
    rows = region.sum(axis=1) > 0
    top = y
    while top > 0 and rows[top - 1] and y - top < 1.5 * lh:
        top -= 1
    return top


def _line_end(region: np.ndarray, y: int, lh: float) -> int:
    rows = region.sum(axis=1) > 0
    end = y
    while end < len(rows) and rows[end] and end - y < 1.5 * lh:
        end += 1
    return end


def _tight(text: np.ndarray, box: Box) -> Box:
    region = text[box.y0:box.y1, box.x0:box.x1]
    rows = np.flatnonzero(region.sum(axis=1) > 0)
    cols = np.flatnonzero(region.sum(axis=0) > 0)
    if not len(rows) or not len(cols):
        return Box(box.x0, box.y0, box.x0, box.y0)
    return Box(box.x0 + int(cols[0]), box.y0 + int(rows[0]), box.x0 + int(cols[-1]) + 1, box.y0 + int(rows[-1]) + 1)


def _glyph(stats: np.ndarray, box: Box) -> float:
    x, y, w, h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    inside = (x >= box.x0) & (x + w <= box.x1) & (y >= box.y0) & (y + h <= box.y1) & (h >= 6)
    return float(np.median(h[inside])) if inside.any() else 0.0


def _roles(blocks: list[tuple[Box, str]], glyphs: np.ndarray, rules: np.ndarray, pictures: list[Box],
           initials: list[Box], lh: float, height: int) -> list[Zone]:
    zones: list[Zone] = []
    pieces = [(b, kind, _glyph(glyphs, b)) for b, kind in blocks if b.w > 0 and b.h > 0.3 * lh]
    columns = [p for p in pieces if p[1] == "columns"]
    widest = max((b.w for b, _, _ in columns), default=0)
    body_bottom = max((b.y1 for b, _, _ in columns if b.w >= 0.45 * widest), default=0)
    rule_tops = [int(r[1]) for r in rules]
    first = min((b.y0 for b, _, _ in pieces), default=0)
    for box, kind, glyph in pieces:
        if kind == "columns" and box.w >= 0.45 * widest:
            role, side = "column", ""
        elif kind == "columns":
            beside = [c for c, k, _ in columns if c.w >= 0.45 * widest and c.y0 < box.y1 and box.y0 < c.y1]
            role = "notes"
            side = "right" if any(c.x1 <= box.x0 for c in beside) else "left"
        elif box.y0 == first and box.h <= 2.5 * lh and box.y1 < 0.12 * height:
            role, side = "header", ""
        elif body_bottom and box.y0 >= body_bottom - 0.5 * lh and (
            glyph and glyph < 0.92 * lh or any(body_bottom - lh <= r <= box.y0 for r in rule_tops)
        ):
            role, side = "footnotes", ""
        else:
            role, side = "text", ""
        zones.append(Zone(role, box, side=side, glyph=glyph))
    # drop capitals: large initials (or an ornamented one, which looks like a small picture) at the
    # left edge of a column or text block, with its text beside them
    small = [b for b in pictures if b.w <= 16 * lh and b.h <= 16 * lh]
    pictures = [b for b in pictures if b not in small]
    for box in initials + small:
        # the initial stands in the left half of a column, within its height
        holder = next((z for z in zones if z.role in ("column", "text")
                       and z.box.x0 - 2 * lh <= box.x0 and (box.x0 + box.x1) / 2 <= z.box.x0 + 0.5 * z.box.w
                       and z.box.y0 - 2 * lh <= box.y0 <= z.box.y1), None)
        if holder:
            zones.append(Zone("dropcap", box))
        elif box in small:
            zones.append(Zone("picture", box))  # an ornament of several parts
        # else: one large letter (a swash capital, a big title letter): text, which the OCR reads
    zones += [Zone("picture", b) for b in pictures]
    # reading order: top to bottom, then left to right within a band of columns
    zones.sort(key=lambda z: (z.box.y0 // max(1, int(2 * lh)), z.box.x0))
    counters: dict[str, int] = {}
    for z in zones:
        z.index = counters.get(z.role, 0)
        counters[z.role] = z.index + 1
    return zones


COLOURS = {"header": (200, 120, 0), "column": (0, 160, 0), "notes": (0, 120, 255), "footnotes": (160, 0, 160),
           "text": (0, 110, 110), "picture": (0, 0, 220), "dropcap": (220, 0, 0)}


def draw(path: Path, layout: PageLayout, out: Path, scale: float = 0.3) -> None:
    """The page with its zones outlined, for checking the analysis by eye."""
    import cv2

    page = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    for z in layout.zones:
        cv2.rectangle(page, (z.box.x0, z.box.y0), (z.box.x1, z.box.y1), COLOURS[z.role], 8)
        cv2.putText(page, z.id, (z.box.x0 + 10, z.box.y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLOURS[z.role], 5)
    small = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), small)
