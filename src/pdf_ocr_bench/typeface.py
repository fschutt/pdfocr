"""The book's own typeface, rebuilt from its scans (`pdf-ocr-bench typeface`).

Times is not the face the book was printed in: at the same x-height it sets lines ~5% wider,
so paragraphs break where the print does not. This finds every letter of the scans, measures
each letter's width and side bearings, averages its shape over hundreds of prints, traces it
and builds a TrueType font with the printed metrics, for `reconstruct --typeface`.

Steps:
  1. per page (cached in work/page_NNN/glyphs.json, from a `reconstruct` run's work): Vision
     finds each glyph's box; the words Vision read there are lined up with the words the model
     read (its text writes s for a long s, Vision reads it f: that tells ſ from f; its markup
     tells italic); each glyph's ink, and the line's baseline and pitch (the text size);
  2. metrics: side bearings from the gaps between letters inside words (least squares), the
     word space from lines that are not justified;
  3. shapes: each letter's prints, scaled to one size and laid on the baseline, averaged;
  4. outlines traced from the averages; a font per style (roman, italic).
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import numpy as np

# glyphs the compositor set as one piece: several letters, one box (longest first)
LIGATURES = ("ſſi", "ſſl", "ffi", "ffl", "ſſ", "ſt", "ſi", "ſl", "ſh", "ſb", "ſk", "ff", "fi", "fl", "ct")
# letters that stand on the baseline (no descender): what a word's baseline is measured on
ON_BASELINE = set("acehiklmnorstuvwxzABCDEFGHIKLMNORSTUVWXZ")
X_HEIGHT = set("acemnorsuvwxz")
SCAN_VERSION = 2  # glyphs.json of another version is scanned again


def glyph_boxes(image: Path) -> list[tuple[int, int, int, int, int]]:
    """The glyph boxes Vision finds on `image` (its text detection, which reads nothing), px,
    (x0, y0, x1, y1, row) with y down, `row` the text line it found them in. A ligature is one
    box; a letter may be two."""
    import objc
    import Vision
    from Foundation import NSURL
    from PIL import Image

    with Image.open(image) as im:
        width, height = im.size
    with objc.autorelease_pool():
        find = Vision.VNDetectTextRectanglesRequest.alloc().init()
        find.setReportCharacterBoxes_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(image)), {})
        ok, error = handler.performRequests_error_([find], None)
        if not ok:
            raise RuntimeError(f"Vision: {error}")
        out = []
        for row, obs in enumerate(find.results() or []):
            for c in obs.characterBoxes() or []:
                r = c.boundingBox()
                x, y, w, h = r.origin.x, r.origin.y, r.size.width, r.size.height
                out.append((round(x * width), round((1 - y - h) * height), round((x + w) * width), round((1 - y) * height), row))
    return out


def printed(read: str, meant: str) -> str | None:
    """The letters as printed of a word Vision read as `read` and the model as `meant` (its
    text: s for a long s, which Vision reads f): "fhould"/"should" -> "ſhould". None when they
    differ otherwise (one of them misread it)."""
    if len(read) != len(meant):
        return None
    out = []
    for r, m in zip(read, meant):
        if r == m:
            out.append(r)
        elif r == "f" and m == "s":
            out.append("ſ")
        else:
            return None
    return "".join(out)


def spell(word: str, boxes: int) -> list[str] | None:
    """`word` (as printed) cut into as many pieces as it has glyph boxes, ligatures taken
    together where it has fewer boxes than letters, leftmost first; None if no way fits."""
    need = len(word) - boxes
    if need < 0:
        return None
    out, i = [], 0
    while i < len(word):
        hit = next((lig for lig in LIGATURES if need >= len(lig) - 1 and word.startswith(lig, i)), None) if need else None
        if hit:
            out.append(hit)
            i += len(hit)
            need -= len(hit) - 1
        else:
            out.append(word[i])
            i += 1
    return out if need == 0 else None


def _key(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower().replace("ſ", "s").replace("f", "s"))


def _marked_words(text: str) -> list[tuple[str, bool]]:
    """The words of an item's text with their style: italic inside <i> (and the item's own
    style outside it)."""
    out, italic = [], False
    for part in re.split(r"(</?i>)", text):
        if part == "<i>":
            italic = True
        elif part == "</i>":
            italic = False
        else:
            out += [(w, italic) for w in re.sub(r"<[^>]+>", "", part).split()]
    return out


def scan_page(pw: Path, ink=None, boxes=None) -> dict:
    """Step 1 for one page's work directory (a `reconstruct` run's work/page_NNN): its glyphs
    as [letters, italic, x0, y0, x1, y1 (ink, px), baseline (px), word, place in the word,
    x-height of the word (px), line], the boxes of its lines, and the page's text pitch (px)."""
    import pickle

    import cv2
    from PIL import Image

    from . import claude_cli
    from .llm_structure import to_items

    res = pickle.loads((pw / "prepared.pkl").read_bytes())["result"]
    lines, layout = res["lines"], res["layout"]
    answer, _ = claude_cli.answer_of(pw / "llm" / "response.jsonl")
    items = (to_items(answer, lines, layout) if answer else None) or []
    if ink is None:
        ink = (np.array(Image.open(pw / "native.png").convert("L")) < 128).astype(np.uint8)
    if boxes is None:
        boxes = glyph_boxes(pw / "native.png")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    # the model's word for each word Vision read: lined up item by item
    by_id = {l.id: l for l in lines}
    meant: dict[tuple[str, int], tuple[str, bool]] = {}
    for item in items:
        if item.kind == "noise":
            continue
        ocr = [(l.id, k, t) for l in sorted((by_id[i] for i in item.lines if i in by_id), key=lambda l: (l.box.y0, l.box.x0))
               for k, (t, _) in enumerate(l.words)]
        theirs = [(w, it or item.italic) for w, it in _marked_words(item.text)]
        match = difflib.SequenceMatcher(None, [_key(t) for _, _, t in ocr], [_key(w) for w, _ in theirs], autojunk=False)
        for a, b, n in match.get_matching_blocks():
            for j in range(n):
                meant[ocr[a + j][:2]] = theirs[b + j]

    lh = layout.line_height
    bx = np.array([b[:4] for b in boxes], dtype=float).reshape(-1, 4)
    cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2

    def glyphs_in(wb) -> list[set]:
        """Its glyphs, left to right, each as the pieces of ink it has: the glyph boxes of its own
        line inside it (Vision's word box may reach into the next line: the boxes whose middles
        stand nearest its top), each piece of ink given to the one box nearest it."""
        hit = np.flatnonzero((cx >= wb.x0 - 0.2 * lh) & (cx <= wb.x1 + 0.2 * lh) & (cy >= wb.y0 - 0.3 * lh) & (cy <= wb.y1 + 0.3 * lh))
        if not len(hit):
            return []
        hit = hit[np.argsort(cy[hit])]
        groups, cur = [], [hit[0]]
        for i in hit[1:]:
            if cy[i] - cy[cur[-1]] > 0.8 * lh:
                groups.append(cur)
                cur = []
            cur.append(i)
        groups.append(cur)
        mine = min(groups, key=lambda g: abs(np.median(bx[g, 1]) - wb.y0))
        mine = sorted(mine, key=lambda i: bx[i, 0])
        x0, y0 = int(bx[mine, 0].min()) - 2, int(bx[mine, 1].min()) - 2
        x1, y1 = int(bx[mine, 2].max()) + 2, int(bx[mine, 3].max()) + 2
        owned: dict[int, set] = {i: set() for i in mine}
        for p in np.unique(labels[max(0, y0):y1, max(0, x0):x1]):
            if not p or stats[p, 4] < 3:
                continue
            px_, py_ = stats[p, 0] + stats[p, 2] / 2, stats[p, 1] + stats[p, 3] / 2
            holders = [i for i in mine if bx[i, 0] - 2 <= px_ <= bx[i, 2] + 2 and bx[i, 1] - 2 <= py_ <= bx[i, 3] + 2]
            if holders:
                owned[min(holders, key=lambda i: abs(cx[i] - px_))].add(int(p))
        return [owned[i] for i in mine if owned[i]]

    glyphs, words_out = [], 0
    why: dict[str, int] = {}
    line_bases: list[tuple[float, float, float]] = []  # (baseline, x0, x1) of each line
    line_boxes = []
    for line_no, line in enumerate(lines):
        line_boxes.append([line.box.x0, line.box.y0, line.box.x1, line.box.y1])
        bases = []
        for k, (text, wb) in enumerate(line.words or []):
            if (line.id, k) not in meant:
                why["not in the model's text"] = why.get("not in the model's text", 0) + 1
                continue
            their, italic = meant[(line.id, k)]
            core = re.sub(r"^[^\w]+|[^\w]+$", "", text)
            their_core = re.sub(r"^[^\w]+|[^\w]+$", "", re.sub(r"<[^>]+>", "", their))
            word = printed(core, their_core)
            if not word:
                why["read otherwise"] = why.get("read otherwise", 0) + 1
                continue
            mine = glyphs_in(wb)
            # with its punctuation, as Vision read it (each mark a glyph too)
            lead, trail = text[:text.find(core)] if core else "", text[text.find(core) + len(core):] if core else ""
            pieces = spell(lead + word + trail, len(mine))
            if pieces is None and lead + trail:  # (a mark the detection did not box)
                pieces = spell(word, len(mine))
            while pieces is None and len(mine) > 1:
                # a letter the scan broke in two, boxed twice ("w"): its halves touch
                ext = [(min(stats[q, 0] for q in g), max(stats[q, 0] + stats[q, 2] for q in g)) for g in mine]
                gaps = [ext[j + 1][0] - ext[j][1] for j in range(len(ext) - 1)]
                j = int(np.argmin(gaps))
                if gaps[j] > 1:
                    break
                mine = mine[:j] + [mine[j] | mine[j + 1]] + mine[j + 2:]
                pieces = spell(lead + word + trail, len(mine)) or (spell(word, len(mine)) if lead + trail else None)
            if not pieces:
                key = "more boxes than letters" if len(mine) > len(word) else "fewer boxes than letters"
                why[key] = why.get(key, 0) + 1
                continue
            word_glyphs = []
            for pos, (letters, ids) in enumerate(zip(pieces, mine)):
                ix0 = int(min(stats[i, 0] for i in ids))
                iy0 = int(min(stats[i, 1] for i in ids))
                ix1 = int(max(stats[i, 0] + stats[i, 2] for i in ids))
                iy1 = int(max(stats[i, 1] + stats[i, 3] for i in ids))
                word_glyphs.append([letters, italic, ix0, iy0, ix1, iy1])
            else:
                low = sorted(g[5] for g in word_glyphs if g[0] in ON_BASELINE)
                xh = sorted(g[5] - g[3] for g in word_glyphs if g[0] in X_HEIGHT)
                base = float(low[len(low) // 2]) if len(low) >= 2 else None
                if base is not None:
                    bases.append(base)
                for pos, g in enumerate(word_glyphs):
                    glyphs.append(g + [base, words_out, pos, float(xh[len(xh) // 2]) if xh else None, line_no])
                words_out += 1
        # a word with too few letters of its own to tell (figures, "a", "of"): its line's baseline
        if bases:
            line_base = float(np.median(bases))
            line_bases.append((line_base, line.box.x0, line.box.x1))
            for g in glyphs:
                if g[10] == line_no and g[6] is None:
                    g[6] = line_base
    # the text's pitch: from one line's baseline to the next below it in the same column
    steps = []
    for base, x0, x1 in line_bases:
        below = [b for b, a0, a1 in line_bases if b > base + 0.5 * lh and min(x1, a1) - max(x0, a0) > 0.5 * min(x1 - x0, a1 - a0)]
        if below:
            steps.append(min(below) - base)
    steps.sort()
    pitch = steps[len(steps) // 2] if steps else 0.0
    return {"version": SCAN_VERSION, "pitch": pitch, "glyphs": glyphs, "words": words_out, "skipped": why, "lines": line_boxes,
            "size": [int(ink.shape[1]), int(ink.shape[0])]}


def scan(work: Path, workers: int = 4, log=print) -> list[dict]:
    """Step 1 for every page of a `reconstruct` run's work directory (in `workers` processes)."""
    from concurrent.futures import ProcessPoolExecutor

    pages = sorted(p for p in work.glob("page_*") if (p / "prepared.pkl").exists())
    out = []
    with ProcessPoolExecutor(workers) as pool:
        for k, result in enumerate(pool.map(page_glyphs, pages, chunksize=4)):
            out.append(result)
            if (k + 1) % 50 == 0 or k + 1 == len(pages):
                log(f"{k + 1}/{len(pages)} pages, {sum(len(r['glyphs']) for r in out)} glyphs")
    return out


def page_glyphs(pw: Path) -> dict:
    """`scan_page`, kept in work/page_NNN/glyphs.json."""
    cache = pw / "glyphs.json"
    if cache.exists():
        try:
            saved = json.loads(cache.read_text())
            if saved.get("version") == SCAN_VERSION:
                return saved
        except ValueError:
            pass
    try:
        out = scan_page(pw)
    except Exception as exc:  # a page without an answer, a scan Vision cannot open: no glyphs
        return {"version": SCAN_VERSION, "pitch": 0.0, "glyphs": [], "words": 0, "lines": [], "error": str(exc)}
    cache.write_text(json.dumps(out))
    return out


# step 2: metrics

def samples(pages: list[dict], size_tol: float = 0.1) -> list[tuple]:
    """Every glyph of the body text, in em of its page's text size (its line pitch: the book is
    set solid): (letters, italic, x0, x1, bottom, top (y up from the baseline), page, word, pos,
    line), only words of the body's x-height (not the smaller notes and footnotes)."""
    out = []
    for k, page in enumerate(pages):
        em = page.get("pitch") or 0.0
        rows = [g for g in page["glyphs"] if g[6] is not None]
        heights = sorted(g[9] for g in rows if g[9])
        if em <= 0 or not heights:
            continue
        body = heights[len(heights) // 2]
        for g in rows:
            if g[9] and abs(g[9] / body - 1) > size_tol:
                continue
            letters, italic, x0, y0, x1, y1, base, word, pos = g[:9]
            out.append((letters, bool(italic), x0 / em, x1 / em, (base - y1) / em, (base - y0) / em, k, word, pos, g[10]))
    return out


def side_bearings(glyphs: list[tuple], least: int = 30) -> tuple[dict, dict, dict]:
    """Left and right side bearings (em) of each glyph of one style, from the gaps between the
    glyphs of a word: gap(a, b) = right(a) + left(b) + kern(a, b), least squares over the pairs
    seen `least` times or more (the medians of their gaps); a round letter's two bearings taken
    alike (o, n, l: else any shift of all lefts against all rights fits as well). And the kerns:
    pairs whose gaps stand off their bearings."""
    gaps: dict[tuple[str, str], list[float]] = {}
    by_word: dict[tuple, list] = {}
    for g in glyphs:
        by_word.setdefault((g[6], g[7]), []).append(g)
    for word in by_word.values():
        word.sort(key=lambda g: g[8])
        for a, b in zip(word, word[1:]):
            if b[8] == a[8] + 1:
                gaps.setdefault((a[0], b[0]), []).append(b[2] - a[3])
    # (not two capitals: the headwords are spaced out, "A L E X A N D R I U M")
    pairs = {p: float(np.median(v)) for p, v in gaps.items() if len(v) >= least and not (p[0].isupper() and p[1].isupper())}
    letters = sorted({c for p in pairs for c in p})
    index = {c: i for i, c in enumerate(letters)}
    n = len(letters)
    rows, rhs, weights = [], [], []
    for (a, b), gap in pairs.items():
        row = np.zeros(2 * n)
        row[n + index[a]] += 1.0  # right of a
        row[index[b]] += 1.0  # left of b
        rows.append(row)
        rhs.append(gap)
        weights.append(np.sqrt(len(gaps[(a, b)])))
    strong = np.sqrt(max(weights, default=1.0))
    for c in letters:
        # a round letter's bearings alike; a capital's, more or less (its left one is seen only
        # after another capital)
        if c in "onlimuxOHI" or (len(c) == 1 and c.isupper()):
            row = np.zeros(2 * n)
            row[index[c]], row[n + index[c]] = 1.0, -1.0
            rows.append(row)
            rhs.append(0.0)
            weights.append(strong if c in "onlimuxOHI" else 0.3 * strong)
    if not rows:
        return {}, {}, {}
    A, y, w = np.array(rows), np.array(rhs), np.array(weights)
    solution, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    left = {c: float(solution[index[c]]) for c in letters}
    right = {c: float(solution[n + index[c]]) for c in letters}
    kerns = {}
    for (a, b), gap in pairs.items():
        off = gap - right[a] - left[b]
        if len(gaps[(a, b)]) >= 2 * least and abs(off) >= 0.015:
            kerns[(a, b)] = off
    return left, right, kerns


def word_space(glyphs: list[tuple], pages: list[dict], left: dict, right: dict) -> dict | None:
    """The word spaces (em): the gap from one word's last glyph to the next one's first, less
    their bearings. "space": the tight end of the justified lines' spaces (a line's mean: the
    compositor spaced a line alike), what a line must fit with: a layout engine fills a line at
    its font's space and only widens it (azul, like CSS, never narrows it); "loose": the spaces
    of the lines that end short of their column."""
    by_line: dict[tuple, list] = {}
    for g in glyphs:
        by_line.setdefault((g[6], g[9]), []).append(g)
    loose, tight = [], []
    for (k, line_no), line in by_line.items():
        boxes = pages[k]["lines"]
        x0, _, x1, _ = boxes[line_no]
        em = pages[k]["pitch"]
        # its column's right edge: where the lines that start where it does mostly end
        ends = sorted(b[2] for b in boxes if abs(b[0] - x0) < 1.5 * em)
        if not ends:
            continue
        edge = ends[int(0.8 * (len(ends) - 1))]
        short = x1 < edge - 3 * em
        if not short and x1 < edge - 0.5 * em:
            continue
        spaces: list[float] = []
        words: dict[int, list] = {}
        for g in line:
            words.setdefault(g[7], []).append(g)
        order = sorted(words.values(), key=lambda w: min(g[2] for g in w))
        for wa, wb in zip(order, order[1:]):
            if wb[0][7] != wa[0][7] + 1:
                continue  # a word between them was not read
            last, first = max(wa, key=lambda g: g[8]), min(wb, key=lambda g: g[8])
            if last[0] in right and first[0] in left:
                spaces.append(first[2] - last[3] - right[last[0]] - left[first[0]])
        if short:
            loose += spaces
        elif len(spaces) >= 3:
            tight.append(float(np.mean(spaces)))
    if len(tight) < 20 or len(loose) < 20:
        return None
    tight.sort()
    return {"space": tight[len(tight) // 20], "loose": float(np.median(loose)), "justified": float(np.median(tight))}


def metrics(pages: list[dict]) -> dict:
    """Step 2: per style, each glyph's ink box (em, medians) and bearings, the kerns and the
    word space."""
    glyphs = samples(pages)
    out = {}
    for italic in (False, True):
        mine = [g for g in glyphs if g[1] == italic]
        left, right, kerns = side_bearings(mine)
        boxes: dict[str, list] = {}
        for g in mine:
            boxes.setdefault(g[0], []).append((g[3] - g[2], g[4], g[5]))
        ink = {c: [float(np.median([v[i] for v in vs])) for i in range(3)] + [len(vs)] for c, vs in boxes.items()}
        out["italic" if italic else "roman"] = {
            "glyphs": {c: {"width": ink[c][0], "bottom": ink[c][1], "top": ink[c][2], "count": ink[c][3],
                           "left": left.get(c), "right": right.get(c)} for c in ink},
            "kerns": {"".join(p): v for p, v in kerns.items()},
            "spaces": word_space(mine, pages, left, right),
        }
    return out


# step 3: shapes

EM_PX = 160  # the averaged glyphs' size: px per em
CANVAS = (240, 320)  # rows, columns: 1.0 em above the baseline, 0.5 below; 2 em wide
BASE_ROW, MID_COL = 160, 160


def _choose(glyphs: list[tuple], most: int, seed: int = 1) -> dict[str, list[tuple]]:
    """Up to `most` prints of each glyph of one style, spread over the pages."""
    import random

    by: dict[str, list[tuple]] = {}
    for g in glyphs:
        by.setdefault(g[0], []).append(g)
    rng = random.Random(seed)
    return {c: (rng.sample(v, most) if len(v) > most else v) for c, v in by.items()}


def _print_of(ink, labels, stats, box, base: float, em: float, others=()):
    """One print on the canvas: its ink (the pieces whose middle is in its box, and nearer its
    middle than the middle of a neighbour's box that has it too: an italic f's box reaches over
    the next letter), scaled to EM_PX, its baseline on BASE_ROW, its middle (of ink) on MID_COL."""
    import cv2

    x0, y0, x1, y1 = box
    pad = 3
    crop_labels = labels[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]

    def mine(p) -> bool:
        px_, py_ = stats[p, 0] + stats[p, 2] / 2, stats[p, 1] + stats[p, 3] / 2
        if not (x0 - 1 <= px_ <= x1 + 1 and y0 - 1 <= py_ <= y1 + 1):
            return False
        d = abs(px_ - (x0 + x1) / 2)
        return all(not (o[0] - 1 <= px_ <= o[2] + 1) or abs(px_ - (o[0] + o[2]) / 2) > d for o in others)

    keep = [p for p in np.unique(crop_labels) if p and mine(p)]
    if not keep:
        return None
    crop = np.isin(crop_labels, keep).astype(np.float32)
    ys, xs = np.nonzero(crop)
    ox, oy = max(0, x0 - pad), max(0, y0 - pad)
    mx = xs.mean() + ox
    s = EM_PX / em
    # page px -> canvas: (x - mx) * s + MID_COL, (y - base) * s + BASE_ROW
    m = np.float32([[s, 0, (ox - mx) * s + MID_COL], [0, s, (oy - base) * s + BASE_ROW]])
    return cv2.warpAffine(crop, m, (CANVAS[1], CANVAS[0]), flags=cv2.INTER_LINEAR)


def shapes(pages: list[dict], dirs: list[Path], most: int = 200, least: int = 12, log=print) -> dict:
    """Step 3: each glyph's average print per style, {(style, letters): (canvas float 0..1, n)},
    and where its ink starts on the canvas relative to MID_COL is in the canvas itself."""
    import cv2
    from PIL import Image

    glyphs = samples(pages)
    want: dict[tuple, list] = {}
    for italic in (False, True):
        for c, gs in _choose([g for g in glyphs if g[1] == italic], most).items():
            if len(gs) >= least:
                want[("italic" if italic else "roman", c)] = gs
    # the prints, page by page (each page's scan read once)
    by_page: dict[int, list] = {}
    for key, gs in want.items():
        for g in gs:
            by_page.setdefault(g[6], []).append((key, g))
    prints: dict[tuple, list] = {key: [] for key in want}
    for n, (k, todo) in enumerate(sorted(by_page.items())):
        page = pages[k]
        ink = (np.array(Image.open(dirs[k] / "native.png").convert("L")) < 128).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        rows = {(g[7], g[8]): g for g in page["glyphs"]}
        for key, g in todo:
            raw = rows.get((g[7], g[8]))
            if raw is None:
                continue
            others = [tuple(rows[(g[7], g[8] + d)][2:6]) for d in (-1, 1) if (g[7], g[8] + d) in rows]
            canvas = _print_of(ink, labels, stats, tuple(raw[2:6]), raw[6], page["pitch"], others)
            if canvas is not None:
                prints[key].append(canvas)
        if (n + 1) % 100 == 0:
            log(f"prints from {n + 1}/{len(by_page)} pages")
    out = {}
    for key, ps in prints.items():
        if len(ps) < least:
            continue
        stack = np.stack(ps)
        mean = np.median(stack, axis=0)
        # each print moved onto the median (phase correlation), then the median again
        moved = []
        for p in ps:
            (dx, dy), _ = cv2.phaseCorrelate(mean.astype(np.float64), p.astype(np.float64))
            if abs(dx) > 8 or abs(dy) > 8:
                continue
            m = np.float32([[1, 0, -dx], [0, 1, -dy]])
            moved.append(cv2.warpAffine(p, m, (CANVAS[1], CANVAS[0]), flags=cv2.INTER_LINEAR))
        if len(moved) >= least:
            mean = np.median(np.stack(moved), axis=0)
        out[key] = (mean, len(ps))
    return out


# step 4: outlines and fonts

UPM = 1000


def trace(binary: np.ndarray, smooth: float = 1.2, corner: float = 55.0) -> list[list[tuple[float, float, bool]]]:
    """The outlines of a glyph's averaged print (canvas px, y down): closed contours of points
    (x, y, on_curve). Corners (turning more than `corner` degrees) are points on the curve; the
    rest of a contour's points, thinned, are the control points of a quadratic B-spline (the
    curve runs through the midpoints between them), which TrueType draws as it is."""
    import cv2

    found, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in found:
        pts = c[:, 0, :].astype(float)
        n = len(pts)
        if n < 8:
            continue
        # smoothed along the contour (a circular Gaussian)
        r = max(1, int(3 * smooth))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
        k /= k.sum()
        ext = np.concatenate([pts[-r:], pts, pts[:r]])
        sm = np.stack([np.convolve(ext[:, i], k, mode="valid") for i in (0, 1)], axis=1)
        # the turn at each point, over a few points either way
        w = 4
        a = sm - np.roll(sm, w, axis=0)
        b = np.roll(sm, -w, axis=0) - sm
        ang = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1))))
        corners = [i for i in range(n) if ang[i] > corner and ang[i] == ang[[(i + d) % n for d in range(-w, w + 1)]].max()]
        # the control points: the smoothed contour thinned (Douglas-Peucker, 0.7 px)
        poly = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2), 0.7, True)[:, 0, :]
        idx = [int(np.argmin(((sm - p) ** 2).sum(axis=1))) for p in poly]
        keep = sorted(set(idx) | set(corners))
        if len(keep) < 3:
            continue
        cset = set(corners)
        out.append([(float(sm[i, 0]), float(sm[i, 1]), i in cset) for i in keep])
    return out


def _font_points(contour, x_shift: float) -> list[tuple[float, float, bool]]:
    """A traced contour in font units (y up), its ink starting `x_shift` units right of the
    glyph's origin."""
    s = UPM / EM_PX
    return [((x - x_shift) * s, (BASE_ROW - y) * s, on) for x, y, on in contour]


def _area(points) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0, _), (x1, y1, _) in zip(points, points[1:] + points[:1]))


def draw(pen, contours) -> None:
    """Contours of (x, y, on_curve) points (font units, outer ones clockwise) onto a TrueType
    pen; off-curve runs between on-curve points are quadratic B-splines."""
    for pts in contours:
        if not any(on for _, _, on in pts):
            pen.qCurveTo(*[(x, y) for x, y, _ in pts], None)
            pen.closePath()
            continue
        start = next(i for i, p in enumerate(pts) if p[2])
        pts = pts[start:] + pts[:start]
        pen.moveTo(pts[0][:2])
        offs = []
        for x, y, on in pts[1:] + [pts[0]]:
            if on:
                if offs:
                    pen.qCurveTo(*offs, (x, y))
                else:
                    pen.lineTo((x, y))
                offs = []
            else:
                offs.append((x, y))
        pen.closePath()


def svg_path(contours) -> str:
    """The contours as an SVG path (font units, y up: draw with a flip)."""
    parts = []
    for pts in contours:
        n = len(pts)
        on = [i for i, p in enumerate(pts) if p[2]]
        start = on[0] if on else None
        seq = pts[start:] + pts[:start] if start is not None else pts

        def mid(a, b):
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if start is None:
            p0 = mid(seq[-1], seq[0])
            d = [f"M{p0[0]:.1f},{p0[1]:.1f}"]
            for i in range(n):
                c, nxt = seq[i], seq[(i + 1) % n]
                e = mid(c, nxt)
                d.append(f"Q{c[0]:.1f},{c[1]:.1f} {e[0]:.1f},{e[1]:.1f}")
        else:
            d = [f"M{seq[0][0]:.1f},{seq[0][1]:.1f}"]
            ring = seq[1:] + [seq[0]]
            prev_on = seq[0]
            offs = []
            for p in ring:
                if p[2]:
                    if not offs:
                        d.append(f"L{p[0]:.1f},{p[1]:.1f}")
                    else:
                        for j, c in enumerate(offs):
                            e = mid(c, offs[j + 1]) if j + 1 < len(offs) else p
                            d.append(f"Q{c[0]:.1f},{c[1]:.1f} {e[0]:.1f},{e[1]:.1f}")
                    offs = []
                else:
                    offs.append(p)
        parts.append(" ".join(d) + "Z")
    return " ".join(parts)


GLYPH_NAMES = {" ": "space", "ſ": "longs", "æ": "ae", "œ": "oe", "&": "ampersand", ",": "comma", ".": "period",
               ";": "semicolon", ":": "colon", "(": "parenleft", ")": "parenright", "-": "hyphen", "'": "quotesingle",
               "?": "question", "!": "exclam", "[": "bracketleft", "]": "bracketright", "*": "asterisk"}


def glyph_name(letters: str) -> str:
    """A PostScript glyph name: the letter's own (a, A, one), or a ligature's (s_t: f_i)."""
    def one(c: str) -> str:
        if c in GLYPH_NAMES:
            return GLYPH_NAMES[c]
        if c.isascii() and c.isalnum():
            return c if not c.isdigit() else ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][int(c)]
        return f"uni{ord(c):04X}"
    return "_".join(one(c) for c in letters)


def build_font(style: str, shapes_: dict, metrics_: dict, family: str, path: Path,
               fallback: dict | None = None, svg_dir: Path | None = None) -> dict:
    """A TrueType font of one style from the averaged prints and the measured metrics (and the
    glyphs this style lacks from `fallback`, the other style's (shapes, metrics)): advances and
    bearings as printed, the word space, kerns. Writes `path` (and each glyph as SVG to
    `svg_dir`); returns {glyph: advance} in em."""
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    m = metrics_[style]
    have = {c: v for (st, c), v in shapes_.items() if st == style}
    sources = {c: (style, have[c][0]) for c in have}
    if fallback is not None:
        other_style, other_shapes = fallback
        for (st, c), v in other_shapes.items():
            if st == other_style and c not in sources:
                sources[c] = (other_style, v[0])
    lefts = [v["left"] for v in m["glyphs"].values() if v.get("left") is not None]
    default_bearing = float(np.median(lefts)) if lefts else 0.02
    glyph_order, cmap, glyphs, hmtx, advances = [".notdef", "space"], {32: "space"}, {}, {}, {}
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0)); pen.lineTo((50, 700)); pen.lineTo((450, 700)); pen.lineTo((450, 0)); pen.closePath()
    glyphs[".notdef"] = pen.glyph()
    hmtx[".notdef"] = (500, 50)
    space = ((m.get("spaces") or {}).get("space") or 0.25) * UPM
    glyphs["space"] = TTGlyphPen(None).glyph()
    hmtx["space"] = (round(space), 0)
    advances[" "] = space / UPM
    svgs = []
    for c, (src_style, canvas) in sorted(sources.items()):
        binary = canvas > 0.5
        cols = np.flatnonzero(binary.any(axis=0))
        if not len(cols):
            continue
        info = metrics_[src_style]["glyphs"].get(c, {})
        left = info.get("left") if info.get("left") is not None else default_bearing
        right = info.get("right") if info.get("right") is not None else default_bearing
        width = info.get("width") or (cols[-1] + 1 - cols[0]) / EM_PX
        # the ink from `left` on; the advance as printed (left + its ink + right)
        x_shift = cols[0] - left * EM_PX
        contours = [_font_points(ct, x_shift) for ct in trace(binary)]
        # TrueType: outer contours clockwise (y up), holes the other way
        fixed = []
        for pts in contours:
            inside = sum(1 for other in contours if other is not pts and _contains(other, pts[0]))
            want_clockwise = inside % 2 == 0
            if (_area(pts) < 0) != want_clockwise:
                pts = pts[::-1]
            fixed.append(pts)
        pen = TTGlyphPen(None)
        draw(pen, fixed)
        name = glyph_name(c)
        glyphs[name] = pen.glyph()
        advance = round((left + width + right) * UPM)
        x_min = min((x for pts in fixed for x, _, _ in pts), default=0)
        hmtx[name] = (max(advance, 1), round(x_min))
        glyph_order.append(name)
        advances[c] = advance / UPM
        if len(c) == 1:
            cmap[ord(c)] = name
        svgs.append((name, c, svg_path(fixed), advance))
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(hmtx)
    tops = [v["top"] for c, v in m["glyphs"].items() if c.isalpha() and v["count"] > 50]
    bottoms = [v["bottom"] for c, v in m["glyphs"].items() if c.isalpha() and v["count"] > 50]
    ascent, descent = round(max(tops, default=0.75) * UPM) + 30, round(-min(bottoms, default=-0.25) * UPM) + 20
    fb.setupHorizontalHeader(ascent=ascent, descent=-descent)
    style_name = "Italic" if style == "italic" else "Regular"
    fb.setupNameTable({"familyName": family, "styleName": style_name, "uniqueFontIdentifier": f"{family}-{style_name}",
                       "fullName": f"{family} {style_name}", "psName": f"{family.replace(' ', '')}-{style_name}",
                       "version": "Version 0.1", "copyright": "Traced from the scans of the 1732 printing"})
    xh = m["glyphs"].get("x", {}).get("top", 0.45)
    cap = m["glyphs"].get("H", {}).get("top", 0.7)
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=-descent, sTypoLineGap=0, usWinAscent=ascent, usWinDescent=descent,
                sxHeight=round(xh * UPM), sCapHeight=round(cap * UPM), fsSelection=0x01 if style == "italic" else 0x40,
                achVendID="PDFO")
    fb.setupPost(italicAngle=-14 if style == "italic" else 0)
    fb.setupHead(macStyle=0x02 if style == "italic" else 0)
    # the ligatures the compositor set (where the text has their letters), and the kerns
    ligs = [c for c in sources if len(c) > 1 and all(ch in sources for ch in c) and "ſ" not in c]
    fea = []
    if ligs:
        fea.append("feature liga {\n" + "".join(
            f"  sub {' '.join(glyph_name(ch) for ch in lig)} by {glyph_name(lig)};\n"
            for lig in sorted(ligs, key=len, reverse=True)) + "} liga;")
    kerns = [(a, b, v) for (a, b), v in ((tuple(k) if len(k) == 2 else (None, None), v) for k, v in m["kerns"].items())
             if a in sources and b in sources and len(a) == 1 and len(b) == 1]
    if kerns:
        fea.append("feature kern {\n" + "".join(f"  pos {glyph_name(a)} {glyph_name(b)} {round(v * UPM)};\n"
                                               for a, b, v in kerns) + "} kern;")
    if fea:
        addOpenTypeFeaturesFromString(fb.font, "\n".join(fea))
    path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(path))
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=True)
        for name, c, d, advance in svgs:
            # (named by code points: a and A are one file name where case is not told apart)
            (svg_dir / f"{style}-{'_'.join(f'{ord(ch):04X}' for ch in c)}.svg").write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-100 -{ascent} {advance + 200} {ascent + descent}">'
                f'<line x1="0" y1="0" x2="{advance}" y2="0" stroke="#c00" stroke-width="4"/>'
                f'<path transform="scale(1,-1)" d="{d}"/></svg>')
    return advances


def _contains(contour, point) -> bool:
    """Whether `point` (x, y, _) is inside the polygon of `contour`'s points (even-odd)."""
    x, y = point[0], point[1]
    inside = False
    for (x0, y0, _), (x1, y1, _) in zip(contour, contour[1:] + contour[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def build(out: Path, family: str = "Book", workers: int = 4, log=print) -> Path:
    """All four steps for a `reconstruct` run in `out` (its work/ directory): writes
    out/typeface/ with FAMILY-Regular.ttf, FAMILY-Italic.ttf, each glyph as SVG, and
    typeface.json (the fonts, their family names, the measured metrics) for
    `reconstruct --typeface`. Returns that directory."""
    import time

    start = time.perf_counter()
    work = out / "work"
    dirs = sorted(p for p in work.glob("page_*") if (p / "prepared.pkl").exists())
    if not dirs:
        raise ValueError(f"{work}: no prepared pages (run `pdf-ocr-bench reconstruct` first)")
    pages = scan(work, workers, log)
    log(f"{sum(len(p['glyphs']) for p in pages)} glyphs on {len(pages)} pages in {time.perf_counter() - start:.0f}s")
    measured = metrics(pages)
    averaged = shapes(pages, dirs, log=log)
    target = out / "typeface"
    files = {"roman": f"{family.replace(' ', '')}-Regular.ttf", "italic": f"{family.replace(' ', '')}-Italic.ttf"}
    names = {"roman": family, "italic": f"{family} Italic"}
    build_font("roman", averaged, measured, names["roman"], target / files["roman"], svg_dir=target / "svg")
    build_font("italic", averaged, measured, names["italic"], target / files["italic"], fallback=("roman", averaged),
               svg_dir=target / "svg")
    counts = {f"{st}:{c}": n for (st, c), (_, n) in averaged.items()}
    (target / "typeface.json").write_text(json.dumps(
        {"families": names, "files": files, "metrics": measured, "prints": counts}, ensure_ascii=False, indent=1))
    log(f"wrote {target} ({len(averaged)} glyphs averaged) in {time.perf_counter() - start:.0f}s")
    return target
