#!/usr/bin/env python3
"""Correct an OCR result page by page with Claude (`claude -p`): spelling, junk and layout.

    scripts/llm_correct.py results/<engine>/pages.zip --images results/images -o corrected.zip
        [--agents 20] [--model haiku] [--pages 1-20] [--pdf corrected.pdf]

1. `html2pdf --layout flow --long-s repair --html-out` turns the engine's positioned words into
   text blocks with paragraphs (every block knows where its lines were: `data-box`).
2. Every page goes to its own `claude -p` (up to `--agents` at once) with the blocks as JSON, the
   page reduced (for the layout) and cut into 8 overlapping full-resolution tiles, all in the
   first message, so a page is usually one request. For a word it still cannot read, the model
   may zoom with ImageMagick (`magick page.png -crop ...`) and Read the crop.
   It returns the blocks corrected, each with a role; `noise` (OCR junk from engravings and
   specks) is dropped, and marginal notes it pulls out of the running text become blocks of
   their own.
3. The corrected pages are written as a new zip (same metadata.json) that html2pdf renders as
   they are: `html2pdf corrected.zip -o corrected.pdf`.

A page whose call fails keeps the uncorrected flow layout. Everything the model saw and said is
kept next to the output in `<output>_work/page_NNN/`.

Needs `claude` (logged in) and ImageMagick's `magick` on PATH, and a built html2pdf.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HTML2PDF = ROOT / "html2pdf" / "target" / "release" / "html2pdf"
OVERVIEW_HEIGHT = 1000  # px: the whole page, for the layout; words are read from the tiles
TILES = (2, 4)  # columns x rows of full-resolution tiles
TILE_OVERLAP = 48  # px, so every line is whole in at least one tile

ROLES = ["text", "note", "header", "footnote", "noise"]
SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "role": {"type": "string", "enum": ROLES},
                    "box_px": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "role", "paragraphs"],
            },
        }
    },
    "required": ["regions"],
}

SYSTEM = """You proofread OCR of one scanned page of a book about the Bible: the English \
translation of Augustin Calmet's "Historical, Critical, Geographical, Chronological, and \
Etymological Dictionary of the Holy Bible", printed in London in 1732. The entries explain the \
persons, places, customs and words of the Old and New Testament, so expect biblical names \
(Aaron, Jochebed, Joshua, Jerusalem, Ægypt), Bible references ("Exod. iv. 14.", "Ch. xxix. 9.", \
"Ver. 6, 7."), citations of Church Fathers and Jewish writers ("Joseph. Antiq. l. 4. c. 2."), \
and Latin, Greek and Hebrew words. Layout: two columns, marginal notes with references, running \
heads, footnotes, sometimes copper-plate pictures. The OCR text is mostly right. Its typical \
errors:
- the long s (ſ) read as f: "Hiftorical" -> "Historical", "Mofes" -> "Moses", "fhall" -> \
"shall", "Condefcenfion" -> "Condescension"; also as l or t ("Moles", "themtelves")
- italic capitals misread: "Fefus" -> "Jesus", "Ifrael"/"1/rael" -> "Israel", "Fer." -> "Jer."
- other letters confused, and junk "words" read from pictures, ornaments and specks
- marginal notes ("Ch. iv. 27.", "Rev. i. 8.") run into the middle of a sentence of the text
- lines of different blocks interleaved

Your job: make every block say exactly what the page says.
- You get the page reduced (layout) and cut into full-resolution tiles (their place on page.png \
is given). Compare every block with the tiles word by word.
- The tiles are the scan at full resolution. Zoom into anything you cannot read with certainty, \
as often as you need: a single-line Bash command that starts with magick and makes the crops, \
e.g. `magick page.png -crop 300x80+410+1220 +repage -resize 200% zoom_1.png && magick page.png \
-crop ... zoom_2.png`, then Read the crops (absolute paths; several Reads in one turn).
- Keep the 1732 text as printed: its spelling (thro', shew, perform'd, compleat), \
capitalisation, punctuation and abbreviations. Only fix what the OCR got wrong. Write the long \
s as s. Never modernise, translate, summarise, add or drop real text.
- Return EVERY block you were given, each with its id, corrected or unchanged: the page is \
rebuilt from your answer. One string per paragraph as on the page; a word hyphenated at a line \
end stays joined as the OCR has it. Text that belongs to another block: move it there, and \
return the emptied block with no paragraphs.
- role: text (running text), note (marginal note), header (running head, page number), \
footnote, noise (junk from a picture or specks: the block is removed).
- A marginal note that ended up inside a text paragraph: take it out of the paragraph and give \
it a block of its own with role note, a new id (n1, n2, ...) and its box_px on page.png.
- Blocks the OCR missed entirely: add them the same way (new id, box_px).
Answer with all blocks."""

NO_ZOOM = """
You have no tools this time: read everything from the tiles you were given and answer directly."""


@dataclass
class Region:
    id: str
    style: str  # the flow layout's inline style: position, font size, line height
    box: tuple[float, float, float, float]  # where its lines were: left top right bottom, % of the page
    paragraphs: list[str]
    indents: list[str]  # each paragraph's own style ("" or "text-indent: ...")


REGION = re.compile(r'<div class="region" data-box="([^"]*)" style="([^"]*)">(.*?)</div>', re.S)
PARAGRAPH = re.compile(r'<p(?: style="([^"]*)")?>(.*?)</p>', re.S)


def parse_regions(page_html: str) -> list[Region]:
    return [
        Region(
            id=f"r{i}",
            style=style,
            box=tuple(float(v) for v in box.split()),
            paragraphs=[html.unescape(text) for _, text in PARAGRAPH.findall(body)],
            indents=[indent for indent, _ in PARAGRAPH.findall(body)],
        )
        for i, (box, style, body) in enumerate(REGION.findall(page_html), 1)
    ]


def page_prompt(regions: list[Region], width: int, height: int, work: Path) -> str:
    blocks = [
        {
            "id": r.id,
            "box_px": [round(r.box[0] / 100 * width), round(r.box[1] / 100 * height),
                       round((r.box[2] - r.box[0]) / 100 * width), round((r.box[3] - r.box[1]) / 100 * height)],
            "paragraphs": r.paragraphs,
        }
        for r in regions
    ]
    return (
        f"Working directory {work}: page.png is the scan ({width}x{height} px); zoom crops go there "
        f"and are Read by absolute path ({work}/zoom_1.jpg). "
        f"The OCR blocks in reading order (box_px = X Y W H on page.png):\n\n"
        + json.dumps(blocks, ensure_ascii=False, indent=1)
    )


def tiles(gray: Image.Image, work: Path) -> list[tuple[str, Path]]:
    """The page cut into overlapping full-resolution tiles: (label with its place, file)."""
    width, height = gray.size
    cols, rows = TILES
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * width // cols - TILE_OVERLAP)
            y0 = max(0, r * height // rows - TILE_OVERLAP)
            x1 = min(width, (c + 1) * width // cols + TILE_OVERLAP)
            y1 = min(height, (r + 1) * height // rows + TILE_OVERLAP)
            tile = gray.crop((x0, y0, x1, y1))
            name = work / f"tile_{r + 1}{'LR'[c] if cols == 2 else c + 1}"
            if bilevel(gray):  # a black-and-white scan: 1-bit PNG is a seventh of the JPEG
                path = name.with_suffix(".png")
                tile.point(lambda v: 255 if v > 128 else 0).convert("1").save(path, optimize=True)
            else:
                path = name.with_suffix(".jpg")
                tile.save(path, quality=75)
            out.append((f"tile {path.stem[5:]}: X {x0} Y {y0} W {x1 - x0} H {y1 - y0} on page.png", path))
    return out


def bilevel(gray: Image.Image) -> bool:
    """Nearly every pixel black or white (a 1-bit scan rendered at its own resolution)."""
    hist = gray.histogram()
    return sum(hist[16:240]) < 0.15 * sum(hist)  # a render smooths a few edge pixels


def image_block(path: Path) -> dict:
    import base64

    media = "image/png" if path.suffix == ".png" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def corrected_html(flow_html: str, regions: list[Region], answer: list[dict], width: int, height: int) -> str:
    """The flow page with the model's blocks: known ids keep their place and size."""
    by_id = {r.id: r for r in regions}
    sizes = sorted(float(m) for r in regions for m in re.findall(r"font-size: ([\d.]+)pt", r.style))
    note_size = sizes[len(sizes) // 4] if sizes else 10.0  # notes are set small
    answered = {b.get("id"): b for b in answer}
    # Every original block in its place, then the new ones. A block the model left out keeps the
    # OCR text, unless the answer holds (nearly) all of the page's text: then it was merged into
    # another block or dropped as junk.
    chars = lambda paragraphs: sum(len(p) for p in paragraphs)  # noqa: E731
    covered = chars(p for b in answer if b.get("role") != "noise" for p in b.get("paragraphs", []))
    complete = covered >= 0.9 * chars(p for r in regions for p in r.paragraphs)
    blocks = [
        answered.get(r.id) or ({"id": r.id, "role": "noise", "paragraphs": []} if complete else
                               {"id": r.id, "role": "text", "paragraphs": r.paragraphs})
        for r in regions
    ]
    blocks += [b for b in answer if b.get("id") not in by_id]
    out = []
    for block in blocks:
        if block.get("role") == "noise" or not any(p.strip() for p in block.get("paragraphs", [])):
            continue
        original = by_id.get(block.get("id", ""))
        if original is not None:
            style = original.style
            indents = original.indents if len(original.indents) == len(block["paragraphs"]) else []
        elif len(block.get("box_px") or []) == 4:
            x, y, w, h = block["box_px"]
            style = (
                f"left: {100 * x / width:.4f}%; top: {100 * y / height:.4f}%; width: {100 * w / width:.4f}%; "
                f"font-size: {note_size:.2f}pt; line-height: {1.15 * note_size:.2f}pt; text-align: left;"
            )
            indents = []
        else:
            continue  # a new block without a place
        role = html.escape(block.get("role", "text"))
        paragraphs = "".join(
            f'<p style="{indents[i]}">{html.escape(p)}</p>' if indents and indents[i] else f"<p>{html.escape(p)}</p>"
            for i, p in enumerate(block["paragraphs"])
        )
        out.append(f'<div class="region" data-role="{role}" style="{style}">{paragraphs}</div>')
    start = flow_html.index('<div class="page">') + len('<div class="page">')
    end = flow_html.rindex("</div>")
    return flow_html[:start] + "\n" + "\n".join(out) + "\n" + flow_html[end:]


def run_answer(path: Path) -> tuple[list[dict] | None, dict]:
    """The blocks of a successful `claude -p` result in a stream-json transcript, and the result event."""
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
    except (OSError, json.JSONDecodeError):
        return None, {}
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    answer = (result.get("structured_output") or {}).get("regions")
    return (answer if answer and not result.get("is_error") else None), result


def cached_answer(work: Path) -> list[dict] | None:
    """A page answered by an earlier run (delete its work dir to ask again)."""
    for name in ("response.jsonl", "response_nozoom.jsonl"):
        answer, _ = run_answer(work / name)
        if answer:
            return answer
    return None


def ask(args, message: str, work: Path, zoom: bool) -> tuple[list[dict] | None, dict]:
    """One `claude -p` for the page; with `zoom` it may crop the scan with magick and Read the crops."""
    tools = ["--tools", "Read", "Bash", "--allowedTools", "Read", "Bash(magick:*)"] if zoom else ["--tools", ""]
    cmd = [
        args.claude, "-p",
        "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--model", args.model,
        "--system-prompt", SYSTEM if zoom else SYSTEM + NO_ZOOM,
        *tools,
        "--json-schema", json.dumps(SCHEMA),
        "--max-turns", str(args.max_turns if zoom else 3),
        "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
    ]  # fmt: skip
    transcript = work / ("response.jsonl" if zoom else "response_nozoom.jsonl")
    try:
        proc = subprocess.run(cmd, cwd=work, input=message, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, {"error": f"timed out after {args.timeout}s"}
    transcript.write_text(proc.stdout, encoding="utf-8")
    answer, result = run_answer(transcript)
    if not result:
        result = {"error": proc.stderr.strip()[-300:] or f"claude exited {proc.returncode}"}
    return answer, result


def correct_page(name: str, flow_html: str, image: Path, work: Path, args) -> tuple[str, str]:
    """(page html, status line). Falls back to the flow page on any failure."""
    work.mkdir(parents=True, exist_ok=True)
    regions = parse_regions(flow_html)
    with Image.open(image) as img:
        gray = img.convert("L")
        gray.save(work / "page.png", optimize=True)
        width, height = gray.size
        scale = OVERVIEW_HEIGHT / height
        gray.resize((round(width * scale), OVERVIEW_HEIGHT), Image.LANCZOS).save(work / "overview.jpg", quality=60)
        page_tiles = tiles(gray, work)
    prompt = page_prompt(regions, width, height, work.resolve())
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    content = [{"type": "text", "text": "The whole page, reduced:"}, image_block(work / "overview.jpg")]
    for label, path in page_tiles:
        content += [{"type": "text", "text": label}, image_block(path)]
    content.append({"type": "text", "text": prompt})
    message = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
    start = time.perf_counter()
    answer, result, how = (None if args.fresh else cached_answer(work)), {}, "cached"
    if answer is None and not args.no_zoom:
        answer, result = ask(args, message, work, zoom=True)
        how = "zoom"
    if answer is None:
        # out of turns while zooming (or --no-zoom): answer from the full-resolution tiles alone
        answer, result = ask(args, message, work, zoom=False)
        how = "no zoom" if args.no_zoom else "retried without zoom"
    if answer is None:
        return flow_html, f"{name}: kept uncorrected ({result.get('subtype') or result.get('error') or 'no answer'})"
    page = corrected_html(flow_html, regions, answer, width, height)
    zooms = len(list(work.glob("zoom*")))
    dropped = sum(1 for b in answer if b.get("role") == "noise")
    added = sum(1 for b in answer if b.get("id") not in {r.id for r in regions})
    return page, (
        f"{name}: {len(regions)} blocks -> {len(answer)} ({dropped} noise, {added} new), {zooms} zooms, "
        f"{how}, {result.get('num_turns', '-')} turns, {time.perf_counter() - start:.0f}s"
    )


def page_numbers(spec: str | None, count: int) -> set[int] | None:
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        pages.update(range(int(lo), int(hi or lo) + 1))
    return {p for p in pages if 1 <= p <= count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zip", type=Path, help="an engine's pages.zip")
    parser.add_argument("--images", type=Path, required=True,
                        help="page renders (page_NNN.png), any DPI: e.g. the engine's input, or the native-resolution render")
    parser.add_argument("-o", "--output", type=Path, required=True, help="corrected zip")
    parser.add_argument("--pdf", type=Path, help="also render the corrected zip to this PDF")
    parser.add_argument("--agents", type=int, default=20, help="pages corrected at once (default 20)")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--pages", help="only these pages (1-based, e.g. 1-5,9); the others stay uncorrected")
    parser.add_argument("--max-turns", type=int, default=60,
                        help="turns for the zooming pass; a page that runs out is asked again without tools")
    parser.add_argument("--no-zoom", action="store_true", help="no tools at all: one request per page, least data")
    parser.add_argument("--fresh", action="store_true", help="ask again for pages answered by an earlier run")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per page")
    parser.add_argument("--claude", default=shutil.which("claude") or "claude")
    args = parser.parse_args()
    for tool in (args.claude, "magick"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} is not on PATH")
    if not HTML2PDF.exists():
        sys.exit(f"{HTML2PDF} missing: cargo build --release --manifest-path html2pdf/Cargo.toml")

    work = args.output.with_name(args.output.stem + "_work")
    flow_dir = work / "flow"
    shutil.rmtree(flow_dir, ignore_errors=True)
    subprocess.run(
        [HTML2PDF, args.zip, "-o", work / "flow.pdf", "--layout", "flow", "--long-s", "repair", "--html-out", flow_dir],
        check=True, capture_output=True,
    )  # fmt: skip

    with zipfile.ZipFile(args.zip) as z:
        metadata = json.loads(z.read("metadata.json"))
    pages = sorted(metadata["pages"], key=lambda p: p["page_num"])
    wanted = page_numbers(args.pages, max(p["page_num"] for p in pages) + 1)
    todo = [p for p in pages if wanted is None or p["page_num"] + 1 in wanted]
    out = {p["html"]: (flow_dir / p["html"]).read_text(encoding="utf-8") for p in pages}

    print(f"[correct] {len(todo)} of {len(pages)} pages with {args.model}, {args.agents} at a time; work dir {work}", flush=True)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.agents) as pool:
        jobs = {
            pool.submit(
                correct_page, p["html"], out[p["html"]],
                args.images / p["html"].replace(".html", ".png"), work / p["html"].removesuffix(".html"), args,
            ): p["html"]
            for p in todo
        }  # fmt: skip
        for job in as_completed(jobs):
            page, status = job.result()
            out[jobs[job]] = page
            print(f"[correct] {status}", flush=True)

    metadata["engine"] = f"{metadata['engine']}+{args.model}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        for name, page in out.items():
            z.writestr(name, page)
    print(f"[correct] wrote {args.output} in {time.perf_counter() - start:.0f}s", flush=True)
    if args.pdf:
        subprocess.run([HTML2PDF, args.output, "-o", args.pdf], check=True)


if __name__ == "__main__":
    main()
