"""Asking Claude about a page through the `claude` CLI (`claude -p`, so a Claude subscription works).

The page goes in the first message: reduced to 1000 px high for the layout, and cut into 8
overlapping tiles at full resolution (1-bit PNG for a black-and-white scan, a seventh of a JPEG).
The answer is structured (`--json-schema`). With `zoom`, the model may crop the scan with
ImageMagick and Read the crops; every turn then re-sends all images, so a page costs several MB.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

OVERVIEW_HEIGHT = 1000
TILES = (2, 4)  # columns x rows
TILE_OVERLAP = 48  # px, so every line is whole in at least one tile

ZOOM_HELP = """- The tiles are the scan at full resolution. Zoom into anything you cannot read with certainty: a \
single-line Bash command that starts with magick and makes the crops, e.g. `magick page.png -crop \
300x80+410+1220 +repage -resize 200% zoom_1.png && magick page.png -crop ... zoom_2.png`, then Read the \
crops (absolute paths; several Reads in one turn)."""
NO_ZOOM = "- Read everything from the tiles you were given (you have no tools)."


def bilevel(gray) -> bool:
    """Nearly every pixel black or white (a 1-bit scan, rendered at about its own resolution)."""
    hist = gray.histogram()
    return sum(hist[16:240]) < 0.15 * sum(hist)


def image_block(path: Path) -> dict:
    media = "image/png" if path.suffix == ".png" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def page_content(image: Path, work: Path) -> tuple[list[dict], tuple[int, int]]:
    """Message content for a page: the reduced page and the tiles, each labelled with its place
    on page.png (written to `work`, for zooming); and page.png's size."""
    from PIL import Image

    work.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as img:
        gray = img.convert("L")
    gray.save(work / "page.png", optimize=True)
    width, height = gray.size
    scale = OVERVIEW_HEIGHT / height
    gray.resize((round(width * scale), OVERVIEW_HEIGHT), Image.LANCZOS).save(work / "overview.jpg", quality=60)
    content = [{"type": "text", "text": "The whole page, reduced:"}, image_block(work / "overview.jpg")]
    one_bit = bilevel(gray)
    cols, rows = TILES
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * width // cols - TILE_OVERLAP)
            y0 = max(0, r * height // rows - TILE_OVERLAP)
            x1 = min(width, (c + 1) * width // cols + TILE_OVERLAP)
            y1 = min(height, (r + 1) * height // rows + TILE_OVERLAP)
            tile = gray.crop((x0, y0, x1, y1))
            name = work / f"tile_{r + 1}{'LR'[c] if cols == 2 else c + 1}"
            if one_bit:
                path = name.with_suffix(".png")
                tile.point(lambda v: 255 if v > 128 else 0).convert("1").save(path, optimize=True)
            else:
                path = name.with_suffix(".jpg")
                tile.save(path, quality=75)
            content += [{"type": "text", "text": f"tile {path.stem[5:]}: X {x0} Y {y0} W {x1 - x0} H {y1 - y0} on page.png"},
                        image_block(path)]
    return content, (width, height)


def answer_of(transcript: Path) -> tuple[dict | None, dict]:
    """The structured answer of a successful run in a stream-json transcript, and its result event."""
    try:
        events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
    except (OSError, json.JSONDecodeError):
        return None, {}
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    answer = result.get("structured_output")
    return (answer if answer and not result.get("is_error") else None), result


def ask(content: list[dict], system: str, schema: dict, work: Path, model: str, zoom: bool = False,
        max_turns: int = 60, timeout: int = 900, name: str = "response") -> tuple[dict | None, dict]:
    """One `claude -p` with `content` as the user message; its transcript is kept in `work`."""
    claude = shutil.which("claude")
    if claude is None:
        return None, {"error": "claude is not on PATH"}
    tools = ["--tools", "Read", "Bash", "--allowedTools", "Read", "Bash(magick:*)"] if zoom else ["--tools", ""]
    cmd = [
        claude, "-p",
        "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--system-prompt", system + "\n" + (ZOOM_HELP if zoom else NO_ZOOM),
        *tools,
        "--json-schema", json.dumps(schema),
        "--max-turns", str(max_turns if zoom else 3),
        "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
    ]  # fmt: skip
    message = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
    transcript = work / f"{name}.jsonl"
    try:
        proc = subprocess.run(cmd, cwd=work, input=message, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, {"error": f"timed out after {timeout}s"}
    transcript.write_text(proc.stdout, encoding="utf-8")
    answer, result = answer_of(transcript)
    if not result:
        result = {"error": proc.stderr.strip()[-300:] or f"claude exited {proc.returncode}"}
    return answer, result
