"""The structure of a page from an LLM: paragraphs, notes and headings from the OCR lines, corrected.

The model gets the page's zones (from `page_layout`) with Vision's lines in each, and the scan
(`claude_cli.page_content`). It answers with items in reading order: which OCR lines make a
paragraph, a marginal note or a heading line, the corrected text of exactly those lines, a drop
capital, the alignment. Lines it calls noise (OCR junk) are dropped. Placement stays with the
geometry: an item is set where its first line was, as wide as its zone.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import claude_cli
from .log import get_logger
from .page_layout import PageLayout

log = get_logger("Structure")

SYSTEM = """You restore one scanned page of a book from its OCR: the text, and how it is grouped.

You get the page as images, and its zones (found from the pixels: running head, text columns, \
marginal-note strips, footnotes, other text blocks such as titles and captions) with the OCR lines \
read in each, as JSON: line ids, text, and their box X Y W H on page.png.

Answer with items in reading order (zone by zone: running head, then the columns left to right, \
with the marginal notes where they stand, then footnotes):
- kind "paragraph": a paragraph of running text; "note": one marginal note (a reference like \
"Ch. iv. 27." or "Rev. i. 8.", a date); "heading": one printed line of a title, a running head, an \
entry heading or a caption, each line its own item; "noise": OCR junk read from a picture, an \
ornament or specks (dropped).
- lines: the ids of the OCR lines the item was read from, in order. Every OCR line belongs to \
exactly one item. If the OCR split one printed line in two, list both ids; if it merged a note into \
a line of the text, give the line to the item it mostly belongs to and leave the note's words out \
of the paragraph.
- text: the item's text as printed, corrected against the images: the long s (ſ) written as s \
("fhould" -> "should", "Mofes" -> "Moses"), misread letters and italic capitals ("Fefus" -> "Jesus") \
fixed, words split by a line-end hyphen joined. Keep the book's own spelling, capitals, punctuation \
and abbreviations (thro', shew, perform'd); never modernise, translate, summarise, add or drop text.
- drop_cap: the letter, when the paragraph opens with a large (drop) initial; the text then starts \
with that letter, though the OCR lines may not contain it.
- align: justify for running text, center for centred lines, left or right otherwise.
- italic: true when the item is (mostly) set in italic type."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "kind": {"type": "string", "enum": ["paragraph", "note", "heading", "noise"]},
                    "lines": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"},
                    "drop_cap": {"type": "string"},
                    "align": {"type": "string", "enum": ["justify", "left", "center", "right"]},
                    "italic": {"type": "boolean"},
                },
                "required": ["zone", "kind", "lines", "text"],
            },
        }
    },
    "required": ["items"],
}


def page_prompt(layout: PageLayout, lines: list, semantic_context: str) -> str:
    zones = []
    for z in layout.zones:
        if z.role in ("picture", "dropcap"):
            zones.append({"zone": z.id, "role": z.role, "box": [z.box.x0, z.box.y0, z.box.w, z.box.h]})
            continue
        mine = [l for l in lines if l.zone == z.id]
        zones.append({
            "zone": z.id, "role": z.role, **({"side": z.side} if z.side else {}),
            "box": [z.box.x0, z.box.y0, z.box.w, z.box.h],
            "lines": [{"id": l.id, "text": l.text, "box": [l.box.x0, l.box.y0, l.box.w, l.box.h]}
                      for l in sorted(mine, key=lambda l: (l.box.y0, l.box.x0))],
        })
    context = f"About the book: {semantic_context}\n\n" if semantic_context else ""
    return context + "The page's zones and OCR lines (boxes X Y W H on page.png):\n\n" + json.dumps(zones, ensure_ascii=False)


def to_items(answer: dict, lines: list, layout: PageLayout):
    """The answer as `reconstruct.Item`s, checked against the OCR lines; None if it lost text."""
    from .reconstruct import Item

    known = {l.id: l for l in lines}
    zones = {z.id for z in layout.zones}
    items, used = [], set()
    for a in answer.get("items", []):
        ids = [i for i in a.get("lines", []) if i in known and i not in used]
        used.update(ids)
        if a.get("kind") == "noise" or not ids or not a.get("text", "").strip():
            continue
        zone = a.get("zone") if a.get("zone") in zones else known[ids[0]].zone
        items.append(Item(zone=zone, text=a["text"].strip(), lines=ids, kind=a.get("kind", "paragraph"),
                          drop_cap=(a.get("drop_cap") or "")[:1], align=a.get("align") or "justify",
                          italic=bool(a.get("italic"))))
    # lines the answer left out: fine when they are a little junk, not when text went missing
    missing = sum(len(l.text) for i, l in known.items() if i not in used)
    total = sum(len(l.text) for l in lines) or 1
    if missing > 0.15 * total:
        return None
    return items


def structure_pages(results: dict[int, dict], work: Path, semantic_context: str, model: str, agents: int,
                    zoom: bool = False) -> dict[int, list]:
    """{page index: items} for the pages the model answered; the others fall back to the heuristic."""
    def one(n: int, r: dict):
        pw = work / r["name"] / "llm"
        pw.mkdir(parents=True, exist_ok=True)
        prompt = page_prompt(r["layout"], r["lines"], semantic_context)
        # an earlier answer counts only for the same zones and lines (and model and context)
        key = f"{model}\n{SYSTEM}\n{prompt}"
        same = (pw / "prompt.txt").exists() and (pw / "prompt.txt").read_text(encoding="utf-8") == key
        answer, result = (claude_cli.answer_of(pw / "response.jsonl")[0] if same else None), {"num_turns": 0}
        if answer is None:
            (pw / "prompt.txt").write_text(key, encoding="utf-8")
            content, _ = claude_cli.page_content(work / r["name"] / "native.png", pw)
            content.append({"type": "text", "text": prompt})
            answer, result = claude_cli.ask(content, SYSTEM, SCHEMA, pw, model, zoom=zoom)
        items = to_items(answer, r["lines"], r["layout"]) if answer else None
        return n, items, result

    out: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=max(1, agents)) as pool:
        for job in as_completed([pool.submit(one, n, r) for n, r in results.items()]):
            n, items, result = job.result()
            name = results[n]["name"]
            if items is None:
                log.warning(f"{name}: no usable answer ({result.get('subtype') or result.get('error') or 'lost text'}); heuristic structure")
                continue
            out[n] = items
            log.info(f"{name}: {len(items)} items from {model} ({result.get('num_turns', 0)} turns)")
    return out
