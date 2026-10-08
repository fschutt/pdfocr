"""The structure of a page from an LLM: paragraphs, notes and headings from the OCR lines, corrected.

The model gets the page's zones (from `page_layout`) with Vision's lines in each, and the scan
(`claude_cli.page_content`). It answers with items in reading order: which OCR lines make a
paragraph, a marginal note or a heading line, the corrected text of exactly those lines, a drop
capital, the alignment. Lines it calls noise (OCR junk) are dropped. Placement stays with the
geometry: an item is set where its first line was, as wide as its zone.

`learnings` turns an earlier run (a larger model on the first pages) into extra instructions for
the rest: the corrections it made most often, and one page worked through.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
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
and abbreviations (thro', shew, perform'd); never modernise, translate, summarise, add or drop text. \
Mark the words printed in the other style with <i>...</i>: in upright text the italic ones (names, \
references such as "<i>Jer.</i> i. 6.", Latin words, emphasis), in an italic item the upright ones. \
No other markup.
- drop_cap: the letter, when the paragraph opens with a large (drop) initial; the text then starts \
with that letter, though the OCR lines may not contain it.
- align: justify for running text, center for centred lines, left or right otherwise.
- italic: true when the item is (mostly) set in italic type (its upright words then marked with <i>)."""

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
    from .reconstruct import Item, clean_marks, normalize_markup, plain

    known = {l.id: l for l in lines}
    zones = {z.id for z in layout.zones}
    answered = answer.get("items", [])
    # a line the answer lists for two items (the last lines of one paragraph and the first of the
    # next) belongs to the one whose text has it; the other would be set where it is not printed
    claims: dict[str, list[int]] = {}
    for k, a in enumerate(answered):
        for i in dict.fromkeys(a.get("lines", [])):
            if i in known:
                claims.setdefault(i, []).append(k)
    owner = {i: max(ks, key=lambda k: (_shared_pairs(known[i].text, plain(answered[k].get("text", ""))), -k))
             for i, ks in claims.items()}
    lines_of = {k: [i for i in dict.fromkeys(a.get("lines", [])) if i in known and owner[i] == k]
                for k, a in enumerate(answered)}
    _reassign(lines_of, answered, known)
    _drop_strays(lines_of, known)
    items, used = [], set()
    for k, a in enumerate(answered):
        ids = lines_of[k]
        text = clean_marks(normalize_markup(a.get("text", "").strip()))
        # (an item with lines and no text read none of them: the model listed a column's 34 lines
        # and the footnotes' with empty texts, and the page lost 40% of its text, p. 247)
        if a.get("kind") == "noise" or plain(text).strip():
            used.update(ids)
        if a.get("kind") == "noise" or not plain(text).strip():
            continue
        # no lines of its own (a margin note or a title line the model read from the scan, which
        # Vision ran into the line beside it or did not read): kept, placed by its words
        zone = a.get("zone") if a.get("zone") in zones else (known[ids[0]].zone if ids else "")
        items.append(Item(zone=zone, text=text, lines=ids, kind=a.get("kind", "paragraph"),
                          drop_cap=(a.get("drop_cap") or "")[:1], align=a.get("align") or "justify",
                          italic=bool(a.get("italic")), label=str(a.get("zone") or "")))
    _join_split(items, known)  # (before the trim: a trimmed paragraph ends mid-sentence)
    _trim_repeated(items, known)
    # lines the answer left out: fine when they are a little junk, not when text went missing
    missing = sum(len(l.text) for i, l in known.items() if i not in used)
    total = sum(len(l.text) for l in lines) or 1
    if missing > 0.15 * total:
        return None
    return items


def _trim_repeated(items: list, known: dict) -> None:
    """An item whose text runs on past what its lines read, into the text of another item (the
    model gave a paragraph's lines in one column its whole text, and its lines in the next
    column, with that text again, an item of their own: 91 words drawn where 5 lines were
    printed, p. 455): its text ends where that other item's begins."""
    import difflib

    from .reconstruct import _norm_word, _split_at, plain

    # (paragraphs, and eight words alike: the chronological notes all begin "In the Year of the
    # World", and one was cut to nothing)
    starts = []
    for item in items:
        head = [k for k in (_norm_word(w) for w in plain(item.text).split()) if k][:8]
        if item.kind == "paragraph" and len(head) == 8:
            starts.append((item, head))
    for item in items:
        if not item.lines or item.kind != "paragraph":
            continue
        words = item.text.split()
        keys = [_norm_word(plain(w)) for w in words]
        read = [k for i in item.lines if i in known for k in (_norm_word(w) for w in known[i].text.split()) if k]
        match = difflib.SequenceMatcher(None, read, keys, autojunk=False)
        end = max((b + n for _, b, n in match.get_matching_blocks() if n), default=0)
        if end < 3 or end > len(words) - 8:
            continue  # its lines read none of it, or it to (nearly) its end
        for other, head in starts:
            if other is item:
                continue
            for t in range(end, len(keys) - 7):
                if keys[t:t + 8] == head:
                    item.text = _split_at(item.text, t)[0]
                    break
            else:
                continue
            break


def _join_split(items: list, known: dict) -> None:
    """A paragraph of a line or two that ends mid-sentence, and the next paragraph, on one printed
    row with it, are one paragraph: Vision read the first line in two pieces (the headword and
    the rest), and the model made "ABIMELECH. The Priest of the" an item and "Lord, who gave
    Goliath's Sword ..." another, with the headword's line; both were set at that row (p. 35,
    p. 792). Or Vision read the row twice ("phetess Huldah. Many have been of" | "opinion, that
    the Lamentations", p. 1003)."""
    from .reconstruct import plain

    k = 0
    while k + 1 < len(items):
        a, b = items[k], items[k + 1]
        la = [known[i] for i in a.lines if i in known]
        lb = [known[i] for i in b.lines if i in known]
        end = plain(a.text).rstrip().rstrip("\"'\u201d\u2019)")
        if (a.kind == b.kind == "paragraph" and "foot" not in (a.label + b.label).lower() and la and lb
                and len(la) <= 2 and end and end[-1] not in ".!?:;"
                and any(abs(x.box.y0 - y.box.y0) < 0.5 * max(x.box.h, 1) and x.zone == y.zone for x in la for y in lb)):
            items[k] = replace(a, text=a.text.rstrip() + " " + b.text.lstrip(), lines=a.lines + [i for i in b.lines if i not in a.lines])
            del items[k + 1]
            continue
        k += 1


def _pairs(text: str) -> set[tuple[str, str]]:
    words = re.findall(r"\w+", text.lower().replace("ſ", "s").replace("f", "s"))
    return set(zip(words, words[1:]))


def _reassign(lines_of: dict[int, list[str]], answered: list[dict], known: dict) -> None:
    """A line the answer gives an item whose text has next to none of it (under 30% of its pairs
    of words), when one other item's text has most of it (60%, or 50% and no other 30%), goes to that one: the line ids of
    a page may be off by a line or two (p. 41 of vol. 1: its first lines given to the running
    head, and two paragraphs a line each in turn), the text the model read is right. It goes
    after that item's nearest line above it in its zone (the item's order: its column first)."""
    from .reconstruct import plain

    texts = [_pairs(plain(a.get("text", ""))) for a in answered]
    moves = []
    for k, ids in lines_of.items():
        for i in ids:
            pairs = _pairs(known[i].text)
            if len(pairs) < 3:
                continue
            own = len(pairs & texts[k]) / len(pairs)
            if own >= 0.3:
                continue
            scores = sorted(((len(pairs & t) / len(pairs), j) for j, t in enumerate(texts) if j != k), reverse=True)
            # (or half of it, no other item a third: OCR misreads and a word hyphenated at the
            # line's start, "zon like the other Stais", p. 41)
            runner_up = scores[1][0] if len(scores) > 1 else 0.0
            if scores and ((scores[0][0] >= 0.6 and runner_up < 0.6) or (scores[0][0] >= 0.5 and runner_up < 0.3)):
                moves.append((i, k, scores[0][1]))
    for i, k, j in moves:
        lines_of[k].remove(i)
        line = known[i]
        above = [n for n, m in enumerate(lines_of[j]) if known[m].zone == line.zone and known[m].box.y0 <= line.box.y0]
        at = max(above, key=lambda n: known[lines_of[j][n]].box.y0) + 1 if above else 0
        lines_of[j].insert(at, i)


def _drop_strays(lines_of: dict[int, list[str]], known: dict) -> None:
    """A line of a word or two the answer gives an item, printed far from the item's other lines
    in its column with another item's lines between, is no part of it (it would set the item's
    top there): a fragment ("In", ten lines above the paragraph it was given to, p. 58), the
    running head (p. 41)."""
    owner = {i: k for k, ids in lines_of.items() for i in ids}
    for k, ids in lines_of.items():
        if len(ids) < 3:
            continue
        heights = sorted(known[i].box.h for i in ids)
        h = heights[len(heights) // 2]
        strays = []
        for i in ids:
            line = known[i]
            if len(line.text.split()) > 2:
                continue

            def beside(o) -> bool:  # in its column (overlapping it across)
                return min(o.box.x1, line.box.x1) > max(o.box.x0, line.box.x0)

            rest = [known[j] for j in ids if j != i and beside(known[j])]
            if not rest:
                continue  # alone in its column: a part of its own
            near = min(rest, key=lambda o: abs(o.box.y0 - line.box.y0))
            if abs(near.box.y0 - line.box.y0) <= 4 * h:
                continue
            lo, hi = sorted((near.box.y0, line.box.y0))
            if any(owner.get(j, k) != k and lo < o.box.y0 < hi and beside(o) for j, o in known.items()):
                strays.append(i)
        for i in strays:
            ids.remove(i)


def _shared_pairs(line: str, text: str) -> int:
    """How many pairs of adjacent words of `line` (OCR, f or ſ for a long s) are in `text`."""
    def pairs(s: str) -> set[tuple[str, str]]:
        words = re.findall(r"\w+", s.lower().replace("ſ", "s").replace("f", "s"))
        return set(zip(words, words[1:]))

    return len(pairs(line) & pairs(text))


# `claude -p` ran into the subscription's limit: asking further pages only uploads them for nothing
# ("You've hit your session limit · resets 6pm (Europe/Berlin)", HTTP 429)
LIMIT_HIT = re.compile(r"\blimit\b|\bresets?\b", re.I)
PUNCT = ".,;:!?()[]'\"*"
WORD = re.compile(r"[A-Za-z][A-Za-z']*")
MARKS = re.compile(r"</?(?:i|sup|sub)>")


def _word_list() -> set[str]:
    try:
        from .reconstruct import WORD_LIST

        return {w.strip().lower() for w in Path(WORD_LIST).read_text(encoding="utf-8").splitlines()}
    except OSError:
        return set()


def learnings(source: Path, glossary: int = 150) -> str:
    """Extra instructions from an earlier run (`source`, a reconstruct output directory): the OCR
    words its model corrected most often, and one of its pages worked through. "" if it has none."""
    fixes: Counter = Counter()
    pages = []
    # only misreadings that are no word: "whole -> whose" or "fame -> same" was right on its page,
    # as a rule it would turn every correct "whole" into a mistake
    words = _word_list()
    for pw in sorted((source / "work").glob("page_*")):
        answer, _ = claude_cli.answer_of(pw / "llm" / "response.jsonl")
        if not answer or not (pw / "lines.json").exists() or not (pw / "llm" / "prompt.txt").exists():
            continue
        text_of = {l["id"]: l["text"] for l in json.loads((pw / "lines.json").read_text(encoding="utf-8"))}
        for item in answer.get("items", []):
            if item.get("kind") == "noise":
                continue
            ocr = " ".join(text_of.get(i, "") for i in item.get("lines", [])).split()
            fixed = MARKS.sub("", item.get("text", "")).split()
            for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=ocr, b=fixed, autojunk=False).get_opcodes():
                if op != "replace" or a1 - a0 != b1 - b0:
                    continue  # a word joined across a line-end hyphen, a note left out: not a misreading
                for x, y in zip(ocr[a0:a1], fixed[b0:b1]):
                    x, y = x.strip(PUNCT), y.strip(PUNCT)
                    if (len(x) >= 3 and x.lower() != y.lower() and abs(len(x) - len(y)) <= 2
                            and WORD.fullmatch(x) and WORD.fullmatch(y) and x.lower() not in words):
                        fixes[(x, y)] += 1
        prompt = (pw / "llm" / "prompt.txt").read_text(encoding="utf-8")
        at = prompt.find("The page's zones and OCR lines")
        if at >= 0:
            kinds = {i.get("kind") for i in answer["items"]} | ({"drop_cap"} if any(i.get("drop_cap") for i in answer["items"]) else set())
            pages.append((len(kinds), -len(prompt), prompt[at:], answer))
    if not fixes and not pages:
        return ""
    parts = []
    if fixes:
        common = sorted(fixes.items(), key=lambda kv: (-kv[1], kv[0]))[:glossary]
        parts.append("Corrections made on earlier pages of this book (OCR -> as printed), most frequent first; "
                     "the same misreadings recur:\n" + ", ".join(f"{x} -> {y}" for (x, y), _ in common))
    if pages:
        # the page that shows the most kinds of item, the shorter of equals
        _, _, prompt, answer = max(pages, key=lambda p: (p[0], p[1]))
        parts.append("An earlier page worked through (its zones and OCR lines, then the answer). Answer every "
                     "page the same way, from its own lines and images:\n<example_page>\n" + prompt +
                     "\n</example_page>\n<example_answer>\n" + json.dumps(answer, ensure_ascii=False) +
                     "\n</example_answer>")
    return "\n\n".join(parts)


class Structurer:
    """Asks the model about each page as soon as it is read (`submit`), `agents` pages at a time;
    `collect` waits for the answers. An answer is reused while the page's zones and lines (and
    the model, instructions and context) stay the same. After the subscription's usage limit is
    hit, pages are no longer sent (they fall back to the heuristic; run again later)."""

    def __init__(self, work: Path, semantic_context: str, model: str, agents: int, zoom: bool = False,
                 guide: str = "", thinking: bool = True):
        self.work, self.context, self.model, self.zoom, self.thinking = work, semantic_context, model, zoom, thinking
        self.system = SYSTEM + (f"\n\n{guide}" if guide else "")
        self.pool = ThreadPoolExecutor(max_workers=max(1, agents))
        self.jobs: dict[int, Future] = {}
        self.limited = threading.Event()

    def submit(self, n: int, r: dict, fresh: bool = False) -> None:
        """Ask about page `n` (`fresh`: even if an answer is kept)."""
        self.jobs[n] = self.pool.submit(self._one, r, fresh)

    def _one(self, r: dict, fresh: bool = False):
        pw = self.work / r["name"] / "llm"
        pw.mkdir(parents=True, exist_ok=True)
        prompt = page_prompt(r["layout"], r["lines"], self.context)
        key = f"{self.model}\n{self.system}\n{prompt}"
        same = not fresh and (pw / "prompt.txt").exists() and (pw / "prompt.txt").read_text(encoding="utf-8") == key
        answer, result = (claude_cli.answer_of(pw / "response.jsonl")[0] if same else None), {"num_turns": 0}
        items = to_items(answer, r["lines"], r["layout"]) if answer else None
        for _ in range(2 if items is None else 0):  # asked again once: an answer may stop halfway
            if self.limited.is_set():
                return None, {"error": "not sent: the usage limit was reached"}
            (pw / "prompt.txt").write_text(key, encoding="utf-8")
            content, _ = claude_cli.page_content(self.work / r["name"] / "native.png", pw)
            content.append({"type": "text", "text": prompt + self._feedback(answer, r)})
            answer, result = claude_cli.ask(content, self.system, SCHEMA, pw, self.model, zoom=self.zoom,
                                            thinking=self.thinking)
            if answer is None and (result.get("api_error_status") == 429
                                   or LIMIT_HIT.search(str(result.get("result") or result.get("error") or ""))):
                self.limited.set()
            items = to_items(answer, r["lines"], r["layout"]) if answer else None
            if items is not None:
                break
        if items is None:
            log.warning(f"{r['name']}: no usable answer ({result.get('subtype') or result.get('error') or result.get('result') or 'lost text'}); heuristic structure")
        else:
            took = f", {result['duration_ms'] / 1000:.0f}s" if result.get("duration_ms") else ""
            log.info(f"{r['name']}: {len(items)} items from {self.model} ({result.get('num_turns', 0)} turns{took})")
        return items, result

    @staticmethod
    def _feedback(answer: dict | None, r: dict) -> str:
        """For a page asked again: what the earlier answer left out (Sonnet sometimes hands in the
        first column of a page and stops)."""
        if not answer:
            return ""
        known = {l.id for l in r["lines"]}
        listed = {i for a in answer.get("items", []) for i in a.get("lines", [])} & known
        empty = {i for a in answer.get("items", []) if a.get("kind") != "noise" and not str(a.get("text") or "").strip()
                 for i in a.get("lines", [])} & known
        read = listed - empty
        zones = sorted({l.zone for l in r["lines"] if l.id not in read})
        also = f" ({len(empty)} of them in items with no text)" if empty else ""
        return (f"\n\nAn earlier answer to this page read only {len(read)} of its {len(known)} OCR lines{also} and "
                f"left out zones {', '.join(zones)}. Answer the whole page: every OCR line in exactly one item, "
                f"each item with its full text.")

    def collect(self) -> dict[int, list]:
        """{page index: items} for the pages the model answered."""
        out = {n: items for n, job in sorted(self.jobs.items()) if (items := job.result()[0]) is not None}
        self.pool.shutdown()
        if self.limited.is_set():
            log.warning(f"The usage limit was reached: {len(self.jobs) - len(out)} pages have the heuristic structure. "
                        "Run the same command again later; answered pages are kept.")
        return out
