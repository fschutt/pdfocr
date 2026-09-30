from __future__ import annotations

import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, TypeVar

from ..languages import Language
from ..log import get_logger
from ..models import BBox, OcrWord, PageImage, PageResult

T = TypeVar("T")


class PageTimeout(Exception):
    pass


@dataclass(frozen=True)
class Route:
    """How an engine handles the requested languages, decided before any engine runs."""

    lang: Any = None  # engine-specific language argument (code, tuple of codes, model family)
    detail: str = ""  # the model choice, for logs and the report
    unsupported: str | None = None  # why the engine cannot handle these languages

    @property
    def ok(self) -> bool:
        return self.unsupported is None


def unsupported(reason: str) -> Route:
    return Route(unsupported=reason)


@dataclass(frozen=True)
class Option:
    """A tunable engine parameter, set with `-O ENGINE.KEY=VALUE` and validated before a run."""

    default: Any
    help: str
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None

    @property
    def kind(self) -> type:
        return type(self.default)

    def parse(self, raw: str) -> Any:
        text = raw.strip()
        if self.kind is bool:
            if text.lower() in ("1", "true", "yes", "on"):
                return True
            if text.lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError("expected true or false")
        try:
            value = self.kind(text)
        except ValueError:
            raise ValueError(f"expected {self.kind.__name__}") from None
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"expected one of {', '.join(map(str, self.choices))}")
        if self.minimum is not None and value < self.minimum or self.maximum is not None and value > self.maximum:
            raise ValueError(f"expected {self.minimum}..{self.maximum}")
        return value

    def domain(self) -> str:
        if self.choices is not None:
            return "|".join(map(str, self.choices))
        if self.kind is bool:
            return "true|false"
        if self.minimum is not None or self.maximum is not None:
            return f"{self.minimum:g}..{self.maximum:g}"
        return self.kind.__name__


class OcrEngine(ABC):
    """One OCR backend. Subclasses implement `route` and `ocr_page`; everything else is shared."""

    name: ClassVar[str]
    display_name: ClassVar[str]
    requires_gpu: ClassVar[bool] = False
    # Subprocess-based engines enforce the timeout themselves (and can kill the child).
    handles_timeout: ClassVar[bool] = False
    # False for engines whose output carries no confidence (PDF text layers, VLM text).
    reports_confidence: ClassVar[bool] = True
    # What the engine runs, for `--help` and `pdf-ocr-bench engines`.
    model: ClassVar[str] = ""
    options: ClassVar[dict[str, Option]] = {}

    def __init__(self, route: Route, timeout: float | None = None, options: dict[str, Any] | None = None):
        if not route.ok:
            raise ValueError(f"{self.display_name}: {route.unsupported}")
        unknown = set(options or {}) - set(self.options)
        if unknown:
            raise ValueError(f"{self.display_name}: unknown option(s) {', '.join(sorted(unknown))}")
        self.route_info = route
        self.lang = route.lang
        self.timeout = timeout or None
        self.opts = {key: option.default for key, option in self.options.items()} | (options or {})
        self.log = get_logger(self.display_name)

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        """Pick the model for `languages`, or say why this engine cannot read them.

        Static: depends only on the languages, never on this machine, so inputs can be
        validated before anything is installed (the workflow does that first).
        """
        return Route(detail="language-independent")

    @classmethod
    def preflight(cls, route: Route) -> Route:
        """Check `route` against this machine (installed models, platform) before a run."""
        return route

    def prepare(self) -> None:
        """Import the backend and load models. Called once before the first page."""

    def close(self) -> None:
        """Release resources (servers, temp dirs)."""

    @abstractmethod
    def ocr_page(self, image: PageImage, lang: Any) -> list[OcrWord]:
        ...

    def run(self, image: PageImage) -> PageResult:
        start = time.perf_counter()
        call = lambda: self.ocr_page(image, self.lang)  # noqa: E731
        words = call() if self.handles_timeout else run_with_timeout(call, self.timeout)
        words = [
            part
            for w in words
            if w.bbox.w > 0 and w.bbox.h > 0
            # a word never contains whitespace; split what an engine returned joined
            for part in split_line(w.text.strip(), w.bbox, w.confidence)
        ]
        return PageResult(
            page_num=image.page_num,
            width_px=image.width_px,
            height_px=image.height_px,
            words=words,
            full_text=words_to_text(words),
            engine_name=self.name,
            elapsed_seconds=time.perf_counter() - start,
        )


def run_with_timeout(fn: Callable[[], T], timeout: float | None) -> T:
    """Run `fn` in a daemon thread. On timeout the thread is abandoned (Python cannot kill it)."""
    if not timeout:
        return fn()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise PageTimeout(f"exceeded {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def group_lines(words: Iterable[OcrWord]) -> list[list[OcrWord]]:
    """Group words into lines by vertical overlap; lines top to bottom, words left to right."""
    lines: list[list[OcrWord]] = []
    for word in sorted(words, key=lambda w: (w.bbox.y + w.bbox.h / 2, w.bbox.x)):
        line = next((l for l in reversed(lines[-3:]) if _same_line(l[-1].bbox, word.bbox)), None)
        if line is None:
            lines.append([word])
            continue
        line.append(word)
    lines.sort(key=lambda l: min(w.bbox.y for w in l))
    return [sorted(l, key=lambda w: w.bbox.x) for l in lines]


def words_to_text(words: Iterable[OcrWord]) -> str:
    return "\n".join(" ".join(w.text for w in line) for line in group_lines(words))


def _same_line(a: BBox, b: BBox) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y, b.y)
    return overlap > 0.5 * min(a.h, b.h)


def split_line(text: str, bbox: BBox, confidence: float) -> list[OcrWord]:
    """Split a line-level detection into word boxes, proportional to character count."""
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return []
    if len(tokens) == 1:
        return [OcrWord(text=tokens[0][2], bbox=bbox, confidence=confidence)]
    span = max(len(text.rstrip()), 1)
    char_w = bbox.w / span
    return [
        OcrWord(
            text=tok,
            bbox=BBox(x=bbox.x + start * char_w, y=bbox.y, w=(end - start) * char_w, h=bbox.h),
            confidence=confidence,
        )
        for start, end, tok in tokens
    ]


def split_block(lines: list[str], bbox: BBox, confidence: float) -> list[OcrWord]:
    """Distribute text lines evenly over a block box, then split each into words."""
    lines = [l for l in lines if l.strip()]
    if not lines:
        return []
    line_h = bbox.h / len(lines)
    longest = max(len(l.strip()) for l in lines)
    return [
        word
        for i, line in enumerate(lines)
        for word in split_line(
            line.strip(),
            BBox(x=bbox.x, y=bbox.y + i * line_h, w=bbox.w * len(line.strip()) / longest, h=line_h),
            confidence,
        )
    ]
