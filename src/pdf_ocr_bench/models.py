from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float  # normalized 0..1, left edge
    y: float  # normalized 0..1, top edge
    w: float  # normalized width
    h: float  # normalized height

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "BBox":
        x0, x1 = sorted((_clamp(x0), _clamp(x1)))
        y0, y1 = sorted((_clamp(y0), _clamp(y1)))
        return cls(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    @classmethod
    def from_pixels(cls, x0: float, y0: float, x1: float, y1: float, width: float, height: float) -> "BBox":
        return cls.from_corners(x0 / width, y0 / height, x1 / width, y1 / height)

    @classmethod
    def from_points(cls, points, width: float, height: float) -> "BBox":
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return cls.from_pixels(min(xs), min(ys), max(xs), max(ys), width, height)

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


class OcrWord(BaseModel):
    text: str
    bbox: BBox
    confidence: float  # 0..1


class PageResult(BaseModel):
    page_num: int  # 0-indexed
    width_px: int  # rendered image dimensions
    height_px: int
    words: list[OcrWord]
    full_text: str
    engine_name: str
    elapsed_seconds: float
    skipped: bool = False
    error: str | None = None

    @property
    def avg_confidence(self) -> float:
        return sum(w.confidence for w in self.words) / len(self.words) if self.words else 0.0


class OcrResult(BaseModel):
    engine_name: str
    pages: list[PageResult]
    total_elapsed: float

    @property
    def total_words(self) -> int:
        return sum(len(p.words) for p in self.pages)

    @property
    def avg_confidence(self) -> float:
        words = [w for p in self.pages for w in p.words]
        return sum(w.confidence for w in words) / len(words) if words else 0.0


class PageImage(BaseModel):
    """A source page rendered once and shared by every engine."""

    page_num: int  # 0-indexed page in the source PDF
    path: Path
    width_px: int
    height_px: int
    width_pt: float  # source PDF page size
    height_pt: float
    dpi: int

    @property
    def file_name(self) -> str:
        return self.path.name


class EngineReport(BaseModel):
    name: str
    display_name: str
    success: bool
    route: str | None = None  # the model chosen for the requested languages
    options: dict[str, Any] = Field(default_factory=dict)  # effective engine options
    error: str | None = None
    total_words: int = 0
    avg_confidence: float | None = None  # None: the engine reports no confidence
    elapsed: float = 0.0
    pages_processed: int = 0
    pages_skipped: list[int] = Field(default_factory=list)
    zip_path: str | None = None


class RankingEntry(BaseModel):
    name: str
    avg_cer: float
    avg_wer: float
    avg_agreement: float
    avg_bbox_iou: float


class Report(BaseModel):
    input_pdf: str
    lang: str
    dpi: int
    preprocess: list[str] = Field(default_factory=list)
    pages: list[int]  # 1-indexed pages processed
    engines: list[EngineReport]
    cer_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    wer_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    agreement_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    bbox_iou_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    ranking: list[RankingEntry] = Field(default_factory=list)
    best_engine: str | None = None
