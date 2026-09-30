"""Pairwise text and layout metrics. Without ground truth, every metric compares two engines."""

from __future__ import annotations

import jiwer
import numpy as np
from rapidfuzz import fuzz

from ..models import OcrWord


def normalize(text: str) -> str:
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, capped at 1.0 so one garbage page cannot dominate an average."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(1.0, float(jiwer.cer(ref, hyp)))


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(1.0, float(jiwer.wer(ref, hyp)))


def symmetric(metric, a: str, b: str) -> float:
    """CER/WER depend on which side is the reference; with no ground truth, average both."""
    return (metric(a, b) + metric(b, a)) / 2


def agreement(a: str, b: str) -> float:
    """Normalized Levenshtein similarity, 0..1."""
    return fuzz.ratio(normalize(a), normalize(b)) / 100.0


def _boxes(words: list[OcrWord]) -> np.ndarray:
    return np.array([[w.bbox.x, w.bbox.y, w.bbox.x1, w.bbox.y1] for w in words], dtype=np.float64).reshape(-1, 4)


def iou_matrix(a: list[OcrWord], b: list[OcrWord]) -> np.ndarray:
    ba, bb = _boxes(a)[:, None, :], _boxes(b)[None, :, :]
    iw = np.clip(np.minimum(ba[..., 2], bb[..., 2]) - np.maximum(ba[..., 0], bb[..., 0]), 0, None)
    ih = np.clip(np.minimum(ba[..., 3], bb[..., 3]) - np.maximum(ba[..., 1], bb[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    union = area(ba) + area(bb) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def bbox_iou(a: list[OcrWord], b: list[OcrWord]) -> float:
    """Mean best-match IoU, averaged over both directions (word segmentation differs per engine)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m = iou_matrix(a, b)
    return float((m.max(axis=1).mean() + m.max(axis=0).mean()) / 2)
