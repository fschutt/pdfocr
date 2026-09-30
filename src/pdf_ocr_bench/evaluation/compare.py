"""Cross-engine comparison: every pair of engines, every page both processed."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from statistics import mean

from ..log import get_logger
from ..models import OcrResult, PageResult
from . import metrics

log = get_logger("Evaluation")

Matrix = dict[str, dict[str, float]]


@dataclass
class Comparison:
    engines: list[str]
    cer: Matrix = field(default_factory=dict)
    wer: Matrix = field(default_factory=dict)
    agreement: Matrix = field(default_factory=dict)
    bbox_iou: Matrix = field(default_factory=dict)


def _usable_pages(result: OcrResult) -> dict[int, PageResult]:
    return {p.page_num: p for p in result.pages if not p.skipped and p.error is None}


def _pair_scores(a: OcrResult, b: OcrResult) -> dict[str, float] | None:
    pa, pb = _usable_pages(a), _usable_pages(b)
    common = sorted(pa.keys() & pb.keys())
    if not common:
        return None
    pairs = [(pa[n], pb[n]) for n in common]
    return {
        "cer": mean(metrics.symmetric(metrics.cer, x.full_text, y.full_text) for x, y in pairs),
        "wer": mean(metrics.symmetric(metrics.wer, x.full_text, y.full_text) for x, y in pairs),
        "agreement": mean(metrics.agreement(x.full_text, y.full_text) for x, y in pairs),
        "bbox_iou": mean(metrics.bbox_iou(x.words, y.words) for x, y in pairs),
    }


def compare(results: dict[str, OcrResult]) -> Comparison:
    names = list(results)
    log.info("Computing cross-engine CER matrix...")
    comp = Comparison(engines=names)
    matrices = {"cer": comp.cer, "wer": comp.wer, "agreement": comp.agreement, "bbox_iou": comp.bbox_iou}
    identity = {"cer": 0.0, "wer": 0.0, "agreement": 1.0, "bbox_iou": 1.0}
    for name in names:
        for key, matrix in matrices.items():
            matrix.setdefault(name, {})[name] = identity[key]
    for a, b in combinations(names, 2):
        scores = _pair_scores(results[a], results[b])
        if scores is None:
            log.warning(f"{a} vs {b}: no common pages, skipped")
            continue
        for key, value in scores.items():
            matrices[key][a][b] = matrices[key][b][a] = round(value, 4)
        log.debug(f"{a} vs {b}: CER {scores['cer']:.4f}, IoU {scores['bbox_iou']:.3f}")
    return comp
