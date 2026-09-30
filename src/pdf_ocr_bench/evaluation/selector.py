"""Rank engines by consensus: lowest average CER against all other engines wins."""

from __future__ import annotations

import sys
from statistics import mean

from rich.console import Console
from rich.table import Table

from ..models import EngineReport, RankingEntry
from .compare import Comparison, Matrix


def _avg_vs_others(matrix: Matrix, name: str) -> float | None:
    values = [v for other, v in matrix.get(name, {}).items() if other != name]
    return mean(values) if values else None


def rank(comp: Comparison) -> list[RankingEntry]:
    entries = [
        RankingEntry(
            name=name,
            avg_cer=round(cer, 4),
            avg_wer=round(_avg_vs_others(comp.wer, name) or 0.0, 4),
            avg_agreement=round(_avg_vs_others(comp.agreement, name) or 0.0, 4),
            avg_bbox_iou=round(_avg_vs_others(comp.bbox_iou, name) or 0.0, 4),
        )
        for name in comp.engines
        if (cer := _avg_vs_others(comp.cer, name)) is not None
    ]
    return sorted(entries, key=lambda e: (e.avg_cer, -e.avg_agreement))


def print_ranking(ranking: list[RankingEntry], engines: list[EngineReport], console: Console | None = None) -> None:
    # fixed width when piped (CI logs) so columns are not truncated to 80 chars
    console = console or Console(width=None if sys.stdout.isatty() else 120)
    by_name = {e.name: e for e in engines}
    table = Table(title="Engine ranking (consensus: lower avg CER vs. other engines is better)")
    for col, justify in [
        ("#", "right"), ("Engine", "left"), ("Avg CER", "right"), ("Avg WER", "right"),
        ("Agreement", "right"), ("BBox IoU", "right"), ("Words", "right"), ("Avg conf", "right"), ("Time (s)", "right"),
    ]:  # fmt: skip
        table.add_column(col, justify=justify)
    for i, entry in enumerate(ranking, 1):
        report = by_name.get(entry.name)
        table.add_row(
            str(i),
            report.display_name if report else entry.name,
            f"{entry.avg_cer:.4f}",
            f"{entry.avg_wer:.4f}",
            f"{entry.avg_agreement:.3f}",
            f"{entry.avg_bbox_iou:.3f}",
            str(report.total_words) if report else "-",
            f"{report.avg_confidence:.3f}" if report and report.avg_confidence is not None else "n/a",
            f"{report.elapsed:.1f}" if report else "-",
        )
    console.print(table)
