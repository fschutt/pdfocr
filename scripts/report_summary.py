"""Render report.json as Markdown for the GitHub Actions job summary."""

from __future__ import annotations

import json
import sys


def fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main(path: str) -> None:
    report = json.load(open(path, encoding="utf-8"))
    preprocess = " → ".join(report.get("preprocess") or []) or "none"
    print(f"Languages: `{report['lang']}`, DPI {report['dpi']}, preprocessing: {preprocess}\n")
    print("| Engine | Model | Words | Avg conf | Time (s) | Status |")
    print("|--------|-------|-------|----------|----------|--------|")
    for e in report.get("engines", []):
        status = "✅" if e.get("success") else f"❌ {(e.get('error') or '')[:100]}"
        print(
            f"| {e['display_name']} | {e.get('route') or '-'} | {e.get('total_words', '-')} "
            f"| {fmt(e.get('avg_confidence'), '.3f')} | {fmt(e.get('elapsed'), '.1f')} | {status} |"
        )
    if report.get("ranking"):
        print("\n### Ranking (by consensus agreement)")
        for i, eng in enumerate(report["ranking"], 1):
            print(f"{i}. **{eng['name']}**: avg CER {eng['avg_cer']:.4f}, bbox IoU {eng['avg_bbox_iou']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
