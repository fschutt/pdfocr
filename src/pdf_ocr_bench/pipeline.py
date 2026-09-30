"""Orchestrator: render pages once, run each engine, zip its HTML, compare engines, write the report."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .engines import OcrEngine, PageTimeout, Route
from .evaluation import compare, print_ranking, rank
from .html_output.renderer import load_template, page_count, render_pdf_pages
from .html_output.zipper import create_engine_zip
from .languages import html_lang
from .log import get_logger
from .models import EngineReport, OcrResult, PageImage, PageResult, Report
from .plan import InputError, check_page_spec, log_plan, make_plan
from .preprocess import apply_filters

log = get_logger("Pipeline")


@dataclass
class PipelineConfig:
    input_pdf: Path
    output_dir: Path = Path("results")
    engines: str = "all"
    lang: str = "eng"
    dpi: int = 300
    pages: str | None = None
    timeout_per_page: int = 300
    include_gpu_engines: bool = False
    report_path: Path | None = None
    preprocess: str | None = None
    engine_options: tuple[str, ...] = ()  # "ENGINE.KEY=VALUE"


def parse_page_range(spec: str | None, count: int) -> list[int]:
    """'1-5' / '1,3,7-10' (1-indexed, inclusive) -> sorted 0-indexed pages. Empty = all."""
    if not spec or not spec.strip() or spec.strip().lower() == "all":
        return list(range(count))
    pages: set[int] = set()
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        start, _, end = part.partition("-")
        lo, hi = int(start), int(end or start)
        if lo < 1 or hi < lo:
            raise InputError(f"invalid page range '{part}'")
        pages.update(range(lo - 1, min(hi, count)))
    if not pages:
        raise InputError(f"page range '{spec}' selects no pages (document has {count})")
    return sorted(pages)


def run(cfg: PipelineConfig) -> Report:
    if not cfg.input_pdf.is_file():
        raise InputError(f"input PDF not found: {cfg.input_pdf}")
    check_page_spec(cfg.pages)
    plan = make_plan(cfg.engines, cfg.lang, cfg.preprocess, cfg.include_gpu_engines, engine_options=cfg.engine_options)
    pages = parse_page_range(cfg.pages, page_count(cfg.input_pdf))
    log.info(f"Input {cfg.input_pdf}: {len(pages)} page(s)")
    log_plan(plan)

    images = render_pdf_pages(cfg.input_pdf, cfg.output_dir / "images", cfg.dpi, pages)
    preprocess_images(images, plan.filters)
    template = load_template()
    page_lang = html_lang(plan.languages)

    results: dict[str, OcrResult] = {}
    reports = [
        EngineReport(name=p.engine.name, display_name=p.engine.display_name, success=False, error=f"skipped: {p.route.unsupported}")
        for p in plan.skipped
    ]
    for i, p in enumerate(plan.runnable, 1):
        cls = p.engine
        log.info(f"=== Engine {i}/{len(plan.runnable)}: {cls.display_name} ===")
        result, report = run_engine(cls, p.route, p.options, cfg, images)
        reports.append(report)
        if result is None:
            continue
        zip_path = cfg.output_dir / cls.name / "pages.zip"
        log.info(f"Creating zip: {zip_path}")
        try:
            create_engine_zip(result, images, zip_path, template, page_lang, cls.reports_confidence)
        except Exception as exc:  # noqa: BLE001 - e.g. disk full; keep the other engines
            report.success = False
            report.error = f"zip failed: {type(exc).__name__}: {exc}"
            log.error(f"{cls.display_name} {report.error}")
            continue
        report.zip_path = str(zip_path)
        results[cls.name] = result

    report = Report(
        input_pdf=str(cfg.input_pdf),
        lang="+".join(lang.code for lang in plan.languages),
        dpi=cfg.dpi,
        preprocess=plan.filters,
        pages=[p + 1 for p in pages],
        engines=reports,
    )
    evaluate(report, results)
    report_path = cfg.report_path or cfg.output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info(f"Report written to {report_path}")
    return report


def preprocess_images(images: list[PageImage], filters: list[str]) -> None:
    if not filters:
        return
    start = time.perf_counter()
    for image in images:
        apply_filters(image.path, filters)
    log.info(f"Preprocessed {len(images)} pages ({' -> '.join(filters)}) in {time.perf_counter() - start:.1f}s")


def run_engine(
    cls: type[OcrEngine], route: Route, options: dict, cfg: PipelineConfig, images: list[PageImage]
) -> tuple[OcrResult | None, EngineReport]:
    """Never raises: any failure is logged and recorded in the returned report."""
    start = time.perf_counter()
    engine = cls(route, timeout=cfg.timeout_per_page, options=options)
    report = EngineReport(
        name=cls.name, display_name=cls.display_name, success=False, route=route.detail, options=engine.opts
    )
    try:
        engine.prepare()
    except Exception as exc:  # noqa: BLE001 - one broken engine must not stop the run
        report.error = f"setup failed: {type(exc).__name__}: {exc}"
        report.elapsed = time.perf_counter() - start
        log.error(f"{cls.display_name} {report.error}")
        _close(engine)
        return None, report

    try:
        pages = [_run_page(engine, image, n, len(images)) for n, image in enumerate(images, 1)]
    finally:
        _close(engine)

    elapsed = time.perf_counter() - start
    result = OcrResult(engine_name=cls.name, pages=pages, total_elapsed=elapsed)
    ok_pages = [p for p in pages if not p.skipped and p.error is None]
    report.success = bool(ok_pages)
    report.total_words = result.total_words
    report.avg_confidence = round(result.avg_confidence, 4) if cls.reports_confidence else None
    report.elapsed = round(elapsed, 3)
    report.pages_processed = len(ok_pages)
    report.pages_skipped = [p.page_num + 1 for p in pages if p.skipped or p.error is not None]
    if not report.success:
        report.error = next((p.error for p in pages if p.error), "no page succeeded")
        log.error(f"{cls.display_name} failed on every page: {report.error}")
        return None, report
    log.info(f"{cls.display_name} complete: {result.total_words} words total, {elapsed:.1f}s")
    return result, report


def _run_page(engine: OcrEngine, image: PageImage, n: int, total: int) -> PageResult:
    start = time.perf_counter()
    try:
        page = engine.run(image)
    except PageTimeout as exc:
        engine.log.warning(f"Page {n}/{total}: skipped, {exc}")
        return _failed_page(engine, image, start, f"timeout: {exc}", skipped=True)
    except Exception as exc:  # noqa: BLE001 - keep going with the next page
        engine.log.error(f"Page {n}/{total}: {type(exc).__name__}: {exc}")
        return _failed_page(engine, image, start, f"{type(exc).__name__}: {exc}")
    conf = f"avg conf {page.avg_confidence:.2f}" if engine.reports_confidence else "no confidence"
    engine.log.info(f"Page {n}/{total}: {len(page.words)} words, {page.elapsed_seconds:.1f}s, {conf}")
    return page


def _failed_page(engine: OcrEngine, image: PageImage, start: float, error: str, skipped: bool = False) -> PageResult:
    return PageResult(
        page_num=image.page_num,
        width_px=image.width_px,
        height_px=image.height_px,
        words=[],
        full_text="",
        engine_name=engine.name,
        elapsed_seconds=time.perf_counter() - start,
        skipped=skipped,
        error=error,
    )


def _close(engine: OcrEngine) -> None:
    try:
        engine.close()
    except Exception as exc:  # noqa: BLE001
        engine.log.warning(f"cleanup failed: {exc}")


def evaluate(report: Report, results: dict[str, OcrResult]) -> None:
    if len(results) < 2:
        log.warning("Fewer than two successful engines: skipping cross-engine comparison")
        report.ranking = []
        report.best_engine = next(iter(results), None)
        return
    comp = compare(results)
    report.cer_matrix, report.wer_matrix = comp.cer, comp.wer
    report.agreement_matrix, report.bbox_iou_matrix = comp.agreement, comp.bbox_iou
    report.ranking = rank(comp)
    report.best_engine = report.ranking[0].name if report.ranking else None
    print_ranking(report.ranking, report.engines)
    if report.best_engine:
        log.info(f"Most consistent engine: {report.best_engine}")
