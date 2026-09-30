from __future__ import annotations

import sys
from pathlib import Path

import click

from .engines import ENGINES, select_engines
from .log import get_logger, setup_logging
from .pipeline import PipelineConfig, run


@click.group()
@click.version_option(package_name="pdf-ocr-bench")
def main() -> None:
    """Run several OCR engines on a scanned PDF and emit positioned-HTML zips per engine."""


@main.command("run")
@click.argument("input_pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("results"), show_default=True, help="Output directory")
@click.option("-e", "--engines", default="all", show_default=True, help=f"Comma-separated engines: {', '.join(ENGINES)}")
@click.option(
    "--lang",
    default="eng",
    show_default=True,
    help="Document language as Tesseract code, e.g. eng, deu, fra, deu_frak, frk, chi_sim, jpn, ara, rus, lat. "
    "Compound: eng+deu (Tesseract tries both).",
)
@click.option("--dpi", type=click.IntRange(36, 1200), default=300, show_default=True, help="Render DPI")
@click.option("--pages", default=None, help='Page range, e.g. "1-5" or "1,3,7-10" [default: all]')
@click.option("--timeout-per-page", type=click.IntRange(0), default=300, show_default=True, help="Seconds before skipping a page for an engine (0 = no limit)")
@click.option("--include-gpu-engines", is_flag=True, help="Also run GPU-requiring engines (olmOCR)")
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Evaluation report path [default: OUTPUT_DIR/report.json]")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def run_cmd(input_pdf, output_dir, engines, lang, dpi, pages, timeout_per_page, include_gpu_engines, report_path, verbose) -> None:
    """OCR INPUT_PDF with every selected engine."""
    setup_logging(verbose)
    cfg = PipelineConfig(
        input_pdf=input_pdf,
        output_dir=output_dir,
        engines=engines,
        lang=lang,
        dpi=dpi,
        pages=pages,
        timeout_per_page=timeout_per_page,
        include_gpu_engines=include_gpu_engines,
        report_path=report_path,
    )
    try:
        report = run(cfg)
    except (ValueError, FileNotFoundError) as exc:
        raise click.UsageError(str(exc)) from exc
    if not any(e.success for e in report.engines):
        get_logger("Pipeline").error("No engine succeeded")
        sys.exit(1)


@main.command("warmup")
@click.option("-e", "--engines", default="all", show_default=True, help="Comma-separated engines")
@click.option("--lang", default="eng", show_default=True, help="Tesseract language code (mapped per engine)")
@click.option("--include-gpu-engines", is_flag=True)
def warmup(engines, lang, include_gpu_engines) -> None:
    """Load every engine once (downloads models). Failures are reported, never fatal."""
    setup_logging(False)
    log = get_logger("Warmup")
    for cls in select_engines(engines, include_gpu_engines):
        engine = cls(lang=lang)
        try:
            engine.prepare()
            log.info(f"{cls.display_name} ok (lang={engine.lang})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"{cls.display_name} unavailable: {type(exc).__name__}: {exc}")
        finally:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass


@main.command("engines")
def list_engines() -> None:
    """List available engines."""
    for name, cls in ENGINES.items():
        click.echo(f"{name:16} {cls.display_name}{'  (GPU, opt-in)' if cls.requires_gpu else ''}")


if __name__ == "__main__":
    main()
