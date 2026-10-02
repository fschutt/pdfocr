from __future__ import annotations

import sys
from pathlib import Path

import click

from .engines import ENGINES
from .languages import LANGUAGES, parse_languages, tesseract_packages
from .log import get_logger, setup_logging
from .pipeline import PipelineConfig, run
from .plan import InputError, check_page_spec, log_plan, make_plan
from .preprocess import FILTERS

LANG_HELP = (
    "Document language(s) as Tesseract codes joined by '+', main language first, e.g. deu or "
    "deu+eng. Every engine is routed to the model that covers all of them; see "
    "`pdf-ocr-bench languages`."
)
ENGINES_HELP = f"'all' or a comma-separated list of: {', '.join(ENGINES)}"
PREPROCESS_HELP = f"Comma-separated image filters, applied in order before OCR: {', '.join(FILTERS)}"
PAGES_HELP = 'Page range, e.g. "1-5" or "1,3,7-10" [default: all]'
OPTION_HELP = "Engine parameter ENGINE.KEY=VALUE, repeatable, e.g. -O macos_vision.level=fast -O tesseract.psm=6"


def engines_reference() -> str:
    """Every engine with its model and its -O parameters (default, allowed values, meaning)."""
    lines = []
    for cls in ENGINES.values():
        gpu = "  [GPU, opt-in]" if cls.requires_gpu else ""
        lines += ["\b", f"{cls.name}{gpu}", f"  model: {cls.model}"]
        if not cls.options:
            lines.append("  options: none")
        for key, opt in cls.options.items():
            default = str(opt.default).lower() if isinstance(opt.default, bool) else (opt.default if opt.default != "" else '""')
            lines.append(f"  -O {cls.name}.{key}={default}  ({opt.domain()}) {opt.help}")
        lines.append("")
    return "\n".join(lines)


RUN_EPILOG = (
    "Engines and their parameters (the language model each one uses for --lang: `pdf-ocr-bench languages`):\n\n"
    + engines_reference()
)


@click.group()
@click.version_option(package_name="pdf-ocr-bench")
def main() -> None:
    """Run several OCR engines on a scanned PDF and emit positioned-HTML zips per engine.

    \b
    pdf-ocr-bench engines      every engine's model and -O parameters
    pdf-ocr-bench languages    the --lang codes and the model each engine uses for them
    pdf-ocr-bench check ...    validate the inputs and show the routing, without running
    pdf-ocr-bench run --help   all run options
    """


@main.command("run", epilog=RUN_EPILOG)
@click.argument("input_pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("results"), show_default=True, help="Output directory")
@click.option("-e", "--engines", default="all", show_default=True, help=ENGINES_HELP)
@click.option("--lang", default="eng", show_default=True, help=LANG_HELP)
@click.option("--dpi", type=click.IntRange(72, 1200), default=300, show_default=True, help="Render DPI of the page images the engines read")
@click.option("--pages", default=None, help=PAGES_HELP)
@click.option("--preprocess", default="", help=PREPROCESS_HELP)
@click.option("-O", "--engine-option", "engine_options", multiple=True, metavar="ENGINE.KEY=VALUE", help=OPTION_HELP)
@click.option("--timeout-per-page", type=click.IntRange(0), default=300, show_default=True, help="Seconds before skipping a page for an engine (0 = no limit)")
@click.option("--include-gpu-engines", is_flag=True, help="With --engines all, also run GPU engines (olmOCR)")
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Evaluation report path [default: OUTPUT_DIR/report.json]")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def run_cmd(input_pdf, output_dir, engines, lang, dpi, pages, preprocess, engine_options, timeout_per_page, include_gpu_engines, report_path, verbose) -> None:
    """OCR INPUT_PDF with every selected engine."""
    setup_logging(verbose)
    cfg = PipelineConfig(
        input_pdf=input_pdf,
        output_dir=output_dir,
        engines=engines,
        lang=lang,
        dpi=dpi,
        pages=pages,
        preprocess=preprocess,
        engine_options=engine_options,
        timeout_per_page=timeout_per_page,
        include_gpu_engines=include_gpu_engines,
        report_path=report_path,
    )
    try:
        report = run(cfg)
    except InputError as exc:
        raise click.UsageError(str(exc)) from exc
    if not any(e.success for e in report.engines):
        get_logger("Pipeline").error("No engine succeeded")
        sys.exit(1)


@main.command("reconstruct")
@click.argument("input_pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("reconstructed"), show_default=True)
@click.option("--pages", default=None, help=PAGES_HELP)
@click.option("--lang", default="eng", show_default=True, help=LANG_HELP)
@click.option("--semantic-context", default="", help="What the book is, for the LLM, e.g. 'an English dictionary of the Bible, printed 1732'")
@click.option("--model", default="sonnet", show_default=True, help="claude -p model for the structure")
@click.option("--agents", type=click.IntRange(1), default=4, show_default=True, help="pages asked at once")
@click.option("--no-llm", is_flag=True, help="structure from the line geometry only (no upload)")
@click.option("--zoom", is_flag=True, help="let the LLM crop and zoom into the scan (several MB of upload per page)")
@click.option("--fit-rounds", type=click.IntRange(0), default=6, show_default=True, help="render-and-shrink rounds (needs html2pdf)")
@click.option("-v", "--verbose", is_flag=True)
def reconstruct_cmd(input_pdf, output_dir, pages, lang, semantic_context, model, agents, no_llm, zoom, fit_rounds, verbose) -> None:
    """Rebuild scanned pages as structured HTML: OpenCV layout, macOS Vision text, an LLM's structure.

    Writes OUTPUT_DIR/pages.zip for html2pdf (columns, notes beside their lines, drop capitals,
    the pictures cut out of the scan) and OUTPUT_DIR/work/ (each page's layout, lines, LLM answer).
    """
    from .html_output.renderer import page_count
    from .languages import html_lang
    from .pipeline import parse_page_range
    from .reconstruct import reconstruct

    setup_logging(verbose)
    try:
        check_page_spec(pages)
        languages = parse_languages(lang)
        selected = parse_page_range(pages, page_count(input_pdf))
    except (ValueError, InputError) as exc:
        raise click.UsageError(str(exc)) from exc
    route = ENGINES["macos_vision"].preflight(ENGINES["macos_vision"].route(languages))
    if not route.ok:
        raise click.UsageError(f"reconstruct reads the text with macOS Vision: {route.unsupported}")
    reconstruct(input_pdf, output_dir, selected, route.lang, html_lang(languages), semantic_context,
                llm=not no_llm, model=model, agents=agents, fit_rounds=fit_rounds, zoom=zoom)


@main.command("check")
@click.option("-e", "--engines", default="all", show_default=True, help=ENGINES_HELP)
@click.option("--lang", default="eng", show_default=True, help=LANG_HELP)
@click.option("--pages", default=None, help=PAGES_HELP)
@click.option("--preprocess", default="", help=PREPROCESS_HELP)
@click.option("-O", "--engine-option", "engine_options", multiple=True, metavar="ENGINE.KEY=VALUE", help=OPTION_HELP)
@click.option("--include-gpu-engines", is_flag=True)
@click.option("--installed", is_flag=True, help="Also check this machine: engine packages, Tesseract models, llama-server, platform")
def check(engines, lang, pages, preprocess, engine_options, include_gpu_engines, installed) -> None:
    """Validate the inputs and show how each engine is routed, without running anything."""
    setup_logging(False)
    try:
        check_page_spec(pages)
        plan = make_plan(engines, lang, preprocess, include_gpu_engines, preflight=installed, engine_options=engine_options)
    except InputError as exc:
        raise click.UsageError(str(exc)) from exc
    log_plan(plan)


@main.command("warmup")
@click.option("-e", "--engines", default="all", show_default=True, help=ENGINES_HELP)
@click.option("--lang", default="eng", show_default=True, help=LANG_HELP)
@click.option("-O", "--engine-option", "engine_options", multiple=True, metavar="ENGINE.KEY=VALUE", help=OPTION_HELP)
@click.option("--include-gpu-engines", is_flag=True)
def warmup(engines, lang, engine_options, include_gpu_engines) -> None:
    """Load every routed engine once (downloads its models). Failures are reported, never fatal."""
    setup_logging(False)
    log = get_logger("Warmup")
    try:
        plan = make_plan(engines, lang, include_gpu_engines=include_gpu_engines, engine_options=engine_options)
    except InputError as exc:
        raise click.UsageError(str(exc)) from exc
    log_plan(plan)
    for p in plan.runnable:
        engine = p.engine(p.route, options=p.options)
        try:
            engine.prepare()
            log.info(f"{p.engine.display_name} ok ({p.route.detail})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"{p.engine.display_name} unavailable: {type(exc).__name__}: {exc}")
        finally:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass


@main.command("languages")
def list_languages() -> None:
    """List the --lang codes and the model each engine uses for them."""
    from rich.console import Console
    from rich.table import Table

    names = ("tesseract", "rapidocr", "paddleocr", "easyocr", "doctr", "macos_vision")
    engines = [ENGINES[n] for n in names]
    table = Table(title="--lang codes and the model each engine uses (- = not supported)")
    table.add_column("code")
    table.add_column("language")
    for cls in engines:
        table.add_column(cls.name)
    for lang in LANGUAGES.values():
        routes = [cls.route([lang]) for cls in engines]
        table.add_row(lang.code, lang.name, *[(r.detail if r.ok else "-") for r in routes])
    console = Console(width=None if sys.stdout.isatty() else 160)
    console.print(table)
    console.print("Surya and olmOCR take no language setting; ocrmypdf uses the Tesseract models.")
    console.print("Combine codes with '+', main language first: every engine must cover all of them.")


@main.command("filters")
def list_filters() -> None:
    """List the --preprocess filters."""
    for name, (_, description) in FILTERS.items():
        click.echo(f"{name:13} {description}")


@main.command("tesseract-packages")
@click.option("--lang", required=True, help=LANG_HELP)
def list_tesseract_packages(lang) -> None:
    """Print the apt packages providing the Tesseract models for --lang, one per line."""
    try:
        languages = parse_languages(lang)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo("\n".join(tesseract_packages(languages)))


@main.command("engines")
def list_engines() -> None:
    """List the engines with their model and parameters (-O ENGINE.KEY=VALUE)."""
    click.echo(engines_reference().replace("\b\n", ""))


if __name__ == "__main__":
    main()
