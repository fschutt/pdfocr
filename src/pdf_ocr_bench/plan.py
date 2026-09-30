"""Validate the inputs and decide what runs, before any page is rendered.

Bad input is rejected here with a message that says what to change: unknown language codes,
language combinations no engine model covers, unknown filters, malformed page ranges, and
engines asked for by name that cannot read the requested languages. With `--engines all`,
engines that cannot are skipped instead, and the report says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .engines import ENGINES, OcrEngine, Route, is_all, select_engines
from .languages import Language, parse_languages
from .log import get_logger
from .preprocess import parse_filters

log = get_logger("Pipeline")

PAGE_SPEC = re.compile(r"^\s*\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*\s*$")


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class EnginePlan:
    engine: type[OcrEngine]
    route: Route
    options: dict[str, Any] = field(default_factory=dict)  # only the ones set with -O


@dataclass(frozen=True)
class Plan:
    languages: list[Language]
    filters: list[str]
    runnable: list[EnginePlan]
    skipped: list[EnginePlan]  # only with --engines all: cannot read these languages here


def check_page_spec(spec: str | None) -> None:
    """Syntax only; the page count is checked once the PDF is open."""
    if not spec or spec.strip().lower() == "all":
        return
    if not PAGE_SPEC.match(spec):
        raise InputError(f'invalid page range {spec!r}: use e.g. "1-5" or "1,3,7-10"')


def parse_engine_options(raw: list[str] | tuple[str, ...], selected: list[type[OcrEngine]]) -> dict[str, dict[str, Any]]:
    """`["macos_vision.level=fast", ...]` -> {"macos_vision": {"level": "fast"}}, validated."""
    selected_names = {cls.name for cls in selected}
    parsed: dict[str, dict[str, Any]] = {}
    for item in raw:
        target, sep, value = item.partition("=")
        engine, dot, key = target.strip().partition(".")
        engine = engine.lower().replace("-", "_")
        if not sep or not dot or not engine or not key:
            raise InputError(f"invalid engine option {item!r}: expected ENGINE.KEY=VALUE, e.g. tesseract.psm=6")
        if engine not in ENGINES:
            raise InputError(f"engine option {item!r}: unknown engine {engine!r} (available: {', '.join(ENGINES)})")
        if engine not in selected_names:
            raise InputError(f"engine option {item!r}: {engine} is not among the selected engines")
        options = ENGINES[engine].options
        if key not in options:
            valid = ", ".join(options) or "none"
            raise InputError(f"engine option {item!r}: {engine} has no option {key!r} (its options: {valid})")
        try:
            parsed.setdefault(engine, {})[key] = options[key].parse(value)
        except ValueError as exc:
            raise InputError(f"engine option {item!r}: {exc}") from None
    return parsed


def make_plan(
    engines: str,
    lang: str,
    preprocess: str | None = None,
    include_gpu_engines: bool = False,
    preflight: bool = True,
    engine_options: list[str] | tuple[str, ...] = (),
) -> Plan:
    """Route every selected engine to the requested languages.

    `preflight=False` checks only what does not depend on this machine (the workflow does that
    before installing the engines); a run also checks installed models and the platform.
    """
    try:
        languages = parse_languages(lang)
        filters = parse_filters(preprocess)
        classes = select_engines(engines, include_gpu_engines)
    except ValueError as exc:
        raise InputError(str(exc)) from exc

    options = parse_engine_options(engine_options, classes)
    plans = []
    for cls in classes:
        route = cls.route(languages)
        plans.append(EnginePlan(cls, cls.preflight(route) if preflight else route, options.get(cls.name, {})))
    runnable = [p for p in plans if p.route.ok]
    skipped = [p for p in plans if not p.route.ok]
    codes = "+".join(lang.code for lang in languages)
    if skipped and not is_all(engines):
        reasons = "; ".join(f"{p.engine.name}: {p.route.unsupported}" for p in skipped)
        raise InputError(f"cannot run the requested engines for --lang {codes}: {reasons}")
    if not runnable:
        reasons = "; ".join(f"{p.engine.name}: {p.route.unsupported}" for p in skipped)
        raise InputError(f"no engine can read --lang {codes}: {reasons}")
    return Plan(languages=languages, filters=filters, runnable=runnable, skipped=skipped)


def log_plan(plan: Plan) -> None:
    names = ", ".join(f"{lang.code} ({lang.name})" for lang in plan.languages)
    log.info(f"Languages: {names}")
    log.info(f"Preprocessing: {' -> '.join(plan.filters) if plan.filters else 'none'}")
    log.info("Engine routing:")
    for p in plan.runnable:
        opts = " ".join(f"{k}={v}" for k, v in p.options.items())
        log.info(f"  {p.engine.name:15} {p.route.detail}" + (f" [{opts}]" if opts else ""))
    for p in plan.skipped:
        log.info(f"  {p.engine.name:15} skipped: {p.route.unsupported}")
