"""Prefixed, always-flushed logging (GitHub Actions shows lines live only if flushed)."""

from __future__ import annotations

import logging
import sys

ROOT = "pdf_ocr_bench"


class _FlushingHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _PrefixFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None) or record.name.rsplit(".", 1)[-1]
        msg = super().format(record)
        level = "" if record.levelno == logging.INFO else f"{record.levelname}: "
        return f"[{tag}] {level}{msg}"


def setup_logging(verbose: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    handler = _FlushingHandler(sys.stdout)
    handler.setFormatter(_PrefixFormatter("%(message)s"))
    root = logging.getLogger(ROOT)
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False


def get_logger(tag: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger(f"{ROOT}.{tag}"), {"tag": tag})
