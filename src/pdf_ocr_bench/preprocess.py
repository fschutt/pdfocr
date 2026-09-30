"""Image filters applied once to the rendered pages, before any engine sees them.

`--preprocess grayscale,deskew,denoise` applies the filters left to right. Every engine reads the
filtered images, so all of them compare on the same input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .log import get_logger

log = get_logger("Preprocess")

MAX_SKEW_DEG = 5.0


def grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def autocontrast(img: Image.Image) -> Image.Image:
    """Stretch the histogram so the darkest ink is black and the paper white (ignoring 1% outliers)."""
    return ImageOps.autocontrast(img, cutoff=1)


def denoise(img: Image.Image) -> Image.Image:
    """3x3 median filter: removes scanner speckle smaller than a stroke."""
    return img.filter(ImageFilter.MedianFilter(3))


def sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3))


def otsu_threshold(gray: np.ndarray) -> int:
    """The gray level that best separates ink from paper (max. between-class variance)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = weight_bg[-1] - weight_bg
    mean_bg = np.cumsum(hist * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_bg = mean_bg / weight_bg
        mu_fg = (mean_bg[-1] - mean_bg) / weight_fg
        between = weight_bg * weight_fg * (mu_bg - mu_fg) ** 2
    return int(np.nanargmax(between))


def binarize(img: Image.Image) -> Image.Image:
    """Otsu threshold to pure black and white."""
    gray = np.asarray(img.convert("L"))
    t = otsu_threshold(gray)
    return Image.fromarray(np.where(gray > t, 255, 0).astype(np.uint8), mode="L")


def skew_angle(img: Image.Image) -> float:
    """Rotation (degrees, counter-clockwise) that makes the text lines horizontal.

    Projection profile: rotate the ink mask over candidate angles and keep the one whose row
    sums vary most, i.e. where lines of ink and the gaps between them line up with the rows.
    Coarse search in 0.5 degree steps, then 0.1 degree steps around the best.
    """
    gray = img.convert("L")
    scale = 1000 / max(gray.size)
    if scale < 1:
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.BILINEAR)
    arr = np.asarray(gray)
    ink = Image.fromarray(np.where(arr <= otsu_threshold(arr), 255, 0).astype(np.uint8), mode="L")

    def score(angle: float) -> float:
        rows = np.asarray(ink.rotate(angle, resample=Image.BILINEAR, fillcolor=0), dtype=np.float64).sum(axis=1)
        return float(np.var(rows))

    coarse = max(np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-9, 0.5), key=score)
    fine = max(np.arange(coarse - 0.5, coarse + 0.5 + 1e-9, 0.1), key=score)
    return round(float(fine), 2)


def deskew(img: Image.Image) -> Image.Image:
    angle = skew_angle(img)
    if abs(angle) < 0.05:
        return img
    log.debug(f"deskew: rotating {angle:+.2f} degrees")
    fill = 255 if img.mode in ("L", "1") else (255,) * len(img.getbands())
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=fill)


FILTERS: dict[str, tuple[Callable[[Image.Image], Image.Image], str]] = {
    "grayscale": (grayscale, "convert to 8-bit gray"),
    "autocontrast": (autocontrast, "stretch contrast so ink is black and paper white"),
    "denoise": (denoise, "3x3 median filter, removes scanner speckle"),
    "sharpen": (sharpen, "unsharp mask, crisper stroke edges"),
    "binarize": (binarize, "Otsu threshold to black and white"),
    "deskew": (deskew, f"straighten text lines (up to ±{MAX_SKEW_DEG:.0f}°)"),
}


class FilterError(ValueError):
    pass


def parse_filters(spec: str | None) -> list[str]:
    """`grayscale,denoise` -> ["grayscale", "denoise"]; empty means no preprocessing."""
    names = [n.strip().lower() for n in (spec or "").split(",") if n.strip()]
    unknown = [n for n in names if n not in FILTERS]
    if unknown:
        raise FilterError(f"unknown filter(s): {', '.join(unknown)}. Available: {', '.join(FILTERS)}")
    return names


def apply_filters(path: Path, filters: list[str]) -> None:
    """Filter the image at `path` in place."""
    if not filters:
        return
    with Image.open(path) as src:
        img = src.convert("RGB") if src.mode not in ("RGB", "L") else src.copy()
    for name in filters:
        img = FILTERS[name][0](img)
    img.save(path, format="PNG")
