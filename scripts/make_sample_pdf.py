"""Generate sample.pdf: an image-only ("scanned") two-page A4 PDF with English and German text."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DPI = 200
A4_PT = (595.28, 841.89)

PAGES = [
    [
        ("title", "Quarterly Report on Document Digitization"),
        ("body", "The archive contains roughly 12,400 scanned pages from the years"),
        ("body", "1952 to 1987. Most documents are typed letters, invoices and"),
        ("body", "technical drawings. This sample page is used to compare OCR engines."),
        ("body", ""),
        ("head", "1. Methods"),
        ("body", "Each page is rendered at 300 DPI and passed to every engine."),
        ("body", "Word boxes are normalized to fractions of the page size, so the"),
        ("body", "HTML overlay lines up with the background image at any zoom."),
        ("body", ""),
        ("head", "2. Results"),
        ("body", "Engine agreement is measured with the character error rate (CER)"),
        ("body", "between every pair of engines; lower means closer to consensus."),
        ("body", "Invoice No. 2024-0815, total amount: 1,234.56 EUR."),
    ],
    [
        ("title", "Bericht über die Digitalisierung"),
        ("body", "Größere Bestände wurden im Frühjahr gescannt. Die Qualität der"),
        ("body", "Vorlagen schwankt stark: Durchschläge, Matrizenabzüge und"),
        ("body", "handschriftliche Notizen erschweren die Texterkennung."),
        ("body", ""),
        ("head", "Übersicht der Schritte"),
        ("body", "Zunächst prüfen wir Umlaute (ä, ö, ü, Ä, Ö, Ü) und das ß."),
        ("body", "Danach vergleichen wir die Ergebnisse aller Verfahren."),
        ("body", "Straße, Maßnahme, Übergröße, Höhe, Äpfel, Öffnungszeiten."),
        ("body", ""),
        ("body", "Mit freundlichen Grüßen"),
        ("body", "Dr. Jürgen Müller, Köln"),
    ],
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/Library/Fonts/Times New Roman.ttf",
    "C:/Windows/Fonts/times.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def render_page(lines: list[tuple[str, str]], rng: random.Random) -> Image.Image:
    w, h = (round(v / 72 * DPI) for v in A4_PT)
    img = Image.new("L", (w, h), 250)
    draw = ImageDraw.Draw(img)
    sizes = {"title": 44, "head": 34, "body": 28}
    y = 180
    for kind, text in lines:
        font = load_font(sizes[kind])
        draw.text((150, y), text, fill=rng.randint(10, 40), font=font)
        y += int(sizes[kind] * 1.9)
    # scanner look: slight skew, blur and speckle noise
    img = img.rotate(rng.uniform(-0.4, 0.4), resample=Image.BICUBIC, fillcolor=250)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    px = img.load()
    for _ in range(w * h // 400):
        x, yy = rng.randrange(w), rng.randrange(h)
        px[x, yy] = rng.choice((30, 200))
    return img


def main(out: Path) -> None:
    rng = random.Random(42)
    doc = pymupdf.open()
    for lines in PAGES:
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        tmp = out.with_suffix(".tmp.png")
        render_page(lines, rng).save(tmp)
        page.insert_image(page.rect, filename=str(tmp))
        tmp.unlink()
    doc.save(out, deflate=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"))
