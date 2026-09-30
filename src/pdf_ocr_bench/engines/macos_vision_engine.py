"""macOS Vision text recognition: the recognizer behind Live Text in Preview and Photos.

VisionKit (what Preview calls) only exposes a plain transcript; Vision's
`VNRecognizeTextRequest` is the same recognizer with per-line results and per-range boxes,
reachable from Python through pyobjc. macOS only.
"""

from __future__ import annotations

import re
import sys

from ..languages import Language
from ..models import BBox, OcrWord, PageImage
from .base import OcrEngine, Option, Route, split_line, unsupported


class MacOSVisionEngine(OcrEngine):
    name = "macos_vision"
    display_name = "macOS Vision"
    model = "Apple Vision VNRecognizeTextRequest (Live Text's recognizer), macOS only"
    options = {
        "level": Option("accurate", "recognition level: accurate (neural, slower) or fast", choices=("accurate", "fast")),
        "language_correction": Option(True, "let Vision correct words with its language model"),
        "min_text_height": Option(0.0, "ignore text smaller than this fraction of the page height (0 = Vision's default)", minimum=0.0, maximum=1.0),
    }

    @classmethod
    def route(cls, languages: list[Language]) -> Route:
        missing = [lang.code for lang in languages if lang.vision is None]
        if missing:
            return unsupported(f"macOS Vision has no recognizer for {', '.join(missing)}")
        tags = tuple(dict.fromkeys(lang.vision for lang in languages))
        return Route(lang=tags, detail=", ".join(tags))

    @classmethod
    def preflight(cls, route: Route) -> Route:
        if route.ok and sys.platform != "darwin":
            return unsupported("macOS only (Apple Vision framework)")
        return route

    def prepare(self) -> None:
        import Vision  # pyobjc-framework-Vision

        request = self._request()
        supported, error = request.supportedRecognitionLanguagesAndReturnError_(None)
        if error is not None:
            raise RuntimeError(f"Vision: {error}")
        missing = [tag for tag in self.lang if tag not in set(supported)]
        if missing:
            raise RuntimeError(f"this macOS cannot recognize {', '.join(missing)}; it supports {', '.join(supported)}")
        self.log.info(f"VNRecognizeTextRequest revision {request.revision()}, languages {', '.join(self.lang)}")
        self._vision = Vision

    def _request(self):
        import Vision

        request = Vision.VNRecognizeTextRequest.alloc().init()
        level = self.opts["level"]
        request.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate if level == "accurate" else Vision.VNRequestTextRecognitionLevelFast
        )
        request.setUsesLanguageCorrection_(self.opts["language_correction"])
        if self.opts["min_text_height"] > 0:
            request.setMinimumTextHeight_(self.opts["min_text_height"])
        return request

    def ocr_page(self, image: PageImage, lang: tuple[str, ...]) -> list[OcrWord]:
        import objc
        from Foundation import NSURL

        with objc.autorelease_pool():
            request = self._request()
            request.setRecognitionLanguages_(list(lang))
            url = NSURL.fileURLWithPath_(str(image.path))
            handler = self._vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
            ok, error = handler.performRequests_error_([request], None)
            if not ok:
                raise RuntimeError(f"Vision request failed: {error}")
            return [word for observation in request.results() or [] for word in _observation_words(observation)]


def _observation_words(observation) -> list[OcrWord]:
    candidates = observation.topCandidates_(1)
    if not candidates:
        return []
    candidate = candidates[0]
    text = str(candidate.string())
    confidence = float(candidate.confidence())
    words = []
    for m in re.finditer(r"\S+", text):
        # Vision ranges are NSString (UTF-16) offsets
        start, length = _utf16_len(text[: m.start()]), _utf16_len(m.group())
        box, error = candidate.boundingBoxForRange_error_((start, length), None)
        if box is None or error is not None:
            return split_line(text, _bbox(observation.boundingBox()), confidence)
        words.append(OcrWord(text=m.group(), bbox=_bbox(box.boundingBox()), confidence=confidence))
    return words


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _bbox(rect) -> BBox:
    """Vision rects are normalized with the origin at the bottom left."""
    x, y, w, h = rect.origin.x, rect.origin.y, rect.size.width, rect.size.height
    return BBox.from_corners(x, 1.0 - (y + h), x + w, 1.0 - y)
