"""The `--lang` vocabulary: supported document languages, validated before anything runs.

Codes are Tesseract's (`eng`, `deu`, `chi_sim`, ...), joined with `+` for multilingual documents.
Each engine turns the parsed list into its own model choice (`OcrEngine.route`); this table
holds the per-engine codes those routes are built from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # Tesseract code: the --lang vocabulary
    name: str
    script: str  # Latin | Cyrillic | Arabic | Han | Japanese | Hangul
    tag: str  # BCP 47 tag, for the HTML `lang` attribute
    paddle: str  # PaddleOCR `lang`
    rapidocr: str  # RapidOCR recognizer family, see engines/rapidocr_engine.py
    easyocr: str  # EasyOCR language code
    vision: str | None  # macOS Vision recognition language; None = not supported
    # Tesseract models to try, in order; the first installed one is used
    tesseract_models: tuple[str, ...]
    # Debian/Ubuntu packages that provide those models
    tesseract_packages: tuple[str, ...]


def _lang(code, name, script, tag, paddle, rapidocr, easyocr, vision, models=None, packages=None) -> Language:
    return Language(
        code=code,
        name=name,
        script=script,
        tag=tag,
        paddle=paddle,
        rapidocr=rapidocr,
        easyocr=easyocr,
        vision=vision,
        tesseract_models=models or (code,),
        tesseract_packages=packages or (f"tesseract-ocr-{code.replace('_', '-')}",),
    )


# There is no LSTM `deu_frak` model any more: German Fraktur is `frk`, and `Fraktur` is the
# script model (installed at the tessdata root on Debian/Ubuntu, under script/ elsewhere).
_FRAKTUR_MODELS = ("frk", "Fraktur", "script/Fraktur")
_FRAKTUR_PACKAGES = ("tesseract-ocr-frk", "tesseract-ocr-script-frak")

LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        _lang("eng", "English", "Latin", "en", "en", "en", "en", "en-US"),
        # Tesseract's Middle English model knows the long s (ſ), which `eng` reads as f; it also
        # suits 16th-18th century English print. The other engines read it with their English model.
        _lang("enm", "English (historical, long s)", "Latin", "en", "en", "en", "en", "en-US"),
        _lang("deu", "German", "Latin", "de", "de", "latin", "de", "de-DE"),
        _lang("deu_frak", "German (Fraktur)", "Latin", "de", "de", "latin", "de", "de-DE", _FRAKTUR_MODELS, _FRAKTUR_PACKAGES),
        _lang("frk", "Fraktur", "Latin", "de", "de", "latin", "de", "de-DE", _FRAKTUR_MODELS, _FRAKTUR_PACKAGES),
        _lang("fra", "French", "Latin", "fr", "fr", "latin", "fr", "fr-FR"),
        _lang("spa", "Spanish", "Latin", "es", "es", "latin", "es", "es-ES"),
        _lang("ita", "Italian", "Latin", "it", "it", "latin", "it", "it-IT"),
        _lang("por", "Portuguese", "Latin", "pt", "pt", "latin", "pt", "pt-BR"),
        _lang("nld", "Dutch", "Latin", "nl", "nl", "latin", "nl", "nl-NL"),
        _lang("pol", "Polish", "Latin", "pl", "pl", "latin", "pl", "pl-PL"),
        _lang("lat", "Latin", "Latin", "la", "la", "latin", "la", None),
        _lang("rus", "Russian", "Cyrillic", "ru", "ru", "eslav", "ru", "ru-RU"),
        _lang("ukr", "Ukrainian", "Cyrillic", "uk", "uk", "eslav", "uk", "uk-UA"),
        _lang("ara", "Arabic", "Arabic", "ar", "ar", "arabic", "ar", "ar-SA"),
        _lang("chi_sim", "Chinese (Simplified)", "Han", "zh-Hans", "ch", "ch", "ch_sim", "zh-Hans"),
        _lang("chi_tra", "Chinese (Traditional)", "Han", "zh-Hant", "chinese_cht", "chinese_cht", "ch_tra", "zh-Hant"),
        _lang("jpn", "Japanese", "Japanese", "ja", "japan", "japan", "ja", "ja-JP"),
        _lang("kor", "Korean", "Hangul", "ko", "korean", "korean", "ko", "ko-KR"),
    )
}


class LanguageError(ValueError):
    pass


def parse_languages(spec: str) -> list[Language]:
    """`deu+eng` -> [German, English]. Order is kept (the first language is the main one)."""
    codes = [c.strip() for c in (spec or "").split("+")]
    if not any(codes) or not all(codes):
        raise LanguageError(f"invalid --lang {spec!r}: expected codes joined by '+', e.g. deu or deu+eng")
    unknown = [c for c in codes if c not in LANGUAGES]
    if unknown:
        raise LanguageError(
            f"unknown language code(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(LANGUAGES)} (see `pdf-ocr-bench languages`)"
        )
    return [LANGUAGES[c] for c in dict.fromkeys(codes)]


def html_lang(languages: list[Language]) -> str:
    return languages[0].tag


def tesseract_packages(languages: list[Language]) -> list[str]:
    """apt packages providing the Tesseract models for `languages`."""
    return list(dict.fromkeys(p for lang in languages for p in lang.tesseract_packages))
