"""Translate Tesseract language codes to the codes each engine expects."""

from __future__ import annotations

LANG_MAP: dict[str, dict[str, str]] = {
    # tesseract_code -> {engine: code}
    "eng":      {"paddleocr": "en",          "easyocr": "en",     "surya": "en", "doctr": "en", "rapidocr": "en",          "ocrmypdf_rapid": "eng"},
    "deu":      {"paddleocr": "german",      "easyocr": "de",     "surya": "de", "doctr": "de", "rapidocr": "latin",       "ocrmypdf_rapid": "deu"},
    "deu_frak": {"paddleocr": "german",      "easyocr": "de",     "surya": "de", "doctr": "de", "rapidocr": "latin",       "ocrmypdf_rapid": "deu"},
    "frk":      {"paddleocr": "latin",       "easyocr": "de",     "surya": "de", "doctr": "de", "rapidocr": "latin",       "ocrmypdf_rapid": "deu"},
    "fra":      {"paddleocr": "fr",          "easyocr": "fr",     "surya": "fr", "doctr": "fr", "rapidocr": "latin",       "ocrmypdf_rapid": "fra"},
    "chi_sim":  {"paddleocr": "ch",          "easyocr": "ch_sim", "surya": "zh", "doctr": "zh", "rapidocr": "ch",          "ocrmypdf_rapid": "chi_sim"},
    "chi_tra":  {"paddleocr": "chinese_cht", "easyocr": "ch_tra", "surya": "zh", "doctr": "zh", "rapidocr": "chinese_cht", "ocrmypdf_rapid": "chi_tra"},
    "jpn":      {"paddleocr": "japan",       "easyocr": "ja",     "surya": "ja", "doctr": "ja", "rapidocr": "japan",       "ocrmypdf_rapid": "jpn"},
    "kor":      {"paddleocr": "korean",      "easyocr": "ko",     "surya": "ko", "doctr": "ko", "rapidocr": "korean",      "ocrmypdf_rapid": "kor"},
    "ara":      {"paddleocr": "ar",          "easyocr": "ar",     "surya": "ar", "doctr": "ar", "rapidocr": "arabic",      "ocrmypdf_rapid": "ara"},
    "rus":      {"paddleocr": "ru",          "easyocr": "ru",     "surya": "ru", "doctr": "ru", "rapidocr": "cyrillic",    "ocrmypdf_rapid": "rus"},
    # the rapidocr plugin has no "lat" entry; any Latin-script code selects the same model
    "lat":      {"paddleocr": "latin",       "easyocr": "la",     "surya": "la", "doctr": "la", "rapidocr": "latin",       "ocrmypdf_rapid": "ita"},
}

# Engines that consume Tesseract codes directly.
TESSERACT_NATIVE = {"tesseract", "ocrmypdf"}


def base_lang(tesseract_code: str) -> str:
    return tesseract_code.split("+")[0].strip()


def get_lang(tesseract_code: str, engine: str) -> str:
    """Map Tesseract lang code to engine-specific code. Falls back to 'en'."""
    if engine in TESSERACT_NATIVE:
        return tesseract_code
    return LANG_MAP.get(base_lang(tesseract_code), {}).get(engine, "en")


def is_known(tesseract_code: str) -> bool:
    return all(part in LANG_MAP for part in tesseract_code.split("+"))


HTML_LANG = {
    "eng": "en", "deu": "de", "deu_frak": "de", "frk": "de", "fra": "fr", "chi_sim": "zh-Hans",
    "chi_tra": "zh-Hant", "jpn": "ja", "kor": "ko", "ara": "ar", "rus": "ru", "lat": "la",
}  # fmt: skip


def html_lang(tesseract_code: str) -> str:
    return HTML_LANG.get(base_lang(tesseract_code), "en")
