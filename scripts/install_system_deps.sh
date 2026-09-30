#!/usr/bin/env bash
# System packages for every engine: Tesseract with the models for the given languages,
# ocrmypdf's tools, llama.cpp (Surya on CPU), Python >= 3.11, and Rust >= 1.88 (html2pdf).
#
#   scripts/install_system_deps.sh deu+eng
#
# macOS: Homebrew. Debian/Ubuntu: apt (sudo). Anything that cannot be installed safely from
# here (rustup, llama.cpp on Linux) is printed as a hint instead.
set -euo pipefail

lang="${1:-eng}"
cd "$(dirname "$0")/.."

hint() { printf '\n  -> %s\n' "$*"; }

have_rust() {
  command -v cargo >/dev/null && [ "$(printf '%s\n' 1.88.0 "$(cargo --version | awk '{print $2}')" | sort -V | head -1)" = 1.88.0 ]
}

case "$(uname -s)" in
  Darwin)
    command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
    # tesseract-lang carries every Tesseract model (frk, Fraktur, chi_sim, ...)
    brew install python@3.12 tesseract tesseract-lang ghostscript qpdf unpaper pngquant llama.cpp
    have_rust || brew install rust
    ;;
  Linux)
    command -v apt-get >/dev/null || {
      echo "Only macOS (Homebrew) and Debian/Ubuntu (apt) are scripted; see README.md for the package list."
      exit 1
    }
    # the models for --lang, from the same table the CLI routes with (no third-party imports)
    packages=$(python3 -c '
import sys
sys.path.insert(0, "src")
from pdf_ocr_bench.languages import parse_languages, tesseract_packages
print(" ".join(tesseract_packages(parse_languages(sys.argv[1]))))
' "$lang")
    sudo apt-get update -qq
    # shellcheck disable=SC2086 # word splitting of the package list is intended
    sudo apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential \
      tesseract-ocr ghostscript qpdf unpaper pngquant libgl1 libglib2.0-0 $packages
    have_rust || hint "Rust >= 1.88 is needed for html2pdf: https://rustup.rs (apt's cargo is too old)"
    command -v llama-server >/dev/null || hint "Surya needs llama-server on PATH: https://github.com/ggml-org/llama.cpp/releases (else Surya is skipped)"
    ;;
  *)
    echo "Unsupported OS $(uname -s); see README.md for the package list."
    exit 1
    ;;
esac

echo
echo "Installed Tesseract models:"
tesseract --list-langs
