# Run everything locally, one step after another:
#
#   make setup OCR_LANG=deu+eng     system packages (Homebrew / apt), .venv with every engine, html2pdf
#   make test                       unit tests (on a Mac also the macOS Vision tests), html2pdf tests
#   make check  PDF=... OCR_LANG=... validate the inputs against this machine, show the routing
#   make ocr    PDF=scan.pdf OCR_LANG=deu+eng [ENGINES=all] [PREPROCESS=grayscale,deskew,denoise]
#               [OPTIONS="tesseract.psm=6 macos_vision.level=fast"] [DPI=300] [PAGES=1-3]
#   make pdfs                       results/<engine>/pages.zip -> results/<engine>.pdf
#   make all    PDF=scan.pdf OCR_LANG=deu+eng    setup, test, ocr, pdfs
#
# OCR_LANG, not LANG: LANG is the locale variable.

PYTHON ?= $(shell for p in python3.13 python3.12 python3.11; do command -v $$p >/dev/null && { echo $$p; break; }; done)
VENV := .venv
BIN := $(VENV)/bin
EXTRAS ?= ci,test

PDF ?= sample.pdf
OCR_LANG ?= eng
ENGINES ?= all
PREPROCESS ?=
OPTIONS ?=
DPI ?= 300
PAGES ?=
OUT ?= results

RUN_ARGS = --lang "$(OCR_LANG)" --engines "$(ENGINES)" --preprocess "$(PREPROCESS)" \
	$(foreach o,$(OPTIONS),-O "$(o)") $(if $(PAGES),--pages "$(PAGES)")

.PHONY: setup system-deps venv html2pdf test check ocr pdfs all clean

setup: system-deps venv html2pdf

system-deps:
	scripts/install_system_deps.sh "$(OCR_LANG)"

venv: $(BIN)/pdf-ocr-bench

$(BIN)/pdf-ocr-bench: pyproject.toml
	@test -n "$(PYTHON)" || { echo "Python >= 3.11 not found (make system-deps installs it)"; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[$(EXTRAS)]"
	@touch $@

html2pdf:
	cargo build --release --locked --manifest-path html2pdf/Cargo.toml

test: venv
	$(BIN)/pytest -v
	cargo test --release --locked --manifest-path html2pdf/Cargo.toml

check: venv
	$(BIN)/pdf-ocr-bench check --installed $(RUN_ARGS)

ocr: venv
	$(BIN)/pdf-ocr-bench run "$(PDF)" -o "$(OUT)" --dpi $(DPI) $(RUN_ARGS) -v

pdfs: html2pdf
	@for zip in $(OUT)/*/pages.zip; do \
		engine=$$(basename "$$(dirname "$$zip")"); \
		html2pdf/target/release/html2pdf "$$zip" -o "$(OUT)/$$engine.pdf" || exit 1; \
	done

all: setup test ocr pdfs

clean:
	rm -rf "$(OUT)"
