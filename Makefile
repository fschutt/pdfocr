# Run everything locally, one step after another:
#
#   make setup OCR_LANG=deu+eng     system packages (Homebrew / apt), .venv with every engine, html2pdf
#              [EXTRAS=macos-vision,tesseract,test]   only these engines (pyproject.toml extras)
#   make test                       unit tests (on a Mac also the macOS Vision tests), html2pdf tests
#   make check  PDF=... OCR_LANG=... validate the inputs against this machine, show the routing
#   make ocr    PDF=scan.pdf OCR_LANG=deu+eng [ENGINES=all] [PREPROCESS=grayscale,deskew,denoise]
#               [OPTIONS="tesseract.psm=6 macos_vision.level=fast"] [DPI=300] [PAGES=1-3]
#   make pdfs                       results/<engine>/pages.zip -> results/<engine>.pdf
#   make all    PDF=scan.pdf OCR_LANG=deu+eng    setup, test, ocr, pdfs
#
# OCR_LANG, not LANG: LANG is the locale variable.

PYTHON ?= $(shell for p in python3.12 python3.11 python3.13; do command -v $$p >/dev/null && { echo $$p; break; }; done)
VENV := .venv
BIN := $(VENV)/bin
# the engines in the venv; defaults to what the last setup installed
DEFAULT_EXTRAS := ci,test
EXTRAS ?= $(or $(shell cat $(VENV)/extras 2>/dev/null),$(DEFAULT_EXTRAS))

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

.PHONY: setup system-deps venv html2pdf test check ocr pdfs all clean FORCE

setup: system-deps venv html2pdf

system-deps:
	scripts/install_system_deps.sh "$(OCR_LANG)"

venv: $(BIN)/pdf-ocr-bench

# rewritten only when EXTRAS changes, which reinstalls
$(VENV)/extras: FORCE
	@mkdir -p $(VENV)
	@test "$$(cat $@ 2>/dev/null)" = "$(EXTRAS)" || echo "$(EXTRAS)" > $@

$(BIN)/pdf-ocr-bench: pyproject.toml $(VENV)/extras
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
	@test -n "$$(ls $(OUT)/*/pages.zip 2>/dev/null)" || { echo "no $(OUT)/*/pages.zip: run make ocr first"; exit 1; }
	@for zip in $(OUT)/*/pages.zip; do \
		engine=$$(basename "$$(dirname "$$zip")"); \
		cargo run -q --release --locked --manifest-path html2pdf/Cargo.toml -- "$$zip" -o "$(OUT)/$$engine.pdf" || exit 1; \
		echo "$(OUT)/$$engine.pdf"; \
	done

all: setup test ocr pdfs

clean:
	rm -rf "$(OUT)"
