#!/usr/bin/env bash
# Builds manuscript_ieee.pdf. Needs a TeX Live with pdflatex, bibtex and the IEEEtran class (package `IEEEtran`;
# the class is NOT shipped here). If your TeX installation lacks it, put IEEEtran.cls next to this script or set TEXINPUTS.
set -euo pipefail
cd "$(dirname "$0")"
export TEXINPUTS="${TEXINPUTS:-}:$PWD:$PWD/vendor:"
pdflatex -interaction=nonstopmode manuscript_ieee.tex >/dev/null
bibtex manuscript_ieee >/dev/null || true
pdflatex -interaction=nonstopmode manuscript_ieee.tex >/dev/null
pdflatex -interaction=nonstopmode manuscript_ieee.tex | grep -E "Output written|Error" || true
