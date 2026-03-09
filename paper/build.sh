#!/bin/bash
# Build LSME paper
# Run: cd paper && bash build.sh

set -e

echo "=== Building main.tex ==="
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "=== Building supp.tex ==="
pdflatex -interaction=nonstopmode supp.tex
bibtex supp
pdflatex -interaction=nonstopmode supp.tex
pdflatex -interaction=nonstopmode supp.tex

echo ""
echo "=== Done ==="
echo "Output: main.pdf, supp.pdf"
