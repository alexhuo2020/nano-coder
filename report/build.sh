#!/usr/bin/env bash
# Rebuild the project report from the run metrics.
#
# Two passes are required: the first writes the .aux that resolves \ref and
# \S references, the second consumes it. A single pass leaves "??" in the
# cross-references and still exits 0, so it looks like it worked.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] figures from runs/metrics/*.jsonl"
python make_figures.py

echo "[2/3] pdflatex pass 1"
pdflatex -interaction=nonstopmode report.tex > build1.log 2>&1 || {
  grep -A5 '^!' build1.log | head -40; exit 1; }

echo "[3/3] pdflatex pass 2"
pdflatex -interaction=nonstopmode report.tex > build2.log 2>&1 || {
  grep -A5 '^!' build2.log | head -40; exit 1; }

if grep -q 'undefined references' build2.log; then
  echo "WARNING: unresolved cross-references remain" >&2
fi
echo "ok -> $(pwd)/report.pdf"
