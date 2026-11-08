#!/usr/bin/env bash
# Downloads the dataset if missing, runs the analysis and the tests.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw
if [ ! -f data/raw/ai4i2020.csv ]; then
  curl -sSL -o data/raw/ai4i.zip "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
  unzip -o -q data/raw/ai4i.zip -d data/raw
fi
python src/analysis.py
python -m pytest -q
