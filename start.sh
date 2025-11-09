#!/usr/bin/env bash
set -e

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install requirements if not already installed (in case build step didn't)
pip install -r requirements.txt

# If index missing, create it
if [ ! -f data_out/assessments.csv ] || [ ! -f data_out/assess_index.faiss ]; then
  echo "data_out missing — running build_index.py (this may take several minutes)..."
  python build_index.py
else
  echo "data_out already present — skipping build_index.py"
fi

# Start Gradio app (app.py) — will bind to 7860
python app.py
