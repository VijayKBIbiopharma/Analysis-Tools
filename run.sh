#!/bin/sh
# Start Analysis Tools on this machine. Your data never leaves it.
#
# First run creates a virtual environment and installs the packages, which takes
# a minute. Every run after that is immediate.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up. This takes about a minute."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi
echo "Starting. Your browser should open at http://localhost:8501"
exec .venv/bin/streamlit run app.py --browser.gatherUsageStats false
