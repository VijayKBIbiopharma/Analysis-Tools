@echo off
REM Start Analysis Tools on this machine. Your data never leaves it.
REM
REM First run creates a virtual environment and installs the packages, which
REM takes a minute. Every run after that is immediate.
cd /d "%~dp0"
if not exist .venv (
  echo First run: setting up. This takes about a minute.
  python -m venv .venv
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\pip install --quiet -r requirements.txt
)
echo Starting. Your browser should open at http://localhost:8501
.venv\Scripts\streamlit run app.py --browser.gatherUsageStats false
pause
