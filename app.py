"""Streamlit entry point.

    streamlit run app.py

Kept deliberately thin. Streamlit executes its entry point as a script rather
than importing it as part of a package, so the real application lives in
analysis_tools/app/main.py where normal import rules apply, and this file does
nothing but put the repository root on the path and call it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_tools.app.main import main  # noqa: E402

main()
