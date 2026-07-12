"""
pytest configuration shared by all tests under backend/tests/.

The application code (scoring/models.py, scoring/safety.py, scoring/constants.py)
is normally run either as bare scripts with `backend/scoring` directly on
sys.path (scripts/train_and_score.py, scoring/train.py), or as the `scoring.*`
package with `backend/` on sys.path (app/main.py, run via
`uvicorn app.main:app` from the backend/ directory).

Tests are invoked from the repo root (see backend/tests/README-less setup:
`backend/venv/Scripts/python.exe -m pytest backend/tests/ -v`), so neither of
those working directories is implicit. This conftest inserts `backend/` onto
sys.path once, for the whole test session, so `from scoring.models import ...`
and `from scoring.safety import ...` resolve the same way they do inside the
running app -- no network calls, no app startup, just import path setup.
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
