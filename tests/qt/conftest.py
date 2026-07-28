"""Pytest fixtures for the Qt GUI test suite.

Runs offscreen — no X server required. Skips cleanly if PySide6 or
pytest-qt is not installed so the rest of the suite still runs.
"""
from __future__ import annotations

from importlib.util import find_spec
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Skipping while this conftest is imported aborts collection of the entire
# repository on pytest 7, leaving pytest with exit code 5 ("no tests
# collected").  Ignore only this directory's test modules when an optional Qt
# test dependency is absent so the non-Qt suite can still run.
_QT_TEST_DEPENDENCIES = ("PySide6", "pytestqt")
collect_ignore_glob = (
    ["test_*.py"]
    if any(find_spec(module) is None for module in _QT_TEST_DEPENDENCIES)
    else []
)


@pytest.fixture(scope="session")
def qt_theme_applied(qapp):
    """Apply the spacr palette + QSS to the shared QApplication once."""
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qapp)
    qapp.setStyleSheet(stylesheet())
    return qapp


@pytest.fixture(autouse=True)
def _skip_first_launch_tour():
    """The first-launch tour attaches a modal overlay to the MainWindow
    the first time it opens. Left alone, it steals focus + adds widgets
    that break test isolation. Mark it "seen" for every Qt test so
    MainWindow constructs without the overlay."""
    try:
        from spacr.qt.first_run import mark_tour_seen, reset_tour_state
        mark_tour_seen()
        yield
        # Leave the "seen" flag alone — tests that specifically want
        # the tour set force=True (see test_onboarding).
    except Exception:
        yield
