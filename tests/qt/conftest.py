"""Pytest fixtures for the Qt GUI test suite.

Runs offscreen — no X server required. Skips cleanly if PySide6 or
pytest-qt is not installed so the rest of the suite still runs.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed; skipping Qt tests")
pytest.importorskip("pytestqt", reason="pytest-qt not installed; skipping Qt tests")


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
