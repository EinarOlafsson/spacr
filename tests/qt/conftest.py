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
