"""A screen built while the window opens it takes breaths, and says so."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import screens, timing  # noqa: E402


@pytest.fixture
def breathing(qapp, monkeypatch):
    marks = []
    monkeypatch.setattr(timing, "mark", lambda name, detail="": marks.append(
        (name, detail)))
    screens._start_breathing_while_a_window_opens(time.perf_counter() - 1.0)
    yield marks
    screens._stop_breathing_while_a_window_opens()


def test_nothing_happens_outside_a_window_open(qapp, monkeypatch):
    marks = []
    monkeypatch.setattr(timing, "mark", lambda *a: marks.append(a))
    monkeypatch.setattr(timing, "ENABLED", True)
    screens._stop_breathing_while_a_window_opens()
    screens._breathe_while_a_window_opens(force=True)
    assert marks == []


def _breathe_on_behalf_of_a_step():
    """A helper named like the breath itself, which the note must skip."""
    screens._breathe_while_a_window_opens(force=True)


def _the_step_that_asked():
    _breathe_on_behalf_of_a_step()


def test_a_breathing_helper_is_skipped_to_name_the_real_step(
        breathing, monkeypatch):
    monkeypatch.setattr(timing, "ENABLED", True)
    _the_step_that_asked()
    assert "_the_step_that_asked" in breathing[0][1]
