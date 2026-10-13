"""Lesson 05 and 07 stalls: restyling a window, and a frameless help strip.

The capture host's stall watcher caught two GUI-thread blocks:

* lesson 07, second 1: the Preferences warm-up re-applied the application
  sheet to every window in one timer slice, and each close mark in a
  window re-measured itself on the style change through ``font_px``,
  which opened the preference store for the font scale, once per mark;
* lesson 05: a help strip made frameless re-measured four rows of text
  with a fresh text document and re-set its editor's sheet on every style
  change.

These tests pin the fixes, self-calibrated where they time anything.
"""
from __future__ import annotations

import contextlib
import time

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets import hint_bar

pytestmark = pytest.mark.qt


def _window_of_close_marks(qtbot, count: int = 200) -> QWidget:
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    for _ in range(count):
        layout.addWidget(theme.close_mark_button(window))
    window.show()
    QApplication.processEvents()
    return window


@pytest.fixture
def parked_sheet():
    """The dark sheet parked on the application, as an Apply leaves it."""
    app = QApplication.instance()
    saved = {name: getattr(app, name) for name in (
        theme._WINDOW_SHEET_ATTRIBUTE, theme._WINDOW_SHEET_SERIAL)
        if hasattr(app, name)}
    setattr(app, theme._WINDOW_SHEET_ATTRIBUTE, theme.stylesheet(theme="dark"))
    yield app
    for name in (theme._WINDOW_SHEET_ATTRIBUTE, theme._WINDOW_SHEET_SERIAL):
        if name in saved:
            setattr(app, name, saved[name])
        elif hasattr(app, name):
            delattr(app, name)


def _sheet(app, window) -> float:
    """Put the parked sheet on ``window`` again; seconds taken."""
    setattr(app, theme._WINDOW_SHEET_SERIAL,
            int(getattr(app, theme._WINDOW_SHEET_SERIAL, 0)) + 1)
    started = time.perf_counter()
    assert theme._sheet_one_window(window)
    return time.perf_counter() - started


def test_sheeting_a_window_reads_the_font_scale_once(qtbot, monkeypatch, parked_sheet):
    app = parked_sheet
    window = _window_of_close_marks(qtbot)
    reads = []
    real = preferences.get_font_scale
    monkeypatch.setattr(preferences, "get_font_scale",
                        lambda: reads.append(1) or real())
    _sheet(app, window)
    assert len(reads) <= 2, f"{len(reads)} reads for one window"
    assert theme._PINNED_FONT_SCALE[0] is None
    assert theme.font_px("body") == theme.font_px("body", real())


def test_the_pinned_scale_makes_sheeting_cheaper(qtbot, monkeypatch, parked_sheet):
    """Self-calibrated: the same window sheeted with and without the pin."""
    app = parked_sheet
    window = _window_of_close_marks(qtbot, 300)
    _sheet(app, window)
    pinned = min(_sheet(app, window) for _ in range(3))
    monkeypatch.setattr(theme, "_font_scale_pinned", contextlib.nullcontext)
    unpinned = min(_sheet(app, window) for _ in range(3))
    assert pinned < unpinned, (pinned, unpinned)


def test_a_nested_pin_keeps_the_outer_scale():
    with theme._font_scale_pinned() as outer:
        with theme._font_scale_pinned() as inner:
            assert inner == outer
        assert theme._PINNED_FONT_SCALE[0] == outer
    assert theme._PINNED_FONT_SCALE[0] is None


def test_the_preferences_warm_up_restyles_in_slices():
    source = open(preferences.__file__, encoding="utf-8").read()
    start = source.index("def _warm_one_slice(")
    body = source[start:source.index("\ndef ", start + 1)]
    assert "apply_preferences_to_app(in_slices=True)" in body
    assert "apply_preferences_to_app()" not in body


def test_a_style_change_does_not_remeasure_the_help_strip(qtbot, monkeypatch):
    """Self-calibrated: a style change against one cold measurement."""
    bar = hint_bar.HintBar("Hover a control to see what it does.")
    qtbot.addWidget(bar)
    bar.show()
    QApplication.processEvents()
    height, sheet = bar.height(), bar._view.styleSheet()
    built = []
    real = hint_bar.QTextDocument
    monkeypatch.setattr(hint_bar, "QTextDocument",
                        lambda *a: built.append(1) or real(*a))
    sets = []
    real_set = bar._view.setStyleSheet
    monkeypatch.setattr(bar._view, "setStyleSheet",
                        lambda text: sets.append(text) or real_set(text))
    started = time.perf_counter()
    for _ in range(20):
        bar.changeEvent(QEvent(QEvent.StyleChange))
    warm = (time.perf_counter() - started) / 20
    assert not built and not sets
    assert (bar.height(), bar._view.styleSheet()) == (height, sheet)
    hint_bar._FOUR_ROWS.clear()
    started = time.perf_counter()
    rows = bar._four_painted_rows()
    cold = time.perf_counter() - started
    assert built == [1]
    assert rows == bar._four_painted_rows() and built == [1]
    assert warm < max(cold, 0.0005) * 4, (warm, cold)


def test_the_help_strip_follows_a_new_font(qtbot):
    bar = hint_bar.HintBar("Hover a control to see what it does.")
    qtbot.addWidget(bar)
    bar.show()
    QApplication.processEvents()
    before = bar.height()
    font = bar.font()
    font.setPixelSize(max(8, font.pixelSize(), font.pointSize()) * 2)
    bar.setFont(font)
    QApplication.processEvents()
    assert bar.height() > before
    assert f"font-size: {bar.fontInfo().pixelSize()}px" in (
        bar._view.styleSheet())
