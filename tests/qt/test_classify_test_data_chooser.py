"""Classify's test-data control sits above ``src`` and offers both routes.

Asked for on 2026-09-01: "in classify in plate sources and workflow above the
src top entry there should be a Load test data ... here there is also the
option to load and stream so go for a similar logic to annotate".

It had been filed under Labels & Classes -- where the labels it brings are
configured, but two sections away from the ``src`` it actually fills.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    s = AppScreen("classify")
    qtbot.addWidget(s)
    return s


def test_the_control_is_filed_with_the_sources_it_fills():
    assert EXAMPLE_DATA_SECTIONS["classify"] == "Plate Sources & Workflow"


def test_classify_builds_the_control(screen):
    assert hasattr(screen, "_annotate_example_button")
    assert "test data" in screen._annotate_example_button.text().lower()


def test_choosing_load_takes_the_crops_route(screen, monkeypatch):
    seen = []
    monkeypatch.setattr(screen, "load_the_annotate_example",
                        lambda **k: seen.append("crops") or {"src": "x"})
    monkeypatch.setattr(screen, "load_the_measure_example",
                        lambda **k: seen.append("merged") or {})

    class Chose:
        chosen = "load"
        def exec(self): return 1

    screen.choose_the_test_data(chooser=Chose())
    assert seen == ["crops"]


def test_choosing_stream_takes_the_merged_route(screen, monkeypatch):
    """Stream trains from the merged arrays the crops were cut from."""
    seen = []
    monkeypatch.setattr(screen, "load_the_annotate_example",
                        lambda **k: seen.append("crops") or {})
    monkeypatch.setattr(screen, "load_the_measure_example",
                        lambda **k: seen.append("merged") or {"src": "y"})

    class Chose:
        chosen = "stream"
        def exec(self): return 1

    screen.choose_the_test_data(chooser=Chose())
    assert seen == ["merged"]


def test_closing_without_choosing_downloads_nothing(screen, monkeypatch):
    seen = []
    monkeypatch.setattr(screen, "load_the_annotate_example",
                        lambda **k: seen.append("crops"))
    monkeypatch.setattr(screen, "load_the_measure_example",
                        lambda **k: seen.append("merged"))

    class Chose:
        chosen = ""
        def exec(self): return 0

    assert screen.choose_the_test_data(chooser=Chose()) == {}
    assert not seen


def test_classify_and_annotate_share_one_chooser():
    """Two screens, one implementation -- and a third is asked for."""
    from spacr.qt.screens.annotate import TestDataChooser as from_annotate
    from spacr.qt.widgets.test_data_chooser import TestDataChooser as shared

    assert from_annotate is shared
