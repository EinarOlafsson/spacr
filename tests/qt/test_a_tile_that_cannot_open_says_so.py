"""A module that fails to build must say so, not look like a dead tile.

GitHub #130, 2026-09-22: "When I look at the tiles, I can click them and
nothing seems to happen". An exception raised while a screen is built happens
inside a Qt slot: Qt prints it to a terminal the user does not have and the
window carries on as if nothing was pressed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _window(qtbot):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_a_missing_package_is_named_rather_than_silent(qtbot, monkeypatch):
    window = _window(qtbot)
    said = []
    monkeypatch.setattr(type(window), "_build_screen",
                        lambda self, key: (_ for _ in ()).throw(
                            ImportError("No module named 'pylibCZIrw'",
                                        name="pylibCZIrw")))
    monkeypatch.setattr(type(window), "_say_a_module_would_not_open",
                        lambda self, key, exc: said.append((key, exc)) or "said")

    window._on_nav_selected("measure")
    assert said and said[0][0] == "measure", "the failure reached the person"
    assert "measure" not in window._screens, "a broken screen is not kept"


def test_the_sentence_names_the_module_and_the_package(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = _window(qtbot)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    said = window._say_a_module_would_not_open(
        "measure", ImportError("No module named 'pylibCZIrw'",
                               name="pylibCZIrw"))
    assert "pylibCZIrw" in said
    assert "measure" in said.lower() or "Measure" in said

    other = window._say_a_module_would_not_open("mask", ValueError("no good"))
    assert "no good" in other and "ValueError" in other
