"""The virtual-stain button stops when either dialog is dismissed."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt.screens import train_cellpose as tc  # noqa: E402


@pytest.fixture
def button(qtbot):
    widget = tc._VirtualStainApply()
    qtbot.addWidget(widget)
    return widget


def test_no_model_chosen_means_no_run(button, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert button.apply() == ""
    assert not button.note.text()


def test_no_folder_chosen_means_no_run(button, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "m.pt"), "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: asked.append(1) or ""))
    assert button.apply(folder=str(tmp_path / "missing")) == ""
    assert asked == [1]


def test_a_failure_is_shown_beside_the_button(button):
    button._on_failed("no weights")
    assert "no weights" in button.note.text()
