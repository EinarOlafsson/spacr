"""The Cellpose workbench's Virtual staining button is alpha-gated.

``CellposeWorkbenchVirtualStain`` is registered in
``spacr.settings.ALPHA_FEATURES``, so it is hidden until Preferences ->
"Show alpha features" is on. Hiding is display only: a folder given while
it is hidden still reaches the training run.

Measured on the CPU only (plate1, nucleus stain predicted from the cell
stain, one held-out well of 4 fields, 20 epochs): F1 0.70 at IoU 0.5
against the real stain's objects, against 0.26 for the input channel
alone; F1 0.37 at IoU 0.75; Pearson r 0.88. The run below is mocked.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.screens import train_cellpose as tc

pytestmark = pytest.mark.qt


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def screen(qtbot, alpha):
    from spacr.qt.job_runner import JobRunner

    widget = tc.CellposeWorkbenchScreen()
    qtbot.addWidget(widget)
    widget._vs_jobs = JobRunner(widget, threaded=False)
    widget._vs_jobs.job_failed.connect(widget._on_virtual_stain_failed)
    return widget


def test_the_button_follows_the_alpha_switch(screen, alpha):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[558]["widgets"] == ("CellposeWorkbenchVirtualStain",)
    button = screen.findChild(QPushButton, "CellposeWorkbenchVirtualStain")
    assert button is not None and button.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not button.isHidden()
    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert button.isHidden()


def test_a_folder_given_while_hidden_still_reaches_the_run(
        screen, monkeypatch, tmp_path):
    import spacr.deep_spacr as ds

    seen = []

    def fake(folder, sources, target, **kwargs):
        seen.append((folder, sources, target, kwargs["epochs"]))
        return None, {"test_fields": 2, "predicted_f1_50": 0.8,
                      "input_baseline_f1_50": 0.2, "predicted_pearson": 0.9}

    monkeypatch.setattr(ds, "_virtual_stain_from_folder", fake)
    assert screen._vs_button.isHidden()
    assert screen._virtual_stain(str(tmp_path), "1, 3 > 0") == str(tmp_path)
    assert seen == [(str(tmp_path), [1, 3], 0, 20)]
    assert screen._vs_summary["predicted_f1_50"] == 0.8
    assert "F1 0.80" in screen._vs_note.text()


def test_the_dialogs_ask_for_what_was_not_given(screen, monkeypatch,
                                                tmp_path):
    """No folder asks for one and no channels ask for them; dismissing
    either dialog starts nothing."""
    import spacr.deep_spacr as ds
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    seen = []
    monkeypatch.setattr(ds, "_virtual_stain_from_folder",
                        lambda folder, sources, target, **kw: seen.append(
                            (folder, sources, target)) or (None, {
                                "test_fields": 1, "predicted_f1_50": 0.5,
                                "input_baseline_f1_50": 0.1,
                                "predicted_pearson": 0.7}))
    folders = iter(["", str(tmp_path), str(tmp_path)])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: next(folders)))
    answers = iter([("2 > 1", False), ("2 > 1", True)])
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: next(answers)))
    assert screen._virtual_stain() == ""
    assert screen._virtual_stain() == ""
    assert seen == []
    assert screen._virtual_stain() == str(tmp_path)
    assert seen == [(str(tmp_path), [2], 1)]


def test_a_failed_virtual_stain_says_why(screen, monkeypatch, tmp_path):
    import spacr.deep_spacr as ds

    def broken(*args, **kwargs):
        raise RuntimeError("no paired fields in the folder")

    monkeypatch.setattr(ds, "_virtual_stain_from_folder", broken)
    screen._virtual_stain(str(tmp_path), "1 > 0")
    assert "no paired fields in the folder" in screen._vs_note.text()
    assert screen._vs_note.text().startswith("Virtual staining failed")
