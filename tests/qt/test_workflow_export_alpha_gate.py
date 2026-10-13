"""Run History's Export workflow button is an alpha feature.

``RunHistoryExportWorkflow`` is registered in
``spacr.settings.ALPHA_FEATURES``, so it is hidden until Preferences ->
"Show alpha features" is on, and the export itself never consults the gate.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def mask_run(tmp_path, monkeypatch):
    from spacr import run_journal as journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(journal, "runs_root", lambda: root)
    with journal.open_run("mask", {"src": str(tmp_path / "plate1")}) as run:
        run.set_status("success")
    return run.dir


def test_the_button_is_hidden_until_alpha_features_are_shown(
        qtbot, qt_theme_applied, monkeypatch, mask_run):
    from spacr.qt import preferences
    from spacr.qt.screens.run_history import RunHistoryScreen
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[575] == {"widgets": ("RunHistoryExportWorkflow",)}
    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        screen = RunHistoryScreen(threaded=False)
        qtbot.addWidget(screen)
        assert screen._export_workflow.isHidden() is (not shown)
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        preferences._apply_alpha_widgets(screen)
        assert screen._export_workflow.isHidden() is shown


def test_export_writes_the_selected_run_even_while_hidden(
        qtbot, qt_theme_applied, monkeypatch, mask_run, tmp_path):
    from spacr.qt import preferences
    from spacr.qt.screens.run_history import RunHistoryScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    screen = RunHistoryScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.refresh()
    assert screen.select_run(mask_run)
    assert screen._export_workflow.isEnabled()
    main = screen._export_selected_workflow("nextflow", str(tmp_path / "out"))
    assert main is not None and main.name == "main.nf"
    assert main.parent.name == f"{mask_run.name}_nextflow"
    settings = json.loads((main.parent / "settings" / "plate1.json").read_text())
    assert settings["src"] == str(tmp_path / "plate1")
    assert str(main) in screen._status.text()
    assert screen._export_selected_workflow("make", str(tmp_path)) is None
    assert "Could not export" in screen._status.text()
