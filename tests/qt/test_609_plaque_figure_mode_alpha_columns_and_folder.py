"""Item 609: Figure mode hides the alpha estimate columns and loads its folder.

Found by spacr-d7 while re-recording tutorials, 2026-10-01.

* The Wells and Plaques tables showed the experimental growth estimate's
  columns ("Estimated pixels per µm", "Estimated time (hours)", "Estimate
  basis") with Preferences -> Show alpha features off. They follow the same
  gate as the Estimate scale / time button that fills them.
* A figure folder chosen in Plaque mode was listed the Plaque way -- no
  images directly inside a folder of paper folders -- and switching to Figure
  mode kept "No images found". The source is listed again for the new mode.
"""
from __future__ import annotations

import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings                                  # noqa: E402
from PySide6.QtWidgets import QApplication                            # noqa: E402

from spacr import plaque_papers as PP                                 # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv                    # noqa: E402

ESTIMATES = ("Estimated pixels per µm", "Estimated time (hours)",
             "Estimate basis")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


@pytest.fixture
def panel(qtbot, monkeypatch, prefs):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _hidden(table, columns) -> list:
    return [table.isColumnHidden(columns.index(name)) for name in ESTIMATES]


def test_the_estimate_columns_are_registered_through_item_501():
    from spacr.settings import ALPHA_FEATURES

    assert "PlaqueEstimateScaleTime" in ALPHA_FEATURES[501]["widgets"]
    assert set(ESTIMATES) <= set(ppv.TABLE_COLUMNS)
    assert set(ESTIMATES) <= set(ppv.PLAQUE_COLUMNS)


def test_the_estimate_columns_follow_show_alpha_features(panel, prefs):
    assert prefs._get_show_alpha_features() is False
    assert _hidden(panel._table, ppv.TABLE_COLUMNS) == [True] * 3
    assert _hidden(panel._plaque_table, ppv.PLAQUE_COLUMNS) == [True] * 3
    assert panel._growth_btn.isHidden()
    others = [n for n in ppv.TABLE_COLUMNS if n not in ESTIMATES]
    assert not any(panel._table.isColumnHidden(ppv.TABLE_COLUMNS.index(n))
                   for n in others)

    prefs._set_show_alpha_features(True)
    panel._apply_alpha_columns()
    assert _hidden(panel._table, ppv.TABLE_COLUMNS) == [False] * 3
    assert _hidden(panel._plaque_table, ppv.PLAQUE_COLUMNS) == [False] * 3

    prefs._set_show_alpha_features(False)
    panel._apply_alpha_columns()
    assert _hidden(panel._table, ppv.TABLE_COLUMNS) == [True] * 3
    assert panel._table.columnCount() == len(ppv.TABLE_COLUMNS)


def test_a_value_in_a_hidden_column_is_kept(panel):
    from PySide6.QtWidgets import QTableWidgetItem

    column = ppv.TABLE_COLUMNS.index("Estimated pixels per µm")
    panel._table.setRowCount(1)
    panel._table.setItem(0, column, QTableWidgetItem("1.25"))
    panel._apply_alpha_columns()
    assert panel._table.isColumnHidden(column)
    assert panel._table.item(0, column).text() == "1.25"


def test_closing_preferences_reapplies_the_gate_on_the_screen(qtbot, prefs,
                                                              monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    screen = AppScreen("analyze_plaques")
    qtbot.addWidget(screen)
    panel = screen._live_preview
    assert _hidden(panel._table, ppv.TABLE_COLUMNS) == [True] * 3
    prefs._set_show_alpha_features(True)
    screen._refresh_alpha_visibility()
    assert _hidden(panel._table, ppv.TABLE_COLUMNS) == [False] * 3
    assert _hidden(panel._plaque_table, ppv.PLAQUE_COLUMNS) == [False] * 3


def _figure_folders(root: Path) -> Path:
    """A folder holding two paper folders, as several read PDFs leave it."""
    for name in ("paper_a", "paper_b"):
        folder = root / "papers" / name
        folder.mkdir(parents=True)
        imageio.imwrite(folder / "fig1.png",
                        np.full((32, 32, 3), 90, dtype=np.uint8))
        (folder / PP.PAPER_FILE).write_text("{}")
    return root / "papers"


def _settle(ms: int = 400) -> None:
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def _listed(panel) -> list:
    return [f"{p.parent.name}/{p.name}" for p in panel._paths]


def test_a_figure_folder_loaded_in_plaque_mode_loads_after_the_switch(
        panel, tmp_path):
    folder = _figure_folders(tmp_path)
    panel.load_source_async(str(folder))
    _settle()
    assert _listed(panel) == []
    assert "No images found" in panel.preview_status()
    panel._mode_switch.button("figure").click()
    _settle()
    assert panel.mode() == "figure"
    assert _listed(panel) == ["paper_a/fig1.png", "paper_b/fig1.png"]
    panel._mode_switch.button("plaque").click()
    _settle()
    assert _listed(panel) == []


def test_the_screens_switch_loads_the_figure_folder_too(qtbot, tmp_path,
                                                        monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda box: 0)
    screen = AppScreen("analyze_plaques")
    qtbot.addWidget(screen)
    screen.show()
    folder = _figure_folders(tmp_path)
    screen._on_preview_switch(True)
    screen._settings_model._widgets["src"].setText(str(folder))
    _settle(900)
    panel = screen._live_preview
    assert _listed(panel) == []
    ppv.choose_plaque_mode(screen, "figure")
    _settle()
    assert _listed(panel) == ["paper_a/fig1.png", "paper_b/fig1.png"]


def test_a_mode_change_without_a_source_lists_nothing(panel):
    panel._mode_switch.button("figure").click()
    assert panel._src == ""
    assert panel._paths == []
