"""Bottom-row activation control for the interactive image UMAP."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.app_screen import AppScreen


def _payload_figure():
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2, 2))
    axes = figure.subplots()
    axes.scatter([0.0, 1.0], [0.0, 1.0])
    figure._spacr_umap_payload = {
        "embedding": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "labels": np.array([0, 1]),
        "records": [
            {"display_name": "cell-1"},
            {"display_name": "cell-2"},
        ],
    }
    return figure


def test_umap_has_interactive_toggle_immediately_beside_ai(
        qtbot, qt_theme_applied):
    screen = AppScreen("umap")
    qtbot.addWidget(screen)

    toggle = screen._interactive_switch
    assert toggle is not None
    assert toggle.text() == "Live"
    assert toggle.isChecked() is False
    assert "click a point" in toggle.toolTip().lower()

    row = screen._ai_switch.parentWidget().layout()
    assert row.indexOf(toggle) + 1 == row.indexOf(screen._ai_switch)


def test_interactive_toggle_is_umap_only(qtbot, qt_theme_applied):
    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    assert screen._interactive_switch is None


def test_toggle_selects_interactive_or_static_view(
        qtbot, qt_theme_applied):
    screen = AppScreen("umap")
    qtbot.addWidget(screen)

    # Users can activate the feature before running UMAP.
    screen._interactive_switch.setChecked(True)
    assert screen._interactive_switch.isChecked()
    assert screen._umap_explorer.isHidden()

    screen._on_figure_ready(_payload_figure())
    assert screen._umap_payload_ready is True
    assert screen._figure_queue.count() == 1
    assert not screen._umap_explorer.isHidden()
    assert screen._figure_queue.isHidden()

    # The static plot was retained, so switching off is immediate.
    screen._interactive_switch.setChecked(False)
    assert screen._umap_explorer.isHidden()
    assert not screen._figure_queue.isHidden()

    # Switching back on reuses the loaded payload without another run.
    screen._interactive_switch.setChecked(True)
    assert not screen._umap_explorer.isHidden()
    assert screen._figure_queue.isHidden()


def test_payload_defaults_to_static_until_interactive_is_enabled(
        qtbot, qt_theme_applied):
    screen = AppScreen("umap")
    qtbot.addWidget(screen)

    screen._on_figure_ready(_payload_figure())

    assert screen._interactive_switch.isChecked() is False
    assert screen._umap_explorer.isHidden()
    assert not screen._figure_queue.isHidden()
    assert screen._figure_queue.count() == 1
