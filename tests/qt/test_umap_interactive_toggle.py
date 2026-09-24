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
    # NOT "Live". "Live" is reserved for the four preview panels that
    # re-render a module's own output from the current settings before a
    # run; this explorer makes an already-computed embedding clickable and
    # no setting changes what it draws. See
    # spacr.qt.widgets.preview_contract for the decision.
    assert toggle.text() == "Interactive"
    assert toggle.isChecked() is False
    assert "select a point" in toggle.toolTip().lower()

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


def test_opening_toggling_and_closing_umap_without_a_payload_builds_no_explorer(
        qtbot, qt_theme_applied):
    from matplotlib.figure import Figure

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    screen.show()
    qtbot.wait(20)
    assert screen._if_built("_umap_explorer") is None
    screen._interactive_switch.setChecked(True)
    screen._interactive_switch.setChecked(False)
    screen._on_figure_ready(Figure(figsize=(2, 2)))
    assert screen._figure_queue.count() == 1
    assert screen._if_built("_umap_explorer") is None
    screen.close()
    assert screen._if_built("_umap_explorer") is None


def test_the_first_payload_builds_one_explorer_and_later_payloads_reuse_it(
        qtbot, qt_theme_applied):
    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    assert screen._if_built("_umap_explorer") is None
    screen._interactive_switch.setChecked(True)
    screen._on_figure_ready(_payload_figure())
    explorer = screen._if_built("_umap_explorer")
    assert explorer is not None
    assert not explorer.isHidden()
    screen._on_figure_ready(_payload_figure())
    assert screen._if_built("_umap_explorer") is explorer
    assert screen._figure_queue.count() == 2
    assert screen._umap_payload_ready
    np.testing.assert_array_equal(explorer._embedding, [[0, 0], [1, 1]])
    assert [row["display_name"] for row in explorer._records] == [
        "cell-1", "cell-2"]


def test_the_deferred_explorer_uses_current_language_and_display_settings(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.i18n import retranslate_widget_tree, tr

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    assert screen._if_built("_umap_explorer") is None
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    retranslate_widget_tree(screen)
    assert screen._settings_model.set_value_for_key("figuresize", 11)
    explorer = screen._umap_explorer
    assert explorer._display_btn.text() == tr("Display settings…", "sv")
    assert explorer._display_btn.text() != "Display settings…"
    assert explorer._settings_getter()["figuresize"] == 11
    explorer._propagate_cb({"figuresize": 12})
    assert screen._settings_model.collect()["figuresize"] == 12
