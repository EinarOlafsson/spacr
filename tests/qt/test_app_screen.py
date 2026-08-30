"""AppScreen (settings + console + usage + actions) tests."""
from __future__ import annotations

import pytest

from spacr.qt.screens.app_screen import APP_TITLES, AppScreen
from spacr.qt.bridge import resolve_pipeline_entry


@pytest.mark.parametrize("app_key", [
    "mask", "measure", "classify", "umap",
    "train_cellpose", "cellpose_masks", "cellpose_all",
    "map_barcodes", "ml_analyze", "regression",
    "recruitment", "activation", "analyze_plaques",
])
def test_app_screen_constructs_for_every_key(qtbot, qt_theme_applied, app_key):
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    # The final layout item is always the stretch that keeps entries anchored
    # at the top.  Construction may already have written an actionable status
    # message (for example, that no optional AI provider is installed), so an
    # absolute-empty assertion races legitimate startup diagnostics.
    entries = screen._console._entries
    assert entries.count() >= 1
    assert entries.itemAt(entries.count() - 1).spacerItem() is not None
    # Run + stop + import + clear buttons exist and start in expected state.
    assert screen._btn_run.isEnabled()
    assert not screen._btn_stop.isEnabled()
    assert screen._btn_clear.text() == "Clear console"
    # Usage bars exist.
    for label in ("_usage_ram", "_usage_gpu", "_usage_vram", "_usage_cpu"):
        assert getattr(screen, label) is not None
    # Hidden stacked pages do not poll. Only the visible page owns a timer.
    assert not screen._usage_timer.isActive()
    screen.show()
    qtbot.waitUntil(screen.isVisible)
    assert screen._usage_timer.isActive()
    screen.hide()
    assert not screen._usage_timer.isActive()


def test_a_sample_finishing_after_hide_does_not_repaint_the_screen(qtbot):
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    assert not screen._usage_timer.isActive()

    in_flight_generation = screen._usage_generation
    screen._usage_generation += 1  # hideEvent invalidates this request.
    screen._apply_usage({"ram": 73}, in_flight_generation)
    assert screen._usage_ram._bar.value() == 0

    screen._apply_usage({"ram": 73}, screen._usage_generation)
    assert screen._usage_ram._bar.value() == 73


def test_app_screen_settings_widgets_populated(qtbot, qt_theme_applied):
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    # Settings model built at least one widget for the mask app.
    assert len(screen._settings_model._widgets) > 0
    # Collected dict includes common keys.
    settings = screen._settings_model.collect()
    assert "src" in settings


def test_regression_settings_build_in_laptop_extra_performance_mode(
        qtbot, qt_theme_applied, monkeypatch):
    """The supported low-resource path reaches a real Regression panel.

    Timing instrumentation once wrapped widget construction with an undefined
    ``_span`` alias.  The screen catches construction errors and displays them
    in its console, so checking only that the outer widget exists would let the
    exact regression pass unnoticed.  Assert both the populated model and the
    timing record produced by the corrected call site.
    """
    from spacr.qt import preferences, timing

    preferences.set_laptop_mode("on")
    preferences.set_spacr_mode("extra_performance")
    monkeypatch.setattr(timing, "ENABLED", True)
    timing._SPANS.clear()

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert screen._settings_model is not None
    assert screen._settings_model._widgets
    assert "regression_type" in screen._settings_model._widgets
    assert any(row["name"] == "build widgets" for row in timing._SPANS)


def test_app_titles_cover_apps():
    for key in ("mask", "measure", "classify", "umap"):
        assert key in APP_TITLES


@pytest.mark.parametrize("key,expected_present", [
    ("mask", True),
    ("measure", True),
    ("classify", True),
    ("umap", True),
    ("annotate", False),      # interactive-only
    ("make_masks", False),    # interactive-only
    ("unknown_key", False),
])
def test_resolve_pipeline_entry(key, expected_present):
    entry = resolve_pipeline_entry(key)
    if expected_present:
        assert callable(entry), f"expected callable for {key}, got {entry!r}"
    else:
        assert entry is None
