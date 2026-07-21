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
    # Console widget exists and starts with only the trailing stretch.
    assert screen._console._entries.count() == 1
    # Run + stop + import + clear buttons exist and start in expected state.
    assert screen._btn_run.isEnabled()
    assert not screen._btn_stop.isEnabled()
    assert screen._btn_clear.text() == "Clear console"
    # Usage bars exist.
    for label in ("_usage_ram", "_usage_gpu", "_usage_vram", "_usage_cpu"):
        assert getattr(screen, label) is not None
    # Timer is running.
    assert screen._usage_timer.isActive()


def test_app_screen_settings_widgets_populated(qtbot, qt_theme_applied):
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    # Settings model built at least one widget for the mask app.
    assert len(screen._settings_model._widgets) > 0
    # Collected dict includes common keys.
    settings = screen._settings_model.collect()
    assert "src" in settings


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
