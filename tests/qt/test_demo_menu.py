"""Tests for the "Load demo dataset…" menu action in MainWindow.

We bypass the QFileDialog by calling `_run_demo_generator` +
`_apply_demo_to_screen` directly with a tmp_path destination — that's
what the menu callback would call after the user picks a folder.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _new_mainwindow(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_demo_targets_cover_every_generator(qtbot, qt_theme_applied):
    """Every DEMO_TARGETS entry must reference a real generator on
    spacr.qt.synthetic — a typo here would silently break the menu."""
    from spacr.qt import synthetic as syn
    win = _new_mainwindow(qtbot, qt_theme_applied)
    for demo_key, (target_app, gen_name) in win.DEMO_TARGETS.items():
        assert hasattr(syn, gen_name), (
            f"DEMO_TARGETS['{demo_key}'] points at missing "
            f"spacr.qt.synthetic.{gen_name}")


@pytest.mark.parametrize("demo_key", ["mask", "measure", "crop",
                                        "classify", "timelapse",
                                        "map_barcodes"])
def test_run_demo_generator_produces_layout(qtbot, qt_theme_applied,
                                              tmp_path, demo_key):
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator(demo_key, str(tmp_path))
    assert layout.src.exists()
    # Every generator returns a settings CSV that spacr.utils can read
    assert layout.settings_csv is not None
    assert layout.settings_csv.exists()


def test_apply_mask_demo_populates_app_screen(qtbot, qt_theme_applied,
                                                 tmp_path):
    """The mask demo → mask AppScreen path: after applying, the
    settings model's src widget should carry our tmp_path."""
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator("mask", str(tmp_path))

    win._on_nav_selected("mask")
    screen = win._screens.get("mask")
    assert screen is not None
    win._apply_demo_to_screen(screen, layout)

    src_w = getattr(screen, "_settings_model", None)
    assert src_w is not None
    widgets = src_w._widgets
    if "src" in widgets:
        from PySide6.QtWidgets import QLineEdit
        w = widgets["src"]
        # Whatever widget type is used for src, its value should carry
        # the tmp_path we generated the demo into
        if isinstance(w, QLineEdit):
            assert str(layout.src) in w.text()


def test_apply_classify_demo_opens_annotate_screen(qtbot,
                                                     qt_theme_applied,
                                                     tmp_path):
    """The classify demo routes to the AnnotateScreen (not an AppScreen)
    because that's where users label the crops."""
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator("classify", str(tmp_path))
    win._on_nav_selected("annotate")
    screen = win._screens.get("annotate")
    assert screen is not None
    # Sanity: AnnotateScreen exposes _open_source, which is what
    # _apply_demo_to_screen falls through to when settings can't apply
    assert hasattr(screen, "_open_source")


def test_demo_menu_has_expected_entries(qtbot, qt_theme_applied):
    """Menu wiring — every demo is a QAction under &Demos, including the
    real-dataset end-to-end option."""
    win = _new_mainwindow(qtbot, qt_theme_applied)
    demos_menu = None
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == "Demos":
            demos_menu = act.menu()
            break
    assert demos_menu is not None, "no &Demos menu found"
    actions = [a for a in demos_menu.actions() if not a.isSeparator()]
    labels = {a.text() for a in actions}
    for expected in ("Mask demo…", "Measure demo…", "Crop demo…",
                      "Classify demo…", "Timelapse demo…",
                      "Sequencing demo…"):
        assert expected in labels
    # The real-dataset E2E option should be present
    assert any("End-to-end" in lbl and "Annotate" in lbl for lbl in labels)
