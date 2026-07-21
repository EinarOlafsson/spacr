"""Tests for the polish pass: EmptyState widget, iconset, updated
typography palette, status bar, and empty-state wiring in the screens."""
from __future__ import annotations

import pytest
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QLabel, QPushButton

from spacr.qt import iconset
from spacr.qt.theme import PALETTE, TYPOGRAPHY, stylesheet
from spacr.qt.widgets import EmptyState
from spacr.qt.widgets.empty_state import _WrappedLabel


# ---------------------------------------------------------------------------
# Iconset
# ---------------------------------------------------------------------------

def test_iconset_returns_qicon_for_known_name(qt_theme_applied):
    ic = iconset.icon("open")
    assert isinstance(ic, QIcon)


def test_iconset_unknown_name_falls_back_to_placeholder(qt_theme_applied):
    ic = iconset.icon("this-doesnt-exist")
    assert isinstance(ic, QIcon)


def test_iconset_accent_and_contrast_variants_are_distinct_qicon(qt_theme_applied):
    a = iconset.accent_icon("brush")
    c = iconset.contrast_icon("brush")
    assert isinstance(a, QIcon)
    assert isinstance(c, QIcon)


def test_iconset_covers_every_app_key(qt_theme_applied):
    from spacr.qt.app import APPS
    for key, *_ in APPS:
        # Just proves the semantic key resolves; a placeholder QIcon is fine
        assert isinstance(iconset.icon(key), QIcon)


# ---------------------------------------------------------------------------
# Theme additions
# ---------------------------------------------------------------------------

def test_typography_roles_defined():
    for role in ("display", "title", "subtitle", "header",
                  "body", "small", "caption", "hero"):
        assert role in TYPOGRAPHY
        entry = TYPOGRAPHY[role]
        assert "size" in entry
        assert "weight" in entry


def test_palette_gains_surface_hi_and_accent_soft():
    assert PALETTE["surface_hi"].startswith("#")
    assert PALETTE["accent_soft"].startswith("#")


def test_stylesheet_references_new_selectors():
    qss = stylesheet()
    for selector in ("#Hero", "#TitleHeading", "#SubtitleSmall",
                     "#Caption"):
        assert selector in qss


# ---------------------------------------------------------------------------
# EmptyState widget
# ---------------------------------------------------------------------------

def test_empty_state_renders_all_slots(qtbot, qt_theme_applied):
    es = EmptyState(
        title="Nothing yet",
        subtitle="Pick a folder to start.",
        icon=iconset.accent_icon("brush"),
        cta_label="Open folder",
    )
    qtbot.addWidget(es)
    labels = [l.text() for l in es.findChildren(QLabel)]
    assert any("Nothing yet" in t for t in labels)
    assert any("Pick a folder" in t for t in labels)
    assert es.cta_button is not None
    assert es.cta_button.text() == "Open folder"


def test_empty_state_skips_missing_slots(qtbot, qt_theme_applied):
    es = EmptyState()
    qtbot.addWidget(es)
    assert es.cta_button is None
    assert not es.findChildren(QLabel)


def test_empty_state_cta_emits_signal_and_runs_callback(qtbot, qt_theme_applied):
    fired = {"count": 0}
    es = EmptyState(cta_label="Go", on_action=lambda: fired.__setitem__("count", fired["count"] + 1))
    qtbot.addWidget(es)
    with qtbot.waitSignal(es.action_triggered, timeout=1000):
        es.cta_button.click()
    assert fired["count"] == 1


def test_wrapped_label_sizes_for_wrap_width(qtbot, qt_theme_applied):
    text = ("This is a wrapped label with many words in it, "
            "specifically enough words to definitely span more than "
            "one line at a moderate width, and force the wrapper to "
            "return a taller-than-single-line size hint.")
    lbl = _WrappedLabel(text, wrap_width=280)
    qtbot.addWidget(lbl)
    hint = lbl.sizeHint()
    assert hint.width() == 280
    # Multi-line wrap at 280px must be taller than single-line height
    assert hint.height() > 30


# ---------------------------------------------------------------------------
# Wiring on the screens
# ---------------------------------------------------------------------------

def test_annotate_screen_starts_on_empty_state(qtbot, qt_theme_applied):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    assert screen._content_stack.currentWidget() is screen._empty_state


def test_make_masks_screen_starts_on_empty_state(qtbot, qt_theme_applied):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._body_stack.currentWidget() is screen._empty_state


def test_make_masks_empty_state_cta_swaps_to_editor(qtbot, qt_theme_applied,
                                                     tmp_path):
    """Clicking the empty-state CTA opens the folder picker; when a valid
    folder is opened programmatically, the stack should switch to the
    editor splitter."""
    from spacr.qt.screens.make_masks import MakeMasksScreen
    import imageio.v2 as imageio
    import numpy as np

    folder = tmp_path / "images"
    folder.mkdir()
    imageio.imwrite(folder / "a.tif",
                    np.zeros((16, 16), dtype=np.uint16))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._body_stack.currentWidget() is screen._empty_state
    screen._open_folder(str(folder))
    assert screen._body_stack.currentWidget() is screen._body_splitter


def test_toolbar_buttons_have_icons(qtbot, qt_theme_applied):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    for btn in (screen._btn_open, screen._btn_settings, screen._btn_prev,
                 screen._btn_next, screen._btn_skip, screen._btn_count,
                 screen._btn_clear):
        assert not btn.icon().isNull(), f"{btn.text()!r} should have an icon"


def test_main_window_status_bar_has_version_and_app_labels(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._status_version_label.text().startswith("SpaCR")
    assert win._status_app_label.text() == "Home"
    win._on_nav_selected("mask")
    assert win._status_app_label.text() == "Mask"
