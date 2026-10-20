"""Shortcut help distinguishes crop focus from annotation page navigation."""

import sqlite3

import pytest
from PIL import Image
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication

from spacr.qt import shortcuts
from spacr.qt.screens.annotate import AnnotateScreen


@pytest.fixture
def screen(qtbot, tmp_path):
    source = tmp_path / "experiment"
    (source / "measurements").mkdir(parents=True)
    paths = []
    for index in range(12):
        path = source / f"crop_{index:02d}.png"
        Image.new("RGB", (24, 24), (index * 10, 40, 90)).save(path)
        paths.append((str(path),))
    with sqlite3.connect(source / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" VALUES (?)', paths)
    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.grid_rows = 2
    widget._settings.grid_cols = 3
    widget._settings.image_size = (24, 24)
    widget._compute_grid_dims = lambda: None
    widget._rebuild_grid()
    widget._open_source(str(source))
    qtbot.waitUntil(lambda: len(widget._page_paths) == 6, timeout=5000)
    yield widget
    shortcuts._save_keymap({}, {})
    if widget._worker is not None:
        widget._worker.stop(wait=True)


@pytest.mark.parametrize("key, target, caption", [
    (Qt.Key_Left, 0, "Move focus left"),
    (Qt.Key_Right, 2, "Move focus right"),
])
def test_annotation_map_describes_the_actual_navigation(qtbot, screen,
                                                        key, target, caption):
    try:
        page = tuple(screen._page_paths)
        offset = screen._offset
        screen._set_focus_slot(1)
        QApplication.sendEvent(screen, QKeyEvent(QEvent.KeyPress, key,
                                                  Qt.NoModifier))
        assert screen._focus_slot == target
        assert screen._offset == offset
        assert tuple(screen._page_paths) == page

        screen._spacr_screen_holders["PageDown"].activated.emit()
        qtbot.waitUntil(lambda: tuple(screen._page_paths) != page,
                        timeout=5000)
        assert screen._offset == offset + screen._settings.page_size
        screen._spacr_screen_holders["PageUp"].activated.emit()
        qtbot.waitUntil(lambda: tuple(screen._page_paths) == page,
                        timeout=5000)
        assert screen._offset == offset
        screen._spacr_screen_holders["Alt+Right"].activated.emit()
        qtbot.waitUntil(lambda: tuple(screen._page_paths) != page,
                        timeout=5000)
        assert screen._offset == offset + screen._settings.page_size
        screen._spacr_screen_holders["Alt+Left"].activated.emit()
        qtbot.waitUntil(lambda: tuple(screen._page_paths) == page,
                        timeout=5000)
        assert screen._offset == offset

        default = QKeySequence(key).toString(QKeySequence.PortableText)
        shown = [spec for spec in shortcuts.mapped()
                 if spec.scope == "the Annotate screen" and spec.keys == default]
        assert len(shown) == 1
        page_labels = {spec.keys: spec.label for spec in shortcuts.mapped()
                       if spec.scope == "the Annotate screen"
                       and spec.keys in ("PageUp", "PageDown", "Alt+Left", "Alt+Right")}
        assert {default: shown[0].label, **page_labels} == {
            default: caption, "PageUp": "Previous page", "Alt+Left": "Previous page",
            "PageDown": "Next page", "Alt+Right": "Next page"}

        shortcuts._save_keymap({}, {"Annotate": {default: "F8"}})
        shortcuts._apply_screen_keymaps()
        screen._set_focus_slot(1)
        QApplication.sendEvent(screen, QKeyEvent(QEvent.KeyPress, Qt.Key_F8,
                                                  Qt.NoModifier))
        assert screen._focus_slot == target
        rebound = [spec for spec in shortcuts.mapped()
                   if spec.scope == "the Annotate screen" and spec.keys == "F8"]
        assert len(rebound) == 1
        assert rebound[0].label == caption
        assert tuple(screen._page_paths) == page
    finally:
        shortcuts._save_keymap({}, {})


def test_other_screen_arrows_keep_their_image_navigation_captions():
    for scope in ("Make Masks", "Field browser"):
        specs = {spec.keys: spec for spec in shortcuts._screen_specs(scope)}
        assert specs["Left"].label == "Previous image"
        assert specs["Right"].label == "Next image"
