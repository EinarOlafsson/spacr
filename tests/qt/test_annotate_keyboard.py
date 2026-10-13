"""Keyboard-only rapid annotation on the Qt annotate screen.

Every test drives :meth:`AnnotateScreen.handle_key` directly rather than
synthesising Qt key events, so the suite exercises the binding logic
without depending on offscreen event delivery. One test still pushes a
real ``QKeyEvent`` through ``keyPressEvent`` to prove the wiring is
connected.

Offscreen, CPU-only, no modal dialogs.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from PySide6.QtCore import Qt

from spacr.qt.screens import annotate as annotate_mod


ROWS, COLS = 2, 3
PAGE = ROWS * COLS          # 6 crops per page
N_CROPS = 12                # → exactly two pages


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def kbd_source(tmp_path: Path) -> Path:
    """A minimal experiment folder: `measurements.db` + N_CROPS tiny PNGs."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    rng = np.random.default_rng(7)
    png_paths: List[str] = []
    for i in range(N_CROPS):
        arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        p = src / "data" / f"crop_{i:02d}.png"
        Image.fromarray(arr).save(p)
        png_paths.append(str(p))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in png_paths])
    return src


def _open_screen(qtbot, src: Path, rows: int = ROWS, cols: int = COLS):
    """Build an AnnotateScreen with a pinned grid and load the first page.

    ``_compute_grid_dims`` normally re-fits the grid to the viewport; that is
    non-deterministic offscreen, so it is pinned here to keep slot indices
    meaningful.
    """
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = rows
    screen._settings.grid_cols = cols
    screen._settings.image_size = (24, 24)
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == rows * cols,
                    timeout=5000)
    return screen


def _stop(screen) -> None:
    if screen._worker is not None:
        screen._worker.stop(wait=True)


def _wait_saved(screen, qtbot, timeout_ms: int = 5000) -> None:
    """Poll until the SaveWorker has drained its queue."""
    for _ in range(timeout_ms // 20):
        qtbot.wait(20)
        w = screen._worker
        if w is not None and not w.busy and w.pending_batches == 0 \
                and w.last_save_ts is not None:
            return


def _db_labels(src: Path) -> dict:
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        return dict(conn.execute(
            'SELECT png_path, annotate FROM "png_list"').fetchall())


def _labels(screen) -> List:
    return [v for _p, v in screen._page_paths]


# ---------------------------------------------------------------------------
# key_token — pure normalisation, no widgets needed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("1", "1"), ("9", "9"), ("0", "0"),
    ("h", "left"), ("j", "down"), ("k", "up"), ("l", "right"),
    ("H", "left"), ("L", "right"),
    ("left", "left"), ("Right", "right"), ("UP", "up"), ("Down", "down"),
    (" ", "space"), ("space", "space"),
    ("backspace", "backspace"), ("u", "undo"),
    ("enter", "enter"), ("return", "enter"), ("?", "help"),
])
def test_key_token_from_text(given, expected):
    assert annotate_mod.key_token(given) == expected


@pytest.mark.parametrize("qt_key,expected", [
    (Qt.Key_1, "1"), (Qt.Key_9, "9"), (Qt.Key_0, "0"),
    (Qt.Key_Left, "left"), (Qt.Key_Right, "right"),
    (Qt.Key_Up, "up"), (Qt.Key_Down, "down"),
    (Qt.Key_H, "left"), (Qt.Key_J, "down"),
    (Qt.Key_K, "up"), (Qt.Key_L, "right"),
    (Qt.Key_Space, "space"), (Qt.Key_Backspace, "backspace"),
    (Qt.Key_U, "undo"), (Qt.Key_Return, "enter"), (Qt.Key_Enter, "enter"),
    (Qt.Key_Question, "help"), (Qt.Key_Escape, "escape"),
])
def test_key_token_from_qt_code(qt_key, expected):
    assert annotate_mod.key_token(qt_key) == expected


@pytest.mark.parametrize("given", ["z", "qq", "", "  ", "F5", None, object()])
def test_key_token_unbound_returns_none(given):
    assert annotate_mod.key_token(given) is None


def test_key_token_falls_back_to_event_text():
    # Qt.Key_F5 has no binding, but a character in `text` still resolves.
    assert annotate_mod.key_token(Qt.Key_F5, "3") == "3"


# ---------------------------------------------------------------------------
# 1..9 assign + persist through the existing write path
# ---------------------------------------------------------------------------

def test_digits_assign_expected_class(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        # Focus starts on the first unlabelled crop (slot 0 on a fresh page).
        assert screen.focus_slot == 0
        for slot, digit in enumerate(("1", "2", "3", "4", "5", "9")):
            assert screen.focus_slot == slot
            assert screen.handle_key(digit) is True
        assert _labels(screen) == [1, 2, 3, 4, 5, 9]
    finally:
        _stop(screen)


def test_digit_assignment_persists_through_save_worker(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("1")
        screen.handle_key("7")
        paths = [p for p, _ in screen._page_paths]
        screen._flush_pending()
        _wait_saved(screen, qtbot)
    finally:
        _stop(screen)
    stored = _db_labels(kbd_source)
    assert stored[paths[0]] == 1
    assert stored[paths[1]] == 7
    for p, v in stored.items():
        if p not in (paths[0], paths[1]):
            assert v is None


def test_digit_uses_the_same_pending_map_as_the_mouse(
        qtbot, qt_theme_applied, kbd_source):
    """Keyboard labels land in `_pending_updates` exactly like clicks do."""
    screen = _open_screen(qtbot, kbd_source)
    try:
        path0 = screen._page_paths[0][0]
        screen.handle_key("4")
        assert screen._pending_updates[path0] == 4
        # And the mouse path still works alongside it.
        screen._on_thumb_right(2)
        assert screen._pending_updates[screen._page_paths[2][0]] == 2
        assert screen._page_paths[2][1] == 2
    finally:
        _stop(screen)


def test_reassigning_a_digit_overwrites_rather_than_toggling(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("1")           # slot 0 -> 1, focus -> 1
        screen._set_focus_slot(0)
        screen.handle_key("3")           # slot 0 -> 3 (NOT cleared)
        assert screen._page_paths[0][1] == 3
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Auto-advance
# ---------------------------------------------------------------------------

def test_auto_advance_skips_already_labelled_crops(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        # Pre-label slots 1 and 2 by mouse; keyboard must hop over them.
        screen._on_thumb_left(1)          # class 1
        screen._on_thumb_right(2)         # class 2
        screen._set_focus_slot(0)
        screen.handle_key("5")
        assert screen._page_paths[0][1] == 5
        assert screen.focus_slot == 3, "should have skipped labelled 1 and 2"
        assert screen._page_paths[1][1] == 1   # untouched
        assert screen._page_paths[2][1] == 2   # untouched
    finally:
        _stop(screen)


def test_focus_lands_on_first_unannotated_on_page_load(
        qtbot, qt_theme_applied, kbd_source):
    """Rows 0 and 1 are already labelled in the DB, so focus starts at 2."""
    db = kbd_source / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute('ALTER TABLE "png_list" ADD COLUMN "annotate" INTEGER')
        rows = [r[0] for r in conn.execute('SELECT png_path FROM "png_list"')]
        conn.executemany('UPDATE "png_list" SET annotate = 1 '
                         'WHERE png_path = ?', [(rows[0],), (rows[1],)])
    screen = _open_screen(qtbot, kbd_source)
    try:
        assert screen.focus_slot == 2
    finally:
        _stop(screen)


def test_end_of_page_does_not_wrap_and_tells_the_user(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        for _ in range(PAGE):
            screen.handle_key("1")
        assert _labels(screen) == [1] * PAGE
        # Focus stayed on the last crop instead of wrapping to slot 0.
        assert screen.focus_slot == PAGE - 1
        assert "End of page" in screen._kbd_hint.text()
        assert "Enter" in screen._kbd_hint.text()
    finally:
        _stop(screen)


def test_end_of_page_reports_unlabelled_crops_left_behind(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        # Label only the last crop; slots 0..4 stay empty and must be named.
        screen._set_focus_slot(PAGE - 1)
        screen.handle_key("2")
        assert screen.focus_slot == PAGE - 1     # no wrap
        hint = screen._kbd_hint.text()
        assert "End of page" in hint
        assert f"{PAGE - 1} unlabelled" in hint
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# 0 clears
# ---------------------------------------------------------------------------

def test_zero_clears_focused_crop(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("1")               # slot 0 -> 1, focus -> 1
        screen._set_focus_slot(0)
        assert screen.handle_key("0") is True
        assert screen._page_paths[0][1] is None
        # Clearing stays put so the crop can be re-keyed immediately.
        assert screen.focus_slot == 0
        assert screen._pending_updates[screen._page_paths[0][0]] is None
    finally:
        _stop(screen)


def test_zero_on_unlabelled_crop_is_harmless(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        assert screen.handle_key("0") is True
        assert screen._page_paths[0][1] is None
    finally:
        _stop(screen)


def test_zero_clear_persists_as_null(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        path0 = screen._page_paths[0][0]
        screen.handle_key("6")
        screen._flush_pending()
        _wait_saved(screen, qtbot)
        screen._set_focus_slot(0)
        screen.handle_key("0")
        screen._flush_pending()
        _wait_saved(screen, qtbot)
    finally:
        _stop(screen)
    assert _db_labels(kbd_source)[path0] is None


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def test_undo_reverts_last_assignment(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("3")
        assert screen._page_paths[0][1] == 3
        assert screen.handle_key("u") is True
        assert screen._page_paths[0][1] is None
        # Undo parks focus back on the crop it repaired.
        assert screen.focus_slot == 0
    finally:
        _stop(screen)


def test_undo_unwinds_several_in_a_row(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        for digit in ("1", "2", "3", "4"):
            screen.handle_key(digit)
        assert _labels(screen)[:4] == [1, 2, 3, 4]
        screen.handle_key("u")
        assert _labels(screen)[:4] == [1, 2, 3, None]
        screen.handle_key("u")
        assert _labels(screen)[:4] == [1, 2, None, None]
        screen.handle_key("u")
        screen.handle_key("u")
        assert _labels(screen)[:4] == [None, None, None, None]
        assert screen.focus_slot == 0
    finally:
        _stop(screen)


def test_undo_past_the_start_is_a_noop(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        # Nothing done yet.
        assert screen.handle_key("u") is True
        assert "Nothing to undo" in screen._kbd_hint.text()
        assert _labels(screen) == [None] * PAGE
        # One assignment, then undo twice — the second must not raise.
        screen.handle_key("8")
        screen.handle_key("u")
        assert screen.handle_key("u") is True
        assert "Nothing to undo" in screen._kbd_hint.text()
        assert _labels(screen) == [None] * PAGE
    finally:
        _stop(screen)


def test_undo_restores_a_previous_value_not_just_null(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("1")             # slot 0 -> 1
        screen._set_focus_slot(0)
        screen.handle_key("5")             # slot 0 -> 5, previous was 1
        assert screen._page_paths[0][1] == 5
        screen.handle_key("u")
        assert screen._page_paths[0][1] == 1
    finally:
        _stop(screen)


def test_undo_covers_the_zero_clear(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("2")
        screen._set_focus_slot(0)
        screen.handle_key("0")
        assert screen._page_paths[0][1] is None
        screen.handle_key("u")
        assert screen._page_paths[0][1] == 2
    finally:
        _stop(screen)


def test_undo_stack_is_bounded(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        for _ in range(annotate_mod.UNDO_LIMIT + 40):
            screen._set_focus_slot(0)
            screen.handle_key("1")
        assert len(screen._undo_stack) == annotate_mod.UNDO_LIMIT
    finally:
        _stop(screen)


def test_undo_stack_clears_on_page_load(qtbot, qt_theme_applied, kbd_source):
    """Slot indices mean different crops after paging; stale undo is dropped."""
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("1")
        assert len(screen._undo_stack) == 1
        screen.handle_key("enter")              # next batch
        qtbot.waitUntil(lambda: screen._offset == PAGE, timeout=5000)
        assert len(screen._undo_stack) == 0
        screen.handle_key("u")
        assert "Nothing to undo" in screen._kbd_hint.text()
        assert _labels(screen) == [None] * PAGE
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Space / Backspace — movement without labelling
# ---------------------------------------------------------------------------

def test_space_advances_without_labelling(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        assert screen.handle_key("space") is True
        assert screen.focus_slot == 1
        screen.handle_key(" ")
        assert screen.focus_slot == 2
        assert _labels(screen) == [None] * PAGE
        assert screen._pending_updates == {}
    finally:
        _stop(screen)


def test_space_does_not_skip_labelled_crops(qtbot, qt_theme_applied,
                                             kbd_source):
    """Space is a plain step — unlike auto-advance it visits every crop."""
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen._on_thumb_left(1)
        screen._set_focus_slot(0)
        screen.handle_key("space")
        assert screen.focus_slot == 1
    finally:
        _stop(screen)


def test_backspace_steps_back_without_labelling(qtbot, qt_theme_applied,
                                                 kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen._set_focus_slot(3)
        assert screen.handle_key("backspace") is True
        assert screen.focus_slot == 2
        screen.handle_key(Qt.Key_Backspace)
        assert screen.focus_slot == 1
        assert _labels(screen) == [None] * PAGE
        assert screen._pending_updates == {}
    finally:
        _stop(screen)


def test_space_and_backspace_stop_at_the_page_edges(qtbot, qt_theme_applied,
                                                     kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        assert screen.focus_slot == 0
        screen.handle_key("backspace")
        assert screen.focus_slot == 0
        assert "Start of page" in screen._kbd_hint.text()
        screen._set_focus_slot(PAGE - 1)
        screen.handle_key("space")
        assert screen.focus_slot == PAGE - 1
        assert "End of page" in screen._kbd_hint.text()
        assert _labels(screen) == [None] * PAGE
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Focus movement bounds
# ---------------------------------------------------------------------------

def test_arrow_and_vi_keys_move_focus_in_the_grid(qtbot, qt_theme_applied,
                                                   kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("right")
        assert screen.focus_slot == 1
        screen.handle_key("down")
        assert screen.focus_slot == 1 + COLS
        screen.handle_key("left")
        assert screen.focus_slot == COLS
        screen.handle_key("up")
        assert screen.focus_slot == 0
        # vi aliases behave identically
        screen.handle_key("l")
        assert screen.focus_slot == 1
        screen.handle_key("j")
        assert screen.focus_slot == 1 + COLS
        screen.handle_key("h")
        assert screen.focus_slot == COLS
        screen.handle_key("k")
        assert screen.focus_slot == 0
        assert _labels(screen) == [None] * PAGE
    finally:
        _stop(screen)


def test_focus_movement_stays_in_bounds_at_every_edge(qtbot, qt_theme_applied,
                                                       kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        edges = [
            (0, "left", 0), (0, "up", 0),                       # top-left
            (COLS - 1, "right", COLS - 1),                      # top-right
            (COLS - 1, "up", COLS - 1),
            (COLS, "left", COLS),                               # bottom-left
            (COLS, "down", COLS),
            (PAGE - 1, "right", PAGE - 1),                      # bottom-right
            (PAGE - 1, "down", PAGE - 1),
        ]
        for start, token, expected in edges:
            screen._set_focus_slot(start)
            assert screen.handle_key(token) is True
            assert screen.focus_slot == expected, \
                f"{token} from slot {start} escaped the grid"
        # Hammering every direction can never leave the valid range.
        for token in ["left", "up", "right", "down"] * 8:
            screen.handle_key(token)
            assert 0 <= screen.focus_slot < PAGE
        assert _labels(screen) == [None] * PAGE
    finally:
        _stop(screen)


def test_focus_movement_respects_a_partially_filled_last_row(
        qtbot, qt_theme_applied, tmp_path):
    """A 3x3 grid over 7 rows leaves slots 7 and 8 empty — focus can't go there."""
    src = tmp_path / "small"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    paths = []
    for i in range(7):
        p = src / "data" / f"c{i}.png"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 3
    screen._settings.grid_cols = 3
    screen._settings.image_size = (16, 16)
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == 7, timeout=5000)
    try:
        assert screen._slot_count() == 7
        screen._set_focus_slot(6)        # only crop in the last row
        screen.handle_key("right")       # slot 7 has no crop
        assert screen.focus_slot == 6
        screen._set_focus_slot(5)
        screen.handle_key("down")        # slot 8 has no crop
        assert screen.focus_slot == 5
        screen._set_focus_slot(4)
        screen.handle_key("down")        # slot 7 has no crop
        assert screen.focus_slot == 4
    finally:
        _stop(screen)


def test_focus_keys_are_safe_before_a_source_is_opened(qtbot, qt_theme_applied):
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    for token in ("right", "down", "left", "up", "space", "backspace",
                  "1", "0", "u"):
        assert screen.handle_key(token) is True
    assert screen.focus_slot == 0
    assert screen._pending_updates == {}


# ---------------------------------------------------------------------------
# Focus visibility
# ---------------------------------------------------------------------------

def test_focused_crop_is_marked_and_only_one_at_a_time(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen._set_focus_slot(1)
        marked = [i for i, t in enumerate(screen._thumbs)
                  if t.property("kbdFocused")]
        assert marked == [1]
        screen.handle_key("right")            # slot 1 -> 2 (still in row 0)
        marked = [i for i, t in enumerate(screen._thumbs)
                  if t.property("kbdFocused")]
        assert marked == [2]
        screen.handle_key("down")             # slot 2 -> 5 (row 1)
        marked = [i for i, t in enumerate(screen._thumbs)
                  if t.property("kbdFocused")]
        assert marked == [5]
    finally:
        _stop(screen)


def test_focus_ring_is_distinct_from_the_class_border(qtbot, qt_theme_applied,
                                                       kbd_source):
    """The ring colour is never produced by `label_to_hex`, so focus and
    label can be read off the same crop at once."""
    from spacr.qt.annotate_engine import label_to_hex
    ring = annotate_mod.current_ring_color().lower()
    assert ring not in {(label_to_hex(v) or "").lower() for v in range(1, 40)}


def test_focus_ring_is_reported_by_the_focused_tile(qtbot, qt_theme_applied,
                                                     kbd_source):
    """Moving focus moves the ring, and only the focused tile carries it."""
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen._set_focus_slot(0)
        assert screen._thumbs[0].is_current()
        assert screen._thumbs[0].ring_color() == \
            annotate_mod.current_ring_color()
        screen._set_focus_slot(1)
        assert not screen._thumbs[0].is_current()
        assert screen._thumbs[0].ring_color() is None
        assert screen._thumbs[1].is_current()
    finally:
        _stop(screen)


def test_focus_visible_before_images_decode(qtbot, qt_theme_applied,
                                             kbd_source):
    """Even with no pixmap yet the focused cell carries a visible ring."""
    screen = _open_screen(qtbot, kbd_source)
    try:
        for i in range(len(screen._thumbs)):
            screen._set_slot_image(i, None)   # undo the decode
        screen._set_focus_slot(0)
        assert screen._thumbs[0].pixmap().isNull()
        assert screen._thumbs[0].is_current()
        assert not screen._thumbs[1].is_current()
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Enter — commit page / next batch
# ---------------------------------------------------------------------------

def test_enter_loads_the_next_batch_like_the_next_button(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        first_paths = [p for p, _ in screen._page_paths]
        screen.handle_key("1")
        assert screen.handle_key("enter") is True
        qtbot.waitUntil(lambda: screen._offset == PAGE, timeout=5000)
        second_paths = [p for p, _ in screen._page_paths]
        assert second_paths and second_paths != first_paths
        # Enter flushes: the label made before it must reach the DB.
        _wait_saved(screen, qtbot)
    finally:
        _stop(screen)
    assert _db_labels(kbd_source)[first_paths[0]] == 1


def test_enter_on_the_last_page_says_so_and_still_saves(
        qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen.handle_key("enter")
        qtbot.waitUntil(lambda: screen._offset == PAGE, timeout=5000)
        path0 = screen._page_paths[0][0]
        screen.handle_key("2")
        assert screen.handle_key("enter") is True
        assert screen._offset == PAGE, "must not paginate past the end"
        assert "last page" in screen._kbd_hint.text().lower()
        _wait_saved(screen, qtbot)
    finally:
        _stop(screen)
    assert _db_labels(kbd_source)[path0] == 2


def test_enter_resets_focus_on_the_new_page(qtbot, qt_theme_applied,
                                             kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen._set_focus_slot(PAGE - 1)
        screen.handle_key("enter")
        qtbot.waitUntil(lambda: screen._offset == PAGE, timeout=5000)
        assert screen.focus_slot == 0
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def test_legend_exists_and_takes_no_focus(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QWidget
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    legend = screen._legend
    assert legend is not None
    assert legend.focusPolicy() == Qt.NoFocus
    children = legend.findChildren(QWidget)
    assert children, "legend should actually contain something"
    for child in children:
        assert child.focusPolicy() == Qt.NoFocus, \
            f"{child.objectName() or child} would steal keyboard focus"


def test_legend_documents_every_binding(qtbot, qt_theme_applied):
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    text = screen._legend_label.text()
    for fragment in ("1", "9", "0", "hjkl", "Space", "Backspace",
                     "u", "Enter"):
        assert fragment in text, f"legend never mentions {fragment!r}"


def test_question_mark_expands_the_legend_without_a_dialog(
        qtbot, qt_theme_applied):
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    compact = screen._legend_label.text()
    assert screen.handle_key("?") is True
    expanded = screen._legend_label.text()
    assert expanded != compact and len(expanded) > len(compact)
    assert screen._legend_expanded is True
    # Escape collapses it again; a second Escape is left to Qt.
    assert screen.handle_key(Qt.Key_Escape) is True
    assert screen._legend_label.text() == compact
    assert screen.handle_key(Qt.Key_Escape) is False


def test_legend_toggle_button_is_wired(qtbot, qt_theme_applied):
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    compact = screen._legend_label.text()
    screen._legend_toggle.click()
    assert screen._legend_label.text() != compact


# ---------------------------------------------------------------------------
# Unbound keys / event plumbing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["z", "Q", "F1", Qt.Key_F5, Qt.Key_Tab,
                                  Qt.Key_Shift, None, 999999, object()])
def test_unbound_keys_are_ignored_not_raised(qtbot, qt_theme_applied,
                                              kbd_source, key):
    screen = _open_screen(qtbot, kbd_source)
    try:
        assert screen.handle_key(key) is False
        assert screen.focus_slot == 0
        assert _labels(screen) == [None] * PAGE
        assert screen._pending_updates == {}
    finally:
        _stop(screen)


def test_real_key_event_reaches_handle_key(qtbot, qt_theme_applied,
                                            kbd_source):
    """keyPressEvent is genuinely wired to handle_key, not just callable."""
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    screen = _open_screen(qtbot, kbd_source)
    try:
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_3, Qt.NoModifier, "3")
        screen.keyPressEvent(ev)
        assert screen._page_paths[0][1] == 3
        assert screen.focus_slot == 1
        assert ev.isAccepted()
        # An unbound key falls through to Qt without exploding.
        ev2 = QKeyEvent(QEvent.KeyPress, Qt.Key_F7, Qt.NoModifier, "")
        screen.keyPressEvent(ev2)
        assert screen._page_paths[0][1] == 3
    finally:
        _stop(screen)


def test_event_filter_intercepts_grid_scroll_keys(qtbot, qt_theme_applied,
                                                   kbd_source):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    screen = _open_screen(qtbot, kbd_source)
    try:
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier, "")
        assert screen.eventFilter(screen._grid_scroll, ev) is True
        assert screen.focus_slot == 1
        # A non-key event passes straight through.
        plain = QEvent(QEvent.Type.Show)
        assert screen.eventFilter(screen._grid_scroll, plain) is False
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Mouse annotation must keep working
# ---------------------------------------------------------------------------

def test_mouse_click_toggle_still_works(qtbot, qt_theme_applied, kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        screen._on_thumb_left(0)
        assert screen._page_paths[0][1] == 1
        screen._on_thumb_left(0)                 # same class again clears
        assert screen._page_paths[0][1] is None
        screen._on_thumb_right(0)
        assert screen._page_paths[0][1] == 2
        screen._on_thumb_left(0)                 # different class replaces
        assert screen._page_paths[0][1] == 1
        # Out-of-range slot is ignored rather than raising.
        screen._on_thumb_left(len(screen._page_paths) + 5)
    finally:
        _stop(screen)


def test_mouse_and_keyboard_share_one_write_path(qtbot, qt_theme_applied,
                                                  kbd_source):
    screen = _open_screen(qtbot, kbd_source)
    try:
        paths = [p for p, _ in screen._page_paths]
        screen._on_thumb_left(0)         # mouse -> class 1
        screen._set_focus_slot(1)
        screen.handle_key("4")           # keyboard -> class 4
        screen._flush_pending()
        _wait_saved(screen, qtbot)
    finally:
        _stop(screen)
    stored = _db_labels(kbd_source)
    assert stored[paths[0]] == 1
    assert stored[paths[1]] == 4
