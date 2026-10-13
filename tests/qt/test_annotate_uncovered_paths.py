"""Annotate-screen paths a clean page of crops never reaches.

The page worker being told to give up, a loader whose signature cannot be
read, an outline cancelled mid-crop, the zoom overlay's guards, the folded
Agreement module, and the several small formatters that have to answer
something rather than raise.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                    # noqa: E402
from PySide6.QtGui import QPixmap                                # noqa: E402

from spacr.qt.annotate_engine import OutlineCancelled            # noqa: E402
from spacr.qt.screens import annotate as annotate_mod            # noqa: E402

pytestmark = pytest.mark.qt

ROWS, COLS = 2, 3
N_CROPS = 12


@pytest.fixture()
def crop_source(tmp_path: Path) -> Path:
    """A minimal experiment folder: ``measurements.db`` plus tiny PNGs."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "cell_png").mkdir(parents=True)
    rng = np.random.default_rng(11)
    paths: List[str] = []
    for i in range(N_CROPS):
        arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        path = src / "data" / "cell_png" / f"crop_{i:02d}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    return src


def _open_screen(qtbot, src: Path, rows: int = ROWS, cols: int = COLS):
    """An AnnotateScreen with a pinned grid, showing the first page."""
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


# ---------------------------------------------------------------------------
# The folded Agreement module
# ---------------------------------------------------------------------------

def test_the_folded_agreement_module_builds_the_whole_agreement_screen(
        qtbot, qapp):
    """The fold opens the real screen, not a summary of it."""
    from spacr.qt.screens.agreement import AgreementScreen

    widget = annotate_mod.FOLD_BUILDERS["agreement"](None)
    qtbot.addWidget(widget)

    assert isinstance(widget, AgreementScreen)


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

def test_a_letter_key_the_screen_does_not_bind_is_left_to_qt():
    """Only the vi-motion letters are claimed; the rest fall through."""
    assert annotate_mod.key_token(int(Qt.Key_A)) is None
    assert annotate_mod.key_token(int(Qt.Key_Z)) is None
    assert annotate_mod.key_token(int(Qt.Key_H)) == "left"


def test_a_key_code_that_cannot_be_turned_into_an_int_is_skipped(monkeypatch):
    """A binding whose enum will not convert costs that key, not the map."""
    class _Unusable:
        Key_Left = object()
        Key_Right = int(Qt.Key_Right)
        Key = None

    monkeypatch.setattr(annotate_mod, "Qt", _Unusable)

    table = annotate_mod._qt_code_tokens()

    assert table == {int(Qt.Key_Right): "right"}


# ---------------------------------------------------------------------------
# Filter fields
# ---------------------------------------------------------------------------

def test_a_filter_bound_that_is_not_a_number_reads_as_an_empty_field():
    """A corrupted bound leaves the field blank rather than showing junk."""
    assert annotate_mod._filter_text(None) == ""
    assert annotate_mod._filter_text(200.0) == "200"
    assert annotate_mod._filter_text("not a number") == ""
    assert annotate_mod._filter_text(object()) == ""


# ---------------------------------------------------------------------------
# The page worker
# ---------------------------------------------------------------------------

def _one_row(path: str):
    return (path, None)


class _OpaqueLoader:
    """A callable whose signature cannot be introspected, as a C one is not."""

    __signature__ = "this is not a Signature"

    def __init__(self):
        self.rows = []

    def __call__(self, row):
        self.rows.append(row)
        return (row, None)


def test_a_loader_whose_signature_cannot_be_read_is_called_with_the_row_alone(
        qapp):
    """An uninspectable loader is used, not refused."""
    import inspect

    loader = _OpaqueLoader()
    with pytest.raises(ValueError):
        inspect.signature(loader)

    worker = annotate_mod._PageLoadWorker(1, ["a", "b"], loader)
    assert worker._load_fn_stops is False
    emitted = []
    worker.done.connect(lambda gen, loaded: emitted.append((gen, loaded)))

    worker.run()

    assert loader.rows == ["a", "b"]
    assert emitted == [(1, [("a", None), ("b", None)])]


def test_a_loader_that_can_be_stopped_is_handed_the_stop_check(qapp):
    """The page loader is asked to give up between crops, not only after."""
    seen = []

    def _loader(row, should_stop=None):
        seen.append((row, should_stop is not None))
        return (row, None)

    worker = annotate_mod._PageLoadWorker(3, ["a", "b"], _loader)
    assert worker._load_fn_stops is True

    emitted = []
    worker.done.connect(lambda gen, loaded: emitted.append((gen, loaded)))
    worker.run()

    assert [row for row, _ in seen] == ["a", "b"]
    assert all(handed for _row, handed in seen)
    assert emitted == [(3, [("a", None), ("b", None)])]


def test_a_page_abandoned_mid_crop_emits_nothing_at_all(qapp):
    """An outline that was cancelled describes a page nobody is looking at."""
    def _loader(row, should_stop=None):
        raise OutlineCancelled("the screen moved on")

    worker = annotate_mod._PageLoadWorker(4, ["a"], _loader)
    emitted = []
    worker.done.connect(lambda *args: emitted.append(args))

    worker.run()

    assert emitted == []


class _AbandonedWhileDecoding(annotate_mod._PageLoadWorker):
    """A worker the screen gives up on while the last crop is decoding.

    ``requestInterruption`` only takes effect on a thread that is running, so
    the flag is modelled directly here and the body is driven in-thread.
    """

    abandoned = False

    def isInterruptionRequested(self):   # noqa: N802 - Qt name
        return self.abandoned


def test_a_worker_abandoned_during_the_last_crop_emits_nothing(qapp):
    """The interruption flag is read again after the decode, not only before."""
    decoded = []

    def _loader(row):
        decoded.append(row)
        worker.abandoned = True          # the screen moved off this page
        return (row, None)

    worker = _AbandonedWhileDecoding(5, ["a"], _loader)
    emitted = []
    worker.done.connect(lambda *args: emitted.append(args))

    worker.run()

    assert decoded == ["a"], "the crop was decoded before the page was dropped"
    assert emitted == [], "a page nobody is looking at is not handed back"


def test_a_worker_whose_c_half_is_gone_reads_as_stopped(qapp):
    """A destroyed wrapper means the page was abandoned, not an error."""
    from shiboken6 import Shiboken

    worker = annotate_mod._PageLoadWorker(6, [], lambda row: row)
    Shiboken.delete(worker)

    assert worker._stop_requested() is True


# ---------------------------------------------------------------------------
# Loading one crop
# ---------------------------------------------------------------------------

def test_a_cancelled_outline_propagates_out_of_the_crop_loader(
        qapp, monkeypatch, crop_source):
    """The cancellation is re-raised so the worker can unwind the page."""
    from spacr.qt.annotate_engine import AnnotateSettings

    def _cancel(**_kwargs):
        raise OutlineCancelled("the screen has gone")

    monkeypatch.setattr(annotate_mod, "outline_image", _cancel)
    settings = AnnotateSettings()
    settings.db_path = str(crop_source / "measurements" / "measurements.db")
    settings.image_size = (24, 24)
    settings.outline = (1,)
    path = str(crop_source / "data" / "cell_png" / "crop_00.png")

    with pytest.raises(OutlineCancelled):
        annotate_mod._load_thumb_image_worker((path, None), None, settings)


def test_an_outline_that_fails_for_another_reason_still_yields_a_crop(
        qapp, monkeypatch, crop_source):
    """Only cancellation unwinds; any other failure loses the outline alone."""
    from spacr.qt.annotate_engine import AnnotateSettings

    def _explode(**_kwargs):
        raise ValueError("that is not an image")

    monkeypatch.setattr(annotate_mod, "outline_image", _explode)
    settings = AnnotateSettings()
    settings.db_path = str(crop_source / "measurements" / "measurements.db")
    settings.image_size = (24, 24)
    settings.outline = (1,)
    path = str(crop_source / "data" / "cell_png" / "crop_00.png")

    image, annotation = annotate_mod._load_thumb_image_worker(
        (path, None), None, settings)

    assert image.size == (24, 24)
    assert annotation is None


# ---------------------------------------------------------------------------
# The zoom overlay
# ---------------------------------------------------------------------------

def test_an_overlay_with_no_picture_claims_no_rectangle_and_paints_nothing(
        qtbot, qapp):
    """A cleared overlay has nothing to place and nothing to draw."""
    overlay = annotate_mod._ZoomOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(200, 120)

    assert overlay.picture_rect().isEmpty()

    # A paint over an empty overlay must return before it touches a painter.
    overlay.render(QPixmap(overlay.size()))


def test_an_overlay_with_a_picture_centres_it_inside_the_margin(qtbot, qapp):
    """The margin is real space, so a click outside the crop has somewhere
    to land."""
    overlay = annotate_mod._ZoomOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(200, 200)
    pixmap = QPixmap(50, 25)
    pixmap.fill(Qt.red)
    overlay.show_pixmap(pixmap, 2)

    box = overlay.picture_rect()

    assert not box.isEmpty()
    assert box.left() >= annotate_mod.SPACING["md"]
    assert box.right() <= 200 - annotate_mod.SPACING["md"]
    assert overlay.slot == 2
    overlay.render(QPixmap(overlay.size()))


# ---------------------------------------------------------------------------
# The settings dialog
# ---------------------------------------------------------------------------

def test_a_chosen_primary_set_is_not_overwritten_by_the_saved_preference(
        qtbot, qapp, monkeypatch):
    """A session that already chose CMY keeps it when the form reopens."""
    def _never():
        raise AssertionError(
            "the saved preference must not be read over an explicit choice")

    monkeypatch.setattr("spacr.qt.preferences.image_display_primaries", _never)
    from spacr.qt.annotate_engine import AnnotateSettings

    settings = AnnotateSettings()
    settings.display_primaries = "cmy"

    dialog = annotate_mod._SettingsDialog(settings)
    qtbot.addWidget(dialog)

    assert dialog._display_primaries.currentData() == "cmy"


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------

def test_the_console_switch_caption_is_recomposed_on_a_language_change(
        qtbot, crop_source):
    """The arrow and the word are one composed caption, so it re-renders."""
    screen = _open_screen(qtbot, crop_source)
    try:
        screen._console_switch.setChecked(True)
        open_text = screen._console_switch.text()

        screen.retranslate_dynamic_content("en")

        assert screen._console_switch.text() == open_text
        assert "Console" in screen._console_switch.text()
    finally:
        _stop(screen)


def test_turning_ai_on_while_the_console_is_already_open_leaves_it_open(
        qtbot, crop_source, monkeypatch):
    """The console is revealed, not toggled, when AI is switched on."""
    from spacr.qt import ai as ai_module

    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    screen = _open_screen(qtbot, crop_source)
    try:
        screen._console_switch.setChecked(True)

        screen._on_ai_switch(True)

        assert screen._console_switch.isChecked()
        assert screen._console._ai_active is True
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# The compatibility crop loader
# ---------------------------------------------------------------------------

def test_the_crop_loader_uses_the_source_and_settings_it_is_handed(
        qtbot, crop_source):
    """Explicit arguments are used as given; nothing is taken off the screen."""
    from copy import deepcopy

    screen = _open_screen(qtbot, crop_source)
    try:
        handed = deepcopy(screen._settings)
        handed.image_size = (8, 8)
        path = str(crop_source / "data" / "cell_png" / "crop_01.png")

        image, _annotation = screen._load_thumb_image(
            (path, None), src=screen._crop_source(), settings=handed)

        assert image.size == (8, 8)
        assert screen._settings.image_size == (24, 24), (
            "the screen's own settings were not consulted or changed")
    finally:
        _stop(screen)


def test_the_crop_loader_falls_back_to_the_screens_own_source_and_settings(
        qtbot, crop_source):
    """Called with nothing, it reads the page the screen is showing."""
    screen = _open_screen(qtbot, crop_source)
    try:
        path = str(crop_source / "data" / "cell_png" / "crop_01.png")

        image, _annotation = screen._load_thumb_image((path, None))

        assert image.size == tuple(screen._settings.image_size)
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Keyboard movement and undo
# ---------------------------------------------------------------------------

def test_moving_down_from_the_bottom_row_stays_where_it_is(qtbot, crop_source):
    """The grid does not wrap, so the focus ring does not move."""
    screen = _open_screen(qtbot, crop_source)
    try:
        bottom = (ROWS - 1) * COLS + 1
        screen._set_focus_slot(bottom)

        assert screen.handle_key("down") is True

        assert screen._focus_slot == bottom
    finally:
        _stop(screen)


def test_an_undo_entry_for_a_crop_that_has_gone_is_skipped(qtbot, crop_source):
    """Writing a stale entry back would label whatever is in the slot now."""
    screen = _open_screen(qtbot, crop_source)
    try:
        screen._set_focus_slot(0)
        assert screen.handle_key("1") is True
        assert screen._page_paths[0][1] == 1
        # The page moved on: slot 0 now holds a different crop, so the entry
        # recorded against the old path no longer applies.
        screen._undo_stack.append(
            (0, "/gone/crop_that_left_the_page.png", 7))
        screen._undo_stack.append((0, "/also/gone.png", 8))

        assert screen.handle_key("u") is True

        assert screen._page_paths[0][1] is None, (
            "the stale entries were skipped and the real one was undone")
        assert len(screen._undo_stack) == 0
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# The zoom overlay on the screen
# ---------------------------------------------------------------------------

def test_zooming_a_slot_that_is_not_on_the_page_does_nothing(
        qtbot, crop_source):
    """There is no crop past the end of the page to blow up."""
    screen = _open_screen(qtbot, crop_source)
    try:
        screen._on_thumb_shift(ROWS * COLS + 5)

        assert screen._zoom_is_open() is False
        assert screen._zoom_overlay.isVisible() is False
    finally:
        _stop(screen)


def test_an_overlay_whose_viewport_has_gone_reads_as_closed_and_is_not_fitted(
        qtbot, crop_source):
    """A destroyed overlay must not raise out of a resize or a key press."""
    from shiboken6 import Shiboken

    screen = _open_screen(qtbot, crop_source)
    try:
        Shiboken.delete(screen._zoom_overlay)

        assert screen._zoom_is_open() is False
        screen._fit_zoom_overlay()          # must not raise
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def test_a_report_the_user_closed_is_replaced_rather_than_asked(
        qtbot, crop_source):
    """A closed report has lost its C++ half; a second press opens a new one."""
    from shiboken6 import Shiboken

    screen = _open_screen(qtbot, crop_source)
    try:
        first = screen._show_report("Coverage", "one")
        qtbot.addWidget(first)
        Shiboken.delete(first)

        second = screen._show_report("Coverage", "two")
        qtbot.addWidget(second)

        assert second is not None
        assert list(screen._reports) == ["Coverage"]
        assert screen._reports["Coverage"] is second
    finally:
        _stop(screen)


def test_a_second_press_rewrites_the_report_that_is_already_up(
        qtbot, crop_source):
    """One window per title, rewritten rather than stacked."""
    screen = _open_screen(qtbot, crop_source)
    try:
        first = screen._show_report("Coverage", "one")
        qtbot.addWidget(first)

        second = screen._show_report("Coverage", "two")

        assert second is first
        assert first._view.toPlainText() == "two"
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# The active-learning strip
# ---------------------------------------------------------------------------

def test_a_round_with_no_per_class_scores_reports_what_it_does_have(
        qtbot, crop_source):
    """A binary report without per-class accuracies still shows the round."""
    from spacr.active_learning import RoundResult

    screen = _open_screen(qtbot, crop_source)
    try:
        screen._last_round = RoundResult(
            round_index=0, n_labels=40, report={"accuracy": 0.812, "n": 40})
        screen._round_index = 1
        screen._stop_verdict = None

        screen._refresh_al_label()

        text = screen._al_label.text()
        assert "Round 1" in text
        assert "40 labels" in text
        assert "held-out 0.812" in text
        assert "worst class" not in text, "there were no per-class scores"
        assert "no model fitted" not in text, "a round was fitted"
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# The fold strip
# ---------------------------------------------------------------------------

def test_a_fold_strip_that_cannot_be_built_costs_the_buttons_not_the_screen(
        qtbot, monkeypatch, crop_source):
    """A screen without its fold buttons is smaller; without the grid it is
    nothing."""
    def _explode(*_args, **_kwargs):
        raise RuntimeError("the fold strip could not be laid out")

    monkeypatch.setattr(annotate_mod, "FoldStrip", _explode)

    screen = _open_screen(qtbot, crop_source)
    try:
        assert getattr(screen, "_fold_strip", None) is None
        assert len(screen._page_paths) == ROWS * COLS, (
            "the crops still loaded"
        )
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# The grid's event filter
# ---------------------------------------------------------------------------

def _zoom_first_crop(screen, qtbot):
    """Blow up slot 0, once the screen is on screen and its crop decoded."""
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.waitUntil(lambda: screen._thumb_pixmaps[0] is not None, timeout=5000)
    screen._on_thumb_shift(0)
    assert screen._zoom_is_open()


def test_escape_folds_a_zoomed_crop_back_before_the_grid_sees_it(
        qtbot, crop_source):
    """A zoomed crop owns Escape; the grid never reads it as 'clear'."""
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    screen = _open_screen(qtbot, crop_source)
    try:
        _zoom_first_crop(screen, qtbot)
        press = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)

        consumed = screen.eventFilter(screen._grid_holder, press)

        assert consumed is True
        assert screen._zoom_is_open() is False
        assert screen._zoom_overlay.slot == -1
    finally:
        _stop(screen)


def test_resizing_the_viewport_keeps_the_zoomed_crop_filling_it(
        qtbot, crop_source):
    """The overlay follows the container it fills, or the margin moves."""
    from PySide6.QtCore import QEvent, QSize
    from PySide6.QtGui import QResizeEvent

    screen = _open_screen(qtbot, crop_source)
    try:
        _zoom_first_crop(screen, qtbot)
        viewport = screen._grid_scroll.viewport()
        viewport.resize(320, 240)
        resize = QResizeEvent(QSize(320, 240), QSize(100, 100))

        screen.eventFilter(viewport, resize)

        assert screen._zoom_overlay.geometry() == viewport.rect()
        assert resize.type() == QEvent.Resize
    finally:
        _stop(screen)


def test_a_second_selection_band_reuses_the_one_already_made(
        qtbot, crop_source):
    """One rubber band per screen, moved rather than remade."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    screen = _open_screen(qtbot, crop_source)
    screen.show()
    qtbot.waitExposed(screen)
    try:
        def _press(x, y):
            point = QPointF(x, y)
            return QMouseEvent(QEvent.MouseButtonPress, point, point, point,
                               Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
                               Qt.MouseEventNotSynthesized)

        def _release(x, y):
            point = QPointF(x, y)
            return QMouseEvent(QEvent.MouseButtonRelease, point, point, point,
                               Qt.LeftButton, Qt.NoButton, Qt.NoModifier,
                               Qt.MouseEventNotSynthesized)

        assert screen.eventFilter(screen._grid_holder, _press(2, 2)) is True
        band = screen._band
        assert band is not None
        assert screen.eventFilter(screen._grid_holder, _release(3, 3)) is True

        assert screen.eventFilter(screen._grid_holder, _press(4, 4)) is True

        assert screen._band is band, "a second band would leak the first"
        assert screen._band.isVisible()
        screen.eventFilter(screen._grid_holder, _release(5, 5))
    finally:
        _stop(screen)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def test_closing_the_screen_tolerates_a_report_the_user_already_closed(
        qtbot, crop_source):
    """A report husk must not raise out of closeEvent."""
    from shiboken6 import Shiboken

    screen = _open_screen(qtbot, crop_source)
    report = screen._show_report("Coverage", "one")
    qtbot.addWidget(report)
    Shiboken.delete(report)

    screen.close()

    assert screen._reports == {}


def test_turning_ai_on_keeps_the_provider_that_is_already_chosen(
        qtbot, crop_source, monkeypatch):
    """A provider is picked once; switching AI on again does not re-pick."""
    from spacr.qt import ai as ai_module

    screen = _open_screen(qtbot, crop_source)
    try:
        screen._console._current_provider_name = "already-chosen"

        def _never():
            raise AssertionError("a chosen provider must not be replaced")

        monkeypatch.setattr(ai_module, "configured_providers", _never)
        screen._on_ai_switch(True)

        assert screen._console._current_provider_name == "already-chosen"
        assert screen._console_switch.isChecked()
    finally:
        _stop(screen)


def test_a_resize_of_something_other_than_the_viewport_leaves_the_zoom_alone(
        qtbot, crop_source):
    """Only the container the overlay fills makes it refit.

    The overlay is nudged off the viewport rect first, so a refit would be
    visible: it snaps the overlay back to that rect. Watching every resize
    instead of the viewport's own would move the picture -- and the margin a
    click has to land in to fold it back -- while the user resized something
    else entirely.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    screen = _open_screen(qtbot, crop_source)
    try:
        _zoom_first_crop(screen, qtbot)
        viewport = screen._grid_scroll.viewport()
        nudged = viewport.rect().adjusted(0, 0, -17, -13)
        screen._zoom_overlay.setGeometry(nudged)
        before = screen._zoom_overlay.geometry()
        assert before != viewport.rect(), "a refit would be visible"

        # The grid holder is watched too, but it is not the container the
        # overlay fills, so its resize must not move the picture.
        screen.eventFilter(screen._grid_holder,
                           QResizeEvent(QSize(500, 400), QSize(100, 100)))

        assert screen._zoom_overlay.geometry() == before
        assert screen._zoom_is_open()

        # The viewport's own resize is the one that does refit it.
        screen.eventFilter(viewport,
                           QResizeEvent(viewport.size(), QSize(100, 100)))

        assert screen._zoom_overlay.geometry() == viewport.rect()
    finally:
        _stop(screen)


def test_a_finished_round_does_not_repaginate_a_pinned_subset(
        qtbot, crop_source):
    """A routed population keeps its page when a retrain re-ranks the queue."""
    from spacr.active_learning import RoundResult
    from spacr.selection import ObjectRequest

    screen = _open_screen(qtbot, crop_source)
    try:
        screen._object_request = ObjectRequest(
            keys=[("plate1", "c1", "f1", 1)], reason="clicked in the UMAP",
            source="umap")
        screen._offset = 6
        result = RoundResult(round_index=0, n_labels=12,
                             report={"accuracy": 0.75, "n": 12})

        screen._on_retrain_done(result)

        assert screen._offset == 6, (
            "a pinned subset is not re-paginated under the heading")
        assert screen._last_round is result
        assert "Round 0" in screen._status_label.text()
    finally:
        _stop(screen)
