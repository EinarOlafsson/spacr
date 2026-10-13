"""Annotate: the answers it gives when the database will not cooperate.

Everything pinned here is on the write path or the path that decides what
gets written, and every one of them is a case where the screen has to say
what went wrong rather than quietly label the wrong population:

* the auto-annotator's measurement preview -- a rule it can run, and a rule
  it cannot read, which must be *reported* and never silently dropped;
* the value hint beside the metadata picker, which is a convenience and must
  not take the dialog down with it when the file it reads is not a database;
* a bulk write whose annotation column cannot be created, which has to stop
  before it queues anything;
* two gestures at the edge of the grid -- a direction the grid does not know,
  and a rubber band released after the widget drawing it has gone.

The last section proves the branches in this module that cannot be taken.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PIL import Image                                             # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt                    # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap                 # noqa: E402

from spacr.qt import annotate_engine as engine                    # noqa: E402
from spacr.qt.screens import annotate as annotate_mod             # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# A project with real metadata and a measurement table
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path: Path) -> Path:
    """png_list joined to a ``cell`` table, the shape the join needs."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)

    rng = np.random.default_rng(4)
    png_rows, cell_rows = [], []
    for index in range(12):
        png = src / "data" / f"cell_{index:02d}.png"
        Image.fromarray(
            rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)).save(png)
        well = "c1" if index < 6 else "c2"
        prcfo = f"plate1_r1_{well}_f1_o{index}"
        png_rows.append((str(png), "plate1", f"r1{well}", "r1", well, "f1",
                         index, None, prcfo, index))
        cell_rows.append((prcfo, "plate1", "r1", well, "f1", index,
                          float(100 + index * 100), float(10 + index)))

    db = src / "measurements" / "measurements.db"
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            'CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, plateID TEXT,'
            ' wellID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT,'
            ' label INTEGER, annotate INTEGER, prcfo TEXT, cell_id INTEGER)')
        connection.executemany(
            'INSERT INTO "png_list" VALUES (?,?,?,?,?,?,?,?,?,?)', png_rows)
        connection.execute(
            'CREATE TABLE "cell" (prcfo TEXT PRIMARY KEY, plateID TEXT,'
            ' rowID TEXT, columnID TEXT, fieldID TEXT, object_label INTEGER,'
            ' cell_area REAL, nucleus_area REAL)')
        connection.executemany(
            'INSERT INTO "cell" VALUES (?,?,?,?,?,?,?,?)', cell_rows)
        connection.commit()
    finally:
        connection.close()
    return src


def _db(project: Path) -> str:
    return str(project / "measurements" / "measurements.db")


def _dialog(qtbot, db_path, src=""):
    settings = engine.AnnotateSettings(src=str(src), db_path=str(db_path),
                                       annotation_column="annotate")
    dlg = annotate_mod._AutoAnnotateDialog(settings)
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# The measurement preview
# ---------------------------------------------------------------------------

def test_a_rule_that_cannot_be_read_is_reported_and_not_dropped(
        qtbot, project, qt_theme_applied):
    """A typo in a rule must not quietly widen the population.

    The rules are ANDed, so a dropped line is a *larger* population than the
    user asked for -- objects labelled by a threshold nobody wrote. The
    dialog therefore refuses the whole preview and says why, and Apply goes
    back to disabled so the previous, approved count cannot be written under
    the new text.
    """
    dlg = _dialog(qtbot, _db(project), project)
    dlg._source.setCurrentIndex(1)                 # measurement thresholds
    assert dlg.source() == "measurement"

    dlg._rules.setPlainText("cell_area > 500")
    dlg._on_preview()

    matched = dlg.matched_paths()
    assert matched, "the readable rule found a population"
    assert len(matched) == 7                       # cell_area 600..1200
    assert "7 object(s) match" in dlg._preview_label.text()
    assert dlg._apply.isEnabled() is True

    dlg._rules.setPlainText("cell_area > 500\ncell_area more_than 500")
    dlg._on_preview()

    assert dlg._preview_label.text().startswith("Could not preview:")
    assert "more_than" in dlg._preview_label.text()
    assert dlg.matched_paths() == []
    assert dlg._apply.isEnabled() is False


# ---------------------------------------------------------------------------
# The metadata value hint
# ---------------------------------------------------------------------------

def test_the_value_hint_goes_quiet_when_the_file_is_not_a_database(
        qtbot, project, tmp_path, qt_theme_applied):
    """The hint reads the database; the dialog must not need it to.

    It fires on every change of the column picker, including while the
    dialog is being built. A user who pointed the screen at the wrong file
    would otherwise get a traceback out of a combo box signal instead of a
    dialog they can correct the path in.
    """
    working = _dialog(qtbot, _db(project), project)
    working._column.setCurrentText("columnID")
    assert "c1" in working._values.placeholderText()

    not_a_db = tmp_path / "notes.txt"
    not_a_db.write_text("this is not a database")
    broken = _dialog(qtbot, not_a_db, project)
    placeholder_before = broken._values.placeholderText()

    broken._column.setCurrentText("columnID")

    assert broken._values.placeholderText() == placeholder_before
    assert broken.source() == "metadata", "the dialog is still usable"


# ---------------------------------------------------------------------------
# A bulk write whose column cannot be made
# ---------------------------------------------------------------------------

def test_a_column_that_cannot_be_created_stops_the_bulk_write(
        qtbot, project, monkeypatch, qt_theme_applied):
    """Queueing a write into a column that does not exist loses the labels.

    The save worker would take the batch, fail on it row by row and report
    nothing, so the annotator would believe several thousand objects had
    been labelled. It has to fail here, in front of them, with the column
    named.
    """
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.db_path = _db(project)
    screen._settings.annotation_column = "auto_class"
    paths = [str(project / "data" / f"cell_{i:02d}.png") for i in range(3)]

    assert screen._apply_bulk_annotation(paths, 2) == 3
    assert screen._pending_updates == {path: 2 for path in paths}

    screen._pending_updates.clear()
    warned = []
    monkeypatch.setattr(annotate_mod.QMessageBox, "warning",
                        lambda *args, **kwargs: warned.append(args))

    # THE STUB MIRRORS THE REAL SIGNATURE, keyword-only `table` included.
    #
    # It used to be `_refuse(_db_path, _column)`. `ensure_annotation_column`
    # has since grown `*, table=DEFAULT_PNG_TABLE`, and the Annotate screen
    # passes it -- so the stub raised TypeError ("unexpected keyword argument
    # 'table'") instead of the OperationalError this test is about. The
    # assertions below still fired, on a message about the wrong exception,
    # and the test failed for a reason that had nothing to do with a readonly
    # database (instruction 345).
    #
    # A stub that accepts `**kwargs` would have hidden the drift instead of
    # failing on it. Naming the parameter is what keeps this test honest the
    # next time the signature moves.
    def _refuse(_db_path, _column, *, table=None):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(engine, "ensure_annotation_column", _refuse)

    assert screen._apply_bulk_annotation(paths, 2) == 0
    assert screen._pending_updates == {}, "nothing was queued"
    assert warned, "the annotator was told"
    assert "auto_class" in warned[0][2]
    assert "readonly database" in warned[0][2]


# ---------------------------------------------------------------------------
# Two gestures at the edge of the grid
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(qtbot, qt_theme_applied, project):
    """An Annotate screen on a pinned 3x3 grid with its first page decoded."""
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    scr._settings.grid_rows = 3
    scr._settings.grid_cols = 3
    scr._settings.image_size = (16, 16)
    scr._compute_grid_dims = lambda: None
    scr._rebuild_grid()
    scr._open_source(str(project))
    qtbot.waitUntil(lambda: len(scr._page_paths) >= 9, timeout=10000)
    yield scr
    if scr._worker is not None:
        scr._worker.stop(wait=True)


def test_a_direction_the_grid_does_not_know_moves_nothing(screen):
    """``_kbd_move`` takes a token, and only four of them mean a direction.

    Driven directly because ``handle_key`` only ever hands it one of the
    four; the guard is what keeps a fifth token -- a rebind, a new keymap
    entry -- from silently moving focus somewhere arbitrary instead of
    doing nothing.
    """
    scr = screen
    scr._set_focus_slot(0)

    assert scr._kbd_move("right") is True
    assert scr._focus_slot == 1, "a direction it knows moves one cell"

    assert scr._kbd_move("diagonal") is True
    assert scr._focus_slot == 1, "one it does not know moves nothing"


def test_a_band_released_after_its_widget_went_still_labels_the_selection(
        screen, monkeypatch):
    """The band is a drawn rectangle; the selection is the user's work.

    The rubber band is parented to the grid holder, so a rebuild between
    press and release takes it away. Hiding it is then impossible -- but the
    crops the user dragged across still have to be labelled, because that
    gesture is the only record of what they selected.
    """
    scr = screen
    asked = []
    monkeypatch.setattr(annotate_mod.AnnotateScreen, "_ask_class",
                        lambda self, count, what: asked.append(count) or 4)

    class _Event:
        def __init__(self, point):
            self._point = point

        def button(self):
            return Qt.LeftButton

        def position(self):
            return self._point

    assert scr._band_event(QEvent.MouseButtonPress,
                           _Event(QPointF(0.0, 0.0))) is True
    assert scr._band is not None

    scr._band.deleteLater()
    scr._band = None            # the grid it was parented to was rebuilt

    assert scr._band_event(QEvent.MouseButtonRelease,
                           _Event(QPointF(4000.0, 4000.0))) is True

    assert asked, "the selection still asked what to call itself"
    assert scr._band_origin is None
    labelled = [value for _path, value in scr._page_paths if value == 4]
    assert labelled, "the crops the band crossed were labelled"
    assert "as 4" in scr._kbd_hint.text()


def test_a_hover_the_cursor_never_left_is_kept_across_a_page_load(screen):
    """The white ring has to stay on the tile the next click will hit.

    A page load swaps the crops under widgets that never move, and Qt only
    re-sends Enter/Leave when the cursor crosses a boundary -- so after one
    the screen re-asks ``underMouse`` rather than trusting the hover it has.
    A cursor still resting on a tile keeps it; one that left without a Leave
    event loses it, which is the same call taking the other answer.
    """
    scr = screen
    thumb = scr._thumbs[2]
    thumb.setAttribute(Qt.WA_UnderMouse, True)
    scr._on_thumb_hover(2, True)
    assert scr.hover_slot == 2

    scr._revalidate_hover()

    assert scr.hover_slot == 2, "the cursor never left the tile"

    thumb.setAttribute(Qt.WA_UnderMouse, False)
    scr._revalidate_hover()
    assert scr.hover_slot is None, "a cursor that left loses the ring"


# ---------------------------------------------------------------------------
# Proved unreachable
# ---------------------------------------------------------------------------

def test_a_zoomed_crop_always_has_somewhere_to_draw(qtbot, qt_theme_applied):
    """Why ``if box.isEmpty()`` in ``_ZoomOverlay.paintEvent`` cannot be true.

    ``paintEvent`` has already returned at the line above unless ``_pixmap``
    is a non-null pixmap, and ``picture_rect`` returns an empty rect in
    exactly that same case. Otherwise its width is ``pm.width() * scale``
    with ``scale = min(box_w / pm.width(), box_h / pm.height())`` over
    ``box_w, box_h >= 1.0``, which is strictly positive for any pixmap Qt
    will hand over -- even in a widget with no size at all, which is the
    case that looks like it might produce one.
    """
    overlay = annotate_mod._ZoomOverlay()
    qtbot.addWidget(overlay)
    assert overlay.picture_rect().isEmpty(), "no pixmap, nothing to draw"

    pixmap = QPixmap(1, 1)
    pixmap.fill(QColor(255, 0, 0))
    overlay.resize(0, 0)
    overlay.show_pixmap(pixmap, 0)

    box = overlay.picture_rect()
    assert not box.isEmpty()
    assert box.width() > 0 and box.height() > 0

    # And the guard really is evaluated on a paint: at a usable size the
    # crop is drawn, so the empty-rect branch is the one that never fires.
    overlay.resize(64, 64)
    canvas = QImage(64, 64, QImage.Format.Format_ARGB32)
    canvas.fill(QColor(0, 255, 0))
    overlay.render(canvas)
    painted = {QColor(canvas.pixel(x, y)).name()
               for x in range(0, 64, 8) for y in range(0, 64, 8)}
    assert "#00ff00" not in painted, "the scrim covered the whole widget"
    assert QColor(canvas.pixel(32, 32)).name() == "#ff0000", (
        "the crop itself was drawn inside the rect")


def test_every_slot_the_grid_offers_can_be_annotated(screen):
    """Why ``if self._set_annotation(...)`` can never be false in
    ``_apply_to_slots`` or ``_toggle_annotation``.

    ``_set_annotation`` returns False only for a slot outside
    ``self._page_paths``. Both of those callers test ``_slot_is_valid``
    first, and that is ``0 <= slot < _slot_count()`` where ``_slot_count``
    is ``min(len(self._page_paths), len(self._thumbs))`` -- so a slot that
    passes it is inside ``_page_paths`` by construction and the write always
    succeeds.
    """
    scr = screen
    assert scr._slot_count() <= len(scr._page_paths)

    for slot in range(scr._slot_count()):
        assert scr._slot_is_valid(slot) is True
        assert scr._set_annotation(slot, 1) is True

    # The mouse path through the same guard: a valid slot always records,
    # so the branch that skips the console line is never taken.
    scr._toggle_annotation(0, 2)
    assert scr._current_value(0) == 2
    scr._toggle_annotation(0, 2)
    assert scr._current_value(0) is None, "the same class again clears"

    # The False the callers cannot reach: a slot past the end of the page.
    beyond = len(scr._page_paths)
    assert scr._slot_is_valid(beyond) is False
    assert scr._set_annotation(beyond, 1) is False
    assert scr._apply_to_slots([beyond], 5) == 0
