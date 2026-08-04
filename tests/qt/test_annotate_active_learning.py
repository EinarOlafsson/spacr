"""Annotate's half of the closed loop, and its object-routing slot.

Two things are pinned here that nothing else can pin:

* **``open_object_request`` shows exactly the requested crops, in the
  requested order, and keeps showing them.** The subset has to survive a
  queue rebuild — a routed request that got quietly replaced by "everything"
  under the same "12 objects · predicted infected" heading is worse than one
  that failed.
* **The loop actually closes inside the screen.** A retrain is offered, runs
  off the GUI thread, updates the round counter and the strip, and re-ranks
  the page — and every label that is saved carries the round it was made in.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr.selection import ObjectRequest


TOTAL_CROPS = 12


@pytest.fixture
def al_source(tmp_path: Path) -> Path:
    """A synthetic experiment with real PNGs and a metadata-carrying png_list."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    images = src / "data" / "images"
    images.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(TOTAL_CROPS):
        array = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        path = images / f"cell_{i:02d}.png"
        Image.fromarray(array).save(path)
        column = f"c{i // 4 + 1}"          # three wells of four crops
        rows.append((str(path), "p1", "r1", column, "f1",
                     f"p1_r1_{column}_f1_o{i}", f"o{i}", None))

    db = src / "measurements" / "measurements.db"
    con = sqlite3.connect(db)
    con.execute(
        'CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, plateID TEXT, '
        'rowID TEXT, columnID TEXT, fieldID TEXT, prcfo TEXT, cell_id TEXT, '
        'annotate INTEGER)')
    con.executemany('INSERT INTO "png_list" VALUES (?,?,?,?,?,?,?,?)', rows)
    con.commit()
    con.close()
    return src


@pytest.fixture
def screen(qtbot, qt_theme_applied, al_source: Path):
    from spacr.qt.screens.annotate import AnnotateScreen
    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.grid_rows = 3
    widget._settings.grid_cols = 3
    widget._settings.image_size = (32, 32)
    widget._compute_grid_dims = lambda: None
    widget._rebuild_grid()
    widget._open_source(str(al_source))
    qtbot.waitUntil(lambda: len(widget._page_paths) > 0, timeout=5000)
    yield widget
    if widget._worker is not None:
        widget._worker.stop(wait=True)
    widget.close()


def _key(index: int, column: str) -> str:
    """The OBJECT_KEY_COLUMNS key for crop ``index`` in ``column``."""
    return f"p1_r1_{column}_f1_{index}"


def _settle(widget, timeout_s: float = 20.0) -> None:
    """Pump until the screen's population count has delivered.

    Counting the population is database work -- with a threshold filter it
    joins every measurement table through ``spacr.io._read_and_join_tables``,
    which measured 4.8 s of frozen window on a 60 000-object database -- so it
    runs on a worker thread and lands on a later turn of the event loop.
    """
    import time
    from PySide6.QtWidgets import QApplication
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        if not widget.is_busy() and widget.active_jobs() == 0:
            return
        QApplication.processEvents()
        time.sleep(0.005)
    raise AssertionError("the population count never finished")


# ---------------------------------------------------------------------------
# open_object_request
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_as_the_object_opener(screen):
    from spacr.qt.linked_selection import has_object_opener, linked_selection
    assert has_object_opener("annotate")
    assert linked_selection()._openers["annotate"].__self__ is screen


def test_open_object_request_shows_exactly_the_requested_keys(screen, qtbot):
    """Three crops out of twelve, in the caller's order, and nothing else."""
    wanted = [_key(9, "c3"), _key(2, "c1"), _key(5, "c2")]
    request = ObjectRequest(keys=wanted, reason="predicted 1, annotated 2",
                            source="classifier_evaluation")

    result = screen.open_object_request(request)
    assert result is screen

    shown = [os.path.basename(p) for p, _ in screen._page_paths]
    assert shown == ["cell_09.png", "cell_02.png", "cell_05.png"], \
        "the caller's order is the answer; table order is not"
    assert screen._total == 3
    assert screen._offset == 0
    # the header says why, or three crops read as the whole screen
    assert "predicted 1, annotated 2" in screen._page_label.text()
    assert "3 objects" in screen._page_label.text()
    qtbot.waitUntil(lambda: screen._raw_thumb_images[0] is not None,
                    timeout=5000)


def test_a_routed_subset_survives_a_queue_rebuild(screen):
    """The population must not be swapped out from under the heading."""
    request = ObjectRequest(keys=[_key(1, "c1"), _key(4, "c2")],
                            reason="clicked in the UMAP", source="umap")
    screen.open_object_request(request)
    assert screen._total == 2

    screen._refresh_total()
    assert screen._total == 2, "a rebuild must not un-pin a routed subset"
    assert [os.path.basename(p) for p, _ in screen._filtered_rows] == \
        ["cell_01.png", "cell_04.png"]

    screen.clear_object_request()
    assert screen._object_request is None
    # Unpinning the subset recounts the whole population, and counting is now
    # database work on a worker thread — see AnnotateScreen._refresh_total.
    _settle(screen)
    assert screen._total == TOTAL_CROPS


def test_open_object_request_reports_keys_this_database_does_not_have(screen):
    request = ObjectRequest(keys=[_key(0, "c1"), "p9_r9_c9_f9_999"],
                            reason="worst errors", source="model_compare")
    screen.open_object_request(request)
    assert screen._total == 1
    assert "1 of them are not in this database" in screen._page_label.text()


def test_an_empty_request_is_a_real_answer_not_an_exception(screen):
    request = ObjectRequest(keys=[], reason="no errors in this cell",
                            source="classifier_evaluation")
    screen.open_object_request(request)
    assert screen._total == 0
    assert "0 objects" in screen._page_label.text()
    assert "no errors in this cell" in screen._page_label.text()


def test_the_request_routes_through_open_objects_without_naming_annotate(
        screen):
    """A caller uses the module function; it never imports this screen."""
    from spacr.qt.linked_selection import open_objects
    frame = pd.DataFrame({
        "plateID": ["p1", "p1"], "rowID": ["r1", "r1"],
        "columnID": ["c2", "c1"], "fieldID": ["f1", "f1"],
        "object_label": [6, 3],
    })
    result = open_objects(frame, reason="lassoed in the UMAP", source="umap")
    assert result is screen
    assert [os.path.basename(p) for p, _ in screen._page_paths] == \
        ["cell_06.png", "cell_03.png"]


def test_closing_withdraws_the_opener(qtbot, qt_theme_applied, al_source):
    from spacr.qt.linked_selection import has_object_opener
    from spacr.qt.screens.annotate import AnnotateScreen
    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    assert has_object_opener("annotate")
    widget.close()
    assert not has_object_opener("annotate")


def test_a_second_screen_keeps_the_registration_when_the_first_closes(
        qtbot, qt_theme_applied):
    from spacr.qt.linked_selection import has_object_opener, linked_selection
    from spacr.qt.screens.annotate import AnnotateScreen
    first = AnnotateScreen()
    qtbot.addWidget(first)
    second = AnnotateScreen()
    qtbot.addWidget(second)
    first.close()
    assert has_object_opener("annotate")
    assert linked_selection()._openers["annotate"].__self__ is second
    second.close()


# ---------------------------------------------------------------------------
# The loop inside the screen
# ---------------------------------------------------------------------------

def test_the_screen_offers_the_loop_controls(screen):
    assert screen._btn_retrain.isEnabled()
    assert screen._btn_coverage.text() == "Coverage"
    assert screen._btn_curve.text() == "Rounds"
    # the strip is visible as soon as a source is open, and says what is
    # missing rather than nothing
    assert screen._al_label.isVisibleTo(screen)
    assert "Round 0" in screen._al_label.text()
    assert "press Retrain" in screen._al_label.text()


def test_saved_labels_carry_the_round_they_were_made_in(screen):
    import spacr.active_learning as al

    screen._toggle_annotation(0, 1)
    screen._toggle_annotation(1, 2)
    screen._flush_pending()
    screen._worker.stop(wait=True)

    rounds = al.label_rounds(screen._settings.db_path, "annotate")
    assert len(rounds) == 2
    assert set(rounds["round"]) == {0}
    assert set(rounds["source"]) == {"manual"}

    # and the source follows how the crops were being served
    screen._worker = None
    screen._settings.queue_by_uncertainty = True
    assert screen._label_source() == "queue"
    screen._settings.queue_by_uncertainty = False
    screen._object_request = object()
    assert screen._label_source() == "object_request"


def test_provenance_failure_never_costs_the_labels(screen, monkeypatch):
    """A broken bookkeeping write must not stop somebody annotating."""
    import spacr.active_learning as al

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(al, "record_labels", boom)
    screen._toggle_annotation(0, 1)
    screen._flush_pending()               # must not raise
    screen._worker.stop(wait=True)

    with sqlite3.connect(screen._settings.db_path) as con:
        saved = con.execute(
            "SELECT COUNT(*) FROM png_list WHERE annotate IS NOT NULL"
        ).fetchone()[0]
    assert saved == 1, "the label itself still reached the database"


def test_retrain_runs_off_the_gui_thread_and_updates_the_strip(screen, qtbot,
                                                               monkeypatch):
    """The whole in-screen round: fit, record, re-rank, report."""
    import spacr.active_learning as al

    # Features the screen cannot read from this synthetic database — inject
    # them, exactly as a real source would supply them from the measurement
    # tables.
    rng = np.random.default_rng(1)
    paths = sorted(
        str(p) for p in (Path(screen._settings.src) / "data" / "images")
        .glob("*.png"))
    features = pd.DataFrame(
        {"png_path": paths,
         "signal": [(1.0 if i % 2 == 0 else 2.0) + rng.normal(0, 0.1)
                    for i in range(len(paths))],
         "noise": rng.normal(0, 1, size=len(paths))}).set_index("png_path")
    monkeypatch.setattr(al, "round_features", lambda *a, **k: features)

    # label every crop so a round can actually be fitted
    with sqlite3.connect(screen._settings.db_path) as con:
        for i, path in enumerate(paths):
            con.execute("UPDATE png_list SET annotate=? WHERE png_path=?",
                        (1 if i % 2 == 0 else 2, path))

    gui_thread = screen.thread()
    ran_on = {}
    original = al.retrain_round

    def spy(*args, **kwargs):
        from PySide6.QtCore import QThread
        ran_on["thread"] = QThread.currentThread()
        return original(*args, **kwargs)

    monkeypatch.setattr(al, "retrain_round", spy)

    screen._on_retrain()
    assert screen._btn_retrain.isEnabled() is False, "no double-fire"
    qtbot.waitUntil(lambda: screen._retrain_worker is None, timeout=30000)

    assert ran_on["thread"] is not gui_thread, \
        "a fit on the GUI thread freezes the grid mid-annotation"
    assert screen._btn_retrain.isEnabled() is True
    result = screen._last_round
    assert result is not None
    assert result.round_index == 0
    assert result.n_labels == TOTAL_CROPS
    assert result.score_columns == ["al_prob_0", "al_prob_1"]
    assert screen._round_index == 1, "the next labels belong to round 1"

    # the strip now carries the round's numbers and the verdict
    text = screen._al_label.text()
    assert "Round 1" in text
    assert "held-out" in text
    assert "worst class" in text

    # the round is on the curve, with a grouped split rule recorded
    curve = al.learning_curve(screen._settings.db_path, "annotate")
    assert list(curve["round"]) == [0]
    assert curve["split_rule"].iloc[0]
    assert curve["n_holdout"].iloc[0] > 0

    # and the database now carries the fresh per-class scores the queue
    # re-ranks on
    with sqlite3.connect(screen._settings.db_path) as con:
        columns = {r[1] for r in con.execute("PRAGMA table_info(png_list)")}
    assert {"al_prob_0", "al_prob_1"} <= columns


def test_retrain_failure_is_reported_rather_than_swallowed(screen, qtbot,
                                                           monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args, **kw: warned.append(args)))

    # nothing is annotated, so the round refuses to fit
    screen._on_retrain()
    qtbot.waitUntil(lambda: screen._retrain_worker is None, timeout=30000)

    assert "Retrain failed" in screen._status_label.text()
    assert screen._btn_retrain.isEnabled() is True, "the button comes back"
    assert screen._last_round is None
    assert len(warned) == 1, "a failed round says so on screen"
    title, body = warned[0][1], warned[0][2]
    assert title == "Retrain failed"
    assert "annotations are untouched" in body
    # and the reason is the module's own, not a generic apology
    assert "labels" in body or "class" in body


def test_retrain_needs_a_source(qtbot, qt_theme_applied, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from spacr.qt.screens.annotate import AnnotateScreen
    asked = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: asked.append(a[1])))
    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._on_retrain()
    widget._on_coverage()
    widget._on_learning_curve()
    assert asked == ["Open a source first"] * 3
    assert widget._retrain_worker is None
    widget.close()


def test_coverage_and_curve_dialogs_render_the_reports(screen, monkeypatch):
    from spacr.qt.screens import annotate as annotate_module
    shown = {}

    class _Dialog:
        def __init__(self, title, body, parent=None):
            shown[title] = body

        def exec(self):
            return 0

    monkeypatch.setattr(annotate_module, "_TextReportDialog", _Dialog)
    screen._toggle_annotation(0, 1)
    screen._flush_pending()
    screen._worker.stop(wait=True)

    screen._on_coverage()
    assert "Annotation coverage" in shown
    assert "Per class" in shown["Annotation coverage"]

    screen._on_learning_curve()
    assert "Active-learning rounds" in shown
    assert "No round recorded yet" in shown["Active-learning rounds"]
