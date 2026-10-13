"""A Suggest round on a small set finishes in seconds, counts its steps, stops on Cancel.

The tutorial recording of 2026-09-30 labelled ten crops of class 1 and ten
of class 2 on one page and pressed Suggest. Every class-1 crop came from
one well, so the round's well-grouped held-out split was impossible, the
fit raised, and the failure was shown with a blocking ``QMessageBox.warning``
from the worker's failure slot. The recorder, pumping events while it
waited for the run to end, sat inside that box's nested event loop for
four hours.

These tests hold the three repairs: the round still suggests when the
labels cannot be split by well (its check falls back to a random split and
says so), the failure box no longer blocks, and a run says which of its
five steps it is on and stops at the next one when Cancel is pressed.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.suggest import SUGGESTION_OFFSET  # noqa: E402

#: Seconds a whole round on the tiny set may take. The real round is well
#: under five seconds on a CPU; the bound only has to separate "finishes"
#: from "hangs".
_BOUND_S = 60.0


def _one_page_of_labels(tmp_path: Path):
    """Sixty crops in two wells, labelled the way the recording labelled.

    Class 1: ten crops, all in well r1c1. Class 2: six in r1c1 and four in
    r1c2. The other forty are open. Features carry the class in ``signal``.
    """
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, feats, labels = [], [], {}
    for i in range(60):
        column = "c1" if i < 30 else "c2"
        path = f"/crops/cell_{i:04d}.png"
        latent = 1 if i % 2 == 0 else 2
        rows.append((path, "p1", "r1", column, "f1", None))
        feats.append({"png_path": path,
                      "signal": latent + rng.normal(0, 0.25),
                      "noise": rng.normal(0, 1.0)})
    for i in range(0, 20, 2):
        labels[rows[i][0]] = 1
    for i in (1, 3, 5, 7, 9, 11, 31, 33, 35, 37):
        labels[rows[i][0]] = 2
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE png_list (png_path TEXT, plateID TEXT, '
                     'rowID TEXT, columnID TEXT, fieldID TEXT, '
                     'annotate INTEGER)')
        conn.executemany('INSERT INTO png_list VALUES (?,?,?,?,?,?)', rows)
        conn.executemany('UPDATE png_list SET annotate=? WHERE png_path=?',
                         [(v, p) for p, v in labels.items()])
    features = pd.DataFrame(feats).set_index("png_path")
    return db, features


def _suggested(db: Path) -> int:
    """How many crops hold a suggestion."""
    with sqlite3.connect(db) as conn:
        return conn.execute(
            'SELECT count(*) FROM png_list WHERE annotate > ?',
            (SUGGESTION_OFFSET,)).fetchone()[0]


def test_the_labels_really_cannot_be_split_by_well(tmp_path):
    """The set reproduces the recording: the grouped check refuses it."""
    import spacr.active_learning as al
    from spacr.classifier_evaluation import _GroupedSplitImpossible

    db, features = _one_page_of_labels(tmp_path)
    with pytest.raises(_GroupedSplitImpossible):
        al.retrain_round(str(db), "annotate", features=features,
                         model_type="gradient_boosting",
                         balance="downsample", round_index=1,
                         save_model=False, write_card=False)
    assert _suggested(db) == 0


def test_a_round_on_one_page_of_labels_finishes_in_seconds(tmp_path):
    """The worker suggests, within the bound, and names each step."""
    from spacr.qt.screens import annotate as mod

    db, features = _one_page_of_labels(tmp_path)
    worker = mod._SuggestWorker(
        str(db), "annotate",
        {"features": features, "round_index": 1,
         "model_type": "gradient_boosting", "balance": "downsample",
         "model_dir": str(tmp_path / "models")})
    done, failed, steps, relaxed = [], [], [], []
    worker.done.connect(done.append)
    worker.failed.connect(failed.append)
    worker.progress.connect(lambda n, total, stage:
                            steps.append((n, total, stage)))
    worker.split_relaxed.connect(relaxed.append)
    started = time.perf_counter()
    worker.run()
    elapsed = time.perf_counter() - started

    assert not failed, failed
    assert elapsed < _BOUND_S, f"the round took {elapsed:.1f} s"
    assert [s[0] for s in steps] == [1, 2, 3, 4, 5]
    assert {s[1] for s in steps} == {mod._SuggestWorker.STEPS}
    assert [s[2] for s in steps] == ["clear", "features", "fit", "rank",
                                     "write"]
    assert relaxed and "well" in relaxed[0]
    proposal, written, _ = done[0]
    assert written == 40 and _suggested(db) == 40


def _screen(qtbot, db: Path):
    """An Annotate screen pointed at ``db`` with no source opened."""
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(db)
    widget._settings.annotation_column = "annotate"
    return widget


def test_the_screen_runs_a_round_to_the_end_without_a_box(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The whole path from the menu entry, on a real thread, in seconds."""
    import spacr.active_learning as al
    from PySide6.QtWidgets import QMessageBox

    db, features = _one_page_of_labels(tmp_path)
    monkeypatch.setattr(al, "round_features", lambda *a, **k: features)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: pytest.fail("a blocking box was opened")))
    screen = _screen(qtbot, db)
    assert screen._btn_suggest_cancel.isHidden(), (
        "Cancel is shown only while a run is going")

    started = time.perf_counter()
    screen._start_suggest(this_page=False)
    assert not screen._btn_suggest_cancel.isHidden()
    qtbot.waitUntil(lambda: screen._suggest_worker is None,
                    timeout=int(_BOUND_S * 1000))
    elapsed = time.perf_counter() - started

    assert elapsed < _BOUND_S
    assert "suggestions written" in screen._status_label.text()
    assert screen.findChild(QMessageBox, "AnnotateSuggestFailedBox") is None
    assert screen._btn_suggest.isEnabled()
    assert screen._btn_suggest_cancel.isHidden()
    assert _suggested(db) == 40


def test_cancel_stops_the_worker_before_it_fits(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Cancel during step 2 of 5: no fit, no suggestion, the thread ends."""
    import spacr.active_learning as al

    db, features = _one_page_of_labels(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    fitted = []

    def slow_features(*args, **kwargs):
        """Hold step 2 open until the test has pressed Cancel."""
        entered.set()
        release.wait(30)
        return features

    monkeypatch.setattr(al, "round_features", slow_features)
    monkeypatch.setattr(al, "retrain_round",
                        lambda *a, **k: fitted.append(1))
    screen = _screen(qtbot, db)
    screen._start_suggest(this_page=False)
    worker = screen._suggest_worker
    ended, cancelled = [], []
    worker.finished.connect(lambda: ended.append(1))
    worker.cancelled.connect(lambda: cancelled.append(1))
    assert entered.wait(30), "the run never reached its second step"
    qtbot.waitUntil(lambda: "step 2 of 5" in screen._status_label.text(),
                    timeout=5000)

    screen._btn_suggest_cancel.click()
    assert worker.isInterruptionRequested()
    assert not screen._btn_suggest_cancel.isEnabled()
    release.set()
    qtbot.waitUntil(lambda: screen._suggest_worker is None, timeout=10000)

    assert ended and cancelled, "the worker did not end as cancelled"
    assert fitted == [], "a cancelled run went on to fit"
    assert _suggested(db) == 0
    assert screen._status_label.text() == "Suggest cancelled."
    assert screen._btn_suggest.isEnabled()
    assert screen._btn_suggest_cancel.isHidden()


def test_cancel_with_no_run_does_nothing(qtbot, qt_theme_applied, tmp_path):
    """A stray press after the run ended changes nothing."""
    db, _ = _one_page_of_labels(tmp_path)
    screen = _screen(qtbot, db)
    before = screen._status_label.text()
    screen._cancel_suggest()
    assert screen._status_label.text() == before
