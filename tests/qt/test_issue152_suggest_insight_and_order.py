"""Suggest shows its progress, orders the pages and re-thresholds live (#152).

The maintainer's review of 2026-10-19: a Suggest run must show a progress bar
with its stage, the pages must be orderable by annotation, and a "Suggestion
model" window must show the model, a cross-validated scorecard against a
majority-class baseline and thresholds that move the suggestions on the pages
without fitting again. He had got every suggestion as one class; the
first tests below hold the cause and the fix.

Every test counts: crops suggested per class, progress steps, rows in an
order, fits made.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.suggest import SUGGESTION_OFFSET  # noqa: E402


def _database(tmp_path: Path, n: int = 120, labels=None, *,
              classes: int = 2, wells: int = 6, seed: int = 0):
    """``n`` crops over ``wells`` wells whose ``signal`` carries the class.

    :param labels: ``{index: class}`` to annotate.
    :returns: ``(db, features, latent)`` with ``latent`` the true class of
        every crop.
    """
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, feats, latent = [], [], []
    for i in range(n):
        cls = 1 + (i % classes)
        latent.append(cls)
        path = f"/crops/cell_{i:04d}.png"
        rows.append((path, "p1", "r1", f"c{i % wells}", "f1", None))
        feats.append({"png_path": path,
                      "signal": cls + rng.normal(0, 0.2),
                      "noise": rng.normal(0, 1.0)})
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE png_list (png_path TEXT, plateID TEXT, '
                     'rowID TEXT, columnID TEXT, fieldID TEXT, '
                     'annotate INTEGER)')
        conn.executemany('INSERT INTO png_list VALUES (?,?,?,?,?,?)', rows)
        conn.executemany('UPDATE png_list SET annotate=? WHERE png_path=?',
                         [(v, rows[i][0]) for i, v in (labels or {}).items()])
    return db, pd.DataFrame(feats).set_index("png_path"), latent


def _suggested(db: Path) -> dict:
    """``{class: crops suggested as it}``."""
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            'SELECT annotate - ?, count(*) FROM png_list WHERE annotate > ? '
            'GROUP BY annotate', (SUGGESTION_OFFSET, SUGGESTION_OFFSET)
        ).fetchall()
    return {int(c): int(k) for c, k in rows}


def _run(db: Path, features, **options):
    """One Suggest run on this thread, as the screen configures it."""
    from spacr.qt.screens import annotate as mod

    worker = mod._SuggestWorker(
        str(db), "annotate",
        {"features": features, "round_index": 1,
         "model_type": "gradient_boosting", "balance": "none",
         "save_model": False, "write_card": False, **options})
    done, failed, steps, scoring = [], [], [], []
    worker.done.connect(done.append)
    worker.failed.connect(failed.append)
    worker.progress.connect(lambda n, total, stage: steps.append(stage))
    worker.scoring.connect(scoring.append)
    worker.run()
    assert not failed, failed
    return worker, done[0], steps, scoring


def _twenty_labels():
    """Ten labels of each class, the recording's first page."""
    return {i: 1 + (i % 2) for i in range(20)}


def test_a_boosted_fit_on_twenty_labels_used_to_learn_nothing():
    """The cause of the single-class result, measured.

    scikit-learn's default leaf of 20 rows needs 40 training rows for one
    split. Twenty labels, downsampled and with a quarter held out, are
    fifteen rows: the trees never split, every crop gets the training prior
    and the arg-max is the same class for all of them.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    import spacr.active_learning as al

    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 8)
    x = np.c_[y + rng.normal(0, 0.2, len(y)), rng.normal(0, 1, len(y))]
    old = HistGradientBoostingClassifier(random_state=0).fit(x, y)
    assert np.ptp(old.predict_proba(x)[:, 1]) == 0.0
    new = al._build_round_model("gradient_boosting", 0, 2,
                                n_samples=len(y)).fit(x, y)
    predicted = new.predict_proba(x).argmax(axis=1)
    assert set(predicted.tolist()) == {0, 1}
    assert al._model_reweights("gradient_boosting")


def test_twenty_labels_now_suggest_both_classes(tmp_path):
    """The recording's ten-and-ten page gives suggestions of both classes."""
    db, features, latent = _database(tmp_path, labels=_twenty_labels())
    worker, (proposal, written, _), _, _ = _run(db, features)
    counts = _suggested(db)
    assert written == 100
    assert counts.get(1, 0) > 30 and counts.get(2, 0) > 30, counts
    assert not worker.insight["flat"]


def test_a_minority_class_is_suggested_on_an_imbalanced_set(tmp_path):
    """Sixty labels of class 1 and eight of class 2: class 2 is suggested.

    Class weights make the eight count as much as the sixty, so the
    minority class is proposed for the crops that carry its signal.
    """
    n = 400
    db, features, latent = _database(tmp_path, n=n, seed=3)
    ones = [i for i in range(n) if latent[i] == 1][:60]
    twos = [i for i in range(n) if latent[i] == 2][:8]
    with sqlite3.connect(db) as conn:
        conn.executemany('UPDATE png_list SET annotate=? WHERE png_path=?',
                         [(1, f"/crops/cell_{i:04d}.png") for i in ones]
                         + [(2, f"/crops/cell_{i:04d}.png") for i in twos])
    worker, (proposal, written, _), _, _ = _run(db, features)
    counts = _suggested(db)
    open_twos = sum(1 for i in range(n) if latent[i] == 2 and i not in twos)
    assert counts.get(2, 0) >= 0.8 * open_twos, counts
    model = worker.insight["model"]
    assert model["class_counts"] == {"1": 60, "2": 8}
    assert model["weights"]["2"] > model["weights"]["1"]


def test_the_scorecard_has_a_control_row_and_a_full_confusion_matrix(tmp_path):
    """Cross-validated on the labels, against the majority-class baseline."""
    labels = {i: 1 + (i % 2) for i in range(40)}
    labels.update({i: 1 for i in range(40, 60) if i % 2 == 0})
    db, features, _ = _database(tmp_path, labels=labels)
    worker, _, _, _ = _run(db, features)
    card = worker.insight["scorecard"]
    assert card["support"] == [30, 20]
    assert sum(map(sum, card["confusion"])) == 50
    assert card["baseline_class"] == 1
    assert card["baseline_accuracy"] == pytest.approx(0.6)
    assert card["accuracy"] > card["baseline_accuracy"]
    assert card["oof"].shape == (50, 2)


def test_the_run_reports_five_steps_and_how_many_crops_it_scores(tmp_path):
    """The progress signals the bar is driven from."""
    db, features, _ = _database(tmp_path, labels=_twenty_labels())
    _, _, steps, scoring = _run(db, features)
    assert steps == ["clear", "features", "fit", "rank", "write"]
    assert scoring == [120]


def _screen(qtbot, db: Path):
    """An Annotate screen pointed at ``db`` with no source opened."""
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(db)
    widget._settings.annotation_column = "annotate"
    return widget


def _run_on_screen(qtbot, screen, monkeypatch, features):
    """Press Suggest for every image and wait for the run to end."""
    import spacr.active_learning as al

    monkeypatch.setattr(al, "round_features", lambda *a, **k: features)
    screen._start_suggest(this_page=False)
    qtbot.waitUntil(lambda: screen._suggest_worker is None, timeout=60000)


def test_the_progress_bar_shows_each_stage_and_hides_at_the_end(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Shown while the run goes, one value per step, hidden afterwards."""
    db, features, _ = _database(tmp_path, labels=_twenty_labels())
    screen = _screen(qtbot, db)
    bar = screen._suggest_progress
    seen = []
    original = screen._on_suggest_progress

    def record(step, total, stage):
        """Note what the bar shows after each step starts."""
        original(step, total, stage)
        seen.append((bar.value(), bar.maximum(),
                     screen._suggest_stage.text(), bar.isVisibleTo(screen)))

    monkeypatch.setattr(screen, "_on_suggest_progress", record)
    assert bar.isHidden()
    _run_on_screen(qtbot, screen, monkeypatch, features)
    assert [v for v, _, _, _ in seen] == [0, 1, 2, 3, 4]
    assert {m for _, m, _, _ in seen} == {5}
    assert all(shown for _, _, _, shown in seen)
    assert seen[1][2] == "Reading features…"
    assert seen[2][2] == "Training and cross-validating…"
    assert bar.isHidden() and screen._suggest_stage.isHidden()
    assert not bar.isTextVisible()
    assert screen._btn_suggest_model.isEnabled()


def test_moving_the_threshold_changes_the_suggestions_without_a_fit(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The slider re-thresholds the cached probabilities, live."""
    import spacr.active_learning as al

    db, features, _ = _database(tmp_path, labels=_twenty_labels())
    screen = _screen(qtbot, db)
    _run_on_screen(qtbot, screen, monkeypatch, features)
    before = _suggested(db)
    assert before.get(1) and before.get(2)

    fits = []
    monkeypatch.setattr(al, "retrain_round",
                        lambda *a, **k: fits.append(1))
    dialog = screen._open_suggest_model()
    assert dialog is not None and dialog.isVisible()
    assert dialog.findChild(object, "AnnotateSuggestThreshold") is not None

    dialog._slider.setValue(0)
    dialog.apply_now()
    everything_two = _suggested(db)
    assert everything_two == {2: 100}

    dialog._slider.setValue(100)
    dialog.apply_now()
    assert _suggested(db).get(2, 0) < before[2]
    assert "class 2:" in dialog.counts_text()

    dialog._slider.setValue(50)
    dialog._band.setChecked(True)
    dialog._low.setValue(0.0)
    dialog._high.setValue(1.0)
    dialog.apply_now()
    assert _suggested(db) == {}
    assert "Withheld by the thresholds: 100" in dialog.counts_text()

    dialog._band.setChecked(False)
    dialog.apply_now()
    assert _suggested(db) == before
    assert fits == [], "a threshold move fitted a model"


def test_a_threshold_move_never_touches_answers_or_rejections(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Answered and rejected crops keep their state through any threshold."""
    from spacr.suggest import ensure_verdict_column

    db, features, _ = _database(tmp_path, labels=_twenty_labels())
    screen = _screen(qtbot, db)
    _run_on_screen(qtbot, screen, monkeypatch, features)
    ensure_verdict_column(str(db), "annotate")
    with sqlite3.connect(db) as conn:
        open_paths = [r[0] for r in conn.execute(
            'SELECT png_path FROM png_list WHERE annotate > ? '
            'ORDER BY png_path', (SUGGESTION_OFFSET,)).fetchall()]
        answered, rejected = open_paths[0], open_paths[1]
        conn.execute('UPDATE png_list SET annotate = 1 WHERE png_path = ?',
                     (answered,))
        conn.execute('UPDATE png_list SET annotate = NULL, '
                     'annotate_verdict = -2 WHERE png_path = ?', (rejected,))
    dialog = screen._open_suggest_model()
    for value in (0, 100, 0):
        dialog._slider.setValue(value)
        dialog.apply_now()
    with sqlite3.connect(db) as conn:
        state = dict(conn.execute(
            'SELECT png_path, annotate FROM png_list WHERE png_path IN (?,?)',
            (answered, rejected)).fetchall())
    assert state == {answered: 1, rejected: None}
    assert _suggested(db) == {2: 98}


def test_more_classes_get_a_minimum_probability_per_class(tmp_path):
    """Three classes: a class's floor withholds only that class."""
    from spacr.suggest import (_default_thresholds, _suggestion_counts,
                               _threshold_suggestions)

    frame = pd.DataFrame({"p_1": [0.9, 0.5, 0.2, 0.1],
                          "p_2": [0.05, 0.3, 0.7, 0.1],
                          "p_3": [0.05, 0.2, 0.1, 0.8]})
    rule = _default_thresholds([1, 2, 3])
    assert _suggestion_counts(_threshold_suggestions(frame, [1, 2, 3], rule),
                              [1, 2, 3]) == {"1": 2, "2": 1, "3": 1,
                                             "withheld": 0}
    rule["minimum"]["1"] = 0.6
    assert _suggestion_counts(_threshold_suggestions(frame, [1, 2, 3], rule),
                              [1, 2, 3]) == {"1": 1, "2": 1, "3": 1,
                                             "withheld": 1}
    rule["margin"] = 0.65
    assert _suggestion_counts(_threshold_suggestions(frame, [1, 2, 3], rule),
                              [1, 2, 3]) == {"1": 1, "2": 0, "3": 1,
                                             "withheld": 2}


def test_the_model_window_shows_the_scorecard_with_its_control_row(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Classes, then the macro mean, then the majority-class baseline."""
    from PySide6.QtWidgets import QTableWidget

    db, features, _ = _database(tmp_path, labels=_twenty_labels())
    screen = _screen(qtbot, db)
    _run_on_screen(qtbot, screen, monkeypatch, features)
    dialog = screen._open_suggest_model()
    table = dialog.findChild(QTableWidget, "AnnotateSuggestScorecard")
    assert table.rowCount() == 4
    assert table.item(3, 0).text() == "Majority-class baseline (control)"
    confusion = dialog.findChild(QTableWidget, "AnnotateSuggestConfusion")
    total = sum(int(confusion.item(r, c).text())
                for r in range(2) for c in range(2))
    assert total == 20
    histogram = dialog._histogram
    assert sum(map(sum, histogram.counts().values())) == 20
    assert histogram.lines() == [0.5]


def _labelled_order_db(tmp_path: Path):
    """Ten crops: answers, suggestions and open crops, with round scores."""
    db, _, _ = _database(tmp_path, n=10)
    values = [None, 2, 11, None, 1, 12, 2, None, 1, 11]
    probs = [0.55, 0.9, 0.8, None, 0.6, 0.99, 0.7, 0.51, 0.95, 0.65]
    with sqlite3.connect(db) as conn:
        conn.execute('ALTER TABLE png_list ADD COLUMN al_prob_0 REAL')
        conn.execute('ALTER TABLE png_list ADD COLUMN al_prob_1 REAL')
        for i, (v, p) in enumerate(zip(values, probs)):
            conn.execute(
                'UPDATE png_list SET annotate=?, al_prob_0=?, al_prob_1=? '
                'WHERE png_path=?',
                (v, p, None if p is None else 1 - p,
                 f"/crops/cell_{i:04d}.png"))
    return db


def _order(screen, qtbot, mode):
    """Choose ``mode`` in the Order control and return the crop indices."""
    combo = screen._sort_order
    screen._filtered_rows = None
    combo.setCurrentIndex(combo.findData(mode))
    qtbot.waitUntil(lambda: screen._filtered_rows is not None, timeout=10000)
    return [int(p[-8:-4]) for p, _ in screen._filtered_rows]


def test_the_pages_can_be_ordered_by_class_and_by_probability(
        qtbot, qt_theme_applied, tmp_path):
    """Each order, counted crop by crop, and kept while paging."""
    db = _labelled_order_db(tmp_path)
    screen = _screen(qtbot, db)
    assert screen.findChild(object, "AnnotateSortOrder") is screen._sort_order

    assert _order(screen, qtbot, "class_last") == [4, 8, 2, 9, 1, 6, 5,
                                                   0, 3, 7]
    assert _order(screen, qtbot, "class_first") == [0, 3, 7, 4, 8, 2, 9,
                                                    1, 6, 5]
    assert _order(screen, qtbot, "probability_high") == [5, 8, 1, 2, 6, 9,
                                                         4, 0, 7, 3]
    assert _order(screen, qtbot, "probability_low") == [7, 0, 4, 9, 6, 2,
                                                        1, 8, 5, 3]

    screen._offset = 4
    screen._load_page()
    assert [int(p[-8:-4]) for p, _ in screen._page_paths][:6] == [
        6, 2, 1, 8, 5, 3]
    assert screen._sort_mode == "probability_low"

    screen._sort_order.setCurrentIndex(screen._sort_order.findData(
        "original"))
    qtbot.waitUntil(lambda: screen._filtered_rows is None, timeout=10000)
    assert screen._offset == 0
