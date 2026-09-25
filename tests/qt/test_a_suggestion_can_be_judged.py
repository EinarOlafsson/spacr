"""A suggestion can be judged, and the judgement is kept (item 512, part 2).

The maintainer: "i dont understand how to tell the algorithm that the
annotations are correct or not other than simply changing the annotation."

What is asserted here:

* a suggested crop wears a badge a confirmed or a rejected crop does not,
  and the badge colours read in both themes;
* one click or one key confirms or rejects, the badge and the counts follow
  at once, and undo takes the judgement back with the label;
* the judgement reaches ``<column>_verdict`` in the database, survives a
  restart, and an old table without that column gains it on open without
  losing a row;
* the next Suggest round is handed the rejections as training examples,
  and a real round fits them as the other class.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr.suggest import SUGGESTION_OFFSET

ROWS = 2
COLS = 3


def _make_source(tmp_path: Path, values, *, with_verdict: bool = False):
    """An experiment folder whose crops carry ``values`` in ``annotate``."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    rng = np.random.default_rng(11)
    paths = []
    for i in range(len(values)):
        arr = rng.integers(0, 255, size=(20, 20, 3), dtype=np.uint8)
        path = src / "data" / f"crop_{i:02d}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    extra = ', "annotate_verdict" INTEGER' if with_verdict else ""
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                     f'annotate INTEGER{extra})')
        conn.executemany('INSERT INTO "png_list" (png_path, annotate) '
                         'VALUES (?, ?)', list(zip(paths, values)))
    return src, paths


def _db(src: Path) -> Path:
    return src / "measurements" / "measurements.db"


def _rows(src: Path):
    """``{png_path: (annotate, annotate_verdict)}`` as stored."""
    with sqlite3.connect(_db(src)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("png_list")')]
        if "annotate_verdict" not in cols:
            return {p: (v, None) for p, v in conn.execute(
                'SELECT png_path, annotate FROM "png_list"')}
        return {p: (v, j) for p, v, j in conn.execute(
            'SELECT png_path, annotate, annotate_verdict FROM "png_list"')}


def _open(qtbot, src: Path):
    """A screen with a pinned 2x3 grid and the source's first page loaded."""
    from spacr.qt.screens import annotate as mod

    screen = mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = ROWS
    screen._settings.grid_cols = COLS
    screen._settings.image_size = (20, 20)
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == ROWS * COLS,
                    timeout=5000)
    return screen


def _saved(screen, qtbot) -> None:
    """Flush the screen's changes and wait for the writer to drain."""
    screen._flush_pending()
    worker = screen._worker
    qtbot.waitUntil(lambda: not worker.busy and worker.pending_batches == 0,
                    timeout=5000)
    qtbot.wait(50)


def _stop(screen) -> None:
    if screen._worker is not None:
        screen._worker.stop(wait=True)


S1 = 1 + SUGGESTION_OFFSET
S2 = 2 + SUGGESTION_OFFSET


@pytest.fixture
def judged(tmp_path, qtbot, qt_theme_applied):
    """Four suggestions, one human label and one blank crop on one page."""
    src, paths = _make_source(tmp_path, [S1, S2, S1, S2, 1, None])
    screen = _open(qtbot, src)
    yield screen, src, paths
    _stop(screen)


# ---------------------------------------------------------------------------
# The badge
# ---------------------------------------------------------------------------

def test_a_suggestion_wears_a_badge_a_decision_does_not(judged):
    screen, _src, _paths = judged
    assert screen._thumbs[0].badge() == "suggested"
    assert screen._thumbs[0].property("judgement") == "suggested"
    assert screen._thumbs[4].badge() is None, "a human label has no badge"
    assert screen._thumbs[5].badge() is None, "a blank crop has no badge"


@pytest.mark.parametrize("dark", [True, False])
def test_the_badge_colours_read_in_both_themes(dark):
    """Glyph on fill at 4.5:1 or better, and three states, three fills."""
    from spacr.qt.screens.annotate import badge_colors
    from spacr.qt.theme import contrast_ratio

    fills = set()
    for state in ("suggested", "confirmed", "rejected"):
        fill, glyph = badge_colors(state, dark=dark)
        assert contrast_ratio(fill, glyph) >= 4.5, (state, fill, glyph)
        fills.add(fill.lower())
    assert len(fills) == 3, "the three states must not share a colour"


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def test_a_click_confirms_and_a_right_click_rejects(judged):
    screen, _src, _paths = judged
    screen._on_thumb_left(0)
    assert screen._current_value(0) == 1, "a confirmed 1 is an ordinary 1"
    assert screen._thumbs[0].badge() == "confirmed"
    assert screen._thumbs[0].is_suggested() is False

    screen._on_thumb_right(1)
    assert screen._current_value(1) is None, "a rejection clears the label"
    assert screen._thumbs[1].badge() == "rejected"

    counts = screen._page_judgements()
    assert counts == {"suggested": 4, "confirmed": 1, "rejected": 1,
                      "left": 2}
    text = screen._judge_label.text()
    for word in ("1 confirmed", "1 rejected", "2 left"):
        assert word in text, (word, text)


def test_a_click_on_a_decided_crop_still_labels_it(judged):
    """The click means confirm only on a suggestion; elsewhere it labels."""
    screen, _src, _paths = judged
    screen._on_thumb_left(5)
    assert screen._current_value(5) == 1
    assert screen._thumbs[5].badge() is None


def test_y_and_n_judge_the_focused_crop_and_move_on(judged):
    screen, _src, _paths = judged
    screen._set_focus_slot(0)
    assert screen.handle_key("y") is True
    assert screen._thumbs[0].badge() == "confirmed"
    assert screen.focus_slot == 1, "focus moves to the next suggestion"
    assert screen.handle_key("n") is True
    assert screen._thumbs[1].badge() == "rejected"
    assert screen.focus_slot == 2
    assert "not class 2" in screen._kbd_hint.text() or \
        "Rejected" in screen._kbd_hint.text()


def test_judging_a_crop_that_is_not_a_suggestion_says_so(judged):
    screen, _src, _paths = judged
    screen._set_focus_slot(4)
    before = screen._current_value(4)
    assert screen.handle_key("y") is True
    assert screen._current_value(4) == before
    assert screen._kbd_hint.text(), "the key must not do nothing silently"


def test_undo_takes_a_judgement_back(judged):
    screen, _src, _paths = judged
    screen._set_focus_slot(1)
    screen.handle_key("n")
    assert screen._judge_totals["rejected"] == 1
    screen.handle_key("u")
    assert screen._current_value(1) == S2
    assert screen._thumbs[1].badge() == "suggested"
    assert screen._judge_totals == {"left": 4, "confirmed": 0,
                                    "rejected": 0}


def test_the_page_buttons_judge_what_is_left(judged):
    screen, _src, _paths = judged
    screen._on_thumb_right(0)
    screen._judge_page(confirm=True)
    assert [screen._thumbs[i].badge() for i in range(4)] == [
        "rejected", "confirmed", "confirmed", "confirmed"]
    assert screen._current_value(4) == 1, "a human label is left alone"


def test_relabelling_a_confirmed_crop_withdraws_the_confirmation(judged):
    screen, _src, _paths = judged
    screen._on_thumb_left(0)
    assert screen._judge_totals["confirmed"] == 1
    screen._set_focus_slot(0)
    screen.handle_key("2")
    assert screen._verdict_value(0) is None
    assert screen._judge_totals["confirmed"] == 0


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------

def test_the_judgement_is_stored_and_survives_a_restart(
        tmp_path, qtbot, qt_theme_applied):
    src, paths = _make_source(tmp_path, [S1, S2, S1, S2, 1, None])
    screen = _open(qtbot, src)
    screen._on_thumb_left(0)
    screen._on_thumb_right(1)
    _saved(screen, qtbot)
    rows = _rows(src)
    assert rows[paths[0]] == (1, 1), "confirmed 1: label 1, verdict +1"
    assert rows[paths[1]] == (None, -2), "rejected 2: no label, verdict -2"
    assert rows[paths[2]] == (S1, None)
    assert rows[paths[4]] == (1, None)
    _stop(screen)

    again = _open(qtbot, src)
    assert again._thumbs[0].badge() == "confirmed"
    assert again._thumbs[1].badge() == "rejected"
    assert again._thumbs[2].badge() == "suggested"
    assert again._judge_totals == {"left": 2, "confirmed": 1, "rejected": 1}
    assert "1 rejected" in again._judge_label.text()
    _stop(again)


def test_an_old_table_gains_the_verdict_column_on_open(
        tmp_path, qtbot, qt_theme_applied):
    """The migration: no verdict column before, one after, no row lost."""
    src, paths = _make_source(tmp_path, [S1, 2, None, 1, S2, None])
    before = _rows(src)
    with sqlite3.connect(_db(src)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("png_list")')]
    assert "annotate_verdict" not in cols

    screen = _open(qtbot, src)
    with sqlite3.connect(_db(src)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("png_list")')]
    assert "annotate_verdict" in cols
    assert _rows(src) == before, "the migration must not change a value"
    screen._on_thumb_left(0)
    _saved(screen, qtbot)
    assert _rows(src)[paths[0]] == (1, 1)
    _stop(screen)


def test_the_migration_is_idempotent_and_refuses_nothing(tmp_path):
    from spacr.suggest import ensure_verdict_column

    src, _paths = _make_source(tmp_path, [S1, None])
    assert ensure_verdict_column(str(_db(src)), "annotate") is True
    assert ensure_verdict_column(str(_db(src)), "annotate") is True
    with sqlite3.connect(_db(src)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("png_list")')]
    assert cols.count("annotate_verdict") == 1


def test_keeping_in_bulk_is_recorded_as_confirming(tmp_path):
    from spacr.suggest import (ensure_verdict_column, judgement_counts,
                               resolve_suggestions)

    src, paths = _make_source(tmp_path, [S1, S2, 1, None])
    ensure_verdict_column(str(_db(src)), "annotate")
    resolve_suggestions(str(_db(src)), "annotate", keep=True)
    rows = _rows(src)
    assert rows[paths[0]] == (1, 1) and rows[paths[1]] == (2, 2)
    assert rows[paths[2]] == (1, None), "a human label is not a judgement"
    assert judgement_counts(str(_db(src)), "annotate") == {
        "left": 0, "confirmed": 2, "rejected": 0}


def test_clearing_the_column_clears_its_judgements(tmp_path):
    from spacr.qt.annotate_engine import clear_column

    src, paths = _make_source(tmp_path, [1, None], with_verdict=True)
    with sqlite3.connect(_db(src)) as conn:
        conn.execute('UPDATE png_list SET annotate_verdict = 1')
    clear_column(str(_db(src)), "annotate")
    assert set(_rows(src).values()) == {(None, None)}


def test_the_verdict_column_is_not_an_annotator(tmp_path):
    """Agreement must not offer ``annotate_verdict`` as a second rater."""
    from spacr.agreement import _is_model_column

    assert _is_model_column("annotate_verdict", ["annotate",
                                                 "annotate_verdict"])
    assert not _is_model_column("annotate", ["annotate", "annotate_verdict"])


# ---------------------------------------------------------------------------
# The next round
# ---------------------------------------------------------------------------

def test_the_next_suggest_round_is_handed_the_rejections(
        tmp_path, monkeypatch):
    """What the user rejected reaches the fit; what they confirmed is a
    label in the column the fit reads anyway."""
    from spacr.qt.screens import annotate as mod
    from spacr.suggest import ensure_verdict_column

    src, paths = _make_source(tmp_path, [1, None, S2, 2])
    db = str(_db(src))
    ensure_verdict_column(db, "annotate")
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE png_list SET annotate_verdict = 1 '
                     'WHERE png_path = ?', (paths[0],))
        conn.execute('UPDATE png_list SET annotate_verdict = -1 '
                     'WHERE png_path = ?', (paths[1],))

    seen = {}

    def fake_retrain(db_path, column, **options):
        seen["options"] = options
        seen["values"] = _rows(src)
        return object()

    class FakeProposal:
        def __init__(self):
            self.frame = pd.DataFrame({"png_path": [], "suggested": [],
                                       "stored": [], "confidence": []})
            self.note = ""
            self.scored = 0
            self.classes = [1, 2]

    import spacr.active_learning as al
    import spacr.suggest as sug
    monkeypatch.setattr(al, "retrain_round", fake_retrain)
    monkeypatch.setattr(sug, "suggest_from_scores",
                        lambda *a, **k: FakeProposal())
    emitted = []
    worker = mod._SuggestWorker(db, "annotate", {})
    worker.done.connect(emitted.append)
    worker.run()

    assert seen["options"]["rejections"] == {paths[1]: 1}
    assert seen["values"][paths[0]] == (1, 1), "the confirmation is a label"
    assert emitted and emitted[0][2] == 1, "the run says how many it sent"


def test_a_real_round_fits_a_rejection_as_the_other_class(tmp_path):
    import spacr.active_learning as al

    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, feats = [], []
    i = 0
    for plate in ("p1", "p2"):
        for row_id in ("r1", "r2"):
            for column in ("c1", "c2", "c3"):
                for _ in range(10):
                    latent = 1 if i % 2 == 0 else 2
                    path = f"/crops/cell_{i:04d}.png"
                    rows.append((path, plate, row_id, column, "f1", None))
                    feats.append({"png_path": path,
                                  "signal": latent + rng.normal(0, 0.25),
                                  "noise": rng.normal(0, 1.0)})
                    i += 1
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE png_list (png_path TEXT, plateID TEXT, '
                     'rowID TEXT, columnID TEXT, fieldID TEXT, '
                     'annotate INTEGER)')
        conn.executemany('INSERT INTO png_list VALUES (?,?,?,?,?,?)', rows)
        conn.executemany('UPDATE png_list SET annotate=? WHERE png_path=?',
                         [(1 if k % 2 == 0 else 2, rows[k][0])
                          for k in range(40)])
    features = pd.DataFrame(feats).set_index("png_path")
    rejected = {rows[k][0]: 1 for k in range(41, 61, 2)}
    rejected[rows[0][0]] = 2

    result = al.retrain_round(str(db), "annotate", features=features,
                              seed=0, save_model=False, write_card=False,
                              rejections=rejected)
    fitted = [n for n in result.notes if "rejected suggestions were fitted"
              in n]
    assert fitted and fitted[0].startswith("10 "), result.notes

    keys, labels, unusable = al._rejections_as_labels(
        {"a": 1, "b": 2, "c": 1}, [1, 2], {"c"}, {"a", "b", "c"})
    assert keys == ["a", "b"] and labels == [2, 1] and unusable == 0
    keys, labels, unusable = al._rejections_as_labels(
        {"a": 1}, [1, 2, 3], set(), {"a"})
    assert keys == [] and unusable == 1, "three classes: no 'other class'"


# ---------------------------------------------------------------------------
# The instructions
# ---------------------------------------------------------------------------

def test_the_keys_are_on_the_map_and_on_the_screen(judged):
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS

    screen, _src, _paths = judged
    annotate = {s.keys for s in SCREEN_SHORTCUTS if s.category == "Annotate"}
    assert {"Y", "N"} <= annotate
    assert screen._judge_bar.isVisibleTo(screen)
    hint = screen._judge_hint.text()
    assert "Y" in hint and "N" in hint, hint


def test_the_bar_stays_out_of_the_way_without_suggestions(
        tmp_path, qtbot, qt_theme_applied):
    src, _paths = _make_source(tmp_path, [1, 2, None, None, 1, None])
    screen = _open(qtbot, src)
    assert not screen._judge_bar.isVisibleTo(screen)
    _stop(screen)
