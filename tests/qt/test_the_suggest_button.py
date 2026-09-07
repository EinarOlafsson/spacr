"""The Suggest button: a model's opinion you can reject, not a bulk edit.

WHAT THIS IS ABOUT. ``spacr.suggest`` was a library with tests and no
caller -- the exact state ``spacrops.py`` was in when instruction 372 was
filed for it. Annotate now calls it, and these are the properties that make
the calling safe rather than the ones that make it work:

* a suggestion is DRAWN as a proposal (dashed) in the class colour it
  proposes, never as a colour of its own -- a third swatch on a two-class
  screen reads as a third class;
* the scope the user picks narrows what is WRITTEN, never what the model
  learns from -- training on one page to save nothing would make a worse
  model for no gain;
* accepting in bulk cannot reach a human's annotation;
* and the fit happens off the GUI thread, because a thirty-second block on
  that thread is what the desktop offers to force-quit.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.suggest import SUGGESTION_OFFSET


def _empty_db(tmp_path: Path) -> Path:
    """A crop table with nothing annotated, which is what the worker needs.

    The worker clears outstanding suggestions before it fits (see
    `test_a_second_run_does_not_train_on_the_first_run_s_suggestions`), and
    that is a real UPDATE -- so a test that hands it a path with no database
    behind it is testing sqlite, not the button.
    """
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.commit()
    con.close()
    return db


def _stop(screen) -> None:
    """Retire a screen's save worker without a full ``close()``.

    THE SAME TEARDOWN ``test_annotate_keyboard`` uses, and for a reason
    found the hard way: ``close()`` runs the whole ``closeEvent``, which
    drains the population-count pool and both model threads with a budget.
    On a screen that never opened a source there is nothing to drain and
    the drain is what hangs -- this file sat at 0.9 % CPU until its
    ``timeout`` killed it. These tests are about the button, not about
    shutdown; ``test_annotate_worker_lifecycle`` owns that.
    """
    if screen._worker is not None:
        screen._worker.stop(wait=True)


def _annotate(db: Path, path_value_pairs) -> None:
    """Write annotation values straight into the crop table."""
    con = sqlite3.connect(db)
    for path, value in path_value_pairs:
        con.execute('UPDATE "png_list" SET annotate = ? WHERE png_path = ?',
                    (value, path))
    con.commit()
    con.close()


def _values(db: Path):
    """Every crop's annotation value, keyed by path."""
    con = sqlite3.connect(db)
    rows = con.execute(
        'SELECT png_path, annotate FROM "png_list"').fetchall()
    con.close()
    return dict(rows)


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------

def test_a_suggestion_is_drawn_in_its_own_class_colour(qtbot,
                                                       qt_theme_applied):
    """A suggested 1 wears class 1's colour, dashed -- not a third colour.

    The offset is a storage detail: an 11 handed raw to ``label_to_hex``
    either falls off the end of the palette or, worse, lands on the colour
    of a class the model never proposed. What separates a proposal from an
    answer is the dash, and the assertion is that the COLOURS are equal
    while the dashes differ.
    """
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._page_paths = [("/a.png", 1), ("/b.png", 1 + SUGGESTION_OFFSET)]

    assert widget._displayed_class(0) == 1
    assert widget._displayed_class(1) == 1, (
        "the suggestion's class must be read through the offset")
    assert widget._border_color_for(0) == widget._border_color_for(1), (
        "a suggested 1 and an answered 1 must be the same colour; the dash "
        "is the difference, because a third colour reads as a third class")
    assert widget._is_suggested_slot(1) is True
    assert widget._is_suggested_slot(0) is False
    _stop(widget)


def test_the_tile_carries_the_suggestion_as_a_property(qtbot,
                                                       qt_theme_applied):
    """``set_suggested`` mirrors onto a Qt property, like ``set_current``.

    So a stylesheet and a test can both ask, and there is exactly one notion
    of "this tile is a proposal" rather than one per reader.
    """
    from spacr.qt.screens.annotate import _Thumbnail

    tile = _Thumbnail(0)
    qtbot.addWidget(tile)
    assert tile.is_suggested() is False
    assert tile.set_suggested(True) is True, "the first change is a change"
    assert tile.set_suggested(True) is False, "a redundant set costs nothing"
    assert tile.is_suggested() is True
    assert tile.property("suggested") is True


# ---------------------------------------------------------------------------
# The scope
# ---------------------------------------------------------------------------

def test_the_page_scope_narrows_what_is_written_not_what_is_fitted(
        monkeypatch, tmp_path):
    """"This page" must not become "fit on this page".

    Training on nine crops to save no time would make a worse model and a
    worse ranking, and the reviewer would have no way to tell. The fit sees
    every label; the scope is applied to the PROPOSAL.
    """
    import pandas as pd

    from spacr.qt.screens import annotate as mod

    fitted = {}
    written = {}

    def fake_retrain(db_path, column, **options):
        """Record that the fit was asked for, with nothing narrowed."""
        fitted["called"] = (db_path, column, options)
        return object()

    frame = pd.DataFrame({
        "png_path": ["/a.png", "/b.png", "/c.png"],
        "suggested": [1, 2, 1],
        "stored": [11, 12, 11],
        "confidence": [0.9, 0.8, 0.7],
    })

    class FakeProposal:
        """The shape ``suggest_from_scores`` returns."""

        def __init__(self):
            self.frame = frame.copy()
            self.note = ""
            self.scored = 3
            self.classes = [1, 2]

    def fake_suggest(db_path, column, png_table="png_list"):
        """Hand back all three, as the real one would."""
        return FakeProposal()

    def fake_write(db_path, column, suggestions, png_table="png_list"):
        """Record exactly which crops were written."""
        written["paths"] = list(suggestions["png_path"])
        return len(suggestions)

    import spacr.active_learning as al
    import spacr.suggest as sug
    monkeypatch.setattr(al, "retrain_round", fake_retrain)
    monkeypatch.setattr(sug, "suggest_from_scores", fake_suggest)
    monkeypatch.setattr(sug, "write_suggestions", fake_write)

    worker = mod._SuggestWorker(
        str(_empty_db(tmp_path)), "annotate",
        {"model_type": "gradient_boosting"},
        only_paths=["/a.png", "/c.png"])
    worker.run()

    assert fitted["called"][1] == "annotate"
    assert "only_paths" not in fitted["called"][2], (
        "the scope must never reach the fit")
    assert written["paths"] == ["/a.png", "/c.png"], (
        "the page scope must narrow the write, and keep the confidence order")


def test_no_scope_writes_every_unanswered_crop(monkeypatch, tmp_path):
    """"Every image" is the absence of a filter, not a different query."""
    import pandas as pd

    from spacr.qt.screens import annotate as mod

    written = {}
    frame = pd.DataFrame({
        "png_path": ["/a.png", "/b.png"],
        "suggested": [1, 2], "stored": [11, 12], "confidence": [0.9, 0.8]})

    class FakeProposal:
        """The shape ``suggest_from_scores`` returns."""

        def __init__(self):
            self.frame = frame.copy()
            self.note = ""
            self.scored = 2
            self.classes = [1, 2]

    import spacr.active_learning as al
    import spacr.suggest as sug
    monkeypatch.setattr(al, "retrain_round", lambda *a, **k: object())
    monkeypatch.setattr(sug, "suggest_from_scores",
                        lambda *a, **k: FakeProposal())
    monkeypatch.setattr(
        sug, "write_suggestions",
        lambda db, col, s, png_table="png_list": written.setdefault(
            "paths", list(s["png_path"])) and 0 or len(s))

    worker = mod._SuggestWorker(str(_empty_db(tmp_path)), "annotate", {})
    worker.run()
    assert written["paths"] == ["/a.png", "/b.png"]


# ---------------------------------------------------------------------------
# The bulk verdicts
# ---------------------------------------------------------------------------

def test_keeping_in_bulk_cannot_reach_a_human_annotation(tmp_path: Path):
    """The one thing that must never happen, asserted on real rows.

    Every value a person put in the column is captured before and demanded
    identical after, both ways round. ``resolve_suggestions`` acts only on
    values above the offset, so a 1 and a 2 are invisible to it -- but that
    is the property, and a property is what a test is for.
    """
    from spacr.suggest import resolve_suggestions

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.executemany(
        'INSERT INTO "png_list" VALUES (?,?)',
        [("/human1.png", 1), ("/human2.png", 2), ("/none.png", None),
         ("/sug1.png", 1 + SUGGESTION_OFFSET),
         ("/sug2.png", 2 + SUGGESTION_OFFSET)])
    con.commit()
    con.close()

    human = {p: v for p, v in _values(db).items() if p.startswith("/human")}

    changed = resolve_suggestions(str(db), "annotate", keep=True)
    after = _values(db)
    assert changed == 2
    assert {p: after[p] for p in human} == human, (
        "a bulk KEEP reached a human's annotation")
    assert after["/sug1.png"] == 1 and after["/sug2.png"] == 2, (
        "accepted suggestions must become ordinary annotations")
    assert after["/none.png"] is None, "an unanswered crop must stay unanswered"


def test_throwing_away_clears_only_the_suggestions(tmp_path: Path):
    """And the other verdict, on the same rows, with the same guarantee."""
    from spacr.suggest import resolve_suggestions

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.executemany(
        'INSERT INTO "png_list" VALUES (?,?)',
        [("/human1.png", 1), ("/sug1.png", 1 + SUGGESTION_OFFSET)])
    con.commit()
    con.close()

    resolve_suggestions(str(db), "annotate", keep=False)
    after = _values(db)
    assert after["/human1.png"] == 1
    assert after["/sug1.png"] is None


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------

def test_the_menu_offers_the_verdicts_only_when_there_is_something_to_judge(
        qtbot, qt_theme_applied, tmp_path: Path):
    """"Keep 0 suggestions" would be worse than no entry at all.

    Which is why the menu is built on every press rather than once at
    construction: the count in the label has to be the count in the column.
    """
    from spacr.qt.screens.annotate import AnnotateScreen

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.executemany('INSERT INTO "png_list" VALUES (?,?)',
                    [("/a.png", None), ("/b.png", None)])
    con.commit()
    con.close()

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(db)
    widget._settings.annotation_column = "annotate"

    # `_build_suggest_menu` rather than `_on_suggest_menu`: the latter calls
    # `QMenu.exec`, which blocks in C++ and does not come back for a
    # monkeypatch on a Shiboken type. A first version of this test patched it
    # and hung until its timeout.
    menu = widget._build_suggest_menu()
    assert menu is not None
    assert len(menu.actions()) == 2, (
        "with nothing outstanding the menu is the two scopes and no verdicts")

    _annotate(db, [("/a.png", 1 + SUGGESTION_OFFSET),
                   ("/b.png", 2 + SUGGESTION_OFFSET)])
    labels = [action.text() for action in
              widget._build_suggest_menu().actions() if action.text()]
    assert any("Keep all 2" in text for text in labels), labels
    assert any("Throw away all 2" in text for text in labels), labels
    _stop(widget)


def test_suggest_without_a_source_asks_for_one_instead_of_raising(
        qtbot, qt_theme_applied, monkeypatch):
    """No source is a sentence, not a traceback."""
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.annotate import AnnotateScreen

    asked = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: asked.append(a))
    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = ""
    widget._on_suggest_menu()
    assert asked, "pressing Suggest with no source must say so"
    _stop(widget)


# ---------------------------------------------------------------------------
# The feedback loop, which is the thing wiring this button could have broken
# ---------------------------------------------------------------------------

def test_a_second_run_does_not_train_on_the_first_run_s_suggestions(
        monkeypatch, tmp_path: Path):
    """The model must never be fitted on its own opinion.

    A suggestion is stored as its class plus ten, and ``retrain_round``
    reads EVERY non-null value in the column as a class label --
    ``_class_value(11)`` is 11. So without this, pressing Suggest twice
    fits the second round on classes 11 and 12, which are labels no human
    ever made, and the model is learning from itself. It is also 379's
    stated rule that a re-run REPLACES the outstanding suggestions rather
    than adding to them, so both answers point the same way.

    Asserted on the real database rather than on a call count: what the fit
    sees is what is in the column when it runs.
    """
    import pandas as pd

    from spacr.qt.screens import annotate as mod

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.executemany(
        'INSERT INTO "png_list" VALUES (?,?)',
        [("/human.png", 1), ("/old_sug.png", 1 + SUGGESTION_OFFSET)])
    con.commit()
    con.close()

    seen = {}

    def fake_retrain(db_path, column, **options):
        """Record the column exactly as the fit would read it."""
        seen["values"] = sorted(
            v for v in _values(Path(db_path)).values() if v is not None)
        return object()

    class FakeProposal:
        """The shape ``suggest_from_scores`` returns."""

        def __init__(self):
            self.frame = pd.DataFrame(
                {"png_path": [], "suggested": [], "stored": [],
                 "confidence": []})
            self.note = ""
            self.scored = 0
            self.classes = [1, 2]

    import spacr.active_learning as al
    import spacr.suggest as sug
    monkeypatch.setattr(al, "retrain_round", fake_retrain)
    monkeypatch.setattr(sug, "suggest_from_scores",
                        lambda *a, **k: FakeProposal())

    mod._SuggestWorker(str(db), "annotate", {}).run()

    assert seen["values"] == [1], (
        "the fit saw a suggestion as a class label; the model would be "
        f"learning from its own output (column held {seen['values']})")
    assert _values(db)["/human.png"] == 1, (
        "clearing suggestions before the fit must not touch an annotation")


def test_retrain_asks_before_it_throws_a_review_queue_away(
        qtbot, qt_theme_applied, tmp_path: Path, monkeypatch):
    """Retrain hits the same trap from the other button, and must ASK.

    Suggest clears silently because replacing outstanding suggestions is
    its documented behaviour. Retrain says nothing about suggestions, so
    discarding a review queue on its behalf would be a surprise -- and
    fitting without discarding would train on classes 11 and 12.
    """
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.annotate import AnnotateScreen

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.executemany('INSERT INTO "png_list" VALUES (?,?)',
                    [("/a.png", 1 + SUGGESTION_OFFSET), ("/b.png", 1)])
    con.commit()
    con.close()

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(db)
    widget._settings.annotation_column = "annotate"

    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: asked.append(a) or QMessageBox.No)
    assert widget._clear_suggestions_before_fitting("Retrain") is False, (
        "declining must stop the fit, not proceed without the suggestions")
    assert asked, "Retrain must ask before discarding a review queue"
    assert _values(db)["/a.png"] == 1 + SUGGESTION_OFFSET, (
        "a declined question must leave the suggestions where they were")

    asked.clear()
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: asked.append(a) or QMessageBox.Yes)
    assert widget._clear_suggestions_before_fitting("Retrain") is True
    after = _values(db)
    assert after["/a.png"] is None, "agreeing must clear the suggestions"
    assert after["/b.png"] == 1, "and must not touch the annotation"
    _stop(widget)


def test_nothing_outstanding_asks_nothing(qtbot, qt_theme_applied,
                                          tmp_path: Path, monkeypatch):
    """The common case must not grow a dialog."""
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.annotate import AnnotateScreen

    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.execute('INSERT INTO "png_list" VALUES (?,?)', ("/a.png", 1))
    con.commit()
    con.close()

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(db)
    widget._settings.annotation_column = "annotate"

    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a) or QMessageBox.Yes)
    assert widget._clear_suggestions_before_fitting("Retrain") is True
    assert not asked, "no suggestions outstanding must mean no question"
    _stop(widget)
