"""Item 288: the Suggest run's life on the Annotate screen, start to verdict.

``test_the_suggest_button.py`` pins what a suggestion IS (drawn dashed, in
its class's colour, never over a human's answer) and what the fit is asked
to do. This file pins what the SCREEN does around a run:

* a second press while a run is going is refused in words, not queued;
* "this page" with nothing on the page is refused before any thread starts;
* a started run disables the button and hands the worker the page's paths
  and the invented-negative count;
* a run that wrote nothing says why; one that wrote something says how many,
  warns when they are a ranking rather than answers, and reloads the page;
* a failed fit says what usually causes it, and writes nothing;
* the thread is retired and the button given back when it finishes;
* keeping or throwing away in bulk is asked first, and a "No" writes
  nothing.

The fit itself is replaced by a stand-in thread: what is under test is the
screen's bookkeeping, and a real fit is pinned in
``test_a_suggestion_can_be_judged.py``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal                   # noqa: E402
from PySide6.QtWidgets import QMessageBox                    # noqa: E402

import spacr.qt.screens.annotate as annotate_screen          # noqa: E402
from spacr.suggest import SUGGESTION_OFFSET                  # noqa: E402


def _db(tmp_path: Path, rows) -> Path:
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER)')
    con.executemany('INSERT INTO "png_list" VALUES (?,?)', rows)
    con.commit()
    con.close()
    return db


def _values(db: Path) -> dict:
    con = sqlite3.connect(db)
    rows = dict(con.execute('SELECT png_path, annotate FROM "png_list"'))
    con.close()
    return rows


class _StandInWorker(QObject):
    """Records how it was made and started; emits only when told to."""

    done = Signal(object)
    failed = Signal(str)
    finished = Signal()
    made = []

    def __init__(self, db_path, column, options, *, png_table="png_list",
                 only_paths=None, parent=None):
        super().__init__(parent)
        self.args = {"db_path": db_path, "column": column,
                     "options": dict(options), "png_table": png_table,
                     "only_paths": only_paths}
        self.started = False
        _StandInWorker.made.append(self)

    def start(self):
        self.started = True


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """An Annotate screen on a small crop table, its reloads recorded."""
    db = _db(tmp_path, [("/a.png", 1), ("/b.png", None), ("/c.png", None)])
    widget = annotate_screen.AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(db)
    widget._settings.annotation_column = "annotate"
    notices = []
    monkeypatch.setattr(
        widget._console, "append_notice",
        lambda text, **fields: notices.append(text.format(**fields)))
    reloads = []
    monkeypatch.setattr(widget, "_recount_judgements",
                        lambda: reloads.append("recount"))
    monkeypatch.setattr(widget, "_refresh_total",
                        lambda then=None: reloads.append("refresh"))
    widget.forget_the_run = lambda: setattr(widget, "_suggest_worker", None)
    widget.notices = notices
    widget.reloads = reloads
    widget.db = db
    _StandInWorker.made = []
    monkeypatch.setattr(annotate_screen, "_SuggestWorker", _StandInWorker)
    return widget


def test_a_second_press_while_a_run_is_going_is_refused(screen):
    screen._suggest_worker = object()
    try:
        screen._start_suggest(this_page=False)
    finally:
        screen._suggest_worker = None
    assert screen._status_label.text() == \
        "A suggestion run is already going."
    assert _StandInWorker.made == []


def test_this_page_with_nothing_on_it_starts_nothing(screen):
    screen._page_paths = []
    screen._start_suggest(this_page=True)
    assert screen._status_label.text() == \
        "There is nothing on this page to suggest for."
    assert _StandInWorker.made == []
    assert screen._suggest_worker is None


def test_a_started_run_holds_the_button_and_names_the_page(screen):
    screen._page_paths = [("/b.png", None), ("/c.png", None)]
    screen._start_suggest(this_page=True)
    held = screen._suggest_worker
    screen.forget_the_run()
    assert len(_StandInWorker.made) == 1
    worker = _StandInWorker.made[0]
    assert worker.started is True
    assert held is worker
    assert not screen._btn_suggest.isEnabled()
    assert worker.args["only_paths"] == ["/b.png", "/c.png"]
    assert worker.args["db_path"] == str(screen.db)
    assert worker.args["options"]["synthetic_negatives"] == 1, (
        "one class annotated once: one negative is invented")
    assert screen._suggestions_are_a_ranking is True
    assert "Fitting on the labels so far" in screen._status_label.text()


def test_a_run_over_the_whole_column_names_no_paths(screen):
    screen._start_suggest(this_page=False)
    screen.forget_the_run()
    assert _StandInWorker.made[0].args["only_paths"] is None


class _Proposal:
    note = "every crop is already answered"


def test_a_run_that_wrote_nothing_says_why(screen):
    screen._on_suggest_done((_Proposal(), 0))
    assert screen._status_label.text() == \
        "No suggestions — every crop is already answered."
    assert any("every crop is already answered" in text
               for text in screen.notices)


def test_a_run_that_wrote_nothing_without_a_note_still_says_so(screen):
    screen._on_suggest_done((object(), 0))
    assert screen._status_label.text() == \
        "No suggestions — nothing was left to suggest a label for."


def test_a_ranking_run_says_how_many_and_that_they_are_not_answers(screen):
    screen._suggestions_are_a_ranking = True
    screen._on_suggest_done((_Proposal(), 1234, 5))
    assert screen._status_label.text().startswith("1,234 suggestions written")
    joined = "".join(screen.notices)
    assert "Wrote 1,234 suggestions" in joined
    assert "INVENTED negatives" in joined
    assert "5 rejected suggestions were handed" in joined
    assert screen.reloads == ["recount", "refresh"]


def test_an_ordinary_run_neither_warns_nor_mentions_rejections(screen):
    screen._suggestions_are_a_ranking = False
    screen._on_suggest_done((_Proposal(), 3))
    joined = "".join(screen.notices)
    assert "Wrote 3 suggestions" in joined
    assert "INVENTED" not in joined
    assert "rejected suggestions" not in joined


def test_a_failed_fit_says_what_usually_causes_it(screen, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda parent, title, text:
                                     warned.append((title, text))))
    screen._on_suggest_failed("only one class")
    assert screen._status_label.text() == "Suggest failed — only one class"
    assert warned and warned[0][0] == "Suggest failed"
    assert "Nothing was written" in warned[0][1]
    assert _values(screen.db) == {"/a.png": 1, "/b.png": None,
                                  "/c.png": None}


def test_a_finished_run_gives_the_button_back_and_retires_the_thread(
        screen, monkeypatch):
    retired = []
    monkeypatch.setattr(annotate_screen, "_retire",
                        lambda obj: retired.append(obj) or True)
    screen._start_suggest(this_page=False)
    worker = screen._suggest_worker
    screen._on_suggest_finished()
    assert screen._suggest_worker is None
    assert screen._btn_suggest.isEnabled()
    assert retired == [worker]
    worker.done.emit((_Proposal(), 9))
    assert not any("Wrote 9" in text for text in screen.notices), (
        "a retired run's late signal must not reach the screen")


def test_finishing_with_no_run_recorded_only_gives_the_button_back(screen):
    screen._btn_suggest.setEnabled(False)
    screen._on_suggest_finished()
    assert screen._btn_suggest.isEnabled()


def test_resolving_with_no_source_does_nothing(screen, monkeypatch):
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a: asked.append(a)))
    screen._settings.db_path = ""
    screen._resolve_suggestions(True)
    assert asked == []


def test_resolving_with_nothing_waiting_says_so(screen):
    screen._resolve_suggestions(True)
    assert screen._status_label.text() == "There are no suggestions waiting."


def _outstanding(screen):
    con = sqlite3.connect(screen.db)
    con.execute('UPDATE "png_list" SET annotate = ? WHERE png_path = ?',
                (2 + SUGGESTION_OFFSET, "/b.png"))
    con.commit()
    con.close()


def test_saying_no_to_keeping_writes_nothing(screen, monkeypatch):
    _outstanding(screen)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda parent, title, text: asked.append((title, text))
        or QMessageBox.No))
    screen._resolve_suggestions(True)
    assert asked[0][0] == "Keep suggestions"
    assert "Accept all 1 suggestions" in asked[0][1]
    assert _values(screen.db)["/b.png"] == 2 + SUGGESTION_OFFSET


def test_keeping_turns_the_suggestion_into_an_answer(screen, monkeypatch):
    _outstanding(screen)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a: QMessageBox.Yes))
    screen._resolve_suggestions(True)
    assert _values(screen.db) == {"/a.png": 1, "/b.png": 2, "/c.png": None}
    assert screen._status_label.text() == "1 suggestions accepted."
    assert screen.reloads == ["recount", "refresh"]


def test_throwing_away_clears_the_suggestion_and_the_ranking(
        screen, monkeypatch):
    _outstanding(screen)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda parent, title, text: asked.append(title) or QMessageBox.Yes))
    screen._suggestions_are_a_ranking = True
    screen._resolve_suggestions(False)
    assert asked == ["Throw away suggestions"]
    assert _values(screen.db) == {"/a.png": 1, "/b.png": None,
                                  "/c.png": None}
    assert screen._suggestions_are_a_ranking is False
    assert screen._status_label.text() == "1 suggestions thrown away."
