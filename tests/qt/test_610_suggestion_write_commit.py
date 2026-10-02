"""Suggest reports committed results and never leaves a partial proposal batch."""
from __future__ import annotations

import sqlite3
import threading

import pandas as pd
import pytest

pytest.importorskip('PySide6')

from spacr import suggest  # noqa: E402
from spacr.qt.screens.annotate import _SuggestWorker  # noqa: E402


def database(tmp_path, *, fail_second=False):
    """Build annotations, an earlier proposal and two unlabelled crops."""
    path = tmp_path / 'measurements.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE png_list (png_path TEXT PRIMARY KEY, annotate INTEGER, score REAL)')
        db.executemany('INSERT INTO png_list VALUES (?,?,NULL)', [
            ('human', 1), ('earlier', suggest.SUGGESTION_OFFSET + 2),
            ('first', None), ('second', None)])
        if fail_second:
            db.execute(f'''CREATE TRIGGER reject_second BEFORE UPDATE OF annotate ON png_list
                        WHEN NEW.png_path = 'second' AND NEW.annotate > {suggest.SUGGESTION_OFFSET}
                        BEGIN SELECT RAISE(ABORT, 'forced second-row failure'); END''')
    return path


def proposal():
    """Use the writer's actual proposal columns and suggestion encoding."""
    return suggest.Suggestions(pd.DataFrame({
        'png_path': ['first', 'second'], 'suggested': [1, 2],
        'stored': [suggest.SUGGESTION_OFFSET + 1, suggest.SUGGESTION_OFFSET + 2]}))


def values(path):
    """Read persisted values after the worker closes its write transaction."""
    with sqlite3.connect(path) as db:
        return {name: (label, score) for name, label, score in db.execute('SELECT * FROM png_list')}


def worker_for(path, monkeypatch):
    """Keep real proposal writes while replacing fitting with a tiny score save."""
    from spacr import active_learning

    def fit(*args, **kwargs):
        """Represent the earlier fit's independently committed round scores."""
        with sqlite3.connect(path) as db:
            db.execute("UPDATE png_list SET score=.8 WHERE png_path='first'")

    monkeypatch.setattr(active_learning, 'retrain_round', fit)
    monkeypatch.setattr(suggest, 'suggest_from_scores', lambda *args, **kwargs: proposal())
    worker = _SuggestWorker(str(path), 'annotate', {'features': pd.DataFrame({'x': [1]})})
    done, failed, cancelled = [], [], []
    worker.done.connect(done.append)
    worker.failed.connect(failed.append)
    worker.cancelled.connect(lambda: cancelled.append(True))
    return worker, done, failed, cancelled


def test_cancel_during_write_reports_the_committed_result(qtbot, tmp_path, monkeypatch):
    """Deliver a real interruption while step 5 is paused, then commit its rows."""
    path = database(tmp_path)
    worker, done, failed, cancelled = worker_for(path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = suggest.write_suggestions

    def held_write(*args, **kwargs):
        """Pause after entering the final step, before the actual SQLite write."""
        entered.set()
        if not release.wait(5):
            raise TimeoutError('test did not release the final write')
        return original(*args, **kwargs)

    monkeypatch.setattr(suggest, 'write_suggestions', held_write)
    worker.start()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        worker.requestInterruption()
        assert worker.isInterruptionRequested()
        release.set()
        qtbot.waitUntil(lambda: bool(done or failed or cancelled), timeout=5000)
        assert done and not failed and not cancelled
        assert done[0][1] == 2
        stored = values(path)
        assert stored['first'][0] == suggest.SUGGESTION_OFFSET + 1
        assert stored['second'][0] == suggest.SUGGESTION_OFFSET + 2
        assert stored['human'][0] == 1
    finally:
        release.set()
        assert worker.wait(5000)


def test_cancel_before_final_write_keeps_no_new_proposals(qtbot, tmp_path, monkeypatch):
    """Step-boundary cancellation remains distinct from finishing a write."""
    path = database(tmp_path)
    worker, done, failed, cancelled = worker_for(path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def held_ranking(*args, **kwargs):
        """Let Cancel arrive while ranking, before the next boundary."""
        entered.set()
        if not release.wait(5):
            raise TimeoutError('test did not release ranking')
        return proposal()

    monkeypatch.setattr(suggest, 'suggest_from_scores', held_ranking)
    worker.start()
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        worker.requestInterruption()
        release.set()
        qtbot.waitUntil(lambda: bool(done or failed or cancelled), timeout=5000)
        assert cancelled and not done and not failed
        stored = values(path)
        assert stored['first'] == (None, .8)
        assert stored['second'][0] is None
        assert stored['earlier'][0] is None
        assert stored['human'][0] == 1
    finally:
        release.set()
        assert worker.wait(5000)


def test_midwrite_error_rolls_back_proposals_but_not_earlier_phases(tmp_path, monkeypatch):
    """A real SQLite trigger aborts row two after row one's UPDATE succeeded."""
    path = database(tmp_path, fail_second=True)
    worker, done, failed, cancelled = worker_for(path, monkeypatch)
    worker.run()
    assert failed and 'forced second-row failure' in failed[0]
    assert not done and not cancelled
    stored = values(path)
    assert stored['first'] == (None, .8), 'proposal rolled back, earlier fit score retained'
    assert stored['second'][0] is None
    assert stored['earlier'][0] is None, 'clearing old proposals is an earlier committed phase'
    assert stored['human'][0] == 1


def test_failure_dialog_does_not_promise_that_earlier_phases_wrote_nothing(qtbot, monkeypatch):
    """The nonblocking dialog names the rollback boundary honestly."""
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._on_suggest_failed('forced second-row failure')
    box = screen.findChild(QMessageBox, 'AnnotateSuggestFailedBox')
    assert box is not None
    assert 'Nothing was written' not in box.text()
    assert 'No new suggestions from this round were saved' in box.text()
    assert 'Earlier suggestions may have been cleared' in box.text()
    assert 'round scores may have been updated' in box.text()
    box.close()
    screen.close()
