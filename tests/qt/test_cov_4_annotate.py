"""Workers and teardown survive the screen being destroyed underneath them.

Qt deletes a widget's C++ half while Python still holds the wrapper, and a
page of thumbnails takes long enough to decode that the screen can be gone by
the time the worker has something to hand back. Every call into a dead C++
object raises ``RuntimeError``, and one raised out of a ``QThread.run``
override does not become a traceback -- PySide6 aborts the process. So each
of these paths is the difference between closing a screen and losing the
session.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6

from spacr.qt.screens import annotate as annotate_mod


class _DeadCppHalf(RuntimeError):
    """What Qt raises once the C++ object behind a wrapper is gone."""


# -- the page loader ---------------------------------------------------------

def test_a_page_that_finishes_after_the_screen_is_gone_is_dropped(qapp):
    """Raised out of run(), this would abort the process, not the thread."""
    state = {"decoded": False}

    class _GoesAwayMidDecode(annotate_mod._PageLoadWorker):
        def isInterruptionRequested(self):
            if state["decoded"]:
                raise _DeadCppHalf("Internal C++ object already deleted")
            return False

    def _decode(row):
        state["decoded"] = True
        return (row, None)

    seen = []
    worker = _GoesAwayMidDecode(3, [("a.png", None)], _decode)
    worker.done.connect(lambda gen, rows: seen.append(gen))
    worker.run()
    assert state["decoded"] is True
    assert seen == [], "results were handed to a screen that is gone"


def test_a_page_whose_images_will_not_load_still_reports_back(qapp):
    """A decode failure must not leave the screen waiting forever."""
    def _boom(_row):
        raise ValueError("not an image")

    seen = []
    worker = annotate_mod._PageLoadWorker(5, [("a.png", None)], _boom)
    worker.done.connect(lambda gen, rows: seen.append((gen, rows)))
    worker.run()
    assert seen == [(5, [])]


# -- the retrain worker ------------------------------------------------------

def test_a_failed_round_reported_to_a_dead_screen_is_dropped(qapp,
                                                             monkeypatch):
    """The failure has nowhere to go, and raising here aborts the process."""
    from spacr import active_learning as al

    def _boom(*_args, **_kwargs):
        raise ValueError("no usable features")

    monkeypatch.setattr(al, "retrain_round", _boom)

    worker = annotate_mod._RetrainWorker("db", "annotate", {})
    reported = []
    worker.failed.connect(reported.append)
    shiboken6.delete(worker)
    worker.run()
    assert reported == [], "a failure was reported to a screen that is gone"


def test_a_finished_round_reported_to_a_dead_screen_is_dropped(qapp,
                                                               monkeypatch):
    """Same rule on the success path: the result has nowhere to land."""
    from spacr import active_learning as al

    monkeypatch.setattr(al, "retrain_round",
                        lambda *_a, **_k: {"round_index": 1})

    class _Gone(annotate_mod._RetrainWorker):
        def isInterruptionRequested(self):
            raise _DeadCppHalf("Internal C++ object already deleted")

    worker = _Gone("db", "annotate", {})
    delivered = []
    worker.done.connect(delivered.append)
    worker.run()
    assert delivered == [], "a result was handed to a screen that is gone"


def test_a_finished_round_reaches_a_live_screen(qapp, monkeypatch):
    """The guards must not swallow the ordinary result."""
    from spacr import active_learning as al

    result = {"round_index": 1}
    monkeypatch.setattr(al, "retrain_round", lambda *_a, **_k: result)
    worker = annotate_mod._RetrainWorker("db", "annotate", {})
    seen = []
    worker.done.connect(seen.append)
    worker.run()
    assert seen == [result]


# -- teardown ----------------------------------------------------------------

class _AlreadyDeleted:
    """A widget whose C++ half went first, counting what was asked of it."""

    def __init__(self):
        self.viewport_asks = 0
        self.filter_removals = 0

    def viewport(self):
        self.viewport_asks += 1
        raise _DeadCppHalf("Internal C++ object already deleted")

    def removeEventFilter(self, _observer):
        self.filter_removals += 1
        raise _DeadCppHalf("Internal C++ object already deleted")


def test_teardown_survives_a_scroll_area_whose_viewport_is_gone(qtbot):
    """A child deleted by its Qt parent must not stop the screen closing, and
    the scroll area itself is still asked to drop the filter."""
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    scroll = _AlreadyDeleted()
    screen._grid_scroll = scroll
    screen._grid_holder = None
    screen._detach_event_filters()
    assert scroll.viewport_asks == 1
    assert scroll.filter_removals == 1


def test_teardown_survives_a_grid_holder_that_is_gone(qtbot):
    """Removing a filter from a deleted widget is not a reason to stay open."""
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    holder = _AlreadyDeleted()
    screen._grid_scroll = None
    screen._grid_holder = holder
    screen._detach_event_filters()
    assert holder.filter_removals == 1
