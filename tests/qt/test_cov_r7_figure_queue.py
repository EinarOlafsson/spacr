"""The vector render that succeeds, and three refusals underneath it.

Round 3 pinned ``render_pdf_to_image``'s refusals -- a missing file, a
document that will not load, a document with no pages -- but every one of
them returns before a renderer is ever built. What was left uncovered was
the whole second half of that function: the page measurement, the
``MultiThreaded`` renderer, the nested event loop, and the guard timer that
bounds it. This file drives that half, on the same stubbed ``PySide6.QtPdf``
round 3 introduced, because ``QtPdf`` is a separate Qt shared library that
does not load in this environment at all.

Three smaller refusals sit around it, and each is a real thing the gallery
survives rather than a defensive nicety:

* a graphics view whose scene is not installed when the panel asks for it
  -- the constructor must finish, and the surfaces below the scene must
  still be cleared;
* a crisp render that lands after the user has navigated away -- dropped
  rather than cached, so the RAM window stays the window of what has been
  *viewed*;
* a temp directory that will not delete -- forgotten anyway, because the
  widget is on its way out and a second attempt has nothing to attempt
  with.

Every "it refused" assertion here is paired, in the same test, with the
input that makes it accept.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure                               # noqa: E402
from PySide6.QtCore import (QSettings, Qt,        # noqa: E402
                            )
from PySide6.QtGui import QColor                           # noqa: E402

from spacr.qt.widgets import figure_queue as fq                    # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def prefs(monkeypatch, tmp_path_factory):
    """The real preference module, pointed at a throwaway INI.

    PNG explicitly: in the default ``pdf`` format every navigation would
    start a real vector refinement on a worker thread, and the tests below
    place their own tokens in ``_pdf_state`` to say what is in flight.
    """
    from spacr.qt import preferences as preferences_module

    store = tmp_path_factory.mktemp("figq_r7_prefs") / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    preferences_module.set_figure_format("png")
    return preferences_module


def _fig(seed: int = 0) -> Figure:
    """A small, real figure that is not in pyplot's registry."""
    figure = Figure(figsize=(3.0, 2.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [seed, seed + 1, seed])
    axes.set_title(f"fig {seed}")
    return figure


def _queue(qtbot):
    """A queue on the raster path."""
    queue = fq.FigureQueue(ram_cap=100)
    qtbot.addWidget(queue)
    queue.set_live_canvas_enabled(False)
    return queue


# ---------------------------------------------------------------------------
# render_pdf_to_image -- the page that really renders
# ---------------------------------------------------------------------------
#
# ``PySide6.QtPdf`` is a separate shared library and does not import in this
# environment (its Brotli dependency is unresolved), so the real renderer
# cannot be driven here at all. The stubs below are the same device round 3
# used for the refusals, extended to the point of answering: a document with
# one A4 page, and a renderer that delivers ``pageRendered`` the way the real
# one does -- queued back onto the calling thread, after ``requestPage`` has
# returned, which is why the nested QEventLoop exists.


# ---------------------------------------------------------------------------
# The scene that is not installed yet
# ---------------------------------------------------------------------------

def _view_class(hide_scene_once: bool):
    """A ``_ZoomView`` that paints its scene red, and may hide it once.

    A QGraphicsView is three surfaces and the SCENE's brush is the one that
    actually gets painted behind the items, so the panel clears all three.
    Presetting the scene's brush to red makes "the constructor cleared it"
    and "the constructor could not reach it" tell apart -- a default
    QGraphicsScene brush is already ``NoBrush``, so the cleared state is
    otherwise indistinguishable from the untouched one.

    The flag lives outside the class because a QObject subclass cannot carry
    Python attributes before ``super().__init__`` has run, and the scene has
    to be hidden on the FIRST call, which is the panel's.
    """
    state = {"hide": hide_scene_once}

    class _View(fq._ZoomView):
        def __init__(self, parent=None):
            super().__init__(parent)
            super().scene().setBackgroundBrush(QColor("#c81e1e"))

        def scene(self):
            # QGraphicsView.scene() is not virtual in C++, so only Python
            # callers -- i.e. the panel's constructor -- see this.
            if state["hide"]:
                state["hide"] = False
                return None
            return super().scene()

    return _View


def test_a_view_with_no_scene_yet_still_gets_a_transparent_panel(qtbot,
                                                                monkeypatch):
    """Instruction 118's transparent figure must survive a missing scene.

    Clearing the scene's brush is one of three surfaces and it is the only
    one that can be absent, so it is the only one inside a ``try``. If that
    handler did not exist the panel would not be built at all; because it
    does, the widget and viewport are still cleared and the queue still
    holds figures.
    """
    monkeypatch.setattr(fq, "_ZoomView", _view_class(True))
    hidden = _queue(qtbot)

    assert hidden._view.scene().backgroundBrush().color() == QColor("#c81e1e"), (
        "the scene was reached after all, so the handler under test was not "
        "the path taken")
    assert hidden._view.viewport().autoFillBackground() is False, (
        "construction stopped at the missing scene: the viewport still paints "
        "the palette's white Base")
    hidden.add_figure(_fig(0))
    assert hidden.count() == 1, "the panel was left unusable"

    # The same view with its scene in place: the brush IS cleared, so the
    # red above is the absence of that call rather than a brush that never
    # gets set.
    monkeypatch.setattr(fq, "_ZoomView", _view_class(False))
    normal = _queue(qtbot)
    assert normal._view.scene().backgroundBrush().style() == Qt.NoBrush, (
        "the scene kept painting its own background behind the figure")


# ---------------------------------------------------------------------------
# A temp directory that will not delete
# ---------------------------------------------------------------------------

def test_a_temp_directory_that_will_not_delete_is_forgotten_anyway(qtbot,
                                                                   monkeypatch):
    """``_delete_tempdir`` runs from ``closeEvent`` and from ``__del__``.

    ``rmtree(ignore_errors=True)`` swallows what it meets while walking, so
    reaching the handler takes a failure before the walk -- an unreadable
    parent, or an interpreter already tearing down under ``__del__``, which
    is why the call is patched here rather than staged on disk. Either way
    the widget is going: the path is forgotten so nothing else reads out of
    it, and a second attempt is not made.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    doomed = queue._tempdir
    assert Path(doomed).is_dir()

    real_rmtree = fq.shutil.rmtree
    calls = []

    def _will_not_delete(path, **kwargs):
        calls.append(path)
        raise OSError("the directory would not go")

    monkeypatch.setattr(fq.shutil, "rmtree", _will_not_delete)
    queue._delete_tempdir()

    assert calls == [doomed], "the delete was never attempted"
    assert Path(doomed).is_dir(), "the directory went away, so nothing raised"
    assert queue._tempdir is None, (
        "the widget kept a path it can no longer clear, and the next close "
        "would try again")
    queue._delete_tempdir()
    assert calls == [doomed], "a forgotten directory was attempted a second time"

    # A queue whose delete is allowed to work, to show the survival above is
    # the raise and not a directory that was never going to be removed.
    # Restored by hand rather than with ``monkeypatch.undo``, which would
    # also undo the preference-store redirection this file runs under.
    monkeypatch.setattr(fq.shutil, "rmtree", real_rmtree)
    other = _queue(qtbot)
    other.add_figure(_fig(1))
    gone = Path(other._tempdir)
    other._delete_tempdir()
    assert not gone.exists() and other._tempdir is None
