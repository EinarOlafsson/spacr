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

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure                               # noqa: E402
from PySide6.QtCore import (QObject, QSettings, QSizeF, Qt,        # noqa: E402
                            QTimer, Signal)
from PySide6.QtGui import QColor, QImage                           # noqa: E402

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


def _an_image(width: int = 8, height: int = 6) -> QImage:
    """A small, genuinely non-null QImage, as a worker would return."""
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#3366aa"))
    assert not image.isNull()
    return image


class _RecordingView:
    """Stands in for the pixmap view so "what was painted?" is answerable."""

    def __init__(self):
        self.shown = []

    def set_pixmap(self, pixmap):
        self.shown.append(pixmap)


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

A4_POINTS = QSizeF(612.0, 792.0)


class _OnePagePdfDocument:
    """A QPdfDocument that loads cleanly and holds a single A4 page."""

    class Error:
        None_ = 0

    def load(self, path):
        return 0

    def pageCount(self):                                   # noqa: N802 (Qt)
        return 1

    def pagePointSize(self, index):                        # noqa: N802 (Qt)
        assert index == 0, "a page other than the first was measured"
        return A4_POINTS


def _renderer_class(answer):
    """A ``QPdfPageRenderer`` stand-in that answers with ``answer``.

    ``answer=None`` means a renderer that never answers at all, which is what
    the guard timer is for. The instance is reachable through ``.made`` so a
    test can assert on the render mode and the requested page size.
    """
    made = []

    class _Renderer(QObject):
        pageRendered = Signal(int, object, object, object, int)

        class RenderMode:
            SingleThreaded = 0
            MultiThreaded = 1

        def __init__(self):
            super().__init__()
            self.mode = None
            self.document = None
            self.requests = []
            made.append(self)

        def setRenderMode(self, mode):                     # noqa: N802 (Qt)
            self.mode = mode

        def setDocument(self, document):                   # noqa: N802 (Qt)
            self.document = document

        def requestPage(self, page, size):                 # noqa: N802 (Qt)
            self.requests.append((page, size))
            if answer is None:
                return
            # Queued, not immediate: quitting a QEventLoop that has not begun
            # does nothing, and the real renderer answers from its own thread.
            QTimer.singleShot(
                0, lambda: self.pageRendered.emit(page, size, answer, None, 7))

    _Renderer.made = made
    return _Renderer


@pytest.fixture
def qtpdf(monkeypatch):
    """Put a stub ``PySide6.QtPdf`` in front of the real one."""
    import PySide6

    module = types.ModuleType("PySide6.QtPdf")
    module.QPdfDocument = _OnePagePdfDocument
    module.QPdfPageRenderer = _renderer_class(None)
    monkeypatch.setitem(sys.modules, "PySide6.QtPdf", module)
    monkeypatch.setattr(PySide6, "QtPdf", module, raising=False)
    return module


def test_a_rendered_page_comes_back_scaled_to_the_long_edge(qtpdf, tmp_path,
                                                            qapp):
    """The whole point of the function: one page in, one QImage out.

    The page is measured in POINTS and the caller asks in PIXELS, so the
    scale is taken from the longer edge and both edges are floored to at
    least one pixel -- a 612x792 pt page asked for at 64 px is 49x64, not
    64x64 and not 612x792. The renderer is put in ``MultiThreaded`` mode
    because the function is documented to run on a worker thread while the
    GUI thread keeps painting.
    """
    page = tmp_path / "figure.pdf"
    page.write_bytes(b"%PDF-1.4\n")
    answer = _an_image(20, 30)
    qtpdf.QPdfPageRenderer = _renderer_class(answer)

    image = fq.render_pdf_to_image(str(page), max_px=64, timeout_ms=5000)

    assert image is answer, "the rendered page was not the value returned"
    renderer = qtpdf.QPdfPageRenderer.made[-1]
    assert renderer.mode == qtpdf.QPdfPageRenderer.RenderMode.MultiThreaded, (
        "the render was not asked for off-thread")
    assert renderer.document is not None, "the renderer was given no document"
    (requested_page, requested_size), = renderer.requests
    assert requested_page == 0
    assert (requested_size.width(), requested_size.height()) == (49, 64), (
        "the page was not scaled to the requested long edge")


def test_a_renderer_that_never_answers_is_ended_by_the_guard(qtpdf, tmp_path,
                                                             qapp):
    """The nested loop is bounded by a timer, not by the renderer.

    ``_shutdown_jobs`` parks a render rather than killing it, so a renderer
    that never delivers a page would otherwise hold a worker thread inside
    ``QEventLoop.exec`` for the life of the process. The timer quits the
    loop and the function reports "no crisp version" -- which the caller
    reads as "keep the raster".
    """
    page = tmp_path / "figure.pdf"
    page.write_bytes(b"%PDF-1.4\n")
    qtpdf.QPdfPageRenderer = _renderer_class(None)

    assert fq.render_pdf_to_image(str(page), max_px=64, timeout_ms=40) is None
    silent = qtpdf.QPdfPageRenderer.made[-1]
    assert silent.requests, "the page was never requested at all"

    # The same file, the same loop, a renderer that answers: proof the None
    # above was the guard firing and not a refusal further up.
    answer = _an_image()
    qtpdf.QPdfPageRenderer = _renderer_class(answer)
    assert fq.render_pdf_to_image(str(page), max_px=64,
                                  timeout_ms=5000) is answer


def test_a_page_that_renders_to_nothing_is_not_a_crisp_version(qtpdf,
                                                               tmp_path,
                                                               qapp):
    """A null QImage is what a renderer that failed mid-page delivers.

    Returning it would put an empty sheet over a figure the user can already
    see, so it is reported the same way a timeout is.
    """
    page = tmp_path / "figure.pdf"
    page.write_bytes(b"%PDF-1.4\n")
    qtpdf.QPdfPageRenderer = _renderer_class(QImage())

    assert fq.render_pdf_to_image(str(page), max_px=64,
                                  timeout_ms=5000) is None

    good = _an_image()
    qtpdf.QPdfPageRenderer = _renderer_class(good)
    assert fq.render_pdf_to_image(str(page), max_px=64,
                                  timeout_ms=5000) is good


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
# A crisp render that lands after the user has moved on
# ---------------------------------------------------------------------------

def test_a_crisp_render_for_a_figure_left_behind_is_dropped_not_cached(qtbot):
    """Navigating away is one of the three things checked in the handler.

    The result is dropped rather than cached, and -- the part that matters --
    the slot is left with NO state at all rather than ``"done"``: the RAM
    window is the sliding window of what has been viewed, and the next visit
    to this figure renders it again instead of finding a "done" that never
    produced a pixmap.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.show_index(1)
    assert queue._current == 1

    view = _RecordingView()
    queue._view = view
    raster = queue._ram[0]
    queue._pdf_state[0] = 41
    queue._on_pdf_rendered((0, 41, _an_image()))

    assert view.shown == [], "a figure the user is not looking at was painted"
    assert 0 not in queue._pdf_state, (
        "the abandoned slot kept a verdict, so the crisp page will never be "
        "rendered again")
    assert queue._ram[0] is raster, (
        "the unviewed render replaced the cached raster, so the RAM window is "
        "no longer the window of what has been viewed")

    # The same payload for the slot actually on screen: accepted, painted,
    # and remembered as done.
    queue._pdf_state[1] = 42
    queue._on_pdf_rendered((1, 42, _an_image()))
    assert len(view.shown) == 1 and not view.shown[0].isNull()
    assert queue._pdf_state[1] == "done"


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
