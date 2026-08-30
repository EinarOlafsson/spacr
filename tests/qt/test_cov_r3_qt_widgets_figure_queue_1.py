"""The figure queue's refusals: resize, restyle and the right-click menu.

*The debounced resize.* ``_rerender_for_size`` runs on every settled drag of
the figure panel and ``_on_resize_rendered`` lands the worker's answer. Both
refuse a great deal -- a container too small to draw into, a figure that will
not report its DPI, a payload that is stale or empty -- and every refusal is
there to stop a bad render from replacing a good picture.

*The restyle.* ``refresh_current_figure`` / ``refresh_figure`` are what the
"Figure settings..." dialog and the context menu call, and callers believe
the bool they return: False means "nothing changed on screen".

*The right-click menu.* ``show_figure_menu`` is the only way to restyle a
figure from the picture itself, and its ``_redraw`` closure doubles as the
swap for "Show as ..." -- which builds a NEW Figure, so anything but a bool
arriving there means "replace, do not redraw".
"""
from __future__ import annotations

import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import pytest                                                     # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets import figure_queue as fq                   # noqa: E402
from spacr.qt.widgets.figure_queue import FigureQueue             # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def queue(qtbot):
    """A live queue on the raster path, torn down with the test.

    The live canvas is off deliberately: it short-circuits every raster
    refusal below.
    """
    widget = FigureQueue()
    qtbot.addWidget(widget)
    widget.set_live_canvas_enabled(False)
    return widget


@pytest.fixture
def figure():
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot([0, 1], [1, 0])
    yield fig
    plt.close(fig)


class _RecordingView:
    """Stands in for the pixmap view so "was it painted?" is answerable."""

    def __init__(self):
        self.shown = []

    def set_pixmap(self, pixmap):
        self.shown.append(pixmap)


# ---------------------------------------------------------------------------
# render_pdf_to_image -- the three refusals before a page is ever rendered.
# ---------------------------------------------------------------------------

class _StubPdfError:
    None_ = 0
    FileNotFound = 1


class _StubPdfDocument:
    """A QPdfDocument that loads cleanly and holds no pages."""

    Error = _StubPdfError

    def load(self, path):
        return _StubPdfError.None_

    def pageCount(self):                                   # noqa: N802 (Qt)
        return 0

    def pagePointSize(self, index):                        # noqa: N802 (Qt)
        raise AssertionError(
            "a document with no pages must be refused before it is measured")


@pytest.fixture
def stub_qtpdf(monkeypatch):
    """Put a stub ``PySide6.QtPdf`` in front of the real one.

    ``QtPdf`` is a separate Qt shared library that does not load in every
    environment the suite runs in; where it does not, ``render_pdf_to_image``
    leaves by its outer handler and the guards below are never consulted.
    """
    import PySide6

    module = types.ModuleType("PySide6.QtPdf")
    module.QPdfDocument = _StubPdfDocument
    module.QPdfPageRenderer = type("QPdfPageRenderer", (), {})
    monkeypatch.setitem(sys.modules, "PySide6.QtPdf", module)
    monkeypatch.setattr(PySide6, "QtPdf", module, raising=False)
    return module


def test_a_vanished_pdf_is_refused_before_the_document_is_built(stub_qtpdf,
                                                                tmp_path):
    """The queue deletes its temp dir on close, under renders in flight, so a
    missing file is the NORMAL end of a render rather than an error -- and it
    is refused before the document is built at all."""
    loads = []

    class _Recording(_StubPdfDocument):
        def load(self, path):
            loads.append(path)
            return _StubPdfError.None_

    stub_qtpdf.QPdfDocument = _Recording
    real = tmp_path / "there.pdf"
    real.write_bytes(b"%PDF-1.4\n")

    assert fq.render_pdf_to_image(str(tmp_path / "gone.pdf"),
                                  timeout_ms=100) is None
    assert loads == [], "a file that is not there was opened anyway"

    # The same stub on a file that IS there gets as far as loading it, so the
    # refusal above was the path check and not a stub refusing everything.
    assert fq.render_pdf_to_image(str(real), timeout_ms=100) is None
    assert loads == [str(real)]


def test_a_document_that_will_not_load_is_refused(stub_qtpdf, tmp_path):
    """A half-written PDF must not be rendered as a blank page.

    ``_request_pdf_refinement`` reads ``None`` as "no crisp version, keep the
    raster"; rendering a document that failed to load would put an empty white
    sheet over a figure the user can already see.
    """
    class _WillNotLoad(_StubPdfDocument):
        def load(self, path):
            return _StubPdfError.FileNotFound

        def pageCount(self):                               # noqa: N802 (Qt)
            raise AssertionError(
                "a document that failed to load was rendered anyway")

    stub_qtpdf.QPdfDocument = _WillNotLoad
    page = tmp_path / "half-written.pdf"
    page.write_bytes(b"%PDF-1.4\n")

    assert fq.render_pdf_to_image(str(page), timeout_ms=100) is None


def test_a_pdf_with_no_pages_is_refused(stub_qtpdf, tmp_path):
    """``pagePointSize(0)`` on an empty document is a crash, not a None, and
    the refinement runs on a worker thread where an exception is unhandled and
    has no window to report itself in. The stub asserts if page 0 is measured
    at all."""
    page = tmp_path / "empty.pdf"
    page.write_bytes(b"%PDF-1.4\n")

    assert fq.render_pdf_to_image(str(page), timeout_ms=100) is None


# ---------------------------------------------------------------------------
# _rerender_for_size -- the resize the user drives with a mouse drag.
# ---------------------------------------------------------------------------

def test_a_container_too_small_to_draw_into_is_left_alone(queue, figure,
                                                          tmp_path):
    """Mid-layout the panel is briefly a few pixels wide.

    Re-rendering there sets the figure to a size nobody asked for and throws
    the real one away; the strip thumbnail is built from that render, so the
    user is left with a smear.
    """
    queue._figures[0] = figure
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "f.png")
    queue._jobs.submit = lambda fn, cb: None

    queue._view.resize(40, 40)
    before = tuple(figure.get_size_inches())
    queue._rerender_for_size()
    assert tuple(figure.get_size_inches()) == before

    queue._view.resize(900, 700)
    queue._rerender_for_size()
    assert tuple(figure.get_size_inches()) != before, (
        "the queue no longer resizes anything, so the refusal proves nothing")


def test_a_figure_that_cannot_be_measured_costs_no_render(queue, figure,
                                                          tmp_path):
    """A figure whose canvas Qt already destroyed cannot report its size.

    Seventy of those are in the maintainer's log. Rendering one produces
    nothing useful and burns a worker per resize frame.
    """
    class _Unmeasurable:
        def get_dpi(self):
            raise RuntimeError("Internal C++ object already deleted")

    submitted = []
    queue._jobs.submit = lambda fn, cb: submitted.append(fn)
    queue._view.resize(880, 660)

    queue._figures[0] = _Unmeasurable()
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "broken.png")
    queue._rerender_for_size()
    assert submitted == [], "a figure that cannot be measured was rendered"

    queue._figures[1] = figure
    queue._current = 1
    queue._png_paths[1] = str(tmp_path / "good.png")
    queue._rerender_for_size()
    assert len(submitted) == 1, "the healthy figure was refused too"


# ---------------------------------------------------------------------------
# _on_resize_rendered -- what the worker hands back.
# ---------------------------------------------------------------------------

def test_a_finished_resize_render_is_cached_and_shown(queue, figure,
                                                      tmp_path):
    """The whole point of moving the render off the GUI thread.

    The pixmap has to reach the RAM cache and the view, and any crisp PDF
    render cached for the slot is of the OLD size: kept, it lands afterwards
    and puts the pre-resize picture back.
    """
    png = tmp_path / "resized.png"
    figure.savefig(str(png))
    view = _RecordingView()
    queue._view = view
    queue._current = 0
    queue._resize_seq = 4
    queue._pdf_state[0] = "done"

    queue._on_resize_rendered((0, 4, True, str(png)))

    assert 0 in queue._ram, "the finished render was not cached"
    assert not queue._ram[0].isNull()
    assert len(view.shown) == 1, "the finished render never reached the view"
    assert 0 not in queue._pdf_state, (
        "a crisp render of the OLD size survived and will repaint over this")


def test_an_empty_or_stale_resize_payload_paints_nothing(queue, figure,
                                                         tmp_path):
    """A drag dispatches a render per settled frame; only the last may show.

    Three ways a payload is not the one to paint: the worker returned nothing,
    the render failed, and the user navigated away while it was drawing. The
    good payload at the end proves the view was reachable all along.
    """
    png = tmp_path / "ok.png"
    figure.savefig(str(png))
    view = _RecordingView()
    queue._view = view
    queue._current = 0
    queue._resize_seq = 9

    queue._on_resize_rendered(None)
    queue._on_resize_rendered(())
    queue._on_resize_rendered((0, 9, False, str(png)))
    queue._on_resize_rendered((3, 9, True, str(png)))
    assert view.shown == [], "a stale or failed resize render was painted"
    assert 0 not in queue._ram

    queue._on_resize_rendered((0, 9, True, str(png)))
    assert len(view.shown) == 1, "the current render was refused as well"


def test_a_resize_render_that_wrote_no_readable_png_paints_nothing(
        queue, figure, tmp_path):
    """``ok`` is True but the file is not an image, so the pixmap is null: a
    blank grey rectangle where the figure was, made permanent for the slot if
    it is cached."""
    rubbish = tmp_path / "truncated.png"
    rubbish.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    good = tmp_path / "whole.png"
    figure.savefig(str(good))
    view = _RecordingView()
    queue._view = view
    queue._current = 0
    queue._resize_seq = 1

    queue._on_resize_rendered((0, 1, True, str(rubbish)))
    assert view.shown == [], "an unreadable render was painted"
    assert 0 not in queue._ram, "an unreadable render was cached"

    queue._on_resize_rendered((0, 1, True, str(good)))
    assert len(view.shown) == 1 and not queue._ram[0].isNull()


# ---------------------------------------------------------------------------
# refresh_current_figure / refresh_figure -- what a restyle returns.
# ---------------------------------------------------------------------------

def test_an_empty_panel_reports_that_nothing_was_refreshed(queue, figure):
    """The settings dialog asks for a redraw before a run has produced one.

    False is the answer callers act on. A True here would leave the dialog
    claiming a figure was restyled while the panel is empty -- so the same
    queue, once it holds a figure, has to answer True.
    """
    assert queue.refresh_current_figure() is False
    assert queue.refresh_current_figure(preview=True) is False

    queue.add_figure(figure)
    assert queue.refresh_current_figure() is True


def test_a_restyle_during_a_draw_is_promised_not_refused(queue, figure,
                                                         tmp_path):
    """A control moving while a preview draws must not queue a second worker.

    A worker per control change spends the interaction copying figures whose
    renders are stale before they land. The change is remembered instead, and
    True keeps the caller off a GUI-thread render.
    """
    queue._figures[0] = figure
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "f.png")
    queue._preview_busy = True
    queue._jobs.submit = lambda fn, cb: pytest.fail(
        "a second preview worker was started while one was drawing")

    assert queue.refresh_current_figure(preview=True) is True
    assert queue._preview_pending is True, (
        "the change was dropped; the picture will never catch up")


def test_a_restyle_that_cannot_be_written_reports_failure(queue, figure,
                                                          tmp_path):
    """The temp directory can be gone -- the queue deletes it on close.

    The render then writes nowhere, and a True would leave the view and the
    strip showing the pre-restyle picture while the dialog says it landed.
    """
    queue._figures[0] = figure
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "deleted-dir" / "f.png")

    assert queue.refresh_current_figure() is False
    assert not (tmp_path / "deleted-dir").exists(), "the render created the dir"

    # The same figure, into a directory that is there: proof the False above
    # was the failed write and not a queue that refuses everything.
    queue._png_paths[0] = str(tmp_path / "written.png")
    assert queue.refresh_current_figure() is True
    assert (tmp_path / "written.png").is_file()


def test_a_restyle_repaints_the_view_and_the_strip_thumbnail(queue, figure,
                                                             tmp_path):
    """Both pictures of a figure have to move together.

    Restyling the view and not the strip thumbnail is the bug the context menu
    was reported for: it appeared to do nothing, so the user did it again.
    """
    queue.add_figure(figure)
    idx = queue._current
    queue._pdf_state[idx] = "done"
    view = _RecordingView()
    queue._view = view

    assert queue.refresh_current_figure() is True
    assert len(view.shown) == 1, "the restyled figure never reached the view"
    assert queue._pdf_state.get(idx) != "done", (
        "the crisp render of the OLD styling was kept; it would repaint the "
        "pre-restyle picture over the new one")
    item = queue._list.item(idx)
    assert item is not None and not item.icon().isNull(), (
        "the strip still shows the pre-restyle thumbnail")


def test_restyling_a_figure_that_is_not_on_screen_reports_failure(
        queue, figure, tmp_path):
    """Restyling from a grid tile touches a figure the view is not showing.

    The grid stays put, so a failed render there is invisible unless the
    return value says so -- and `replace_figure` hands it straight back as
    the result a "Show as violin" menu entry reports.
    """
    second, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 0], [0, 1])
    try:
        queue.add_figure(figure)
        queue.add_figure(second)
        queue.show_index(1)
        queue._png_paths[0] = str(tmp_path / "no-such-dir" / "f.png")

        assert queue.refresh_figure(0) is False
        # The figure it could not write is still the one at slot 0: the
        # failure is the render, not a queue that lost the figure.
        assert queue.figure_for(0) is figure
    finally:
        plt.close(second)


# ---------------------------------------------------------------------------
# The right-click menu, and the closure that doubles as the figure swap.
# ---------------------------------------------------------------------------

class _StubMenu:
    def __init__(self):
        self.executed_at = []

    def exec(self, position):
        self.executed_at.append(position)


@pytest.fixture
def menu_seam(monkeypatch):
    """Capture what ``show_figure_menu`` builds, without opening a menu.

    ``menu.exec()`` spins its own modal event loop, so a real menu would hang
    the suite.
    """
    from spacr.qt.widgets import figure_settings

    built = {}

    def _build(owner, figure, on_change=None, open_settings=None):
        built["owner"] = owner
        built["figure"] = figure
        built["on_change"] = on_change
        built["open_settings"] = open_settings
        built["menu"] = _StubMenu()
        return built["menu"]

    monkeypatch.setattr(figure_settings, "build_figure_context_menu", _build)
    return built


def test_right_clicking_a_thumbnail_navigates_to_that_figure(queue, figure,
                                                             menu_seam):
    """The strip's menu acts on the figure under the cursor, not the shown one.

    Right-clicking thumbnail #2 while #1 is on screen and restyling #1 is the
    worst kind of wrong: the figure they did not touch changes instead.
    """
    from PySide6.QtCore import QPoint

    second, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([0, 1], [0, 1])
    try:
        queue.add_figure(figure)
        queue.add_figure(second)
        queue.show_index(0)

        menu = queue.show_figure_menu(QPoint(3, 4), idx=1)

        assert queue._current == 1, "the menu did not navigate to the figure"
        assert menu_seam["figure"] is second
        assert menu is menu_seam["menu"]
        assert menu.executed_at == [QPoint(3, 4)], (
            "the menu was not opened where the user clicked")
    finally:
        plt.close(second)


def test_the_grid_menu_does_not_move_the_selection(queue, figure, menu_seam):
    """A grid is for comparing figures; jumping to one loses the comparison.

    ``navigate=False`` is what the grid passes, and the menu still has to be
    built for the tile that was clicked.
    """
    from PySide6.QtCore import QPoint

    second, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([0, 1], [1, 1])
    try:
        queue.add_figure(figure)
        queue.add_figure(second)
        queue.show_index(0)

        queue.show_figure_menu(QPoint(0, 0), idx=1, navigate=False)

        assert queue._current == 0, "the grid's menu moved the selection"
        assert menu_seam["figure"] is second
    finally:
        plt.close(second)


def test_a_bool_redraws_and_a_figure_replaces(queue, figure, menu_seam):
    """"Show as ..." builds a NEW Figure, so the callback doubles as the swap.

    ``figure_settings._replot`` hands the replacement to the same ``on_change``
    a colour toggle calls with a bool. Anything that is not a bool is the new
    figure, and everything holding the old one has to be pointed at it
    together, or the tile keeps the old picture and the menu looks broken.
    """
    from PySide6.QtCore import QPoint

    replacement, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([2, 3], [3, 2])
    try:
        queue.add_figure(figure)
        queue.show_figure_menu(QPoint(0, 0), idx=0)
        redraw = menu_seam["on_change"]

        assert redraw(False) is True, "a toggle did not redraw the figure"
        assert queue.figure_for(0) is figure

        assert redraw(replacement) is True, "the new figure was not adopted"
        assert queue.figure_for(0) is replacement, (
            "the queue still holds the figure the menu replaced")
    finally:
        plt.close(replacement)


def test_figure_settings_refuses_to_open_on_nothing(queue, monkeypatch):
    """The button is reachable before a run has produced a figure, and a
    dialog built on ``None`` raises the moment it reads a facecolor -- a
    traceback in the user's face for pressing a button they were offered."""
    from spacr.qt.widgets import figure_settings

    opened = []

    class _Dialog:
        def __init__(self, fig, *args, **kwargs):
            opened.append(fig)

        def exec(self):
            return 0

    monkeypatch.setattr(figure_settings, "FigureSettingsDialog", _Dialog)

    queue._open_figure_settings()
    assert opened == [], "a settings dialog was built on no figure"

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([0, 1], [0, 1])
    try:
        queue.add_figure(fig)
        queue._open_figure_settings()
        assert opened == [fig], "the dialog did not open on the real figure"
    finally:
        plt.close(fig)
