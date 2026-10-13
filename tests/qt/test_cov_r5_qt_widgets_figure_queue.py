"""FigureQueue when the layer under it fails, and the settings dialog's edges.

Round 3 pinned the gallery's ordinary refusals -- a stale resize payload, a
render that wrote no file, a menu built on nothing. What is left is the tier
below that: the file operations that raise rather than return False, the
worker-side preview draw, the RAM-budget accounting the process-wide policy
drives, and the half of ``_FigureSettingsDialog`` that only runs when the
preference store cannot be read or a colour dialog is cancelled.

Three themes run through it.

*Nothing is allowed to take the picture away.* A vector page that will not
delete, a live frame that will not move, a figure that will not pickle, a
runner that will not stop -- each is a real failure with a log line, and in
every case the gallery must still be holding a figure afterwards. The tests
assert on the pixmap and the strip icon, not on "it did not raise".

*The budget must answer honestly.* ``cache_budget_entries`` /
``drop_cache_budget_entry`` are called by a process-wide sweep that will
happily evict whatever they say is droppable. A figure a worker is drawing,
or the one on screen, is pinned; a size it has not measured yet is measured
rather than guessed.

*A dialog with no store behind it is still usable.* Every read of
``spacr.qt.preferences`` in the dialog is inside a ``try``, and the fallbacks
are what a user with an unreadable INI actually gets.
"""
from __future__ import annotations

import shutil as _real_shutil
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")

import numpy as np                                                 # noqa: E402
from matplotlib.figure import Figure                               # noqa: E402
from PySide6.QtCore import QPoint, QSettings                       # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap                  # noqa: E402
from PySide6.QtWidgets import QDialog                              # noqa: E402

from spacr.qt.job_runner import JobRunner                          # noqa: E402
from spacr.qt.widgets import figure_queue as fq                    # noqa: E402
from spacr.qt.widgets import figure_settings as fs                 # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def prefs(monkeypatch, tmp_path_factory):
    """The real preference module, pointed at a throwaway INI.

    The getters and setters ARE what several of these tests drive, so they
    are not stubbed; only the one accessor they all funnel through is
    redirected. Without it ``render_figure_to_png`` and
    ``_FigureSettingsDialog._apply_and_accept`` rewrite the developer's own
    Preferences. PNG explicitly: the default ``pdf`` writes a sibling vector
    page beside every raster, and several tests below are about a slot that
    has none, or one that has a page the format says should not be there.
    """
    from spacr.qt import preferences as preferences_module

    store = tmp_path_factory.mktemp("figq_r5_prefs") / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    preferences_module.set_figure_format("png")
    return preferences_module


def _fig(seed: int = 0) -> Figure:
    """A small, real, picklable figure that is not in pyplot's registry."""
    figure = Figure(figsize=(3.0, 2.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [seed, seed + 1, seed])
    axes.set_title(f"fig {seed}")
    return figure


def _queue(qtbot, *, live: bool = False, ram_cap: int = 100):
    """A queue on the raster path unless a test is about the live canvas."""
    queue = fq.FigureQueue(ram_cap=ram_cap)
    qtbot.addWidget(queue)
    queue.set_live_canvas_enabled(live)
    return queue


class _RecordingView:
    """Stands in for the pixmap view so "what was painted?" is answerable."""

    def __init__(self):
        self.shown = []

    def set_pixmap(self, pixmap):
        self.shown.append(pixmap)


def _an_image(width: int = 8, height: int = 6) -> QImage:
    """A small, genuinely non-null QImage, as a worker would return."""
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#3366aa"))
    assert not image.isNull()
    return image


class _NullConversion:
    """Stands in for ``QPixmap`` where ``fromImage`` yields nothing.

    Qt returns a null pixmap when it cannot allocate one for an image the
    worker built -- the case both PDF and preview delivery check for. There
    is no image that reliably provokes it on demand, so the conversion
    itself is the seam.
    """

    @staticmethod
    def fromImage(image):
        return QPixmap()


def _boom(*_args, **_kwargs):
    raise RuntimeError("the preference store is unreachable")


# ---------------------------------------------------------------------------
# The two context-menu entry points
# ---------------------------------------------------------------------------

def _stub_menu(monkeypatch):
    """Replace the real context menu with one that records and never execs.

    ``QMenu.exec`` spins its own modal event loop, which would hang the run.
    """
    seen = {}

    class _Menu:
        def exec(self, position):
            seen["at"] = position

    def _build(parent, figure, on_change=None, open_settings=None):
        seen["figure"] = figure
        return _Menu()

    monkeypatch.setattr(fs, "build_figure_context_menu", _build)
    return seen


def test_the_two_context_menu_hooks_pick_the_figure_under_the_cursor(
        qtbot, monkeypatch):
    """Qt hands both hooks a point in the WIDGET's coordinates.

    The view's menu is about the figure on screen and must open where the
    cursor is, so the point has to be mapped to global coordinates -- a menu
    posted at widget coordinates lands in the corner of the screen. The
    strip's menu is about the thumbnail under the cursor, which is usually
    not the figure being shown; right-clicking #1 and restyling #2 is the
    worst kind of wrong. Below the last row there is no thumbnail, and the
    menu then belongs to the figure on screen rather than to nothing.
    """
    seen = _stub_menu(monkeypatch)
    queue = _queue(qtbot)
    first, second = _fig(0), _fig(1)
    queue.add_figure(first)
    queue.add_figure(second)
    queue.show_index(1)

    queue._view_context_menu(QPoint(4, 5))
    assert seen["figure"] is second, "the view's menu took the wrong figure"
    assert seen["at"] == queue._view.mapToGlobal(QPoint(4, 5)), (
        "the menu was posted in widget coordinates, not where the user "
        "clicked")

    row = queue._list.item(0)
    over_row = queue._list.visualItemRect(row).center()
    assert queue._list.itemAt(over_row) is row, "the strip is not laid out"
    queue._list_context_menu(over_row)
    assert seen["figure"] is first, (
        "right-clicking thumbnail #1 built the menu for the shown figure")
    assert queue._current == 0, "the strip's menu did not navigate"

    queue.show_index(1)
    below_the_rows = QPoint(4, queue._list.viewport().height() + 4000)
    assert queue._list.itemAt(below_the_rows) is None
    queue._list_context_menu(below_the_rows)
    assert seen["figure"] is second, (
        "a right-click on empty strip built a menu for no figure")


# ---------------------------------------------------------------------------
# add_figure / _refresh_live_figure: the file operations that raise
# ---------------------------------------------------------------------------

class _ShutilThatCannotMove:
    """``shutil`` with a ``move`` that always fails.

    The bridge renders into its own directory and the queue moves the result
    into the queue's; the two can be on different volumes, and the queue
    deletes its own directory while renders are in flight. Either way ``move``
    raises rather than returning anything the caller can test.
    """

    rmtree = staticmethod(_real_shutil.rmtree)

    @staticmethod
    def move(src, dst):
        raise OSError("cross-device link: the temp volume went away")


def test_a_prerendered_png_that_cannot_be_moved_is_drawn_here_instead(
        qtbot, monkeypatch, tmp_path):
    """Adopting the worker's PNG is a fast path, not a required one.

    The move is the whole saving -- a file rename instead of a second
    ``savefig`` on the GUI thread. When it fails, dropping the figure would
    lose a plot the run produced and renumber every figure after it, so the
    render happens here and the figure keeps its slot and its picture.
    """
    queue = _queue(qtbot)
    worker_png = tmp_path / "worker_0.png"
    _fig(0).savefig(worker_png, dpi=60)

    monkeypatch.setattr(fq, "shutil", _ShutilThatCannotMove)
    index = queue.add_figure(_fig(0), prerendered_png=str(worker_png))
    assert worker_png.is_file(), "the move that raised took the file anyway"
    assert queue._ram[index] is not None and not queue._ram[index].isNull(), (
        "a figure whose prerender could not be adopted lost its picture")
    assert not queue._list.item(index).icon().isNull()

    # The same call with a working ``move``: proof the branch above was the
    # failed move and not a queue that renders everything from source.
    monkeypatch.setattr(fq, "shutil", _real_shutil)
    other_png = tmp_path / "worker_1.png"
    _fig(1).savefig(other_png, dpi=60)
    other = queue.add_figure(_fig(1), prerendered_png=str(other_png))
    assert not other_png.exists(), "the prerender was copied, not adopted"
    assert not queue._ram[other].isNull()


def test_a_stale_vector_page_that_will_not_delete_still_lets_the_frame_land(
        qtbot, monkeypatch, tmp_path):
    """The training monitor overwrites one slot every epoch, for hours.

    A replacement raster that arrives without a sibling ``.pdf`` leaves the
    slot's previous page behind, and the refinement would rasterise it and
    paint the superseded epoch back over the new one. Deleting it is what
    keeps the pairing honest -- but a delete that fails must not cost the
    frame, because the raster is what the user is watching.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    stale_page = fq._sibling_pdf(Path(queue._png_paths[0]))
    stale_page.write_bytes(b"%PDF-1.4\n% the previous epoch\n")
    before = queue._ram[0]

    refuse = {"pdf": True}
    real_unlink = Path.unlink

    def _unlink(self, *args, **kwargs):
        # Narrowed to the vector page: a read-only temp directory is the
        # real cause, and everything else in the test still needs to delete.
        if refuse["pdf"] and self.suffix == ".pdf":
            raise PermissionError("the temp directory is read-only")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink)

    epoch_2 = tmp_path / "epoch_2.png"
    _fig(5).savefig(epoch_2, dpi=60)
    queue._refresh_live_figure(0, str(epoch_2))

    assert stale_page.is_file(), "the unlink did not actually fail"
    assert queue._ram[0] is not before, "the new epoch never reached the slot"
    assert not queue._list.item(0).icon().isNull()

    refuse["pdf"] = False
    epoch_3 = tmp_path / "epoch_3.png"
    _fig(7).savefig(epoch_3, dpi=60)
    queue._refresh_live_figure(0, str(epoch_3))
    assert not stale_page.exists(), (
        "the orphaned page survived a delete that was allowed to work")


def test_a_live_frame_that_cannot_be_moved_keeps_the_picture_it_has(
        qtbot, tmp_path):
    """The queue deletes its temp directory on close, under a run.

    The next epoch then has nowhere to land. Keeping the previous frame is
    the right answer: it is a real plot of a real epoch, where a half-moved
    file or a cleared slot is an empty panel under the figure's caption.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    before = queue._ram[0]
    target = Path(queue._png_paths[0])
    queue._delete_tempdir()
    assert not target.parent.exists()

    epoch = tmp_path / "epoch.png"
    _fig(5).savefig(epoch, dpi=60)
    queue._refresh_live_figure(0, str(epoch))

    assert queue._ram[0] is before, "a frame that never arrived was shown"
    assert epoch.is_file(), "the failed move consumed the source file"
    assert queue.count() == 1, "the figure lost its slot"

    # The directory back: proof the refusal was the missing directory.
    target.parent.mkdir(parents=True)
    queue._refresh_live_figure(0, str(epoch))
    assert queue._ram[0] is not before
    assert not epoch.exists()


# ---------------------------------------------------------------------------
# forget_run: the empty-section guard
# ---------------------------------------------------------------------------

def test_a_run_that_drew_nothing_is_never_reported_as_a_section(qtbot):
    """``run_sections`` is the only thing ``forget_run`` reads.

    A run that started and drew nothing leaves a mark, and the mark is real
    -- but a SECTION is only emitted where ``end > start``, so every section
    reported has at least one figure in it. That invariant is what
    ``forget_run`` relies on when it renumbers, and it is why the label of a
    run that drew nothing is simply not found.

    It is also why ``forget_run``'s own ``if count <= 0`` can never fire: the
    only ``count`` it ever sees comes out of the generator above, whose
    ``if end > start`` guarantees ``count >= 1``. The assertion below is
    that guarantee, stated where it is produced rather than where it is
    relied on.
    """
    queue = _queue(qtbot)
    queue.mark_run("first")
    queue.add_figure(_fig(0))
    queue.mark_run("drew nothing")
    queue.mark_run("second")
    queue.add_figure(_fig(1))

    sections = queue.run_sections()
    assert [name for name, _, _ in sections] == ["first", "second"]
    assert all(count >= 1 for _, _, count in sections), (
        "run_sections emitted an empty section; forget_run's renumbering "
        "assumes it cannot")
    assert queue.forget_run("drew nothing") == 0
    assert queue.count() == 2, "a label with no section removed figures"

    assert queue.forget_run("second") == 1
    assert queue.count() == 1
    assert [name for name, _, _ in queue.run_sections()] == ["first"]


# ---------------------------------------------------------------------------
# clear(): closing figures that will not close
# ---------------------------------------------------------------------------

def test_a_figure_that_will_not_close_does_not_stop_the_clear(qtbot):
    """Clear deletes the temp directory, and that must happen.

    ``plt.close`` is what releases matplotlib's own reference; it raises on
    anything that is not a Figure, and a queue can be holding a restored
    spill or a replacement handed in by "Show as ...". A clear that stopped
    there would leave the temp directory on disk and the strip populated
    while the user has just been told the gallery is empty.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    tempdir = queue._tempdir
    # Not a Figure: `plt.close` raises TypeError on it, which is the shape of
    # the failure a stale entry produces.
    queue._figures[1] = object()

    queue.clear()

    assert queue.count() == 0 and queue._list.count() == 0
    assert queue._current == -1
    assert queue._figures == {} and queue._ram == {}
    assert queue._tempdir is None and not tempdir.exists(), (
        "the temp directory outlived the clear")


def test_a_clear_without_pyplot_still_empties_the_gallery(qtbot, monkeypatch):
    """pyplot is imported inside ``clear``, not at module scope.

    It is the one matplotlib import this widget can do without -- the render
    path is pure Agg -- so an environment where importing it fails must still
    be able to clear. The figures are dropped either way; only matplotlib's
    own registry entry is left behind.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    tempdir = queue._tempdir
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    queue.clear()

    assert queue.count() == 0 and queue._list.count() == 0
    assert queue._figures == {}
    assert queue._tempdir is None and not tempdir.exists()


# ---------------------------------------------------------------------------
# The vector-page refinement
# ---------------------------------------------------------------------------

def test_a_store_that_will_not_answer_is_not_pdf_mode(qtbot, monkeypatch,
                                                      prefs):
    """The format decides whether a worker is started per navigation.

    An unreadable store answering "pdf" would dispatch a 2200 px rasterise
    for a page that was never written. False is the safe answer, and the
    positive half here is what proves the guard is reading the store at all.
    """
    prefs.set_figure_format("pdf")
    assert fq.FigureQueue._figure_format_is_pdf() is True

    monkeypatch.setattr(prefs, "get_figure_format", _boom)
    assert fq.FigureQueue._figure_format_is_pdf() is False


def test_an_old_vector_page_is_rasterised_even_though_the_format_is_png(
        qtbot, monkeypatch, prefs):
    """"Dynamic figures" is a promise about pages already on disk.

    A figure drawn while the preference said PDF keeps its vector page when
    the preference is switched to PNG, and once its live Figure has been
    released that page is the only way left to show it sharply -- a
    display-capped raster enlarged to the panel is the "soft" picture the
    option exists to avoid. So the refinement is dispatched on the format
    being PNG, and only for a slot that is genuinely just a picture.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    page = fq._sibling_pdf(Path(queue._png_paths[0]))
    page.write_bytes(b"%PDF-1.4\n% written under the pdf preference\n")
    # No live Figure and no cached pixmap: the state a navigation back to an
    # old figure arrives in.
    assert queue.drop_cache_budget_entry(("figure", 0)) is False  # it is shown
    queue._current = -1
    queue._evict_live_figure(0)
    queue._current = 0
    queue._ram.pop(0, None)
    queue._pdf_state.pop(0, None)

    submitted = []
    monkeypatch.setattr(queue._jobs, "submit",
                        lambda fn, cb=None: submitted.append(fn) or True)

    monkeypatch.setattr(prefs, "get_figure_dynamic", lambda: False)
    queue.show_index(0)
    assert submitted == [], (
        "the crisp render was dispatched with dynamic figures turned off")

    queue._ram.pop(0, None)
    queue._pdf_state.pop(0, None)
    monkeypatch.setattr(prefs, "get_figure_dynamic", lambda: True)
    queue.show_index(0)
    assert len(submitted) == 1, "the vector page on disk was never read"
    assert isinstance(queue._pdf_state.get(0), int), (
        "the slot did not record a render in flight, so a second navigation "
        "would start another one")


def test_only_a_three_part_payload_that_converts_is_a_crisp_render(
        qtbot, monkeypatch):
    """``_on_pdf_rendered`` is a signal handler: it is handed whatever came
    back, including ``None`` from a worker that failed before it built a
    tuple. A failed conversion is recorded as ``"failed"`` so the same broken
    page is not rasterised again on every navigation to this figure.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    view = _RecordingView()
    queue._view = view

    queue._pdf_state[0] = 11
    queue._on_pdf_rendered(None)
    queue._on_pdf_rendered((0, 11))
    assert view.shown == [] and queue._pdf_state[0] == 11, (
        "a payload that is not a result was treated as one")

    monkeypatch.setattr(fq, "QPixmap", _NullConversion)
    queue._on_pdf_rendered((0, 11, _an_image()))
    assert view.shown == []
    assert queue._pdf_state[0] == "failed", (
        "a page that would not convert will be rasterised again on every "
        "visit")

    monkeypatch.setattr(fq, "QPixmap", QPixmap)
    queue._pdf_state[0] = 12
    queue._on_pdf_rendered((0, 12, _an_image()))
    assert len(view.shown) == 1 and queue._pdf_state[0] == "done"


def test_a_crisp_render_for_a_slot_with_no_strip_row_still_reaches_the_view(
        qtbot):
    """The strip is rebuilt independently of the figure store.

    Forgetting a run takes rows out of it while a refinement for the figure
    on screen is still in flight. The picture matters more than its
    thumbnail: the view is repainted and the slot is marked done either way.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    view = _RecordingView()
    queue._view = view
    # blockSignals so removing the row does not also move the current index:
    # the state under test is "the view is on a slot whose row has gone".
    queue._list.blockSignals(True)
    queue._list.takeItem(0)
    queue._list.blockSignals(False)
    assert queue._list.item(0) is None and queue._current == 0

    queue._pdf_state[0] = 3
    queue._on_pdf_rendered((0, 3, _an_image()))

    assert len(view.shown) == 1 and not view.shown[0].isNull()
    assert queue._pdf_state[0] == "done"
    assert 0 in queue._ram


# ---------------------------------------------------------------------------
# The live canvas
# ---------------------------------------------------------------------------

def test_a_canvas_that_cannot_be_built_falls_back_to_the_raster(qtbot):
    """The raster path is the fallback for everything the canvas cannot do.

    ``FigureCanvasQTAgg`` raises on anything that is not a Figure -- a
    restored spill that unpickled into something else, a replacement handed
    in by a plugin -- and a half-built canvas left in the stack would show
    the PREVIOUS figure under this one's caption. False sends the caller to
    the picture instead.
    """
    queue = _queue(qtbot, live=True)
    figure = _fig(0)
    assert queue.show_live_canvas(figure) is True
    assert queue._stack.currentIndex() == 1 and queue._canvas is not None

    assert queue.show_live_canvas(object()) is False
    assert queue._canvas is None, "a half-built canvas was left in place"
    assert queue._canvas_toolbar is None
    assert queue._canvas_layout.count() == 0, (
        "the canvas that WAS up is still in the host, so the panel now shows "
        "the previous figure under this one's caption")


def test_turning_the_live_canvas_back_on_does_not_tear_down_the_raster(qtbot):
    """Only turning it OFF forces the raster view.

    Turning it on is a permission, not a switch: the figure on screen may be
    one that has no Figure left, and dropping it back to a canvas that cannot
    be built for it would blank the panel.
    """
    queue = _queue(qtbot, live=True)
    figure = _fig(0)
    queue.add_figure(figure)
    assert queue._stack.currentIndex() == 1

    queue.set_live_canvas_enabled(False)
    assert queue._live_canvas_enabled is False
    assert queue._stack.currentIndex() == 0
    assert queue.show_live_canvas(figure) is False

    queue.set_live_canvas_enabled(True)
    assert queue._live_canvas_enabled is True
    assert queue._stack.currentIndex() == 0, (
        "enabling the canvas moved the stack on its own")
    assert queue.show_live_canvas(figure) is True
    assert queue._stack.currentIndex() == 1


def test_teardown_disconnects_callbacks_that_point_at_the_dying_widgets(
        qtbot):
    """A callback bound to a widget Qt is about to delete is a crash.

    ``deleteLater`` destroys the C++ side while the figure's callback
    registry -- which the Figure owns and which outlives both widgets --
    still holds a proxy for it. The next mouse move then calls into freed
    memory, once per event. Callbacks belonging to anything else must
    survive: matplotlib keeps its own entries in that registry and the
    figure goes on being used after the canvas has gone.

    Connect through Matplotlib itself so the test exercises the real
    ``WeakMethod`` / ``_StrongRef`` proxy shapes rather than a hand-built
    stand-in that can drift away from the dependency.
    """
    queue = _queue(qtbot, live=True)
    figure = _fig(0)
    assert queue.show_live_canvas(figure) is True
    canvas, toolbar = queue._canvas, queue._canvas_toolbar
    signal = "button_press_event"

    class _Survivor:
        def __call__(self, _event):
            return None

    survivor = _Survivor()
    canvas_cid = canvas.mpl_connect(signal, canvas.draw_idle)
    toolbar_cid = canvas.mpl_connect(signal, toolbar.set_message)
    survivor_cid = canvas.mpl_connect(signal, survivor)

    registry = canvas.callbacks.callbacks
    assert registry[signal][canvas_cid].__class__.__name__ == "WeakMethod"
    assert registry[signal][toolbar_cid].__class__.__name__ == "WeakMethod"
    assert registry[signal][survivor_cid].__class__.__name__ == "_StrongRef"

    queue.set_live_canvas_enabled(False)          # tears the canvas down

    remaining = registry.get(signal, {})
    assert canvas_cid not in remaining, (
        "a canvas-bound callback outlived the canvas")
    assert toolbar_cid not in remaining, (
        "a toolbar-bound callback outlived it too")
    assert remaining.get(survivor_cid) is not None, (
        "a callback belonging to something else was disconnected as well")
    assert queue._canvas is None and queue._canvas_toolbar is None


# ---------------------------------------------------------------------------
# The live preview: the worker draw and what lands from it
# ---------------------------------------------------------------------------

def test_a_figure_that_will_not_copy_is_drawn_on_the_gui_thread(
        qtbot, monkeypatch):
    """The worker draw needs a pickled copy, and not every figure has one.

    A custom artist or a live callback makes ``pickle.dumps`` raise. Falling
    back to a synchronous draw costs a frame; returning False without one
    would leave the control the user is dragging attached to a picture that
    never changes.
    """
    queue = _queue(qtbot)
    stubborn = _fig(0)
    stubborn._spacr_callback = lambda: None      # a lambda cannot be pickled
    queue.add_figure(stubborn)
    before = queue._ram[0]

    submitted = []
    monkeypatch.setattr(queue._jobs, "submit",
                        lambda fn, cb=None: submitted.append(fn) or True)

    assert queue.refresh_current_figure(preview=True) is True
    assert submitted == [], "a figure that will not copy was handed to a worker"
    assert queue._ram[0] is not before, (
        "the inline draw produced nothing, so the restyle never showed")

    # A figure that DOES copy goes to the worker: proof the refusal above was
    # the pickle and not a queue that never uses one.
    queue.add_figure(_fig(1))
    assert queue.refresh_current_figure(preview=True) is True
    assert len(submitted) == 1


def test_a_preview_drawn_off_the_gui_thread_reaches_the_view_and_the_strip(
        qtbot):
    """The whole point of moving the draw off the GUI thread.

    An Agg draw of a real figure is ~110 ms, felt as lag on every control
    change. The worker unpickles its own copy, sizes it to the panel and
    returns a QImage -- QPixmap is GUI-thread-only -- and the result has to
    reach the RAM cache, the view AND the strip thumbnail. Any crisp render
    cached for the slot is of the OLD styling and would repaint over it.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    view = _RecordingView()
    queue._view = view
    before = queue._ram[0]
    queue._pdf_state[0] = "done"
    # An inline runner: the same JobRunner, running the same callable and the
    # same completion handler in the same order, on this thread.
    queue._jobs = JobRunner(queue, threaded=False, app_key="figures")

    assert queue.refresh_current_figure(preview=True) is True

    assert queue._preview_busy is False, "the worker slot was never released"
    assert queue._ram[0] is not before, "the drawn preview was not cached"
    assert len(view.shown) == 1 and not view.shown[0].isNull()
    assert not queue._list.item(0).icon().isNull()
    assert 0 not in queue._pdf_state, (
        "a crisp render of the OLD styling survived and would repaint over "
        "the preview")


def test_only_the_newest_usable_preview_is_painted(qtbot, monkeypatch):
    """A drag emits a change per frame and each one is a draw.

    Four ways a payload is not the one to paint: it is not a result at all,
    it has been superseded, the user navigated away while it drew, and the
    image would not convert to a pixmap. Painting any of them makes the
    figure flicker backwards to a state the controls have already left.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.show_index(0)
    view = _RecordingView()
    queue._view = view
    before = queue._ram[0]
    queue._preview_seq = 6
    queue._pdf_state[0] = "done"

    queue._paint_preview(None)
    queue._paint_preview((0, 6))
    queue._paint_preview((0, 5, _an_image()))      # superseded
    queue._paint_preview((1, 6, _an_image()))      # another figure
    queue._paint_preview((0, 6, None))             # the draw produced nothing
    queue._paint_preview((0, 6, QImage()))         # ... or a null image
    assert view.shown == [] and queue._ram[0] is before
    assert queue._pdf_state[0] == "done", "a refused preview cleared the slot"

    monkeypatch.setattr(fq, "QPixmap", _NullConversion)
    queue._paint_preview((0, 6, _an_image()))
    assert view.shown == [], "an image that would not convert was painted"

    monkeypatch.setattr(fq, "QPixmap", QPixmap)
    queue._paint_preview((0, 6, _an_image()))
    assert len(view.shown) == 1, "the current preview was refused as well"
    assert queue._ram[0] is not before
    assert 0 not in queue._pdf_state


def test_a_preview_for_a_slot_with_no_strip_row_still_paints_the_view(qtbot):
    """Same ledge as the crisp render, reached from the settings dialog.

    A restyle can be in flight when a run is forgotten. The figure the user
    is looking at must still follow the controls; only its thumbnail is lost.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    view = _RecordingView()
    queue._view = view
    queue._list.blockSignals(True)
    queue._list.takeItem(0)
    queue._list.blockSignals(False)
    queue._preview_seq = 2

    queue._paint_preview((0, 2, _an_image()))

    assert len(view.shown) == 1 and not view.shown[0].isNull()
    assert not queue._ram[0].isNull()


def test_a_change_that_arrives_with_no_figure_left_is_dropped(qtbot,
                                                              monkeypatch):
    """A control moved while the draw was running, and then the run was
    cleared. The remembered change has to be forgotten with it: dispatching a
    draw for an index that no longer names a figure spends a worker on
    nothing and leaves ``_preview_busy`` set for good.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    submitted = []
    monkeypatch.setattr(queue._jobs, "submit",
                        lambda fn, cb=None: submitted.append(fn) or True)

    queue._preview_busy = True
    queue._preview_pending = True
    queue._current = -1                       # what `clear()` leaves behind
    queue._on_preview_rendered(None)
    assert queue._preview_busy is False
    assert queue._preview_pending is False, "the change was remembered forever"
    assert submitted == [], "a draw was started for a figure that is gone"

    # The same handler with a figure still there does start the pending draw.
    queue._current = 0
    queue._preview_busy = True
    queue._preview_pending = True
    queue._on_preview_rendered(None)
    assert len(submitted) == 1


# ---------------------------------------------------------------------------
# Measuring what the RAM budget is allowed to reclaim
# ---------------------------------------------------------------------------

def test_a_pixmap_that_cannot_be_measured_sizes_as_nothing(qtbot):
    """The budget divides by these numbers; it must not raise on one.

    Measured without copying the pixels -- width, height and depth -- so
    anything that answers those with nonsense sizes as zero rather than
    taking the whole sweep down with it. Zero is the right answer: an entry
    whose size is unknown must not be chosen for eviction on the strength of
    a guess.
    """
    real = QPixmap(20, 10)
    assert fq.FigureQueue._pixmap_bytes(real) > 0

    assert fq.FigureQueue._pixmap_bytes(None) == 0
    assert fq.FigureQueue._pixmap_bytes(QPixmap()) == 0

    # A stand-in for a pixmap whose accessors answer nonsense; the method is
    # a staticmethod and reads nothing else.
    class _Nonsense:
        def isNull(self):
            return False

        def width(self):
            return "not a number"

        def height(self):
            return 10

        def depth(self):
            return 32

    assert fq.FigureQueue._pixmap_bytes(_Nonsense()) == 0


def test_a_figure_whose_arrays_cannot_be_read_measures_as_its_shell():
    """Sizing a Figure must never draw it or serialise it.

    Every array is read through the artist that owns it, and any of those
    reads can raise -- a canvas Qt has already destroyed, an image whose
    data was released, a collection that answers neither of the two readers.
    The measurement then falls back to the object's own shallow size, which
    is honest: the budget under-counts rather than crashing the sweep that
    every screen's memory policy runs.
    """
    class _DeadCanvas:
        def get_width_height(self):
            raise RuntimeError("Internal C++ object already deleted")

    class _UnreadableLine:
        def get_xdata(self):
            raise RuntimeError("the data was released")

        def get_ydata(self):
            raise AssertionError(
                "unreachable: both readers are in one try block, and "
                "get_xdata raises first")

    class _PlainListLine:
        """Readable, but its data is a list -- there is no ``nbytes``."""

        def get_xdata(self):
            return [0, 1, 2]

        def get_ydata(self):
            return [2, 1, 0]

    class _UnreadableImage:
        def get_array(self):
            raise RuntimeError("the array was released")

    class _UnreadableCollection:
        def get_array(self):
            raise RuntimeError("no array")

        def get_offsets(self):
            raise RuntimeError("no offsets")

    axis = types.SimpleNamespace(
        lines=(_UnreadableLine(), _PlainListLine()),
        images=(_UnreadableImage(),),
        collections=(_UnreadableCollection(),))
    figure = types.SimpleNamespace(canvas=_DeadCanvas(), axes=(axis,))

    measured = fq.FigureQueue._measure_live_figure_bytes(figure)
    assert measured == sys.getsizeof(figure), (
        "something that could not be read was counted anyway")

    # A real figure with real arrays measures MORE than its shell, which is
    # what makes the fallback above a fallback rather than the only answer.
    real = _fig(0)
    real.add_subplot(212).imshow(np.zeros((40, 40), dtype=np.float64))
    assert fq.FigureQueue._measure_live_figure_bytes(real) > sys.getsizeof(real)


def test_the_budget_measures_an_entry_it_has_not_measured_before(qtbot):
    """The policy is handed sizes, not objects, and acts on them.

    Both ledgers are filled when an entry arrives, but an entry can outlive
    its measurement -- ``forget_run`` rebuilds every map, and a restored
    spill is a new object in an old slot. A missing size is measured here
    and remembered, rather than reported as zero and never reclaimed.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue._figure_bytes.pop(0)
    queue._ram_bytes.pop(0)

    rows = dict((key, size) for key, size, _, _ in queue.cache_budget_entries())

    assert rows[("figure", 0)] > 0, "an unmeasured figure sized as nothing"
    assert rows[("pixmap", 0)] > 0, "an unmeasured pixmap sized as nothing"
    assert queue._figure_bytes[0] == rows[("figure", 0)], (
        "the measurement was thrown away and will be redone every sweep")
    assert queue._ram_bytes[0] == rows[("pixmap", 0)]


def test_the_budget_refuses_to_drop_what_is_in_use_or_not_there(qtbot):
    """The sweep evicts whatever this says is droppable, so it must be right.

    The figure on screen is pinned -- dropping it blanks the panel the user
    is looking at. Every editable Figure is pinned while a worker is drawing
    one, because spilling it closes matplotlib state under that worker. And a
    key naming something the queue does not hold has to answer False rather
    than KeyError: the policy holds keys across sweeps.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    assert queue._current == 1

    assert queue.drop_cache_budget_entry(("pixmap", 1)) is False, (
        "the pixmap on screen was offered up")
    assert queue.drop_cache_budget_entry(("figure", 1)) is False

    queue._preview_busy = True
    assert queue.drop_cache_budget_entry(("figure", 0)) is False, (
        "a Figure was spilled while a preview worker was using one")
    assert queue.has_live_figure(0) is True

    queue._preview_busy = False
    assert queue.drop_cache_budget_entry(("figure", 0)) is True
    assert queue.has_live_figure(0) is False
    assert queue.drop_cache_budget_entry(("figure", 0)) is False, (
        "a Figure that has already gone was reported as dropped again")

    assert queue.drop_cache_budget_entry(("pixmap", 0)) is True
    assert 0 not in queue._ram
    assert queue.drop_cache_budget_entry(("pixmap", 0)) is False
    assert queue.drop_cache_budget_entry(("thumbnail", 0)) is False, (
        "an entry kind this queue does not own was accepted")


# ---------------------------------------------------------------------------
# Spilling and restoring
# ---------------------------------------------------------------------------

def test_a_figure_that_cannot_be_spilled_is_released_anyway(qtbot):
    """Failing to spill must never cost the cap.

    The cap exists to stop live Figures accumulating; a spill is how an
    evicted one stays editable. When there is nowhere to write -- the temp
    directory is gone, which is what closing does -- the figure is still
    released and falls back to its rendered page. Keeping it because it
    could not be written would turn a full disk into a memory leak.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))

    # The directory the queue would spill into, gone but still recorded: the
    # state a deleted temp dir leaves a widget that is still being used.
    _real_shutil.rmtree(queue._tempdir)
    assert queue.drop_cache_budget_entry(("figure", 0)) is True
    assert queue.has_live_figure(0) is False
    assert queue.is_restorable(0) is False, "a spill was written after all"

    # And with no temp directory recorded at all there is nowhere to even
    # name a spill file.
    queue._tempdir = None
    assert queue._spill_path(1) is None
    queue._current = 0
    assert queue.drop_cache_budget_entry(("figure", 1)) is True
    assert queue.has_live_figure(1) is False
    assert queue.is_restorable(1) is False


def test_a_spill_that_will_not_unpickle_is_not_a_figure(qtbot):
    """A pickled Figure restores as a REAL Figure -- when it restores.

    A truncated file is what a spill written while the disk filled looks
    like, and unpickling one raises rather than returning something odd.
    ``None`` sends the caller to the picture; anything else would put a
    half-built object in front of the restyling menu.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))

    assert queue.drop_cache_budget_entry(("figure", 0)) is True
    spill = queue._spill_path(0)
    assert spill.is_file(), "nothing was spilled, so nothing is being restored"
    restored = queue.figure_for(0)
    assert isinstance(restored, Figure), "the spill did not come back"

    # Spill it again, then corrupt the file under it.
    queue._current = 1
    assert queue.drop_cache_budget_entry(("figure", 0)) is True
    spill.write_bytes(b"not a pickle at all")

    assert queue.is_restorable(0) is True, "the file is there to be tried"
    assert queue.figure_for(0) is None, (
        "an unreadable spill was handed to the restyling menu")


# ---------------------------------------------------------------------------
# Replacing a figure
# ---------------------------------------------------------------------------

def test_a_figure_past_the_live_window_can_still_be_replaced(qtbot):
    """"Show as violin" builds a NEW Figure, on a slot that may be spilled.

    There is no previous live object to un-index in that case, and the new
    one still has to become the slot's figure -- and its own key in the
    id-to-index map, or the next emit of it opens a second gallery entry.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    assert queue.drop_cache_budget_entry(("figure", 0)) is True
    assert queue.has_live_figure(0) is False

    replacement = _fig(9)
    assert queue.replace_figure(0, replacement) is True
    assert queue.figure_for(0) is replacement
    assert queue._fig_index[id(replacement)] == 0
    assert queue._current == 1, "replacing an off-screen figure moved the view"


def test_replacing_a_figure_looks_for_no_spill_when_there_is_no_temp_dir(
        qtbot, tmp_path):
    """The spill holds a pickle of the OLD figure and must go with it.

    Left behind, a later eviction would restore the picture this call just
    replaced. There is nothing to drop when the temp directory has already
    been deleted -- closing does that -- and the swap still has to happen.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    # A page outside the temp directory, so the redraw can still be written
    # after the directory is gone.
    queue._png_paths[0] = str(tmp_path / "kept.png")
    queue._delete_tempdir()
    assert queue._spill_path(0) is None

    replacement = _fig(9)
    assert queue.replace_figure(0, replacement) is True
    assert queue.figure_for(0) is replacement
    assert (tmp_path / "kept.png").is_file(), "the replacement was not drawn"


# ---------------------------------------------------------------------------
# Reading a raster back off disk
# ---------------------------------------------------------------------------

def test_a_raster_qt_cannot_open_says_so_instead_of_opening_blank(qtbot):
    """Reported as "i can not see #2 when i click on it this run".

    A file that is there but is not a picture loads as a null pixmap. Caching
    it would make the empty panel permanent for the slot, and leaving the
    view untouched is worse than empty -- it keeps the PREVIOUS figure, so
    clicking figure 2 shows figure 1 and the user reads it as figure 2.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    page = Path(queue._png_paths[0])
    view = _RecordingView()
    queue._view = view

    queue._ram.pop(0)
    page.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    queue.show_index(0)

    assert 0 not in queue._ram, "an unreadable raster was cached"
    assert len(view.shown) == 1 and not view.shown[0].isNull()
    assert view.shown[0].size().toTuple() == (720, 360), (
        "the panel opened blank instead of explaining itself")
    assert "could not be read" in queue._why_not_shown(0)
    assert queue.all_pixmaps() == [None]

    # A readable page in the same slot: proof the refusal was the file.
    _fig(0).savefig(page, dpi=60)
    queue.show_index(0)
    assert not queue._ram[0].isNull()
    assert len(view.shown) == 2 and view.shown[1].size().toTuple() != (720, 360)


# ---------------------------------------------------------------------------
# Shutting down
# ---------------------------------------------------------------------------

def test_a_runner_that_will_not_stop_does_not_stop_the_clear(qtbot):
    """Clear deletes the temp directory a worker is reading out of.

    The runner is asked to stop first, and it can refuse -- ``JobRunner`` is
    reached from ``__del__`` too, where the C++ half may already be gone. The
    directory still has to be removed and the gallery still has to empty, or
    a failing shutdown leaks a temp directory per run.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    tempdir = queue._tempdir
    calls = []

    class _RunnerThatWillNotStop:
        def shutdown(self, timeout_ms=0):
            calls.append(timeout_ms)
            raise RuntimeError("Internal C++ object already deleted")

    queue._jobs = _RunnerThatWillNotStop()
    queue.clear()

    assert calls == [2000], "the runner was never asked to stop"
    assert queue.count() == 0 and queue._tempdir is None
    assert not tempdir.exists()

    # A widget whose runner was never built at all -- ``__del__`` reaches
    # this on a queue whose construction did not finish.
    queue.add_figure(_fig(1))
    second_tempdir = queue._tempdir
    del queue._jobs
    queue.clear()
    assert calls == [2000], "the missing runner was called anyway"
    assert queue.count() == 0 and not second_tempdir.exists()


def test_every_teardown_step_runs_even_when_the_one_before_it_raises(qtbot):
    """``__del__`` is the last chance to remove the temp directory.

    It runs during interpreter teardown and on a widget Qt may already have
    half-destroyed, so each of the three steps can raise. They are separately
    guarded for exactly that reason: a canvas that will not tear down must
    not cost the directory removal, which is the only one of the three that
    leaks something the user can see.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    calls = []

    def _raiser(name):
        def _step(*_args, **_kwargs):
            calls.append(name)
            raise RuntimeError(f"{name} is already gone")
        return _step

    # Plain setattr with an explicit restore rather than ``monkeypatch``: the
    # widget's own closeEvent calls two of these three when qtbot retires it,
    # and it does so before a monkeypatch registered in this test is undone.
    steps = ("_teardown_canvas", "_shutdown_jobs", "_delete_tempdir")
    for step in steps:
        setattr(queue, step, _raiser(step.strip("_").split("_")[0]))
    try:
        queue.__del__()
    finally:
        for step in steps:
            delattr(queue, step)

    assert calls == ["teardown", "shutdown", "delete"], (
        "a step that raised stopped the ones after it")
    assert queue._tempdir is not None and queue._tempdir.exists(), (
        "the temp directory went away, so the third step did not raise")


# ---------------------------------------------------------------------------
# _FigureSettingsDialog: what it does with no store behind it
# ---------------------------------------------------------------------------

def _payload() -> dict:
    """The Image UMAP payload a run leaves on its figure."""
    rng = np.random.default_rng(7)
    embedding = rng.normal(size=(24, 2))
    labels = np.arange(24) % 3
    return {
        "embedding": embedding,
        "labels": labels,
        "plot_labels": labels,
        "records": [{"image": None, "display_name": str(i)} for i in range(24)],
        "display": {},
        "settings": {
            "dot_size": 33, "point_color": "cluster", "point_alpha": 0.5,
            "outline_width": 1.0, "figuresize": 5.0, "image_nr": 7,
            "img_zoom": 0.4, "plot_images": False, "plot_points": True,
            "plot_outlines": False, "smooth_lines": False,
            "n_neighbors": 42, "min_dist": 0.2, "metric": "euclidean",
            "clustering": "dbscan", "black_background": False,
        },
        "theme_colors": None,
    }


def _umap_figure(payload):
    figure = Figure(figsize=(4.0, 4.0))
    axes = figure.subplots()
    axes.scatter(payload["embedding"][:, 0], payload["embedding"][:, 1],
                 s=33, c=payload["labels"])
    figure._spacr_umap_payload = payload
    return figure


def test_the_dialog_opens_on_auto_when_the_store_cannot_be_read(
        qtbot, monkeypatch, prefs):
    """Every colour read is a token, and a failed read must not invent one.

    The dialog WRITES ITS SEED BACK on OK. Seeding from a resolved colour is
    what froze "#ffffff" into every future figure (instruction 152 A), and
    seeding a size of 10 into a store that holds 0 is issue #108's "the font
    size is by default too large" -- so the fallback is the token and a
    stored size of zero, with 10 only SHOWN.
    """
    prefs.set_figure_text_size(17)
    seeded = fq._FigureSettingsDialog(_fig(0))
    qtbot.addWidget(seeded)
    assert seeded._stored_size == 17 and seeded._size.value() == 17

    monkeypatch.setattr(prefs, "get_figure_color_tokens", _boom)
    dialog = fq._FigureSettingsDialog(_fig(0))
    qtbot.addWidget(dialog)

    assert (dialog._bg, dialog._fg, dialog._line) == ("auto", "auto", "auto")
    assert dialog._stored_size == 0, "an unreadable store seeded a real size"
    assert dialog._size.value() == 10
    assert dialog._size_touched is False
    assert dialog._auto_btn.isEnabled() is False, (
        "'Follow the theme' offered work it would not do")


def test_the_colour_helpers_answer_without_the_preference_module(
        monkeypatch, prefs):
    """Both helpers are read on every repaint of the three colour buttons.

    Raising there would leave the dialog half-built. The fallbacks have to be
    the same answers the store gives: "auto" is the token, and everything
    else is a colour.
    """
    monkeypatch.setattr(prefs, "figure_color_is_auto", lambda token: True)
    assert fq._FigureSettingsDialog._is_auto("#ff0000") is True, (
        "the helper is not reading the preference module at all")

    monkeypatch.setattr(prefs, "figure_color_is_auto", _boom)
    assert fq._FigureSettingsDialog._is_auto("#ff0000") is False
    assert fq._FigureSettingsDialog._is_auto("  AUTO ") is True

    monkeypatch.setattr(prefs, "auto_figure_colors",
                        lambda: ("#123456", "#abcdef"))
    assert fq._FigureSettingsDialog._auto_preview() == ("#123456", "#abcdef")

    monkeypatch.setattr(prefs, "auto_figure_colors", _boom)
    assert fq._FigureSettingsDialog._auto_preview() == ("none", "#000000")


def test_the_line_colour_follows_the_font_only_while_it_is_automatic(qtbot):
    """Instruction 152 B: the axes and the labels are two controls.

    "Automatic" on the line half means "the same as the text", which is what
    a figure looked like before the split -- so a store nobody has touched
    renders exactly as it did. A line colour the user actually chose must
    survive a font colour that is something else entirely.
    """
    dialog = fq._FigureSettingsDialog(_fig(0))
    qtbot.addWidget(dialog)
    dialog._fg = "#112233"

    dialog._line = "auto"
    assert dialog._resolved_line() == "#112233"

    dialog._line = "#445566"
    assert dialog._resolved_line() == "#445566", (
        "a chosen line colour was overwritten by the font colour")


def test_picking_a_colour_starts_from_the_one_shown_and_honours_cancel(
        qtbot, monkeypatch):
    """The picker opens on what the button is showing, not on "auto".

    A dialog that opened on black every time would make the current colour
    unfindable. And Cancel must leave the token alone: an invalid QColor is
    what a cancelled picker returns, and writing it would turn "no change"
    into a colour nobody chose.
    """
    from spacr.qt.widgets import colour_picker

    dialog = fq._FigureSettingsDialog(_fig(0))
    qtbot.addWidget(dialog)
    dialog._fg = "#00ff00"
    dialog._paint_colour_buttons()
    seen = {}

    def _picked(parent, current, title):
        seen["current"], seen["title"] = current, title
        return QColor("#123456")

    monkeypatch.setattr(colour_picker, "pick_colour", _picked)
    dialog._pick("_fg", dialog._fg_btn)

    assert seen["current"] == "#00ff00", (
        "the picker opened on the theme's answer rather than the colour shown")
    assert seen["title"] == "Font colour"
    assert dialog._fg == "#123456"
    assert dialog._fg_btn.text() == "#123456"

    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *a, **k: QColor())     # a cancelled picker
    dialog._pick("_fg", dialog._fg_btn)
    assert dialog._fg == "#123456", "Cancel changed the colour"


def test_ok_restyles_the_figure_even_when_the_store_refuses_the_write(
        qtbot, monkeypatch, prefs):
    """The figure in front of the user is the point; the store is the memory.

    A read-only or missing INI must not swallow the restyle -- the user
    pressed OK on a colour they can see, and reporting nothing while doing
    nothing is the failure this guard exists to avoid.
    """
    figure = _fig(0)
    dialog = fq._FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)
    dialog._bg, dialog._fg, dialog._line = "#101010", "#eeeeee", "#00ff00"
    dialog._apply_and_accept()
    assert prefs.get_figure_color_tokens() == ("#101010", "#eeeeee"), (
        "the tokens were not persisted, so the store is not being written")

    other = _fig(1)
    broken = fq._FigureSettingsDialog(other)
    qtbot.addWidget(broken)
    broken._bg, broken._fg, broken._line = "#202020", "#dddddd", "#ff0000"
    monkeypatch.setattr(prefs, "set_figure_colors", _boom)
    broken._apply_and_accept()

    assert broken.result() == QDialog.Accepted, "OK did not close the window"
    assert other.patch.get_facecolor() == matplotlib.colors.to_rgba("#202020"), (
        "the figure was not restyled when the store refused the write")


def test_ok_flushes_an_image_umap_value_still_on_the_debounce_timer(
        qtbot, prefs):
    """Every UMAP edit is debounced by 250 ms before it reaches the figure.

    A user who types a value and presses OK promptly is inside that window,
    and without the flush their last edit is dropped by the very button that
    means "keep this".
    """
    payload = _payload()
    figure = _umap_figure(payload)
    dialog = fq._FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)
    assert dialog._umap_settings is not None

    emitted = []
    dialog._umap_settings.settings_changed.connect(emitted.append)
    # No public setter for one field; the editors are what the user types in.
    dialog._umap_settings._editors["dot_size"].setValue(77)
    assert emitted == [], "the change reached the figure without the debounce"

    dialog._apply_and_accept()

    assert len(emitted) == 1, "a typed value was lost by pressing OK"
    assert emitted[0]["dot_size"] == 77


def test_a_umap_change_survives_a_re_render_that_raises(qtbot):
    """The figure is styled first and re-rasterised second.

    The re-render is the queue's, and it fails for the ordinary reasons a
    render fails -- no temp directory, a dead canvas. The setting has already
    landed on the figure by then, so forgetting it would make the next change
    compute its delta against a state the figure is not in.
    """
    payload = _payload()
    figure = _umap_figure(payload)
    rendered = []

    def _render():
        rendered.append(True)
        raise RuntimeError("the temp directory has gone")

    dialog = fq._FigureSettingsDialog(figure, render_callback=_render)
    qtbot.addWidget(dialog)
    dialog._umap_applied = {}

    dialog._on_umap_changed({"dot_size": 77})

    assert rendered == [True], "the figure was never asked to re-render"
    assert dialog._umap_applied == {"dot_size": 77}, (
        "a failed re-render lost the setting that had already been applied")


def test_propagate_needs_a_callback_and_survives_one_that_raises(qtbot):
    """The button is offered on every figure, and owned by a module screen.

    A queue built outside one has no settings panel to write into, so the
    button says so instead of doing nothing. And the callback runs foreign
    code -- a screen writing into its own widgets -- which can raise; the
    settings window must not go down with it.
    """
    orphan = fq._FigureSettingsDialog(_fig(0))
    qtbot.addWidget(orphan)
    assert orphan._propagate_btn.isEnabled() is False
    assert "module screen" in orphan._propagate_btn.toolTip()
    orphan._propagate()                     # the button is reachable anyway

    sent = []
    wired = fq._FigureSettingsDialog(_fig(0), propagate_callback=sent.append)
    qtbot.addWidget(wired)
    wired._bg, wired._fg, wired._line = "#101010", "#eeeeee", "#00ff00"
    wired._propagate()
    assert len(sent) == 1
    assert sent[0]["figure_background"] == "#101010"
    assert sent[0]["figure_line_color"] == "#00ff00"

    attempted = []

    def _explodes(values):
        attempted.append(values)
        raise RuntimeError("the settings panel is gone")

    broken = fq._FigureSettingsDialog(_fig(0), propagate_callback=_explodes)
    qtbot.addWidget(broken)
    broken._propagate()
    assert len(attempted) == 1, "the callback was never called"


def test_cancel_puts_back_only_a_umap_setting_that_actually_moved(qtbot):
    """Cancel costs one redraw, and a redraw of a montage is not free.

    Live apply with no way out is a trap, so Cancel restores what the window
    opened on -- but only when something moved. Replotting an untouched
    figure on every Cancel is a tenth of a second spent producing the picture
    that is already there.
    """
    payload = _payload()
    figure = _umap_figure(payload)
    untouched = fq._FigureSettingsDialog(figure)
    qtbot.addWidget(untouched)
    restored = []
    untouched._on_umap_changed = lambda values: restored.append(values)

    untouched.reject()
    assert restored == [], "an untouched figure was replotted on Cancel"

    changed = fq._FigureSettingsDialog(_umap_figure(_payload()))
    qtbot.addWidget(changed)
    put_back = []
    initial = changed._umap_settings.initial_values()
    changed._on_umap_changed = lambda values: put_back.append(values)
    changed._umap_applied = dict(initial, dot_size=77)

    changed.reject()
    assert len(put_back) == 1, "a changed figure was left changed by Cancel"
    assert put_back[0] == initial
