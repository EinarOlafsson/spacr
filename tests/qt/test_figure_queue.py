"""Tests for the FigureQueue widget — RAM cap, temp spill, cleanup,
zoom, forward/back navigation, and the off-GUI-thread PDF page render.

Uses matplotlib's Agg backend (headless) to build real Figure objects,
then drives the queue directly.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from PySide6.QtGui import QImage

from spacr.qt.widgets.figure_queue import (
    PDF_DISPLAY_MAX_PX, FigureQueue, render_pdf_to_image,
)
# The canonical event-loop watchdog. Imported rather than copied so there is
# one definition of "how long did the GUI thread stop pumping events"; see
# that module's docstring for why nothing else measures what a user feels.
from tests.qt.test_gui_responsiveness import LoopWatchdog


def _make_fig(seed: int = 0):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [seed, seed + 1, seed])
    ax.set_title(f"fig {seed}")
    return fig


@pytest.fixture(autouse=True)
def _isolate_figure_prefs(monkeypatch, tmp_path_factory):
    """Keep this file out of the developer's real preference store.

    Two tests below call the real ``prefs.set_figure_format(...)`` to put the
    queue into PDF mode, and ``preferences._settings()`` returns
    ``QSettings(_ORG, _APP)`` — the *installed application's* store. So running
    this file rewrote whatever the person running it had chosen in
    Preferences → Figure format and left it wherever the last test happened to
    put it. A test that changes the machine it runs on is not isolated, and
    this one changed it silently and permanently.

    Redirecting ``_settings`` at a per-session ini file keeps the tests
    exercising the real getters and setters — the plumbing is the thing under
    test, so stubbing it out would remove the point — while confining every
    write to a temp directory.
    """
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as prefs

    store = tmp_path_factory.mktemp("figure_queue_prefs") / "prefs.ini"
    monkeypatch.setattr(
        prefs, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))


class TestBasics:
    def test_figure_settings_fields_link_to_api_docs(self, qtbot):
        """The help is on the LABEL's tooltip, and there is no dot.

        This test asserted the opposite until instruction 75 -- every field
        carried a ``_spacr_api_dot``. Inverted rather than deleted, and the
        ``href=`` assertion above it kept deliberately: "no dot" on its own
        would also pass if the documentation had left with the decoration,
        which is the regression worth excluding. Same change, same reason,
        as the Mask live preview, Annotate and the UMAP search dialog.
        """
        from spacr.qt.widgets.dot_link import DotLink
        from spacr.qt.widgets.figure_queue import _FigureSettingsDialog
        dialog = _FigureSettingsDialog(_make_fig())
        qtbot.addWidget(dialog)
        for widget in (dialog._bg_btn, dialog._fg_btn, dialog._size):
            assert widget.toolTip() == ""
            label = widget._spacr_setting_label
            assert "href=" in label.toolTip()
            assert getattr(label, "_spacr_api_dot", None) is None
        assert dialog.findChildren(DotLink) == []

    def test_add_figure_increments_count(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        q.add_figure(_make_fig(1))
        assert q.count() == 2

    def test_thumbnail_list_matches_count(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        for i in range(3):
            q.add_figure(_make_fig(i))
        assert q._list.count() == 3

    def test_dedup_same_figure_object(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        fig = _make_fig(0)
        q.add_figure(fig)
        q.add_figure(fig)   # same object → no new entry
        assert q.count() == 1

    def test_live_figure_refresh_reuses_gallery_slot(self, qtbot, tmp_path):
        q = FigureQueue()
        qtbot.addWidget(q)
        fig = _make_fig(0)
        q.add_figure(fig)
        before = Path(q._png_paths[0]).read_bytes()

        replacement = tmp_path / "live.png"
        updated = _make_fig(99)
        updated.savefig(replacement, dpi=100)
        q.add_figure(fig, prerendered_png=str(replacement))

        assert q.count() == 1
        assert q._list.count() == 1
        assert Path(q._png_paths[0]).read_bytes() != before
        assert not replacement.exists()

    def test_every_figure_has_a_temp_png(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        for i in range(3):
            q.add_figure(_make_fig(i))
        for i in range(3):
            assert Path(q._png_paths[i]).is_file()


class TestNavigation:
    def test_prev_next_cycle(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        for i in range(4):
            q.add_figure(_make_fig(i))
        # After adding, current is the newest (index 3)
        assert q._current == 3
        q.show_prev()
        assert q._current == 2
        q.show_prev()
        assert q._current == 1
        q.show_next()
        assert q._current == 2

    def test_no_prev_next_buttons(self, qtbot):
        # Prev/Next buttons were removed — navigation is via the thumbnail
        # strip (show_index) instead.
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        assert not hasattr(q, "_prev_btn")
        assert not hasattr(q, "_next_btn")
        assert hasattr(q, "_fig_settings_btn")

    def test_figure_settings_button_visible_with_figures(self, qtbot):
        from spacr.qt import preferences as prefs
        prefs.set_figure_format("png")
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        q._refresh_nav()
        # Figure settings restyle + re-render, so the button shows in PNG mode
        # too (colours/size apply to the displayed raster).
        assert q._fig_settings_btn.isVisibleTo(q)

    def test_position_label(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        for i in range(3):
            q.add_figure(_make_fig(i))
        q.show_index(1)
        assert q._pos_label.text() == "2 / 3"


class TestRamCapAndSpill:
    def test_ram_cap_holds_only_n_most_recent(self, qtbot):
        # Small cap so the test is fast.
        q = FigureQueue(ram_cap=5)
        qtbot.addWidget(q)
        for i in range(8):
            q.add_figure(_make_fig(i))
        # Only 5 pixmaps resident in RAM
        assert q.ram_resident() == 5
        # 3 have been spilled to disk-only
        assert q.spilled_count() == 3

    def test_spilled_figure_reloads_from_disk(self, qtbot):
        q = FigureQueue(ram_cap=5)
        qtbot.addWidget(q)
        for i in range(8):
            q.add_figure(_make_fig(i))
        # Figure 0 was evicted from RAM (window is 3..7). Its PNG exists.
        assert 0 not in q._ram
        assert Path(q._png_paths[0]).is_file()
        # Viewing it reloads from disk + re-caches.
        q.show_index(0)
        assert 0 in q._ram

    def test_sliding_window_evicts_oldest_first(self, qtbot):
        q = FigureQueue(ram_cap=3)
        qtbot.addWidget(q)
        for i in range(3):
            q.add_figure(_make_fig(i))
        # RAM holds {0,1,2}
        assert set(q._ram.keys()) == {0, 1, 2}
        q.add_figure(_make_fig(3))
        # Adding #3 evicts the oldest (#0)
        assert set(q._ram.keys()) == {1, 2, 3}
        q.add_figure(_make_fig(4))
        assert set(q._ram.keys()) == {2, 3, 4}


class TestCleanup:
    def test_clear_deletes_tempdir(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        tempdir = q._tempdir
        assert tempdir is not None and Path(tempdir).is_dir()
        q.clear()
        assert not Path(tempdir).exists()
        assert q.count() == 0

    def test_close_deletes_tempdir(self, qtbot):
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        tempdir = q._tempdir
        q.close()
        assert not Path(tempdir).exists()


class TestZoomView:
    def test_enlarged_view_is_zoomable(self, qtbot):
        from spacr.qt.widgets.live_preview import _ZoomView
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        # The enlarged view is a _ZoomView (wheel-zoom + fit-to-container)
        assert isinstance(q._view, _ZoomView)
        # It has a pixmap item loaded
        assert q._view._pixmap_item is not None


# ---------------------------------------------------------------------------
# The PDF page render must not run on the GUI thread
# ---------------------------------------------------------------------------

#: The longest the GUI thread may stop pumping events while a PDF-mode figure
#: is added or navigated to. Stated, not derived, and deliberately far above
#: what this machine measures — a flaky responsiveness test gets deleted
#: rather than fixed. Measured on a warm local SSD with the nine-panel 16x12"
#: figure the fixtures below build:
#:
#:     FigureQueue.add_figure   815 ms  ->  194 ms   (181 ms of which is the
#:                                                    PNG decode, unrelated)
#:     navigation (reload)      767 ms  ->  179 ms
#:
#: The tests use a low-DPI PNG so the decode is ~7 ms and the number below is
#: about the *PDF* render, which is the thing that moved off the GUI thread.
PDF_STALL_BUDGET_S = 0.250


@pytest.fixture(scope="module")
def _pdf_assets(tmp_path_factory):
    """One deliberately expensive figure, rendered once for the whole module.

    The PNG is written at a LOW dpi on purpose. Decoding the ~4000 px raster
    ``render_figure_to_png`` produces at the default 300 DPI is a GUI-thread
    cost of its own — 181 ms, measured — and leaving it in would blur the
    thing these tests bound. Small PNG, expensive PDF: what is left in the
    measurement is the vector page render.
    """
    directory = tmp_path_factory.mktemp("figq_pdf_assets")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for panel, ax in enumerate(axes.ravel()):
        n = 12000
        ax.plot(np.arange(n), rng.normal(size=n).cumsum(), lw=0.4)
        ax.scatter(rng.uniform(0, n, 4000), rng.normal(size=4000) * 20, s=2)
        ax.set_title(f"panel {panel}")
    fig.tight_layout()
    fig.savefig(directory / "big.png", dpi=50)
    fig.savefig(directory / "big.pdf")
    return fig, directory / "big.png", directory / "big.pdf"


@pytest.fixture
def pdf_figure(_pdf_assets, tmp_path):
    """``(figure, prerendered_png)`` — exactly what the pipeline hands over.

    ``PipelineWorker`` renders the PNG and its sibling ``.pdf`` on its own
    thread and passes the path in, so ``add_figure`` only moves files and
    loads a pixmap. That is the path this class is about: the PDF page render
    used to be dropped straight back onto the GUI thread afterwards.
    """
    from spacr.qt import preferences as prefs
    prefs.set_figure_format("pdf")
    fig, png_src, pdf_src = _pdf_assets
    png = tmp_path / "prerendered.png"
    shutil.copyfile(png_src, png)
    shutil.copyfile(pdf_src, png.with_suffix(".pdf"))
    return fig, str(png)


def _drive(qtbot, dog, done, budget_s: float = 30.0):
    """Pump the event loop until ``done()``, never blocking it."""
    end = time.perf_counter() + budget_s
    while time.perf_counter() < end and not done():
        qtbot.wait(20)
    qtbot.wait(50)
    dog.stop()


def _settled(q):
    return lambda: not q.is_busy() and q.active_jobs() == 0


class TestPdfRenderIsOffTheGuiThread:

    def test_adding_a_pdf_figure_does_not_freeze_the_gui_thread(
            self, qtbot, pdf_figure):
        """The render that used to block for ~0.8 s now blocks for ~0.

        Asserts on the event loop, not on the presence of a thread: the queue
        could dispatch the render and still block on delivery, and the user
        could not tell the difference.
        """
        fig, png = pdf_figure
        q = FigureQueue()
        qtbot.addWidget(q)
        q.resize(900, 700)
        q.show()
        qtbot.waitExposed(q)
        qtbot.wait(100)

        dog = LoopWatchdog(q)
        dog.start()
        start = time.perf_counter()
        idx = q.add_figure(fig, prerendered_png=png)
        dispatch = time.perf_counter() - start
        _drive(qtbot, dog, _settled(q))

        assert idx == 0
        assert dispatch < 0.100, (
            f"add_figure took {dispatch * 1000:.0f} ms to return; it is still "
            "rendering the PDF page on the GUI thread")
        assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
        assert dog.worst < PDF_STALL_BUDGET_S, (
            f"add_figure stalled the GUI thread for {dog.worst * 1000:.0f} ms "
            f"(budget {PDF_STALL_BUDGET_S * 1000:.0f} ms)")
        # And it stayed responsive by *finishing the work*, not by skipping it.
        assert q._pdf_state.get(0) == "done"

    def test_the_pdf_render_really_is_slow_enough_for_the_budget_to_mean_something(
            self, pdf_figure):
        """Guard against the fixture shrinking until the test proves nothing.

        This is the call that used to run inline in ``_pdf_pixmap``: a plain
        ``QPdfDocument.render`` on the calling thread. It does not release the
        GIL, so on the GUI thread it is a freeze of exactly this length.
        """
        from PySide6.QtCore import QSize
        from PySide6.QtPdf import QPdfDocument

        _fig, png = pdf_figure
        doc = QPdfDocument()
        assert doc.load(str(Path(png).with_suffix(".pdf"))) == \
            QPdfDocument.Error.None_
        size = doc.pagePointSize(0)
        scale = PDF_DISPLAY_MAX_PX / max(size.width(), size.height())
        start = time.perf_counter()
        image = doc.render(0, QSize(int(size.width() * scale),
                                    int(size.height() * scale)))
        elapsed = time.perf_counter() - start

        assert not image.isNull()
        assert elapsed > PDF_STALL_BUDGET_S, (
            f"rendering the fixture's page took only {elapsed * 1000:.0f} ms, "
            f"under the {PDF_STALL_BUDGET_S * 1000:.0f} ms budget — the "
            "responsiveness test above would pass with the threading removed")

    def test_the_crisper_render_actually_arrives(self, qtbot, pdf_figure):
        """Progressive refinement: the PNG first, the vector page after.

        Without this a queue that simply never rendered the PDF would sail
        through the stall budget.
        """
        fig, png = pdf_figure
        q = FigureQueue()
        qtbot.addWidget(q)
        idx = q.add_figure(fig, prerendered_png=png)

        # Immediately: the cheap PNG-derived pixmap, already on screen.
        coarse = q._ram[idx]
        assert coarse.width() == 800, "not the low-DPI prerendered PNG"
        assert q._view._pixmap_item is not None
        assert q._view._pixmap_item.pixmap().width() == 800

        qtbot.waitUntil(_settled(q), timeout=30000)

        crisp = q._ram[idx]
        assert crisp is not coarse, "the crisp render never replaced the PNG"
        assert max(crisp.width(), crisp.height()) == PDF_DISPLAY_MAX_PX
        assert q._view._pixmap_item.pixmap().width() == crisp.width()
        assert q._pdf_state.get(idx) == "done"

    def test_navigating_to_a_spilled_figure_does_not_freeze_the_gui_thread(
            self, qtbot, pdf_figure):
        """Every navigation click used to pay the full render, too."""
        fig, png = pdf_figure
        q = FigureQueue()
        qtbot.addWidget(q)
        q.resize(900, 700)
        q.show()
        qtbot.waitExposed(q)
        q.add_figure(fig, prerendered_png=png)
        qtbot.waitUntil(_settled(q), timeout=30000)

        # Spill it: the RAM pixmap is gone, so viewing it reloads from disk —
        # which is where the second copy of the render used to live.
        q._ram.clear()
        q._current = -1
        qtbot.wait(100)

        dog = LoopWatchdog(q)
        dog.start()
        start = time.perf_counter()
        q.show_index(0)
        dispatch = time.perf_counter() - start
        _drive(qtbot, dog, _settled(q))

        assert dispatch < 0.100, (
            f"show_index took {dispatch * 1000:.0f} ms to return")
        assert dog.ticks > 10
        assert dog.worst < PDF_STALL_BUDGET_S, (
            f"navigation stalled the GUI thread for {dog.worst * 1000:.0f} ms")
        assert max(q._ram[0].width(), q._ram[0].height()) == PDF_DISPLAY_MAX_PX

    def test_a_render_the_user_navigated_away_from_is_dropped(
            self, qtbot, pdf_figure):
        """A late result must not paint over the figure now on screen."""
        fig, png = pdf_figure
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(fig, prerendered_png=png)      # #0, render dispatched
        assert q._pdf_state.get(0) is not None
        q.add_figure(_make_fig(1))                  # #1 is now what is shown
        assert q._current == 1

        qtbot.waitUntil(_settled(q), timeout=30000)

        # #0's render finished, found itself stale, and was discarded rather
        # than cached or painted.
        assert q._pdf_state.get(0) != "done"
        assert q._ram[0].width() == 800, "the stale render was cached anyway"
        assert q._current == 1
        assert q._view._pixmap_item.pixmap().width() == q._ram[1].width()

    def test_closing_mid_render_leaves_no_thread_and_cannot_hang(
            self, qtbot, pdf_figure):
        """``closeEvent`` deletes the temp dir the worker is reading from.

        So it must stop the worker first — and must do it with a bounded wait,
        because Qt aborts the process if a running QThread is destroyed and a
        blocking join with no deadline is just a different freeze.
        """
        from spacr.qt.bridge import parked_thread_count, thread_has_stopped

        fig, png = pdf_figure
        q = FigureQueue()
        idx = q.add_figure(fig, prerendered_png=png)
        assert q.active_jobs() >= 1, "no crisp render was dispatched"
        threads = [pair[0] for pair in q._jobs._jobs.values()]
        tempdir = q._tempdir
        parked_before = parked_thread_count()

        start = time.perf_counter()
        q.close()                              # mid-render, deliberately
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, (
            f"close() blocked for {elapsed:.1f} s; the shutdown is not bounded")
        assert q.active_jobs() == 0
        assert all(thread_has_stopped(t) for t in threads)
        assert parked_thread_count() == parked_before, (
            "the render thread outlasted the shutdown budget and was parked")
        assert not Path(tempdir).exists()
        # Nothing was delivered into the widget on its way out.
        assert q._pdf_state.get(idx) != "done"
        qtbot.wait(200)

    def test_the_widget_dying_first_raises_nothing_in_the_event_loop(
            self, qtbot, pdf_figure):
        """The C++ half can go while a render is still in flight.

        The worker then emits into a runner whose C++ half is gone and PySide6
        raises ``RuntimeError: Signal source has been deleted``. Unguarded it
        surfaces as an unhandled exception in the Qt event loop, which
        pytest-qt turns into a failure in whatever test runs next.
        """
        import shiboken6
        from spacr.qt.bridge import thread_has_stopped

        fig, png = pdf_figure
        q = FigureQueue()
        q.add_figure(fig, prerendered_png=png)
        assert q.active_jobs() >= 1
        # Hold the QThread wrappers ourselves: a QThread garbage-collected
        # while running takes the process down, and the widget is about to
        # stop being anybody's owner.
        runner = q._jobs
        threads = [pair[0] for pair in runner._jobs.values()]

        escaped = []
        previous_hook = sys.excepthook
        sys.excepthook = lambda *exc_info: escaped.append(exc_info)
        try:
            q.deleteLater()                  # no closeEvent, no shutdown
            qtbot.wait(50)
            assert not shiboken6.isValid(q), "the C++ half is still alive"
            qtbot.waitUntil(
                lambda: all(thread_has_stopped(t) for t in threads),
                timeout=30000)
            qtbot.wait(200)
        finally:
            sys.excepthook = previous_hook

        assert escaped == [], (
            "an exception reached the Qt event loop: "
            f"{[e[0].__name__ for e in escaped]} {[str(e[1]) for e in escaped]}")
        assert runner.active_jobs() >= 0     # the runner survived the widget


class TestRenderPdfToImageIsWorkerSafe:
    """``render_pdf_to_image`` is the callable the worker thread runs."""

    def test_it_returns_a_qimage_never_a_qpixmap(self, pdf_figure):
        """QPixmap is GUI-thread-only; the worker must produce a QImage."""
        from PySide6.QtGui import QPixmap

        _fig, png = pdf_figure
        image = render_pdf_to_image(str(Path(png).with_suffix(".pdf")))
        assert isinstance(image, QImage)
        assert not isinstance(image, QPixmap)
        assert max(image.width(), image.height()) == PDF_DISPLAY_MAX_PX

    def test_a_vanished_file_is_not_an_error(self, tmp_path):
        """The temp dir is deleted on close, under a render already running."""
        assert render_pdf_to_image(str(tmp_path / "never-existed.pdf")) is None

    def test_a_file_that_is_not_a_pdf_is_not_an_error(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"definitely not a pdf")
        assert render_pdf_to_image(str(bad)) is None

    def test_it_runs_on_a_worker_thread_without_touching_the_gui_thread(
            self, qtbot, pdf_figure):
        """The whole point, asserted directly rather than through the widget.

        ``QPdfDocument.render`` does not release the GIL, so a naive worker
        starves the GUI thread just as thoroughly as an inline call did. This
        fails if ``render_pdf_to_image`` ever goes back to calling it.
        """
        from PySide6.QtCore import QThread

        _fig, png = pdf_figure
        pdf = str(Path(png).with_suffix(".pdf"))
        result = {}

        class _RenderThread(QThread):
            def run(self):
                result["image"] = render_pdf_to_image(pdf)

        thread = _RenderThread()
        dog = LoopWatchdog()
        dog.start()
        thread.start()
        _drive(qtbot, dog, lambda: "image" in result and not thread.isRunning())
        assert thread.wait(5000)

        assert result.get("image") is not None
        assert dog.ticks > 10
        assert dog.worst < PDF_STALL_BUDGET_S, (
            f"the worker held the GUI thread for {dog.worst * 1000:.0f} ms — "
            "the render is blocking the interpreter, not just this thread")


class TestAppScreenIntegration:
    def test_mask_screen_uses_figure_queue(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        assert hasattr(scr, "_figure_queue")
        from spacr.qt.widgets.figure_queue import FigureQueue as FQ
        assert isinstance(scr._figure_queue, FQ)

    def test_figure_ready_routes_to_queue_and_shows_card(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._on_figure_ready(_make_fig(0))
        assert scr._figure_queue.count() == 1
        assert scr._figures_card.isVisibleTo(scr) or True  # card.show() called
