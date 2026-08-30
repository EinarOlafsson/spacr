"""FigureQueue's refusal paths: what the gallery does once something is wrong.

Nearly every branch here is one a user meets only after something else has
failed -- a render that wrote no file, a raster Qt will not load, a strip row
that is gone, a vector page that never arrived -- or a navigation with
nowhere left to go. They exist so the gallery degrades instead of crashing,
or of showing figure 1 under the caption for figure 2. Real figures and real
files throughout: "did the queue keep a picture" is answered by the pixmap.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QPoint, QSettings  # noqa: E402

from spacr.qt.widgets import figure_queue as fq  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_figure_prefs(monkeypatch, tmp_path_factory):
    """Keep the real preference store out of it, and pin the format.

    ``render_figure_to_png`` reads the installed app's QSettings, so without
    this the file rewrites the developer's Preferences. PNG explicitly: the
    default ``pdf`` writes a sibling vector page beside every raster, and
    several branches below are about a slot that has none.
    """
    from spacr.qt import preferences as prefs

    store = tmp_path_factory.mktemp("figq_r3_prefs") / "prefs.ini"
    monkeypatch.setattr(
        prefs, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    prefs.set_figure_format("png")


def _fig(seed: int = 0) -> Figure:
    """A small, real, picklable figure that is not in pyplot's registry."""
    figure = Figure(figsize=(3.0, 2.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [seed, seed + 1, seed])
    axes.set_title(f"fig {seed}")
    return figure


def _fail_renders(monkeypatch):
    """Make every full render fail; the returned dict switches it back on.

    ``monkeypatch.undo()`` is not usable for this: the ``monkeypatch``
    fixture is shared with ``_isolate_figure_prefs``, so undoing here would
    put the developer's real preference store back mid-test.
    """
    real = fq.render_figure_to_png
    state = {"failing": True}
    monkeypatch.setattr(
        fq, "render_figure_to_png",
        lambda *a, **k: False if state["failing"] else real(*a, **k))
    return state


def _queue(qtbot, ram_cap: int = 100):
    """A raster-mode queue: the path a spilled or PDF-only figure uses."""
    queue = fq.FigureQueue(ram_cap=ram_cap)
    queue.set_live_canvas_enabled(False)
    qtbot.addWidget(queue)
    return queue


class _Legend:
    def __init__(self, texts, title):
        self._texts, self._title = texts, title

    def get_texts(self):
        return list(self._texts)

    def get_title(self):
        return self._title


class _Ax:
    def __init__(self, legend):
        self.title, self.texts = "title", []
        self.xaxis = types.SimpleNamespace(label="x")
        self.yaxis = types.SimpleNamespace(label="y")
        self._legend = legend

    def get_xticklabels(self):
        return []

    def get_yticklabels(self):
        return []

    def get_legend(self):
        return self._legend


def test_a_legend_with_no_title_object_still_yields_its_entry_texts():
    """A titleless legend must not cost that legend its restyling.

    ``figure_text_items`` is what the text controls walk, duck-typed so it
    can run on a worker. Falling out at a legend reporting no title object
    leaves its entry labels unstyled while the rest of the figure changes.
    """
    titled = _Ax(_Legend(["a"], "LEGEND TITLE"))
    untitled = _Ax(_Legend(["b"], None))
    figure = types.SimpleNamespace(axes=[titled, untitled], texts=["suptitle"])

    found = fq.figure_text_items(figure)

    assert "a" in found and "b" in found
    assert found.count("LEGEND TITLE") == 1
    assert "suptitle" in found


def _install_pdf_stub(monkeypatch, *, load_ok=True, pages=1):
    """A stand-in ``PySide6.QtPdf``, recording what the reader asked of it.

    Stubbed rather than driven with real PDFs because ``PySide6.QtPdf`` does
    not import in every environment (a system libbrotlidec mismatch is one
    way), and the gates under test are about paths, load results and page
    counts -- none of which need a real rasteriser.
    """
    log = types.SimpleNamespace(loads=[], counted=0, renderers=0)

    class _Error:
        None_ = "ok"
        FileNotFoundError = "missing"

    class _Doc:
        Error = _Error

        def load(self, path):
            log.loads.append(path)
            return _Error.None_ if load_ok else _Error.FileNotFoundError

        def pageCount(self):
            log.counted += 1
            return pages

    class _Renderer:
        def __init__(self):
            log.renderers += 1

    module = types.ModuleType("PySide6.QtPdf")
    module.QPdfDocument = _Doc
    module.QPdfPageRenderer = _Renderer
    monkeypatch.setitem(sys.modules, "PySide6.QtPdf", module)
    return log


def test_a_pdf_page_is_refused_at_each_gate_before_anything_is_rendered(
        monkeypatch, tmp_path):
    """Each gate must refuse cheaply, because this runs on a worker thread.

    A missing page is *normal* -- the queue deletes its temp dir while renders
    are in flight -- and an unloadable or empty document must stop before a
    page renderer and a nested event loop are set up, or every navigation
    leaves a worker blocked on a loop.
    """
    missing = tmp_path / "gone.pdf"
    log = _install_pdf_stub(monkeypatch)
    assert fq.render_pdf_to_image(str(missing)) is None
    assert log.loads == []          # never opened a document for a missing file

    present = tmp_path / "there.pdf"
    present.write_bytes(b"not really a pdf")
    log = _install_pdf_stub(monkeypatch, load_ok=False)
    assert fq.render_pdf_to_image(str(present)) is None
    assert log.loads == [str(present)] and log.counted == 0

    log = _install_pdf_stub(monkeypatch, pages=0)
    assert fq.render_pdf_to_image(str(present)) is None
    assert log.counted == 1 and log.renderers == 0


def test_only_the_newest_resize_render_of_the_current_figure_is_shown(
        qtbot, monkeypatch, tmp_path):
    """The resize pipeline must discard far more than it paints.

    The debounce fires on every layout change, including ones that squeeze
    the pane to nothing -- rendering into a 40 px box is pure cost. And a
    drag dispatches a render per settled size while the user goes on
    navigating, so a superseded render, one belonging to another figure, or
    a PNG Qt could not load would put the wrong picture -- or a blank --
    under the current figure's caption.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    good_png = queue._png_paths[0]
    submitted = []
    monkeypatch.setattr(queue._jobs, "submit",
                        lambda fn, cb=None: submitted.append(fn) or True)

    queue._view.resize(40, 40)
    queue._rerender_for_size()
    assert submitted == []
    queue._view.resize(700, 500)
    queue._rerender_for_size()
    assert len(submitted) == 1

    queue._resize_seq = 7
    before = queue._ram[0]
    junk = tmp_path / "torn.png"
    junk.write_bytes(b"\x89PNG truncated")

    queue._on_resize_rendered(None)                     # worker returned nothing
    queue._on_resize_rendered((0, 7, False, good_png))  # the render failed
    queue._on_resize_rendered((5, 7, True, good_png))   # another figure
    queue._on_resize_rendered((0, 3, True, good_png))   # superseded token
    queue._on_resize_rendered((0, 7, True, str(junk)))  # unreadable raster
    assert queue._ram[0] is before

    queue._on_resize_rendered((0, 7, True, good_png))
    assert queue._ram[0] is not before


def test_restyling_reports_failure_rather_than_pretending(qtbot, monkeypatch):
    """The settings dialog reads this bool to decide whether to say anything.

    Two ways it is False: nothing on screen, and a render that wrote no file.
    True in either case leaves the user looking at the old picture with no
    sign that their change never landed.
    """
    empty = _queue(qtbot)
    assert empty.refresh_current_figure() is False

    queue = _queue(qtbot)
    queue.add_figure(_fig(1))
    renders = _fail_renders(monkeypatch)
    assert queue.refresh_current_figure() is False

    renders["failing"] = False
    assert queue.refresh_current_figure() is True
    assert not queue._list.item(0).icon().isNull()


def test_a_change_arriving_mid_draw_is_remembered_not_dropped(qtbot):
    """A slider dragged faster than the preview renders must still settle.

    One draw runs at a time; a change arriving during one is recorded and
    drawn when the worker frees up. Reporting "nothing started" makes the
    caller render on the GUI thread -- the freeze this exists to remove.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(2))
    queue._preview_busy = True

    assert queue.refresh_current_figure(preview=True) is True
    assert queue._preview_pending is True


def test_a_missing_strip_row_does_not_stop_the_refresh(qtbot):
    """The picture matters more than its thumbnail.

    The strip is rebuilt independently of the figure store -- forgetting a run
    takes rows out of it -- so a refresh can arrive for a slot with no row. It
    must still report success, or a good restyle is treated as a failure.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    assert queue.refresh_current_figure() is True
    assert not queue._list.item(1).icon().isNull()

    # blockSignals so removing the row does not also move the current index:
    # the state under test is "the view is on a slot whose row has gone".
    queue._list.blockSignals(True)
    queue._list.takeItem(1)
    queue._list.blockSignals(False)
    assert queue._current == 1 and queue._list.item(1) is None

    assert queue.refresh_current_figure() is True
    assert not queue._ram[1].isNull()


def test_refreshing_a_figure_off_screen_survives_both_of_its_ledges(
        qtbot, monkeypatch):
    """Restyling from a grid tile redraws a figure nobody is looking at.

    Without a truthful return the grid keeps the old picture and the user
    clicks again, and again. A render that wrote no file is False; a slot
    whose row has gone is still True, and the current index must not move.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.show_index(0)

    renders = _fail_renders(monkeypatch)
    assert queue.refresh_figure(1) is False

    renders["failing"] = False
    queue._list.takeItem(1)
    assert queue.refresh_figure(1) is True
    assert queue._current == 0


def test_figure_settings_do_not_open_on_nothing(qtbot, monkeypatch):
    """An empty gallery must not raise a dialog bound to no figure.

    A cleared queue's context menu reaches here, and every control in that
    dialog writes straight onto the figure it was handed.
    """
    built = []
    monkeypatch.setattr(
        fs, "FigureSettingsDialog",
        lambda figure, *a, **k: built.append(figure)
        or types.SimpleNamespace(exec=lambda: None))

    empty = _queue(qtbot)
    empty._open_figure_settings()
    assert built == []

    queue = _queue(qtbot)
    figure = _fig(3)
    queue.add_figure(figure)
    queue._open_figure_settings()
    assert built == [figure]


def _stub_menu(monkeypatch):
    """Replace the real context menu with one that records and never execs."""
    seen = {}

    class _Menu:
        def exec(self, position):
            seen["at"] = position

    def _build(parent, figure, on_change=None, open_settings=None):
        seen["figure"] = figure
        seen["on_change"] = on_change
        return _Menu()

    monkeypatch.setattr(fs, "build_figure_context_menu", _build)
    return seen


def test_the_context_menu_navigates_only_when_the_caller_asks(
        qtbot, monkeypatch):
    """A grid tile's menu must not jump the queue to that tile.

    The strip wants a right-click to select what was clicked; the grid does
    not, because moving the current index loses the comparison being made.
    """
    seen = _stub_menu(monkeypatch)
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    assert queue._current == 1

    queue.show_figure_menu(QPoint(1, 2), 0)
    assert queue._current == 0
    assert seen["at"] == QPoint(1, 2)

    queue.show_figure_menu(QPoint(3, 4), 1, navigate=False)
    assert queue._current == 0


def test_the_menu_callback_swaps_a_new_figure_and_redraws_an_old_one(
        qtbot, monkeypatch):
    """"Show as violin" hands back a NEW Figure, not a toggle.

    One callback carries both: anything not a bool is the replacement, and
    queue, strip and tile have to be pointed at it together. Read as a bool
    it merely redraws the figure just replaced, and the menu looks broken.
    """
    seen = _stub_menu(monkeypatch)
    queue = _queue(qtbot)
    original = _fig(0)
    queue.add_figure(original)
    queue.show_figure_menu(QPoint(0, 0), 0)
    redraw = seen["on_change"]

    assert redraw(False) is True
    assert queue._figures[0] is original

    replacement = _fig(9)
    assert redraw(replacement) is True
    assert queue._figures[0] is replacement
    assert queue._fig_index[id(replacement)] == 0


def test_re_emitting_an_evicted_figure_reuses_its_slot(qtbot):
    """A re-emitted figure must never become a second gallery entry.

    The training monitor re-emits the same Figure every epoch, by which time
    the cap may have spilled it. Keying only on the live set would treat it as
    new and grow the gallery by a tile per epoch, all showing one plot.
    """
    queue = _queue(qtbot)
    figure = _fig(0)
    queue.add_figure(figure)
    assert queue.add_figure(figure) == 0        # still live: same slot
    assert queue.has_live_figure(0) is True

    queue._evict_live_figure(0)
    assert queue.has_live_figure(0) is False
    assert queue.add_figure(figure) == 0        # spilled: still the same slot
    assert queue.count() == 1


def test_a_prerendered_png_is_adopted_and_a_broken_one_is_redrawn(
        qtbot, tmp_path):
    """The worker-rendered PNG is a fast path, not a trusted one.

    Adopting the bridge's file is a move plus a cheap load instead of a second
    savefig on the GUI thread. A truncated file loads as nothing, and dropping
    the figure would lose a plot the run produced -- so it is rendered here.
    """
    queue = _queue(qtbot)

    good = tmp_path / "worker_0.png"
    _fig(0).savefig(good, dpi=60)
    idx = queue.add_figure(_fig(0), prerendered_png=str(good))
    assert not good.exists()                     # moved, not copied
    assert not queue._ram[idx].isNull()

    broken = tmp_path / "worker_1.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    other = queue.add_figure(_fig(1), prerendered_png=str(broken))
    assert not queue._ram[other].isNull()
    assert not queue._list.item(other).icon().isNull()


def test_a_figure_whose_render_fails_still_takes_a_slot(qtbot, monkeypatch):
    """A figure that could not be drawn must still be counted.

    Dropping it renumbers every later figure and loses the run's account of
    what it produced. The slot exists with no picture -- what
    ``_why_not_shown`` explains -- and the next figure keeps its icon.
    """
    queue = _queue(qtbot)
    renders = _fail_renders(monkeypatch)
    first = queue.add_figure(_fig(0))
    assert queue.count() == 1
    assert queue._ram.get(first) is None
    assert queue._list.item(first).icon().isNull()
    assert queue._png_paths[first].endswith(".png")

    renders["failing"] = False
    second = queue.add_figure(_fig(1))
    assert queue._ram.get(second) is not None
    assert not queue._list.item(second).icon().isNull()


def test_a_live_update_lands_in_its_own_slot_or_nowhere(qtbot, tmp_path):
    """The training monitor overwrites one slot every epoch, for hours.

    A frame caught mid-write loads as a null pixmap; caching it would replace
    a readable plot with an empty panel until the next epoch. A frame for a
    figure the user is not looking at belongs in its own slot -- whose strip
    row may no longer exist -- but must not repaint the view, or the user
    watching figure 1 sees figure 2 appear under figure 1's caption.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.show_index(0)
    assert not fq._sibling_pdf(Path(queue._png_paths[0])).is_file()
    before_0, before_1 = queue._ram[0], queue._ram[1]

    torn = tmp_path / "epoch_1.png"
    torn.write_bytes(b"\x89PNG\r\n\x1a\n half")
    queue._refresh_live_figure(0, str(torn))
    assert queue._ram[0] is before_0

    fresh = tmp_path / "epoch_2.png"
    _fig(5).savefig(fresh, dpi=60)
    queue._refresh_live_figure(0, str(fresh))
    assert queue._ram[0] is not before_0

    # Now one for the figure off screen, whose strip row has been taken away.
    queue._list.takeItem(1)
    other = tmp_path / "epoch_3.png"
    _fig(7).savefig(other, dpi=60)
    queue._refresh_live_figure(1, str(other))
    assert queue._ram[1] is not before_1
    assert queue._current == 0


def test_navigation_stops_at_both_ends_and_ignores_an_index_off_the_list(
        qtbot):
    """Walking off either end must be a no-op, not a wrong figure.

    Back and Forward get held down. Past the ends the index goes negative or
    past the count, and picture and caption then disagree: the raster is
    looked up by index and so is the strip row.
    """
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    assert queue._current == 1

    queue.show_next()
    assert queue._current == 1
    queue.show_prev()
    assert queue._current == 0
    queue.show_prev()
    assert queue._current == 0

    queue.show_index(99)
    assert queue._current == 0
    queue.show_index(-1)
    assert queue._current == 0


def test_the_grid_reads_an_evicted_raster_from_disk_without_promoting_it(
        qtbot):
    """Building the grid must not evict the figure the user is looking at.

    It is a bulk read of everything. Fetching each missing pixmap through the
    LRU would touch every index and leave the cache holding the tail of the
    grid instead of the current figure.
    """
    queue = _queue(qtbot, ram_cap=1)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    assert set(queue._ram) == {1}

    pixmaps = queue.all_pixmaps()
    assert len(pixmaps) == 2
    assert all(pm is not None and not pm.isNull() for pm in pixmaps)
    assert set(queue._ram) == {1}


def test_forgetting_a_run_moves_only_what_is_after_it(qtbot):
    """Forget is reached from a heading, and headings outlive their figures.

    Marks are recorded when a run STARTS, so one that failed before its first
    plot leaves a mark with no section -- forgetting it must answer 0 rather
    than raise. And the current index is a position in a dense list: shifting
    it for a run removed *after* it would slide the view onto a different
    figure than the one being read.
    """
    queue = _queue(qtbot)
    queue.mark_run("first")
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.mark_run("second")
    queue.add_figure(_fig(2))
    queue.mark_run("third")          # started, drew nothing
    queue.show_index(0)

    assert queue.forget_run("third") == 0
    assert queue.forget_run("never marked") == 0
    assert queue.count() == 3

    assert queue.forget_run("second") == 1
    assert queue.count() == 2
    assert queue._current == 0
    assert [name for name, _, _ in queue.run_sections()] == ["first"]
