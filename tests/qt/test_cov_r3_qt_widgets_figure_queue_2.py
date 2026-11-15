"""FigureQueue's refusal paths: what the gallery does once something is wrong.

Nearly every branch here is one a user meets only after something else has
failed -- a render that wrote no file, a raster Qt will not load, a strip row
that is gone, a vector page that never arrived -- or a navigation with
nowhere left to go. They exist so the gallery degrades instead of crashing,
or of showing figure 1 under the caption for figure 2. Real figures and real
files throughout: "did the queue keep a picture" is answered by the pixmap.
"""
from __future__ import annotations

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
