"""Figure queue: resize renders, restyles, the context menu and figure slots.

The gallery degrades rather than crashes, and never shows one figure's picture
under another's caption.  Only the newest readable resize render of the
current figure is painted and cached; a restyle returns True only when the
picture on screen (and its strip thumbnail) really changed, or when a change
was remembered during a draw; the right-click menu acts on the figure under
the cursor and navigates only when asked; every figure takes a slot even when
its render fails, a re-emitted figure keeps its slot, the grid reads spilled
rasters without promoting them, and a crisp page for a figure left behind is
dropped.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.  Real figures and real files throughout.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QPoint, QSettings  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402

from spacr.qt.widgets import figure_queue as fq  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolate_figure_prefs(monkeypatch, tmp_path_factory):
    """Point preferences at a throwaway INI and pin PNG, so no vector page is written."""
    from spacr.qt import preferences as prefs

    store = tmp_path_factory.mktemp("figq_slots_prefs") / "prefs.ini"
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    prefs.set_figure_format("png")


def _fig(seed=0):
    figure = Figure(figsize=(3.0, 2.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [seed, seed + 1, seed])
    axes.set_title(f"fig {seed}")
    return figure


def _queue(qtbot, ram_cap=100):
    queue = fq.FigureQueue(ram_cap=ram_cap)
    queue.set_live_canvas_enabled(False)
    qtbot.addWidget(queue)
    return queue


class _RecordingView:
    """Stands in for the pixmap view so "was it painted?" is answerable."""

    def __init__(self):
        self.shown = []

    def set_pixmap(self, pixmap):
        self.shown.append(pixmap)


def _fail_renders(monkeypatch):
    """Make every full render fail; flip ``state['failing']`` to render again."""
    real = fq.render_figure_to_png
    state = {"failing": True}
    monkeypatch.setattr(fq, "render_figure_to_png",
                        lambda *a, **k: False if state["failing"] else real(*a, **k))
    return state


def _an_image():
    image = QImage(8, 6, QImage.Format_RGB32)
    image.fill(QColor("#3366aa"))
    return image


def test_resize_renders_paint_only_the_newest_readable_one(qtbot, monkeypatch, tmp_path, subtests):
    """Tiny panes submit nothing; empty, failed, foreign, stale or unreadable payloads are not painted or cached."""
    good = tmp_path / "whole.png"
    _fig(0).savefig(str(good))
    junk = tmp_path / "torn.png"
    junk.write_bytes(b"\x89PNG\r\n\x1a\n truncated")

    with subtests.test(msg="tiny_pane_submits_nothing"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(0))
        submitted = []
        monkeypatch.setattr(queue._jobs, "submit",
                            lambda fn, cb=None: submitted.append(fn) or True)
        queue._view.resize(40, 40)
        queue._rerender_for_size()
        assert submitted == []
        queue._view.resize(700, 500)
        queue._rerender_for_size()
        assert len(submitted) == 1

    refused = {
        "nothing": None,
        "empty_tuple": (),
        "render_failed": (0, 9, False, str(good)),
        "another_figure": (3, 9, True, str(good)),
        "superseded_token": (0, 4, True, str(good)),
        "unreadable_png": (0, 9, True, str(junk)),
    }
    for case_id, payload in refused.items():
        with subtests.test(msg=f"refused_{case_id}"):
            queue = _queue(qtbot)
            view = _RecordingView()
            queue._view = view
            queue._current, queue._resize_seq = 0, 9
            queue._pdf_state[0] = "done"
            queue._on_resize_rendered(payload)
            assert view.shown == []
            assert 0 not in queue._ram
            assert queue._pdf_state[0] == "done"
    with subtests.test(msg="current_render_cached_and_shown"):
        queue = _queue(qtbot)
        view = _RecordingView()
        queue._view = view
        queue._current, queue._resize_seq = 0, 9
        queue._pdf_state[0] = "done"
        queue._on_resize_rendered((0, 9, True, str(good)))
        assert not queue._ram[0].isNull()
        assert len(view.shown) == 1
        assert 0 not in queue._pdf_state


def test_restyles_report_true_only_when_the_picture_changed(qtbot, monkeypatch, tmp_path, subtests):
    """Empty panels and unwritten renders are False; a landed restyle repaints view and strip; mid-draw changes are remembered."""
    with subtests.test(msg="empty_panel"):
        queue = _queue(qtbot)
        assert queue.refresh_current_figure() is False
        assert queue.refresh_current_figure(preview=True) is False
    with subtests.test(msg="mid_draw_change_remembered"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(2))
        queue._preview_busy = True
        queue._jobs.submit = lambda fn, cb=None: pytest.fail(
            "a second preview worker was started while one was drawing")
        assert queue.refresh_current_figure(preview=True) is True
        assert queue._preview_pending is True
    with subtests.test(msg="unwritable_directory_then_written"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(0))
        queue._png_paths[0] = str(tmp_path / "deleted-dir" / "f.png")
        assert queue.refresh_current_figure() is False
        assert not (tmp_path / "deleted-dir").exists()
        queue._png_paths[0] = str(tmp_path / "written.png")
        assert queue.refresh_current_figure() is True
        assert (tmp_path / "written.png").is_file()
    with subtests.test(msg="failed_render_then_landed"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(1))
        renders = _fail_renders(monkeypatch)
        assert queue.refresh_current_figure() is False
        renders["failing"] = False
        assert queue.refresh_current_figure() is True
        assert not queue._list.item(0).icon().isNull()
    with subtests.test(msg="landed_restyle_repaints_view_and_strip"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(0))
        idx = queue._current
        queue._pdf_state[idx] = "done"
        view = _RecordingView()
        queue._view = view
        assert queue.refresh_current_figure() is True
        assert len(view.shown) == 1
        assert idx not in queue._pdf_state
        assert not queue._list.item(idx).icon().isNull()
    with subtests.test(msg="missing_strip_row_still_true"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(0))
        queue.add_figure(_fig(1))
        queue._list.blockSignals(True)
        queue._list.takeItem(1)
        queue._list.blockSignals(False)
        assert queue._current == 1 and queue._list.item(1) is None
        assert queue.refresh_current_figure() is True
        assert not queue._ram[1].isNull()
    with subtests.test(msg="off_screen_unwritable_is_false"):
        queue = _queue(qtbot)
        first, second = _fig(0), _fig(1)
        queue.add_figure(first)
        queue.add_figure(second)
        queue.show_index(1)
        queue._png_paths[0] = str(tmp_path / "no-such-dir" / "f.png")
        assert queue.refresh_figure(0) is False
        assert queue.figure_for(0) is first


def test_context_menu_acts_on_the_clicked_figure_and_navigates_on_request(qtbot, monkeypatch, subtests):
    """The strip's menu navigates to the clicked figure and opens at the click; the grid's does not navigate."""
    built = {}

    class _Menu:
        def __init__(self):
            self.executed_at = []

        def exec(self, position):
            self.executed_at.append(position)

    def _build(owner, figure, on_change=None, open_settings=None):
        built.update(figure=figure, menu=_Menu())
        return built["menu"]

    monkeypatch.setattr(fs, "build_figure_context_menu", _build)
    for case_id, navigate, current in (("strip_navigates", True, 1),
                                       ("grid_stays", False, 0)):
        with subtests.test(msg=case_id):
            queue = _queue(qtbot)
            first, second = _fig(0), _fig(1)
            queue.add_figure(first)
            queue.add_figure(second)
            queue.show_index(0)
            menu = queue.show_figure_menu(QPoint(3, 4), idx=1, navigate=navigate)
            assert queue._current == current
            assert built["figure"] is second
            assert menu is built["menu"]
            assert menu.executed_at == [QPoint(3, 4)]


def test_every_figure_keeps_one_slot_and_the_cache_follows_what_was_viewed(qtbot, monkeypatch, tmp_path, subtests):
    """Failed renders still take a slot, prerendered PNGs are adopted or redrawn, re-emits reuse slots, the grid and crisp pages leave the cache alone."""
    with subtests.test(msg="failed_render_still_takes_a_slot"):
        queue = _queue(qtbot)
        renders = _fail_renders(monkeypatch)
        first = queue.add_figure(_fig(0))
        assert (queue.count(), queue._ram.get(first)) == (1, None)
        assert queue._list.item(first).icon().isNull()
        assert queue._png_paths[first].endswith(".png")
        renders["failing"] = False
        second = queue.add_figure(_fig(1))
        assert second == 1
        assert not queue._ram[second].isNull()
        assert not queue._list.item(second).icon().isNull()
    with subtests.test(msg="prerendered_png_adopted_or_redrawn"):
        queue = _queue(qtbot)
        good = tmp_path / "worker_0.png"
        _fig(0).savefig(good, dpi=60)
        idx = queue.add_figure(_fig(0), prerendered_png=str(good))
        assert not good.exists()
        assert not queue._ram[idx].isNull()
        broken = tmp_path / "worker_1.png"
        broken.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
        other = queue.add_figure(_fig(1), prerendered_png=str(broken))
        assert not queue._ram[other].isNull()
        assert not queue._list.item(other).icon().isNull()
    with subtests.test(msg="re_emitted_figure_reuses_its_slot"):
        queue = _queue(qtbot)
        figure = _fig(0)
        assert queue.add_figure(figure) == 0
        assert queue.add_figure(figure) == 0
        assert queue.has_live_figure(0) is True
        queue._evict_live_figure(0)
        assert queue.has_live_figure(0) is False
        assert queue.add_figure(figure) == 0
        assert queue.count() == 1
    with subtests.test(msg="grid_reads_spilled_raster_without_promoting"):
        queue = _queue(qtbot, ram_cap=1)
        queue.add_figure(_fig(0))
        queue.add_figure(_fig(1))
        assert set(queue._ram) == {1}
        pixmaps = queue.all_pixmaps()
        assert len(pixmaps) == 2
        assert [pm.isNull() for pm in pixmaps] == [False, False]
        assert set(queue._ram) == {1}
    with subtests.test(msg="crisp_page_for_figure_left_behind_dropped"):
        queue = _queue(qtbot)
        queue.add_figure(_fig(0))
        queue.add_figure(_fig(1))
        queue.show_index(1)
        view = _RecordingView()
        queue._view = view
        raster = queue._ram[0]
        queue._pdf_state[0] = 41
        queue._on_pdf_rendered((0, 41, _an_image()))
        assert view.shown == []
        assert 0 not in queue._pdf_state
        assert queue._ram[0] is raster
        queue._pdf_state[1] = 42
        queue._on_pdf_rendered((1, 42, _an_image()))
        assert len(view.shown) == 1 and not view.shown[0].isNull()
        assert queue._pdf_state[1] == "done"

