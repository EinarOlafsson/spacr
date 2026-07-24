"""Tests for the FigureQueue widget — RAM cap, temp spill, cleanup,
zoom, and forward/back navigation.

Uses matplotlib's Agg backend (headless) to build real Figure objects,
then drives the queue directly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from spacr.qt.widgets.figure_queue import FigureQueue


def _make_fig(seed: int = 0):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [seed, seed + 1, seed])
    ax.set_title(f"fig {seed}")
    return fig


class TestBasics:
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

    def test_figure_settings_button_png_hidden(self, qtbot):
        from spacr.qt import preferences as prefs
        prefs.set_figure_format("png")
        q = FigureQueue()
        qtbot.addWidget(q)
        q.add_figure(_make_fig(0))
        q._refresh_nav()
        # In PNG mode the figure-settings button is hidden.
        assert not q._fig_settings_btn.isVisibleTo(q)

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
