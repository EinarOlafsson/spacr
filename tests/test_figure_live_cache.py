"""The Figures panel keeps recent figures EDITABLE, not just visible.

The panel displays a pixmap, so what the user sees is a picture of a figure
and has no legend to toggle or axis to rescale. The live matplotlib Figures
were retained -- but never capped, so a long run accumulated every one.

These tests pin both halves: the cap is the user's number, and a figure past
it is still viewable and (with dynamic figures on) still loads from its
vector page.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    widget = FigureQueue()
    qtbot.addWidget(widget)
    yield widget
    plt.close("all")


def _add(widget, n):
    for i in range(n):
        figure = plt.figure()
        figure.gca().plot([0, 1], [i, i])
        widget.add_figure(figure)


def test_the_live_figure_cap_is_the_users_number(queue, monkeypatch):
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 5)
    _add(queue, 12)
    assert queue.live_figure_count() == 5
    # The newest are the restylable ones -- those are the figures a user is
    # looking at when they want to change something.
    assert all(queue.has_live_figure(i) for i in range(7, 12))
    assert not any(queue.has_live_figure(i) for i in range(0, 5))


def test_an_evicted_figure_is_still_viewable(queue, monkeypatch):
    """Releasing the Figure must not lose the figure."""
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 3)
    _add(queue, 8)
    assert queue.count() == 8
    # Every slot still has its rendered page on disk.
    assert all(queue._png_paths.get(i) for i in range(8))


def test_raising_the_cap_retains_more(queue, monkeypatch):
    caps = {"n": 2}
    monkeypatch.setattr(queue, "live_figure_cap", lambda: caps["n"])
    _add(queue, 6)
    assert queue.live_figure_count() == 2
    caps["n"] = 10
    _add(queue, 1)
    assert queue.live_figure_count() == 3


def test_a_cap_of_one_is_legal(queue, monkeypatch):
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 1)
    _add(queue, 4)
    assert queue.live_figure_count() == 1


def test_the_preferences_round_trip():
    from spacr.qt.preferences import (
        MAX_FIG_LIVE_CACHE, MIN_FIG_LIVE_CACHE, get_figure_dynamic,
        get_figure_live_cache, set_figure_dynamic, set_figure_live_cache,
    )

    original, original_dynamic = get_figure_live_cache(), get_figure_dynamic()
    try:
        set_figure_live_cache(37)
        assert get_figure_live_cache() == 37
        set_figure_dynamic(False)
        assert get_figure_dynamic() is False
        set_figure_dynamic(True)
        assert get_figure_dynamic() is True
        # Out of range is refused rather than silently clamped on write.
        with pytest.raises(ValueError):
            set_figure_live_cache(MAX_FIG_LIVE_CACHE + 1)
        with pytest.raises(ValueError):
            set_figure_live_cache(MIN_FIG_LIVE_CACHE - 1)
    finally:
        set_figure_live_cache(original)
        set_figure_dynamic(original_dynamic)


def test_both_controls_exist_in_the_preferences_dialog(qtbot):
    """A setting the user cannot reach is not a setting."""
    from PySide6.QtWidgets import QCheckBox, QSpinBox

    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    spins = [w for w in dialog.findChildren(QSpinBox)
             if "restylable" in (w.toolTip() or "")]
    checks = [w for w in dialog.findChildren(QCheckBox)
              if "load its PDF page" in (w.toolTip() or "")]
    assert len(spins) == 1 and len(checks) == 1
    assert spins[0].minimum() >= 1


# ------------------------------------------------- restoring an old figure


def test_an_evicted_figure_comes_back_fully_editable(queue, monkeypatch):
    """The point of spilling a Figure rather than only its picture.

    A saved vector page allows a stroke to be recoloured, a width changed, a
    font resized. It does NOT allow anything data-bound: a log axis has to
    recompute every position. A restored Figure allows all of it.
    """
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 3)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: True)
    _add(queue, 10)

    assert not queue.has_live_figure(0), "figure 0 should have been evicted"
    assert queue.is_restorable(0)

    figure = queue.figure_for(0)
    assert figure is not None
    axis = figure.axes[0]

    # Every appearance change the user asked for...
    axis.grid(False)
    axis.spines["left"].set_linewidth(3)
    axis.tick_params(labelsize=16)
    for line in axis.lines:
        line.set_color("crimson")
    # ...plus the data-bound one that a PDF could never give back.
    axis.set_yscale("log")
    assert axis.get_yscale() == "log"


def test_restoring_puts_the_figure_back_in_the_live_set(queue, monkeypatch):
    """Repeated edits must not re-read the disk each time."""
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 3)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: True)
    _add(queue, 8)
    queue.figure_for(0)
    assert queue.has_live_figure(0)
    # And the cap still holds afterwards.
    assert queue.live_figure_count() <= 3


def test_dynamic_figures_off_does_not_restore(queue, monkeypatch):
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 2)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: False)
    _add(queue, 6)
    assert queue.is_restorable(0), "the spill is still on disk"
    assert queue.figure_for(0) is None, "but the option says do not use it"


def test_an_unpicklable_figure_does_not_break_the_cap(queue, monkeypatch):
    """Failing to spill must never stop old figures being released."""
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 2)
    monkeypatch.setattr(queue, "_spill_figure", lambda idx, fig: False)
    _add(queue, 7)
    assert queue.live_figure_count() == 2
    assert queue.figure_for(0) is None
    # It is still viewable from its rendered page.
    assert queue._png_paths.get(0)
