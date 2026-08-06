"""The Classify (CV) training curves: one figure that grows, on nothing.

Two defects were reported together. One was real in the display layer and
one was not real at all, which is worth recording because the second cost
the most time to establish:

* **"a completely new graph for each epoch."** The PIPELINE was already
  correct -- `_plot_training_curves` takes the previous figure back and
  redraws into it, and `id(fig)` is stable across epochs. Measured before
  changing anything. So the duplication, if it is still seen, is in the
  gallery slot lookup, NOT here. These tests pin the pipeline half so the
  next person can skip it.

* **"the graphs have a black background."** Real, and not in this module
  either: `preferences.get_figure_colors()` resolved "auto" to #000000 on a
  dark theme, so every rendered figure got an opaque black page. `bg` is the
  WINDOW colour and a figure is not a window (INVARIANTS 2). It resolves to
  transparent now, so the container shows through and the page-opacity
  preference reaches the plot.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from spacr.deep_spacr import (            # noqa: E402
    _CLASS_CURVE_COLORS, _TRAIN_CURVE_COLOR, _VAL_CURVE_COLOR,
    _plot_training_curves,
)


def _history(n):
    train = [{"epoch": e, "loss": 1.0 / e, "accuracy": 0.5 + e * 0.05}
             for e in range(1, n + 1)]
    val = [{"epoch": e, "loss": 1.1 / e, "accuracy": 0.4 + e * 0.05}
           for e in range(1, n + 1)]
    return train, val


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def test_one_figure_survives_many_epochs():
    """THE assertion. Twenty epochs must be one curve to read, not twenty
    pictures to scroll through -- you cannot see whether the loss is still
    falling if each epoch is a separate figure."""
    figure = None
    seen = []
    for epoch in range(1, 11):
        train, val = _history(epoch)
        figure = _plot_training_curves(train, val, 10, figure)
        seen.append(id(figure))

    assert len(set(seen)) == 1, "a new figure was created during training"
    assert len(plt.get_fignums()) == 1, (
        f"{len(plt.get_fignums())} figures are open; the training monitor "
        f"must keep exactly one")


def test_each_epoch_adds_a_point_rather_than_replacing_the_curve():
    figure = None
    for epoch in range(1, 6):
        train, val = _history(epoch)
        figure = _plot_training_curves(train, val, 10, figure)

    loss_axis = figure.axes[0]
    assert loss_axis.lines, "no curve was drawn"
    assert len(loss_axis.lines[0].get_xdata()) == 5


def test_the_figure_is_marked_for_live_update():
    """The flag the bridge reads to re-emit in place instead of appending a
    new gallery entry. Without it the display duplicates however well the
    pipeline behaves."""
    figure = _plot_training_curves(*_history(1), 10, None)
    assert getattr(figure, "_spacr_live_update", False) is True


# ---------------------------------------------------------------------------
# The background
# ---------------------------------------------------------------------------

def test_nothing_opaque_is_baked_into_the_figure():
    """A white or black page cannot be undone by restyling later, so the
    plot has to start transparent and let the container show through."""
    figure = _plot_training_curves(*_history(3), 10, None)
    assert figure.patch.get_alpha() == 0.0
    for axis in figure.axes:
        assert axis.patch.get_alpha() == 0.0


def test_it_stays_transparent_when_the_figure_is_reused():
    """fig.clear() resets the patch, so the second epoch would otherwise
    come back opaque -- and the user would see it flash."""
    figure = None
    for epoch in range(1, 4):
        figure = _plot_training_curves(*_history(epoch), 10, figure)
    assert figure.patch.get_alpha() == 0.0
    for axis in figure.axes:
        assert axis.patch.get_alpha() == 0.0


def test_auto_figure_colours_are_transparent_not_black(monkeypatch):
    """The actual cause of the reported black box."""
    pytest.importorskip("PySide6")
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "dark")
    monkeypatch.setattr(preferences, "_settings",
                        lambda: _Store({"prefs/figure_bg": "auto",
                                        "prefs/figure_fg": "auto"}))
    bg, fg = preferences.get_figure_colors()
    assert preferences.figure_bg_is_transparent(bg), bg
    assert bg != "#000000"
    assert fg == "#ffffff"


def test_an_explicit_background_is_still_honoured(monkeypatch):
    """Only the "auto" resolution changed. A user who has chosen a colour
    must keep it."""
    pytest.importorskip("PySide6")
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "dark")
    monkeypatch.setattr(preferences, "_settings",
                        lambda: _Store({"prefs/figure_bg": "#123456",
                                        "prefs/figure_fg": "auto"}))
    bg, _fg = preferences.get_figure_colors()
    assert bg == "#123456"
    assert not preferences.figure_bg_is_transparent(bg)


class _Store:
    def __init__(self, values):
        self._values = values

    def value(self, key, default=None):
        return self._values.get(key, default)


# ---------------------------------------------------------------------------
# The palette
# ---------------------------------------------------------------------------

def test_the_curves_use_the_requested_palette():
    """Teal, blue, purple, grey -- asked for by name."""
    assert _CLASS_CURVE_COLORS[:4] == ('#2aa198', '#4A9EFF', '#9b7fd4',
                                       '#8a8f98')
    assert _TRAIN_CURVE_COLOR in _CLASS_CURVE_COLORS
    assert _VAL_CURVE_COLOR in _CLASS_CURVE_COLORS
    assert _TRAIN_CURVE_COLOR != _VAL_CURVE_COLOR


def test_no_curve_colour_is_pure_white_or_pure_black():
    """The figure is transparent and the pipeline does not know which theme
    is behind it, so a curve at either extreme vanishes on one of them."""
    for colour in _CLASS_CURVE_COLORS:
        assert colour.lower() not in ("#ffffff", "#000000", "#fff", "#000")
