"""Two arms of ``spacr.plot`` that only a caller from outside can reach.

Round 5's remaining ``spacr.plot`` targets were almost all guards that the
module's own callers have already made true -- a ``None`` default replaced at
the top of the function and re-checked at the bottom, a closure whose single
call site passes the argument explicitly, a branch chain that enumerates
exactly the values an earlier ``raise`` allows through. Those are written up
in the round's report rather than contorted into tests.

Two are genuinely reachable, and both are about a helper being handed
something its in-module caller never hands it:

* ``_chrome`` walking a figure whose ``axes`` list holds something that is
  not an ``Axes``. The theme walker recurses into every entry, and a
  non-Axes one has no spines, ticks or title to recolour.
* ``spacrGraph._standerdize_figure_format`` with a graph type it has no
  layout rule for. The shared work -- figure size, x limits, tick sizes --
  still has to happen; only the per-type artist tweak is skipped.

Both are paired against the input that DOES produce the effect, so neither
assertion is about an absence on its own.
"""
from __future__ import annotations

import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("seaborn")
pytest.importorskip("statsmodels")


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _chrome: the figure-level walk
# ---------------------------------------------------------------------------

class _FigureWithAStranger:
    """A figure-shaped object whose ``axes`` list holds one non-Axes entry.

    ``_chrome`` is duck-typed on ``patch``/``texts``/``legends``/``axes``, and
    the ``isinstance`` filter in the figure-level walk exists for exactly this
    -- an entry that has no chrome to recolour must be stepped over rather
    than recursed into, which would raise on the first ``.spines`` access.
    """

    def __init__(self, figure, stranger):
        self.patch = figure.patch
        self.texts = list(figure.texts)
        self.legends = list(getattr(figure, "legends", []))
        self.axes = [stranger, *figure.axes]


def test_the_theme_walk_steps_over_something_that_is_not_an_axes():
    from spacr.plot import _chrome

    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("a title")
    figure.suptitle("a suptitle")

    real = [kind for kind, _artist, _get, _set in _chrome(figure)]
    assert "chrome" in real and "text" in real and "ground" in real

    stranger = object()
    with_stranger = [kind for kind, _artist, _get, _set
                     in _chrome(_FigureWithAStranger(figure, stranger))]

    # The stranger contributed nothing and cost nothing: the same chrome
    # comes back, in the same order, and no artist in it is the stranger.
    assert with_stranger == real
    assert all(artist is not stranger for _kind, artist, _get, _set
               in _chrome(_FigureWithAStranger(figure, stranger)))


# ---------------------------------------------------------------------------
# spacrGraph._standerdize_figure_format: a type with no layout rule
# ---------------------------------------------------------------------------

def _bar_axes():
    figure, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0, 1], [1.0, 2.0], width=0.8)
    return figure, ax


def test_a_graph_type_with_no_layout_rule_still_gets_the_shared_layout():
    """The per-type step narrows bars, shifts jitter or resizes boxes. A type
    with no rule of its own must still come out square, limited and ticked --
    the layout the method exists to standardise."""
    from spacr.plot import spacrGraph

    frame = pd.DataFrame({"grp": ["a"] * 5 + ["b"] * 5,
                          "val": [float(v) for v in range(10)]})
    graph = spacrGraph(frame.copy(), "grp", "val", graph_type="bar")

    known_fig, known_ax = _bar_axes()
    unknown_fig, unknown_ax = _bar_axes()
    assert [p.get_width() for p in known_ax.patches] == [0.8, 0.8]

    graph._standerdize_figure_format(ax=known_ax, num_groups=2,
                                     graph_type="bar")
    graph._standerdize_figure_format(ax=unknown_ax, num_groups=2,
                                     graph_type="scatter")

    # 'bar' has a rule: min(0.8, 1.5/2) / 4.
    assert [p.get_width() for p in known_ax.patches] == [0.1875, 0.1875]
    # 'scatter' has none, so the bars are left exactly as they were...
    assert [p.get_width() for p in unknown_ax.patches] == [0.8, 0.8]
    # ... but the shared layout was applied to both.
    assert tuple(unknown_fig.get_size_inches()) == (10.0, 10.0)
    assert tuple(known_fig.get_size_inches()) == (10.0, 10.0)
    assert unknown_ax.get_xlim() == (-0.5, 1.5)
    assert known_ax.get_xlim() == (-0.5, 1.5)


def test_a_line_graph_is_left_out_of_the_layout_pass_entirely(capsys):
    """The early return is the one type that opts out of even the shared
    work, and it says so; every other type is resized."""
    from spacr.plot import spacrGraph

    frame = pd.DataFrame({"grp": ["a"] * 5 + ["b"] * 5,
                          "val": [float(v) for v in range(10)]})
    graph = spacrGraph(frame.copy(), "grp", "val", graph_type="bar")

    line_fig, line_ax = _bar_axes()
    graph._standerdize_figure_format(ax=line_ax, num_groups=2,
                                     graph_type="line")

    assert tuple(line_fig.get_size_inches()) == (4.0, 4.0)
    assert "Skipping layout adjustment for line graphs." in \
        capsys.readouterr().out

    other_fig, other_ax = _bar_axes()
    graph._standerdize_figure_format(ax=other_ax, num_groups=2,
                                     graph_type="scatter")
    assert tuple(other_fig.get_size_inches()) == (10.0, 10.0)
