"""The translator's remaining refusals: the artist it cannot paint, place or read.

Every branch here is one the scene translation takes when an artist does not
give it what it asked for -- a colour that is not paint, a position that is
nowhere, an axis that raises when asked what its ticks say, a bar drawn in
axes fraction that converts to infinity. The rule the module lives by is that
NONE of those may be fatal and none may be invented around: the mark is left
off and the page is still written.

Each test drives the refusal and the ordinary case side by side, because "no
item was added" is only interesting next to the input that does add one.
"""


import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt          # noqa: E402

from spacr.figures import scene as sc    # noqa: E402
from spacr.figures.scene import pyqtgraph_ready  # noqa: E402


@pytest.fixture
def closed_figures():
    """Close whatever a test drew, whichever way it ends."""
    yield
    plt.close("all")


@pytest.fixture
def qt_ready(qapp):
    ready, why = pyqtgraph_ready()
    if not ready:
        pytest.skip(why)
    return True


class _Collector:
    """The one method of a pyqtgraph ``PlotItem`` the translators call."""

    def __init__(self):
        self.items = []

    def addItem(self, item, **_ignored):
        self.items.append(item)


def _look():
    return sc._Look(None)


# ---------------------------------------------------------------------------
# starting a QApplication where there is no display
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# a Line2D
# ---------------------------------------------------------------------------

def test_a_one_point_line_is_not_a_reference_line(closed_figures):
    """A reference line is a direction, and one point has none.

    ``axhline`` puts two points down; a single-point marker drawn in the same
    blended transform is a mark, and treating it as an infinite line would
    draw a rule across the whole panel.
    """
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots()
    assert sc._reference_line(Line2D([0.5], [0.5]), axes) is None
    assert sc._reference_line(Line2D([0.0, 1.0], [0.5, 0.5]), axes) == "h"


# ---------------------------------------------------------------------------
# a scatter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# a LineCollection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# a Rectangle drawn in axes fraction
# ---------------------------------------------------------------------------

def test_a_backdrop_that_converts_to_nowhere_is_dropped(qt_ready,
                                                        closed_figures):
    """A fraction that lands off the number line is not a place to draw.

    `_skip_box` draws its backdrop in axes fraction; a corner that converts
    to infinity would reach `BarGraphItem` as a NaN geometry and take the
    whole bar set with it.
    """
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    nowhere = Rectangle((float("inf"), 0.0), 1.0, 1.0,
                        transform=axes.transAxes, facecolor="#DDDDDD")
    axes.add_patch(nowhere)
    somewhere = Rectangle((0.1, 0.1), 0.5, 0.5,
                          transform=axes.transAxes, facecolor="#DDDDDD")
    axes.add_patch(somewhere)

    plot = _Collector()
    assert sc._add_rectangles(plot, [nowhere], axes, _look()) == 0
    assert plot.items == []
    assert sc._add_rectangles(plot, [somewhere], axes, _look()) == 1


# ---------------------------------------------------------------------------
# mathtext
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# a Text
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# an axis that will not describe itself
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# a colour bar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# an artist the translator knows about but does not draw
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# a legend
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# laying the scene out
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tidying up after a render
# ---------------------------------------------------------------------------
