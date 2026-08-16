"""A fast plot is transparent, and its ink follows the theme.

Instruction 118: "figures should not have a background not black not white
just transparent".

The background half was already right -- pyqtgraph was configured with
``background=None``. Two things were not:

  * ``foreground="k"`` hardcoded BLACK axes, ticks and labels. On a dark
    theme that is black ink on a transparent page over a dark surface, i.e.
    invisible axes. The matplotlib renderer had resolved this correctly for
    a while through ``preferences.get_figure_colors``; pyqtgraph never asked.
  * the QWidget around the plot still painted the theme's ``bg`` under the
    blanket QWidget rule, so a "transparent" plot sat on an opaque slab
    anyway.

And the export wrote pyqtgraph's config background onto the saved image, so
the one place transparency mattered most -- the file the user puts in a
paper -- was the one place it was lost.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt

pytest.importorskip("pyqtgraph")


@pytest.fixture
def plot(qtbot):
    from spacr.qt.widgets.fast_plots import FastPlot

    widget = FastPlot(title="t", x_label="x", y_label="y")
    qtbot.addWidget(widget)
    return widget


def test_the_ink_is_not_hardcoded_black(plot):
    """The defect, stated as a test: black axes on a dark theme are no
    axes."""
    assert plot._foreground != "k"
    assert plot._foreground.startswith("#")


def test_the_ink_comes_from_the_same_place_matplotlib_gets_it(plot):
    """One source, so a theme switch cannot move one renderer and not the
    other."""
    from spacr.qt.preferences import get_figure_colors

    _bg, fg = get_figure_colors()
    assert plot._foreground == fg


def test_the_axes_actually_carry_that_colour(plot):
    """Storing the value is not the same as drawing with it."""
    for edge in ("bottom", "left"):
        pen = plot.plot.getAxis(edge).pen()
        assert pen.color().name().lower() == plot._foreground.lower()


def test_the_widget_does_not_paint_its_own_background(plot):
    """A transparent pyqtgraph background is undone by the QWidget around
    it painting the theme's bg."""
    assert plot.autoFillBackground() is False


def test_restyle_re_inks_a_live_plot(plot):
    """pyqtgraph resolves foreground at construction, so without this a
    theme switch leaves every open plot drawing its old ink -- and on a
    dark-to-light switch that ink is invisible."""
    plot.restyle(foreground="#ff0000")

    assert plot._foreground == "#ff0000"
    assert plot.plot.getAxis("bottom").pen().color().name() == "#ff0000"


def test_restyle_with_no_arguments_re_reads_the_preferences(plot):
    from spacr.qt.preferences import get_figure_colors

    plot.restyle(foreground="#ff0000")
    plot.restyle()

    _bg, fg = get_figure_colors()
    assert plot._foreground == fg


def test_an_exported_png_keeps_its_transparency(plot, tmp_path):
    """The one place transparency matters most is the file that leaves the
    app, and it was the one place it was lost."""
    from PIL import Image

    plot.add_scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    target = tmp_path / "plot.png"
    plot.export(str(target))

    assert target.is_file()
    image = Image.open(target)
    assert image.mode == "RGBA", f"exported as {image.mode}, not RGBA"
    # The corners are outside the axes, so they must be fully transparent.
    width, height = image.size
    for corner in ((0, 0), (width - 1, 0), (0, height - 1)):
        assert image.getpixel(corner)[3] == 0, (
            f"pixel {corner} is opaque -- the export painted a background")


def test_the_colour_helper_survives_no_settings_store():
    """Reached in a headless render or a bare unit test. White, because
    every spaCR theme but one is dark and invisible axes are worse than
    slightly wrong ones."""
    import builtins

    from spacr.qt.widgets import fast_plots

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):
        if "preferences" in name:
            raise ImportError("no settings store")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _fail
    try:
        background, foreground = fast_plots._figure_colors()
    finally:
        builtins.__import__ = real_import

    assert background == "none"
    assert foreground == fast_plots._FALLBACK_FOREGROUND
