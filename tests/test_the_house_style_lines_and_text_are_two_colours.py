"""Lines and text are two colours in the house style, and both are the user's.

The maintainer's ask, in their own words: "i want one color option when
picking color for the graph elements. called line color which should cahnge
the color of all lines including axis lines and ticks. and then a font color
theat controlls the color of all font in the graph."

Three renderers draw spaCR figures and all three have to obey it. This file is
the third one -- the publication house style, the figures that go into the
manuscript. A control that reaches the live plot and stops before the saved
figure is the report this exists to keep answered.

THE SPLIT IS BY WHAT A MARK IS. The axis spines and the tick MARKS are lines;
the title, the axis labels, the tick LABELS and the annotations are text. The
tick is where the two meet, which is why it has a test of its own.

TWO PROPERTIES ARE EASY TO BREAK TOGETHER and both are pinned here:

* an untouched settings store must still draw the measured publication ink,
  not the theme's answer -- reading the RESOLVED preference instead of the
  stored token is what changes every figure in the package for people who
  never chose anything;
* the DATA keeps its own colours. A preference pushed over every series
  flattens every multi-series figure the moment a theme is read.
"""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from spacr.figures import style  # noqa: E402

preferences = pytest.importorskip("spacr.qt.preferences")

BLUE = "#0000ff"
RED = "#ff0000"


@pytest.fixture
def colours():
    """The figure colour preferences, handed back exactly as they were found.

    One sandboxed QSettings store serves the whole session and the suite is
    randomly ordered, so a test that leaves an explicit colour behind
    repaints the figures of a different set of tests every run. Restoring the
    raw values rather than calling the "follow the theme" setter matters too:
    that setter WRITES tokens, which is not the same state as a key that was
    never there.
    """
    keys = (preferences._KEY_FIG_BG, preferences._KEY_FIG_FG,
            preferences._KEY_FIG_LINE, preferences._KEY_FIG_COLORS_EXPLICIT,
            preferences._KEY_FIG_COLOR_SCALE, preferences._KEY_FIG_STYLE,
            preferences._KEY_FIG_STYLE_PER_GRAPH)
    settings = preferences._settings()
    before = {key: settings.value(key, None) for key in keys}
    yield preferences
    for key, value in before.items():
        if value is None:
            settings.remove(key)
        else:
            settings.setValue(key, value)
    settings.sync()


def _drawn(**kwargs):
    """Render one figure in the house style and read the colours off it.

    Off the ARTISTS, after a draw, rather than out of the rcParams dict: a
    parameter that is set and a spine that is painted are two different
    claims and only the second one is what the user sees.
    """
    with style.figure_style("print", **kwargs):
        figure, axes = plt.subplots()
        try:
            axes.plot([0, 1], [0, 1], color=style.ROLES["up"])
            axes.set_xlabel("x")
            axes.set_title("t")
            figure.canvas.draw()
            return {
                "spine": to_hex(axes.spines["left"].get_edgecolor()),
                "tick_mark": to_hex(axes.xaxis.get_ticklines()[0].get_color()),
                "tick_label": to_hex(axes.get_xticklabels()[0].get_color()),
                "axis_label": to_hex(axes.xaxis.label.get_color()),
                "title": to_hex(axes.title.get_color()),
                "data": to_hex(axes.lines[0].get_color()),
            }
        finally:
            plt.close(figure)


# --------------------------------------------------------------------------- #
#  The control reaches the figure
# --------------------------------------------------------------------------- #

def test_a_chosen_line_colour_reaches_the_spines_and_the_tick_marks(colours):
    """The report this exists for: the control changed the live plot only.

    Rendered and read back off the artists, because the whole complaint was
    about a figure that came out in the old ink.
    """
    colours.set_figure_line_colour(BLUE)

    painted = _drawn()

    assert painted["spine"] == BLUE
    assert painted["tick_mark"] == BLUE


def test_the_tick_mark_is_a_line_and_the_tick_label_is_text(colours):
    """The one place the two controls meet, decided and pinned.

    The little dash beside the axis follows the line colour; the number
    printed next to it follows the font colour. Anything else makes "all font
    in the graph" untrue, and the two are one rcParam apart -- `xtick.color`
    and `xtick.labelcolor` -- so they come back equal unless both are named.
    """
    colours.set_figure_line_colour(BLUE)
    colours.set_figure_colors(colours.AUTO_FIGURE_COLOR, RED)

    painted = _drawn()

    assert painted["tick_mark"] == BLUE
    assert painted["tick_label"] == RED
    assert painted["axis_label"] == RED and painted["title"] == RED
    assert painted["spine"] == BLUE


def test_the_data_keeps_its_own_colour_when_the_chrome_is_repainted(colours):
    """A line colour is the figure's frame, not a repaint of every series.

    A preference that reached the data would flatten every multi-series
    figure in the package into one ink the first time a theme was read.
    """
    colours.set_figure_line_colour(BLUE)

    assert _drawn()["data"] == style.ROLES["up"].lower()


# --------------------------------------------------------------------------- #
#  What an untouched store must still draw
# --------------------------------------------------------------------------- #

def test_an_untouched_store_draws_the_published_ink(colours):
    """Nothing chosen means the measured house ink, in every role.

    This fails if the line colour is read through the RESOLVED preference
    rather than the stored token: "auto" resolves to the theme's flat black
    or white, so every figure in the package would change colour for every
    user who never picked anything.
    """
    colours.set_figure_colors_auto()

    painted = _drawn()

    assert set(painted.values()) == {style.INK_PRINT.lower(),
                                     style.ROLES["up"].lower()}


def test_the_lines_follow_the_text_until_a_line_colour_is_chosen(colours):
    """One colour chosen paints the whole figure, as it did before the split.

    The split is only allowed to cost somebody a changed figure once they ask
    for two colours.
    """
    colours.set_figure_colors_auto()
    colours.set_figure_colors(colours.AUTO_FIGURE_COLOR, RED)

    painted = _drawn()

    assert painted["spine"] == RED and painted["tick_mark"] == RED
    assert painted["tick_label"] == RED


def test_follow_the_theme_puts_the_published_ink_back(colours):
    """The way out. A preference you can only ever set is a trap."""
    untouched = style.rc("print")

    colours.set_figure_line_colour(BLUE)
    colours.set_figure_colors(colours.AUTO_FIGURE_COLOR, RED)
    assert style.rc("print") != untouched

    colours.set_figure_colors_auto()
    assert style.rc("print") == untouched


# --------------------------------------------------------------------------- #
#  Who outranks whom
# --------------------------------------------------------------------------- #

def test_a_caller_that_names_the_colours_outranks_the_preference(colours):
    """A panel that already knows its colours never has to ask the store."""
    colours.set_figure_line_colour(BLUE)
    colours.set_figure_colors(colours.AUTO_FIGURE_COLOR, RED)

    params = style.rc("print", ink="#123456", line="#654321")

    assert params["axes.edgecolor"] == "#654321"
    assert params["xtick.color"] == "#654321"
    assert params["text.color"] == "#123456"
    assert params["xtick.labelcolor"] == "#123456"


def test_the_figure_colours_outrank_the_graph_style_panels_foreground(colours):
    """Two settings surfaces name a colour; the more specific one wins.

    The graph-style panel's `foreground` is one general control that resolves
    to `xtick.color` among other things, so without an order it silently
    repaints the tick marks the line control was just told to own -- and the
    user would be looking at the colour they did not pick in the dialog they
    did not use.
    """
    colours.set_figure_style({"foreground": "#00aa00"})
    colours.set_figure_line_colour(BLUE)
    colours.set_figure_colors(colours.AUTO_FIGURE_COLOR, RED)

    params = style.rc("print", kind="volcano")

    assert params["xtick.color"] == BLUE
    assert params["axes.edgecolor"] == BLUE
    assert params["text.color"] == RED
    assert params["xtick.labelcolor"] == RED


def test_the_panels_foreground_still_reaches_a_figure_nobody_recoloured(
        colours):
    """The order above must not turn the general control off.

    While both figure-colour halves follow the theme the panel's own
    foreground is the only colour anybody chose, so it has to arrive.
    """
    colours.set_figure_colors_auto()
    colours.set_figure_style({"foreground": "#00aa00"})

    assert style.rc("print")["text.color"] == "#00aa00"


# --------------------------------------------------------------------------- #
#  No settings store at all
# --------------------------------------------------------------------------- #

def test_a_settings_store_that_raises_leaves_the_published_ink(monkeypatch):
    """Figures are drawn from headless workers and from bare unit runs.

    A preference read that throws must not take the plot down with it, and a
    figure drawn with no colour at all is a figure with invisible axes.
    """
    def _explode(*args, **kwargs):
        raise RuntimeError("no settings store")

    monkeypatch.setattr(preferences, "get_figure_color_tokens", _explode)
    monkeypatch.setattr(preferences, "get_figure_line_token", _explode)

    assert style.chosen_ink() is None
    assert style.chosen_line_ink() is None
    assert style.resolve_ink("print") == style.INK_PRINT
    assert style.resolve_line_ink("print") == style.INK_PRINT


def test_the_ink_survives_preferences_not_being_importable(monkeypatch):
    """Qt is optional at render time; the house style is not."""
    monkeypatch.setitem(__import__("sys").modules, "spacr.qt.preferences",
                        None)

    assert style.chosen_ink() is None
    assert style.chosen_line_ink() is None
    assert style.rc("screen")["axes.edgecolor"] == style.INK_SCREEN
