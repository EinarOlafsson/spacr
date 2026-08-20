"""Instruction 150 C: one decision, asked by both renderers.

The matplotlib half shipped at 80e72695. What these cover is the part that
makes the pyqtgraph half a CALL rather than a second implementation:
``spacr.figure_style.export_colour`` answers "what should this artist be
painted for the save", ``illegible_colours`` answers section D, and
``spacr.plot.print_ready`` -- the matplotlib application -- now goes through
both rather than keeping its own copy of the rule.

Nothing here imports matplotlib or Qt for the decision itself, deliberately:
if the rule needed a renderer to state it, the two renderers could not share
it.
"""
import pytest

from spacr import figure_style as style


@pytest.fixture()
def printing():
    return style.saved_figure_appearance("print")


# ------------------------------------------------- the chrome flips ...

@pytest.mark.parametrize("kind", ["chrome", "grid"])
def test_illegible_chrome_is_repainted(kind, printing):
    assert style.export_colour("#FFFFFF", kind, printing) is not None


def test_a_grid_is_repainted_faint_and_not_in_the_ink(printing):
    """A grid repainted in the ink is a cage over the data."""
    assert style.export_colour("#FFFFFF", "grid", printing) == style.PRINT_GRID
    assert style.export_colour("#FFFFFF", "chrome", printing) == style.PRINT_INK


def test_chrome_that_can_already_be_read_is_left_alone(printing):
    """The property that makes 'print' safe as the default: a light-mode save
    changes nothing at all."""
    assert style.export_colour("#222222", "chrome", printing) is None
    assert style.export_colour("#DDDDDD", "grid", printing) is None


def test_a_dark_ground_becomes_the_page(printing):
    assert style.export_colour("#1E1E1E", "ground", printing) == style.PRINT_GROUND


def test_a_light_ground_is_somebody_s_choice(printing):
    """A deliberately tinted light background is not a bug to be corrected."""
    assert style.export_colour("#FFF8E7", "ground", printing) is None


# ------------------------------------------------- ... and the data does not

def test_a_white_data_point_stays_white(printing):
    """On a volcano, black is the colour of 'not a hit'."""
    assert style.export_colour("#FFFFFF", "data", printing) is None


def test_no_kind_of_data_is_ever_repainted(printing):
    for colour in ("#FFFFFF", "#000000", "none", (1.0, 1.0, 1.0),
                   (1.0, 1.0, 1.0, 0.0), "#F0E442"):
        assert style.export_colour(colour, "data", printing) is None


def test_screen_mode_repaints_nothing_at_all():
    """That is the whole meaning of the mode."""
    look = style.saved_figure_appearance("screen")
    for kind in style.ARTIST_KINDS:
        assert style.export_colour("#FFFFFF", kind, look) is None


def test_transparent_uses_the_theme_ink_and_drops_the_ground(monkeypatch):
    monkeypatch.setattr(style, "theme_ink", lambda: ("#F2F2F2", "#555555"))
    look = style.saved_figure_appearance("transparent")
    assert style.export_colour("#FFFFFF", "chrome", look) == "#F2F2F2"
    assert style.export_colour("#FFFFFF", "grid", look) == "#555555"
    assert style.export_colour("#1E1E1E", "ground", look) is None


def test_a_colour_that_cannot_be_read_is_left_alone(printing):
    """The safe direction: never repaint something that was never white."""
    for colour in ("none", "transparent", (1.0, 1.0, 1.0, 0.0), object()):
        assert style.export_colour(colour, "chrome", printing) is None


def test_the_default_look_is_asked_for_when_none_is_given(monkeypatch):
    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "screen")
    assert style.export_colour("#FFFFFF", "chrome") is None
    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "print")
    assert style.export_colour("#FFFFFF", "chrome") == style.PRINT_INK


# ------------------------------------------------------ section D, shared

def test_a_colour_chosen_for_a_dark_ground_is_named():
    named = style.illegible_colours(["#FFFFFF", "#F0E442", "#2E7D4F"])
    assert named == ["#F0E442", "#FFFFFF"]


def test_the_house_style_s_own_greys_stay_quiet():
    """A warning that fires on every figure is a warning nobody reads."""
    from spacr.figures.style import ROLES

    assert style.illegible_colours(list(ROLES.values())) == []


def test_de_emphasis_is_not_held_to_a_legibility_floor():
    """`figures.plates` lays its 'never measured' wash down at 9% opacity."""
    assert style.illegible_colours([(1.0, 1.0, 1.0, 0.09)]) == []
    assert style.illegible_colours([(1.0, 1.0, 1.0, 1.0)]) == ["#FFFFFF"]


def test_the_same_palette_produces_the_same_sentence_twice():
    palette = ["#FFFFFF", "#F0E442", "#FFFFFF"]
    assert style.illegible_colours(palette) == style.illegible_colours(
        list(reversed(palette)))


def test_the_colour_is_named_and_never_substituted():
    message = style.illegible_colour_warning(["#FFFFFF"])
    assert "#FFFFFF" in message
    assert "NOT being changed" in message


def test_nothing_illegible_says_nothing():
    assert style.illegible_colour_warning([]) == ""
    assert style.illegible_colours([]) == []


def test_an_unreadable_entry_is_skipped_rather_than_guessed_at():
    assert style.illegible_colours(["not a colour", None, "#FFFFFF"]) == [
        "#FFFFFF"]


# ------------------------------- the matplotlib half asks the shared question

def test_the_matplotlib_half_uses_the_shared_decision(monkeypatch):
    """A CALL, not a copy: if plot.py kept its own rule, patching the shared
    one here would leave the figure's chrome alone."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from spacr.plot import print_ready

    figure, ax = plt.subplots()
    ax.set_title("title", color="#FFFFFF")
    try:
        monkeypatch.setattr("spacr.figure_style.export_colour",
                            lambda current, kind, look=None:
                            "#FF00FF" if kind == "chrome" else None)
        with print_ready(figure, mode="print", announce=False):
            assert ax.title.get_color() == "#FF00FF"
        # And nothing survives the block.
        assert ax.title.get_color() == "#FFFFFF"
    finally:
        plt.close(figure)


def test_section_d_reads_the_shared_floor(monkeypatch):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from spacr.plot import illegible_data_colours

    figure, ax = plt.subplots()
    ax.scatter([0, 1], [0, 1], c="#FFFFFF")
    try:
        assert illegible_data_colours(figure, "#FFFFFF") == ["#FFFFFF"]
    finally:
        plt.close(figure)
