"""The three figure-style controls that the frame, the mark and the page use.

`chrome_colour`, `marker_style` and `page_shape` are offered by the figure
settings and by Preferences, so each of them has to move something on a drawn
figure. The path is the only one a figure has:
`spacr.figures.style.user_overrides` builds its overrides by DIFFING two
`rc_params` dicts, so a setting that function does not emit cannot reach a
figure at all -- the control is drawn, it is set, and nothing happens.
"""
import pytest


@pytest.fixture()
def base():
    from spacr.figure_style import resolve

    return resolve(None)


def _diff(base, changed):
    from spacr.figure_style import rc_params

    before, after = rc_params(base), rc_params(changed)
    return {key: value for key, value in after.items()
            if before.get(key) != value}


class TestTheFrameIsOneInk:
    """D: the spines, the tick marks and the grid are the same furniture."""

    def test_it_colours_the_spines(self, base):
        diff = _diff(base, {**base, "chrome_colour": "#FF0000"})
        assert diff.get("axes.edgecolor") == "#FF0000"

    def test_it_colours_the_ticks(self, base):
        diff = _diff(base, {**base, "chrome_colour": "#FF0000"})
        assert diff.get("xtick.color") == "#FF0000"
        assert diff.get("ytick.color") == "#FF0000"

    def test_it_colours_the_grid(self, base):
        diff = _diff(base, {**base, "chrome_colour": "#FF0000"})
        assert diff.get("grid.color") == "#FF0000"

    def test_a_chosen_grid_colour_still_wins(self, base):
        """The one control is the fallback; a per-element choice outranks it."""
        diff = _diff(base, {**base, "chrome_colour": "#FF0000",
                            "grid_colour": "#00FF00"})
        assert diff.get("grid.color") == "#00FF00"
        assert diff.get("axes.edgecolor") == "#FF0000"

    def test_saying_nothing_changes_nothing(self, base):
        assert _diff(base, dict(base)) == {}


class TestTheMark:
    """C: the marker shape is a control, not a matplotlib code to memorise."""

    def test_a_chosen_shape_reaches_the_figure(self, base):
        assert _diff(base, {**base, "marker_style": "^"}).get(
            "lines.marker") == "^"

    def test_the_default_shape_is_not_forced_on_every_line(self, base):
        """These params are pushed into the GLOBAL rcParams, so naming the
        default would put a marker on every line ever drawn."""
        from spacr.figure_style import rc_params

        assert "lines.marker" not in rc_params(base)


class TestTheShapeOfThePage:
    """E: a named ratio, and one number in gives two out."""

    def test_square_is_square(self, base):
        width, height = _diff(base, {**base, "page_shape": "square"})[
            "figure.figsize"]
        assert width == pytest.approx(height)

    def test_wide_is_wider_than_landscape(self, base):
        wide = _diff(base, {**base, "page_shape": "wide"})["figure.figsize"]
        tall = _diff(base, {**base, "page_shape": "portrait"})["figure.figsize"]
        assert wide[0] / wide[1] > tall[0] / tall[1]

    def test_custom_keeps_the_callers_inches(self, base):
        """`custom` means the caller's own size -- overwriting it silently
        would be the worse failure."""
        assert "figure.figsize" not in _diff(base, {**base,
                                                   "page_shape": "custom"})


def test_every_emitted_key_is_a_real_rcparam(base):
    """`apply` pushes this dict into matplotlib, which raises on a key it does
    not have."""
    import matplotlib

    from spacr.figure_style import rc_params

    for style in ({**base, "chrome_colour": "#FF0000"},
                  {**base, "marker_style": "^"},
                  {**base, "page_shape": "square"}):
        for key in rc_params(style):
            assert key in matplotlib.rcParams, key
