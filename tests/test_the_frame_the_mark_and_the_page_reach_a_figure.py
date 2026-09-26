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
                  {**base, "page_shape": "square"},
                  {**base, "mark_colouring": "random"}):
        for key in rc_params(style):
            assert key in matplotlib.rcParams, key


# -- mark_colouring: it was the one control of the four still inert --------

def _cycle(params):
    return [entry["color"] for entry in params["axes.prop_cycle"]]


class TestWhichColourEachMarkTakes:
    """`mark_colouring` reaches a figure two ways: the colour cycle every
    renderer that names no colour draws from, and the grouped renderers that
    do name theirs (`spacr.figures.style._group_colours`)."""

    def test_group_keeps_the_palette_in_its_order(self, base):
        from spacr.figure_style import palette_colours, rc_params

        assert _cycle(rc_params(base)) == palette_colours(base["palette"])

    def test_uniform_is_one_colour_and_reaches_the_overrides(self, base):
        from spacr.figure_style import palette_colours, rc_params

        uniform = {**base, "mark_colouring": "uniform"}
        assert _cycle(rc_params(uniform)) == \
            palette_colours(base["palette"])[:1]
        assert "axes.prop_cycle" in _diff(base, uniform)

    def test_random_reorders_the_palette_the_same_way_every_time(self, base):
        from spacr.figure_style import palette_colours, rc_params

        random_style = {**base, "mark_colouring": "random"}
        drawn = _cycle(rc_params(random_style))
        palette = palette_colours(base["palette"])
        assert sorted(drawn) == sorted(palette)
        assert drawn != palette
        assert drawn == _cycle(rc_params(random_style))

    def test_apply_does_not_put_the_full_palette_back(self, base):
        """`apply` used to re-set the palette after `rc_params`, which would
        have undone the rule on the one path that writes global rcParams."""
        import matplotlib

        from spacr.figure_style import apply, palette_colours

        with matplotlib.rc_context():
            apply(None, {"mark_colouring": "uniform"})
            cycle = [entry["color"]
                     for entry in matplotlib.rcParams["axes.prop_cycle"]]
        assert cycle == palette_colours(base["palette"])[:1]


@pytest.fixture()
def chosen_colouring(monkeypatch):
    """Set the user's `mark_colouring` preference without touching disk."""
    import spacr.qt.preferences as preferences

    def choose(rule):
        monkeypatch.setattr(preferences, "get_figure_style",
                            lambda: {"mark_colouring": rule})
        monkeypatch.setattr(preferences, "get_figure_style_per_graph",
                            lambda: {})
    return choose


class TestTheGroupedRenderersReadIt:

    def test_group_leaves_the_house_rule_alone(self, chosen_colouring):
        from spacr.figures.style import _group_colours

        chosen_colouring("group")
        assert _group_colours(4, ["#111111", "#222222"]) is None

    def test_uniform_is_the_data_ink_for_every_group(self, chosen_colouring):
        from spacr.figures.style import ROLES, _group_colours

        chosen_colouring("uniform")
        assert _group_colours(3, ["#111111", "#222222"]) == [ROLES["data"]] * 3

    def test_random_is_the_palette_reordered_and_stable(self,
                                                        chosen_colouring):
        from spacr.figures.style import _group_colours

        palette = ["#000001", "#000002", "#000003", "#000004", "#000005"]
        chosen_colouring("random")
        drawn = _group_colours(5, palette)
        assert sorted(drawn) == palette and drawn != palette
        assert _group_colours(5, palette) == drawn

    def test_spacr_graph_draws_by_the_rule(self, chosen_colouring):
        """The one-grey house rule for a single column, replaced by the
        user's choice when they made one."""
        from types import SimpleNamespace

        from spacr.figures.style import ROLES
        from spacr.plot import spacrGraph

        graph = SimpleNamespace(colors=None, data_column=["value"],
                                sns_palette=["#000001", "#000002",
                                             "#000003"])
        chosen_colouring("group")
        assert spacrGraph._plot_palette(graph, 3) == [ROLES["data"]] * 3
        chosen_colouring("random")
        drawn = spacrGraph._plot_palette(graph, 3)
        assert len(set(drawn)) == 3
        explicit = SimpleNamespace(colors=["red"], data_column=["value"],
                                   sns_palette=[])
        assert spacrGraph._plot_palette(explicit, 2) == ["red", "red"]
