"""108 points 1 and 2: the shared style base, and the renderer contract.

    1. THE SHARED STYLE BASE (spacr/figure_style.py holds general/per-graph
       DICTS today, not a dataclass hierarchy; volcano_style.VolcanoStyle is
       the only dataclass).
    2. THE RENDERER CONTRACT `render(data, style, figure=None,
       save_path=None)` per figure type -- only the volcano has it.

A BASE WITH ONE SUBCLASS PROVES NOTHING, which is why both land together and
why the tests below check the two styles against EACH OTHER rather than
against a list. The payoff is stated in the base's own docstring: a font size
chosen on a volcano reaches the comparison figure beside it, and the volcano's
effect-size threshold does not follow it there.
"""
from __future__ import annotations

import dataclasses
import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr.gene_measurement_compare import (Comparison, ComparisonStyle,
                                            render_comparison)
from spacr.style_base import (SHARED_CHOICES, FigureStyle, apply_page,
                              style_kind)
from spacr.volcano_style import VolcanoStyle, render_volcano


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def comparison():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "unit": [f"w{i}" for i in range(120)],
        "group": ["the gene"] * 40 + ["the rest"] * 80,
        "value": np.concatenate([rng.normal(1.0, 0.2, 40),
                                 rng.normal(0.0, 0.2, 80)]),
    })
    return Comparison(measurement="cell_area", level="well", frame=frame)


# --------------------------------------------------------------------------- #
#  1. The shared base
# --------------------------------------------------------------------------- #

class TestBothStylesShareOneVocabulary:

    def test_the_shared_style_documents_every_editable_field(self):
        """Every portable control is explained where subclasses inherit it."""
        documentation = FigureStyle.__doc__ or ""
        missing = [
            item.name
            for item in dataclasses.fields(FigureStyle)
            if f":param {item.name}:" not in documentation
        ]
        assert not missing, missing

    def test_the_comparison_style_documents_every_constructor_parameter(self):
        """Inherited and comparison-specific controls all reach its API page."""
        documentation = inspect.getdoc(ComparisonStyle) or ""
        assert "Parameters\n----------" in documentation
        missing = [
            item.name
            for item in dataclasses.fields(ComparisonStyle)
            if f"\n{item.name} :" not in documentation
        ]
        assert not missing, missing

    def test_both_are_figure_styles(self):
        assert isinstance(VolcanoStyle(), FigureStyle)
        assert isinstance(ComparisonStyle(), FigureStyle)

    def test_the_general_fields_are_on_both_under_the_same_names(self):
        shared = {f.name for f in dataclasses.fields(FigureStyle())}
        for style in (VolcanoStyle(), ComparisonStyle()):
            have = {f.name for f in dataclasses.fields(style)}
            assert shared <= have, sorted(shared - have)

    def test_and_they_agree_on_the_defaults(self):
        """Two figures whose 'font size 10' means two things is the drift a
        shared base exists to prevent."""
        for name in ("font_size", "grid", "grid_axis", "dpi", "legend",
                     "hide_top_right_spines", "transparent"):
            assert getattr(VolcanoStyle(), name) \
                == getattr(ComparisonStyle(), name), name

    def test_a_house_style_crosses_between_them(self):
        volcano = VolcanoStyle(font_size=14.0, grid=False, dpi=600)

        shared = volcano.shared_with(ComparisonStyle())
        applied = ComparisonStyle(**shared)

        assert applied.font_size == 14.0
        assert applied.grid is False
        assert applied.dpi == 600

    def test_but_a_volcano_only_setting_does_not_cross(self):
        """An effect-size threshold on a jitter plot is the 'present but
        inert' control instruction 106 forbids."""
        volcano = VolcanoStyle(effect_threshold=1.5, threshold_method="mad")

        shared = volcano.shared_with(ComparisonStyle())

        assert "effect_threshold" not in shared
        assert "threshold_method" not in shared

    def test_the_volcano_keeps_every_field_it_had(self):
        """Inheritance must not quietly drop one: the restyle menu is built
        from `dataclasses.fields`, so a lost field is a lost menu entry."""
        names = {f.name for f in dataclasses.fields(VolcanoStyle())}

        for name in ("x_column", "y_column", "effect_threshold",
                     "threshold_method", "significant_color", "color_by",
                     "shape_by", "annotations", "split_axis", "x_label",
                     "font_size", "dpi", "transparent", "legend_location"):
            assert name in names, name

    def test_the_volcanos_own_default_label_survived(self):
        """A base class cannot know what this figure's x axis is."""
        assert VolcanoStyle().x_label == "Standardized marginal effect"

    def test_choices_is_not_a_field(self):
        """Annotated as an ordinary type it becomes a dataclass FIELD, and
        the restyle menu -- which skips nothing -- would offer to edit the
        list of choices itself."""
        for style in (FigureStyle(), VolcanoStyle(), ComparisonStyle()):
            assert "CHOICES" not in {f.name
                                     for f in dataclasses.fields(style)}

    def test_a_subclass_extends_the_shared_choices_rather_than_replacing(self):
        assert set(SHARED_CHOICES) <= set(ComparisonStyle.CHOICES)
        assert "kind" in ComparisonStyle.CHOICES

    def test_the_kind_is_derived_from_the_class(self):
        assert style_kind(VolcanoStyle()) == "volcano"
        assert style_kind(ComparisonStyle()) == "comparison"


# --------------------------------------------------------------------------- #
#  2. The renderer contract
# --------------------------------------------------------------------------- #

class TestBothRenderersHonourTheContract:

    def test_the_signatures_match(self):
        import inspect

        for renderer in (render_volcano, render_comparison):
            parameters = inspect.signature(renderer).parameters
            assert list(parameters)[:2] == ["results", "style"] \
                or list(parameters)[:2] == ["comparison", "style"], renderer
            for name in ("figure", "save_path"):
                assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY

    def test_it_returns_a_figure_and_axes(self, comparison):
        figure, axes = render_comparison(comparison)

        assert figure is not None and axes is not None

    def test_nothing_to_draw_is_two_Nones_not_a_crash(self):
        empty = Comparison(measurement="x", level="well",
                           frame=pd.DataFrame(columns=["group", "value"]))

        assert render_comparison(empty) == (None, None)

    def test_it_draws_into_a_figure_it_was_given(self, comparison):
        """A live canvas redrawn IN PLACE is what keeps a restyle from
        resetting the zoom, and it is why the parameter is in the contract."""
        mine = plt.figure(figsize=(3, 2))

        figure, _axes = render_comparison(comparison, figure=mine)

        assert figure is mine

    def test_and_the_old_content_is_gone(self, comparison):
        """One axes, and none of it is the previous drawing. Not "no lines":
        the default kind is a box plot, whose whiskers, caps and medians ARE
        lines -- ten of them."""
        mine = plt.figure()
        mine.add_subplot(111).plot([0.0, 1.0], [1.0, 0.0], label="the old one")

        _figure, axes = render_comparison(comparison, figure=mine)

        assert len(mine.axes) == 1
        assert all(line.get_label() != "the old one" for line in axes.lines)

    def test_save_path_writes_through_the_one_writer(self, comparison,
                                                     tmp_path):
        target = tmp_path / "deep" / "comparison.png"

        render_comparison(comparison, save_path=str(target))

        assert target.is_file()
        assert target.stat().st_size > 0


class TestTheStyleReachesTheFigure:

    def test_the_page_size_is_the_styles(self, comparison):
        figure, _axes = render_comparison(
            comparison, ComparisonStyle(figure_width=8.0, figure_height=3.0))

        assert tuple(figure.get_size_inches()) == pytest.approx((8.0, 3.0))

    def test_the_title_and_labels_are_the_styles(self, comparison):
        _figure, axes = render_comparison(
            comparison, ComparisonStyle(title="a title", y_label="a y"))

        assert axes.get_title() == "a title"
        assert axes.get_ylabel() == "a y"

    def test_a_blank_label_leaves_the_measurements_name(self, comparison):
        """"Leave it alone" and "set it to nothing" have to stay different."""
        _figure, axes = render_comparison(comparison, ComparisonStyle())

        assert axes.get_ylabel() == "cell_area"

    def test_the_grid_turns_OFF(self, comparison):
        """`grid(False, linewidth=...)` ENABLES the grid -- matplotlib warns
        and does it anyway. Spelled once in `apply_page` so a second renderer
        cannot meet it again."""
        _figure, axes = render_comparison(comparison,
                                          ComparisonStyle(grid=False))

        assert not axes.xaxis._major_tick_kw.get("gridOn")
        assert not axes.yaxis._major_tick_kw.get("gridOn")

    def test_and_on(self, comparison):
        _figure, axes = render_comparison(
            comparison, ComparisonStyle(grid=True, grid_axis="both"))

        assert axes.yaxis._major_tick_kw.get("gridOn")

    def test_grid_axis_none_is_off_whatever_grid_says(self, comparison):
        _figure, axes = render_comparison(
            comparison, ComparisonStyle(grid=True, grid_axis="none"))

        assert not axes.yaxis._major_tick_kw.get("gridOn")

    def test_the_spines_follow_the_style(self, comparison):
        _figure, axes = render_comparison(
            comparison, ComparisonStyle(hide_top_right_spines=True))

        assert not axes.spines["top"].get_visible()
        assert not axes.spines["right"].get_visible()

    def test_every_plot_kind_draws(self, comparison):
        from spacr.gene_measurement_compare import PLOTS

        for kind, _label in PLOTS:
            figure, _axes = render_comparison(comparison,
                                              ComparisonStyle(kind=kind))
            assert figure is not None, kind

    def test_showing_one_class_is_a_filter_on_the_DRAW(self, comparison):
        _figure, axes = render_comparison(
            comparison, ComparisonStyle(only="the gene"))

        assert len(axes.get_xticklabels()) == 1

    def test_the_counts_can_be_turned_off(self, comparison):
        _figure, axes = render_comparison(comparison,
                                          ComparisonStyle(show_counts=False))

        assert all("n=" not in t.get_text() for t in axes.get_xticklabels())


class TestApplyPageIsSharedNotCopied:
    """The actual payoff: two renderers, one implementation of the furniture."""

    def test_it_works_on_any_figure(self):
        figure, axes = plt.subplots()
        axes.plot([0, 1], [0, 1])

        apply_page(figure, axes, FigureStyle(title="t", x_label="x",
                                             figure_width=5.0,
                                             figure_height=2.0))

        assert axes.get_title() == "t"
        assert axes.get_xlabel() == "x"
        assert tuple(figure.get_size_inches()) == pytest.approx((5.0, 2.0))

    def test_a_bad_scale_is_skipped_rather_than_raising(self):
        """A style loaded from a file someone hand-edited must still draw."""
        figure, axes = plt.subplots()

        apply_page(figure, axes, FigureStyle(x_scale="sideways"))

        assert axes.get_xscale() == "linear"

    def test_the_limits_and_inversions_are_applied(self):
        figure, axes = plt.subplots()

        apply_page(figure, axes, FigureStyle(x_lim=(0.0, 2.0), invert_y=True))

        assert axes.get_xlim() == pytest.approx((0.0, 2.0))
        assert axes.get_ylim()[0] > axes.get_ylim()[1]
