"""The other ways of looking at the sweep (instruction 175).

Asked for on 2026-08-19: "i want you to implement afew ways of visualizing
this data, come up with 10 ways, ill tell you what i like." These are the
three that answer a question the heatmap cannot:

  * effect against representation -- "220950 is waaay over represented and is
    therefore beeing attributed with a tone of mesauremnt effects";
  * measurement families -- WHAT KIND of thing a gene moves, which is a
    sentence about biology where "41 significant measurements" is not;
  * guide concordance -- do a gene's own guides agree, which is the only
    internal control this design has.

Every one of them must return None rather than an empty axis when there is
nothing to draw. A picture of nothing is the failure mode these replace: it
reads as "no effect" when it means "no data".
"""
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr.gene_measurement_sweep import (MEASUREMENT_FAMILIES,
                                          measurement_family,
                                          plot_effect_against_representation,
                                          plot_guide_concordance,
                                          plot_measurement_families, sweep)


@pytest.fixture()
def screen():
    """120 wells on 3 plates. TGGT1_111 is in every well; TGGT1_222 is rare.

    Built so representation and effect are DIFFERENT things: the rare gene
    has the larger true effect, so a plot that only ranked by q would put the
    ubiquitous one first and a reader would never know why.
    """
    rng = np.random.default_rng(7)
    n = 120
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    everywhere = rng.uniform(0.15, 0.25, n)          # in all 120 wells
    rare = np.zeros(n)
    rare[rng.choice(n, 18, replace=False)] = rng.uniform(0.3, 0.6, 18)
    control = np.zeros(n)
    control[rng.choice(n, 24, replace=False)] = 0.2

    wells = pd.DataFrame({
        "pathogen_area": rare * 6.0 + rng.normal(0, 0.3, n),
        "pathogen_channel_1_mean_intensity": rare * 4.0 + rng.normal(0, .4, n),
        "cell_area": everywhere * 5.0 + rng.normal(0, 0.3, n),
        "nucleus_eccentricity": rng.normal(0, 1, n),
        "cell_channel_0_median_intensity": everywhere * 3.0
                                           + rng.normal(0, 0.4, n),
    }, index=index)
    fractions = pd.DataFrame({
        "TGGT1_111_1": everywhere,
        "TGGT1_222_1": rare,
        "TGGT1_333_1": control,
    }, index=index)
    plates = [i.split("_")[0] for i in index]
    return wells, fractions, plates


# ----------------------------------------------------------- the families


def test_the_first_family_that_matches_wins():
    """`pathogen_area` is a pathogen measurement, not a shape one."""
    assert measurement_family("pathogen_area") == "pathogen"
    assert measurement_family("cell_area") == "cell"
    assert measurement_family("solidity") == "shape"
    assert measurement_family("nucleus_eccentricity") == "nucleus"


def test_a_column_in_no_family_is_other_not_an_error():
    """A screen may measure something none of these names -- that is an
    answer, not a failure."""
    assert measurement_family("wibble") == "other"


def test_the_families_are_few_enough_to_read():
    assert len(MEASUREMENT_FAMILIES) <= 8


def test_a_family_is_matched_on_tokens_not_substrings():
    """The bug this module already made once: "path" is inside "pathogen"."""
    assert measurement_family("pathogen_area") == "pathogen"
    assert measurement_family("path_length") != "pathogen"


# ------------------------------------------- effect against representation


def test_representation_is_drawn_against_the_hits(screen, tmp_path):
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")

    figure = plot_effect_against_representation(
        result, path=str(tmp_path / "rep.png"))

    assert figure is not None
    axes = figure.axes[0]
    assert "effective wells" in axes.get_xlabel()
    assert "past BH" in axes.get_ylabel()


def test_the_ubiquitous_gene_is_further_right_than_the_rare_one(screen):
    """The whole point: the x axis has to separate them, IN THAT ORDER.

    And `share` is not the axis that does it. `share` is the median fraction
    a gene takes of the wells it is IN, so a gene in 18 of 120 wells scores
    0.44 on it and a gene in all 120 scores 0.20 -- the reverse of the
    ordering a reader asking about over-representation means. The
    participation ratio is what the p-value was actually computed on.
    """
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")

    weight = result.table.groupby("guide")["effective_wells"].first()
    assert weight["111"] > weight["222"], weight

    share = result.table.groupby("guide")["share"].first()
    assert share["111"] < share["222"], (
        "the fixture no longer separates abundance from breadth, so this "
        "test cannot show which axis is the right one")


def test_a_control_is_drawn_and_not_dropped(screen, tmp_path):
    """"these are removed for regression are they removed here? i dont think
    they should" -- a control that moves twenty measurements is the
    calibration for everything else on the plot."""
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene",
                   controls=["333"])

    figure = plot_effect_against_representation(result)

    assert figure is not None
    labels = " ".join(t.get_text() for t in figure.axes[0].get_legend().texts)
    assert "control" in labels, labels


def test_no_trend_line_is_drawn_through_two_points(tmp_path):
    """Two points make a line through themselves and say nothing; drawing it
    would put a confident diagonal on a plot with no evidence for one."""
    rng = np.random.default_rng(0)
    n = 40
    index = [f"plate1_r{i}_c1" for i in range(n)]
    wells = pd.DataFrame({"cell_area": rng.normal(0, 1, n)}, index=index)
    fractions = pd.DataFrame({"TGGT1_1_1": rng.random(n),
                              "TGGT1_2_1": rng.random(n)}, index=index)
    result = sweep(wells, fractions, blocks=["plate1"] * n, level="gene")

    figure = plot_effect_against_representation(result)

    if figure is not None:
        assert not figure.axes[0].get_lines(), "a trend was fitted to 2 genes"


def test_an_empty_sweep_draws_nothing(screen):
    """None, not an empty axis: a picture of nothing reads as 'no effect'."""
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")
    assert plot_effect_against_representation(result, alpha=0.0) is None


# ---------------------------------------------------------- the families bar


def test_the_family_bars_say_what_kind_of_thing_moved(screen, tmp_path):
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")

    figure = plot_measurement_families(
        result, path=str(tmp_path / "fam.png"))

    assert figure is not None
    legend = {t.get_text() for t in figure.axes[0].get_legend().texts}
    assert legend <= {f for f, _ in MEASUREMENT_FAMILIES} | {"other"}
    assert legend, "no family reached the legend"


def test_the_family_bars_draw_nothing_when_nothing_survived(screen):
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")
    assert plot_measurement_families(result, alpha=0.0) is None


def test_the_family_bars_are_capped(screen):
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")

    figure = plot_measurement_families(result, top=1)

    if figure is not None:
        assert len(figure.axes[0].get_yticklabels()) <= 1


# ------------------------------------------------------- guide concordance


def test_concordance_needs_guide_rows(screen):
    """A gene-level table has nothing to compare, and says so by drawing
    nothing rather than by drawing an empty axis."""
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="gene")

    assert plot_guide_concordance(result) is None


def test_a_gene_with_one_guide_cannot_agree_with_itself(screen):
    """Every gene in this fixture has exactly one guide, so there is nothing
    to draw even at guide level."""
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates, level="guide")

    assert plot_guide_concordance(result) is None


def test_two_guides_agreeing_are_drawn_as_agreeing(tmp_path):
    rng = np.random.default_rng(3)
    n = 90
    index = [f"plate{1 + i // 30}_r{i}_c1" for i in range(n)]
    signal = rng.random(n)
    wells = pd.DataFrame({
        "pathogen_area": signal * 5.0 + rng.normal(0, 0.2, n),
        "cell_area": rng.normal(0, 1, n),
    }, index=index)
    # TWO guides against one gene, both carrying the same signal.
    fractions = pd.DataFrame({
        "TGGT1_444_1": signal * 0.5,
        "TGGT1_444_2": signal * 0.5,
        "TGGT1_555_1": rng.random(n),
    }, index=index)
    plates = [i.split("_")[0] for i in index]
    result = sweep(wells, fractions, blocks=plates, level="guide")

    figure = plot_guide_concordance(result, path=str(tmp_path / "c.png"))

    assert figure is not None
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert any("444" in label for label in labels), labels
    widths = [p.get_width() for p in figure.axes[0].patches]
    assert widths and max(widths) == pytest.approx(1.0), widths


def test_the_agreement_axis_cannot_exceed_one(tmp_path):
    """A share is a share: an axis running past 1.0 would be unreadable."""
    rng = np.random.default_rng(4)
    n = 90
    index = [f"plate{1 + i // 30}_r{i}_c1" for i in range(n)]
    signal = rng.random(n)
    wells = pd.DataFrame({"pathogen_area": signal * 5 + rng.normal(0, .2, n)},
                         index=index)
    fractions = pd.DataFrame({"TGGT1_444_1": signal * 0.5,
                              "TGGT1_444_2": signal * 0.5}, index=index)
    result = sweep(wells, fractions, blocks=[i.split("_")[0] for i in index],
                   level="guide")

    figure = plot_guide_concordance(result)

    if figure is not None:
        assert figure.axes[0].get_xlim()[1] <= 1.05


# --------------------------------------------------- reachable from the panel


def test_the_panel_offers_every_picture(qtbot):
    """A picture nobody can choose is a picture that does not exist -- the
    same defect as `set_target_cell_width` having no caller."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    panel = SweepPanel()
    qtbot.addWidget(panel)

    offered = {panel.picture.itemData(i) for i in range(panel.picture.count())}
    assert offered == {kind for kind, _label in SweepPanel.PICTURES}
    assert panel.picture.currentData() == "heatmap"


def test_each_picture_draws_from_the_panel(qtbot, screen):
    """Every kind reaches its function, and none of them raises."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions, blocks=plates, level="guide")

    for kind, _label in SweepPanel.PICTURES:
        figure = panel.figure(kind=kind)
        assert figure is None or hasattr(figure, "axes"), kind


def test_the_saved_picture_is_the_chosen_one(qtbot, screen, tmp_path):
    """Save writes the figure beside the table; it must be the one on screen."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions, blocks=plates, level="gene")

    index = next(i for i in range(panel.picture.count())
                 if panel.picture.itemData(i) == "families")
    panel.picture.setCurrentIndex(index)
    out = tmp_path / "fam.png"
    figure = panel.figure(path=str(out))

    assert figure is not None
    assert out.exists()
    legend = figure.axes[0].get_legend()
    assert legend is not None, "the heatmap was drawn instead of the bars"


def test_concordance_ignores_the_panel_s_gene_default(qtbot, screen):
    """This picture IS the guide comparison, so the panel's gene level must
    not be passed through -- it would leave nothing to compare."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    rng = np.random.default_rng(11)
    n = 90
    index = [f"plate{1 + i // 30}_r{i}_c1" for i in range(n)]
    signal = rng.random(n)
    wells = pd.DataFrame({"pathogen_area": signal * 5 + rng.normal(0, .2, n)},
                         index=index)
    fractions = pd.DataFrame({"TGGT1_444_1": signal * 0.5,
                              "TGGT1_444_2": signal * 0.5}, index=index)

    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions,
                          blocks=[i.split("_")[0] for i in index],
                          level="guide")
    assert str(panel.level.currentData()) == "gene"      # the panel default

    assert panel.figure(kind="concordance") is not None
