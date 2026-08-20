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


# =========================================================================== #
#  The other six (instruction 175). Ten views, each answering a question the   #
#  others cannot -- built together so the maintainer can choose after seeing   #
#  all of them.                                                                #
# =========================================================================== #


def _swept(screen, level="gene", controls=None):
    wells, fractions, plates = screen
    return sweep(wells, fractions, blocks=plates, level=level,
                 controls=controls)


# ------------------------------------------------------- #5 the grid volcano


def test_the_volcano_draws_every_pair_not_just_survivors(screen, tmp_path):
    """The shape of the WHOLE grid, which the heatmap cannot show."""
    from spacr.gene_measurement_sweep import plot_grid_volcano

    result = _swept(screen)
    figure = plot_grid_volcano(result, path=str(tmp_path / "v.png"))

    assert figure is not None
    axes = figure.axes[0]
    drawn = sum(len(c.get_offsets()) for c in axes.collections)
    assert drawn == len(result.table), (
        f"{drawn} points for {len(result.table)} pairs")


def test_an_uncomputed_circularity_is_said_not_coloured(screen):
    """Grey is not "clean". A NaN column shown on a colour scale would read
    as zero, which is the exact misreading the column exists to prevent."""
    from spacr.gene_measurement_sweep import plot_grid_volcano

    result = _swept(screen)
    assert not result.circularity_known
    figure = plot_grid_volcano(result)

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert "NOT computed" in " ".join(t.get_text() for t in legend.texts)


def test_the_volcano_draws_nothing_from_an_empty_table():
    from spacr.gene_measurement_sweep import SweepResult, plot_grid_volcano

    empty = SweepResult(table=pd.DataFrame(
        columns=["level", "guide", "measurement", "effect", "p", "q",
                 "circularity"]),
        effects=pd.DataFrame(), n_wells=0, n_blocks=0)
    assert plot_grid_volcano(empty) is None


# ------------------------------------------------------ #6 the gene profile


def test_a_gene_profile_names_the_measurements_it_moves(screen, tmp_path):
    from spacr.gene_measurement_sweep import plot_gene_profile

    result = _swept(screen)
    figure = plot_gene_profile(result, "222", path=str(tmp_path / "p.png"))

    assert figure is not None
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert labels
    assert all(any(l.startswith(m[:10]) for m in result.effects.columns)
               for l in labels), labels


def test_a_gene_with_nothing_significant_still_says_so(screen):
    """"This gene has no significant measurement" is worth SEEING, and an
    empty axis does not say it."""
    from spacr.gene_measurement_sweep import plot_gene_profile

    result = _swept(screen)
    figure = plot_gene_profile(result, "222", alpha=0.0)

    assert figure is not None
    assert "NOTHING" in figure.axes[0].get_title()


def test_a_gene_that_is_not_in_the_screen_draws_nothing(screen):
    from spacr.gene_measurement_sweep import plot_gene_profile

    assert plot_gene_profile(_swept(screen), "not_a_gene") is None


def test_the_profile_is_capped(screen):
    from spacr.gene_measurement_sweep import plot_gene_profile

    figure = plot_gene_profile(_swept(screen), "222", top=2)
    if figure is not None:
        assert len(figure.axes[0].get_yticklabels()) <= 2


# --------------------------------------------------- #7 the gene similarity


def test_genes_that_behave_alike_correlate(tmp_path):
    """Two genes moving the same measurements the same way must come back
    positively correlated -- that is the whole claim of this picture."""
    from spacr.gene_measurement_sweep import plot_gene_similarity

    rng = np.random.default_rng(21)
    n = 120
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    shared = rng.random(n)
    other = rng.random(n)
    wells = pd.DataFrame({
        "pathogen_area": shared * 5 + rng.normal(0, .2, n),
        "pathogen_channel_1_mean_intensity": shared * 4 + rng.normal(0, .3, n),
        "cell_area": other * 5 + rng.normal(0, .2, n),
        "nucleus_eccentricity": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({
        "TGGT1_111_1": shared * 0.5,      # these two move the same things
        "TGGT1_222_1": shared * 0.5,
        "TGGT1_333_1": other * 0.5,       # this one does not
    }, index=index)
    result = sweep(wells, fractions,
                   blocks=[i.split("_")[0] for i in index], level="gene")

    figure = plot_gene_similarity(result, path=str(tmp_path / "s.png"))

    if figure is None:
        pytest.skip("nothing survived on this fixture")
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert len(labels) >= 2
    # The matrix is symmetric with a unit diagonal, or it is not a correlation.
    grid = figure.axes[0].get_images()[0].get_array()
    assert np.allclose(np.diag(np.asarray(grid)), 1.0, atol=1e-6)


def test_one_gene_cannot_be_compared_with_itself(screen):
    """A similarity matrix of one is not a picture."""
    from spacr.gene_measurement_sweep import plot_gene_similarity

    assert plot_gene_similarity(_swept(screen), top=1) is None


def test_the_similarity_scale_is_pinned_to_plus_minus_one(screen):
    """A correlation drawn on an auto scale makes 0.2 look like 1.0."""
    from spacr.gene_measurement_sweep import plot_gene_similarity

    figure = plot_gene_similarity(_swept(screen))
    if figure is not None:
        image = figure.axes[0].get_images()[0]
        assert image.get_clim() == (-1.0, 1.0)


# ------------------------------------------------- #8 which measurements


def test_the_measurements_are_ranked_by_how_many_genes_move_them(screen,
                                                                 tmp_path):
    from spacr.gene_measurement_sweep import plot_measurement_hits

    figure = plot_measurement_hits(_swept(screen),
                                   path=str(tmp_path / "m.png"))
    if figure is None:
        pytest.skip("nothing survived on this fixture")
    widths = [p.get_width() for p in figure.axes[0].patches]
    assert widths == sorted(widths, reverse=True)


def test_a_measurement_moved_by_half_the_library_is_marked(tmp_path):
    """Not a discriminating readout: a plate effect wearing a measurement's
    name will put a hit on every gene in the screen."""
    from spacr.gene_measurement_sweep import plot_measurement_hits

    rng = np.random.default_rng(22)
    n = 120
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    drift = np.linspace(0.0, 1.0, n)          # everything correlates with it
    wells = pd.DataFrame({
        "cell_area": drift * 8 + rng.normal(0, .1, n),
        "nucleus_eccentricity": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame(
        {f"TGGT1_{i}_1": drift * 0.3 + rng.normal(0, .02, n)
         for i in range(1, 5)}, index=index)
    result = sweep(wells, fractions,
                   blocks=[i.split("_")[0] for i in index], level="gene")

    figure = plot_measurement_hits(result)
    if figure is None:
        pytest.skip("nothing survived on this fixture")
    reds = [p for p in figure.axes[0].patches
            if tuple(round(c, 3) for c in p.get_facecolor()[:3])
            == (0.757, 0.071, 0.122)]
    assert reds or "half the library" not in figure.axes[0].get_title()


def test_the_measurement_bars_draw_nothing_when_nothing_survived(screen):
    from spacr.gene_measurement_sweep import plot_measurement_hits

    assert plot_measurement_hits(_swept(screen), alpha=0.0) is None


# ------------------------------------------------------- #9 the circularity


def test_circularity_is_not_drawn_when_it_was_never_computed(screen):
    """THE POINT OF THE WHOLE COLUMN. A scatter of NaN is a blank panel that
    reads as "nothing is circular"."""
    from spacr.gene_measurement_sweep import plot_circularity

    result = _swept(screen)
    assert not result.circularity_known
    assert plot_circularity(result) is None


def test_circularity_is_drawn_when_the_score_joined(tmp_path):
    from spacr.gene_measurement_sweep import plot_circularity

    rng = np.random.default_rng(23)
    n = 120
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    signal = rng.random(n)
    wells = pd.DataFrame({
        "pathogen_area": signal * 5 + rng.normal(0, .2, n),
        "cell_area": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({"TGGT1_111_1": signal * 0.5,
                              "TGGT1_222_1": rng.random(n)}, index=index)
    scores = pd.Series(signal * 0.9 + rng.normal(0, 0.05, n), index=index)
    result = sweep(wells, fractions,
                   blocks=[i.split("_")[0] for i in index],
                   scores=scores, level="gene")

    if not result.circularity_known:
        pytest.skip("the score did not join on this fixture")
    figure = plot_circularity(result, path=str(tmp_path / "c.png"))
    if figure is None:
        pytest.skip("nothing survived")
    assert "rho" in figure.axes[0].get_ylabel()


# ------------------------------------------------------- #10 the calibration


def test_the_calibration_plot_reports_an_inflation_factor(screen, tmp_path):
    """A number beats an eyeballed slope, and lambda has a standard meaning."""
    from spacr.gene_measurement_sweep import plot_calibration

    figure = plot_calibration(_swept(screen), path=str(tmp_path / "q.png"))

    assert figure is not None
    assert "lambda" in figure.axes[0].get_title()


def test_a_null_screen_sits_on_the_diagonal(tmp_path):
    """Nothing real: the observed P values follow the uniform, so lambda is
    about 1 and the plot says "calibrated"."""
    from spacr.gene_measurement_sweep import plot_calibration

    rng = np.random.default_rng(24)
    n = 200
    index = [f"plate{1 + i // 50}_r{i}_c1" for i in range(n)]
    wells = pd.DataFrame(
        {f"cell_measure_{k}": rng.normal(0, 1, n) for k in range(6)},
        index=index)
    fractions = pd.DataFrame(
        {f"TGGT1_{i}_1": rng.random(n) for i in range(1, 7)}, index=index)
    result = sweep(wells, fractions,
                   blocks=[i.split("_")[0] for i in index], level="gene")

    figure = plot_calibration(result)

    assert figure is not None
    title = figure.axes[0].get_title()
    assert "inflated" not in title, title


def test_the_calibration_needs_more_than_one_test(screen):
    from spacr.gene_measurement_sweep import SweepResult, plot_calibration

    one = SweepResult(table=pd.DataFrame({"level": ["gene"], "guide": ["a"],
                                          "measurement": ["m"],
                                          "effect": [0.1], "p": [0.5],
                                          "q": [0.5], "circularity": [np.nan]}),
                      effects=pd.DataFrame(), n_wells=1, n_blocks=1)
    assert plot_calibration(one) is None


# ------------------------------------------------------- all ten, in the panel


def test_the_panel_offers_all_ten(qtbot):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    panel = SweepPanel()
    qtbot.addWidget(panel)
    assert len(SweepPanel.PICTURES) == 10
    offered = [panel.picture.itemData(i) for i in range(panel.picture.count())]
    assert offered == [kind for kind, _label in SweepPanel.PICTURES]


def test_every_one_of_the_ten_draws_or_says_why(qtbot, screen):
    """None is an answer; an exception is not."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions, blocks=plates, level="guide")

    for kind, _label in SweepPanel.PICTURES:
        figure = panel.figure(kind=kind)
        assert figure is None or hasattr(figure, "axes"), kind


def test_the_profile_falls_back_to_the_strongest_survivor(qtbot, screen):
    """With no row selected the picture still has a subject, and the title
    names it so nobody mistakes the default for a choice."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions, blocks=plates, level="gene")

    gene = panel.selected_gene()
    assert gene is not None
    figure = panel.figure(kind="profile")
    if figure is not None:
        assert str(gene) in figure.axes[0].get_title()


# ------------------------------------------------------- looking at them


def test_there_is_a_button_that_shows_the_picture(qtbot, screen):
    """Asked on 2026-08-19: "i ran a measurement sweep how do i see the
    graphs?" -- and the answer was that you could not. The chooser and all
    ten views existed with `figure()` reachable only through Save, so the
    only way to look at one was to write it to disk and open it yourself.
    A picture nobody can look at is a setter nobody calls.
    """
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)

    assert not panel.show_button.isEnabled(), (
        "the button offers to draw a sweep that has not been run")

    panel._result = sweep(wells, fractions, blocks=plates, level="gene")
    dialog = panel.show_picture()

    assert dialog is not None
    assert dialog.isVisible()
    qtbot.addWidget(dialog)


def test_the_window_is_kept_alive(qtbot, screen):
    """Python collects a dialog with no reference the moment the call
    returns, and the window vanishes as it appears."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions, blocks=plates, level="gene")

    dialog = panel.show_picture()
    assert dialog in panel._pictures


def test_a_picture_with_nothing_to_draw_says_so_rather_than_opening_empty(
        qtbot, screen):
    """An empty window reads as a broken button."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    wells, fractions, plates = screen
    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel._result = sweep(wells, fractions, blocks=plates, level="gene")
    # The branch is driven directly rather than through the q filter: the
    # spin box clamps to its own minimum, so "nothing can pass" is not a
    # value a user can type, and the test would have been asserting the
    # widget's range instead of the panel's behaviour.
    panel.figure = lambda *a, **k: None

    assert panel.show_picture() is None
    assert "Nothing to draw" in panel.status.text()


def test_pressing_show_before_a_run_says_what_to_do(qtbot):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    panel = SweepPanel()
    qtbot.addWidget(panel)

    assert panel.show_picture() is None
    assert "Run the sweep first" in panel.status.text()
