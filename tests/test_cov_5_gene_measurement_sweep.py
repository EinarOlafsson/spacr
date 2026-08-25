"""A picture of nothing is the failure this module exists to avoid.

Every view here answers "which genes move which measurements". The dangerous
outcome is not a traceback -- it is an axis with no points on it, which reads
as a measured absence of effect when it means there was nothing to measure.
These drive each of the "there is nothing to draw" exits, plus the name
parsing and the clustering fallback that decide what is on the axis at all.
"""
from __future__ import annotations

import builtins

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_sweep import (SweepResult, _one_level,
                                          _order_like_neighbours, _readable,
                                          gene_fractions, gene_of_guide,
                                          is_measurement, measurement_columns,
                                          plot_circularity,
                                          plot_effect_against_representation,
                                          plot_gene_profile,
                                          plot_guide_concordance,
                                          plot_measurement_families,
                                          plot_measurement_hits, plot_sweep,
                                          sweep)


@pytest.fixture()
def screen():
    """80 wells on two plates, one guide with a real effect on one column."""
    rng = np.random.default_rng(3)
    n = 80
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    driver = np.zeros(n)
    driver[rng.choice(n, 30, replace=False)] = rng.uniform(0.3, 0.6, 30)
    quiet = rng.uniform(0.1, 0.2, n)
    wells = pd.DataFrame({
        "cell_area": driver * 8.0 + rng.normal(0, 0.2, n),
        "nucleus_eccentricity": driver * 6.0 + rng.normal(0, 0.3, n),
        "pathogen_area": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({
        "TGGT1_111_1": driver,
        "TGGT1_111_2": driver * 0.9 + 0.01,
        "TGGT1_222_1": quiet,
    }, index=index)
    plates = [i.split("_")[0] for i in index]
    return wells, fractions, plates


def _table(**columns) -> pd.DataFrame:
    """A sweep table with every column the pictures read, defaults filled."""
    n = len(next(iter(columns.values())))
    base = {"level": ["guide"] * n, "guide": [f"g{i}" for i in range(n)],
            "measurement": ["cell_area"] * n, "effect": [0.5] * n,
            "p": [0.001] * n, "q": [0.001] * n, "circularity": [0.05] * n,
            "n_wells": [40] * n, "effective_wells": [35.0] * n,
            "share": [0.2] * n, "ubiquitous": [False] * n,
            "control": [False] * n}
    base.update(columns)
    return pd.DataFrame(base)


# --------------------------------------------------------------------------- #
#  Naming: what is a measurement, and which gene is a guide
# --------------------------------------------------------------------------- #

def test_a_name_with_no_letters_is_not_a_measurement():
    """A column whose name is only punctuation measures nothing.

    The classifier splits on non-alphanumerics; a name that leaves no tokens
    would otherwise pass the "no identifier token" test vacuously and be
    swept as a measurement.
    """
    assert is_measurement("___") is False
    assert is_measurement("") is False
    assert is_measurement("cell_area") is True


def test_a_frame_of_constants_offers_no_measurement():
    """Columns with fewer than three distinct values are not swept.

    A correlation needs something to vary. Two-valued columns would produce
    p-values off a comparison of two groups of ties, which is not the
    question the sweep is asking and would fill the table with noise.
    """
    frame = pd.DataFrame({"object_label": [1, 2, 3, 4],
                          "cell_area": [5.0, 5.0, 5.0, 5.0],
                          "nucleus_area": [1.0, 2.0, 1.0, 2.0]})

    assert measurement_columns(frame) == []


def test_a_guide_with_no_name_belongs_to_no_gene():
    """A blank guide id resolves to None rather than an empty gene.

    An empty-string gene would collect every unnamed guide in the screen into
    one row and report their summed fraction as a gene's.
    """
    assert gene_of_guide("") is None
    assert gene_of_guide(None) is None
    assert gene_of_guide("   ") is None


def test_a_library_from_another_organism_still_gives_up_its_gene():
    """``<organism>_<gene>_<guide>`` reads the gene from the middle.

    The shipped prefix list is Toxoplasma's. A Plasmodium or human library
    is named the same way and must parse without being listed anywhere --
    otherwise every guide in it resolves to the organism and the whole
    library pools into one "gene".
    """
    assert gene_of_guide("TGGT1_233460_1") == "233460"
    assert gene_of_guide("PF3D7_0100100_1") == "0100100"
    assert gene_of_guide("HGNC_RAB11_A_1") == "RAB11_A"


def test_a_gene_resolver_that_raises_loses_only_that_guide():
    """One guide whose name cannot be parsed does not stop the rest.

    The resolver can be a caller's own function over a library CSV, and a
    single unexpected name in a library of thousands must not take the whole
    gene-level sweep down.
    """
    fractions = pd.DataFrame({"TGGT1_111_1": [0.1, 0.2],
                              "??": [0.3, 0.4],
                              "TGGT1_111_2": [0.2, 0.1]})

    def strict(name):
        if not name.startswith("TGGT1"):
            raise ValueError(f"unparseable guide {name!r}")
        return name.split("_")[1]

    genes = gene_fractions(fractions, gene_of=strict)

    assert list(genes.columns) == ["111"]
    assert genes["111"].tolist() == [0.30000000000000004, 0.30000000000000004]


def test_a_library_no_gene_can_be_read_from_gives_an_empty_frame():
    """When no guide resolves to a gene the result is empty, not wrong.

    Returning the guide columns under their own names would silently make a
    gene-level sweep a guide-level one, and the reader would believe they
    were looking at genes.
    """
    fractions = pd.DataFrame({"a": [0.1, 0.2]}, index=["w1", "w2"])

    genes = gene_fractions(fractions, gene_of=lambda name: None)

    assert list(genes.columns) == []
    assert list(genes.index) == ["w1", "w2"]


# --------------------------------------------------------------------------- #
#  Too little data to sweep at all
# --------------------------------------------------------------------------- #

def test_two_wells_are_not_a_screen():
    """Fewer than three shared wells returns an empty table, not a p-value.

    A correlation over two points is exactly 1 or -1 whatever the data, so a
    sweep that ran anyway would report every measurement as a perfect hit.
    """
    wells = pd.DataFrame({"cell_area": [1.0, 2.0]}, index=["w1", "w2"])
    fractions = pd.DataFrame({"TGGT1_111_1": [0.1, 0.9]}, index=["w1", "w2"])

    result = sweep(wells, fractions)

    assert result.n_wells == 2
    assert result.n_blocks == 0
    assert list(result.table.columns) == ["guide", "measurement", "effect",
                                          "p", "q", "circularity", "n_wells"]
    assert len(result.table) == 0


# --------------------------------------------------------------------------- #
#  Chrome and layout helpers
# --------------------------------------------------------------------------- #

def test_a_missing_panel_is_skipped_when_the_theme_is_applied():
    """A None among the axes is ignored rather than styled.

    Several figures here build an optional second panel, and the helper is
    handed whatever they produced. An AttributeError at this point would lose
    a figure that had already been drawn.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    try:
        ink = _readable(figure, None, axes, None)
        assert isinstance(ink, str) and ink
        assert axes.title.get_color() == ink
    finally:
        plt.close(figure)


def test_the_grid_is_still_ordered_without_scipy(monkeypatch):
    """Without scipy the rows are ordered by their mean, not by arrival.

    A heatmap whose rows are in the order they happened to arrive hides every
    block structure it exists to show, so falling back to arrival order would
    be worse than not drawing it.
    """
    grid = pd.DataFrame(
        [[0.9, 0.8, 0.1], [-0.9, -0.8, 0.0], [0.1, 0.0, 0.9],
         [-0.2, -0.1, 0.8]],
        index=["up1", "down1", "other1", "other2"],
        columns=["m1", "m2", "m3"])

    real_import = builtins.__import__

    def _no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("scipy is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scipy)
    try:
        ordered = _order_like_neighbours(grid)
    finally:
        monkeypatch.undo()

    assert set(ordered.index) == set(grid.index)
    assert set(ordered.columns) == set(grid.columns)
    means = ordered.mean(axis=1).to_numpy()
    assert (np.diff(means) >= 0).all()


def test_a_table_with_no_level_column_draws_every_row():
    """Without a level column there is no level to single out.

    A guide-only sweep records no level, and filtering on a column that is
    not there would drop every row and leave an empty picture.
    """
    assert _one_level(pd.DataFrame({"q": [0.01]}), "gene") == ""
    assert _one_level(pd.DataFrame({"level": ["gene", "guide"], "q": [0, 0]}),
                      "") == "gene"


# --------------------------------------------------------------------------- #
#  Every picture refuses rather than drawing an empty axis
# --------------------------------------------------------------------------- #

def test_a_heatmap_of_no_rows_is_not_drawn(screen):
    """Asking for the top zero survivors draws nothing at all.

    The grid is built by selecting the top guides and measurements; an empty
    selection must return None rather than a labelled but empty heatmap.
    """
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates)

    assert len(result.survivors(alpha=0.05)) > 0
    assert plot_sweep(result, top=0) is None


def test_a_screen_with_no_statistical_weight_recorded_is_not_plotted():
    """Genes whose effective well count is unknown are not put on the axis.

    The x axis IS the effective well count. Plotting rows without one would
    place them at an arbitrary position and invite exactly the comparison
    between representation and effect the figure exists to make.
    """
    result = SweepResult(
        table=_table(effective_wells=[np.nan, np.nan], guide=["g1", "g2"]),
        effects=pd.DataFrame([[0.5], [0.5]], index=["g1", "g2"],
                             columns=["cell_area"]),
        n_wells=40, n_blocks=2)

    assert plot_effect_against_representation(result) is None


def test_no_family_counts_means_no_family_picture():
    """With nothing to count per family the figure is not drawn.

    The families view answers "what KIND of thing does this gene move"; with
    no survivors that sentence has no subject, and an empty stacked bar reads
    as "this gene moves nothing", which is a different claim.
    """
    result = SweepResult(table=_table(q=[0.001, 0.001]),
                         effects=pd.DataFrame([[0.5], [0.5]],
                                              index=["g0", "g1"],
                                              columns=["cell_area"]),
                         n_wells=40, n_blocks=2)

    assert plot_measurement_families(result, top=0) is None


def test_concordance_needs_guides_that_belong_to_a_gene():
    """Guide names no gene can be read from give no concordance picture.

    Concordance is the only internal control this design has: it compares a
    gene's own guides. Without a gene there is nothing to compare, and a
    figure drawn anyway would be comparing unrelated guides.
    """
    result = SweepResult(table=_table(guide=["", "   "]),
                         effects=pd.DataFrame([[0.5], [0.5]], index=["a", "b"],
                                              columns=["cell_area"]),
                         n_wells=40, n_blocks=2)

    assert plot_guide_concordance(result) is None


def test_concordance_needs_something_past_the_correction():
    """With nothing significant there is no agreement to report.

    Agreement among guides that all failed the correction is agreement about
    noise, and reporting it as concordance would be the strongest possible
    endorsement of a null result.
    """
    result = SweepResult(
        table=_table(guide=["TGGT1_111_1", "TGGT1_111_2"], q=[0.4, 0.6]),
        effects=pd.DataFrame([[0.5], [0.5]],
                             index=["TGGT1_111_1", "TGGT1_111_2"],
                             columns=["cell_area"]),
        n_wells=40, n_blocks=2)

    assert plot_guide_concordance(result) is None


def test_one_guide_moving_a_measurement_is_not_agreement():
    """A gene/measurement pair with a single signed effect is skipped.

    Agreement is a statement about two or more guides. Counting a lone guide
    as unanimous would put every single-guide hit at perfect concordance,
    which is where a reader looks for the most trustworthy genes.
    """
    result = SweepResult(
        table=_table(guide=["TGGT1_111_1", "TGGT1_111_2"],
                     measurement=["cell_area", "nucleus_area"],
                     effect=[0.5, 0.4]),
        effects=pd.DataFrame([[0.5, np.nan], [np.nan, 0.4]],
                             index=["TGGT1_111_1", "TGGT1_111_2"],
                             columns=["cell_area", "nucleus_area"]),
        n_wells=40, n_blocks=2)

    assert plot_guide_concordance(result) is None


def test_a_gene_profile_of_nothing_is_not_drawn(screen):
    """Asking a gene profile for zero measurements returns None.

    The profile is a bar per measurement; with none selected the figure would
    be a labelled empty axis under the gene's name, which reads as "this gene
    moves nothing".
    """
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates)

    assert plot_gene_profile(result, "TGGT1_111_1", top=0) is None


def test_a_measurement_hit_count_of_nothing_is_not_drawn(screen):
    """With no measurements selected the bubble plot is not drawn.

    The picture ranks measurements by how many genes move them; an empty
    ranking is not a ranking, and the axis would carry a legend for a scale
    nothing is plotted on.
    """
    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates)

    assert plot_measurement_hits(result, top=0) is None


def test_circularity_needs_a_finite_pair_to_plot(screen):
    """No survivor means no circularity picture, even when the score joined.

    Circularity says whether a measurement is already tracked by the score.
    An empty panel here reads as "nothing is circular", which is the exact
    misreading this column exists to prevent.
    """
    wells, fractions, plates = screen
    rng = np.random.default_rng(5)
    scores = rng.uniform(0, 1, len(wells.index))
    result = sweep(wells, fractions, blocks=plates, scores=scores)

    assert result.circularity_known is True
    assert plot_circularity(result, alpha=0.0) is None
