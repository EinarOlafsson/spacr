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

from spacr.gene_measurement_sweep import (SweepResult, _one_level,
                                          _order_like_neighbours, _readable,
                                          gene_fractions, is_measurement, measurement_columns,
                                          plot_effect_against_representation,
                                          plot_measurement_families)


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
