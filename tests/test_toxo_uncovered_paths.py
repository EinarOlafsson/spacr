"""The volcano's optional halves: a given x range, an empty call set, a missing gene.

Every figure this module draws has parts that only appear when the data asks
for them -- point labels, an in-panel legend, a highlighted gene. The tests
here drive the *other* side of each of those decisions and check the figure
that comes back, because "nothing was drawn" is as much a claim about the
picture as "something was".
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from spacr import toxo as T


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _results(p_values, coefficients):
    """A regression table in the shape ``custom_volcano_plot`` parses."""
    genes = [220000 + index for index in range(len(p_values))]
    return pd.DataFrame({
        'feature': [f"grna[{gene}_1]" for gene in genes],
        'coefficient': list(coefficients),
        'p_value': list(p_values),
    })


def _metadata(frame):
    """One localisation per gene_nr, the many-to-one side of the join."""
    genes = frame['feature'].str.extract(r'\[(.*?)\]')[0].str.split('_').str[0]
    return pd.DataFrame({
        'gene_nr': list(dict.fromkeys(genes)),
        'tagm_location': ['cytosol'] * genes.nunique(),
    })


# ---------------------------------------------------------------------------
# The x range
# ---------------------------------------------------------------------------

def test_an_explicit_x_lim_replaces_the_default_half_unit_window():
    """``x_lim`` is honoured; only its absence produces ``[-0.5, 0.5]``.

    The default is narrow enough to crop a real coefficient column, so a
    caller that widens it has to get the window it asked for rather than the
    house default applied over the top.
    """
    data = _results([0.01, 0.5], [1.4, -1.4])
    metadata = _metadata(data)

    T.custom_volcano_plot(data, metadata, figsize=4, threshold=0,
                          x_lim=[-2.0, 2.0])
    assert plt.gcf().axes[0].get_xlim() == (-2.0, 2.0)

    plt.close("all")
    T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)
    assert plt.gcf().axes[0].get_xlim() == (-0.5, 0.5)


# ---------------------------------------------------------------------------
# A volcano with nothing to say
# ---------------------------------------------------------------------------

def test_a_volcano_with_no_called_points_draws_neither_labels_nor_a_legend():
    """When nothing clears p<=0.05 the panel carries no text at all.

    The point labels and the in-panel legend are both written as ``Axes``
    text, and both are conditional on there being a call to name. A legend
    reading "called, positive (0); called, negative (0)" over an unlabelled
    grey cloud is an index of nothing, so neither is drawn.
    """
    data = _results([0.4, 0.6, 0.9], [0.1, -0.2, 0.05])
    metadata = _metadata(data)

    hits = T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)

    assert hits == []
    axis = plt.gcf().axes[0]
    assert [text.get_text() for text in axis.texts] == []
    # The grey cloud is still there -- this is an empty legend, not an empty
    # figure.
    assert sum(len(c.get_offsets()) for c in axis.collections) == 3


def test_a_volcano_with_called_points_labels_them_and_indexes_the_directions():
    """The other side of the same two decisions, for contrast.

    Two called points, one either side of zero, produce their two names plus
    the legend lines that say how many went each way.
    """
    data = _results([0.001, 0.002, 0.9], [0.3, -0.3, 0.01])
    metadata = _metadata(data)

    hits = T.custom_volcano_plot(data, metadata, figsize=4, threshold=0)

    assert hits == ['220000_1', '220001_1']
    written = [text.get_text() for text in plt.gcf().axes[0].texts]
    assert '220000_1' in written
    assert '220001_1' in written
    assert any('called, positive (1)' in text for text in written)
    assert any('called, negative (1)' in text for text in written)


# ---------------------------------------------------------------------------
# A gene that is not in the table
# ---------------------------------------------------------------------------

def test_a_requested_gene_missing_from_the_table_is_skipped_not_faked():
    """Highlighting a gene the phenotype table does not carry draws nothing.

    ``ml.perform_regression`` hands this function the volcano's hit list, and
    a hit can be absent here -- the GT1 phenotype table does not score every
    gene. The absent name must not become a point at rank 0, and must not
    stop the genes that ARE present from being highlighted.
    """
    values = np.linspace(-2.0, 2.0, 5)
    frame = pd.DataFrame({
        "Gene ID": [f"g{index}" for index in range(5)],
        "T.gondii GT1 CRISPR Phenotype - Mean Phenotype": values,
        "T.gondii GT1 CRISPR Phenotype - Standard Error": np.full(5, 0.1),
    })

    T.plot_gene_phenotypes(frame, gene_list=["g2", "g_not_scored"])

    axis = plt.gcf().axes[0]
    highlighted = {collection.get_label(): np.asarray(collection.get_offsets())
                   for collection in axis.collections
                   if collection.get_label().startswith("Highlighted Gene: ")}
    assert set(highlighted) == {"Highlighted Gene: g2"}
    assert highlighted["Highlighted Gene: g2"][0] == pytest.approx(
        [3, values[2]])
    assert [text.get_text() for text in axis.texts] == ["g2"]


def test_a_gene_list_of_only_absent_genes_still_draws_the_ranked_curve():
    """None of them match, so the figure is the grey curve and nothing else."""
    frame = pd.DataFrame({
        "Gene ID": ["g0", "g1", "g2"],
        "T.gondii GT1 CRISPR Phenotype - Mean Phenotype": [1.0, 2.0, 3.0],
        "T.gondii GT1 CRISPR Phenotype - Standard Error": [0.1, 0.1, 0.1],
    })

    T.plot_gene_phenotypes(frame, gene_list=["nope", "also_nope"])

    axis = plt.gcf().axes[0]
    assert len(axis.lines) == 1
    assert list(np.asarray(axis.lines[0].get_ydata())) == [1.0, 2.0, 3.0]
    assert [c for c in axis.collections
            if c.get_label().startswith("Highlighted Gene: ")] == []
    assert list(axis.texts) == []


# ---------------------------------------------------------------------------
# Making room for a legend that can no longer be measured
# ---------------------------------------------------------------------------

def test_a_legend_detached_from_its_figure_leaves_the_layout_untouched():
    """A legend that cannot be measured is not worth an exception.

    ``Artist.remove()`` clears the legend's ``figure`` reference, so asking it
    for a window extent raises ``AttributeError`` rather than returning a box.
    A caller that dropped its legend before adjusting for it gets the figure
    it already had -- the subplot parameters are left exactly where they
    were, and nothing propagates out of a layout helper.
    """
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], label="series")
    legend = axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    legend.remove()

    with pytest.raises(AttributeError):
        legend.get_window_extent()  # the failure the helper has to absorb

    before = figure.subplotpars.right
    assert T._fit_outside_legend(figure, legend) is None
    assert figure.subplotpars.right == before
