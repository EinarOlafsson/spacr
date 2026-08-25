"""Toxo figure helpers on the inputs that carry no signal.

The enrichment and volcano panels are read as evidence, so the degenerate cases
have to draw something honest rather than divide by zero: a column with no
spread maps to one marker size, a hit list that matches no gene enriches
nothing, and a legend that does not exist takes no room from the data.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import toxo as T

GO_COLUMNS = ['Computed GO Processes', 'Curated GO Components',
              'Curated GO Functions', 'Curated GO Processes']


def test_an_empty_column_has_no_marker_sizes_to_scale():
    """Scaling no values returns no sizes rather than a nanmax over nothing.

    An enrichment table with no called terms reaches the scatter with an empty
    column; ``np.nanmax`` of it warns and returns NaN, which would size every
    later marker as NaN.
    """
    out = T._scaled_sizes(np.array([]))
    assert out.shape == (0,)


def test_a_column_with_no_spread_maps_to_the_smallest_marker():
    """A constant column gets the low size, not a division by a zero range.

    Every term equally enriched is a real result; drawing it as NaN-sized
    markers would show an empty panel and read as "nothing was tested".
    """
    out = T._scaled_sizes(np.array([2.0, 2.0, 2.0]), low=50.0, high=200.0)
    assert out.tolist() == [50.0, 50.0, 50.0]

    infinite = T._scaled_sizes(np.array([1.0, np.inf]), low=50.0, high=200.0)
    assert infinite.tolist() == [50.0, 50.0]


def test_a_volcano_reads_its_regression_table_from_disk(tmp_path):
    """A path argument is read as a table, not just accepted as a DataFrame.

    The volcano is called from the GUI with the path of the regression CSV the
    run just wrote; only the in-memory form is exercised by the fast tests.
    """
    genes = [str(220000 + i) for i in range(8)]
    data = pd.DataFrame({
        'feature': [f'{g}_1' for g in genes],
        'coefficient': [0.6, -0.6, 0.01, 0.02, 0.5, -0.5, 0.0, 0.03],
        'p_value': [1e-6, 1e-6, 0.9, 0.8, 1e-4, 1e-4, 0.7, 0.6],
    })
    data_path = tmp_path / 'regression.csv'
    data.to_csv(data_path, index=False)
    metadata = pd.DataFrame({'gene_nr': genes,
                             'tagm_location': ['cytosol'] * 8})

    hits = T.custom_volcano_plot(str(data_path), metadata, draw=False)
    assert hits == [f'{g}_1' for g in
                    ['220000', '220001', '220004', '220005']]


def test_hits_that_match_no_gene_enrich_no_go_term(tmp_path):
    """With no hit in the metadata every term scores zero and none is plotted.

    A gene-id spelling mismatch between the screen and the annotation table is
    the usual cause; inventing enrichment scores from an empty hit set would
    put terms on the panel that nothing supports.
    """
    genes = [f'TGGT1_{220000 + i}' for i in range(6)]
    metadata = pd.DataFrame({'Gene ID': genes})
    for column in GO_COLUMNS:
        metadata[column] = ['metabolism;transport'] * 6
    meta_path = tmp_path / 'go_meta.csv'
    metadata.to_csv(meta_path, index=False)

    # n_gene values that appear in no Gene ID, plus a missing one that must be
    # dropped before the join.
    significant = pd.DataFrame({'n_gene': ['999999', None]})

    plt.close('all')
    T.go_term_enrichment_by_column(significant, str(meta_path))

    # One panel per default GO column, plus the combined panel, and every one
    # of them empty.
    figures = [plt.figure(i) for i in plt.get_fignums()]
    assert len(figures) == len(GO_COLUMNS) + 1
    titles = [fig.axes[0].get_title() for fig in figures[:-1]]
    assert titles == [f'GO Term Enrichment Analysis for {c}'
                      for c in GO_COLUMNS]
    for figure in figures:
        for collection in figure.axes[0].collections:
            assert len(collection.get_offsets()) == 0
    plt.close('all')


class RecordingFigure:
    """A figure stub that records the axes adjustment it was asked for."""

    def __init__(self, width):
        self.calls = []
        self.dpi = 100.0
        self._width = width
        self.canvas = self

    def draw(self):
        return None

    def get_figwidth(self):
        return self._width

    def subplots_adjust(self, **kwargs):
        self.calls.append(kwargs)


class WideLegend:
    def get_window_extent(self):
        return matplotlib.transforms.Bbox.from_bounds(0, 0, 400, 20)


def test_no_legend_means_no_room_taken_from_the_axes():
    """``_fit_outside_legend`` with no legend leaves the layout alone.

    The panels call it unconditionally, and a figure drawn without a legend
    must keep the full width for the data rather than being shrunk for nothing.
    """
    figure = RecordingFigure(width=8.0)
    T._fit_outside_legend(figure, None)
    assert figure.calls == []


def test_a_figure_with_no_width_is_left_alone_rather_than_divided_by():
    """A zero-width figure returns before the legend fraction is computed.

    Dividing the legend's pixel width by zero yields inf, and the clamp would
    then push the axes to its minimum on a figure that has no size problem.
    """
    figure = RecordingFigure(width=0.0)
    T._fit_outside_legend(figure, WideLegend())
    assert figure.calls == []
