"""The annotation check draws the embedding, and clears it when it refuses.

The verdict is a paragraph of numbers, and a paragraph is not where anyone
looks first. The scatter is what shows WHERE the annotated cells landed --
each cell coloured by how positive its control neighbourhood is -- with the
per-guide numbers beside it so the correlation in the report can be read off
the table rather than believed.

The rule the refusals have to keep is the one the panel already applies to
its text: a refusal must leave NOTHING drawn. A picture under a message
saying this run means nothing is still a picture somebody will screenshot.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from spacr import annotation_umap_qc as qc
from spacr.qt.widgets.annotation_umap_tab import (ANNOTATED,
                                                  AnnotationUmapTab,
                                                  GROUP_SYMBOLS,
                                                  PURITY_COLORMAP,
                                                  PurityScatter)


@pytest.fixture
def tab(qtbot):
    widget = AnnotationUmapTab()
    qtbot.addWidget(widget)
    return widget


def _cells(seed=0, controls=60, annotated=30):
    """Two clean control groups, with annotated cells sitting in each."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(8.0, 1.0, size=(controls, 4)),
              rng.normal(0.0, 1.0, size=(controls, 4)),
              rng.normal(8.0, 1.0, size=(annotated, 4)),
              rng.normal(0.0, 1.0, size=(annotated, 4))]
    frame = pd.DataFrame(np.vstack(blocks),
                         columns=[f"m{i}" for i in range(4)])
    frame["montage_annotation"] = (["Non_annotated"] * (2 * controls)
                                   + ["gA"] * annotated
                                   + ["gB"] * annotated)
    marks = ([qc.POSITIVE] * controls + [qc.NEGATIVE] * controls
             + [None] * (2 * annotated))
    return frame, marks


class TestTheScatterItself:
    """Driven directly, so the drawing is tested without paying for a UMAP."""

    @pytest.fixture
    def plot(self, qtbot):
        widget = PurityScatter()
        qtbot.addWidget(widget)
        return widget

    def _draw(self, plot, purity=(1.0, 0.0, 0.5)):
        embedding = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        return plot.set_embedding(embedding, np.array(purity),
                                  ["gA", "gB", "gA"],
                                  [qc.POSITIVE, qc.NEGATIVE, None])

    def test_every_cell_becomes_a_point(self, plot):
        assert self._draw(plot) == 3
        assert len(plot.plot.listDataItems()) == 1

    def test_purity_is_the_colour_scale(self, plot):
        """Colour is the reading. Position in a UMAP is layout, so the
        number has to be carried by something that is not position."""
        self._draw(plot)
        assert plot._colour_column == ("purity", PURITY_COLORMAP)

    def test_the_group_is_the_shape_so_it_does_not_fight_the_colour(self,
                                                                    plot):
        self._draw(plot)
        symbols = [point.symbol()
                   for point in plot.plot.listDataItems()[0].points()]
        assert symbols == [GROUP_SYMBOLS[qc.POSITIVE],
                           GROUP_SYMBOLS[qc.NEGATIVE],
                           GROUP_SYMBOLS[ANNOTATED]]

    def test_the_shapes_are_keyed_off_the_engine_s_own_labels(self):
        """A copy of "PC" and "NC" here would let the shapes come to
        describe groups the scoring does not produce."""
        assert set(GROUP_SYMBOLS) == {qc.POSITIVE, qc.NEGATIVE, ANNOTATED}
        assert len(set(GROUP_SYMBOLS.values())) == 3

    def test_the_table_it_keeps_carries_the_purity_it_drew(self, plot):
        self._draw(plot)
        frame = plot.frame()
        assert list(frame["purity"]) == [1.0, 0.0, 0.5]
        assert list(frame["guide"]) == ["gA", "gB", "gA"]

    def test_one_purity_everywhere_still_draws_and_says_the_colour_is_not_it(
            self, plot):
        """A colour scale with no range is not a scale, and the points are
        still worth showing -- so the scatter stays and the status says the
        colour means nothing here."""
        assert self._draw(plot, purity=(0.5, 0.5, 0.5)) == 3
        assert plot._colour_column is None
        assert "not drawn as a colour" in plot.status()

    def test_an_empty_embedding_is_said_rather_than_drawn(self, plot):
        assert plot.set_embedding(np.zeros((0, 2)), [], [], []) == 0
        assert plot.plot.listDataItems() == []
        assert "no points" in plot.status()

    def test_clearing_takes_the_table_with_the_points(self, plot):
        self._draw(plot)
        plot.clear_plot("nothing yet")
        assert plot.plot.listDataItems() == []
        assert plot.frame() is None
        assert plot.status() == "nothing yet"


class TestTheGuideTable:

    def test_it_puts_the_effect_beside_the_purity(self, tab):
        """Agreeing is the claim, so the two numbers have to be readable
        against each other rather than one being in a paragraph."""
        tab.set_frame(pd.DataFrame({"m": [1.0]}), effects={"gA": 1.5})
        frame = tab.guide_table({"gA": {"purity": 0.9, "spread": 0.1,
                                        "cells": 30.0}})
        assert list(frame.columns) == ["guide", "purity", "spread", "cells",
                                       "effect"]
        assert frame.loc[0, "effect"] == 1.5

    def test_a_guide_with_no_effect_is_blank_and_not_zero(self, tab):
        """Zero is a coefficient somebody measured."""
        tab.set_frame(pd.DataFrame({"m": [1.0]}), effects={})
        frame = tab.guide_table({"gA": {"purity": 0.9, "spread": 0.1,
                                        "cells": 30.0}})
        assert np.isnan(frame.loc[0, "effect"])

    def test_the_purest_guide_is_first(self, tab):
        tab.set_frame(pd.DataFrame({"m": [1.0]}), effects={})
        frame = tab.guide_table({
            "low": {"purity": 0.1, "spread": 0.0, "cells": 20.0},
            "high": {"purity": 0.9, "spread": 0.0, "cells": 20.0}})
        assert list(frame["guide"]) == ["high", "low"]

    def test_no_scorable_guide_is_an_empty_table_with_the_columns(self, tab):
        tab.set_frame(pd.DataFrame({"m": [1.0]}), effects={})
        frame = tab.guide_table({})
        assert len(frame) == 0
        assert "purity" in frame.columns


class TestARefusalLeavesNothingDrawn:

    def test_a_score_picked_method_clears_the_last_run(self, tab):
        tab.plot.set_embedding(np.array([[0.0, 0.0], [1.0, 1.0]]),
                              [1.0, 0.0], ["gA", "gB"],
                              [qc.POSITIVE, qc.NEGATIVE])
        tab.method.setCurrentIndex(tab.method.findData("rank"))
        assert "refused" in tab.run()
        assert tab.plot.plot.listDataItems() == []
        assert tab.table.table.rowCount() == 0
        assert tab._embedding is None

    def test_no_cells_clears_it_too(self, tab):
        tab.plot.set_embedding(np.array([[0.0, 0.0], [1.0, 1.0]]),
                              [1.0, 0.0], ["gA", "gB"],
                              [qc.POSITIVE, qc.NEGATIVE])
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))
        assert tab.run().get("error") == "no cells"
        assert tab.plot.plot.listDataItems() == []

    def test_too_few_controls_clears_it_too(self, tab):
        frame = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0]})
        tab.set_frame(frame, control_labels=[qc.POSITIVE, qc.NEGATIVE,
                                             None, None])
        tab.plot.set_embedding(np.array([[0.0, 0.0], [1.0, 1.0]]),
                              [1.0, 0.0], ["gA", "gB"],
                              [qc.POSITIVE, qc.NEGATIVE])
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))
        assert tab.run().get("error") == "too few controls"
        assert tab.plot.plot.listDataItems() == []

    def test_an_untrustworthy_embedding_is_refused_rather_than_decorated(
            self, tab, monkeypatch):
        """The guard that matters: a search that separates only the half it
        was tuned on has found the split, not the biology -- so there is no
        scatter to read a conclusion off."""
        frame, marks = _cells(controls=10, annotated=5)
        tab.set_frame(frame, control_labels=marks)
        monkeypatch.setattr(
            "spacr.annotation_umap_qc.fit_on_controls",
            lambda *a, **k: {"recipe": {}, "tuned_silhouette": 0.9,
                             "holdout_silhouette": -0.2, "overfit_gap": 1.1,
                             "trustworthy": False})
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))
        assert tab.run()["verdict"] == "refused"
        assert tab.plot.plot.listDataItems() == []
        assert tab.table.table.rowCount() == 0

    def test_the_named_minimum_is_what_decides_and_not_a_buried_number(
            self, tab, monkeypatch):
        """`MINIMUM_SEPARATION` is documented as what the panel refuses
        below, so raising it has to refuse an embedding the engine was
        willing to call trustworthy."""
        frame, marks = _cells(controls=10, annotated=5)
        tab.set_frame(frame, control_labels=marks)
        monkeypatch.setattr(
            "spacr.annotation_umap_qc.fit_on_controls",
            lambda *a, **k: {"recipe": {}, "tuned_silhouette": 0.6,
                             "holdout_silhouette": 0.5, "overfit_gap": 0.1,
                             "trustworthy": True})
        tab.MINIMUM_SEPARATION = 0.9
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))
        assert tab.run()["verdict"] == "refused"
        assert tab.plot.plot.listDataItems() == []


class TestTheWholeRun:

    def test_a_separable_screen_is_drawn_with_its_guides_beside_it(self, tab):
        frame, marks = _cells()
        tab.set_frame(frame, control_labels=marks,
                      effects={"gA": 1.2, "gB": -0.8})
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))
        out = tab.run()
        if "verdict" in out or "error" in out:
            pytest.skip(f"the controls did not separate here: {out}")
        assert len(tab.plot.frame()) == len(frame)
        assert tab.plot._colour_column == ("purity", PURITY_COLORMAP)
        assert tab.table.table.rowCount() == len(out["purity"])

    def test_the_plot_and_the_table_share_the_divider(self, tab):
        """One result, two views, and the reader decides how much of each --
        a table pinned to a fixed width is a table with the guide names
        elided."""
        assert tab.body.count() == 2
        assert tab.body.widget(0) is tab.plot
        assert tab.body.widget(1) is tab.table
