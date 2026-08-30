"""The annotation UMAP panel's refusals, its well grouping, and its verdict.

This panel is the only place a user can ask whether the cells they annotated
by hand actually sit where the controls say they should. Everything it prints
is read as a claim about the screen, so the paths exercised here are the ones
where a wrong answer is invisible:

* the split level it hands the tuner -- sibling control cells from one well on
  both sides of the held-out split separate because they are siblings, and the
  silhouette then reports that as biology;
* a tuner that failed, which must end as a refusal with the reason showing and
  nothing drawn, because a scatter under "this means nothing" is still a
  scatter somebody screenshots;
* the agreement paragraph itself, which is the answer the user came for.

The tuning and the embedding are stubbed rather than run: a real UMAP search
costs seconds per test and is not what any of these assertions is about, and
the stubs let the cells be placed where the purity they should produce is
known in advance.
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

#: Where each guide's annotated cells are placed along the control gradient.
#: The controls run from all-negative at x=0 to all-positive at x=19.5, so a
#: guide's neighbour purity is a known function of where it sits.
GUIDE_X = {"gA": 19.5, "gB": 15.0, "gC": 12.5,
           "gD": 10.0, "gE": 7.5, "gF": 0.0}

#: Guide effects ordered the same way as GUIDE_X, so purity and effect agree.
EFFECTS = {"gA": 1.5, "gB": 1.2, "gC": 0.8,
           "gD": -0.4, "gE": -1.2, "gF": -1.5}

CONTROLS = 40
CELLS_PER_GUIDE = 12

#: What a tuner that found a real separation returns.
TRUSTWORTHY = {"recipe": {"n_neighbors": 15, "min_dist": 0.1},
               "tuned_silhouette": 0.62, "holdout_silhouette": 0.55,
               "overfit_gap": 0.07, "trustworthy": True}


def _screen(*, wells=True):
    """A screen whose neighbour purity is decided by the x coordinate.

    Negative controls fill the low half of the axis and positives the high
    half, so a cell's share of positive control neighbours rises with ``m0``.
    Half the control cells come from one well and half from another.
    """
    xs, marks, guides = [], [], []
    for i in range(CONTROLS):
        xs.append(i * 0.5)
        marks.append(qc.NEGATIVE if i < CONTROLS // 2 else qc.POSITIVE)
        guides.append("Non_annotated")
    for guide, x in GUIDE_X.items():
        xs.extend([x] * CELLS_PER_GUIDE)
        marks.extend([None] * CELLS_PER_GUIDE)
        guides.extend([guide] * CELLS_PER_GUIDE)
    frame = pd.DataFrame({"m0": xs, "m1": [0.0] * len(xs)})
    frame["montage_annotation"] = guides
    if wells:
        frame["plateID"] = "p1"
        frame["rowID"] = ["A" if mark == qc.NEGATIVE else "B"
                          for mark in marks]
        frame["columnID"] = "3"
    return frame, marks


@pytest.fixture
def tab(qtbot):
    widget = AnnotationUmapTab()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def stub_embed(monkeypatch):
    """Use the first two feature columns as the embedding.

    The panel's job here is what it does WITH an embedding; running a real
    UMAP would only make the purity it scores unpredictable.
    """
    monkeypatch.setattr(
        "spacr.hyperparam._default_umap_embed",
        lambda features, recipe, seed: np.asarray(features,
                                                  dtype=float)[:, :2])


def _stub_tuner(monkeypatch, result):
    """Replace the tuner with one that records how it was called."""
    calls = []

    def fake(features, labels, **kwargs):
        calls.append(dict(kwargs, features=np.asarray(features),
                          labels=list(labels)))
        return dict(result)

    monkeypatch.setattr("spacr.annotation_umap_qc.fit_on_controls", fake)
    return calls


def _draw_something(tab):
    """Leave a scatter on the plot so a later clear has work to do."""
    tab.plot.set_embedding(np.array([[0.0, 0.0], [1.0, 1.0]]), [1.0, 0.0],
                           ["gA", "gB"], [qc.POSITIVE, qc.NEGATIVE])
    assert tab.plot.plot.listDataItems()


class TestWhichSplitLevelTheTunerIsGiven:
    """A held-out score is only a score if the halves are really apart."""

    def test_a_tab_with_no_cells_yet_answers_cell_rather_than_raising(
            self, tab):
        """The grouping question is asked before the frame is checked in
        some call orders, and a tab a user has only just opened has no
        frame. Raising there would take down the whole panel on a click
        instead of falling back to the finest, most conservative level."""
        assert tab._frame is None
        assert tab._control_groups([0, 1]) == (None, "cell")

        frame, marks = _screen()
        tab.set_frame(frame, control_labels=marks)
        groups, level = tab._control_groups([0, 1, CONTROLS - 1])
        assert level == "well"
        assert len(groups) == 3

    def test_control_cells_from_one_well_carry_one_group_id(self, tab):
        """Sibling cells from a single well are near-duplicates. Split
        between the tuning and held-out halves they separate because they
        are siblings, and the panel then reports that as the controls
        separating -- the exact false pass this check exists to catch."""
        frame, marks = _screen()
        tab.set_frame(frame, control_labels=marks)
        rows = [i for i, mark in enumerate(marks) if mark is not None]

        groups, level = tab._control_groups(rows)
        assert level == "well"
        assert len(groups) == CONTROLS
        # Two wells: the negatives share one row letter, the positives the
        # other, so every control cell's id is one of exactly two values.
        assert len(set(map(str, groups))) == 2
        assert str(groups[0]) == str(groups[1])
        assert str(groups[0]) != str(groups[-1])

    def test_a_frame_that_cannot_name_a_well_is_split_per_cell(self, tab):
        """The leakiest rung, and the panel takes it only when the data
        cannot support a better one -- never silently because somebody
        forgot to group. A frame WITH the metadata in the same test proves
        the 'cell' answer is a property of the frame, not of the code path
        always giving up."""
        with_wells, marks = _screen()
        rows = [i for i, mark in enumerate(marks) if mark is not None]
        tab.set_frame(with_wells, control_labels=marks)
        assert tab._control_groups(rows)[1] == "well"

        tab.set_frame(with_wells.drop(columns=["plateID", "rowID",
                                               "columnID"]),
                      control_labels=marks)
        assert tab._control_groups(rows) == (None, "cell")

    def test_the_well_identity_is_what_reaches_the_tuner(self, tab,
                                                         monkeypatch):
        """Deriving the grouping and then not passing it on would leave the
        panel documenting a guarantee it does not keep."""
        frame, marks = _screen()
        tab.set_frame(frame, control_labels=marks)
        calls = _stub_tuner(monkeypatch, dict(TRUSTWORTHY,
                                              trustworthy=False,
                                              holdout_silhouette=-0.1))
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))

        assert tab.run()["verdict"] == "refused"
        assert len(calls) == 1
        assert calls[0]["group_by"] == "well"
        assert len(set(map(str, calls[0]["groups"]))) == 2
        assert calls[0]["features"].shape == (CONTROLS, 2)


class TestATunerThatFailed:

    def test_the_tuning_error_is_shown_and_nothing_is_left_drawn(
            self, tab, monkeypatch):
        """When the search itself failed there is no embedding at all, so
        the panel has to say so and take the previous run's scatter down
        with it -- a stale picture under a failure message is the one thing
        a reader will take away from the screen."""
        frame, marks = _screen()
        tab.set_frame(frame, control_labels=marks)
        _stub_tuner(monkeypatch,
                    {"error": "every control cell carries the same label"})
        _draw_something(tab)
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))

        out = tab.run()

        assert out == {"error": "every control cell carries the same label"}
        assert ("The embedding could not be tuned: every control cell "
                "carries the same label") in tab.report.toPlainText()
        assert tab.plot.plot.listDataItems() == []
        assert tab.table.table.rowCount() == 0
        assert tab._embedding is None
        assert "could not be tuned" in tab.plot.status()


class TestTheVerdictParagraph:

    def _run(self, tab, monkeypatch, effects):
        frame, marks = _screen()
        tab.set_frame(frame, control_labels=marks, effects=effects)
        _stub_tuner(monkeypatch, TRUSTWORTHY)
        tab.method.setCurrentIndex(tab.method.findData("sudoku"))
        return tab.run()

    def test_an_agreeing_screen_reports_the_correlation_it_measured(
            self, tab, monkeypatch, stub_embed):
        """This paragraph IS the answer the user opened the panel for. The
        numbers in it have to be the ones that were computed, not a
        re-derivation, and the sentence at the end has to follow the
        permutation test rather than the sign of the correlation alone."""
        out = self._run(tab, monkeypatch, EFFECTS)
        agreement = out["agreement"]
        report = tab.report.toPlainText()

        assert agreement["correlation"] > 0.9
        assert agreement["p_value"] < 0.05
        assert agreement["separated"] is True
        assert f"rho = {agreement['correlation']:+.3f}" in report
        assert f"p = {agreement['p_value']:.4f}" in report
        assert f"({agreement['permutations']:,} permutations)" in report
        assert (f"positive-effect guides mean purity "
                f"{agreement['positive_effect_purity']:.3f}") in report
        assert (f"negative-effect guides mean purity "
                f"{agreement['negative_effect_purity']:.3f}") in report
        assert ("The annotated cells land where their guide's effect says "
                "they should.") in report
        assert "do NOT land" not in report
        assert ("Held-out separation of the controls: 0.550 "
                "(gap to tuned 0.070).") in report
        assert "6 guide(s) had enough annotated cells to score." in report

    def test_guides_that_land_the_wrong_way_round_are_said_so(
            self, tab, monkeypatch, stub_embed):
        """Cells land somewhere whatever the annotation says. A panel that
        printed the agreeing sentence for every screen would be a panel
        whose verdict carries no information."""
        out = self._run(tab, monkeypatch,
                        {guide: -weight for guide, weight
                         in EFFECTS.items()})

        assert out["agreement"]["correlation"] < -0.9
        assert out["agreement"]["separated"] is False
        report = tab.report.toPlainText()
        assert "The annotated cells do NOT land where" in report
        assert "shuffled between guides" in report

    def test_too_few_guides_with_an_effect_says_so_instead_of_a_number(
            self, tab, monkeypatch, stub_embed):
        """A Spearman correlation over two points is not a measurement.
        Printing one anyway would put a rho and a p-value on screen that a
        reader has no way of knowing to distrust. The full-effects run in
        the same test shows the correlation is printed when it is earned."""
        full = self._run(tab, monkeypatch, EFFECTS)
        assert "rho = " in tab.report.toPlainText()
        assert "correlation" in full["agreement"]

        out = self._run(tab, monkeypatch, {"gA": 1.5, "gB": 1.2})

        report = tab.report.toPlainText()
        assert "rho = " not in report
        assert "guide(s) have both a purity and an effect" in report
        assert "correlation" not in out["agreement"]
        # The run still happened: the per-guide table is there to read.
        assert len(out["purity"]) == 6
        assert tab.table.table.rowCount() == 6

    def test_the_scatter_carries_every_cell_with_its_group_as_the_shape(
            self, tab, monkeypatch, stub_embed):
        """The picture is what says WHERE the annotated cells landed, so it
        has to hold the controls and the annotated cells together and stay
        readable as which is which -- purity is already using the colour."""
        out = self._run(tab, monkeypatch, EFFECTS)
        total = CONTROLS + CELLS_PER_GUIDE * len(GUIDE_X)

        assert tab._embedding.shape == (total, 2)
        assert len(tab.plot.frame()) == total
        assert tab.plot._colour_column == ("purity", PURITY_COLORMAP)
        symbols = [point.symbol()
                   for point in tab.plot.plot.listDataItems()[0].points()]
        assert symbols.count(GROUP_SYMBOLS[qc.POSITIVE]) == CONTROLS // 2
        assert symbols.count(GROUP_SYMBOLS[qc.NEGATIVE]) == CONTROLS // 2
        assert (symbols.count(GROUP_SYMBOLS[ANNOTATED])
                == CELLS_PER_GUIDE * len(GUIDE_X))
        assert f"{total:,} cells." in tab.plot.status()
        assert "Triangles are positive controls" in tab.plot.status()
        assert out["separation"]["trustworthy"] is True

    def test_the_purest_guide_is_the_one_at_the_positive_end(
            self, tab, monkeypatch, stub_embed):
        """The whole claim is that a guide's cells sit among the controls
        its effect predicts. If the guide placed deepest in the positive
        controls were not the purest row, the number in the table would not
        be describing the picture beside it."""
        out = self._run(tab, monkeypatch, EFFECTS)
        table = tab.guide_table(out["purity"])

        assert list(table["guide"])[-1] == "gF"
        assert table.loc[0, "purity"] > table["purity"].iloc[-1]
        assert table.loc[0, "purity"] == pytest.approx(0.8)
        assert table["purity"].iloc[-1] == pytest.approx(0.2)
        assert table.loc[0, "cells"] == CELLS_PER_GUIDE


class TestTheGuideTableWithoutARun:

    def test_no_scorable_guide_leaves_the_columns_a_reader_expects(
            self, tab):
        """An empty result still has to be a table with headings: the same
        widget shows both, and a frame with no columns makes the panel look
        broken rather than empty. Sorting one real row in the same test
        shows the empty case is skipping the sort, not the whole build."""
        tab.set_frame(pd.DataFrame({"m": [1.0]}), effects={"gA": 2.0})

        filled = tab.guide_table({
            "gA": {"purity": 0.2, "spread": 0.0, "cells": 12.0},
            "gB": {"purity": 0.9, "spread": 0.0, "cells": 15.0}})
        assert list(filled["guide"]) == ["gB", "gA"]
        assert filled.loc[1, "effect"] == 2.0

        empty = tab.guide_table({})
        assert len(empty) == 0
        assert list(empty.columns) == ["guide", "purity", "spread", "cells",
                                       "effect"]


class TestTheScatterOnItsOwn:

    @pytest.fixture
    def plot(self, qtbot):
        widget = PurityScatter()
        qtbot.addWidget(widget)
        return widget

    def _draw(self, plot, purity):
        return plot.set_embedding(
            np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
            np.array(purity), ["gA", "gB", "gA"],
            [qc.POSITIVE, qc.NEGATIVE, None])

    def test_an_embedding_with_no_points_replaces_the_last_one(self, plot):
        """A run that produced nothing must not leave the previous run's
        cells on screen wearing this run's status line."""
        assert self._draw(plot, (1.0, 0.0, 0.5)) == 3
        assert plot.plot.listDataItems() != []

        assert plot.set_embedding(np.zeros((0, 2)), [], [], []) == 0
        assert plot.plot.listDataItems() == []
        assert plot.frame() is None
        assert "no points to draw" in plot.status()

    def test_a_purity_that_never_varies_is_still_drawn_without_a_scale(
            self, plot):
        """A colour scale over one repeated value is not a scale, and
        painting the cells anywhere on it would state a difference between
        them that was not measured. The cells are still worth seeing, so
        they stay and the status says the colour is not the reading."""
        assert self._draw(plot, (0.4, 0.4, 0.4)) == 3
        assert plot._colour_column is None
        assert "Purity is not drawn as a colour" in plot.status()
        assert len(plot.frame()) == 3

        assert self._draw(plot, (1.0, 0.0, 0.5)) == 3
        assert plot._colour_column == ("purity", PURITY_COLORMAP)
        assert "3 cells." in plot.status()
