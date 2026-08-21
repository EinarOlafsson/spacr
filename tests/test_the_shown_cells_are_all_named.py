"""Every shown cell carries a label, and the fraction can be raw.

Instruction 207, reported as "a tone of dotts at 1 and a tone of datapoints
at 0" for guide 225160.

THREE CAUSES, ALL OF THEM REAL:

  * `show_all_in_well` was OFF, so the only cells on screen were the ones
    already annotated to the guide -- every visible cell a hit, every
    visible fraction 1. A view that shows only the cells agreeing with the
    annotation cannot disagree with it.
  * the cells that WERE shown but not picked carried no label, so they were
    counted by the eye and by nothing else -- and a fraction computed over
    only the annotated cells is the fraction that came out as 1.
  * the share was always normalised by what survived `fraction_threshold`,
    which redistributes the filtered guides' reads onto the survivors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.cell_montage import (ANNOTATION_COLUMN, NOT_ANNOTATED,
                                select_montage)


@pytest.fixture
def well():
    rng = np.random.default_rng(0)
    objects = pd.DataFrame([
        {"prc": "p1_w1", "prcfo": f"p1_w1_f1_o{i}",
         "pred": float(rng.random())} for i in range(100)])
    counts = pd.DataFrame([
        {"prc": "p1_w1", "grna": "A_1", "gene": "A", "fraction": 0.4},
        {"prc": "p1_w1", "grna": "B_1", "gene": "B", "fraction": 0.6}])
    return objects, counts


def _frame(plan):
    return getattr(plan, "objects", None)


def _notes(plan):
    """The per-well notes. `MontagePlan.notes` carries the run-level ones --
    the arithmetic is on each WELL, which is where it belongs, since it is a
    different sum per well."""
    return " ".join(str(w.note) for w in (plan.wells or []))


class TestTheDefaultShowsTheWholeWell:

    def test_show_all_in_well_is_on_by_default(self):
        from spacr.picture_settings import OWN_DEFAULTS

        assert OWN_DEFAULTS["show_all_in_well"] is True

    def test_the_help_says_why(self):
        import spacr.settings as settings

        said = settings.tooltips["show_all_in_well"]
        assert "cannot disagree" in said or "every visible" in said


class TestEveryShownCellIsNamed:

    def test_the_unpicked_are_labelled_rather_than_blank(self, well):
        objects, counts = well
        plan = select_montage(objects, counts, "A_1", 1.0,
                              score_column="pred", show_all=True,
                              normalise_fraction=False)
        frame = _frame(plan)
        assert frame is not None
        counts_by_label = dict(frame[ANNOTATION_COLUMN].value_counts())
        assert counts_by_label.get("A_1") == 40
        assert counts_by_label.get(NOT_ANNOTATED) == 60

    def test_the_column_is_there_either_way(self, well):
        """A consumer must not have to know which view produced the frame."""
        objects, counts = well
        plan = select_montage(objects, counts, "A_1", 1.0,
                              score_column="pred", show_all=False,
                              normalise_fraction=False)
        frame = _frame(plan)
        assert ANNOTATION_COLUMN in frame.columns
        assert set(frame[ANNOTATION_COLUMN]) == {"A_1"}

    def test_it_is_spelled_the_way_sudoku_abstains(self):
        """Two annotation routes, one word in one column."""
        from spacr.sudoku import ABSTAIN

        assert NOT_ANNOTATED == ABSTAIN

    def test_the_note_says_the_rest_are_not_annotated(self, well):
        objects, counts = well
        plan = select_montage(objects, counts, "A_1", 1.0,
                              score_column="pred", show_all=True,
                              normalise_fraction=False)
        assert NOT_ANNOTATED in _notes(plan)


class TestTheFractionCanBeRaw:
    """Normalising divides by what SURVIVED the threshold, so the filtered
    guides' reads are handed to the survivors."""

    def test_normalising_inflates_a_partial_well_to_the_whole_of_it(self):
        """One guide surviving means its normalised share is 1.0 -- the
        whole well, from a guide that is 40% of the reads."""
        rng = np.random.default_rng(0)
        objects = pd.DataFrame([
            {"prc": "p1_w1", "prcfo": f"p1_w1_f1_o{i}",
             "pred": float(rng.random())} for i in range(100)])
        counts = pd.DataFrame([
            {"prc": "p1_w1", "grna": "A_1", "gene": "A", "fraction": 0.4}])

        normalised = select_montage(objects, counts, "A_1", 1.0,
                                    score_column="pred",
                                    normalise_fraction=True)
        raw = select_montage(objects, counts, "A_1", 1.0,
                             score_column="pred", normalise_fraction=False)

        assert sum(w.n_expected for w in normalised.wells) == 100
        assert sum(w.n_expected for w in raw.wells) == 40

    def test_the_note_says_which_fraction_was_used(self):
        """A count that cannot be traced to a fraction is a count nobody can
        check."""
        rng = np.random.default_rng(0)
        objects = pd.DataFrame([
            {"prc": "p1_w1", "prcfo": f"p1_w1_f1_o{i}",
             "pred": float(rng.random())} for i in range(40)])
        counts = pd.DataFrame([
            {"prc": "p1_w1", "grna": "A_1", "gene": "A", "fraction": 0.5}])

        raw = select_montage(objects, counts, "A_1", 1.0, score_column="pred",
                             normalise_fraction=False)
        assert "RAW" in _notes(raw)

        normalised = select_montage(objects, counts, "A_1", 1.0,
                                    score_column="pred",
                                    normalise_fraction=True)
        assert "normalised by" in _notes(normalised)

    def test_normalising_is_still_the_default(self):
        """This changes what a count MEANS, so it does not change quietly."""
        import spacr.settings as settings

        assert settings.get_perform_regression_default_settings(
            {})["normalise_fraction"] is True

    def test_the_setting_is_documented_with_the_cost(self):
        import spacr.settings as settings

        said = settings.tooltips["normalise_fraction"]
        assert "0.5526" in said or "1.8" in said
        assert "conservative" in said
