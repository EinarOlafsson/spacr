"""Do annotated cells land where their guide's effect says they should?

Proposed 2026-08-21 as "a quality controll and a proof of sorts that the
annotation is working". It is a good check because it is INDEPENDENT: the
annotation is made from sequencing fractions plus a phenotype call, and this
asks where the cell sits among the controls using every measurement at once.

THESE TESTS ARE MOSTLY ABOUT THE CHECK BEING UNABLE TO AGREE WITH WHATEVER
IT IS SHOWN. A UMAP tuned until two labelled groups separate always
separates them; a picture of clusters is not a measurement; and cells land
somewhere whatever the annotation says.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import annotation_umap_qc as module


def _two_blobs(n=200, separation=6.0, seed=0):
    """Controls in two clean groups, in two dimensions."""
    rng = np.random.default_rng(seed)
    positive = rng.normal(separation, 1.0, size=(n, 2))
    negative = rng.normal(0.0, 1.0, size=(n, 2))
    points = np.vstack([positive, negative])
    labels = [module.POSITIVE] * n + [module.NEGATIVE] * n
    return points, labels


class TestTheReadoutIsANumber:
    """"Do they cluster with PC" is answered by counting neighbours. Cluster
    sizes and the distances between them are artefacts of the projection;
    who is near whom is all it preserves."""

    def test_a_cell_among_the_positives_scores_one(self):
        points, labels = _two_blobs()
        query = np.array([[6.0, 6.0]])
        purity = module.neighbour_purity(
            np.vstack([points, query]), list(labels) + [None], k=15)
        assert purity[-1] == pytest.approx(1.0)

    def test_a_cell_among_the_negatives_scores_zero(self):
        points, labels = _two_blobs()
        query = np.array([[0.0, 0.0]])
        purity = module.neighbour_purity(
            np.vstack([points, query]), list(labels) + [None], k=15)
        assert purity[-1] == pytest.approx(0.0)

    def test_a_cell_between_them_scores_between(self):
        points, labels = _two_blobs()
        query = np.array([[3.0, 3.0]])
        purity = module.neighbour_purity(
            np.vstack([points, query]), list(labels) + [None], k=40)
        assert 0.1 < purity[-1] < 0.9

    def test_a_control_does_not_count_itself(self):
        """So controls can be scored on the same footing as everything
        else -- which is what makes them a usable reference for what a pure
        score looks like here."""
        points, labels = _two_blobs(n=50)
        purity = module.neighbour_purity(points, labels, k=10)
        # A positive control still scores near 1 from its neighbours alone.
        assert np.nanmean(purity[:50]) > 0.9
        assert np.nanmean(purity[50:]) < 0.1


class TestTheHyperparametersAreNotChosenOnWhatIsJudged:

    def test_it_reports_a_held_out_score(self):
        points, labels = _two_blobs(n=60)
        out = module.fit_on_controls(
            points, labels,
            recipes=[{"n_neighbors": 10, "min_dist": 0.1}])
        if "error" in out:
            pytest.skip(out["error"])
        assert "holdout_silhouette" in out
        assert "overfit_gap" in out

    def test_separable_controls_survive_the_holdout(self):
        points, labels = _two_blobs(n=80, separation=8.0)
        out = module.fit_on_controls(
            points, labels,
            recipes=[{"n_neighbors": 15, "min_dist": 0.1}])
        if "error" in out:
            pytest.skip(out["error"])
        assert out["holdout_silhouette"] > 0.0

    def test_controls_that_do_not_differ_do_not_separate(self):
        """The case the guard exists for: labels assigned at random to one
        cloud. A search that 'found structure' here found the split."""
        rng = np.random.default_rng(1)
        points = rng.normal(size=(160, 2))
        labels = [module.POSITIVE if i % 2 else module.NEGATIVE
                  for i in range(160)]
        out = module.fit_on_controls(
            points, labels,
            recipes=[{"n_neighbors": 15, "min_dist": 0.1}])
        if "error" in out:
            pytest.skip(out["error"])
        assert out["holdout_silhouette"] < 0.15
        assert not out["trustworthy"]

    def test_one_label_is_refused(self):
        points, _ = _two_blobs(n=20)
        out = module.fit_on_controls(points, [module.POSITIVE] * 40,
                                     recipes=[{"n_neighbors": 5}])
        assert "error" in out


class TestTheNull:

    @staticmethod
    def _guides(agreeing, seed=0):
        """Twelve guides whose purity either tracks their effect or does
        not."""
        rng = np.random.default_rng(seed)
        effects, purity = {}, {}
        for index in range(12):
            effect = float(rng.normal())
            effects[f"g{index}"] = effect
            base = 0.5 + (0.4 * np.sign(effect) if agreeing else 0.0)
            purity[f"g{index}"] = {
                "purity": float(np.clip(base + rng.normal(0, 0.05), 0, 1)),
                "spread": 0.1, "cells": 40.0}
        return purity, effects

    def test_an_annotation_that_agrees_is_detected(self):
        purity, effects = self._guides(agreeing=True, seed=2)
        out = module.effect_agreement(purity, effects, permutations=299)
        assert out["separated"]
        assert out["correlation"] > 0.5
        assert out["positive_effect_purity"] > out["negative_effect_purity"]

    def test_an_annotation_that_does_not_is_not(self):
        """Cells land somewhere whatever the annotation says."""
        purity, effects = self._guides(agreeing=False, seed=3)
        out = module.effect_agreement(purity, effects, permutations=299)
        assert not out["separated"]

    def test_too_few_guides_is_an_error_not_a_correlation(self):
        out = module.effect_agreement(
            {"a": {"purity": 0.9, "cells": 20.0}}, {"a": 1.0})
        assert "error" in out


class TestAGuideNeedsEnoughCellsToBeScored:

    def test_a_thin_guide_is_left_out_rather_than_reported_noisily(self):
        purity = np.array([1.0, 0.0, 1.0] + [0.9] * 20)
        guides = ["thin"] * 3 + ["thick"] * 20
        out = module.purity_by_guide(purity, guides, minimum_cells=10)
        assert "thin" not in out
        assert "thick" in out

    def test_abstentions_are_not_a_guide(self):
        purity = np.full(20, 0.5)
        guides = ["Non_annotated"] * 20
        assert module.purity_by_guide(purity, guides) == {}


class TestTheCircularityItCannotEscape:
    """Which methods may be judged this way at all."""

    def test_a_score_picked_method_is_refused(self):
        """`rank` takes the top-scoring cells in the well, so its cells
        sitting near the positive controls restates how it chose them."""
        said = module.circularity_warning("rank")
        assert said
        assert "validates nothing" in said

    def test_a_method_that_did_not_use_the_score_is_allowed(self):
        assert module.circularity_warning("sudoku") == ""
        assert module.circularity_warning("assigned") == ""

    def test_putting_the_score_in_the_features_is_flagged_too(self):
        said = module.circularity_warning("sudoku", score_in_features=True)
        assert "both sides" in said
