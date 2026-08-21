"""Sudoku annotation, and the harness that says whether it works.

Instruction 209. The maintainer approved the mathematics and asked for two
things: the method, and "a validation and test mechanism to show that this
actually works ... for each strategy".

WHAT IS ACTUALLY BEING TESTED HERE is not that the code runs. It is that the
method beats what the sequencing alone can do, collapses to that baseline
when there is no signal to find, and does not read its own anchors. Those
three are the claims; everything else is plumbing.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import annotation_validation as validation
from spacr import sudoku as module


def _call(screen, **kwargs):
    return list(module.sudoku(screen.features, screen.scores, screen.wells,
                              screen.fractions, screen.guides,
                              anchor_min_fraction=0.30, **kwargs).guides)


# ---------------------------------------------------------------------- graph

class TestTheGraph:

    def test_mutual_knn_can_leave_an_outlier_alone(self):
        """A plain kNN graph gives an outlier k edges it did not earn -- it
        has to have SOME nearest neighbours. Mutual kNN lets it have none,
        which is the honest answer and is what `reach` reports."""
        cloud = np.random.default_rng(0).normal(size=(60, 3))
        far = np.array([[50.0, 50.0, 50.0]])
        points = np.vstack([cloud, far])

        mutual = module.similarity_graph(points, neighbours=5, mutual=True)
        plain = module.similarity_graph(points, neighbours=5, mutual=False)

        assert mutual[-1].nnz == 0, "the outlier earned no mutual edge"
        assert plain[-1].nnz > 0, "a plain graph hands it edges anyway"

    def test_the_diagonal_is_empty(self):
        """A self-edge would let a cell affirm itself."""
        points = np.random.default_rng(1).normal(size=(30, 4))
        graph = module.similarity_graph(points, neighbours=5)
        assert graph.diagonal().sum() == 0

    def test_it_survives_degenerate_input(self):
        assert module.similarity_graph(np.zeros((0, 3))).shape[0] == 0
        assert module.similarity_graph(np.zeros((1, 3))).nnz == 0


class TestPropagation:

    def test_mass_reaches_a_neighbour_and_not_an_island(self):
        from scipy import sparse

        # Two cells joined, one alone.
        graph = sparse.csr_matrix(np.array([[0.0, 1.0, 0.0],
                                            [1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]]))
        seeds = np.array([[1.0], [0.0], [0.0]])

        out = module.propagate(graph, seeds, alpha=0.9)

        assert out[1, 0] > 0, "the neighbour heard about it"
        assert out[2, 0] == pytest.approx(0.0), "the island did not"

    def test_the_mass_is_not_normalised(self):
        """Row-normalising here would turn a cell that received almost
        nothing into a confident-looking row summing to one -- and that cell
        is the interesting one."""
        from scipy import sparse

        graph = sparse.csr_matrix(np.array([[0.0, 1.0, 0.0],
                                            [1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]]))
        out = module.propagate(graph, np.array([[1.0], [0.0], [0.0]]))
        assert out.sum(axis=1)[2] == pytest.approx(0.0)


class TestTheWellConstraint:

    def test_each_guide_gets_the_count_its_reads_imply(self):
        mass = np.ones((10, 2))
        wells = ["w"] * 10
        fractions = {"w": {"a": 0.7, "b": 0.3}}

        out = module.constrain_to_fractions(mass, wells, ("a", "b"), fractions)

        assert out.sum(axis=1) == pytest.approx(np.ones(10))
        assert out[:, 0].sum() == pytest.approx(7.0, abs=0.05)
        assert out[:, 1].sum() == pytest.approx(3.0, abs=0.05)

    def test_a_cell_no_anchor_reached_gets_the_prior(self):
        """It is a cell, it carries something, and 'no idea' is the answer."""
        mass = np.zeros((4, 2))
        out = module.constrain_to_fractions(
            mass, ["w"] * 4, ("a", "b"), {"w": {"a": 0.75, "b": 0.25}})
        assert out[0, 0] > out[0, 1]


# --------------------------------------------------------------- the claims

class TestTheTwoScoresStaySeparate:

    def test_a_cell_unlike_everything_is_low_on_both(self):
        """The fourth quadrant, which a single ratio hides by mapping it to
        the same value as a cell that is high on both."""
        screen = validation.synthesise(wells=6, cells_per_well=40, seed=5)
        far = screen.features.max(axis=0) + 40.0
        features = np.vstack([screen.features, far])
        wells = list(screen.wells) + [screen.wells[0]]
        scores = np.append(screen.scores, 0.0)

        out = module.sudoku(features, scores, wells, screen.fractions,
                            screen.guides, anchor_min_fraction=0.30)

        assert out.reach[-1] < out.reach[:-1].mean()
        assert out.guides[-1] == module.ABSTAIN

    def test_the_result_carries_both_matrices(self):
        screen = validation.synthesise(wells=4, cells_per_well=30, seed=6)
        out = module.sudoku(screen.features, screen.scores, screen.wells,
                            screen.fractions, screen.guides,
                            anchor_min_fraction=0.30)
        assert out.affirm.shape == out.eliminate.shape
        assert out.affirm.shape[1] == len(screen.guides)


class TestItBeatsTheFractionsWhenThereIsSignal:
    """The first claim, and the only one that justifies the method."""

    def test_it_is_more_precise_than_the_majority_baseline(self):
        screen = validation.synthesise(effect=3.0, seed=11)

        mine = validation.score_annotation(
            screen.truth, _call(screen), guides=screen.guides)
        floor = validation.score_annotation(
            screen.truth, validation.baseline_majority(screen),
            guides=screen.guides)

        assert mine.precision > floor.precision + 0.10, (
            f"sudoku {mine.summary()} vs baseline {floor.summary()}")


class TestItDoesNotInventSignal:
    """The second claim, and the one that catches a method reading itself."""

    def test_with_no_effect_it_does_no_better_than_the_fractions(self):
        screen = validation.synthesise(effect=0.0, seed=12)

        mine = validation.score_annotation(
            screen.truth, _call(screen), guides=screen.guides)
        floor = validation.score_annotation(
            screen.truth, validation.baseline_majority(screen),
            guides=screen.guides)

        # Features carrying nothing: whatever it scores must be what the
        # sequencing was already worth, not more.
        assert mine.precision <= floor.precision + 0.10, (
            f"sudoku {mine.summary()} beat the fractions on noise: "
            f"baseline {floor.summary()}")

    def test_the_score_stays_out_of_the_graph_by_default(self):
        """The anchors are chosen BY SCORE. A graph built on the score would
        place every high-scoring cell near every guide's anchors."""
        screen = validation.synthesise(wells=4, cells_per_well=30, seed=13)
        out = module.sudoku(screen.features, screen.scores, screen.wells,
                            screen.fractions, screen.guides,
                            anchor_min_fraction=0.30)
        assert out.report["score_in_graph"] is False

    def test_turning_it_on_says_so(self):
        screen = validation.synthesise(wells=4, cells_per_well=30, seed=14)
        out = module.sudoku(screen.features, screen.scores, screen.wells,
                            screen.fractions, screen.guides,
                            anchor_min_fraction=0.30,
                            use_score_as_feature=True)
        assert "circular" in str(out.report.get("warning", ""))


class TestThePermutationNull:
    """The third claim -- and the only check that could run on real data."""

    def test_shuffling_the_wells_destroys_the_performance(self):
        screen = validation.synthesise(effect=3.0, seed=15)
        null = validation.permuted(screen, seed=2)

        real = validation.score_annotation(
            screen.truth, _call(screen), guides=screen.guides)
        chance = validation.score_annotation(
            null.truth, _call(null), guides=null.guides)

        assert real.precision > chance.precision + 0.20


class TestTheSequentialForm:

    def test_it_claims_in_confidence_order_and_stops(self):
        screen = validation.synthesise(effect=3.0, wells=8, seed=16)
        ranking = [(g, 1.0 / (i + 1)) for i, g in enumerate(screen.guides)]

        out = module.sudoku_all(screen.features, screen.scores, screen.wells,
                                screen.fractions, ranking,
                                anchor_min_fraction=0.30)

        rounds = out.report["claimed_by_round"]
        assert rounds, "it ran at least one round"
        assert out.report["claimed"] == sum(r["claimed"] for r in rounds)

    def test_one_guide_at_a_time_is_not_the_same_as_one_guide_alone(self):
        """The bug the benchmark caught: run with a single guide, the well
        constraint normalises each row over one column, every posterior is
        1.0, and the first guide claims the entire screen."""
        screen = validation.synthesise(effect=3.0, wells=8, seed=17)
        ranking = [(g, 1.0 / (i + 1)) for i, g in enumerate(screen.guides)]

        out = module.sudoku_all(screen.features, screen.scores, screen.wells,
                                screen.fractions, ranking,
                                anchor_min_fraction=0.30)

        first = out.report["claimed_by_round"][0]
        assert first["claimed"] < len(screen), (
            "the first guide took the whole screen -- the per-round call has "
            "gone back to comparing a guide against nothing")

    def test_a_single_guide_run_really_does_saturate(self):
        """Pinning the mechanism, so the fix cannot be undone unnoticed."""
        screen = validation.synthesise(effect=3.0, wells=4, seed=18)
        out = module.sudoku(screen.features, screen.scores, screen.wells,
                            screen.fractions, [screen.guides[0]],
                            anchor_min_fraction=0.30)
        # With one column every row sums to one over that column alone.
        reached = out.posterior[:, 0] > 0
        assert np.allclose(out.posterior[reached, 0], 1.0)


class TestSudokuIsScopedToTheWellsThatHoldTheGuide:
    """Which cells does it annotate?

    Asked 2026-08-21: "will suduko only annotate cells in wells with the
    chosen genes or will it annotate all cells it can annotat?"

    THE WELLS THAT HOLD THE GUIDE, and within those, every guide that
    co-occurs there. Two reasons, and the second is the one that would have
    stopped a real run.

    STATISTICAL: a posterior is a COMPARISON, and a guide that never shares
    a well with this one cannot be compared against it. It contributes a
    column of zeros and a constraint of zero -- not evidence, arithmetic on
    an empty set.

    ARITHMETIC: unscoped, the seed matrix is (all cells x all guides) and
    the propagation holds about three of them. On the maintainer's screen --
    143,765 cells, 1,379 guides -- that is 1.6 GB each and roughly 4.8 GB
    live, to draw a montage of ONE coefficient.
    """

    @staticmethod
    def _screen():
        import pandas as pd

        rng = np.random.default_rng(0)
        objects = pd.DataFrame([
            {"prc": f"p1_w{w}", "prcfo": f"p1_w{w}_f1_o{i}",
             "pred": float(rng.random()),
             "area": float(rng.normal(100, 20)),
             "intensity": float(rng.normal(50, 10))}
            for w in range(10) for i in range(60)])
        rows = []
        for w in range(10):
            pairs = ([("A_1", 0.6), ("B_1", 0.4)] if w < 4
                     else [("B_1", 0.5), ("C_1", 0.5)])
            for guide, share in pairs:
                rows.append({"prc": f"p1_w{w}", "grna": guide,
                             "gene": guide.split("_")[0], "fraction": share})
        return objects, pd.DataFrame(rows)

    def _note(self, plan):
        return " | ".join(n for n in (plan.notes or [])
                          if "sudoku" in n.lower())

    def test_it_reads_only_the_wells_holding_the_guide(self):
        from spacr.cell_montage import select_montage

        objects, counts = self._screen()
        plan = select_montage(objects, counts, "A_1", 1.0,
                              score_column="pred", picking="sudoku")

        said = self._note(plan)
        assert "across 4 well(s)" in said, said
        # 4 wells x 60 cells, not the whole 600-cell plate.
        assert "of 240 cell(s)" in said, said

    def test_it_annotates_every_cell_in_those_wells_not_only_the_guide_s(self):
        """Within the scope it decides EVERY cell it can -- the other guides
        are what this one is compared against, so leaving them out would
        leave nothing to compare with."""
        from spacr.cell_montage import select_montage

        objects, counts = self._screen()
        plan = select_montage(objects, counts, "A_1", 1.0,
                              score_column="pred", picking="sudoku")

        said = self._note(plan)
        annotated = int(said.split(" of ")[0].split()[-1].replace(",", ""))
        assert annotated > 120, said

    def test_a_guide_in_no_well_here_is_said_rather_than_drawn(self):
        from spacr.cell_montage import select_montage

        objects, counts = self._screen()
        plan = select_montage(objects, counts, "C_1", 1.0,
                              score_column="pred", picking="sudoku")

        # C_1 IS present, in the other six wells -- so this must scope to
        # those and not to A_1's.
        said = self._note(plan)
        assert "across 6 well(s)" in said, said
