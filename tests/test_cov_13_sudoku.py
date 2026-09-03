"""Degenerate screens must abstain rather than invent an assignment.

The sudoku assignment turns sequencing fractions plus cell morphology into a
guide name per cell, and every path here is a case where the evidence runs
out: a well whose label is missing, a well whose cells carry no score, a well
sequencing says nothing about, a run with no guides, a run with one cell. Each
must come back as an abstention or an empty result. Filling those in with a
plausible guess is the failure mode that matters, because a fabricated
per-cell label is indistinguishable from a real one downstream.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.sudoku import (
    ABSTAIN,
    SudokuResult,
    anchors_for,
    constrain_to_fractions,
    propagate,
    similarity_graph,
    sudoku,
    sudoku_all,
)


def _cells(n_per_well: int = 12, seed: int = 3):
    """Two wells of morphologically separable cells, with scores."""
    rng = np.random.default_rng(seed)
    features = np.vstack([
        rng.normal(loc=(0.0, 0.0), scale=0.3, size=(n_per_well, 2)),
        rng.normal(loc=(5.0, 5.0), scale=0.3, size=(n_per_well, 2)),
    ])
    wells = ["A01"] * n_per_well + ["A02"] * n_per_well
    scores = np.linspace(0.1, 0.9, len(wells))
    return features, np.asarray(wells), scores


def test_a_well_with_no_score_anywhere_contributes_no_anchors():
    """Anchors are chosen by score, so a scoreless well has none to give.

    The quantile of an all-NaN column is NaN, and every comparison against it
    is False -- which would silently produce an empty selection anyway. Making
    the well drop out explicitly keeps a later change to the quantile from
    turning "no data" into "take everything".
    """
    wells = ["A01", "A01", "A02", "A02"]
    scores = np.array([np.nan, np.nan, 0.2, 0.9])
    fractions = {"A01": {"g1": 1.0}, "A02": {"g1": 1.0}}

    picked = anchors_for("g1", wells, fractions, scores, quantile=0.5)

    assert picked.tolist() == [3]


def test_a_cell_whose_well_label_is_missing_anchors_nothing():
    """A NaN well label matches no well, and must not become its own group.

    Wells arrive from a merged table where the plate column can be absent for
    some rows. Such a row belongs to no sequencing well, so there is no
    fraction that could justify calling it an anchor.
    """
    wells = np.array([np.nan, np.nan, np.nan], dtype=float)
    scores = np.array([0.1, 0.5, 0.9])

    picked = anchors_for("g1", wells, {}, scores, min_fraction=0.0)

    assert picked.tolist() == []


def test_one_well_cannot_flood_the_anchor_set():
    """A well contributes at most ``max_per_well`` anchors, highest score first.

    Without the cap, one deeply sampled well decides the whole propagation and
    the graph learns that well's batch effects rather than the guide's
    phenotype.
    """
    wells = ["A01"] * 6
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7])

    picked = anchors_for("g1", wells, {"A01": {"g1": 1.0}}, scores,
                         quantile=0.0, max_per_well=2)

    assert picked.tolist() == [1, 3]


def test_propagating_into_a_graph_of_the_wrong_size_returns_no_mass():
    """Mismatched shapes must yield zeros, never a broadcast answer.

    ``sudoku_all`` slices the cell pool each round and rebuilds the graph from
    the survivors. A stale graph reaching this with the previous round's size
    would otherwise propagate one round's anchors onto another round's cells.
    """
    graph = similarity_graph(np.random.default_rng(0).normal(size=(6, 2)),
                             neighbours=3)
    seeds = np.zeros((4, 2), dtype=np.float32)
    seeds[0, 0] = 1.0

    mass = propagate(graph, seeds)

    assert mass.shape == (4, 2)
    assert not mass.any()


def test_propagating_no_cells_at_all_returns_an_empty_answer():
    """An emptied pool is a normal end state for the sequential rounds."""
    from scipy import sparse

    mass = propagate(sparse.csr_matrix((0, 0)), np.zeros((0, 3)))

    assert mass.shape == (0, 3)


def test_a_well_sequencing_says_nothing_about_splits_evenly_between_guides():
    """With no fractions for a well, the well constraint must not prefer a guide.

    Sequencing can fail for a well while its images are fine. The constraint
    then has no counts to project onto, and the only defensible target is an
    equal share per guide -- the well's cells still get told apart by the
    graph, but the well as a whole cannot vote for one guide over the other.
    """
    mass = np.array([[3.0, 1.0], [4.0, 0.0], [0.5, 2.0], [1.0, 1.0]])
    wells = ["A01"] * 4

    posterior = constrain_to_fractions(mass, wells, ("g1", "g2"), {})

    assert np.allclose(posterior.sum(axis=1), 1.0)
    per_guide = posterior.sum(axis=0)
    assert per_guide[0] == pytest.approx(per_guide[1], abs=1e-3)


def test_a_well_whose_fractions_are_all_zero_splits_evenly_too():
    """Zero counts for every guide carry no information, same as no counts.

    A well can be sequenced and return nothing for the guides being tested.
    Treating a zero total as a real distribution would divide by it; treating
    it as "no counts" is what the missing-well case already does.
    """
    mass = np.array([[3.0, 1.0], [4.0, 0.0], [0.5, 2.0], [1.0, 1.0]])
    wells = ["A01"] * 4
    fractions = {"A01": {"g1": 0.0, "g2": 0.0}}

    posterior = constrain_to_fractions(mass, wells, ("g1", "g2"), fractions)

    per_guide = posterior.sum(axis=0)
    assert per_guide[0] == pytest.approx(per_guide[1], abs=1e-3)


def test_real_fractions_move_the_well_away_from_the_even_split():
    """The flat-prior fallback must not be what a well with counts also gets.

    Without this the two tests above would pass against a constraint that
    ignored ``fractions`` entirely.
    """
    mass = np.array([[3.0, 1.0], [4.0, 0.0], [0.5, 2.0], [1.0, 1.0]])
    wells = ["A01"] * 4
    fractions = {"A01": {"g1": 0.75, "g2": 0.25}}

    posterior = constrain_to_fractions(mass, wells, ("g1", "g2"), fractions)

    per_guide = posterior.sum(axis=0)
    assert per_guide[0] == pytest.approx(3.0, abs=1e-2)
    assert per_guide[1] == pytest.approx(1.0, abs=1e-2)


def test_a_missing_well_label_leaves_its_row_of_the_posterior_at_zero():
    """A row matching no well is not constrained, and must not be invented."""
    mass = np.array([[3.0, 1.0], [4.0, 0.0]])
    wells = np.array([np.nan, np.nan], dtype=float)

    posterior = constrain_to_fractions(mass, wells, ("g1", "g2"), {})

    assert not posterior.any()


def test_a_run_with_no_guides_returns_an_empty_result_that_says_why():
    """Nothing to assign is an answer, and the report has to carry the reason."""
    features, wells, scores = _cells(4)

    result = sudoku(features, scores, wells, {}, ())

    assert result.names == ()
    assert result.guides == ()
    assert result.affirm.shape == (8, 0)
    assert result.report["reason"] == "no cells or no guides"
    assert "relative to the median positive reach" in " ".join(
        (SudokuResult.__doc__ or "").split())


def test_a_run_with_no_cells_returns_an_empty_result_that_says_why():
    """The same for an empty cell table, which a hard filter can produce."""
    result = sudoku(np.zeros((0, 2)), np.zeros(0), [], {}, ("g1",))

    assert result.guides == ()
    assert result.report["reason"] == "no cells or no guides"


def test_a_single_measurement_per_cell_is_accepted_as_one_column():
    """A 1-D feature array is one feature, not one cell.

    Callers pass a single intensity or area column straight through. Read as a
    row vector it would become a one-cell screen, and the assignment would be
    computed over a graph of one node.
    """
    _, wells, scores = _cells(6)
    intensities = np.concatenate([np.full(6, 0.2), np.full(6, 4.0)])

    result = sudoku(intensities, scores, wells,
                    {"A01": {"g1": 1.0}, "A02": {"g2": 1.0}},
                    ("g1", "g2"), neighbours=3)

    assert len(result.guides) == 12
    assert result.affirm.shape == (12, 2)


def test_explicit_anchors_override_the_score_based_selection():
    """A caller who already knows its anchors must not have them re-chosen.

    Supplying them is how a run reuses a curated set; falling back to the
    score would silently swap in a different anchor population and change
    every downstream assignment.
    """
    features, wells, scores = _cells(6)

    result = sudoku(features, scores, wells,
                    {"A01": {"g1": 1.0}, "A02": {"g2": 1.0}},
                    ("g1", "g2"), anchors={"g1": [0, 1], "g2": [9, 10, 99]},
                    neighbours=3)

    assert result.report["anchors"] == {"g1": 2, "g2": 2}


def test_a_screen_where_no_well_reaches_the_threshold_says_nothing_was_annotated():
    """Every abstention together is a run-level failure and must be reported.

    A result of all-abstain with an empty report reads as "the cells were
    ambiguous". The real cause is that no well's sequencing gave any guide a
    large enough share to anchor on, which is a different problem with a
    different fix.
    """
    features, wells, scores = _cells(6)
    fractions = {"A01": {"g1": 0.1, "g2": 0.1}, "A02": {"g1": 0.1, "g2": 0.1}}

    result = sudoku(features, scores, wells, fractions, ("g1", "g2"),
                    neighbours=3)

    assert result.report["anchors"] == {"g1": 0, "g2": 0}
    assert "no guide reached the anchor threshold" in result.report["warning"]
    assert set(result.guides) == {ABSTAIN}


def test_a_sequential_run_stops_before_comparing_a_single_cell():
    """One unclaimed cell cannot be compared against anything.

    A posterior is a comparison between guides over a pool. Running a round on
    a pool of one normalises that cell over the guide columns and hands it to
    whichever guide is being processed, so the loop has to stop first.
    """
    result = sudoku_all(np.array([0.5]), np.array([0.9]), ["A01"],
                        {"A01": {"g1": 1.0}}, [("g1", 3.0), ("g2", 1.0)])

    assert result.guides == (ABSTAIN,)
    assert result.report["guides_considered"] == 0
    assert result.report["guides_offered"] == 2


def test_a_sequential_run_over_one_measurement_column_still_reads_it_as_a_feature():
    """``sudoku_all`` must reshape a 1-D feature array the same way ``sudoku`` does."""
    _, wells, scores = _cells(6)
    intensities = np.concatenate([np.full(6, 0.2), np.full(6, 4.0)])

    result = sudoku_all(intensities, scores, wells,
                        {"A01": {"g1": 1.0}, "A02": {"g2": 1.0}},
                        [("g1", 3.0), ("g2", 1.0)], neighbours=3)

    assert result.report["cells"] == 12
    assert len(result.guides) == 12
