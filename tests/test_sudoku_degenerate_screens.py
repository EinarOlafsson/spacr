"""Degenerate screens make the sudoku assignment abstain rather than invent.

The sudoku assignment turns sequencing fractions plus cell morphology into a
guide name per cell. Every case here is one where the evidence runs out -- a
well whose label is missing, a well whose cells carry no score, a well
sequencing says nothing about, a run with no guides, a run with one cell --
and must come back as an abstention or an empty result, because a fabricated
per-cell label is indistinguishable from a real one downstream. The four
units are :func:`~spacr.sudoku.anchors_for`,
:func:`~spacr.sudoku.constrain_to_fractions`, :func:`~spacr.sudoku.sudoku`
and :func:`~spacr.sudoku.sudoku_all`. All inputs are seeded, so the calls and
reports are pinned exactly.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spacr import sudoku as sudoku_module
from spacr.sudoku import (ABSTAIN, SudokuResult, anchors_for,
                          constrain_to_fractions, sudoku, sudoku_all)

_A = ABSTAIN
_BASE_REPORT = {"decision": 0.55, "reach_floor": 0.15, "score_in_graph": False}
_NO_ANCHORS = ("no guide reached the anchor threshold: no well gives any of "
               "them a large enough share, so nothing was annotated")


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


def _one_column(n_per_well: int = 6):
    """A single intensity per cell: low in A01, high in A02."""
    return np.concatenate([np.full(n_per_well, 0.2), np.full(n_per_well, 4.0)])


def test_anchors_for_picks_high_scoring_cells_of_high_fraction_wells(subtests):
    """Scoreless wells, unlabelled cells and the per-well cap give no flood."""
    cases = [
        ("a well with no score anywhere contributes nothing",
         ["A01", "A01", "A02", "A02"], np.array([np.nan, np.nan, 0.2, 0.9]),
         {"A01": {"g1": 1.0}, "A02": {"g1": 1.0}}, {"quantile": 0.5}, [3]),
        ("a NaN well label anchors nothing",
         np.array([np.nan, np.nan, np.nan], dtype=float),
         np.array([0.1, 0.5, 0.9]), {}, {"min_fraction": 0.0}, []),
        ("one well is capped at max_per_well, highest first",
         ["A01"] * 6, np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7]),
         {"A01": {"g1": 1.0}}, {"quantile": 0.0, "max_per_well": 2}, [1, 3]),
        ("a well below min_fraction is skipped",
         ["A01", "A01", "A02", "A02"], np.array([0.9, 0.8, 0.2, 0.3]),
         {"A01": {"g1": 0.4}, "A02": {"g1": 0.6}}, {"quantile": 0.0}, [2, 3]),
    ]
    for case_id, wells, scores, fractions, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            picked = anchors_for("g1", wells, fractions, scores, **kwargs)
            assert picked.dtype == int
            assert picked.tolist() == expected


def test_constrain_to_fractions_projects_onto_the_sequenced_counts(subtests):
    """No or zero counts split evenly; real counts move the well; NaN wells stay 0."""
    mass = np.array([[3.0, 1.0], [4.0, 0.0], [0.5, 2.0], [1.0, 1.0]])
    even = np.array([[0.580696781801, 0.419303218199],
                     [1.0, 0.0],
                     [0.103467939885, 0.896532060115],
                     [0.315835281714, 0.684164718286]])
    three_to_one = np.array([[0.885892080626, 0.114107919374],
                             [1.0, 0.0],
                             [0.392824203134, 0.607175796866],
                             [0.721283713072, 0.278716286928]])
    wells = ["A01"] * 4
    cases = [
        ("no fractions for the well split evenly", mass, wells, {}, even),
        ("all-zero fractions split evenly too", mass, wells,
         {"A01": {"g1": 0.0, "g2": 0.0}}, even),
        ("fractions for unknown guides split evenly", mass, wells,
         {"A01": {"g9": 1.0}}, even),
        ("a non-finite count is no count", mass, wells,
         {"A01": {"g1": float("nan")}}, even),
        ("real fractions move the well", mass, wells,
         {"A01": {"g1": 0.75, "g2": 0.25}}, three_to_one),
        ("a missing well label leaves its rows at zero", mass[:2],
         np.array([np.nan, np.nan], dtype=float), {}, np.zeros((2, 2))),
        ("a row with no mass takes the well's share", np.zeros((2, 2)),
         ["A01", "A01"], {"A01": {"g1": 0.75, "g2": 0.25}},
         np.array([[0.75, 0.25], [0.75, 0.25]])),
    ]
    for case_id, values, labels, fractions, expected in cases:
        with subtests.test(msg=case_id):
            posterior = constrain_to_fractions(values, labels, ("g1", "g2"),
                                               fractions)
            np.testing.assert_allclose(posterior, expected, atol=1e-9)


def test_sudoku_abstains_when_the_evidence_runs_out(subtests):
    """Exact calls, names, shapes and reports for each degenerate screen."""
    features, wells, scores = _cells(6)
    split = {"A01": {"g1": 1.0}, "A02": {"g2": 1.0}}

    with subtests.test(msg="no guides returns an empty result with a reason"):
        few, few_wells, few_scores = _cells(4)
        result = sudoku(few, few_scores, few_wells, {}, ())
        assert (result.guides, result.names) == ((), ())
        assert result.affirm.shape == (8, 0)
        assert result.reach.tolist() == [0.0] * 8
        assert result.report == {"reason": "no cells or no guides"}
        assert "relative to the median positive reach" in " ".join(
            (SudokuResult.__doc__ or "").split())

    with subtests.test(msg="no cells returns an empty result with a reason"):
        result = sudoku(np.zeros((0, 2)), np.zeros(0), [], {}, ("g1",))
        assert (result.guides, result.names) == ((), ("g1",))
        assert result.affirm.shape == (0, 1)
        assert result.report == {"reason": "no cells or no guides"}

    cases = [
        ("one measurement per cell is one column", _one_column(), split,
         ("g1", "g2"), {},
         (_A,) * 5 + ("g1",) + (_A,) * 5 + ("g2",),
         {"cells": 12, "guides": 2, "anchors": {"g1": 1, "g2": 1},
          "edges": 6, "isolated": 6, "abstained": 10,
          "abstained_for_reach": 10}),
        ("explicit anchors override the score, out-of-range dropped",
         features, split, ("g1", "g2"),
         {"anchors": {"g1": [0, 1], "g2": [9, 10, 99]}},
         ("g1",) * 4 + (_A,) + ("g1",) + ("g2",) * 6,
         {"cells": 12, "guides": 2, "anchors": {"g1": 2, "g2": 2},
          "edges": 12, "isolated": 3, "abstained": 1,
          "abstained_for_reach": 1}),
        ("no well reaches the threshold: nothing annotated", features,
         {"A01": {"g1": 0.1, "g2": 0.1}, "A02": {"g1": 0.1, "g2": 0.1}},
         ("g1", "g2"), {}, (_A,) * 12,
         {"cells": 12, "guides": 2, "anchors": {"g1": 0, "g2": 0},
          "edges": 12, "isolated": 3, "abstained": 12,
          "abstained_for_reach": 12, "warning": _NO_ANCHORS}),
        ("the score as a feature is flagged as circular", features, split,
         ("g1", "g2"), {"use_score_as_feature": True}, None,
         {"score_in_graph": True,
          "warning": ("the classification score is a graph feature AND "
                      "chooses the anchors, so affirmation is partly "
                      "circular")}),
    ]
    for case_id, values, fractions, guides, kwargs, called, report in cases:
        with subtests.test(msg=case_id):
            result = sudoku(values, scores, wells, fractions, guides,
                            neighbours=3, **kwargs)
            assert result.names == guides
            assert result.affirm.shape == (12, len(guides))
            if called is not None:
                assert result.guides == called
                assert result.report == {**_BASE_REPORT, **report}
            else:
                assert {k: result.report[k] for k in report} == report

    rng = np.random.default_rng(7)
    with subtests.test(msg="the names are exactly the guides given"):
        result = sudoku(rng.normal(size=(12, 2)), rng.uniform(size=12),
                        [f"w{i % 3}" for i in range(12)],
                        {f"w{i}": {"g1": 0.5, "g2": 0.5} for i in range(3)},
                        ("g1", "g2", "g3"))
        assert result.names == ("g1", "g2", "g3")
        assert result.guides == (_A,) * 12
        assert result.report["anchors"] == {"g1": 3, "g2": 3, "g3": 0}

    rng = np.random.default_rng(8)
    with subtests.test(msg="a guide with no well fraction keeps its column"):
        result = sudoku(rng.normal(size=(10, 2)), rng.uniform(size=10),
                        [f"w{i % 2}" for i in range(10)],
                        {f"w{i}": {"g1": 1.0} for i in range(2)},
                        ("g1", "unheard_of"))
        assert result.names == ("g1", "unheard_of")
        assert result.affirm.shape == (10, 2)
        assert result.guides == ("g1",) * 10
        assert result.report["anchors"] == {"g1": 2, "unheard_of": 0}


def test_sudoku_all_runs_ranked_rounds_until_nothing_is_left(subtests,
                                                             monkeypatch):
    """Ranked rounds, single-cell stops and per-round columns are exact."""
    _, wells, scores = _cells(6)

    with subtests.test(msg="a single cell is never compared"):
        result = sudoku_all(np.array([0.5]), np.array([0.9]), ["A01"],
                            {"A01": {"g1": 1.0}}, [("g1", 3.0), ("g2", 1.0)])
        assert result.guides == (_A,)
        assert result.names == ("g1", "g2")
        assert result.report == {
            "cells": 1, "guides_considered": 0, "guides_offered": 2,
            "claimed": 0, "abstained": 1, "claimed_by_round": [],
            "stopped_early": False, "decision": 0.55}

    with subtests.test(msg="one measurement column is read as a feature"):
        result = sudoku_all(_one_column(), scores, wells,
                            {"A01": {"g1": 1.0}, "A02": {"g2": 1.0}},
                            [("g2", 1.0), ("g1", 3.0)], neighbours=3)
        assert result.names == ("g1", "g2")
        assert result.guides == (_A,) * 5 + ("g1",) + (_A,) * 5 + ("g2",)
        assert result.report == {
            "cells": 12, "guides_considered": 2, "guides_offered": 2,
            "claimed": 2, "abstained": 10,
            "claimed_by_round": [
                {"guide": "g1", "confidence": 3.0, "claimed": 1, "left": 11},
                {"guide": "g2", "confidence": 1.0, "claimed": 1, "left": 10}],
            "stopped_early": False, "decision": 0.55}

    with subtests.test(msg="every ranked guide is a result column"):
        rng = np.random.default_rng(2)
        result = sudoku_all(rng.normal(size=(24, 3)), rng.uniform(0.0, 1.0, 24),
                            [f"w{i % 4}" for i in range(24)],
                            {f"w{i}": {"g1": 0.6, "g2": 0.4} for i in range(4)},
                            ranking=[("g1", 0.9), ("g2", 0.5)])
        assert result.names == ("g1", "g2")
        assert result.affirm.shape == (24, 2)
        assert result.guides == ("g1",) * 24
        assert result.report["claimed_by_round"] == [
            {"guide": "g1", "confidence": 0.9, "claimed": 24, "left": 0}]
        assert result.report["guides_considered"] == 1

    with subtests.test(msg="max_guides truncates the ranking"):
        result = sudoku_all(_one_column(), scores, wells,
                            {"A01": {"g1": 1.0}, "A02": {"g2": 1.0}},
                            [("g1", 3.0), ("g2", 1.0)], max_guides=1,
                            neighbours=3)
        assert result.names == ("g1",)
        assert result.report["guides_offered"] == 1

    with subtests.test(msg="each round reads its own named column"):
        calls = []

        def inner(features, scores, wells, fractions, guides, **_kwargs):
            calls.append(tuple(guides))
            n = len(features)
            guide = "g1" if len(calls) == 1 else "g2"
            evidence = np.tile(np.array([11.0, 22.0]), (n, 1))
            return SimpleNamespace(
                names=tuple(guides), guides=[guide, *([_A] * (n - 1))],
                affirm=evidence, eliminate=evidence + 1,
                posterior=evidence + 2, reach=np.zeros(n))

        monkeypatch.setattr(sudoku_module, "sudoku", inner)
        result = sudoku_module.sudoku_all(
            np.zeros((4, 1)), np.zeros(4), ["w"] * 4,
            {"w": {"g1": 0.5, "g2": 0.5}}, [("g1", 0.9), ("g2", 0.8)])
        assert calls == [("g1", "g2"), ("g1", "g2")]
        assert result.guides == ("g1", "g2", _A, _A)
        assert result.affirm[:, 0].tolist() == [11.0] * 4
        assert result.affirm[:, 1].tolist() == [0.0, 22.0, 22.0, 22.0]
        assert result.eliminate[:, 1].tolist() == [1.0, 23.0, 23.0, 23.0]
        assert result.posterior[:, 1].tolist() == [0.0, 24.0, 24.0, 24.0]

