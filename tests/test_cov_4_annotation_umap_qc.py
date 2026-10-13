"""Control-based annotation QC reports why it could not answer.

Every early return here is a refusal a user has to see. Too few controls, a
recipe the embedding library rejected, or an embedding with no control cells
at all must each come back as a stated reason -- an exception would take down
the QC panel, and a silent zero would read as "the annotation agrees with
nothing", which is a finding rather than a missing measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import hyperparam
from spacr.annotation_umap_qc import (NEGATIVE, POSITIVE, fit_on_controls,
                                      neighbour_purity)


def test_too_few_control_cells_is_a_stated_refusal():
    """Under eight controls cannot be split into fit and hold-out halves."""
    features = np.arange(12, dtype=float).reshape(4, 3)
    result = fit_on_controls(features, [POSITIVE, NEGATIVE] * 2,
                             recipes=[{"n_neighbors": 5}])
    assert result == {"error": "too few control cells to split"}


def test_a_label_per_row_is_required_before_splitting():
    """Mismatched lengths mean the labels do not describe these cells."""
    features = np.arange(60, dtype=float).reshape(20, 3)
    result = fit_on_controls(features, [POSITIVE] * 19,
                             recipes=[{"n_neighbors": 5}])
    assert result == {"error": "too few control cells to split"}


def test_grouped_fit_requires_one_group_per_control_cell():
    """A shorter group vector cannot silently misalign the control rows."""
    features = np.arange(60, dtype=float).reshape(20, 3)
    labels = [POSITIVE] * 10 + [NEGATIVE] * 10

    result = fit_on_controls(
        features,
        labels,
        groups=[f"well-{index}" for index in range(19)],
        recipes=[{"n_neighbors": 5}],
    )

    assert result == {"error": "one group is needed per control cell"}


def test_a_grouped_split_refusal_keeps_the_requested_level():
    """Invalid split settings return an actionable error, not an exception."""
    features = np.arange(60, dtype=float).reshape(20, 3)
    labels = [POSITIVE] * 10 + [NEGATIVE] * 10

    result = fit_on_controls(
        features,
        labels,
        groups=[f"well-{index}" for index in range(20)],
        group_by="well",
        holdout=1.0,
        recipes=[{"n_neighbors": 5}],
    )

    assert result["split_level"] == "well"
    assert "strictly between 0 and 1" in result["error"]


def test_a_recipe_the_embedding_rejects_is_recorded_and_the_run_continues(
        monkeypatch):
    """A bad recipe names its own failure instead of ending the search."""
    def _explode(*_args, **_kwargs):
        raise ValueError("n_neighbors must be smaller than the sample size")

    monkeypatch.setattr(hyperparam, "_default_umap_embed", _explode)
    features = np.random.default_rng(0).normal(size=(20, 4))
    labels = [POSITIVE] * 10 + [NEGATIVE] * 10
    result = fit_on_controls(features, labels,
                             recipes=[{"n_neighbors": 500},
                                      {"n_neighbors": 900}])
    assert result["error"] == "no recipe produced a scorable embedding"
    assert len(result["trials"]) == 2
    assert all("ValueError" in str(t["error"]) for t in result["trials"])
    assert [t["recipe"] for t in result["trials"]] == [
        {"n_neighbors": 500}, {"n_neighbors": 900}]


def test_an_embedding_with_no_controls_scores_every_cell_as_unknown():
    """Purity with nothing to compare against is nan, never zero."""
    points = np.random.default_rng(1).normal(size=(6, 2))
    out = neighbour_purity(points, [None] * 6)
    assert out.shape == (6,)
    assert np.all(np.isnan(out))


def test_a_label_per_point_is_required_for_purity():
    """A shorter label list does not describe this embedding."""
    points = np.random.default_rng(2).normal(size=(6, 2))
    out = neighbour_purity(points, [POSITIVE, None])
    assert np.all(np.isnan(out))


def test_the_only_control_cell_gets_no_purity_of_its_own():
    """A control excluded from its own neighbourhood has nobody left."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = neighbour_purity(points, [POSITIVE, None, None])
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)
