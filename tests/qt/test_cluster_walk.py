"""Walk over DBSCAN's eps: try the space, show the candidates.

`cluster_walk` and `cluster_walk_steps` sat in GateEditorSettings with ZERO
readers anywhere in spaCR -- editable, saved, reloaded and used by nothing.
Instruction 48 needs Walk to actually work before the Gate Editor lesson can
demonstrate it, so the fix was to implement it rather than delete a feature
that had been asked for.

These tests are on the pure functions, not the dialog. That is the point of
splitting them out: a search whose only expression is a modal dialog cannot
be tested at all under the offscreen platform.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    ClusterError, best_cluster_candidate, cluster_gates,
    cluster_walk_candidates,
)


def two_blobs(separation=6.0, spread=0.30, n=220, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0], spread, (n, 2))
    b = rng.normal([separation, separation], spread, (n, 2))
    return pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])


def test_the_sweep_is_geometric_and_centred_on_the_given_eps():
    """eps is a DISTANCE, so its useful range spans orders of magnitude.

    An arithmetic sweep spends most of its steps in one region and reports
    the same clustering several times over.
    """
    cands = cluster_walk_candidates(two_blobs(), "x", "y", eps=0.5,
                                    min_samples=10, steps=7, span=3.0)
    radii = [c.eps for c in cands]

    assert len(radii) == 7
    assert radii[0] == pytest.approx(0.5 / 3.0)
    assert radii[-1] == pytest.approx(0.5 * 3.0)
    assert radii == sorted(radii), "candidates must come back ordered by eps"

    ratios = [b / a for a, b in zip(radii, radii[1:])]
    assert max(ratios) - min(ratios) < 1e-9, "steps are not geometric"


def test_a_radius_that_finds_nothing_is_reported_rather_than_dropped():
    """"Nothing below here works" is the most useful part of the answer.

    Dropping the empty rows would leave a list that looks like every setting
    worked, which is how a user picks the smallest eps and gets one blob.
    """
    # Blobs far apart and TIGHT, so a small enough radius finds no core point.
    cands = cluster_walk_candidates(two_blobs(spread=0.9), "x", "y",
                                    eps=0.05, min_samples=40,
                                    steps=6, span=2.0)
    assert any(c.clusters == 0 for c in cands)
    assert all(c.silhouette is None for c in cands if c.clusters < 2)


def test_the_walk_finds_the_two_blobs_that_are_really_there():
    cands = cluster_walk_candidates(two_blobs(), "x", "y", eps=0.5,
                                    min_samples=10, steps=9)
    best = best_cluster_candidate(cands)
    assert best is not None
    assert best.clusters == 2
    assert best.silhouette > 0.8


def test_best_refuses_a_radius_that_calls_most_of_the_plate_noise():
    """The half that matters, and the reason silhouette alone is not enough.

    Silhouette is maximised by keeping a few tight points and discarding
    everything else -- a near-perfect score for a result that answers no
    question. The noise ceiling is what rules it out.
    """
    from spacr.qt.widgets.gate_spec import ClusterCandidate

    greedy = ClusterCandidate(eps=0.05, clusters=2, noise_fraction=0.93,
                              silhouette=0.99)
    honest = ClusterCandidate(eps=0.50, clusters=2, noise_fraction=0.04,
                              silhouette=0.71)
    assert best_cluster_candidate([greedy, honest]) is honest


def test_best_is_none_when_nothing_is_defensible():
    """None, not a shrug. The caller has to be able to say so to the user."""
    from spacr.qt.widgets.gate_spec import ClusterCandidate

    only_noise = [ClusterCandidate(eps=e, clusters=0, noise_fraction=1.0,
                                   silhouette=None) for e in (0.1, 0.2, 0.4)]
    assert best_cluster_candidate(only_noise) is None


def test_ties_break_toward_the_larger_radius():
    """Merging is the conservative direction when two settings score alike."""
    from spacr.qt.widgets.gate_spec import ClusterCandidate

    small = ClusterCandidate(eps=0.2, clusters=2, noise_fraction=0.0,
                             silhouette=0.80)
    large = ClusterCandidate(eps=0.9, clusters=2, noise_fraction=0.0,
                             silhouette=0.80)
    assert best_cluster_candidate([small, large]) is large


def test_the_walk_and_the_run_prepare_the_data_identically():
    """A search that scaled differently would recommend a misleading number.

    eps means a different distance either side of standardisation, so the
    radius the walk endorses has to be the radius the real run then uses.
    """
    frame = two_blobs()
    cands = cluster_walk_candidates(frame, "x", "y", eps=0.5, min_samples=10,
                                    steps=9, scale=True)
    best = best_cluster_candidate(cands)

    gates = cluster_gates(frame, "x", "y", eps=best.eps, min_samples=10,
                          scale=True)
    assert len(gates) == best.clusters


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"steps": 1}, "at least 2 steps"),
    ({"span": 1.0}, "greater than 1"),
    ({"eps": 0.0}, "must be positive"),
    ({"min_samples": 1}, "at least 2"),
])
def test_a_walk_that_cannot_be_described_says_so(kwargs, message):
    params = {"eps": 0.5, "min_samples": 10, "steps": 6}
    params.update(kwargs)
    with pytest.raises(ClusterError, match=message):
        cluster_walk_candidates(two_blobs(), "x", "y", **params)


def test_a_flat_measurement_is_refused_by_the_walk_too():
    """Shared preparation means shared refusals -- that is why it is shared."""
    frame = two_blobs()
    frame["flat"] = 3.0
    with pytest.raises(ClusterError, match="same value for every object"):
        cluster_walk_candidates(frame, "x", "flat", eps=0.5, min_samples=10)
