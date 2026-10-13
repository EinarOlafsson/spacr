"""PCAResult readers no test had ever named.

Instruction 60. Eight public members of ``PCAResult`` had never appeared in a
test, and every one of them is either what a CSV export writes or what the
biplot draws -- so a wrong answer here is a published figure or a shared
table that nobody can reproduce.

They are read off a real decomposition rather than a hand-built result: a
PCAResult assembled in a test could carry loadings and correlations that no
input could produce, and then these readers would be checked against a shape
the code never sees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def result():
    """A decomposition of data with a known structure.

    Two correlated features and one independent one, so the first component
    has to carry most of the variance and the ordering assertions below mean
    something.
    """
    from spacr.qt.widgets.pca_model import pca

    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    frame = pd.DataFrame({
        "a": base + rng.normal(scale=0.05, size=200),
        "b": base * 2.0 + rng.normal(scale=0.05, size=200),
        "c": rng.normal(size=200),
        "d": rng.normal(size=200),
    })
    return pca(frame)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def test_one_name_per_component(result):
    assert len(result.component_names) == result.n_components


def test_the_names_round_trip_through_their_index(result):
    """The viewer stores an axis choice by NAME and looks it up by index; a
    pair that does not round trip puts the user on a different component
    from the one the menu says."""
    from spacr.qt.widgets.pca_model import component_index

    for index, name in enumerate(result.component_names):
        assert component_index(name) == index


def test_an_unknown_name_is_none_rather_than_zero():
    """Falling back to the first component would silently show PC1 wherever
    a stale saved choice was read."""
    from spacr.qt.widgets.pca_model import component_index

    assert component_index("not a component") is None


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------

def test_the_cumulative_ratio_only_goes_up(result):
    assert list(result.cumulative_ratio) == sorted(result.cumulative_ratio)


def test_the_cumulative_ratio_ends_at_the_retained_share(result):
    """The last cumulative entry IS the share the returned components carry;
    two numbers that disagree would put two different totals on one scree
    plot."""
    assert float(result.cumulative_ratio[-1]) == \
        pytest.approx(result.retained_ratio)


def test_no_component_carries_more_than_everything(result):
    assert 0.0 < result.retained_ratio <= 1.0 + 1e-9


def test_the_first_component_carries_the_most(result):
    """Not a property of PCA in general -- a property of THIS input, which
    was built with two features that move together."""
    ratios = list(result.explained_variance_ratio)
    assert ratios[0] == max(ratios)


def test_the_variance_frame_is_one_row_per_component(result):
    frame = result.variance_frame()
    assert len(frame) == result.n_components
    assert list(frame["component"]) == list(result.component_names)
    assert set(frame.columns) >= {"explained_variance",
                                  "explained_variance_ratio",
                                  "cumulative_ratio"}


# ---------------------------------------------------------------------------
# Loadings
# ---------------------------------------------------------------------------

def test_the_loadings_frame_is_one_row_per_feature(result):
    frame = result.loadings_frame()
    assert len(frame) == len(result.features)
    assert list(frame["feature"]) == list(result.features)


def test_every_component_gets_a_loading_and_a_correlation(result):
    """The pair is what the biplot draws: the loading is the direction and
    the correlation is how much of the feature is actually in this plane."""
    frame = result.loadings_frame()
    for name in result.component_names:
        assert f"{name}_loading" in frame.columns
        assert f"{name}_r" in frame.columns


def test_a_correlation_is_a_correlation(result):
    frame = result.loadings_frame()
    for name in result.component_names:
        values = frame[f"{name}_r"].to_numpy(dtype=float)
        assert np.all(np.abs(values) <= 1.0 + 1e-9), name


# ---------------------------------------------------------------------------
# Which arrows the biplot draws
# ---------------------------------------------------------------------------

def test_the_plane_returns_at_most_what_was_asked_for(result):
    assert len(result.plane_features(0, 1, count=2)) <= 2


def test_it_never_returns_more_features_than_there_are(result):
    assert len(result.plane_features(0, 1, count=999)) <= len(result.features)


def test_the_features_are_ranked_by_how_visible_they_are_here(result):
    """Ranked by r_x squared plus r_y squared. A short arrow means "this
    feature points somewhere you are not looking", not "this feature does
    not matter" -- so the longest are the honest ones to draw."""
    picked = result.plane_features(0, 1, count=3)
    strength = (result.correlations[:, 0] ** 2 +
                result.correlations[:, 1] ** 2)
    best = list(np.argsort(strength)[::-1][:3])
    assert list(picked) == best


def test_the_indices_are_into_the_feature_list(result):
    """Returning names would be a second spelling of the same thing and the
    caller indexes `features` with these."""
    for index in result.plane_features(0, 1, count=4):
        assert 0 <= int(index) < len(result.features)
