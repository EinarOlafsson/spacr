"""Edges of the PCA spec and of a result that is read rather than computed.

Three of these are about what a spec accepts from a file and from a caller
who did not tidy their feature list; one is about a result whose loadings
carry no sign at all, which the decomposition cannot produce but the
dataclass can be handed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.pca_model import (NAN_MEAN, SCALE_NONE, PCAResult,
                                        PCASpec, pca)


def measured(n: int = 40, seed: int = 0, columns: int = 3) -> pd.DataFrame:
    """Correlated continuous measurements, no NaN anywhere."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {f"cell_feature_{i}": base * (i + 1) + rng.normal(0, 0.5, n)
         for i in range(columns)})


# ---------------------------------------------------------------------------
# The spec's own tidying
# ---------------------------------------------------------------------------

def test_a_blank_feature_name_is_dropped_rather_than_asked_for():
    """A list built from a form or a saved file carries empty strings.

    Left in, an empty name reaches ``_select_features`` and is reported as
    "not a column of this table" -- a refusal note about a feature the user
    never chose.
    """
    spec = PCASpec(features=("area", "", "perimeter", None, "area"))

    assert spec.features == ("area", "perimeter")


def test_a_saved_spec_that_names_no_features_keeps_the_default_of_all():
    """``features`` is optional in the payload, and its absence means "every
    continuous column", not "no columns"."""
    spec = PCASpec.from_dict({"n_components": 4, "scaling": SCALE_NONE})

    assert spec.features == ()
    assert spec.n_components == 4
    assert spec.scaling == SCALE_NONE

    # And it is the same object a round trip through JSON produces.
    assert PCASpec.from_json(spec.to_json()) == spec


# ---------------------------------------------------------------------------
# Mean imputation with nothing to impute
# ---------------------------------------------------------------------------

def test_asking_for_mean_imputation_on_a_complete_table_fabricates_nothing():
    """The policy is honoured, but there are no holes, so no value is
    invented and the report says nothing about imputation."""
    frame = measured()

    result = pca(frame, PCASpec(nan_policy=NAN_MEAN))

    assert len(result) == len(frame)
    assert result.dropped_rows == 0
    assert not any("replaced by their" in note for note in result.notes)
    assert "replaced by their" not in result.report()


def test_mean_imputation_says_so_when_it_does_fill_a_hole():
    """The other side of the same decision: a filled cell is declared."""
    frame = measured()
    frame.loc[0, "cell_feature_1"] = np.nan

    result = pca(frame, PCASpec(nan_policy=NAN_MEAN))

    assert len(result) == len(frame)
    assert any("replaced by their" in note for note in result.notes)


# ---------------------------------------------------------------------------
# A component with no direction at all
# ---------------------------------------------------------------------------

def a_flat_result(loadings: np.ndarray) -> PCAResult:
    """A result carrying ``loadings`` and otherwise plausible arrays."""
    n_features, k = loadings.shape
    return PCAResult(
        features=tuple(f"f{i}" for i in range(n_features)),
        rows=np.arange(5),
        scores=np.zeros((5, k)),
        loadings=loadings,
        correlations=np.zeros((n_features, k)),
        explained_variance=np.ones(k),
        explained_variance_ratio=np.full(k, 1.0 / k),
        total_variance=float(k),
        centre=np.zeros(n_features),
        scale=np.ones(n_features),
        rank=k, n_rows_in=5, n_features_in=n_features)


def test_a_component_with_no_loading_at_all_is_reported_as_one_sided():
    """``sign_agreement`` divides by the component's squared loading.

    A column of zeros is not something the decomposition returns -- its
    columns are unit-norm -- but a ``PCAResult`` can be built from stored
    arrays, and a division by zero here would take the headline of every
    component down with it.
    """
    result = a_flat_result(np.zeros((3, 2)))

    assert result.sign_agreement(0) == 1.0
    assert result.sign_agreement(1) == 1.0
    # And every reader built on top of it still answers.
    assert result.dominant(0) == ("f0", 0.0)
    assert "PC1" in result.headline(0)
    assert result.report().startswith("PCA of 5 objects")


def test_a_component_whose_loadings_are_mostly_negative_is_read_from_that_side():
    """The agreement is the larger sign's share, whichever sign that is.

    Nothing distinguishes the two ends of a component -- the sign of a
    loading vector is arbitrary, and a decomposition may hand back either.
    Reading only the positive share would call this axis 2% one-sided when
    98% of its squared loading sits on one side.
    """
    loadings = np.array([[-0.6], [-0.5], [0.1]])

    result = a_flat_result(loadings)

    assert result.sign_agreement(0) == pytest.approx(0.61 / 0.62)
    assert result.sign_agreement(0) >= 0.5


def test_a_one_sided_component_is_read_as_a_size_axis():
    """Every feature loading the same way is the classic magnitude axis, and
    the agreement is 1.0 because none of the squared loading is on the other
    sign."""
    loadings = np.array([[0.6, 0.7], [0.5, -0.7], [0.62, 0.14]])

    result = a_flat_result(loadings)

    assert result.sign_agreement(0) == pytest.approx(1.0)
    assert result.sign_agreement(1) < 1.0
    assert result.is_size_like(0) is True
