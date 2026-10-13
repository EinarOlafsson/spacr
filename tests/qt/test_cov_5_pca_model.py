"""PCA says what it decomposed, or refuses -- it never quietly changes it.

A component is read as a statement about the objects in the table. Every
refusal here is a place where running anyway would produce a picture of a
different set of objects, or of a different set of features, than the one the
user selected -- and nothing on the axes would say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.pca_model import (NAN_COMPLETE, NAN_DROP_FEATURES,
                                        SCALE_NONE, SCALE_ZSCORE, PCAError,
                                        PCASpec, component_index, pca)


def _measured(n=40, seed=0, columns=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    data = {f"cell_feature_{i}": base * (i + 1) + rng.normal(0, 0.5, n)
            for i in range(columns)}
    return pd.DataFrame(data)


def test_a_name_that_is_not_a_component_has_no_index():
    """``PCx`` and ``PC`` resolve to None rather than raising.

    Component names are read back off saved figure settings and off menu
    entries. A name that is not one has to answer None so the caller falls
    back, instead of taking the axis selection down.
    """
    assert component_index("PC1") == 0
    assert component_index("PCx") is None
    assert component_index("PC") is None
    assert component_index("area") is None


def test_a_spec_is_edited_by_replacement_not_in_place():
    """Each ``with_`` edit returns a new spec and leaves the original alone.

    The spec is what a saved analysis records. A menu that mutated it would
    change the record of an analysis that had already been run and captioned.
    """
    spec = PCASpec(features=("a", "b"), n_components=3)

    assert spec.with_features(["x", "y", "z"]).features == ("x", "y", "z")
    assert spec.with_scaling(SCALE_NONE).scaling == SCALE_NONE
    assert spec.with_nan_policy(NAN_COMPLETE).nan_policy == NAN_COMPLETE
    assert spec.with_components(5).n_components == 5

    assert spec.features == ("a", "b")
    assert spec.n_components == 3
    assert spec.scaling == SCALE_ZSCORE


def test_a_spec_describes_its_features_its_scaling_and_its_policy():
    """The caption names all four settings, including "every column".

    The caption is what makes a PCA figure reproducible. An empty feature
    list means "every continuous column", which is a different analysis on a
    different table and has to be said rather than left blank.
    """
    every = PCASpec().describe()
    assert "every continuous column" in every
    assert "standardised" in every

    chosen = PCASpec(features=("a", "b", "c"), scaling=SCALE_NONE,
                     n_components=2).describe()
    assert "3 features" in chosen
    assert "centred only" in chosen
    assert "2 components" in chosen


def test_asking_for_a_component_that_was_not_computed_says_which_exist():
    """An out-of-range component raises and lists the ones there are.

    Component indices come from menus and from saved settings, and a saved
    figure of PC4 opened against a two-component result has to say so rather
    than draw PC1 under PC4's label.
    """
    result = pca(_measured(), PCASpec(n_components=2))

    with pytest.raises(PCAError) as excinfo:
        result.dominant(7)

    message = str(excinfo.value)
    assert "PC8" in message
    assert "PC1" in message and "PC2" in message


def test_features_that_are_not_columns_are_named_in_the_caveats():
    """Unknown feature names are dropped, counted and listed, six at a time.

    A saved spec is applied to another plate's table, which may not have
    every column. Running on what is left without saying so gives a PCA of
    two features under a caption that names nine.
    """
    frame = _measured(columns=3)
    ghosts = [f"ghost_{i}" for i in range(8)]
    result = pca(frame, PCASpec(features=tuple(frame.columns) + tuple(ghosts)))

    caveat = next(c for c in result.caveats() if "were not used" in c)
    assert "8 feature(s) were not used" in caveat
    assert "(+2 more)" in caveat
    assert "ghost_0" in caveat
    assert set(result.dropped_features) == set(ghosts)


def test_a_policy_that_leaves_one_feature_is_not_a_pca():
    """Dropping features for missingness until one is left raises, with names.

    One feature has one component, which is the feature. The message has to
    name the columns that were missing most often, or the user cannot tell
    which of their forty features caused it.
    """
    frame = _measured(columns=3)
    frame.loc[0, "cell_feature_1"] = np.nan
    frame.loc[0, "cell_feature_2"] = np.nan

    with pytest.raises(PCAError) as excinfo:
        pca(frame, PCASpec(nan_policy=NAN_DROP_FEATURES))

    message = str(excinfo.value)
    assert "left 1 feature(s)" in message
    assert "cell_feature_1" in message
    assert NAN_COMPLETE in message


def test_one_complete_object_is_not_enough_to_decompose():
    """Complete-case PCA with a single usable object raises, naming the worst.

    "PCA of one object" is not a picture of anything, and the blame list is
    what tells the user which feature to drop to get their objects back.
    """
    frame = _measured(n=6, columns=3)
    frame.loc[1:, "cell_feature_2"] = np.nan

    with pytest.raises(PCAError) as excinfo:
        pca(frame, PCASpec(features=tuple(frame.columns),
                           nan_policy=NAN_COMPLETE))

    message = str(excinfo.value)
    assert "only 1 object" in message
    assert "cell_feature_2" in message


def test_one_varying_feature_is_not_a_pca():
    """When all but one feature is constant the analysis is refused.

    A constant column has no variance to decompose and standardising it is a
    division by zero; with one left there is nothing to rotate, and a
    one-component "PCA" would just be that column rescaled.
    """
    frame = _measured(columns=3)
    frame["cell_feature_1"] = 5.0
    frame["cell_feature_2"] = 7.0

    with pytest.raises(PCAError, match="only 1 feature varies"):
        pca(frame, PCASpec(features=tuple(frame.columns)))


def test_a_table_of_one_object_is_refused_before_anything_is_computed():
    """A single row raises with the count it was given.

    One object has no covariance. Running would return a component of zeros
    that plots as a single point at the origin, which reads as a real result.
    """
    frame = _measured(n=1)

    with pytest.raises(PCAError, match="at least two objects"):
        pca(frame, PCASpec(features=tuple(frame.columns)))


def test_two_objects_are_decomposed_without_a_collinearity_scan():
    """With fewer than three objects no feature pair can be called collinear.

    A correlation over two points is exactly ±1 for every pair, so a scan
    would report every feature as a duplicate of every other one.
    """
    frame = _measured(n=2)
    result = pca(frame, PCASpec(features=tuple(frame.columns),
                                n_components=2))

    assert result.collinear_groups == ()
    assert not any("Perfectly correlated" in c for c in result.caveats())


def test_the_collinearity_scan_says_when_it_skipped_itself():
    """Past the feature ceiling the scan is skipped and the skip is reported.

    The scan is O(p²). Silently not running it would leave the caveats saying
    nothing about collinearity, which reads as "there is none".
    """
    n, p = 4, 1_600
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(0, 1, (n, p)),
        columns=[f"cell_feature_{i}" for i in range(p)])

    result = pca(frame, PCASpec(features=tuple(frame.columns),
                                n_components=2))

    assert result.collinear_groups == ()
    assert any("collinearity scan was skipped" in note
               for note in result.notes)


def test_a_tall_table_decomposes_through_the_gram_matrix():
    """Above the row limit the result is the same as the direct route.

    A million-object table cannot have its full design matrix decomposed, so
    the p×p Gram matrix is used instead. Nothing downstream is told which ran,
    so the two have to agree.
    """
    n = 20_001
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "cell_feature_0": base + rng.normal(0, 0.2, n),
        "cell_feature_1": 2 * base + rng.normal(0, 0.2, n),
        "cell_feature_2": rng.normal(0, 1, n),
    })

    tall = pca(frame, PCASpec(n_components=2))
    short = pca(frame.iloc[:1000], PCASpec(n_components=2))

    assert len(tall) == n
    assert tall.n_components == 2
    # Both routes put almost all of the variance on the correlated pair.
    assert (tall.explained_variance_ratio[0]
            > tall.explained_variance_ratio[1])
    assert tall.explained_variance_ratio[0] == pytest.approx(
        short.explained_variance_ratio[0], abs=0.05)
    assert np.isfinite(tall.scores).all()
