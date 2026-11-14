"""PCA says what it decomposed, or refuses -- it never quietly changes it.

A component is read as a statement about the objects in the table. Every
refusal here is a place where running anyway would produce a picture of a
different set of objects, or of a different set of features, than the one the
user selected -- and nothing on the axes would say so.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.pca_model import (NAN_COMPLETE, SCALE_NONE,
                                        SCALE_ZSCORE, PCASpec,
                                        component_index)


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
