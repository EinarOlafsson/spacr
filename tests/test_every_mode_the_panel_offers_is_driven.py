"""Every value of every mode dropdown, driven -- and a guard that keeps it so.

A mode is a setting with a FIXED set of values: the dropdowns the settings
panel builds out of ``_APP_COMBO_OPTIONS`` and ``FIXED_ALPHABETS``, plus the
two inventories those tables only stub and fill from the owning module at
widget-build time (``ml.REGRESSION_TYPES``,
``multiple_testing.method_choices``). Across the eleven apps that is 34
alphabets and 127 values, and an untested one is a control the user can pick
that nobody has ever selected.

Seven were untested when this file was written -- ``cov_type`` HC0/HC1/HC2,
``isomap_path_method`` FW, and ``pca_svd_solver`` arpack/covariance_eigh/
randomized. They are driven here, and each is checked for the thing that
makes the choice worth offering rather than merely for not raising:

* the four heteroskedasticity-robust covariances must give standard errors
  that differ from the classical one AND from each other, because a
  ``cov_type`` that quietly did nothing would look exactly like one that
  worked;
* the PCA solvers are four routes to ONE decomposition, so they must AGREE
  with ``full`` rather than merely return an array of the right shape; the
  same holds for Floyd-Warshall and Dijkstra in isomap, which compute the
  same shortest paths by different algorithms.

:func:`test_every_mode_value_is_named_somewhere_in_the_suite` is the part
that does not rot: it re-derives the alphabets from the panel on every run,
so a mode added to any dropdown fails here until somebody tests it.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# cov_type -- the robust covariance estimators
# ---------------------------------------------------------------------------

COV_TYPES = ("HC0", "HC1", "HC2", "HC3")


@pytest.fixture(scope="module")
def _ols_design():
    """A design with enough rows for the sandwich estimators to be stable."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    X.insert(0, "Intercept", 1.0)
    y = pd.Series(0.4 * X["a"] - 0.2 * X["b"] + rng.normal(0, 0.1, n))
    return X, y


def _bse(X, y, cov_type):
    from spacr.ml import regression_model
    model = regression_model(X, y, regression_type="ols", cov_type=cov_type)
    return np.asarray(model.bse, dtype=float)


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_a_robust_cov_type_changes_the_standard_errors(_ols_design, cov_type):
    """It has to move the number it exists to move."""
    X, y = _ols_design
    robust = _bse(X, y, cov_type)
    classical = _bse(X, y, None)
    assert robust.shape == classical.shape
    assert np.all(np.isfinite(robust))
    assert not np.allclose(robust, classical), (
        f"cov_type={cov_type!r} gave the classical standard errors, so the "
        f"setting did nothing")


def test_the_four_robust_cov_types_are_four_different_estimators(_ols_design):
    """HC0..HC3 differ only in a finite-sample correction, but they DO differ."""
    X, y = _ols_design
    seen = [tuple(np.round(_bse(X, y, c), 12)) for c in COV_TYPES]
    assert len(set(seen)) == len(COV_TYPES), (
        "two cov_type values produced identical standard errors; one of them "
        "is not reaching statsmodels")
    # HC0 < HC1 < HC2 < HC3 for the slope: each correction inflates the last.
    slopes = [_bse(X, y, c)[1] for c in COV_TYPES]
    assert slopes == sorted(slopes)


# ---------------------------------------------------------------------------
# the reduction solvers -- several routes to one answer
# ---------------------------------------------------------------------------

def _embed(method, options):
    from spacr.utils import reduction_and_clustering
    rng = np.random.default_rng(0)
    embedding = reduction_and_clustering(
        rng.random((60, 8)), n_neighbors=5, min_dist=0.1, metric="euclidean",
        clustering="kmeans", eps=0.5, min_samples=5, reduction_method=method,
        n_jobs=1, verbose=False, reducer_options=options, random_seed=42)
    if isinstance(embedding, tuple):
        embedding = embedding[0]
    return np.asarray(embedding, dtype=float)


@pytest.mark.parametrize("solver", ["auto", "full", "covariance_eigh",
                                    "arpack", "randomized"])
def test_every_pca_svd_solver_finds_the_same_components(solver):
    """Four routes to one decomposition: the answer may not depend on the route.

    Compared as absolute values because a principal component's SIGN is
    arbitrary -- two correct solvers may return v and -v.
    """
    reference = _embed("pca", {"whiten": False, "svd_solver": "full"})
    got = _embed("pca", {"whiten": False, "svd_solver": solver})
    assert got.shape == reference.shape == (60, 2)
    assert np.allclose(np.abs(got), np.abs(reference), atol=1e-8), (
        f"pca_svd_solver={solver!r} disagrees with 'full'")


@pytest.mark.parametrize("path_method", ["auto", "FW", "D"])
def test_every_isomap_path_method_finds_the_same_geodesics(path_method):
    """Floyd-Warshall and Dijkstra are two algorithms for one shortest path."""
    reference = _embed("isomap", {"n_neighbors": 5, "path_method": "FW"})
    got = _embed("isomap", {"n_neighbors": 5, "path_method": path_method})
    assert got.shape == reference.shape == (60, 2)
    assert np.allclose(np.abs(got), np.abs(reference), atol=1e-8), (
        f"isomap_path_method={path_method!r} disagrees with Floyd-Warshall")


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

def _alphabets():
    """Every fixed-value setting the panel offers, and its values.

    Re-derived from the panel rather than listed here, which is the whole
    point: a value added to a dropdown is in scope the moment it is added.
    """
    from spacr.qt.screens.settings_model import (FIXED_ALPHABETS,
                                                 _APP_COMBO_OPTIONS)

    def values(entries):
        return {str(e[0] if isinstance(e, tuple) else e) for e in entries}

    alphabets: dict[str, set[str]] = {}
    for options in _APP_COMBO_OPTIONS.values():
        for key, entries in options.items():
            alphabets.setdefault(key, set()).update(values(entries))
    for key, entries in FIXED_ALPHABETS.items():
        alphabets.setdefault(key, set()).update(values(entries))

    # The two the static table only stubs; the panel fills them in
    # ``_widget_for`` from the module that owns the inventory.
    from spacr.ml import REGRESSION_TYPES
    from spacr.multiple_testing import method_choices
    alphabets["regression_type"] = {"auto", *REGRESSION_TYPES}
    alphabets["multiple_testing_method"] = {str(m) for m in method_choices()}
    return alphabets


def _suite_text():
    here = pathlib.Path(__file__).parent
    chunks = []
    for base, _dirs, names in os.walk(here):
        for name in names:
            if name.endswith(".py"):
                chunks.append(pathlib.Path(base, name)
                              .read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def test_every_mode_value_is_named_somewhere_in_the_suite():
    """A dropdown value no test ever selects is a mode nobody has run.

    Deliberately a NAME check and not a coverage check: it is cheap enough
    to run on every commit, and it fails for the case that matters -- a mode
    added to the panel and to the dispatch and to nothing else. Being named
    in a test is the floor, not the goal; the drivers above are what
    actually exercise the seven that were missing.
    """
    blob = _suite_text()
    missing = {}
    for key, values in sorted(_alphabets().items()):
        absent = sorted(v for v in values
                        if v not in ("None", "")
                        and f"'{v}'" not in blob and f'"{v}"' not in blob)
        if absent:
            missing[key] = absent
    assert not missing, (
        "these mode values are offered by the settings panel and named by no "
        f"test: {missing}")


def test_the_guard_is_actually_looking_at_something():
    """A guard that found no alphabets would pass for the wrong reason."""
    alphabets = _alphabets()
    assert len(alphabets) > 25
    assert sum(len(v) for v in alphabets.values()) > 100
    assert "regression_type" in alphabets and len(alphabets["regression_type"]) > 10
