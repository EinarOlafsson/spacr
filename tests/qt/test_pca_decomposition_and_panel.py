"""PCA says what it decomposed, or refuses -- and its panel idles safely.

A component is read as a statement about the objects and features the user
selected.  Every refusal below is a place where running anyway would picture a
different table than the one chosen; every result row pins the features,
components, caveats and notes the decomposition reports.  The panel cases are
the ordinary things done before a fit exists or while one is in flight.  One
general test per unit, each walking a case table with pytest's ``subtests``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.pca_model import (  # noqa: E402
    NAN_COMPLETE, NAN_DROP_FEATURES, PCAError, PCASpec, pca,
)

pytestmark = pytest.mark.qt


def _measured(n=40, seed=0, columns=3) -> pd.DataFrame:
    """Columns that share one latent factor, scaled 1x, 2x, 3x, plus noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return pd.DataFrame({f"cell_feature_{i}": base * (i + 1)
                         + rng.normal(0, 0.5, n) for i in range(columns)})


def _with(frame, **columns):
    frame = frame.copy()
    for name, value in columns.items():
        frame[name] = value
    return frame


def _nan_at(frame, rows, *columns):
    frame = frame.copy()
    for column in columns:
        frame.loc[rows, column] = np.nan
    return frame


FEATURES3 = ("cell_feature_0", "cell_feature_1", "cell_feature_2")


def test_pca_refuses_a_table_it_cannot_decompose_and_says_why(subtests):
    """Each refusal is a PCAError whose message names the way out."""
    cases = [
        ("one object", _measured(n=1), PCASpec(features=FEATURES3),
         "PCA needs at least two objects and this table has 1."),
        ("one complete object", _nan_at(_measured(n=6), slice(1, None),
                                        "cell_feature_2"),
         PCASpec(features=FEATURES3, nan_policy=NAN_COMPLETE),
         "only 1 object has a value for every feature; PCA needs at least "
         "two. Worst offenders: cell_feature_2 (83% missing), cell_feature_1 "
         "(0% missing), cell_feature_0 (0% missing)."),
        ("a NaN policy leaving one feature",
         _nan_at(_measured(), 0, "cell_feature_1", "cell_feature_2"),
         PCASpec(nan_policy=NAN_DROP_FEATURES),
         "the NaN policy 'drop_features' left 1 feature(s), which is not a "
         "PCA. The features missing most often are: cell_feature_2 (2% "
         "missing), cell_feature_1 (2% missing), cell_feature_0 (0% "
         "missing). Try nan_policy='complete' to keep the features and drop "
         "the objects instead."),
        ("one varying feature",
         _with(_measured(), cell_feature_1=5.0, cell_feature_2=7.0),
         PCASpec(features=FEATURES3),
         "only 1 feature varies over the analysed objects (cell_feature_0); "
         "PCA needs at least two."),
        ("every feature constant",
         pd.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10}),
         PCASpec(features=("a", "b"), n_components=1),
         "every selected feature is constant over the analysed objects, so "
         "there is no variance to decompose."),
    ]
    for case_id, frame, spec, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(PCAError) as caught:
                pca(frame, spec)
            assert str(caught.value) == message

    with subtests.test(msg="a component that was not computed"):
        result = pca(_measured(), PCASpec(n_components=2))
        with pytest.raises(PCAError) as caught:
            result.dominant(7)
        assert str(caught.value) == (
            "there is no component PC8; this result has 2 (PC1, PC2)")


def test_pca_reports_the_features_components_and_caveats_it_used(subtests):
    """Dropped, constant and collinear features are named; rank caps output."""
    ghosts = tuple(f"ghost_{i}" for i in range(8))
    cases = [
        # case, frame, spec, features used, n components, caveats
        ("the components asked for", _measured(), PCASpec(n_components=2),
         FEATURES3, 2, ()),
        ("never more than the data has", _measured(),
         PCASpec(features=FEATURES3[:2], n_components=10),
         FEATURES3[:2], 2, ()),
        ("unknown features are listed six at a time", _measured(),
         PCASpec(features=FEATURES3 + ghosts), FEATURES3, 3,
         ("8 feature(s) were not used: ghost_0, ghost_1, ghost_2, ghost_3, "
          "ghost_4, ghost_5 (+2 more).",)),
        ("a constant column is dropped by name",
         _with(_measured(), cell_constant=5.0),
         PCASpec(features=("cell_feature_0", "cell_constant",
                           "cell_feature_1"), n_components=2),
         ("cell_feature_0", "cell_feature_1"), 2,
         ("1 feature(s) were not used: cell_constant.",)),
        ("collinear columns are kept and reported",
         _with(_measured(), cell_copy=_measured()["cell_feature_0"]),
         PCASpec(features=("cell_feature_0", "cell_copy", "cell_feature_1"),
                 n_components=3),
         ("cell_feature_0", "cell_copy", "cell_feature_1"), 2,
         ("Perfectly correlated features: cell_feature_0 = cell_copy. They "
          "are kept, but each such group weighs once per member in a "
          "standardised PCA.",
          "The 3 features span only 2 independent directions, so there are 2 "
          "components and no more; the rest would be floating-point noise.")),
        ("two objects skip the collinearity scan", _measured(n=2),
         PCASpec(features=FEATURES3, n_components=2), FEATURES3, 1,
         ("The 3 features span only 1 independent directions, so there are 1 "
          "components and no more; the rest would be floating-point noise.",)),
    ]
    for case_id, frame, spec, features, components, caveats in cases:
        with subtests.test(msg=case_id):
            result = pca(frame, spec)
            assert tuple(result.features) == features
            assert result.n_components == components
            assert result.loadings.shape == (len(features), components)
            assert result.caveats() == caveats
            assert float(np.max(result.explained_variance)) > 0.0
            if case_id.startswith("unknown"):
                assert set(result.dropped_features) == set(ghosts)
            if case_id.startswith("collinear"):
                assert result.collinear_groups == (
                    ("cell_feature_0", "cell_copy"),)
            if case_id.startswith("two objects"):
                assert result.collinear_groups == ()

    with subtests.test(msg="past the feature ceiling the scan says it skipped"):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.normal(0, 1, (4, 1_600)),
                             columns=[f"cell_feature_{i}"
                                      for i in range(1_600)])
        result = pca(frame, PCASpec(features=tuple(frame.columns),
                                    n_components=2))
        assert result.collinear_groups == ()
        assert [note for note in result.notes
                if "collinearity scan was skipped" in note] != []

    with subtests.test(msg="a tall table agrees with the direct route"):
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
        assert tall.explained_variance_ratio[0] > tall.explained_variance_ratio[1]
        assert tall.explained_variance_ratio[0] == pytest.approx(
            short.explained_variance_ratio[0], abs=0.05)
        assert np.isfinite(tall.scores).all()


def test_the_pca_panel_is_safe_before_a_fit_and_on_close(qtbot, monkeypatch,
                                                         subtests):
    """A scree click or view switch with no fit is harmless; close stops work."""
    from spacr.qt.widgets.pca_view import PCAPanel

    def _panel():
        widget = PCAPanel()
        qtbot.addWidget(widget)
        return widget

    for case_id, act in [
            ("a scree click", lambda p: p.scree.component_picked.emit(2)),
            ("flipping the biplot switch",
             lambda p: p._biplot.setChecked(not p._biplot.isChecked()))]:
        with subtests.test(msg=f"{case_id} before a fit"):
            panel = _panel()
            act(panel)
            assert panel.result is None
            assert panel.report.text() == ""

    with subtests.test(msg="closing abandons the fit in flight"):
        panel = _panel()
        stopped = []
        monkeypatch.setattr(panel._jobs, "shutdown",
                            lambda *a, **k: stopped.append(True))
        panel.close()
        assert stopped == [True]
