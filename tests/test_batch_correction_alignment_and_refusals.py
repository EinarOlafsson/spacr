"""Table-driven tests of ``spacr.batch_correction`` on the inputs a tidy frame never produces.

Every case decides whether a plate effect is removed, refused, or removed
together with the biology: control values that arrived as bytes, labels,
controls, covariates or metadata in another row order, a covariate column that
never varies, a constant feature inside one batch, collinear features that
collapse the ComBat prior, and the TVN / Harmony input checks.  Corrected
frames are compared exactly (to 1e-6) against the values the current
estimator produces, and every refusal by its full message.  Harmony runs
through a stand-in ``harmonypy`` that hands the PCA scores back unchanged.
"""
from __future__ import annotations

import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import batch_correction as bc
from spacr.batch_correction import (NO_COVARIATE, correct_batch_effects,
                                    correct_from_metadata, spherize_features)


def _features(n=6):
    return pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0][:n],
         "f2": [4.0, 9.0, 6.0, 14.0, 15.0, 26.0][:n]})


def _batch(n=6):
    return pd.Series(["A", "A", "A", "B", "B", "B"][:n])


def _frame(n=30, k=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, k)),
                        columns=[f"f{i}" for i in range(k)])


CONTROLS = ["neg", "neg", "trt", "neg", "neg", "trt"]
CONDITION = ["x", "y", "x", "y", "x", "y"]
CENTERED = {"f1": [6.0, 7.0, 8.0, 6.0, 7.0, 8.0],
            "f2": [10.0, 15.0, 12.0, 8.0, 9.0, 20.0]}
COMBAT_CONDITION = {
    "f1": [5.888553, 6.945935, 8.003316, 6.364183, 7.077618, 7.791053],
    "f2": [8.995862, 14.248003, 12.004423, 10.262691, 9.507745, 18.433717]}
COMBAT_NO_COVARIATE = {
    "f1": [5.92551, 6.944364, 7.963218, 6.348935, 7.075877, 7.80282],
    "f2": [9.725575, 15.819609, 12.163188, 8.887707, 9.592236, 17.342053]}
COLLAPSED_PRIOR = (
    "2 batch(es) gave the same scale estimate for every feature -- a single "
    "feature, or features that are linear copies of one another -- so the "
    "empirical-Bayes prior carried no information and those batches were "
    "shrunk fully to the pooled scale.")
WHICH_BIOLOGY = (
    "batch_correction='combat' needs to know which biology to keep. ComBat "
    "estimates the plate effect from whatever the design does not explain, "
    "so a contrast that is not in the design is removed as if it were noise "
    "-- and the run looks cleaner for it. Pass covariate= the "
    "condition/treatment column, or "
    "covariate=spacr.batch_correction.NO_COVARIATE to state that there is "
    "nothing to preserve.")


def _check(corrected, report, expected, *, batches=("A", "B"), controls=0,
           covariates=(), warnings=()):
    """The corrected frame and every report field match exactly."""
    assert list(corrected.index) == list(range(len(corrected)))
    np.testing.assert_allclose(corrected[["f1", "f2"]].to_numpy(),
                               np.column_stack([expected["f1"],
                                                expected["f2"]]),
                               atol=1e-6)
    assert report.batches == list(batches)
    assert report.controls == controls
    assert report.covariate_columns == list(covariates)
    assert report.warnings == list(warnings)
    assert report.features == ["f1", "f2"]


def test_correct_batch_effects_realigns_by_index_and_records_what_it_did(
        monkeypatch, subtests):
    """Rows pair by index, not position, and each correction gives its exact table."""
    # (case, method, batch, kwargs, expected frame, report fields)
    cases = [
        ("bytes_control_value_is_decoded", "control_center", _batch(),
         dict(control=pd.Series(CONTROLS), control_values=b"neg",
              min_samples=2),
         {"f1": [6.0, 7.0, 8.0, 6.0, 7.0, 8.0],
          "f2": [9.0, 14.0, 11.0, 11.0, 12.0, 23.0]}, dict(controls=4)),
        # The same flags in reversed index order mark other rows, so the
        # f2 column moves to a different answer than positional zipping gives.
        ("control_column_in_another_order", "control_center", _batch(),
         dict(control=pd.Series(CONTROLS, index=[5, 4, 3, 2, 1, 0]),
              control_values="neg", min_samples=2),
         {"f1": [6.0, 7.0, 8.0, 6.0, 7.0, 8.0],
          "f2": [8.5, 13.5, 10.5, 5.5, 6.5, 17.5]}, dict(controls=4)),
        ("batch_series_in_another_order", "center",
         pd.Series(["B", "B", "B", "A", "A", "A"], index=[3, 4, 5, 0, 1, 2]),
         dict(min_samples=3), CENTERED, {}),
        ("constant_covariate_column_is_kept_but_adds_no_term", "combat",
         _batch(),
         dict(covariate=pd.DataFrame({"dose": [1.0] * 6,
                                      "condition": CONDITION}),
              min_samples=3),
         COMBAT_CONDITION, dict(covariates=("dose", "condition"))),
        ("covariate_series_in_another_order", "combat", _batch(),
         dict(covariate=pd.Series(CONDITION, index=[5, 4, 3, 2, 1, 0]),
              min_samples=3),
         COMBAT_CONDITION, dict(covariates=("covariate",))),
        ("constant_feature_in_one_batch_borrows_the_global_scale", "zscore",
         _batch(), dict(min_samples=3),
         {"f1": [0.795163, 7.0, 13.204837, 0.795163, 7.0, 13.204837],
          "f2": [3.5, 3.5, 3.5, 1.531498, 3.5, 5.468502]},
         dict(warnings=("A: 1 constant feature(s) used global scale.",))),
        ("collinear_features_never_come_back_nan", "combat", _batch(),
         dict(covariate=pd.DataFrame({"condition": CONDITION}),
              min_samples=3),
         {"f1": [6.183503, 7.0, 7.816497, 6.183503, 7.0, 7.816497],
          "f2": [9.183503, 10.0, 10.816497, 9.183503, 10.0, 10.816497]},
         dict(covariates=("condition",), warnings=(COLLAPSED_PRIOR,))),
    ]
    for case_id, method, batch, kwargs, expected, fields in cases:
        with subtests.test(msg=case_id):
            features = _features()
            if case_id.startswith("constant_feature"):
                features["f2"] = [5.0, 5.0, 5.0, 1.0, 2.0, 3.0]
            if case_id.startswith("collinear"):
                features["f2"] = [4.0, 5.0, 6.0, 14.0, 15.0, 16.0]
            corrected, report = correct_batch_effects(
                features, batch, method=method, **kwargs)
            _check(corrected, report, expected, **fields)

    with subtests.test(msg="constant_covariate_design_has_one_term"):
        design, terms, sources = bc._covariate_design(
            pd.DataFrame({"dose": [1.0] * 6, "condition": CONDITION}))
        np.testing.assert_array_equal(design, [[0.0], [1.0]] * 3)
        assert terms == ["condition=y"]
        assert sources == ["dose", "condition"]

    with subtests.test(msg="harmony_returns_the_embedding_and_says_so"):
        _fake_harmonypy(monkeypatch)
        out, report = correct_batch_effects(
            _frame(n=30, k=6), pd.Series(["a", "b", "c"] * 10),
            method="harmony", harmony_n_components=2)
        assert list(out.columns) == ["harmony_1", "harmony_2"]
        assert report.features == ["harmony_1", "harmony_2"]
        assert [w for w in report.warnings
                if "2-component corrected embedding" in w] == [
            report.warnings[0]]

    with subtests.test(msg="tvn_realigns_a_control_series"):
        feats = _frame(n=40)
        feats.index = [f"w{i}" for i in range(40)]
        batch = pd.Series(["a", "b"] * 20, index=feats.index)
        control = pd.Series(["DMSO"] * 40, index=list(feats.index[::-1]))
        out, report = correct_batch_effects(
            feats, batch, method="tvn", control=control,
            control_values="DMSO")
        aligned, _ = correct_batch_effects(
            feats, batch, method="tvn", control=control.reindex(feats.index),
            control_values="DMSO")
        assert report.controls == 40
        assert list(out.index) == list(feats.index)
        pd.testing.assert_frame_equal(out, aligned)


def test_correct_batch_effects_refuses_what_it_cannot_estimate(subtests):
    """Each impossible request is a ValueError carrying its full sentence."""
    tvn_features = _frame(n=40)
    # (case, call, exact message)
    cases = [
        ("empty_feature_frame",
         lambda: correct_batch_effects(pd.DataFrame(), _batch(),
                                       method="center"),
         "features must be a non-empty pandas DataFrame."),
        ("batch_below_min_samples",
         lambda: correct_batch_effects(_features(), pd.Series(list("AAAAAB")),
                                       method="zscore", min_samples=3),
         "plateID batch(es) have fewer than min_samples=3: {'B': 1}."),
        ("control_center_without_controls",
         lambda: correct_batch_effects(_features(), _batch(),
                                       method="control_center",
                                       control=None, control_values=None),
         "control_center requires batch_control_column and at least one "
         "batch_control_value (normally the negative control)."),
        # Values that cannot be listed drop the "The column holds" hint
        # instead of failing inside the error path.
        ("unhashable_control_column",
         lambda: correct_batch_effects(
             _features(), _batch(), method="control_center",
             control=pd.Series([[1], [2], [3], [4], [5], [6]]),
             control_values="neg", min_samples=1),
         "Only 0 total reference-control row(s) matched ['neg']; "
         "need at least 1."),
        ("unknown_tvn_scope",
         lambda: correct_batch_effects(
             tvn_features, pd.Series(["a", "b"] * 20), method="tvn",
             control=pd.Series(["DMSO"] * 40), control_values="DMSO",
             tvn_scope="galaxy"),
         f"Unknown tvn scope 'galaxy'. Choose one of {bc.TVN_SCOPES}."),
    ]
    for case_id, call, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                call()
            assert str(excinfo.value) == message


def test_correct_from_metadata_reads_named_columns_off_aligned_metadata(
        subtests):
    """Metadata realigns by index; columns, lists or data name the covariate."""
    plates = pd.DataFrame({"plateID": list(_batch())})
    with_condition = plates.assign(condition=CONDITION)
    # (case, metadata, kwargs, expected frame, report fields)
    cases = [
        ("metadata_in_another_order",
         pd.DataFrame({"plateID": ["B", "B", "B", "A", "A", "A"]},
                      index=[3, 4, 5, 0, 1, 2]),
         dict(batch_correction="center", batch_min_samples=3), CENTERED, {}),
        ("covariate_handed_over_as_data", plates,
         dict(batch_correction="combat",
              batch_covariate_column=pd.Series(CONDITION, name="condition"),
              batch_min_samples=3),
         COMBAT_CONDITION, dict(covariates=("condition",))),
        ("covariate_named_by_a_list", with_condition,
         dict(batch_correction="combat", batch_covariate_column=["condition"],
              batch_min_samples=3),
         COMBAT_CONDITION, dict(covariates=("condition",))),
        ("covariate_named_by_a_string", with_condition,
         dict(batch_correction="combat", batch_covariate_column="condition",
              batch_min_samples=3),
         COMBAT_CONDITION, dict(covariates=("condition",))),
        ("no_covariate_declared", plates,
         dict(batch_correction="combat", batch_covariate_column=NO_COVARIATE,
              batch_min_samples=3),
         COMBAT_NO_COVARIATE, {}),
    ]
    for case_id, metadata, kwargs, expected, fields in cases:
        with subtests.test(msg=case_id):
            corrected, report = correct_from_metadata(_features(), metadata,
                                                      **kwargs)
            _check(corrected, report, expected, **fields)

    assert bc._resolve_covariate(plates, []) is None
    # (case, features, metadata, kwargs, exact message)
    refusals = [
        ("metadata_not_a_frame", _features(), ["plateID"],
         dict(batch_correction="center"),
         "metadata must be a pandas DataFrame."),
        ("duplicated_metadata_index", _features(3),
         pd.DataFrame({"plateID": ["A", "A", "B"]}, index=[0, 0, 1]),
         dict(batch_correction="center"),
         "Batch-correction metadata cannot be aligned to feature rows; "
         "ensure their indexes are identical and unique."),
        ("named_control_column_absent", _features(), plates,
         dict(batch_correction="control_center",
              batch_control_column="condition", batch_control_values="neg"),
         "batch_control_column='condition' is absent from the input metadata."),
        ("empty_covariate_list_is_unanswered", _features(), plates,
         dict(batch_correction="combat", batch_covariate_column=[],
              batch_min_samples=3),
         WHICH_BIOLOGY),
    ]
    for case_id, features, metadata, kwargs, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                correct_from_metadata(features, metadata, **kwargs)
            assert str(excinfo.value) == message


def test_spherize_features_whitens_on_the_aligned_reference(subtests):
    """The reference realigns by index, every row is the default, bad input is refused."""
    with subtests.test(msg="no_reference_whitens_every_row"):
        feats = _frame(n=25)
        out, info = spherize_features(feats, method="ZCA")
        assert info["n_reference"] == 25
        assert list(out.columns) == list(feats.columns)
        np.testing.assert_allclose(np.cov(out.to_numpy(), rowvar=False),
                                   np.eye(5), atol=1e-3)

    with subtests.test(msg="reference_in_another_order_is_realigned"):
        feats = _frame(n=20)
        feats.index = [f"w{i}" for i in range(20)]
        flags = [True] * 10 + [False] * 10
        reversed_reference = pd.Series(flags, index=list(feats.index[::-1]))
        out, info = spherize_features(feats, reversed_reference, method="ZCA")
        aligned, _ = spherize_features(
            feats, reversed_reference.reindex(feats.index), method="ZCA")
        positional, _ = spherize_features(
            feats, pd.Series(flags, index=feats.index), method="ZCA")
        assert info["n_reference"] == 10
        pd.testing.assert_frame_equal(out, aligned)
        assert not np.allclose(out.to_numpy(), positional.to_numpy())

    holed = _frame()
    holed.iloc[0, 0] = np.nan
    holed.iloc[1, 1] = np.inf
    # (case, features, kwargs, exact message or pattern)
    cases = [
        ("unknown_method", _frame(), dict(method="nonsense"),
         f"Unknown tvn whitening 'nonsense'. Choose one of {bc.TVN_WHITENINGS}."),
        ("not_a_frame", _frame().to_numpy(), {},
         "features must be a non-empty pandas DataFrame."),
        ("empty_frame", pd.DataFrame(), {},
         "features must be a non-empty pandas DataFrame."),
        ("missing_values_are_counted", holed, {},
         re.compile(r"missing or infinite values.*\(2 found\)\.$")),
    ]
    for case_id, features, kwargs, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                spherize_features(features, **kwargs)
            if isinstance(message, str):
                assert str(excinfo.value) == message
            else:
                assert message.search(str(excinfo.value))


def _fake_harmonypy(monkeypatch, transposed=True, version="9.9"):
    """Install a ``harmonypy`` that returns the PCA scores unchanged."""
    calls = {}

    class Result:
        def __init__(self, z):
            self.Z_corr = z.T if transposed else z

    def run_harmony(scores, meta, vars_use, **kwargs):
        calls.update(scores=np.asarray(scores), meta=meta, vars_use=vars_use,
                     kwargs=kwargs)
        return Result(np.asarray(scores))

    module = types.ModuleType("harmonypy")
    module.run_harmony = run_harmony
    if version is not None:
        module.__version__ = version
    monkeypatch.setitem(sys.modules, "harmonypy", module)
    return calls


def test_harmony_embedding_returns_the_corrected_scores_and_its_info(
        monkeypatch, subtests):
    """Columns, shape, info and the Harmony call are exact; bad input is refused."""
    # (case, n, k, batch labels, kwargs, transposed, version, keep, theta)
    cases = [
        ("three_components_transposed", 30, 6, ["a", "b", "c"] * 10,
         dict(n_components=3, theta=1.5, seed=4), True, "9.9", 3, 1.5),
        ("capped_by_rank_untransposed_no_version", 20, 4, ["a", "b"] * 10,
         dict(n_components=50, standardize=False), False, None, 4, 2.0),
    ]
    for case_id, n, k, labels, kwargs, transposed, version, keep, theta in cases:
        with subtests.test(msg=case_id):
            calls = _fake_harmonypy(monkeypatch, transposed, version)
            feats = _frame(n=n, k=k)
            out, info = bc.harmony_embedding(feats, pd.Series(labels),
                                             **kwargs)
            assert list(out.columns) == [f"harmony_{i + 1}"
                                         for i in range(keep)]
            assert out.shape == (n, keep)
            np.testing.assert_array_equal(out.to_numpy(), calls["scores"])
            assert info["n_components"] == keep
            assert info["theta"] == theta
            assert info["harmonypy_version"] == (version or "")
            assert calls["vars_use"] == ["batch"]
            assert calls["meta"]["batch"].tolist() == labels
            assert calls["kwargs"]["random_state"] == kwargs.get("seed", 0)
            assert calls["kwargs"]["max_iter_harmony"] == 10
            # The fake returns the scores, so the explained share is theirs.
            centred = feats.to_numpy() - feats.to_numpy().mean(axis=0)
            if kwargs.get("standardize", True):
                centred = centred / centred.std(axis=0)
            assert info["explained_variance_ratio"] == pytest.approx(
                float((out.to_numpy() ** 2).sum() / (centred ** 2).sum()))
            if keep == k:
                assert info["explained_variance_ratio"] == pytest.approx(1.0)
            else:
                assert info["explained_variance_ratio"] < 1.0

    calls = _fake_harmonypy(monkeypatch)
    feats = _frame(n=20, k=4)
    feats.index = [f"w{i}" for i in range(20)]
    with subtests.test(msg="batch_in_another_order_is_realigned"):
        labels = ["a"] * 5 + ["b"] * 15
        batch = pd.Series(labels, index=list(feats.index[::-1]))
        out, _ = bc.harmony_embedding(feats, batch, n_components=2)
        assert list(out.index) == list(feats.index)
        assert calls["meta"]["batch"].tolist() == labels[::-1]
    holed = feats.copy()
    holed.iloc[0, 0] = np.nan
    # (case, features, batch, exact message)
    refusals = [
        ("not_a_frame", feats.to_numpy(), pd.Series(["a", "b"] * 10),
         "features must be a non-empty pandas DataFrame."),
        ("missing_values", holed, pd.Series(["a", "b"] * 10, index=feats.index),
         "harmony cannot embed missing or infinite values; impute or drop "
         "them first."),
        ("one_batch", feats, pd.Series(["x"] * 20, index=feats.index),
         "harmony needs at least two batches."),
    ]
    for case_id, features, batch, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                bc.harmony_embedding(features, batch)
            assert str(excinfo.value) == message
