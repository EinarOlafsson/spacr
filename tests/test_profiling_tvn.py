"""Measure profiling with profiling_normalization='tvn' (item 863).

'tvn' is mad_robustize per plate, then, after feature selection, the kept
features whitened on the negative-control wells through spaCR's one TVN core
(batch_correction.spherize_features -> ops_tvn._control_basis). At epsilon 0
the profiles must equal that core applied directly to the mad_robustize
profiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s
from spacr.batch_correction import spherize_features
from tests.test_image_based_profiling import _plate


def _screen(seed=0, plates=("p1", "p2", "p3"), controls=24, features=10):
    rng = np.random.default_rng(seed)
    signatures = {name: rng.normal(0, 1, features) for name in ("t1", "t2", "t3")}
    wells = pd.concat([_plate(p, rng, signatures, controls=controls, features=features)
                       for p in plates], ignore_index=True)
    # a shared nuisance direction in every well, which TVN should remove
    mix = rng.normal(0, 1, (len(wells), 1)) * rng.normal(0, 1, (1, features)) * 3
    cols = [c for c in wells.columns if c.startswith("cell_f")]
    wells[cols] = wells[cols].to_numpy() + mix
    return wells, cols


def _run(wells, features, normalization, epsilon=1e-6):
    return s._profile_wells(wells, features, group_columns=["treatment"],
                            negative_controls=["neg"], normalization=normalization,
                            tvn_epsilon=epsilon, null_size=200, report=None)


def test_tvn_at_epsilon_zero_is_the_core_applied_to_the_robust_z_profiles():
    wells, features = _screen()
    plain = _run(wells, features, "mad_robustize")
    tvn = _run(wells, features, "tvn", epsilon=0.0)
    assert tvn.kept == plain.kept
    kept = plain.kept
    negative = plain.selected["treatment"].astype(str).eq("neg")
    expected, info = spherize_features(plain.selected[kept], negative, method="ZCA-cor",
                                       epsilon=0.0, n_components="auto")
    np.testing.assert_array_equal(tvn.selected[kept].to_numpy(), expected.to_numpy())
    pd.testing.assert_frame_equal(tvn.selected.drop(columns=kept),
                                  plain.selected.drop(columns=kept))
    assert tvn.options["normalization"] == "tvn" and tvn.options["tvn_epsilon"] == 0.0
    assert any(note.startswith("TVN:") for note in tvn.notes)
    # the controls end with identity correlation
    ref = tvn.selected.loc[negative, kept].to_numpy()
    np.testing.assert_allclose(np.cov(ref, rowvar=False), np.eye(len(kept)), atol=1e-8)


def test_tvn_uses_the_shared_core(monkeypatch):
    from spacr import ops_tvn

    calls = []
    real = ops_tvn._control_basis

    def spy(fit, **kwargs):
        calls.append(fit.shape)
        return real(fit, **kwargs)

    monkeypatch.setattr(ops_tvn, "_control_basis", spy)
    wells, features = _screen(seed=1)
    _run(wells, features, "tvn")
    assert len(calls) == 1 and calls[0][0] == 72


def test_tvn_removes_variation_the_controls_share():
    wells, features = _screen(seed=2)
    plain = _run(wells, features, "mad_robustize")
    tvn = _run(wells, features, "tvn")
    kept = plain.kept
    ctrl = plain.selected["treatment"].eq("neg")

    def first_share(frame):
        values = frame.loc[ctrl, kept].to_numpy(float)
        ev = np.linalg.eigvalsh(np.cov(values, rowvar=False))
        return ev[-1] / ev.sum()

    assert first_share(tvn.selected) < 0.5 * first_share(plain.selected)


def test_tvn_needs_a_negative_control():
    wells, features = _screen(seed=3)
    with pytest.raises(s._ProfilingError, match="negative-control"):
        s._profile_wells(wells, features, group_columns=["treatment"],
                         normalization="tvn", report=None)


def test_tvn_is_offered_in_the_measure_panel():
    from spacr.settings_spec import convert_settings_dict_for_gui

    kind, choices, default = convert_settings_dict_for_gui(
        {"profiling_normalization": "mad_robustize"})["profiling_normalization"]
    assert kind == "combo" and "tvn" in choices and default == "mad_robustize"
