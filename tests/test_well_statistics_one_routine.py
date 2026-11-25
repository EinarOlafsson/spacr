"""Merge 9b: one per-well centre-scale routine (schema 962)."""
import numpy as np
import pytest

from spacr import gene_measurement_compare as gmc
from spacr import ops_tvn, settings as S


def _data(seed=0, n=600, d=7, offset=1e4):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, d)) * rng.uniform(0.5, 3, d) + offset).astype(np.float32)
    wells = rng.choice(["a", "b", "c"], n)
    return x, wells


def _reference(x, wells):
    mean, sd = {}, {}
    for w in np.unique(wells):
        v = x[wells == w].astype(np.float64)
        mean[w], sd[w] = v.mean(0), v.std(0)
    return mean, sd


def test_routine_matches_a_float64_reference():
    x, wells = _data()
    mean, sd = ops_tvn.batch_mean_sd(x, np.arange(len(x)), wells, 10 ** 6)
    ref_mean, ref_sd = _reference(x, wells)
    for w in ref_mean:
        np.testing.assert_allclose(mean[w], ref_mean[w], rtol=1e-13)
        np.testing.assert_allclose(sd[w], ref_sd[w], rtol=1e-12)


@pytest.mark.parametrize("chunk", [1, 7, 64, 599])
def test_streaming_equals_in_memory(chunk):
    x, wells = _data(1)
    rows = np.arange(len(x))
    whole = ops_tvn.batch_mean_sd(x, rows, wells, len(x))
    part = ops_tvn.batch_mean_sd(x, rows, wells, chunk)
    for a, b in zip(whole, part):
        for w in a:
            np.testing.assert_allclose(a[w], b[w], rtol=1e-12, atol=0)


def test_zero_spread_policy():
    x, wells = _data(2)
    x[:, 3] = 1234.5678
    x[:, 4] = 0.0
    x[:, 5] = np.nan
    mean, sd = ops_tvn.batch_mean_sd(x, np.arange(len(x)), wells, 50)
    for w in sd:
        assert sd[w][3] == 1.0 and sd[w][4] == 1.0 and sd[w][5] == 1.0
        assert np.isfinite(mean[w]).all()
    out = ops_tvn._centerscale_by_batch(x.copy(), wells)
    assert np.allclose(out[:, 3], 0, atol=1e-3) and np.allclose(out[:, 4], 0)
    assert np.isnan(out[:, 5]).all()


def test_centerscale_gives_zero_mean_unit_sd_in_input_dtype():
    x, wells = _data(3)
    out = ops_tvn._centerscale_by_batch(x.copy(), wells)
    assert out.dtype == np.float32
    for w in np.unique(wells):
        v = out[wells == w].astype(np.float64)
        assert np.abs(v.mean(0)).max() < 1e-3
        np.testing.assert_allclose(v.std(0), 1, atol=1e-3)


def test_legacy_method_is_the_standard_scaler():
    from sklearn.preprocessing import StandardScaler

    x, wells = _data(4)
    out = ops_tvn._centerscale_by_batch(x.copy(), wells, method="legacy")
    for w in np.unique(wells):
        np.testing.assert_array_equal(
            out[wells == w], StandardScaler().fit_transform(x[wells == w]))
    with pytest.raises(ValueError):
        ops_tvn._centerscale_by_batch(x.copy(), wells, method="nope")


def test_gmc_and_ops_tvn_share_the_routine():
    x, wells = _data(5, n=300)
    rows = np.arange(len(x))
    control = np.random.default_rng(5).random(len(x)) < 0.5
    for w in np.unique(wells):
        control[np.flatnonzero(wells == w)[0]] = True
    mean, sd, _m, _d = gmc._well_statistics(x, rows, wells, control, 41)
    ref_mean, ref_sd = ops_tvn.batch_mean_sd(x, rows, wells, 10 ** 6)
    for w in mean:
        np.testing.assert_allclose(mean[w], ref_mean[w], rtol=1e-12)
        np.testing.assert_allclose(sd[w], ref_sd[w], rtol=1e-12)


def test_schema_962_replays_the_old_method_for_old_files():
    assert S.SETTINGS_SCHEMA_VERSION >= 962
    assert S.DEFAULT_CHANGES[962] == {"ops_well_statistics": "legacy"}
    old = {"ops_profile_normalisation": "tvn"}
    S._fold_older_defaults(old, 959, known={"ops_well_statistics"})
    assert old["ops_well_statistics"] == "legacy"
    new = {}
    S._fold_older_defaults(new, 962, known={"ops_well_statistics"})
    assert "ops_well_statistics" not in new
    assert ops_tvn.tvn_options(old)["well_statistics"] == "legacy"
    assert ops_tvn.tvn_options({})["well_statistics"] == "two_pass"


def test_legacy_tvn_is_repeatable_and_differs_slightly_from_new():
    x, wells = _data(6, n=400, d=12, offset=50.0)
    control = np.zeros(len(x), bool)
    control[::3] = True
    old, _ = ops_tvn.tvn_in_memory(x, wells, control, pca_sample=300,
                                   well_statistics="legacy")
    new, _ = ops_tvn.tvn_in_memory(x, wells, control, pca_sample=300)
    again, _ = ops_tvn.tvn_in_memory(x, wells, control, pca_sample=300,
                                     well_statistics="legacy")
    assert old.shape == new.shape
    np.testing.assert_array_equal(old, again)
