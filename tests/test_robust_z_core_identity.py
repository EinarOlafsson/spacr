"""Every robust z / centre-scale kernel gives the bytes it gave before merge 9.

The kernels now share ``batch_correction._center`` and ``_scale``, each
with its own policy (centre, scale, MAD constant, zero-scale rule). Each
test keeps a verbatim copy of the old kernel and compares it with the new
call on random inputs that include zero-MAD, constant, NaN and (where the
kernel can see them) infinite columns. Equality is exact: same values,
same dtype, NaN in the same places.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import assay_report, batch_correction as bc, embeddings, ops_clusters, ops_mitotic, sp_stats
from spacr import gene_measurement_compare as gmc

SEEDS = range(6)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape
    if a.dtype == object:
        assert a.tolist() == b.tolist()
    else:
        assert np.array_equal(a, b, equal_nan=True)


def _same_frame(a, b):
    if isinstance(a, pd.DataFrame):
        assert list(a.columns) == list(b.columns)
        pd.testing.assert_index_equal(a.index, b.index)
        for column in a.columns:
            _same(a[column].to_numpy(), b[column].to_numpy())
    else:
        pd.testing.assert_index_equal(a.index, b.index)
        _same(a.to_numpy(), b.to_numpy())


def _matrix(seed, n=60, inf=False, dtype=float):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)) * rng.uniform(0.1, 5, size=8)
    x[:, 1] = 3.0
    x[:, 2] = 0.0
    x[: n // 2 + 3, 2] = 0.0
    x[n // 2 + 3:, 2] = rng.normal(size=n - n // 2 - 3)
    x[:, 3] = np.where(rng.random(n) < 0.8, 1.0, 2.0)
    x[rng.random(n) < 0.15, 4] = np.nan
    x[:, 5] = np.nan
    x[rng.random(n) < 0.5, 6] = np.nan
    if inf:
        x[rng.integers(n), 7] = np.inf
    return x.astype(dtype)


def _frame(seed, n=60, inf=False):
    return pd.DataFrame(_matrix(seed, n, inf), columns=[f"f{j}" for j in range(8)])


def _old_mad(values, *, scale=True):
    data = sp_stats._finite(values)
    if not data.size:
        return float("nan")
    raw = float(np.median(np.abs(data - np.median(data))))
    return raw * sp_stats.MAD_SCALE if scale else raw


def _old_robust_z_scores(values, reference):
    values = np.asarray(values, dtype=float)
    ref = sp_stats._finite(reference)
    if ref.size < 2:
        return np.full(values.shape, np.nan)
    spread = _old_mad(ref)
    if not np.isfinite(spread) or spread <= 0:
        return np.full(values.shape, np.nan)
    return (values - float(np.median(ref))) / spread


@pytest.mark.parametrize("seed", SEEDS)
def test_sp_stats_mad_and_robust_z_scores(seed):
    x = _matrix(seed, inf=True)
    for j in range(x.shape[1]):
        for scale in (True, False):
            old, new = _old_mad(x[:, j], scale=scale), sp_stats.mad(x[:, j], scale=scale)
            assert type(old) is type(new) and (old == new or (old != old and new != new))
        _same(_old_robust_z_scores(x[:, 0], x[:, j]), sp_stats.robust_z_scores(x[:, 0], x[:, j]))
    _same(_old_robust_z_scores(x[:, 0], [1.0]), sp_stats.robust_z_scores(x[:, 0], [1.0]))


def _old_robust_z_to_controls(cells, features, *, control, group="well"):
    control = pd.Series(control, index=cells.index).fillna(False).astype(bool)
    values = cells.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(np.nan, index=cells.index, columns=list(features))
    for _key, rows in cells.groupby(group, sort=False).groups.items():
        rows = pd.Index(rows)
        reference = values.loc[rows[control.loc[rows].to_numpy()]]
        if reference.empty:
            continue
        centre = reference.median()
        spread = (reference - centre).abs().median() * 1.4826
        spread = spread.where(spread > 0)
        out.loc[rows] = ((values.loc[rows] - centre) / spread).to_numpy()
    return out


def _cells(seed, n=90):
    rng = np.random.default_rng(seed)
    cells = _frame(seed, n, inf=True)
    cells["well"] = rng.choice(["A1", "A2", "B1"], n)
    control = pd.Series(rng.random(n) < 0.4, index=cells.index)
    control[cells["well"] == "B1"] = False
    return cells, control


@pytest.mark.parametrize("seed", SEEDS)
def test_gmc_robust_z_to_controls(seed):
    cells, control = _cells(seed)
    features = [f"f{j}" for j in range(8)]
    _same_frame(_old_robust_z_to_controls(cells, features, control=control),
                gmc.robust_z_to_controls(cells, features, control=control))


def _old_well_statistics(embedding, rows, wells, control, chunk):
    names = sorted(pd.unique(wells))
    width = np.asarray(embedding[rows[:1]]).shape[1]
    first = {w: np.zeros(width) for w in names}
    second = {w: np.zeros(width) for w in names}
    count = {w: 0 for w in names}
    controls = {w: [] for w in names}
    for start in range(0, len(rows), int(chunk)):
        x = np.asarray(embedding[rows[start:start + int(chunk)]], dtype=np.float32)
        here = wells[start:start + int(chunk)]
        for w in names:
            k = here == w
            if k.any():
                first[w] += x[k].sum(0)
                second[w] += (x[k].astype(np.float64) ** 2).sum(0)
                count[w] += int(k.sum())
                kc = k & control[start:start + int(chunk)]
                if kc.any():
                    controls[w].append(x[kc])
    mean = {w: first[w] / count[w] for w in names}
    sd = {w: np.sqrt(np.clip(second[w] / count[w] - mean[w] ** 2, 1e-12, None))
          for w in names}
    median, mad = {}, {}
    for w in names:
        values = np.vstack(controls[w])
        median[w] = np.median(values, 0)
        mad[w] = 1.4826 * np.median(np.abs(values - median[w]), 0)
        mad[w][mad[w] == 0] = 1
    return mean, sd, median, mad


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gmc_well_statistics_wraps_ops_tvn(seed, dtype):
    rng = np.random.default_rng(seed)
    embedding = _matrix(seed, n=200, dtype=dtype)
    embedding[:, 5] = 2.5
    rows = np.sort(rng.choice(200, 150, replace=False))
    wells = rng.choice(["w1", "w2", "w3"], len(rows))
    control = rng.random(len(rows)) < 0.5
    for w in ("w1", "w2", "w3"):
        control[np.flatnonzero(wells == w)[0]] = True
    old = _old_well_statistics(embedding, rows, wells, control, 37)
    new = gmc._well_statistics(embedding, rows, wells, control, 37, "legacy")
    for a, b in zip(old, new):
        assert list(a) == list(b)
        for w in a:
            _same(a[w], b[w])


def _old_normalise_to_controls(frame, features, controls, how, plate_column, whitening):
    values = frame[features].astype(float)
    groups = (frame[plate_column] if plate_column else pd.Series(0, index=frame.index))
    dropped = set()
    out = values.copy()
    for _, index in groups.groupby(groups, sort=False, dropna=False).groups.items():
        block = values.loc[index]
        reference = block[controls.loc[index].to_numpy()]
        if how == "robust_zscore":
            centre = reference.median()
            spread = (reference - centre).abs().median() * 1.4826
        else:
            centre = reference.mean()
            spread = reference.std(ddof=1)
        if how in ("center", "tvn"):
            out.loc[index] = block - centre
            continue
        flat = ~(spread > 0)
        dropped.update(spread.index[flat])
        out.loc[index] = (block - centre) / spread.where(~flat, 1.0)
    kept = [column for column in features if column not in dropped]
    return out[kept], kept


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("how", ["robust_zscore", "zscore", "center"])
@pytest.mark.parametrize("plate", [None, "plate"])
def test_embeddings_normalise_to_controls(seed, how, plate):
    rng = np.random.default_rng(seed)
    frame = _frame(seed, 80, inf=True)
    frame["plate"] = rng.choice(["p1", "p2"], 80)
    controls = pd.Series(rng.random(80) < 0.5, index=frame.index)
    features = [f"f{j}" for j in range(8)]
    old = _old_normalise_to_controls(frame, features, controls, how, plate, "ZCA")
    new = embeddings._normalise_to_controls(frame, features, controls, how, plate, "ZCA")
    assert old[1] == new[1]
    _same_frame(old[0], new[0])


def _old_mad_robustize(values, labels, reference_mask, epsilon):
    result = values.copy()
    tiny = 10 * np.finfo(float).eps
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for code in np.unique(labels):
                rows = labels == code
                fit = values[rows & reference_mask]
                centre = np.nanmedian(fit, axis=0)
                spread = np.nanmedian(np.abs(fit - centre), axis=0) * sp_stats.MAD_SCALE
                flat = (spread < tiny) & (np.nan_to_num(np.nanstd(fit, axis=0), nan=0.0) < tiny)
                spread = np.where(flat, 1.0, spread + epsilon)
                result[rows] = (values[rows] - centre) / spread
    return result


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("epsilon", [1e-18, 0.0, 1e-3])
def test_sp_stats_normalize_profiles_mad_robustize(seed, epsilon):
    rng = np.random.default_rng(seed)
    frame = _frame(seed, 70, inf=True)
    frame["plateID"] = rng.choice(["a", "b"], 70)
    reference = pd.Series(rng.random(70) < 0.6, index=frame.index)
    features = [f"f{j}" for j in range(8)]
    labels = pd.factorize(frame["plateID"].astype(str), sort=True)[0]
    old = _old_mad_robustize(frame[features].to_numpy(dtype=float), labels,
                             reference.to_numpy(bool), epsilon)
    new = sp_stats._normalize_profiles(frame, features, reference=reference, epsilon=epsilon)
    _same(old, new[features].to_numpy())


def _old_moa_scale(reference, method):
    if method == "zscore":
        centre, scale = reference.mean(), reference.std(ddof=1)
    else:
        centre = reference.median()
        scale = 1.4826 * (reference - centre).abs().median()
    return centre, scale.where(np.isfinite(scale) & (scale > 0))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("method", ["robust", "zscore"])
def test_sp_stats_moa_normalise(seed, method):
    rng = np.random.default_rng(seed)
    frame = _frame(seed, 60, inf=True)
    features = [f"f{j}" for j in range(8)]
    frame["plate"] = rng.choice(["p1", "p2"], 60)
    control = rng.random(60) < 0.5
    order = frame.assign(_control=control)
    order = order.iloc[np.argsort(order["plate"].to_numpy(), kind="stable")]
    control_rows = order.pop("_control").to_numpy()
    order = order.reset_index(drop=True)
    out, bad = order.copy(), set()
    values = order[features].astype(float)
    for plate in sorted(order["plate"].unique()):
        rows = (order["plate"] == plate).to_numpy()
        centre, scale = _old_moa_scale(values[rows & control_rows], method)
        bad |= set(scale.index[scale.isna()]) | set(centre.index[~np.isfinite(centre)])
        out.loc[rows, features] = ((values[rows] - centre) / scale).to_numpy()
    keep = [name for name in features if name not in bad]
    out = out.drop(columns=[name for name in features if name in bad])
    out[keep] = out[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    new, new_keep, new_bad = sp_stats._moa_normalise(frame, features, method, control, None)
    assert new_keep == keep and new_bad == sorted(bad)
    _same_frame(out, new)


def _old_standardised_measurements(frame, measurements):
    values = frame[list(measurements)].apply(pd.to_numeric, errors='coerce').astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    keep = [name for name in values.columns if values[name].notna().mean() > 0.5]
    values = values[keep]
    centre = values.median()
    spread = (values - centre).abs().median() * 1.4826
    fallback = values.std(ddof=0)
    spread = spread.where(spread > 0, fallback)
    usable = [name for name in values.columns if np.isfinite(spread[name]) and spread[name] > 0]
    scaled = (values[usable] - centre[usable]) / spread[usable]
    return scaled.fillna(0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_sp_stats_standardised_measurements(seed):
    frame = _frame(seed, 50, inf=True)
    features = list(frame.columns)
    _same_frame(_old_standardised_measurements(frame, features),
                sp_stats._standardised_measurements(frame, features))


def _old_outlier_deviation(block):
    centre = block.median()
    spread = (block - centre).abs().median() * 1.4826
    spread = spread.where(spread > 0, 1.0)
    return ((block - centre) / spread).abs().max(axis=1)


@pytest.mark.parametrize("seed", SEEDS)
def test_sp_stats_typical_in_focus_outlier_block(seed):
    scaled = _frame(seed, 30).fillna(0.0)
    scaled.iloc[:, 5] = 0.0
    members = list(scaled.index[::2])
    deviation = _old_outlier_deviation(scaled.loc[members])
    frame = pd.DataFrame(index=scaled.index)
    for threshold in np.unique(deviation.to_numpy()):
        kept, excluded = sp_stats._typical_in_focus(
            frame, members, focus_column=None, saturation_columns=(),
            outlier_z=float(threshold), max_saturation=1.0, scaled=scaled)
        assert kept == [m for m in members if not deviation[m] > threshold]
        assert sorted(excluded) == sorted(m for m in members if deviation[m] > threshold)


def _old_anomaly_spread(centred_negative):
    spread = 1.4826 * np.median(np.abs(centred_negative), axis=0)
    fallback = centred_negative.std(axis=0)
    return np.where(spread > 0, spread, fallback)


@pytest.mark.parametrize("seed", SEEDS)
def test_sp_stats_score_anomalies_spread(seed):
    x = _matrix(seed)
    x = np.where(np.isfinite(x), x, 0.0)
    x[:, 6] = -0.0
    x = x - np.median(x, axis=0)
    _same(_old_anomaly_spread(x),
          sp_stats._frame_scale(x, True, centre=0.0, skipna=False, zero_scale="sd"))


def _old_well_robust_z(cells, features, group):
    if group is None:
        return cells[list(features)].to_numpy(np.float32)
    groups = pd.factorize(cells[group])[0]
    rows_of = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    out = np.empty((len(cells), len(features)), np.float32)
    for j, feature in enumerate(features):
        column = cells[feature].to_numpy(np.float64)
        column = np.where(np.isfinite(column), column, np.nan)
        for rows in rows_of:
            block = column[rows]
            median = np.nanmedian(block) if len(block) else np.nan
            mad = 1.4826 * np.nanmedian(np.abs(block - median)) if len(block) else np.nan
            out[rows, j] = (block - median) / mad if mad > 0 else np.nan
    return out


def _old_fit_population(cells, features, group):
    fitted = {}
    for key, rows in ops_mitotic._normalization_groups(cells, group).items():
        median = np.full(len(features), np.nan)
        scale = np.full(len(features), np.nan)
        for j, feature in enumerate(features):
            values = cells[feature].iloc[rows].to_numpy(np.float64, na_value=np.nan)
            values = values[np.isfinite(values)]
            if len(values):
                median[j] = np.median(values)
                mad = 1.4826 * np.median(np.abs(values - median[j]))
                if mad > 0:
                    scale[j] = mad
        fitted[key] = {"median": median, "scale": scale, "cells": int(len(rows))}
    return fitted


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("seed", SEEDS)
def test_ops_mitotic_robust_z_and_population_fit(seed):
    cells, _control = _cells(seed)
    cells.loc[cells.index[:5], "well"] = None
    features = [f"f{j}" for j in range(8)]
    _same(_old_well_robust_z(cells, features, "well"), ops_mitotic._well_robust_z(cells, features, "well"))
    old = _old_fit_population(cells, features, "well")
    new = ops_mitotic._fit_population_normalization(cells, features, "well")["groups"]
    assert list(old) == list(new)
    for key in old:
        assert old[key]["cells"] == new[key]["cells"]
        _same(old[key]["median"], new[key]["median"])
        _same(old[key]["scale"], new[key]["scale"])


def _old_phase_view(names, mask, guides, medians, seed, width):
    rng = None if seed is None else np.random.default_rng(int(seed))
    view = np.zeros((len(names), int(width)))
    has = np.zeros(len(names), bool)
    for i, name in enumerate(names):
        own = list(guides.get(str(name), ()))
        if mask[i] and rng is not None and own:
            own = [own[p] for p in rng.choice(len(own), len(own), replace=True)]
        found = [medians[f"{name}|{g}"] for g in own if f"{name}|{g}" in medians]
        if found:
            weight = np.array([n for _, n in found], float)
            view[i] = (weight[:, None] * np.vstack([v for v, _ in found])).sum(0) / weight.sum()
            has[i] = True
    controls = (~np.asarray(mask, bool)) & has
    if not controls.any():
        return np.where(has[:, None], view, 0)
    centre = np.median(view[controls], 0)
    spread = 1.4826 * np.median(np.abs(view[controls] - centre), 0)
    return np.where(has[:, None], (view - centre) / np.where(spread > 0, spread, 1), 0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("draw", [None, 3])
def test_ops_clusters_phase_view(seed, draw):
    rng = np.random.default_rng(seed)
    names = [f"g{i}" for i in range(12)]
    mask = np.array([i >= 4 for i in range(12)])
    guides = {n: [f"s{k}" for k in range(3)] for n in names[:-1]}
    medians = {}
    for n in names[:-1]:
        for g in guides[n]:
            v = rng.normal(size=6)
            v[1] = 2.0
            v[2] = 0.0 if rng.random() < 0.8 else 1.0
            medians[f"{n}|{g}"] = (v, int(rng.integers(1, 9)))
    _same(_old_phase_view(names, mask, guides, medians, draw, 6),
          ops_clusters.phase_view(names, mask, guides, medians, draw, 6))


def _old_scale(values, robust):
    if robust:
        medians = values.median(axis=0)
        scale = (values.subtract(medians).abs().median(axis=0) * 1.4826)
    else:
        scale = values.std(axis=0, ddof=0)
    return scale.replace([np.inf, -np.inf, 0], np.nan)


def _old_center(values, robust):
    return values.median(axis=0) if robust else values.mean(axis=0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("robust", [True, False])
def test_batch_correction_center_and_scale(seed, robust):
    frame = _frame(seed, 40, inf=True)
    _same_frame(_old_center(frame, robust), bc._center(frame, robust))
    _same_frame(_old_scale(frame, robust), bc._scale(frame, robust))


def _old_assay_robust_z(v, ref):
    med = v.groupby(ref).transform("median")
    dev = (v - med).abs().groupby(ref).transform("median")
    return (v - med) / (1.4826 * dev.replace(0, np.nan))


@pytest.mark.parametrize("seed", SEEDS)
def test_assay_report_robust_z(seed):
    rng = np.random.default_rng(seed)
    frame = _frame(seed, 70, inf=True)
    ref = pd.Series(rng.choice(["p1\x1fa", "p1\x1fb", "p2"], 70), index=frame.index)
    ref[ref.index[:3]] = None
    for column in frame.columns:
        _same_frame(_old_assay_robust_z(frame[column], ref),
                    assay_report._robust_z(frame[column], ref))


def test_unknown_zero_scale_policy_is_refused():
    with pytest.raises(ValueError, match="zero_scale"):
        bc._scale(_frame(0), True, zero_scale="drop")
