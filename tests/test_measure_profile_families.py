"""The profiling measurement families of spacr.measure against reference implementations.

Each family is checked on synthetic objects against an independent
computation: Haralick and PFTAS against mahotas on the same quantised
object, weighted Hu moments against skimage, Manders-type and rank
coefficients against per-object NumPy/SciPy, radial fractions on planted
rings, edge intensity on a planted rim. The OPS phenotype table must keep its
columns when the setting is off and gain exactly the declared ones when on,
and reruns must be identical.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spacr import measure as M

mahotas = pytest.importorskip("mahotas")


def _field(seed=0):
    """Three non-touching objects (disc, square, ellipse) and a 2-channel image."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((80, 90), np.int64)
    rr, cc = np.indices(labels.shape)
    labels[(rr - 20) ** 2 + (cc - 20) ** 2 <= 12 ** 2] = 1
    labels[45:70, 10:35] = 2
    labels[((rr - 40) / 10.0) ** 2 + ((cc - 65) / 18.0) ** 2 <= 1] = 3
    image = rng.gamma(4.0, 50.0, size=labels.shape + (2,))
    image[..., 1] += 0.6 * image[..., 0]
    return labels, image


def test_families_parse_presets_and_reject_unknown_names():
    assert M.profile_feature_families(None) == ()
    assert M.profile_feature_families("") == ()
    assert M.profile_feature_families([]) == ()
    assert M.profile_feature_families("profiling") == M.PROFILE_FEATURE_DEFAULT
    assert M.profile_feature_families("all") == M.PROFILE_FEATURE_FAMILIES
    assert M.profile_feature_families("granularity, edge_intensity") == (
        "edge_intensity", "granularity")
    with pytest.raises(ValueError, match="not a measurement family"):
        M.profile_feature_families(["haralick"])


def test_haralick_and_pftas_match_mahotas_on_the_same_quantised_object():
    labels, image = _field()
    table = M.measure_texture(labels, image)
    for row, label in enumerate(table["label"]):
        mask = labels == label
        box = np.argwhere(mask)
        (r0, c0), (r1, c1) = box.min(0), box.max(0) + 1
        sub = mask[r0:r1, c0:c1]
        for c in range(2):
            values = image[r0:r1, c0:c1, c][sub]
            low, high = values.min(), values.max()
            q32 = np.zeros(sub.shape, np.uint8)
            q32[sub] = np.minimum(np.floor((values - low) / (high - low) * 32), 31) + 1
            want = mahotas.features.haralick(q32, ignore_zeros=True, distance=5,
                                             return_mean=True)
            got = table.loc[row, [f"channel_{c}_haralick_{n}"
                                  for n in M._HARALICK_NAMES]].to_numpy(float)
            np.testing.assert_allclose(got, want, rtol=1e-10)
            q255 = np.zeros(sub.shape, np.uint8)
            q255[sub] = np.minimum(np.floor((values - low) / (high - low) * 255), 254) + 1
            want = mahotas.features.pftas(q255, T=mahotas.otsu(q255, ignore_zeros=True))
            got = table.loc[row, [f"channel_{c}_pftas_{k}" for k in range(54)]]
            np.testing.assert_allclose(got.to_numpy(float), want, rtol=1e-10)


def test_haralick_detects_planted_stripes():
    """Stripes one pixel wide anti-correlate at an odd offset; noise does not."""
    labels = np.zeros((40, 80), np.int64)
    labels[5:35, 5:35] = 1
    labels[5:35, 45:75] = 2
    rng = np.random.default_rng(3)
    image = rng.uniform(0, 1, labels.shape)
    image[:, 5:35] = (np.indices((40, 30))[1] % 2).astype(float)
    table = M.measure_texture(labels, image[..., None], pftas=False, distance=1)
    contrast = table["channel_0_haralick_contrast"].to_numpy()
    assert contrast[0] > 3 * contrast[1]


def test_weighted_hu_moments_match_skimage():
    from skimage.measure import moments_central, moments_hu, moments_normalized

    labels, image = _field(1)
    table = M.measure_intensity_moments(labels, image)
    for row, label in enumerate(table["label"]):
        for c in range(2):
            weighted = np.where(labels == label, image[..., c], 0.0)
            want = moments_hu(moments_normalized(moments_central(weighted)))
            got = table.loc[row, [f"channel_{c}_weighted_hu_{k}"
                                  for k in range(7)]].to_numpy(float)
            np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-14)


def test_mass_displacement_of_a_planted_off_centre_spot():
    labels = np.zeros((41, 41), np.int64)
    labels[10:31, 10:31] = 1
    image = np.zeros((41, 41, 1))
    image[10:31, 10:31] = 1.0
    image[12, 28] = 1e6
    table = M.measure_intensity_moments(labels, image)
    expected = np.hypot(12 - 20, 28 - 20)
    assert table["channel_0_mass_displacement"].iloc[0] == pytest.approx(
        expected, rel=1e-3)


def test_shape_extended_on_a_disc_and_skimage_hu():
    from skimage.measure import moments_central, moments_hu, moments_normalized

    labels, _image = _field()
    table = M.measure_shape_extended(labels)
    disc = table.iloc[0]
    assert disc["compactness"] == pytest.approx(1.0, abs=0.02)
    assert 0.85 < disc["form_factor"] <= 1.1
    assert disc["max_radius"] == pytest.approx(12.0, abs=1.0)
    for row, label in enumerate(table["label"]):
        mask = (labels == label).astype(float)
        want = moments_hu(moments_normalized(moments_central(mask)))
        got = table.loc[row, [f"hu_{k}" for k in range(7)]].to_numpy(float)
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-14)


def test_radial_fractions_follow_planted_rings():
    """All intensity in the outer quarter of a disc lands in the outer ring."""
    labels = np.zeros((61, 61), np.int64)
    rr, cc = np.indices(labels.shape)
    radius = np.hypot(rr - 30, cc - 30)
    labels[radius <= 24] = 1
    image = np.zeros((61, 61, 2))
    image[..., 0] = np.where(radius > 19, 1.0, 0.0)
    image[..., 1] = np.where(radius <= 5, 1.0, 0.0)
    table = M.measure_radial_distribution(labels, image)
    outer = table[[f"channel_0_frac_at_d_{k}" for k in range(4)]].iloc[0].to_numpy()
    inner = table[[f"channel_1_frac_at_d_{k}" for k in range(4)]].iloc[0].to_numpy()
    assert outer.sum() == pytest.approx(1.0)
    assert outer[3] > 0.95
    assert inner[0] > 0.95
    assert table["channel_0_mean_frac_3"].iloc[0] > 1.5
    flat = np.ones((61, 61, 1))
    even = M.measure_radial_distribution(labels, flat)
    np.testing.assert_allclose(
        even[[f"channel_0_mean_frac_{k}" for k in range(4)]].iloc[0], 1.0)
    np.testing.assert_allclose(
        even[[f"channel_0_radial_cv_{k}" for k in range(4)]].iloc[0], 0.0,
        atol=1e-12)


def test_edge_intensity_on_a_planted_rim():
    labels = np.zeros((30, 30), np.int64)
    labels[5:15, 5:15] = 1
    labels[0:10, 20:30] = 2          # touches the field border
    image = np.zeros((30, 30, 1))
    image[5:15, 5:15] = 2.0
    image[6:14, 6:14] = 1.0           # interior is dimmer than the rim
    table = M.measure_edge_intensity(labels, image)
    first = table.iloc[0]
    assert first["channel_0_mean_edge"] == pytest.approx(2.0)
    assert first["channel_0_integrated_edge"] == pytest.approx(2.0 * 36)
    assert first["channel_0_std_edge"] == pytest.approx(0.0)
    from skimage.segmentation import find_boundaries
    padded = np.pad(labels == 2, 1)
    expected = find_boundaries(padded, connectivity=1, mode="inner")[1:-1, 1:-1]
    assert table.iloc[1]["channel_0_integrated_edge"] == pytest.approx(
        image[..., 0][expected & (labels == 2)].sum())


def test_colocalisation_against_per_object_references():
    labels, image = _field(2)
    table = M.measure_colocalisation(labels, image)
    for row, label in enumerate(table["label"]):
        x = image[..., 0][labels == label]
        y = image[..., 1][labels == label]
        assert table.loc[row, "channel_0_channel_1_spearman"] == pytest.approx(
            stats.spearmanr(x, y).statistic, rel=1e-10)
        icq = np.mean((x - x.mean()) * (y - y.mean()) > 0) - 0.5
        assert table.loc[row, "channel_0_channel_1_icq"] == pytest.approx(icq)
        tx = np.median(x) + 3 * 1.4826 * np.median(np.abs(x - np.median(x)))
        ty = np.median(y) + 3 * 1.4826 * np.median(np.abs(y - np.median(y)))
        a, b = np.clip(x - tx, 0, None), np.clip(y - ty, 0, None)
        k1 = (a * b).sum() / (a * a).sum() if (a * a).sum() else 0.0
        assert table.loc[row, "channel_0_channel_1_k1"] == pytest.approx(k1)
        ra = stats.rankdata(x, method="dense")
        rb = stats.rankdata(y, method="dense")
        top = max(ra.max(), rb.max())
        w = (top - np.abs(ra - rb)) / top
        both = (x > tx) & (y > ty)
        rwc1 = (a * w)[both].sum() / a.sum() if a.sum() else 0.0
        assert table.loc[row, "channel_0_channel_1_rwc1"] == pytest.approx(rwc1)


def test_rank_weighted_colocalisation_is_one_for_identical_channels():
    labels, image = _field(4)
    image[..., 0] = np.where(np.indices(labels.shape)[0] % 3 == 0, 900.0, 10.0) \
        + image[..., 0] * 0.01
    image[..., 1] = image[..., 0]
    table = M.measure_colocalisation(labels, image)
    np.testing.assert_allclose(table["channel_0_channel_1_rwc1"], 1.0)
    np.testing.assert_allclose(table["channel_0_channel_1_spearman"], 1.0)


def test_granularity_sees_planted_granules():
    """Two-pixel granules on a flat cell are sieved out in the first steps."""
    labels = np.zeros((60, 60), np.int64)
    labels[5:55, 5:55] = 1
    image = np.full((60, 60, 1), 10.0)
    for r in range(10, 50, 8):
        for c in range(10, 50, 8):
            image[r:r + 2, c:c + 2] = 200.0
    table = M.measure_granularity(labels, image, spectrum_length=4,
                                  background_radius=6)
    spectrum = table[[f"channel_0_granularity_{k}" for k in range(1, 5)]].iloc[0]
    assert spectrum.iloc[0] > 90
    assert spectrum.iloc[1:].abs().max() < 5


def test_profile_features_declares_its_columns_and_reruns_identically():
    labels, image = _field(5)
    first = M.measure_profile_features(labels, image, "all")
    second = M.measure_profile_features(labels, image, "all")
    pd.testing.assert_frame_equal(first, second)
    assert list(first["label"]) == [1, 2, 3]
    families = {
        "edge_intensity": M.measure_edge_intensity(labels, image),
        "intensity_moments": M.measure_intensity_moments(labels, image),
        "radial_distribution": M.measure_radial_distribution(labels, image),
        "shape_extended": M.measure_shape_extended(labels),
        "texture_haralick": M.measure_texture(labels, image, pftas=False),
        "texture_pftas": M.measure_texture(labels, image, haralick=False),
        "texture_lbp": M.measure_lbp(labels, image),
        "texture_gabor": M.measure_gabor(labels, image),
        "periphery": M.measure_periphery(labels, image),
        "outside": M.measure_outside(labels, image),
        "colocalisation": M.measure_colocalisation(labels, image),
        "colocalisation_extra": M.measure_colocalisation_extra(labels, image,
                                                               costes=False),
        "colocalisation_costes": M.measure_costes(labels, image),
        "neighbours": M.measure_neighbours(labels),
        "granularity": M.measure_granularity(labels, image),
    }
    expected = ["label"] + [c for f in M.PROFILE_FEATURE_FAMILIES
                            for c in families[f].columns if c != "label"]
    assert list(first.columns) == expected
    assert len(set(expected)) == len(expected)


def _ops_field():
    """Two paired cells with nuclei in a 2-channel field."""
    rng = np.random.default_rng(7)
    cells = np.zeros((70, 70), np.int32)
    nuclei = np.zeros_like(cells)
    cells[5:35, 5:35] = 1
    cells[35:65, 30:65] = 2
    nuclei[12:26, 12:26] = 1
    nuclei[43:57, 40:55] = 2
    image = rng.gamma(3.0, 40.0, size=(70, 70, 2)).astype(np.float32)
    return nuclei, cells, image


def test_ops_table_is_unchanged_without_the_setting_and_gains_the_families():
    from spacr.ops_sample import measure_phenotype_field

    nuclei, cells, image = _ops_field()
    legacy = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"])
    explicit = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                       profile_features=None)
    pd.testing.assert_frame_equal(legacy, explicit)
    extended = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                       profile_features="profiling")
    again = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                    profile_features="profiling")
    pd.testing.assert_frame_equal(extended, again)
    assert set(legacy.columns) <= set(extended.columns)
    pd.testing.assert_frame_equal(extended[legacy.columns], legacy)
    added = [c for c in extended.columns if c not in legacy.columns]
    for prefix in ("nucleus", "cell", "cytoplasm"):
        assert f"{prefix}_dapi_haralick_contrast" in added
        assert f"{prefix}_dapi_pftas_53" in added
        assert f"{prefix}_gfp_frac_at_d_3" in added
        assert f"{prefix}_dapi_gfp_rwc1" in added
        assert f"{prefix}_dapi_gfp_manders_m1" in added
        assert f"{prefix}_gfp_mean_edge" in added
        assert f"{prefix}_form_factor" in added
    assert "cell_nearest_neighbor_distance" in added
    assert not any("granularity" in c for c in added)


def test_ops_profile_setting_is_registered_and_off_by_default():
    from spacr.ops_engine import _ph_profile_features
    from spacr.ops_settings import ops_defaults

    assert ops_defaults()["ops_ph_profile_features"] is None
    assert _ph_profile_features({}) is None
    assert _ph_profile_features({"ops_ph_profile_features": "profiling"}) == list(
        M.PROFILE_FEATURE_DEFAULT)


def test_lbp_shares_match_skimage_codes_per_object():
    from skimage.feature import local_binary_pattern

    labels, image = _field(8)
    table = M.measure_lbp(labels, image)
    for c in range(2):
        codes = local_binary_pattern(image[..., c], 16, 2.0, method="uniform")
        for row, label in enumerate(table["label"]):
            inside = codes[labels == label]
            want = [np.mean(inside == k) for k in range(18)]
            got = table.loc[row, [f"channel_{c}_lbp_r2_{k}" for k in range(18)]]
            np.testing.assert_allclose(got.to_numpy(float), want)
    np.testing.assert_allclose(
        table[[f"channel_0_lbp_r1_{k}" for k in range(10)]].sum(axis=1), 1.0)


def test_lbp_is_invariant_to_brightness():
    labels, image = _field(9)
    a = M.measure_lbp(labels, image)
    b = M.measure_lbp(labels, 3.0 * image + 7.0)
    pd.testing.assert_frame_equal(a, b)


def test_gabor_matches_skimage_kernel_and_finds_planted_orientation():
    from scipy.signal import fftconvolve
    from skimage.filters import gabor_kernel

    labels, image = _field(10)
    table = M.measure_gabor(labels, image, frequencies=(0.25,), orientations=2)
    kernel = gabor_kernel(0.25, theta=np.pi / 2)
    magnitude = np.abs(fftconvolve(image[..., 1], kernel, mode="same"))
    for row, label in enumerate(table["label"]):
        mask = labels == label
        want = magnitude[mask].mean() / image[..., 1][mask].mean()
        assert table.loc[row, "channel_1_gabor_f0_o1_mean"] == pytest.approx(
            want, rel=1e-6)
    stripes = np.zeros((60, 60, 1))
    stripes[..., 0] = 1.0 + np.sin(2 * np.pi * 0.25 * np.indices((60, 60))[1])
    square = np.zeros((60, 60), np.int64)
    square[10:50, 10:50] = 1
    out = M.measure_gabor(square, stripes, frequencies=(0.25,), orientations=2)
    assert out["channel_0_gabor_f0_o0_mean"].iloc[0] > 5 * out[
        "channel_0_gabor_f0_o1_mean"].iloc[0]


def test_field_families_give_the_same_values_for_several_masks_at_once():
    labels, image = _field(11)
    other = np.where(labels == 2, 0, labels)
    for single, many in (
            (M.measure_granularity, M._granularity_frames),
            (M.measure_lbp, M._lbp_frames),
            (M.measure_gabor, M._gabor_frames)):
        together = many([labels, other], image)
        pd.testing.assert_frame_equal(together[0], single(labels, image))
        pd.testing.assert_frame_equal(together[1], single(other, image))


def test_ops_table_with_field_families_matches_the_public_functions():
    from spacr.ops_sample import measure_phenotype_field

    nuclei, cells, image = _ops_field()
    table = measure_phenotype_field(
        nuclei, cells, image, ["DAPI", "GFP"],
        profile_features=["texture_lbp", "granularity", "edge_intensity"])
    gran = M.measure_granularity(cells, image)
    np.testing.assert_allclose(
        table["cell_gfp_granularity_3"].to_numpy(),
        gran["channel_1_granularity_3"].to_numpy(), equal_nan=True)
    lbp = M.measure_lbp(nuclei, image)
    np.testing.assert_allclose(table["nucleus_dapi_lbp_r1_4"].to_numpy(),
                               lbp["channel_0_lbp_r1_4"].to_numpy())
    assert "cytoplasm_gfp_mean_edge" in table


def test_ops_table_has_no_column_that_repeats_the_label():
    """The colocalisation block's own label column must not come out as a feature."""
    from spacr.ops_sample import measure_phenotype_field

    nuclei, cells, image = _ops_field()
    relabel = {1: 7, 2: 12}
    nuclei = np.vectorize(lambda v: relabel.get(v, 0))(nuclei).astype(np.int32)
    cells = np.vectorize(lambda v: relabel.get(v, 0))(cells).astype(np.int32)
    table = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                    profile_features="all")
    assert list(table["label"]) == [7, 12]
    assert not [c for c in table.columns if "label_correlation" in c]
    for column in table.columns:
        if column == "label":
            continue
        values = table[column].to_numpy(float)
        assert not np.array_equal(values, [7.0, 12.0]), column


def test_radial_rings_of_a_ring_shaped_object_are_finite():
    """A cytoplasm-like ring has an empty innermost ring: 0, not NaN."""
    labels = np.zeros((61, 61), np.int64)
    rr, cc = np.indices(labels.shape)
    radius = np.hypot(rr - 30, cc - 30)
    labels[(radius <= 24) & (radius > 12)] = 1
    image = np.random.default_rng(12).uniform(1, 2, (61, 61, 1))
    table = M.measure_radial_distribution(labels, image)
    assert np.isfinite(table.drop(columns="label").to_numpy(float)).all()
    assert table["channel_0_frac_at_d_0"].iloc[0] == 0.0
    assert table["channel_0_mean_frac_0"].iloc[0] == 0.0


def test_periphery_and_outside_repeat_measures_own_helpers():
    labels, image = _field(13)
    periphery = M.measure_periphery(labels, image)
    outside = M.measure_outside(labels, image)
    for c in range(2):
        want = M._periphery_intensity(labels, image[..., c])
        got = periphery[[f"channel_{c}_periphery_{s}" for s in M._RING_STATS]]
        np.testing.assert_allclose(got.to_numpy(float),
                                   np.array([row[1:] for row in want], float))
        want = M._outside_intensity(labels, image[..., c],
                                    exclude_foreground=True)
        got = outside[[f"channel_{c}_outside_{s}" for s in M._RING_STATS]]
        np.testing.assert_allclose(got.to_numpy(float),
                                   np.array([row[1:] for row in want], float))


def test_outside_ring_of_a_planted_halo():
    labels = np.zeros((40, 40), np.int64)
    labels[15:25, 15:25] = 1
    image = np.zeros((40, 40, 1))
    image[12:28, 12:28] = 5.0
    image[15:25, 15:25] = 1.0
    table = M.measure_outside(labels, image, distance=3)
    assert table["channel_0_outside_mean"].iloc[0] == pytest.approx(5.0)


def test_colocalisation_extra_on_planted_patterns():
    """Identical channels: NMI 1, Costes M1 = M2 = 1, every spot overlaps."""
    labels = np.zeros((50, 50), np.int64)
    labels[5:45, 5:45] = 1
    rng = np.random.default_rng(14)
    base = rng.normal(100.0, 2.0, (50, 50))
    for r in range(10, 40, 9):
        for c in range(10, 40, 9):
            base[r:r + 3, c:c + 3] += 500.0
    image = np.stack([base, base], axis=-1)
    table = M.measure_colocalisation_extra(labels, image)
    assert table["channel_0_channel_1_nmi"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert table["channel_0_channel_1_costes_m1"].iloc[0] == pytest.approx(1.0)
    assert table["channel_0_channel_1_puncta_overlap_ij"].iloc[0] == 1.0
    assert 16 <= table["channel_0_puncta_count"].iloc[0] <= 20  # + 3-sigma noise pixels
    other = np.stack([base, rng.permutation(base.ravel()).reshape(50, 50)], -1)
    shuffled = M.measure_colocalisation_extra(labels, other)
    assert shuffled["channel_0_channel_1_nmi"].iloc[0] < 0.05
    assert shuffled["channel_0_channel_1_puncta_overlap_ij"].iloc[0] < 0.5


def test_mutual_information_matches_a_per_object_reference():
    labels, image = _field(15)
    image[..., 1] = (image[..., 0] - image[..., 0].mean()) ** 2  # non-monotone
    table = M.measure_colocalisation_extra(labels, image)
    for row, label in enumerate(table["label"]):
        x = image[..., 0][labels == label]
        y = image[..., 1][labels == label]
        n = x.size
        qx = np.minimum(((stats.rankdata(x) - 0.5) / n * 4).astype(int), 3)
        qy = np.minimum(((stats.rankdata(y) - 0.5) / n * 4).astype(int), 3)
        p = np.histogram2d(qx, qy, bins=[np.arange(5)] * 2)[0] / n
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        mi = np.nansum(np.where(p > 0, p * np.log(p / (px * py)), 0)) / np.log(4)
        assert table.loc[row, "channel_0_channel_1_nmi"] == pytest.approx(mi)


def test_costes_thresholds_split_a_planted_colocalised_population():
    """Background noise uncorrelated, a correlated bright population above it."""
    rng = np.random.default_rng(16)
    labels = np.zeros((60, 60), np.int64)
    labels[:, :] = 1
    x = rng.normal(100, 5, (60, 60))
    y = rng.normal(100, 5, (60, 60))
    signal = rng.uniform(200, 800, (60, 60))
    bright = rng.uniform(size=(60, 60)) < 0.3
    x[bright] += signal[bright]
    y[bright] += 0.8 * signal[bright]
    table = M.measure_colocalisation_extra(labels, np.stack([x, y], -1))
    assert table["channel_0_channel_1_costes_m1"].iloc[0] > 0.9
    assert table["channel_0_channel_1_costes_m2"].iloc[0] > 0.9


def test_an_enclosed_cell_keeps_its_row_with_nan_ring_and_flag():
    from spacr.ops_engine import _ph_outside_ring_column
    from spacr.ops_sample import measure_phenotype_field

    cells = np.zeros((60, 60), np.int32)
    nuclei = np.zeros_like(cells)
    cells[0:60, 0:20] = 1
    cells[0:20, 20:40] = 2
    cells[20:40, 20:40] = 3          # enclosed on every side
    cells[40:60, 20:40] = 4
    cells[0:60, 40:60] = 5
    for label, (r, c) in enumerate(((30, 10), (10, 30), (30, 30), (50, 30),
                                    (30, 50)), start=1):
        nuclei[r - 3:r + 3, c - 3:c + 3] = label
    cells[0:5, 0:20] = 0              # cell 1 touches background
    image = np.random.default_rng(17).uniform(1, 2, (60, 60, 2)).astype(np.float32)
    table = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                    profile_features=["outside"])
    row = table.set_index("label").loc[3]
    assert row["cell_enclosed"] == 1
    assert np.isnan(row["cell_dapi_outside_mean"])
    assert table.set_index("label").loc[1, "cell_enclosed"] == 0
    assert len(table) == 5
    assert _ph_outside_ring_column("cell_dapi_outside_percentile_95")
    assert _ph_outside_ring_column("cell_dapi_outside_mean")
    assert not _ph_outside_ring_column("cell_enclosed")
    assert not _ph_outside_ring_column("cell_dapi_mean_intensity")


def test_costes_family_equals_the_costes_columns_of_the_extra_function():
    labels, image = _field(18)
    full = M.measure_colocalisation_extra(labels, image)
    lite = M.measure_colocalisation_extra(labels, image, costes=False)
    costes = M.measure_costes(labels, image)
    pd.testing.assert_frame_equal(full[costes.columns], costes)
    assert set(full.columns) == set(lite.columns) | set(costes.columns)


def test_aggregate_keeps_cells_whose_only_gap_is_the_outside_ring(tmp_path):
    """ops_drop_nonfinite_rows drops label 8 of each well (a NaN cell area),
    never the cells whose background ring is empty."""
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings

    db, wells, _ = _plate(tmp_path)
    for well in wells:
        path = tmp_path / f"ops_phenotype_cells_{well}.parquet"
        cells = pd.read_parquet(path)
        cells["cell_dapi_outside_mean"] = np.where(
            cells["label"] % 4 == 0, np.nan, 1.0)
        cells["cell_enclosed"] = (cells["label"] % 4 == 0).astype(np.int64)
        cells.to_parquet(path)
    settings = _settings(tmp_path, ops_bootstrap=0)
    result = ops_engine._ph_aggregate(
        db, PLATE, wells, {**settings, "ops_drop_nonfinite_rows": True},
        str(tmp_path))
    assert result["dropped_nonfinite_rows"] == 3


def test_ops_measure_task_carries_the_profile_families(tmp_path):
    """The ph_measure worker passes ops_ph_profile_features through to the table."""
    from spacr import ops_engine

    nuclei, cells, image = _ops_field()
    np.save(tmp_path / "stack.npy", image)
    np.savez(tmp_path / "masks.npz", nuclei=nuclei, cells=cells)
    base = {"site": 3, "files": {}, "channels": ["DAPI", "GFP"],
            "mask_path": str(tmp_path / "masks.npz"),
            "cell_path": str(tmp_path / "site-3.parquet"),
            "texture": True, "staged": str(tmp_path / "stack.npy")}
    legacy, _u, _s = ops_engine._ph_measure_table(base)
    settings = {"ops_ph_profile_features": "profiling"}
    task = {**base, "profile_features": ops_engine._ph_profile_features(settings),
            "granularity_subsample": ops_engine._ph_granularity_subsample(settings)}
    table, _u, _s = ops_engine._ph_measure_table(task)
    assert ops_engine._ph_granularity_subsample(settings) is None
    assert set(legacy.columns) < set(table.columns)
    pd.testing.assert_frame_equal(table[legacy.columns], legacy)
    features = ops_engine._ph_feature_columns(table.columns)
    assert "cell_dapi_haralick_contrast" in features
    assert "nucleus_dapi_gfp_nmi" in features
    assert not [c for c in table.columns if "costes" in c or "granularity" in c]
