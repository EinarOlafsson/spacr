"""Image-level measurements on synthetic images with known values."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from spacr.measure import _image_feature_masks, _image_level_features


def _two_squares(size=128, low=100, high=1100, dtype=np.uint16):
    image = np.full((size, size), low, dtype=dtype)
    image[16:48, 16:48] = high
    image[80:112, 64:96] = high
    return image


def test_intensity_statistics_and_saturation():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    image[0, :5] = 255
    record = _image_level_features({0: image}, confluency=False)
    values = image.astype(float).ravel()
    assert record["channel_0_intensity_mean"] == pytest.approx(values.mean())
    assert record["channel_0_intensity_median"] == pytest.approx(np.median(values))
    assert record["channel_0_intensity_std"] == pytest.approx(values.std())
    assert record["channel_0_intensity_integrated"] == pytest.approx(values.sum())
    assert record["channel_0_intensity_p95"] == pytest.approx(np.percentile(values, 95))
    assert record["channel_0_intensity_p1"] == pytest.approx(np.percentile(values, 1))
    assert record["channel_0_saturation_level"] == 255
    assert record["channel_0_saturation_fraction"] == pytest.approx(0.05)


def test_nonfinite_pixels_are_excluded_and_float_saturation_needs_a_level():
    image = np.full((8, 8), 2.0)
    image[0, 0] = np.nan
    image[1, 1] = 10.0
    record = _image_level_features({0: image}, confluency=False)
    assert record["channel_0_intensity_integrated"] == pytest.approx(2.0 * 62 + 10.0)
    assert np.isnan(record["channel_0_saturation_fraction"])
    leveled = _image_level_features({0: image}, saturation_levels={0: 10.0}, confluency=False)
    assert leveled["channel_0_saturation_fraction"] == pytest.approx(1 / 63)


def test_focus_scores_match_their_definitions_and_drop_with_blur():
    ramp = np.tile(np.arange(64, dtype=np.float64) * 3.0, (64, 1))
    record = _image_level_features({0: ramp}, confluency=False)
    assert record["channel_0_focus_brenner"] == pytest.approx(0.5 * (6.0 ** 2 + 0.0))
    assert record["channel_0_focus_laplacian_variance"] == pytest.approx(
        np.var(np.r_[np.full(64, 6.0), np.zeros(62 * 64), np.full(64, -6.0)]), rel=1e-6)
    flat = _image_level_features({0: np.full((32, 32), 7.0)}, confluency=False)
    for name in ("laplacian_variance", "brenner", "tenengrad"):
        assert flat[f"channel_0_focus_{name}"] == 0
    rng = np.random.default_rng(1)
    sharp = rng.normal(1000, 200, (128, 128))
    blurred = gaussian_filter(sharp, 3)
    sharp_record = _image_level_features({0: sharp}, confluency=False)
    blurred_record = _image_level_features({0: blurred}, confluency=False)
    for name in ("laplacian_variance", "brenner", "tenengrad"):
        assert blurred_record[f"channel_0_focus_{name}"] < 0.2 * sharp_record[f"channel_0_focus_{name}"]


def test_texture_of_flat_and_checkerboard_images():
    flat = _image_level_features({0: np.full((32, 32), 5, np.uint16)}, confluency=False)
    assert flat["channel_0_texture_glcm_contrast"] == 0
    assert flat["channel_0_texture_glcm_energy"] == pytest.approx(1.0)
    assert flat["channel_0_texture_glcm_homogeneity"] == pytest.approx(1.0)
    assert flat["channel_0_texture_glcm_entropy"] == pytest.approx(0.0)
    board = (np.indices((32, 32)).sum(axis=0) % 2 * 1000).astype(np.uint16)
    rough = _image_level_features({0: board}, confluency=False)
    assert rough["channel_0_texture_glcm_contrast"] > 100
    assert rough["channel_0_texture_glcm_homogeneity"] == pytest.approx(0.5, abs=0.01)


def test_texture_is_bounded_on_large_images():
    image = np.zeros((2048, 1024), np.uint16)
    record = _image_level_features({0: image}, confluency=False)
    assert record["channel_0_texture_glcm_downsample"] == 4


def test_foreground_background_snr_and_objects_from_otsu():
    rng = np.random.default_rng(0)
    image = _two_squares().astype(np.float64) + rng.normal(0, 10, (128, 128))
    record = _image_level_features({0: image}, confluency=False)
    assert record["channel_0_otsu_area_fraction"] == pytest.approx(2 * 32 * 32 / 128 ** 2, abs=0.002)
    assert record["channel_0_background_median"] == pytest.approx(100, abs=2)
    assert record["channel_0_background_std"] == pytest.approx(10, abs=1)
    assert record["channel_0_snr"] == pytest.approx(100, rel=0.15)
    assert record["channel_0_signal_to_background"] == pytest.approx(11, rel=0.05)
    assert record["channel_0_objects_count"] == 2
    assert record["channel_0_objects_area_mean"] == pytest.approx(1024, rel=0.02)
    assert record["channel_0_objects_solidity_mean"] == pytest.approx(1.0, abs=0.02)
    assert record["channel_0_objects_eccentricity_mean"] == pytest.approx(0.0, abs=0.2)
    assert record["channel_0_objects_intensity_mean"] == pytest.approx(1100, abs=5)


def test_existing_masks_replace_the_otsu_objects():
    image = _two_squares()
    labels = np.zeros(image.shape, np.int32)
    labels[0:64, 0:64] = 1
    labels[64:128, 0:32] = 2
    record = _image_level_features({0: image}, {"cell": labels}, confluency=False)
    assert record["cell_objects_count"] == 2
    assert record["cell_objects_area_mean"] == pytest.approx((64 * 64 + 64 * 32) / 2)
    assert record["cell_objects_eccentricity_mean"] > 0
    assert record["cell_objects_source"] == "mask"
    expected = (np.mean(image[0:64, 0:64]) + np.mean(image[64:128, 0:32])) / 2
    assert record["channel_0_cell_object_intensity_mean"] == pytest.approx(expected)
    assert "channel_0_objects_count" not in record


def test_confluency_tracks_covered_area():
    rng = np.random.default_rng(2)
    image = np.full((256, 256), 500.0) + rng.normal(0, 1, (256, 256))
    image[:, :128] += rng.normal(0, 150, (256, 128))
    record = _image_level_features({0: image})
    assert record["channel_0_confluency"] == pytest.approx(0.5, abs=0.06)
    assert record["channel_0_confluency_downsample"] == 1


def test_colocalisation_identical_disjoint_and_anticorrelated_channels():
    a = _two_squares()
    b = np.full(a.shape, 100, np.uint16)
    b[80:112, 64:96] = 1100
    c = _two_squares()[::-1, ::-1].copy()
    c[:] = 100
    c[60:70, 100:120] = 900
    record = _image_level_features({1: a, 2: a.copy(), 3: b, 4: c}, confluency=False)
    assert record["channels_1_2_pearson"] == pytest.approx(1.0)
    assert record["channels_1_2_manders_m1"] == pytest.approx(1.0)
    assert record["channels_1_2_manders_m2"] == pytest.approx(1.0)
    assert record["channels_1_3_manders_m1"] == pytest.approx(0.5)
    assert record["channels_1_3_manders_m2"] == pytest.approx(1.0)
    assert record["channels_1_3_overlap_fraction"] == pytest.approx(0.5)
    assert record["channels_1_4_manders_m1"] == 0
    assert record["channels_1_4_manders_m2"] == 0
    inverse = (1200 - a.astype(np.int32)).astype(np.uint16)
    pair = _image_level_features({0: a, 1: inverse}, confluency=False)
    assert pair["channels_0_1_pearson"] == pytest.approx(-1.0)


def test_shapes_must_match_and_planes_must_be_two_dimensional():
    with pytest.raises(ValueError):
        _image_level_features({0: np.zeros((4, 4)), 1: np.zeros((4, 5))})
    with pytest.raises(ValueError):
        _image_level_features({0: np.zeros((2, 4, 4))})
    assert _image_level_features({}) == {}


def test_masks_are_found_beside_the_images(tmp_path):
    image = tmp_path / "plate1_A01_1.tif"
    image.write_bytes(b"")
    stack = tmp_path / "masks" / "nucleus_mask_stack"
    stack.mkdir(parents=True)
    labels = np.zeros((8, 8), np.uint16)
    labels[2:4, 2:4] = 1
    np.save(stack / "plate1_A01_1.npy", labels)
    np.save(stack / "other.npy", labels)
    found = _image_feature_masks((("1", str(image)),), "plate1", "A01", "1", (8, 8))
    assert list(found) == ["nucleus"] and found["nucleus"].sum() == 4
    assert _image_feature_masks((("1", str(image)),), "plate1", "A01", "1", (9, 9)) == {}


def test_channels_without_finite_pixels_are_refused():
    with pytest.raises(ValueError, match="no finite"):
        _image_level_features({0: np.full((4, 4), np.nan)}, confluency=False)


def test_flat_channels_have_undefined_correlation_and_no_objects():
    flat = np.full((16, 16), 3, np.uint16)
    record = _image_level_features({0: flat, 1: flat.copy()}, confluency=False)
    assert np.isnan(record["channels_0_1_pearson"])
    assert record["channel_0_objects_count"] == 0
    assert np.isnan(record["channel_0_objects_area_mean"])
    assert np.isnan(record["channel_0_objects_intensity_mean"])
    assert np.isnan(record["channel_0_otsu_threshold"])


def test_empty_masks_summarise_to_zero_objects():
    record = _image_level_features({0: _two_squares(16, dtype=np.uint16)},
                                   {"cell": np.zeros((16, 16), np.int32)}, confluency=False)
    assert record["cell_objects_count"] == 0
    assert np.isnan(record["cell_objects_area_mean"])
    assert np.isnan(record["channel_0_cell_object_intensity_mean"])


def test_tiff_masks_are_read(tmp_path):
    import tifffile
    image = tmp_path / "f.tif"
    stack = tmp_path / "masks" / "cell_mask_stack"
    stack.mkdir(parents=True)
    tifffile.imwrite(stack / "f.tif", np.ones((6, 6), np.uint16))
    found = _image_feature_masks((("1", str(image)), ("2", str(image))), "p", "A01", "1", (6, 6))
    assert list(found) == ["cell"] and int(found["cell"].sum()) == 36
