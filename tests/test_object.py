"""
Tests for spacr.object — segmentation helpers that don't require GPU
or trained model weights.

Cellpose / stardist / U-Net paths are gated behind GPU + weight downloads
and are excluded here.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacr.object as O


# ---------------------------------------------------------------------------
# _validate_organelle_settings — raises on bad combos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("morph,method", [
    ("spots", "log"),
    ("spots", "dog"),
    ("spots", "otsu"),
    ("network", "hysteresis"),
    ("network", "ridge"),
    ("irregular", "otsu"),
    ("ring", "log"),
])
def test_validate_organelle_settings_accepts_valid_combos(morph, method):
    # Should not raise.
    O._validate_organelle_settings(morph, method)


def test_validate_organelle_settings_rejects_unknown_morphology():
    with pytest.raises(ValueError) as excinfo:
        O._validate_organelle_settings("mesh", "otsu")
    assert "organelle_morphology" in str(excinfo.value)


def test_validate_organelle_settings_rejects_method_for_morphology():
    # 'log' is only valid for 'spots' / 'ring', not for 'network'.
    with pytest.raises(ValueError) as excinfo:
        O._validate_organelle_settings("network", "log")
    assert "morphology='network'" in str(excinfo.value)


def test_validate_organelle_settings_rejects_stardist():
    """Stardist was removed to avoid the TensorFlow dependency; it must
    not be silently accepted as a segmentation method anywhere."""
    for morph in ("spots", "network", "irregular", "ring"):
        with pytest.raises(ValueError):
            O._validate_organelle_settings(morph, "stardist")


# ---------------------------------------------------------------------------
# _build_object_settings — dict extraction
# ---------------------------------------------------------------------------

def test_build_object_settings_uses_all_expected_keys():
    src = {
        "organelle_model_name": "cyto3",
        "organelle_diameter": 30,
        "organelle_min_size": 12,
        "organelle_max_size": 500,
        "organelle_resample": False,
        "organelle_remove_border": True,
    }
    out = O._build_object_settings(src)
    assert out["model_name"] == "cyto3"
    assert out["diameter"] == 30
    assert out["minimum_size"] == 12
    assert out["maximum_size"] == 500
    assert out["resample"] is False
    assert out["remove_border_objects"] is True
    # These are hard-coded to False in the builder.
    assert out["filter_size"] is False
    assert out["filter_intensity"] is False
    assert out["merge"] is False


def test_build_object_settings_missing_key_raises():
    """Underspecified settings should raise KeyError, not silently succeed."""
    with pytest.raises(KeyError):
        O._build_object_settings({"organelle_model_name": "cyto3"})


# ---------------------------------------------------------------------------
# _extract_classical_settings — subset with graceful missing keys
# ---------------------------------------------------------------------------

def test_extract_classical_settings_keeps_only_known_keys():
    src = {
        "organelle_morphology": "spots",
        "organelle_method": "log",
        "organelle_log_min_sigma": 1.0,
        "organelle_log_max_sigma": 5.0,
        # Non-classical keys that should be dropped:
        "unrelated_setting": "should_be_dropped",
        "organelle_model_name": "cyto3",  # not in the classical whitelist
    }
    out = O._extract_classical_settings(src)
    assert "unrelated_setting" not in out
    assert "organelle_model_name" not in out
    assert out["organelle_morphology"] == "spots"
    assert out["organelle_log_min_sigma"] == 1.0


def test_extract_classical_settings_missing_keys_silently_dropped():
    """The extractor should not raise when settings is sparse — just
    return whatever keys are actually present."""
    out = O._extract_classical_settings({"organelle_morphology": "network"})
    assert out == {"organelle_morphology": "network"}


# ---------------------------------------------------------------------------
# _segment_spots — classical (otsu) path on synthetic blobs
# ---------------------------------------------------------------------------

def _spot_settings(**overrides):
    base = {
        "organelle_tophat_radius": 3,
        "organelle_watershed_spots": False,
        "organelle_min_size": 5,
        "organelle_adaptive_block_size": 51,
        "organelle_adaptive_offset": -0.01,
        "organelle_log_min_sigma": 1.5,
        "organelle_log_max_sigma": 6.0,
        "organelle_log_num_sigma": 5,
        "organelle_log_threshold": 0.02,
        "organelle_dog_sigma_low": 1.5,
        "organelle_dog_sigma_high": 4.0,
    }
    base.update(overrides)
    return base


def test_segment_spots_otsu_returns_valid_label_image(synth_image_2d):
    """The otsu path must return a valid int label image of the same shape
    without raising. Detection count depends on the tophat radius vs the
    synthetic blob size, so we don't hard-assert count > 0."""
    labeled = O._segment_spots(synth_image_2d, "otsu", _spot_settings())
    assert labeled.dtype.kind in "iu"
    assert labeled.shape == synth_image_2d.shape
    assert (labeled >= 0).all()


def test_segment_spots_otsu_finds_blobs_with_matched_tophat(synth_image_2d):
    """With a tophat radius that matches synthetic blob size (~10 px), the
    otsu path should find at least one blob."""
    labeled = O._segment_spots(
        synth_image_2d,
        "otsu",
        _spot_settings(organelle_tophat_radius=15),
    )
    n_labels = len(np.unique(labeled)) - (1 if 0 in np.unique(labeled) else 0)
    assert n_labels >= 1, "expected ≥1 blob with tophat radius matched to blob size"


def test_segment_spots_log_returns_zero_when_no_blobs():
    """A flat image has nothing to detect."""
    flat = np.full((128, 128), 100, dtype=np.uint16)
    labeled = O._segment_spots(flat, "log", _spot_settings())
    assert labeled.shape == flat.shape
    assert (labeled == 0).all()


def test_segment_spots_unknown_method_raises(synth_image_2d):
    with pytest.raises(ValueError):
        O._segment_spots(synth_image_2d, "not_a_method", _spot_settings())


# ---------------------------------------------------------------------------
# _blobs_to_labels — coordinate → label conversion
# ---------------------------------------------------------------------------

def test_blobs_to_labels_places_markers_at_coords():
    img_norm = np.zeros((32, 32), dtype=np.float32)
    blobs = np.array([[10, 10, 2.0], [20, 15, 3.0]])
    out = O._blobs_to_labels(blobs, img_norm, use_watershed=False)
    assert out.shape == img_norm.shape
    # Two labels expected.
    unique = np.unique(out)
    unique = unique[unique != 0]
    assert len(unique) == 2


def test_blobs_to_labels_ignores_out_of_bounds():
    img_norm = np.zeros((16, 16), dtype=np.float32)
    # Second blob is outside the image; must not cause an IndexError.
    blobs = np.array([[5, 5, 1.0], [50, 50, 1.0]])
    out = O._blobs_to_labels(blobs, img_norm, use_watershed=False)
    assert out.shape == img_norm.shape
