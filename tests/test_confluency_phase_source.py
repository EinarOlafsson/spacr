"""The learned phase-contrast confluency source.

``confluency_source='phase'`` classifies every pixel with a small perceptron
over multi-scale texture features, trained on LIVECell phase-contrast fields.
These synthetic fields imitate phase contrast -- granular cell bodies, a dark
rim, a bright halo and bright cell-cell walls on flat plastic -- with an
exactly known covered area, and the source is held to the 5-point agreement
the confluency brief asks for. A field rescaled two-fold is read the same
when ``confluency_window`` is scaled with it, bare plastic reads empty, and
the source runs end to end through Measure.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from skimage.draw import ellipse

from spacr import measure
from spacr.measure import (
    _CONFLUENCY_TABLE,
    _confluency_phase_features,
    _confluency_phase_network,
    _field_confluency,
    _phase_coverage,
    _resolve_confluency_source,
)

SHAPE = (384, 384)
TOLERANCE = 0.05


def _phase_field(fraction, seed=777, shape=SHAPE):
    """Phase-contrast-like cells covering about ``fraction`` of the field."""
    rng = np.random.default_rng(seed)
    covered = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    count = 0
    while covered.mean() < fraction and count < 20000:
        count += 1
        rr, cc = ellipse(rng.integers(0, shape[0]), rng.integers(0, shape[1]),
                         rng.uniform(12, 26), rng.uniform(8, 15), shape=shape,
                         rotation=rng.uniform(0, np.pi))
        covered[rr, cc] = True
        labels[rr, cc] = count
    granules = sum(ndi.gaussian_filter(rng.standard_normal(shape), s) * s
                   for s in (1.0, 3.0, 6.0))
    granules /= granules.std()
    halo = ndi.binary_dilation(covered, iterations=1) & ~covered
    rim = covered & ~ndi.binary_erosion(covered, iterations=2)
    walls = (ndi.maximum_filter(labels, 3)
             != ndi.minimum_filter(labels, 3)) & covered
    image = (110.0 + rng.normal(0.0, 2.0, shape) + covered * 12.0 * granules
             - 20.0 * rim + 25.0 * halo + 20.0 * walls)
    image = np.clip(ndi.gaussian_filter(image, 0.6), 0, 255)
    return image.astype(np.uint8), covered


@pytest.mark.parametrize("fraction", (0.1, 0.35, 0.6, 0.9))
def test_phase_source_recovers_known_phase_coverage(fraction):
    image, truth = _phase_field(fraction)
    result = _phase_coverage(image)
    assert result.source == "phase" and not result.uniform
    assert abs(result.confluency - truth.mean()) <= TOLERANCE


def test_bare_plastic_and_a_flat_field_read_empty():
    rng = np.random.default_rng(4)
    bare = np.clip(128.0 + rng.normal(0.0, 2.0, SHAPE), 0, 255)
    assert _phase_coverage(bare.astype(np.uint8)).confluency <= 0.02
    flat = _phase_coverage(np.full((64, 64), 7, dtype=np.uint16))
    assert flat.confluency == 0.0 and flat.uniform


def test_the_window_rescales_a_field_imaged_at_a_finer_pixel_size():
    image, truth = _phase_field(0.35)
    finer = ndi.zoom(image, 2, order=1)
    result = _phase_coverage(finer, window=30)
    assert result.covered.shape == finer.shape
    assert abs(result.confluency - truth.mean()) <= TOLERANCE


def test_the_network_reads_every_feature_plane_it_is_given():
    layers = _confluency_phase_network()
    features = _confluency_phase_features(np.random.default_rng(0).random(
        (48, 48)))
    assert layers[0][0].shape[0] == features.shape[-1]
    for (weights, bias), (following, _) in zip(layers, layers[1:]):
        assert weights.shape[1] == bias.size == following.shape[0]
    assert layers[-1][0].shape[1] == 1
    assert all(np.isfinite(w).all() and np.isfinite(b).all()
               for w, b in layers)


def test_phase_is_a_source_the_settings_and_dispatch_accept():
    assert _resolve_confluency_source(
        {"confluency_source": "phase", "cell_mask_dim": None}) == "phase"
    image, _truth = _phase_field(0.35)
    result = _field_confluency(image, source="phase", channel=0)
    assert result.source == "phase" and result.channel == 0


def test_measure_writes_phase_confluency_per_field(tmp_path):
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / "merged"
    merged.mkdir()
    truths = {}
    for number, fraction in enumerate((0.2, 0.7), start=1):
        image, truth = _phase_field(fraction, seed=900 + number)
        labels, _count = ndi.label(truth)
        name = f"plate1_A01_{number}"
        np.save(merged / f"{name}.npy",
                np.stack([image.astype(np.uint16),
                          labels.astype(np.uint16)], axis=-1))
        truths[name] = truth
    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0], "cell_mask_dim": 1,
        "nucleus_mask_dim": None, "pathogen_mask_dim": None,
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1, "confluency": True,
        "confluency_source": "phase", "confluency_channel": 0,
    })
    measure.measure_crop(settings)

    db = tmp_path / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        fields = pd.read_sql_query(f"SELECT * FROM {_CONFLUENCY_TABLE}", conn)
    assert set(fields["confluency_source"]) == {"phase"}
    assert len(fields) == 2
    for _, row in fields.iterrows():
        assert abs(row["confluency"] - truths[row["file_name"]].mean()) <= (
            TOLERANCE)
