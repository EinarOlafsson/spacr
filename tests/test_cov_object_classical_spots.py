"""Direct tests of ``spacr.object._spots_dog``: the empty result and the sigma settings.

The rest of the classical spots block is tested in
``tests/test_object_classical_segmentation.py``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared synthetic data helpers (module level so the multiprocessing Pool
# path can pickle whatever it needs).
# ---------------------------------------------------------------------------

def _classical_settings(**over):
    """A complete classical-segmentation settings dict (spots/otsu by default)."""
    s = {
        "organelle_morphology": "spots",
        "organelle_method": "otsu",
        "organelle_tophat_radius": 5,
        "organelle_watershed_spots": False,
        "organelle_min_area": 4,
        "organelle_adaptive_block_size": 11,
        "organelle_adaptive_offset": 0.0,
        "organelle_log_min_sigma": 1.0,
        "organelle_log_max_sigma": 5.0,
        "organelle_log_num_sigma": 5,
        "organelle_log_threshold": 0.1,
        "organelle_dog_sigma_low": 1.0,
        "organelle_dog_sigma_high": 3.0,
        "organelle_morph_radius": 2,
        "organelle_fill_holes": 0,
        "organelle_skeletonize": False,
        "organelle_ridge_sigmas": [1, 2],
        "organelle_ridge_filter": "frangi",
        "organelle_network_threshold": "otsu",
        "organelle_hysteresis_low": 0.2,
        "organelle_hysteresis_high": 0.6,
        "organelle_ring_sigma_inner": 1.0,
        "organelle_ring_sigma_outer": 3.0,
        "organelle_ring_min_prominence": 0.05,
        "organelle_ring_fill_method": "flood",
    }
    s.update(over)
    return s


def _disks(shape, centers, radius, value=40000, bg=100):
    """Flat uint16 background with bright filled disks at ``centers``."""
    img = np.full(shape, bg, dtype=np.uint16)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = value
    return img


# ---------------------------------------------------------------------------
# _spots_log / _spots_dog called directly
# ---------------------------------------------------------------------------

def test_spots_dog_empty_result_is_zeros_of_input_shape():
    from spacr.object import _spots_dog

    out = _spots_dog(np.full((24, 20), 3, dtype=np.uint16),
                     _classical_settings(), use_watershed=False)

    assert out.shape == (24, 20)
    assert out.dtype == np.int32
    assert out.max() == 0


def test_spots_dog_respects_sigma_settings(monkeypatch):
    """The DoG sigmas come from settings (with documented defaults)."""
    import spacr.object as O

    seen = {}

    def _fake_blob_dog(image, min_sigma, max_sigma, threshold):
        seen["min_sigma"] = min_sigma
        seen["max_sigma"] = max_sigma
        seen["threshold"] = threshold
        return np.empty((0, 3))

    monkeypatch.setattr(O, "blob_dog", _fake_blob_dog)

    out = O._spots_dog(_disks((32, 32), [(16, 16)], 4),
                       {"organelle_log_threshold": 0.33},
                       use_watershed=False)

    # No sigma keys present -> the function's own defaults are used.
    assert seen == {"min_sigma": 1.0, "max_sigma": 3.0, "threshold": 0.33}
    assert out.shape == (32, 32) and out.max() == 0
