"""Numerical examples behind Make Masks' local-threshold parameter help."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import mask_engine as engine


def _field(scale=1.0):
    return (np.random.default_rng(47).uniform(0.2, 0.8, (31, 31))
            * scale).astype(np.float32)


@pytest.mark.parametrize("bright", [True, False])
def test_increasing_niblack_k_expands_bright_and_contracts_dark_foreground(bright):
    image = _field()
    masks = [engine._otsu_instances(image, algorithm="niblack", local_k=k,
                                   window=7, bright=bright) > 0
             for k in (-0.2, 0.2)]
    smaller, larger = masks if bright else masks[::-1]
    assert np.all(~smaller | larger)
    assert smaller.sum() < larger.sum()


def test_sauvola_can_label_an_entire_constant_positive_background():
    image = np.full((15, 15), 0.5, dtype=np.float32)
    levels = engine._local_level_map(image, "sauvola", window=7, k=0.2)
    np.testing.assert_allclose(levels, 0.4)
    assert np.all(engine._otsu_instances(
        image, algorithm="sauvola", local_k=0.2, window=7) > 0)
    assert not engine._otsu_instances(
        image, algorithm="sauvola", local_k=0, window=7).any()


def test_sauvola_k_direction_depends_on_intensity_scale_because_r_is_one():
    for scale, increasing in ((1.0, False), (100.0, True)):
        image = _field(scale)
        low = engine._local_level_map(image, "sauvola", window=7, k=0.1)
        high = engine._local_level_map(image, "sauvola", window=7, k=0.3)
        assert np.all(high > low) if increasing else np.all(high < low)


@pytest.mark.parametrize("algorithm", ["sauvola", "niblack"])
@pytest.mark.parametrize("bright", [True, False])
def test_local_correction_factor_direction_for_positive_thresholds(algorithm, bright):
    image = _field()
    masks = [engine._otsu_instances(image, algorithm=algorithm, local_k=0.2,
                                   correction=factor, window=7, bright=bright) > 0
             for factor in (0.9, 1.1)]
    smaller, larger = masks[::-1] if bright else masks
    assert np.all(~smaller | larger)
    assert smaller.sum() < larger.sum()


def test_adaptive_offset_increases_foreground_in_the_real_organelle_engine():
    from spacr.qt.organelle_modes import MethodParams, segment

    image = _field()
    masks = [segment(image, "adaptive", MethodParams(
        adaptive_block=9, adaptive_offset=offset, morph_radius=0,
        fill_holes=0), min_area=0) > 0 for offset in (-0.05, 0.05)]
    assert np.all(~masks[0] | masks[1])
    assert masks[0].sum() < masks[1].sum()
