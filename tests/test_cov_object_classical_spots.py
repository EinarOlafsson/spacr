"""CPU-only coverage for the classical "spots" segmentation block of
``spacr.object`` plus the U-Net semantic-segmentation helper.

Covered symbols:
    _segment_unet, _segment_classical_parallel, _segment_single_image,
    _segment_spots, _spots_log, _spots_dog, _blobs_to_labels, _circle_coords

Everything here is deterministic: synthetic disks on a flat background and
constant-output stand-in "U-Nets" so every assertion is an exact pixel count
or an exact label set rather than a smoke check.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


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
        "organelle_min_size": 4,
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


def _spot_batch(n=3, shape=(32, 32)):
    return np.stack([_disks(shape, [(8, 8), (24, 24)], 4) for _ in range(n)])


def _n_labels(labeled):
    vals = np.unique(labeled)
    return int((vals != 0).sum())


def _const_unet(logits):
    """A stand-in U-Net that ignores its input and returns fixed logits.

    Records every tensor it was fed in ``.seen`` so tests can assert what the
    normalisation step actually handed the network. Carries one real
    ``nn.Parameter`` because ``_segment_unet`` does
    ``next(model.parameters()).device``.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _ConstUNet(nn.Module):
        def __init__(self, arr):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.register_buffer("fixed", torch.from_numpy(arr))
            self.seen = []

        def forward(self, x):
            self.seen.append(x.detach().cpu().numpy().copy())
            return self.fixed.clone()

    return _ConstUNet(np.ascontiguousarray(logits, dtype=np.float32))


# ---------------------------------------------------------------------------
# _segment_unet
# ---------------------------------------------------------------------------

def test_segment_unet_multichannel_prediction_uses_first_channel():
    """A (B, 2, H, W) prediction must be sliced down to channel 0.

    Channel 0 is strongly positive and channel 1 strongly negative, so the
    whole field is foreground iff the slice happened.
    """
    from spacr.object import _segment_unet

    h = w = 8
    logits = np.stack([np.full((h, w), 8.0), np.full((h, w), -8.0)])[None]
    model = _const_unet(logits)
    img = np.random.default_rng(0).random((1, h, w)).astype(np.float32)

    masks = _segment_unet(img, model, {"organelle_min_size": 1})

    assert len(masks) == 1
    assert masks[0].shape == (h, w)
    # Whole frame is one label -> channel 0 (positive) was the one used.
    assert np.array_equal(np.unique(masks[0]), np.array([1]))


def test_segment_unet_threshold_setting_controls_foreground_area():
    """``organelle_unet_threshold`` is applied to the sigmoid probability."""
    from spacr.object import _segment_unet

    lo = float(np.log(0.3 / 0.7))   # sigmoid -> 0.3
    hi = float(np.log(0.7 / 0.3))   # sigmoid -> 0.7
    arr = np.full((8, 8), lo)
    arr[:, :4] = hi
    logits = arr[None, None]
    img = np.random.default_rng(1).random((1, 8, 8)).astype(np.float32)

    default = _segment_unet(img, _const_unet(logits), {"organelle_min_size": 1})
    lowered = _segment_unet(
        img, _const_unet(logits),
        {"organelle_min_size": 1, "organelle_unet_threshold": 0.2},
    )

    assert int((default[0] > 0).sum()) == 32     # only the p=0.7 half
    assert int((lowered[0] > 0).sum()) == 64     # both halves clear 0.2


def test_segment_unet_flat_image_is_zeroed_not_divided_by_zero():
    """std == 0 must take the ``np.zeros_like`` branch, never 0/0 -> NaN."""
    from spacr.object import _segment_unet

    img = np.full((1, 8, 8), 7.0, dtype=np.float32)
    model = _const_unet(np.full((1, 1, 8, 8), -6.0))

    masks = _segment_unet(img, model, {"organelle_min_size": 1})

    seen = model.seen[0]
    assert seen.shape == (1, 1, 8, 8)
    assert np.isfinite(seen).all()
    assert (seen == 0).all()
    # sigmoid(-6) ~ 0.0025 -> nothing passes the 0.5 threshold.
    assert masks[0].max() == 0


def test_segment_unet_skeletonize_thins_the_mask():
    """``organelle_skeletonize`` must shrink the foreground of a thick band."""
    from spacr.object import _segment_unet

    h = w = 16
    arr = np.full((h, w), -8.0)
    arr[4:13, :] = 8.0                       # 9-row band across the frame
    logits = arr[None, None]
    img = np.random.default_rng(2).random((1, h, w)).astype(np.float32)

    plain = _segment_unet(img, _const_unet(logits),
                          {"organelle_min_size": 1})[0]
    skel = _segment_unet(img, _const_unet(logits),
                         {"organelle_min_size": 1,
                          "organelle_skeletonize": True})[0]

    assert int((plain > 0).sum()) == 9 * 16
    assert 0 < int((skel > 0).sum()) < int((plain > 0).sum())
    assert plain.max() == 1 and skel.max() == 1


def test_segment_unet_min_size_removes_small_predictions():
    """remove_small_objects is driven by ``organelle_min_size``."""
    from spacr.object import _segment_unet

    arr = np.full((12, 12), -8.0)
    arr[1:3, 1:3] = 8.0                      # a 4-pixel blob
    logits = arr[None, None]
    img = np.random.default_rng(3).random((1, 12, 12)).astype(np.float32)

    kept = _segment_unet(img, _const_unet(logits), {"organelle_min_size": 4})[0]
    dropped = _segment_unet(img, _const_unet(logits), {"organelle_min_size": 5})[0]

    assert int((kept > 0).sum()) == 4
    assert int((dropped > 0).sum()) == 0


def test_segment_unet_empty_batch_returns_empty_list():
    from spacr.object import _segment_unet

    model = _const_unet(np.zeros((1, 1, 4, 4), dtype=np.float32))
    masks = _segment_unet(np.zeros((0, 4, 4), dtype=np.float32), model,
                          {"organelle_min_size": 1})

    assert masks == []
    assert model.seen == []          # the model was never invoked


# ---------------------------------------------------------------------------
# _segment_classical_parallel
# ---------------------------------------------------------------------------

def test_segment_classical_parallel_pool_matches_serial():
    """The multiprocessing branch must produce byte-identical masks."""
    from spacr.object import _segment_classical_parallel

    batch = _spot_batch(3)
    settings = _classical_settings(organelle_tophat_radius=6)

    serial = _segment_classical_parallel(batch, settings, n_jobs=1)
    parallel = _segment_classical_parallel(batch, settings, n_jobs=2)

    assert len(serial) == 3 and len(parallel) == 3
    assert _n_labels(serial[0]) == 2          # the two synthetic disks
    for a, b in zip(serial, parallel):
        assert np.array_equal(a, b)


def test_segment_classical_parallel_single_image_never_spawns_a_pool(monkeypatch):
    """n_images == 1 short-circuits to the serial branch even with n_jobs > 1."""
    import spacr.object as O

    def _boom(*args, **kwargs):
        raise AssertionError("Pool must not be used for a single image")

    monkeypatch.setattr(O, "Pool", _boom)

    batch = _spot_batch(1)
    out = O._segment_classical_parallel(
        batch, _classical_settings(organelle_tophat_radius=6), n_jobs=8)

    assert len(out) == 1
    assert out[0].shape == (32, 32)
    assert _n_labels(out[0]) == 2


def test_segment_classical_parallel_njobs_one_never_spawns_a_pool(monkeypatch):
    import spacr.object as O

    def _boom(*args, **kwargs):
        raise AssertionError("Pool must not be used when n_jobs == 1")

    monkeypatch.setattr(O, "Pool", _boom)

    batch = _spot_batch(3)
    out = O._segment_classical_parallel(
        batch, _classical_settings(organelle_tophat_radius=6), n_jobs=1)

    assert len(out) == 3
    assert all(_n_labels(m) == 2 for m in out)


def test_segment_classical_parallel_caps_workers_at_image_count(monkeypatch):
    """effective_jobs = min(n_jobs, n_images, cpu_count())."""
    import spacr.object as O

    recorded = {}

    class _FakePool:
        def __init__(self, processes):
            recorded["processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items):
            return [fn(i) for i in items]

    monkeypatch.setattr(O, "Pool", _FakePool)

    batch = _spot_batch(2)
    out = O._segment_classical_parallel(
        batch, _classical_settings(organelle_tophat_radius=6), n_jobs=64)

    assert recorded["processes"] == 2          # clamped to the 2 images
    assert len(out) == 2
    assert all(_n_labels(m) == 2 for m in out)


# ---------------------------------------------------------------------------
# _segment_single_image
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "morphology,target",
    [
        ("spots", "_segment_spots"),
        ("network", "_segment_network"),
        ("irregular", "_segment_irregular"),
        ("ring", "_segment_ring"),
    ],
)
def test_segment_single_image_dispatches_to_expected_routine(
        monkeypatch, morphology, target):
    import spacr.object as O

    sentinel = np.full((4, 4), 77, dtype=np.int32)
    calls = {}

    def _fake(img, method, settings):
        calls["method"] = method
        calls["morphology"] = settings["organelle_morphology"]
        calls["shape"] = img.shape
        return sentinel

    monkeypatch.setattr(O, target, _fake)

    img = np.zeros((4, 4), dtype=np.uint16)
    out = O._segment_single_image(
        img, _classical_settings(organelle_morphology=morphology,
                                 organelle_method="otsu"))

    assert out is sentinel
    assert calls == {"method": "otsu", "morphology": morphology,
                     "shape": (4, 4)}


def test_segment_single_image_unknown_morphology_raises():
    from spacr.object import _segment_single_image

    with pytest.raises(ValueError, match="Unknown morphology: blobby"):
        _segment_single_image(
            np.zeros((4, 4), dtype=np.uint16),
            _classical_settings(organelle_morphology="blobby"))


# ---------------------------------------------------------------------------
# _segment_network early dispatch (the two `return` lines that sit at the tail
# of this chunk's line range; the ridge/hysteresis internals live elsewhere).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method,target",
    [("ridge", "_network_ridge"), ("hysteresis", "_network_hysteresis")],
)
def test_segment_network_delegates_before_thresholding(monkeypatch, method, target):
    import spacr.object as O

    sentinel = np.full((6, 6), 5, dtype=np.int32)
    calls = {}

    def _fake(img, settings):
        calls["shape"] = img.shape
        calls["filter"] = settings["organelle_ridge_filter"]
        return sentinel

    monkeypatch.setattr(O, target, _fake)
    # If dispatch failed we would fall through to the otsu path, which would
    # neither return the sentinel nor record a call.
    monkeypatch.setattr(O, "threshold_otsu",
                        lambda *a, **k: pytest.fail("fell through to otsu"))

    out = O._segment_network(_disks((6, 6), [(3, 3)], 1), method,
                             _classical_settings())

    assert out is sentinel
    assert calls == {"shape": (6, 6), "filter": "frangi"}


# ---------------------------------------------------------------------------
# _segment_spots
# ---------------------------------------------------------------------------

def test_segment_spots_otsu_labels_every_disk():
    from spacr.object import _segment_spots

    img = _disks((48, 48), [(10, 10), (10, 36), (36, 10)], radius=4)
    out = _segment_spots(img, "otsu",
                         _classical_settings(organelle_tophat_radius=6,
                                             organelle_min_size=4))

    assert out.shape == (48, 48)
    assert out.dtype.kind in "iu"
    assert _n_labels(out) == 3
    for cy, cx in ((10, 10), (10, 36), (36, 10)):
        assert out[cy, cx] != 0


def test_segment_spots_watershed_splits_touching_disks():
    """The watershed branch must split a merged blob the label pass leaves as one."""
    from spacr.object import _segment_spots

    img = _disks((40, 40), [(20, 14), (20, 24)], radius=7)
    base = _classical_settings(organelle_tophat_radius=10,
                               organelle_min_size=10)

    merged = _segment_spots(img, "otsu",
                            {**base, "organelle_watershed_spots": False})
    split = _segment_spots(img, "otsu",
                           {**base, "organelle_watershed_spots": True})

    assert _n_labels(merged) == 1
    assert _n_labels(split) == 2
    assert split[20, 14] != split[20, 24]


def test_segment_spots_adaptive_keeps_disk_centres():
    from spacr.object import _segment_spots

    img = _disks((48, 48), [(12, 12), (12, 34), (34, 12)], radius=4)
    out = _segment_spots(
        img, "adaptive",
        _classical_settings(organelle_method="adaptive",
                            organelle_tophat_radius=6,
                            organelle_adaptive_block_size=11,
                            organelle_adaptive_offset=0.0))

    assert out.shape == (48, 48)
    assert _n_labels(out) >= 1
    assert out[12, 12] != 0
    assert out[34, 12] != 0
    # The flat background produces a zero top-hat, so it stays background.
    assert out[0, 0] == 0


def test_segment_spots_log_paints_a_circle_at_each_blob():
    from spacr.object import _segment_spots

    centres = [(16, 16), (16, 48), (48, 16), (48, 48)]
    img = _disks((64, 64), centres, radius=3)
    out = _segment_spots(img, "log", _classical_settings())

    assert out.shape == (64, 64)
    assert _n_labels(out) == 4
    for cy, cx in centres:
        assert out[cy, cx] != 0


def test_segment_spots_dog_flat_image_returns_int32_zeros():
    from spacr.object import _segment_spots

    flat = np.full((32, 32), 500, dtype=np.uint16)
    out = _segment_spots(flat, "dog", _classical_settings())

    assert out.shape == (32, 32)
    assert out.dtype == np.int32
    assert (out == 0).all()


def test_segment_spots_log_flat_image_returns_int32_zeros():
    from spacr.object import _segment_spots

    flat = np.full((32, 32), 500, dtype=np.uint16)
    out = _segment_spots(flat, "log", _classical_settings())

    assert out.dtype == np.int32
    assert (out == 0).all()


def test_segment_spots_dog_detects_blobs():
    from spacr.object import _segment_spots

    centres = [(16, 16), (16, 48), (48, 16), (48, 48)]
    img = _disks((64, 64), centres, radius=3)
    out = _segment_spots(img, "dog",
                         _classical_settings(organelle_dog_sigma_low=1.0,
                                             organelle_dog_sigma_high=4.0,
                                             organelle_log_threshold=0.1))

    assert _n_labels(out) >= 1
    assert out[16, 16] != 0


def test_segment_spots_unsupported_method_raises():
    from spacr.object import _segment_spots

    with pytest.raises(ValueError, match="Unsupported spot method: sobel"):
        _segment_spots(_disks((16, 16), [(8, 8)], 3), "sobel",
                       _classical_settings())


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


def test_spots_log_empty_result_is_zeros_of_input_shape():
    from spacr.object import _spots_log

    out = _spots_log(np.full((24, 20), 3, dtype=np.uint16),
                     _classical_settings(), use_watershed=False)

    assert out.shape == (24, 20)
    assert out.dtype == np.int32
    assert out.max() == 0


def test_spots_log_watershed_grows_regions_beyond_the_circles():
    from spacr.object import _spots_log

    img = _disks((64, 64), [(16, 16), (48, 48)], radius=4)
    settings = _classical_settings()

    circles = _spots_log(img, settings, use_watershed=False)
    grown = _spots_log(img, settings, use_watershed=True)

    assert _n_labels(circles) == 2
    assert _n_labels(grown) == 2
    # Watershed floods the whole intensity mask, the circle painter does not.
    assert int((grown > 0).sum()) > int((circles > 0).sum())


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


# ---------------------------------------------------------------------------
# _blobs_to_labels
# ---------------------------------------------------------------------------

def test_blobs_to_labels_paints_discs_sized_by_sigma():
    """Radius is ``round(sigma * sqrt(2))`` -> exact pixel counts."""
    from spacr.object import _blobs_to_labels

    img_norm = np.zeros((32, 32), dtype=np.float64)
    blobs = np.array([[10.0, 10.0, 2.0], [20.0, 20.0, 3.0]])

    out = _blobs_to_labels(blobs, img_norm, use_watershed=False)

    assert out.shape == (32, 32)
    assert int((out == 1).sum()) == 29    # radius 3 filled circle
    assert int((out == 2).sum()) == 49    # radius 4 filled circle
    assert out[10, 10] == 1 and out[20, 20] == 2


def test_blobs_to_labels_tiny_sigma_still_paints_at_least_radius_one():
    from spacr.object import _blobs_to_labels

    img_norm = np.zeros((16, 16), dtype=np.float64)
    out = _blobs_to_labels(np.array([[8.0, 8.0, 0.1]]), img_norm,
                           use_watershed=False)

    # max(round(0.1*sqrt(2)), 1) == 1 -> a 5-pixel plus shape.
    assert int((out == 1).sum()) == 5
    assert set(zip(*np.where(out == 1))) == {
        (7, 8), (8, 7), (8, 8), (8, 9), (9, 8)}


def test_blobs_to_labels_out_of_bounds_blob_is_clipped_to_the_corner():
    from spacr.object import _blobs_to_labels

    img_norm = np.zeros((16, 16), dtype=np.float64)
    blobs = np.array([[5.0, 5.0, 1.0], [50.0, 50.0, 1.0]])

    out = _blobs_to_labels(blobs, img_norm, use_watershed=False)

    assert int((out == 1).sum()) == 5
    assert set(zip(*np.where(out == 2))) == {(15, 15)}


def test_blobs_to_labels_watershed_seeds_one_label_per_blob():
    from spacr.object import _blobs_to_labels

    img_norm = np.zeros((32, 32), dtype=np.float64)
    yy, xx = np.mgrid[:32, :32]
    for cy, cx in ((8, 8), (24, 24)):
        img_norm[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] = 1.0

    out = _blobs_to_labels(np.array([[8.0, 8.0, 2.0], [24.0, 24.0, 2.0]]),
                           img_norm, use_watershed=True)

    assert out.shape == (32, 32)
    assert _n_labels(out) == 2
    assert out[8, 8] == 1
    assert out[24, 24] == 2


def test_blobs_to_labels_watershed_skips_out_of_bounds_markers():
    """A blob outside the frame must not seed a marker in the watershed path."""
    from spacr.object import _blobs_to_labels

    img_norm = np.zeros((32, 32), dtype=np.float64)
    yy, xx = np.mgrid[:32, :32]
    img_norm[(yy - 8) ** 2 + (xx - 8) ** 2 <= 16] = 1.0

    out = _blobs_to_labels(np.array([[8.0, 8.0, 2.0], [99.0, 99.0, 2.0]]),
                           img_norm, use_watershed=True)

    assert 2 not in np.unique(out)
    assert out[8, 8] == 1


# ---------------------------------------------------------------------------
# _circle_coords
# ---------------------------------------------------------------------------

def test_circle_coords_radius_one_is_a_plus_shape():
    from spacr.object import _circle_coords

    rows, cols = _circle_coords(10, 10, 1, (32, 32))

    assert len(rows) == len(cols) == 5
    assert set(zip(rows.tolist(), cols.tolist())) == {
        (9, 10), (10, 9), (10, 10), (10, 11), (11, 10)}


def test_circle_coords_clamps_instead_of_dropping_out_of_range_pixels():
    from spacr.object import _circle_coords

    rows, cols = _circle_coords(0, 0, 2, (20, 20))

    # The radius-2 disc has 13 offsets; clipping folds them onto 6 pixels.
    assert len(rows) == len(cols) == 13
    assert rows.min() == 0 and cols.min() == 0
    assert set(zip(rows.tolist(), cols.tolist())) == {
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}


def test_circle_coords_clamps_at_the_far_border():
    from spacr.object import _circle_coords

    rows, cols = _circle_coords(19, 19, 2, (20, 20))

    assert rows.max() == 19 and cols.max() == 19
    assert set(zip(rows.tolist(), cols.tolist())) == {
        (17, 19), (18, 18), (18, 19), (19, 17), (19, 18), (19, 19)}
