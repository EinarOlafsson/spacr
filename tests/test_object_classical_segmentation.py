"""Table-driven tests of the classical organelle segmentation back-ends in ``spacr.object``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the shared helpers (``_normalize_01``,
``_circle_coords``, ``_blobs_to_labels``, ``_watershed_split``, the ring fill
and ring filter), the U-Net semantic segmenter, the batch dispatcher and the
four morphology back-ends (spots, network, irregular, ring).  Inputs are
synthetic disks, bars and annuli with fixed seeds, so every row pins an exact
label count, foreground area and dtype rather than a smoke check.  CPU-only,
offline; the U-Net is a constant-logit double.
"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.object as O  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# shared builders (module level so the Pool path can pickle them)
# ---------------------------------------------------------------------------

def _settings(**over):
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


_FOUR = ((16, 16), (16, 48), (48, 16), (48, 48))


def _u16_disks(shape, centers, radius):
    """uint16 background of 100 with filled disks of 40000 at ``centers``."""
    img = np.full(shape, 100, dtype=np.uint16)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 40000
    return img


def _disks(radius=6, centers=_FOUR, fg=1.0, noise=0.02, size=64, seed=0):
    """float32 disks of ``fg`` over faint uniform noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.zeros((size, size), dtype=np.float32)
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = fg
    return img + rng.random((size, size)).astype(np.float32) * noise * max(fg, 1.0)


def _filaments(size=64, seed=1):
    """A cross of thin bright bars - the shape ridge filters are built for."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    img[10:54, 20:23] = 1.0
    img[30:33, 5:60] = 1.0
    return img + rng.random((size, size)).astype(np.float32) * 0.02


def _annulus(size, center, r_outer, r_inner):
    """Boolean annulus with the given outer/inner radii."""
    yy, xx = np.mgrid[:size, :size]
    d2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return (d2 <= r_outer ** 2) & (d2 > r_inner ** 2)


def _rings(size=64, seed=2):
    """Two hollow rings (bright wall, dark lumen) on a dark noisy background."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    img[_annulus(size, (18, 18), 9, 4)] = 1.0
    img[_annulus(size, (46, 46), 9, 4)] = 1.0
    return img + rng.random((size, size)).astype(np.float32) * 0.02


def _summary(mask):
    """(label count, foreground pixels, dtype name) of a label image.

    Also asserts the labels are consecutive ``0..N`` (what ``sk_label``
    guarantees), so ``max`` really is the object count.
    """
    vals = [v for v in np.unique(mask).tolist() if v != 0]
    assert vals == list(range(1, int(mask.max()) + 1)), vals
    return int(mask.max()), int((mask > 0).sum()), mask.dtype.name


def _square_ring_edges(size=32, lo=8, hi=24, gap=False):
    """One-pixel square outline; ``gap`` opens the bottom wall."""
    edges = np.zeros((size, size), dtype=bool)
    edges[lo:hi, lo] = True
    edges[lo:hi, hi - 1] = True
    edges[lo, lo:hi] = True
    edges[hi - 1, lo:(lo + hi) // 2 if gap else hi] = True
    return edges


def _const_unet(logits):
    """A stand-in U-Net returning fixed logits and recording every input."""
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


def _bias_conv(bias):
    """A real 3x3 Conv2d whose output is the constant ``bias`` everywhere."""
    torch = pytest.importorskip("torch")
    conv = torch.nn.Conv2d(1, 1, 3, padding=1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.fill_(bias)
    return conv


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def test_normalize_01_maps_the_1_99_percentile_window_onto_unit_range(subtests):
    """Values are rescaled between the 1st/99th percentiles and clipped; flat input gives zeros."""
    ramp = np.arange(100, dtype=np.uint16).reshape(10, 10)
    noisy = (np.random.default_rng(1).random((16, 16)) * 1000).astype(np.uint16)
    lo, hi = np.percentile(noisy.astype(np.float64), (1, 99))
    cases = {
        # percentiles of 0..99 are 0.99 and 98.01 -> only the two ends clip
        "ramp": (ramp, np.clip((np.arange(100) - 0.99) / 97.02, 0, 1).reshape(10, 10)),
        "noisy_uint16": (noisy, np.clip((noisy - lo) / (hi - lo), 0, 1)),
        "flat_float": (np.full((8, 8), 5.0), np.zeros((8, 8))),
        "flat_uint16": (np.full((4, 6), 7, dtype=np.uint16), np.zeros((4, 6))),
    }
    for name, (img, expected) in cases.items():
        with subtests.test(msg=name):
            out = O._normalize_01(img)
            assert out.dtype == np.float64
            np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)
    with subtests.test(msg="ramp_clips_exactly_one_pixel_per_tail"):
        out = O._normalize_01(ramp)
        assert int((out == 0).sum()) == 1 and int((out == 1).sum()) == 1
        assert np.all(np.diff(out.ravel()) >= 0)


def test_circle_coords_returns_the_filled_disc_clamped_into_the_frame(subtests):
    """Offsets of the radius-r disc are shifted to the centre and clamped (not dropped) at borders."""
    cases = {
        "radius_1_plus": ((10, 10, 1, (32, 32)), 5,
                          {(9, 10), (10, 9), (10, 10), (10, 11), (11, 10)}),
        "near_corner_folds_13_offsets_onto_6": (
            (0, 0, 2, (20, 20)), 13,
            {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}),
        "far_corner": ((19, 19, 2, (20, 20)), 13,
                       {(17, 19), (18, 18), (18, 19), (19, 17), (19, 18), (19, 19)}),
        "radius_3_interior": (
            (5, 5, 3, (16, 16)), 29,
            {(y, x) for y in range(2, 9) for x in range(2, 9)
             if (y - 5) ** 2 + (x - 5) ** 2 <= 9}),
    }
    for name, (args, n, pixels) in cases.items():
        with subtests.test(msg=name):
            rows, cols = O._circle_coords(*args)
            assert len(rows) == len(cols) == n
            assert set(zip(rows.tolist(), cols.tolist())) == pixels


def test_blobs_to_labels_paints_discs_or_grows_half_maximum_basins(subtests):
    """Without watershed each blob is a sigma*sqrt(2) disc; with it each grows to half its height, never flooding."""
    zeros32 = np.zeros((32, 32))
    two_discs = np.zeros((32, 32))
    one_disc = np.zeros((32, 32))
    yy, xx = np.mgrid[:32, :32]
    for cy, cx in ((8, 8), (24, 24)):
        two_discs[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] = 1.0
    one_disc[(yy - 8) ** 2 + (xx - 8) ** 2 <= 16] = 1.0
    disks_norm = O._normalize_01(_disks())
    # (blobs, image, use_watershed) -> {label: pixel count}, {pixel: label} probes
    cases = {
        "discs_sized_by_sigma": (
            [[10, 10, 2.0], [20, 20, 3.0]], zeros32, False,
            {1: 29, 2: 49}, {(10, 10): 1, (20, 20): 2}),
        "tiny_sigma_paints_radius_one": (
            [[8, 8, 0.1]], np.zeros((16, 16)), False,
            {1: 5}, {(7, 8): 1, (8, 7): 1, (8, 9): 1, (9, 8): 1, (6, 8): 0}),
        "out_of_frame_blob_clamped_to_corner": (
            [[5, 5, 1.0], [50, 50, 1.0]], np.zeros((16, 16)), False,
            {1: 5, 2: 1}, {(15, 15): 2}),
        "disc_reach_on_real_disks": (
            [[16, 16, 3.0], [48, 48, 3.0]], disks_norm, False,
            {1: 49, 2: 49}, {(16, 20): 1, (16, 21): 0, (48, 48): 2}),
        "watershed_one_label_per_blob": (
            [[8, 8, 2.0], [24, 24, 2.0]], two_discs, True,
            {1: 45, 2: 45}, {(8, 8): 1, (24, 24): 2, (0, 0): 0, (16, 16): 0}),
        "watershed_skips_out_of_frame_marker": (
            [[8, 8, 2.0], [99, 99, 2.0]], one_disc, True,
            {1: 45}, {(8, 8): 1}),
        "watershed_on_real_disks": (
            [[16, 16, 3.0], [48, 48, 3.0]], disks_norm, True,
            {1: 109, 2: 109}, {(16, 16): 1, (48, 48): 2, (16, 48): 0}),
    }
    for name, (blobs, img, ws, areas, probes) in cases.items():
        with subtests.test(msg=name):
            out = O._blobs_to_labels(np.array(blobs, dtype=float), img, use_watershed=ws)
            assert out.shape == img.shape and out.dtype == np.int32
            labels, counts = np.unique(out[out > 0], return_counts=True)
            assert dict(zip(labels.tolist(), counts.tolist())) == areas
            for (y, x), lab in probes.items():
                assert out[y, x] == lab, (y, x)


def test_watershed_split_separates_fused_blobs_and_falls_back_to_plain_labels(subtests):
    """Distance-peak markers split touching blobs; with no peak the mask is just labelled."""
    yy, xx = np.mgrid[:48, :48]
    fused = (((yy - 24) ** 2 + (xx - 16) ** 2) <= 100) | (((yy - 24) ** 2 + (xx - 32) ** 2) <= 100)
    square = np.zeros((32, 32), dtype=bool)
    square[8:24, 8:24] = True
    cases = {
        "two_fused_discs": (fused, (2, 601, "int32"), [(24, 16), (24, 32)]),
        "one_square": (square, (1, 256, "int32"), [(16, 16)]),
        "empty_has_no_peaks": (np.zeros((16, 16), dtype=bool), (0, 0, "int32"), []),
    }
    for name, (binary, summary, centres) in cases.items():
        with subtests.test(msg=name):
            out = O._watershed_split(binary, binary.astype(float))
            assert _summary(out) == summary
            assert len({int(out[c]) for c in centres}) == len(centres)
            assert int((out > 0).sum()) == int(binary.sum())


def test_ring_fill_closes_only_lumens_the_method_can_see(subtests):
    """Flood fill fills enclosed lumens only; convex fill also closes an open ring but stays inside its hull."""
    closed = _square_ring_edges()
    open_c = _square_ring_edges(gap=True)
    cases = {
        "flood_closed_square": (O._fill_rings_flood, closed, 256, True),
        "flood_open_ring_unfilled": (O._fill_rings_flood, open_c, int(open_c.sum()), False),
        "convex_closed_square": (O._fill_rings_convex, closed, 256, True),
        "convex_open_ring_filled": (O._fill_rings_convex, open_c, 256, True),
        "convex_empty": (O._fill_rings_convex, np.zeros((16, 16), dtype=bool), 0, False),
    }
    for name, (fill, edges, area, centre) in cases.items():
        with subtests.test(msg=name):
            out = fill(edges)
            assert out.dtype == bool and out.shape == edges.shape
            assert int(out.sum()) == area
            assert bool(out[16 % edges.shape[0], 16 % edges.shape[1]]) is centre
            assert not out[0, 0]
            assert np.all(out[edges])          # the walls are always kept


def test_filter_non_rings_keeps_objects_with_ring_contrast(subtests):
    """Objects survive on wall/lumen prominence or a >=30% edge share; survivors are relabelled 1..N."""
    square = np.zeros((32, 32), dtype=np.int32)
    square[10:20, 10:20] = 1
    walls = _square_ring_edges(lo=10, hi=20)
    bright_walls = np.where(walls, 1.0, 0.0)
    wall_only = np.where(walls, 1, 0).astype(np.int32)
    two = np.zeros((32, 32), dtype=np.int32)
    two[2:8, 2:8] = 5
    two[20:28, 20:28] = 9
    two_edges = np.zeros((32, 32), dtype=bool)
    two_edges[2:8, 2] = True
    two_edges[20:28, 20] = True
    no_interior = np.zeros((32, 32), dtype=np.int32)
    no_interior[8:16, 8:16] = 1
    one_column = np.zeros((32, 32), dtype=bool)
    one_column[8:16, 8] = True
    rand = np.random.default_rng(2).random((32, 32))
    # labeled, edges, img_norm, min_prominence -> (count, area)
    cases = {
        "no_edge_pixels_dropped": (square, np.zeros((32, 32), bool), np.full((32, 32), 0.5), 0.05, (0, 0)),
        "all_edge_object_kept": (wall_only, walls, np.full((32, 32), 0.5), 0.05, (1, int(walls.sum()))),
        "high_contrast_ring_kept": (square, walls, bright_walls, 0.5, (1, 100)),
        "low_contrast_below_threshold_dropped": (square, walls, bright_walls, 10.0, (0, 0)),
        "zero_intensity_has_zero_prominence": (square, walls, np.zeros((32, 32)), 0.05, (0, 0)),
        "survivors_relabelled_consecutively": (two, two_edges, np.where(two_edges, 1.0, 0.0), 0.1, (2, 100)),
        "random_texture_kept": (no_interior, one_column, rand, 0.01, (1, 64)),
    }
    for name, (labeled, edges, img, prom, (count, area)) in cases.items():
        with subtests.test(msg=name):
            out = O._filter_non_rings(labeled, edges, img, prom)
            assert _summary(out)[:2] == (count, area)
    with subtests.test(msg="input_not_mutated"):
        before = two.copy()
        O._filter_non_rings(two, two_edges, np.zeros((32, 32)), 0.1)
        assert np.array_equal(two, before)


# ---------------------------------------------------------------------------
# U-Net and batch dispatch
# ---------------------------------------------------------------------------

def test_segment_unet_thresholds_the_first_channel_sigmoid(subtests):
    """Images are z-scored, channel 0 of the logits is sigmoid-thresholded, small objects removed, optional skeleton."""
    rng = np.random.default_rng(0)
    img8 = rng.random((1, 8, 8)).astype(np.float32)
    lo, hi = float(np.log(0.3 / 0.7)), float(np.log(0.7 / 0.3))
    half = np.full((8, 8), lo)
    half[:, :4] = hi
    band = np.full((16, 16), -8.0)
    band[4:13, :] = 8.0
    blob = np.full((12, 12), -8.0)
    blob[1:3, 1:3] = 8.0
    two_ch = np.stack([np.full((8, 8), 8.0), np.full((8, 8), -8.0)])[None]
    # model, batch, settings -> per-image (count, area)
    cases = {
        "multichannel_uses_channel_0": (_const_unet(two_ch), img8, {"organelle_min_area": 1}, [(1, 64)]),
        "default_threshold_is_0_5": (_const_unet(half[None, None]), img8, {"organelle_min_area": 1}, [(1, 32)]),
        "threshold_setting_lowers_cut": (_const_unet(half[None, None]), img8,
                                         {"organelle_min_area": 1, "organelle_unet_threshold": 0.2}, [(1, 64)]),
        "min_area_keeps_4px_blob": (_const_unet(blob[None, None]), rng.random((1, 12, 12)).astype(np.float32),
                                    {"organelle_min_area": 4}, [(1, 4)]),
        "min_area_drops_4px_blob": (_const_unet(blob[None, None]), rng.random((1, 12, 12)).astype(np.float32),
                                    {"organelle_min_area": 5}, [(0, 0)]),
        "band_plain": (_const_unet(band[None, None]), rng.random((1, 16, 16)).astype(np.float32),
                       {"organelle_min_area": 1}, [(1, 144)]),
        "band_skeletonized": (_const_unet(band[None, None]), rng.random((1, 16, 16)).astype(np.float32),
                              {"organelle_min_area": 1, "organelle_skeletonize": True}, [(1, 32)]),
        "real_conv_two_images_plus_flat": (
            _bias_conv(2.0),
            np.stack([rng.random((16, 16)), np.zeros((16, 16))]).astype(np.float32),
            {"organelle_min_area": 2, "organelle_skeletonize": True}, [(1, 11), (1, 11)]),
        "real_conv_negative_bias": (_bias_conv(-2.0), rng.random((2, 16, 16)).astype(np.float32),
                                    {"organelle_min_area": 2}, [(0, 0), (0, 0)]),
        "empty_batch": (_const_unet(np.zeros((1, 1, 4, 4))), np.zeros((0, 4, 4), np.float32),
                        {"organelle_min_area": 1}, []),
    }
    for name, (model, batch, settings, expected) in cases.items():
        with subtests.test(msg=name):
            masks = O._segment_unet(batch, model, settings)
            assert [_summary(m)[:2] for m in masks] == expected
            assert all(m.shape == batch.shape[1:] for m in masks)
    with subtests.test(msg="flat_image_is_fed_as_zeros_not_nan"):
        model = _const_unet(np.full((1, 1, 8, 8), -6.0))
        masks = O._segment_unet(np.full((1, 8, 8), 7.0, np.float32), model, {"organelle_min_area": 1})
        assert len(model.seen) == 1 and model.seen[0].shape == (1, 1, 8, 8)
        assert np.array_equal(model.seen[0], np.zeros((1, 1, 8, 8), np.float32))
        assert masks[0].max() == 0
    with subtests.test(msg="input_is_z_scored"):
        model = _const_unet(np.zeros((1, 1, 8, 8)))
        O._segment_unet(img8, model, {"organelle_min_area": 1})
        expected = (img8[0] - img8[0].mean()) / img8[0].std()
        np.testing.assert_allclose(model.seen[0][0, 0], expected, rtol=1e-5, atol=1e-6)
    with subtests.test(msg="empty_batch_never_calls_the_model"):
        model = _const_unet(np.zeros((1, 1, 4, 4)))
        assert O._segment_unet(np.zeros((0, 4, 4), np.float32), model, {"organelle_min_area": 1}) == []
        assert model.seen == []


class _RecordingPool:
    """Pool double recording ``processes`` and mapping in-process."""

    created = []

    def __init__(self, processes):
        _RecordingPool.created.append(processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items):
        return [fn(i) for i in items]


def _no_pool(*args, **kwargs):
    raise AssertionError("Pool must not be used")


def test_segment_classical_parallel_runs_serially_or_in_a_capped_pool(subtests, monkeypatch):
    """One job or one image stays serial; otherwise the pool gets min(n_jobs, n_images, cpus) workers, same masks."""
    settings = _settings(organelle_tophat_radius=6)

    def batch(n):
        return np.stack([_u16_disks((32, 32), [(8, 8), (24, 24)], 4) for _ in range(n)])

    serial_one = _summary(O._segment_single_image(batch(1)[0], settings))
    assert serial_one == (2, 98, "int32")
    # pool double, n_images, n_jobs -> processes recorded (None = no pool)
    cases = {
        "njobs_one_is_serial": (_no_pool, 3, 1, None),
        "single_image_is_serial": (_no_pool, 1, 8, None),
        "workers_capped_at_image_count": (_RecordingPool, 2, 64, 2),
        "workers_capped_at_njobs": (_RecordingPool, 3, 2, 2),
    }
    for name, (pool, n, n_jobs, processes) in cases.items():
        with subtests.test(msg=name):
            _RecordingPool.created = []
            monkeypatch.setattr(O, "Pool", pool)
            out = O._segment_classical_parallel(batch(n), settings, n_jobs=n_jobs)
            assert [_summary(m) for m in out] == [serial_one] * n
            assert _RecordingPool.created == ([] if processes is None else [processes])
    monkeypatch.undo()
    with subtests.test(msg="real_pool_matches_serial"):
        b = batch(3)
        serial = O._segment_classical_parallel(b, settings, n_jobs=1)
        pooled = O._segment_classical_parallel(b, settings, n_jobs=2)
        assert len(pooled) == 3
        assert all(np.array_equal(a, p) for a, p in zip(serial, pooled))


def test_segment_single_image_dispatches_on_morphology(subtests, monkeypatch):
    """Each morphology reaches its back-end with the configured method; unknown ones raise."""
    targets = {"spots": "_segment_spots", "network": "_segment_network",
               "irregular": "_segment_irregular", "ring": "_segment_ring"}
    img = _disks()
    for morphology, target in targets.items():
        with subtests.test(msg=f"{morphology}_real"):
            s = _settings(organelle_morphology=morphology, organelle_method="otsu")
            assert np.array_equal(O._segment_single_image(img, s),
                                  getattr(O, target)(img, "otsu", s))
        with subtests.test(msg=f"{morphology}_routed"):
            sentinel = np.full((4, 4), 77, dtype=np.int32)
            calls = []
            with monkeypatch.context() as m:
                m.setattr(O, target, lambda im, meth, st: calls.append(
                    (im.shape, meth, st["organelle_morphology"])) or sentinel)
                out = O._segment_single_image(
                    np.zeros((4, 4), np.uint16),
                    _settings(organelle_morphology=morphology, organelle_method="adaptive"))
            assert out is sentinel
            assert calls == [((4, 4), "adaptive", morphology)]
    with subtests.test(msg="unknown_morphology"):
        with pytest.raises(ValueError, match="^Unknown morphology: blobby$"):
            O._segment_single_image(img, _settings(organelle_morphology="blobby"))


# ---------------------------------------------------------------------------
# morphology back-ends
# ---------------------------------------------------------------------------

def test_segment_spots_labels_each_spot_by_every_method(subtests):
    """otsu/adaptive top-hat thresholds and LoG/DoG blob detection each find every spot; watershed splits touching ones."""
    three = [(10, 10), (10, 36), (36, 10)]
    img3 = _u16_disks((48, 48), three, 4)
    img4 = _u16_disks((64, 64), _FOUR, 3)
    touching = _u16_disks((40, 40), [(20, 14), (20, 24)], 7)
    flat = np.full((32, 32), 500, dtype=np.uint16)
    big = dict(organelle_tophat_radius=10, organelle_min_area=10)
    # img, method, settings overrides -> summary, centres expected labelled in order
    cases = {
        "otsu": (img3, "otsu", dict(organelle_tophat_radius=6), (3, 147, "int32"), three),
        "adaptive": (img3, "adaptive", dict(organelle_tophat_radius=6), (3, 147, "int32"), three),
        "otsu_watershed": (img3, "otsu", dict(organelle_tophat_radius=6, organelle_watershed_spots=True),
                           (3, 147, "int32"), three),
        "touching_merged_without_watershed": (touching, "otsu", big, (1, 273, "int32"), [(20, 14)]),
        "touching_split_by_watershed": (touching, "otsu", dict(big, organelle_watershed_spots=True),
                                        (2, 273, "int32"), [(20, 14), (20, 24)]),
        "log_circles": (img4, "log", {}, (4, 116, "int32"), list(_FOUR)),
        "log_watershed": (img4, "log", dict(organelle_watershed_spots=True), (4, 100, "int32"), list(_FOUR)),
        "dog_circles": (img4, "dog", dict(organelle_dog_sigma_high=4.0), (4, 52, "int32"), list(_FOUR)),
        "dog_watershed": (img4, "dog", dict(organelle_dog_sigma_high=4.0, organelle_watershed_spots=True),
                          (4, 100, "int32"), list(_FOUR)),
        "log_flat_is_empty": (flat, "log", {}, (0, 0, "int32"), []),
        "dog_flat_is_empty": (flat, "dog", {}, (0, 0, "int32"), []),
    }
    for name, (img, method, over, summary, centres) in cases.items():
        with subtests.test(msg=name):
            out = O._segment_spots(img, method, _settings(**over))
            assert out.shape == img.shape
            assert _summary(out) == summary
            assert [int(out[c]) for c in centres] == list(range(1, len(centres) + 1))
            assert out[0, 0] == 0
    with subtests.test(msg="unsupported_method"):
        with pytest.raises(ValueError, match="^Unsupported spot method: sobel$"):
            O._segment_spots(img3, "sobel", _settings())


def test_spots_log_outlines_each_spot_without_flooding(subtests):
    """LoG spots: empty field -> int32 zeros; watershed outlines stop at half height instead of tiling the field."""
    yy, xx = np.mgrid[:128, :128]
    cell = 0.15 * np.exp(-(((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 40.0 ** 2)))
    droplets = [(20 + 18 * i, 20 + 18 * j) for i in range(5) for j in range(5)]
    for cy, cx in droplets:
        cell = cell + 0.8 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2)))
    cell = np.clip(cell + np.random.default_rng(0).normal(0, 0.01, cell.shape), 0, None)
    pair = _u16_disks((64, 64), [(16, 16), (48, 48)], 4)
    # img, threshold, use_watershed -> (count, area, dtype), (min, max) droplet area
    cases = {
        "flat_uint16": (np.full((24, 20), 3, np.uint16), 0.1, False, (0, 0, "int32"), None),
        "zeros_float": (np.zeros((32, 32), np.float32), 0.1, True, (0, 0, "int32"), None),
        "two_discs_circles": (pair, 0.1, False, (2, 98, "int32"), (49, 49)),
        # each disc is 49 px; the half-maximum outline keeps 45 of them
        "two_discs_watershed": (pair, 0.1, True, (2, 90, "int32"), (45, 45)),
        "lipid_droplets_on_a_noisy_cell": (cell, 0.05, True, (25, 519, "int32"), (17, 21)),
    }
    for name, (img, thr, ws, summary, size_range) in cases.items():
        with subtests.test(msg=name):
            out = O._spots_log(img, _settings(organelle_log_threshold=thr), use_watershed=ws)
            assert out.shape == img.shape
            assert _summary(out) == summary
            if size_range is not None:
                areas = np.bincount(out.ravel())[1:]
                assert (int(areas.min()), int(areas.max())) == size_range
    with subtests.test(msg="every_droplet_centre_is_labelled"):
        out = O._spots_log(cell, _settings(organelle_log_threshold=0.05), use_watershed=True)
        assert all(out[c] > 0 for c in droplets)


def test_segment_network_thresholds_closes_and_optionally_skeletonizes(subtests, monkeypatch):
    """otsu/adaptive on the smoothed image, min-area filter, skeleton; ridge/hysteresis delegate; unknown raises."""
    img = _disks()
    small = _disks(radius=3, centers=((16, 16), (48, 48)))
    cases = {
        "otsu": (img, "otsu", {}, (4, 455, "int32")),
        "otsu_skeleton": (img, "otsu", dict(organelle_skeletonize=True), (4, 20, "int32")),
        "adaptive": (img, "adaptive", {}, (5, 1931, "int32")),
        "adaptive_skeleton": (img, "adaptive", dict(organelle_skeletonize=True), (5, 1431, "int32")),
        "min_area_keeps": (small, "otsu", dict(organelle_min_area=4), (2, 74, "int32")),
        "min_area_drops": (small, "otsu", dict(organelle_min_area=5000), (0, 0, "int32")),
    }
    for name, (im, method, over, summary) in cases.items():
        with subtests.test(msg=name):
            out = O._segment_network(im, method, _settings(**over))
            assert _summary(out) == summary
            if summary[0] == 4 and not over:
                assert all(out[c] != 0 for c in _FOUR)
    for method, target in (("ridge", "_network_ridge"), ("hysteresis", "_network_hysteresis")):
        with subtests.test(msg=f"{method}_delegates"):
            s = _settings()
            assert np.array_equal(O._segment_network(_filaments(), method, s),
                                  getattr(O, target)(_filaments(), s))
            sentinel = np.full((6, 6), 5, dtype=np.int32)
            with monkeypatch.context() as m:
                m.setattr(O, target, lambda im, st: sentinel)
                m.setattr(O, "threshold_otsu", lambda *a, **k: pytest.fail("fell through to otsu"))
                assert O._segment_network(np.zeros((6, 6)), method, s) is sentinel
    with subtests.test(msg="unknown_method"):
        with pytest.raises(ValueError, match="^Unsupported network method: bogus$"):
            O._segment_network(img, "bogus", _settings())


def test_network_ridge_enhances_filaments_with_each_filter(subtests):
    """frangi/sato/meijering x otsu/adaptive (unknown threshold = otsu), optional skeleton; unknown filter raises."""
    img = _filaments()
    expected = {
        ("frangi", "otsu"): (3, 275), ("frangi", "adaptive"): (27, 1345), ("frangi", "li"): (3, 275),
        ("sato", "otsu"): (1, 296), ("sato", "adaptive"): (5, 1911), ("sato", "li"): (1, 296),
        ("meijering", "otsu"): (1, 292), ("meijering", "adaptive"): (5, 1965), ("meijering", "li"): (1, 292),
    }
    for (filt, thr), summary in expected.items():
        with subtests.test(msg=f"{filt}_{thr}"):
            out = O._network_ridge(img, _settings(organelle_ridge_filter=filt,
                                                  organelle_network_threshold=thr))
            assert _summary(out) == summary + ("int32",)
    with subtests.test(msg="skeleton_thins"):
        assert _summary(O._network_ridge(img, _settings(organelle_skeletonize=True))) == (3, 270, "int32")
    with subtests.test(msg="unknown_filter"):
        with pytest.raises(ValueError, match=r"^organelle_ridge_filter must be one of "
                                             r"\['frangi', 'sato', 'meijering'\], got 'hessian'$"):
            O._network_ridge(img, _settings(organelle_ridge_filter="hessian"))


def test_network_hysteresis_reads_bounds_as_percentiles_below_one(subtests):
    """Bounds < 1 are percentiles of the smoothed image, >= 1 absolute intensities; optional skeleton."""
    img = _disks()
    bright = _disks(fg=1000.0, noise=0.0)
    cases = {
        "percentiles": (img, dict(), (1, 3593, "int32"), True),
        "percentiles_skeleton": (img, dict(organelle_skeletonize=True), (1, 1856, "int32"), False),
        "absolute_intensities": (bright, dict(organelle_hysteresis_low=200.0, organelle_hysteresis_high=600.0),
                                 (4, 580, "int32"), True),
        "absolute_above_signal": (img, dict(organelle_hysteresis_low=5.0, organelle_hysteresis_high=9.0),
                                  (0, 0, "int32"), False),
    }
    for name, (im, over, summary, centres_on) in cases.items():
        with subtests.test(msg=name):
            out = O._network_hysteresis(im, _settings(**over))
            assert _summary(out) == summary
            if centres_on:
                assert all(out[c] != 0 for c in _FOUR)


def test_segment_irregular_thresholds_cleans_fills_and_splits(subtests):
    """otsu/adaptive on a morph-radius blur, open/close, optional hole filling, watershed split; unknown raises."""
    disks8 = _disks(radius=8)
    lumen = np.zeros((64, 64), dtype=np.float32)
    lumen[_annulus(64, (32, 32), 14, 6)] = 1.0
    lumen = lumen + np.random.default_rng(3).random((64, 64)).astype(np.float32) * 0.02
    cases = {
        "otsu": (disks8, "otsu", {}, (4, 743, "int32"), {c: True for c in _FOUR}),
        "adaptive": (disks8, "adaptive", {}, (12, 1859, "int32"), {}),
        "lumen_open_without_fill": (lumen, "otsu", dict(organelle_fill_holes=0), (8, 504, "int32"),
                                    {(32, 32): False}),
        "lumen_filled": (lumen, "otsu", dict(organelle_fill_holes=400), (1, 613, "int32"),
                         {(32, 32): True}),
    }
    for name, (im, method, over, summary, probes) in cases.items():
        with subtests.test(msg=name):
            out = O._segment_irregular(im, method, _settings(**over))
            assert _summary(out) == summary
            for c, on in probes.items():
                assert bool(out[c]) is on, c
    with subtests.test(msg="unknown_method"):
        with pytest.raises(ValueError, match="^Unsupported irregular method: bogus$"):
            O._segment_irregular(disks8, "bogus", _settings())


def test_segment_ring_finds_and_fills_hollow_rings(subtests):
    """otsu/log/dog share the Otsu edge threshold, adaptive differs; flood/convex fill (unknown = flood); filter."""
    img = _rings()
    log = dict(organelle_log_max_sigma=4, organelle_log_num_sigma=3, organelle_log_threshold=0.05)
    expected = {
        ("otsu", "flood"): (2, 874), ("otsu", "convex"): (2, 890), ("otsu", "parachute"): (2, 874),
        ("log", "flood"): (2, 874), ("log", "convex"): (2, 890),
        ("dog", "flood"): (2, 874), ("dog", "convex"): (2, 890),
        ("adaptive", "flood"): (10, 2434), ("adaptive", "convex"): (8, 2987),
        ("adaptive", "parachute"): (10, 2434),
    }
    for (method, fill), summary in expected.items():
        with subtests.test(msg=f"{method}_{fill}"):
            out = O._segment_ring(img, method, _settings(organelle_ring_min_prominence=0.0,
                                                         organelle_ring_fill_method=fill, **log))
            assert _summary(out) == summary + ("int32",)
            if method != "adaptive":
                assert out[18, 18] != 0 and out[46, 46] != 0     # lumens filled
    with subtests.test(msg="defaults_when_ring_keys_absent"):
        bare = {k: v for k, v in _settings().items() if not k.startswith("organelle_ring_")}
        assert np.array_equal(O._segment_ring(img, "otsu", bare),
                              O._segment_ring(img, "otsu", _settings(organelle_ring_min_prominence=0.1)))
    with subtests.test(msg="all_wall_objects_ignore_prominence"):
        # the DoG walls cover every filled object, so the edge-share rule decides
        strict = O._segment_ring(img, "otsu", _settings(organelle_ring_min_prominence=1e6))
        assert np.array_equal(strict, O._segment_ring(img, "otsu", _settings(organelle_ring_min_prominence=0.0)))
    with subtests.test(msg="log_without_blobs_is_empty"):
        out = O._segment_ring(np.zeros((32, 32), np.float32), "log", _settings())
        assert out.shape == (32, 32) and _summary(out) == (0, 0, "int32")
    with subtests.test(msg="unknown_method"):
        with pytest.raises(ValueError, match="^Unsupported ring method: bogus$"):
            O._segment_ring(img, "bogus", _settings())
