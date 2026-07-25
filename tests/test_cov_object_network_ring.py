"""
Branch-level coverage for the tail of ``spacr.object`` (classical
segmentation back-ends for NETWORK / IRREGULAR / RING organelle
morphologies plus the shared post-processing helpers).

Covered symbols
---------------
  * ``_segment_network``   otsu / adaptive / ridge+hysteresis delegation /
                           skeletonize / unsupported-method raise
  * ``_network_ridge``     frangi | sato | meijering, otsu / adaptive /
                           fallback thresholding, skeletonize, bad filter
  * ``_network_hysteresis`` fractional (percentile) vs absolute thresholds,
                           skeletonize, empty-result path
  * ``_segment_irregular`` otsu / adaptive / hole filling / unsupported
  * ``_segment_ring``      otsu | adaptive | log (with and without blobs) |
                           dog | unsupported, flood / convex / fallback fill
  * ``_fill_rings_flood``  / ``_fill_rings_convex``
  * ``_filter_non_rings``  every prominence / degenerate-mask branch
  * ``_normalize_01``, ``_watershed_split``, ``_postprocess_masks``
  * ``_blobs_to_labels`` tail (circle painting + watershed)

Everything here is CPU-only, offline and deterministic (fixed seeds, fixed
geometry) so the assertions are exact rather than "shape is not None".
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / synthetic image builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Nothing here should plot, but never let a figure leak into the run."""
    yield
    plt.close("all")


def _settings(**over):
    """The organelle-settings dict the classical back-ends read from."""
    s = {
        "organelle_morphology": "network",
        "organelle_method": "otsu",
        "organelle_min_size": 4,
        "organelle_max_size": 10000,
        "organelle_tophat_radius": 5,
        "organelle_watershed_spots": False,
        "organelle_log_min_sigma": 1,
        "organelle_log_max_sigma": 4,
        "organelle_log_num_sigma": 3,
        "organelle_log_threshold": 0.05,
        "organelle_dog_sigma_low": 1.0,
        "organelle_dog_sigma_high": 3.0,
        "organelle_ridge_sigmas": [1, 2],
        "organelle_ridge_filter": "frangi",
        "organelle_skeletonize": False,
        "organelle_network_threshold": "otsu",
        "organelle_hysteresis_low": 0.2,
        "organelle_hysteresis_high": 0.6,
        "organelle_adaptive_block_size": 11,
        "organelle_adaptive_offset": 0.0,
        "organelle_morph_radius": 2,
        "organelle_fill_holes": 0,
        "organelle_ring_sigma_inner": 1.0,
        "organelle_ring_sigma_outer": 3.0,
        "organelle_ring_min_prominence": 0.05,
        "organelle_ring_fill_method": "flood",
    }
    s.update(over)
    return s


def _disk_image(size=64, radius=6,
                centers=((16, 16), (16, 48), (48, 16), (48, 48)),
                fg=1.0, noise=0.02, seed=0):
    """Bright solid disks on a faintly noisy dark background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.zeros((size, size), dtype=np.float32)
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = fg
    img = img + (rng.random((size, size)).astype(np.float32) * noise * max(fg, 1.0))
    return img


def _filament_image(size=64, seed=1):
    """A cross of thin bright bars — the shape ridge filters are built for."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    img[10:54, 20:23] = 1.0     # vertical bar
    img[30:33, 5:60] = 1.0      # horizontal bar
    img = img + rng.random((size, size)).astype(np.float32) * 0.02
    return img


def _annulus(size, center, r_outer, r_inner):
    cy, cx = center
    yy, xx = np.mgrid[:size, :size]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (d2 <= r_outer ** 2) & (d2 > r_inner ** 2)


def _ring_image(size=64, seed=2):
    """Two hollow rings (bright wall, dark lumen) on a dark background."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    img[_annulus(size, (18, 18), 9, 4)] = 1.0
    img[_annulus(size, (46, 46), 9, 4)] = 1.0
    img = img + rng.random((size, size)).astype(np.float32) * 0.02
    return img


def _is_contiguous_labeling(mask):
    """Labels must be 0..N with no gaps (what sk_label guarantees)."""
    vals = np.unique(mask)
    return list(vals) == list(range(int(vals.max()) + 1))


# ===========================================================================
# _blobs_to_labels tail (circle painting vs watershed)
# ===========================================================================

def test_blobs_to_labels_paints_one_circle_per_blob():
    from spacr.object import _blobs_to_labels, _normalize_01

    img_norm = _normalize_01(_disk_image())
    blobs = np.array([[16.0, 16.0, 3.0], [48.0, 48.0, 3.0]])

    labeled = _blobs_to_labels(blobs, img_norm, use_watershed=False)

    assert labeled.shape == img_norm.shape
    assert set(np.unique(labeled)) == {0, 1, 2}
    # each blob centre carries its own id
    assert labeled[16, 16] == 1
    assert labeled[48, 48] == 2
    # radius = round(sigma*sqrt(2)) = 4 -> the disk reaches 4 px out
    assert labeled[16, 20] == 1
    assert labeled[16, 22] == 0


def test_blobs_to_labels_watershed_covers_more_than_the_markers():
    from spacr.object import _blobs_to_labels, _normalize_01

    img_norm = _normalize_01(_disk_image())
    blobs = np.array([[16.0, 16.0, 3.0], [48.0, 48.0, 3.0]])

    labeled = _blobs_to_labels(blobs, img_norm, use_watershed=True)

    assert labeled.shape == img_norm.shape
    # watershed grows each single-pixel marker into a basin
    assert labeled.max() == 2
    assert (labeled > 0).sum() > 2


# ===========================================================================
# _segment_network
# ===========================================================================

def test_segment_network_otsu_labels_every_disk():
    from spacr.object import _segment_network

    out = _segment_network(_disk_image(), "otsu", _settings())

    assert out.shape == (64, 64)
    assert out.max() == 4                      # four disks -> four labels
    assert _is_contiguous_labeling(out)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        assert out[cy, cx] != 0


def test_segment_network_adaptive_uses_local_threshold():
    from spacr.object import _segment_network

    s = _settings(organelle_adaptive_block_size=11, organelle_adaptive_offset=0.0)
    out = _segment_network(_disk_image(), "adaptive", s)
    otsu_out = _segment_network(_disk_image(), "otsu", _settings())

    assert out.shape == (64, 64)
    assert out.max() >= 1
    assert _is_contiguous_labeling(out)
    # the local threshold responds to edges, so it cannot equal the global one
    assert not np.array_equal(out > 0, otsu_out > 0)


def test_segment_network_skeletonize_thins_the_objects():
    from spacr.object import _segment_network

    img = _disk_image()
    solid = _segment_network(img, "otsu", _settings(organelle_skeletonize=False))
    thin = _segment_network(img, "otsu", _settings(organelle_skeletonize=True))

    assert thin.max() >= 1
    assert (thin > 0).sum() < (solid > 0).sum()
    assert _is_contiguous_labeling(thin)


def test_segment_network_delegates_ridge_to_network_ridge():
    from spacr.object import _segment_network, _network_ridge

    img = _filament_image()
    s = _settings()
    assert np.array_equal(_segment_network(img, "ridge", s), _network_ridge(img, s))


def test_segment_network_delegates_hysteresis_to_network_hysteresis():
    from spacr.object import _segment_network, _network_hysteresis

    img = _disk_image()
    s = _settings()
    assert np.array_equal(_segment_network(img, "hysteresis", s),
                          _network_hysteresis(img, s))


def test_segment_network_rejects_unknown_method():
    from spacr.object import _segment_network

    with pytest.raises(ValueError, match="Unsupported network method: bogus"):
        _segment_network(_disk_image(), "bogus", _settings())


def test_segment_network_min_size_drops_small_objects():
    from spacr.object import _segment_network

    img = _disk_image(radius=3, centers=((16, 16), (48, 48)))
    kept = _segment_network(img, "otsu", _settings(organelle_min_size=4))
    dropped = _segment_network(img, "otsu", _settings(organelle_min_size=5000))

    assert kept.max() == 2
    assert dropped.max() == 0


# ===========================================================================
# _network_ridge
# ===========================================================================

@pytest.mark.parametrize("filter_name", ["frangi", "sato", "meijering"])
def test_network_ridge_each_filter_finds_the_filaments(filter_name):
    from spacr.object import _network_ridge

    out = _network_ridge(_filament_image(),
                         _settings(organelle_ridge_filter=filter_name))

    assert out.shape == (64, 64)
    assert out.max() >= 1
    assert _is_contiguous_labeling(out)


def test_network_ridge_rejects_unknown_filter():
    from spacr.object import _network_ridge

    with pytest.raises(ValueError, match="organelle_ridge_filter must be one of"):
        _network_ridge(_filament_image(),
                       _settings(organelle_ridge_filter="hessian"))


def test_network_ridge_adaptive_threshold_differs_from_otsu():
    from spacr.object import _network_ridge

    img = _filament_image()
    adaptive = _network_ridge(img, _settings(organelle_network_threshold="adaptive"))
    otsu = _network_ridge(img, _settings(organelle_network_threshold="otsu"))

    assert adaptive.shape == (64, 64)
    assert _is_contiguous_labeling(adaptive)
    assert not np.array_equal(adaptive > 0, otsu > 0)


def test_network_ridge_unknown_threshold_falls_back_to_otsu():
    from spacr.object import _network_ridge

    img = _filament_image()
    fallback = _network_ridge(img, _settings(organelle_network_threshold="li"))
    otsu = _network_ridge(img, _settings(organelle_network_threshold="otsu"))

    assert np.array_equal(fallback, otsu)
    assert otsu.max() >= 1


def test_network_ridge_skeletonize_thins_the_result():
    from spacr.object import _network_ridge

    img = _filament_image()
    solid = _network_ridge(img, _settings(organelle_skeletonize=False))
    thin = _network_ridge(img, _settings(organelle_skeletonize=True))

    assert thin.max() >= 1
    assert (thin > 0).sum() < (solid > 0).sum()


# ===========================================================================
# _network_hysteresis
# ===========================================================================

def test_network_hysteresis_fractional_values_are_percentiles():
    from spacr.object import _network_hysteresis

    # 0.2 / 0.6 -> 20th / 60th percentile of the smoothed image.
    out = _network_hysteresis(_disk_image(),
                              _settings(organelle_hysteresis_low=0.2,
                                        organelle_hysteresis_high=0.6))
    assert out.shape == (64, 64)
    assert out.max() >= 1
    assert _is_contiguous_labeling(out)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        assert out[cy, cx] != 0


def test_network_hysteresis_absolute_values_are_intensities():
    from spacr.object import _network_hysteresis

    # fg = 1000 so low/high >= 1.0 are taken literally, not as percentiles.
    img = _disk_image(fg=1000.0, noise=0.0)
    out = _network_hysteresis(img, _settings(organelle_hysteresis_low=200.0,
                                             organelle_hysteresis_high=600.0))
    assert out.max() == 4
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        assert out[cy, cx] != 0


def test_network_hysteresis_absolute_threshold_above_signal_gives_empty_mask():
    from spacr.object import _network_hysteresis

    img = _disk_image(fg=1.0, noise=0.02)
    out = _network_hysteresis(img, _settings(organelle_hysteresis_low=5.0,
                                             organelle_hysteresis_high=9.0))
    assert out.shape == (64, 64)
    assert out.max() == 0


def test_network_hysteresis_skeletonize_thins_the_result():
    from spacr.object import _network_hysteresis

    img = _disk_image()
    solid = _network_hysteresis(img, _settings(organelle_skeletonize=False))
    thin = _network_hysteresis(img, _settings(organelle_skeletonize=True))

    assert thin.max() >= 1
    assert (thin > 0).sum() < (solid > 0).sum()


# ===========================================================================
# _segment_irregular
# ===========================================================================

def test_segment_irregular_otsu_returns_one_label_per_disk():
    from spacr.object import _segment_irregular

    out = _segment_irregular(_disk_image(radius=8), "otsu", _settings())

    assert out.shape == (64, 64)
    assert out.max() >= 4
    assert _is_contiguous_labeling(out)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        assert out[cy, cx] != 0


def test_segment_irregular_adaptive_differs_from_otsu():
    from spacr.object import _segment_irregular

    img = _disk_image(radius=8)
    adaptive = _segment_irregular(img, "adaptive", _settings())
    otsu = _segment_irregular(img, "otsu", _settings())

    assert adaptive.shape == (64, 64)
    assert _is_contiguous_labeling(adaptive)
    assert not np.array_equal(adaptive > 0, otsu > 0)


def test_segment_irregular_fill_holes_closes_the_lumen():
    from spacr.object import _segment_irregular

    # one thick annulus -> a genuine hole that survives opening/closing
    img = np.zeros((64, 64), dtype=np.float32)
    img[_annulus(64, (32, 32), 14, 6)] = 1.0
    img = img + np.random.default_rng(3).random((64, 64)).astype(np.float32) * 0.02

    unfilled = _segment_irregular(img, "otsu", _settings(organelle_fill_holes=0))
    filled = _segment_irregular(img, "otsu", _settings(organelle_fill_holes=400))

    assert (unfilled > 0).sum() > 0
    assert (filled > 0).sum() > (unfilled > 0).sum()
    assert filled[32, 32] != 0          # lumen filled in
    assert unfilled[32, 32] == 0        # lumen still open


def test_segment_irregular_rejects_unknown_method():
    from spacr.object import _segment_irregular

    with pytest.raises(ValueError, match="Unsupported irregular method: bogus"):
        _segment_irregular(_disk_image(), "bogus", _settings())


# ===========================================================================
# _segment_ring
# ===========================================================================

def test_segment_ring_otsu_recovers_both_rings_as_solid_objects():
    from spacr.object import _segment_ring

    out = _segment_ring(_ring_image(), "otsu",
                        _settings(organelle_ring_min_prominence=0.0))

    assert out.shape == (64, 64)
    assert out.max() >= 2
    assert _is_contiguous_labeling(out)
    # the lumen is filled, so the ring centres belong to an object
    assert out[18, 18] != 0
    assert out[46, 46] != 0


def test_segment_ring_log_and_dog_and_otsu_agree_on_the_edge_threshold():
    from spacr.object import _segment_ring

    img = _ring_image()
    s = _settings(organelle_ring_min_prominence=0.0)
    otsu = _segment_ring(img, "otsu", s)
    log = _segment_ring(img, "log", s)
    dog = _segment_ring(img, "dog", s)

    # all three branches end up thresholding the DoG image with Otsu
    assert np.array_equal(otsu, dog)
    assert np.array_equal(log, dog)
    assert dog.max() >= 2


def test_segment_ring_log_without_blobs_returns_empty_int32_mask():
    from spacr.object import _segment_ring

    flat = np.zeros((32, 32), dtype=np.float32)
    out = _segment_ring(flat, "log", _settings())

    assert out.shape == (32, 32)
    assert out.dtype == np.int32
    assert out.max() == 0


def test_segment_ring_adaptive_differs_from_otsu():
    from spacr.object import _segment_ring

    img = _ring_image()
    s = _settings(organelle_ring_min_prominence=0.0)
    adaptive = _segment_ring(img, "adaptive", s)
    otsu = _segment_ring(img, "otsu", s)

    assert adaptive.shape == (64, 64)
    assert _is_contiguous_labeling(adaptive)
    assert not np.array_equal(adaptive > 0, otsu > 0)


def test_segment_ring_convex_fill_covers_at_least_the_flood_fill():
    from spacr.object import _segment_ring

    img = _ring_image()
    flood = _segment_ring(img, "otsu",
                          _settings(organelle_ring_fill_method="flood",
                                    organelle_ring_min_prominence=0.0))
    convex = _segment_ring(img, "otsu",
                           _settings(organelle_ring_fill_method="convex",
                                     organelle_ring_min_prominence=0.0))

    assert convex.max() >= 1
    assert (convex > 0).sum() >= (flood > 0).sum()


def test_segment_ring_unknown_fill_method_falls_back_to_flood():
    from spacr.object import _segment_ring

    img = _ring_image()
    weird = _segment_ring(img, "otsu",
                          _settings(organelle_ring_fill_method="parachute",
                                    organelle_ring_min_prominence=0.0))
    flood = _segment_ring(img, "otsu",
                          _settings(organelle_ring_fill_method="flood",
                                    organelle_ring_min_prominence=0.0))

    assert np.array_equal(weird, flood)


def test_segment_ring_rejects_unknown_method():
    from spacr.object import _segment_ring

    with pytest.raises(ValueError, match="Unsupported ring method: bogus"):
        _segment_ring(_ring_image(), "bogus", _settings())


def test_segment_ring_prominence_filter_never_adds_pixels():
    from spacr.object import _segment_ring

    img = _ring_image()
    permissive = _segment_ring(img, "otsu",
                               _settings(organelle_ring_min_prominence=0.0))
    strict = _segment_ring(img, "otsu",
                           _settings(organelle_ring_min_prominence=1e6))

    assert (strict > 0).sum() <= (permissive > 0).sum()
    # For this image the DoG walls cover the whole filled object, so every
    # object has an empty interior and takes the edge_ratio short-circuit
    # (ratio 1.0 >= 0.3) — min_prominence is never consulted.
    assert np.array_equal(strict, permissive)
    assert permissive.max() == 2


# ===========================================================================
# _fill_rings_flood / _fill_rings_convex
# ===========================================================================

def test_fill_rings_flood_fills_the_lumen_but_not_the_background():
    from spacr.object import _fill_rings_flood

    edges = np.zeros((32, 32), dtype=bool)
    edges[8:24, 8] = True
    edges[8:24, 23] = True
    edges[8, 8:24] = True
    edges[23, 8:24] = True

    filled = _fill_rings_flood(edges)

    assert filled.dtype == bool
    assert filled[16, 16]                  # lumen filled
    assert filled[8, 8]                    # original wall preserved
    assert not filled[0, 0]                # border-connected background kept
    assert not filled[31, 31]
    assert filled.sum() > edges.sum()
    # the filled object is exactly the 16x16 square
    assert filled.sum() == 16 * 16


def test_fill_rings_flood_leaves_an_open_ring_unfilled():
    from spacr.object import _fill_rings_flood

    edges = np.zeros((32, 32), dtype=bool)
    edges[8:24, 8] = True
    edges[8:24, 23] = True
    edges[8, 8:24] = True
    edges[23, 8:16] = True                 # gap in the bottom wall

    filled = _fill_rings_flood(edges)

    # interior leaks to the image border, so nothing gets filled
    assert not filled[16, 16]
    assert filled.sum() == edges.sum()


def test_fill_rings_convex_fills_an_open_ring_that_flood_cannot():
    from spacr.object import _fill_rings_flood, _fill_rings_convex

    edges = np.zeros((32, 32), dtype=bool)
    edges[8:24, 8] = True
    edges[8:24, 23] = True
    edges[8, 8:24] = True
    edges[23, 8:16] = True                 # same C-shape as above

    flood = _fill_rings_flood(edges)
    convex = _fill_rings_convex(edges)

    assert convex[16, 16]
    assert convex.sum() > flood.sum()
    assert not convex[0, 0]                # hull stays inside the bbox


def test_fill_rings_convex_on_empty_input_returns_empty():
    from spacr.object import _fill_rings_convex

    empty = np.zeros((16, 16), dtype=bool)
    out = _fill_rings_convex(empty)
    assert out.shape == (16, 16)
    assert not out.any()


# ===========================================================================
# _filter_non_rings
# ===========================================================================

def _square_label(size=32, lo=10, hi=20):
    labeled = np.zeros((size, size), dtype=np.int32)
    labeled[lo:hi, lo:hi] = 1
    return labeled


def _square_edges(size=32, lo=10, hi=20):
    edges = np.zeros((size, size), dtype=bool)
    edges[lo:hi, lo] = True
    edges[lo:hi, hi - 1] = True
    edges[lo, lo:hi] = True
    edges[hi - 1, lo:hi] = True
    return edges


def test_filter_non_rings_drops_objects_with_no_edge_pixels():
    from spacr.object import _filter_non_rings

    labeled = _square_label()
    edges = np.zeros((32, 32), dtype=bool)     # no edge overlaps the object
    img_norm = np.full((32, 32), 0.5)

    out = _filter_non_rings(labeled, edges, img_norm, min_prominence=0.05)

    assert out.max() == 0                       # edge_ratio 0.0 < 0.3 -> dropped


def test_filter_non_rings_keeps_an_all_edge_object():
    from spacr.object import _filter_non_rings

    edges = _square_edges()
    labeled = np.zeros((32, 32), dtype=np.int32)
    labeled[edges] = 1                          # object == its own edges
    img_norm = np.full((32, 32), 0.5)

    out = _filter_non_rings(labeled, edges, img_norm, min_prominence=0.05)

    # interior_mask is empty but edge_ratio == 1.0 >= 0.3 -> object survives
    assert out.max() == 1
    assert (out > 0).sum() == edges.sum()


def test_filter_non_rings_keeps_high_contrast_rings():
    from spacr.object import _filter_non_rings

    labeled = _square_label()
    edges = _square_edges()
    img_norm = np.zeros((32, 32))
    img_norm[edges] = 1.0                       # bright wall, dark lumen

    out = _filter_non_rings(labeled, edges, img_norm, min_prominence=0.5)

    assert out.max() == 1
    assert (out > 0).sum() == 100               # whole 10x10 square kept


def test_filter_non_rings_drops_low_contrast_objects():
    from spacr.object import _filter_non_rings

    labeled = _square_label()
    edges = _square_edges()
    img_norm = np.zeros((32, 32))
    img_norm[edges] = 1.0

    # prominence is ~2.8 here, so a threshold of 10 must reject the object
    out = _filter_non_rings(labeled, edges, img_norm, min_prominence=10.0)

    assert out.max() == 0


def test_filter_non_rings_zero_intensity_object_gets_zero_prominence():
    from spacr.object import _filter_non_rings

    labeled = _square_label()
    edges = _square_edges()
    img_norm = np.zeros((32, 32))               # object_mean == 0 branch

    out = _filter_non_rings(labeled, edges, img_norm, min_prominence=0.05)

    assert out.max() == 0


def test_filter_non_rings_relabels_survivors_consecutively():
    from spacr.object import _filter_non_rings

    labeled = np.zeros((32, 32), dtype=np.int32)
    labeled[2:8, 2:8] = 5                       # non-consecutive input ids
    labeled[20:28, 20:28] = 9
    edges = np.zeros((32, 32), dtype=bool)
    edges[2:8, 2] = True
    edges[20:28, 20] = True
    img_norm = np.zeros((32, 32))
    img_norm[edges] = 1.0

    out = _filter_non_rings(labeled, edges, img_norm, min_prominence=0.1)

    assert sorted(np.unique(out).tolist()) == [0, 1, 2]


# ===========================================================================
# _normalize_01
# ===========================================================================

def test_normalize_01_clips_to_the_1_99_percentile_window():
    from spacr.object import _normalize_01

    img = np.arange(100, dtype=np.uint16).reshape(10, 10)
    out = _normalize_01(img)

    assert out.dtype == np.float64
    assert out.min() == 0.0
    assert out.max() == 1.0
    # monotone ramp stays monotone
    assert np.all(np.diff(out.ravel()) >= 0)
    # the extreme tails are clipped, not merely rescaled
    assert out.ravel()[0] == 0.0
    assert out.ravel()[-1] == 1.0


def test_normalize_01_flat_image_returns_zeros():
    from spacr.object import _normalize_01

    out = _normalize_01(np.full((8, 8), 5.0))
    assert out.shape == (8, 8)
    assert out.dtype == np.float64
    assert np.count_nonzero(out) == 0


# ===========================================================================
# _watershed_split
# ===========================================================================

def test_watershed_split_separates_two_touching_blobs():
    from spacr.object import _watershed_split

    yy, xx = np.mgrid[:48, :48]
    binary = (((yy - 24) ** 2 + (xx - 16) ** 2) <= 10 ** 2) | \
             (((yy - 24) ** 2 + (xx - 32) ** 2) <= 10 ** 2)
    from skimage.measure import label as sk_label
    assert sk_label(binary).max() == 1           # they really are fused

    out = _watershed_split(binary, binary.astype(float))

    assert out.max() == 2
    assert out[24, 16] != out[24, 32]
    assert (out > 0).sum() == binary.sum()


def test_watershed_split_without_peaks_falls_back_to_plain_labeling():
    from spacr.object import _watershed_split

    empty = np.zeros((16, 16), dtype=bool)
    out = _watershed_split(empty, np.zeros((16, 16)))
    assert out.shape == (16, 16)
    assert out.max() == 0


# ===========================================================================
# _postprocess_masks
# ===========================================================================

def test_postprocess_masks_min_size_removes_small_objects():
    from spacr.object import _postprocess_masks

    m = np.zeros((32, 32), dtype=np.int32)
    m[0, 0] = 1                 # 1 px
    m[10:20, 10:20] = 2         # 100 px

    out = _postprocess_masks([m], min_size=5)

    assert len(out) == 1
    assert out[0].max() == 1
    assert (out[0] > 0).sum() == 100


def test_postprocess_masks_max_size_removes_large_objects():
    from spacr.object import _postprocess_masks

    m = np.zeros((32, 32), dtype=np.int32)
    m[2:6, 2:6] = 1             # 16 px
    m[8:28, 8:28] = 2           # 400 px

    out = _postprocess_masks([m], min_size=0, max_size=100)

    assert out[0].max() == 1
    assert (out[0] > 0).sum() == 16


def test_postprocess_masks_remove_border_drops_touching_objects():
    from spacr.object import _postprocess_masks

    m = np.zeros((16, 16), dtype=np.int32)
    m[0:4, 0:4] = 1             # top-left corner -> touches border
    m[7:11, 7:11] = 2           # interior
    m[12:16, 12:16] = 3         # bottom-right corner -> touches border

    out = _postprocess_masks([m], min_size=0, remove_border=True)

    assert out[0].max() == 1
    assert (out[0] > 0).sum() == 16
    assert out[0][8, 8] == 1


def test_postprocess_masks_no_size_filter_only_relabels():
    from spacr.object import _postprocess_masks

    m = np.zeros((16, 16), dtype=np.int32)
    m[2:5, 2:5] = 7             # sparse, non-consecutive ids
    m[9:12, 9:12] = 3

    out = _postprocess_masks([m], min_size=0, max_size=None)

    assert sorted(np.unique(out[0]).tolist()) == [0, 1, 2]
    assert (out[0] > 0).sum() == 18


def test_postprocess_masks_processes_every_mask_and_does_not_mutate_input():
    from spacr.object import _postprocess_masks

    a = np.zeros((16, 16), dtype=np.int32)
    a[2:6, 2:6] = 1
    a[9, 9] = 2                 # 1 px, dropped by min_size
    b = np.zeros((16, 16), dtype=np.int32)
    b[4:12, 4:12] = 4

    before = a.copy()
    out = _postprocess_masks([a, b], min_size=4)

    assert len(out) == 2
    assert out[0].max() == 1 and (out[0] > 0).sum() == 16
    assert out[1].max() == 1 and (out[1] > 0).sum() == 64
    assert np.array_equal(a, before)     # inputs are copied, not edited
