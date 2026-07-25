"""Coverage for :func:`spacr.object.merge_split_filter_masks`.

Focuses on the input-normalisation branches that decide how ``masks`` and
``intensity_images`` are unpacked into per-FOV lists, in particular the
4-D (N, H, W, C) intensity layout that ``generate_cellpose_masks_sam``
actually passes in (it hands over the raw ``batch``), and the guard that
rejects any other rank.

Everything here is pure numpy on tiny 64x64 fields - no cellpose, no torch
device work, no disk IO.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Three disjoint 10x10 squares, well clear of the image border so the
# border filter can never be the reason an object disappears.
_BOXES = {
    1: (slice(5, 15), slice(5, 15)),      # probe pixel (10, 10)
    2: (slice(5, 15), slice(25, 35)),     # probe pixel (10, 30)
    3: (slice(5, 15), slice(45, 55)),     # probe pixel (10, 50)
}
_PROBES = {1: (10, 10), 2: (10, 30), 3: (10, 50)}


def _three_box_mask(shape=(64, 64)):
    """Label image with three disjoint, non-touching, non-border squares."""
    m = np.zeros(shape, dtype=np.int32)
    for lbl, sl in _BOXES.items():
        m[sl] = lbl
    return m


def _intensity_plane(values, shape=(64, 64)):
    """Flat-valued intensity image: ``values[label]`` inside each box, 0 outside."""
    img = np.zeros(shape, dtype=np.float32)
    for lbl, sl in _BOXES.items():
        img[sl] = values[lbl]
    return img


def _surviving_labels(out_mask):
    """Original box ids (1/2/3) that still have a non-zero pixel at their probe."""
    return {lbl for lbl, (y, x) in _PROBES.items() if out_mask[y, x] != 0}


@pytest.fixture(autouse=True)
def _no_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 4-D intensity images: (N, H, W, C)
# ---------------------------------------------------------------------------

def test_merge_split_filter_masks_accepts_4d_channel_last_intensity():
    """A 4-D (N, H, W, C) intensity stack is split per-FOV and channel 0 is used.

    Channel 0 and channel 1 carry *opposite* brightness orderings, so the
    identity of the object dropped by the intensity-percentile filter proves
    which channel was read.  With three objects and
    ``min_intensity_percentile=50`` the threshold is the median object mean,
    so only the strictly-dimmest object is removed.
    """
    from spacr.object import merge_split_filter_masks

    masks = np.stack([_three_box_mask(), _three_box_mask()])          # (2, 64, 64)

    # FOV 0 -> box 1 dimmest on channel 0 (but brightest on channel 1).
    fov0 = np.stack(
        [_intensity_plane({1: 10.0, 2: 50.0, 3: 90.0}),
         _intensity_plane({1: 90.0, 2: 50.0, 3: 10.0})],
        axis=-1,
    )
    # FOV 1 -> box 3 dimmest on channel 0 (orderings swapped again).
    fov1 = np.stack(
        [_intensity_plane({1: 90.0, 2: 50.0, 3: 10.0}),
         _intensity_plane({1: 10.0, 2: 50.0, 3: 90.0})],
        axis=-1,
    )
    intensity = np.stack([fov0, fov1])                                # (2, 64, 64, 2)
    assert intensity.ndim == 4

    out = merge_split_filter_masks(
        masks, intensity,
        {"cell_min_intensity_percentile": 50}, "cell",
    )

    assert isinstance(out, list)
    assert len(out) == 2, "one output mask per FOV of the 4-D intensity stack"
    for arr in out:
        assert arr.shape == (64, 64)
        assert arr.dtype == np.uint16

    # Channel 0 drove the filter in each FOV, independently.
    assert _surviving_labels(out[0]) == {2, 3}, "FOV0: dimmest box (1) removed"
    assert _surviving_labels(out[1]) == {1, 2}, "FOV1: dimmest box (3) removed"

    # Exactly two objects left in each FOV, sequentially relabelled from 1.
    for arr in out:
        assert np.unique(arr).tolist() == [0, 1, 2]


def test_merge_split_filter_masks_4d_single_fov_matches_3d_equivalent():
    """(1, H, W, C) and (1, H, W) intensity stacks give the identical result.

    Guards the 4-D branch against silently selecting the wrong axis: the
    channel-last stack whose channel 0 equals the 3-D plane must filter the
    very same objects.
    """
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()[None, ...]                               # (1, 64, 64)
    plane = _intensity_plane({1: 10.0, 2: 50.0, 3: 90.0})
    settings = {"cell_min_intensity_percentile": 50}

    out_3d = merge_split_filter_masks(masks, plane[None, ...], settings, "cell")
    # Channel 0 == plane; channel 1 is deliberately garbage.
    stack_4d = np.stack([plane, np.full_like(plane, 999.0)], axis=-1)[None, ...]
    assert stack_4d.shape == (1, 64, 64, 2)
    out_4d = merge_split_filter_masks(masks, stack_4d, settings, "cell")

    assert len(out_3d) == len(out_4d) == 1
    np.testing.assert_array_equal(out_4d[0], out_3d[0])
    assert _surviving_labels(out_4d[0]) == {2, 3}


def test_merge_split_filter_masks_4d_length_must_match_masks():
    """The 4-D branch still feeds the mask/intensity count check."""
    from spacr.object import merge_split_filter_masks

    masks = np.stack([_three_box_mask(), _three_box_mask()])           # 2 FOVs
    intensity = np.zeros((3, 64, 64, 2), dtype=np.float32)             # 3 FOVs
    with pytest.raises(ValueError) as ei:
        merge_split_filter_masks(
            masks, intensity, {"cell_min_area": 5}, "cell")
    msg = str(ei.value)
    assert "(2)" in msg and "(3)" in msg


# ---------------------------------------------------------------------------
# Unsupported intensity rank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(8,), (2, 2, 2, 2, 2)])
def test_merge_split_filter_masks_rejects_bad_intensity_ndim(shape):
    """1-D and 5-D intensity arrays raise, and the message names the rank."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()                     # 2-D -> accepted
    intensity = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError) as ei:
        merge_split_filter_masks(
            masks, intensity, {"cell_min_area": 5}, "cell")
    assert f"Unsupported intensity_images ndim: {len(shape)}" in str(ei.value)


def test_merge_split_filter_masks_bad_intensity_ndim_checked_before_length():
    """A 5-D intensity array is rejected for its rank, not for its length."""
    from spacr.object import merge_split_filter_masks

    masks = [_three_box_mask()]
    intensity = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError) as ei:
        merge_split_filter_masks(
            masks, intensity, {"cell_min_area": 5}, "cell")
    assert "Unsupported intensity_images ndim: 5" in str(ei.value)
    assert "does not match" not in str(ei.value)


# ---------------------------------------------------------------------------
# needs_work gate + settings-key plumbing
# ---------------------------------------------------------------------------

def test_merge_split_filter_masks_noop_returns_same_object_identity():
    """No enabled operation -> the caller's array is handed straight back."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()
    settings = {
        "cell_perimeter_fraction": 0,
        "cell_intensity_merge": False,
        "cell_intensity_split": False,
        "cell_min_object_area": 0,
        "cell_min_area": 0,
        "cell_max_area": 0,
        "cell_remove_border_objects": False,
        "cell_min_intensity_percentile": 0,
        "cell_max_intensity_percentile": 100,
    }
    out = merge_split_filter_masks(masks, None, settings, "cell")
    assert out is masks


def test_merge_split_filter_masks_max_intensity_percentile_alone_enables_work():
    """``max_intensity_percentile < 100`` on its own must trip needs_work."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()
    intensity = _intensity_plane({1: 10.0, 2: 50.0, 3: 90.0})

    out = merge_split_filter_masks(
        masks, intensity,
        {"cell_max_intensity_percentile": 50}, "cell",
    )
    assert isinstance(out, list) and len(out) == 1
    # Median threshold -> only the strictly brightest object exceeds it.
    assert _surviving_labels(out[0]) == {1, 2}


def test_merge_split_filter_masks_honours_misspelled_perimiter_key():
    """The legacy ``<type>_perimiter_fraction`` spelling is still read.

    Two objects sharing a long boundary merge into one when the misspelled
    key is supplied, proving the fallback lookup is live.
    """
    from spacr.object import merge_split_filter_masks

    # Two touching 20x10 halves of one 20x20 block -> long shared boundary.
    m = np.zeros((64, 64), dtype=np.int32)
    m[20:40, 20:30] = 1
    m[20:40, 30:40] = 2
    intensity = np.zeros((64, 64), dtype=np.float32)

    merged = merge_split_filter_masks(
        m.copy(), intensity, {"cell_perimiter_fraction": 0.1}, "cell")
    assert len(np.unique(merged[0])) - 1 == 1, "the two halves merged"

    # Sanity: without the key nothing merges (needs_work driven by min_area=1,
    # which removes nothing since both objects are 200 px).
    untouched = merge_split_filter_masks(
        m.copy(), intensity, {"cell_min_area": 1}, "cell")
    assert len(np.unique(untouched[0])) - 1 == 2


def test_merge_split_filter_masks_none_masks_short_circuits():
    """``masks=None`` returns None once an operation is enabled."""
    from spacr.object import merge_split_filter_masks

    assert merge_split_filter_masks(
        None, None, {"cell_min_area": 5}, "cell") is None


def test_merge_split_filter_masks_uses_object_type_prefix():
    """Settings for another object type are ignored for this object_type."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()
    # A nucleus-scoped setting must not switch on work for 'cell'.
    out = merge_split_filter_masks(
        masks, None, {"nucleus_min_area": 500}, "cell")
    assert out is masks

    # ...but it does for 'nucleus'.
    out2 = merge_split_filter_masks(
        masks.copy(), _intensity_plane({1: 1.0, 2: 1.0, 3: 1.0}),
        {"nucleus_min_area": 500}, "nucleus")
    assert isinstance(out2, list)
    # every box is 100 px < 500 -> all removed
    assert out2[0].max() == 0


def test_merge_split_filter_masks_list_inputs_and_batch_filenames():
    """List-of-2D inputs are accepted and processed one FOV at a time."""
    from spacr.object import merge_split_filter_masks

    masks = [_three_box_mask(), _three_box_mask()]
    intensity = [
        _intensity_plane({1: 10.0, 2: 50.0, 3: 90.0}),
        _intensity_plane({1: 10.0, 2: 50.0, 3: 90.0}),
    ]
    out = merge_split_filter_masks(
        masks, intensity, {"cell_min_intensity_percentile": 50}, "cell",
        batch_filenames=["fov_a.npy", "fov_b.npy"],
    )
    assert len(out) == 2
    for arr in out:
        assert _surviving_labels(arr) == {2, 3}
    assert len(masks) == 2


def test_merge_split_filter_masks_rejects_bad_mask_ndim():
    """A 4-D mask array is rejected (only 2-D/3-D label images are meaningful)."""
    from spacr.object import merge_split_filter_masks

    bad = np.zeros((2, 4, 4, 4), dtype=np.int32)
    with pytest.raises(ValueError) as ei:
        merge_split_filter_masks(
            bad, np.zeros((2, 4, 4), dtype=np.float32),
            {"cell_min_area": 3}, "cell")
    assert "Unsupported masks ndim: 4" in str(ei.value)
