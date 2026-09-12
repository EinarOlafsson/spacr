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


# --- a touching pair, for the boundary-intensity merge ---------------------
#
# WHY THE DISJOINT BOXES ABOVE CAN NO LONGER SHOW INTENSITY DOING ANYTHING.
# The absolute-intensity change left exactly one intensity-driven operation
# in this pipeline, and it is the merge: it measures the MEAN ALONG A SHARED
# BOUNDARY and joins the two labels when that mean reaches the threshold.
# Nothing drops an object for being dim any more.  Three boxes that do not
# touch have no shared boundary to measure, so they come back untouched at
# every threshold -- measured, not assumed; the guard below pins it.
#
# These two labels are the two halves of one block, so there is exactly one
# boundary, it runs down the middle of the block, and the mean along it is
# simply the value the block carries.
_SEAM_BLOCK = (slice(20, 40), slice(20, 40))
_SEAM_BRIGHT = 900.0      # boundary mean 900 >= 500 -> the pair merges
_SEAM_DIM = 100.0         # boundary mean 100 <  500 -> the pair stays apart
_SEAM_THRESHOLD = 500.0   # strictly between the two, in raw image units

#: The merge switch plus the threshold it needs; the switch is what trips
#: ``needs_work``, the threshold only parameterises it.
_MERGE_ON = {
    "cell_intensity_merge": True,
    "cell_intensity_threshold": _SEAM_THRESHOLD,
}


def _touching_pair_mask(shape=(64, 64)):
    """One 20x20 block cut down the middle into labels 1 and 2."""
    m = np.zeros(shape, dtype=np.int32)
    m[20:40, 20:30] = 1
    m[20:40, 30:40] = 2
    return m


def _seam_plane(value, shape=(64, 64)):
    """Flat ``value`` across the whole block, 0 outside.

    The shared boundary is interior to the block, so the boundary mean the
    merge measures is exactly ``value``.
    """
    img = np.zeros(shape, dtype=np.float32)
    img[_SEAM_BLOCK] = value
    return img


def _n_objects(out_mask):
    """Object count of a label image, background excluded."""
    return len(np.unique(out_mask)) - 1


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

    Channel 0 and channel 1 carry *opposite* seam brightness, so the fate of
    the touching pair proves which channel was read: against the absolute
    threshold 500, a seam of 900 merges the pair into one object and a seam
    of 100 leaves two.  The two FOVs disagree, so a shared verdict would mean
    they were not judged independently.

    RE-POINTED, AND HERE IS THE ARITHMETIC.  This used to set
    ``cell_min_intensity_percentile=50`` on three disjoint boxes and watch the
    dimmest disappear.  That threshold was the MEDIAN OF THE THREE OBJECT
    MEANS in this very field, so the filter removed one of three however
    bright the field was -- it could not decline to fire.  The replacement is
    an absolute threshold in raw image units and it belongs to the MERGE,
    which judges the mean along a shared boundary.  On disjoint boxes it
    therefore does nothing at any value (10/50/90 with thresholds from 5 to
    1000 all leave three objects), so the fixture had to grow a boundary for
    the threshold to have something to measure.
    """
    from spacr.object import merge_split_filter_masks

    masks = np.stack([_touching_pair_mask(), _touching_pair_mask()])   # (2, 64, 64)

    # FOV 0 -> bright seam on channel 0 (dim on channel 1).
    fov0 = np.stack([_seam_plane(_SEAM_BRIGHT), _seam_plane(_SEAM_DIM)], axis=-1)
    # FOV 1 -> orderings swapped again.
    fov1 = np.stack([_seam_plane(_SEAM_DIM), _seam_plane(_SEAM_BRIGHT)], axis=-1)
    intensity = np.stack([fov0, fov1])                                # (2, 64, 64, 2)
    assert intensity.ndim == 4

    out = merge_split_filter_masks(masks, intensity, dict(_MERGE_ON), "cell")

    assert isinstance(out, list)
    assert len(out) == 2, "one output mask per FOV of the 4-D intensity stack"
    for arr in out:
        assert arr.shape == (64, 64)
        assert arr.dtype == np.uint16

    # Channel 0 drove the merge in each FOV, independently.
    assert _n_objects(out[0]) == 1, "FOV0: seam 900 >= 500 on channel 0 -> merged"
    assert _n_objects(out[1]) == 2, "FOV1: seam 100 < 500 on channel 0 -> kept apart"

    # Sequentially relabelled from 1 either way.
    assert np.unique(out[0]).tolist() == [0, 1]
    assert np.unique(out[1]).tolist() == [0, 1, 2]


def test_merge_split_filter_masks_4d_single_fov_matches_3d_equivalent():
    """(1, H, W, C) and (1, H, W) intensity stacks give the identical result.

    Guards the 4-D branch against silently selecting the wrong axis: the
    channel-last stack whose channel 0 equals the 3-D plane must reach the
    same verdict on the same pair.  Channel 1 is not merely garbage, it is a
    plane that would reach the OPPOSITE verdict, so reading it instead could
    not go unnoticed.

    RE-POINTED for the same reason as the test above: the percentile band it
    used to drive was removed, and the absolute threshold that replaced it
    measures a shared boundary, which disjoint boxes do not have.
    """
    from spacr.object import merge_split_filter_masks

    masks = _touching_pair_mask()[None, ...]                           # (1, 64, 64)
    plane = _seam_plane(_SEAM_BRIGHT)
    settings = dict(_MERGE_ON)

    out_3d = merge_split_filter_masks(masks, plane[None, ...], settings, "cell")
    # Channel 0 == plane (merges); channel 1 is a dim seam (would not merge).
    stack_4d = np.stack([plane, _seam_plane(_SEAM_DIM)], axis=-1)[None, ...]
    assert stack_4d.shape == (1, 64, 64, 2)
    out_4d = merge_split_filter_masks(masks, stack_4d, settings, "cell")

    assert len(out_3d) == len(out_4d) == 1
    np.testing.assert_array_equal(out_4d[0], out_3d[0])
    assert _n_objects(out_4d[0]) == 1, "channel 0 seam 900 >= 500 -> merged"


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
        "cell_min_split_area": 0,
        "cell_min_area": 0,
        "cell_max_area": 0,
        "cell_remove_border_objects": False,
        "cell_min_intensity_percentile": 0,
        "cell_max_intensity_percentile": 100,
    }
    out = merge_split_filter_masks(masks, None, settings, "cell")
    assert out is masks


def test_merge_split_filter_masks_intensity_percentile_band_stays_gone():
    """The percentile band neither enables work nor drops anything any more.

    REPLACES a test that asserted ``max_intensity_percentile < 100`` on its
    own tripped ``needs_work``, which it did, and that the band then kept only
    ``{1, 2}`` of three boxes, which it also did.  Both statements were true
    of a filter that has been removed, and the reason it was removed is
    visible in that old expectation: the cut was the 50th percentile of the
    three object means IN THIS FIELD, so it took one of the three whatever
    their raw intensities were.  Rescaling every box by ten would not have
    changed which one went.  A filter that cannot decline to fire is a quota.

    Two halves, and the second is the one with teeth.  Neither key is read
    now, so on its own each leaves the caller's array untouched; and supplied
    ALONGSIDE an operation that does run, they must still drop nothing, so all
    three boxes survive where the band would have kept only the median one.
    """
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()
    intensity = _intensity_plane({1: 10.0, 2: 50.0, 3: 90.0})

    for dead_key in ("cell_min_intensity_percentile",
                     "cell_max_intensity_percentile"):
        out = merge_split_filter_masks(masks, intensity, {dead_key: 50}, "cell")
        assert out is masks, f"{dead_key} must not switch work on by itself"

    # min_area=1 removes nothing -- every box is 100 px -- but it does enable
    # the pipeline, so the dead keys travel all the way through a real run.
    out = merge_split_filter_masks(
        masks, intensity,
        {"cell_min_area": 1,
         "cell_min_intensity_percentile": 50,
         "cell_max_intensity_percentile": 50},
        "cell",
    )
    assert isinstance(out, list) and len(out) == 1
    assert _surviving_labels(out[0]) == {1, 2, 3}, \
        "no object is dropped for being the dimmest of its field"


def test_merge_split_filter_masks_absolute_threshold_needs_the_merge_switch():
    """``intensity_threshold`` alone is a no-op; the merge switch enables it.

    The replacement setting is not an operation of its own -- it is the
    boundary mean at which the merge joins two touching labels -- so with the
    merge off it has nothing to parameterise and correctly leaves the caller's
    array alone.  This is the documented contract: the setting reads "Ignored
    unless cell_intensity_merge is True".  With the merge on, that same number
    decides the pair.
    """
    from spacr.object import merge_split_filter_masks

    masks = _touching_pair_mask()
    bright = _seam_plane(_SEAM_BRIGHT)

    out = merge_split_filter_masks(
        masks, bright, {"cell_intensity_threshold": _SEAM_THRESHOLD}, "cell")
    assert out is masks, "a threshold with the merge off has nothing to do"

    out = merge_split_filter_masks(masks, bright, dict(_MERGE_ON), "cell")
    assert isinstance(out, list) and len(out) == 1
    assert _n_objects(out[0]) == 1, "seam 900 >= threshold 500 -> merged"


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
    """List-of-2D inputs are accepted and processed one FOV at a time.

    The two FOVs carry seams on opposite sides of the absolute threshold, so a
    shared verdict would mean the list was not walked per-FOV.

    RE-POINTED off ``cell_min_intensity_percentile``, which no longer exists;
    the surviving intensity operation is the boundary merge, and it needs
    objects that touch.
    """
    from spacr.object import merge_split_filter_masks

    masks = [_touching_pair_mask(), _touching_pair_mask()]
    intensity = [_seam_plane(_SEAM_BRIGHT), _seam_plane(_SEAM_DIM)]
    out = merge_split_filter_masks(
        masks, intensity, dict(_MERGE_ON), "cell",
        batch_filenames=["fov_a.npy", "fov_b.npy"],
    )
    assert len(out) == 2
    assert _n_objects(out[0]) == 1, "fov_a: seam 900 >= 500 -> merged"
    assert _n_objects(out[1]) == 2, "fov_b: seam 100 < 500 -> kept apart"
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
