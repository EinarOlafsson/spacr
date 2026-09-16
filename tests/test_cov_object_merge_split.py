"""Coverage for :func:`spacr.object.merge_split_filter_masks`.

Focuses on the input-normalisation branches that decide how ``masks`` and
``intensity_images`` are unpacked into per-FOV lists, in particular the
4-D (N, H, W, C) intensity layout that ``generate_cellpose_masks_sam``
passes in with the object's original channel first, and the guards that
reject unsupported ranks and mismatched field counts.

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


_MEAN_BOUNDS = {
    "cell_min_intensity": 50.25,
    "cell_max_intensity": 90.5,
}


def _expected_mask(kept):
    """Independent full label image for an explicit set of surviving boxes."""
    expected = np.zeros((64, 64), dtype=np.uint16)
    for new_label, original_label in enumerate(kept, 1):
        expected[_BOXES[original_label]] = new_label
    return expected


@pytest.fixture(autouse=True)
def _no_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 4-D intensity images: (N, H, W, C)
# ---------------------------------------------------------------------------

def test_merge_split_filter_masks_accepts_4d_channel_last_intensity():
    """Each FOV uses channel 0's raw means and keeps equality at both bounds."""
    from spacr.object import merge_split_filter_masks

    masks = np.stack([_three_box_mask(), _three_box_mask()])
    first = _intensity_plane({1: 10.25, 2: 50.25, 3: 90.5})
    second = _intensity_plane({1: 80.5, 2: 120.25, 3: 20.5})
    # The other channel would give the opposite field's surviving labels.
    fov0 = np.stack([first, second], axis=-1)
    fov1 = np.stack([second, first], axis=-1)
    intensity = np.stack([fov0, fov1])
    assert intensity.ndim == 4

    out = merge_split_filter_masks(masks, intensity, dict(_MEAN_BOUNDS), "cell")

    assert isinstance(out, list)
    assert len(out) == 2, "one output mask per FOV of the 4-D intensity stack"
    for arr in out:
        assert arr.shape == (64, 64)
        assert arr.dtype == np.uint16

    np.testing.assert_array_equal(out[0], _expected_mask((2, 3)))
    np.testing.assert_array_equal(out[1], _expected_mask((1,)))
    np.testing.assert_array_equal(masks[0], _three_box_mask())
    np.testing.assert_array_equal(masks[1], _three_box_mask())


def test_merge_split_filter_masks_4d_single_fov_matches_3d_equivalent():
    """(1, H, W, C) and (1, H, W) give the same independently expected labels."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()[None, ...]
    plane = _intensity_plane({1: 10.25, 2: 50.25, 3: 90.5})
    settings = dict(_MEAN_BOUNDS)

    out_3d = merge_split_filter_masks(masks, plane[None, ...], settings, "cell")
    other = _intensity_plane({1: 80.5, 2: 120.25, 3: 20.5})
    stack_4d = np.stack([plane, other], axis=-1)[None, ...]
    assert stack_4d.shape == (1, 64, 64, 2)
    out_4d = merge_split_filter_masks(masks, stack_4d, settings, "cell")

    assert len(out_3d) == len(out_4d) == 1
    np.testing.assert_array_equal(out_4d[0], out_3d[0])
    np.testing.assert_array_equal(out_4d[0], _expected_mask((2, 3)))


def test_merge_split_filter_masks_channel_last_layout_does_not_depend_on_channel_count():
    """A large channel axis remains the explicit last axis."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()[None, ...]
    intensity = np.zeros((1, 64, 64, 16), dtype=np.float64)
    intensity[0, ..., 0] = _intensity_plane({1: 10.25, 2: 50.25, 3: 90.5})
    out = merge_split_filter_masks(masks, intensity, dict(_MEAN_BOUNDS), "cell")
    np.testing.assert_array_equal(out[0], _expected_mask((2, 3)))


def test_merge_split_filter_masks_does_not_guess_channel_first_layout():
    """A channel-first field is refused rather than silently reinterpreted."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()[None, ...]
    intensity = np.zeros((1, 2, 64, 64), dtype=np.float64)
    with pytest.raises(ValueError, match="same shape as the mask"):
        merge_split_filter_masks(masks, intensity, dict(_MEAN_BOUNDS), "cell")


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
        "cell_min_area": 0,
        "cell_max_area": 0,
        "cell_remove_border_objects": False,
        "cell_min_intensity": 0,
        "cell_max_intensity": 0,
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


@pytest.mark.parametrize("settings,kept", [
    ({"cell_min_intensity": 50.25}, (2, 3)),
    ({"cell_max_intensity": 50.25}, (1, 2)),
    ({"cell_min_intensity": 91}, ()),
    ({"cell_max_intensity": 100}, (1, 2, 3)),
])
def test_each_mean_bound_enables_filtering_without_other_operations(settings, kept):
    """Either bound is sufficient; absolute limits can retain all or none."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()
    intensity = _intensity_plane({1: 10.25, 2: 50.25, 3: 90.5})
    out = merge_split_filter_masks(masks, intensity, settings, "cell")
    assert isinstance(out, list) and len(out) == 1
    np.testing.assert_array_equal(out[0], _expected_mask(kept))


def test_area_and_border_filters_do_not_require_an_intensity_image():
    """Disabled mean bounds permit None while every other filter still runs."""
    from spacr.object import merge_split_filter_masks

    masks = _three_box_mask()
    masks[0:5, 20:25] = 4            # 25 px: rejected only by border contact
    masks[25:27, 25:27] = 5          # 4 px: below the area floor
    masks[30:41, 30:41] = 6          # 121 px: above the area ceiling
    out = merge_split_filter_masks(
        masks, None, {"cell_min_area": 10, "cell_max_area": 100,
                      "cell_remove_border_objects": True,
                      "cell_min_intensity": 0, "cell_max_intensity": 0}, "cell")
    np.testing.assert_array_equal(out[0], _expected_mask((1, 2, 3)))


def test_enabled_mean_bounds_require_the_intensity_image():
    from spacr.object import merge_split_filter_masks

    with pytest.raises(ValueError, match="same shape as the mask"):
        merge_split_filter_masks(
            _three_box_mask(), None, dict(_MEAN_BOUNDS), "cell")


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
    merged = merge_split_filter_masks(
        m.copy(), None, {"cell_perimiter_fraction": 0.1}, "cell")
    assert len(np.unique(merged[0])) - 1 == 1, "the two halves merged"

    # Sanity: without the key nothing merges (needs_work driven by min_area=1,
    # which removes nothing since both objects are 200 px).
    untouched = merge_split_filter_masks(
        m.copy(), None, {"cell_min_area": 1}, "cell")
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


def test_merge_split_filter_masks_list_inputs_and_batch_filenames(monkeypatch):
    """List inputs preserve field order and report progress once per field."""
    from spacr.object import merge_split_filter_masks
    import spacr.utils as utils

    progress = []

    def capture_progress(current, total, **kwargs):
        progress.append((current, total, dict(kwargs, time_ls=list(kwargs["time_ls"]))))

    monkeypatch.setattr(utils, "print_progress", capture_progress)
    masks = [_three_box_mask(), _three_box_mask()]
    intensity = [_intensity_plane({1: 10.25, 2: 50.25, 3: 90.5}),
                 _intensity_plane({1: 80.5, 2: 120.25, 3: 20.5})]
    out = merge_split_filter_masks(
        masks, intensity, dict(_MEAN_BOUNDS), "cell",
        batch_filenames=["fov_a.npy", "fov_b.npy"],
    )
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], _expected_mask((2, 3)))
    np.testing.assert_array_equal(out[1], _expected_mask((1,)))
    assert len(masks) == 2
    assert [(current, total) for current, total, _ in progress] == [(1, 2), (2, 2)]
    for index, (_, _, kwargs) in enumerate(progress, 1):
        assert kwargs["n_jobs"] == 1
        assert kwargs["batch_size"] is None
        assert kwargs["operation_type"] == "merge_cell"
        assert len(kwargs["time_ls"]) == index
        assert all(duration >= 0 for duration in kwargs["time_ls"])


def test_merge_split_filter_masks_rejects_bad_mask_ndim():
    """A 4-D mask array is rejected (only 2-D/3-D label images are meaningful)."""
    from spacr.object import merge_split_filter_masks

    bad = np.zeros((2, 4, 4, 4), dtype=np.int32)
    with pytest.raises(ValueError) as ei:
        merge_split_filter_masks(
            bad, np.zeros((2, 4, 4), dtype=np.float32),
            {"cell_min_area": 3}, "cell")
    assert "Unsupported masks ndim: 4" in str(ei.value)
