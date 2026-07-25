"""CPU coverage for the label merge/split/filter block of ``spacr.utils``.

Covers the small IO helpers (``_load_image`` / ``_save_image``), the
union-find plumbing, the defensive ``continue`` branches inside
``_merge_by_intensity`` and ``_split_by_watershed``, every intensity-image
shape branch of ``_process_single_fov_in_memory``, and the on-disk
``merge_split_objects`` / ``_process_single_fov`` pair.

Everything runs on tiny synthetic label images so the whole file is
sub-second and never touches the GPU, the network, or a display.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# synthetic label-image builders
# ---------------------------------------------------------------------------

def _two_touching_blocks(size=16):
    """Two rectangles sharing a 10-px vertical edge (labels 1 and 2).

    Perimeters are 28 px each and the shared boundary is 20 px, so a
    ``perimeter_fraction`` below ~0.71 merges them.
    """
    m = np.zeros((size, size), dtype=np.uint16)
    m[2:12, 2:8] = 1
    m[2:12, 8:14] = 2
    return m


def _dumbbell():
    """One label: two discs joined by a thin neck -> watershed splits it."""
    m = np.zeros((40, 80), dtype=np.uint16)
    yy, xx = np.ogrid[:40, :80]
    m[(xx - 20) ** 2 + (yy - 20) ** 2 < 150] = 1
    m[(xx - 60) ** 2 + (yy - 20) ** 2 < 150] = 1
    m[18:22, 20:60] = 1
    return m


def _big_square_plus_speck():
    """A 30x30 solid square (label 1) and a 2x2 speck (label 2)."""
    m = np.zeros((60, 60), dtype=np.uint16)
    m[10:40, 10:40] = 1
    m[50:52, 50:52] = 2
    return m


def _n_objects(label_img):
    return int(len(np.unique(label_img)) - (1 if 0 in np.unique(label_img) else 0))


def _in_memory_kwargs(**overrides):
    """Neutral (all-phases-off) kwargs for ``_process_single_fov_in_memory``."""
    kw = dict(
        mask=None,
        intensity_img=None,
        intensity_channel=None,
        do_split=False,
        do_perimeter_merge=False,
        do_intensity_merge=False,
        perimeter_fraction=0.5,
        area_multiplier=2.0,
        min_distance=10,
        min_object_area=100,
        intensity_threshold_method="mean",
        intensity_percentile=75,
        min_area=0,
        max_area=0,
        remove_border_objects=False,
        min_intensity_percentile=0,
        max_intensity_percentile=100,
    )
    kw.update(overrides)
    return kw


def _fov_args(mask_path, intensity_path=None, intensity_channel=None,
              do_split=False, do_perimeter_merge=True, do_intensity_merge=False,
              perimeter_fraction=0.5, area_multiplier=2.0, min_distance=10,
              min_object_area=100, intensity_threshold_method="mean",
              intensity_percentile=75, min_area=0, max_area=0,
              remove_border_objects=False, min_intensity_percentile=0,
              max_intensity_percentile=100, **kw):
    """Positional argument tuple for the file-based ``_process_single_fov``."""
    return (
        (mask_path, intensity_path, intensity_channel,
         do_split, do_perimeter_merge, do_intensity_merge,
         perimeter_fraction, area_multiplier, min_distance,
         min_object_area, intensity_threshold_method,
         intensity_percentile, min_area, max_area,
         remove_border_objects, min_intensity_percentile,
         max_intensity_percentile),
        kw,
    )


@pytest.fixture(autouse=True)
def _no_stray_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# _load_image / _save_image
# ---------------------------------------------------------------------------

def test_load_save_image_npy_roundtrip(tmp_path):
    from spacr.utils import _load_image, _save_image
    img = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = str(tmp_path / "mask.npy")
    _save_image(path, img)
    assert os.path.exists(path)
    back = _load_image(path)
    assert back.dtype == np.uint16
    assert np.array_equal(back, img)


def test_load_save_image_tif_roundtrip(tmp_path):
    from spacr.utils import _load_image, _save_image
    img = (np.arange(9, dtype=np.uint16) * 7).reshape(3, 3)
    path = str(tmp_path / "mask.tif")
    _save_image(path, img)
    back = _load_image(path)
    assert back.shape == (3, 3)
    assert np.array_equal(back, img)


def test_load_image_tiff_extension_and_case_insensitive(tmp_path):
    from spacr.utils import _load_image
    img = np.eye(4, dtype=np.uint16) * 3
    path = str(tmp_path / "mask.TIFF")
    tifffile.imwrite(path, img)
    assert np.array_equal(_load_image(path), img)


def test_load_image_unsupported_extension_returns_none(tmp_path):
    from spacr.utils import _load_image
    path = tmp_path / "mask.png"
    path.write_bytes(b"not an image")
    assert _load_image(str(path)) is None


def test_save_image_falls_back_to_tiff_for_unknown_extension(tmp_path):
    """Anything that is not ``.npy`` is written as TIFF regardless of suffix."""
    from spacr.utils import _load_image, _save_image
    img = np.arange(4, dtype=np.uint16).reshape(2, 2)
    path = str(tmp_path / "mask.png")
    _save_image(path, img)
    # _load_image refuses the extension, but the bytes on disk are a TIFF.
    assert _load_image(path) is None
    assert np.array_equal(tifffile.imread(path), img)


# ---------------------------------------------------------------------------
# _apply_union_find
# ---------------------------------------------------------------------------

def test_apply_union_find_passes_through_labels_absent_from_parent():
    """Label ids present in the id-space but missing from ``parent`` map to
    themselves (the ``else`` branch of the mapping loop)."""
    from spacr.utils import _apply_union_find
    m = np.zeros((8, 8), dtype=np.uint16)
    m[1:3, 1:3] = 1
    m[5:7, 5:7] = 3           # id 2 is a hole in the label space
    parent = {1: 1, 3: 3}     # ... and is absent from parent

    out = _apply_union_find(m, parent)

    assert out.dtype == np.uint16
    assert out.shape == m.shape
    assert sorted(np.unique(out).tolist()) == [0, 1, 2]
    assert out[1, 1] != out[5, 5]
    assert int(np.sum(out == out[1, 1])) == 4
    assert int(np.sum(out == out[5, 5])) == 4


def test_apply_union_find_merges_united_labels():
    from spacr.utils import _apply_union_find, _union_find_merge
    m = _two_touching_blocks()
    parent = {1: 1, 2: 2}
    _union_find_merge(parent, 1, 2)

    out = _apply_union_find(m, parent)

    assert sorted(np.unique(out).tolist()) == [0, 1]
    assert int(np.sum(out > 0)) == int(np.sum(m > 0))


# ---------------------------------------------------------------------------
# _merge_by_intensity defensive branches
# ---------------------------------------------------------------------------

def test_merge_by_intensity_skips_pair_without_boundary_coords(monkeypatch):
    """A touching pair whose boundary lookup comes back empty is skipped."""
    import spacr.utils as U
    m = _two_touching_blocks()
    intensity = np.ones(m.shape, dtype=np.float32)  # uniform => would merge
    monkeypatch.setattr(U, "_get_boundary_coords", lambda *a, **k: [])

    parent = {1: 1, 2: 2}
    U._merge_by_intensity(m, intensity, parent)

    assert parent == {1: 1, 2: 2}
    assert U._union_find_root(parent, 1) != U._union_find_root(parent, 2)


def test_merge_by_intensity_skips_pair_with_label_missing_from_stats(monkeypatch):
    """A shared-boundary pair naming a label absent from the image is skipped."""
    import spacr.utils as U
    m = _two_touching_blocks()
    intensity = np.ones(m.shape, dtype=np.float32)
    # label 99 does not exist in `m`, so its intensity stats are missing
    monkeypatch.setattr(U, "_compute_shared_boundaries", lambda img: {(1, 99): 20})
    monkeypatch.setattr(U, "_get_boundary_coords",
                        lambda img, la, lb: [(5, 7), (5, 8)])

    parent = {1: 1, 2: 2, 99: 99}
    U._merge_by_intensity(m, intensity, parent)

    assert parent == {1: 1, 2: 2, 99: 99}


def test_merge_by_intensity_percentile_method_keeps_dark_boundary_split():
    """A genuinely dark boundary is *not* merged under the percentile rule."""
    from spacr.utils import _merge_by_intensity, _union_find_root
    m = _two_touching_blocks()
    intensity = np.zeros(m.shape, dtype=np.float32)
    intensity[m > 0] = 100.0
    intensity[:, 7:9] = 0.0     # dark seam exactly on the shared boundary

    parent = {1: 1, 2: 2}
    _merge_by_intensity(m, intensity, parent,
                        intensity_threshold_method="percentile",
                        intensity_percentile=50)

    assert _union_find_root(parent, 1) != _union_find_root(parent, 2)


# ---------------------------------------------------------------------------
# _split_by_watershed skip branches
# ---------------------------------------------------------------------------

def test_split_by_watershed_skips_small_and_single_peak_objects(capsys):
    """Small objects fall under the area threshold; a convex object above the
    threshold has a single distance maximum and is left alone."""
    from spacr.utils import _split_by_watershed
    m = _big_square_plus_speck()

    out = _split_by_watershed(m, area_multiplier=0.5, min_distance=10,
                              min_object_area=10)

    # threshold = max(0.5 * median(900, 4), 10) = 226
    #   label 2 (4 px)   -> below threshold, skipped
    #   label 1 (900 px) -> above threshold but a single peak, skipped
    assert np.array_equal(out, m)
    assert sorted(np.unique(out).tolist()) == [0, 1, 2]


def test_split_by_watershed_splits_dumbbell():
    from spacr.utils import _split_by_watershed
    m = _dumbbell()
    assert _n_objects(m) == 1

    out = _split_by_watershed(m, area_multiplier=0.1, min_distance=5,
                              min_object_area=10)

    assert _n_objects(out) >= 2
    # splitting only re-labels foreground; it never grows or erodes it
    assert np.array_equal(out > 0, m > 0)


# ---------------------------------------------------------------------------
# _process_single_fov_in_memory
# ---------------------------------------------------------------------------

def test_in_memory_none_mask_returns_none():
    from spacr.utils import _process_single_fov_in_memory
    assert _process_single_fov_in_memory(**_in_memory_kwargs(mask=None)) is None


def test_in_memory_empty_mask_short_circuits(capsys):
    from spacr.utils import _process_single_fov_in_memory
    empty = np.zeros((12, 12), dtype=np.int32)

    out = _process_single_fov_in_memory(
        **_in_memory_kwargs(mask=empty, do_perimeter_merge=True, fov_index=7))

    assert out.dtype == np.uint16
    assert out.shape == (12, 12)
    assert not out.any()
    assert "empty mask" in capsys.readouterr().out


def test_in_memory_perimeter_merge_and_progress_callback():
    from spacr.utils import _process_single_fov_in_memory
    calls = []

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=_two_touching_blocks(),
        do_perimeter_merge=True,
        perimeter_fraction=0.5,
        progress_callback=lambda i, t, d, n: calls.append((i, t, d, n)),
        fov_index=3, total_fovs=9, op_name="merge_cell",
    ))

    assert _n_objects(out) == 1
    assert int(np.sum(out > 0)) == 120
    assert len(calls) == 1
    idx, total, duration, name = calls[0]
    assert (idx, total, name) == (3, 9, "merge_cell")
    assert duration >= 0.0


def test_in_memory_split_phase_increases_object_count(capsys):
    from spacr.utils import _process_single_fov_in_memory

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=_dumbbell(),
        do_split=True,
        area_multiplier=0.1,
        min_distance=5,
        min_object_area=10,
        fov_index=1,
    ))

    assert _n_objects(out) >= 2
    assert "split: 1 →" in capsys.readouterr().out


def test_in_memory_intensity_merge_2d_intensity(capsys):
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m,
        intensity_img=np.full(m.shape, 5.0, dtype=np.float64),
        do_intensity_merge=True,
        fov_index=2,
    ))

    assert _n_objects(out) == 1
    assert "merge: 2 → 1" in capsys.readouterr().out


def test_in_memory_intensity_channel_last_small_stack():
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, 3), dtype=np.uint16)
    raw[..., 1] = 9        # only channel 1 is bright -> uniform -> merge

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=1, do_intensity_merge=True))

    assert _n_objects(out) == 1


def test_in_memory_intensity_channel_last_out_of_bounds():
    from spacr.utils import _process_single_fov_in_memory
    with pytest.raises(ValueError, match="channel-last"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=_two_touching_blocks(),
            intensity_img=np.zeros((16, 16, 3), dtype=np.uint16),
            intensity_channel=5,
            do_intensity_merge=True,
        ))


def test_in_memory_intensity_channel_first_small_stack():
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((3, 16, 16), dtype=np.uint16)
    raw[1] = 4

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=1, do_intensity_merge=True))

    assert _n_objects(out) == 1


def test_in_memory_intensity_channel_first_out_of_bounds():
    from spacr.utils import _process_single_fov_in_memory
    with pytest.raises(ValueError, match="channel-first"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=_two_touching_blocks(),
            intensity_img=np.zeros((3, 16, 16), dtype=np.uint16),
            intensity_channel=5,
            do_intensity_merge=True,
        ))


def test_in_memory_intensity_many_channel_stack_uses_last_axis():
    """Neither axis looks like a channel axis (both > 4) -> channel-last."""
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, 6), dtype=np.uint16)
    raw[..., 5] = 11

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=5, do_intensity_merge=True))

    assert _n_objects(out) == 1


def test_in_memory_intensity_many_channel_stack_out_of_bounds():
    from spacr.utils import _process_single_fov_in_memory
    with pytest.raises(ValueError, match=r"out of bounds for image with shape"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=_two_touching_blocks(),
            intensity_img=np.zeros((16, 16, 6), dtype=np.uint16),
            intensity_channel=10,
            do_intensity_merge=True,
        ))


def test_in_memory_intensity_3d_without_channel_is_passed_through():
    """``intensity_channel=None`` on a 3-D stack falls to the generic cast."""
    from spacr.utils import _process_single_fov_in_memory
    m = np.zeros((16, 16), dtype=np.uint16)
    m[3:9, 3:9] = 1                                  # a single object, so the
    raw = np.ones((3, 16, 16), dtype=np.uint16)      # percentile filter no-ops

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=None,
        min_intensity_percentile=5))

    assert _n_objects(out) == 1
    assert out.shape == (16, 16)
    assert int(np.sum(out > 0)) == 36


def test_in_memory_area_filter_drops_speck():
    from spacr.utils import _process_single_fov_in_memory

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=_big_square_plus_speck(), min_area=100))

    assert _n_objects(out) == 1
    assert int(np.sum(out > 0)) == 900


# ---------------------------------------------------------------------------
# _process_single_fov (file based)
# ---------------------------------------------------------------------------

def test_process_single_fov_unreadable_mask_returns_none(tmp_path):
    from spacr.utils import _process_single_fov
    path = tmp_path / "mask.png"
    path.write_bytes(b"junk")
    args, kw = _fov_args(str(path))

    assert _process_single_fov(*args, **kw) is None
    assert path.read_bytes() == b"junk"      # untouched


def test_process_single_fov_merges_and_overwrites_npy(tmp_path):
    from spacr.utils import _process_single_fov
    path = tmp_path / "mask.npy"
    np.save(path, _two_touching_blocks())
    calls = []
    args, kw = _fov_args(str(path), perimeter_fraction=0.5,
                         progress_callback=lambda *a: calls.append(a),
                         fov_index=4, total_fovs=5, op_name="merge_nucleus")

    _process_single_fov(*args, **kw)

    out = np.load(path)
    assert out.dtype == np.uint16
    assert sorted(np.unique(out).tolist()) == [0, 1]
    assert int(np.sum(out > 0)) == 120
    assert len(calls) == 1 and calls[0][0] == 4 and calls[0][3] == "merge_nucleus"


def test_process_single_fov_split_writes_more_objects(tmp_path):
    from spacr.utils import _process_single_fov
    path = tmp_path / "mask.tif"
    tifffile.imwrite(str(path), _dumbbell())
    args, kw = _fov_args(str(path), do_split=True, do_perimeter_merge=False,
                         perimeter_fraction=0.0, area_multiplier=0.1,
                         min_distance=5, min_object_area=10)

    _process_single_fov(*args, **kw)

    out = tifffile.imread(str(path))
    assert _n_objects(out) >= 2
    assert np.array_equal(out > 0, _dumbbell() > 0)


def test_process_single_fov_channel_first_intensity_merge(tmp_path):
    from spacr.utils import _process_single_fov
    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    tifffile.imwrite(str(mask_path), _two_touching_blocks())
    raw = np.zeros((3, 16, 16), dtype=np.float32)
    raw[0] = 7.0
    tifffile.imwrite(str(int_path), raw)
    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=0,
                         do_perimeter_merge=False, do_intensity_merge=True,
                         perimeter_fraction=0.0)

    _process_single_fov(*args, **kw)

    out = tifffile.imread(str(mask_path))
    assert sorted(np.unique(out).tolist()) == [0, 1]


def test_process_single_fov_intensity_without_channel(tmp_path):
    """``intensity_channel=None`` uses the whole intensity image."""
    from spacr.utils import _process_single_fov
    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    m = _two_touching_blocks()
    tifffile.imwrite(str(mask_path), m)
    intensity = np.zeros(m.shape, dtype=np.float32)
    intensity[m == 1] = 10.0
    intensity[m == 2] = 200.0
    tifffile.imwrite(str(int_path), intensity)
    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=None,
                         do_perimeter_merge=False, perimeter_fraction=0.0,
                         min_intensity_percentile=50)

    _process_single_fov(*args, **kw)

    out = tifffile.imread(str(mask_path))
    # the dim object (label 1) sits below the 50th percentile and is dropped
    assert sorted(np.unique(out).tolist()) == [0, 1]
    assert int(np.sum(out > 0)) == 60
    assert bool(np.all(out[m == 1] == 0))


def test_process_single_fov_unreadable_intensity_is_ignored(tmp_path):
    from spacr.utils import _process_single_fov
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), _two_touching_blocks())
    bad_intensity = tmp_path / "intensity.png"
    bad_intensity.write_bytes(b"junk")
    args, kw = _fov_args(str(mask_path), str(bad_intensity), intensity_channel=0,
                         do_intensity_merge=True, perimeter_fraction=0.5)

    _process_single_fov(*args, **kw)

    # intensity merge silently disabled; perimeter merge still ran
    out = tifffile.imread(str(mask_path))
    assert sorted(np.unique(out).tolist()) == [0, 1]


def test_process_single_fov_2d_intensity_with_channel(tmp_path):
    from spacr.utils import _process_single_fov, _process_single_fov_in_memory
    m = _two_touching_blocks()
    intensity = np.full(m.shape, 5.0, dtype=np.float32)

    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    tifffile.imwrite(str(mask_path), m)
    tifffile.imwrite(str(int_path), intensity)

    expected = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=intensity, intensity_channel=0,
        do_intensity_merge=True, perimeter_fraction=0.0))

    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=0,
                         do_perimeter_merge=False, do_intensity_merge=True,
                         perimeter_fraction=0.0)
    _process_single_fov(*args, **kw)

    assert np.array_equal(tifffile.imread(str(mask_path)), expected)


def test_process_single_fov_channel_last_intensity(tmp_path):
    from spacr.utils import _process_single_fov, _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, 3), dtype=np.float32)
    raw[..., 1] = 6.0

    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    tifffile.imwrite(str(mask_path), m)
    tifffile.imwrite(str(int_path), raw)

    expected = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=1,
        do_intensity_merge=True, perimeter_fraction=0.0))

    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=1,
                         do_perimeter_merge=False, do_intensity_merge=True,
                         perimeter_fraction=0.0)
    _process_single_fov(*args, **kw)

    assert np.array_equal(tifffile.imread(str(mask_path)), expected)


# ---------------------------------------------------------------------------
# merge_split_objects (directory driver)
# ---------------------------------------------------------------------------

def test_merge_split_objects_no_masks_is_a_noop(tmp_path):
    from spacr.utils import merge_split_objects
    (tmp_path / "notes.txt").write_text("no masks here")

    assert merge_split_objects(str(tmp_path)) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_merge_split_objects_processes_every_mask_in_place(tmp_path):
    from spacr.utils import merge_split_objects
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(str(mask_dir / "A01_f01.tif"), _two_touching_blocks())
    np.save(mask_dir / "A01_f02.npy", _two_touching_blocks())
    (mask_dir / "README.md").write_text("ignored by the extension filter")

    calls = []
    merge_split_objects(str(mask_dir), perimeter_fraction=0.5, n_jobs=1,
                        progress_callback=lambda *a: calls.append(a),
                        op_name="merge_cell")

    tif_out = tifffile.imread(str(mask_dir / "A01_f01.tif"))
    npy_out = np.load(mask_dir / "A01_f02.npy")
    for out in (tif_out, npy_out):
        assert out.dtype == np.uint16
        assert sorted(np.unique(out).tolist()) == [0, 1]
        assert int(np.sum(out > 0)) == 120
    assert (mask_dir / "README.md").read_text().startswith("ignored")
    assert [c[0] for c in calls] == [0, 1]
    assert {c[1] for c in calls} == {2}
    assert {c[3] for c in calls} == {"merge_cell"}


def test_merge_split_objects_with_intensity_directory(tmp_path):
    from spacr.utils import merge_split_objects
    mask_dir = tmp_path / "masks"
    int_dir = tmp_path / "intensity"
    mask_dir.mkdir()
    int_dir.mkdir()
    m = _two_touching_blocks()
    tifffile.imwrite(str(mask_dir / "A01_f01.tif"), m)
    raw = np.zeros((3, 16, 16), dtype=np.float32)
    raw[2] = 3.0                       # uniform bright channel -> merge
    tifffile.imwrite(str(int_dir / "A01_f01.tif"), raw)

    merge_split_objects(str(mask_dir), intensity_img_src=str(int_dir),
                        intensity_channel=2, perimeter_fraction=0.0,
                        intensity_merge=True, n_jobs=1)

    out = tifffile.imread(str(mask_dir / "A01_f01.tif"))
    assert sorted(np.unique(out).tolist()) == [0, 1]
    assert int(np.sum(out > 0)) == 120


def test_merge_split_objects_split_and_area_filter(tmp_path):
    from spacr.utils import merge_split_objects
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(str(mask_dir / "d.tif"), _dumbbell())

    merge_split_objects(str(mask_dir), perimeter_fraction=0.0,
                        intensity_split=True, area_multiplier=0.1,
                        min_distance=5, min_object_area=10, min_area=20,
                        n_jobs=1)

    out = tifffile.imread(str(mask_dir / "d.tif"))
    assert _n_objects(out) >= 2
    # every surviving object clears the 20 px minimum area
    ids, counts = np.unique(out[out > 0], return_counts=True)
    assert bool(np.all(counts >= 20))
