"""CPU coverage for the label merge/filter block of ``spacr.utils``.

Covers the small IO helpers (``_load_image`` / ``_save_image``), the
union-find plumbing, original own-channel mean-intensity filtering in
``_process_single_fov_in_memory``, and the on-disk
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
        do_perimeter_merge=False,
        perimeter_fraction=0.5,
        min_area=0,
        max_area=0,
        remove_border_objects=False,
    )
    kw.update(overrides)
    return kw


def _fov_args(mask_path, intensity_path=None, intensity_channel=None,
              do_perimeter_merge=True, perimeter_fraction=0.5,
              min_area=0, max_area=0,
              remove_border_objects=False, **kw):
    """Positional argument tuple for the file-based ``_process_single_fov``."""
    return (
        (mask_path, intensity_path, intensity_channel,
         do_perimeter_merge, perimeter_fraction,
         min_area, max_area, remove_border_objects),
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


def test_in_memory_intensity_bounds_use_object_mean_not_maximum(capsys):
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    intensity = np.zeros(m.shape, dtype=np.float64)
    intensity[m == 1] = 5
    intensity[2, 8] = 1200  # label 2: 60 pixels, mean 20, maximum 1200

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m,
        intensity_img=intensity,
        min_intensity=10, max_intensity=25,
        fov_index=2,
    ))

    np.testing.assert_array_equal(out, (m == 2).astype(np.uint16))
    assert "Intensity filter: removed 1" in capsys.readouterr().out


def test_in_memory_intensity_channel_last_preserves_raw_precision():
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, 3), dtype=np.uint32)
    exact = 2**24 + 1  # float32 would round this below the inclusive bound
    raw[m == 1, 1] = exact
    raw[m == 2, 1] = exact + 2
    original = raw.copy()

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=1,
        min_intensity=exact, max_intensity=exact))

    np.testing.assert_array_equal(out, (m == 1).astype(np.uint16))
    np.testing.assert_array_equal(raw, original)


def test_in_memory_intensity_channel_last_out_of_bounds():
    from spacr.utils import _process_single_fov_in_memory
    with pytest.raises(IndexError, match="out of bounds"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=_two_touching_blocks(),
            intensity_img=np.zeros((16, 16, 3), dtype=np.uint16),
            intensity_channel=5,
            min_intensity=1,
        ))


def test_in_memory_channel_first_requires_explicit_axis_conversion():
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((3, 16, 16), dtype=np.uint16)
    raw[1, m == 1] = 9
    raw[1, m == 2] = 3

    with pytest.raises(ValueError, match="same shape"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=m, intensity_img=raw, intensity_channel=1, min_intensity=5))

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=np.moveaxis(raw, 0, -1),
        intensity_channel=1, min_intensity=5))

    np.testing.assert_array_equal(out, (m == 1).astype(np.uint16))


def test_in_memory_short_spatial_axis_is_not_guessed_to_be_channels():
    from spacr.utils import _process_single_fov_in_memory
    m = np.zeros((3, 16), dtype=np.uint16)
    m[1, 2:6] = 1
    m[1, 10:14] = 2
    raw = np.zeros((3, 16, 6), dtype=np.uint16)
    raw[m == 1, 5] = 4
    raw[m == 2, 5] = 14

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=5, min_intensity=7))

    np.testing.assert_array_equal(out, (m == 2).astype(np.uint16))


def test_in_memory_intensity_many_channel_stack_uses_last_axis():
    """Explicit channel-last selection also works beyond four channels."""
    from spacr.utils import _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, 6), dtype=np.uint16)
    raw[m == 1, 5] = 5
    raw[m == 2, 5] = 11

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=5,
        min_intensity=10, max_intensity=12))

    np.testing.assert_array_equal(out, (m == 2).astype(np.uint16))


def test_in_memory_intensity_many_channel_stack_out_of_bounds():
    from spacr.utils import _process_single_fov_in_memory
    with pytest.raises(IndexError, match="out of bounds"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=_two_touching_blocks(),
            intensity_img=np.zeros((16, 16, 6), dtype=np.uint16),
            intensity_channel=10,
            min_intensity=1,
        ))


def test_in_memory_multichannel_intensity_requires_an_explicit_channel():
    from spacr.utils import _process_single_fov_in_memory
    m = np.zeros((16, 16), dtype=np.uint16)
    m[3:9, 3:9] = 1
    raw = np.zeros((16, 16, 3), dtype=np.uint16)
    raw[..., 1] = 9

    with pytest.raises(ValueError, match="explicit own-channel index"):
        _process_single_fov_in_memory(**_in_memory_kwargs(
            mask=m, intensity_img=raw, min_intensity=8))

    out = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=1,
        min_intensity=8, max_intensity=10))

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


def test_process_single_fov_rejects_channel_first_without_overwriting(tmp_path):
    from spacr.utils import _process_single_fov
    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    m = _two_touching_blocks()
    tifffile.imwrite(str(mask_path), m)
    raw = np.zeros((3, 16, 16), dtype=np.float32)
    raw[0, m == 1] = 3
    raw[0, m == 2] = 7
    tifffile.imwrite(str(int_path), raw)
    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=0,
                         do_perimeter_merge=False, perimeter_fraction=0.0,
                         min_intensity=5)

    before = mask_path.read_bytes()
    with pytest.raises(ValueError, match="same shape"):
        _process_single_fov(*args, **kw)
    assert mask_path.read_bytes() == before

    tifffile.imwrite(str(int_path), np.moveaxis(raw, 0, -1))
    _process_single_fov(*args, **kw)

    out = tifffile.imread(str(mask_path))
    np.testing.assert_array_equal(out, (m == 2).astype(np.uint16))


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
                         min_intensity=100, max_intensity=300)

    _process_single_fov(*args, **kw)

    out = tifffile.imread(str(mask_path))
    np.testing.assert_array_equal(out, (m == 2).astype(np.uint16))
    assert int(np.sum(out > 0)) == 60


def test_process_single_fov_requires_readable_intensity_only_for_active_bounds(tmp_path):
    from spacr.utils import _process_single_fov
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), _two_touching_blocks())
    bad_intensity = tmp_path / "intensity.png"
    bad_intensity.write_bytes(b"junk")
    args, kw = _fov_args(str(mask_path), str(bad_intensity), intensity_channel=0,
                         perimeter_fraction=0.5)

    _process_single_fov(*args, **kw)

    # A perimeter-only operation does not need the unreadable intensity file.
    out = tifffile.imread(str(mask_path))
    assert sorted(np.unique(out).tolist()) == [0, 1]
    assert int(np.sum(out > 0)) == 120
    before = mask_path.read_bytes()
    with pytest.raises(ValueError, match="intensity plane.*same shape"):
        _process_single_fov(*args, **kw, min_intensity=1)
    assert mask_path.read_bytes() == before


def test_process_single_fov_2d_intensity_with_channel(tmp_path):
    from spacr.utils import _process_single_fov, _process_single_fov_in_memory
    m = _two_touching_blocks()
    intensity = np.full(m.shape, 5.0, dtype=np.float32)
    intensity[m == 2] = 15

    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    tifffile.imwrite(str(mask_path), m)
    tifffile.imwrite(str(int_path), intensity)

    expected = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=intensity, intensity_channel=0,
        min_intensity=10, max_intensity=20, perimeter_fraction=0.0))
    np.testing.assert_array_equal(expected, (m == 2).astype(np.uint16))

    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=0,
                         do_perimeter_merge=False, perimeter_fraction=0.0,
                         min_intensity=10, max_intensity=20)
    _process_single_fov(*args, **kw)

    assert np.array_equal(tifffile.imread(str(mask_path)), expected)


def test_process_single_fov_channel_last_intensity(tmp_path):
    from spacr.utils import _process_single_fov, _process_single_fov_in_memory
    m = _two_touching_blocks()
    raw = np.zeros((16, 16, 3), dtype=np.uint32)
    exact = 2**24 + 1
    raw[m == 1, 1] = exact
    raw[m == 2, 1] = exact + 2

    mask_path = tmp_path / "mask.tif"
    int_path = tmp_path / "intensity.tif"
    tifffile.imwrite(str(mask_path), m)
    tifffile.imwrite(str(int_path), raw)

    expected = _process_single_fov_in_memory(**_in_memory_kwargs(
        mask=m, intensity_img=raw, intensity_channel=1,
        min_intensity=exact, max_intensity=exact, perimeter_fraction=0.0))
    np.testing.assert_array_equal(expected, (m == 1).astype(np.uint16))

    args, kw = _fov_args(str(mask_path), str(int_path), intensity_channel=1,
                         do_perimeter_merge=False, perimeter_fraction=0.0,
                         min_intensity=exact, max_intensity=exact)
    _process_single_fov(*args, **kw)

    assert np.array_equal(tifffile.imread(str(mask_path)), expected)
    np.testing.assert_array_equal(tifffile.imread(str(int_path)), raw)


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
    raw = np.zeros((16, 16, 3), dtype=np.float32)
    raw[m == 1, 2] = 3
    raw[m == 2, 2] = 9
    tifffile.imwrite(str(int_dir / "A01_f01.tif"), raw)

    merge_split_objects(str(mask_dir), intensity_img_src=str(int_dir),
                        intensity_channel=2, perimeter_fraction=0.0,
                        min_intensity=6, max_intensity=10,
                        n_jobs=1)

    out = tifffile.imread(str(mask_dir / "A01_f01.tif"))
    np.testing.assert_array_equal(out, (m == 2).astype(np.uint16))
    assert int(np.sum(out > 0)) == 60


def test_merge_split_objects_area_filter(tmp_path):
    from spacr.utils import merge_split_objects
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    m = _big_square_plus_speck()
    tifffile.imwrite(str(mask_dir / "d.tif"), m)

    merge_split_objects(str(mask_dir), perimeter_fraction=0.0,
                        min_area=20,
                        n_jobs=1)

    out = tifffile.imread(str(mask_dir / "d.tif"))
    np.testing.assert_array_equal(out, (m == 1).astype(np.uint16))
    # every surviving object clears the 20 px minimum area
    ids, counts = np.unique(out[out > 0], return_counts=True)
    assert bool(np.all(counts >= 20))
