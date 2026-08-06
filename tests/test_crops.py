"""Tests for :mod:`spacr.crops` -- on-demand single-object crops from ``merged/*.npy``.

The load-bearing test in this file is
:func:`test_merged_crop_is_pixel_identical_to_png_path`: a crop cut from the
merged array has to be byte-for-byte what ``spacr.measure._measure_crop_core``
would have written to the PNG folder. If those two ever diverge, an annotation
made on a PNG stops being comparable with a model trained on an on-demand crop,
which is the whole point of having an alternative source.
"""

import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from spacr import crops
from spacr.crops import (
    CorruptMergedFile,
    CropError,
    CropSpec,
    LabelMissing,
    MaskPlaneMissing,
    MergedCropSource,
    MergedFileMissing,
    MergedField,
    PngCropSource,
    extract_crop,
    extract_crops,
    open_merged_field,
    png_view,
    resolve_crop_source,
)

CELL_DIM, NUC_DIM, PATH_DIM = 4, 5, 6
MASK_DIMS = {"cell": CELL_DIM, "nucleus": NUC_DIM, "pathogen": PATH_DIM}


# ---------------------------------------------------------------------------
# Synthetic fields
# ---------------------------------------------------------------------------

def _make_field(h=96, w=112, n_channels=4, seed=0, objects=None, edge=False):
    """Return a merged array: ``n_channels`` intensity planes + cell/nucleus/pathogen.

    ``objects`` is a list of ``(label, y0, y1, x0, x1)`` rectangles written into
    the cell plane; nucleus/pathogen get a smaller concentric rectangle each.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 4000, size=(h, w, n_channels + 3)).astype(np.uint16)
    for d in (CELL_DIM, NUC_DIM, PATH_DIM):
        data[:, :, d] = 0
    if objects is None:
        objects = [(1, 10, 30, 12, 34), (2, 50, 78, 60, 92), (3, 40, 52, 10, 26)]
    if edge:
        objects = list(objects) + [(9, 0, 8, 0, 9)]
    for label, y0, y1, x0, x1 in objects:
        data[y0:y1, x0:x1, CELL_DIM] = label
        data[y0 + 2:y1 - 2, x0 + 2:x1 - 2, NUC_DIM] = label
        data[y0 + 3:y0 + 6, x0 + 3:x0 + 6, PATH_DIM] = label
    return data


def _write_field(tmp_path, data, name="plate1_A01_1.npy"):
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    path = merged / name
    np.save(path, data)
    return str(path)


def _reference_png_crop(data, object_type, label, *, png_dims=(0, 1, 2),
                        width=48, height=48, normalize=False,
                        normalize_by="png", use_bounding_box=False,
                        dilate=False, dilate_ratio=0.2, mask_dims=None):
    """Reproduce ``_measure_crop_core``'s PNG branch using the real spacr helpers.

    This is a transcription of ``spacr/measure.py`` lines 1124-1174 -- the
    reference the merged source has to match.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    from spacr.utils import (_crop_center, _find_bounding_box, _get_percentiles,
                             normalize_to_dtype)

    mask_dims = mask_dims or MASK_DIMS
    data_type = data.dtype
    crop_mask = data[:, :, mask_dims[object_type]].astype(data_type)

    region = (crop_mask == label)
    if use_bounding_box:
        region = _find_bounding_box(crop_mask, label, buffer=10)
    if dilate:
        # count_nonzero, not sum: _find_bounding_box fills the rectangle with
        # the LABEL VALUE, so np.sum scaled the "area" by the label id and
        # object 100 dilated ten times further than object 1.
        region_area = np.count_nonzero(region)
        approximate_diameter = np.sqrt(region_area)
        dialate_png_px = int(approximate_diameter * dilate_ratio)
        # A radius of 0 means no dilation. scipy reads iterations=0 as "repeat
        # until nothing changes", which used to grow the region to the whole
        # field.
        if dialate_png_px > 0:
            struct = generate_binary_structure(2, 2)
            region = binary_dilation(region, structure=struct,
                                     iterations=dialate_png_px)

    png_channels = data[:, :, list(png_dims)].astype(data_type)
    percentile_list = None
    if normalize_by == "fov" and normalize is not False:
        percentile_list = _get_percentiles(png_channels, normalize[0], normalize[1])

    png_channels = _crop_center(png_channels, region, new_width=width,
                                new_height=height)
    if isinstance(normalize, list):
        if normalize_by == "png":
            png_channels = normalize_to_dtype(png_channels, normalize[0], normalize[1])
        if normalize_by == "fov":
            png_channels = normalize_to_dtype(png_channels, normalize[0], normalize[1],
                                              percentile_list=percentile_list)
    else:
        png_channels = normalize_to_dtype(png_channels, 0, 100)

    if png_channels.shape[2] == 2:
        dummy_channel = np.zeros_like(png_channels[:, :, 0])
        png_channels = np.dstack((png_channels, dummy_channel))
    return png_channels


# ---------------------------------------------------------------------------
# THE fidelity test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", ["cell", "nucleus", "pathogen"])
@pytest.mark.parametrize("normalize,normalize_by", [
    (False, "png"),
    ([2, 98], "png"),
    ([2, 98], "fov"),
])
def test_merged_crop_is_pixel_identical_to_png_path(tmp_path, object_type,
                                                    normalize, normalize_by):
    """A crop cut from merged/*.npy == the crop the PNG path would have written."""
    data = _make_field(seed=3)
    path = _write_field(tmp_path, data)

    for label in (1, 2, 3):
        expected = _reference_png_crop(
            data, object_type, label, png_dims=(0, 1, 2), width=48, height=48,
            normalize=normalize, normalize_by=normalize_by)
        got = extract_crop(path, object_type, label, channels=(0, 1, 2),
                           size=(48, 48), mask_dims=MASK_DIMS,
                           normalize=normalize, normalize_by=normalize_by)
        assert got.dtype == expected.dtype == np.uint16
        assert got.shape == expected.shape == (48, 48, 3)
        assert np.array_equal(got, expected), (
            f"{object_type} label {label} differs from the PNG path")


def test_merged_crop_matches_png_written_and_read_back(tmp_path):
    """png_view() of an on-demand crop == what a consumer reads off the PNG folder.

    UPDATED, DELIBERATELY, to the corrected contract. This test used to assert
    the bug: it wrote the crop with a bare ``cv2.imwrite`` (which reads a
    3-channel array as BGR), read it back with ``PIL.Image.open(...)`` (which
    reads RGB) and pinned ``png_view`` to the *reversed* result. That made
    ``png_dims[0]`` the blue channel of every crop spaCR ever wrote.

    The writer now hands cv2 the reversed array (``crops.to_cv2_bgr``) so the
    file's red channel is ``png_dims[0]``, and the reader
    (``crops.read_crop_png``) knows which format it is looking at. Both paths
    now land on ``png_view``: channel i in, channel i out.
    """
    cv2 = pytest.importorskip("cv2")

    data = _make_field(seed=11)
    path = _write_field(tmp_path, data)
    folder = tmp_path / "cell_png"
    folder.mkdir()
    crops.stamp_crop_folder(str(folder))

    for label in (1, 2, 3):
        expected = _reference_png_crop(data, "cell", label, png_dims=(0, 1, 2),
                                       width=64, height=64)
        png_file = str(folder / f"ref_{label}.png")
        cv2.imwrite(png_file, crops.to_cv2_bgr(expected))       # the new writer
        from_disk = crops.read_crop_png(png_file)

        crop = extract_crop(path, "cell", label, channels=(0, 1, 2),
                            size=(64, 64), mask_dims=MASK_DIMS)
        assert np.array_equal(png_view(crop), from_disk)
        # ...and the file really does carry png_dims[0] in its red slot.
        assert np.array_equal(from_disk[:, :, 0], expected[:, :, 0] // 256)

        src = MergedCropSource(
            spec=CropSpec(merged_path="", channels=(0, 1, 2), size=(64, 64),
                          mask_dims=MASK_DIMS))
        row = {"path_name": path, "object_label": label}
        assert np.array_equal(src.get(row), from_disk)


def test_png_source_and_merged_source_agree_on_the_same_object(tmp_path):
    """PngCropSource and MergedCropSource return the same pixels for one object."""
    cv2 = pytest.importorskip("cv2")
    data = _make_field(seed=5)
    path = _write_field(tmp_path, data)

    png_dir = tmp_path / "data" / "plate1_A01" / "cell_png"
    png_dir.mkdir(parents=True)
    png_file = png_dir / "plate1_A01_1_2.png"
    # Written the way the current writer writes: the crop is already in
    # colour order, `to_cv2_bgr` reverses it once for cv2's BGR reading, and
    # the folder is stamped so the reader does not have to guess. Writing it
    # with a bare cv2.imwrite would produce a legacy-layout file inside an
    # unmarked folder, which is a state spaCR no longer creates.
    crops.stamp_crop_folder(str(png_dir))
    cv2.imwrite(str(png_file),
                crops.to_cv2_bgr(_reference_png_crop(data, "cell", 2,
                                                     width=48, height=48)))

    png_src = PngCropSource()
    merged_src = MergedCropSource(
        spec=CropSpec(merged_path="", channels=(0, 1, 2), size=(48, 48),
                      mask_dims=MASK_DIMS))
    assert np.array_equal(
        png_src.get({"png_path": str(png_file)}),
        merged_src.get({"path_name": path, "object_label": 2}))


# ---------------------------------------------------------------------------
# Region / geometry
# ---------------------------------------------------------------------------

def test_bbox_from_mask_plane_matches_bbox_stored_in_db(tmp_path):
    """The derived bounding box equals a regionprops bbox recorded in the db."""
    from skimage.measure import regionprops_table

    data = _make_field(seed=7)
    path = _write_field(tmp_path, data)
    props = regionprops_table(data[:, :, CELL_DIM].astype(np.int32),
                              properties=("label", "bbox"))

    db_path = str(tmp_path / "measurements.db")
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE cell (object_label INTEGER, path_name TEXT, '
                 '"bbox-0" INTEGER, "bbox-1" INTEGER, "bbox-2" INTEGER, "bbox-3" INTEGER)')
    for i, label in enumerate(props["label"]):
        conn.execute('INSERT INTO cell VALUES (?,?,?,?,?,?)',
                     (int(label), path, int(props["bbox-0"][i]),
                      int(props["bbox-1"][i]), int(props["bbox-2"][i]),
                      int(props["bbox-3"][i])))
    conn.commit()
    conn.close()

    fld = open_merged_field(path, MASK_DIMS)
    idx = fld.label_index("cell")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM cell").fetchall()
    conn.close()

    src = MergedCropSource(spec=CropSpec(merged_path="", channels=(0, 1, 2),
                                         size=(48, 48), mask_dims=MASK_DIMS))
    for row in rows:
        label = int(row["object_label"])
        # (y0, y1, x0, x1) half-open, i.e. skimage's (bbox-0, bbox-2, bbox-1, bbox-3).
        assert idx.bbox(label) == (row["bbox-0"], row["bbox-2"],
                                   row["bbox-1"], row["bbox-3"])
        # A db-supplied bbox must give the same crop as a derived one.
        spec = src.spec_for(dict(row))
        assert spec.bbox == (row["bbox-0"], row["bbox-2"],
                             row["bbox-1"], row["bbox-3"])
        assert np.array_equal(
            extract_crop(path, spec=spec),
            extract_crop(path, "cell", label, channels=(0, 1, 2),
                         size=(48, 48), mask_dims=MASK_DIMS))


def test_object_at_the_array_edge_is_padded_like_the_png_path(tmp_path):
    """An object touching (0, 0) gets the same zero padding as _crop_center gives."""
    data = _make_field(seed=13, edge=True)
    path = _write_field(tmp_path, data)

    expected = _reference_png_crop(data, "cell", 9, width=64, height=64)
    got = extract_crop(path, "cell", 9, channels=(0, 1, 2), size=(64, 64),
                       mask_dims=MASK_DIMS)
    assert np.array_equal(got, expected)
    # The object sits in the top-left corner, so the top rows and left columns
    # of the crop are pure padding.
    assert got.shape == (64, 64, 3)
    assert not got[:20, :, :].any()
    assert not got[:, :20, :].any()


def test_non_square_size_uses_width_first(tmp_path):
    """size=(width, height) matches the PNG path's width/height argument order."""
    data = _make_field(seed=17)
    path = _write_field(tmp_path, data)
    expected = _reference_png_crop(data, "cell", 1, width=40, height=72)
    got = extract_crop(path, "cell", 1, channels=(0, 1, 2), size=(40, 72),
                       mask_dims=MASK_DIMS)
    assert got.shape == (72, 40, 3)
    assert np.array_equal(got, expected)


def test_use_bounding_box_matches_png_path(tmp_path):
    """use_bounding_box reproduces _find_bounding_box's padded rectangle."""
    data = _make_field(seed=19)
    path = _write_field(tmp_path, data)
    for label in (1, 2, 3):
        expected = _reference_png_crop(data, "cell", label, width=64, height=64,
                                       use_bounding_box=True)
        got = extract_crop(path, "cell", label, channels=(0, 1, 2), size=(64, 64),
                           mask_dims=MASK_DIMS, use_bounding_box=True)
        assert np.array_equal(got, expected)


def test_dilation_matches_png_path(tmp_path):
    """dialate_pngs reproduces scipy's 8-connected binary_dilation."""
    data = _make_field(seed=23)
    path = _write_field(tmp_path, data)
    for label in (1, 2, 3):
        expected = _reference_png_crop(data, "cell", label, width=64, height=64,
                                       dilate=True, dilate_ratio=0.3)
        got = extract_crop(path, "cell", label, channels=(0, 1, 2), size=(64, 64),
                           mask_dims=MASK_DIMS, dilate=True, dilate_ratio=0.3)
        assert np.array_equal(got, expected)


def test_a_dilation_radius_that_rounds_to_zero_means_no_dilation(tmp_path):
    """A radius of 0 must leave the object alone, not swallow the field.

    scipy reads ``iterations=0`` as "dilate until nothing changes", so both the
    PNG path and this one used to turn every object under about 25 px -- which
    on a Toxoplasma screen means the parasites -- into an unmasked window
    centred on the middle of the field. It looked like a real crop. Both
    sources guard the radius now, so a tiny object crops exactly as it would
    with dilation switched off.
    """
    data = _make_field(seed=29, objects=[(1, 40, 43, 40, 43)])
    path = _write_field(tmp_path, data)
    expected = _reference_png_crop(data, "cell", 1, width=32, height=32,
                                   dilate=True, dilate_ratio=0.2)
    got = extract_crop(path, "cell", 1, channels=(0, 1, 2), size=(32, 32),
                       mask_dims=MASK_DIMS, dilate=True, dilate_ratio=0.2)
    assert np.array_equal(got, expected)
    # The object is isolated, not a window on the background.
    assert (got > 0).mean() < 0.05
    masked = extract_crop(path, "cell", 1, channels=(0, 1, 2), size=(32, 32),
                          mask_dims=MASK_DIMS, dilate=False)
    assert np.array_equal(got, masked)


def test_cytoplasm_plane_is_derived_from_cell_minus_children(tmp_path):
    """Cytoplasm has no plane on disk; it is cell minus nucleus/pathogen/organelle."""
    data = _make_field(seed=31)
    path = _write_field(tmp_path, data)
    fld = open_merged_field(path, MASK_DIMS)

    interior = (data[:, :, NUC_DIM] != 0) | (data[:, :, PATH_DIM] != 0)
    expected_plane = np.where(interior, 0, data[:, :, CELL_DIM])
    assert np.array_equal(fld.mask_plane("cytoplasm"), expected_plane)

    with pytest.raises(MaskPlaneMissing):
        fld.mask_dim("cytoplasm")

    ref_data = data.copy()
    ref_data[:, :, CELL_DIM] = expected_plane           # crop by the derived plane
    expected = _reference_png_crop(ref_data, "cell", 2, width=48, height=48)
    got = extract_crop(path, "cytoplasm", 2, channels=(0, 1, 2), size=(48, 48),
                       mask_dims=MASK_DIMS)
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [(0,), (0, 1), (0, 1, 2), (3, 1, 0), (2, 2, 2)])
def test_channel_selection_and_ordering(tmp_path, channels):
    """Channels are selected and emitted in the requested order, like png_dims."""
    data = _make_field(seed=37)
    path = _write_field(tmp_path, data)
    expected = _reference_png_crop(data, "cell", 1, png_dims=channels,
                                   width=48, height=48)
    got = extract_crop(path, "cell", 1, channels=channels, size=(48, 48),
                       mask_dims=MASK_DIMS)
    assert np.array_equal(got, expected)
    # Two channels are padded to three, exactly like the PNG path does.
    assert got.shape[2] == (3 if len(channels) == 2 else len(channels))


def test_reversed_channels_actually_reverse_the_pixels(tmp_path):
    """A reversed channel request is not silently ignored."""
    data = _make_field(seed=41)
    path = _write_field(tmp_path, data)
    forward = extract_crop(path, "cell", 2, channels=(0, 1, 2), size=(48, 48),
                           mask_dims=MASK_DIMS)
    reverse = extract_crop(path, "cell", 2, channels=(2, 1, 0), size=(48, 48),
                           mask_dims=MASK_DIMS)
    assert np.array_equal(forward[:, :, ::-1], reverse)
    assert not np.array_equal(forward, reverse)


def test_out_of_range_channel_is_an_error(tmp_path):
    data = _make_field(seed=43)
    path = _write_field(tmp_path, data)
    with pytest.raises(CropError, match="channel 99"):
        extract_crop(path, "cell", 1, channels=(0, 99), size=(16, 16),
                     mask_dims=MASK_DIMS)


# ---------------------------------------------------------------------------
# Memory-map bound + label-index cache
# ---------------------------------------------------------------------------

class _CountingArray:
    """Wraps an array and records how many elements each read materialises.

    :class:`MergedField` is only allowed to touch ``shape`` / ``dtype`` /
    ``ndim`` / ``__getitem__``, which is exactly what a ``np.memmap`` needs to
    stay lazy -- so this proxy both counts reads and pins that contract.
    """

    def __init__(self, arr):
        self._a = arr
        self.read_elems = 0
        self.reads = 0

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def ndim(self):
        return self._a.ndim

    def __getitem__(self, key):
        out = self._a[key]
        self.read_elems += int(np.size(out))
        self.reads += 1
        return out


@pytest.fixture(scope="module")
def big_field(tmp_path_factory):
    """A 2048x2048x5 uint16 field (40 MB) with one 64x64 object."""
    d = tmp_path_factory.mktemp("bigfield")
    h = w = 2048
    data = np.zeros((h, w, 5), dtype=np.uint16)
    data[:, :, :4] = 1000
    data[900:964, 1000:1064, 4] = 7          # one 64x64 object in the mask plane
    data[1500:1560, 300:360, 4] = 8
    path = str(d / "plate1_A01_1.npy")
    np.save(path, data)
    return path


def test_open_merged_field_memory_maps_the_npy(big_field):
    fld = open_merged_field(big_field, {"cell": 4}, use_cache=False)
    assert isinstance(fld.array, np.memmap)
    assert fld.array.mode == "r"


def test_one_object_does_not_materialise_the_whole_field(big_field):
    """Cutting one 64x64 object touches the mask plane + the window, not 40 MB."""
    raw = np.load(big_field, mmap_mode="r")
    counting = _CountingArray(raw)
    fld = MergedField(big_field, array=counting, mask_dims={"cell": 4})

    crop = extract_crop(big_field, "cell", 7, channels=(0, 1, 2), size=(128, 128),
                        mask_dims={"cell": 4}, field=fld)
    assert crop.shape == (128, 128, 3)

    total = 2048 * 2048 * 5
    one_plane = 2048 * 2048
    # One mask plane (to find the object) + the mask window + 3 channel windows.
    assert counting.read_elems <= one_plane + 8 * 128 * 128
    assert counting.read_elems < total // 4


def test_a_db_supplied_bbox_avoids_scanning_the_mask_plane(big_field):
    """With a bbox in hand the read is bounded by the window alone."""
    raw = np.load(big_field, mmap_mode="r")
    counting = _CountingArray(raw)
    fld = MergedField(big_field, array=counting, mask_dims={"cell": 4})

    spec = CropSpec(merged_path=big_field, object_type="cell", label=7,
                    channels=(0, 1, 2), size=(128, 128), mask_dims={"cell": 4},
                    bbox=(900, 964, 1000, 1064))
    crop = extract_crop(big_field, spec=spec, field=fld)
    assert crop.shape == (128, 128, 3)
    assert counting.read_elems <= 8 * 128 * 128
    assert counting.read_elems < 2048 * 2048

    # ... and it is the same crop the mask-plane scan produces.
    plain = extract_crop(big_field, "cell", 7, channels=(0, 1, 2),
                         size=(128, 128), mask_dims={"cell": 4})
    assert np.array_equal(crop, plain)


def test_label_index_is_computed_once_for_many_objects(tmp_path):
    """N objects from one field scan the mask plane once, not N times."""
    data = _make_field(seed=47, objects=[(i, 4 * i, 4 * i + 3, 4 * i, 4 * i + 3)
                                         for i in range(1, 21)])
    path = _write_field(tmp_path, data)
    counting = _CountingArray(np.load(path, mmap_mode="r"))
    fld = MergedField(path, array=counting, mask_dims=MASK_DIMS)

    calls = {"n": 0}
    real_init = crops._LabelIndex.__init__

    def counting_init(self, mask):
        calls["n"] += 1
        real_init(self, mask)

    crops._LabelIndex.__init__ = counting_init
    try:
        for label in range(1, 21):
            extract_crop(path, "cell", label, channels=(0, 1, 2), size=(24, 24),
                         mask_dims=MASK_DIMS, field=fld)
    finally:
        crops._LabelIndex.__init__ = real_init

    assert calls["n"] == 1
    plane_elems = data.shape[0] * data.shape[1]
    # One full-plane scan for the label index, then only per-crop windows:
    # 3 channel windows of 24x24 plus a 3x3 mask window per object.
    assert counting.read_elems <= plane_elems + 20 * (4 * 24 * 24 + 3 * 3)
    assert counting.read_elems >= plane_elems
    # Twenty independent plane scans would have cost an order of magnitude more.
    assert counting.read_elems < 20 * plane_elems // 4


def test_extract_crops_opens_the_file_once(tmp_path, monkeypatch):
    """Batched access memory-maps the .npy a single time for the whole batch."""
    data = _make_field(seed=53)
    path = _write_field(tmp_path, data)
    crops.clear_field_cache()

    opens = {"n": 0}
    real_load = np.load

    def counting_load(*args, **kwargs):
        if kwargs.get("mmap_mode"):
            opens["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(crops.np, "load", counting_load)

    specs = [CropSpec(merged_path=path, object_type="cell", label=lbl,
                      channels=(0, 1, 2), size=(48, 48), mask_dims=MASK_DIMS)
             for lbl in (1, 2, 3)]
    out = extract_crops(path, specs)
    assert opens["n"] == 1
    assert len(out) == 3
    for spec, got in zip(specs, out):
        assert np.array_equal(got, _reference_png_crop(
            data, "cell", spec.label, width=48, height=48))


def test_field_cache_is_keyed_on_file_contents(tmp_path):
    """A regenerated merged file is never served from the cache."""
    data = _make_field(seed=59)
    path = _write_field(tmp_path, data)
    crops.clear_field_cache()
    first = open_merged_field(path, MASK_DIMS)
    assert open_merged_field(path, MASK_DIMS) is first

    data2 = _make_field(seed=61)
    os.utime(path, (0, 0))
    np.save(path, data2)
    assert open_merged_field(path, MASK_DIMS) is not first


def test_get_many_groups_rows_by_file(tmp_path):
    """MergedCropSource.get_many cuts every object of a field from one open file."""
    data_a = _make_field(seed=67)
    data_b = _make_field(seed=71)
    path_a = _write_field(tmp_path, data_a, "plate1_A01_1.npy")
    path_b = _write_field(tmp_path, data_b, "plate1_A01_2.npy")
    crops.clear_field_cache()

    src = MergedCropSource(spec=CropSpec(merged_path="", channels=(0, 1, 2),
                                         size=(48, 48), mask_dims=MASK_DIMS))
    rows = [{"path_name": path_a, "object_label": 1},
            {"path_name": path_b, "object_label": 2},
            {"path_name": path_a, "object_label": 3}]
    out = src.get_many(rows)
    assert len(out) == 3
    assert np.array_equal(out[0], png_view(_reference_png_crop(
        data_a, "cell", 1, width=48, height=48)))
    assert np.array_equal(out[1], png_view(_reference_png_crop(
        data_b, "cell", 2, width=48, height=48)))
    assert np.array_equal(out[2], png_view(_reference_png_crop(
        data_a, "cell", 3, width=48, height=48)))


# ---------------------------------------------------------------------------
# Failure modes -- every one of these must be loud
# ---------------------------------------------------------------------------

def test_missing_merged_file_raises(tmp_path):
    with pytest.raises(MergedFileMissing, match="merged array not found"):
        extract_crop(str(tmp_path / "nope.npy"), "cell", 1, mask_dims=MASK_DIMS)


def test_corrupt_npy_raises(tmp_path):
    bad = tmp_path / "truncated.npy"
    good = _write_field(tmp_path, _make_field(seed=73), "ok.npy")
    raw = open(good, "rb").read()
    bad.write_bytes(raw[:len(raw) // 3])
    with pytest.raises(CorruptMergedFile):
        extract_crop(str(bad), "cell", 1, mask_dims=MASK_DIMS)


def test_two_dimensional_npy_raises(tmp_path):
    flat = tmp_path / "flat.npy"
    np.save(flat, np.zeros((10, 10), dtype=np.uint16))
    with pytest.raises(CorruptMergedFile, match=r"\(H, W, C\)"):
        extract_crop(str(flat), "cell", 1, mask_dims=MASK_DIMS)


def test_missing_label_raises(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=79))
    with pytest.raises(LabelMissing, match="not present"):
        extract_crop(path, "cell", 999, channels=(0, 1, 2), size=(32, 32),
                     mask_dims=MASK_DIMS)


def test_background_label_zero_raises(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=83))
    with pytest.raises(LabelMissing, match="background"):
        extract_crop(path, "cell", 0, channels=(0, 1, 2), size=(32, 32),
                     mask_dims=MASK_DIMS)
    with pytest.raises(LabelMissing, match="positive integer"):
        extract_crop(path, "cell", -4, channels=(0, 1, 2), size=(32, 32),
                     mask_dims=MASK_DIMS)


def test_unknown_object_type_raises(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=89))
    with pytest.raises(CropError, match="unknown object_type"):
        extract_crop(path, "mitochondrion", 1, mask_dims=MASK_DIMS)


def test_object_type_without_a_plane_raises(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=97))
    with pytest.raises(MaskPlaneMissing, match="no mask plane recorded"):
        extract_crop(path, "organelle", 1, channels=(0, 1, 2), size=(32, 32),
                     mask_dims={"cell": CELL_DIM})


def test_mask_plane_out_of_range_raises(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=101))
    with pytest.raises(MaskPlaneMissing, match="out of range"):
        extract_crop(path, "cell", 1, channels=(0, 1), size=(32, 32),
                     mask_dims={"cell": 99})


def test_bbox_outside_the_field_raises(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=103))
    spec = CropSpec(merged_path=path, object_type="cell", label=1,
                    channels=(0, 1, 2), size=(32, 32), mask_dims=MASK_DIMS,
                    bbox=(0, 5000, 0, 10))
    with pytest.raises(CropError, match="runs outside"):
        extract_crop(path, spec=spec)


def test_bad_size_and_empty_channels_raise(tmp_path):
    path = _write_field(tmp_path, _make_field(seed=107))
    with pytest.raises(CropError, match="size must be positive"):
        extract_crop(path, "cell", 1, channels=(0,), size=(0, 10),
                     mask_dims=MASK_DIMS)
    with pytest.raises(CropError, match="channels is empty"):
        extract_crop(path, "cell", 1, channels=(), size=(10, 10),
                     mask_dims=MASK_DIMS)


def test_bad_normalize_by_raises():
    with pytest.raises(CropError, match="normalize_by"):
        CropSpec(merged_path="x", normalize_by="sideways")


def test_extract_crops_on_error_none_returns_sentinels(tmp_path):
    data = _make_field(seed=109)
    path = _write_field(tmp_path, data)
    specs = [CropSpec(merged_path=path, object_type="cell", label=lbl,
                      channels=(0, 1, 2), size=(32, 32), mask_dims=MASK_DIMS)
             for lbl in (1, 4242, 2)]
    out = extract_crops(path, specs, on_error="none")
    assert out[1] is None
    assert out[0] is not None and out[2] is not None
    assert extract_crops(str(tmp_path / "gone.npy"), specs, on_error="none") == [None] * 3
    with pytest.raises(ValueError, match="on_error"):
        extract_crops(path, specs, on_error="shrug")
    assert extract_crops(path, []) == []


def test_png_source_missing_file_raises(tmp_path):
    with pytest.raises(MergedFileMissing, match="crop PNG not found"):
        PngCropSource().get({"png_path": str(tmp_path / "missing.png")})
    with pytest.raises(CropError, match="no 'png_path'"):
        PngCropSource().get({})


def test_merged_source_row_without_identity_raises(tmp_path):
    src = MergedCropSource(spec=CropSpec(merged_path="", mask_dims=MASK_DIMS))
    with pytest.raises(CropError, match="object_label"):
        src.get({"path_name": "/x/y.npy"})
    with pytest.raises(CropError, match="path_name"):
        src.get({"object_label": 1})
    src2 = MergedCropSource(spec=CropSpec(merged_path="", mask_dims=MASK_DIMS),
                            merged_root=str(tmp_path))
    with pytest.raises(CropError, match="not enough metadata"):
        src2.spec_for({"object_label": 1})


# ---------------------------------------------------------------------------
# Row plumbing
# ---------------------------------------------------------------------------

def test_merged_source_rebuilds_the_path_from_well_metadata(tmp_path):
    """A row with only plate/row/column/field still finds its merged array."""
    data = _make_field(seed=113)
    path = _write_field(tmp_path, data, "plate1_A01_1.npy")
    src = MergedCropSource(spec=CropSpec(merged_path="", channels=(0, 1, 2),
                                         size=(48, 48), mask_dims=MASK_DIMS),
                           merged_root=os.path.dirname(path))
    row = {"plateID": "plate1", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
           "object_label": 1}
    assert src.spec_for(row).merged_path == path
    assert np.array_equal(src.get(row),
                          png_view(_reference_png_crop(data, "cell", 1,
                                                       width=48, height=48)))


def test_merged_source_reanchors_a_path_from_another_machine(tmp_path):
    data = _make_field(seed=127)
    path = _write_field(tmp_path, data, "plate1_A01_1.npy")
    src = MergedCropSource(spec=CropSpec(merged_path="", mask_dims=MASK_DIMS),
                           merged_root=os.path.dirname(path))
    stale = "/somewhere/else/merged/plate1_A01_1.npy"
    assert src.spec_for({"path_name": stale, "object_label": 1}).merged_path == path


def test_merged_source_accepts_a_pandas_row(tmp_path):
    pd = pytest.importorskip("pandas")
    data = _make_field(seed=131)
    path = _write_field(tmp_path, data)
    df = pd.DataFrame([{"path_name": path, "object_label": 2}])
    src = MergedCropSource(spec=CropSpec(merged_path="", channels=(0, 1, 2),
                                         size=(48, 48), mask_dims=MASK_DIMS))
    got = src.get(df.iloc[0])
    assert np.array_equal(got, png_view(_reference_png_crop(
        data, "cell", 2, width=48, height=48)))


def test_png_source_reanchors_paths_like_correct_paths(tmp_path):
    cv2 = pytest.importorskip("cv2")
    png_dir = tmp_path / "data" / "plate1_A01" / "cell_png"
    png_dir.mkdir(parents=True)
    png_file = png_dir / "a.png"
    cv2.imwrite(str(png_file), np.zeros((4, 4, 3), np.uint8))
    src = PngCropSource(root=str(tmp_path))
    stale = "/other/machine/data/plate1_A01/cell_png/a.png"
    assert src.resolve(stale) == str(png_file)
    assert src.get(stale).shape == (4, 4, 3)


def test_crop_source_get_image_returns_pil(tmp_path):
    data = _make_field(seed=137)
    path = _write_field(tmp_path, data)
    src = MergedCropSource(spec=CropSpec(merged_path="", channels=(0, 1, 2),
                                         size=(32, 32), mask_dims=MASK_DIMS))
    img = src.get_image({"path_name": path, "object_label": 1})
    assert img.mode == "RGB"
    assert img.size == (32, 32)


def test_abstract_crop_source_get_is_not_implemented():
    with pytest.raises(NotImplementedError):
        crops.CropSource().get({})


# ---------------------------------------------------------------------------
# resolve_crop_source
# ---------------------------------------------------------------------------

def _experiment(tmp_path, with_png=True, with_merged=True, settings=None):
    root = tmp_path / "exp"
    (root / "measurements").mkdir(parents=True)
    if with_merged:
        (root / "merged").mkdir()
        np.save(root / "merged" / "plate1_A01_1.npy", _make_field(seed=139))
    if with_png:
        (root / "data" / "plate1_A01" / "cell_png").mkdir(parents=True)
    if settings is not None:
        conn = sqlite3.connect(str(root / "measurements" / "measurements.db"))
        conn.execute("CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")
        conn.executemany("INSERT INTO settings VALUES (?,?)",
                         [(k, str(v)) for k, v in settings.items()])
        conn.commit()
        conn.close()
    return str(root)


def test_resolve_picks_png_when_a_png_folder_exists(tmp_path):
    root = _experiment(tmp_path)
    src = resolve_crop_source(root)
    assert src.kind == "png"
    assert "PNG" in src.reason
    assert "png" in src.describe()


def test_resolve_picks_merged_when_there_is_no_png_folder(tmp_path):
    root = _experiment(tmp_path, with_png=False)
    src = resolve_crop_source(root)
    assert src.kind == "merged"
    assert "merged" in src.reason


def test_resolve_honours_an_explicit_request(tmp_path):
    root = _experiment(tmp_path)
    assert resolve_crop_source(root, prefer="merged").kind == "merged"
    assert resolve_crop_source({"src": root, "crop_source": "merged"}).kind == "merged"
    assert resolve_crop_source({"src": [root], "crop_source": "png"}).kind == "png"
    assert resolve_crop_source(root, prefer="merged").reason.startswith("requested")


def test_resolve_accepts_the_merged_folder_as_src(tmp_path):
    root = _experiment(tmp_path, with_png=False)
    src = resolve_crop_source(os.path.join(root, "merged"))
    assert src.kind == "merged"
    assert src.merged_root == os.path.join(root, "merged")


def test_resolve_recovers_crop_settings_from_the_database(tmp_path):
    """The merged source reuses the very settings that shaped the PNG folder."""
    root = _experiment(tmp_path, with_png=False, settings={
        "png_dims": [1, 2, 3], "png_size": [64, 64], "normalize": [2, 98],
        "normalize_by": "fov", "crop_mode": ["nucleus"], "use_bounding_box": True,
        "cell_mask_dim": CELL_DIM, "nucleus_mask_dim": NUC_DIM,
        "pathogen_mask_dim": PATH_DIM, "organelle_mask_dim": None,
        "dialate_pngs": False, "dialate_png_ratios": [0.2],
    })
    src = resolve_crop_source(root)
    assert src.kind == "merged"
    assert "measurements.db" in src.reason
    spec = src.spec
    # COLOUR order, so png_dims=[1,2,3] -- entry 0 is blue -- becomes
    # (red, green, blue) = (3, 2, 1). See crops.channels_from_settings.
    assert spec.channels == (3, 2, 1)
    assert spec.size == (64, 64)
    assert spec.normalize == [2, 98]
    assert spec.normalize_by == "fov"
    assert spec.object_type == "nucleus"
    assert spec.use_bounding_box is True
    assert spec.mask_dims == {"cell": CELL_DIM, "nucleus": NUC_DIM,
                              "pathogen": PATH_DIM}


def test_resolve_lets_explicit_settings_beat_the_saved_snapshot(tmp_path):
    root = _experiment(tmp_path, with_png=False, settings={"png_size": [64, 64]})
    src = resolve_crop_source({"src": root, "crop_source": "merged",
                               "png_size": [16, 16]})
    assert src.spec.size == (16, 16)


def test_resolve_errors_when_nothing_is_available(tmp_path):
    root = _experiment(tmp_path, with_png=False, with_merged=False)
    with pytest.raises(CropError, match="no crop source available"):
        resolve_crop_source(root)
    with pytest.raises(CropError, match="no 'src'"):
        resolve_crop_source({})
    with pytest.raises(CropError, match="crop_source must be"):
        resolve_crop_source(root, prefer="telepathy")


def test_crop_settings_from_db_handles_a_missing_table(tmp_path):
    root = _experiment(tmp_path)
    db = os.path.join(root, "measurements", "measurements.db")
    sqlite3.connect(db).close()
    assert crops.crop_settings_from_db(db) == {}
    with pytest.raises(MergedFileMissing):
        crops.crop_settings_from_db(os.path.join(root, "nope.db"))


def test_crop_settings_from_db_round_trips_python_values(tmp_path):
    root = _experiment(tmp_path, settings={"png_dims": [0, 1, 2], "normalize": False,
                                           "normalize_by": "png",
                                           "organelle_mask_dim": None})
    saved = crops.crop_settings_from_db(
        os.path.join(root, "measurements", "measurements.db"))
    assert saved["png_dims"] == [0, 1, 2]
    assert saved["normalize"] is False
    assert saved["normalize_by"] == "png"
    assert saved["organelle_mask_dim"] is None


def test_crop_spec_from_settings_handles_per_crop_mode_lists():
    spec = crops.crop_spec_from_settings(
        {"crop_mode": ["cell", "nucleus"], "png_size": [[64, 64], [32, 32]],
         "dialate_pngs": [True, False], "dialate_png_ratios": [0.1, 0.4]},
        object_type="nucleus")
    assert spec.size == (32, 32)
    assert spec.dilate is False
    assert spec.dilate_ratio == 0.4

    spec_cell = crops.crop_spec_from_settings(
        {"crop_mode": ["cell", "nucleus"], "png_size": [[64, 64], [32, 32]],
         "dialate_pngs": [True, False], "dialate_png_ratios": [0.1, 0.4]},
        object_type="cell")
    assert spec_cell.size == (64, 64)
    assert spec_cell.dilate is True
    assert spec_cell.dilate_ratio == 0.1


def test_crop_spec_from_settings_disables_dilation_for_cytoplasm():
    spec = crops.crop_spec_from_settings(
        {"crop_mode": "cytoplasm", "dialate_pngs": True}, object_type=None)
    assert spec.object_type == "cytoplasm"
    assert spec.dilate is False


def test_mask_dims_from_settings_falls_back_to_the_default_layout():
    assert crops.mask_dims_from_settings({}) == crops.DEFAULT_MASK_DIMS
    assert crops.mask_dims_from_settings(
        {"cell_mask_dim": 2, "nucleus_mask_dim": "None"}) == {"cell": 2}
    assert crops.mask_dims_from_settings(
        {"cell_mask_dim": "not-a-number"}) == crops.DEFAULT_MASK_DIMS


# ---------------------------------------------------------------------------
# Normalisation helpers match the originals
# ---------------------------------------------------------------------------

def test_rescale_intensity_clone_matches_skimage():
    from skimage.exposure import rescale_intensity

    rng = np.random.default_rng(5)
    for _ in range(20):
        img = rng.integers(0, 65535, size=(9, 7)).astype(np.uint16)
        lo, hi = sorted(rng.uniform(0, 65535, size=2))
        out_hi = 65535
        assert np.array_equal(
            crops._rescale_intensity(img, (lo, hi), (0, out_hi)),
            rescale_intensity(img, in_range=(lo, hi), out_range=(0, out_hi)))
    # Degenerate range: skimage clips straight to the output range.
    img = np.full((4, 4), 7, dtype=np.uint16)
    assert np.array_equal(
        crops._rescale_intensity(img, (7.0, 7.0), (0, 65535)),
        rescale_intensity(img, in_range=(7.0, 7.0), out_range=(0, 65535)))


def test_normalize_and_percentile_clones_match_spacr_utils():
    from spacr.utils import _get_percentiles, normalize_to_dtype

    rng = np.random.default_rng(9)
    for dtype in (np.uint8, np.uint16):
        top = np.iinfo(dtype).max
        arr = rng.integers(0, top, size=(11, 13, 3)).astype(dtype)
        arr[3:6, 3:6, :] = 0                     # zeros exercise the nonzero branch
        assert np.array_equal(crops._normalize_to_dtype(arr, 2, 98),
                              normalize_to_dtype(arr, 2, 98))
        assert np.array_equal(crops._normalize_to_dtype(arr, 0, 100),
                              normalize_to_dtype(arr, 0, 100))
        pl = _get_percentiles(arr, 2, 98)
        assert crops._get_percentiles(arr, 2, 98) == pl
        assert np.array_equal(
            crops._normalize_to_dtype(arr, 2, 98, percentile_list=pl),
            normalize_to_dtype(arr, 2, 98, percentile_list=pl))
    # All-zero channel: both fall back to percentiles of the raw image.
    zeros = np.zeros((5, 5, 2), dtype=np.uint16)
    assert np.array_equal(crops._normalize_to_dtype(zeros, 2, 98),
                          normalize_to_dtype(zeros, 2, 98))
    assert crops._get_percentiles(zeros, 2, 98) == _get_percentiles(zeros, 2, 98)


def test_binary_dilate_clone_matches_scipy():
    from scipy.ndimage import binary_dilation, generate_binary_structure

    rng = np.random.default_rng(13)
    struct = generate_binary_structure(2, 2)
    for _ in range(6):
        region = rng.random((17, 19)) > 0.9
        for it in (1, 2, 3):
            assert np.array_equal(
                crops._binary_dilate(region, it),
                binary_dilation(region, structure=struct, iterations=it))


def test_png_view_matches_a_real_write_and_read(tmp_path):
    """png_view == what read_crop_png gives back, for a real file round trip.

    UPDATED, DELIBERATELY. The old version wrote with a bare ``cv2.imwrite``
    and read with ``PIL...convert('RGB')`` and asserted png_view reproduced
    *that* — i.e. it pinned both halves of the bug: the reversed channel order
    and PIL's split narrowing, which CLIPS a 16-bit single-channel PNG at 255
    (case 3 below is exactly such a crop, and it used to come back solid
    white). The round trip is now writer + reader, and the narrowing is one
    rule: the high byte, for every channel count.
    """
    cv2 = pytest.importorskip("cv2")

    folder = tmp_path / "cell_png"
    folder.mkdir()
    crops.stamp_crop_folder(str(folder))

    rng = np.random.default_rng(17)
    cases = [
        rng.integers(0, 256, size=(6, 5, 3)).astype(np.uint8),
        rng.integers(0, 65536, size=(6, 5, 3)).astype(np.uint16),
        rng.integers(0, 256, size=(6, 5, 1)).astype(np.uint8),
        rng.integers(300, 40000, size=(6, 5, 1)).astype(np.uint16),
    ]
    for i, arr in enumerate(cases):
        p = str(folder / f"case{i}.png")
        cv2.imwrite(p, crops.to_cv2_bgr(arr))
        assert np.array_equal(png_view(arr), crops.read_crop_png(p)), f"case {i}"

    # The single-channel 16-bit case is the one PIL used to flatten: prove it
    # comes back as the high byte, not as 255.
    assert crops.read_crop_png(str(folder / "case3.png"))[:, :, 0].max() < 255
    assert np.array_equal(crops.read_crop_png(str(folder / "case3.png"))[:, :, 0],
                          cases[3][:, :, 0] // 256)


def test_legacy_png_view_still_reproduces_the_old_round_trip(tmp_path):
    """legacy_png_view is the inverse of the format-1 write, and stays exact.

    This is the assertion the file used to make about ``png_view``. It is kept
    under the name that says what it is, because ``read_crop_png`` has to undo
    precisely this to open an old dataset.
    """
    cv2 = pytest.importorskip("cv2")
    from PIL import Image

    rng = np.random.default_rng(17)
    cases = [
        rng.integers(0, 256, size=(6, 5, 3)).astype(np.uint8),
        rng.integers(0, 65536, size=(6, 5, 3)).astype(np.uint16),
        rng.integers(0, 256, size=(6, 5, 1)).astype(np.uint8),
        rng.integers(0, 300, size=(6, 5, 1)).astype(np.uint16),
    ]
    for i, arr in enumerate(cases):
        p = str(tmp_path / f"case{i}.png")
        cv2.imwrite(p, arr)                    # the OLD writer, verbatim
        with Image.open(p) as img:
            expected = np.array(img.convert("RGB"))
        assert np.array_equal(crops.legacy_png_view(arr), expected), f"case {i}"


def test_png_view_accepts_a_2d_array():
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = png_view(arr)
    assert out.shape == (2, 2, 3)
    assert (out[:, :, 0] == out[:, :, 2]).all()


def test_png_view_narrows_float_input():
    arr = np.full((2, 2, 3), 300.0)
    assert png_view(arr).max() == 255


# ---------------------------------------------------------------------------
# Field introspection + misc
# ---------------------------------------------------------------------------

def test_field_reports_shape_dtype_and_labels(tmp_path):
    data = _make_field(seed=149)
    path = _write_field(tmp_path, data)
    fld = open_merged_field(path, MASK_DIMS)
    assert fld.shape == data.shape
    assert fld.dtype == np.dtype(np.uint16)
    assert fld.crop_dtype == np.dtype(np.uint16)
    assert fld.labels("cell") == [1, 2, 3]
    assert fld.labels("nucleus") == [1, 2, 3]


def test_float_field_is_cropped_as_uint16_like_the_png_path(tmp_path):
    """_measure_crop_core promotes non-uint arrays to uint16 before cropping."""
    data = _make_field(seed=151).astype(np.float32)
    path = _write_field(tmp_path, data, "float.npy")
    fld = open_merged_field(path, MASK_DIMS)
    assert fld.dtype == np.dtype(np.float32)
    assert fld.crop_dtype == np.dtype(np.uint16)
    got = extract_crop(path, "cell", 1, channels=(0, 1, 2), size=(32, 32),
                       mask_dims=MASK_DIMS)
    assert got.dtype == np.uint16
    expected = _reference_png_crop(data.astype(np.uint16), "cell", 1,
                                   width=32, height=32)
    assert np.array_equal(got, expected)


def test_empty_mask_plane_has_no_labels(tmp_path):
    data = _make_field(seed=157)
    data[:, :, CELL_DIM] = 0
    path = _write_field(tmp_path, data, "empty.npy")
    fld = open_merged_field(path, MASK_DIMS)
    assert fld.labels("cell") == []
    with pytest.raises(LabelMissing):
        fld.label_index("cell").bbox(1)
    with pytest.raises(LabelMissing):
        extract_crop(path, "cell", 1, channels=(0,), size=(8, 8),
                     mask_dims=MASK_DIMS)


def test_label_index_exposes_area_and_centroid(tmp_path):
    data = _make_field(seed=163, objects=[(4, 10, 20, 30, 44)])
    path = _write_field(tmp_path, data, "one.npy")
    idx = open_merged_field(path, MASK_DIMS).label_index("cell")
    assert 4 in idx
    assert 5 not in idx
    assert idx.area(4) == 10 * 14
    assert idx.centroid(4) == pytest.approx((14.5, 36.5))
    assert idx.bbox(4) == (10, 20, 30, 44)


def test_crop_spec_with_returns_a_modified_copy():
    spec = CropSpec(merged_path="a.npy", label=1)
    other = spec.with_(label=9)
    assert spec.label == 1 and other.label == 9
    assert other.merged_path == "a.npy"


def test_extract_crop_kwargs_override_a_supplied_spec(tmp_path):
    data = _make_field(seed=167)
    path = _write_field(tmp_path, data)
    spec = CropSpec(merged_path=path, object_type="cell", label=1,
                    channels=(0, 1, 2), size=(48, 48), mask_dims=MASK_DIMS)
    a = extract_crop(path, spec=spec, label=2)
    b = extract_crop(path, "cell", 2, channels=(0, 1, 2), size=(48, 48),
                     mask_dims=MASK_DIMS)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Dependency hygiene
# ---------------------------------------------------------------------------

def test_module_does_not_import_torch_or_cellpose():
    """spacr.crops must stay light enough to sit on a GUI thumbnail path."""
    module_path = os.path.abspath(crops.__file__)
    # The sys.modules delta, not its absolute contents: the coverage runner's
    # sitecustomize pre-imports torch into every interpreter in this env, so
    # only what *this module's* import adds is meaningful.
    code = (
        "import importlib.util, sys\n"
        "before = set(sys.modules)\n"
        f"spec = importlib.util.spec_from_file_location('crops_probe', {module_path!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['crops_probe'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "added = set(sys.modules) - before\n"
        "banned = sorted(m for m in added\n"
        "                if m.split('.')[0] in ('torch', 'cellpose', 'tensorflow',\n"
        "                                       'skimage', 'scipy', 'cv2', 'spacr'))\n"
        "assert not banned, banned\n"
        "assert hasattr(mod, 'extract_crop')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_merged_field_rejects_a_non_3d_array_it_is_handed():
    with pytest.raises(CorruptMergedFile, match=r"\(H, W, C\)"):
        MergedField("in-memory.npy", array=np.zeros((4, 4), dtype=np.uint16))


def test_merged_field_constructed_from_a_missing_path_raises(tmp_path):
    with pytest.raises(MergedFileMissing):
        MergedField(str(tmp_path / "absent.npy"))


def test_cytoplasm_skips_child_planes_that_are_out_of_range(tmp_path):
    """A stale nucleus_mask_dim must not break the derived cytoplasm plane."""
    data = _make_field(seed=173)
    path = _write_field(tmp_path, data, "stale.npy")
    fld = open_merged_field(path, {"cell": CELL_DIM, "nucleus": 99,
                                   "pathogen": PATH_DIM}, use_cache=False)
    expected = np.where(data[:, :, PATH_DIM] != 0, 0, data[:, :, CELL_DIM])
    assert np.array_equal(fld.mask_plane("cytoplasm"), expected)


def test_db_bbox_that_does_not_contain_the_label_raises(tmp_path):
    data = _make_field(seed=179)
    path = _write_field(tmp_path, data)
    spec = CropSpec(merged_path=path, object_type="cell", label=1,
                    channels=(0, 1, 2), size=(32, 32), mask_dims=MASK_DIMS,
                    bbox=(80, 90, 80, 90))         # empty patch of background
    with pytest.raises(LabelMissing, match="not present"):
        extract_crop(path, spec=spec)


def test_extract_crop_object_type_argument_overrides_a_spec(tmp_path):
    data = _make_field(seed=181)
    path = _write_field(tmp_path, data)
    spec = CropSpec(merged_path=path, object_type="cell", label=2,
                    channels=(0, 1, 2), size=(48, 48), mask_dims=MASK_DIMS)
    got = extract_crop(path, "nucleus", 2, spec=spec)
    assert np.array_equal(got, _reference_png_crop(data, "nucleus", 2,
                                                   width=48, height=48))


def test_extract_crops_raises_by_default(tmp_path):
    data = _make_field(seed=191)
    path = _write_field(tmp_path, data)
    specs = [CropSpec(merged_path=path, object_type="cell", label=4242,
                      channels=(0, 1, 2), size=(32, 32), mask_dims=MASK_DIMS)]
    with pytest.raises(LabelMissing):
        extract_crops(path, specs)
    with pytest.raises(MergedFileMissing):
        extract_crops(str(tmp_path / "gone.npy"), specs)


def test_png_view_of_a_two_channel_crop(tmp_path):
    """UPDATED, DELIBERATELY: the empty plane of a 2-channel crop is BLUE.

    The PNG path pads a 2-channel crop with a zero third plane, so the crop's
    channel 2 is empty. Under the bug the write reversed everything and that
    empty plane came back as RED — the assertion below used to be on
    ``[:, :, 0]``. With the writer corrected it is where the user put it.
    """
    cv2 = pytest.importorskip("cv2")

    data = _make_field(seed=193)
    path = _write_field(tmp_path, data)
    folder = tmp_path / "cell_png"
    folder.mkdir()
    crops.stamp_crop_folder(str(folder))
    crop = extract_crop(path, "cell", 1, channels=(0, 1), size=(32, 32),
                        mask_dims=MASK_DIMS)
    assert crop.shape == (32, 32, 3)          # padded to RGB by the PNG path
    png_file = str(folder / "two.png")
    cv2.imwrite(png_file, crops.to_cv2_bgr(crop))
    assert np.array_equal(png_view(crop), crops.read_crop_png(png_file))

    raw_two = crop[:, :, :2]
    assert png_view(raw_two).shape == (32, 32, 3)
    assert not png_view(raw_two)[:, :, 2].any()
    assert png_view(raw_two)[:, :, 0].any()


def test_coerce_passes_non_strings_through():
    assert crops._coerce(7) == 7
    assert crops._coerce(None) is None
    assert crops._coerce("") is None
    assert crops._coerce("None") is None
    assert crops._coerce("not python") == "not python"


def test_crop_spec_from_settings_survives_malformed_settings():
    """A settings snapshot from a different crop_mode must not explode."""
    spec = crops.crop_spec_from_settings(
        {"crop_mode": ["cell"], "png_size": [[64, 64]], "dialate_pngs": [True],
         "dialate_png_ratios": 0.5}, object_type="pathogen")
    assert spec.size == (64, 64)            # falls back to the first entry
    assert spec.dilate is True
    assert spec.dilate_ratio == 0.5

    spec2 = crops.crop_spec_from_settings(
        {"crop_mode": [], "dialate_pngs": [], "dialate_png_ratios": [],
         "normalize": [1, 2, 3]})
    assert spec2.object_type == "cell"
    assert spec2.dilate is False
    assert spec2.dilate_ratio == 0.2
    assert spec2.normalize is False         # a 3-element normalize is not a pair


def test_row_get_handles_nan_and_attributes(tmp_path):
    pd = pytest.importorskip("pandas")

    row = pd.Series({"path_name": np.nan, "merged_path": "/x/y.npy",
                     "object_label": 3})
    assert crops._row_get(row, "path_name", "merged_path") == "/x/y.npy"

    class _Obj:
        object_label = 5

    assert crops._row_get(_Obj(), "object_label") == 5
    assert crops._row_get(_Obj(), "nothing", default="fallback") == "fallback"


def test_base_crop_source_get_many_walks_rows(tmp_path):
    cv2 = pytest.importorskip("cv2")
    png_dir = tmp_path / "data" / "plate1_A01" / "cell_png"
    png_dir.mkdir(parents=True)
    paths = []
    for i in range(3):
        p = png_dir / f"o{i}.png"
        cv2.imwrite(str(p), np.full((4, 4, 3), 10 * i + 1, np.uint8))
        paths.append({"png_path": str(p)})
    out = PngCropSource().get_many(paths)
    assert len(out) == 3
    assert [int(a[0, 0, 0]) for a in out] == [1, 11, 21]


def test_merged_source_object_type_argument_overrides_the_spec(tmp_path):
    data = _make_field(seed=197)
    path = _write_field(tmp_path, data)
    src = MergedCropSource(spec=CropSpec(merged_path="", object_type="cell",
                                         channels=(0, 1, 2), size=(48, 48),
                                         mask_dims=MASK_DIMS),
                           object_type="pathogen")
    assert src.spec.object_type == "pathogen"
    assert np.array_equal(src.get_array({"path_name": path, "object_label": 1}),
                          _reference_png_crop(data, "pathogen", 1,
                                              width=48, height=48))


def test_merged_source_rebuilds_the_path_from_file_name(tmp_path):
    data = _make_field(seed=199)
    path = _write_field(tmp_path, data, "plate1_A01_1.npy")
    src = MergedCropSource(spec=CropSpec(merged_path="", mask_dims=MASK_DIMS),
                           merged_root=os.path.dirname(path))
    spec = src.spec_for({"file_name": "plate1_A01_1", "object_label": 1})
    assert spec.merged_path == path


def test_has_png_folder_stops_descending_and_reports_false(tmp_path):
    root = tmp_path / "exp"
    deep = root / "data" / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    assert crops._has_png_folder(str(root)) is False
    assert crops._has_png_folder(str(tmp_path / "no-such-root")) is False


def test_resolve_tolerates_an_unreadable_settings_table(tmp_path, monkeypatch):
    root = _experiment(tmp_path, with_png=False, settings={"png_size": [64, 64]})

    def boom(_db):
        raise CropError("nope")

    monkeypatch.setattr(crops, "crop_settings_from_db", boom)
    src = resolve_crop_source(root)
    assert src.kind == "merged"
    assert "measurements.db" not in src.reason
    assert src.spec.size == (224, 224)          # shipped default


def test_module_source_mentions_no_heavy_imports():
    with open(crops.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    for banned in ("import torch", "from torch", "import cellpose",
                   "from cellpose", "import tensorflow", "from skimage",
                   "import scipy", "from scipy"):
        assert banned not in source, banned
