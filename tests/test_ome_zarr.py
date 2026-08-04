"""OME-Zarr read/write — round-trips, hand-written fixtures, and the refusals.

Three things this file is careful about.

**Nothing here is mocked.** Every round-trip writes real chunk files and reads
them back off disk, and every "spaCR honours the spec" claim is tested against
a directory this file builds itself with :func:`json.dump` and
:meth:`numpy.ndarray.tobytes` — not against spaCR's own writer. A reader and a
writer that agree with each other and with nothing else would pass a symmetric
test suite perfectly.

**zarr and numcodecs are not installed, and that is the point.** The whole
module is exercised with the optional extra absent, which is the state a plain
``pip install spacr`` leaves an environment in. The missing-dependency paths
are called directly rather than skipped — a test that skips when the extra is
missing tests nothing in the environment where it matters.

**Laziness is counted, not asserted.** ``test_a_small_region_does_not_decode
_every_chunk`` instruments :func:`spacr.ome_zarr._read_chunk_bytes`, the single
function every byte of chunk data passes through, and compares the call count
for a one-chunk region against the count for the whole array. "Chunked and
lazily readable" is the claim the format exists for, so it is the one claim
that gets a number.
"""
from __future__ import annotations

import gzip
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from spacr import ome_zarr
from spacr.layers import Spacing
from spacr.ome_zarr import (Axis, OmeZarrError, OmeZarrImage, ZarrExtraMissing,
                            axes_from_spacing, ngff_unit_to_spacr,
                            read_ome_zarr, read_ome_zarr_array, require_codec,
                            require_zarr, spacing_from_axes,
                            spacr_unit_to_ngff, write_ome_zarr)


# ---------------------------------------------------------------------------
# Fixtures built by hand — deliberately not through spaCR's writer
# ---------------------------------------------------------------------------

def _write_raw_v2_array(path: Path, array: np.ndarray, chunks, *,
                        separator=".", order="C", compressor=None,
                        fill_value=0, skip=()):
    """Write a zarr-format-2 array with json.dump and ndarray.tobytes().

    This is the independent implementation the reader is checked against, so
    it uses nothing from :mod:`spacr.ome_zarr`. ``skip`` names chunk grid
    indices to leave unwritten, which is how zarr represents "all fill_value".
    """
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 2,
        "shape": list(array.shape),
        "chunks": list(chunks),
        "dtype": array.dtype.str,
        "compressor": compressor,
        "fill_value": fill_value,
        "order": order,
        "filters": None,
        "dimension_separator": separator,
    }
    (path / ".zarray").write_text(json.dumps(meta), encoding="utf-8")

    grid = [range(-(-n // c)) for n, c in zip(array.shape, chunks)]
    for index in np.ndindex(*[len(list(g)) for g in grid]):
        if tuple(index) in skip:
            continue
        block = np.full(tuple(chunks), fill_value, dtype=array.dtype)
        cut = tuple(slice(i * c, min((i + 1) * c, n))
                    for i, c, n in zip(index, chunks, array.shape))
        piece = array[cut]
        block[tuple(slice(0, s) for s in piece.shape)] = piece
        raw = block.tobytes(order=order)
        if compressor and compressor["id"] == "zlib":
            raw = zlib.compress(raw)
        key = separator.join(str(i) for i in index)
        target = path.joinpath(*key.split("/")) if separator == "/" \
            else path / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def _write_raw_group(root: Path, attrs: dict):
    """Write a zarr-format-2 group's .zgroup and .zattrs by hand."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}),
                                  encoding="utf-8")
    (root / ".zattrs").write_text(json.dumps(attrs), encoding="utf-8")


def _multiscale(axes, datasets, name="hand-written", version="0.4"):
    """One `multiscales` entry, spelled out."""
    return {"multiscales": [{"version": version, "name": name, "axes": axes,
                             "datasets": datasets}]}


def _dataset(path, scale, translation=None):
    """One `datasets` entry with its coordinateTransformations."""
    transforms = [{"type": "scale", "scale": list(scale)}]
    if translation is not None:
        transforms.append({"type": "translation",
                           "translation": list(translation)})
    return {"path": path, "coordinateTransformations": transforms}


UM = "micrometer"


# ---------------------------------------------------------------------------
# 1. Write -> read round-trip
# ---------------------------------------------------------------------------

def test_write_read_round_trip_is_bit_identical_and_keeps_the_spacing(tmp_path):
    """The core promise: what went in comes back out, with its voxel size."""
    data = (np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9) * 7 + 11)
    spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")

    written = write_ome_zarr(tmp_path / "a.zarr", data, spacing=spacing,
                             levels=1, chunks=(1, 4, 4))
    reopened = read_ome_zarr(tmp_path / "a.zarr")

    back = reopened.read(0)
    assert back.dtype == data.dtype
    assert np.array_equal(back, data)                # bit-identical, not close

    for image in (written, reopened):
        assert image.spacing.scale == (2.0, 0.65, 0.65)
        assert image.spacing.translate == (0.0, 0.0, 0.0)
        assert image.spacing.axes == ("z", "y", "x")
        assert image.spacing.units == "um"
        assert image.units_declared is True
        assert image.axis_names == ("z", "y", "x")
        assert [a.unit for a in image.axes] == [UM, UM, UM]


def test_a_translated_spacing_round_trips_through_the_translation_transform(tmp_path):
    """A crop keeps its place in the mosaic: translate survives the file."""
    data = np.ones((4, 5), dtype=np.uint8)
    spacing = Spacing.from_map({"y": 0.5, "x": 0.5},
                               origin={"y": 12.5, "x": -3.25}, units="um")

    image = write_ome_zarr(tmp_path / "crop.zarr", data, spacing=spacing)

    assert image.spacing.translate == (12.5, -3.25)
    assert image.levels[0].translation == (12.5, -3.25)
    assert image.spacing.to_world((0, 0)) == (12.5, -3.25)


def test_pixel_units_are_written_as_no_unit_and_read_back_as_pixels(tmp_path):
    """NGFF has no pixel unit, so px writes nothing and nothing reads as px."""
    image = write_ome_zarr(tmp_path / "px.zarr", np.zeros((4, 4), np.uint8))

    attrs = json.loads((tmp_path / "px.zarr" / ".zattrs").read_text())
    axes = attrs["multiscales"][0]["axes"]
    assert all("unit" not in axis for axis in axes)

    assert image.spacing.units == "px"
    assert image.units_declared is False
    assert "file declares no unit" in image.describe()


def test_omero_channel_names_round_trip(tmp_path):
    """spaCR has channel names; a viewer should not have to invent them."""
    data = np.zeros((2, 4, 4), dtype=np.uint16)
    data[1] = 900
    image = write_ome_zarr(
        tmp_path / "ch.zarr", data,
        spacing=Spacing.from_map({"y": 0.65, "x": 0.65}, units="um"),
        axes=("c", "y", "x"), channel_names=("DAPI", "GFP"))

    assert image.channel_names == ("DAPI", "GFP")
    assert image.channel_axis is not None
    assert image.channel_axis.name == "c"
    assert image.channel_axis.unit is None          # never a unit on channels
    windows = [c["window"] for c in image.omero["channels"]]
    assert windows[0]["end"] == 0.0                 # measured, not assumed
    assert windows[1]["end"] == 900.0
    assert "DAPI, GFP" in image.describe()


# ---------------------------------------------------------------------------
# 2. Multiscale — the numbers written out by hand
# ---------------------------------------------------------------------------

def test_the_pyramid_has_the_levels_shapes_and_scales_the_rule_implies(tmp_path):
    """Level count, per-level shape, and scale == level 0 x 2^k. By hand.

    Shapes follow ``ceil(n / 2)`` on y and x only: (3, 8, 9) -> (3, 4, 5) ->
    (3, 2, 3). z is untouched, because halving a 2 um z-step three times gives
    a 16 um step on an image whose pixels are 0.65 um.
    """
    data = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")

    image = write_ome_zarr(tmp_path / "p.zarr", data, spacing=spacing, levels=3)

    assert [lv.shape for lv in image.levels] == [(3, 8, 9), (3, 4, 5), (3, 2, 3)]
    assert [lv.path for lv in image.levels] == ["0", "1", "2"]

    attrs = json.loads((tmp_path / "p.zarr" / ".zattrs").read_text())
    datasets = attrs["multiscales"][0]["datasets"]
    scales = [d["coordinateTransformations"][0]["scale"] for d in datasets]
    assert scales == [[2.0, 0.65, 1.3 / 2], [2.0, 1.3, 1.3], [2.0, 2.6, 2.6]]
    # ... spelled out again without the arithmetic, so a wrong constant cannot
    # hide behind a matching expression:
    assert scales[0] == [2.0, 0.65, 0.65]
    assert scales[1] == [2.0, 1.3, 1.3]
    assert scales[2] == [2.0, 2.6, 2.6]

    translations = [d["coordinateTransformations"][1]["translation"]
                    for d in datasets]
    # A block mean puts level k's element 0 at the centre of the 2^k level-0
    # elements it averaged: 0.65 * (2^k - 1) / 2. approx only for level 2,
    # where 0.65 * 1.5 is not exactly the double nearest 0.975 — the number is
    # right, the decimal is just not representable.
    assert translations[0] == [0.0, 0.0, 0.0]
    assert translations[1] == [0.0, 0.325, 0.325]
    assert translations[2] == pytest.approx([0.0, 0.975, 0.975])

    assert image.multiscale["type"] == "local mean"
    assert image.multiscale["metadata"]["downsample_axes"] == ["y", "x"]


def test_a_strided_pyramid_is_not_translated_and_keeps_its_labels(tmp_path):
    """Striding samples element 0 exactly, so there is no half-pixel shift.

    And it is the only correct rule for labels: the mean of 3 and 5 is 4, an
    object that is not in the image.
    """
    labels = np.zeros((4, 4), dtype=np.uint16)
    labels[0:2, 0] = 3            # a 2x1 object of label 3...
    labels[0:2, 1] = 5            # ...touching a 2x1 object of label 5
    spacing = Spacing.from_map({"y": 1.0, "x": 1.0}, units="um")

    image = write_ome_zarr(tmp_path / "lab.zarr", labels, spacing=spacing,
                           levels=2, downsample="stride")

    assert image.levels[1].scale == (2.0, 2.0)
    assert image.levels[1].translation == (0.0, 0.0)
    coarse = image.read(1)
    assert set(np.unique(coarse)) <= {0, 3, 5}       # no invented label 4
    assert coarse[0, 0] == 3
    assert image.multiscale["type"] == "nearest (stride)"

    mean_image = write_ome_zarr(tmp_path / "mean.zarr", labels,
                                spacing=spacing, levels=2, downsample="mean")
    assert mean_image.read(1)[0, 0] == 4             # exactly what to avoid


def test_level_for_size_picks_the_coarsest_level_that_is_still_big_enough(tmp_path):
    """The multiscale payoff: draw 200 px without decoding 40k."""
    data = np.zeros((32, 32), dtype=np.uint8)
    image = write_ome_zarr(tmp_path / "s.zarr", data, levels=4)

    assert [lv.shape for lv in image.levels] == [(32, 32), (16, 16), (8, 8),
                                                 (4, 4)]
    assert image.level_for_size(32) == 0
    assert image.level_for_size(16) == 1
    assert image.level_for_size(9) == 1              # 8 would need upsampling
    assert image.level_for_size(8) == 2
    assert image.level_for_size(1) == 3
    assert image.level_for_size(4096) == 0           # bigger than the image

    with pytest.raises(OmeZarrError, match="no axis 'q'"):
        image.level_for_size(8, axes=("q",))


def test_a_world_box_names_the_matching_pixels_at_every_level(tmp_path):
    """What the per-level transformations buy: one box, any resolution."""
    data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    spacing = Spacing.from_map({"y": 1.0, "x": 1.0}, units="um")
    image = write_ome_zarr(tmp_path / "w.zarr", data, spacing=spacing, levels=3)

    box = {"y": (0.0, 8.0), "x": (0.0, 8.0)}
    assert image.read(0, world_region=box).shape == (8, 8)
    assert image.read(1, world_region=box).shape == (4, 4)
    assert image.read(2, world_region=box).shape == (2, 2)
    # The same corner of the same object, at three resolutions.
    assert image.read(0, world_region=box)[0, 0] == data[0, 0]


# ---------------------------------------------------------------------------
# 3. Reading an NGFF spaCR did not write
# ---------------------------------------------------------------------------

def test_reads_a_hand_written_5d_ngff_and_keeps_time_out_of_the_spacing(tmp_path):
    """The test that proves the spec is honoured rather than the writer.

    Built with json.dump and tobytes: a (t, c, z, y, x) image whose time axis
    is in seconds and whose space axes are in micrometers. The Spacing must
    contain the three space axes and nothing else — a spacing holding a
    seconds-per-step next to a micrometers-per-step would answer
    LayerStack's unit check with a string that is wrong for one of them.
    """
    root = tmp_path / "hand.zarr"
    array = np.arange(2 * 2 * 3 * 4 * 5, dtype="<u2").reshape(2, 2, 3, 4, 5)
    _write_raw_v2_array(root / "0", array, (1, 1, 2, 2, 3))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "t", "type": "time", "unit": "second"},
              {"name": "c", "type": "channel"},
              {"name": "z", "type": "space", "unit": UM},
              {"name": "y", "type": "space", "unit": UM},
              {"name": "x", "type": "space", "unit": UM}],
        datasets=[_dataset("0", [30.0, 1.0, 2.0, 0.65, 0.65])]))

    image = read_ome_zarr(root)

    assert image.ngff_version == "0.4"
    assert image.axis_names == ("t", "c", "z", "y", "x")

    spacing = image.spacing
    assert spacing.axes == ("z", "y", "x")           # ONLY the space axes
    assert spacing.scale == (2.0, 0.65, 0.65)
    assert spacing.units == "um"

    assert [a.name for a in image.other_axes] == ["t", "c"]
    assert image.time_axis is not None
    assert image.time_axis.unit == "second"          # reported, not converted
    assert image.time_axis.scale == 30.0
    assert image.time_axis.spacr_units() == "s"
    assert image.channel_axis is not None
    assert image.channel_axis.unit is None

    assert np.array_equal(image.read(0), array)
    assert np.array_equal(image.read(0, {"t": 1, "c": 0}), array[1:2, 0:1])


def test_a_group_level_transformation_composes_with_the_dataset_one(tmp_path):
    """`multiscales.coordinateTransformations` is applied after the dataset's.

    Files that carry one are rare, which is why ignoring it is such a quiet
    bug: everything works until the plate that has a group-level voxel size.
    """
    root = tmp_path / "composed.zarr"
    array = np.zeros((4, 4), dtype="<u2")
    _write_raw_v2_array(root / "0", array, (2, 2))
    attrs = _multiscale(
        axes=[{"name": "y", "type": "space", "unit": UM},
              {"name": "x", "type": "space", "unit": UM}],
        datasets=[_dataset("0", [2.0, 2.0], [1.0, 1.0])])
    attrs["multiscales"][0]["coordinateTransformations"] = [
        {"type": "scale", "scale": [0.5, 0.5]},
        {"type": "translation", "translation": [10.0, 0.0]},
    ]
    _write_raw_group(root, attrs)

    image = read_ome_zarr(root)

    # scale = ms * ds; translation = ms_scale * ds_translation + ms_translation
    assert image.levels[0].scale == (1.0, 1.0)
    assert image.levels[0].translation == (10.5, 0.5)


def test_an_axes_list_of_bare_names_and_a_missing_type_are_inferred(tmp_path):
    """Pre-0.4 files and sloppy 0.4 files still open.

    `type` is SHOULD, not MUST, and the axis name carries the answer for every
    axis NGFF actually defines.
    """
    root = tmp_path / "loose.zarr"
    array = np.zeros((2, 4, 4), dtype="|u1")
    _write_raw_v2_array(root / "0", array, (1, 4, 4))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "c"}, {"name": "y", "unit": UM}, {"name": "x", "unit": UM}],
        datasets=[_dataset("0", [1.0, 0.5, 0.5])]))

    image = read_ome_zarr(root)
    assert image.channel_axis is not None            # inferred from the name
    assert image.spacing.axes == ("y", "x")


def test_reads_a_zarr_v3_ngff_0_5_group(tmp_path):
    """0.5 is tolerated: zarr.json, metadata under `ome`, c/-prefixed chunks."""
    root = tmp_path / "v3.zarr"
    array = np.arange(4 * 6, dtype="<u2").reshape(4, 6)
    (root / "0").mkdir(parents=True)
    (root / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "name": "v3",
            "axes": [{"name": "y", "type": "space", "unit": UM},
                     {"name": "x", "type": "space", "unit": UM}],
            "datasets": [_dataset("0", [0.5, 0.5])]}]}},
    }), encoding="utf-8")
    (root / "0" / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array",
        "shape": [4, 6], "data_type": "uint16",
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [2, 3]}},
        "chunk_key_encoding": {"name": "default",
                               "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                   {"name": "gzip", "configuration": {"level": 5}}],
    }), encoding="utf-8")
    for i in range(2):
        for j in range(2):
            block = array[i * 2:(i + 1) * 2, j * 3:(j + 1) * 3]
            target = root / "0" / "c" / str(i) / str(j)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.compress(block.tobytes()))

    image = read_ome_zarr(root)

    assert image.ngff_version == "0.5"
    assert image.levels[0].zarr_format == 3
    assert image.spacing.units == "um"
    assert image.spacing.scale == (0.5, 0.5)
    assert np.array_equal(image.read(0), array)
    assert np.array_equal(image.read(0, {"y": (2, 4)}), array[2:4])


# ---------------------------------------------------------------------------
# 4. Regions, and the count that proves laziness
# ---------------------------------------------------------------------------

def test_a_region_read_equals_slicing_the_whole_array(tmp_path):
    """Correctness first: the cheap answer must be the same answer."""
    data = np.arange(3 * 9 * 11, dtype=np.uint16).reshape(3, 9, 11)
    image = write_ome_zarr(tmp_path / "r.zarr", data, chunks=(1, 4, 4))

    whole = image.read(0)
    assert np.array_equal(whole, data)
    assert np.array_equal(image.read(0, {"y": (2, 7), "x": (3, 10)}),
                          data[:, 2:7, 3:10])
    assert np.array_equal(image.read(0, (slice(1, 3), slice(0, 5), None)),
                          data[1:3, 0:5, :])
    assert np.array_equal(image.read(0, {"z": 2}), data[2:3])
    assert np.array_equal(image.read(0, {"z": -1}), data[-1:])
    # A box that lands exactly on chunk boundaries, and one that lands on none.
    assert np.array_equal(image.read(0, {"y": (4, 8), "x": (4, 8)}),
                          data[:, 4:8, 4:8])
    assert np.array_equal(image.read(0, {"y": (1, 2), "x": (5, 6)}),
                          data[:, 1:2, 5:6])
    assert image.read(0, {"y": (3, 3)}).shape == (3, 0, 11)


def test_a_small_region_does_not_decode_every_chunk(tmp_path):
    """The claim the whole format exists for, as a number.

    ``_read_chunk_bytes`` is the single function every byte of chunk data
    passes through, so counting its calls counts the chunks touched. A
    (3, 8, 8) array in (1, 4, 4) chunks is a 3 x 2 x 2 = 12 chunk grid; one
    chunk's worth of region must cost one chunk.
    """
    data = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8) + 1
    image = write_ome_zarr(tmp_path / "lazy.zarr", data, chunks=(1, 4, 4))
    assert image.levels[0].n_chunks == 12

    opened: list = []
    real = ome_zarr._read_chunk_bytes

    def counting(path):
        opened.append(Path(path))
        return real(path)

    ome_zarr._read_chunk_bytes = counting
    try:
        opened.clear()
        one = image.read(0, {"z": (0, 1), "y": (0, 4), "x": (0, 4)})
        n_one = len(opened)

        opened.clear()
        image.read(0)
        n_all = len(opened)

        opened.clear()
        image.read(0, {"z": (0, 1)})
        n_plane = len(opened)
    finally:
        ome_zarr._read_chunk_bytes = real

    assert np.array_equal(one, data[0:1, 0:4, 0:4])
    assert n_one == 1, f"a one-chunk region opened {n_one} chunks"
    assert n_all == 12
    assert n_plane == 4
    assert n_one < n_all                      # the point, stated plainly

    # And a region straddling a boundary costs exactly the chunks it touches.
    ome_zarr._read_chunk_bytes = counting
    try:
        opened.clear()
        image.read(0, {"z": (0, 1), "y": (3, 5), "x": (3, 5)})
    finally:
        ome_zarr._read_chunk_bytes = real
    assert len(opened) == 4
    assert len({p for p in opened}) == 4       # four distinct chunk files


def test_reading_the_metadata_touches_no_chunk_at_all(tmp_path):
    """Levels, shapes, voxel sizes and channels for the price of some JSON."""
    data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    write_ome_zarr(tmp_path / "meta.zarr", data, levels=3, chunks=(4, 4))

    opened: list = []
    real = ome_zarr._read_chunk_bytes
    ome_zarr._read_chunk_bytes = lambda path: (opened.append(path), real(path))[1]
    try:
        image = read_ome_zarr(tmp_path / "meta.zarr")
        _ = (image.describe(), image.spacing, [lv.shape for lv in image.levels],
             image.level_for_size(8), image.levels[0].nbytes)
    finally:
        ome_zarr._read_chunk_bytes = real
    assert opened == []


# ---------------------------------------------------------------------------
# 5. Storage details: separators, fill_value, order, dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("separator", [".", "/"])
def test_dimension_separator_round_trips_both_ways(tmp_path, separator):
    """`.` is flat, `/` is nested; both are zarr, and both must read."""
    data = np.arange(6 * 6, dtype=np.uint16).reshape(6, 6)
    out = tmp_path / f"sep{'dot' if separator == '.' else 'slash'}.zarr"
    image = write_ome_zarr(out, data, chunks=(3, 3),
                           dimension_separator=separator)

    meta = json.loads((out / "0" / ".zarray").read_text())
    assert meta["dimension_separator"] == separator
    assert (out / "0" / ("0.0" if separator == "." else "0/0")).is_file()
    assert np.array_equal(image.read(0), data)
    assert np.array_equal(image.read(0, {"y": (3, 6), "x": (3, 6)}),
                          data[3:6, 3:6])


@pytest.mark.parametrize("separator", [".", "/"])
def test_a_hand_written_array_reads_with_either_separator(tmp_path, separator):
    """Same claim, against a directory spaCR's writer never touched."""
    root = tmp_path / f"raw{len(separator)}{separator!r}".replace("'", "") \
        .replace("/", "slash").replace(".", "dot")
    array = np.arange(6 * 6, dtype=">u2").reshape(6, 6)
    _write_raw_v2_array(root / "0", array, (3, 3), separator=separator)
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space", "unit": UM},
              {"name": "x", "type": "space", "unit": UM}],
        datasets=[_dataset("0", [1.0, 1.0])]))

    assert np.array_equal(read_ome_zarr_array(root), array)


def test_a_missing_chunk_reads_back_as_fill_value(tmp_path):
    """An unwritten chunk is not an error; it is empty space, and it is free."""
    root = tmp_path / "sparse.zarr"
    array = np.full((4, 4), 7, dtype="<u2")
    _write_raw_v2_array(root / "0", array, (2, 2), fill_value=99,
                        skip={(0, 1), (1, 0)})
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])]))

    out = read_ome_zarr_array(root)
    expected = array.copy()
    expected[0:2, 2:4] = 99
    expected[2:4, 0:2] = 99
    assert np.array_equal(out, expected)

    # Reading only the missing chunk still costs no decode and is still 99.
    assert np.array_equal(read_ome_zarr_array(root, region={"y": (0, 2),
                                                            "x": (2, 4)}),
                          np.full((2, 2), 99, dtype="<u2"))


def test_the_writer_omits_all_fill_chunks_and_they_read_back(tmp_path):
    """A sparse mask should not cost what a dense one does."""
    data = np.zeros((8, 8), dtype=np.uint8)
    data[0:4, 0:4] = 5
    image = write_ome_zarr(tmp_path / "empty.zarr", data, chunks=(4, 4))

    stored = sorted(p.name for p in (tmp_path / "empty.zarr" / "0").iterdir()
                    if not p.name.startswith("."))
    assert stored == ["0"]                       # one nested directory, "0/0"
    assert image.levels[0].n_chunks == 4         # of four possible chunks
    assert np.array_equal(image.read(0), data)

    dense = write_ome_zarr(tmp_path / "dense.zarr", data, chunks=(4, 4),
                           write_empty_chunks=True)
    assert np.array_equal(dense.read(0), data)


@pytest.mark.parametrize("order", ["C", "F"])
def test_chunk_memory_order_round_trips(tmp_path, order):
    """F-order chunks are legal zarr and are read as F-order, not transposed."""
    data = np.arange(4 * 6, dtype=np.uint16).reshape(4, 6)
    out = tmp_path / f"order{order}.zarr"
    image = write_ome_zarr(out, data, chunks=(2, 3), order=order)

    assert json.loads((out / "0" / ".zarray").read_text())["order"] == order
    assert np.array_equal(image.read(0), data)
    assert np.array_equal(image.read(0, {"y": (2, 4), "x": (0, 3)}),
                          data[2:4, 0:3])


def test_a_hand_written_f_order_array_reads_correctly(tmp_path):
    """The one that catches a reader that ignored `order` and got lucky."""
    root = tmp_path / "forder.zarr"
    array = np.arange(4 * 6, dtype="<u2").reshape(4, 6)
    _write_raw_v2_array(root / "0", array, (4, 6), order="F")
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])]))

    assert np.array_equal(read_ome_zarr_array(root), array)
    # Contrast: read as C order the same bytes are a transpose, so the test
    # above really does discriminate.
    raw = (root / "0" / "0.0").read_bytes()
    assert not np.array_equal(
        np.frombuffer(raw, dtype="<u2").reshape(4, 6, order="C"), array)


def test_big_endian_dtype_round_trips_byte_order_and_all(tmp_path):
    """`>u2` is a real dtype in real microscope files; it is kept, not cast."""
    data = (np.arange(4 * 5, dtype=">u2").reshape(4, 5) * 1000).astype(">u2")
    image = write_ome_zarr(tmp_path / "be.zarr", data, levels=2)

    assert json.loads((tmp_path / "be.zarr" / "0" / ".zarray").read_text())[
        "dtype"] == ">u2"
    assert image.dtype == np.dtype(">u2")
    back = image.read(0)
    assert back.dtype == np.dtype(">u2")
    assert np.array_equal(back, data)
    assert image.read(1).dtype == np.dtype(">u2")


def test_a_hand_written_big_endian_array_is_not_read_as_little(tmp_path):
    """Byte order read wrong yields plausible numbers, never an exception."""
    root = tmp_path / "beraw.zarr"
    array = np.array([[1, 2], [3, 4]], dtype=">u2")
    _write_raw_v2_array(root / "0", array, (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])]))

    out = read_ome_zarr_array(root)
    assert out.dtype == np.dtype(">u2")
    assert out.tolist() == [[1, 2], [3, 4]]
    # The same bytes read little-endian would be 256, 512, 768, 1024.
    raw = (root / "0" / "0.0").read_bytes()
    assert np.frombuffer(raw, dtype="<u2").tolist() == [256, 512, 768, 1024]


def test_float_arrays_and_a_nan_fill_value_survive(tmp_path):
    """NaN has to go through JSON as a string; the spec says which one."""
    data = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    image = write_ome_zarr(tmp_path / "f.zarr", data, chunks=(2, 2),
                           fill_value=float("nan"))

    assert json.loads((tmp_path / "f.zarr" / "0" / ".zarray").read_text())[
        "fill_value"] == "NaN"
    assert np.array_equal(image.read(0), data)


@pytest.mark.parametrize("codec", [None, "zlib", "gzip", "bz2", "lzma"])
def test_every_stdlib_codec_round_trips_without_the_extra(tmp_path, codec):
    """The reason the extra is optional: these need nothing but the stdlib."""
    data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    image = write_ome_zarr(tmp_path / f"c_{codec}.zarr", data,
                           compressor=codec, chunks=(4, 4))

    stored = json.loads(
        (tmp_path / f"c_{codec}.zarr" / "0" / ".zarray").read_text())
    assert (stored["compressor"] or {}).get("id") == codec
    assert image.levels[0].compressor == codec
    assert np.array_equal(image.read(0), data)


# ---------------------------------------------------------------------------
# 6. The missing extra — exercised directly, never skipped
# ---------------------------------------------------------------------------

def test_zarr_and_numcodecs_really_are_absent():
    """The premise of every test below it, checked rather than assumed."""
    for name in ("zarr", "numcodecs"):
        with pytest.raises(ImportError):
            __import__(name)
    assert ome_zarr._zarr_is_installed() is False


def test_require_zarr_names_the_extra_and_the_command():
    """Called directly: the message is the feature, so it is asserted."""
    with pytest.raises(ZarrExtraMissing) as excinfo:
        require_zarr()
    message = str(excinfo.value)
    assert 'python -m pip install "spacr[zarr]"' in message
    assert "missing module: zarr" in message
    assert isinstance(excinfo.value, ImportError)      # for except ImportError
    assert isinstance(excinfo.value, OmeZarrError)     # and for except ours


@pytest.mark.parametrize("codec", ["blosc", "lz4", "zstandard"])
def test_require_codec_names_the_codec_and_the_extra(codec):
    """"Install the extra" without saying what for cannot be acted on."""
    with pytest.raises(ZarrExtraMissing) as excinfo:
        require_codec(codec, {"id": codec})
    message = str(excinfo.value)
    assert codec in message
    assert 'python -m pip install "spacr[zarr]"' in message
    assert "zlib" in message                    # what works without the extra


def test_zstd_uses_the_standard_library_on_314_and_the_extra_before_it():
    """One codec whose availability depends on the interpreter, not the install.

    ``compression.zstd`` is standard from Python 3.14, so the answer differs
    per interpreter and the test asserts the branch that is actually taken
    rather than asserting the one this machine happens to have.
    """
    import sys

    try:
        module = ome_zarr._stdlib_zstd()
    except ImportError:
        assert sys.version_info < (3, 14)
        with pytest.raises(ZarrExtraMissing) as excinfo:
            require_codec("zstd", {"id": "zstd"})
        assert "zstd" in str(excinfo.value)
        assert 'pip install "spacr[zarr]"' in str(excinfo.value)
    else:
        assert sys.version_info >= (3, 14)
        blob = module.compress(b"hello", 3)
        assert require_codec("zstd")(blob) == b"hello"


def test_require_codec_returns_a_working_decoder_for_the_stdlib_ones():
    """The other half: no exception, and the bytes actually come back."""
    assert require_codec("zlib")(zlib.compress(b"hello")) == b"hello"
    assert require_codec("gzip")(gzip.compress(b"hello")) == b"hello"
    with pytest.raises(OmeZarrError, match="needs a name"):
        require_codec("")


def test_a_blosc_array_reads_its_metadata_and_refuses_only_its_pixels(tmp_path):
    """Shapes and voxel size without numcodecs; pixels with an actionable error.

    Refusing to even open a blosc file would make the extra mandatory for the
    half of the format that is pure JSON.
    """
    root = tmp_path / "blosc.zarr"
    (root / "0").mkdir(parents=True)
    (root / "0" / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [4, 4], "chunks": [2, 2], "dtype": "<u2",
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5,
                       "shuffle": 1},
        "fill_value": 0, "order": "C", "filters": None,
        "dimension_separator": ".",
    }), encoding="utf-8")
    (root / "0" / "0.0").write_bytes(b"not really blosc, and never decoded")
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space", "unit": UM},
              {"name": "x", "type": "space", "unit": UM}],
        datasets=[_dataset("0", [0.65, 0.65])]))

    image = read_ome_zarr(root)                    # metadata: fine
    assert image.levels[0].shape == (4, 4)
    assert image.levels[0].compressor == "blosc"
    assert image.spacing.scale == (0.65, 0.65)

    with pytest.raises(ZarrExtraMissing) as excinfo:
        image.read(0)
    message = str(excinfo.value)
    assert "blosc" in message
    assert 'python -m pip install "spacr[zarr]"' in message


def test_a_filter_pipeline_is_refused_with_the_install_hint(tmp_path):
    """spaCR applies no filters; saying so beats returning wrong numbers."""
    root = tmp_path / "filtered.zarr"
    (root / "0").mkdir(parents=True)
    (root / "0" / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [2, 2], "chunks": [2, 2], "dtype": "<u2",
        "compressor": None, "fill_value": 0, "order": "C",
        "filters": [{"id": "delta", "dtype": "<u2"}],
    }), encoding="utf-8")
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])]))

    with pytest.raises(OmeZarrError, match="filter pipeline"):
        read_ome_zarr(root)


# ---------------------------------------------------------------------------
# 7. Refusals, each with its message asserted
# ---------------------------------------------------------------------------

def test_an_unknown_unit_is_refused_and_not_quietly_read_as_pixels(tmp_path):
    """The refusal this module exists for."""
    root = tmp_path / "furlong.zarr"
    _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space", "unit": "furlong"},
              {"name": "x", "type": "space", "unit": "furlong"}],
        datasets=[_dataset("0", [0.65, 0.65])]))

    with pytest.raises(OmeZarrError) as excinfo:
        read_ome_zarr(root)
    message = str(excinfo.value)
    assert "furlong" in message
    assert "micrometer" in message
    assert "million" in message                 # says what the fallback costs


def test_a_legal_ngff_unit_spacr_cannot_tokenise_says_so_differently(tmp_path):
    """Two problems, two fixes: your file is wrong vs. convert your file."""
    with pytest.raises(OmeZarrError) as excinfo:
        ngff_unit_to_spacr("parsec", axis="x")
    assert "legal OME-NGFF unit" in str(excinfo.value)
    assert "no short token" in str(excinfo.value)

    with pytest.raises(OmeZarrError) as excinfo:
        ngff_unit_to_spacr("furlong", axis="x")
    assert "not an OME-NGFF unit" in str(excinfo.value)


def test_space_axes_in_different_units_are_refused_by_name(tmp_path):
    """One Spacing has one unit; there is no honest way to hold y=um, x=nm."""
    root = tmp_path / "mixed.zarr"
    _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space", "unit": UM},
              {"name": "x", "type": "space", "unit": "nanometer"}],
        datasets=[_dataset("0", [0.65, 650.0])]))

    with pytest.raises(OmeZarrError) as excinfo:
        read_ome_zarr(root)
    message = str(excinfo.value)
    assert "y=um" in message and "x=nm" in message
    assert "not all in the same unit" in message


def test_a_group_without_multiscales_is_refused_with_a_hint(tmp_path):
    """A plate group is a zarr group too; say where the images are."""
    root = tmp_path / "plate.zarr"
    _write_raw_group(root, {"plate": {"columns": [], "rows": []}})

    with pytest.raises(OmeZarrError) as excinfo:
        read_ome_zarr(root)
    message = str(excinfo.value)
    assert "no `multiscales` metadata" in message
    assert "<row>/<column>/<field>" in message
    assert "plate" in message                   # names what it did find


def test_an_unsupported_zarr_format_is_refused_by_number(tmp_path):
    """Read the format field rather than assuming it."""
    root = tmp_path / "future.zarr"
    root.mkdir()
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 42}),
                                  encoding="utf-8")
    with pytest.raises(OmeZarrError, match="zarr_format 42"):
        read_ome_zarr(root)

    other = tmp_path / "future3.zarr"
    other.mkdir()
    (other / "zarr.json").write_text(
        json.dumps({"zarr_format": 4, "node_type": "group", "attributes": {}}),
        encoding="utf-8")
    with pytest.raises(OmeZarrError) as excinfo:
        read_ome_zarr(other)
    assert "zarr_format 4" in str(excinfo.value)
    assert "(2, 3)" in str(excinfo.value)


def test_a_directory_that_is_not_a_zarr_group_at_all_is_refused(tmp_path):
    """The most common mistake: pointing at the plate, or at a folder of TIFs."""
    (tmp_path / "images").mkdir()
    with pytest.raises(OmeZarrError, match="not a zarr group"):
        read_ome_zarr(tmp_path / "images")
    with pytest.raises(OmeZarrError, match="does not exist"):
        read_ome_zarr(tmp_path / "nope.zarr")
    (tmp_path / "a.tif").write_bytes(b"II*\x00")
    with pytest.raises(OmeZarrError, match="is a file"):
        read_ome_zarr(tmp_path / "a.tif")


def test_writing_over_an_existing_group_is_refused_unless_asked(tmp_path):
    """Half-overwriting a pyramid leaves the old deep levels readable."""
    out = tmp_path / "twice.zarr"
    first = write_ome_zarr(out, np.zeros((8, 8), np.uint8), levels=3)
    assert len(first.levels) == 3

    with pytest.raises(OmeZarrError) as excinfo:
        write_ome_zarr(out, np.zeros((8, 8), np.uint8), levels=1)
    assert "already a zarr group" in str(excinfo.value)
    assert "overwrite=True" in str(excinfo.value)
    assert len(read_ome_zarr(out).levels) == 3        # untouched

    second = write_ome_zarr(out, np.ones((8, 8), np.uint8), levels=1,
                            overwrite=True)
    assert len(second.levels) == 1
    assert not (out / "1").exists()                   # the old level 1 is gone
    assert np.array_equal(second.read(0), np.ones((8, 8), np.uint8))


def test_a_non_empty_directory_that_is_not_a_group_is_not_deleted(tmp_path):
    """spaCR will not rmtree a directory it did not recognise."""
    out = tmp_path / "mydata"
    out.mkdir()
    (out / "precious.csv").write_text("keep me", encoding="utf-8")

    with pytest.raises(OmeZarrError, match="not empty"):
        write_ome_zarr(out, np.zeros((4, 4), np.uint8))
    assert (out / "precious.csv").read_text() == "keep me"


def test_an_impossible_axis_layout_is_refused_before_anything_is_written(tmp_path):
    """NGFF 0.4 fixes the axis order; relabelling is not the fix."""
    data = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(OmeZarrError, match="space axes must come last"):
        write_ome_zarr(tmp_path / "bad.zarr", data,
                       axes=(Axis.space("y", 1.0), Axis.channel("c"),
                             Axis.space("x", 1.0)))
    assert not (tmp_path / "bad.zarr").exists()

    with pytest.raises(OmeZarrError, match="ordered time, then channel"):
        write_ome_zarr(tmp_path / "bad1.zarr",
                       np.zeros((2, 2, 4, 4), dtype=np.uint8),
                       axes=(Axis.channel("c"), Axis.time("t"),
                             Axis.space("y", 1.0), Axis.space("x", 1.0)))

    with pytest.raises(OmeZarrError, match="2 or 3 space axes"):
        write_ome_zarr(tmp_path / "bad2.zarr", data,
                       axes=(Axis.channel("c"), Axis.channel("c2"),
                             Axis.space("x", 1.0)))

    with pytest.raises(OmeZarrError, match="2 to 5 dimensional"):
        write_ome_zarr(tmp_path / "bad3.zarr", np.zeros(4, dtype=np.uint8))

    with pytest.raises(OmeZarrError, match="unique"):
        write_ome_zarr(tmp_path / "bad4.zarr", np.zeros((4, 4), np.uint8),
                       axes=("y", "y"))


def test_bad_write_arguments_are_refused_with_the_fix_in_the_message(tmp_path):
    """Each of these has a plausible wrong behaviour; none of them is taken."""
    data = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(OmeZarrError, match="downsample="):
        write_ome_zarr(tmp_path / "a.zarr", data, downsample="gaussian")
    with pytest.raises(OmeZarrError, match="at least level 0"):
        write_ome_zarr(tmp_path / "b.zarr", data, levels=0)
    with pytest.raises(OmeZarrError, match="spacr.layers.Spacing"):
        write_ome_zarr(tmp_path / "c.zarr", data, spacing=(0.65, 0.65))
    with pytest.raises(OmeZarrError, match="spaCR writes OME-NGFF 0.4"):
        write_ome_zarr(tmp_path / "d.zarr", data, ngff_version="0.5")
    with pytest.raises(OmeZarrError, match="chunks"):
        write_ome_zarr(tmp_path / "e.zarr", data, chunks=(2, 2, 2))
    with pytest.raises(OmeZarrError, match="downsample_axes names"):
        write_ome_zarr(tmp_path / "f.zarr", data, levels=2,
                       downsample_axes=("q",))
    with pytest.raises(OmeZarrError, match="no channel axis"):
        write_ome_zarr(tmp_path / "g.zarr", data, channel_names=("DAPI",))
    with pytest.raises(OmeZarrError, match="cannot write dtype"):
        write_ome_zarr(tmp_path / "h.zarr", np.array([["a", "b"], ["c", "d"]]))


def test_a_spacing_unit_with_no_ngff_name_is_refused_on_write(tmp_path):
    """The write-side mirror of the read-side refusal."""
    weird = Spacing.from_map({"y": 1.0, "x": 1.0}, units="furlong")
    with pytest.raises(OmeZarrError) as excinfo:
        write_ome_zarr(tmp_path / "u.zarr", np.zeros((2, 2), np.uint8),
                       spacing=weird)
    assert "no OME-NGFF equivalent" in str(excinfo.value)
    assert "um" in str(excinfo.value)


def test_region_requests_that_cannot_mean_anything_are_refused(tmp_path):
    """An index box off the end is a level mix-up, so it says so."""
    image = write_ome_zarr(tmp_path / "reg.zarr",
                           np.zeros((8, 8), np.uint8), levels=2)

    with pytest.raises(OmeZarrError, match="outside the level's extent"):
        image.read(0, {"y": (0, 99)})
    with pytest.raises(OmeZarrError, match="region names axis 'q'"):
        image.read(0, {"q": (0, 1)})
    with pytest.raises(OmeZarrError, match="one per axis"):
        image.read(0, (slice(0, 1),))
    with pytest.raises(OmeZarrError, match="has step"):
        image.read(0, {"y": slice(0, 8, 2)})
    with pytest.raises(OmeZarrError, match="use a slice"):
        image.read(0, {"y": (0, 1, 2)})
    with pytest.raises(OmeZarrError, match="no level 7"):
        image.read(7)
    with pytest.raises(OmeZarrError, match="no level with path 'nope'"):
        image.read("nope")
    with pytest.raises(OmeZarrError, match="both region= and world_region="):
        image.read(0, {"y": (0, 2)}, world_region={"y": (0.0, 2.0)})
    with pytest.raises(OmeZarrError, match="not a space axis"):
        image.read(0, world_region={"t": (0.0, 2.0)})


def test_a_truncated_chunk_is_refused_rather_than_padded(tmp_path):
    """Zarr stores full chunks; a short one means the file is damaged."""
    root = tmp_path / "short.zarr"
    _write_raw_v2_array(root / "0", np.zeros((4, 4), "<u2"), (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])]))
    (root / "0" / "0.0").write_bytes(b"\x00\x00")

    with pytest.raises(OmeZarrError, match="decodes to 2 bytes"):
        read_ome_zarr_array(root)


def test_metadata_that_contradicts_the_arrays_is_refused(tmp_path):
    """The axes list and the array have to agree about how many axes there are."""
    root = tmp_path / "wrongaxes.zarr"
    _write_raw_v2_array(root / "0", np.zeros((4, 4), "<u2"), (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "z", "type": "space"}, {"name": "y", "type": "space"},
              {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0, 1.0])]))
    with pytest.raises(OmeZarrError, match="lists 3 axes"):
        read_ome_zarr(root)

    other = tmp_path / "wrongscale.zarr"
    _write_raw_v2_array(other / "0", np.zeros((4, 4), "<u2"), (2, 2))
    _write_raw_group(other, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0, 1.0])]))
    with pytest.raises(OmeZarrError, match="one value per axis"):
        read_ome_zarr(other)

    empty = tmp_path / "nodatasets.zarr"
    _write_raw_group(empty, {"multiscales": [{"version": "0.4", "axes": [],
                                              "datasets": []}]})
    with pytest.raises(OmeZarrError, match="lists no `datasets`"):
        read_ome_zarr(empty)


def test_an_unknown_coordinate_transformation_is_refused(tmp_path):
    """An unapplied affine is a wrong picture, not a missing feature."""
    root = tmp_path / "affine.zarr"
    _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[{"path": "0", "coordinateTransformations": [
            {"type": "affine", "affine": [[1, 0, 0], [0, 1, 0]]}]}]))

    with pytest.raises(OmeZarrError, match="coordinateTransformations type"):
        read_ome_zarr(root)


# ---------------------------------------------------------------------------
# 8. The vocabulary itself
# ---------------------------------------------------------------------------

def test_the_unit_table_is_a_strict_inverse_where_it_claims_to_be():
    """Every writable token reads back as itself. No lossy pair survives."""
    for token, ngff in ome_zarr.SPACR_UNIT_TO_NGFF.items():
        assert ngff_unit_to_spacr(ngff) == token
        assert spacr_unit_to_ngff(token) == ngff
    assert spacr_unit_to_ngff("px") is None
    assert ngff_unit_to_spacr(None) == "px"
    assert ngff_unit_to_spacr("") == "px"
    assert ngff_unit_to_spacr("micron") == "um"      # off-spec, tolerated
    assert ngff_unit_to_spacr("µm") == "um"
    assert spacr_unit_to_ngff("micrometer") == "micrometer"


def test_axis_records_what_the_file_said_and_writes_what_the_spec_wants():
    """A channel axis with a unit is read faithfully and written without one."""
    axis = Axis(name="c", type="channel", unit="micrometer")
    assert axis.unit == "micrometer"                  # faithful record
    assert "unit" not in axis.to_ngff()               # spec-clean output
    assert Axis.space("x", 0.65, UM).to_ngff() == {
        "name": "x", "type": "space", "unit": UM}
    assert "0.65" in Axis.space("x", 0.65, UM).describe()
    assert "pixel (undeclared)" in Axis.space("x", 0.65).describe()
    # A channel index is not a measurement, so it is not given a step.
    assert Axis.channel("c").describe() == "c: channel"
    assert "/step" not in Axis.channel("c").describe()

    with pytest.raises(OmeZarrError, match="collapses the axis"):
        Axis.space("x", 0.0)
    with pytest.raises(OmeZarrError, match="needs a name"):
        Axis(name="  ")
    with pytest.raises(OmeZarrError, match="non-finite translation"):
        Axis.space("x", 1.0, translate=float("inf"))


def test_spacing_from_axes_refuses_an_image_with_no_space_axes():
    """Two space axes is the floor; without them there is no voxel size."""
    with pytest.raises(OmeZarrError, match="no spatial axes"):
        spacing_from_axes([Axis.time(), Axis.channel()])


def test_axes_from_spacing_fills_in_the_canonical_leading_axes():
    """A (t, c, z, y, x) array written with a (z, y, x) spacing needs no help."""
    spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")

    assert [a.name for a in axes_from_spacing(spacing, 5)] == \
        ["t", "c", "z", "y", "x"]
    assert [a.name for a in axes_from_spacing(spacing, 4)] == \
        ["c", "z", "y", "x"]
    assert [a.type for a in axes_from_spacing(spacing, 5)] == \
        ["time", "channel", "space", "space", "space"]
    assert all(a.unit == UM for a in axes_from_spacing(spacing, 3))

    with pytest.raises(OmeZarrError, match="the array has 2"):
        axes_from_spacing(spacing, 2)
    with pytest.raises(OmeZarrError, match="2 axis names for a 3-dimensional"):
        axes_from_spacing(spacing, 3, names=("y", "x"))
    two_d = Spacing.from_map({"y": 1.0, "x": 1.0}, units="um")
    with pytest.raises(OmeZarrError, match="not in the spacing"):
        axes_from_spacing(two_d, 3)


def test_describe_says_when_the_spacing_cannot_be_formed():
    """A report must not raise; it must say what is wrong instead."""
    image = OmeZarrImage(
        path="/tmp/mixed.zarr",
        axes=(Axis.space("y", 1.0, UM), Axis.space("x", 1.0, "nanometer")),
        levels=(ome_zarr.Level(path="0", shape=(2, 2), chunks=(2, 2),
                               dtype="<u2", scale=(1.0, 1.0),
                               translation=(0.0, 0.0)),))
    text = image.describe()
    assert "spacing: unavailable" in text
    assert "not all in the same unit" in text


def test_an_image_whose_levels_disagree_with_its_axes_is_refused():
    """The invariant OmeZarrImage exists to hold."""
    level = ome_zarr.Level(path="0", shape=(2, 2), chunks=(2, 2), dtype="<u2",
                           scale=(1.0, 1.0), translation=(0.0, 0.0))
    with pytest.raises(OmeZarrError, match="one axis entry per array dimension"):
        OmeZarrImage(path="/tmp/x.zarr", axes=(Axis.space("x", 1.0),),
                     levels=(level,))
    with pytest.raises(OmeZarrError, match="lists no datasets"):
        OmeZarrImage(path="/tmp/x.zarr", axes=(Axis.space("x", 1.0),),
                     levels=())
    with pytest.raises(OmeZarrError, match="one per axis"):
        ome_zarr.Level(path="0", shape=(2, 2), chunks=(2, 2), dtype="<u2",
                       scale=(1.0,), translation=(0.0, 0.0))
    with pytest.raises(OmeZarrError, match="different ranks"):
        ome_zarr.Level(path="0", shape=(2, 2), chunks=(2,), dtype="<u2",
                       scale=(1.0, 1.0), translation=(0.0, 0.0))


def test_level_reports_its_own_size_without_reading_it():
    """`nbytes` and `n_chunks` are arithmetic on the metadata, nothing more."""
    level = ome_zarr.Level(path="0", shape=(4, 100, 100), chunks=(1, 32, 32),
                           dtype="<u2", scale=(1.0, 1.0, 1.0),
                           translation=(0.0, 0.0, 0.0), compressor="zlib")
    assert level.nbytes == 4 * 100 * 100 * 2
    assert level.n_chunks == 4 * 4 * 4
    assert level.ndim == 3
    assert "zlib" in level.describe()


def test_the_public_api_is_what_dunder_all_says_it_is():
    """A name in __all__ that does not exist is a broken `from ... import *`."""
    for name in ome_zarr.__all__:
        assert hasattr(ome_zarr, name), f"__all__ names a missing {name!r}"
    assert ome_zarr.__all__ == sorted(ome_zarr.__all__)


def test_spacr_records_its_own_version_in_the_multiscale_metadata(tmp_path):
    """So a pyramid written before a rule change can be told from one after."""
    image = write_ome_zarr(tmp_path / "v.zarr", np.zeros((4, 4), np.uint8))
    recorded = image.multiscale["metadata"]["version"]
    assert recorded == ome_zarr._spacr_version()
    assert recorded and recorded != "unknown"


# ---------------------------------------------------------------------------
# 9. The rest of the storage layer, one malformed fixture at a time
# ---------------------------------------------------------------------------

def _bare_group(root: Path, meta: dict, shape=(2, 2), axes=None):
    """A group whose single array's .zarray is exactly ``meta``."""
    (root / "0").mkdir(parents=True, exist_ok=True)
    (root / "0" / ".zarray").write_text(json.dumps(meta), encoding="utf-8")
    _write_raw_group(root, _multiscale(
        axes=axes or [{"name": "y", "type": "space"},
                      {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0] * len(shape))]))


def _zarray(**overrides):
    """A minimal valid zarr-format-2 .zarray, with fields overridden."""
    meta = {"zarr_format": 2, "shape": [2, 2], "chunks": [2, 2],
            "dtype": "<u2", "compressor": None, "fill_value": 0, "order": "C",
            "filters": None, "dimension_separator": "."}
    meta.update(overrides)
    return meta


@pytest.mark.parametrize("overrides,message", [
    ({"zarr_format": 3}, "zarr_format 3"),
    ({"compressor": "zlib"}, "neither null"),
    ({"order": "Z"}, "not C or F"),
    ({"dimension_separator": ":"}, "dimension_separator"),
    ({"shape": None}, "no \"shape\""),
    ({"chunks": [2]}, "differ in\nrank"),
    ({"chunks": [0, 2]}, "zero edge"),
    ({"dtype": [["a", "<u2"]]}, "structured or nested"),
    ({"dtype": "not-a-dtype"}, "not a numpy dtype"),
    ({"dtype": "|O"}, "object dtype"),
    ({"fill_value": "NaN"}, "fill_value 'NaN'"),
    ({"fill_value": "!!not base64!!"}, "nor valid base64"),
    ({"fill_value": "AAAAAA=="}, "decodes to 4 bytes"),
])
def test_a_malformed_zarray_is_refused_with_the_field_named(tmp_path,
                                                            overrides, message):
    """Each field of .zarray is read, and each is checked."""
    root = tmp_path / f"bad{abs(hash(str(overrides)))}.zarr"
    _bare_group(root, _zarray(**overrides))
    with pytest.raises(OmeZarrError, match=message.replace("\n", "\\s+")):
        read_ome_zarr(root)


def test_a_null_fill_value_means_zero_and_a_base64_one_means_its_bytes(tmp_path):
    """Both are legal zarr, and both have to survive a missing chunk."""
    root = tmp_path / "fills.zarr"
    _bare_group(root, _zarray(fill_value=None))          # no chunk written
    assert read_ome_zarr_array(root).tolist() == [[0, 0], [0, 0]]

    other = tmp_path / "b64.zarr"
    raw = np.array(513, dtype="<u2").tobytes()
    import base64 as b64
    _bare_group(other, _zarray(fill_value=b64.b64encode(raw).decode()))
    assert read_ome_zarr_array(other).tolist() == [[513, 513], [513, 513]]


def test_an_inf_fill_value_round_trips_through_json(tmp_path):
    """JSON has no infinity; the spec names it, so the writer uses the name."""
    data = np.zeros((2, 2), dtype=np.float32)
    write_ome_zarr(tmp_path / "inf.zarr", data, fill_value=float("-inf"),
                   write_empty_chunks=True)
    assert json.loads((tmp_path / "inf.zarr" / "0" / ".zarray").read_text())[
        "fill_value"] == "-Infinity"

    write_ome_zarr(tmp_path / "half.zarr", data, fill_value=0.5)
    assert json.loads((tmp_path / "half.zarr" / "0" / ".zarray").read_text())[
        "fill_value"] == 0.5


def test_a_boolean_array_round_trips(tmp_path):
    """Masks are boolean, and JSON booleans are not JSON numbers."""
    data = np.zeros((4, 4), dtype=bool)
    data[1:3, 1:3] = True
    image = write_ome_zarr(tmp_path / "mask.zarr", data, downsample="stride",
                           levels=2)
    assert json.loads((tmp_path / "mask.zarr" / "0" / ".zarray").read_text())[
        "fill_value"] is False
    assert np.array_equal(image.read(0), data)
    assert image.read(1).dtype == np.dtype(bool)


def test_a_chunk_key_that_is_a_directory_is_treated_as_absent(tmp_path):
    """A stray directory where a chunk should be is empty space, not a crash."""
    root = tmp_path / "dirchunk.zarr"
    _bare_group(root, _zarray(fill_value=42))
    (root / "0" / "0.0").mkdir(parents=True)
    assert read_ome_zarr_array(root).tolist() == [[42, 42], [42, 42]]


def test_unreadable_metadata_files_are_reported_by_path(tmp_path):
    """`.zattrs` is JSON; when it is not, say which file and why."""
    root = tmp_path / "brokenjson.zarr"
    root.mkdir()
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}),
                                  encoding="utf-8")
    (root / ".zattrs").write_text("{not json", encoding="utf-8")
    with pytest.raises(OmeZarrError, match="is not valid JSON"):
        read_ome_zarr(root)

    other = tmp_path / "listattrs.zarr"
    other.mkdir()
    (other / ".zgroup").write_text(json.dumps({"zarr_format": 2}),
                                   encoding="utf-8")
    (other / ".zattrs").write_text("[]", encoding="utf-8")
    with pytest.raises(OmeZarrError, match="holds list, not an object"):
        read_ome_zarr(other)

    empty = tmp_path / "nozarray.zarr"
    (empty / "0").mkdir(parents=True)
    _write_raw_group(empty, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])]))
    with pytest.raises(OmeZarrError, match="is not a zarr array"):
        read_ome_zarr(empty)


def test_malformed_multiscales_blocks_are_refused(tmp_path):
    """Every field of the block spaCR relies on is checked before it is used."""
    def _group(attrs, name):
        root = tmp_path / name
        _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
        _write_raw_group(root, attrs)
        return root

    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    with pytest.raises(OmeZarrError, match="not the list NGFF requires"):
        read_ome_zarr(_group({"multiscales": {"axes": axes}}, "notalist.zarr"))
    with pytest.raises(OmeZarrError, match="multiscale_index 3"):
        read_ome_zarr(_group(_multiscale(axes, [_dataset("0", [1.0, 1.0])]),
                             "oneonly.zarr"), multiscale_index=3)
    with pytest.raises(OmeZarrError, match="is not an object"):
        read_ome_zarr(_group({"multiscales": ["not an object"]}, "str.zarr"))
    with pytest.raises(OmeZarrError, match="has no `path`"):
        read_ome_zarr(_group(_multiscale(axes, [{"scale": [1.0]}]), "nop.zarr"))
    with pytest.raises(OmeZarrError, match="axes\\[0\\] is 7"):
        read_ome_zarr(_group(_multiscale([7, 8], [_dataset("0", [1.0, 1.0])]),
                             "int.zarr"))
    with pytest.raises(OmeZarrError, match="has no \"name\""):
        read_ome_zarr(_group(_multiscale([{"type": "space"}, {"name": "x"}],
                                         [_dataset("0", [1.0, 1.0])]),
                             "noname.zarr"))


def test_axes_given_as_bare_strings_and_omitted_entirely_still_open(tmp_path):
    """0.1-0.3 files have no `axes` at all; the canonical tail is what they meant."""
    root = tmp_path / "bare.zarr"
    _write_raw_v2_array(root / "0", np.zeros((2, 3, 4), "<u2"), (1, 3, 4))
    _write_raw_group(root, _multiscale(
        axes=["z", "y", "x"], datasets=[_dataset("0", [2.0, 0.5, 0.5])]))
    assert read_ome_zarr(root).spacing.axes == ("z", "y", "x")

    old = tmp_path / "noaxes.zarr"
    _write_raw_v2_array(old / "0", np.zeros((2, 3, 4), "<u2"), (1, 3, 4))
    _write_raw_group(old, {"multiscales": [
        {"version": "0.3", "datasets": [_dataset("0", [2.0, 0.5, 0.5])]}]})
    image = read_ome_zarr(old)
    assert image.axis_names == ("z", "y", "x")
    assert image.spacing.units == "px"


def test_levels_that_do_not_describe_the_same_data_are_refused(tmp_path):
    """A pyramid whose levels have different ranks is not a pyramid."""
    root = tmp_path / "ranks.zarr"
    _write_raw_v2_array(root / "0", np.zeros((4, 4), "<u2"), (2, 2))
    _write_raw_v2_array(root / "1", np.zeros((2, 2, 2), "<u2"), (1, 2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0]), _dataset("1", [2.0, 2.0])]))
    with pytest.raises(OmeZarrError, match="describes the same data"):
        read_ome_zarr(root)


def test_an_omero_block_that_is_not_an_object_is_ignored(tmp_path):
    """Rendering metadata is optional; malformed optional metadata is not fatal."""
    root = tmp_path / "badomero.zarr"
    _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
    attrs = _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[_dataset("0", [1.0, 1.0])])
    attrs["omero"] = "not an object"
    _write_raw_group(root, attrs)

    image = read_ome_zarr(root)
    assert image.channel_names == ()
    assert dict(image.omero) == {}


def test_identity_and_repeated_transformations_compose(tmp_path):
    """`identity` is legal and means nothing; two scales multiply."""
    root = tmp_path / "identity.zarr"
    _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        datasets=[{"path": "0", "coordinateTransformations": [
            {"type": "identity"},
            {"type": "scale", "scale": [2.0, 2.0]},
            {"type": "scale", "scale": [0.5, 0.5]},
            {"type": "translation", "translation": [1.0, 1.0]}]}]))
    image = read_ome_zarr(root)
    assert image.levels[0].scale == (1.0, 1.0)
    assert image.levels[0].translation == (1.0, 1.0)


def test_malformed_transformations_are_refused(tmp_path):
    """A transformation with the wrong arity is silently wrong if applied."""
    def _group(transforms, name):
        root = tmp_path / name
        _write_raw_v2_array(root / "0", np.zeros((2, 2), "<u2"), (2, 2))
        _write_raw_group(root, _multiscale(
            axes=[{"name": "y", "type": "space"},
                  {"name": "x", "type": "space"}],
            datasets=[{"path": "0", "coordinateTransformations": transforms}]))
        return root

    with pytest.raises(OmeZarrError, match="is not an object"):
        read_ome_zarr(_group(["scale"], "t1.zarr"))
    with pytest.raises(OmeZarrError, match="translation transformation needs"):
        read_ome_zarr(_group([{"type": "scale", "scale": [1.0, 1.0]},
                              {"type": "translation", "translation": [1.0]}],
                             "t2.zarr"))


def test_a_negative_scale_axis_reads_a_world_box_the_right_way_round(tmp_path):
    """A flipped axis is a real acquisition; the box still has to land on it."""
    root = tmp_path / "flip.zarr"
    array = np.arange(4 * 4, dtype="<u2").reshape(4, 4)
    _write_raw_v2_array(root / "0", array, (2, 2))
    _write_raw_group(root, _multiscale(
        axes=[{"name": "y", "type": "space", "unit": UM},
              {"name": "x", "type": "space", "unit": UM}],
        datasets=[_dataset("0", [-1.0, 1.0], [3.0, 0.0])]))

    image = read_ome_zarr(root)
    assert image.spacing.scale == (-1.0, 1.0)
    # World y from 2.0 to 3.0 is data rows 0..1 because y counts downwards.
    assert np.array_equal(image.read(0, world_region={"y": (2.0, 3.0)}),
                          array[0:1])
    # Bounds handed over in the wrong order are sorted, not refused.
    assert np.array_equal(image.read(0, world_region={"y": (3.0, 2.0)}),
                          array[0:1])


def test_a_zarr_v3_transpose_and_crc32c_chain_decodes(tmp_path):
    """Two v3 codecs the standard library can honour, and one it only strips."""
    root = tmp_path / "v3codecs.zarr"
    array = np.arange(4 * 6, dtype="<u2").reshape(4, 6)
    (root / "0").mkdir(parents=True)
    (root / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [{"name": "y", "type": "space", "unit": UM},
                     {"name": "x", "type": "space", "unit": UM}],
            "datasets": [_dataset("0", [1.0, 1.0])]}]}},
    }), encoding="utf-8")
    (root / "0" / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array",
        "shape": [4, 6], "data_type": "uint16",
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [4, 6]}},
        "chunk_key_encoding": {"name": "v2", "configuration": {"separator": "."}},
        "fill_value": 0,
        "codecs": [{"name": "transpose", "configuration": {"order": [1, 0]}},
                   {"name": "bytes", "configuration": {"endian": "big"}},
                   {"name": "crc32c"}],
    }), encoding="utf-8")
    stored = np.transpose(array, (1, 0)).astype(">u2")
    (root / "0" / "0.0").write_bytes(stored.tobytes() + b"\x00\x00\x00\x00")

    image = read_ome_zarr(root)
    assert image.dtype == np.dtype(">u2")
    assert np.array_equal(image.read(0), array)
    assert ome_zarr._strip_crc32c(b"abcdefgh") == b"abcd"


@pytest.mark.parametrize("meta,message", [
    ({"zarr_format": 2}, "declares zarr_format 2"),
    ({"zarr_format": 3, "node_type": "group"}, "not an array"),
    ({"zarr_format": 3, "chunk_grid": {"name": "rectangular"}},
     "chunk grid"),
    ({"zarr_format": 3, "chunk_key_encoding": {"name": "weird"}},
     "chunk key encoding"),
    ({"zarr_format": 3, "codecs": [{"name": "bytes",
                                    "configuration": {"endian": "middle"}}]},
     "endian"),
    ({"zarr_format": 3, "codecs": [{"name": "sharding_indexed"}]}, "sharding"),
])
def test_a_malformed_v3_array_is_refused(tmp_path, meta, message):
    """zarr-format 3 gets the same treatment: read the fields, check them."""
    root = tmp_path / f"v3bad{abs(hash(str(meta)))}.zarr"
    (root / "0").mkdir(parents=True)
    full = {"zarr_format": 3, "node_type": "array", "shape": [2, 2],
            "data_type": "uint16",
            "chunk_grid": {"name": "regular",
                           "configuration": {"chunk_shape": [2, 2]}},
            "chunk_key_encoding": {"name": "default"}, "fill_value": 0,
            "codecs": [{"name": "bytes"}]}
    full.update(meta)
    (root / "0" / "zarr.json").write_text(json.dumps(full), encoding="utf-8")
    (root / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [{"name": "y", "type": "space"},
                     {"name": "x", "type": "space"}],
            "datasets": [_dataset("0", [1.0, 1.0])]}]}}}), encoding="utf-8")
    with pytest.raises(OmeZarrError, match=message):
        read_ome_zarr(root)


def test_a_zarr_json_that_is_an_array_is_not_a_group(tmp_path):
    """Point at the group, not at one of its arrays — and be told which."""
    root = tmp_path / "arraynotgroup.zarr"
    root.mkdir()
    (root / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": [2, 2],
        "data_type": "uint16"}), encoding="utf-8")
    with pytest.raises(OmeZarrError, match="not a group"):
        read_ome_zarr(root)


# ---------------------------------------------------------------------------
# 10. The remaining accessors and writer branches
# ---------------------------------------------------------------------------

def test_the_image_accessors_answer_from_metadata_alone(tmp_path):
    """Small surface, but it is the surface a viewer actually calls."""
    data = np.zeros((2, 2, 4, 4), dtype=np.uint16)
    image = write_ome_zarr(
        tmp_path / "acc.zarr", data,
        spacing=Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um"),
        axes=(Axis.channel("c"), Axis.space("z", 2.0, UM),
              Axis.space("y", 0.65, UM), Axis.space("x", 0.65, UM)),
        levels=2, channel_names=("DAPI", "GFP"))

    assert image.shape == (2, 2, 4, 4)
    assert [a.name for a in image.space_axes] == ["z", "y", "x"]
    assert image.time_axis is None                 # no t axis in this image
    assert image.level("1") is image.levels[1]
    assert image.level(1) is image.levels[1]
    assert image.spacing_at(1).scale == (2.0, 1.3, 1.3)
    assert image.spacing_at("1").units == "um"
    assert image.dtype == np.dtype("<u2")

    plain = write_ome_zarr(tmp_path / "plain.zarr", np.zeros((4, 4), np.uint8))
    assert plain.channel_axis is None
    assert plain.other_axes == ()


def test_the_pyramid_stops_when_there_is_nothing_left_to_halve(tmp_path):
    """Asking for ten levels of a 2x2 image gets the levels that exist."""
    image = write_ome_zarr(tmp_path / "tiny.zarr", np.zeros((2, 2), np.uint8),
                           levels=10)
    assert [lv.shape for lv in image.levels] == [(2, 2), (1, 1)]

    flat = write_ome_zarr(tmp_path / "flat.zarr", np.zeros((1, 4), np.uint8),
                          levels=2)
    assert [lv.shape for lv in flat.levels] == [(1, 4), (1, 2)]


def test_a_float_pyramid_keeps_its_float_dtype(tmp_path):
    """No rounding on a float image: the mean of 0.0 and 1.0 is 0.5."""
    data = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    image = write_ome_zarr(tmp_path / "fp.zarr", data, levels=2)
    assert image.read(1).dtype == np.dtype(np.float32)
    assert image.read(1)[0, 0] == pytest.approx(0.5)


def test_explicit_downsample_axes_are_honoured(tmp_path):
    """Halving z is refused by default and available on request."""
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    spacing = Spacing.from_map({"z": 1.0, "y": 1.0, "x": 1.0}, units="um")

    default = write_ome_zarr(tmp_path / "dz0.zarr", data, spacing=spacing,
                             levels=2)
    assert default.levels[1].shape == (4, 2, 2)

    asked = write_ome_zarr(tmp_path / "dz1.zarr", data, spacing=spacing,
                           levels=2, downsample_axes=("z", "y", "x"))
    assert asked.levels[1].shape == (2, 2, 2)
    assert asked.levels[1].scale == (2.0, 2.0, 2.0)


def test_explicit_axis_objects_carry_a_time_step_through(tmp_path):
    """The reason `axes` takes Axis objects: t has a step and a unit too."""
    data = np.zeros((3, 4, 4), dtype=np.uint8)
    image = write_ome_zarr(
        tmp_path / "t.zarr", data,
        axes=(Axis.time("t", scale=30.0, unit="second"),
              Axis.space("y", 0.65, UM), Axis.space("x", 0.65, UM)))

    assert image.time_axis is not None
    assert image.time_axis.scale == 30.0
    assert image.time_axis.unit == "second"
    assert image.spacing.axes == ("y", "x")        # still space only
    assert "t: time, 30 second/step" in image.describe()

    with pytest.raises(OmeZarrError, match="4 axes for a 3-dimensional"):
        write_ome_zarr(tmp_path / "t2.zarr", data,
                       axes=(Axis.time("t"), Axis.space("z", 1.0),
                             Axis.space("y", 1.0), Axis.space("x", 1.0)))


def test_channel_metadata_errors_are_refused_before_writing(tmp_path):
    """Names and colours have to match the channel axis they describe."""
    data = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(OmeZarrError, match="1 channel names for 2 channels"):
        write_ome_zarr(tmp_path / "c1.zarr", data, axes=("c", "y", "x"),
                       channel_names=("only",))
    with pytest.raises(OmeZarrError, match="1 channel colours for 2"):
        write_ome_zarr(tmp_path / "c2.zarr", data, axes=("c", "y", "x"),
                       channel_names=("a", "b"), channel_colors=("FF0000",))
    image = write_ome_zarr(tmp_path / "c3.zarr", data, axes=("c", "y", "x"),
                           channel_names=("a", "b"),
                           channel_colors=("#ff0000", "00ff00"))
    assert [c["color"] for c in image.omero["channels"]] == ["FF0000", "00FF00"]


def test_writing_onto_a_path_that_is_a_file_is_refused(tmp_path):
    """A file where a group should be is a mistake, not something to delete."""
    target = tmp_path / "notadir.zarr"
    target.write_text("hello", encoding="utf-8")
    with pytest.raises(OmeZarrError, match="is a directory"):
        write_ome_zarr(target, np.zeros((2, 2), np.uint8))
    assert target.read_text() == "hello"


def test_an_empty_directory_is_written_into_without_complaint(tmp_path):
    """`mkdir out && write` is what everyone does; it must not need a flag."""
    target = tmp_path / "empty"
    target.mkdir()
    image = write_ome_zarr(target, np.zeros((2, 2), np.uint8))
    assert image.levels[0].shape == (2, 2)


def test_spacing_from_axes_reports_a_spacing_that_cannot_be_built():
    """spacr.layers refuses it for its own reasons; the message says whose."""
    with pytest.raises(OmeZarrError, match="do not make a usable spacing"):
        spacing_from_axes([Axis.space("x", 1.0), Axis.space("x", 2.0)])


def test_axes_from_spacing_gives_up_rather_than_inventing_axis_names():
    """There are only five canonical names; a sixth axis has to be named."""
    spacing = Spacing.from_map({"z": 1.0, "y": 1.0, "x": 1.0}, units="um")
    with pytest.raises(OmeZarrError, match="cannot name 3 extra axes"):
        axes_from_spacing(spacing, 6)


@pytest.mark.parametrize("stored", [None, "none", "stored", "raw", "null"])
def test_uncompressed_chunks_are_written_verbatim(tmp_path, stored):
    """`compressor: null` is a zarr codec too, and the fastest one to read."""
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out = tmp_path / f"stored_{stored}.zarr"
    image = write_ome_zarr(out, data, compressor=stored, chunks=(4, 4))

    assert json.loads((out / "0" / ".zarray").read_text())["compressor"] is None
    assert (out / "0" / "0" / "0").read_bytes() == data.tobytes()
    assert np.array_equal(image.read(0), data)


def test_writing_with_a_third_party_codec_refuses_the_same_way_reading_does(tmp_path):
    """The write side must not produce a file this install cannot read back."""
    with pytest.raises(ZarrExtraMissing) as excinfo:
        write_ome_zarr(tmp_path / "blosc.zarr", np.zeros((4, 4), np.uint8),
                       compressor="blosc")
    message = str(excinfo.value)
    assert "blosc" in message
    assert 'python -m pip install "spacr[zarr]"' in message


def test_the_defensive_checks_that_nothing_upstream_should_ever_reach():
    """Called directly, because "unreachable" is a claim with a shelf life."""
    with pytest.raises(OmeZarrError, match="translation transformation has"):
        ome_zarr.Level(path="0", shape=(2, 2), chunks=(2, 2), dtype="<u2",
                       scale=(1.0, 1.0), translation=(0.0,))
    with pytest.raises(OmeZarrError, match="does not exist"):
        ome_zarr._read_json(Path("/definitely/not/a/file.json"))
    with pytest.raises(OmeZarrError, match="2 axes .* for a 3-dimensional"):
        ome_zarr._validate_ngff_axes(
            (Axis.space("y", 1.0), Axis.space("x", 1.0)), (2, 2, 2))


def test_the_version_string_never_fails_a_write(monkeypatch):
    """A version is documentation; documentation must not raise."""
    import sys
    import types

    broken = types.ModuleType("spacr.version")      # no __version__ on it
    monkeypatch.setitem(sys.modules, "spacr.version", broken)
    assert ome_zarr._spacr_version() == "unknown"


def test_no_module_scope_import_of_the_optional_extra():
    """The extra must not make `import spacr.ome_zarr` fail — it is optional.

    tests/test_declared_dependencies_match_imports.py asserts this across the
    package; asserted here too because this module is the one with the
    strongest temptation to import zarr at the top.
    """
    import ast

    source = Path(ome_zarr.__file__).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module.split(".")[0]]
        else:
            continue
        assert "zarr" not in names and "numcodecs" not in names, \
            f"module-scope import of {names} would make the extra mandatory"
