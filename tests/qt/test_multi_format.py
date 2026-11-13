"""Real-file tests for :mod:`spacr.qt.multi_format`.

Every accepted single-file dataset extension is exercised against a file
actually written to disk (npz / npy / single- and multi-page tif), plus the
rejection paths: unsupported extension, a directory, a path that does not
exist, and a file whose bytes are garbage for its extension.

``.lif`` / ``.nd2`` need vendor hardware formats we cannot synthesise, so the
vendor *reader* (a true externality) is substituted while the product's own
shape/metadata mapping — the thing under test — runs for real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt.multi_format import (
    DatasetDescription, describe_file, _describe_lif, _describe_nd2,
)


# ---------------------------------------------------------------------------
# DatasetDescription.summary
# ---------------------------------------------------------------------------

def test_summary_minimal_has_only_format_fields_channels():
    d = DatasetDescription(path=Path("/x/y.npz"), kind="npz")
    assert d.summary() == "format=npz  fields=1  channels=1"


def test_summary_includes_t_z_shape_dtype_and_notes():
    d = DatasetDescription(
        path=Path("/x/y.lif"), kind="lif", n_fields=3, n_channels=2,
        n_timepoints=7, n_slices=5, shape=(64, 48), dtype="uint16",
        notes=["series=['a']", "extra"],
    )
    assert d.summary() == (
        "format=lif  fields=3  channels=2  T=7  Z=5  "
        "HxW=64x48  dtype=uint16  · series=['a']; extra"
    )


def test_summary_omits_t_and_z_when_singleton():
    d = DatasetDescription(path=Path("/x/y.npy"), kind="npy",
                           n_timepoints=1, n_slices=1, shape=(4, 5))
    s = d.summary()
    assert "T=" not in s and "Z=" not in s
    assert s.endswith("HxW=4x5")


# ---------------------------------------------------------------------------
# describe_file — rejection paths
# ---------------------------------------------------------------------------

def test_describe_file_returns_none_for_missing_path(tmp_path):
    assert describe_file(tmp_path / "nope.npz") is None


def test_describe_file_returns_none_for_directory(tmp_path):
    d = tmp_path / "a_folder.npz"        # directory that LOOKS like a dataset
    d.mkdir()
    assert describe_file(d) is None


@pytest.mark.parametrize("name", ["notes.txt", "reads.fastq", "img.png",
                                  "table.csv", "noext"])
def test_describe_file_returns_none_for_unsupported_extension(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"whatever")
    assert describe_file(p) is None


def test_describe_file_accepts_str_path(tmp_path):
    p = tmp_path / "one.npy"
    np.save(p, np.zeros((6, 7), np.uint8))
    d = describe_file(str(p))
    assert d is not None and d.kind == "npy" and d.shape == (6, 7)


# ---------------------------------------------------------------------------
# .npz
# ---------------------------------------------------------------------------

def test_npz_2d_single_array(tmp_path):
    p = tmp_path / "flat.npz"
    np.savez(p, plane=np.zeros((32, 40), np.uint16))
    d = describe_file(p)
    assert d.kind == "npz"
    assert (d.n_fields, d.n_channels) == (1, 1)
    assert d.shape == (32, 40)
    assert d.dtype == "uint16"
    assert d.notes == ["arrays=['plane']"]


def test_npz_4d_is_fields_h_w_channels(tmp_path):
    p = tmp_path / "stack.npz"
    np.savez(p, cube=np.zeros((6, 30, 25, 3), np.float32))
    d = describe_file(p)
    assert (d.n_fields, d.n_channels) == (6, 3)
    assert d.shape == (30, 25)
    assert d.dtype == "float32"


@pytest.mark.parametrize(
    "shape, fields_channels, plane",
    [
        ((5, 40, 50), (5, 1), (40, 50)),    # 5 < 20, 40/50 > 20
        ((40, 50, 3), (1, 3), (40, 50)),    # 40 not < 20
    ],
    ids=["small_leading_axis_is_fields", "big_leading_axis_is_h_w_channels"],
)
def test_npz_3d_leading_axis_reads_as_fields_or_h_w_channels(
        tmp_path, shape, fields_channels, plane):
    p = tmp_path / "a.npz"
    np.savez(p, a=np.zeros(shape, np.uint8))
    d = describe_file(p)
    assert (d.n_fields, d.n_channels) == fields_channels
    assert d.shape == plane


def test_npz_1d_array_leaves_shape_unknown(tmp_path):
    p = tmp_path / "vec.npz"
    np.savez(p, v=np.arange(9, dtype=np.int64))
    d = describe_file(p)
    assert d.shape is None
    assert (d.n_fields, d.n_channels) == (1, 1)
    assert d.dtype == "int64"


@pytest.mark.parametrize(
    "n_keys, prefix, notes",
    [
        # 1 field from shape, 4 from keys
        (4, "f", ["arrays=['f0', 'f1', 'f2', 'f3']"]),
        (7, "k", ["arrays=['k0', 'k1', 'k2', 'k3', 'k4']…"]),
    ],
    ids=["four_keys_listed", "more_than_five_keys_truncated_with_ellipsis"],
)
def test_npz_key_count_raises_n_fields(tmp_path, n_keys, prefix, notes):
    p = tmp_path / "many.npz"
    np.savez(p, **{f"{prefix}{i}": np.zeros((8, 9), np.uint8)
                   for i in range(n_keys)})
    d = describe_file(p)
    assert d.n_fields == n_keys
    assert d.notes == notes


def test_npz_keeps_larger_shape_derived_field_count(tmp_path):
    """Two keys but a 9-field cube → the shape wins (max of the two)."""
    p = tmp_path / "cube2.npz"
    np.savez(p, a=np.zeros((9, 30, 30), np.uint8),
             b=np.zeros((9, 30, 30), np.uint8))
    d = describe_file(p)
    assert d.n_fields == 9


def test_npz_empty_archive_returns_none(tmp_path):
    p = tmp_path / "empty.npz"
    np.savez(p)
    assert describe_file(p) is None


@pytest.mark.parametrize(
    "name, payload",
    [
        ("broken.npz", b"this is definitely not a zip archive"),
        ("broken.npy", b"not a numpy header at all"),
        # Valid TIFF magic, nonsense page offset -> tifffile reports 0 pages.
        ("broken.tif", b"II*\x00garbage-not-really-a-tiff"),
        # No TIFF magic -> tifffile raises; the describer must swallow it.
        ("fake.tif", b"hello world, this is plain text with a .tif name"),
    ],
    ids=["npz_garbage", "npy_garbage", "tif_truncated_page_table",
         "tif_not_a_tiff_at_all"],
)
def test_garbage_bytes_return_none(tmp_path, name, payload):
    p = tmp_path / name
    p.write_bytes(payload)
    assert describe_file(p) is None


# ---------------------------------------------------------------------------
# .npy
# ---------------------------------------------------------------------------

def test_npy_2d(tmp_path):
    p = tmp_path / "plane.npy"
    np.save(p, np.zeros((12, 13), np.uint16))
    d = describe_file(p)
    assert d.kind == "npy"
    assert d.shape == (12, 13)
    assert (d.n_fields, d.n_channels) == (1, 1)
    assert d.dtype == "uint16"
    assert d.notes == ["npy_shape=(12, 13)"]


def test_npy_4d(tmp_path):
    p = tmp_path / "cube.npy"
    np.save(p, np.zeros((4, 20, 21, 2), np.uint8))
    d = describe_file(p)
    assert (d.n_fields, d.n_channels) == (4, 2)
    assert d.shape == (20, 21)


@pytest.mark.parametrize(
    "shape, fields_channels, plane",
    [
        ((3, 44, 45), (3, 1), (44, 45)),
        ((44, 45, 3), (1, 3), (44, 45)),
    ],
    ids=["small_leading_axis_is_fields", "big_leading_axis_is_h_w_channels"],
)
def test_npy_3d_leading_axis_reads_as_fields_or_h_w_channels(
        tmp_path, shape, fields_channels, plane):
    p = tmp_path / "a.npy"
    np.save(p, np.zeros(shape, np.uint8))
    d = describe_file(p)
    assert (d.n_fields, d.n_channels) == fields_channels
    assert d.shape == plane


def test_npy_scalar_leaves_shape_unknown(tmp_path):
    p = tmp_path / "scalar.npy"
    np.save(p, np.float64(3.5))
    d = describe_file(p)
    assert d.shape is None
    assert d.dtype == "float64"
    assert d.notes == ["npy_shape=()"]


def test_npy_does_not_load_pixels_into_memory(tmp_path):
    """The describer must mmap, not read — a big array stays cheap."""
    p = tmp_path / "big.npy"
    np.save(p, np.zeros((2, 512, 512), np.uint16))
    d = describe_file(p)
    assert d.shape == (512, 512) and d.n_fields == 2


# ---------------------------------------------------------------------------
# .tif / .tiff
# ---------------------------------------------------------------------------

def test_single_page_tif_is_not_a_multi_format_dataset(tmp_path):
    p = tmp_path / "one.tif"
    tifffile.imwrite(str(p), np.zeros((16, 17), np.uint16))
    assert describe_file(p) is None


def test_multipage_tif_counts_pages_and_reports_axes(tmp_path):
    p = tmp_path / "stack.tif"
    tifffile.imwrite(str(p), np.zeros((5, 16, 17), np.uint16))
    d = describe_file(p)
    assert d.kind == "tif_multi"
    assert d.n_fields == 5
    assert d.shape == (16, 17)
    assert d.dtype == "uint16"
    assert d.notes[0] == "pages=5"
    # Regression: axis detection used to probe attributes TiffFile never has,
    # so the axes note was never emitted for ANY file.
    assert d.notes[1] == "axes=QYX"


def test_multipage_imagej_tif_reports_real_zc_axes(tmp_path):
    p = tmp_path / "ij.tiff"
    tifffile.imwrite(str(p), np.zeros((3, 2, 16, 17), np.uint16), imagej=True)
    d = describe_file(p)
    assert d.kind == "tif_multi"
    assert d.n_fields == 6                     # 3 Z x 2 C pages
    assert d.notes == ["pages=6", "axes=ZCYX"]


def test_tiff_extension_variant_is_accepted(tmp_path):
    p = tmp_path / "stack.TIFF"
    tifffile.imwrite(str(p), np.zeros((2, 8, 9), np.uint8))
    d = describe_file(p)
    assert d is not None and d.kind == "tif_multi" and d.n_fields == 2


# ---------------------------------------------------------------------------
# .lif — vendor reader substituted, product mapping runs for real
# ---------------------------------------------------------------------------

class _Dims:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeLifImage:
    def __init__(self, name, channels=1, nz=1, nt=1, x=64, y=48):
        self.name = name
        self.channels = channels
        self.nz = nz
        self.nt = nt
        self.dims = _Dims(x, y)


def _install_fake_lif(monkeypatch, images):
    class _FakeLifFile:
        def __init__(self, path):
            self.path = path

        def get_iter_image(self):
            return iter(images)

    monkeypatch.setattr("readlif.reader.LifFile", _FakeLifFile)


def test_lif_maps_series_channels_z_and_t(tmp_path, monkeypatch):
    p = tmp_path / "plate.lif"
    p.write_bytes(b"\x70\x00\x00\x00")
    _install_fake_lif(monkeypatch, [
        _FakeLifImage("Series001", channels=3, nz=7, nt=4, x=512, y=256),
        _FakeLifImage("Series002", channels=3, nz=7, nt=4),
    ])
    d = describe_file(p)
    assert d.kind == "lif"
    assert d.n_fields == 2
    assert d.n_channels == 3
    assert d.n_slices == 7
    assert d.n_timepoints == 4
    assert d.shape == (256, 512)            # (y, x) — H then W
    assert d.notes == ["series=['Series001', 'Series002']"]


def test_lif_more_than_five_series_truncates_note(tmp_path, monkeypatch):
    p = tmp_path / "big.lif"
    p.write_bytes(b"\x70\x00\x00\x00")
    _install_fake_lif(monkeypatch,
                      [_FakeLifImage(f"S{i}") for i in range(6)])
    d = describe_file(p)
    assert d.n_fields == 6
    assert d.notes == ["series=['S0', 'S1', 'S2', 'S3', 'S4']…"]


def test_lif_with_no_series_returns_none(tmp_path, monkeypatch):
    p = tmp_path / "empty.lif"
    p.write_bytes(b"\x70\x00\x00\x00")
    _install_fake_lif(monkeypatch, [])
    assert describe_file(p) is None


def test_lif_coerces_zero_and_none_attributes_to_one(tmp_path, monkeypatch):
    p = tmp_path / "sparse.lif"
    p.write_bytes(b"\x70\x00\x00\x00")
    img = _FakeLifImage("S", channels=0, nz=None, nt=0)
    _install_fake_lif(monkeypatch, [img])
    d = describe_file(p)
    assert (d.n_channels, d.n_slices, d.n_timepoints) == (1, 1, 1)


def test_lif_unreadable_file_returns_none_via_real_reader(tmp_path):
    """No monkeypatch: the genuine readlif reader rejects these bytes."""
    p = tmp_path / "junk.lif"
    p.write_bytes(b"definitely not a Leica image file")
    assert _describe_lif(p) is None


# ---------------------------------------------------------------------------
# .nd2 — vendor reader substituted, product mapping runs for real
# ---------------------------------------------------------------------------

def _install_fake_nd2(monkeypatch, sizes, axes):
    class _FakeND2:
        def __init__(self, path):
            self.path = path
            self.sizes = sizes
            self.axes = axes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("nd2reader.ND2Reader", _FakeND2)


def test_nd2_maps_v_c_t_z_sizes(tmp_path, monkeypatch):
    p = tmp_path / "run.nd2"
    p.write_bytes(b"\x00\x00\x00\x00")
    _install_fake_nd2(monkeypatch,
                      {"x": 1024, "y": 768, "v": 9, "c": 2, "t": 30, "z": 11},
                      ["x", "y", "c", "t", "z", "v"])
    d = describe_file(p)
    assert d.kind == "nd2"
    assert d.n_fields == 9
    assert d.n_channels == 2
    assert d.n_timepoints == 30
    assert d.n_slices == 11
    assert d.shape == (768, 1024)
    assert d.notes == ["axes=['x', 'y', 'c', 't', 'z', 'v']"]


def test_nd2_missing_axes_default_to_one(tmp_path, monkeypatch):
    p = tmp_path / "min.nd2"
    p.write_bytes(b"\x00\x00\x00\x00")
    _install_fake_nd2(monkeypatch, {"x": 100, "y": 200}, ["x", "y"])
    d = describe_file(p)
    assert (d.n_fields, d.n_channels, d.n_timepoints, d.n_slices) == (1, 1, 1, 1)
    assert d.shape == (200, 100)


def test_nd2_without_xy_leaves_shape_none(tmp_path, monkeypatch):
    p = tmp_path / "noxy.nd2"
    p.write_bytes(b"\x00\x00\x00\x00")
    _install_fake_nd2(monkeypatch, {"c": 2}, ["c"])
    d = describe_file(p)
    assert d.shape is None
    assert d.n_channels == 2


def test_nd2_unreadable_file_returns_none_via_real_reader(tmp_path):
    """No monkeypatch: the genuine nd2reader rejects these bytes."""
    p = tmp_path / "junk.nd2"
    p.write_bytes(b"definitely not a Nikon ND2 file")
    assert _describe_nd2(p) is None
