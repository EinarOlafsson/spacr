"""Chunk codecs and the writer in ``spacr.ome_zarr``, with the full stack.

``require_codec`` turns a zarr ``compressor`` block into a ``bytes -> bytes``
decoder (standard library first, then numcodecs), and ``write_ome_zarr``
records the exact block it encoded with.  Both are pinned against stand-in
numcodecs/zarr modules in ``sys.modules`` (to reach each import site on any
install) and against the real numcodecs when it is installed: the written
block must reopen the exact pixels.
"""
from __future__ import annotations

import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

from spacr import ome_zarr as oz
from spacr.ome_zarr import OmeZarrError, ZarrExtraMissing


class _ZlibCodec:
    """A numcodecs-shaped codec backed by zlib; decodes to a bytearray."""

    def __init__(self, spec):
        self.spec = dict(spec)

    def decode(self, raw):
        return bytearray(zlib.decompress(bytes(raw)))


def _fake_numcodecs(factory):
    """A ``numcodecs`` module whose ``get_codec`` is ``factory``."""
    module = types.ModuleType("numcodecs")
    module.get_codec = factory
    return module


def _hide(monkeypatch, *modules):
    """Make ``import <module>`` raise ImportError whatever is installed."""
    import builtins

    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in modules:
            raise ImportError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_require_codec_builds_a_bytes_decoder_or_names_what_is_missing(
        subtests, monkeypatch):
    """zlib comes from the standard library; any other id is built by
    numcodecs from the lower-cased id plus the rest of the block and always
    decodes to ``bytes``; an empty id, a missing numcodecs and a codec
    numcodecs refuses each raise a message naming the codec."""
    payload = zlib.compress(b"chunk bytes")

    with subtests.test(msg="standard-library zlib"):
        _hide(monkeypatch, "numcodecs")
        assert oz.require_codec("zlib")(payload) == b"chunk bytes"
        with pytest.raises(ZarrExtraMissing, match="'blosc'"):
            oz.require_codec("blosc", {"cname": "lz4"})
        monkeypatch.undo()

    for case_id, codec_id in (("empty", ""), ("blank", "  "), ("not text", 7)):
        with subtests.test(msg=f"{case_id} id"):
            with pytest.raises(OmeZarrError, match="a chunk codec needs a name"):
                oz.require_codec(codec_id)

    with subtests.test(msg="numcodecs decoder, spec and bytes"):
        seen = []

        def factory(spec):
            seen.append(dict(spec))
            return _ZlibCodec(spec)

        monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(factory))
        decode = oz.require_codec(" BLOSC ", {"cname": "lz4", "clevel": 5})
        raw = decode(payload)
        assert raw == b"chunk bytes" and type(raw) is bytes
        assert seen == [{"id": "blosc", "cname": "lz4", "clevel": 5}]

    with subtests.test(msg="numcodecs refuses the codec"):
        def refusing(spec):
            raise ValueError("no such codec registered")

        monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(refusing))
        with pytest.raises(OmeZarrError) as raised:
            oz.require_codec("imagecodecs_jpeg", {"quality": 90})
        assert str(raised.value) == (
            "numcodecs is installed but could not build the "
            "'imagecodecs_jpeg' codec from {'quality': 90, 'id': "
            "'imagecodecs_jpeg'}: no such codec registered")
        assert not isinstance(raised.value, ZarrExtraMissing)


def test_write_ome_zarr_records_a_block_that_reopens_the_exact_pixels(
        tmp_path, subtests, monkeypatch):
    """A region read through zarr gets the right slices and the same voxels
    as the pure-Python reader; real blosc and numcodecs-zstd blocks (the
    latter with no standard-library zstd) are what numcodecs itself reports
    and decode back to the written uint16 pixels."""
    with subtests.test(msg="region read through zarr"):
        volume = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
        oz.write_ome_zarr(tmp_path / "img.zarr", volume, axes=("z", "y", "x"))
        image = oz.read_ome_zarr(tmp_path / "img.zarr")
        region = ((1, 3), (2, 4), (0, 8))
        expected = image.read(0, region, prefer_zarr=False)
        np.testing.assert_array_equal(expected, volume[1:3, 2:4, 0:8])

        log = []

        class _Array:
            def __getitem__(self, key):
                log.append(("getitem", key))
                return volume[key]

        zarr = types.ModuleType("zarr")
        zarr.__version__ = "0.0-stand-in"
        zarr.open = lambda path, mode="r": (log.append(("open", str(path),
                                                        mode)) or _Array())
        monkeypatch.setitem(sys.modules, "zarr", zarr)
        through_zarr = image.read(0, region, prefer_zarr=True)
        monkeypatch.undo()
        assert [entry[0] for entry in log] == ["open", "getitem"]
        assert Path(log[0][1]).as_posix().endswith("img.zarr/0")
        assert log[0][2] == "r"
        assert log[1][1] == (slice(1, 3), slice(2, 4), slice(0, 8))
        assert type(through_zarr) is np.ndarray
        np.testing.assert_array_equal(through_zarr, expected)

    numcodecs = pytest.importorskip("numcodecs")
    blocky = np.zeros((32, 48), dtype=np.uint16)
    blocky[2:18, 4:27] = 4095
    blocky[18:29, 27:45] = 65535
    ramp = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    cases = [
        ("blosc level 4", "blosc", 4, blocky, False,
         numcodecs.Blosc(clevel=4).get_config()),
        ("zstd level 7 without stdlib zstd", "zstd", 7, ramp, True,
         numcodecs.Zstd(level=7).get_config()),
    ]
    for case_id, codec, level, data, no_stdlib, block in cases:
        with subtests.test(msg=case_id):
            if no_stdlib:
                def unavailable():
                    raise ImportError("No compression.zstd",
                                      name="compression.zstd")

                monkeypatch.setattr(oz, "_stdlib_zstd", unavailable)
            before = data.copy()
            folder = tmp_path / f"{codec}.ome.zarr"
            written = oz.write_ome_zarr(folder, data, compressor=codec,
                                        compression_level=level,
                                        chunks=data.shape,
                                        dimension_separator=".")
            metadata = json.loads((folder / "0" / ".zarray").read_text())
            assert metadata["compressor"] == block
            chunk = (folder / "0" / "0.0").read_bytes()
            assert len(chunk) < data.nbytes
            decoded = numcodecs.get_codec(metadata["compressor"]).decode(chunk)
            np.testing.assert_array_equal(
                np.frombuffer(decoded, dtype=data.dtype).reshape(data.shape),
                data)
            np.testing.assert_array_equal(written.read(), data)
            np.testing.assert_array_equal(oz.read_ome_zarr(folder).read(), data)
            np.testing.assert_array_equal(data, before)
            monkeypatch.undo()
