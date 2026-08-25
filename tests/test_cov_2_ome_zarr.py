"""The OME-Zarr paths that only exist when the optional stack IS installed.

Neither :mod:`zarr` nor :mod:`numcodecs` is installed in this environment, and
:mod:`spacr.ome_zarr` is written so that most files can be read without them.
That leaves the other half untested: what happens on the machine that HAS
them. Those branches are the ones a user with a full install runs every time,
so they are driven here against stand-in modules placed in ``sys.modules`` --
the import site is exactly what is being exercised, and a stand-in is the only
way to reach it without adding a dependency.

The same trick covers ``compression.zstd``, which is standard from Python 3.14
and absent on the interpreter this suite runs on.
"""
from __future__ import annotations

import sys
import types
import zlib

import numpy as np
import pytest

from spacr import ome_zarr as oz
from spacr.ome_zarr import OmeZarrError, ZarrExtraMissing


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------

class _RecordingArray:
    """A zarr-like array that serves ``source`` and records how it was sliced."""

    def __init__(self, source, log):
        self._source = source
        self._log = log

    def __getitem__(self, key):
        self._log.append(("getitem", key))
        return self._source[key]


def _fake_zarr(source, log):
    """A module that answers ``zarr.open(path, mode=...)`` like zarr does."""
    module = types.ModuleType("zarr")
    module.__version__ = "0.0-stand-in"

    def _open(path, mode="r"):
        log.append(("open", str(path), mode))
        return _RecordingArray(source, log)

    module.open = _open
    return module


class _ZlibCodec:
    """A numcodecs-shaped codec that really round-trips, backed by zlib."""

    def __init__(self, spec):
        self.spec = dict(spec)

    def decode(self, raw):
        # A bytearray on purpose: numcodecs returns buffers of several types
        # and the wrapper must normalise whatever it gets to bytes.
        return bytearray(zlib.decompress(bytes(raw)))

    def encode(self, raw):
        return bytearray(zlib.compress(bytes(raw)))

    def get_config(self):
        return dict(self.spec)


def _fake_numcodecs(factory):
    module = types.ModuleType("numcodecs")
    module.get_codec = factory
    return module


def _fake_stdlib_zstd():
    """A stand-in for ``compression.zstd``, which needs Python 3.14."""
    module = types.ModuleType("compression.zstd")
    module.compress = lambda data, level=3: zlib.compress(bytes(data), 6)
    module.decompress = lambda data: zlib.decompress(bytes(data))
    return module


# ---------------------------------------------------------------------------
# require_zarr
# ---------------------------------------------------------------------------

def test_require_zarr_hands_back_the_module_when_it_is_installed(monkeypatch):
    """With the extra installed the helper must return, not raise.

    Every zarr-preferring path calls this first, so a helper that raised
    anyway -- or that returned ``None`` -- would make the extra useless on the
    machine that paid to install it.
    """
    monkeypatch.setitem(sys.modules, "zarr", _fake_zarr(np.zeros((1,)), []))

    module = oz.require_zarr()

    assert module is sys.modules["zarr"]
    assert oz._zarr_is_installed() is True


def test_zarr_absence_is_reported_as_false_not_as_an_exception():
    """``_zarr_is_installed`` is the guard, so it must never raise itself.

    It is called in the middle of a read to decide which reader to use; an
    ImportError escaping from it would turn "no optional extra" into a failed
    read of a file the fallback can handle perfectly well.
    """
    assert oz._zarr_is_installed() is False
    with pytest.raises(ZarrExtraMissing):
        oz.require_zarr()


# ---------------------------------------------------------------------------
# reading through zarr
# ---------------------------------------------------------------------------

def test_a_region_read_prefers_zarr_and_hands_it_the_right_slices(monkeypatch,
                                                                  tmp_path):
    """When zarr is installed the region is translated into slices for it.

    The pure-Python reader is the fallback; zarr is the intended path because
    it handles sharding and every codec. What must not change across that
    switch is WHICH voxels come back, so the box is asserted twice: once as
    the slices zarr was handed, and once as the pixels, against the fallback
    reader on the same file.
    """
    volume = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
    oz.write_ome_zarr(tmp_path / "img.zarr", volume, axes=("z", "y", "x"))
    image = oz.read_ome_zarr(tmp_path / "img.zarr")

    region = ((1, 3), (2, 4), (0, 8))
    expected = image.read(0, region, prefer_zarr=False)

    log = []
    monkeypatch.setitem(sys.modules, "zarr", _fake_zarr(volume, log))
    through_zarr = image.read(0, region, prefer_zarr=True)

    assert [entry[0] for entry in log] == ["open", "getitem"]
    assert log[0][1].endswith("img.zarr/0"), "zarr is opened on the level, not the group"
    assert log[1][1] == (slice(1, 3), slice(2, 4), slice(0, 8))
    assert isinstance(through_zarr, np.ndarray)
    np.testing.assert_array_equal(through_zarr, expected)


# ---------------------------------------------------------------------------
# require_codec through numcodecs
# ---------------------------------------------------------------------------

def test_a_numcodecs_decoder_is_wrapped_so_it_always_returns_bytes(monkeypatch):
    """The decoder contract is ``bytes -> bytes``, whatever numcodecs returns.

    numcodecs hands back buffers of several types depending on the codec, and
    the chunk assembler slices the result as bytes. A decoder that leaked a
    bytearray or a memoryview through would work for some codecs and fail for
    others, which is the worst kind of format bug to diagnose.
    """
    monkeypatch.setitem(sys.modules, "numcodecs",
                        _fake_numcodecs(_ZlibCodec))

    decode = oz.require_codec("blosc", {"cname": "lz4", "clevel": 5})

    raw = decode(zlib.compress(b"chunk bytes"))
    assert raw == b"chunk bytes"
    assert type(raw) is bytes


def test_the_codec_spec_carries_the_id_and_the_rest_of_the_block(monkeypatch):
    """The ``compressor`` block from ``.zarray`` is passed on with its ``id``.

    numcodecs selects the codec from ``id`` and configures it from the rest.
    Dropping the config would silently build a differently-parameterised codec
    and decode the file into noise.
    """
    seen = {}

    def _factory(spec):
        seen.update(spec)
        return _ZlibCodec(spec)

    monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(_factory))

    oz.require_codec("BLOSC", {"cname": "lz4", "clevel": 5})

    assert seen == {"id": "blosc", "cname": "lz4", "clevel": 5}, \
        "the id is lower-cased and the rest of the block is kept"


def test_a_codec_numcodecs_cannot_build_says_so_with_the_spec(monkeypatch):
    """An installed numcodecs that still refuses is a different problem.

    Reporting it as "install the extra" would send the user to reinstall
    something they already have. The message names the codec and the spec that
    was rejected, which is what identifies the offending ``.zarray``.
    """
    def _factory(spec):
        raise ValueError("no such codec registered")

    monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(_factory))

    with pytest.raises(OmeZarrError) as excinfo:
        oz.require_codec("imagecodecs_jpeg", {"quality": 90})

    message = str(excinfo.value)
    assert "numcodecs is installed" in message
    assert "imagecodecs_jpeg" in message
    assert "quality" in message
    assert "no such codec registered" in message
    assert not isinstance(excinfo.value, ZarrExtraMissing)


# ---------------------------------------------------------------------------
# the writer's encoder
# ---------------------------------------------------------------------------

def test_zstd_falls_through_to_numcodecs_when_the_stdlib_has_none():
    """Before Python 3.14 ``compression.zstd`` does not exist, and that is fine.

    The writer must not crash on an ImportError raised inside its own
    availability probe; it must carry on to numcodecs and, if that is absent
    too, say which extra to install.
    """
    assert sys.version_info < (3, 14), "this asserts the pre-3.14 branch"
    with pytest.raises(ImportError):
        oz._stdlib_zstd_compress(b"", 3)

    with pytest.raises(ZarrExtraMissing) as excinfo:
        oz._encoder("zstd", 5)
    assert "zstd" in str(excinfo.value)
    assert 'pip install "spacr[zarr]"' in str(excinfo.value)


def test_zstd_uses_the_standard_library_when_the_interpreter_has_it(monkeypatch):
    """On Python 3.14 zstd needs no extra, and the block written says ``zstd``.

    The stand-in stands for ``compression.zstd``. What is being asserted is
    that the writer takes the stdlib branch rather than reaching for
    numcodecs, and that the ``compressor`` block it puts in ``.zarray`` names
    the codec and its level -- a block that named something else would produce
    a file no reader could decode.
    """
    monkeypatch.setattr(oz, "_stdlib_zstd", _fake_stdlib_zstd)
    # numcodecs stays absent, so taking its branch would raise instead.
    block, encode = oz._encoder("zstd", 7)

    assert block == {"id": "zstd", "level": 7}
    assert oz._stdlib_zstd().decompress(encode(b"payload")) == b"payload"


def test_a_numcodecs_encoder_reports_its_own_config_as_the_block(monkeypatch):
    """What goes in ``.zarray`` is the codec's config, not what we asked for.

    numcodecs normalises and completes a spec, and a reader configures itself
    from the block in the file. Writing the request rather than the resolved
    config would produce a file whose header disagrees with its chunks.
    """
    monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(_ZlibCodec))

    block, encode = oz._encoder("blosc", 4)

    assert block == {"id": "blosc", "level": 4}
    payload = encode(b"payload")
    assert type(payload) is bytes
    assert zlib.decompress(payload) == b"payload"

    # A codec that takes no level is not given one.
    other_block, _ = oz._encoder("delta", 4)
    assert other_block == {"id": "delta"}


def test_an_encoder_numcodecs_refuses_to_build_names_the_codec(monkeypatch):
    """A write must fail before it produces chunks nothing can read.

    The same distinction as on the read side: numcodecs is here, and it still
    said no, so the message points at the codec name rather than at the
    install.
    """
    def _factory(spec):
        raise RuntimeError("delta needs a dtype")

    monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(_factory))

    with pytest.raises(OmeZarrError) as excinfo:
        oz._encoder("delta", 4)

    message = str(excinfo.value)
    assert "numcodecs is installed" in message
    assert "delta" in message
    assert "delta needs a dtype" in message
