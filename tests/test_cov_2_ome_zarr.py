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



def _hide(monkeypatch, *modules):
    """Make ``import <module>`` raise, whatever this machine has installed.

    The absence of an optional extra is a fact about a user's install, not
    about spaCR, so a test that merely asserts the extra is missing here is
    asserting something about the test runner -- and goes green or red as the
    environment drifts. Intercepting the import states the condition instead.
    """
    import builtins

    hidden = set(modules)
    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in hidden:
            raise ImportError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_zarr_absence_is_reported_as_false_not_as_an_exception(monkeypatch):
    """``_zarr_is_installed`` is the guard, so it must never raise itself.

    It is called in the middle of a read to decide which reader to use; an
    ImportError escaping from it would turn "no optional extra" into a failed
    read of a file the fallback can handle perfectly well.

    ``require_zarr`` is the other half: where the guard returns False quietly,
    the loader raises something that says which extra to install, because by
    then the caller has asked for a path only zarr can serve.
    """
    _hide(monkeypatch, "zarr")

    assert oz._zarr_is_installed() is False

    with pytest.raises(ZarrExtraMissing) as excinfo:
        oz.require_zarr()
    assert 'pip install "spacr[zarr]"' in str(excinfo.value)


# ---------------------------------------------------------------------------
# the writer's encoder
# ---------------------------------------------------------------------------

def test_zstd_falls_through_to_numcodecs_when_the_stdlib_has_none(monkeypatch):
    """Before Python 3.14 ``compression.zstd`` does not exist, and that is fine.

    The writer must not crash on an ImportError raised inside its own
    availability probe; it must carry on to numcodecs and, if that is absent
    too, say which extra to install.

    Both absences are stated rather than inherited from the interpreter and the
    environment: the probe is made to raise, and numcodecs is hidden. Keying
    this off ``sys.version_info`` would silently stop testing the fall-through
    on 3.14, which is exactly the version that changes the branch.
    """
    def no_stdlib_zstd():
        raise ImportError("No module named 'compression.zstd'",
                          name="compression.zstd")

    monkeypatch.setattr(oz, "_stdlib_zstd", no_stdlib_zstd)
    _hide(monkeypatch, "numcodecs")

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
    """The canonical Blosc request and its resolved defaults describe the chunks."""
    class ConfiguredBlosc(_ZlibCodec):
        def __init__(self, spec):
            assert spec == {"id": "blosc", "clevel": 4}
            super().__init__(spec)
            self.spec.update(cname="lz4", shuffle=1, blocksize=0)

    def codec(spec):
        return ConfiguredBlosc(spec) if spec["id"] == "blosc" else _ZlibCodec(spec)

    monkeypatch.setitem(sys.modules, "numcodecs", _fake_numcodecs(codec))

    block, encode = oz._encoder("blosc", 4)

    assert block == {"id": "blosc", "clevel": 4, "cname": "lz4",
                     "shuffle": 1, "blocksize": 0}
    payload = encode(b"payload")
    assert type(payload) is bytes
    assert zlib.decompress(payload) == b"payload"

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
