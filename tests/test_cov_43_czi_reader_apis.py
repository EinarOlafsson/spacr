"""Both czifile APIs, whichever one this environment happens to install.

czifile 2026 exposes ``CziFile.scenes`` (one image per scene with its own
sizes); czifile 2019 exposes one array with an ``S`` axis and a subblock
directory. spaCR reads both, and a test environment only ever installs one,
so the other API is driven here through a stand-in with exactly the
attributes spaCR reads. The real-file acceptance for each API stays in
tests/test_convert.py.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from spacr import convert as cv


class _Scene:
    def __init__(self, pixels, axes):
        self._pixels = pixels
        self.axes = axes
        self.shape = pixels.shape
        self.dtype = pixels.dtype
        self.sizes = dict(zip(axes, pixels.shape))

    def asarray(self):
        return self._pixels


class _Handle:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _reader(monkeypatch, handle):
    module = types.SimpleNamespace(CziFile=lambda path: handle)
    monkeypatch.setattr(cv, "_import_reader", lambda ext: module)


def _scenes():
    first = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    second = np.arange(5 * 6, dtype=np.uint16).reshape(1, 5, 6) + 100
    return {0: _Scene(first, "CYX"), 3: _Scene(second, "CYX")}


def test_current_czifile_scenes_are_described_and_read_by_scene(monkeypatch):
    scenes = _scenes()
    _reader(monkeypatch, _Handle(scenes=scenes))
    described = cv._describe_czi("plate.czi")
    assert described["n_series"] == 2
    assert [s["czi_scene"] for s in described["per_series"]] == [0, 3]
    assert [s["n_c"] for s in described["per_series"]] == [2, 1]
    by_index = cv.SourceImage(path="plate.czi", plate="p", well="w",
                              field="f", meta={"series": 1}, n_channels=1)
    data = cv._read_czi(by_index)
    assert data.max() == scenes[3].asarray().max()
    by_scene = cv.SourceImage(path="plate.czi", plate="p", well="w",
                              field="f", meta={"czi_scene": 0},
                              n_channels=2)
    assert cv._read_czi(by_scene).max() == scenes[0].asarray().max()


def test_a_czi_with_no_scenes_or_no_subblocks_contains_no_images(
        monkeypatch):
    from spacr.errors import ConfigurationError

    _reader(monkeypatch, _Handle(scenes={}))
    with pytest.raises(ConfigurationError, match="contains no images"):
        cv._describe_czi("empty.czi")
    _reader(monkeypatch, _Handle(filtered_subblock_directory=[]))
    monkeypatch.setattr(cv, "_legacy_czi_series", lambda handle: [])
    with pytest.raises(ConfigurationError, match="contains no images"):
        cv._describe_czi("empty.czi")


def test_a_legacy_scene_keeps_the_axes_its_bounds_do_not_name(monkeypatch):
    pixels = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    handle = _Handle(asarray=lambda: pixels, axes="SCYX", start=(0, 0, 0, 0))
    _reader(monkeypatch, handle)
    source = cv.SourceImage(
        path="legacy.czi", plate="p", well="w", field="f", n_channels=3,
        meta={"czi_scene": 1, "czi_bounds": {"Y": (0, 2), "X": (1, 5)}})
    data = cv._read_czi(source)
    assert data.shape[-2:] == (2, 4)
    assert data.max() == pixels[1, :, :2, 1:5].max()
