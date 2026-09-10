"""Edges of the napari round trip: an empty field, and a broken import system.

Two paths the fidelity rules have to hold on even though nobody plans for
them: a labels array with no elements at all, and an environment where
``importlib`` cannot answer whether napari exists.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from spacr import napari_bridge


class _Layer:
    """The two attributes :func:`labels_from_viewer` reads off a layer."""

    _type_string = "labels"

    def __init__(self, name, data):
        self.name = name
        self.data = data


class _Viewer:
    """A duck-typed stand-in for ``napari.Viewer``."""

    def __init__(self, layers):
        self.layers = list(layers)


def test_an_empty_label_array_converts_without_inspecting_its_labels():
    """A zero-element array has no minimum and no maximum to range-check.

    The guard is skipped rather than reached with an empty array, and what
    comes back is still spaCR's mask dtype and the shape it went in with.
    """
    converted = napari_bridge.to_spacr_mask(np.zeros((0, 5), dtype=np.int32))

    assert converted.shape == (0, 5)
    assert converted.dtype == napari_bridge.MASK_DTYPE
    assert converted.size == 0


def test_an_empty_float_layer_is_taken_back_as_an_empty_uint16_mask():
    """The fractional-value check also has nothing to look at when empty."""
    viewer = _Viewer([_Layer("mask", np.zeros((0, 0), dtype=np.float32))])

    taken = napari_bridge.labels_from_viewer(viewer)

    assert taken.shape == (0, 0)
    assert taken.dtype == napari_bridge.MASK_DTYPE


def test_an_unspecced_napari_in_sys_modules_reads_as_unavailable(monkeypatch):
    """A half-imported ``napari`` makes ``find_spec`` raise, not answer.

    ``importlib.util.find_spec`` raises ``ValueError`` for a name that is in
    ``sys.modules`` with no ``__spec__`` -- which is what a partly initialised
    or hand-installed module looks like. The button that asks this question
    must grey itself out, not propagate the exception into a settings panel.
    """
    broken = types.ModuleType("napari")
    broken.__spec__ = None
    monkeypatch.setitem(sys.modules, "napari", broken)

    with pytest.raises(ValueError):
        importlib.util.find_spec("napari")

    assert napari_bridge.napari_available() is False
