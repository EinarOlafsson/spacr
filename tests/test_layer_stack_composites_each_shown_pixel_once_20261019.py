"""The layer composite is made once per data pixel shown, with the same bytes.

Curate (lesson 42) composited a 256 px mask at the canvas's 1500 px, about
150 ms a dab. These tests hold the faster paths to the old output byte for
byte and count the composited pixels instead of timing them.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import layers as L
from spacr.layers import Blending, Canvas, LayerStack, Spacing


def _old(stack, canvas):
    return np.clip(stack.render(canvas) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _old_label_colors(labels, seed=0):
    arr = np.asarray(labels)
    if arr.size == 0:
        return np.zeros(arr.shape + (3,), dtype=np.float32)
    unique = np.unique(arr)
    lut = np.array([L.label_color(int(u), seed=seed) for u in unique],
                   dtype=np.float32)
    return lut[np.searchsorted(unique, arr)]


def _stack(shape=(48, 40), spacing=None, opacity=0.6, blending="translucent",
           seed=0, channels=1):
    rng = np.random.default_rng(seed)
    spacing = spacing or Spacing.isotropic(len(shape), 0.5, units="um")
    stack = LayerStack()
    data = rng.integers(0, 3000, (channels,) + shape if channels > 1 else shape)
    kwargs = dict(channel_axis=0) if channels > 1 else {}
    stack.add_image(data.astype(np.uint16), name="image", spacing=spacing, **kwargs)
    labels = rng.integers(0, 9, shape) * (rng.random(shape) > 0.4)
    layer = stack.add_labels(labels, name="mask", spacing=spacing, seed=seed)
    layer.opacity = opacity
    layer.blending = blending
    return stack


def _canvases(stack):
    base = Canvas.covering(stack, height=97, width=131, margin=0.05)
    yield base
    yield base.zoomed(3.7, centre=(20.0, 30.0))
    yield base.zoomed(0.4)
    yield base.panned(-40.0, 55.0)
    yield base.resized(13, 300)
    yield base.zoomed(11.0, centre=(1.0, 1.0)).panned(-3.0, -2.0)


@pytest.mark.parametrize("blending", list(Blending.ALL))
@pytest.mark.parametrize("opacity", [1.0, 0.35])
def test_the_gathered_composite_is_the_old_bytes(blending, opacity):
    stack = _stack(opacity=opacity, blending=blending)
    for canvas in _canvases(stack):
        assert stack._render_uint8_gathered(canvas) is not None
        assert np.array_equal(stack.render_uint8(canvas), _old(stack, canvas))


def test_a_volume_slice_and_side_views_are_the_old_bytes():
    spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
    stack = _stack(shape=(7, 30, 26), spacing=spacing)
    for axes in (("y", "x"), ("z", "x"), ("z", "y")):
        canvas = Canvas.covering(stack, height=64, width=80, axes=axes)
        for depth in (0.0, 4.1, 6.0, 99.0):
            moved = canvas.at_depth(**{a: depth for a in ("z", "y", "x")
                                       if a not in axes})
            assert np.array_equal(stack.render_uint8(moved), _old(stack, moved))


def test_multichannel_images_and_hidden_layers_are_the_old_bytes():
    stack = _stack(channels=3)
    stack["image"].set_channel_visible(1, False)
    for canvas in _canvases(stack):
        assert np.array_equal(stack.render_uint8(canvas), _old(stack, canvas))
    stack["mask"].visible = False
    for canvas in _canvases(stack):
        assert np.array_equal(stack.render_uint8(canvas), _old(stack, canvas))


def test_layers_on_different_grids_take_the_old_path():
    stack = _stack()
    stack.add_labels(np.ones((10, 10), np.int64), name="coarse",
                     spacing=Spacing.isotropic(2, 2.0, units="um"))
    canvas = Canvas.covering(stack, height=50, width=60)
    assert stack._render_uint8_gathered(canvas) is None
    assert np.array_equal(stack.render_uint8(canvas), _old(stack, canvas))


def test_each_shown_data_pixel_is_composited_once(monkeypatch):
    stack = _stack(shape=(32, 32))
    sizes = []
    original = Blending.apply

    def counted(dst, *args, **kwargs):
        sizes.append(dst.shape[0] * dst.shape[1])
        return original(dst, *args, **kwargs)
    monkeypatch.setattr(Blending, "apply", staticmethod(counted))
    canvas = Canvas.covering(stack, height=900, width=1400)
    stack.render_uint8(canvas)
    assert sizes and max(sizes) <= 32 * 32


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_label_colours_by_table_are_the_old_colours(dtype):
    rng = np.random.default_rng(4)
    labels = (rng.integers(0, 300, (50, 70)) * (rng.random((50, 70)) > .5)).astype(dtype)
    for seed in (0, 7):
        got = L.label_colors(labels, seed=seed)
        assert got.dtype == np.float32
        assert np.array_equal(got, _old_label_colors(labels, seed))


def test_label_colours_outside_the_table_fall_back():
    labels = np.array([[-3, 0, 5], [2 ** 40, 7, 0]], dtype=np.int64)
    assert np.array_equal(L.label_colors(labels), _old_label_colors(labels))
    assert L._dense_label_colors(labels, 0) is None
    floats = np.array([[0.0, 2.0]])
    assert np.array_equal(L.label_colors(floats), _old_label_colors(floats))


def test_the_separable_sample_is_the_broadcast_one():
    spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
    stack = _stack(shape=(5, 20, 18), spacing=spacing)
    layer = stack["mask"]
    canvas = Canvas.covering(stack, height=41, width=33, axes=("z", "x"))
    canvas = canvas.at_depth(y=3.0).panned(-6.0, 4.0)
    index, valid = layer._sample_index(canvas)
    vectors = layer._sample_vectors(canvas)
    h, w = canvas.shape
    shapes = {0: (h, 1), 1: (1, w)}
    for got, (role, idx, _ok) in zip(index, vectors):
        want = np.broadcast_to(np.reshape(idx, shapes.get(role, ())), (h, w))
        assert np.array_equal(got, want)
    ok = np.ones((h, w), bool)
    for role, _idx, flags in vectors:
        ok &= np.broadcast_to(np.reshape(flags, shapes.get(role, ())), (h, w))
    assert np.array_equal(valid, ok)
