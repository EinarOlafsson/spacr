"""The once-per-data-pixel composite declines or simplifies in the cases the other tests skip."""
from __future__ import annotations

import numpy as np

from spacr.layers import Canvas, LayerStack, Spacing


def _old(stack, canvas):
    return np.clip(stack.render(canvas) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _flat_stack(shape=(20, 16), **kwargs):
    rng = np.random.default_rng(3)
    spacing = Spacing.isotropic(2, 0.5, units="um")
    stack = LayerStack()
    stack.add_image(rng.integers(0, 3000, shape).astype(np.uint16), name="image",
                    spacing=spacing)
    stack.add_labels(rng.integers(0, 6, shape), name="mask", spacing=spacing)
    return stack


def test_a_stack_with_nothing_shown_is_left_to_the_plain_renderer():
    stack = _flat_stack()
    canvas = Canvas.covering(stack, height=30, width=24)
    for layer in list(stack):
        layer.visible = False
    assert stack._render_uint8_gathered(canvas) is None
    assert not stack.render_uint8(canvas).any()
    empty = LayerStack()
    assert empty._render_uint8_gathered(canvas) is None


def test_a_layer_that_is_not_an_image_or_labels_is_left_to_the_plain_renderer():
    stack = _flat_stack()
    canvas = Canvas.covering(stack, height=30, width=24)
    layer = stack["mask"]
    original = type(layer)._draw

    class Odd(type(layer)):
        def _draw(self, *args, **kwargs):
            return original(self, *args, **kwargs)

    layer.__class__ = Odd
    assert stack._render_uint8_gathered(canvas) is None
    assert np.array_equal(stack.render_uint8(canvas), _old(stack, canvas))


def test_same_shape_layers_that_sample_different_pixels_are_left_to_the_plain_renderer():
    rng = np.random.default_rng(4)
    stack = LayerStack()
    stack.add_image(rng.integers(0, 3000, (20, 16)).astype(np.uint16), name="a",
                    spacing=Spacing.isotropic(2, 0.5, units="um"))
    stack.add_labels(rng.integers(0, 6, (20, 16)), name="b",
                     spacing=Spacing(axes=("y", "x"), scale=(0.5, 0.5),
                                     translate=(1.5, 0.0), units="um"))
    canvas = Canvas.covering({"y": (0.0, 10.0), "x": (0.0, 8.0)}, height=30, width=24)
    assert stack._render_uint8_gathered(canvas) is None
    assert np.array_equal(stack.render_uint8(canvas), _old(stack, canvas))


def test_a_canvas_with_no_row_axis_or_no_column_axis_in_the_data_is_still_exact():
    stack = _flat_stack()
    extent = {"z": (0.0, 5.0), "y": (0.0, 10.0), "x": (0.0, 8.0)}
    for axes in (("z", "x"), ("y", "z")):
        canvas = Canvas.covering(extent, height=20, width=26, axes=axes)
        got = stack._render_uint8_gathered(canvas)
        assert got is not None
        assert np.array_equal(got, _old(stack, canvas))
        assert got.any()
