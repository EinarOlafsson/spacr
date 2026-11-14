"""The once-per-data-pixel composite declines or simplifies in the cases the other tests skip."""
from __future__ import annotations

import numpy as np

from spacr.layers import Canvas, LayerStack, Spacing


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
