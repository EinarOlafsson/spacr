"""Table-driven tests of the once-per-data-pixel composite and the curation seams in ``spacr.layers``.

``LayerStack._render_uint8_gathered`` is the fast composite: it either gives
exactly what the plain renderer gives, or declines with ``None`` and lets
``render_uint8`` fall back.  ``MaskCuration`` over a ``LabelsLayer`` built from
``Spacing.from_map`` must let a careless view (double subscribe, stray
unsubscribe, re-opened stroke) leave the session intact.  One general test per
unit, each walking a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np

from spacr.curation import MaskCuration
from spacr.layers import Canvas, LabelsLayer, LayerStack, Spacing


def _plain(stack, canvas):
    """What the plain float renderer gives, quantised the way uint8 is."""
    return np.clip(stack.render(canvas) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _flat_stack():
    rng = np.random.default_rng(3)
    spacing = Spacing.isotropic(2, 0.5, units="um")
    stack = LayerStack()
    stack.add_image(rng.integers(0, 3000, (20, 16)).astype(np.uint16),
                    name="image", spacing=spacing)
    stack.add_labels(rng.integers(0, 6, (20, 16)), name="mask",
                     spacing=spacing)
    return stack


def _odd_layer_stack():
    """A labels layer whose class is a subclass the composite does not know."""
    stack = _flat_stack()
    layer = stack["mask"]
    original = type(layer)._draw

    class Odd(type(layer)):
        def _draw(self, *args, **kwargs):
            return original(self, *args, **kwargs)

    layer.__class__ = Odd
    return stack


def _shifted_stack():
    """Same-shape layers whose spacings sample different world pixels."""
    rng = np.random.default_rng(4)
    stack = LayerStack()
    stack.add_image(rng.integers(0, 3000, (20, 16)).astype(np.uint16),
                    name="a", spacing=Spacing.isotropic(2, 0.5, units="um"))
    stack.add_labels(rng.integers(0, 6, (20, 16)), name="b",
                     spacing=Spacing(axes=("y", "x"), scale=(0.5, 0.5),
                                     translate=(1.5, 0.0), units="um"))
    return stack


def test_the_gathered_composite_is_exact_or_declines(subtests):
    """Either the plain renderer's exact pixels, or ``None`` and the plain fallback."""
    flat_extent = {"y": (0.0, 10.0), "x": (0.0, 8.0)}
    third_axis = {"z": (0.0, 5.0), "y": (0.0, 10.0), "x": (0.0, 8.0)}
    # (case, stack factory, canvas, gathered declines?)
    cases = [
        ("unknown_layer_class_declines", _odd_layer_stack,
         lambda stack: Canvas.covering(stack, height=30, width=24), True),
        ("layers_sampling_different_pixels_decline", _shifted_stack,
         lambda stack: Canvas.covering(flat_extent, height=30, width=24),
         True),
        ("canvas_rows_on_an_axis_the_data_lacks", _flat_stack,
         lambda stack: Canvas.covering(third_axis, height=20, width=26,
                                       axes=("z", "x")), False),
        ("canvas_columns_on_an_axis_the_data_lacks", _flat_stack,
         lambda stack: Canvas.covering(third_axis, height=20, width=26,
                                       axes=("y", "z")), False),
    ]
    for case_id, make_stack, make_canvas, declines in cases:
        with subtests.test(msg=case_id):
            stack = make_stack()
            canvas = make_canvas(stack)
            expected = _plain(stack, canvas)
            assert expected.any()
            gathered = stack._render_uint8_gathered(canvas)
            if declines:
                assert gathered is None
            else:
                np.testing.assert_array_equal(gathered, expected)
            np.testing.assert_array_equal(stack.render_uint8(canvas), expected)


def _session():
    """A flat, isotropic mask session -- the geometry is not what is tested."""
    layer = LabelsLayer(np.zeros((1, 11, 11), dtype=np.int64), name="mask",
                        spacing=Spacing.from_map({"z": 1.0, "y": 1.0,
                                                  "x": 1.0}, units="um"))
    return MaskCuration(layer, artifact="mask.tif")


class _Panel:
    """A view that records the ledger entries it is told about."""

    def __init__(self):
        self.seen = []

    def on_edit(self, edit):
        """Receive one :class:`CurationEdit`."""
        self.seen.append(edit)


def test_a_careless_view_cannot_corrupt_a_curation_session(subtests):
    """Double subscribe, stray unsubscribe and a re-opened stroke are no-ops."""
    with subtests.test(msg="from_map_spacing"):
        spacing = _session().layer.spacing
        assert tuple(spacing.axes) == ("z", "y", "x")
        assert tuple(spacing.scale) == (1.0, 1.0, 1.0)

    with subtests.test(msg="same_bound_method_twice_notified_once"):
        session = _session()
        first, second = _Panel(), _Panel()
        session.subscribe(first.on_edit)
        session.subscribe(first.on_edit)
        session.subscribe(second.on_edit)
        session.paint({"z": 0.0, "y": 5.0, "x": 5.0}, label=3, radius=1.0)
        assert len(first.seen) == 1
        assert len(second.seen) == 1
        assert first.seen[0] is second.seen[0]
        assert first.seen[0].kind == "paint"

    with subtests.test(msg="unsubscribing_a_stranger_removes_nobody"):
        session = _session()
        panel, stranger = _Panel(), _Panel()
        session.subscribe(panel.on_edit)
        session.unsubscribe(stranger.on_edit)
        session.paint({"z": 0.0, "y": 5.0, "x": 5.0}, label=3, radius=1.0)
        assert (len(panel.seen), stranger.seen) == (1, [])
        session.unsubscribe(panel.on_edit)
        session.paint({"z": 0.0, "y": 5.0, "x": 6.0}, label=3, radius=1.0)
        assert len(panel.seen) == 1

    with subtests.test(msg="second_begin_stroke_keeps_the_open_dabs"):
        session = _session()
        session.begin_stroke()
        session.paint({"z": 0.0, "y": 5.0, "x": 4.0}, label=2, radius=1.0)
        session.begin_stroke()
        session.paint({"z": 0.0, "y": 5.0, "x": 7.0}, label=2, radius=1.0)
        entry = session.end_stroke()
        painted = int(np.count_nonzero(session.layer.data))
        assert len(session.log) == 1
        assert entry.n_changed == painted
        assert painted > 0
        # Two dabs four pixels apart: both halves are in the one entry.
        assert session.layer.data[0, 5, 4] == 2
        assert session.layer.data[0, 5, 7] == 2
        session.undo()
        assert int(np.count_nonzero(session.layer.data)) == 0
