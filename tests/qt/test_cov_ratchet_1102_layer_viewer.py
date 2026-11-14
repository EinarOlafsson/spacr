"""The layer panel follows, detaches from and refuses edits for its stack."""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import Colormap, LayerError, LayerStack  # noqa: E402
from spacr.qt import layer_viewer as lv  # noqa: E402

pytestmark = pytest.mark.qt


def _field():
    stack = LayerStack()
    image = np.full((40, 40), 1000, dtype=np.uint16)
    image[20:30, 20:30] = 4000
    stack.add_image(image, name="image", contrast_limits=(0.0, 4095.0))
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[20:30, 20:30] = 17
    stack.add_labels(mask, name="mask")
    return stack


def _panel(qtbot, stack):
    panel = lv._LayerPanel(stack)
    qtbot.addWidget(panel)
    return panel


def test_the_panel_exposes_its_stack_and_stops_listening_when_detached(qtbot):
    stack = _field()
    panel = _panel(qtbot, stack)
    stack.select("image")
    image = stack.selected
    assert panel.stack is stack

    image.opacity = 0.25
    assert panel.opacity_slider.value() == 25

    panel.detach()
    image.opacity = 0.75
    assert panel.opacity_slider.value() == 25, "a detached panel still followed the stack"


def test_a_float_layer_gets_decimal_limits_scaled_to_its_span(qtbot):
    stack = LayerStack()
    stack.add_image(np.linspace(-3.0, 5.0, 100).reshape(10, 10).astype(np.float32),
                    name="float", contrast_limits=(-3.0, 5.0))
    panel = _panel(qtbot, stack)
    stack.select("float")
    panel.sync()

    assert panel.low_spin.decimals() == 4
    assert panel.high_spin.maximum() == pytest.approx(5000.0)
    assert panel.low_spin.minimum() == pytest.approx(-5000.0)
    assert panel.low_spin.value() == pytest.approx(-3.0)
    assert panel.high_spin.value() == pytest.approx(5.0)

    stack.select(None)
    stack.add_image(np.arange(100, dtype=np.uint16).reshape(10, 10), name="int",
                    contrast_limits=(0.0, 99.0))
    stack.select("int")
    panel.sync()
    assert panel.low_spin.decimals() == 0
    assert panel.high_spin.maximum() == float(np.iinfo(np.uint16).max)


def test_a_colormap_the_list_does_not_know_is_added_and_shown(qtbot):
    stack = _field()
    panel = _panel(qtbot, stack)
    stack.select("image")
    custom = Colormap("my ramp", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.5)])
    assert panel.colormap_combo.findText("my ramp") < 0

    stack.selected.set_colormap(custom)

    assert panel.colormap_combo.findText("my ramp") >= 0
    assert panel.colormap_combo.currentText() == "my ramp"


def test_a_blending_the_combo_does_not_list_leaves_the_choice_alone(qtbot):
    stack = _field()
    panel = _panel(qtbot, stack)
    stack.select("image")
    panel.sync()
    panel.blending_combo.blockSignals(True)
    panel.blending_combo.setCurrentIndex(2)
    panel.blending_combo.blockSignals(False)
    stack.selected._blending = "not-a-listed-mode"

    panel.sync()

    assert panel.blending_combo.currentIndex() == 2


def test_controls_that_do_not_apply_to_the_active_layer_kind_change_nothing(qtbot):
    stack = _field()
    panel = _panel(qtbot, stack)
    image, labels = stack[0], stack[1]

    stack.select("mask")
    contour = labels.contour
    colormap = image.colormap.name
    limits = image.contrast_limits(0)
    panel._on_colormap("gray")
    panel._on_limits()
    panel._on_auto()
    assert (image.colormap.name, image.contrast_limits(0)) == (colormap, limits)

    stack.select("image")
    panel._on_labels_display(1)
    assert labels.contour == contour

    blending = image.blending
    panel._on_blending(-1)          # no item behind the index: no mode to apply
    panel._on_colormap("")          # no name: nothing to look up
    assert image.blending == blending
    assert image.colormap.name == colormap


def test_an_empty_contrast_range_is_left_alone(qtbot):
    stack = _field()
    panel = _panel(qtbot, stack)
    stack.select("image")
    panel.sync()
    before = stack.selected.contrast_limits(0)

    panel.high_spin.setValue(panel.low_spin.value())

    assert stack.selected.contrast_limits(0) == before


def test_a_colormap_the_layer_refuses_is_logged_and_not_applied(qtbot, monkeypatch):
    stack = _field()
    panel = _panel(qtbot, stack)
    stack.select("image")
    layer = stack.selected
    before = layer.colormap.name

    def _refuse(*_args, **_kwargs):
        raise LayerError("unknown colormap")

    monkeypatch.setattr(type(layer), "set_colormap", _refuse)
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    lv.LOG.addHandler(handler)
    try:
        panel._on_colormap("green")
    finally:
        lv.LOG.removeHandler(handler)

    assert layer.colormap.name == before
    assert [r.getMessage() for r in records] == ["Could not set the colormap"]


def test_auto_resets_the_limits_from_the_layers_own_intensities(qtbot):
    stack = _field()
    panel = _panel(qtbot, stack)
    stack.select("image")
    layer = stack.selected
    panel.sync()
    assert panel.high_spin.value() == pytest.approx(4095.0)

    panel._on_auto()

    low, high = layer.contrast_limits(0)
    assert high < 4095.0
    assert panel.low_spin.value() == pytest.approx(low)
    assert panel.high_spin.value() == pytest.approx(high)
