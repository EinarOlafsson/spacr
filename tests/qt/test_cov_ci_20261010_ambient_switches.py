"""Ripple and field-effect switches on backdrops whose engines lack them."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient


def test_a_classic_backdrop_accepts_ripple_and_effect_switches(qtbot):
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", background="#101418", seed=5)
    qtbot.addWidget(widget)
    widget.resize(120, 80)
    assert widget._art_input is None
    assert getattr(widget.engine, "set_ripples_enabled", None) is None
    assert getattr(widget.engine, "set_field_effects", None) is None
    widget.set_ripples_enabled(not widget._ripples_enabled)
    flipped = widget._ripples_enabled
    widget.set_ripples_enabled(flipped)
    assert widget._ripples_enabled is flipped
    effects = {key: not value for key, value in dict(widget._field_effects).items()} or {"vortex": True}
    widget.set_field_effects(effects)
    assert widget._field_effects == effects
    assert np.isfinite(widget._ripple_intensity)
