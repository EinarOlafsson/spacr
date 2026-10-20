"""Ripple and field-effect switches on backdrops whose engines lack them."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient


@pytest.mark.parametrize("palette", ["spacr", "random"])
def test_a_light_aurora_inverts_its_ray_colours(palette):
    dark = ambient.make_engine("aurora", palette, "#101418", seed=3)
    light = ambient.make_engine("aurora", palette, "#f4f4f4", seed=3)
    assert dark.dark and not light.dark
    assert light._random_palette is (palette == "random")
    for engine in (dark, light):
        engine.set_time(4.0)
    first = dark.shade(96, 64)
    second = light.shade(96, 64)
    assert first is not None and second is not None
    assert first.size() == second.size()


def test_disabling_ripples_keeps_other_field_materials():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418", seed=4)
    engine._material_cache[("point_atlas", 1)] = ("kept",)
    engine._material_cache[("impulse_lens", 2)] = (None, None, None, [1, 2])
    engine._popup_waves.append((0.0, (0.5, 0.5)))
    engine.set_ripples_enabled(False)
    assert not engine.ripples_enabled
    assert not engine._popup_waves
    assert engine._material_cache[("point_atlas", 1)] == ("kept",)
    assert engine._material_cache[("impulse_lens", 2)][3] == []


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
