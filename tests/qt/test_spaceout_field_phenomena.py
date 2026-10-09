"""Spaceout phenomena deform a bounded native field without changing ordinary field."""

import math

import numpy as np
import pytest

from spacr.qt.widgets import ambient


def _engine(palette='spacr', background='#101418', **kwargs):
    return ambient._SpaceoutFieldEngine(ambient.PALETTE_SETS[palette].colors,
                                        background, seed=42, blur=0, **kwargs)


def _only(engine, key):
    engine.set_field_effects(dict.fromkeys(engine._effect_keys, False))
    engine.set_field_effects({key: True})


def _event(engine, kind, time=12):
    engine.set_time(time)
    row = (kind, 2., 36., ((.3, .4, .24, 1., .2), (.7, .6, .26, -1., 1.)), .7)
    engine._phenomena_schedule = (tuple(math.floor(time / period) for period in engine._lane_periods), (row,))
    return row


def _pixels(engine, width=640, height=360):
    return bytes(engine.shade(width, height).constBits())


@pytest.fixture(autouse=True)
def _cpu_fallback(monkeypatch):
    monkeypatch.setattr(ambient, '_ready_colored_scatter', lambda: None)
    monkeypatch.setattr(ambient, '_ready_packed_scatter', lambda: None)


@pytest.mark.parametrize('palette,background', [('spacr', '#101418'), ('spacr', '#eeeeee'),
                                               ('random', '#101418'), ('random', '#eeeeee')])
def test_disabled_effects_preserve_ordinary_field_full_frames(palette, background):
    made = _engine(palette, background, density=.8, size=.8)
    reference = ambient._DataArtEngine(ambient.PALETTE_SETS[palette].colors, background,
                                       seed=42, blur=0, family='impulse_lens',
                                       density=.8, size=.8)
    made.set_field_effects(dict.fromkeys(made._effect_keys, False))
    for engine in (made, reference):
        engine.set_gravity_radius(.3)
        engine.set_pointer((.4, .6))
        engine._add_ripple((.2, .5))
        engine._set_field_grab(((.6, .5), (.08, .02)))
    for clock in (.4, 2.5, 19., 128.5):
        made.set_time(clock)
        reference.set_time(clock)
        assert _pixels(made) == _pixels(reference)


def test_scheduler_randomizes_bounded_events_and_has_calm_intervals():
    made, same, other = _engine(), _engine(), _engine()
    other._art_seed += 1
    seen, calm, origins, differences = set(), 0, set(), 0
    for clock in range(0, 2200):
        for engine in (made, same, other):
            engine.set_time(clock)
        active = made._field_events()
        assert active == same._field_events()
        assert len(active) <= 2
        assert len(made._phenomena_schedule[1]) <= 2
        other._field_events()
        differences += made._phenomena_schedule != other._phenomena_schedule
        calm += not active
        for row in active:
            assert 2 <= len(row[3]) <= 5
            seen.add(row[0])
            origins.add(row[1])
    assert differences > 1000
    assert calm > 100
    assert seen == set(made._event_keys)
    intervals = np.diff(sorted(origins))
    assert len(np.unique(np.rint(intervals))) > 10
    assert len(origins) > 20
    made.set_time(129.5)
    direct = made._field_events()
    made.set_time(2e8)
    made._field_events()
    made.set_time(129.5)
    assert made._field_events() == direct


@pytest.mark.parametrize('kind', ['attractors', 'vortex', 'density_pulses', 'density_waves'])
def test_independent_geometry_effects_are_finite_and_preserve_grain_population(kind):
    made = _engine()
    _only(made, kind)
    _event(made, kind)
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 60))
    x, y = xx.ravel(), yy.ravel()
    px, py = made._bend_pointer(x, y, 1200, 600)
    assert px.shape == x.shape and py.shape == y.shape
    assert np.isfinite(px).all() and np.isfinite(py).all()
    assert np.count_nonzero((px != x) | (py != y)) > 50
    assert np.max(np.hypot((px - x) * 2, py - y)) < .15
    made.set_field_effects({kind: False})
    off_x, off_y = made._bend_pointer(x, y, 1200, 600)
    assert np.array_equal(off_x, x) and np.array_equal(off_y, y)


def test_multi_attractor_potential_has_both_mountains_and_valleys_and_compact_reach():
    made = _engine()
    _only(made, 'attractors')
    event = _event(made, 'attractors')
    elapsed = made.time - event[1]
    centers = [(cx + .022 * math.sin(elapsed * .12 + phase),
                cy + .022 * math.cos(elapsed * .09 + phase))
               for cx, cy, radius, polarity, phase in event[3]]
    x = np.array([centers[0][0] + .03, centers[1][0] + .03, -.8])
    y = np.array([centers[0][1], centers[1][1], -.8])
    px, py = made._bend_pointer(x, y, 600, 600)
    assert px[0] < x[0] and px[1] > x[1]
    assert px[2] == x[2] and py[2] == y[2]


@pytest.mark.parametrize('kind', ['color_waves', 'spirals'])
@pytest.mark.parametrize('background', ['#101418', '#eeeeee'])
def test_traveling_color_effects_change_owned_crisp_pixels_and_disable_exactly(kind, background):
    made = _engine(background=background, density=.8)
    _only(made, kind)
    _event(made, kind)
    first = made.shade(640, 360)
    frozen = bytes(first.constBits())
    ordinary = _engine(background=background, density=.8)
    ordinary.set_field_effects(dict.fromkeys(ordinary._effect_keys, False))
    ordinary.set_time(made.time)
    assert frozen != _pixels(ordinary)
    made.set_time(15)
    assert _pixels(made) != frozen
    assert bytes(first.constBits()) == frozen
    raster = np.frombuffer(frozen, np.uint32)
    assert np.all(raster >> 24 == 255)
    assert len(np.unique(raster & 0xffffff)) > 20
    made.set_field_effects({kind: False})
    ordinary.set_time(15)
    assert _pixels(made) == _pixels(ordinary)
    assert sum(table.nbytes for table in made._material_cache['spaceout_hues']) == 393216


def test_color_kernel_partial_write_failure_falls_back_and_stops_retries(monkeypatch):
    reference, made = _engine(), _engine()
    for engine in (reference, made):
        _only(engine, 'spirals')
        _event(engine, 'spirals')
    expected = _pixels(reference)
    calls = []

    def broken(output, *args):
        calls.append(True)
        output.fill(0x11223344)
        raise RuntimeError('controlled partial kernel failure')

    monkeypatch.setattr(ambient, '_ready_colored_scatter', lambda: broken)
    assert _pixels(made) == expected
    assert _pixels(made) == expected
    assert calls == [True]
    assert made._phenomena_kernel_failed


def test_relaxation_and_elastic_release_are_independent_and_smooth_at_boundaries():
    made = _engine()
    row = _event(made, 'attractors')
    row = (*row[:4], math.pi)
    made.set_time(2)
    assert made._event_strength(row) == 0
    made.set_time(38)
    assert made._event_strength(row) == 0
    made.set_time(2 + 36 * .3)
    assert made._event_strength(row) == pytest.approx(1)
    made.set_time(2 + 36 * .6)
    elastic = made._event_strength(row)
    made.set_field_effects({'elastic_release': False})
    relaxed = made._event_strength(row)
    made.set_field_effects({'relaxation': False})
    simple = made._event_strength(row)
    assert elastic != relaxed != simple


def test_elastic_grab_release_is_bounded_continuous_and_partition_independent():
    made, split = _engine(), _engine()
    for engine in (made, split):
        engine._set_field_grab(((.5, .5), (.12, .04)))
        engine.advance(.6)
        engine._step_field_grab()
        held = engine._field_grab_offset
        engine._set_field_grab(None)
        assert engine._field_grab_offset == held
    made.advance(.6)
    made._step_field_grab()
    for _ in range(6):
        split.advance(.1)
        split._step_field_grab()
    assert made._field_grab_offset == pytest.approx(split._field_grab_offset, abs=1e-12)
    assert made._field_grab_velocity == pytest.approx(split._field_grab_velocity, abs=1e-12)
    assert made._field_grab_offset[0] < 0
    for _ in range(60):
        made.advance(.1)
        made._step_field_grab()
        assert math.hypot(*made._field_grab_offset) <= .18
    assert math.hypot(*made._field_grab_offset) < 1e-6


def test_switch_mapping_is_copied_and_unknown_keys_cannot_enable_an_event():
    made = _engine()
    switches = {'vortex': False, 'unknown': True}
    made.set_field_effects(switches)
    switches['vortex'] = True
    assert not made.field_effects['vortex']
    assert 'unknown' not in made.field_effects
    assert made.field_effects['attractors']


def test_restyle_and_resize_discard_hue_tables_without_changing_seed():
    made = _engine()
    _only(made, 'color_waves')
    _event(made, 'color_waves')
    first = _pixels(made)
    seed = made._art_seed
    made._restyle()
    assert made._material_cache == {}
    assert _pixels(made) == first
    made.set_resolution(2)
    assert made._material_cache == {}
    assert _pixels(made) == first
    made.set_max_pixels(3840 * 2160)
    assert made.buffer_size(3840, 2160) == (3840, 2160)
    assert made._art_seed == seed


@pytest.mark.parametrize('kind', ['density_pulses', 'density_waves'])
def test_density_events_fade_stable_grains_smoothly_without_changing_reservoir(monkeypatch, kind):
    made = _engine(density=1)
    _only(made, kind)
    _event(made, kind)
    captured = []
    original = ambient._DataArtEngine._point_material

    def record(engine, width, height, x, y, light, spread=False):
        captured.append((np.asarray(x).copy(), np.asarray(y).copy(), np.asarray(light).copy()))
        return original(engine, width, height, x, y, light, spread)

    monkeypatch.setattr(ambient._DataArtEngine, '_point_material', record)
    made.shade(960, 540)
    first = captured[-1]
    reservoir = made._material_cache['spaceout_population']
    made.shade(960, 540)
    assert np.array_equal(captured[-1][2], first[2])
    assert made._material_cache['spaceout_population'] is reservoir
    made.set_time(12.01)
    made.shade(960, 540)
    next_values = captured[-1][2]
    assert len(next_values) == len(first[2])
    assert np.max(np.abs(next_values - first[2])) < .05
    assert not np.array_equal(next_values, first[2])
    made.set_time(18)
    made.shade(960, 540)
    assert np.count_nonzero(captured[-1][2] > .1) != np.count_nonzero(first[2] > .1)
    assert reservoir[1].nbytes == reservoir[0] * 4
    made.set_density(2)
    made.shade(960, 540)
    assert made._material_cache['spaceout_population'][0] > reservoir[0]


@pytest.mark.parametrize('kind', ['attractors', 'vortex', 'density_pulses', 'density_waves'])
def test_phenomena_preserve_full_viewport_boundary_and_no_clock_history(kind):
    made = _engine()
    _only(made, kind)
    _event(made, kind)
    t = np.linspace(0, 1, 40)
    x = np.concatenate((t, t, np.zeros(40), np.ones(40), [-.01, 1.01]))
    y = np.concatenate((np.zeros(40), np.ones(40), t, t, [-.01, 1.01]))
    px, py = made._bend_pointer(x, y, 3840, 2160)
    assert np.allclose(px, x, atol=1e-14) and np.allclose(py, y, atol=1e-14)
    made.set_time(10**9)
    made.shade(320, 180)
    assert len(made._phenomena_schedule[1]) <= 2
    made.set_time(12)
    _event(made, kind)
    again_x, again_y = made._bend_pointer(x, y, 3840, 2160)
    assert np.array_equal(again_x, px) and np.array_equal(again_y, py)


def test_native_sampler_population_follows_density_independently_of_detail(monkeypatch):
    made = _engine(density=1, resolution=1)
    made.set_max_pixels(3840 * 2160)
    made.set_field_effects(dict.fromkeys(made._effect_keys, False))
    samples = []

    def record(width, height, x, y, light, spread=False):
        from PySide6.QtGui import QImage
        samples.append((width, height, np.size(x)))
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(0xff000000)
        return image

    monkeypatch.setattr(made, '_point_material', record)
    for density in (.01, .1, 1, 2, 3):
        made.set_density(density)
        made.set_resolution(1)
        made.shade(3840, 2160)
        made.set_resolution(2)
        made.shade(3840, 2160)
        assert samples[-2] == samples[-1]
        assert samples[-1][:2] == (3840, 2160)
    assert all(samples[i][2] < samples[i + 2][2] for i in range(0, 8, 2))


def test_density_visibility_is_real_rendered_population_and_keeps_owned_frames():
    made, reference = _engine(density=1), _engine(density=1)
    _only(made, 'density_pulses')
    reference.set_field_effects(dict.fromkeys(reference._effect_keys, False))
    _event(made, 'density_pulses', time=13)
    reference.set_time(13)
    first = made.shade(960, 540)
    frozen = bytes(first.constBits())
    ordinary = np.frombuffer(_pixels(reference, 960, 540), np.uint32)
    pulsed = np.frombuffer(frozen, np.uint32)
    assert np.count_nonzero(pulsed & 0xffffff) < np.count_nonzero(ordinary & 0xffffff)
    made.set_time(18)
    assert _pixels(made, 960, 540) != frozen
    assert bytes(first.constBits()) == frozen


@pytest.mark.parametrize('palette', ['spacr', 'random'])
@pytest.mark.parametrize('background', ['#101418', '#eeeeee'])
def test_color_endpoints_match_original_palette_without_baseline_hue_jump(palette, background):
    made, ordinary = _engine(palette, background), _engine(palette, background)
    _only(made, 'color_waves')
    ordinary.set_field_effects(dict.fromkeys(ordinary._effect_keys, False))
    for clock in (2.000001, 37.999999):
        _event(made, 'color_waves', clock)
        ordinary.set_time(clock)
        assert _pixels(made) == _pixels(ordinary)
    _event(made, 'color_waves', 2.05)
    ordinary.set_time(2.05)
    near = np.frombuffer(_pixels(made), np.uint8).astype(np.int16)
    plain = np.frombuffer(_pixels(ordinary), np.uint8).astype(np.int16)
    assert np.max(np.abs(near - plain)) <= 1


def test_both_release_switches_choose_gradual_or_elastic_per_seeded_event():
    made = _engine()
    row = _event(made, 'attractors', 23)
    gradual = (*row[:4], 0.)
    elastic = (*row[:4], math.pi)
    assert made._event_strength(gradual) > 0
    assert made._event_strength(elastic) < 0
    made.set_field_effects({'elastic_release': False})
    assert made._event_strength(gradual) == made._event_strength(elastic)
    made.set_field_effects({'elastic_release': True, 'relaxation': False})
    assert made._event_strength(gradual) == made._event_strength(elastic)


@pytest.mark.parametrize('background', ['#101418', '#eeeeee'])
@pytest.mark.parametrize('spread', [False, True])
def test_color_scatter_kernel_and_numpy_match_exactly_with_overlap(monkeypatch, background, spread):
    made = _engine(background=background)
    _only(made, 'spirals')
    _event(made, 'spirals')
    x = np.array([0, 0, 1, 1, 10, 10, 23, 23, -1, 24], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 10, 10, 15, 15, 0, 15], dtype=np.float32)
    light = np.linspace(.1, .95, len(x), dtype=np.float32)
    reference = made._point_material(24, 16, x, y, light, spread)
    monkeypatch.setattr(ambient, '_ready_colored_scatter', lambda: ambient._scatter_colored_grains)
    actual = made._point_material(24, 16, x, y, light, spread)
    assert bytes(actual.constBits()) == bytes(reference.constBits())


@pytest.mark.parametrize('background', ['#101418', '#eeeeee'])
def test_random_color_peak_and_custom_palette_endpoints_preserve_owned_opaque_frames(background):
    made = _engine('random', background)
    _only(made, 'color_waves')
    _event(made, 'color_waves', 12.8)
    first = made.shade(640, 360)
    frozen = bytes(first.constBits())
    assert np.all(np.frombuffer(frozen, np.uint32) >> 24 == 255)
    _event(made, 'color_waves', 20)
    assert _pixels(made) != frozen
    assert bytes(first.constBits()) == frozen
    colors = ['#00c8ff', '#ffee30', '#fb2090', '#31cf77', '#dd6611']
    custom = ambient._SpaceoutFieldEngine(colors, background, seed=42, blur=0)
    ordinary = ambient._DataArtEngine(colors, background, seed=42, blur=0, family='impulse_lens')
    _only(custom, 'spirals')
    for clock in (2.000001, 37.999999):
        _event(custom, 'spirals', clock)
        ordinary.set_time(clock)
        assert _pixels(custom) == _pixels(ordinary)


def test_elastic_release_no_elapsed_does_not_mutate_and_large_seek_retires_handle():
    made = _engine()
    made._set_field_grab(((.5, .5), (.1, .02)))
    made.advance(.4)
    made._step_field_grab()
    made._set_field_grab(None)
    state = made._field_grab_offset, made._field_grab_velocity
    made._step_field_grab()
    assert state == (made._field_grab_offset, made._field_grab_velocity)
    made.advance(10000)
    made._step_field_grab()
    assert made._field_grab_offset == (0, 0)
    assert made._field_grab_center is None


def test_float32_cyclic_hue_rounding_cannot_index_beyond_palette():
    made = _engine('random')
    _only(made, 'color_waves')
    made._art_seed = 0
    row = _event(made, 'color_waves', 2.4)
    row = (*row[:4], 0.)
    made._phenomena_schedule = (tuple(math.floor(made.time / p) for p in made._lane_periods), (row,))
    column = np.float32((made.time - row[1]) * .22 / 7 * 100)
    column = np.nextafter(column, np.float32(-math.inf))
    x = np.array([np.nextafter(column, np.float32(-math.inf))], np.float32)
    y = np.zeros(1, np.float32)
    flow = made._event_strength(row) * .45 * np.sin(x / 100 * 7 - (made.time - row[1]) * .22)
    assert np.mod(flow, 1.0)[0] == 1.0
    image = made._point_material(100, 40, x, y, np.array([.8], np.float32), True)
    words = np.frombuffer(image.constBits(), np.uint32)
    assert np.all(words >> 24 == 255)
    assert np.any(words & 0xffffff)


def test_relief_has_signed_depth_and_slopes_and_disables_exactly():
    made = _engine()
    _only(made, 'attractors')
    event = _event(made, 'attractors')
    elapsed = made.time - event[1]
    x = np.array([cx + .022 * math.sin(elapsed * .12 + phase) + .04
                  for cx, cy, radius, polarity, phase in event[3]])
    y = np.array([cy + .022 * math.cos(elapsed * .09 + phase)
                  for cx, cy, radius, polarity, phase in event[3]])
    depth, gx, gy = made._field_depth(x, y, 600, 600)
    assert depth[0] > .03 and depth[1] < -.03
    assert gx[0] < -.1 and gx[1] > .1
    assert np.isfinite(gy).all()
    made.set_field_effects({'attractors': False})
    assert all(np.array_equal(values, np.zeros_like(x))
               for values in made._field_depth(x, y, 600, 600))


def test_real_scheduler_shows_mountains_and_vortices_in_first_minute():
    for seed in (1, 19, 42, 137):
        made = _engine()
        made._art_seed = seed
        seen = set()
        for clock in range(60):
            made.set_time(clock)
            seen.update(event[0] for event in made._field_events())
        assert {'attractors', 'vortex'} <= seen
