"""Deforming native waves repel the pointer; cached facets spin only locally."""

import hashlib
import math

import numpy as np
import pytest
from PySide6.QtGui import QImage, QTransform

from spacr.qt.widgets import ambient


def _engine(family, background="#101418", palette="spacr", **kwargs):
    kwargs.setdefault("resolution", 1)
    return ambient.make_engine("data_art_" + family, palette, background,
                               seed=42, blur=0, **kwargs)


def _shade(engine, width=640, height=360):
    engine.set_max_pixels(width * height)
    image = engine.shade(width, height)
    assert (image.width(), image.height()) == (width, height)
    return image


def _words(image):
    return np.frombuffer(image.constBits(), dtype=np.uint32).copy()


def _capture(engine, monkeypatch, width=640, height=360):
    captured = []

    def material(w, h, x, y, light, spread=False):
        captured.append(tuple(np.asarray(value).copy() for value in (x, y, light)))
        image = QImage(w, h, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, "_point_material", material)
    engine._frame_point_atlas(width, height)
    return captured[-1]


def test_waves_change_shape_instead_of_only_translating(monkeypatch):
    engine = _engine("point_atlas")
    engine.set_time(2)
    first = _capture(engine, monkeypatch)
    geometry = next(iter(engine._material_cache.values()))
    for stamp in (2.25, 6, 19):
        engine.set_time(stamp)
        second = _capture(engine, monkeypatch)
        displacement = np.vstack((second[0] - first[0], second[1] - first[1]))
        deformation = displacement - displacement.mean(axis=1, keepdims=True)
        assert np.max(np.linalg.norm(deformation, axis=0)) > 1
        assert np.std(displacement[1]) > .3
        assert not np.array_equal(first[2], second[2])
        assert next(iter(engine._material_cache.values())) is geometry
    rigid_displacement = np.full_like(displacement, 3)
    assert np.max(np.abs(rigid_displacement - rigid_displacement.mean(
        axis=1, keepdims=True))) == 0


@pytest.mark.parametrize("width,height", [(640, 360), (360, 640)])
@pytest.mark.parametrize("radius", [.01, .1, .5])
def test_wave_gravity_repels_inside_physical_radius_only(
        monkeypatch, width, height, radius):
    engine = _engine("point_atlas")
    engine.set_time(8)
    engine.set_gravity_radius(radius)
    first = _capture(engine, monkeypatch, width, height)
    centre = (width * .5, height * .5)
    if radius == .01:
        nearest = np.argmin(np.hypot(first[0] - centre[0], first[1] - centre[1]))
        centre = (first[0][nearest] + 1, first[1][nearest] + 1)
    engine.set_pointer((centre[0] / width, centre[1] / height))
    second = _capture(engine, monkeypatch, width, height)
    before = np.hypot(first[0].astype(np.float64) - centre[0],
                      first[1].astype(np.float64) - centre[1])
    after = np.hypot(second[0].astype(np.float64) - centre[0],
                     second[1].astype(np.float64) - centre[1])
    influenced = (first[0] != second[0]) | (first[1] != second[1])
    assert influenced.any()
    core = influenced & (before < .95 * radius * min(width, height))
    assert core.any()
    assert np.all(after[core] > before[core])
    assert np.all(after[influenced] >= before[influenced] - .0001)
    outside = before >= radius * min(width, height)
    assert np.array_equal(first[0][outside], second[0][outside])
    assert np.array_equal(first[1][outside], second[1][outside])
    assert np.array_equal(first[2], second[2])


def test_wave_repulsion_does_not_reverse_lens_gravity():
    coordinates = np.array([.51, .55, .6, .9], dtype=np.float32)
    rows = np.full_like(coordinates, .5)
    for family, direction in (("point_atlas", 1), ("impulse_lens", -1)):
        engine = _engine(family)
        engine.set_gravity_radius(.5)
        engine.set_pointer((.5, .5))
        x, y = engine._bend_pointer(coordinates, rows, 640, 360)
        changed = x != coordinates
        assert np.all((x[changed] - coordinates[changed]) * direction > 0)
        assert np.array_equal(y, rows)


@pytest.mark.parametrize("size,density,pointer", [(1, 1, (0, 0)),
                                                  (.25, 3, (1, 1))])
def test_deforming_native_waves_preserve_all_uncropped_edge_pixels(
        size, density, pointer):
    width, height = 3840, 2160
    engine = _engine("point_atlas", size=size, density=density)
    full = _engine("point_atlas", size=size, density=density)
    for material in (engine, full):
        material.set_max_pixels(width * height)
        material.set_gravity_radius(1)
        material.set_pointer(pointer)
    spacing = max(2.4, 4.6 * size / np.sqrt(ambient.DENSITY_RANGE[1]))
    columns = min(900, max(48, int(np.ceil(width * 1.65 / spacing))))
    rows = min(520, max(32, int(np.ceil(height * 1.85 / spacing))))
    population = np.sqrt(full.effective_density() / ambient.DENSITY_RANGE[1])
    columns = max(3, int(np.ceil(columns * population)))
    rows = max(3, int(np.ceil(rows * population)))
    x, z = np.meshgrid(np.linspace(-.33, 1.33, columns, dtype=np.float32),
                       np.linspace(-.43, 1.43, rows, dtype=np.float32))
    rng = np.random.default_rng(full._art_seed)
    jitter = rng.uniform(-.17, .17, size=(2, x.size)).astype(np.float32)
    x = x.ravel() + jitter[0] / columns
    z = z.ravel() + jitter[1] / rows
    full._material_cache[("point_atlas", width, height, full.size, full.density)] = (
        x, z, 9 * x + 6.1 * z, 12.3 * z - 4.2 * x,
        18 * x + 8 * z + full._anchors[0][0] * np.pi * 2)
    for stamp in (19, 1023.75):
        engine.set_time(stamp)
        full.set_time(stamp)
        first = _shade(engine, width, height)
        retained = hashlib.sha256(first.constBits()).hexdigest()
        assert np.array_equal(_words(first), _words(_shade(full, width, height)))
        assert hashlib.sha256(first.constBits()).hexdigest() == retained
        assert np.all((_words(first) >> 24) == 255)


@pytest.mark.parametrize("palette", ["spacr", "random"])
def test_native_wave_density_changes_grain_population_and_pixels_at_full_detail(
        monkeypatch, palette):
    width, height = 3840, 2160
    images, counts = [], []
    for density in (1, 2, 3):
        engine = _engine("point_atlas", palette=palette, density=density, resolution=2)
        engine.set_time(8)
        engine.set_gravity_radius(.5)
        engine.set_pointer((.5, .5))
        material = engine._point_material

        def counted(w, h, x, y, light, spread=False, _material=material):
            counts.append(len(x))
            return _material(w, h, x, y, light, spread=spread)

        monkeypatch.setattr(engine, "_point_material", counted)
        frame = _shade(engine, width, height)
        images.append(_words(frame))
        assert engine.effective_density() == density
        assert np.all(images[-1] >> 24 == 255)
    assert counts[0] < counts[1] < counts[2] <= 900 * 520
    assert np.count_nonzero(images[0] != images[1]) > 1000
    assert np.count_nonzero(images[1] != images[2]) > 1000


@pytest.mark.parametrize("density", [.01, .1, .5, 1, 2, 3])
def test_native_wave_grain_population_depends_on_density_and_not_extra_detail(
        monkeypatch, density):
    counts = []
    for detail in (1, 2):
        engine = _engine("point_atlas", density=density, resolution=detail)
        engine.set_max_pixels(3840 * 2160)
        engine.set_gravity_radius(.5)
        captured = _capture(engine, monkeypatch, 3840, 2160)
        counts.append(len(captured[0]))
        assert engine.buffer_size(3840, 2160) == (3840, 2160)
    assert counts[0] == counts[1]
    assert 0 < counts[0] <= 900 * 520


@pytest.mark.parametrize("width,height", [(640, 360), (3840, 2160)])
def test_requested_wave_density_is_graded_across_full_range(monkeypatch, width, height):
    counts = []
    for density in (.01, .1, .5, 1, 2, 3):
        engine = _engine("point_atlas", density=density)
        counts.append(len(_capture(engine, monkeypatch, width, height)[0]))
    assert all(left < right for left, right in zip(counts, counts[1:]))
    assert counts[-1] <= 900 * 520


def test_facet_angular_speed_scales_with_animation_speed():
    engines = [_engine("tissue_facets", speed=value) for value in (1, 4)]
    for engine in engines:
        engine.set_gravity_radius(.5)
        engine.set_pointer((.5, .5))
        _shade(engine)
        engine.advance(.1)
        _shade(engine)
    assert np.array(_rotation(engines[1])[1]) == pytest.approx(
        np.array(_rotation(engines[0])[1]) * 4)


def test_facet_rotation_is_fastest_at_pointer_and_fades_to_radius():
    engine = _engine("tissue_facets")
    engine.set_gravity_radius(.6)
    engine.set_pointer((.5, .5))
    _shade(engine)
    material = next(value for key, value in engine._material_cache.items()
                    if key[0] == "tissue_facets")
    engine.advance(.1)
    _shade(engine)
    angles = np.asarray(_rotation(engine)[1])
    distance = np.array([math.hypot(cell[0] - 320, cell[1] - 180) / 360
                         for cell in material])
    expected = 24 * np.maximum(0, 1 - distance / .6) ** 2
    assert angles == pytest.approx(expected)
    assert np.max(angles) > 18
    assert np.count_nonzero((angles > 0) & (angles < 1)) > 0
    assert np.all(angles[distance >= .6] == 0)


def test_facet_clock_jump_clears_rotation_instead_of_integrating_backwards():
    engine = _engine("tissue_facets")
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    resting_pointer = _words(_shade(engine))
    engine.advance(.1)
    assert not np.array_equal(resting_pointer, _words(_shade(engine)))
    engine.set_time(-10)
    assert np.array_equal(resting_pointer, _words(_shade(engine)))
    assert not any(_rotation(engine)[1])


@pytest.mark.parametrize("palette", ["spacr", "random", "custom"])
def test_local_spin_keeps_saved_palette_and_native_opaque_output(palette, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_ambient_custom_colors", lambda: ("#13d5ca", "#ef67bb"))
    engine = _engine("tissue_facets", palette=palette)
    colors = tuple(color.rgba() for color in engine._colors)
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    first = _shade(engine)
    engine.advance(.1)
    second = _shade(engine)
    assert not np.array_equal(_words(first), _words(second))
    assert np.all((_words(second) >> 24) == 255)
    assert tuple(color.rgba() for color in engine._colors) == colors


@pytest.mark.parametrize("background", ["#101418", "#f6f7f9"])
def test_facets_have_no_global_clocked_lighting_or_icon_sweep(background):
    engine = _engine("tissue_facets", background)
    first = _words(_shade(engine))
    for pointer in (None, (.5, .5)):
        engine.set_pointer(pointer)
        for stamp in (2, 9, 1234):
            engine.set_time(stamp)
            assert np.array_equal(first, _words(_shade(engine)))
    engine.set_gravity_radius(.5)
    engine.set_pointer(None)
    engine.set_time(2000)
    assert np.array_equal(first, _words(_shade(engine)))


def _rotation(engine):
    return next(value for key, value in engine._material_cache.items()
                if key[0] == "tissue_rotation")


def _cells(engine):
    return next(value for key, value in engine._material_cache.items()
                if key[0] == "tissue_facets")


def test_stationary_pointer_spins_facets_with_graded_local_angular_speed():
    engine = _engine("tissue_facets")
    baseline = _words(_shade(engine))
    cells = _cells(engine)
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    lifted = _words(_shade(engine))
    engine.advance(.1)
    spinning = _words(_shade(engine))
    assert not np.array_equal(lifted, spinning)
    distance = np.hypot([(cell[0] - 320) / 360 for cell in cells],
                        [(cell[1] - 180) / 360 for cell in cells])
    angles = np.array(_rotation(engine)[1])
    assert np.all(angles[distance >= .5] == 0)
    close, farther = distance < .1, (distance > .3) & (distance < .4)
    assert close.any() and farther.any()
    assert angles[close].min() > angles[farther].max() > 0
    higher = _engine("tissue_facets")
    _shade(higher)
    higher.set_gravity_radius(.8)
    higher.set_pointer((.5, .5))
    _shade(higher)
    higher.advance(.1)
    _shade(higher)
    high_angles = np.array(_rotation(higher)[1])
    assert np.count_nonzero(high_angles) > np.count_nonzero(angles)
    assert np.all(high_angles[close] > angles[close])
    assert _cells(engine) is cells
    engine.set_gravity_radius(0)
    assert np.array_equal(baseline, _words(_shade(engine)))


def test_facet_spin_preserves_owned_frames_and_same_clock_is_idempotent():
    engine = _engine("tissue_facets")
    resting = _words(_shade(engine))
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    frame = _shade(engine)
    before = hashlib.sha256(frame.constBits()).hexdigest()
    engine.advance(.1)
    changed = _shade(engine)
    assert hashlib.sha256(changed.constBits()).hexdigest() != before
    assert hashlib.sha256(frame.constBits()).hexdigest() == before
    angles = list(_rotation(engine)[1])
    assert np.array_equal(_words(changed), _words(_shade(engine)))
    assert _rotation(engine)[1] == angles
    engine.set_pointer(None)
    assert np.array_equal(resting, _words(_shade(engine)))


def test_failed_rotated_native_tile_restores_painter_transform():
    engine = _engine("tissue_facets")
    _shade(engine)
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    _shade(engine)
    engine.advance(.1)

    class FailedReplay:
        def __init__(self):
            self.transform = QTransform()
            self.saved = []

        def drawImage(self, *_args):
            if not self.transform.isIdentity():
                raise RuntimeError("injected native tile failure")

        def save(self):
            self.saved.append(QTransform(self.transform))

        def translate(self, x, y):
            self.transform.translate(x, y)

        def rotate(self, degrees):
            self.transform.rotate(degrees)

        def setRenderHint(self, *_args):
            pass

        def restore(self):
            self.transform = self.saved.pop()

    painter = FailedReplay()
    with pytest.raises(RuntimeError, match="native tile"):
        engine._paint_tissue_facets(painter, 640, 360)
    assert painter.transform.isIdentity()
    assert painter.saved == []


@pytest.mark.parametrize("family", ["point_atlas", "tissue_facets"])
def test_new_motion_resize_and_controls_do_not_accumulate_material(family):
    engine = _engine(family)
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    for width, height in ((640, 360), (360, 640), (640, 360)):
        _shade(engine, width, height)
        engine.advance(.1)
        _shade(engine, width, height)
        assert all(key[1:3] == (width, height) for key in engine._material_cache)
        assert len(engine._material_cache) <= 2
    engine.set_density(.5)
    assert not engine._material_cache
    engine.set_size(1.5)
    assert not engine._material_cache


def _buffered_facet_frame(engine, width=640, height=360):
    image = QImage(width, height, QImage.Format_RGB32)
    painter = ambient.QPainter(image)
    try:
        painter.fillRect(image.rect(), engine.identity)
        painter.setCompositionMode(engine.mode)
        painter.setPen(ambient.Qt.NoPen)
        engine._paint_field(painter, width, height)
    finally:
        painter.end()
    return image


@pytest.mark.parametrize("background", ["#101418", "#f6f7f9"])
def test_resting_facet_frames_use_copy_on_write_and_never_share_caller_mutation(
        background, monkeypatch):
    engine = _engine("tissue_facets", background, density=3, resolution=2)
    first = _shade(engine)
    original = _words(first)
    material = _cells(engine)
    original_cache_bytes = sum(cell[5].sizeInBytes() for cell in material)

    def no_identical_redraw(*_args):
        raise AssertionError("resting facets must not be rerasterized")

    monkeypatch.setattr(engine, "_paint_field", no_identical_redraw)
    engine.advance(100)
    second = _shade(engine)
    assert first is not second and second is not engine._buffer
    assert np.array_equal(original, _words(second))
    assert _rotation(engine)[0] == engine.time
    first.fill("red")
    words = np.frombuffer(second.bits(), dtype=np.uint32)
    words[:] = 0xff00ff00
    assert np.array_equal(original, _words(_shade(engine)))
    assert _cells(engine) is material
    assert sum(cell[5].sizeInBytes() for cell in material) == original_cache_bytes
    assert sum(isinstance(value, QImage) for value in engine._material_cache.values()) == 0


@pytest.mark.parametrize("background", ["#101418", "#f6f7f9"])
def test_resting_tick_entry_leave_and_reentry_match_original_rotation_clock(background):
    direct, reference = [_engine("tissue_facets", background, density=3)
                         for _ in range(2)]
    resting = _words(_shade(direct))
    _buffered_facet_frame(reference)
    for elapsed in (.01, 20, 100, .02):
        direct.advance(elapsed)
        reference.advance(elapsed)
        assert np.array_equal(_words(_shade(direct)),
                              _words(_buffered_facet_frame(reference)))
    for engine in (direct, reference):
        engine.set_gravity_radius(.5)
        engine.set_pointer((.5, .5))
        engine.advance(.01)
    assert np.array_equal(_words(_shade(direct)),
                          _words(_buffered_facet_frame(reference)))
    assert _rotation(direct)[1] == _rotation(reference)[1]
    assert 0 < max(_rotation(direct)[1]) <= .01 * 240
    for engine in (direct, reference):
        engine.set_pointer(None)
    assert np.array_equal(resting, _words(_shade(direct)))
    _buffered_facet_frame(reference)
    for engine in (direct, reference):
        engine.advance(100)
    _shade(direct)
    _buffered_facet_frame(reference)
    for engine in (direct, reference):
        engine.set_pointer((.5, .5))
        engine.advance(.01)
    assert np.array_equal(_words(_shade(direct)),
                          _words(_buffered_facet_frame(reference)))


@pytest.mark.parametrize("change", ["density", "size", "resolution", "colors", "background"])
def test_resting_facet_control_changes_invalidate_exact_completed_material(change):
    engine = _engine("tissue_facets", density=3, resolution=2)
    first = _shade(engine)
    original = _words(first)
    getattr(engine, "set_" + change)({
        "density": .5, "size": 2, "resolution": .75,
        "colors": ("#ffa500", "#00ffff"), "background": "#f6f7f9",
    }[change])
    assert not engine._material_cache
    actual = engine.shade(640, 360)
    bw, bh = engine.buffer_size(640, 360)
    reference = _engine("tissue_facets", density=engine.density, size=engine.size,
                         resolution=engine.resolution, background=engine._background)
    reference.set_colors(engine._colors)
    expected = _buffered_facet_frame(reference, bw, bh)
    assert np.array_equal(_words(actual), _words(expected))
    assert np.array_equal(original, _words(first))


def test_failed_active_facet_render_ends_owned_painter_and_preserves_published_frame(
        monkeypatch):
    engine = _engine("tissue_facets")
    first = _shade(engine)
    original = _words(first)
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    painters = []
    paint_field = engine._paint_field

    def fail(painter, *_args):
        painters.append(painter)
        painter.fillRect(0, 0, 50, 50, ambient.QColor("red"))
        raise RuntimeError("injected owned facet failure")

    monkeypatch.setattr(engine, "_paint_field", fail)
    with pytest.raises(RuntimeError, match="owned facet failure"):
        engine.shade(640, 360)
    assert all(not painter.isActive() for painter in painters)
    assert np.array_equal(original, _words(first))
    assert not any(key[0] == "tissue_resting" for key in engine._material_cache)
    monkeypatch.setattr(engine, "_paint_field", paint_field)
    assert _shade(engine) is not None
    engine.set_pointer(None)
    assert np.array_equal(original, _words(_shade(engine)))
    assert engine.shade(0, 360) is None and engine.shade(640, 0) is None


def test_legacy_material_still_copies_its_reusable_buffer_without_resting_cache(monkeypatch):
    monkeypatch.setattr(ambient._SATIN_COMPILER, "ready", lambda: None)
    engine = ambient._DataArtEngine(
        ambient.palette_colors("data_art_tissue_facets", "spacr"), "#101418",
        family="chromatin_ribbon", seed=42, resolution=1, blur=0)
    engine.set_max_pixels(640 * 360)
    first = engine.shade(640, 360)
    original = _words(first)
    engine.advance(.2)
    second = engine.shade(640, 360)
    assert first is not second and second is not engine._buffer
    assert np.array_equal(original, _words(first))
    assert not any(key[0] == "tissue_resting" for key in engine._material_cache)
