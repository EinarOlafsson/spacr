"""Deforming native waves repel the pointer; cached facets spin only locally."""

import hashlib

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
