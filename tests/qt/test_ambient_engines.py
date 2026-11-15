"""The ambient backdrop engines from make_engine: settings, caches, frames and the graphics choice.

The backdrop paints behind every module screen while a pipeline runs, so
nothing in an engine may raise and nothing may grow without bound.  Each case
drives one engine family through a Preferences knob, an edge-sized canvas, a
long run of frames or a graphics renderer that refuses, and asserts the exact
clamp, cache state, buffer size, wave list or refusal sentence.  The GPU cases
use the recording ``moderngl`` stand-in; nothing touches a graphics driver.

One general test per engine family, each walking a case table with pytest's
``subtests`` fixture; every case builds its own engine.
"""
from __future__ import annotations

import math
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt.widgets import ambient as amb  # noqa: E402
from spacr.qt.widgets.ambient import (AURORA_PULSE_CACHE, AURORA_TILE_CACHE,  # noqa: E402
                                      BUFFER_MIN_EDGE, make_engine)
from tests.qt.test_cov_ci3_ambient_flow_renderer import _fake_moderngl  # noqa: E402

pytestmark = pytest.mark.qt

DARK = "#101418"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Private preferences and no shared animation policy."""
    settings = QSettings(str(tmp_path / "ambient.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(amb, "_RESOURCE_POLICY", None)
    return settings


def _engine(theme, palette=None, background=DARK, seed=7, **kwargs):
    palette = amb.default_palette_for(theme) if palette is None else palette
    return make_engine(theme, palette, background, seed=seed, **kwargs)


def _paint(engine, width=320, height=200) -> QImage:
    """Paint one frame exactly as the widget would."""
    image = QImage(width, height, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), QColor(DARK))
    try:
        engine.paint(painter, width, height)
    finally:
        painter.end()
    return image


def _on_producer(function):
    """Run ``function`` on a thread named like the shading producer."""
    box = {}

    def run():
        try:
            box["value"] = function()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            box["error"] = error

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_engine_setters_clamp_and_drop_only_what_they_sized(subtests):
    """set_blur/resolution/size/direction/max_pixels/ripples clamp and drop exactly their caches."""
    with subtests.test(msg="setters_take_and_clamp_values"):
        engine = _engine("blobs", "spacr")
        engine.set_blur(2.0)
        engine.set_resolution(1.5)
        engine.set_size(2.0)
        assert (engine.blur, engine.resolution, engine.size) == (2.0, 1.5, 2.0)
        engine.set_blur(99.0)
        engine.set_resolution(-1.0)
        engine.set_size(0.0)
        assert (engine.blur, engine.resolution, engine.size) == (
            amb.BLUR_RANGE[1], amb.RESOLUTION_RANGE[0], amb.SIZE_RANGE[0])

    with subtests.test(msg="direction_known_and_unknown"):
        engine = _engine("drift", "spacr", direction="up")
        assert engine.direction == "up"
        engine.set_direction("down")
        assert engine.direction == "down"
        engine.set_direction("sideways")
        assert engine.direction == "down"

    with subtests.test(msg="new_resolution_drops_the_buffer"):
        engine = _engine("blobs", "spacr")
        engine.shade(320, 200)
        assert engine._buffer is not None
        engine.set_resolution(0.5)
        assert engine._buffer is None

    with subtests.test(msg="every_starfield_setting_drops_the_pens"):
        engine = _engine("drift", "spacr")
        for setter, value in [("set_blur", 1.5), ("set_resolution", 0.5), ("set_size", 2.0)]:
            _paint(engine)
            assert engine._pens
            getattr(engine, setter)(value)
            assert engine._pens == {}, setter

    with subtests.test(msg="unchanged_ceiling_keeps_the_buffer"):
        engine = _engine("fractal", "rainbow")
        _paint(engine)
        assert engine._buffer is not None
        engine.set_max_pixels(engine.max_pixels)
        assert engine._buffer is not None
        smaller = BUFFER_MIN_EDGE ** 2
        engine.set_max_pixels(smaller)
        assert engine.max_pixels == smaller
        assert engine._buffer is None

    with subtests.test(msg="disabling_ripples_keeps_other_materials"):
        engine = _engine("data_art_impulse_lens", "spacr", seed=4)
        engine._material_cache[("point_atlas", 1)] = ("kept",)
        engine._material_cache[("impulse_lens", 2)] = (None, None, None, [1, 2])
        engine._popup_waves.append((0.0, (0.5, 0.5)))
        engine.set_ripples_enabled(False)
        assert not engine.ripples_enabled
        assert not engine._popup_waves
        assert engine._material_cache[("point_atlas", 1)] == ("kept",)
        assert engine._material_cache[("impulse_lens", 2)][3] == []


def test_engine_buffers_fit_their_budget_and_empty_canvases_shade_nothing(qapp, subtests):
    """buffer_scale/blur_scale/shade/blit/buds bound the buffer and do nothing for an empty canvas."""
    with subtests.test(msg="huge_canvas_shaded_coarser_than_the_edge_ratio"):
        engine = _engine("aurora", "borealis")
        engine.set_resolution(amb.RESOLUTION_RANGE[1])
        width, height = 3840, 2400
        from_the_ratio = math.ceil(max(width, height) / engine.resolution_edge())
        assert engine.buffer_scale(width, height) > from_the_ratio
        buffer_width, buffer_height = engine.buffer_size(width, height)
        assert buffer_width * buffer_height <= amb.BUFFER_MAX_PIXELS

    with subtests.test(msg="blur_shrinks_the_buffer"):
        engine = _engine("blobs", "spacr")
        assert engine.blur_scale(320, 200) == 1.0
        engine.set_blur(3.0)
        assert engine.blur_scale(320, 200) > 1.0
        softened = engine.shade(320, 200)
        assert softened.width() < 320

    for theme in ("blobs", "drift"):
        with subtests.test(msg=f"empty_canvas_{theme}"):
            engine = _engine(theme, "spacr")
            assert engine.shade(0, 200) is None and engine.shade(320, 0) is None
            assert engine.shade(320, -1) is None
            canvas = QImage(4, 4, QImage.Format_RGB32)
            canvas.fill(QColor("#123456"))
            painter = QPainter(canvas)
            try:
                mode = painter.compositionMode()
                engine.blit(painter, None, 4, 4)
                engine.blit(painter, QImage(), 4, 4)
                engine.blit(painter, QImage(2, 2, QImage.Format_RGB32), 0, 4)
                assert painter.compositionMode() == mode
            finally:
                painter.end()
            assert canvas.pixelColor(0, 0) == QColor("#123456")

    with subtests.test(msg="stratified_stars_shade_a_premultiplied_buffer"):
        engine = _engine("drift")
        engine.set_max_pixels(160 * 100)
        image = engine.shade(320, 200)
        assert (image.width(), image.height()) == engine.buffer_size(320, 200)
        assert image.width() < 320
        assert image.format() == QImage.Format_ARGB32_Premultiplied

    with subtests.test(msg="fractal_grows_no_buds_on_an_empty_canvas"):
        engine = _engine("fractal", "rainbow")
        assert engine.buds(0, 200) == ()
        assert engine.buds(320, 0) == ()
        assert len(engine.buds(320, 200)) >= 1

    for palette in ("spacr", "random"):
        with subtests.test(msg=f"light_aurora_shades_like_dark_{palette}"):
            dark = _engine("aurora", palette, "#101418", seed=3)
            light = _engine("aurora", palette, "#f4f4f4", seed=3)
            assert dark.dark and not light.dark
            assert light._random_palette is (palette == "random")
            for engine in (dark, light):
                engine.set_time(4.0)
            first, second = dark.shade(96, 64), light.shade(96, 64)
            assert first.size() == second.size()
            assert (first.width(), first.height()) == dark.buffer_size(96, 64)


def test_the_aurora_moves_along_its_arc_and_keeps_its_caches_bounded(qapp, monkeypatch, subtests):
    """AuroraEngine fold/pulse are finite travelling values; tile and surge caches reset at their ceiling."""
    with subtests.test(msg="fold_slides_along_the_arc"):
        engine = _engine("aurora", "borealis")
        curtain = engine.curtains[0]
        still = [engine.fold(curtain, u / 20.0, 0.0) for u in range(21)]
        later = [engine.fold(curtain, u / 20.0, 3.0) for u in range(21)]
        assert any(abs(a - b) > 1e-9 for a, b in zip(still, later))
        assert all(math.isfinite(value) for value in still + later)

    with subtests.test(msg="surge_pulse_stays_in_0_1"):
        engine = _engine("aurora", "borealis")
        curtain = engine.curtains[0]
        values = [engine.pulse(curtain, u / 12.0, t / 4.0) for u in range(13) for t in range(9)]
        assert all(0.0 <= value <= 1.0 for value in values)
        assert max(values) > min(values)

    with subtests.test(msg="resizing_drops_both_pixel_caches"):
        engine = _engine("aurora", "borealis", density=1)
        _paint(engine)
        assert engine._ray_material
        curtain = engine.curtains[0]
        engine._tile(curtain, 1.0, 64, 48)
        engine._surge(curtain, 1.0)
        assert engine._tiles and engine._surges
        prior = _paint(engine)
        engine.set_size(2.0)
        assert engine._tiles == {} and engine._surges == {}
        assert _paint(engine) != prior
        for width in range(240, 600, 13):
            _paint(engine, width, 200)
            assert len(engine._ray_material) <= 24

    with subtests.test(msg="caches_reset_at_their_ceiling"):
        engine = _engine("aurora", "borealis")
        curtain = engine.curtains[0]
        engine._tiles = {("filler", i): None for i in range(AURORA_TILE_CACHE)}
        engine._surges = {("filler", i): None for i in range(AURORA_PULSE_CACHE)}
        tile = engine._tile(curtain, 1.0, 64, 48)
        surge = engine._surge(curtain, 1.0)
        assert not tile.isNull() and not surge.isNull()
        assert (len(engine._tiles), len(engine._surges)) == (1, 1)

    with subtests.test(msg="full_length_ray_is_not_cut_short"):
        engine = _engine("aurora", "borealis")
        with monkeypatch.context() as mp:
            mp.setattr(type(engine), "ray_lengths",
                       lambda self, c: tuple(1.0 for _ in amb.AURORA_TILE_RAYS))
            tile = engine._tile(engine.curtains[0], 1.0, 64, 48)
        assert not tile.isNull()
        assert (tile.width(), tile.height()) == (64, 48)

    with subtests.test(msg="curtain_with_one_sample_is_skipped"):
        engine = _engine("aurora", "borealis")
        with monkeypatch.context() as mp:
            mp.setattr(type(engine), "geometry", lambda self, width, height: ((0.0, 0.0),))
            image = _paint(engine)
        assert (image.width(), image.height()) == (320, 200)


def test_the_starfield_halo_density_and_random_headings(qapp, subtests):
    """DriftEngine draws its halo only with blur, keeps alpha at 1 when dense, and scatters random headings."""
    with subtests.test(msg="halo_is_the_dot_until_blur"):
        engine = _engine("drift", "spacr")
        assert engine.halo_size(0) == engine.dot_size(0)
        engine.set_blur(3.0)
        assert engine.dot_size(0) < engine.halo_size(0) <= amb.DRIFT_HALO_MAX_PX

    with subtests.test(msg="denser_does_not_dim"):
        engine = _engine("drift", "spacr")
        engine.set_density(3.0)
        assert engine.alpha_scale() == 1.0

    with subtests.test(msg="random_direction_scatters_headings"):
        engine = _engine("drift", "spacr", direction="random", density=1)
        engine.set_time(4.0)
        moved = engine.geometry(320, 200)
        engine.set_time(0.0)
        start = engine.geometry(320, 200)
        assert len(moved) == len(start)
        angles = {round(math.atan2(my - sy, mx - sx), 2)
                  for (mx, my, _s), (sx, sy, _t) in zip(moved, start)
                  if abs(mx - sx) + abs(my - sy) > 1e-6}
        assert len(angles) > 4

    with subtests.test(msg="halo_pass_painted_only_with_blur"):
        plain = _engine("drift", "spacr")
        plain.set_time(2.0)
        blurred = _engine("drift", "spacr")
        blurred.set_blur(3.0)
        blurred.set_time(2.0)
        assert _paint(plain) != _paint(blurred)


def test_the_fractal_capacity_guard_follows_measured_costs(qapp, subtests):
    """FractalEngine.afford moves only on measured passes, and _spent is a 16-entry window."""
    with subtests.test(msg="unmeasured_frame_leaves_the_guard"):
        engine = _engine("fractal", "rainbow")
        start = engine.afford()
        engine.advance(0.1)
        assert engine.afford() == start
        engine._spent = [1000.0]
        engine.advance(0.1)
        assert engine.afford() < start

    with subtests.test(msg="measured_costs_are_a_window"):
        engine = _engine("fractal", "rainbow")
        engine._spent = [0.0] * 10
        _paint(engine)
        assert len(engine._spent) == 11
        engine._spent = [0.0] * 20
        _paint(engine)
        assert len(engine._spent) == 16
        assert engine._spent[-1] > 0.0


def _lens(theme="data_art_impulse_lens"):
    engine = make_engine(theme, "spacr", "#191919", seed=684, density=1,
                         blink_percent=0, popup_wave_frequency=0)
    engine.set_time(10.0)
    return engine


def test_column_waves_accept_only_usable_columns(subtests):
    """_add_column_wave clamps and sorts a valid column and refuses malformed ones or disabled ripples."""
    with subtests.test(msg="valid_column_is_clamped_and_sorted"):
        engine = _lens()
        engine._add_column_wave((0.8, -0.5, 1.7))
        assert engine._column_waves == [(10.0, (0.0, 0.8, 1.0), 1.0)]

    unusable = [None, (0.1, 0.5), ("left", 0.5, 0.5), (0.1, object(), 0.5), 7,
                (math.nan, 0.5, 0.5), (0.1, math.inf, 0.5), (0.1, 0.5, -math.inf),
                (0.4, 0.4, 0.5), (1.2, 1.5, 0.5), (-0.3, -0.1, 0.2),
                (.2, .8, .5, .5, .1)]
    for column in unusable:
        with subtests.test(msg=f"unusable_{column!r}"):
            engine = _lens()
            engine._add_column_wave(column)
            assert engine._column_waves == []

    with subtests.test(msg="ripples_off"):
        engine = _lens()
        engine.set_ripples_enabled(False)
        engine._add_column_wave((0.1, 0.5, 0.5))
        assert engine._column_waves == []

    with subtests.test(msg="not_the_impulse_lens"):
        engine = _lens("data_art_tissue_facets")
        engine._add_column_wave((0.1, 0.5, 0.5))
        assert engine._column_waves == []


class _Renderer:
    made = 0

    def __init__(self):
        _Renderer.made += 1
        self._owner = threading.get_ident()
        self.closed = False

    def _close(self):
        self.closed = True


def _warnings(caplog):
    return [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]


def test_the_graphics_choice_is_measured_cached_and_refusals_are_named(qapp, monkeypatch, caplog, subtests):
    """_graphics_automatic/_graphics_renderer follow the cost table, reuse renderers, and name each refusal once."""
    caplog.set_level("WARNING", logger=amb.LOG.name)

    with subtests.test(msg="automatic_follows_the_table_first"):
        measured = _engine("data_art_genetic_advection", density=1.0)
        assert measured._graphics_unmeasured(3840, 2160, work=10 ** 6) is True
        assert measured._graphics_automatic(3840, 2160, work=10 ** 6) is False
        lens = _engine("data_art_impulse_lens", density=0.25)
        assert lens._graphics_unmeasured(1920, 1080, work=10 ** 6) is False
        assert lens._graphics_automatic(1920, 1080) is True

    with subtests.test(msg="unmeasured_rule_wants_dense_4k_work"):
        ribbon = _engine("data_art_point_atlas")
        assert ribbon._graphics_unmeasured(3840, 2160, work=100000) is True
        assert ribbon._graphics_unmeasured(3840, 2160, work=99999) is False
        assert ribbon._graphics_unmeasured(1920, 1080, work=10 ** 7) is False
        paper = _engine("data_art_tissue_facets")
        assert paper._graphics_unmeasured(3840, 2160) is True
        assert paper._graphics_unmeasured(3839, 2160) is False

    with subtests.test(msg="aurora_follows_its_measured_rows"):
        aurora = _engine("aurora", density=1.0)
        assert aurora._graphics_unmeasured(3840, 2160) is True
        assert aurora._graphics_unmeasured(1920, 1080) is False
        assert aurora._graphics_automatic(1920, 1080) is True
        assert amb._measured_graphics_win("aurora", 1920 * 1080, 1.0) is True

    with subtests.test(msg="kept_renderer_reused_until_the_idle_check_expires"):
        checks = []
        with monkeypatch.context() as mp:
            mp.setattr(amb, "_flow_graphics_preflight", lambda stop_event=None: True)
            mp.setattr(amb, "_flow_compute_idle", lambda: checks.append(1) or True)
            mp.setattr(amb, "_FlowPointGraphics", _Renderer)
            _Renderer.made = 0
            engine = _engine("blobs")
            engine._graphics_backend = "gpu"

            def frames():
                first = engine._graphics_renderer(320, 200)
                assert engine._graphics_idle_until > amb.time.monotonic()
                second = engine._graphics_renderer(320, 200)
                engine._graphics_idle_until = 0.0
                third = engine._graphics_renderer(320, 200)
                engine._release_point_graphics()
                return first, second, third

            first, second, third = _on_producer(frames)
        assert first is second is third
        assert _Renderer.made == 1 and len(checks) == 2
        assert first.closed and engine._point_graphics is None

    with subtests.test(msg="constructor_error_is_remembered"):
        attempts = []

        def broken():
            attempts.append(1)
            raise RuntimeError("no context")

        with monkeypatch.context() as mp:
            mp.setattr(amb, "_flow_graphics_preflight", lambda stop_event=None: True)
            mp.setattr(amb, "_flow_compute_idle", lambda: True)
            mp.setattr(amb, "_FlowPointGraphics", broken)
            engine = _engine("aurora")
            engine._graphics_backend = "gpu"
            assert _on_producer(lambda: engine._graphics_renderer(320, 200)) is None
            assert engine._graphics_failed
            assert _on_producer(lambda: engine._graphics_renderer(320, 200)) is None
        assert attempts == [1]

    with subtests.test(msg="each_refusal_named_and_logged_once"):
        caplog.clear()
        with monkeypatch.context() as mp:
            mp.setattr(amb, "_GRAPHICS_REFUSALS_LOGGED", set())
            mp.setattr(amb, "_flow_compute_idle", lambda: True)
            mp.setattr(amb, "_flow_graphics_preflight", lambda stop_event=None: True)
            mp.setattr(amb, "_FlowPointGraphics", _Renderer)
            engine = _engine("blobs")
            engine._graphics_backend = "gpu"

            def reason(width=320, height=200):
                assert engine._graphics_renderer(width, height) is None
                return engine._graphics_refusal

            too_big = "canvas larger than 3840x2160 or big-endian host"
            assert _on_producer(lambda: reason(3841, 2160)) == too_big
            assert _on_producer(lambda: reason(3841, 2160)) == too_big
            mp.setattr(amb, "_decorative_gpu_blocked", lambda: True)
            assert _on_producer(reason) == "Dynamic animation holds the GPU for analysis"
            mp.setattr(amb, "_decorative_gpu_blocked", lambda: False)
            mp.setattr(amb, "_flow_compute_idle", lambda: False)
            assert _on_producer(reason) == "another process is computing on the GPU"
            assert _warnings(caplog) == [
                f"blobs animation stays on the CPU: {too_big}",
                "blobs animation stays on the CPU: Dynamic animation holds the GPU for analysis",
                "blobs animation stays on the CPU: another process is computing on the GPU"]
            engine._graphics_idle_until = 0.0
            mp.setattr(amb, "_flow_compute_idle", lambda: True)

            def accept_then_cpu():
                made = engine._graphics_renderer(320, 200)
                accepted = engine._graphics_refusal
                engine._graphics_backend = "cpu"
                assert engine._graphics_renderer(320, 200) is None
                return made, accepted

            renderer, accepted = _on_producer(accept_then_cpu)
        assert isinstance(renderer, _Renderer) and accepted is None
        assert renderer.closed and engine._graphics_refusal is None

    with subtests.test(msg="automatic_choosing_the_cpu_is_not_logged"):
        caplog.clear()
        with monkeypatch.context() as mp:
            mp.setattr(amb, "_GRAPHICS_REFUSALS_LOGGED", set())
            engine = _engine("drift")
            engine._graphics_backend = "auto"
            assert _on_producer(lambda: engine._graphics_renderer(1920, 1080)) is None
        assert engine._graphics_refusal == "automatic chose the CPU"
        assert _warnings(caplog) == []

    with subtests.test(msg="probe_constructor_and_draw_failures_say_what_failed"):
        caplog.clear()
        with monkeypatch.context() as mp:
            mp.setattr(amb, "_GRAPHICS_REFUSALS_LOGGED", set())
            mp.setattr(amb, "_flow_compute_idle", lambda: True)
            mp.setattr(amb, "_flow_graphics_preflight", lambda stop_event=None: False)
            probed = _engine("aurora")
            probed._graphics_backend = "gpu"
            assert _on_producer(lambda: probed._graphics_renderer(320, 200)) is None
            assert probed._graphics_refusal == amb._graphics_probe_refusal()
            assert probed._graphics_failed
            mp.setattr(amb, "_flow_graphics_preflight", lambda stop_event=None: True)

            def egl_broken():
                raise OSError("EGL_BAD_DISPLAY")

            mp.setattr(amb, "_FlowPointGraphics", egl_broken)
            built = _engine("aurora")
            built._graphics_backend = "gpu"
            assert _on_producer(lambda: built._graphics_renderer(320, 200)) is None
            assert built._graphics_refusal == "context creation failed: OSError: EGL_BAD_DISPLAY"

            class Failing(_Renderer):
                def _draw_blobs(self, *args):
                    raise KeyError("x" * 1000)

            mp.setattr(amb, "_FlowPointGraphics", Failing)
            drawn = _engine("blobs")
            drawn._graphics_backend = "gpu"
            assert _on_producer(lambda: drawn._graphics_frame("_draw_blobs", 320, 200)) is None
        reason = drawn._graphics_refusal
        assert reason.startswith("_draw_blobs failed in _draw_blobs line ")
        assert ": KeyError: " in reason and len(reason) < 700
        assert drawn._graphics_failed and drawn._point_graphics is None
        assert len(_warnings(caplog)) == 3

    with subtests.test(msg="random_palette_points_release_a_renderer"):
        class Recorder:
            def __init__(self):
                self._owner = threading.get_ident()
                self.closed = False
                self.draws = []

            def _draw(self, *args):
                self.draws.append(args)

            def _close(self):
                self.closed = True

        engine = make_engine("data_art_impulse_lens", "random", "#191919", seed=3)
        engine.set_max_pixels(320 * 200)
        engine._graphics_backend = "gpu"

        def run():
            renderer = engine._point_graphics = Recorder()
            return engine.shade(320, 200), renderer

        frame, renderer = _on_producer(run)
        assert renderer.closed and renderer.draws == []
        assert engine._point_graphics is None
        assert (frame.width(), frame.height()) == engine.buffer_size(320, 200)


@pytest.fixture
def gpu(monkeypatch):
    """The recording moderngl stand-in, with every GPU gate open."""
    monkeypatch.setattr(amb, "_mac_radeon_graphics", lambda: True)
    monkeypatch.setattr(amb, "_FLOW_GRAPHICS_PROBE", None, raising=False)
    monkeypatch.setattr(amb, "_graphics_context_options",
                        lambda: {"standalone": True, "require": 430})
    module = _fake_moderngl()
    monkeypatch.setitem(sys.modules, "moderngl", module)
    monkeypatch.setattr(amb, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(amb, "_flow_graphics_preflight", lambda stop_event=None: True)
    return module


def _gpu_engine(theme, **kwargs):
    engine = make_engine(theme, amb.default_palette_for(theme), "#191919", seed=684, **kwargs)
    engine.set_max_pixels(320 * 200)
    engine.set_time(9.5)
    engine._graphics_backend = "gpu"
    return engine


def _shade_and_record(engine):
    def run():
        image = engine.shade(320, 200)
        renderer = engine._point_graphics
        renders = list(renderer._context.renders)
        names = {id(program): name
                 for name, program in renderer.__dict__.get("_programs", {}).items()}
        names[id(renderer._program)] = "points"
        engine._release_point_graphics()
        return image, [names[id(render.program)] for render in renders], renders
    return _on_producer(run)


def test_gpu_capable_engines_hand_their_frame_data_to_the_renderer(qapp, gpu, subtests):
    """Drift, fungal growth and impulse-lens frames reach the renderer as the programs they need."""
    with subtests.test(msg="stratified_stars_are_a_transparent_disc_layer"):
        image, names, renders = _shade_and_record(_gpu_engine("drift"))
        assert image.format() == QImage.Format_ARGB32_Premultiplied
        assert image.pixelColor(0, 0).alpha() == 0
        assert set(names) == {"discs"}
        assert sum(render.instances for render in renders) > 0

    with subtests.test(msg="growth_strokes_and_tips"):
        engine = _gpu_engine("data_art_fungal_growth")
        engine.set_time(41.5)
        image, names, _renders = _shade_and_record(engine)
        assert (image.width(), image.height()) == engine.buffer_size(320, 200)
        assert names[-1] == "discs"
        assert {"hairlines", "capsules"} & set(names)

    with subtests.test(msg="point_material_frames_flash_their_blinking_dots"):
        engine = _gpu_engine("data_art_impulse_lens", blink_percent=100.0, density=0.25)
        image, names, renders = _shade_and_record(engine)
        render, = renders
        assert names == ["subpixel_grains"]
        assert render.mode == "TRIANGLE_STRIP" and render.instances > 0
        points = np.floor(np.frombuffer(render.data, np.float32).reshape(-1, 3)).astype(np.int64)
        width, height = engine.buffer_size(320, 200)
        inside = points[(points[:, 0] >= 0) & (points[:, 0] < width)
                        & (points[:, 1] >= 0) & (points[:, 1] < height)]
        assert len(inside)
        assert (width, height) == (image.width(), image.height())
        pixels = np.frombuffer(image.constBits(), np.uint32,
                               count=width * height).reshape(height, width) & 0xFFFFFF
        lit = np.argwhere(pixels != 0)
        assert len(lit)
        near = np.zeros((height, width), bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xs, ys = inside[:, 0] + dx, inside[:, 1] + dy
                keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                near[ys[keep], xs[keep]] = True
        assert near[lit[:, 0], lit[:, 1]].all()


def _lookup(dark):
    ramp = np.arange(256, dtype=np.uint32)
    if not dark:
        ramp = 255 - ramp
    return (ramp * np.uint32(0x010101)) | np.uint32(0xff000000)


def _stamp(kernel, xs, ys, values, width, height, dark):
    lookup = _lookup(dark)
    flat = np.full(width * height, lookup[0], dtype=np.uint32)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if kernel is None:
        amb._subpixel_grains_numpy(np, flat, xs, ys, values, lookup, width, height, dark)
    else:
        kernel(flat, xs, ys, values, lookup, width, height, dark, np.zeros(16, dtype=np.float64))
    return flat


def test_subpixel_grain_kernels_agree_and_keep_the_brighter_value(monkeypatch, subtests):
    """_scatter_subpixel_grains equals its numpy twin, keeps the max, and the field shades without the compiled kernel."""
    for dark in (True, False):
        with subtests.test(msg=f"python_matches_numpy_dark_{dark}"):
            rng = np.random.default_rng(12)
            width, height = 24, 18
            xs = np.concatenate([rng.uniform(-1.5, width + 1.5, 120),
                                 [3.0, 3.5, 7.0, 0.0, width + 0.9, -1.2]])
            ys = np.concatenate([rng.uniform(-1.5, height + 1.5, 120),
                                 [4.0, 4.5, 7.0, 0.0, -0.4, height + 0.7]])
            values = np.concatenate([rng.uniform(0.0, 2.5, 120), [1.0, 1.0, 3.0, 0.5, 1.0, 1.0]])
            python = _stamp(amb._scatter_subpixel_grains, xs, ys, values, width, height, dark)
            twin = _stamp(None, xs, ys, values, width, height, dark)
            assert np.array_equal(python, twin)
            assert (python != _lookup(dark)[0]).sum() > 100
            assert (python == _lookup(dark)[255]).any()

    with subtests.test(msg="stamped_twice_keeps_the_brighter"):
        once = _stamp(amb._scatter_subpixel_grains, [6.0], [6.0], [2.0], 12, 12, True)
        twice = _stamp(amb._scatter_subpixel_grains, [6.0, 6.3], [6.0, 6.2], [2.0, 0.2], 12, 12, True)
        assert (twice >= once).all()
        assert twice[5 * 12 + 5] == once[5 * 12 + 5]

    for case_id, kernel in [("no_grains_numpy", None), ("no_grains_python", amb._scatter_subpixel_grains)]:
        with subtests.test(msg=case_id):
            flat = _stamp(kernel, [], [], [], 8, 8, True)
            assert (flat == _lookup(True)[0]).all()

    with subtests.test(msg="field_without_the_compiled_kernel"):
        engine = make_engine("data_art_impulse_lens", "spacr", DARK, seed=7, density=0.25)
        with monkeypatch.context() as mp:
            mp.setattr(amb, "_ready_subpixel_scatter", lambda: None)
            engine.advance(0.4)
            image = engine.shade(160, 100)
        assert image.width() == 160


def _grain_frames(engine, width=9, height=9):
    """A full grain with spread, plus reference grains at the axial and diagonal levels."""
    alpha = engine.alpha_scale()
    full = engine._point_material(width, height, [4], [4], [1.0 / alpha], spread=True)
    bare = engine._point_material(width, height, [4], [4], [1.0 / alpha], spread=False)
    axial = engine._point_material(width, height, [4], [4], [173 / 255 / alpha], spread=False)
    diagonal = engine._point_material(width, height, [4], [4], [61 / 255 / alpha], spread=False)
    empty = engine._point_material(width, height, [], [], [], spread=False)
    return full, bare, axial, diagonal, empty


def test_cpu_point_materials_stamp_exact_grains_without_compiled_kernels(qapp, monkeypatch, subtests):
    """_point_material/_colored_point_material spread a grain to 0.68/0.24 neighbours on the NumPy path."""
    for palette in ("spacr", "random"):
        for background in (DARK, "#f4f4f4"):
            with subtests.test(msg=f"numpy_path_{palette}_{background}"):
                with monkeypatch.context() as mp:
                    mp.setattr(amb, "_ready_packed_scatter", lambda: None)
                    mp.setattr(amb, "_ready_colored_scatter", lambda: None)
                    engine = _engine("data_art_point_atlas", palette, background, blink_percent=0)
                    engine._graphics_backend = "cpu"
                    full, bare, axial, diagonal, empty = _grain_frames(engine)
                assert engine._random_palette is (palette == "random")
                background_pixel = empty.pixel(0, 0)
                assert background_pixel == (0xff000000 if background == DARK else 0xffffffff)
                assert full.pixel(4, 4) == bare.pixel(4, 4) != background_pixel
                for x, y in ((3, 4), (5, 4), (4, 3), (4, 5)):
                    assert full.pixel(x, y) == axial.pixel(4, 4)
                    assert bare.pixel(x, y) == background_pixel
                for x, y in ((3, 3), (5, 5), (3, 5), (5, 3)):
                    assert full.pixel(x, y) == diagonal.pixel(4, 4)
                assert full.pixel(0, 0) == background_pixel

    for background in (DARK, "#f4f4f4"):
        for spread in (False, True):
            with subtests.test(msg=f"raising_colour_kernel_falls_back_{background}_spread_{spread}"):
                def raising(*args):
                    raise RuntimeError("kernel rejected its arguments")

                with monkeypatch.context() as mp:
                    mp.setattr(amb, "_COLORED_SCATTER", raising)
                    mp.setattr(amb, "_COLORED_SCATTER_FAILED", False)
                    mp.setattr(amb, "_ready_colored_scatter", lambda: raising)
                    engine = _engine("data_art_point_atlas", "random", background, blink_percent=0)
                    engine._graphics_backend = "cpu"
                    failed = engine._point_material(9, 9, [4], [4], [1.0 / engine.alpha_scale()],
                                                    spread=spread)
                    assert amb._COLORED_SCATTER is None and amb._COLORED_SCATTER_FAILED is True
                    mp.setattr(amb, "_ready_colored_scatter", lambda: None)
                    fallback = engine._point_material(9, 9, [4], [4], [1.0 / engine.alpha_scale()],
                                                      spread=spread)
                assert failed == fallback

    with subtests.test(msg="ready_colored_scatter_warms_once_then_stays_off_after_failure"):
        warmed = []
        with monkeypatch.context() as mp:
            mp.setattr(amb, "_COLORED_SCATTER", None)
            mp.setattr(amb, "_COLORED_SCATTER_FAILED", False)
            mp.setattr(amb, "_ready_packed_scatter", lambda: warmed.append(1))
            assert amb._ready_colored_scatter() is None
            assert warmed == [1]
            mp.setattr(amb, "_COLORED_SCATTER_FAILED", True)
            assert amb._ready_colored_scatter() is None
            assert warmed == [1]


def _power_supply(root, entries):
    root.mkdir()
    for name, attributes in entries.items():
        folder = root / name
        folder.mkdir()
        for attribute, value in attributes.items():
            (folder / attribute).write_text(value + "\n", encoding="ascii")
    return root


def test_the_battery_state_holds_the_backdrop_only_when_unplugged(tmp_path, monkeypatch, subtests):
    """_linux_on_battery/_on_battery answer True only for a real discharging battery with no charger."""
    battery = {"type": "Battery", "status": "Discharging"}
    cases = [
        ("device_scope_battery_is_ignored", {"BAT_DEV": dict(battery, scope="Device")}, False),
        ("mains_only", {"AC": {"type": "Mains", "online": "1"}}, False),
        ("discharging_with_charger_online", {"BAT0": battery, "AC": {"type": "Mains", "online": "1"}}, False),
        ("discharging_with_charger_offline", {"BAT0": battery, "AC": {"type": "USB_C", "online": "0"}}, True),
        ("discharging_battery_only", {"BAT0": battery}, True),
        ("charging_battery_only", {"BAT0": {"type": "Battery", "status": "Charging"}}, False),
    ]
    for case_id, entries, expected in cases:
        with subtests.test(msg=f"linux_{case_id}"):
            assert amb._linux_on_battery(_power_supply(tmp_path / case_id, entries)) is expected

    with subtests.test(msg="linux_unreadable_folder"):
        assert amb._linux_on_battery(tmp_path / "missing") is False

    def on_battery(mp, platform, linux=None, psutil=None):
        mp.delenv("SPACR_BATTERY_PAUSE", raising=False)
        mp.setattr(amb, "_BATTERY", [float("-inf"), False])
        mp.setattr(amb.sys, "platform", platform)
        if linux is not None:
            mp.setattr(amb, "_linux_on_battery", linux)
        if psutil is not None:
            mp.setitem(sys.modules, "psutil", psutil)
        return amb._on_battery()

    with subtests.test(msg="linux_answer_is_cached_for_the_ttl"):
        with monkeypatch.context() as mp:
            assert on_battery(mp, "linux", linux=lambda: True) is True
            mp.setattr(amb, "_linux_on_battery", lambda: False)
            assert amb._on_battery() is True
            assert amb._BATTERY[1] is True

    from types import SimpleNamespace

    for case_id, reading, expected in [("psutil_unplugged", SimpleNamespace(power_plugged=False), True),
                                       ("psutil_plugged", SimpleNamespace(power_plugged=True), False),
                                       ("psutil_unknown_plug", SimpleNamespace(power_plugged=None), False),
                                       ("psutil_no_battery", None, False)]:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                fake = SimpleNamespace(sensors_battery=lambda: reading)
                assert on_battery(mp, "darwin", psutil=fake) is expected

    with subtests.test(msg="a_failing_reading_is_not_on_battery"):
        def broken():
            raise OSError("power supply unreadable")

        with monkeypatch.context() as mp:
            assert on_battery(mp, "linux", linux=broken) is False
            assert amb._BATTERY[1] is False

    with subtests.test(msg="the_pause_switch_keeps_the_animation"):
        with monkeypatch.context() as mp:
            mp.setenv("SPACR_BATTERY_PAUSE", "0")
            mp.setattr(amb, "_BATTERY", [float("inf"), True])
            assert amb._on_battery() is False
