"""N684: graphics paths for blobs, stratified, aurora, growth and spinn keep a CPU fallback.

No graphics driver is used here: renderers are stand-ins that record which
thread and method they were given. The shaders themselves are checked with
tools/verify_animation_gpu_parity.py (Mesa software device or a real GPU).
"""
import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QImage

from spacr.qt import preferences
from spacr.qt.widgets import ambient

CPU_THEMES = {
    "blobs": "_draw_blobs",
    "drift": "_draw_stars",
    "aurora": "_draw_aurora",
    "data_art_fungal_growth": "_draw_filaments",
    "data_art_tissue_facets": "_draw_tiles",
}


@pytest.fixture(autouse=True)
def no_policy(monkeypatch):
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)


def make(theme, background="#191919", density=1.0):
    engine = ambient.make_engine(theme, ambient.default_palette_for(theme), background,
                                 seed=684, density=density)
    engine.set_max_pixels(320 * 200)
    engine.set_time(9.5)
    if theme == "data_art_tissue_facets":
        engine.pointer = (0.5, 0.5)
        engine.gravity_radius = 0.4
    return engine


def on_producer(function):
    box = {}

    def run():
        box["value"] = function()

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    return box["value"]


class Recorder:
    """A renderer stand-in returning a solid frame from any theme method."""

    made = []

    def __init__(self):
        self._owner = threading.get_ident()
        self.calls = []
        self.closed = False
        Recorder.made.append(self)

    def __getattr__(self, name):
        if not name.startswith("_draw"):
            raise AttributeError(name)

        def draw(width, height, *args):
            self.calls.append((name, threading.get_ident(), args))
            image = QImage(width, height, QImage.Format_RGB32)
            image.fill(QColor("magenta"))
            return image
        return draw

    def _close(self):
        self.closed = True


@pytest.fixture
def recorder(monkeypatch):
    Recorder.made = []
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)
    monkeypatch.setattr(ambient, "_FlowPointGraphics", Recorder)
    return Recorder


@pytest.mark.parametrize("theme,method", sorted(CPU_THEMES.items()))
def test_gpu_mode_draws_the_theme_on_the_producer_thread(theme, method, recorder, qapp):
    engine = make(theme)
    engine._graphics_backend = "gpu"
    def frame_then_release():
        image = engine.shade(320, 200)
        engine._release_point_graphics()
        return image

    image = on_producer(frame_then_release)
    renderer, = recorder.made
    assert [call[0] for call in renderer.calls] == [method]
    assert renderer.calls[0][1] == renderer._owner != threading.get_ident()
    assert image.pixelColor(5, 5) in (QColor("magenta"), QColor("#ff00ff"))
    assert renderer.closed and engine._point_graphics is None


@pytest.mark.parametrize("theme", sorted(CPU_THEMES))
@pytest.mark.parametrize("mode", ["cpu", "auto"])
def test_cpu_and_automatic_keep_small_frames_on_the_cpu(theme, mode, monkeypatch, qapp):
    expected = on_producer(lambda: make(theme).shade(320, 200))
    engine = make(theme)
    engine._graphics_backend = mode

    def forbidden(*args, **kwargs):
        pytest.fail("a CPU frame must not probe or build graphics")

    monkeypatch.setattr(ambient, "_flow_graphics_preflight", forbidden)
    monkeypatch.setattr(ambient, "_FlowPointGraphics", forbidden)
    assert on_producer(lambda: engine.shade(320, 200)) == expected


@pytest.mark.parametrize("theme", sorted(CPU_THEMES))
def test_a_failing_draw_falls_back_to_identical_cpu_pixels(theme, monkeypatch, qapp):
    expected = on_producer(lambda: make(theme).shade(320, 200))
    engine = make(theme)
    engine._graphics_backend = "gpu"
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)

    class Broken(Recorder):
        def __getattr__(self, name):
            def draw(*args):
                raise RuntimeError("shader compile failed")
            return draw

    monkeypatch.setattr(ambient, "_FlowPointGraphics", Broken)
    assert on_producer(lambda: engine.shade(320, 200)) == expected
    assert engine._graphics_failed
    assert engine._point_graphics is None
    assert on_producer(lambda: engine.shade(320, 200)) is not None


@pytest.mark.parametrize("theme", sorted(CPU_THEMES))
def test_gui_thread_shading_never_uses_graphics(theme, recorder, qapp):
    engine = make(theme)
    engine._graphics_backend = "gpu"
    engine.shade(320, 200)
    assert recorder.made == []


@pytest.mark.parametrize("theme", sorted(CPU_THEMES))
def test_dynamic_animation_block_releases_and_refuses(theme, recorder, monkeypatch, qapp):
    engine = make(theme)
    engine._graphics_backend = "gpu"

    def both():
        first = engine.shade(320, 200)
        made = engine._point_graphics
        monkeypatch.setattr(ambient, "_decorative_gpu_blocked", lambda: True)
        second = engine.shade(320, 200)
        return first, made, second

    first, made, second = on_producer(both)
    assert made is not None and made.closed
    assert engine._point_graphics is None and not engine._graphics_failed
    assert second != first


@pytest.mark.parametrize("theme", sorted(CPU_THEMES))
def test_busy_external_gpu_keeps_cpu(theme, recorder, monkeypatch, qapp):
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: False)
    expected = on_producer(lambda: make(theme).shade(320, 200))
    engine = make(theme)
    engine._graphics_backend = "gpu"
    assert on_producer(lambda: engine.shade(320, 200)) == expected
    assert recorder.made == []


def test_automatic_selects_only_4k_aurora_and_spinning_paper(qapp):
    for theme in CPU_THEMES:
        engine = make(theme)
        assert not engine._graphics_automatic(1920, 1080)
        assert not engine._graphics_automatic(320, 200)
        assert engine._graphics_automatic(3840, 2160) is (
            theme in ("aurora", "data_art_tissue_facets"))


def test_graphics_only_preparation_is_skipped_on_the_cpu(monkeypatch, qapp):
    engine = make("data_art_fungal_growth")
    engine._graphics_backend = "cpu"
    monkeypatch.setattr(engine, "_fungal_strokes",
                        lambda *a: pytest.fail("strokes are built only for graphics"))
    blobs = make("blobs")
    monkeypatch.setattr(blobs, "_blob_rows",
                        lambda *a: pytest.fail("rows are built only for graphics"))
    on_producer(lambda: (engine.shade(320, 200), blobs.shade(320, 200)))


def test_spinn_layout_advances_once_per_frame_even_after_refusal(monkeypatch, qapp):
    engine = make("data_art_tissue_facets")
    engine._graphics_backend = "gpu"
    calls = []
    original = engine._tissue_layout

    def counted(width, height):
        calls.append((width, height))
        return original(width, height)

    monkeypatch.setattr(engine, "_tissue_layout", counted)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: False)
    on_producer(lambda: engine.shade(320, 200))
    assert len(calls) == 1 and engine._graphics_failed


def test_aurora_gpu_layer_matches_the_cpu_strips(qapp):
    engine = make("aurora")
    layers = engine._aurora_layers(320, 200)
    assert layers
    for layer in layers:
        columns, lookup, per_column, vertices = engine._aurora_gpu_layer(layer)
        assert columns.shape == (layer["width"], 4)
        assert vertices.shape == (6 * len(layer["strips"]), 4)
        m11, m12, m21, m22, dx, dy, source_x, _width = layer["strips"][0]
        assert vertices[0].tolist() == pytest.approx([dx, dy, source_x, 0.0])
        assert not per_column and lookup.shape == (256, 3)


def test_growth_strokes_follow_qt_groups_and_caps(qapp):
    engine = make("data_art_fungal_growth")
    engine.set_time(41.5)
    groups, segments, tips, dark = engine._fungal_strokes(320, 200)
    paths, _mature, cpu_tips = engine._fungal_paths(320, 200)
    assert len(groups) == len(paths) and len(tips) == len(cpu_tips) and dark
    assert sum(group[1] for group in groups) == len(segments)
    for first, count, *_rest in groups:
        caps = segments[first:first + count, 5]
        assert caps.min() >= 0 and caps.max() <= 3
        assert (caps % 2 == 1).sum() == (caps >= 2).sum()


def test_flattening_keeps_straight_curves_whole_and_splits_bends():
    straight = ambient._flatten_quadratic(0.0, 0.0, 5.0, 0.01, 10.0, 0.0)
    assert straight == [(0.0, 0.0), (10.0, 0.0)]
    bent = ambient._flatten_quadratic(0.0, 0.0, 50.0, 80.0, 100.0, 0.0)
    assert len(bent) > 3 and bent[0] == (0.0, 0.0) and bent[-1] == (100.0, 0.0)


def test_stratified_cpu_frame_composites_like_the_direct_paint(qapp):
    from PySide6.QtGui import QPainter

    engine = make("drift")
    direct = QImage(320, 200, QImage.Format_RGB32)
    direct.fill(QColor("#191919"))
    painter = QPainter(direct)
    engine.paint(painter, 320, 200)
    painter.end()
    shaded = QImage(320, 200, QImage.Format_RGB32)
    shaded.fill(QColor("#191919"))
    painter = QPainter(shaded)
    engine.blit(painter, engine.shade(320, 200), 320, 200)
    painter.end()
    assert direct == shaded


@pytest.fixture
def store(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "gpu.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: settings)
    return settings


def _stratified(qtbot, backend, possible, monkeypatch):
    from PySide6.QtWidgets import QWidget

    monkeypatch.setattr(ambient, "_flow_graphics_possible", lambda: possible)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: False)
    preferences._set_flow_graphics_backend(backend)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 200)
    widget = ambient.AmbientWidget(host, theme="drift", seed=3, blink_percent=0)
    widget.resize(host.size())
    host.show()
    widget.show()
    qtbot.waitExposed(widget)
    return host, widget


@pytest.mark.parametrize("backend,possible,threaded", [
    ("gpu", True, True), ("gpu", False, False), ("auto", True, False), ("cpu", True, False)])
def test_stratified_gets_a_shading_thread_only_for_graphics(
        backend, possible, threaded, store, qtbot, monkeypatch):
    _host, widget = _stratified(qtbot, backend, possible, monkeypatch)
    widget.start()
    try:
        assert widget.shading_thread_alive() is threaded
    finally:
        widget.stop()
    assert not widget.shading_thread_alive()


def test_stratified_backend_change_moves_it_on_and_off_its_thread(store, qtbot, monkeypatch):
    _host, widget = _stratified(qtbot, "cpu", True, monkeypatch)
    widget.start()
    try:
        assert not widget.shading_thread_alive()
        widget._set_graphics_backend("gpu")
        assert widget.shading_thread_alive()
        deadline = time.monotonic() + 5
        while widget.frames_shaded() < 2 and time.monotonic() < deadline:
            qtbot.wait(20)
        assert widget.frames_shaded() >= 2
        assert widget._engine._graphics_failed
        widget._set_graphics_backend("cpu")
        assert not widget.shading_thread_alive()
    finally:
        widget.stop()


@pytest.mark.parametrize("theme", sorted(CPU_THEMES))
def test_widget_hands_the_backend_choice_to_every_capable_theme(theme, store, qtbot, monkeypatch):
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    preferences._set_flow_graphics_backend("auto")
    widget = ambient.AmbientWidget(theme=theme, seed=1)
    qtbot.addWidget(widget)
    assert widget._engine._graphics_backend == "auto"
    widget._set_graphics_backend("cpu")
    assert widget._engine._graphics_backend == "cpu"
    widget._theme = "blobs" if theme != "blobs" else "aurora"
    widget._rebuild_engine()
    assert widget._engine._graphics_backend == "cpu"
