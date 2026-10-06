"""Image and pointer contracts for the six offered data-art materials."""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QCursor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt.widgets import ambient


FAMILIES = (
    "point_atlas", "tissue_facets", "chromatin_ribbon",
    "genetic_advection", "impulse_lens", "fungal_growth",
)
INTERACTIVE = frozenset(("point_atlas", "genetic_advection",
                         "impulse_lens"))
BACKGROUND = "#101418"


def _engine(family: str, *, seed: int = 41, palette: str = "spacr"):
    """Build one deterministic material with explicit backdrop controls."""
    return ambient.make_engine(f"data_art_{family}", palette, BACKGROUND,
                               seed=seed)


def _frame(engine, width: int = 384, height: int = 216) -> QImage:
    """Composite one owned frame on the same page colour as the widget."""
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(BACKGROUND))
    painter = QPainter(image)
    engine.paint(painter, width, height)
    painter.end()
    return image


def _digest(image: QImage) -> str:
    """Hash all rendered pixels so subtle animated material changes count."""
    return hashlib.sha256(image.bits().tobytes()).hexdigest()


def test_data_art_registry_has_six_distinct_named_materials():
    """The catalog exposes six families without pointer duplicates."""
    keys = {theme for theme in ambient.AMBIENT_THEMES
            if theme.startswith("data_art_")}
    assert keys == {f"data_art_{family}" for family in FAMILIES}
    for family in FAMILIES:
        key = f"data_art_{family}"
        engine = _engine(family)
        if family != "fungal_growth":
            assert engine.family == family
        assert engine.name == key
        assert getattr(engine, "interactive", False) is (family in INTERACTIVE)
        assert ambient.animation_label(key) != key
        assert ambient.animation_note(key).endswith(".")
        assert set(ambient.palettes_for(key)) == (
            set(ambient.PALETTE_SETS) - {ambient.SPACEOUT_PALETTE})


@pytest.mark.parametrize("family", FAMILIES)
def test_every_material_is_seeded_clocked_and_reproducible(family):
    """A material's pixels follow its seed and clock without random drift."""
    first = _engine(family, seed=71)
    same = _engine(family, seed=71)
    other = _engine(family, seed=72)
    for engine in (first, same, other):
        engine.set_time(8.0)
    image = _digest(_frame(first))
    assert _digest(_frame(same)) == image
    assert _digest(_frame(other)) != image
    first.set_time(17.0)
    assert _digest(_frame(first)) != image


def test_material_keys_produce_unique_full_frame_hashes():
    """Separate registrations cannot silently resolve to one engine frame."""
    digests = set()
    for family in FAMILIES:
        engine = _engine(family, seed=21)
        engine.set_time(5.0)
        digests.add(_digest(_frame(engine)))
    assert len(digests) == len(FAMILIES)


@pytest.mark.parametrize("family", tuple(sorted(INTERACTIVE)))
def test_pointer_bends_interactive_materials_and_lens_wake_expires(family):
    """The gravity lens keeps a brief wake; other pointers clear at once."""
    engine = _engine(family, seed=17)
    engine.set_time(6.0)
    idle = _digest(_frame(engine))
    engine.set_pointer((0.22, 0.68))
    assert engine.pointer == (0.22, 0.68)
    assert _digest(_frame(engine)) != idle
    engine.set_pointer(None)
    if family == "impulse_lens":
        assert _digest(_frame(engine)) != idle
        engine.set_time(11.1)
        reference = _engine(family, seed=17)
        reference.set_time(11.1)
        assert _digest(_frame(engine)) == _digest(_frame(reference))
    else:
        assert _digest(_frame(engine)) == idle


def test_pointer_rejects_nonfinite_values_and_static_material_ignores_it():
    """Bad or irrelevant cursor coordinates cannot enter material state."""
    moving = _engine("impulse_lens")
    moving.set_pointer((float("nan"), 0.5))
    assert moving.pointer is None
    moving.set_pointer((-1.0, 2.0))
    assert moving.pointer == (0.0, 1.0)
    static = _engine("tissue_facets")
    baseline = _digest(_frame(static))
    static.set_pointer((0.2, 0.8))
    assert static.pointer is None
    assert _digest(_frame(static)) == baseline


def test_data_art_keeps_native_shade_resolution_within_existing_pixel_cap():
    """4K display shading retains full 1080p detail without raising limits."""
    for family in FAMILIES:
        engine = _engine(family)
        assert engine.buffer_size(3840, 2160) == (1920, 1080)
        assert engine.buffer_size(1920, 1080) == (1920, 1080)
        assert engine.max_pixels == ambient.BUFFER_MAX_PIXELS


def test_material_cache_follows_size_density_and_resolution_controls():
    """Changing a detail control discards the old size-bound material."""
    engine = _engine("tissue_facets")
    _frame(engine)
    assert engine._material_cache
    engine.set_density(1.4)
    assert not engine._material_cache
    _frame(engine)
    engine.set_size(1.3)
    assert not engine._material_cache
    _frame(engine)
    engine.set_resolution(1.2)
    assert not engine._material_cache


@pytest.mark.parametrize("family", ("tissue_facets",))
def test_canvas_resizes_release_previous_material_images_and_grids(family):
    """Repeated window sizes cannot accumulate old full-resolution rasters."""
    engine = _engine(family)
    for width, height in ((384, 216), (512, 288), (640, 360), (384, 216)):
        engine.shade(width, height)
        assert engine._material_cache
        sizes = {(key[1], key[2]) for key in engine._material_cache
                 if isinstance(key, tuple) and len(key) >= 3
                 and isinstance(key[1], int) and isinstance(key[2], int)}
        assert sizes == {engine.buffer_size(width, height)}


@pytest.mark.parametrize("family", ("point_atlas", "impulse_lens"))
def test_density_changes_grain_population_and_restores_default_frame(family):
    """Density adds actual points and can return to its seeded composition."""
    engine = _engine(family, seed=11, palette="mono")
    engine.set_time(7.0)
    default = _digest(_frame(engine))
    engine.set_density(0.25)
    sparse = _digest(_frame(engine))
    engine.set_density(3.0)
    dense = _digest(_frame(engine))
    assert len({sparse, default, dense}) == 3
    engine.set_density(1.0)
    assert _digest(_frame(engine)) == default


def test_geometry_and_isolated_grain_material_have_bounded_edges():
    """Empty canvases stay empty and an isolated grain clips at the edge."""
    engine = _engine("point_atlas")
    assert engine.geometry(0, 20) == ()
    assert engine.geometry(20, 0) == ()
    assert len(engine.geometry(320, 200)) == 64
    image = engine._point_material(12, 9, [-5, 0, 11, 20],
                                   [0, 0, 8, 8], [1.0] * 4)
    assert image.width() == 12 and image.height() == 9
    assert image.pixelColor(0, 0) != QColor("#000000")
    assert image.pixelColor(11, 8) != QColor("#000000")
    core_only = engine._point_material(12, 9, [6], [4], [1.0], spread=False)
    assert core_only.pixelColor(6, 4) != QColor("#000000")
    assert core_only.pixelColor(7, 4) == QColor("#000000")


@pytest.mark.parametrize("family", ("tissue_facets", "chromatin_ribbon"))
def test_light_page_material_preserves_multiplicative_identity(family):
    """A material remains visible while its light buffer stays page-safe."""
    engine = ambient.make_engine(f"data_art_{family}", "mono", "#f6f7f9",
                                 seed=41)
    engine.set_time(7.0)
    image = QImage(384, 216, QImage.Format_RGB32)
    image.fill(QColor("#f6f7f9"))
    painter = QPainter(image)
    engine.paint(painter, 384, 216)
    painter.end()
    assert len({image.pixelColor(x, y).rgb()
                for x in range(0, 384, 24)
                for y in range(0, 216, 24)}) > 1


def test_unknown_private_material_family_is_rejected_before_configuration():
    """A misspelled family must not silently select a different material."""
    with pytest.raises(ValueError, match="unknown data art family"):
        ambient._DataArtEngine(["#ffffff"], BACKGROUND, family="unknown")


@pytest.mark.parametrize("family", ("tissue_facets", "chromatin_ribbon"))
def test_raster_materials_survive_tiny_and_narrow_canvases(family):
    """A transient 1-pixel or narrow resize cannot poison the next frame."""
    engine = _engine(family)
    for width, height in ((1, 1), (3, 400)):
        image = _frame(engine, width, height)
        assert image.width() == width and image.height() == height
    assert _frame(engine).width() == 384


def test_a_failed_worker_shade_releases_its_qpainter(monkeypatch):
    """A rendering exception leaves the reusable QImage safe to repaint."""
    engine = _engine("tissue_facets")

    def fail_field(painter, width, height):
        """Simulate one failed material calculation inside an active painter."""
        raise RuntimeError("injected shade failure")

    monkeypatch.setattr(engine, "_paint_field", fail_field)
    with pytest.raises(RuntimeError, match="injected shade failure"):
        engine.shade(32, 32)
    painter = QPainter(engine._buffer)
    assert painter.isActive()
    painter.end()


def test_pointer_poll_is_limited_to_active_window_and_widget(qtbot, monkeypatch):
    """Hovering another window or outside the backdrop gives no pointer."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 200)
    widget = ambient.AmbientWidget(host, theme="data_art_point_atlas",
                                   palette="midnight", background=BACKGROUND)
    widget.setGeometry(host.rect())
    host.show()
    QApplication.processEvents()
    widget._animating = True
    monkeypatch.setattr(ambient.QApplication, "activeWindow", lambda: host)
    monkeypatch.setattr(ambient.QApplication, "widgetAt", lambda position: widget)
    monkeypatch.setattr(QCursor, "pos", lambda: widget.mapToGlobal(QPoint(160, 100)))
    assert widget._data_art_pointer_for_tick() == pytest.approx(
        ((160.5 / 320), (100.5 / 200)))
    monkeypatch.setattr(ambient.QApplication, "activeWindow", lambda: None)
    assert widget._data_art_pointer_for_tick() is None
    monkeypatch.setattr(ambient.QApplication, "activeWindow", lambda: host)
    monkeypatch.setattr(QCursor, "pos", lambda: widget.mapToGlobal(QPoint(400, 100)))
    assert widget._data_art_pointer_for_tick() is None
    monkeypatch.setattr(ambient.QApplication, "widgetAt", lambda position: None)
    monkeypatch.setattr(QCursor, "pos", lambda: widget.mapToGlobal(QPoint(160, 100)))
    assert widget._data_art_pointer_for_tick() is None
    monkeypatch.setattr(ambient.QApplication, "widgetAt", lambda position: widget)
    monkeypatch.setattr(QCursor, "pos", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert widget._data_art_pointer_for_tick() is None
    monkeypatch.setattr(QCursor, "pos", lambda: widget.mapToGlobal(QPoint(160, 100)))
    widget.stop()
    widget._animating = True
    monkeypatch.setattr(widget, "_follow_the_run", lambda: None)
    widget._on_tick()
    assert widget._engine.pointer == pytest.approx(
        ((160.5 / 320), (100.5 / 200)))
    widget._animating = False
    assert widget._data_art_pointer_for_tick() is None
    widget.stop()


def test_static_material_tick_never_polls_cursor(qtbot, monkeypatch):
    """A noninteractive material advances without cursor or window work."""
    widget = ambient.AmbientWidget(theme="data_art_tissue_facets",
                                   palette="lowsun", background=BACKGROUND)
    qtbot.addWidget(widget)
    widget.stop()
    monkeypatch.setattr(widget, "_follow_the_run", lambda: None)

    def unexpected_poll():
        """Fail if a static theme performs an unnecessary cursor lookup."""
        raise AssertionError("static material polled the cursor")

    monkeypatch.setattr(widget, "_data_art_pointer_for_tick", unexpected_poll)
    before = widget._engine.frames
    widget._on_tick()
    assert widget._engine.frames == before + 1
