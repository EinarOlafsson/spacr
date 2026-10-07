"""Scalar Qt overloads retain exact curves and native palette pixels."""

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainterPath

from spacr.qt import preferences
from spacr.qt.widgets import ambient


class _PointArgumentPath(QPainterPath):
    """Route production scalar coordinates through the preceding Qt overload."""

    def moveTo(self, x, y):
        return super().moveTo(QPointF(x, y))

    def quadTo(self, cx, cy, x, y):
        return super().quadTo(QPointF(cx, cy), QPointF(x, y))


@pytest.mark.parametrize("progress", [0.125, 0.5, 1.0])
def test_scalar_paths_keep_fractional_negative_and_large_coordinate_geometry(
        monkeypatch, progress):
    engine = ambient.make_engine("data_art_fungal_growth", "spacr", "#101418", seed=42)
    edges = (
        (-17.125, 0.0000125, 135.98765, -8.25, 701.125, 190.375,
         progress, 0.2, 0.7, 0),
        (2**24 + 0.125, -2**24 + 0.375, 2**24 + 3.625, -2**24 + 8.125,
         2**24 + 9.75, -2**24 + 13.625, progress, 0.2, 0.7, 0),
    )
    monkeypatch.setattr(engine, "geometry", lambda width, height: edges)
    scalar_paths, scalar_maturity, scalar_tips = engine._fungal_paths(3840, 2160)
    monkeypatch.setattr(ambient, "QPainterPath", _PointArgumentPath)
    legacy_paths, legacy_maturity, legacy_tips = engine._fungal_paths(3840, 2160)
    assert scalar_maturity == legacy_maturity and scalar_tips == legacy_tips
    assert list(scalar_paths) == list(legacy_paths)
    assert all(path == legacy_paths[key] for key, path in scalar_paths.items())
    assert sum(path.elementCount() for path in scalar_paths.values()) == 8


@pytest.mark.parametrize("palette", ["spacr", "random", "custom"])
@pytest.mark.parametrize("background", ["#101418", "#f3f3f3"])
def test_native_scalar_frame_is_exact_with_light_dark_and_custom_palette(
        qapp, monkeypatch, palette, background):
    monkeypatch.setattr(preferences, "_ambient_custom_colors",
                        lambda: ("#3b82f6", "#ff00ff"))

    def engine():
        result = ambient.make_engine("data_art_fungal_growth", palette, background,
                                     seed=42, resolution=2, density=3, size=2.5, blur=0)
        result.set_max_pixels(3840 * 2160)
        result.set_time(95.25)
        return result

    scalar = engine()
    first = scalar.shade(3840, 2160)
    assert first.width() == 3840 and first.height() == 2160
    before = first.constBits().tobytes()
    monkeypatch.setattr(ambient, "QPainterPath", _PointArgumentPath)
    legacy = engine()
    original = legacy.shade(3840, 2160)
    assert before == original.constBits().tobytes()
    scalar.set_time(95.5)
    scalar.shade(3840, 2160)
    assert first.constBits().tobytes() == before
    assert scalar._owned_fungal_image is None and legacy._owned_fungal_image is None
