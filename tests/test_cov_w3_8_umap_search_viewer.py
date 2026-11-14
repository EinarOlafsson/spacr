"""The native UMAP viewer: what it refuses, and what it actually paints.

Every assertion here comes from a real widget. The renderer is exercised
through ``QWidget.grab()``, which runs ``paintEvent`` synchronously on the
offscreen platform, and the resulting pixels are counted rather than
inspected by eye -- a marker that draws nothing would otherwise pass a test
that only checked the call did not raise.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QImage, QMouseEvent, QWheelEvent,
)
from PySide6.QtWidgets import QDialogButtonBox  # noqa: E402

from spacr.qt.widgets.umap_search_viewer import (  # noqa: E402
    BACKGROUND, UmapAppearanceDialog,
    UmapEmbeddingView, UmapExplorer, _coordinates,
    available_colormaps,
)

pytestmark = pytest.mark.qt


def _embedding(dimensions: int = 3, rows: int = 40) -> np.ndarray:
    """A reproducible cloud that is genuinely spread in every dimension."""
    rng = np.random.default_rng(3)
    return rng.normal(size=(rows, dimensions))


def _rendered(widget) -> np.ndarray:
    """The widget's own painting, as an (H, W, 3) RGB array."""
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8)
    rows = raw.reshape(image.height(), image.bytesPerLine() // 4, 4)
    return rows[:, :image.width(), :3][:, :, ::-1]


def _non_background_pixels(widget) -> int:
    """How many pixels the widget painted over its own black ground."""
    ground = np.array((BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue()),
                      dtype=np.uint8)
    return int(np.any(_rendered(widget) != ground, axis=-1).sum())


# --------------------------------------------------------------------------
# Coordinate validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad, message", [
    (np.zeros((4, 5)), "shape"),
    (np.zeros(6), "shape"),
    (np.zeros((0, 2)), "empty"),
    (np.array([[0.0, 0.0], [np.nan, 1.0]]), "NaN"),
    (np.array([[0.0, 0.0], [np.inf, 1.0]]), "NaN"),
])
def test_a_map_that_cannot_be_drawn_says_why(bad, message):
    with pytest.raises(ValueError, match=message):
        _coordinates(bad)


def test_a_two_dimensional_map_is_lifted_onto_a_flat_third_axis():
    coords = _coordinates(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert coords.shape == (2, 3)
    assert np.array_equal(coords[:, 2], np.zeros(2))


# --------------------------------------------------------------------------
# Colour maps
# --------------------------------------------------------------------------


def test_the_colour_map_list_survives_matplotlib_being_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    names = available_colormaps()
    assert names[0] == "spaCR"
    assert "viridis" in names


def test_the_colour_map_list_offers_the_installed_maps():
    names = available_colormaps()
    assert names[0] == "spaCR"
    assert len(names) > 5
    assert names[1:] == sorted(names[1:])


# --------------------------------------------------------------------------
# The appearance dialog
# --------------------------------------------------------------------------

def test_the_appearance_dialog_opens_on_the_values_in_force(qtbot):
    dialog = UmapAppearanceDialog(
        {"marker": "cross", "size": 9.0, "alpha": 0.25, "cmap": "viridis"})
    qtbot.addWidget(dialog)
    assert dialog.values() == {
        "marker": "cross", "size": 9.0, "alpha": 0.25, "cmap": "viridis"}


def test_apply_publishes_the_values_without_closing_the_dialog(qtbot):
    dialog = UmapAppearanceDialog({})
    qtbot.addWidget(dialog)
    dialog.show()
    seen = []
    dialog.applied.connect(seen.append)
    dialog.marker.setCurrentText("square")
    dialog.size.setValue(6.5)
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.Apply).click()
    assert seen == [{"marker": "square", "size": 6.5,
                     "alpha": 0.86, "cmap": "spaCR"}]
    assert dialog.isVisible()


def test_the_close_button_closes_the_appearance_dialog(qtbot):
    dialog = UmapAppearanceDialog({})
    qtbot.addWidget(dialog)
    dialog.show()
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.Close).click()
    assert not dialog.isVisible()


# --------------------------------------------------------------------------
# The view's own state
# --------------------------------------------------------------------------


def test_labels_for_an_empty_view_are_dropped_not_raised(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_labels([1, 2, 3])
    assert view.labels is None


def test_relabelling_a_loaded_map_needs_one_label_per_point(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(2, rows=5))
    with pytest.raises(ValueError, match="one value per UMAP point"):
        view.set_labels([0, 1])
    view.set_labels([0, 1, 1, -1, 0])
    assert view.labels.tolist() == [0, 1, 1, -1, 0]
    view.set_labels(None)
    assert view.labels is None


# --------------------------------------------------------------------------
# Mouse and wheel
# --------------------------------------------------------------------------

def _mouse(kind, position, button):
    return QMouseEvent(kind, QPointF(position), QPointF(position),
                       button, button, Qt.NoModifier)


def test_dragging_an_empty_view_does_nothing(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.mousePressEvent(_mouse(QEvent.MouseButtonPress,
                                QPoint(10, 10), Qt.LeftButton))
    assert view._drag_at is None


def _wheel(view, steps):
    angle = QPoint(0, int(120 * steps))
    return QWheelEvent(QPointF(10.0, 10.0), view.mapToGlobal(QPointF(10.0, 10.0)),
                       QPoint(0, 0), angle, Qt.NoButton, Qt.NoModifier,
                       Qt.NoScrollPhase, False)


def test_the_wheel_zooms_a_loaded_map_between_hard_limits(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(2))
    view.wheelEvent(_wheel(view, 1))
    assert view._zoom == pytest.approx(1.12)
    view.wheelEvent(_wheel(view, 60))
    assert view._zoom == 8.0
    view.wheelEvent(_wheel(view, -200))
    assert view._zoom == 0.2


def test_the_wheel_over_an_empty_view_changes_nothing(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.wheelEvent(_wheel(view, 3))
    assert view._zoom == 1.0


# --------------------------------------------------------------------------
# Painting
# --------------------------------------------------------------------------

def test_an_empty_view_paints_its_caption_on_black(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(320, 280)
    image = view.grab().toImage()
    corner = image.pixelColor(0, 0)
    assert (corner.red(), corner.green(), corner.blue()) == (
        BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue())
    assert _non_background_pixels(view) > 0


def _ink(widget, top, bottom) -> int:
    """Painted pixels in a horizontal strip of the widget's own render."""
    ground = np.array((BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue()),
                      dtype=np.uint8)
    strip = _rendered(widget)[top:bottom]
    return int(np.any(strip != ground, axis=-1).sum())


def test_the_backend_is_named_beside_the_caption(qtbot):
    """The header strip gains ink when a backend is supplied; nothing else does.

    Measured rather than intercepted: ``QPainter.drawText`` is a Shiboken
    method and cannot be replaced, so the two renders are made identical in
    every respect except the backend string and the pixels are counted.
    """
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(320, 280)
    coords = _embedding(2)
    view.set_embedding(coords, caption="map A")
    plain_header = _ink(view, 6, 28)
    plain_body = _ink(view, 40, 240)
    view.set_embedding(coords, caption="map A", backend="cuml")
    assert _ink(view, 6, 28) > plain_header
    assert _ink(view, 40, 240) == plain_body


# --------------------------------------------------------------------------
# The explorer shell
# --------------------------------------------------------------------------

def test_the_reset_button_puts_the_camera_back(qtbot):
    explorer = UmapExplorer()
    qtbot.addWidget(explorer)
    explorer.view.set_embedding(_embedding(3))
    explorer.view._yaw = 1.5
    explorer.view._zoom = 5.0
    explorer.reset.click()
    assert explorer.view._yaw == pytest.approx(0.22)
    assert explorer.view._pitch == pytest.approx(-0.16)
    assert explorer.view._zoom == 1.0
