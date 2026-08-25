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

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QContextMenuEvent, QImage, QKeyEvent, QMouseEvent, QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

from spacr.qt.widgets.umap_search_viewer import (  # noqa: E402
    BACKGROUND, CLUSTER_COLORS, NOISE, POINT, UmapAppearanceDialog,
    UmapEmbeddingView, UmapExplorer, _colormap_rgb, _coordinates,
    available_colormaps, colors_for_labels,
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

def test_the_native_palette_repeats_instead_of_running_out():
    _colormap_rgb.cache_clear()
    palette = _colormap_rgb("spaCR", len(CLUSTER_COLORS) + 3)
    assert palette[:len(CLUSTER_COLORS)] == CLUSTER_COLORS
    assert palette[len(CLUSTER_COLORS)] == CLUSTER_COLORS[0]


def test_a_matplotlib_map_is_sampled_across_its_whole_range():
    _colormap_rgb.cache_clear()
    palette = _colormap_rgb("viridis", 5)
    assert len(palette) == 5
    assert len(set(palette)) == 5
    assert all(0 <= channel <= 255 for rgb in palette for channel in rgb)


def test_a_broken_colour_map_falls_back_to_the_native_palette(monkeypatch):
    """A name Matplotlib rejects must still colour the points."""
    _colormap_rgb.cache_clear()
    from matplotlib import colormaps

    def refuse(_name):
        raise ValueError("no such colormap")

    monkeypatch.setattr(colormaps, "get_cmap", refuse)
    assert _colormap_rgb("not-a-real-map", 3) == CLUSTER_COLORS[:3]
    _colormap_rgb.cache_clear()


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


def test_unlabelled_points_all_share_one_colour_at_the_asked_opacity():
    colours = colors_for_labels(None, 4, cmap="spaCR", alpha=0.5)
    assert len({colour.name() for colour in colours}) == 1
    assert colours[0].red() == POINT.red()
    assert colours[0].alpha() == int(round(255 * 0.5))


def test_unlabelled_points_still_follow_a_chosen_colour_map():
    colours = colors_for_labels(None, 4, cmap="viridis", alpha=1.0)
    assert len({colour.name() for colour in colours}) == 4


def test_labels_from_another_map_are_ignored_rather_than_mis_coloured():
    colours = colors_for_labels([0, 1], 5)
    assert len(colours) == 5
    assert {colour.name() for colour in colours} == {POINT.name()}


def test_hdbscan_noise_is_grey_and_clusters_are_not():
    colours = colors_for_labels([-1, 0, 1, 0], 4, cmap="spaCR", alpha=1.0)
    assert (colours[0].red(), colours[0].green(), colours[0].blue()) == (
        NOISE.red(), NOISE.green(), NOISE.blue())
    assert colours[1].name() == colours[3].name()
    assert colours[1].name() != colours[2].name()
    assert colours[1].name() != colours[0].name()


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


def test_the_editor_reaches_the_view_and_lets_go_when_it_closes(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(2))
    dialog = view.open_appearance_editor()
    qtbot.addWidget(dialog)
    assert view._appearance_dialog is dialog
    dialog.marker.setCurrentText("diamond")
    dialog._apply()
    assert view.appearance["marker"] == "diamond"
    dialog.reject()
    assert view._appearance_dialog is None


def _drive_open_menu(index):
    """Answer the next popup menu the way a user would: keyboard, not a mock.

    ``QMenu.exec`` cannot be monkeypatched -- Shiboken types reject the
    assignment and the real modal loop runs, wedging the run -- and closing
    the popup from a timer makes ``exec`` return None whatever was clicked.
    Highlighting the row and pressing Return is the path that actually sets
    the action ``exec`` returns; Escape (``index`` None) dismisses it.
    """
    def answer():
        popup = QApplication.activePopupWidget()
        if popup is None:
            QTimer.singleShot(5, answer)
            return
        if index is None:
            key = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        else:
            popup.setActiveAction(popup.actions()[index])
            key = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
        QApplication.sendEvent(popup, key)

    QTimer.singleShot(0, answer)


def _context_menu_event():
    return QContextMenuEvent(
        QContextMenuEvent.Mouse, QPoint(5, 5), QPoint(105, 105))


def test_reset_view_from_the_context_menu_puts_the_camera_back(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(3))
    view._zoom = 4.0
    view._yaw = 1.9
    _drive_open_menu(1)
    view.contextMenuEvent(_context_menu_event())
    assert view._zoom == 1.0
    assert view._yaw == pytest.approx(0.22)
    assert view._appearance_dialog is None


def test_appearance_from_the_context_menu_opens_the_editor(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(3))
    _drive_open_menu(0)
    view.contextMenuEvent(_context_menu_event())
    assert isinstance(view._appearance_dialog, UmapAppearanceDialog)
    qtbot.addWidget(view._appearance_dialog)


def test_dismissing_the_context_menu_changes_nothing(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(3))
    view._zoom = 4.0
    _drive_open_menu(None)
    view.contextMenuEvent(_context_menu_event())
    assert view._zoom == 4.0
    assert view._appearance_dialog is None


# --------------------------------------------------------------------------
# The view's own state
# --------------------------------------------------------------------------

def test_the_view_hands_out_copies_not_its_own_arrays(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    coords = _embedding(2, rows=6)
    view.set_embedding(coords, labels=np.zeros(6, dtype=int))
    handed = view.coordinates
    handed[0, 0] = 999.0
    labels = view.labels
    labels[0] = 7
    assert view.coordinates[0, 0] != 999.0
    assert view.labels[0] == 0


def test_an_unknown_marker_is_refused_rather_than_silently_ignored(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    with pytest.raises(ValueError, match="Unknown point rendering"):
        view.set_appearance({"marker": "star"})
    assert view.appearance["marker"] == "circle"


def test_appearance_values_are_clamped_to_what_can_be_drawn(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_appearance({"size": 500.0, "alpha": 9.0, "cmap": "made-up"})
    assert view.appearance["size"] == 24.0
    assert view.appearance["alpha"] == 1.0
    assert view.appearance["cmap"] == "spaCR"


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


def test_clearing_the_view_keeps_the_message_it_was_given(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(3))
    view.clear("Search cancelled.")
    assert view.coordinates is None
    assert view.dimensions == 0
    assert view._caption == "Search cancelled."


# --------------------------------------------------------------------------
# Mouse and wheel
# --------------------------------------------------------------------------

def _mouse(kind, position, button):
    return QMouseEvent(kind, QPointF(position), QPointF(position),
                       button, button, Qt.NoModifier)


def test_a_right_drag_does_not_spin_the_map(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(400, 320)
    view.set_embedding(_embedding(3))
    before = (view._yaw, view._pitch)
    view.mousePressEvent(_mouse(QEvent.MouseButtonPress,
                                QPoint(100, 100), Qt.RightButton))
    assert view._drag_at is None
    view.mouseMoveEvent(_mouse(QEvent.MouseMove,
                               QPoint(160, 140), Qt.NoButton))
    assert (view._yaw, view._pitch) == before


def test_a_two_dimensional_map_does_not_spin_when_dragged(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(400, 320)
    view.set_embedding(_embedding(2))
    before = (view._yaw, view._pitch)
    view.mousePressEvent(_mouse(QEvent.MouseButtonPress,
                                QPoint(100, 100), Qt.LeftButton))
    assert view._drag_at is not None
    view.mouseMoveEvent(_mouse(QEvent.MouseMove,
                               QPoint(180, 150), Qt.NoButton))
    assert (view._yaw, view._pitch) == before
    view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease,
                                  QPoint(180, 150), Qt.LeftButton))
    assert view._drag_at is None


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


@pytest.mark.parametrize("marker", ["circle", "square", "diamond", "cross"])
def test_every_marker_actually_puts_ink_on_the_canvas(qtbot, marker):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(300, 260)
    view.set_embedding(_embedding(3), labels=np.array(
        [-1 if index % 7 == 0 else index % 3 for index in range(40)]),
        caption="test map", backend="cuml")
    view.set_appearance({"marker": marker, "size": 8.0})
    assert _non_background_pixels(view) > 200


def test_two_markers_do_not_paint_the_same_picture(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(300, 260)
    view.set_embedding(_embedding(2))
    view.set_appearance({"marker": "circle", "size": 9.0})
    circles = view.grab().toImage()
    view.set_appearance({"marker": "square", "size": 9.0})
    squares = view.grab().toImage()
    assert circles != squares


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


def test_only_a_three_dimensional_map_offers_the_drag_hint(qtbot):
    """The 3-D footer carries the extra 'drag to spin' clause, so it is wider.

    Both renders use the same first two coordinates and a zoomed-out cloud
    that stays clear of the footer strip, so the strip holds the hint alone.
    """
    flat = _embedding(2)
    lifted = np.column_stack((flat, np.zeros(len(flat))))

    def footer(coords):
        view = UmapEmbeddingView()
        qtbot.addWidget(view)
        view.resize(600, 400)
        view.set_embedding(coords)
        view._zoom = 0.2
        view._pitch = 0.0
        return _ink(view, 370, 396)

    two_d = footer(flat)
    three_d = footer(lifted)
    assert two_d > 0
    assert three_d > two_d


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
