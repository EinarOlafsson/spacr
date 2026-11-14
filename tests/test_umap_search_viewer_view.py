"""The native UMAP viewer: its colours, its state, its menu and its pixels.

Every assertion comes from a real widget or a real palette.  Painting is
exercised through ``QWidget.grab()``, which runs ``paintEvent`` synchronously
on the offscreen platform, and pixels are counted rather than inspected by
eye.  One general test per unit, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QContextMenuEvent, QImage, QKeyEvent, QMouseEvent,
)
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets.umap_search_viewer import (  # noqa: E402
    BACKGROUND, CLUSTER_COLORS, NOISE, POINT, UmapAppearanceDialog,
    UmapEmbeddingView, _colormap_rgb, colors_for_labels,
)

pytestmark = pytest.mark.qt


def _embedding(dimensions: int = 3, rows: int = 40) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(size=(rows, dimensions))


def _view(qtbot, coords=None, size=None, **kwargs) -> UmapEmbeddingView:
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    if size is not None:
        view.resize(*size)
    if coords is not None:
        view.set_embedding(coords, **kwargs)
    return view


def _ink(widget, top=None, bottom=None) -> int:
    """Pixels the widget painted over its own black ground, in a strip."""
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8)
    rows = raw.reshape(image.height(), image.bytesPerLine() // 4, 4)
    rgb = rows[:, :image.width(), :3][:, :, ::-1][top:bottom]
    ground = np.array((BACKGROUND.red(), BACKGROUND.green(),
                       BACKGROUND.blue()), dtype=np.uint8)
    return int(np.any(rgb != ground, axis=-1).sum())


def _rgb(colour):
    return (colour.red(), colour.green(), colour.blue())


# --------------------------------------------------------------------------
# colour maps and label colours
# --------------------------------------------------------------------------

def test_a_colour_map_is_sampled_or_falls_back_to_the_native_palette(
        monkeypatch, subtests):
    """spaCR repeats; Matplotlib maps span their range; a broken one falls back."""
    _colormap_rgb.cache_clear()
    with subtests.test(msg="the native palette repeats"):
        palette = _colormap_rgb("spaCR", len(CLUSTER_COLORS) + 3)
        assert palette == tuple(CLUSTER_COLORS) + tuple(CLUSTER_COLORS[:3])

    with subtests.test(msg="viridis is sampled end to end"):
        palette = _colormap_rgb("viridis", 5)
        assert len(set(palette)) == 5
        assert palette[0] == (68, 1, 84)
        assert palette[-1] == (253, 231, 37)

    with subtests.test(msg="a map Matplotlib rejects"):
        from matplotlib import colormaps

        def refuse(_name):
            raise ValueError("no such colormap")

        monkeypatch.setattr(colormaps, "get_cmap", refuse)
        assert _colormap_rgb("not-a-real-map", 3) == tuple(CLUSTER_COLORS[:3])
    _colormap_rgb.cache_clear()


def test_points_are_coloured_by_cluster_with_noise_in_grey(subtests):
    """No labels share one colour; noise is grey; foreign labels are ignored."""
    with subtests.test(msg="unlabelled points share POINT at the opacity"):
        colours = colors_for_labels(None, 4, cmap="spaCR", alpha=0.5)
        assert {(*_rgb(c), c.alpha()) for c in colours} == {
            (*_rgb(POINT), 128)}

    with subtests.test(msg="unlabelled points follow a chosen map"):
        colours = colors_for_labels(None, 4, cmap="viridis", alpha=1.0)
        assert [_rgb(c) for c in colours] == list(_colormap_rgb("viridis", 4))
        assert {c.alpha() for c in colours} == {255}

    with subtests.test(msg="labels from another map are ignored"):
        colours = colors_for_labels([0, 1], 5)
        assert [c.name() for c in colours] == [POINT.name()] * 5

    with subtests.test(msg="HDBSCAN noise is grey, clusters are not"):
        colours = colors_for_labels([-1, 0, 1, 0], 4, cmap="spaCR", alpha=1.0)
        assert _rgb(colours[0]) == _rgb(NOISE)
        assert [_rgb(c) for c in colours[1:]] == [
            CLUSTER_COLORS[0], CLUSTER_COLORS[1], CLUSTER_COLORS[0]]


# --------------------------------------------------------------------------
# the view's own state
# --------------------------------------------------------------------------

def test_the_view_guards_its_state_and_clamps_its_appearance(qtbot, subtests):
    """Copies out, unknown markers refused, values clamped, captions kept."""
    with subtests.test(msg="coordinates and labels are handed out as copies"):
        view = _view(qtbot, _embedding(2, rows=6),
                     labels=np.zeros(6, dtype=int))
        view.coordinates[0, 0] = 999.0
        view.labels[0] = 7
        assert view.coordinates[0, 0] != 999.0
        assert view.labels.tolist() == [0] * 6

    with subtests.test(msg="an unknown marker is refused"):
        view = _view(qtbot)
        with pytest.raises(ValueError,
                           match=r"^Unknown point rendering: 'star'\.$"):
            view.set_appearance({"marker": "star"})
        assert view.appearance["marker"] == "circle"

    for case_id, values, expected in [
            ("values above the range", {"size": 500.0, "alpha": 9.0,
                                        "cmap": "made-up"},
             (24.0, 1.0, "spaCR")),
            ("values below the range", {"size": 0.0, "alpha": 0.0,
                                        "cmap": "viridis"},
             (1.0, 0.05, "viridis"))]:
        with subtests.test(msg=f"{case_id} are clamped"):
            view = _view(qtbot)
            view.set_appearance(values)
            assert (view.appearance["size"], view.appearance["alpha"],
                    view.appearance["cmap"]) == expected

    with subtests.test(msg="clearing keeps the message it was given"):
        view = _view(qtbot, _embedding(3))
        view.clear("Search cancelled.")
        assert view.coordinates is None
        assert view.dimensions == 0
        assert view._caption == "Search cancelled."


# --------------------------------------------------------------------------
# the context menu, the appearance editor and the mouse
# --------------------------------------------------------------------------

def _drive_open_menu(index):
    """Answer the next popup by keyboard: Return on ``index``, else Escape.

    ``QMenu.exec`` cannot be monkeypatched (Shiboken rejects it), and closing
    the popup from a timer makes ``exec`` return None whatever was clicked.
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


def _mouse(kind, position, button):
    return QMouseEvent(kind, QPointF(position), QPointF(position),
                       button, button, Qt.NoModifier)


def test_the_menu_editor_and_mouse_move_only_the_camera_they_should(
        qtbot, subtests):
    """Reset, Appearance and Escape on the menu; drags that must not spin."""
    menu_event = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(5, 5),
                                   QPoint(105, 105))

    with subtests.test(msg="Reset view puts the camera back"):
        view = _view(qtbot, _embedding(3))
        view._zoom, view._yaw = 4.0, 1.9
        _drive_open_menu(1)
        view.contextMenuEvent(menu_event)
        assert (view._zoom, view._yaw) == (1.0, pytest.approx(0.22))
        assert view._appearance_dialog is None

    with subtests.test(msg="Appearance opens the editor"):
        view = _view(qtbot, _embedding(3))
        _drive_open_menu(0)
        view.contextMenuEvent(menu_event)
        assert isinstance(view._appearance_dialog, UmapAppearanceDialog)
        qtbot.addWidget(view._appearance_dialog)

    with subtests.test(msg="dismissing the menu changes nothing"):
        view = _view(qtbot, _embedding(3))
        view._zoom = 4.0
        _drive_open_menu(None)
        view.contextMenuEvent(menu_event)
        assert view._zoom == 4.0
        assert view._appearance_dialog is None

    with subtests.test(msg="the editor reaches the view and lets go"):
        view = _view(qtbot, _embedding(2))
        dialog = view.open_appearance_editor()
        qtbot.addWidget(dialog)
        assert view._appearance_dialog is dialog
        dialog.marker.setCurrentText("diamond")
        dialog._apply()
        assert view.appearance["marker"] == "diamond"
        dialog.reject()
        assert view._appearance_dialog is None

    for case_id, dims, button, grabbed in [
            ("a right drag on a 3-D map", 3, Qt.RightButton, False),
            ("a left drag on a 2-D map", 2, Qt.LeftButton, True)]:
        with subtests.test(msg=f"{case_id} does not spin"):
            view = _view(qtbot, _embedding(dims), size=(400, 320))
            before = (view._yaw, view._pitch)
            view.mousePressEvent(_mouse(QEvent.MouseButtonPress,
                                        QPoint(100, 100), button))
            assert (view._drag_at is not None) is grabbed
            view.mouseMoveEvent(_mouse(QEvent.MouseMove, QPoint(180, 150),
                                       Qt.NoButton))
            assert (view._yaw, view._pitch) == before
            view.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease,
                                          QPoint(180, 150), button))
            assert view._drag_at is None


# --------------------------------------------------------------------------
# painting
# --------------------------------------------------------------------------

def test_every_marker_paints_its_own_picture(qtbot, subtests):
    """Each marker inks the canvas, two markers differ, 3-D adds a hint."""
    labels = np.array([-1 if index % 7 == 0 else index % 3
                       for index in range(40)])
    for marker in ("circle", "square", "diamond", "cross"):
        with subtests.test(msg=f"{marker} puts ink on the canvas"):
            view = _view(qtbot, _embedding(3), size=(300, 260), labels=labels,
                         caption="test map", backend="cuml")
            view.set_appearance({"marker": marker, "size": 8.0})
            assert _ink(view) > 200

    with subtests.test(msg="circles and squares differ"):
        view = _view(qtbot, _embedding(2), size=(300, 260))
        view.set_appearance({"marker": "circle", "size": 9.0})
        circles = view.grab().toImage()
        view.set_appearance({"marker": "square", "size": 9.0})
        assert view.grab().toImage() != circles

    with subtests.test(msg="only a 3-D map carries the drag hint"):
        flat = _embedding(2)

        def footer(coords):
            view = _view(qtbot, coords, size=(600, 400))
            view._zoom, view._pitch = 0.2, 0.0
            return _ink(view, 370, 396)

        two_d = footer(flat)
        three_d = footer(np.column_stack((flat, np.zeros(len(flat)))))
        assert 0 < two_d < three_d
