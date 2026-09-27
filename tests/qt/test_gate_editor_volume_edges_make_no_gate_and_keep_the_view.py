"""Gate editor, 3D volume: the edges where no gate is made and nothing breaks.

Pinned here, on a real Gate Editor screen in 3D:

* a mouse event whose button is not a number is not a right-button drag;
* with snap-to-axis on, releasing a DRAW gesture leaves the view where it
  was -- only a spin is squared up;
* a spin locked to the axis that points out of the screen still turns the
  volume about that axis, by the sideways drag;
* moving the pointer with no outline started draws nothing;
* the camera is not read from a flat plot, a camera that cannot be read, one
  that is not a finite 4 x 4, or one that puts the volume's centre at
  infinity -- and no view gate is made from it;
* an outline narrower than three pixels, one the axes cannot map back, or a
  gate that refuses its own name makes no gate (the refusal is said);
* a view gate whose camera sees nothing at a finite depth, or cannot be
  solved back into the volume, is drawn as nothing rather than failing;
* picking a volume shape while already drawing keeps the drawing mode;
* the screen: folding an already folded console, and lining up a side
  panel that has no page.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets import gate_editor as GE  # noqa: E402
from spacr.qt.widgets.gate_settings import GateEditorSettings  # noqa: E402
from spacr.qt.widgets.gate_spec import ViewGate  # noqa: E402
from spacr.qt.widgets.volume_view import view_axes  # noqa: E402

pytestmark = pytest.mark.qt


def _table(n=300, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(0, 1, n),
                         "b": rng.normal(5, 2, n),
                         "c": rng.normal(-3, 0.5, n)})


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1100, 800)
    widget.set_frame(_table())
    widget._x.setCurrentText("a")
    widget._y.setCurrentText("b")
    widget.gates.mode_requested.emit("3D")
    widget._z.setCurrentText("c")
    names = iter(f"view{i}" for i in range(1, 100))
    widget.gates.set_namer(lambda: next(names))
    return widget


class _Mouse:
    def __init__(self, ax, x, y, button=1):
        self.inaxes, self.x, self.y = ax, float(x), float(y)
        self.xdata = self.ydata = 0.0
        self.button = button
        self.step = 0


def _turn(canvas, elev, azim, roll):
    canvas._view_angles = (elev, azim, roll)
    canvas.render_now()
    canvas._canvas.draw()
    return canvas.axes_at(0, 0)


def _centre(ax):
    box = ax.bbox
    return (box.x0 + box.width / 2.0, box.y0 + box.height / 2.0)


def _angles(ax):
    return (float(ax.elev), float(ax.azim), float(ax.roll))


def _outline(ax, radius=0.2, sides=8):
    cx, cy = _centre(ax)
    size = min(ax.bbox.width, ax.bbox.height) * radius
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return [(cx + size * np.cos(t), cy + size * np.sin(t)) for t in angles]


# ---------------------------------------------------------------------------
# gestures
# ---------------------------------------------------------------------------

def test_a_button_that_is_not_a_number_is_not_the_right_button():
    assert GE._is_right_button(types.SimpleNamespace(button="left")) is False
    assert GE._is_right_button(types.SimpleNamespace(button=3)) is True


def test_releasing_a_drawn_outline_does_not_snap_the_view(screen):
    canvas = screen.gates.canvas
    canvas.apply_settings(GateEditorSettings(snap_to_axis=True))
    ax = _turn(canvas, 33.0, -61.0, 17.0)
    outline = _outline(ax)
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("lasso")
    canvas._on_press(_Mouse(ax, *outline[0]))
    for point in outline[1:]:
        canvas._on_motion(_Mouse(ax, *point))
    release = _Mouse(ax, *outline[-1])
    canvas._on_release(release)
    canvas._on_button_release(release)
    assert screen.gates.gates.names == ("view1",)
    assert _angles(canvas.axes_at(0, 0)) == pytest.approx(
        (33.0, -61.0, 17.0)), "a drawing gesture is not a spin to square up"


def test_a_spin_about_the_axis_facing_the_viewer_follows_the_sideways_drag(
        screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 90.0, 0.0, 0.0)
    canvas.set_spin_axis("z")
    _u, _v, w_before = view_axes(*_angles(ax))
    canvas.set_drag_mode("spin")
    canvas._on_press(_Mouse(ax, 100, 100))
    canvas._on_motion(_Mouse(ax, 140, 100))
    canvas._on_release(_Mouse(ax, 140, 100))
    after = _angles(canvas.axes_at(0, 0))
    _u, _v, w_after = view_axes(*after)
    assert abs(float(w_before @ w_after)) == pytest.approx(1.0, abs=1e-6), (
        "turning about z while looking down z keeps looking down z")
    assert after != pytest.approx((90.0, 0.0, 0.0)), "but it did turn"


def test_moving_with_no_outline_started_draws_nothing(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape("lasso")
    ghosts = len(canvas._ghost)
    canvas._extend_lasso(_Mouse(ax, *_centre(ax)))
    assert canvas._lasso is None
    assert len(canvas._ghost) == ghosts


# ---------------------------------------------------------------------------
# the camera
# ---------------------------------------------------------------------------

def test_a_flat_plot_has_no_camera_and_makes_no_view_gate(screen):
    screen.gates.mode_requested.emit("2D")
    canvas = screen.gates.canvas
    assert canvas.view_projection() is None
    assert canvas.gate_from_screen_outline(
        [(10, 10), (80, 10), (40, 90)]) is None


class _Camera:
    """An axes stand-in whose camera answers what it is given."""

    def __init__(self, matrix=None, error=None):
        self._matrix, self._error = matrix, error

    def get_proj(self):
        if self._error is not None:
            raise self._error
        return self._matrix

    def get_xlim3d(self):
        return (-1.0, 1.0)

    def get_ylim3d(self):
        return (-1.0, 1.0)

    def get_zlim3d(self):
        return (-1.0, 1.0)


def test_a_camera_that_cannot_be_used_is_no_camera(screen):
    canvas = screen.gates.canvas
    assert canvas.view_projection(_Camera(error=RuntimeError("no view"))) \
        is None
    assert canvas.view_projection(_Camera(np.full((4, 4), np.nan))) is None
    assert canvas.view_projection(_Camera(np.eye(3))) is None
    at_infinity = np.eye(4)
    at_infinity[3] = 0.0
    assert canvas.view_projection(_Camera(at_infinity)) is None
    flipped = np.eye(4) * -1.0
    assert canvas.view_projection(_Camera(flipped))[3, 3] == 1.0, (
        "a camera behind the volume is turned to face it")


def test_no_view_gate_through_a_camera_that_cannot_be_read(
        screen, monkeypatch):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    monkeypatch.setattr(canvas, "view_projection", lambda ax=None: None)
    assert canvas.gate_from_screen_outline(_outline(ax)) is None


def test_the_same_column_twice_makes_no_view_gate(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    screen._z.setCurrentText("a")
    assert canvas.gate_from_screen_outline(_outline(ax)) is None


def test_an_outline_under_three_pixels_wide_makes_no_gate(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    cx, cy = _centre(ax)
    assert canvas.gate_from_screen_outline(
        [(cx, cy), (cx + 2, cy + 30), (cx, cy + 60)]) is None


def test_an_outline_the_axes_cannot_map_back_makes_no_gate(
        screen, monkeypatch):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)

    class _Unmappable:
        def inverted(self):
            raise ValueError("singular transform")

    outline = _outline(ax)
    monkeypatch.setattr(canvas, "axes_at", lambda row, col: ax)
    monkeypatch.setattr(ax, "transData", _Unmappable())
    try:
        assert canvas.gate_from_screen_outline(outline) is None
    finally:
        monkeypatch.undo()


def test_a_gate_that_refuses_its_name_says_why(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    said = []
    canvas.depth_requested.connect(said.append)
    assert canvas.gate_from_screen_outline(_outline(ax), name="") is None
    assert said and "name" in said[-1].lower()


# ---------------------------------------------------------------------------
# drawing a view gate
# ---------------------------------------------------------------------------

def _view_gate(projection):
    return ViewGate(name="odd", x_column="a", y_column="b", z_column="c",
                    projection=projection,
                    vertices=((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)))


def test_a_camera_that_sees_nothing_at_a_finite_depth_draws_nothing(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    blind = np.eye(4)
    blind[3] = 0.0
    before = len(canvas._artists)
    canvas._draw_view_gate(ax, _view_gate(blind), "#ff0000")
    assert len(canvas._artists) == before


def test_a_camera_that_cannot_be_solved_back_draws_nothing(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    singular = np.eye(4)
    singular[1] = singular[0]
    before = len(canvas._artists)
    canvas._draw_view_gate(ax, _view_gate(singular), "#ff0000")
    assert len(canvas._artists) == before


def test_a_volume_without_limits_draws_no_view_gate(screen):
    canvas = screen.gates.canvas

    class _NoLimits:
        def get_xlim3d(self):
            raise AttributeError("not a 3D axes")

    before = len(canvas._artists)
    canvas._draw_view_gate(_NoLimits(), _view_gate(np.eye(4)), "#ff0000")
    assert len(canvas._artists) == before


def test_a_stored_view_gate_that_sees_nothing_still_renders_the_volume(
        screen):
    canvas = screen.gates.canvas
    blind = np.eye(4)
    blind[3] = 0.0
    screen.gates.gates.add(_view_gate(blind))
    canvas.set_gates(screen.gates.gates)
    canvas.render_now()
    assert hasattr(canvas.axes_at(0, 0), "get_zlim")


# ---------------------------------------------------------------------------
# the panel and the screen
# ---------------------------------------------------------------------------

def test_picking_a_shape_while_drawing_keeps_drawing(screen):
    panel = screen.gates
    panel._drag_buttons["draw"].setChecked(True)
    panel._on_drag_mode("draw")
    index = panel._volume_shape.findData("view_rect")
    assert index >= 0
    panel._volume_shape.setCurrentIndex(index)
    panel._on_volume_shape_picked()
    assert panel.drag_mode() == "draw"
    assert panel.canvas.volume_shape() == "view_rect"


def test_folding_a_folded_console_leaves_it_folded(screen):
    from spacr.qt.screens import gate_editor as GS

    body = screen._body
    assert body.is_collapsed(GS.CONSOLE_PANE)
    GS._start_console_folded(body)
    assert body.is_collapsed(GS.CONSOLE_PANE)


def test_a_side_panel_with_no_page_is_not_offset(screen, qtbot):
    screen.show()
    qtbot.waitExposed(screen)
    tabs = screen.side_tabs
    pages = [(tabs.widget(i), tabs.tabText(i)) for i in range(tabs.count())]
    try:
        while tabs.count():
            tabs.removeTab(0)
        assert tabs.currentWidget() is None
        assert screen.align_side_panel() == 0
    finally:
        for page, text in pages:
            tabs.addTab(page, text)
    assert tabs.count() == len(pages)
