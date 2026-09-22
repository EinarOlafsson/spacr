"""A gate drawn on the 3D view at any angle keeps what projects inside it.

The maintainer, 2026-09-21: "it is not possible to draw propper gates in 3d".
Every 3D shape before this was drawn on one of the three axis planes and
needed a second drag for its depth, and with the view snapping square-on the
plane was often edge-on, so the depth could not be read at all.

The lasso (and the rectangle) through the view is drawn on the screen at
whatever angle the volume was turned to and swept straight through the
volume along the line of sight. The gate carries the camera it was drawn
through, so it re-applies, saves and loads without matplotlib.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.gate_spec import (
    GateSet, VIEW_LASSO, ViewGate, gate_from_dict, points_in_polygon,
)


def _table(n=600, seed=4):
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
    names = iter(f"lasso{i}" for i in range(1, 100))
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


def _centre_pixels(ax):
    box = ax.bbox
    return (box.x0 + box.width / 2.0, box.y0 + box.height / 2.0)


def _outline(ax, radius=0.18, sides=9):
    cx, cy = _centre_pixels(ax)
    size = min(ax.bbox.width, ax.bbox.height) * radius
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    wobble = 1.0 + 0.3 * np.sin(3 * angles)
    return [(cx + size * w * np.cos(t), cy + size * 0.8 * w * np.sin(t))
            for t, w in zip(angles, wobble)]


def _draw(canvas, outline, shape="lasso"):
    canvas.set_drag_mode("draw")
    canvas.set_volume_shape(shape)
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, *outline[0]))
    for point in outline[1:]:
        canvas._on_motion(_Mouse(ax, *point))
    canvas._on_release(_Mouse(ax, *outline[-1]))


def _pixels_of_rows(ax, frame):
    """Where matplotlib puts each row on screen: the ground truth."""
    from mpl_toolkits.mplot3d import proj3d

    xs, ys, _ = proj3d.proj_transform(frame["a"].to_numpy(float),
                                      frame["b"].to_numpy(float),
                                      frame["c"].to_numpy(float),
                                      ax.get_proj())
    return ax.transData.transform(np.column_stack([xs, ys]))


@pytest.mark.parametrize("angles", [(33.0, -61.0, 17.0), (-52.0, 128.0, -40.0),
                                    (71.0, 12.0, 95.0)])
def test_a_lasso_at_any_angle_keeps_what_projects_inside_it(screen, angles):
    canvas = screen.gates.canvas
    ax = _turn(canvas, *angles)
    outline = _outline(ax)
    frame = canvas.population()
    truth_px = _pixels_of_rows(ax, frame)
    truth = points_in_polygon(truth_px[:, 0], truth_px[:, 1], outline)

    _draw(canvas, outline)

    gates = screen.gates.gates
    assert gates.names == ("lasso1",)
    gate = gates.get("lasso1")
    assert isinstance(gate, ViewGate) and gate.kind == VIEW_LASSO
    assert gate.view == pytest.approx(angles)
    got = gate.mask(frame)
    assert 0 < int(truth.sum()) < len(frame)
    assert np.array_equal(got, truth)


def test_the_gate_still_selects_the_same_objects_after_the_view_turns(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 25.0, 40.0, 0.0)
    _draw(canvas, _outline(ax))
    frame = canvas.population()
    before = screen.gates.gates.get("lasso1").mask(frame)
    _turn(canvas, -70.0, 200.0, 33.0)
    after = screen.gates.gates.get("lasso1").mask(frame)
    assert np.array_equal(before, after)


def test_a_rectangle_through_the_view(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 12.0, 77.0, -8.0)
    cx, cy = _centre_pixels(ax)
    corners = [(cx - 60, cy - 40), (cx + 70, cy + 50)]
    frame = canvas.population()
    truth_px = _pixels_of_rows(ax, frame)
    inside = ((truth_px[:, 0] > cx - 60) & (truth_px[:, 0] < cx + 70)
              & (truth_px[:, 1] > cy - 40) & (truth_px[:, 1] < cy + 50))
    _draw(canvas, corners, shape="view_rect")
    gate = screen.gates.gates.get("lasso1")
    assert len(gate.vertices) == 4
    assert np.array_equal(gate.mask(frame), inside)


def test_the_gate_is_listed_with_its_count(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    _draw(canvas, _outline(ax))
    gate = screen.gates.gates.get("lasso1")
    n = int(gate.mask(canvas.population()).sum())
    tree = screen.gates.tree
    texts = []
    stack = [tree.tree.topLevelItem(i)
             for i in range(tree.tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        texts.append(" ".join(item.text(c) for c in range(item.columnCount())))
        stack.extend(item.child(i) for i in range(item.childCount()))
    row = next(t for t in texts if "lasso1" in t)
    assert str(n) in row


def test_a_view_gate_is_drawn_in_the_volume(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    _draw(canvas, _outline(ax))
    ax = canvas.axes_at(0, 0)
    lines = [line for line in ax.get_lines()]
    assert len(lines) >= 2, "the outline was not drawn into the volume"


def test_a_tiny_or_flat_outline_makes_no_gate(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    cx, cy = _centre_pixels(ax)
    _draw(canvas, [(cx, cy), (cx + 1, cy), (cx + 1, cy + 1)])
    assert screen.gates.gates.names == ()


def test_save_and_load_round_trip(screen, tmp_path):
    canvas = screen.gates.canvas
    ax = _turn(canvas, -40.0, 150.0, 60.0)
    _draw(canvas, _outline(ax))
    gates = screen.gates.gates
    frame = canvas.population()
    path = gates.save(str(tmp_path / "gates.json"))
    loaded = GateSet.load(path)
    original = gates.get("lasso1")
    again = loaded.get("lasso1")
    assert again == original
    assert np.array_equal(again.mask(frame), original.mask(frame))
    payload = json.loads(json.dumps(original.to_dict()))
    assert gate_from_dict(payload) == original


def test_a_child_gate_inside_a_view_gate(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    _draw(canvas, _outline(ax, radius=0.3))
    frame = canvas.population()
    parent = screen.gates.gates.get("lasso1").mask(frame)
    child = ViewGate(name="inner", parent="lasso1",
                     x_column="a", y_column="b", z_column="c",
                     projection=screen.gates.gates.get("lasso1").projection,
                     vertices=((-1e9, -1e9), (1e9, -1e9), (0.0, 1e9)))
    gates = screen.gates.gates.add(child)
    assert np.array_equal(gates.mask(frame, "inner"), parent)


def test_a_projection_that_is_not_a_camera_is_refused():
    from spacr.qt.widgets.gate_spec import GateError

    with pytest.raises(GateError):
        ViewGate(name="g", x_column="a", y_column="b", z_column="c",
                 projection=((1, 0, 0, 0),), vertices=((0, 0), (1, 0), (0, 1)))
    with pytest.raises(GateError):
        ViewGate(name="g", x_column="a", y_column="a", z_column="c",
                 projection=np.eye(4), vertices=((0, 0), (1, 0), (0, 1)))


def test_an_orthographic_gate_is_a_prism_along_the_view():
    """Moving a point along the line of sight never changes its answer."""
    matrix = np.eye(4)
    gate = ViewGate(name="g", x_column="a", y_column="b", z_column="c",
                    projection=matrix,
                    vertices=((0, 0), (1, 0), (1, 1), (0, 1)))
    frame = pd.DataFrame({"a": [0.5, 0.5, 2.0], "b": [0.5, 0.5, 0.5],
                          "c": [-100.0, 100.0, 0.0]})
    assert gate.mask(frame).tolist() == [True, True, False]


def test_2d_is_untouched_by_a_view_gate(screen):
    canvas = screen.gates.canvas
    ax = _turn(canvas, 30.0, -30.0, 10.0)
    _draw(canvas, _outline(ax))
    screen.gates.mode_requested.emit("2D")
    assert not hasattr(canvas.axes_at(0, 0), "get_zlim")
    assert screen.gates.gates.names == ("lasso1",)
