"""N673 follow-up (2026-10-24): what one drag move costs in True 3D.

Windows acceptance passed for every backend, but a drag cost about 90 ms per
mouse move. Profiling one move found three costs, each paid on every move:

* matplotlib 3.8 depth-sorts a boundary mesh face by face in Python, which
  was most of every redraw while the view turned;
* pulling an anchor or a surface validated the whole closed boundary (every
  pair of nearby triangles) and then rebuilt the whole scene, axes included;
* the anchors listed under a press were drawn one collection each.

These checks count work instead of timing it, so they hold on any machine:
the mesh is sorted and painted exactly as matplotlib sorts and paints it, a
burst of moves rebuilds nothing, never validates the whole boundary and is
painted at most once, a move that would fold the boundary still says why at
once, and the release still validates once and keeps exactly the edit the
old path kept. No GPU and no network: the small synthetic stack of
test_673_napari_true3d, drawn on the CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import gate_editor as GE
from spacr.qt.widgets import gate_spec

from .test_673_napari_true3d import WAIT, _settle, cpu_view, screen, view

__all__ = ["cpu_view", "screen", "view"]

MOVES = 12


def _projected(collection):
    paths = [(path.vertices.copy(), None if path.codes is None else path.codes.copy())
             for path in collection.get_paths()]
    return paths, np.asarray(collection._facecolors2d), np.asarray(collection._edgecolors2d)


@pytest.mark.parametrize("zsort", ["average", "min", "max"])
def test_the_mesh_is_sorted_and_painted_as_matplotlib_paints_it(zsort):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rng = np.random.default_rng(673)
    triangles = rng.uniform(-1.0, 1.0, (600, 3, 3))
    triangles[100:120] = triangles[99]
    colours = rng.uniform(0.0, 1.0, (600, 4))
    figure = Figure(figsize=(4, 4), dpi=80)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(projection="3d")
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_axis_off()
    for elevation, azimuth in ((20.0, -60.0), (89.0, 0.0), (-45.0, 130.0)):
        ax.view_init(elevation, azimuth)
        pictures, projections = [], []
        for make in (GE._triangle_mesh, Poly3DCollection):
            artist = make(triangles, facecolors=colours, edgecolors=(0.1, 0.2, 0.9, 0.7),
                          linewidths=0.5, zsort=zsort)
            ax.add_collection3d(artist)
            figure.canvas.draw()
            pictures.append(np.asarray(figure.canvas.buffer_rgba()).copy())
            projections.append((artist.do_3d_projection(), _projected(artist)))
            artist.remove()
        (fast_z, (fast_paths, fast_face, fast_edge)), (slow_z, (slow_paths, slow_face, slow_edge)) = projections
        assert type(GE._triangle_mesh(triangles)).__mro__[1] is Poly3DCollection
        assert fast_z == slow_z
        assert len(fast_paths) == len(slow_paths) == len(triangles)
        for (fast_xy, fast_codes), (slow_xy, slow_codes) in zip(fast_paths, slow_paths):
            assert np.array_equal(fast_xy, slow_xy) and np.array_equal(fast_codes, slow_codes)
        assert np.array_equal(fast_face, slow_face) and np.array_equal(fast_edge, slow_edge)
        assert np.array_equal(pictures[0], pictures[1])


def _chosen(stage, qtbot, label=1):
    stage._labels.setCurrentIndex(stage._labels.findData(label))
    qtbot.waitUntil(lambda: stage._mesher.idle() and label in stage._meshes, timeout=WAIT)
    _settle()
    qtbot.waitUntil(stage.render_idle, timeout=WAIT)
    _settle()
    canvas = stage._panel.canvas
    assert canvas.drag_mode() == "edit"
    canvas._canvas.draw()
    _settle()
    return canvas


def _move(canvas, x, y):
    widget = canvas._canvas
    ratio = widget.device_pixel_ratio
    local = QPointF(x / ratio, widget.figure.bbox.height / ratio - y / ratio)
    QApplication.sendEvent(widget, QMouseEvent(QEvent.MouseMove, local, widget.mapToGlobal(local),
                                               Qt.NoButton, Qt.LeftButton, Qt.NoModifier))


class _At:
    def __init__(self, canvas, x, y):
        self.x, self.y, self.inaxes, self.button = x, y, canvas.axes_at(0, 0), 1
        self.key = None


def _counted(monkeypatch, canvas):
    counts = {"rebuilds": 0, "validations": 0, "draws": 0}
    render_now = canvas.render_now
    validate = gate_spec._validate_volume_mesh

    def rebuilt():
        counts["rebuilds"] += 1
        return render_now()

    def validated(*args, **kwargs):
        counts["validations"] += 1
        return validate(*args, **kwargs)

    monkeypatch.setattr(canvas, "render_now", rebuilt)
    monkeypatch.setattr(gate_spec, "_validate_volume_mesh", validated)
    canvas._canvas.mpl_connect("draw_event", lambda _event: counts.__setitem__("draws", counts["draws"] + 1))
    return counts


def _outward(canvas, mesh, start):
    """A unit screen direction from the boundary's middle through ``start``.

    Pulling an anchor or surface a few pixels away from the middle of a
    rounded boundary cannot fold it, whatever size and pixel ratio the
    canvas has; pulling it inward on a small canvas can.
    """
    order = [mesh.columns.index(column) for column in canvas._plot_columns()]
    middle = GE._screen_points(canvas.axes_at(0, 0),
                               [np.asarray(mesh.vertices).mean(axis=0)[order]])[0]
    away = np.asarray(start, float) - middle
    return away / max(float(np.hypot(*away)), 1e-9)


def _burst(canvas, start, counts, direction):
    """Send the moves back to back, then let the event loop run once.

    :param direction: the unit screen direction the pointer moves in.
    :returns: the last move as matplotlib read it, which is where the
        release lands.
    """
    seen = []
    watch = canvas._canvas.mpl_connect("motion_notify_event", lambda event: seen.append((event.x, event.y)))
    for step in range(1, MOVES + 1):
        _move(canvas, start[0] + step * 0.3 * direction[0], start[1] + step * 0.3 * direction[1])
    canvas._canvas.mpl_disconnect(watch)
    assert len(seen) == MOVES
    assert counts["rebuilds"] == 0 and counts["validations"] == 0
    assert counts["draws"] == 0, "a move painted before the event loop ran"
    _settle()
    assert counts["rebuilds"] == 0 and counts["validations"] == 0
    assert counts["draws"] == 1, f"{MOVES} moves painted {counts['draws']} times"
    return seen[-1]


def _drawn_corners(canvas, name):
    """The triangles the artist holds, in either matplotlib's layout.

    Before 3.10 a TriangleMesh kept them as ``_vec``; from 3.10 matplotlib
    calls ``_get_vector`` and keeps them as ``_faces`` itself, already in
    one NumPy array, so the mesh's own fast path is not taken there.
    """
    artist = canvas._mesh_artists[name]
    vec = getattr(artist, "_vec", None)
    if vec is not None:
        return np.asarray(vec[:3]).T.reshape(-1, 3, 3)
    return np.asarray(artist._faces)


def _expected_corners(canvas, mesh):
    order = [mesh.columns.index(column) for column in canvas._plot_columns()]
    return np.asarray(mesh.vertices)[:, order][np.asarray(mesh.faces)]


@pytest.mark.parametrize("weights", [1, 2])
def test_a_burst_of_anchor_moves_rebuilds_nothing_and_releases_the_old_edit(cpu_view, qtbot, monkeypatch, weights):
    stage = cpu_view
    canvas = _chosen(stage, qtbot)
    gate = canvas._handle_gate()
    handles = canvas._volume_handles(gate)
    role, point = next((role, point) for role, point in handles
                       if role.startswith("anchor:") and role.count(",") == weights - 1)
    start = GE._screen_points(canvas.axes_at(0, 0), [point])[0]
    assert canvas._start_volume_handle(gate, role, point, float(start[0]), float(start[1]))
    counts = _counted(monkeypatch, canvas)
    end = _burst(canvas, start, counts, _outward(canvas, canvas._editable_boundary(gate), start))
    expected = canvas._handle_moved(_At(canvas, *end))
    assert counts["validations"] == 1
    assert isinstance(expected, gate_spec.EditableVolumeGate) and expected != gate
    assert np.array_equal(_drawn_corners(canvas, gate.name), _expected_corners(canvas, expected))
    assert not any(artist.get_visible() for artist in canvas._handle_artists)
    edited = []
    canvas.gate_edited.connect(edited.append)
    canvas._volume_release(_At(canvas, *end))
    assert edited == [expected]
    qtbot.waitUntil(lambda: stage._worker.idle() and not stage.busy, timeout=WAIT)
    _settle()


def test_a_burst_of_surface_moves_rebuilds_nothing_and_releases_the_old_edit(cpu_view, qtbot, monkeypatch):
    stage = cpu_view
    canvas = _chosen(stage, qtbot)
    gate = canvas._handle_gate()
    mesh = canvas._editable_boundary(gate)
    group = mesh.surface_groups[len(mesh.surface_groups) // 2]
    canvas._surface_gate, canvas._selected_surfaces = mesh.name, {group}
    centre = np.asarray(mesh.vertices)[sorted(mesh._surface_vertices([group]))].mean(axis=0)
    order = [mesh.columns.index(column) for column in canvas._plot_columns()]
    start = GE._screen_points(canvas.axes_at(0, 0), [centre[order]])[0]
    assert canvas._drag_selected_surfaces(_At(canvas, float(start[0]), float(start[1])))
    counts = _counted(monkeypatch, canvas)
    end = _burst(canvas, start, counts, _outward(canvas, canvas._editable_boundary(gate), start))
    expected = canvas._surface_moved(_At(canvas, *end))
    assert counts["validations"] == 1
    assert isinstance(expected, gate_spec.EditableVolumeGate) and expected != mesh
    assert np.array_equal(_drawn_corners(canvas, mesh.name), _expected_corners(canvas, expected))
    edited = []
    canvas.gate_edited.connect(edited.append)
    canvas._volume_release(_At(canvas, *end))
    assert edited == [expected]
    qtbot.waitUntil(lambda: stage._worker.idle() and not stage.busy, timeout=WAIT)
    _settle()


def test_listed_anchors_are_drawn_as_one_ring_collection_and_one_chosen_anchor(cpu_view, qtbot):
    stage = cpu_view
    canvas = _chosen(stage, qtbot)
    gate = canvas._handle_gate()
    handles = [(role, point) for role, point in canvas._volume_handles(gate) if role.startswith("anchor:")][:40]
    canvas._overlap = [GE._OverlapHit("anchor", gate.name, role, 0, 0.0, 0.0, index + 1, point)
                       for index, (role, point) in enumerate(handles)]
    canvas._overlap_choice = 3
    before = len(canvas.axes_at(0, 0).collections)
    canvas._handle_artists = []
    canvas._draw_overlap_anchors(canvas.axes_at(0, 0), dict(handles))
    assert len(canvas.axes_at(0, 0).collections) - before == 2
    rings, chosen = canvas._handle_artists
    assert len(rings.get_offsets()) == len(handles) - 1 and len(chosen.get_offsets()) == 1
    canvas._overlap, canvas._overlap_choice = [], 0


def test_a_refused_anchor_edit_is_still_refused_on_release_and_the_view_rebuilt(cpu_view, qtbot, monkeypatch):
    stage = cpu_view
    canvas = _chosen(stage, qtbot)
    gate = canvas._handle_gate()
    role, point = next((role, point) for role, point in canvas._volume_handles(gate)
                       if role.startswith("anchor:") and role.count(",") == 1)
    start = GE._screen_points(canvas.axes_at(0, 0), [point])[0]
    assert canvas._start_volume_handle(gate, role, point, float(start[0]), float(start[1]))
    counts = _counted(monkeypatch, canvas)
    said = []
    canvas.depth_requested.connect(said.append)
    end = (float(start[0]) + 400.0, float(start[1]) + 300.0)
    _move(canvas, *end)
    _settle()
    assert counts["rebuilds"] == 0 and counts["validations"] == 0
    assert said, "the move did not say why the edit would be refused"
    assert np.array_equal(_drawn_corners(canvas, gate.name),
                          _expected_corners(canvas, canvas._editable_boundary(gate)))
    said.clear()
    edited = []
    canvas.gate_edited.connect(edited.append)
    canvas._volume_release(_At(canvas, *end))
    assert counts["validations"] == 1 and counts["rebuilds"] == 1
    assert edited == [] and said
    assert np.array_equal(_drawn_corners(canvas, gate.name),
                          _expected_corners(canvas, canvas._editable_boundary(gate)))
