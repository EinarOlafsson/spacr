"""The volume itself: what gets drawn in it, and where a click in it lands.

The 3D view is not decoration. It is the only view in which a third
measurement exists, so everything about it is load-bearing: the cloud has to
be the objects the table holds, the gates have to be drawn where their numbers
say, and -- above all -- a point clicked on a TURNED volume has to come back as
the measurement the user was pointing at, because that reading is what the next
gate is made of.

These drive a REAL ``Axes3D`` rather than a stand-in wherever the answer is
matplotlib's to give. A stand-in can only confirm the arithmetic somebody
already wrote down; the question here is whether a click on the picture the
user is actually looking at lands on the object under the cursor, and only the
real projection can answer that.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import GateCanvas, _project
from spacr.qt.widgets.gate_settings import GateEditorSettings
from spacr.qt.widgets.gate_spec import BoxGate, GateSet, RectGate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _table(n=300, seed=0):
    """Three measurements on three different scales.

    Different scales on purpose: a bug that reads pixels as data, or draws a
    shape before the limits are known, is invisible when every axis happens to
    run 0..1.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"area": rng.normal(4000.0, 600.0, n),
                         "ratio": rng.normal(5.0, 2.0, n),
                         "depth": rng.normal(-3.0, 1.0, n)})


@pytest.fixture
def canvas(qtbot):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def volume(canvas):
    """A canvas showing a real three-measurement volume."""
    canvas.set_frame(_table())
    canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
    canvas.set_mode("3D", z_column="depth")
    return canvas


class _Click:
    """The bits of a matplotlib mouse event the volume reads: pixels."""

    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)


def _pixels_of(axes, point):
    """Where a data point lands on screen, in the canvas' own pixels."""
    return axes.transData.transform(_project(axes, point))


def _clouds_of(axes):
    """Every 3D scatter drawn, in the order it was drawn."""
    return [[np.asarray(a, dtype=float) for a in c._offsets3d]
            for c in axes.collections if hasattr(c, "_offsets3d")]


def _cloud_of(axes):
    """The (x, y, z) the objects themselves were drawn at."""
    clouds = _clouds_of(axes)
    assert clouds, "nothing was drawn in the volume"
    return clouds[0]


def _ringed(axes):
    """How many objects each gate's highlight rings, the cloud excluded."""
    return [len(cloud[0]) for cloud in _clouds_of(axes)[1:]]


# ---------------------------------------------------------------------------
# Asking for the volume
# ---------------------------------------------------------------------------

def test_asking_for_the_third_measurement_draws_a_volume(volume):
    """A third measurement the user picked and a flat plot back is the whole
    complaint the 3D mode exists to answer."""
    axes = volume.axes_at(0, 0)
    assert hasattr(axes, "get_zlim3d")
    assert axes.get_zlabel() == "depth"
    assert (axes.get_xlabel(), axes.get_ylabel()) == ("area", "ratio")


def test_a_mode_nobody_offers_leaves_the_flat_plot_alone(canvas):
    """Only 2D, 3D and xD are drawable. Anything else is a caller's typo, and
    a typo must not leave the editor with no picture at all."""
    canvas.set_frame(_table())
    canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
    canvas.set_mode("sideways", z_column="depth")
    assert not hasattr(canvas.axes_at(0, 0), "get_zlim3d")
    assert canvas.axes_at(0, 0).get_xlabel() == "area"


def test_the_third_measurement_survives_a_mode_switch(volume):
    """Turning the volume off and on again must not lose the measurement the
    user picked -- re-picking it every time is the same click twice."""
    volume.set_mode("2D")
    assert not hasattr(volume.axes_at(0, 0), "get_zlim3d")
    volume.set_mode("3D")                       # no column named this time
    assert volume.axes_at(0, 0).get_zlabel() == "depth"


def test_a_third_measurement_the_table_does_not_have_falls_back_to_the_plot(
        volume):
    """A stale z column -- picked on one table, kept when another was loaded
    -- must cost the third dimension and nothing else. A blank canvas would
    read as the editor having crashed."""
    volume.set_mode("3D", z_column="not_a_column")
    axes = volume.axes_at(0, 0)
    assert not hasattr(axes, "get_zlim3d")
    assert axes.get_xlabel() == "area"          # the 2D plot really drew


def test_a_volume_with_no_rows_falls_back_to_the_plot(canvas):
    canvas.set_frame(_table().iloc[0:0])
    canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
    canvas.set_mode("3D", z_column="depth")
    assert not hasattr(canvas.axes_at(0, 0), "get_zlim3d")


# ---------------------------------------------------------------------------
# What the volume shows
# ---------------------------------------------------------------------------

def test_every_object_in_the_table_is_in_the_cloud(volume):
    xs, ys, zs = _cloud_of(volume.axes_at(0, 0))
    assert len(xs) == 300
    assert xs.min() == pytest.approx(volume.population()["area"].min())


def test_an_object_missing_a_measurement_is_left_out_rather_than_placed_at_zero(
        volume):
    """A row with no depth has no position in the volume. Drawing it anyway
    -- at zero, which is what a coerced NaN becomes -- would invent a cluster
    on the floor that no object is in."""
    frame = _table()
    frame.loc[frame.index[:20], "depth"] = np.nan
    frame["ratio"] = frame["ratio"].astype(object)
    frame.loc[frame.index[20:30], "ratio"] = "not a number"
    volume.set_frame(frame)
    volume.set_mode("3D", z_column="depth")
    xs, _ys, zs = _cloud_of(volume.axes_at(0, 0))
    assert len(xs) == 270
    assert np.isfinite(zs).all()


def test_the_view_the_user_turned_to_survives_a_redraw(volume):
    """Redrawing after a gate is added must not spin the volume back to the
    default -- the user turned it to see something."""
    volume.snap_to_nearest_axis()
    volume._view_angles = (30.0, 40.0)
    volume.render_now()
    axes = volume.axes_at(0, 0)
    assert (round(float(axes.elev)), round(float(axes.azim))) == (30, 40)


def test_a_zoomed_volume_redraws_still_zoomed(volume):
    """Same reason: the zoom is part of the view, and a redraw that threw it
    away would make every gate cost the zoom again."""
    wide = volume.axes_at(0, 0).get_zlim3d()
    volume._volume_zoom = 4.0
    volume.render_now()
    close = volume.axes_at(0, 0).get_zlim3d()
    assert (close[1] - close[0]) < (wide[1] - wide[0]) / 2.0


def test_matplotlibs_own_free_rotation_is_taken_over(volume):
    """The axis lock exists to replace free rotation. Leaving matplotlib's own
    drag-rotation connected would let the two fight over one drag."""
    assert volume.axes_at(0, 0)._rotate_btn == []


def test_the_volume_is_keyed_like_every_other_panel(volume):
    """`panel_axes()` is a contract: nothing downstream should have to know
    this panel is three-dimensional."""
    panels = volume.panel_axes()
    assert list(panels) == [(0, 0)]
    assert panels[(0, 0)] is volume.axes_at(0, 0)


def test_the_anchor_aura_lands_on_the_data_and_not_on_a_unit_square(volume):
    """Drawn AFTER the data, so the plane is sized in the measurement's own
    units. Drawing it first left a 0..1 quad on a measurement running in the
    thousands: technically present, visually absent."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    quads = [a for a in volume._artists if isinstance(a, Poly3DCollection)]
    assert len(quads) == 1
    corners = np.asarray(quads[0]._vec, dtype=float)
    low, high = volume.axes_at(0, 0).get_xlim3d()
    across = corners[0].max() - corners[0].min()
    assert across > 0.5 * (high - low), "the aura is not on the data's scale"


# ---------------------------------------------------------------------------
# Voxels -- the same volume when there are too many objects to draw one by one
# ---------------------------------------------------------------------------

class _Recorder:
    """Enough of an Axes3D to see what would have been drawn."""

    def __init__(self):
        self.calls = []

    def scatter(self, x, y, z, **kw):
        self.calls.append((np.asarray(x), kw))


def test_a_crowd_too_large_to_draw_one_by_one_is_drawn_as_occupancy(canvas):
    """Past the threshold every dot is drawn over by the ones in front of it,
    so the picture stops improving and only the frame rate changes."""
    canvas.apply_settings(GateEditorSettings(voxel_bins=8))
    canvas.set_frame(_table(canvas.VOXEL_THRESHOLD + 1))
    canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
    canvas.set_mode("3D", z_column="depth")
    xs, _ys, _zs = _cloud_of(canvas.axes_at(0, 0))
    # Voxels, not objects: far fewer marks than rows, and never more than the
    # grid has cells.
    assert 0 < len(xs) < 8 ** 3
    assert len(xs) < canvas.VOXEL_THRESHOLD


def test_a_measurement_too_extreme_to_bin_is_drawn_as_points_instead(canvas):
    """Values near the top of what a float can hold make the bin edges
    themselves overflow. The volume still has to appear -- a measurement with
    one absurd value is a data problem to SEE, not a blank canvas."""
    canvas.apply_settings(GateEditorSettings(voxel_bins=8))
    count = canvas.VOXEL_THRESHOLD + 1
    extreme = np.where(np.arange(count) % 2, 1e308, -1e308)
    axes = _Recorder()
    with np.errstate(over="ignore", invalid="ignore"):
        # The overflow is the POINT of the test; numpy shouting about it in
        # the run log is not.
        assert canvas._draw_voxels(axes, extreme, extreme, extreme) is False
    assert not axes.calls, "nothing may be drawn from bins that never formed"


@pytest.mark.parametrize("cloud", [
    "identical",        # every object at the same place
    "one_outlier",      # a single object a long way from the rest
    "two_clumps",
])
def test_every_object_lands_in_a_voxel(canvas, cloud):
    """The occupancy grid is a re-statement of the table, so the counts in it
    have to add up to the table. This is also why the "no voxel was filled"
    refusal never fires: with the grid taken from the data's own extent, there
    is no object that falls outside it."""
    canvas.apply_settings(GateEditorSettings(voxel_bins=6))
    count = canvas.VOXEL_THRESHOLD + 1
    rng = np.random.default_rng(1)
    if cloud == "identical":
        values = np.full(count, 7.0)
    elif cloud == "one_outlier":
        values = rng.normal(0.0, 1.0, count)
        values[0] = 1e6
    else:
        values = np.where(np.arange(count) % 2, rng.normal(-50, 1, count),
                          rng.normal(50, 1, count))
    axes = _Recorder()
    assert canvas._draw_voxels(axes, values, values, values) is True
    weights = np.asarray(axes.calls[0][1]["c"], dtype=float)
    assert weights.sum() == count


# ---------------------------------------------------------------------------
# Pixels to measurements
# ---------------------------------------------------------------------------

class _StubAxes3D:
    """A volume whose projection is ours to break.

    Only for the refusals: the questions below are "what does the editor do
    when matplotlib cannot answer", and matplotlib cannot be asked to fail.
    """

    def __init__(self, limits=((0.0, 10.0), (0.0, 20.0), (0.0, 30.0)),
                 transform=None):
        self._limits = limits

        class _TransData:
            @staticmethod
            def transform(point):
                if transform is not None:
                    return transform(point)
                return point

        self.transData = _TransData()
        self.elev = self.azim = 0.0

    def get_xlim3d(self):
        return self._limits[0]

    def get_ylim3d(self):
        return self._limits[1]

    def get_zlim3d(self):
        return self._limits[2]

    get_zlim = get_zlim3d

    def get_proj(self):
        return np.eye(4)


def test_a_flat_plot_has_no_pixels_to_volume_map(canvas):
    """There is no third measurement to place a point in, so refusing is the
    only honest answer -- guessing one would put a gate on a number nobody
    can see."""
    canvas.set_frame(_table())
    canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
    canvas.set_mode("2D")
    assert canvas.volume_axis_map() is None
    assert canvas.screen_to_volume(_Click(100, 100)) is None


def test_a_volume_missing_its_third_column_has_no_map(volume):
    """The mode says 3D but nothing names the depth: there is no plane to
    draw on."""
    volume._z_column = ""
    volume.axes_at = lambda *_a, **_k: _StubAxes3D()
    assert volume.volume_axis_map() is None


def test_a_measurement_with_no_extent_has_no_map(volume):
    """A measurement whose axis spans nothing cannot be read off the screen:
    every pixel would be the same value, so the inverse does not exist."""
    volume.axes_at = lambda *_a, **_k: _StubAxes3D(
        limits=((0.0, 10.0), (5.0, 5.0), (0.0, 30.0)))
    assert volume.volume_axis_map() is None


def test_a_projection_that_refuses_the_view_refuses_the_map(volume):
    """Built by measuring where the corners land rather than by trusting a
    formula -- so when the corners cannot be placed, the answer is "no map"
    and not a drag that silently lands in the wrong measurement."""
    def broken(_point):
        raise RuntimeError("this matplotlib moved the projection again")

    volume.axes_at = lambda *_a, **_k: _StubAxes3D(transform=broken)
    assert volume.volume_axis_map() is None


def test_one_corner_the_projection_cannot_place_refuses_the_whole_map(volume):
    """Three corners are measured. Two of them and a guess is exactly the
    silently-wrong drag the measurement approach exists to prevent."""
    seen = []

    def sometimes(point):
        seen.append(point)
        if len(seen) > 2:                       # the origin and one corner
            raise OverflowError("that corner is off the projection")
        return point

    volume.axes_at = lambda *_a, **_k: _StubAxes3D(transform=sometimes)
    assert volume.volume_axis_map() is None
    assert len(seen) == 3, "it stopped at the corner it could not place"


def test_a_plane_seen_exactly_edge_on_is_refused_rather_than_guessed(volume):
    """Edge-on, the plane's two measurements collapse onto one line of pixels
    and one pixel means a whole row of data points. Matplotlib's orthographic
    mode is where that is exact."""
    axes = volume.axes_at(0, 0)
    axes.set_proj_type("ortho")
    axes.view_init(elev=0.0, azim=0.0)
    volume._canvas.draw()
    volume.set_anchor_axis("z")                 # the flat X/Y plane...
    assert volume.volume_axis_map() is None     # ...is exactly edge-on here
    volume.set_anchor_axis("x")                 # the plane facing the camera
    assert volume.volume_axis_map() is not None
    # And the refusal is about the CAMERA, not about the plane: turn the
    # volume and the same plane reads again.
    axes.view_init(elev=30.0, azim=40.0)
    volume._canvas.draw()
    volume.set_anchor_axis("z")
    assert volume.volume_axis_map() is not None


@pytest.mark.parametrize("elev,azim", [(0.0, 0.0), (30.0, 40.0),
                                       (17.0, 213.0), (-25.0, 97.0)])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_a_gate_drawn_on_a_turned_volume_lands_where_the_user_drew_it(
        volume, elev, azim, axis):
    """The reading the next gate is made of.

    Perspective is not affine across a plane, so reading the cursor by
    stretching a flat inverse across the face lands the footprint a percent or
    two off -- visible as a gate that does not sit under the mouse. Inverting
    the camera ray and intersecting it with the chosen face is exact from
    every angle, which is what this asserts: within a ten-thousandth of the
    axis, not within a percent.
    """
    axes = volume.axes_at(0, 0)
    axes.view_init(elev=elev, azim=azim)
    volume._canvas.draw()
    volume.set_anchor_axis(axis)
    first, second, normal = volume.anchor_plane()

    index = {"area": 0, "ratio": 1, "depth": 2}
    limits = (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d())
    point = [0.0, 0.0, 0.0]
    for column, fraction in ((first, 0.25), (second, 0.4)):
        low, high = limits[index[column]]
        point[index[column]] = low + fraction * (high - low)
    point[index[normal]] = limits[index[normal]][0]     # on the anchor face

    x, y = _pixels_of(axes, point)
    got = volume.screen_to_volume(_Click(x, y))

    assert (got[0], got[2]) == (first, second)
    for name, value in ((got[0], got[1]), (got[2], got[3])):
        low, high = limits[index[name]]
        assert abs(value - point[index[name]]) < (high - low) * 1e-4


def test_a_click_before_the_first_paint_still_lands_on_the_object(volume):
    """The volume is drawn with `draw_idle`, so a click can arrive before
    matplotlib has stamped its projection matrix onto the axes -- `M` is None
    until the paint happens, which is exactly the state a redraw leaves it in
    until the event loop next runs. Refusing then would make the first click
    after every redraw do nothing."""
    axes = volume.axes_at(0, 0)
    axes.M = None                               # not painted yet
    volume.set_anchor_axis("z")
    limits = (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d())
    point = [limits[0][0] + 0.3 * (limits[0][1] - limits[0][0]),
             limits[1][0] + 0.6 * (limits[1][1] - limits[1][0]),
             limits[2][0]]
    x, y = _pixels_of(axes, point)
    got = volume.screen_to_volume(_Click(x, y))
    assert got[0] == "area" and got[2] == "ratio"
    assert got[1] == pytest.approx(point[0], rel=1e-6)
    assert got[3] == pytest.approx(point[1], rel=1e-6)


# ---------------------------------------------------------------------------
# Keeping what is in view
# ---------------------------------------------------------------------------

def test_the_view_is_the_gesture(volume):
    """Spinning and zooming until a population fills the box is already the
    act of choosing it. A rectangle dragged on a rotated projection has no
    defined extent along the axis pointing at the viewer, so reading one off
    would invent a number."""
    axes = volume.axes_at(0, 0)
    gate = volume.box_from_view()
    assert isinstance(gate, BoxGate)
    assert gate.columns == ("area", "ratio", "depth")
    assert (gate.x_low, gate.x_high) == pytest.approx(tuple(axes.get_xlim3d()))
    assert (gate.z_low, gate.z_high) == pytest.approx(tuple(axes.get_zlim3d()))


def test_what_is_in_view_is_what_the_gate_keeps(volume):
    """The claim the gate makes has to be the picture: every object drawn
    inside the frame is in it, and one moved outside is not."""
    gate = volume.box_from_view()
    frame = volume.population()
    assert gate.mask(frame).all()
    outside = frame.copy()
    outside.loc[outside.index[0], "depth"] = 1e6
    assert not gate.mask(outside)[0]


def test_there_is_nothing_to_keep_on_a_flat_plot(canvas):
    canvas.set_frame(_table())
    canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
    canvas.set_mode("2D")
    assert canvas.box_from_view() is None


def test_there_is_nothing_to_keep_without_a_third_measurement(volume):
    volume._z_column = ""
    assert volume.box_from_view() is None


# ---------------------------------------------------------------------------
# Gates, seen in the volume
# ---------------------------------------------------------------------------

def _box(name="kept", **bounds):
    fields = dict(x_column="area", y_column="ratio", z_column="depth",
                  x_low=3500.0, x_high=4500.0, y_low=4.0, y_high=6.0,
                  z_low=-4.0, z_high=-2.0)
    fields.update(bounds)
    return BoxGate(name=name, **fields)


def _edges_of(axes):
    """The line segments drawn in the volume, as (x, y, z) triples."""
    out = []
    for line in axes.lines:
        xs, ys = line.get_data()
        zs = np.asarray(line.get_data_3d()[2], dtype=float) \
            if hasattr(line, "get_data_3d") else np.zeros(len(xs))
        out.append((np.asarray(xs, float), np.asarray(ys, float), zs))
    return out


def test_a_box_gate_is_drawn_as_the_twelve_edges_of_a_box(volume):
    volume.set_gates(GateSet().add(_box()))
    assert len(_edges_of(volume.axes_at(0, 0))) == 12


def test_the_edges_are_the_numbers_the_gate_holds(volume):
    """An outline that is not where the gate's numbers are is worse than no
    outline: it says the wrong objects are in it."""
    volume.set_gates(GateSet().add(_box()))
    xs = np.concatenate([e[0] for e in _edges_of(volume.axes_at(0, 0))])
    ys = np.concatenate([e[1] for e in _edges_of(volume.axes_at(0, 0))])
    zs = np.concatenate([e[2] for e in _edges_of(volume.axes_at(0, 0))])
    assert sorted(set(np.round(xs, 6))) == [3500.0, 4500.0]
    assert sorted(set(np.round(ys, 6))) == [4.0, 6.0]
    assert sorted(set(np.round(zs, 6))) == [-4.0, -2.0]


def test_a_box_left_open_is_drawn_out_to_the_data_it_covers(volume):
    """An unbounded side is a statement about the measurements it DOES name,
    so it is drawn spanning everything -- a box stopping at zero would say the
    gate ends there."""
    volume.set_gates(GateSet().add(_box(z_low=None, z_high=None)))
    zs = np.concatenate([e[2] for e in _edges_of(volume.axes_at(0, 0))])
    depth = volume.population()["depth"]
    assert min(zs) == pytest.approx(float(depth.min()))
    assert max(zs) == pytest.approx(float(depth.max()))


def test_a_gate_toggled_off_is_not_drawn_in_the_volume(volume):
    """Off means not drawn -- never deleted. The gate comes back exactly as
    it was."""
    volume.set_gates(GateSet().add(_box()))
    volume.set_gate_enabled("kept", False)
    assert _edges_of(volume.axes_at(0, 0)) == []
    volume.set_gate_enabled("kept", True)
    assert len(_edges_of(volume.axes_at(0, 0))) == 12


def test_a_two_measurement_gate_marks_its_objects_without_inventing_a_depth(
        volume):
    """A 2D gate is a statement about two of the three measurements, so in a
    volume it is a COLUMN through the cloud. Marking its objects says exactly
    that; drawing a box would claim a depth the gate never mentioned."""
    gates = GateSet().add(RectGate(name="flat", x_column="area",
                                   y_column="ratio", x_low=3500.0,
                                   x_high=4500.0, y_low=4.0, y_high=6.0))
    volume.set_gates(gates)
    axes = volume.axes_at(0, 0)
    assert _edges_of(axes) == []
    inside = int(gates.mask(volume.population(), "flat").sum())
    assert 0 < inside < 300, "the gate has to select some of the objects"
    assert _ringed(axes) == [inside]


def test_a_box_on_a_different_third_measurement_is_not_drawn_as_a_box(volume):
    """Its z range is about a measurement this volume is not showing, so
    drawing the box would put its faces at depths that mean nothing here. The
    objects are still marked."""
    frame = volume.population().copy()
    frame["other"] = frame["depth"] * 2.0
    volume.set_frame(frame)
    volume.set_mode("3D", z_column="depth")
    gates = GateSet().add(_box(z_column="other"))
    volume.set_gates(gates)
    axes = volume.axes_at(0, 0)
    assert _edges_of(axes) == []
    inside = int(gates.mask(volume.population(), "kept").sum())
    assert inside > 0
    assert _ringed(axes) == [inside], "the objects are still ringed"


def test_a_gate_that_selects_nothing_is_not_drawn(volume):
    """Nothing to ring, and an outline round an empty region reads as a
    population that is there."""
    volume.set_gates(GateSet().add(_box(x_low=1e9, x_high=2e9)))
    axes = volume.axes_at(0, 0)
    assert _edges_of(axes) == []
    assert _ringed(axes) == []


def test_a_gate_this_table_cannot_answer_costs_that_gate_and_not_the_view(
        volume):
    """A gate carried over from another table names a measurement this one
    does not have. Skipping it leaves the volume and every other gate on
    screen, which is what someone comparing two tables needs."""
    gates = GateSet()
    gates.add(RectGate(name="stale", x_column="gone", y_column="ratio",
                       x_low=0.0, x_high=1.0, y_low=0.0, y_high=1.0))
    gates.add(_box(name="here"))
    volume.set_gates(gates)
    axes = volume.axes_at(0, 0)
    assert len(_edges_of(axes)) == 12, "the good gate is still drawn"
    assert len(_cloud_of(axes)[0]) == 300, "and so is the cloud"
    # One highlight, and it is the gate the table CAN answer.
    assert _ringed(axes) == [int(gates.mask(volume.population(),
                                            "here").sum())]


# ---------------------------------------------------------------------------
# Spin speed, when matplotlib will not co-operate
# ---------------------------------------------------------------------------

class _Axes:
    def __init__(self):
        self._sx, self._sy = 100, 100
        self.seen = []

    def _on_move(self, event):
        self.seen.append((event.x, event.y))


class _ReadOnlyEvent:
    """A drag matplotlib will not let us rewrite."""

    def __init__(self, x, y):
        self._x, self._y = x, y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


def test_a_drag_the_wrapper_cannot_rewrite_still_turns_the_volume(canvas):
    """The speed is a preference; turning at all is the feature. An event
    object that will not take a new position leaves the rotation at
    matplotlib's own speed rather than swallowing the drag."""
    canvas.apply_settings(GateEditorSettings(spin_speed=3.0))
    axes = _Axes()
    canvas._apply_spin_speed(axes)
    axes._on_move(_ReadOnlyEvent(140, 100))
    assert axes.seen == [(140, 100)], "the drag reached matplotlib unscaled"


class _SealedAxes:
    """A matplotlib whose drag handler cannot be replaced."""

    def __init__(self):
        self._sx, self._sy = 100, 100
        self.seen = []

    def _record(self, event):
        self.seen.append((event.x, event.y))

    @property
    def _on_move(self):
        return self._record


class _Move:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_a_matplotlib_that_will_not_take_the_wrap_spins_at_its_own_speed(
        canvas):
    """A setting not taking effect, rather than a volume that will not turn.
    Wrapping is the only way to change the speed -- matplotlib has no public
    setting for it -- so this is the failure that has to be survivable."""
    canvas.apply_settings(GateEditorSettings(spin_speed=2.0))
    axes = _SealedAxes()
    canvas._apply_spin_speed(axes)
    axes._on_move(_Move(110, 100))
    assert axes.seen == [(110, 100)]
    assert not getattr(axes._on_move, "_spacr_wrapped", False)


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------

def test_snapping_with_nothing_on_screen_turns_nothing(canvas):
    """The release handler asks for a snap on every drag that ends. Before
    anything is drawn there is no view to square up, and an exception here
    would be raised on a mouse-up in an empty editor."""
    assert canvas.axes_at(0, 0) is None
    assert canvas.snap_to_nearest_axis() == (0.0, 0.0)


def test_snapping_a_real_volume_leaves_one_measurement_flat(volume):
    """The point of snapping: a 3D gate is finally judged from a view where
    one measurement is flat."""
    axes = volume.axes_at(0, 0)
    axes.view_init(elev=23.0, azim=47.0)
    assert volume.snap_to_nearest_axis() == (0.0, 90.0)
    assert (float(axes.elev), float(axes.azim)) == (0.0, 90.0)
    assert volume._view_angles == (0.0, 90.0)
