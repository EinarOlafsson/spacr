"""The working table set, the 3D volume and xD, through the screen.

These go through the SCREEN rather than the widgets because every one of them
was reported as "there is a button and nothing happens" -- the wiring is the
thing under test, not the geometry underneath it.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection


def _cells(n=3):
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["A"] * n,
        "columnID": ["1"] * n, "fieldID": ["f1"] * n,
        "object_label": range(1, n + 1),
        "area": np.linspace(100.0, 300.0, n),
        "mean_intensity": np.linspace(10.0, 30.0, n),
    })


def _pathogens():
    return pd.DataFrame({
        "plateID": ["p1"] * 4, "rowID": ["A"] * 4,
        "columnID": ["1"] * 4, "fieldID": ["f1"] * 4,
        "cell_id": [1, 1, 2, 3], "object_label": [1, 2, 1, 1],
        "area": [10.0, 20.0, 5.0, 7.0],
        "min_intensity": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as conn:
        _cells().to_sql("cell", conn, index=False)
        _pathogens().to_sql("pathogen", conn, index=False)
    return path


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# The working set
# ---------------------------------------------------------------------------

def test_picking_a_table_adds_it_rather_than_switching(screen, db):
    """"dont just switch to nuclei, add nuclei". Before this, picking a second
    table replaced the first and both axes came from whichever was loaded --
    a plot that looked like the one the user wanted and was not."""
    screen.load_path(db, "cell")
    assert screen._tables == ["cell"]

    screen._table_picker.setCurrentText("pathogen")
    screen._on_table_added(0)

    assert screen._tables == ["cell", "pathogen"]
    columns = list(screen.gates._frame.columns)
    assert "cell_area" in columns
    assert "pathogen_area" in columns


def test_a_measurement_from_each_object_can_go_on_its_own_axis(screen, db):
    """"i want to be able to have a cell measurement on one axis nuclear on
    another and pathogen on a thired"."""
    screen.load_path(db, "cell")
    screen._table_picker.setCurrentText("pathogen")
    screen._on_table_added(0)

    options = [screen._x.itemText(i) for i in range(screen._x.count())]
    assert "cell_area" in options and "pathogen_area" in options


def test_each_table_in_the_set_has_a_removable_chip(screen, db, qtbot):
    from spacr.qt.widgets.table_chip import TableChip

    screen.load_path(db, "cell")
    screen._table_picker.setCurrentText("pathogen")
    screen._on_table_added(0)

    chips = screen.findChildren(TableChip)
    assert {c.name for c in chips} == {"cell", "pathogen"}


def test_removing_a_chip_takes_its_measurements_away(screen, db):
    screen.load_path(db, "cell")
    screen._table_picker.setCurrentText("pathogen")
    screen._on_table_added(0)
    screen.remove_table("pathogen")

    assert screen._tables == ["cell"]
    assert not any(c.startswith("pathogen_")
                   for c in screen.gates._frame.columns)


def test_the_last_table_cannot_be_removed(screen, db):
    """A gate editor with no table is a screen with nothing on it."""
    screen.load_path(db, "cell")
    screen.remove_table("cell")
    assert screen._tables == ["cell"]


def test_adding_the_same_table_twice_does_nothing(screen, db):
    screen.load_path(db, "cell")
    screen._table_picker.setCurrentText("cell")
    screen._on_table_added(0)
    assert screen._tables == ["cell"]


def test_a_single_table_is_not_renamed_by_a_merge(screen, db):
    """Merging one table onto itself only renames its columns, and would make
    every gate saved in a single-table session stop matching."""
    screen.load_path(db, "cell")
    assert "area" in screen.gates._frame.columns
    assert "cell_area" not in screen.gates._frame.columns


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def _three_columns(screen):
    rng = np.random.default_rng(0)
    screen.set_frame(pd.DataFrame({"a": rng.normal(0, 1, 200),
                                   "b": rng.normal(0, 1, 200),
                                   "c": rng.normal(0, 1, 200)}))
    screen._x.setCurrentText("a")
    screen._y.setCurrentText("b")


def test_the_z_picker_appears_in_3d_and_is_remembered_in_2d(screen):
    _three_columns(screen)
    assert not screen._z.isVisible()

    screen.gates.mode_requested.emit("3D")
    assert screen._z.isVisibleTo(screen)
    screen._z.setCurrentText("c")

    screen.gates.mode_requested.emit("2D")
    assert screen._z.currentText() == "c", "the third measurement was lost"


def test_3d_actually_draws_a_third_axis(screen):
    """"there is a thired column in 3d but nothing hapens when i pressit ...
    no extra axis is added"."""
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    screen._z.setCurrentText("c")

    ax = screen.gates.canvas.axes_at(0, 0)
    assert hasattr(ax, "zaxis"), "the plot is still two-dimensional"
    assert ax.collections, "nothing was drawn in the volume"


def test_the_volume_can_be_turned(screen):
    """"there is no way to turn the graph". Rotation is the whole reason to
    be in 3D -- a fixed view of a volume tells you less than two scatters."""
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    screen._z.setCurrentText("c")

    ax = screen.gates.canvas.axes_at(0, 0)
    before = (ax.elev, ax.azim)
    ax.view_init(elev=35.0, azim=145.0)
    assert (ax.elev, ax.azim) != before


def test_snapping_turns_the_volume_square_on(screen):
    """A volume stopped at an arbitrary angle cannot be read off at all."""
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    screen._z.setCurrentText("c")

    ax = screen.gates.canvas.axes_at(0, 0)
    ax.view_init(elev=12.0, azim=100.0)
    elevation, azimuth = screen.gates.canvas.snap_to_nearest_axis()
    assert elevation in (0.0, 90.0, -90.0)
    assert azimuth in (0.0, 90.0, 180.0, 270.0)


def test_going_back_to_2d_restores_the_flat_axes(screen):
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    screen._z.setCurrentText("c")
    screen.gates.mode_requested.emit("2D")
    assert not hasattr(screen.gates.canvas.axes_at(0, 0), "zaxis")


def test_3d_without_a_third_measurement_stays_flat(screen):
    """Rather than an empty volume, which reads as the mode being broken."""
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    assert not hasattr(screen.gates.canvas.axes_at(0, 0), "zaxis")


# ---------------------------------------------------------------------------
# xD
# ---------------------------------------------------------------------------

def _wide(screen, n=300):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, n)
    screen.set_frame(pd.DataFrame({
        "a": base + rng.normal(0, 0.05, n), "b": base * 2.0,
        "c": rng.normal(0, 1, n), "d": rng.normal(0, 1, n)}))


def test_xd_projects_onto_components_and_puts_them_on_the_axes(screen):
    """"xD has a button but further than that no implementation"."""
    _wide(screen)
    screen.gates.projection_requested.emit(True)

    columns = list(screen.gates._frame.columns)
    assert {"PC1", "PC2", "PC3"} <= set(columns)
    assert screen._x.currentText() == "PC1"
    assert screen._y.currentText() == "PC2"
    assert screen._z.currentText() == "PC3"


def test_the_components_say_how_much_they_explain(screen):
    """"PC1" alone says nothing about whether it is the data or the noise."""
    _wide(screen)
    screen.gates.projection_requested.emit(True)
    assert "%" in screen._source.text()


def test_every_gate_tool_works_on_a_component(screen):
    """The point of returning ordinary columns: a gate on PC1 vs PC2 is the
    same kind of object as a gate on area vs intensity."""
    from spacr.qt.widgets.gate_spec import RectGate

    _wide(screen)
    screen.gates.projection_requested.emit(True)
    frame = screen.gates._frame
    gate = RectGate(name="g", x_column="PC1", y_column="PC2",
                    x_low=-100.0, x_high=100.0, y_low=-100.0, y_high=100.0)
    assert gate.mask(frame).all()


def test_projecting_twice_does_not_stack_component_columns(screen):
    _wide(screen)
    screen.gates.projection_requested.emit(True)
    screen.gates.projection_requested.emit(True)
    columns = list(screen.gates._frame.columns)
    assert columns.count("PC1") == 1


def test_xd_with_nothing_to_project_says_so(screen):
    screen.set_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    message = screen.reduce_to_components()
    assert message and "two" in message


# --- the follow-up round -------------------------------------------------

def test_a_settings_reload_keeps_the_merged_tables(screen, db):
    """Sampling re-reads the table. Before this it re-read only the FIRST
    one, so every settings change silently unmerged the working set."""
    screen.load_path(db, "cell")
    screen._table_picker.setCurrentText("pathogen")
    screen._on_table_added(0)

    screen.apply_settings(screen.settings().replaced(sample_fraction=0.5))

    assert screen._tables == ["cell", "pathogen"]
    assert "pathogen_area" in screen.gates._frame.columns


def test_reset_view_undoes_a_zoom(screen):
    _three_columns(screen)
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)
    before = ax.get_xlim()

    class _Scroll:
        inaxes = ax
        xdata = sum(before) / 2
        ydata = sum(ax.get_ylim()) / 2
        step = 3
        button = "up"

    canvas._on_scroll(_Scroll())
    assert canvas.axes_at(0, 0).get_xlim() != before

    screen.gates.reset_view()
    assert canvas.axes_at(0, 0).get_xlim() == pytest.approx(before)


def test_reset_view_undoes_a_spin(screen):
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    screen._z.setCurrentText("c")
    canvas = screen.gates.canvas
    before = (canvas.axes_at(0, 0).elev, canvas.axes_at(0, 0).azim)

    canvas.axes_at(0, 0).view_init(elev=70.0, azim=15.0)
    canvas._view_angles = (70.0, 15.0)
    screen.gates.reset_view()

    after = (canvas.axes_at(0, 0).elev, canvas.axes_at(0, 0).azim)
    assert after == pytest.approx(before)


class _Mouse:
    def __init__(self, ax, x, y, step=0):
        self.inaxes, self.x, self.y, self.step = ax, x, y, step
        self.xdata = self.ydata = 0.0
        self.button = 1


def _volume(screen):
    _three_columns(screen)
    screen.gates.mode_requested.emit("3D")
    screen._z.setCurrentText("c")
    return screen.gates.canvas


def test_the_volume_spins_without_a_tool_armed(screen):
    """"i cant zoom in or spin on any of the axees. if i press pollygon and
    tried to draw a gate, then i could all of a suded spinn the graph" -- the
    2D press handler was eating the drag, and only the polygon tool, which
    ignores drags, let it through."""
    canvas = _volume(screen)
    canvas.set_tool("")          # no tool armed: a drag is navigation
    ax = canvas.axes_at(0, 0)
    before = float(ax.azim)

    canvas._on_press(_Mouse(ax, 100, 100))
    canvas._on_motion(_Mouse(ax, 160, 100))
    canvas._on_release(_Mouse(ax, 160, 100))

    assert float(ax.azim) != before, "a drag in the volume did not spin it"


def test_draw_mode_draws_instead_of_spinning(screen):
    """Spin/Draw is the source of truth.  An old 2D tool remains armed while
    spinning, so tool presence cannot decide what a volume drag means.
    """
    from spacr.qt.widgets.gate_spec import RECTANGLE

    canvas = _volume(screen)
    canvas.set_tool(RECTANGLE)
    canvas.set_drag_mode("draw")
    ax = canvas.axes_at(0, 0)
    azimuth = float(ax.azim)

    canvas._on_press(_Mouse(ax, 100, 100))
    assert canvas._volume_drag is not None, "an armed tool did not start a gate"
    canvas._on_motion(_Mouse(ax, 200, 180))
    assert canvas._ghost, "nothing followed the mouse"
    assert float(ax.azim) == azimuth, "drawing spun the volume as well"
    canvas._volume_drag = None   # drop it rather than raising the name prompt


def test_spinning_about_z_leaves_the_horizon_level(screen):
    canvas = _volume(screen)
    canvas.set_tool("")
    ax = canvas.axes_at(0, 0)
    canvas.set_spin_axis("z")
    elevation = float(ax.elev)

    canvas._on_press(_Mouse(ax, 100, 100))
    canvas._on_motion(_Mouse(ax, 160, 140))
    canvas._on_release(_Mouse(ax, 160, 140))

    assert float(ax.elev) == pytest.approx(elevation), (
        "a z-locked spin changed the elevation, so it is not locked")


def test_spinning_about_x_leaves_the_azimuth_alone(screen):
    canvas = _volume(screen)
    canvas.set_tool("")
    ax = canvas.axes_at(0, 0)
    canvas.set_spin_axis("x")
    azimuth = float(ax.azim)

    canvas._on_press(_Mouse(ax, 100, 100))
    canvas._on_motion(_Mouse(ax, 160, 140))
    canvas._on_release(_Mouse(ax, 160, 140))

    assert float(ax.azim) == pytest.approx(azimuth)


def test_the_wheel_zooms_the_volume(screen):
    canvas = _volume(screen)
    ax = canvas.axes_at(0, 0)
    before = ax.get_zlim()
    canvas._on_scroll(_Mouse(ax, 100, 100, step=1))
    assert ax.get_zlim() != before, "the wheel did nothing in 3D"


def test_the_spin_axis_buttons_only_appear_in_3d(screen):
    """A dead control is worse than no control."""
    _three_columns(screen)
    assert not screen.gates._spin_buttons["x"].isVisibleTo(screen.gates)
    screen.gates.mode_requested.emit("3D")
    assert screen.gates._spin_buttons["x"].isVisibleTo(screen.gates)
    screen.gates.mode_requested.emit("2D")
    assert not screen.gates._spin_buttons["x"].isVisibleTo(screen.gates)


def test_a_spin_button_reaches_the_canvas(screen):
    _volume(screen)
    screen.gates.spin_axis_changed.emit("y")
    assert screen.gates.canvas._spin_axis == "y"


def test_xd_projects_a_table_with_scattered_missing_values(screen):
    """The reason xD looked unimplemented: dropping every row with any NaN
    leaves nothing on a table with hundreds of columns."""
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame({f"m{i}": rng.normal(0, 1, n) for i in range(40)})
    for i in range(40):
        frame.loc[rng.choice(n, 10, replace=False), f"m{i}"] = np.nan
    screen.set_frame(frame)

    assert screen.reduce_to_components() is None, "the projection failed"
    assert screen.gates._frame["PC1"].notna().sum() == n


# --- gating in the volume, and merging awkward keys ----------------------

def test_a_box_gate_is_made_from_the_view(screen):
    """"i cannot draw gates in 3D". A rectangle dragged on a rotated
    projection has no defined extent along the axis pointing at the viewer,
    so the view itself is the gesture: frame a population, keep it."""
    from spacr.qt.widgets.gate_spec import BoxGate

    canvas = _volume(screen)
    screen.gates.set_namer(lambda: "blob")
    screen.gates.gate_from_view()

    gate = screen.gates.gates.get("blob")
    assert isinstance(gate, BoxGate)
    assert gate.columns == ("a", "b", "c"), "the box does not use all three"


def test_a_box_gate_narrows_when_the_view_does(screen):
    canvas = _volume(screen)
    ax = canvas.axes_at(0, 0)
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)

    screen.gates.set_namer(lambda: "tight")
    screen.gates.gate_from_view()
    gate = screen.gates.gates.get("tight")
    selected = int(gate.mask(screen.gates._frame).sum())
    assert 0 < selected < len(screen.gates._frame), (
        f"the box selected {selected} of everything; it did not follow the view")


def test_a_box_gate_is_editable_in_2d_as_its_rectangle(screen):
    """Its outline, handles and drag all work unchanged, and the depth the
    flat view cannot express is left alone rather than reset."""
    canvas = _volume(screen)
    screen.gates.set_namer(lambda: "blob")
    screen.gates.gate_from_view()

    screen.gates.mode_requested.emit("2D")
    gate = screen.gates.gates.get("blob")
    assert canvas._gate_is_on_these_axes(gate)
    assert canvas._handles_for(canvas.axes_at(0, 0), gate), "no handles in 2D"


def test_a_box_gate_survives_a_save_and_load(screen, tmp_path):
    from spacr.qt.widgets.gate_spec import BoxGate, GateSet

    _volume(screen)
    screen.gates.set_namer(lambda: "blob")
    screen.gates.gate_from_view()

    path = screen.save_gates(str(tmp_path / "g.json"))
    loaded = GateSet.load(path).get("blob")
    assert isinstance(loaded, BoxGate)
    assert loaded.z_column == "c"


def test_the_box_is_not_offered_as_a_drag_tool(screen):
    """It would promise a gesture that cannot work on a rotated projection."""
    tools = [screen.gates._tool.itemData(i)
             for i in range(screen.gates._tool.count())]
    assert "box" not in tools


def test_xd_shows_three_axes_not_two(screen):
    """"xD has three axees in the feilds above called PC1, PC2, PC3 but i
    cannot see the axees on the graph the graph is 2D"."""
    _wide(screen)
    screen.gates.projection_requested.emit(True)
    screen.gates.mode_requested.emit("3D")
    ax = screen.gates.canvas.axes_at(0, 0)
    assert hasattr(ax, "zaxis"), "xD drew a flat scatter of two components"


def test_xd_can_be_spun_like_3d(screen):
    _wide(screen)
    screen.gates.projection_requested.emit(True)
    screen.gates.mode_requested.emit("3D")
    canvas = screen.gates.canvas
    canvas.set_tool("")
    ax = canvas.axes_at(0, 0)
    before = float(ax.azim)
    canvas._on_press(_Mouse(ax, 100, 100))
    canvas._on_motion(_Mouse(ax, 160, 100))
    canvas._on_release(_Mouse(ax, 160, 100))
    assert float(ax.azim) != before
