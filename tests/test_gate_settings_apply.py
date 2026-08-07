"""The Gate Editor settings have to reach the DRAWING.

Every test here asserts something about the artists on the axes, not about the
settings object. The defect it exists for is exactly a settings window whose
fields were tested and whose drawing was not: "cmap dosnt seem to be allpied
to the data , they are always blue ... none of the other general settings seem
to be applied."
"""
import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_settings import GateEditorSettings


@pytest.fixture
def canvas(qtbot):
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.graph_builder import GraphSpec

    rng = np.random.default_rng(0)
    widget = GateCanvas()
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"a": rng.normal(5, 1, 300),
                                   "b": rng.normal(5, 1, 300)}))
    widget.set_spec(GraphSpec(x="a", y="b"))
    return widget


def _axes(canvas):
    return list(canvas.panel_axes().values())[0]


def test_the_colour_map_reaches_the_points(canvas):
    """They were always one flat blue whatever the map said."""
    canvas.apply_settings(GateEditorSettings().replaced(colour_map="magma"))
    scatters = [c for c in _axes(canvas).collections
                if c.get_cmap() is not None and c.get_array() is not None]
    assert scatters, "nothing on the axes is coloured by a map"
    assert any(s.get_cmap().name == "magma" for s in scatters), (
        "the chosen colour map did not reach the drawing")


def test_an_unknown_colour_map_does_not_take_the_plot_with_it(canvas):
    """matplotlib's registry changes between versions; decoration is never
    load-bearing (INVARIANTS 10)."""
    canvas.apply_settings(GateEditorSettings().replaced(colour_map="nope"))
    assert _axes(canvas).collections, "the plot went away"


def test_points_are_coloured_by_density_when_there_is_no_colour_column(canvas):
    """A cytometry scatter has no colour axis, so a single colour hides the
    overlap -- which on a crowded plot is the whole reading."""
    canvas.apply_settings(GateEditorSettings())
    arrays = [c.get_array() for c in _axes(canvas).collections
              if c.get_array() is not None]
    assert arrays, "no artist carries a per-point value"
    assert len(np.unique(arrays[0])) > 1, "every point got the same value"


def test_point_size_and_opacity_reach_the_points(canvas):
    canvas.apply_settings(GateEditorSettings().replaced(
        point_size=12.0, point_opacity=0.25))
    scatter = _axes(canvas).collections[0]
    assert abs(float(np.max(scatter.get_sizes())) - 144.0) < 1e-6
    assert abs(float(scatter.get_alpha()) - 0.25) < 1e-6


def test_the_grid_setting_reaches_the_axes(canvas):
    canvas.apply_settings(GateEditorSettings().replaced(show_grid=True))
    ax = _axes(canvas)
    assert any(line.get_visible() for line in ax.get_xgridlines())

    canvas.apply_settings(GateEditorSettings().replaced(show_grid=False))
    ax = _axes(canvas)
    assert not any(line.get_visible() for line in ax.get_xgridlines())


def test_log_axes_reach_the_axes(canvas):
    canvas.apply_settings(GateEditorSettings().replaced(log_x=True, log_y=True))
    ax = _axes(canvas)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"


def test_a_log_axis_over_data_that_reaches_zero_is_skipped(qtbot):
    """A log axis over non-positive data draws nothing, which reads as the
    plot having broken rather than the setting being inapplicable."""
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.graph_builder import GraphSpec

    widget = GateCanvas()
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"a": [-5.0, 0.0, 5.0], "b": [1.0, 2.0, 3.0]}))
    widget.set_spec(GraphSpec(x="a", y="b"))
    widget.apply_settings(GateEditorSettings().replaced(log_x=True))
    assert list(widget.panel_axes().values())[0].get_xscale() == "linear"


@pytest.mark.parametrize("mode", ["hexbin", "histogram", "density"])
def test_the_binned_resolution_modes_replace_the_points(canvas, mode):
    """Binning is what makes a very large table draw at all."""
    canvas.apply_settings(GateEditorSettings().replaced(resolution_mode=mode))
    ax = _axes(canvas)
    assert ax.collections or ax.images, f"{mode} drew nothing"


def test_the_gate_line_width_defaults_to_half_a_point():
    assert GateEditorSettings().gate_line_width == 0.5


# ---------------------------------------------------------------------------
# Colour-coded gates
# ---------------------------------------------------------------------------

def _gates(n):
    from spacr.qt.widgets.gate_spec import GateSet, RectGate

    gates = GateSet()
    for i in range(n):
        gates = gates.add(RectGate(name=f"g{i}", x_column="a", y_column="b",
                                   x_low=float(i), x_high=float(i) + 1.0,
                                   y_low=0.0, y_high=10.0))
    return gates


def test_each_gate_gets_its_own_colour(canvas):
    canvas.set_gates(_gates(4))
    colours = {canvas.gate_colour(f"g{i}") for i in range(4)}
    assert len(colours) == 4, "two gates on one plot share a colour"


def test_a_gates_colour_survives_another_being_added(canvas):
    """Colour is the only thing telling two gates apart on the plot, so it
    cannot shuffle when the set changes."""
    canvas.set_gates(_gates(2))
    before = canvas.gate_colour("g0")
    canvas.set_gates(_gates(3))
    assert canvas.gate_colour("g0") == before


def test_an_unknown_gate_falls_back_rather_than_raising(canvas):
    assert canvas.gate_colour("never-drawn")


def test_the_gate_list_shows_each_gates_colour(qtbot):
    """A colour on the plot with nothing to look it up in is not a legend."""
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.graph_builder import GraphSpec

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    panel.set_frame(pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]}))
    panel.canvas.set_spec(GraphSpec(x="a", y="b"))
    panel.set_gates(_gates(3))

    seen = []
    for i in range(panel.tree.tree.topLevelItemCount()):
        item = panel.tree.tree.topLevelItem(i)
        seen.append(item.foreground(0).color().name())
    assert len(set(seen)) == 3, f"the list does not colour its rows: {seen}"
    assert seen[0].lower() == panel.canvas.gate_colour("g0").lower(), (
        "the list and the plot disagree about a gate's colour")


def test_every_column_of_the_gate_list_is_visible_on_startup(qtbot):
    """The counts are the reason the tree has columns at all."""
    from spacr.qt.widgets.gate_editor import GateTree

    tree = GateTree()
    qtbot.addWidget(tree)
    tree.resize(320, 200)
    tree.set_gates(_gates(2), pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]}))
    header = tree.tree.header()
    for column in range(4):
        assert not header.isSectionHidden(column)
        assert header.sectionSize(column) > 0, f"column {column} has no width"


# ---------------------------------------------------------------------------
# The cluster bug
# ---------------------------------------------------------------------------

def test_clustering_sees_the_axes_that_are_chosen(canvas, qtbot, monkeypatch):
    """"when i press cluster i get 'Clustering needs an X and a Y
    measurement.' when both are cohosen" -- it read canvas.x_column, which has
    never existed, so getattr always returned the default."""
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.graph_builder import GraphSpec

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    panel.set_frame(pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]}))
    panel.canvas.set_spec(GraphSpec(x="a", y="b"))

    complained = []
    import PySide6.QtWidgets as W
    monkeypatch.setattr(W.QMessageBox, "information",
                        lambda *a, **k: complained.append(a[-1]))
    # Stop before the dialog: the assertion is about the axis check above it.
    monkeypatch.setattr(
        "spacr.qt.widgets.gate_editor._ClusterSettingsDialog.exec",
        lambda self: 0)
    panel._on_cluster()
    assert not any("needs an X and a Y" in str(m) for m in complained), (
        f"clustering refused axes that are set: {complained}")
