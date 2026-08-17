"""The gate list and the gating surface, from the user's side.

Coverage of the widget half of ``spacr.qt.widgets.gate_editor``:
:class:`GateTree`, the cluster settings modal and :class:`GateEditorPanel`.
The canvas half has its own files.

TWO REAL DEFECTS ARE PINNED HERE and the tests that state them FAIL until
the source is fixed. Both are in ``GateEditorPanel.run_cluster``:

1. ``parent=self.canvas.active_gate()`` calls a **property**. ``GateTree``
   has ``active_gate()`` as a method and the canvas has it as a property, and
   the two got confused -- the same mix-up the file already records for
   ``gates`` ("`gates` is a PROPERTY on both this panel and the canvas.
   Calling it raised TypeError on every drag"). So the Cluster… button has
   never produced a gate: it raises ``TypeError: 'NoneType' object is not
   callable`` the moment it gets past the parameter dialog.

2. The clusters are added to ``self.canvas.gates``, which is a DIFFERENT
   ``GateSet`` from ``self._gates`` until something calls
   ``GateEditorPanel.set_gates`` -- and on a fresh session nothing does; the
   screen only calls it when a saved strategy is loaded. So the populations
   land somewhere the gate list never reads, ``save_gates`` never writes and
   ``gates_changed`` never announces, and the next hand-drawn gate replaces
   the canvas's set and deletes them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt                                    # noqa: E402
from PySide6.QtWidgets import (                                  # noqa: E402
    QDialog, QMessageBox, QTreeWidgetItem,
)

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.qt.widgets import gate_editor as G                    # noqa: E402
from spacr.qt.widgets.gate_editor import (                       # noqa: E402
    GateEditorPanel, GateTree, _ClusterSettingsDialog,
)
from spacr.qt.widgets.gate_settings import GateEditorSettings    # noqa: E402
from spacr.qt.widgets.gate_spec import (                         # noqa: E402
    POLYGON, RECTANGLE, BoxGate, CylinderGate, GateSet, PolygonGate,
    RectGate, ThresholdGate,
)
from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec    # noqa: E402


# ---------------------------------------------------------------------------
# tables and fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid() -> pd.DataFrame:
    """Four objects on a diagonal, so every gate below selects a listable set."""
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                         "b": [1.0, 2.0, 3.0, 4.0],
                         "c": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def keyed() -> pd.DataFrame:
    """Six objects carrying the object keys a shared highlight needs."""
    n = 6
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["A"] * n, "columnID": ["1"] * n,
        "fieldID": ["1"] * n,
        "object_label": [str(i + 1) for i in range(n)],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


@pytest.fixture
def two_blobs() -> pd.DataFrame:
    """Two well-separated populations, plus a measurement that never varies."""
    rng = np.random.default_rng(3)
    a = np.concatenate([rng.normal(0.0, 0.25, 60), rng.normal(8.0, 0.25, 60)])
    b = np.concatenate([rng.normal(0.0, 0.25, 60), rng.normal(8.0, 0.25, 60)])
    return pd.DataFrame({"a": a, "b": b, "flat": np.ones(120)})


@pytest.fixture
def one_blob() -> pd.DataFrame:
    """One population, so no radius can honestly split it in two."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(0.0, 1.0, 100),
                         "b": rng.normal(0.0, 1.0, 100)})


def _pair(name="one", parent=None, x="a", y="b", low=0.0, high=2.5):
    return RectGate(name=name, parent=parent, x_column=x, y_column=y,
                    x_low=low, x_high=high, y_low=low, y_high=high)


@pytest.fixture
def tree(qtbot) -> GateTree:
    widget = GateTree()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def make_panel(qtbot):
    """Build a panel on ``frame``, plotting ``x`` against ``y``."""
    def build(frame, x="a", y="b"):
        panel = GateEditorPanel(link=LinkedSelection())
        qtbot.addWidget(panel)
        panel.set_frame(frame)
        roles = {c: CONTINUOUS for c in frame.columns
                 if pd.api.types.is_numeric_dtype(frame[c])}
        panel.set_spec(GraphSpec(x=x, y=y, roles=roles))
        return panel
    return build


@pytest.fixture
def panel(make_panel, grid):
    return make_panel(grid)


@pytest.fixture
def boxes(monkeypatch):
    """Capture the modal messages. Left real they would hang a headless run."""
    seen = []

    def _grab(_parent, title, text, *_a, **_k):
        seen.append((title, text))

    monkeypatch.setattr(QMessageBox, "information", staticmethod(_grab))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_grab))
    return seen


# ---------------------------------------------------------------------------
# GateTree: the list itself
# ---------------------------------------------------------------------------

def test_gates_that_do_not_fit_this_table_are_still_listed_without_counts(
        tree, grid):
    """A strategy loaded onto the wrong table must still be visible.

    The counts cannot be computed -- the measurement is not there -- but
    dropping the rows would leave the user with an empty panel and no way to
    see, rename or delete the gates they just loaded.
    """
    gates = GateSet()
    gates.add(_pair(name="ghost", x="not_measured"))
    tree.set_gates(gates, grid)

    assert tree.tree.topLevelItemCount() == 1
    item = tree.tree.topLevelItem(0)
    assert item.text(0) == "ghost"
    assert (item.text(1), item.text(2), item.text(3)) == ("", "", ""), \
        "an uncomputable count was printed as a number"


def test_the_gates_own_colour_is_shown_on_its_name(tree, grid):
    """A colour on the plot that is not also in the list has nothing to look
    it up in."""
    gates = GateSet()
    gates.add(_pair())
    tree.set_gates(gates, grid)
    tree.set_colour_source(lambda _name: "#ff0000")

    assert tree.tree.topLevelItem(0).foreground(0).color().name() == "#ff0000"


def test_a_colour_source_that_fails_leaves_the_row_readable(tree, grid):
    """The tree only DISPLAYS the colour; whoever draws the gates owns it.

    A colour that cannot be worked out is not worth losing the gate list
    over, so the row is drawn in the ordinary text colour instead.
    """
    gates = GateSet()
    gates.add(_pair())
    tree.set_gates(gates, grid)

    def broken(_name):
        raise RuntimeError("no palette yet")

    tree.set_colour_source(broken)

    assert tree._colour_for("one") == ""
    assert tree.tree.topLevelItemCount() == 1
    assert tree.tree.topLevelItem(0).text(0) == "one"


def test_unticking_a_gate_in_the_tree_reports_it_as_hidden(tree, grid):
    gates = GateSet()
    gates.add(_pair(name="one"))
    gates.add(_pair(name="two", high=3.5))
    tree.set_gates(gates, grid)
    seen = []
    tree.enabled_changed.connect(lambda name, on: seen.append((name, on)))

    tree.tree.topLevelItem(0).setCheckState(0, Qt.Unchecked)

    assert seen == [("one", False)]
    assert tree.is_enabled("one") is False
    assert tree.is_enabled("two") is True
    assert tree.is_enabled("never_drawn") is True, "an unknown gate is on"


def test_reticking_a_gate_puts_it_back_on(tree, grid):
    gates = GateSet()
    gates.add(_pair())
    tree.set_gates(gates, grid)
    item = tree.tree.topLevelItem(0)
    item.setCheckState(0, Qt.Unchecked)
    seen = []
    tree.enabled_changed.connect(lambda name, on: seen.append((name, on)))

    item.setCheckState(0, Qt.Checked)

    assert seen == [("one", True)]
    assert tree.is_enabled("one") is True


def test_rebuilding_the_list_does_not_report_every_gate_as_freshly_ticked(
        tree, grid):
    """Recounting the gates is not the user touching the ticks.

    Announcing one would tell the canvas to hide and show gates nobody
    touched, and would do it on every edit, since every edit recounts.
    """
    gates = GateSet()
    gates.add(_pair())
    tree.set_gates(gates, grid)
    seen = []
    tree.enabled_changed.connect(lambda name, on: seen.append((name, on)))

    tree.refresh()
    assert seen == []

    # ...and a tick that moves WHILE the rebuild is running is the rebuild's
    # doing, not the user's, so it is neither announced nor recorded.
    item = tree.tree.topLevelItem(0)
    tree._rebuilding = True
    try:
        item.setCheckState(0, Qt.Unchecked)
    finally:
        tree._rebuilding = False

    assert seen == []
    assert tree.is_enabled("one") is True, "a rebuild switched a gate off"


def test_a_row_that_names_no_gate_toggles_nothing(tree, grid):
    """Only rows that stand for a gate can hide one."""
    gates = GateSet()
    gates.add(_pair())
    tree.set_gates(gates, grid)
    seen = []
    tree.enabled_changed.connect(lambda name, on: seen.append((name, on)))

    spacer = QTreeWidgetItem(["", "", "", ""])
    tree.tree.addTopLevelItem(spacer)
    spacer.setCheckState(0, Qt.Unchecked)

    assert seen == []
    assert tree.is_enabled("one") is True


def test_delete_gate_with_nothing_selected_deletes_nothing(tree, grid):
    """The button is always there; pressing it before choosing a gate must
    not remove one at random, nor tell everyone the strategy changed."""
    gates = GateSet()
    gates.add(_pair())
    tree.set_gates(gates, grid)
    tree.tree.setCurrentItem(None)
    announced = []
    tree.gates_changed.connect(lambda: announced.append(1))

    tree.remove_selected()

    assert gates.names == ("one",)
    assert announced == []


# ---------------------------------------------------------------------------
# GateTree: the per-gate thresholds
# ---------------------------------------------------------------------------

class _UnreadableGate(PolygonGate):
    """A gate kind whose bounds cannot be worked out."""

    def thresholds(self):
        raise RuntimeError("this shape cannot say what it is bounded by")


def test_a_gate_whose_bounds_cannot_be_read_shows_no_threshold_rows(
        tree, grid):
    """Selecting it must leave the rest of the panel usable.

    The rows are the only part that depends on the answer, so the gate is
    listed and selectable and simply offers nothing to type into.
    """
    gates = GateSet()
    gates.add(_UnreadableGate(name="odd", x_column="a", y_column="b",
                              vertices=((0.0, 0.0), (4.0, 0.0), (0.0, 4.0))))
    tree.set_gates(gates, grid)

    tree._rebuild_thresholds("odd")

    assert tree._threshold_rows == {}
    assert not tree._thresholds.isVisibleTo(tree)


def test_typing_a_threshold_for_a_gate_that_has_gone_is_a_no_op(tree, grid):
    """The rows outlive the gate if it is deleted while one is being filled
    in, and the number must not be written onto whatever is selected now."""
    gates = GateSet()
    gates.add(ThresholdGate(name="cut", column="a", low=1.0, high=3.0))
    tree.set_gates(gates, grid)
    tree._rebuild_thresholds("cut")
    announced = []
    tree.gates_changed.connect(lambda: announced.append(1))

    gates.remove("cut")
    tree._threshold_rows["a"][0].setText("2")
    tree._apply_threshold("a")

    assert gates.names == ()
    assert announced == []


def test_a_threshold_the_gate_can_no_longer_take_is_refused_not_written(
        tree, grid):
    """The gate was replaced by one bounded on other measurements.

    Writing the number anyway would leave the panel showing a bound the gate
    does not honour -- which is exactly what `with_threshold` refuses to do.
    """
    gates = GateSet()
    gates.add(CylinderGate(name="c", u_column="a", v_column="b",
                           axis_column="c", u_radius=1.0, v_radius=1.0))
    tree.set_gates(gates, grid)
    tree._rebuild_thresholds("c")
    assert set(tree._threshold_rows) == {"c"}
    announced = []
    tree.gates_changed.connect(lambda: announced.append(1))

    # Re-drawn as a cut on `a` -- same name, no bound on `c` at all.
    gates.add(ThresholdGate(name="c", column="a", low=0.0, high=1.0))
    tree._threshold_rows["c"][0].setText("2")
    tree._apply_threshold("c")

    assert gates.get("c") == ThresholdGate(name="c", column="a",
                                           low=0.0, high=1.0)
    assert announced == [], "a refused edit was announced as a change"


# ---------------------------------------------------------------------------
# The cluster settings modal
# ---------------------------------------------------------------------------

def test_the_cluster_dialog_opens_even_where_the_window_hint_fails(
        qtbot, monkeypatch):
    """Asking the window manager to let the modal be dragged is a nicety.

    A platform that refuses must not be the reason clustering cannot be set
    up at all, so the request is made and the answer is not waited on.
    """
    import spacr.qt.dialogs as dialogs

    def refuse(_dialog):
        raise RuntimeError("no window manager here")

    monkeypatch.setattr(dialogs, "detach_from_window_manager", refuse)

    dialog = _ClusterSettingsDialog(settings=GateEditorSettings(cluster_eps=1.5))
    qtbot.addWidget(dialog)

    assert dialog.eps() == pytest.approx(1.5)


def test_the_cluster_dialog_carries_the_algorithm_chosen_in_settings(qtbot):
    """The algorithm is a Gate Settings decision and this modal is the
    per-run tuning of it, so it is carried rather than re-asked -- and
    carrying it is the whole point: the picker existed while `cluster_gates`
    ran DBSCAN regardless."""
    dialog = _ClusterSettingsDialog(
        settings=GateEditorSettings(cluster_method="hdbscan"))
    qtbot.addWidget(dialog)

    assert dialog.method() == "hdbscan"


# ---------------------------------------------------------------------------
# GateEditorPanel: the tool picker
# ---------------------------------------------------------------------------

def test_choosing_a_tool_in_the_dropdown_changes_what_a_drag_draws(panel):
    """The picker is the only way to change the tool, so a picker that does
    not reach the canvas is a picker that does nothing."""
    panel._tool.setCurrentIndex(panel._tool.findData(POLYGON))
    assert panel.canvas.gate_from_drag(1.0, 1.0, 3.0, 3.0) is None, \
        "a drag still drew a shape after Polygon was chosen"

    panel._tool.setCurrentIndex(panel._tool.findData(RECTANGLE))
    drawn = panel.canvas.gate_from_drag(1.0, 1.0, 3.0, 3.0)
    assert isinstance(drawn, RectGate)
    assert (drawn.x_low, drawn.x_high) == (1.0, 3.0)


# ---------------------------------------------------------------------------
# GateEditorPanel: clustering
# ---------------------------------------------------------------------------

def test_clustering_with_no_table_loaded_says_so_instead_of_running(
        make_panel, boxes):
    panel = make_panel(pd.DataFrame({"a": [], "b": []}))

    panel.run_cluster(ask=False)

    assert [title for title, _ in boxes] == ["Nothing to cluster"]
    assert panel.gates.is_empty


def test_clustering_needs_both_axes_before_it_will_ask_for_anything(
        make_panel, grid, boxes):
    panel = make_panel(grid, x="a", y=None)

    panel.run_cluster(ask=False)

    assert [title for title, _ in boxes] == ["Pick two measurements"]
    assert panel.gates.is_empty


def test_a_walk_that_finds_nothing_says_so_and_does_not_cluster_anyway(
        make_panel, one_blob, boxes):
    """A walk that found nothing defensible is a result about the DATA.

    Clustering at the radius that was typed in would present it as if the
    search had endorsed it, and the user would read the populations as the
    search's answer when the search declined to give one.
    """
    panel = make_panel(one_blob)
    panel.apply_settings(GateEditorSettings(
        cluster_eps=0.5, cluster_min_samples=40,
        cluster_walk=True, cluster_walk_steps=4))

    panel.run_cluster(ask=False)

    assert len(boxes) == 1
    title, text = boxes[0]
    assert title == "The walk found nothing to recommend"
    assert "Tried: " in text, "the radii it tried were not reported"
    assert panel.gates.is_empty
    assert panel.canvas.gates.is_empty, "it clustered anyway"


def test_a_measurement_that_never_varies_is_refused_with_a_reason(
        make_panel, two_blobs, boxes):
    """A silent empty result reads as a broken button."""
    panel = make_panel(two_blobs, x="a", y="flat")
    panel.apply_settings(GateEditorSettings(cluster_eps=0.5,
                                            cluster_min_samples=5))

    panel.run_cluster(ask=False)

    assert len(boxes) == 1
    title, text = boxes[0]
    assert title == "Could not cluster"
    assert "flat" in text
    assert panel.gates.is_empty


def test_settings_too_strict_to_find_a_population_say_which_to_relax(
        make_panel, two_blobs, boxes):
    panel = make_panel(two_blobs)
    panel.apply_settings(GateEditorSettings(cluster_eps=0.01,
                                            cluster_min_samples=50))

    panel.run_cluster(ask=False)

    assert len(boxes) == 1
    title, text = boxes[0]
    assert title == "No clusters"
    assert "eps" in text and "min_samples" in text
    assert panel.gates.is_empty


def test_clustering_puts_the_populations_it_found_in_the_gate_list(
        make_panel, two_blobs, boxes):
    """FAILS TODAY -- see the module docstring, defects 1 and 2.

    A cluster is meant to be a REAL gate: editable, nestable, saveable and
    usable as a filter the moment it appears. All of that is reached through
    the gate list and through `GateEditorPanel.gates`, which is what the
    screen saves -- so a population that is only on the canvas is a
    population the user cannot do any of those things with, and the next
    hand-drawn gate deletes it.
    """
    panel = make_panel(two_blobs)
    panel.apply_settings(GateEditorSettings(cluster_eps=0.5,
                                            cluster_min_samples=5))
    announced = []
    panel.gates_changed.connect(lambda: announced.append(1))

    panel.run_cluster(ask=False)

    assert boxes == []
    assert panel.gates.names == ("cluster 1", "cluster 2")
    assert panel.tree.tree.topLevelItemCount() == 2, "the gate list is empty"
    assert panel.canvas.gates is panel.gates
    assert panel.canvas.active_gate == "cluster 1"
    assert announced, "the strategy changed and nobody was told"
    assert "2 of 2 gate(s) shown" in panel.status()


def test_the_clusters_survive_the_next_hand_drawn_gate(
        make_panel, two_blobs, boxes):
    """FAILS TODAY -- see the module docstring, defect 2.

    Drawing a gate hands `self._gates` to the canvas, so anything the canvas
    was holding on its own is dropped. Two populations found by DBSCAN
    disappearing because the user then drew a rectangle is the user losing
    work without being told.
    """
    panel = make_panel(two_blobs)
    panel.set_namer(lambda: "by hand")
    panel.apply_settings(GateEditorSettings(cluster_eps=0.5,
                                            cluster_min_samples=5))
    panel.run_cluster(ask=False)

    panel.canvas.gate_drawn.emit(
        panel.canvas.gate_from_drag(-1.0, -1.0, 1.0, 1.0))

    assert set(panel.gates.names) == {"cluster 1", "cluster 2", "by hand"}


def test_the_button_clusters_at_the_numbers_the_modal_was_left_on(
        make_panel, two_blobs, boxes, monkeypatch):
    """FAILS TODAY -- see the module docstring, defect 1.

    The Cluster… button asks first and the Search tab does not, but both must
    end up running the same five numbers. Pressing OK is what makes the
    modal's numbers the run's numbers.
    """
    panel = make_panel(two_blobs)
    panel.apply_settings(GateEditorSettings(cluster_eps=0.5,
                                            cluster_min_samples=5))
    used = {}

    class _Ok(_ClusterSettingsDialog):
        def exec(self):
            used["eps"] = self.eps()
            used["min_samples"] = self.min_samples()
            return QDialog.Accepted

    monkeypatch.setattr(G, "_ClusterSettingsDialog", _Ok)

    panel._on_cluster()

    assert used == {"eps": pytest.approx(0.5), "min_samples": 5}
    assert panel.gates.names == ("cluster 1", "cluster 2")


def test_cancelling_the_modal_clusters_nothing(make_panel, two_blobs, boxes,
                                               monkeypatch):
    panel = make_panel(two_blobs)
    panel.apply_settings(GateEditorSettings(cluster_eps=0.5,
                                            cluster_min_samples=5))

    class _Cancelled(_ClusterSettingsDialog):
        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(G, "_ClusterSettingsDialog", _Cancelled)

    panel._on_cluster()

    assert panel.gates.is_empty
    assert panel.canvas.gates.is_empty


def test_a_walk_reports_the_radius_it_settled_on(make_panel, two_blobs,
                                                 boxes):
    """FAILS TODAY -- see the module docstring, defect 1.

    A search that silently substitutes a parameter is worse than one that
    never ran: the number has to come back in the units the user typed in, so
    it can be carried to Gate Settings by hand.
    """
    panel = make_panel(two_blobs)
    panel.apply_settings(GateEditorSettings(
        cluster_eps=0.5, cluster_min_samples=5,
        cluster_walk=True, cluster_walk_steps=5))

    panel.run_cluster(ask=False)

    assert panel.gates.names == ("cluster 1", "cluster 2")
    assert len(boxes) == 1
    title, text = boxes[0]
    assert title == "Walk finished"
    assert "populations" in text and "eps" in text


# ---------------------------------------------------------------------------
# GateEditorPanel: drawing, selecting and publishing
# ---------------------------------------------------------------------------

def test_declining_to_name_a_gate_throws_the_drawing_away(panel, monkeypatch):
    """A gate is not a gate until it is named.

    The refused shape has to come off the plot too -- left there it looks
    like a gate that exists and cannot be selected.
    """
    panel.set_namer(lambda: "")
    repainted = []
    monkeypatch.setattr(panel.canvas, "render_now",
                        lambda: repainted.append(1))

    panel.canvas.gate_drawn.emit(
        panel.canvas.gate_from_drag(1.0, 1.0, 3.0, 3.0))

    assert panel.gates.is_empty
    assert repainted, "the abandoned outline was left on the plot"


def test_a_gate_drawn_inside_a_gate_that_is_gone_is_refused_with_a_reason(
        panel):
    """A child whose parent no longer exists is a gate on a population that
    does not exist, and the panel says which parent is missing rather than
    adding a gate that means nothing."""
    panel.set_namer(lambda: "orphan")

    panel.canvas.gate_drawn.emit(_pair(name="(unnamed)", parent="deleted"))

    assert panel.gates.is_empty
    assert "deleted" in panel.status()


def test_selecting_a_gate_asks_the_screen_for_its_measurements(panel, grid):
    """A gate is drawn on two named measurements, so selecting one whose axes
    are off screen used to select something invisible."""
    gates = GateSet()
    gates.add(_pair(name="one", x="a", y="b"))
    panel.set_gates(gates)
    asked = []
    panel.axes_requested.connect(lambda x, y: asked.append((x, y)))

    panel.tree.select("one")

    assert asked == [("a", "b")]


def test_clicking_a_gate_that_has_been_deleted_does_not_move_the_axes(panel):
    """The row can outlive the gate -- the strategy is shared and can be
    edited from elsewhere. Clicking a stale row must leave the plot alone
    rather than take the screen down."""
    gates = GateSet()
    gates.add(_pair(name="one", x="a", y="b"))
    gates.add(_pair(name="two", x="b", y="c"))
    panel.set_gates(gates)
    panel.tree.select("one")
    asked = []
    panel.axes_requested.connect(lambda x, y: asked.append((x, y)))

    gates.remove("two")            # gone from the strategy, still in the list
    panel.tree.select("two")

    assert asked == [], "the axes moved to a gate that no longer exists"


def test_publishing_with_nothing_selected_says_which_step_is_missing(panel):
    gates = GateSet()
    gates.add(_pair())
    panel.set_gates(gates)
    panel.tree.select("")

    assert panel.publish() is None
    assert "Select a gate" in panel.status()


def test_publishing_without_a_table_says_so(panel):
    """The table can go away underneath a selection -- the screen reloads it
    on a worker. Saying so beats a traceback from an empty highlight."""
    gates = GateSet()
    gates.add(_pair())
    panel.set_gates(gates)
    panel.tree.select("one")
    panel._frame = None

    assert panel.publish() is None
    assert panel.status() == "Load a table first."


def test_publishing_a_gate_this_table_cannot_take_names_the_measurement(
        panel):
    """Re-applying a gate to a table without the measurement is a mistake
    worth a sentence, not a silently empty population."""
    gates = GateSet()
    gates.add(_pair(name="ghost", x="not_measured"))
    panel.set_gates(gates)
    panel.tree.select("ghost")

    assert panel.publish() is None
    assert "not_measured" in panel.status()


def test_publishing_a_gate_highlights_its_objects_for_every_other_view(
        make_panel, keyed):
    """Applying a gate rings the objects inside it and leaves the rest of the
    graph on screen -- it is a SELECTION, never a filter."""
    panel = make_panel(keyed)
    gates = GateSet()
    gates.add(_pair(name="mid", low=1.5, high=4.5))
    panel.set_gates(gates)
    panel.tree.select("mid")
    link = panel.canvas.link

    assert panel.publish() is None
    assert panel.status() == "3 object(s) highlighted by mid"
    assert list(link.selection.keys) == ["p1_A_1_1_2", "p1_A_1_1_3",
                                         "p1_A_1_1_4"]
    assert link.filter is None or link.filter.is_empty, "it filtered instead"


# ---------------------------------------------------------------------------
# GateEditorPanel: the 3D controls
# ---------------------------------------------------------------------------

def test_the_panel_reports_the_plane_a_shape_would_land_on(panel):
    """The plane is picked, not read off the camera, so the panel can always
    say which one is armed."""
    assert panel.anchor_axis() == "z"

    panel._plane_buttons["x"].click()

    assert panel.anchor_axis() == "x"
    assert panel.canvas.anchor_axis() == "x"


def test_with_no_plane_ticked_the_editor_still_names_one(panel):
    """The three plane buttons are an exclusive group, so one is always
    ticked while the group holds. The fallback is what keeps `anchor_axis`
    answerable for a caller that reaches it when it does not."""
    group = panel._plane_buttons["z"].group()
    group.setExclusive(False)
    for button in panel._plane_buttons.values():
        button.setChecked(False)

    assert panel.anchor_axis() == "z"


def test_with_no_drag_mode_ticked_the_editor_still_names_one(panel):
    """As above: a drag has to mean something, and spinning is the safe one
    because it cannot create a gate nobody asked for."""
    group = panel._drag_buttons["spin"].group()
    group.setExclusive(False)
    for button in panel._drag_buttons.values():
        button.setChecked(False)

    assert panel.drag_mode() == "spin"


def test_from_view_needs_a_third_measurement_and_says_which(panel):
    """On a flat plot there is no depth to read, and inventing one would make
    a box that means something the user never framed."""
    panel.set_namer(lambda: "framed")

    panel.gate_from_view()

    assert panel.gates.is_empty
    assert "choose a Z" in panel.status()


def test_from_view_turns_the_framed_volume_into_a_gate(make_panel, grid):
    """The view IS the gesture: spin and zoom until a population fills the
    box, then keep what was framed."""
    panel = make_panel(grid)
    panel.canvas._mode = "3D"
    panel.canvas._z_column = "c"
    panel.canvas.render_now()
    panel.set_namer(lambda: "framed")

    panel.gate_from_view()

    gate = panel.gates.get("framed")
    assert isinstance(gate, BoxGate)
    assert gate.columns == ("a", "b", "c"), "the box did not use all three"


def test_reset_view_undoes_a_zoom_and_a_spin_together(panel):
    """From the user's side there is one problem -- "the graph is not where
    it was" -- and having to know whether they zoomed or rotated to get out
    of it is a distinction only the implementation cares about."""
    panel.canvas._zoom = ((0.0, 1.0), (0.0, 1.0))
    panel.canvas._volume_zoom = 4.0
    panel.canvas._view_angles = (10.0, 20.0)

    panel.reset_view()

    assert panel.canvas._zoom is None
    assert panel.canvas._volume_zoom == 1.0
    assert panel.canvas._view_angles is None
