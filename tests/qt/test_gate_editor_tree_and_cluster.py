"""The gate list, the cluster modal and the panel's Cluster button.

:class:`GateTree` lists every gate -- including ones this table cannot count --
keeps its ticks honest across rebuilds and refuses edits to rows that name no
gate.  ``GateEditorPanel.run_cluster`` turns DBSCAN/HDBSCAN populations into
real gates in the panel's own gate set, or says by name why it found none.
``_distinct_from`` decides which outline anchors are new.  One general test per
unit, each walking a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QDialog, QMessageBox, QTreeWidgetItem  # noqa: E402

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets import gate_editor as G  # noqa: E402
from spacr.qt.widgets.gate_editor import (  # noqa: E402
    GateEditorPanel, GateTree, _ClusterSettingsDialog, _distinct_from,
)
from spacr.qt.widgets.gate_settings import GateEditorSettings  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    GateSet, PolygonGate, RectGate, ThresholdGate,
)
from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec  # noqa: E402

pytestmark = pytest.mark.qt

GRID = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0],
                     "c": [1.0, 2.0, 3.0, 4.0]})


def _two_blobs():
    """Two well-separated populations, plus a measurement that never varies."""
    rng = np.random.default_rng(3)
    a = np.concatenate([rng.normal(0.0, 0.25, 60), rng.normal(8.0, 0.25, 60)])
    b = np.concatenate([rng.normal(0.0, 0.25, 60), rng.normal(8.0, 0.25, 60)])
    return pd.DataFrame({"a": a, "b": b, "flat": np.ones(120)})


def _one_blob():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(0.0, 1.0, 100),
                         "b": rng.normal(0.0, 1.0, 100)})


def _pair(name="one", x="a"):
    return RectGate(name=name, x_column=x, y_column="b",
                    x_low=0.0, x_high=2.5, y_low=0.0, y_high=2.5)


class _UnreadableGate(PolygonGate):
    """A gate kind whose bounds cannot be worked out."""

    def thresholds(self):
        raise RuntimeError("this shape cannot say what it is bounded by")


@pytest.fixture
def make_tree(qtbot):
    """A GateTree listing ``gates`` over ``frame``, plus its enabled_changed log."""
    def build(*gates, frame=GRID):
        tree = GateTree()
        qtbot.addWidget(tree)
        gate_set = GateSet()
        for gate in gates:
            gate_set.add(gate)
        tree.set_gates(gate_set, frame)
        seen: list = []
        tree.enabled_changed.connect(lambda name, on: seen.append((name, on)))
        announced: list = []
        tree.gates_changed.connect(lambda: announced.append(1))
        return tree, gate_set, seen, announced
    return build


@pytest.fixture
def make_panel(qtbot, monkeypatch):
    """A GateEditorPanel plotting ``x`` against ``y``, with modal boxes captured."""
    boxes: list = []

    def _grab(_parent, title, text, *_a, **_k):
        boxes.append((title, text))

    monkeypatch.setattr(QMessageBox, "information", staticmethod(_grab))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_grab))

    def build(frame, x="a", y="b", **settings):
        boxes.clear()
        panel = GateEditorPanel(link=LinkedSelection())
        qtbot.addWidget(panel)
        panel.set_frame(frame)
        roles = {c: CONTINUOUS for c in frame.columns
                 if pd.api.types.is_numeric_dtype(frame[c])}
        panel.set_spec(GraphSpec(x=x, y=y, roles=roles))
        panel.apply_settings(GateEditorSettings(**settings))
        return panel, boxes
    return build


def test_the_gate_tree_lists_every_gate_and_keeps_ticks_honest(make_tree, subtests):
    """Rows, counts, colours and ticks of the gate list; no-ops without a gate."""
    with subtests.test(msg="a_gate_this_table_cannot_count_is_listed_unavailable"):
        tree, *_ = make_tree(_pair(name="ghost", x="not_measured"))
        assert tree.tree.topLevelItemCount() == 1
        item = tree.tree.topLevelItem(0)
        assert [item.text(i) for i in range(4)] == ["ghost", GateTree.UNAVAILABLE, "", ""]
        assert "not_measured" in item.toolTip(0)
    with subtests.test(msg="a_missing_column_costs_one_count_not_the_others"):
        tree, *_ = make_tree(ThresholdGate(name="big", column="area", low=1.0),
                             ThresholdGate(name="bright", column="intensity", low=5.0),
                             frame=pd.DataFrame({"area": [0.5, 2.0, 3.0]}))
        rows = {tree.tree.topLevelItem(i).text(0): tree.tree.topLevelItem(i).text(1)
                for i in range(tree.tree.topLevelItemCount())}
        assert rows == {"big": "2", "bright": tree.UNAVAILABLE}
    with subtests.test(msg="the_gates_own_colour_is_on_its_name"):
        tree, *_ = make_tree(_pair())
        tree.set_colour_source(lambda _name: "#ff0000")
        assert tree.tree.topLevelItem(0).foreground(0).color().name() == "#ff0000"
    with subtests.test(msg="a_failing_colour_source_leaves_the_row_readable"):
        tree, *_ = make_tree(_pair())

        def broken(_name):
            raise RuntimeError("no palette yet")

        tree.set_colour_source(broken)
        assert tree._colour_for("one") == ""
        assert [tree.tree.topLevelItem(i).text(0)
                for i in range(tree.tree.topLevelItemCount())] == ["one"]
    with subtests.test(msg="reticking_a_gate_puts_it_back_on"):
        tree, _gates, seen, _ = make_tree(_pair())
        item = tree.tree.topLevelItem(0)
        item.setCheckState(0, Qt.Unchecked)
        seen.clear()
        item.setCheckState(0, Qt.Checked)
        assert seen == [("one", True)]
        assert tree.is_enabled("one") is True
    with subtests.test(msg="a_rebuild_is_not_the_user_ticking"):
        tree, _gates, seen, _ = make_tree(_pair())
        tree.refresh()
        tree._rebuilding = True
        try:
            tree.tree.topLevelItem(0).setCheckState(0, Qt.Unchecked)
        finally:
            tree._rebuilding = False
        assert seen == []
        assert tree.is_enabled("one") is True
    with subtests.test(msg="a_row_that_names_no_gate_toggles_nothing"):
        tree, _gates, seen, _ = make_tree(_pair())
        spacer = QTreeWidgetItem(["", "", "", ""])
        tree.tree.addTopLevelItem(spacer)
        spacer.setCheckState(0, Qt.Unchecked)
        assert seen == []
        assert tree.is_enabled("one") is True
    with subtests.test(msg="delete_with_nothing_selected"):
        tree, gates, _seen, announced = make_tree(_pair())
        tree.tree.setCurrentItem(None)
        tree.remove_selected()
        assert (gates.names, announced) == (("one",), [])
    with subtests.test(msg="unreadable_bounds_show_no_threshold_rows"):
        tree, *_ = make_tree(_UnreadableGate(
            name="odd", x_column="a", y_column="b",
            vertices=((0.0, 0.0), (4.0, 0.0), (0.0, 4.0))))
        tree._rebuild_thresholds("odd")
        assert tree._threshold_rows == {}
        assert tree._thresholds.isVisibleTo(tree) is False
    with subtests.test(msg="typing_a_threshold_for_a_deleted_gate"):
        tree, gates, _seen, announced = make_tree(
            ThresholdGate(name="cut", column="a", low=1.0, high=3.0))
        tree._rebuild_thresholds("cut")
        gates.remove("cut")
        tree._threshold_rows["a"][0].setText("2")
        tree._apply_threshold("a")
        assert (gates.names, announced) == ((), [])


def test_the_cluster_dialog_carries_the_settings_it_was_opened_with(qtbot, monkeypatch, subtests):
    """The modal shows the Gate Settings radius and algorithm, window hint or not."""
    import spacr.qt.dialogs as dialogs

    with subtests.test(msg="a_refused_window_hint_still_opens_it"):
        def refuse(_dialog):
            raise RuntimeError("no window manager here")

        with monkeypatch.context() as mp:
            mp.setattr(dialogs, "detach_from_window_manager", refuse)
            dialog = _ClusterSettingsDialog(settings=GateEditorSettings(cluster_eps=1.5))
            qtbot.addWidget(dialog)
        assert dialog.eps() == pytest.approx(1.5)
    with subtests.test(msg="the_algorithm_chosen_in_settings"):
        dialog = _ClusterSettingsDialog(settings=GateEditorSettings(cluster_method="hdbscan"))
        qtbot.addWidget(dialog)
        assert dialog.method() == "hdbscan"


def test_clustering_makes_real_gates_or_says_why_not(make_panel, monkeypatch, subtests):
    """``run_cluster`` / ``_on_cluster``: populations become listed gates, or a named refusal."""
    refusals = [
        # case_id, frame, y, settings, title, fragments
        ("a_walk_that_finds_nothing", _one_blob(), "b",
         dict(cluster_eps=0.5, cluster_min_samples=40, cluster_walk=True,
              cluster_walk_steps=4),
         "The walk found nothing to recommend", ["Tried: "]),
        ("a_measurement_that_never_varies", _two_blobs(), "flat",
         dict(cluster_eps=0.5, cluster_min_samples=5),
         "Could not cluster", ["flat"]),
        ("settings_too_strict", _two_blobs(), "b",
         dict(cluster_eps=0.01, cluster_min_samples=50),
         "No clusters", ["eps", "min_samples"]),
    ]
    for case_id, frame, y, settings, title, fragments in refusals:
        with subtests.test(msg=case_id):
            panel, boxes = make_panel(frame, y=y, **settings)
            panel.run_cluster(ask=False)
            assert [t for t, _ in boxes] == [title]
            assert all(f in boxes[0][1] for f in fragments)
            assert panel.gates.is_empty and panel.canvas.gates.is_empty
    with subtests.test(msg="populations_land_in_the_gate_list"):
        panel, boxes = make_panel(_two_blobs(), cluster_eps=0.5, cluster_min_samples=5)
        announced: list = []
        panel.gates_changed.connect(lambda: announced.append(1))
        panel.run_cluster(ask=False)
        assert boxes == []
        assert panel.gates.names == ("cluster 1", "cluster 2")
        assert panel.tree.tree.topLevelItemCount() == 2
        assert panel.canvas.gates is panel.gates
        assert panel.canvas.active_gate == "cluster 1"
        assert announced != []
        assert "2 of 2 gate(s) shown" in panel.status()
    with subtests.test(msg="clusters_survive_the_next_hand_drawn_gate"):
        panel, _boxes = make_panel(_two_blobs(), cluster_eps=0.5, cluster_min_samples=5)
        panel.set_namer(lambda: "by hand")
        panel.run_cluster(ask=False)
        panel.canvas.gate_drawn.emit(panel.canvas.gate_from_drag(-1.0, -1.0, 1.0, 1.0))
        assert set(panel.gates.names) == {"cluster 1", "cluster 2", "by hand"}
    with subtests.test(msg="a_walk_reports_the_radius_it_settled_on"):
        panel, boxes = make_panel(_two_blobs(), cluster_eps=0.5, cluster_min_samples=5,
                                  cluster_walk=True, cluster_walk_steps=5)
        panel.run_cluster(ask=False)
        assert panel.gates.names == ("cluster 1", "cluster 2")
        assert [t for t, _ in boxes] == ["Walk finished"]
        assert "populations" in boxes[0][1] and "eps" in boxes[0][1]
    with subtests.test(msg="the_button_runs_the_numbers_the_modal_was_left_on"):
        used = {}

        class _Ok(_ClusterSettingsDialog):
            def exec(self):
                used["eps"] = self.eps()
                used["min_samples"] = self.min_samples()
                return QDialog.Accepted

        panel, _boxes = make_panel(_two_blobs(), cluster_eps=0.5, cluster_min_samples=5)
        with monkeypatch.context() as mp:
            mp.setattr(G, "_ClusterSettingsDialog", _Ok)
            panel._on_cluster()
        assert used == {"eps": pytest.approx(0.5), "min_samples": 5}
        assert panel.gates.names == ("cluster 1", "cluster 2")
    with subtests.test(msg="cancelling_the_modal_clusters_nothing"):
        class _Cancelled(_ClusterSettingsDialog):
            def exec(self):
                return QDialog.Rejected

        panel, _boxes = make_panel(_two_blobs(), cluster_eps=0.5, cluster_min_samples=5)
        with monkeypatch.context() as mp:
            mp.setattr(G, "_ClusterSettingsDialog", _Cancelled)
            panel._on_cluster()
        assert panel.gates.is_empty and panel.canvas.gates.is_empty


def test_distinct_from_keeps_each_new_point_once(subtests):
    """``_distinct_from`` is True for candidates not existing and not repeated."""
    cases = [
        ("nothing_exists_yet", [[0, 0], [1, 0], [0, 1]], np.empty((0, 2)), {},
         [True, True, True]),
        ("a_candidate_on_an_existing_point", [[0, 0], [5, 5]], [[0, 0]], {},
         [False, True]),
        ("a_repeat_is_kept_once_across_blocks", [[1, 1], [2, 2], [1, 1], [3, 3], [2, 2]],
         np.empty((0, 2)), dict(block=2), [True, True, False, True, False]),
        ("repeats_of_an_existing_point_all_refused", [[0, 0], [7, 7], [7, 7]], [[7, 7]],
         dict(block=2), [True, False, False]),
    ]
    for case_id, points, existing, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert _distinct_from(points, existing, **kwargs).tolist() == expected
