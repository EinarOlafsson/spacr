"""The gates panel lets each gate's thresholds be edited. Point 4 of 52."""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import GateTree
from spacr.qt.widgets.gate_spec import (
    CylinderGate, GateSet, PolygonGate, ThresholdGate,
)


@pytest.fixture
def tree(qtbot):
    widget = GateTree()
    qtbot.addWidget(widget)
    return widget


def _with(tree, gate):
    gates = GateSet()
    gates.add(gate)
    tree.set_gates(gates, pd.DataFrame({"a": [0.0, 1.0, 2.0],
                                        "b": [0.0, 1.0, 2.0],
                                        "z": [0.0, 5.0, 9.0]}))
    tree._rebuild_thresholds(gate.name)
    return gates


def _cylinder(**kw):
    base = dict(name="c", u_column="a", v_column="b", axis_column="z",
                u_radius=1.0, v_radius=1.0)
    base.update(kw)
    return CylinderGate(**base)


def test_a_cylinder_offers_a_row_for_its_normal(tree):
    _with(tree, _cylinder())
    assert set(tree._threshold_rows) == {"z"}
    assert tree._thresholds.isVisibleTo(tree)


def test_an_unbounded_normal_shows_blank_and_not_zero(tree):
    """Blank means unbounded; 0 would mean a gate that selects nothing."""
    _with(tree, _cylinder())
    low, high = tree._threshold_rows["z"]
    assert low.text() == "" and high.text() == ""


def test_typing_a_pair_bounds_the_gate(tree):
    gates = _with(tree, _cylinder())
    low, high = tree._threshold_rows["z"]
    low.setText("1")
    high.setText("6")
    tree._apply_threshold("z")
    assert gates.get("c").thresholds() == {"z": (1.0, 6.0)}


def test_clearing_a_field_puts_the_bound_back_to_unbounded(tree):
    gates = _with(tree, _cylinder(axis_low=1.0, axis_high=6.0))
    low, high = tree._threshold_rows["z"]
    low.setText("")
    high.setText("")
    tree._apply_threshold("z")
    assert gates.get("c").thresholds() == {"z": (None, None)}


def test_a_reversed_pair_is_ordered_rather_than_refused(tree):
    gates = _with(tree, _cylinder())
    low, high = tree._threshold_rows["z"]
    low.setText("9")
    high.setText("2")
    tree._apply_threshold("z")
    assert gates.get("c").thresholds() == {"z": (2.0, 9.0)}


def test_text_that_is_not_a_number_reads_as_unbounded(tree):
    gates = _with(tree, _cylinder(axis_low=1.0, axis_high=6.0))
    low, _high = tree._threshold_rows["z"]
    low.setText("banana")
    tree._apply_threshold("z")
    assert gates.get("c").thresholds()["z"][0] is None


def test_a_gate_with_no_thresholds_shows_no_rows(tree):
    _with(tree, PolygonGate(name="p", x_column="a", y_column="b",
                            vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
    assert tree._threshold_rows == {}
    assert not tree._thresholds.isVisibleTo(tree)


def test_selecting_nothing_clears_the_rows(tree):
    _with(tree, _cylinder())
    tree._rebuild_thresholds("")
    assert tree._threshold_rows == {}


def test_a_threshold_gate_offers_its_one_column(tree):
    _with(tree, ThresholdGate(name="t", column="a", low=0.0, high=1.0))
    assert set(tree._threshold_rows) == {"a"}
    assert tree._threshold_rows["a"][0].text() == "0"


def test_editing_announces_the_change(tree, qtbot):
    gates = _with(tree, _cylinder())
    with qtbot.waitSignal(tree.gates_changed, timeout=500):
        tree._threshold_rows["z"][0].setText("1")
        tree._apply_threshold("z")


def test_editing_a_column_the_gate_cannot_take_is_a_no_op(tree):
    gates = _with(tree, _cylinder())
    before = gates.get("c")
    tree._threshold_rows["z"] = tree._threshold_rows["z"]
    tree._apply_threshold("a")          # not offered
    assert gates.get("c") == before
