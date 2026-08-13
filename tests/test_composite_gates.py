"""Gates combined with each other. Instruction 52, point 5.

    "if the user draws another gate on the same 3d graph they should be able
     to set the new gate as being its own gate, subtracting or add[ing]
     from/to the other gates in view"

"The bright, small, round ones" is three measurements at once, and answering
it today means gating twice and intersecting by hand -- an intersection that
exists only in whatever the user did next. A gate that IS "this cylinder
minus that box" is a statement someone else can re-run.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    COMPOSITE_OPS,
    CompositeGate,
    GateError,
    GateSet,
    ThresholdGate,
    gate_from_dict,
)


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def gates():
    gate_set = GateSet()
    gate_set.add(ThresholdGate(name="lo", column="a", low=0.0, high=2.0))
    gate_set.add(ThresholdGate(name="hi", column="a", low=3.0, high=4.0))
    gate_set.add(ThresholdGate(name="mid", column="a", low=1.0, high=3.0))
    return gate_set


# ---------------------------------------------------------------------------
# The three operations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation,expected", [
    ("union", [1, 1, 1, 1, 1]),
    ("intersect", [0, 0, 0, 0, 0]),
    ("subtract", [1, 1, 1, 0, 0]),
])
def test_each_operation_selects_what_it_says(gates, frame, operation, expected):
    gates.add(CompositeGate(name="c", operation=operation,
                            operands=("lo", "hi")))
    assert gates.mask(frame, "c").astype(int).tolist() == expected


def test_subtract_keeps_the_order_it_was_given(gates, frame):
    """A minus B is not B minus A, and a set that sorted its operands would
    silently change which."""
    gates.add(CompositeGate(name="forward", operation="subtract",
                            operands=("mid", "lo")))
    gates.add(CompositeGate(name="backward", operation="subtract",
                            operands=("lo", "mid")))
    assert gates.mask(frame, "forward").astype(int).tolist() == [0, 0, 0, 1, 0]
    assert gates.mask(frame, "backward").astype(int).tolist() == [1, 0, 0, 0, 0]


def test_more_than_two_operands_combine_in_order(gates, frame):
    gates.add(CompositeGate(name="c", operation="union",
                            operands=("lo", "hi", "mid")))
    assert gates.mask(frame, "c").all()


def test_describe_reads_as_the_sentence_it_is(gates):
    for operation, word in (("union", " or "), ("intersect", " and "),
                            ("subtract", " minus ")):
        gate = CompositeGate(name="c", operation=operation,
                             operands=("lo", "hi"))
        assert gate.describe() == f"lo{word}hi"


# ---------------------------------------------------------------------------
# One source of truth
# ---------------------------------------------------------------------------

def test_editing_an_operand_changes_the_composite(gates, frame):
    """The reason operands are names and not copies: a copy would leave the
    composite showing the old shape, silently, until somebody looked."""
    gates.add(CompositeGate(name="c", operation="union", operands=("lo", "hi")))
    before = gates.mask(frame, "c").sum()
    gates.add(ThresholdGate(name="lo", column="a", low=0.0, high=0.0))
    assert gates.mask(frame, "c").sum() < before


def test_an_operand_carries_its_own_ancestors(frame):
    """A gate drawn inside another MEANS the pair; combining it as the shape
    alone would include rows its parent excluded."""
    gates = GateSet()
    gates.add(ThresholdGate(name="parent", column="a", low=0.0, high=2.0))
    gates.add(ThresholdGate(name="child", column="a", low=1.0, high=4.0,
                            parent="parent"))
    gates.add(ThresholdGate(name="other", column="a", low=4.0, high=4.0))
    gates.add(CompositeGate(name="c", operation="union",
                            operands=("child", "other")))
    # child alone would take a=3; child-with-parent stops at 2.
    assert gates.mask(frame, "c").astype(int).tolist() == [0, 1, 1, 0, 1]


def test_a_deleted_operand_is_an_error_not_a_quiet_answer(gates, frame):
    gates.add(CompositeGate(name="c", operation="union", operands=("lo", "hi")))
    gates.remove("hi", cascade=False)
    with pytest.raises(GateError, match="no longer exists"):
        gates.mask(frame, "c")


def test_a_loop_is_a_sentence_naming_it_not_a_recursion_error(frame):
    gates = GateSet()
    gates.add(ThresholdGate(name="t", column="a", low=0.0, high=9.0))
    gates.gates.append(CompositeGate(name="x", operation="union",
                                     operands=("t", "y")))
    gates.gates.append(CompositeGate(name="y", operation="union",
                                     operands=("t", "x")))
    with pytest.raises(GateError, match="loop"):
        gates.mask(frame, "x")


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_combining_needs_two_gates():
    with pytest.raises(GateError, match="at least two"):
        CompositeGate(name="c", operation="union", operands=("lo",))


def test_a_gate_cannot_combine_itself():
    with pytest.raises(GateError, match="combines itself"):
        CompositeGate(name="c", operation="union", operands=("c", "lo"))


def test_the_same_operand_twice_is_refused_with_the_reason():
    """A gate unioned with itself is itself and subtracted from itself is
    empty; neither is likely to be meant."""
    with pytest.raises(GateError, match="same gate twice"):
        CompositeGate(name="c", operation="union", operands=("lo", "lo"))


def test_an_unknown_operation_names_the_ones_that_exist():
    with pytest.raises(GateError, match="union, intersect, subtract"):
        CompositeGate(name="c", operation="xor", operands=("lo", "hi"))


def test_evaluating_one_alone_says_to_ask_the_set(frame):
    gate = CompositeGate(name="c", operation="union", operands=("lo", "hi"))
    with pytest.raises(GateError, match="ask the gate set"):
        gate.mask(frame)


def test_it_reports_no_columns_of_its_own():
    """They are its operands', and only the set knows them; a guess would be
    a guess about gates it cannot see."""
    assert CompositeGate(name="c", operation="union",
                         operands=("lo", "hi")).columns == ()


# ---------------------------------------------------------------------------
# Not the same thing as a parent
# ---------------------------------------------------------------------------

def test_a_parent_is_sequential_gating_and_this_is_set_algebra(gates, frame):
    """A parent is always an intersection and always a tree. This is
    neither, which is why both exist."""
    gates.add(CompositeGate(name="either", operation="union",
                            operands=("lo", "hi")))
    nested = GateSet()
    nested.add(ThresholdGate(name="lo", column="a", low=0.0, high=2.0))
    nested.add(ThresholdGate(name="hi", column="a", low=3.0, high=4.0,
                             parent="lo"))
    assert gates.mask(frame, "either").sum() == 5
    assert nested.mask(frame, "hi").sum() == 0


def test_a_composite_can_still_have_a_parent(gates, frame):
    gates.add(CompositeGate(name="c", operation="union",
                            operands=("lo", "hi"), parent="mid"))
    # mid is 1..3, so the union is narrowed to it.
    assert gates.mask(frame, "c").astype(int).tolist() == [0, 1, 1, 1, 0]


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation", COMPOSITE_OPS)
def test_a_composite_survives_json(operation):
    gate = CompositeGate(name="c", operation=operation, operands=("lo", "hi"))
    assert gate_from_dict(json.loads(json.dumps(gate.to_dict()))) == gate


def test_a_whole_set_survives_json(gates, frame):
    gates.add(CompositeGate(name="c", operation="subtract",
                            operands=("mid", "lo")))
    restored = GateSet.from_dict(json.loads(json.dumps(gates.to_dict())))
    assert np.array_equal(restored.mask(frame, "c"), gates.mask(frame, "c"))


def test_moving_a_composite_moves_nothing(gates):
    """Its operands are other gates with their own users."""
    gate = CompositeGate(name="c", operation="union", operands=("lo", "hi"))
    assert gate.translated(5.0, 5.0) == gate
    assert gate.scaled(2.0) == gate
