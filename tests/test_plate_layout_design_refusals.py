"""``PlateDesign`` refuses a layout or edge policy it cannot honour.

A misspelled ``layout="randmo"`` or ``edge_policy="skip"`` must fail at
construction: accepted, :func:`assign_wells` would fall through to its default
fill order and hand back a plate that silently contradicts the design.  The
message names the offending value and lists the legal ones, because the
caller is typing these into a GUI field.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.plate_layout import (  # noqa: E402
    EDGE_LEAVE_EMPTY, EDGE_USE, LAYOUTS, ROLE_TREATMENT,
    Condition, PlateDesign,
)


def test_plate_design_accepts_every_advertised_layout_and_names_the_legal_ones(
        subtests):
    """Each layout in ``LAYOUTS`` constructs; a bad layout or edge policy is refused by name."""
    conditions = (Condition("drug_a", 4, ROLE_TREATMENT),)
    for layout in sorted(LAYOUTS):
        with subtests.test(msg=f"layout-{layout}"):
            design = PlateDesign(plate_id="plate1", plate_format=96,
                                 conditions=conditions, layout=layout)
            assert (design.layout, design.plate_id) == (layout, "plate1")
    layout_refusal = ("layout='randmo'; choose one of "
                      "('random', 'row', 'column', 'block').")
    refusals = {
        "bad-layout": ({"layout": "randmo"}, layout_refusal),
        "bad-edge-policy": ({"edge_policy": "skip"},
                            "edge_policy='skip'; choose 'use' or 'leave_empty'."),
        # Both wrong: the layout is reported first, so the first fix is the real one.
        "both-bad-layout-first": ({"layout": "randmo", "edge_policy": "skip"},
                                  layout_refusal),
    }
    assert LAYOUTS == ("random", "row", "column", "block")
    assert (EDGE_USE, EDGE_LEAVE_EMPTY) == ("use", "leave_empty")
    for case, (kwargs, message) in refusals.items():
        with subtests.test(msg=case):
            with pytest.raises(ValueError) as excinfo:
                PlateDesign(plate_id="plate1", plate_format=96,
                            conditions=conditions, **kwargs)
            assert str(excinfo.value) == message
