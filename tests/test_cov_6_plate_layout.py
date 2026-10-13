"""``PlateDesign`` rejects a layout or edge policy it cannot honour.

A misspelled ``layout="randmo"`` or ``edge_policy="skip"`` must fail at
construction, while the design still holds the values the caller passed.
If it were accepted, :func:`assign_wells` would fall through to its
default fill order and hand back a plate that silently contradicts the
design it was asked for -- a confound nobody would see until analysis.
The message has to name the offending value and list the legal ones,
because the caller is typing these into a GUI field.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.plate_layout import (  # noqa: E402
    EDGE_LEAVE_EMPTY, EDGE_USE, LAYOUTS, ROLE_TREATMENT,
    Condition, PlateDesign,
)


def _conditions():
    return (Condition("drug_a", 4, ROLE_TREATMENT),)


def test_an_unknown_layout_is_refused_and_the_legal_ones_are_named():
    """A typo in ``layout`` must not fall through to the default fill."""
    with pytest.raises(ValueError) as excinfo:
        PlateDesign(plate_id="plate1", plate_format=96,
                    conditions=_conditions(), layout="randmo")
    message = str(excinfo.value)
    assert "randmo" in message
    for legal in LAYOUTS:
        assert legal in message


def test_an_unknown_edge_policy_is_refused_and_both_choices_are_named():
    """Only ``use`` and ``leave_empty`` decide the fate of the outer ring."""
    with pytest.raises(ValueError) as excinfo:
        PlateDesign(plate_id="plate1", plate_format=96,
                    conditions=_conditions(), edge_policy="skip")
    message = str(excinfo.value)
    assert "skip" in message
    assert EDGE_USE in message
    assert EDGE_LEAVE_EMPTY in message


def test_the_layout_check_runs_before_the_edge_policy_check():
    """Both wrong: the layout is reported, so the first fix is the real one."""
    with pytest.raises(ValueError) as excinfo:
        PlateDesign(plate_id="plate1", plate_format=96,
                    conditions=_conditions(), layout="randmo",
                    edge_policy="skip")
    assert "randmo" in str(excinfo.value)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_every_advertised_layout_constructs(layout):
    """``LAYOUTS`` is the list the error message offers; each must work."""
    design = PlateDesign(plate_id="plate1", plate_format=96,
                         conditions=_conditions(), layout=layout)
    assert design.layout == layout
