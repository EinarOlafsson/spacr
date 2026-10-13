"""A trellis control shows the spec, or falls back — it never keeps a stale value.

Instruction 310, entry A56. ``TrellisPanelWidget._sync`` set each picker only
when the spec's value was found in it::

    index = self._kind.findData(spec.graph.kind or "")
    if index >= 0:
        self._kind.setCurrentIndex(index)

so a spec naming a kind the picker has no item for left the combo showing
whatever it happened to be showing before. That alone would be cosmetic. The
damage is that ``_on_controls_changed`` reads the PICKER rather than the spec,
so the next touch of any control wrote the stale value back into the spec.

Reproduced before fixing, exactly as the entry describes: with the picker on
Histogram, pushing a spec with ``kind="empty"`` left the picker reading
"Histogram" while ``spec.graph.kind`` was ``"empty"``; moving the Bins box
then made ``spec.graph.kind == "histogram"``. A plot kind the user never chose,
arriving from a restored layout, with nothing on screen to explain it.

The same ``>= 0`` pattern guarded both scale pickers, so they are covered here
too.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.trellis_view import (EMPTY, GraphSpec, TrellisPanelWidget,
                                           TrellisSpec)


@pytest.fixture()
def panel(qtbot):
    widget = TrellisPanelWidget()
    qtbot.addWidget(widget)
    return widget


def _spec(kind: str, bins: int = 7) -> TrellisSpec:
    return TrellisSpec(graph=GraphSpec(x="area", kind=kind, bins=bins))


def test_a_known_kind_is_shown(panel):
    panel.set_spec(_spec("histogram"))
    assert panel._kind.currentData() == "histogram"
    assert panel.spec.graph.kind == "histogram"


def test_an_unknown_kind_falls_back_instead_of_keeping_the_last_one(panel):
    """The picker must not go on claiming a kind the spec does not name."""
    panel.set_spec(_spec("histogram"))
    assert panel._kind.currentText() == "Histogram"

    panel.set_spec(_spec(EMPTY))
    assert panel._kind.currentData() == "", (
        f"the picker still reads {panel._kind.currentText()!r} for a spec "
        f"whose kind is {panel.spec.graph.kind!r}"
    )
    assert panel._kind.currentText() == "Automatic"


def test_touching_a_control_cannot_resurrect_the_stale_kind(panel):
    """The cost the entry names, asserted directly.

    Before the fix this turned the spec into a histogram the user never chose.
    """
    panel.set_spec(_spec("histogram"))
    panel.set_spec(_spec(EMPTY))
    panel._bins.setValue(9)
    assert panel.spec.graph.kind != "histogram", (
        "moving the Bins box rewrote the spec with the kind the picker was "
        "stale on, which is the A56 defect"
    )


def test_the_spec_itself_refuses_an_unknown_scale_mode(panel):
    """The scale pickers carried the same ``>= 0`` guard, and it is
    unreachable: ``TrellisSpec`` validates the mode before a picker ever sees
    it.

    Written after trying to exercise the fallback and finding it impossible --
    the constructor raises. The fallback there is now belt-and-braces rather
    than a fix for a live path, and saying so is worth more than a test that
    pretends otherwise. The PLOT KIND is the half that was really reachable,
    because ``kind`` accepts values the picker has no item for; that is what
    the tests above cover.
    """
    from dataclasses import replace

    from spacr.qt.widgets.trellis_spec import SpecError

    with pytest.raises(SpecError):
        panel.set_spec(replace(_spec("histogram"), scale_x="not-a-mode"))

    # A mode the spec does accept still reaches its picker.
    panel.set_spec(replace(_spec("histogram"), scale_x="free"))
    assert panel._scale_x.currentData() == "free"
