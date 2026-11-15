"""Trellis view — the three guards that only a degenerate input reaches.

``tests/qt/test_trellis.py`` and ``tests/test_cov_w2_4_trellis_view.py`` cover
the ordinary grid: panels, shared limits, brushing, the highlight. What is
left is the handful of branches that only fire when the input is *not*
ordinary, and each one is a guard whose absence would be a visible break
rather than a crash:

* the blank slot at the end of a wrapped grid, which must never be captioned:
  a slot with no group is the remainder of a division, and ``n = 0`` on it
  would read as a group that was measured and came back empty;
* a spec carrying a plot kind or a scale mode that this build's picker does
  not offer — ``EMPTY`` is a real :data:`PLOT_KINDS` member that the Plot
  picker deliberately omits — where the picker has to keep the item it has
  rather than blank itself.

``seen`` is four measured rows faceted across ``gene`` (three in *a*, one in *b*).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.theme import active_palette
from spacr.qt.widgets.graph_spec import CONTINUOUS, EMPTY, HISTOGRAM, SCATTER, GraphSpec
from spacr.qt.widgets.trellis_spec import (SCALE_FREE, SCALE_MODES,
                                          SCALE_SHARED, TrellisSpec)
from spacr.qt.widgets.trellis_view import TrellisCanvas, TrellisPanelWidget

#: ``classify_columns`` calls a numeric column with twelve or fewer distinct
#: values a category, and every frame here is tiny on purpose. Saying that
#: ``area`` is a measurement keeps the bins above from becoming tick labels.
NUMERIC = {"area": CONTINUOUS}


@pytest.fixture
def seen() -> pd.DataFrame:
    """Four measured objects, three in gene *a* and one in gene *b*."""
    return pd.DataFrame({"gene": ["a", "a", "a", "b"],
                         "area": [1.0, 2.0, 3.0, 10.0]})


@pytest.fixture
def panel(qtbot) -> TrellisPanelWidget:
    widget = TrellisPanelWidget(link=LinkedSelection())
    qtbot.addWidget(widget)
    return widget



# ---------------------------------------------------------------------------
# The blank slot at the end of a wrapped grid
# ---------------------------------------------------------------------------

def test_the_blank_slot_of_a_wrapped_grid_is_never_given_a_caption(qtbot):
    """Seven plates three wide leaves two slots that are not groups.

    Every panel of a trellis carries its n, and that is the whole reason a
    blank slot must carry nothing: captioned, it would read ``n = 0``, which
    in this grid means "this plate was measured and every object was filtered
    out" — a fact about the data. The remainder of 7 / 3 is not a fact about
    the data. The labeller is driven both ways over the same Axes: the real
    panel replaces the sentinel title with its group and its n, the blank slot
    leaves whatever was there untouched.
    """
    frame = pd.DataFrame({"plateID": [f"p{i}" for i in range(1, 8)],
                          "area": np.arange(7.0)})
    canvas = TrellisCanvas(link=LinkedSelection())
    qtbot.addWidget(canvas)
    canvas.set_frame(frame)
    canvas.set_trellis_spec(TrellisSpec(
        graph=GraphSpec(x="area", facet_col="plateID", roles=NUMERIC), wrap=3))
    result = canvas.trellis
    assert result.shape == (3, 3)

    blanks = [p for p in result.panels if not p.occupied]
    assert [(p.row, p.col) for p in blanks] == [(2, 1), (2, 2)]
    assert [p.label() for p in blanks] == ["", ""]
    # The renderer hides them rather than drawing an empty chart.
    assert not canvas.panel_axes()[(2, 1)].get_visible()

    axes = canvas.panel_axes()[(0, 0)]
    axes.set_title("SENTINEL")
    canvas._label_trellis_panel(axes, blanks[0], result, active_palette())
    assert axes.get_title() == "SENTINEL"

    canvas._label_trellis_panel(axes, result.panel(0, 0), result,
                                active_palette())
    assert axes.get_title() == "p1  ·  n = 1"


# ---------------------------------------------------------------------------
# A spec the pickers cannot show
# ---------------------------------------------------------------------------

def test_a_plot_kind_the_picker_cannot_offer_falls_back_to_automatic(
        panel, seen):
    """``EMPTY`` is a real kind, and the Plot picker deliberately omits it.

    The picker lists "Automatic" plus every :data:`PLOT_KINDS` member except
    ``EMPTY`` — there is nothing to override when nothing has been dropped —
    so a spec pushed in from a saved layout can name a kind the picker has no
    item for. Selecting index -1 would blank the box, and KEEPING THE LAST
    KIND is worse than blank: `_on_controls_changed` reads the PICKER, so a
    box still reading "Histogram" under a spec that says ``empty`` turns the
    spec into a histogram the moment any other control is touched. That was
    the defect instruction 310 A56 reported. "Automatic" is the honest
    fallback — it claims no kind the spec did not ask for.

    The rest of the sync has to happen anyway, which is what the bins and wrap
    assertions below pin: the fallback replaces one control's value, it does
    not abandon the method.
    """
    panel.set_frame(seen)
    panel.zone("x").set_column("area")
    panel._kind.setCurrentIndex(panel._kind.findData(HISTOGRAM))
    assert panel.spec.graph.kind == HISTOGRAM

    panel.set_spec(TrellisSpec(graph=GraphSpec(x="area", kind=EMPTY, bins=7),
                               wrap=4))

    assert panel.spec.graph.kind == EMPTY
    assert panel._kind.currentIndex() == 0
    assert panel._kind.currentData() == ""
    assert panel._bins.value() == 7
    assert panel._wrap.value() == 4

    # ...and a kind the picker does have moves it, so the branch above is a
    # skip rather than a picker that never follows the spec.
    panel.set_spec(TrellisSpec(graph=GraphSpec(x="area", y="area",
                                               kind=SCATTER, bins=5),
                               wrap=4))
    assert panel._kind.currentData() == SCATTER
    assert panel._bins.value() == 5


def test_a_scale_mode_the_picker_cannot_offer_falls_back_to_the_first(
        panel, seen):
    """A mode from a newer spaCR must not blank this build's scale box.

    ``TrellisSpec`` validates its modes, so the only way a picker meets one it
    cannot show is a spec written elsewhere — a saved trellis from a build
    that offers a mode this one does not. The X-scale box must not empty
    itself, and it must not keep the mode it happened to be showing either:
    `_on_controls_changed` reads the PICKER, so a stale box republishes its
    own value as the spec's the next time any control moves. It falls back to
    the first entry, and the rest of the sync still runs — the Y box and the
    bin count below both follow the incoming spec.
    """
    panel._scale_x.setCurrentIndex(panel._scale_x.findData(SCALE_FREE))
    assert panel.spec.scale_x == SCALE_FREE

    # Written past __post_init__ on purpose: the constructor rejects the very
    # input this guard exists for, so a valid spec cannot carry it here.
    future = TrellisSpec(graph=GraphSpec(bins=9), scale_y=SCALE_SHARED)
    object.__setattr__(future, "scale_x", "logarithmic")
    panel.set_spec(future)

    assert panel.spec.scale_x == "logarithmic"
    assert panel._scale_x.currentIndex() == 0
    assert panel._scale_x.currentData() == SCALE_MODES[0]
    assert panel._scale_y.currentData() == SCALE_SHARED
    assert panel._bins.value() == 9

    # ...and a mode the picker does have still moves it.
    panel.set_spec(TrellisSpec(graph=GraphSpec(bins=9), scale_x=SCALE_SHARED))
    assert panel._scale_x.currentData() == SCALE_SHARED
