"""The interactive plots without their library, and the restyle dialogs.

Two separable things live here.

**pyqtgraph is optional.** Its absence must cost the plots and nothing else,
which means the plot widget still has to BUILD -- with every attribute its
callers touch already set, and a notice saying what to install. A half-built
widget that raises on its third method is worse than one that raises on its
first, because the traceback then names a symptom instead of the cause.

**The restyle menu is a set of one-question dialogs.** Each one has the same
two outcomes and both matter: an answer reaches every artist on the plot, and
a cancel changes nothing at all. A cancelled dialog that half-applies is
worse than one that does nothing, because the user has no way back to what
they had.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPointF, Qt                # noqa: E402
from PySide6.QtWidgets import QMenu     # noqa: E402

from spacr.qt.widgets import fast_plots as fp         # noqa: E402

pytestmark = pytest.mark.qt


N = 40


def _frame(seed: int = 0) -> pd.DataFrame:
    """A coefficient table with one real hit, so a threshold line is drawn."""
    rng = np.random.default_rng(seed)
    p_values = rng.uniform(0.001, 0.99, N)
    p_values[0] = 1e-8
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(N)],
        "coefficient": rng.normal(0, 0.5, N),
        "p_value": p_values,
        "well_count": rng.integers(4, 96, N).astype("float64"),
        "n_guides": rng.integers(1, 4, N),
        "condition": list(rng.choice(["nc", "pc", "other"], N,
                                     p=[0.1, 0.1, 0.8])),
    })


@pytest.fixture
def volcano(qtbot):
    """A drawn volcano — the restyle dialogs need artists to act on."""
    plot = fp.VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(), effect_threshold=0.6)
    assert plot._scatter_items(), "the fixture drew no points to restyle"
    return plot


# ---------------------------------------------------------------------------
# pyqtgraph absent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the restyle dialogs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# "Show as", when the graph-type table or the redraw is unavailable
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure styles: what the menu can edit, and what it refuses to
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScaleLocationStyle:
    """A style with one field of every shape the menu has to sort out."""

    y_lim: tuple = (0.0, 1.0)
    point_colour: str = "#112233"
    dpi: int = 300
    marker: str = dataclasses.field(
        default="o", metadata={"choices": ("o", "s", "d")})
    corners: tuple = (0, 0, 1, 1)

    CHOICES = {"point_colour": ()}


@pytest.mark.parametrize("name,value,declared,expected", [
    ("y_lim", (0.0, 1.0), "", "pair"),
    ("corners", (0, 0, 1, 1), "", "unsupported"),
    ("labels", ("a", "b"), "", "unsupported"),
    ("split", None, "tuple[tuple[float, float], ...]", "unsupported"),
    ("mapping", None, "dict[str, str]", "unsupported"),
    ("grid", None, "bool", "flag"),
    ("visible", None, "bool | None", "flag"),
    ("x_lim", None, "tuple[float, float] | None", "pair"),
    ("dpi", None, "int | None", "number"),
    ("title", None, "str", "text"),
    ("line_colour", None, "str", "colour"),
    ("y_lims", None, "", "pair"),
    ("edge_color", None, "", "colour"),
    ("caption", None, "", "text"),
])
def test_a_style_field_is_sorted_into_the_control_that_can_edit_it(
        name, value, declared, expected):
    """A field offered as the wrong control writes a value nothing can read."""
    assert fp.style_field_kind(name, value, None, declared) == expected


def test_a_fields_own_declared_choices_are_the_closed_set_it_offers():
    """Declared beside the field is the most specific place to say it."""
    style = ScaleLocationStyle()

    assert fp.style_field_choices(style, "marker") == ("o", "s", "d")
    assert fp.style_field_choices(style, "marker", {"marker": ("x",)}) == ("x",)
    assert fp.style_field_choices(style, "dpi") == ()


def test_a_pair_is_labelled_as_the_range_it_is():
    """"0 to 1" is readable on a menu; a repr of a tuple is not."""
    assert fp.style_field_label("y_lim", (0.0, 1.5), "pair") == \
        "Y lim: 0 to 1.5…"


def test_a_style_kind_is_the_snake_case_of_its_class():
    """The kind keys the saved default, so one lab's style stays on one figure."""
    assert fp.style_kind(ScaleLocationStyle()) == "scale_location"


def test_a_group_with_no_fields_gets_no_submenu(qtbot):
    """An empty group would be a menu entry that opens onto nothing."""
    menu = QMenu()
    added = fp.add_style_entries(menu, ScaleLocationStyle())

    names = [action.text() for action in menu.actions()]
    assert "Data" not in names, "this style has no Data field to offer"
    assert "Axes" in names and "Size" in names
    assert added, "every editable field is still offered"


# ---------------------------------------------------------------------------
# Saving, loading and defaulting a whole style
# ---------------------------------------------------------------------------

def _entry(actions, fragment):
    for action in actions:
        if fragment in action.text():
            return action
    raise AssertionError(f"no entry containing {fragment!r}: "
                         f"{[a.text() for a in actions]}")


def test_loading_a_style_asks_for_a_path_and_applies_it(tmp_path, monkeypatch):
    """A round trip through the file is what makes a house style portable."""
    from PySide6.QtWidgets import QFileDialog

    saved = ScaleLocationStyle(dpi=72, point_colour="#ff0000")
    path = fp.save_style(saved, str(tmp_path / "house.json"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    style = ScaleLocationStyle()
    said = []
    menu = QMenu()
    actions = fp.add_style_file_entries(menu, style, note=said.append)

    _entry(actions, "Load style").trigger()

    assert style.dpi == 72
    assert style.point_colour == "#ff0000"


def test_a_house_default_can_be_set_and_then_cleared():
    """Per KIND, not per figure -- that is what makes it a house style.

    And the way back is not optional: a default that can only be set is the
    same trap as a colour that can only be set. Clearing twice is here too,
    because the menu is built once and the second press has nothing to
    remove -- which has to be a sentence rather than silence.
    """
    style = ScaleLocationStyle(dpi=150)
    said = []
    menu = QMenu()
    actions = fp.add_style_file_entries(menu, style, note=said.append)

    _entry(actions, "Use as the default").trigger()

    fresh = ScaleLocationStyle()
    assert fp.apply_default_style(fresh) == ["dpi"]
    assert fresh.dpi == 150

    # Rebuilt, because "Clear the default" doubles as the readout for
    # "is a house style in force here?" and is greyed when none is.
    said.clear()
    rebuilt = QMenu()
    actions = fp.add_style_file_entries(rebuilt, style, note=said.append)
    clear = _entry(actions, "Clear the default")
    assert clear.isEnabled() is True
    clear.trigger()

    assert fp.apply_default_style(ScaleLocationStyle()) == []
    assert any("gone" in message for message in said)

    said.clear()
    clear.trigger()

    assert any("no saved" in message for message in said)


# ---------------------------------------------------------------------------
# Modifier-drag selection, wrapped around pyqtgraph's own pan and zoom
# ---------------------------------------------------------------------------

class _Drag:
    """The parts of a pyqtgraph drag event the wrapper actually reads."""

    def __init__(self, modifier, down, now, finish=True,
                 button=Qt.LeftButton):
        self._modifier = modifier
        self._down = down
        self._now = now
        self._finish = finish
        self._button = button
        self.accepted = False

    def modifiers(self):
        return self._modifier

    def button(self):
        return self._button

    def accept(self):
        self.accepted = True

    def buttonDownPos(self):
        return self._down

    def pos(self):
        return self._now

    def isFinish(self):
        return self._finish


@pytest.fixture
def band(volcano):
    """A view box whose original drag handler is a recorder we can see."""
    box = volcano.plot.getViewBox()
    handed_back = []
    box.mouseDragEvent = lambda event, axis=None: handed_back.append(event)
    box._spacr_band = False
    volcano._install_rubber_band()
    return volcano, box, handed_back


def test_a_right_button_drag_is_handed_back_even_with_the_modifier(band):
    """Right-drag is pyqtgraph's own scale gesture and stays that way."""
    _volcano, box, handed_back = band
    event = _Drag(Qt.ControlModifier, QPointF(0, 0), QPointF(10, 10),
                  button=Qt.RightButton)

    box.mouseDragEvent(event)

    assert handed_back == [event]


