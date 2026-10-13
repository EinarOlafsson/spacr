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

import builtins
import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPointF, Qt                # noqa: E402
from PySide6.QtGui import QColor                      # noqa: E402
from PySide6.QtWidgets import QInputDialog, QMenu     # noqa: E402

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


def _answer(monkeypatch, **canned):
    """Make each QInputDialog getter answer without opening anything."""
    for name, value in canned.items():
        monkeypatch.setattr(QInputDialog, name, staticmethod(
            lambda *a, _v=value, **k: _v))


# ---------------------------------------------------------------------------
# pyqtgraph absent
# ---------------------------------------------------------------------------

_ATTRIBUTES = (
    "plot", "_background", "_foreground", "_status", "_headline", "_note",
    "_labels", "_legend_colours", "_items", "_keys", "_key_rows", "_row_xy",
    "_selected_key", "_selected_keys", "_extra_highlights", "_highlight",
    "_refit", "_baselines", "_compartments", "_corrections", "_encodings",
    "_p_values", "_thresholds", "_marks", "_frame", "_style_note",
    "_legend_box",
)


def test_without_pyqtgraph_the_plot_still_builds_and_says_what_to_install(
        qtbot, monkeypatch):
    """An optional library's absence may cost the plot, never the module."""
    monkeypatch.setattr(fp, "HAVE_PYQTGRAPH", False)

    plot = fp.FastPlot(title="Guide effects")
    qtbot.addWidget(plot)

    from PySide6.QtWidgets import QLabel
    texts = " ".join(label.text() for label in plot.findChildren(QLabel))
    assert "Guide effects" in texts
    assert fp.PYQTGRAPH_MISSING_MESSAGE in texts
    for name in _ATTRIBUTES:
        assert hasattr(plot, name), f"{name} was left unset by the fallback"
    assert plot._legend_box.isEnabled() is False
    assert plot._thresholds == ([], None, None)


def test_without_pyqtgraph_the_absent_plot_absorbs_the_calls_made_on_it(
        qtbot, monkeypatch):
    """The stand-in has to survive the chained calls the subclasses make."""
    monkeypatch.setattr(fp, "HAVE_PYQTGRAPH", False)

    plot = fp.FastPlot()
    qtbot.addWidget(plot)

    assert list(plot.plot.listDataItems()) == []
    assert bool(plot.plot) is False
    assert plot.plot.scene().sigMouseClicked is not None
    assert "absent" in repr(plot.plot)


def test_a_caller_with_no_fallback_is_told_what_to_install(monkeypatch):
    """Where degrading is impossible, the message still names the extra."""
    monkeypatch.setattr(fp, "HAVE_PYQTGRAPH", False)

    with pytest.raises(RuntimeError) as caught:
        fp._require_pyqtgraph()

    assert str(caught.value) == fp.PYQTGRAPH_MISSING_MESSAGE


# ---------------------------------------------------------------------------
# the restyle dialogs
# ---------------------------------------------------------------------------

def test_a_point_size_answer_reaches_every_scatter(volcano, monkeypatch):
    """One answer, applied to the data artists and not to the cursor."""
    _answer(monkeypatch, getDouble=(21.0, True))

    volcano._ask_point_size()

    assert [item.opts["size"] for item in volcano._scatter_items()] == \
        [21.0] * len(volcano._scatter_items())


def test_a_cancelled_point_size_changes_nothing(volcano, monkeypatch):
    """Cancel is a real answer and it means "leave it as it was"."""
    before = [item.opts["size"] for item in volcano._scatter_items()]
    _answer(monkeypatch, getDouble=(21.0, False))

    volcano._ask_point_size()

    assert [item.opts["size"] for item in volcano._scatter_items()] == before


def test_an_opacity_answer_reaches_every_scatter(volcano, monkeypatch):
    """0 is invisible and 1 is solid; anything between has to arrive intact."""
    _answer(monkeypatch, getDouble=(0.25, True))

    volcano._ask_opacity()

    for item in volcano._scatter_items():
        assert item.opacity() == pytest.approx(0.25)


def test_a_cancelled_opacity_changes_nothing(volcano, monkeypatch):
    """The contrast that makes the applied case mean something."""
    before = [item.opacity() for item in volcano._scatter_items()]
    _answer(monkeypatch, getDouble=(0.25, False))

    volcano._ask_opacity()

    assert [item.opacity() for item in volcano._scatter_items()] == before


def test_a_point_colour_overrides_whatever_the_points_were_coloured_by(
        volcano, monkeypatch):
    """A deliberate single colour is allowed to displace a categorical one."""
    monkeypatch.setattr(fp, "pick_colour",
                        lambda *a, **k: QColor("#ff0055"))

    volcano._ask_point_colour()

    for item in volcano._scatter_items():
        assert item.opts["brush"].color().name() == "#ff0055"


def test_a_cancelled_colour_dialog_leaves_the_colouring_alone(volcano,
                                                              monkeypatch):
    """An invalid colour is what a cancelled colour dialog hands back."""
    before = [item.opts["brush"] for item in volcano._scatter_items()]
    monkeypatch.setattr(fp, "pick_colour", lambda *a, **k: QColor())

    volcano._ask_point_colour()

    assert [item.opts["brush"] for item in volcano._scatter_items()] == before


def test_both_axis_labels_are_asked_for_and_both_are_applied(volcano,
                                                             monkeypatch):
    """Two questions, one change: a half-renamed pair of axes is a wrong plot."""
    answers = iter([("effect size", True), ("-log10 p", True)])
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: next(answers)))

    volcano._ask_labels()

    assert volcano.plot.getAxis("bottom").labelText == "effect size"
    assert volcano.plot.getAxis("left").labelText == "-log10 p"


def test_cancelling_the_second_axis_label_leaves_the_first_one_alone(
        volcano, monkeypatch):
    """Renaming x and then giving up must not rename x either."""
    before_x = volcano.plot.getAxis("bottom").labelText
    before_y = volcano.plot.getAxis("left").labelText
    answers = iter([("effect size", True), ("-log10 p", False)])
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: next(answers)))

    volcano._ask_labels()

    assert volcano.plot.getAxis("bottom").labelText == before_x
    assert volcano.plot.getAxis("left").labelText == before_y


def test_cancelling_the_first_axis_label_asks_nothing_further(volcano,
                                                              monkeypatch):
    """A user who cancels is not made to answer the second question."""
    asked = []

    def once(*_args, **_kwargs):
        asked.append(1)
        return ("", False)

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(once))

    volcano._ask_labels()

    assert len(asked) == 1


def test_an_effect_size_cut_is_handed_to_the_caller_that_asked_for_it(
        volcano, monkeypatch):
    """The dialog does not apply the cut; it reports the number chosen."""
    _answer(monkeypatch, getDouble=(2.5, True))
    chosen = []

    volcano._ask_threshold_multiplier(1.0, chosen.append)

    assert chosen == [2.5]


def test_a_cancelled_effect_size_cut_calls_nobody(volcano, monkeypatch):
    """Cancel must not re-apply the current value as if it were new."""
    _answer(monkeypatch, getDouble=(2.5, False))
    chosen = []

    volcano._ask_threshold_multiplier(1.0, chosen.append)

    assert chosen == []


def test_a_y_axis_split_takes_both_ends_before_it_is_applied(volcano,
                                                             monkeypatch):
    """A split is a band; one number is not a band."""
    answers = iter([(4.0, True), (7.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    volcano._ask_y_split()

    assert volcano.y_split() == (4.0, 7.0), volcano.style_note()


def test_cancelling_either_end_of_a_split_leaves_the_axis_whole(volcano,
                                                               monkeypatch):
    """Neither half of the answer alone is a split."""
    first = iter([(4.0, False)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(first)))
    volcano._ask_y_split()
    assert volcano.y_split() is None

    second = iter([(4.0, True), (7.0, False)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(second)))
    volcano._ask_y_split()

    assert volcano.y_split() is None


# ---------------------------------------------------------------------------
# "Show as", when the graph-type table or the redraw is unavailable
# ---------------------------------------------------------------------------

def test_a_redraw_that_fails_leaves_the_plot_as_it_was(volcano):
    """A plot that cannot become a bar chart stays the plot it is."""
    before = len(volcano._scatter_items())

    volcano._show_as_kind("bar")

    assert len(volcano._scatter_items()) == before


def test_without_the_graph_type_table_no_show_as_menu_is_offered(volcano,
                                                                 monkeypatch):
    """An unavailable table costs the submenu, never the rest of the menu."""
    volcano.spec = types.SimpleNamespace(
        frame=_frame(), group="condition", value="coefficient", kind="bar")
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if level and name == "graph_types":
            raise ImportError("graph types are unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    menu = QMenu()

    assert volcano._offer_graph_kinds(menu) is None
    assert menu.actions() == []


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


def test_saving_a_style_asks_for_a_path_and_says_where_it_went(tmp_path,
                                                               monkeypatch):
    """A save that says nothing is a save the user repeats."""
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "house.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    said = []
    menu = QMenu()
    actions = fp.add_style_file_entries(menu, ScaleLocationStyle(),
                                        note=said.append)

    _entry(actions, "Save style").trigger()

    assert target.is_file()
    assert any(str(target) in message for message in said)


def test_a_style_that_cannot_be_written_names_the_reason(tmp_path,
                                                         monkeypatch):
    """"Could not write" with the errno beats a menu entry that does nothing."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "s.json"),
                                                      "")))

    def read_only(*_args, **_kwargs):
        raise OSError("Read-only file system")

    monkeypatch.setattr(fp, "save_style", read_only)
    said = []
    menu = QMenu()
    actions = fp.add_style_file_entries(menu, ScaleLocationStyle(),
                                        note=said.append)

    _entry(actions, "Save style").trigger()

    assert any("Could not write the style" in m and "Read-only" in m
               for m in said)


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


def test_a_cancelled_load_changes_nothing(monkeypatch):
    """Cancel is answered with an empty path, which is not a file."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    style = ScaleLocationStyle()
    said = []
    menu = QMenu()
    actions = fp.add_style_file_entries(menu, style, note=said.append)

    _entry(actions, "Load style").trigger()

    assert style == ScaleLocationStyle()
    assert said == []


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


def test_clearing_is_greyed_out_when_no_default_is_in_force():
    """The entry is the readout: greyed means "this figure is the package's"."""
    said = []
    menu = QMenu()
    actions = fp.add_style_file_entries(menu, ScaleLocationStyle(),
                                        note=said.append)

    clear = _entry(actions, "Clear the default")

    assert clear.isEnabled() is False
    assert "scale_location" in clear.toolTip()


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


def test_a_plain_drag_is_handed_straight_back_to_pyqtgraph(band):
    """A plot that lost its pan and zoom to gain a selection would be worse."""
    volcano, box, handed_back = band
    event = _Drag(Qt.NoModifier, QPointF(0, 0), QPointF(10, 10))

    box.mouseDragEvent(event)

    assert handed_back == [event]
    assert event.accepted is False
    assert volcano.selected_keys() == []


def test_a_right_button_drag_is_handed_back_even_with_the_modifier(band):
    """Right-drag is pyqtgraph's own scale gesture and stays that way."""
    _volcano, box, handed_back = band
    event = _Drag(Qt.ControlModifier, QPointF(0, 0), QPointF(10, 10),
                  button=Qt.RightButton)

    box.mouseDragEvent(event)

    assert handed_back == [event]


def test_a_modified_drag_in_progress_draws_the_band_and_selects_nothing_yet(
        band):
    """Selection lands on release; a drag still moving is just a rectangle."""
    volcano, box, handed_back = band
    event = _Drag(Qt.ControlModifier, QPointF(0, 0), QPointF(10, 10),
                  finish=False)

    assert box.mouseDragEvent(event) is None

    assert handed_back == [], "the wrapper took this one"
    assert event.accepted is True
    assert volcano.selected_keys() == []


def test_a_finished_modified_drag_selects_the_points_it_enclosed(band):
    """The whole point of the gesture: a rectangle becomes a selection."""
    volcano, box, _handed_back = band
    (low_x, high_x), (low_y, high_y) = volcano.axis_limits()
    span_x, span_y = high_x - low_x, high_y - low_y
    event = _Drag(
        Qt.ShiftModifier,
        box.mapFromView(QPointF(low_x - span_x, low_y - span_y)),
        box.mapFromView(QPointF(high_x + span_x, high_y + span_y)))

    box.mouseDragEvent(event)

    assert event.accepted is True
    assert volcano.selected_keys(), "a band over the whole plot selected nothing"


def test_the_drag_wrapper_is_installed_once_not_once_per_call(band):
    """Wrapping a wrapper would run the selection twice on one drag."""
    volcano, box, _handed_back = band
    wrapped = box.mouseDragEvent

    volcano._install_rubber_band()

    assert box.mouseDragEvent is wrapped
