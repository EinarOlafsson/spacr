"""The restyle menu is built from the style's fields, not from a list.

Instruction 108, point 3:

    "A mixin for the Qt canvas that adds a right-click menu built FROM THE
     STYLE OBJECT'S FIELDS rather than from a hand-written list per figure:
     a bool renders a checkbox, a bounded float a spin box, a field with a
     declared choice set a submenu. Then 'as many settings as possible,
     depending on the graph' is automatic -- a style gains a field, the menu
     gains an entry, and the two cannot fall out of step."

and its acceptance test, quoted:

    "asserted by comparing the menu's actions against
     `dataclasses.fields(style)`, so a new field cannot be silently missing
     from the menu."

THAT COMPARISON IS ONLY MEANINGFUL IF NOTHING IS SKIPPED. A field the menu
has no dialog for -- the split axis's pair of PAIRS, the per-point annotation
map -- is still listed, greyed, and says why: a setting silently absent is one
the user has been told about and cannot find, which is the failure instruction
106 exists to prevent.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@dataclasses.dataclass
class _Style:
    """A style of every kind the mechanism has to answer for."""

    title: str = "Counts per well"
    legend: bool = True
    grid: bool = False
    marker_size: float = 8.0
    dpi: int = 200
    line_color: str = "#404040"
    x_lim: tuple | None = None
    effect_threshold: float | None = None
    line_style: str = "--"
    annotations: dict = dataclasses.field(default_factory=dict)

    CHOICES = {"line_style": ("-", "--", ":")}


#: Menus built here are held for the test's lifetime. A QMenu with no parent
#: is Python-owned, so `_action(_menu(style), ...)` would collect the menu --
#: and every QAction in it -- between building it and triggering one, which
#: surfaces as "Internal C++ object (QAction) already deleted". The
#: application does not have the problem: `build_style_menu` parents its menu
#: to the plot widget.
_KEEP: list = []


def _menu(style=None, on_change=None, choices=None):
    from PySide6.QtWidgets import QMenu

    from spacr.qt.widgets.fast_plots import add_style_entries

    menu = QMenu()
    _KEEP.append(menu)
    add_style_entries(menu, style if style is not None else _Style(),
                      on_change, choices=choices)
    return menu


def _reachable(menu):
    """Every name a reader meets: entries and the groups holding them."""
    from spacr.qt.widgets.fast_plots import menu_entries, menu_groups

    return ([action.text() for action in menu_entries(menu)]
            + list(menu_groups(menu)))


def _pretty(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _entry(menu, name: str):
    for text in _reachable(menu):
        if text.startswith(_pretty(name)):
            return text
    raise AssertionError(f"{name} is not on the menu: {_reachable(menu)}")


def _action(menu, name: str):
    from spacr.qt.widgets.fast_plots import menu_entries

    for action in menu_entries(menu):
        if action.text().startswith(_pretty(name)):
            return action
    raise AssertionError(f"{name} has no action: {_reachable(menu)}")


# --------------------------------------------------------------------------- #
#  The acceptance test instruction 108 names
# --------------------------------------------------------------------------- #

def test_every_field_of_the_style_is_on_the_menu(qtbot):
    style = _Style()
    menu = _menu(style)

    reachable = _reachable(menu)
    for field in dataclasses.fields(style):
        assert any(text.startswith(_pretty(field.name))
                   for text in reachable), (
            f"{field.name} is missing from the menu: {reachable}")


def test_the_real_volcano_style_is_covered_field_for_field(qtbot):
    """Sixty-three fields, and the mechanism was written against none of
    them by name."""
    from spacr.volcano_style import VolcanoStyle

    style = VolcanoStyle()
    menu = _menu(style)

    reachable = _reachable(menu)
    missing = [field.name for field in dataclasses.fields(style)
               if not any(text.startswith(_pretty(field.name))
                          for text in reachable)]
    assert missing == [], missing


def test_a_field_added_to_the_style_appears_without_touching_the_menu(qtbot):
    """The whole point: the two cannot fall out of step."""
    grown = dataclasses.make_dataclass(
        "_Grown", [("watermark", str, dataclasses.field(default="draft"))],
        bases=(_Style,))

    reachable = _reachable(_menu(grown()))

    assert any(text.startswith("Watermark") for text in reachable), reachable


# --------------------------------------------------------------------------- #
#  A field is edited with the control its kind wants
# --------------------------------------------------------------------------- #

def test_a_flag_is_a_tick_that_shows_its_state(qtbot):
    menu = _menu(_Style())

    assert _action(menu, "legend").isCheckable()
    assert _action(menu, "legend").isChecked()
    assert not _action(menu, "grid").isChecked()


def test_toggling_a_flag_off_and_on_puts_the_style_back(qtbot):
    """Instruction 108's own acceptance: the style is the single source of
    truth and nothing is lost on the way round."""
    style = _Style()

    _action(_menu(style), "legend").trigger()
    assert style.legend is False

    _action(_menu(style), "legend").trigger()
    assert style.legend is True


def test_a_closed_set_is_a_submenu_of_its_values(qtbot):
    """Declared on the style's class here; the field's own metadata and the
    caller's mapping are read too, because the styles in this package use all
    three and reading one would turn a closed set into a free-text box."""
    from spacr.qt.widgets.fast_plots import menu_entries

    menu = _menu(_Style())

    offered = {action.text() for action in menu_entries(menu)
               if action.text() in ("-", "--", ":")}
    assert offered == {"-", "--", ":"}
    ticked = [action.text() for action in menu_entries(menu)
              if action.isCheckable() and action.isChecked()
              and action.text() in ("-", "--", ":")]
    assert ticked == ["--"]


def test_choosing_a_value_off_a_closed_set_writes_it(qtbot):
    from spacr.qt.widgets.fast_plots import menu_entries

    style = _Style()
    menu = _menu(style)

    next(a for a in menu_entries(menu) if a.text() == ":").trigger()

    assert style.line_style == ":"


def test_a_field_with_no_dialog_is_greyed_and_says_why(qtbot):
    """`annotations` is a map of per-point notes. Offering it as a text box
    would write a value the renderer cannot read."""
    action = _action(_menu(_Style()), "annotations")

    assert not action.isEnabled()
    assert "not one the menu can edit" in action.text()
    assert action.toolTip()


def test_the_current_value_is_in_the_label(qtbot):
    """A menu of settings that does not say what they are set to is one the
    user has to open every entry of to read."""
    menu = _menu(_Style())

    assert _entry(menu, "marker_size") == "Marker size: 8…"
    assert _entry(menu, "dpi") == "Dpi: 200…"
    assert _entry(menu, "x_lim") == "X lim: automatic…"


def test_a_number_dialog_writes_the_number(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    style = _Style()
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (14.5, True)))

    _action(_menu(style), "marker_size").trigger()

    assert style.marker_size == 14.5


def test_an_integer_field_stays_an_integer(qtbot, monkeypatch):
    """`dpi=300.0` is a float where matplotlib wants a count, and the kind of
    difference that surfaces three layers away."""
    from PySide6.QtWidgets import QInputDialog

    style = _Style()
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (300.0, True)))

    _action(_menu(style), "dpi").trigger()

    assert style.dpi == 300
    assert isinstance(style.dpi, int)


def test_a_colour_field_opens_a_colour_dialog(qtbot, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    style = _Style()
    monkeypatch.setattr(QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor("#ff0000")))

    _action(_menu(style), "line_color").trigger()

    assert style.line_color == "#ff0000"


def test_a_nullable_number_is_asked_for_as_a_number(qtbot, monkeypatch):
    """`effect_threshold: float | None = None` holds nothing, so its
    annotation is the only thing left that says it is a number."""
    from PySide6.QtWidgets import QInputDialog

    style = _Style()
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (0.8, True)))

    _action(_menu(style), "effect_threshold").trigger()

    assert style.effect_threshold == 0.8


def test_a_pair_asks_twice_and_a_cancelled_second_writes_nothing(qtbot,
                                                                 monkeypatch):
    """The rule the axis-limit dialog already follows: a half-set pair is a
    range nobody chose."""
    from PySide6.QtWidgets import QInputDialog

    style = _Style()
    answers = iter([(-1.0, True), (1.0, False)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(_menu(style), "x_lim").trigger()

    assert style.x_lim is None


def test_a_pair_that_is_answered_is_written(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    style = _Style()
    answers = iter([(-1.0, True), (1.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))

    _action(_menu(style), "x_lim").trigger()

    assert style.x_lim == (-1.0, 1.0)


def test_the_host_is_told_what_changed(qtbot):
    """Where the host redraws."""
    told = []
    style = _Style()
    menu = _menu(style, lambda name, value: told.append((name, value)))

    _action(menu, "legend").trigger()

    assert told == [("legend", False)]


# --------------------------------------------------------------------------- #
#  Grouped the way the plot's own menu is
# --------------------------------------------------------------------------- #

def test_the_fields_are_grouped_and_the_groups_are_the_plots_own(qtbot):
    from spacr.qt.widgets.fast_plots import menu_groups

    groups = menu_groups(_menu(_Style()))

    for wanted in ("Axes", "Appearance", "Size"):
        assert wanted in groups, groups


def test_a_greyed_field_inside_a_group_can_still_show_its_reason(qtbot):
    """`setToolTipsVisible` is per menu, and these are all submenus."""
    menu = _menu(_Style())

    submenus = [action.menu() for action in menu.actions()
                if action.menu() is not None]
    assert submenus
    for submenu in submenus:
        assert submenu.toolTipsVisible(), submenu.title()


# --------------------------------------------------------------------------- #
#  On a real plot
# --------------------------------------------------------------------------- #

def test_a_plot_offered_a_style_carries_it_on_its_right_click_menu(qtbot):
    from spacr.qt.widgets.fast_plots import QQPlot, menu_groups

    import numpy as np

    plot = QQPlot()
    qtbot.addWidget(plot)
    plot.set_p_values(np.random.default_rng(0).random(40))
    style = _Style()

    plot.offer_style(style)

    groups = menu_groups(plot.build_style_menu())
    assert "Figure style" in groups, groups


def test_a_plot_offered_no_style_grows_no_such_group(qtbot):
    """A widget that always showed the heading would be one the reader has to
    learn to ignore."""
    import numpy as np

    from spacr.qt.widgets.fast_plots import QQPlot, menu_groups

    plot = QQPlot()
    qtbot.addWidget(plot)
    plot.set_p_values(np.random.default_rng(0).random(40))

    assert "Figure style" not in menu_groups(plot.build_style_menu())
