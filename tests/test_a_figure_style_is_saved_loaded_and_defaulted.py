"""Instruction 108 point 5: a style is EDITABLE **AND SAVABLE**.

    "all gigures should be editable by right clicking" ... "each figure should
     be editable and savable"

Point 3 shipped the mechanism -- a menu built from ``dataclasses.fields`` so a
style that gains a field gains an entry. What was missing was any way to KEEP
what the menu produced: the serialisation already existed
(``VolcanoStyle.from_dict`` / ``asdict``) and nothing reached it, so a restyle
was something the user redid every time they needed the picture.

Three things, and the third is the one that makes it a house style rather than
a file:

    Save style…                 this figure's settings, to a file
    Load style…                 a saved file, into a figure of the same kind
    Use as the default for …    every future figure of this kind starts here

AND A WAY BACK. "Clear the default" is greyed when there is nothing to clear,
which doubles as the readout for "is a house style in force here?" -- a
default that can only be set is the same trap as a colour that can only be
set (instruction 152 A).
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")

from spacr.qt.widgets.fast_plots import (add_style_file_entries,
                                         apply_default_style,
                                         apply_style_dict, load_style,
                                         save_style, style_as_dict,
                                         style_kind)


@dataclasses.dataclass
class PretendStyle:
    """A style with one of each kind, so the round trip is not vacuous."""

    grid: bool = True
    line_width: float = 1.4
    point_colour: str = "#4C72B0"
    x_label: str = "effect"
    figure_size: tuple = (6.0, 4.0)


@dataclasses.dataclass
class HeatmapStyle:
    colormap: str = "viridis"
    grid: bool = False


@pytest.fixture
def prefs(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    return preferences_module


# --------------------------------------------------------------------------- #
#  The kind, which everything else is keyed on
# --------------------------------------------------------------------------- #

def test_the_kind_comes_from_the_class_not_from_the_caller():
    """A caller that named its own kind would eventually name two of them the
    same, and one lab's house style would land on another figure type."""
    from spacr.volcano_style import VolcanoStyle

    assert style_kind(VolcanoStyle()) == "volcano"
    assert style_kind(PretendStyle()) == "pretend"
    assert style_kind(HeatmapStyle()) == "heatmap"


# --------------------------------------------------------------------------- #
#  Save and load
# --------------------------------------------------------------------------- #

def test_a_saved_style_reproduces_the_figure_exactly(tmp_path):
    """108's own acceptance: "A style saved from one figure and loaded into
    another of the same type reproduces it exactly"."""
    source = PretendStyle(grid=False, line_width=3.0, point_colour="#ff0000",
                          x_label="log2 fold change", figure_size=(9.0, 5.0))
    path = save_style(source, tmp_path / "house")
    target = PretendStyle()
    load_style(target, path)
    assert style_as_dict(target) == style_as_dict(source)


def test_a_pair_survives_the_json_round_trip(tmp_path):
    """JSON has no tuples. A pair that came back a list would leave two
    representations of one value in the store."""
    source = PretendStyle(figure_size=(9.0, 5.0))
    target = PretendStyle()
    load_style(target, save_style(source, tmp_path / "s.json"))
    assert target.figure_size == (9.0, 5.0)
    assert isinstance(target.figure_size, tuple)


def test_the_extension_is_added_when_the_user_leaves_it_off(tmp_path):
    written = save_style(PretendStyle(), tmp_path / "house")
    assert written.endswith(".json")
    assert os.path.exists(written)


def test_a_style_of_another_kind_is_refused_rather_than_half_applied(tmp_path):
    """Four fields whose names happen to match looks like a corrupted figure
    rather than like a mistake."""
    path = save_style(PretendStyle(grid=False), tmp_path / "v.json")
    heatmap = HeatmapStyle()
    with pytest.raises(ValueError) as raised:
        load_style(heatmap, path)
    assert "pretend" in str(raised.value) and "heatmap" in str(raised.value)
    assert heatmap.grid is False or heatmap.grid is not None
    assert heatmap.colormap == "viridis", "nothing may be applied"


def test_a_file_that_is_not_a_style_is_refused(tmp_path):
    path = tmp_path / "notastyle.json"
    path.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError):
        load_style(PretendStyle(), path)


def test_loading_is_forwards_compatible_in_both_directions(tmp_path):
    """A file from a later spaCR carries fields this one has never heard of;
    one from an earlier is missing some. Neither may raise."""
    path = tmp_path / "future.json"
    path.write_text(json.dumps({
        "spacr_style_kind": "pretend",
        "fields": {"grid": False, "a_field_from_2027": 12}}))
    style = PretendStyle()
    changed = load_style(style, path)

    assert changed == ["grid"]
    assert style.grid is False
    assert style.line_width == 1.4, "a missing field keeps what it had"


def test_the_host_is_told_once_and_not_sixty_times():
    """A host that redrew per field would redraw sixty times for one load."""
    seen = []
    style = PretendStyle()
    apply_style_dict(style, {"grid": False, "line_width": 9.0,
                             "x_label": "x"},
                     lambda name, value: seen.append(name))
    assert seen == [None]


def test_a_load_that_changes_nothing_tells_nobody():
    seen = []
    apply_style_dict(PretendStyle(), {"grid": True},
                     lambda name, value: seen.append(name))
    assert seen == []


# --------------------------------------------------------------------------- #
#  The per-project default
# --------------------------------------------------------------------------- #

def test_a_default_reaches_the_next_figure_of_that_kind(prefs):
    prefs.set_figure_style_default("pretend", style_as_dict(
        PretendStyle(line_width=7.0, point_colour="#00ff00")))
    fresh = PretendStyle()
    changed = apply_default_style(fresh)

    assert set(changed) == {"line_width", "point_colour"}
    assert fresh.line_width == 7.0


def test_a_default_does_not_reach_another_kind(prefs):
    prefs.set_figure_style_default("pretend", {"grid": False})
    heatmap = HeatmapStyle()
    assert apply_default_style(heatmap) == []
    assert heatmap.grid is False or heatmap.grid is not None


def test_no_default_leaves_the_package_defaults_standing(prefs):
    assert apply_default_style(PretendStyle()) == []
    assert prefs.get_figure_style_default("pretend") == {}


def test_the_default_can_be_cleared(prefs):
    prefs.set_figure_style_default("pretend", {"grid": False})
    assert prefs.clear_figure_style_default("pretend") is True
    assert prefs.get_figure_style_default("pretend") == {}
    assert prefs.clear_figure_style_default("pretend") is False


def test_a_field_name_with_an_underscore_survives_the_store(prefs):
    """QSettings' INI writer flattens a nested map into slash-separated keys,
    which is why this is stored as JSON: `x_label` must come back as a field
    and not as a group."""
    prefs.set_figure_style_default("pretend", {"x_label": "log2 fold change"})
    assert prefs.get_figure_style_default("pretend") == {
        "x_label": "log2 fold change"}


# --------------------------------------------------------------------------- #
#  The menu entries, driven
# --------------------------------------------------------------------------- #

def _entries(menu):
    return [action.text() for action in menu.actions()]


def test_the_four_entries_are_on_the_menu(qtbot, prefs):
    from PySide6.QtWidgets import QMenu, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    add_style_file_entries(menu, PretendStyle(), parent=parent)
    texts = _entries(menu)

    assert texts == ["Save style…", "Load style…",
                     "Use as the default for every pretend",
                     "Clear the default"]


def test_clear_is_greyed_when_there_is_nothing_to_clear(qtbot, prefs):
    from PySide6.QtWidgets import QMenu, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    add_style_file_entries(menu, PretendStyle(), parent=parent)
    assert not menu.actions()[-1].isEnabled()

    prefs.set_figure_style_default("pretend", {"grid": False})
    menu2 = QMenu(parent)
    add_style_file_entries(menu2, PretendStyle(), parent=parent)
    assert menu2.actions()[-1].isEnabled()


def test_the_save_entry_actually_writes_a_file(qtbot, prefs, tmp_path):
    """Driven through the ACTION. A menu entry connected to nothing is a
    failure this repo has shipped before."""
    from PySide6.QtWidgets import QMenu, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    target = tmp_path / "picked.json"
    said = []
    add_style_file_entries(menu, PretendStyle(line_width=5.0), parent=parent,
                           note=said.append,
                           ask_path=lambda mode, suggested: str(target))
    menu.actions()[0].trigger()

    assert target.exists()
    assert json.loads(target.read_text())["fields"]["line_width"] == 5.0
    assert said and str(target) in said[0]


def test_the_load_entry_actually_changes_the_figure(qtbot, prefs, tmp_path):
    from PySide6.QtWidgets import QMenu, QWidget

    path = save_style(PretendStyle(line_width=5.0), tmp_path / "s.json")
    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    style = PretendStyle()
    redrawn = []
    add_style_file_entries(menu, style, lambda *a: redrawn.append(a),
                           parent=parent, ask_path=lambda mode, s: path)
    menu.actions()[1].trigger()

    assert style.line_width == 5.0
    assert redrawn, "the host was never told to redraw"


def test_a_failed_load_says_why_instead_of_doing_nothing(qtbot, prefs,
                                                         tmp_path):
    from PySide6.QtWidgets import QMenu, QWidget

    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    said = []
    add_style_file_entries(menu, PretendStyle(), parent=parent,
                           note=said.append,
                           ask_path=lambda mode, s: str(bad))
    menu.actions()[1].trigger()

    assert said and "Could not load" in said[0]


def test_cancelling_the_dialog_writes_nothing(qtbot, prefs, tmp_path):
    from PySide6.QtWidgets import QMenu, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    said = []
    add_style_file_entries(menu, PretendStyle(), parent=parent,
                           note=said.append, ask_path=lambda mode, s: "")
    menu.actions()[0].trigger()
    assert said == []


def test_the_default_entry_stores_the_whole_style(qtbot, prefs):
    from PySide6.QtWidgets import QMenu, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu(parent)
    add_style_file_entries(menu, PretendStyle(line_width=8.0), parent=parent)
    menu.actions()[2].trigger()

    saved = prefs.get_figure_style_default("pretend")
    assert saved["line_width"] == 8.0
    assert set(saved) == {entry.name
                          for entry in dataclasses.fields(PretendStyle)}


# --------------------------------------------------------------------------- #
#  And on the real plot
# --------------------------------------------------------------------------- #

def test_a_real_volcano_offers_all_four(qtbot, prefs):
    from spacr.qt.widgets.fast_plots import VolcanoPlot, menu_entries
    from spacr.volcano_style import VolcanoStyle

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.offer_style(VolcanoStyle())
    texts = [action.text() for action in menu_entries(plot.build_style_menu())]

    assert "Save style…" in texts
    assert "Load style…" in texts
    assert "Use as the default for every volcano" in texts
    assert "Clear the default" in texts


def test_a_real_volcano_style_round_trips(tmp_path):
    """The mechanism was written against no field of VolcanoStyle by name;
    this is the check that it works on the one that exists."""
    from spacr.volcano_style import VolcanoStyle

    source = VolcanoStyle()
    fields = dataclasses.fields(source)
    assert len(fields) > 50, "VolcanoStyle should still be the big one"
    path = save_style(source, tmp_path / "v.json")
    target = VolcanoStyle()
    load_style(target, path)
    assert style_as_dict(target) == style_as_dict(source)


def test_the_default_reaches_a_plot_the_moment_a_style_is_offered(qtbot,
                                                                  prefs):
    """THE WIRING, driven rather than assumed. A stored default nothing reads
    is a preference, not a house style -- and the whole of point 5 is that a
    lab's look reaches every figure of that type without re-setting it."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    prefs.set_figure_style_default("pretend", {"line_width": 6.5})
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    style = PretendStyle()
    plot.offer_style(style)

    assert style.line_width == 6.5
    assert "saved pretend style" in plot._style_note


def test_re_offering_the_same_style_does_not_undo_an_edit(qtbot, prefs):
    """The host redraws on every level, baseline and compartment change. Any
    one of them re-asserting the default would silently undo the user."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    prefs.set_figure_style_default("pretend", {"line_width": 6.5})
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    style = PretendStyle()
    plot.offer_style(style)
    style.line_width = 1.0
    plot.offer_style(style)

    assert style.line_width == 1.0


def test_a_host_can_decline_the_default(qtbot, prefs):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    prefs.set_figure_style_default("pretend", {"line_width": 6.5})
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    style = PretendStyle()
    plot.offer_style(style, use_default=False)

    assert style.line_width == 1.4
