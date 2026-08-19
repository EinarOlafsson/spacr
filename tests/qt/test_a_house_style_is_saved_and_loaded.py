"""Instruction 108 point 5, for the figures that have no style dataclass.

    "Save style / Load style as the Volcano Explorer already has, plus a
     per-project default so a lab's house style is applied to every figure of
     that type without re-setting it each time."

restated by the maintainer on 2026-08-16 as "each figure should be editable
AND SAVABLE".

WHAT WAS ALREADY TRUE on 2026-08-18: `fast_plots.save_style`, `load_style`,
`apply_default_style` and `add_style_file_entries` all existed, and they take
a style DATACLASS. The only style dataclass in the package is `VolcanoStyle`,
`offer_style` had NO CALLER anywhere, and every matplotlib figure -- which is
nearly all of them -- had no style object at all. So the savable half was
built and nothing a user could click reached it.

WHAT THIS ADDS is the same two menu entries over instruction 118's
vocabulary, which is the style every matplotlib figure already resolves
through. It invents no keys: the file holds the DELTAS the Figures tab
stores, so loading one is indistinguishable afterwards from having set every
control by hand, and `figure_style.resolve` / `figures.style.user_overrides`
pick it up with nothing else wired.

AND IT CARRIES NO COLOURS BEYOND THE ONES 118 ALREADY STORES. The obvious
extra -- capture what this figure looks like right now -- would sample the
ink the THEME resolved, and saving that is instruction 152 section A word for
word: a resolved default written back, invisible the first time the user
switches theme.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.widgets.figure_settings import (GRAPH_STYLE_FILE_KIND,
                                              FigureStylePreferences,
                                              apply_graph_style,
                                              build_figure_context_menu,
                                              graph_style_as_dict,
                                              load_graph_style,
                                              save_graph_style)


@pytest.fixture
def prefs(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    return preferences_module


@pytest.fixture
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [1, 0], label="x")
    ax.legend()
    yield fig
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  it is reachable
# --------------------------------------------------------------------------- #

def test_the_right_click_menu_offers_both(qtbot, figure):
    """The gap this closes: the helpers existed and no menu had them."""
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = build_figure_context_menu(parent, figure)
    labels = [action.text() for action in menu.actions()]
    assert "Save graph style…" in labels, labels
    assert "Load graph style…" in labels, labels


def test_the_preferences_panel_offers_both(qtbot):
    from PySide6.QtWidgets import QPushButton

    panel = FigureStylePreferences()
    qtbot.addWidget(panel)
    labels = [b.text() for b in panel.findChildren(QPushButton)]
    assert "Save style…" in labels, labels
    assert "Load style…" in labels, labels


def test_a_figure_that_cannot_be_restyled_offers_nothing(qtbot):
    """The evicted-figure menu stays a single disabled sentence."""
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = build_figure_context_menu(parent, None)
    assert [a.text() for a in menu.actions()] == \
        ["This figure can no longer be restyled"]


# --------------------------------------------------------------------------- #
#  the file
# --------------------------------------------------------------------------- #

def test_the_file_says_what_it_is(tmp_path):
    path = str(tmp_path / "house.json")
    save_graph_style(path, {"font_size": 17.0}, {"volcano": {"marker_size": 44.0}})
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["spacr_style_kind"] == GRAPH_STYLE_FILE_KIND


def test_a_round_trip_reproduces_the_style(tmp_path):
    general = {"font_size": 17.0, "palette": "muted", "grid": False}
    per_graph = {"volcano": {"marker_size": 44.0}}
    path = str(tmp_path / "house.json")
    save_graph_style(path, general, per_graph)
    assert load_graph_style(path) == (general, per_graph)


def test_a_file_that_is_not_a_style_is_refused(tmp_path):
    """REFUSED, NOT PARTIALLY APPLIED. A dict whose keys happen to match
    would set a few settings and leave the rest, which looks like a corrupted
    house style rather than like a mistake."""
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({"general": {"font_size": 17.0}}),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_graph_style(str(path))


def test_a_key_this_build_does_not_know_is_kept(tmp_path):
    """Forwards-compatible. Dropping it would make opening and re-saving a
    colleague's house style silently lose the parts this build is behind on
    -- and the panel already shows such a value as "(not offered)"."""
    path = tmp_path / "house.json"
    path.write_text(json.dumps({
        "spacr_style_kind": GRAPH_STYLE_FILE_KIND,
        "general": {"font_size": 17.0, "a_setting_from_2027": 3},
        "per_graph": {},
    }), encoding="utf-8")
    general, _per_graph = load_graph_style(str(path))
    assert general["a_setting_from_2027"] == 3


def test_saving_reads_the_store_when_it_is_not_given(prefs, tmp_path):
    prefs.set_figure_style({"font_size": 17.0})
    prefs.set_figure_style_per_graph({"volcano": {"marker_size": 44.0}})
    assert graph_style_as_dict()["general"] == {"font_size": 17.0}


# --------------------------------------------------------------------------- #
#  it becomes this project's default
# --------------------------------------------------------------------------- #

def test_loading_makes_it_the_project_default(prefs, tmp_path):
    path = str(tmp_path / "house.json")
    save_graph_style(path, {"font_size": 17.0}, {"volcano": {"marker_size": 44.0}})
    general, per_graph = load_graph_style(path)
    apply_graph_style(general, per_graph)
    assert prefs.get_figure_style() == {"font_size": 17.0}
    assert prefs.get_figure_style_per_graph() == {"volcano": {"marker_size": 44.0}}


def test_the_loaded_style_reaches_the_figures(prefs, tmp_path):
    """THE ACCEPTANCE CRITERION: "applied to every figure of that type
    without re-setting it each time". Driven through `resolve`, which is what
    a plotting function asks."""
    from spacr.figure_style import resolve

    path = str(tmp_path / "house.json")
    save_graph_style(path, {"font_size": 17.0}, {"volcano": {"marker_size": 44.0}})
    apply_graph_style(*load_graph_style(path))
    style = resolve("volcano", prefs.get_figure_style(),
                    prefs.get_figure_style_per_graph())
    assert style["font_size"] == 17.0
    assert style["marker_size"] == 44.0
    # …and the per-graph half stays per-graph.
    other = resolve("plate_heatmap", prefs.get_figure_style(),
                    prefs.get_figure_style_per_graph())
    assert other["marker_size"] != 44.0


def test_a_loaded_style_reaches_the_house_style_too(prefs, tmp_path):
    """`figures.style.user_overrides` is the other reader, and it is the one
    the publication figures go through."""
    from spacr.figures.style import user_overrides

    path = str(tmp_path / "house.json")
    save_graph_style(path, {"font_size": 17.0}, {})
    apply_graph_style(*load_graph_style(path))
    assert user_overrides()["font.size"] == 17.0


# --------------------------------------------------------------------------- #
#  the panel round-trip
# --------------------------------------------------------------------------- #

def test_the_panel_takes_a_loaded_style(qtbot):
    panel = FigureStylePreferences()
    qtbot.addWidget(panel)
    panel.apply_values({"font_size": 17.0}, {"volcano": {"marker_size": 44.0}})
    assert panel.values() == ({"font_size": 17.0},
                              {"volcano": {"marker_size": 44.0}})


def test_a_setting_the_file_omits_goes_back_to_the_default(qtbot):
    """Loading a house style that left half of somebody else's settings
    standing would not be that house style."""
    panel = FigureStylePreferences({"font_size": 17.0, "palette": "muted"})
    qtbot.addWidget(panel)
    panel.apply_values({"palette": "muted"}, {})
    assert panel.values() == ({"palette": "muted"}, {})


def test_the_panel_saves_what_is_on_screen(qtbot, tmp_path):
    """Not what is stored: a file that disagreed with the controls in front
    of the user would be the worse of the two answers."""
    panel = FigureStylePreferences()
    qtbot.addWidget(panel)
    panel.apply_values({"font_size": 17.0}, {})
    path = str(tmp_path / "house.json")
    general, per_graph = panel.values()
    save_graph_style(path, general, per_graph)
    assert load_graph_style(path)[0] == {"font_size": 17.0}


# --------------------------------------------------------------------------- #
#  driven through the action, not through the helper it calls
# --------------------------------------------------------------------------- #
#
# "GREEN TESTS DO NOT MEAN THE FEATURE WORKS" (HANDOFF 0b). The helpers above
# were all green before any menu reached them, which is the whole reason this
# section exists: these two press the menu entry.

def test_pressing_load_on_the_menu_changes_the_project_default(
        qtbot, figure, prefs, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QWidget

    path = str(tmp_path / "house.json")
    save_graph_style(path, {"font_size": 17.0}, {"volcano": {"marker_size": 44.0}})

    parent = QWidget()
    qtbot.addWidget(parent)
    redrawn = []
    menu = build_figure_context_menu(
        parent, figure, on_change=lambda **_kw: redrawn.append(True))
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (path, "")))
    action = next(a for a in menu.actions() if a.text() == "Load graph style…")
    action.trigger()

    assert prefs.get_figure_style() == {"font_size": 17.0}
    assert prefs.get_figure_style_per_graph() == \
        {"volcano": {"marker_size": 44.0}}
    assert redrawn, "the figure was never asked to redraw"


def test_pressing_save_on_the_menu_writes_the_store(qtbot, figure, prefs,
                                                    tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QWidget

    prefs.set_figure_style({"font_size": 17.0})
    path = str(tmp_path / "written.json")
    parent = QWidget()
    qtbot.addWidget(parent)
    menu = build_figure_context_menu(parent, figure)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (path, "")))
    next(a for a in menu.actions()
         if a.text() == "Save graph style…").trigger()
    assert load_graph_style(path)[0] == {"font_size": 17.0}


def test_a_refused_file_says_so_and_changes_nothing(qtbot, figure, prefs,
                                                    tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps({"general": {"font_size": 99.0}}),
                   encoding="utf-8")
    parent = QWidget()
    qtbot.addWidget(parent)
    menu = build_figure_context_menu(parent, figure)
    warned = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(bad), "")))
    # A static modal runs its event loop in C++ and hangs a headless run —
    # tests/qt/conftest.py enforces that, so it is replaced rather than
    # allowed to open.
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warned.append(a)))
    next(a for a in menu.actions()
         if a.text() == "Load graph style…").trigger()
    assert warned, "a refused file said nothing"
    assert prefs.get_figure_style() == {}


def test_cancelling_the_file_dialog_changes_nothing(qtbot, figure, prefs,
                                                    monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = build_figure_context_menu(parent, figure)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    next(a for a in menu.actions()
         if a.text() == "Load graph style…").trigger()
    assert prefs.get_figure_style() == {}
