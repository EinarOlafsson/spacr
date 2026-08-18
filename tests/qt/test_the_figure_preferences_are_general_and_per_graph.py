"""Instruction 118: figure preferences, general AND per graph type.

    "in the general app preferences in the figure tab theere should be general
     graph settings and specialized settings for al the possible different
     sets of graphs"

The MODEL existed before this: `spacr.figure_style` holds GENERAL_DEFAULTS,
GRAPH_DEFAULTS and `resolve`, and `spacr.figures.style.user_overrides` lays a
user's deltas over the publication house style. What did not exist was any way
to SET them. The Figures tab held format, DPI, cache size and the dynamic
switch -- everything about the FILE and nothing about how a plot looks, which
is what "the graphs look pretty ugly" means in practice.

Three things are asserted here, and the third is the acceptance criterion the
instruction itself names:

1. the panel is built FROM `figure_style`'s own tables, so a key added there
   gains a control without this panel being edited;
2. the store keeps DELTAS -- a user who never touched a setting stores
   nothing, which is what keeps the house style intact for them;
3. "a volcano-specific point size does not alter the plate heatmaps".
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.figure_style import (GENERAL_DEFAULTS, GRAPH_DEFAULTS, GRAPH_KINDS,
                                resolve)
from spacr.qt.widgets.figure_settings import (FigureStylePreferences,
                                              style_choices_for)


@pytest.fixture
def panel(qtbot):
    widget = FigureStylePreferences()
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  1. built from the model, not from a list
# --------------------------------------------------------------------------- #

def test_every_general_setting_has_a_control(panel):
    """A key added to GENERAL_DEFAULTS gains a control without this panel
    being touched -- the same rule `add_style_entries` follows for a style
    dataclass, and the reason both are worth asserting."""
    assert set(panel._general_controls) == set(GENERAL_DEFAULTS)


def test_every_graph_kind_has_a_page_and_every_setting_a_control(panel):
    assert set(panel._kind_controls) == set(GRAPH_KINDS)
    for kind in GRAPH_KINDS:
        assert set(panel._kind_controls[kind]) == set(GRAPH_DEFAULTS[kind]), \
            kind


def test_a_setting_that_is_a_closed_set_is_a_combo_not_a_text_box(panel):
    """A free-text box for `spines` is a setting the user can spell wrong,
    and `resolve` would hand matplotlib a preset that does not exist."""
    from PySide6.QtWidgets import QComboBox

    assert style_choices_for("spines") == tuple(
        __import__("spacr.figure_style", fromlist=["x"]).SPINE_PRESETS)
    combos = [w for w in panel.findChildren(QComboBox)]
    texts = {tuple(c.itemData(i) for i in range(c.count())) for c in combos}
    assert any("left_bottom" in t for t in texts), texts


def test_a_colour_setting_gets_a_colour_button(panel):
    from PySide6.QtWidgets import QPushButton

    buttons = [b.text() for b in panel.findChildren(QPushButton)]
    assert GENERAL_DEFAULTS["grid_colour"].lower() in \
        [text.lower() for text in buttons], buttons


# --------------------------------------------------------------------------- #
#  2. the store keeps deltas
# --------------------------------------------------------------------------- #

def test_a_panel_nobody_touched_stores_nothing(panel):
    """The contract that keeps `figures.style.user_overrides` returning {},
    and with it the published house style, for every user who has never
    opened this tab. Writing every control back would freeze today's defaults
    into everybody's settings."""
    general, per_graph = panel.values()
    assert general == {}
    assert per_graph == {}


def test_a_changed_general_setting_is_the_only_thing_stored(panel):
    panel._general_controls["font_size"][1](18.0)
    general, per_graph = panel.values()

    assert general == {"font_size": 18.0}
    assert per_graph == {}


def test_a_float_that_a_spin_box_cannot_hold_exactly_is_not_a_change(panel):
    """`grid_width` is 0.6 and a two-decimal spin box round-trips it as
    0.6000000000000001. A panel comparing exactly would mark every user as
    having overridden a setting they never touched."""
    panel._general_controls["grid_width"][1](
        GENERAL_DEFAULTS["grid_width"])
    general, _per_graph = panel.values()
    assert "grid_width" not in general


def test_a_stored_setting_comes_back_into_the_control(qtbot):
    widget = FigureStylePreferences({"palette": "muted"},
                                    {"volcano": {"marker_size": 40.0}})
    qtbot.addWidget(widget)
    general, per_graph = widget.values()

    assert general["palette"] == "muted"
    assert per_graph["volcano"]["marker_size"] == 40.0


def test_reset_puts_every_control_back(qtbot):
    widget = FigureStylePreferences({"palette": "muted", "font_size": 22.0,
                                     "grid_colour": "#ff0000"},
                                    {"volcano": {"marker_size": 40.0}})
    qtbot.addWidget(widget)
    widget.reset()
    assert widget.values() == ({}, {})


def test_reset_repaints_the_colour_button(qtbot):
    """A reset the user cannot see did not happen. The button paints from its
    own state, so writing the value alone would leave the old swatch."""
    from PySide6.QtWidgets import QPushButton

    widget = FigureStylePreferences({"grid_colour": "#ff0000"})
    qtbot.addWidget(widget)
    assert any(b.text().lower() == "#ff0000"
               for b in widget.findChildren(QPushButton))
    widget.reset()
    assert any(b.text().lower() == GENERAL_DEFAULTS["grid_colour"].lower()
               for b in widget.findChildren(QPushButton))


def test_a_stored_value_the_package_no_longer_offers_is_kept(qtbot):
    """Silently snapping a user's setting to the first entry, in the dialog
    that is meant to show them their settings, is the worst place to do it."""
    widget = FigureStylePreferences({"palette": "a_palette_from_2027"})
    qtbot.addWidget(widget)
    general, _ = widget.values()
    assert general["palette"] == "a_palette_from_2027"


# --------------------------------------------------------------------------- #
#  3. the acceptance criterion the instruction names
# --------------------------------------------------------------------------- #

def test_a_volcano_point_size_does_not_alter_the_plate_heatmaps(panel):
    """The instruction's own "HOW TO KNOW IT IS DONE", driven end to end
    through `resolve`."""
    panel._kind_controls["volcano"]["marker_size"][1](44.0)
    general, per_graph = panel.values()

    volcano = resolve("volcano", general, per_graph)
    heatmap = resolve("plate_heatmap", general, per_graph)

    assert volcano["marker_size"] == 44.0
    assert heatmap["marker_size"] == GENERAL_DEFAULTS["marker_size"]


def test_changing_the_general_palette_reaches_every_graph_kind(panel):
    """The other half of it: "Changing the palette in Preferences changes the
    next run's figures without any per-figure work"."""
    panel._general_controls["palette"][1]("muted")
    general, per_graph = panel.values()

    for kind in GRAPH_KINDS:
        assert resolve(kind, general, per_graph)["palette"] == "muted"


def test_a_per_graph_setting_beats_the_general_one(panel):
    panel._general_controls["marker_size"][1](30.0)
    panel._kind_controls["histogram"]["log_y"][1](True)
    panel._kind_controls["scatter"]["marker_size"][1](5.0)
    general, per_graph = panel.values()

    assert resolve("scatter", general, per_graph)["marker_size"] == 5.0
    assert resolve("residuals", general, per_graph)["marker_size"] == \
        GRAPH_DEFAULTS["residuals"]["marker_size"]
    assert resolve("histogram", general, per_graph)["log_y"] is True


# --------------------------------------------------------------------------- #
#  And it is actually on the Figures tab
# --------------------------------------------------------------------------- #

def test_the_panel_is_on_the_preferences_figures_tab(qtbot, monkeypatch,
                                                     tmp_path):
    """THE WIRING. A panel nobody can reach is a panel that does not exist,
    and this repo has shipped a described-but-unreachable control before."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))

    dialog = preferences_module.PreferencesDialog()
    qtbot.addWidget(dialog)
    panels = dialog.findChildren(FigureStylePreferences)
    assert len(panels) == 1, "exactly one graph-style panel"

    tabs = panels[0]
    parent_names = []
    widget = tabs
    while widget is not None:
        parent_names.append(widget.objectName())
        widget = widget.parentWidget()
    assert "PreferencesTabFigures" in parent_names, parent_names


def test_saving_the_dialog_writes_the_style_through(qtbot, monkeypatch,
                                                    tmp_path):
    """Driven through the dialog's own Save, not through the setter it
    calls: the point of 118 is that the value reaches the store from the
    tab."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QDialogButtonBox
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    monkeypatch.setattr(preferences_module, "apply_preferences_to_app",
                        lambda *a, **k: None)

    dialog = preferences_module.PreferencesDialog()
    qtbot.addWidget(dialog)
    panel = dialog.findChildren(FigureStylePreferences)[0]
    panel._general_controls["palette"][1]("muted")
    panel._kind_controls["volcano"]["marker_size"][1](44.0)

    box = dialog.findChildren(QDialogButtonBox)[0]
    box.button(QDialogButtonBox.Save).click()

    assert preferences_module.get_figure_style()["palette"] == "muted"
    assert preferences_module.get_figure_style_per_graph()["volcano"][
        "marker_size"] == 44.0
