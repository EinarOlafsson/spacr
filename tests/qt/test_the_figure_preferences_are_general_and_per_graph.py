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


# --------------------------------------------------------------------------- #
#  4. "not black not white just transparent" -- the 2026-08-16 restatement
# --------------------------------------------------------------------------- #
#
# "figures should not have a background not black not white just transparent".
#
# The panel could not express that: `background` is a colour and every colour
# a colour button returns is opaque. It now has a Transparent box beside it,
# storing matplotlib's own spelling -- `to_rgba("none")` is (0, 0, 0, 0) --
# which `figure_style.rc_params` already forwards into `figure.facecolor` and
# `axes.facecolor`. So the vocabulary `spacr/figure_style.py` needed turned
# out to be one it already had, and that module (another territory) is
# untouched.
#
# MEASURED THROUGH THE WHOLE CHAIN, not off the widget:
#     panel ticked            -> {'background': 'none'}
#     rcParams                -> figure.facecolor none, axes.facecolor none
#     figures.style.user_overrides('volcano')
#                             -> {'figure.facecolor': 'none',
#                                 'axes.facecolor': 'none'}
#     the saved page's corner -> (255, 255, 255, 0)   <- alpha zero

def _transparent_box(panel):
    """The Transparent checkbox, found the way a user finds it."""
    from PySide6.QtWidgets import QCheckBox

    boxes = [b for b in panel.findChildren(QCheckBox)
             if b.text() == "Transparent"]
    assert len(boxes) == 1, [b.text() for b in panel.findChildren(QCheckBox)]
    return boxes[0]


def test_the_background_can_be_asked_to_be_transparent(panel):
    """Asserted through what a figure gets, not through the delta.

    The store keeps deltas from GENERAL_DEFAULTS, and the shipped default
    is transparent now -- so ticking transparent stores NOTHING, which is
    correct and is the opposite of what this used to assert. The panel
    used to ship white, where the same mechanism meant that choosing
    white deliberately was the one thing it could not express.
    """
    from spacr.figure_style import rc_params

    _transparent_box(panel).setChecked(True)
    general, per_graph = panel.values()
    assert per_graph == {}
    assert rc_params(resolve("volcano", general))["figure.facecolor"] == "none"


def test_transparent_reaches_matplotlib(panel):
    """Through `resolve` and `rc_params`, which is the path a drawn figure
    takes -- not through the widget's own getter."""
    from spacr.figure_style import rc_params

    _transparent_box(panel).setChecked(True)
    general, per_graph = panel.values()
    params = rc_params(resolve("volcano", general, per_graph))
    assert params["figure.facecolor"] == "none"
    assert params["axes.facecolor"] == "none"


def test_a_transparent_page_really_has_no_ground(panel, tmp_path):
    """THE PIXELS. A rcParam that is set and a page that is transparent are
    two different claims, and only the second one is the request."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rc_context

    from spacr.figure_style import rc_params

    _transparent_box(panel).setChecked(True)
    general, per_graph = panel.values()
    path = tmp_path / "page.png"
    with rc_context(rc_params(resolve(None, general, per_graph))):
        figure, axis = plt.subplots(figsize=(2, 2))
        axis.plot([0, 1], [0, 1])
        figure.savefig(path)
        plt.close(figure)
    from PIL import Image

    corner = np.asarray(Image.open(path).convert("RGBA"))[0, 0]
    assert int(corner[3]) == 0, tuple(int(v) for v in corner)


def test_unticking_puts_the_colour_back(panel):
    """GREYED, NOT REMOVED. The colour the user had is still there, so the
    way back is one click and stores nothing."""
    box = _transparent_box(panel)
    box.setChecked(True)
    box.setChecked(False)
    assert panel.values() == ({}, {})


def test_the_colour_button_is_greyed_while_transparent(panel):
    from PySide6.QtWidgets import QPushButton

    box = _transparent_box(panel)
    # THE BUTTON IS FOUND BY ITS ROW, not by matching its caption against
    # the default -- the default is 'none' now and no colour button is
    # captioned that.
    button = [b for b in panel.findChildren(QPushButton)
              if b.text().startswith("#")]
    assert button, [b.text() for b in panel.findChildren(QPushButton)]
    box.setChecked(True)
    assert not button[0].isEnabled()
    box.setChecked(False)
    assert button[0].isEnabled()


def test_a_stored_transparent_comes_back_ticked(qtbot):
    """A setting that could be saved and not restored would be worse than
    one that could not be saved."""
    widget = FigureStylePreferences({"background": "none"})
    qtbot.addWidget(widget)
    assert _transparent_box(widget).isChecked()
    # Nothing is stored, because transparent is what ships -- and the
    # figure is transparent either way, which is the thing that matters.
    general, per_graph = widget.values()
    assert per_graph == {}
    assert resolve("volcano", general)["background"] == "none"


def test_reset_untickss_transparent(qtbot):
    widget = FigureStylePreferences({"background": "#FFFFFF"})
    qtbot.addWidget(widget)
    assert not _transparent_box(widget).isChecked()
    widget.reset()
    # Reset goes back to the shipped default, which IS transparent, so the
    # box comes back TICKED. It was seeded white here for that reason: a
    # reset from the default to the default proves nothing.
    assert _transparent_box(widget).isChecked()
    assert widget.values() == ({}, {})


def test_only_the_background_offers_transparency(panel):
    """Invisible text is not a style, and `foreground` is a colour button."""
    from spacr.qt.widgets.figure_settings import TRANSPARENT_CAPABLE

    assert TRANSPARENT_CAPABLE == ("background",)
    assert "foreground" not in TRANSPARENT_CAPABLE
