"""The volcano explorer's edges: a panel that is not the one it expects.

Pins the paths the explorer takes when something it reads is not the shape
it was written for -- a renderer complaint that names a setting without
spelling it ``name=``, an "automatic" limit that cannot be expressed as a
number, a results table with an unnamed column, a numeric setting that is
``None``, a split that cannot be sized, and a writer that reports it wrote
nothing. Every one of them has to leave the reader with a figure and a
sentence rather than an exception.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QWidget

from spacr.qt.widgets.volcano_explorer import (
    _PROBLEM_INK, VolcanoExplorer, _setting_named_in)
from spacr.volcano_style import VolcanoStyle


def _results(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "guide": [f"G{i:03d}" for i in range(n)],
        "gene": [f"TGGT1_{i // 3:06d}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(size=n),
        "adjusted_p_value": rng.random(n) * 0.5 + 1e-6,
        "p_value": rng.random(n),
    })


@pytest.fixture
def explorer(qapp, qtbot):
    widget = VolcanoExplorer(_results())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def localised(qapp, qtbot):
    """An explorer whose genes the bundled localisation table recognises."""
    from spacr.localisation import table

    lookup = table()
    if not lookup:
        pytest.skip("no bundled localisation table to colour by")
    genes = []
    for compartment in ("dense granules", "rhoptries 1"):
        genes += [f"TGGT1_{key}" for key, place in lookup.items()
                  if place == compartment][:10]
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "guide": [f"{gene}_1" for gene in genes],
        "gene": genes,
        "standardized_marginal_effect": rng.normal(size=len(genes)),
        "adjusted_p_value": rng.random(len(genes)) * 0.5 + 1e-6,
    })
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------
# which setting a complaint blames
# --------------------------------------------------------------------------

def test_a_complaint_that_only_mentions_a_setting_still_names_it():
    """The renderer usually writes ``x_column='foo' is not a column``, and
    the ``name=`` form is preferred -- but a sentence that merely mentions a
    setting has to point at that control too, or the reader is shown a red
    line under the plot with nothing turning red beside it."""
    assert _setting_named_in("x_column='nope' is not a column") == "x_column"
    assert _setting_named_in("the colormap could not be resolved") == "colormap"


def test_a_complaint_about_nothing_the_panel_holds_blames_no_control():
    """The explanation is still printed; it just has no control to redden.
    Blaming the nearest field instead would turn a setting red that the
    reader had not touched."""
    assert _setting_named_in("cannot open the file you asked for") == ""
    # ... and the same sentence with one setting named does blame it, so the
    # empty answer above is a real absence rather than a broken search.
    assert _setting_named_in("cannot open the font_family you asked for") == \
        "font_family"


# --------------------------------------------------------------------------
# a limit that is "automatic"
# --------------------------------------------------------------------------

def test_taking_the_tick_off_an_axis_limit_writes_the_numbers_into_the_style(
        explorer):
    """The tick IS the third state: with it on the limit is ``None``, and
    taking it off has to hand the spin boxes' numbers to the style -- and
    redraw -- or the panel and the plot disagree about the axis."""
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox

    control = explorer._controls["x_lim"]
    assert explorer.style().x_lim is None
    spins = control.findChildren(QDoubleSpinBox)
    spins[0].setValue(-3.0)
    spins[1].setValue(3.0)

    told = []
    explorer.style_changed.connect(lambda: told.append(1))
    # The tick is the widget's only public way to leave "automatic".
    auto = control.findChild(QCheckBox)
    auto.setChecked(False)
    assert told, "the panel changed and the plot was not told"
    assert explorer.style().x_lim == (-3.0, 3.0)
    assert all(spin.isEnabled() for spin in spins)

    auto.setChecked(True)
    assert explorer.style().x_lim is None
    assert not any(spin.isEnabled() for spin in spins)


def test_a_limit_that_is_not_a_number_reads_as_automatic(explorer):
    """A style file can carry anything. A value the control cannot express
    has to read as "automatic" rather than as a number nobody typed."""
    style = dataclasses.replace(explorer.style(), effect_threshold=2.5)
    explorer.set_style(style)
    assert explorer._controls["effect_threshold"].value() == 2.5

    explorer.set_style(dataclasses.replace(style, effect_threshold="rubbish"))
    assert explorer._controls["effect_threshold"].value() is None


# --------------------------------------------------------------------------
# ticking a compartment
# --------------------------------------------------------------------------

def test_ticking_a_compartment_in_the_panel_reaches_the_style(localised):
    """Filling the list is not a change to report -- the panel would tell the
    plot to redraw while it was being built -- but a tick by hand is."""
    compartments = localised._controls["localizations"]
    assert compartments.options() == ["dense granules", "rhoptries 1"]

    seen = []
    compartments.changed.connect(lambda: seen.append(1))
    compartments.setValues(())            # refilling reports nothing
    assert seen == []

    compartments.item(0).setCheckState(Qt.Checked)
    assert seen == [1]
    assert localised.style().localizations == ("dense granules",)


# --------------------------------------------------------------------------
# what the right-click menu is allowed to offer
# --------------------------------------------------------------------------

def test_a_column_with_no_name_is_offered_by_what_it_says(qapp, qtbot):
    """A row that carries no data falls back to its own text, so a column
    the frame did not name is still offerable -- while the "— none —" row,
    whose data IS ``None``, keeps it, because that is how a colour-by column
    is taken back off."""
    frame = _results()
    frame[None] = np.arange(len(frame), dtype=float)
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)

    choices = widget._style_choices()
    assert "None" in choices["x_column"]
    assert None in choices["color_by"]


def test_a_menu_offers_no_choice_for_a_picker_that_has_none(qapp, qtbot):
    """The column pickers of an empty screen hold nothing. Offering an empty
    list on the menu would put a setting there with no way to set it."""
    widget = VolcanoExplorer(pd.DataFrame())
    qtbot.addWidget(widget)

    choices = widget._style_choices()
    assert "x_column" not in choices
    # The pickers that do not depend on the results are still offered, so the
    # absence above is this picker being empty and not the menu being empty.
    assert choices["marker"][0] == "o"


# --------------------------------------------------------------------------
# a panel that is not carrying every control
# --------------------------------------------------------------------------

def test_the_column_menus_refill_even_with_no_compartment_list(explorer):
    """`_controls` is looked up by name throughout, and every lookup is
    guarded -- a screen that carries a subset of the panel must still get its
    column pickers refilled. Reached through the registry because there is no
    public way to build a partial panel."""
    explorer._controls.pop("localizations")
    explorer.set_results(_results(12))
    assert explorer._controls["x_column"].count() > 0


def test_an_unknown_control_is_stepped_over_rather_than_stopping_the_panel(
        explorer):
    """Writing the style into the panel and reading it back both walk every
    control by type. A control of a type neither walk knows has to be passed
    over, or one unrecognised widget silently costs every control after it
    its value."""
    probe = QWidget(explorer)
    explorer._controls["unknown_probe"] = probe
    # Moved to the end so a known control sits AFTER the unknown one, which
    # is what proves the walk carried on rather than stopped.
    explorer._controls["title"] = explorer._controls.pop("title")

    explorer.set_style(dataclasses.replace(explorer.style(), title="Kept"))
    assert explorer._controls["title"].text() == "Kept"

    explorer._controls["title"].setText("Typed by hand")
    explorer._pull_style_from_controls()
    assert explorer.style().title == "Typed by hand"


def test_a_numeric_setting_of_none_leaves_the_spin_box_where_it_was(explorer):
    """A spin box has no way to show ``None``. Handing it one would snap the
    control to its minimum and then read that minimum back as the user's
    answer -- 50 dpi for a figure nobody said 50 about."""
    explorer.set_style(dataclasses.replace(explorer.style(), dpi=600))
    assert explorer._controls["dpi"].value() == 600

    explorer.set_style(dataclasses.replace(explorer.style(), dpi=None))
    assert explorer._controls["dpi"].value() == 600


# --------------------------------------------------------------------------
# compartments, splits and the house red
# --------------------------------------------------------------------------

def test_an_empty_screen_has_no_compartments_to_colour_by(localised, qapp,
                                                          qtbot):
    """No results, no genes, no compartments -- and asking the reference
    table anyway would be asking it about an empty column."""
    assert localised.compartments() == ["dense granules", "rhoptries 1"]

    empty = VolcanoExplorer(pd.DataFrame())
    qtbot.addWidget(empty)
    assert empty.compartments() == []


def test_a_reference_table_that_will_not_be_read_costs_only_the_tick_boxes(
        localised, monkeypatch):
    """No reference table, no colouring -- but still a volcano. A screen of a
    different organism has no reason to carry that file, and a lookup that
    throws must not take the plot down with it."""
    assert localised.compartments()

    from spacr import localisation

    def _unreadable():
        raise OSError("the bundled table is not there")

    monkeypatch.setattr(localisation, "table", _unreadable)
    assert localised.compartments() == []
    localised.set_results(localised.results())
    assert localised._controls["localizations"].count() == 0
    assert localised._controls["x_column"].count() > 0


def test_a_split_that_cannot_be_sized_is_left_unset(explorer):
    """The suggestion is read off the y column, which is one of the settings
    that can be broken. The y column's own complaint is the one the reader is
    shown; the split does not add a second."""
    style = dataclasses.replace(explorer.style(), split_axis=True,
                                split_y_lims=None)
    explorer.set_style(style)
    assert explorer.style().split_y_lims is not None

    broken = dataclasses.replace(explorer.style(), split_axis=True,
                                 split_y_lims=None, y_column="not_a_column")
    explorer.set_style(broken)
    assert explorer.style().split_y_lims is None
    assert "y_column" in explorer.problems()


def test_the_last_whole_style_that_drew_is_offered_after_the_current_one(
        explorer):
    """Three chances to keep something on the canvas, in that order. The
    whole of the last style that drew is only worth offering once something
    has drawn."""
    fresh = VolcanoExplorer(pd.DataFrame())
    offered = list(fresh._drawable_styles({}))
    assert [style is fresh.style() for style in offered] == [True, False]

    # `explorer` has drawn once already, so it has a remembered style.
    offered = list(explorer._drawable_styles({}))
    assert len(offered) == 3
    assert offered[0] is explorer.style()
    assert offered[1] == explorer._last_good_style
    assert offered[2] == VolcanoStyle()


def test_the_house_red_answers_when_the_theme_cannot(monkeypatch):
    """A bare widget built with no application palette still has to be able
    to turn a label red."""
    from spacr.qt import theme

    house = str(theme.active_palette().get("error") or _PROBLEM_INK)
    assert VolcanoExplorer._error_ink() == house

    def _no_palette():
        raise RuntimeError("no theme here")

    monkeypatch.setattr(theme, "active_palette", _no_palette)
    assert VolcanoExplorer._error_ink() == _PROBLEM_INK


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_an_export_that_wrote_nothing_says_so_and_names_no_file(
        explorer, tmp_path, monkeypatch):
    """SVG goes through the vector writer, which reports the path it actually
    wrote. An empty answer means no file exists, and handing that path back
    would name a file that was never created."""
    from spacr.qt.widgets import figure_settings

    said = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args: said.append(args[2])))

    target = tmp_path / "volcano.svg"
    monkeypatch.setattr(figure_settings, "save_figure_as",
                        lambda *args, **kwargs: str(target))
    assert explorer.export("svg", str(target)) == str(target)
    assert said == []

    monkeypatch.setattr(figure_settings, "save_figure_as",
                        lambda *args, **kwargs: "")
    assert explorer.export("svg", str(target)) is None
    assert said and str(target) in said[0]
