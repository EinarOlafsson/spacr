"""3D and Time: two switches that say which axes the plate actually has.

"to the left of the Live button which sitts to the left of the AI button, i
want to have 2 new buttons called 3D and Time which toggle the visability of
the volumetric and Timelapse related settings. these rules should apply for
Mask generation and measure."

What these tests pin:

  * the two switches sit in the action row in that order, immediately left
    of Live, on Mask Generation and on Measure;
  * each reveals ITS OWN settings and only those -- asserted as a set read
    back off the driven form, never by eye;
  * they are STATES: pressing one leaves it lit, and it stays lit across the
    panel refresh its own press causes;
  * hiding a setting does not lose it. The value is still collected and
    still goes to the run, which is the difference between a form that
    stops asking a question and a form that throws the answer away;
  * a settings file that asks for a volumetric or a timelapse run lights the
    matching switch, so an import never fills in settings it then hides;
  * and which settings count as volumetric or timelapse is read off
    :mod:`spacr.settings`, not off a list written here.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFormLayout

from spacr.qt.screens import mask as mask_folds
from spacr.qt.screens import measure as measure_folds
from spacr.qt.screens.app_screen import (
    DIMENSION_TOGGLE_APPS, DIMENSION_TOGGLE_MIN_PX, AppScreen,
    dimension_settings, setting_dimension,
)


#: ``app key -> the function that installs that screen's fold switches``.
#:
#: The folds are installed by the main window as the stack reaches a screen,
#: so a screen built on its own has none. Installing them here means these
#: tests are looking at the masthead a user sees rather than at half of it.
INSTALL_FOLDS = {"mask": mask_folds.install_folds,
                 "measure": measure_folds.install_folds}


def _screen(qtbot, app_key: str) -> AppScreen:
    """One module screen, shown, with its fold switches on, ready to drive.

    SHOWN, because "visible" is asked of the real widgets below and Qt
    reports every widget on an unshown page as invisible -- which would let
    a test pass on a form that never hid anything at all.
    """
    screen = AppScreen(app_key=app_key)
    qtbot.addWidget(screen)
    INSTALL_FOLDS[app_key](screen)
    screen.show()
    qtbot.waitExposed(screen)
    return screen


def _action_row_labels(screen: AppScreen) -> list:
    """The text of every toggle in the action row, in the order drawn."""
    switch = screen.dimension_switch("z")
    assert switch is not None, "the screen carries no 3D switch to find"
    layout = switch.parentWidget().layout()
    labels = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if widget is None or type(widget).__name__ != "AiToggleLabel":
            continue
        labels.append(widget.text())
    return labels


def _rendered(screen: AppScreen, dimension: str) -> set:
    """Every key of ``dimension`` on this screen's OWN settings form.

    Walked over the screen's own categories rather than over the model's
    ``key -> widget`` map, because a fold adds ITS module's controls to that
    map while mounting them in cards of its own. Those cards belong to the
    fold's switch (see the boundary test at the foot of this file), so
    counting them here would be measuring a control the 3D and Time
    switches do not own.
    """
    keys = dimension_settings()[dimension]
    widgets = getattr(screen._settings_model, "_widgets", {}) or {}
    by_widget = {id(widget): key for key, widget in widgets.items()}
    found = set()
    for section in screen._settings_sections:
        form = getattr(section, "_form", None)
        if not isinstance(form, QFormLayout):
            continue
        for index in range(form.rowCount()):
            item = form.itemAt(index, QFormLayout.FieldRole)
            field = item.widget() if item is not None else None
            key = by_widget.get(id(field)) if field is not None else None
            if key in keys:
                found.add(key)
    return found


def _visible(screen: AppScreen, keys) -> set:
    """The subset of ``keys`` whose row is on the form right now."""
    return {key for key in keys if screen.setting_row_is_visible(key)}


# ---------------------------------------------------------------------------
# where they are
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", sorted(INSTALL_FOLDS))
def test_the_switches_sit_in_that_order_immediately_left_of_live(
        qtbot, qt_theme_applied, app_key):
    """3D, then Time, then Live -- with nothing between them.

    Read off the built row rather than off the construction order, because
    the row is assembled by several unrelated blocks (preview, GPU,
    hyperparameter search, AI) and only the finished order is the promise.
    """
    screen = _screen(qtbot, app_key)

    labels = _action_row_labels(screen)

    assert "Live" in labels, (
        f"{app_key} has no Live toggle, so 'left of Live' means nothing")
    live = labels.index("Live")
    assert labels[live - 2:live + 1] == ["3D", "Time", "Live"], (
        f"{app_key} action row reads {labels}, not 3D, Time, Live")


def test_a_module_that_is_neither_mask_nor_measure_carries_no_switch(
        qtbot, qt_theme_applied):
    """The request named two screens, and the switches are on those two.

    Classify has beta settings and an action row of its own, so it would
    have grown a pair of dead switches had the gate been "any screen".
    """
    assert "classify" not in DIMENSION_TOGGLE_APPS
    screen = AppScreen(app_key="classify")
    qtbot.addWidget(screen)

    assert screen.dimension_switch("z") is None
    assert screen.dimension_switch("t") is None


def test_a_screen_with_no_switch_hides_nothing(qtbot, qt_theme_applied):
    """Nothing is ever hidden where nothing can bring it back.

    The Timelapse module's own screen renders the same z and t settings Mask
    does -- its Acquisition & Axes category is nothing else -- and it carries
    no switches, because it is not one of the two screens the request named.
    A gate keyed on the switch STATE alone therefore took that whole category
    off a screen with no control able to return it. It is keyed on the
    switch's existence instead.
    """
    screen = AppScreen(app_key="timelapse")
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)

    assert screen.dimension_switch("z") is None
    for key in ("z_stack", "t_stack", "anisotropy", "frame_interval_s"):
        assert screen.setting_row_is_visible(key), (
            f"{key} is hidden on a screen that cannot show it again")


@pytest.mark.parametrize("app_key", sorted(INSTALL_FOLDS))
def test_the_switches_do_not_starve_the_settings_column(
        qtbot, qt_theme_applied, app_key):
    """Two more toggles in that row cost the settings form its width.

    A QHBoxLayout cannot go below the sum of its children's minimum widths,
    and the action row's minimum is the screen's, which the body splitter
    then satisfies out of the window. Unaided, "3D" asks for 69 px and
    "Time" for 92 px, which took Mask's settings column from 260 px to 83 px
    at a 1200 px window -- a 624 px card in an 83 px viewport, its labels
    hanging out of the right. That is the failure ``ELIDE_ABOVE_PX`` was
    written for, reached from two short labels instead of one long one.

    Measured at the size the module smoke test opens every screen at, and
    asserted against the card rather than against a remembered number: what
    matters is that the labels fit inside the panel they are drawn on.
    """
    screen = AppScreen(app_key=app_key)
    qtbot.addWidget(screen)
    screen.resize(1200, 720)
    screen.show()
    qtbot.waitExposed(screen)

    for dimension in ("z", "t"):
        switch = screen.dimension_switch(dimension)
        assert switch.minimumSizeHint().width() > DIMENSION_TOGGLE_MIN_PX, (
            "the premise is that the label is wider than the cap")
        assert switch.minimumWidth() == DIMENSION_TOGGLE_MIN_PX
        assert switch.sizeHint().width() > DIMENSION_TOGGLE_MIN_PX, (
            "the full label is no longer asked for, so a wide row clips it")

    label, _field = screen._settings_sections[0]._row_widgets[0]
    assert screen._settings_scroll.width() >= label.parentWidget().width(), (
        f"{app_key}: the settings viewport is "
        f"{screen._settings_scroll.width()} px and a setting label needs "
        f"{label.parentWidget().width()}")


# ---------------------------------------------------------------------------
# what they reveal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", sorted(INSTALL_FOLDS))
@pytest.mark.parametrize("dimension", ("z", "t"))
def test_a_switch_reveals_its_own_settings_and_only_its_own(
        qtbot, qt_theme_applied, app_key, dimension):
    """Pressing one shows every key of that dimension and no key of the other.

    Asserted as a set both ways round: the point of the switch is not that
    something appeared, it is that exactly the right settings did.
    """
    other = "t" if dimension == "z" else "z"
    screen = _screen(qtbot, app_key)
    mine, theirs = _rendered(screen, dimension), _rendered(screen, other)
    assert mine, f"{app_key} renders no {dimension} settings to reveal"

    assert _visible(screen, mine | theirs) == set(), (
        "a dimensional setting is on the form before either switch is on")

    screen.dimension_switch(dimension).setChecked(True)

    assert _visible(screen, mine) == mine
    assert _visible(screen, theirs) == set(), (
        f"the {dimension} switch also revealed {other} settings")


@pytest.mark.parametrize("app_key", sorted(INSTALL_FOLDS))
def test_switching_one_off_again_puts_its_settings_away(
        qtbot, qt_theme_applied, app_key):
    """Off is off. A toggle that only ever adds is a button, not a state."""
    screen = _screen(qtbot, app_key)
    switch = screen.dimension_switch("z")
    volumetric = _rendered(screen, "z")

    switch.setChecked(True)
    assert _visible(screen, volumetric) == volumetric
    switch.setChecked(False)

    assert _visible(screen, volumetric) == set()


def test_a_card_of_nothing_but_one_dimension_goes_with_it(
        qtbot, qt_theme_applied):
    """Mask's Volumetric Processing card is z settings and nothing else.

    Hiding its rows one at a time would leave an empty titled card sitting
    on the form, which says a category exists and then shows none of it.
    """
    screen = _screen(qtbot, "mask")
    card = _section_titled(screen, "Volumetric Processing (Beta)")

    assert not card.isVisible()
    screen.dimension_switch("z").setChecked(True)
    assert card.isVisible()


def test_a_card_that_merely_contains_some_keeps_its_other_rows(
        qtbot, qt_theme_applied):
    """Measure files ``timelapse`` beside the channel mapping; the card stays.

    That card is where a user sets which mask is which. Taking it away with
    the two timelapse rows inside it would hide the whole of Measure's
    channel setup behind a switch about time.
    """
    screen = _screen(qtbot, "measure")
    card = _section_titled(screen, "Mask & Channel Mapping")

    assert card.isVisible()
    assert not screen.setting_row_is_visible("timelapse")
    assert screen.setting_row_is_visible("cell_mask_dim"), (
        "the channel mapping went away with the timelapse rows")

    screen.dimension_switch("t").setChecked(True)

    assert screen.setting_row_is_visible("timelapse")
    assert screen.setting_row_is_visible("cell_mask_dim")


def _section_titled(screen: AppScreen, title: str):
    """The settings card written down as ``title`` on this screen."""
    for section in screen._settings_sections:
        if str(section.property("settingsCategorySource")) == title:
            return section
    raise AssertionError(f"{screen.app_key} has no {title!r} category")


# ---------------------------------------------------------------------------
# they are states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dimension", ("z", "t"))
def test_a_switch_stays_lit_while_it_is_on(qtbot, qt_theme_applied, dimension):
    """On is a colour that survives the panel rebuild the press causes.

    The fold switches on Mask hold their state by filling with their stage
    colour; these hold theirs the way Live and AI in the same row do, by
    inking the label in the accent. Measured against the palette rather
    than looked at, and read AFTER the toggle has driven the settings panel,
    because that is the redraw a momentary highlight would not survive.
    """
    from spacr.qt.theme import active_palette

    screen = _screen(qtbot, "mask")
    switch = screen.dimension_switch(dimension)
    off_sheet = switch.styleSheet()

    switch.setChecked(True)

    assert switch.isChecked()
    assert screen.dimension_is_on(dimension)
    assert active_palette()["button_accent"] in switch.styleSheet()
    assert switch.styleSheet() != off_sheet, (
        "the switch looks the same on as off, so nothing says it is on")


# ---------------------------------------------------------------------------
# hidden, not deleted
# ---------------------------------------------------------------------------

def test_a_hidden_setting_keeps_its_value_and_still_reaches_the_run(
        qtbot, qt_theme_applied):
    """Switching 3D off must not throw the voxel geometry away.

    A setting that leaves the form while staying in the dict is how a run
    gets a value nobody can see; a setting that leaves the dict as well is
    how a user loses the number they typed. Neither happens: the row goes,
    the answer stays, and turning the switch back on shows the same answer.
    """
    screen = _screen(qtbot, "measure")
    screen.dimension_switch("z").setChecked(True)
    screen.apply_settings_dict({"voxel_size_z_um": 0.7})

    screen.dimension_switch("z").setChecked(False)

    assert not screen.setting_row_is_visible("voxel_size_z_um")
    assert screen._settings_model.collect()["voxel_size_z_um"] == 0.7
    screen.dimension_switch("z").setChecked(True)
    assert screen._settings_model.collect()["voxel_size_z_um"] == 0.7


# ---------------------------------------------------------------------------
# a settings file moves the switches
# ---------------------------------------------------------------------------

def test_a_volumetric_settings_file_lights_the_3d_switch(
        qtbot, qt_theme_applied):
    """``z_stack=True`` is a file about a z axis, so the z settings appear.

    Without this an import would fill in every voxel size the file names and
    then hide all of them -- from the user's side, an import that did
    nothing.
    """
    screen = _screen(qtbot, "mask")
    assert not screen.dimension_is_on("z")

    screen.apply_settings_dict({"z_stack": True, "voxel_size_z_um": 0.5})

    assert screen.dimension_is_on("z")
    assert screen.setting_row_is_visible("voxel_size_z_um")
    assert not screen.dimension_is_on("t"), (
        "a file about z also switched the time settings on")


def test_a_timelapse_settings_file_lights_the_time_switch(
        qtbot, qt_theme_applied):
    """``timelapse=True`` is a file about a time axis, on either screen.

    Measure renders ``timelapse`` itself, and it is what the mask pipeline
    reads to group a plate into time stacks -- so it is the announcement on
    both, whichever screen the file was written by.
    """
    screen = _screen(qtbot, "measure")

    screen.apply_settings_dict({"timelapse": True})

    assert screen.dimension_is_on("t")
    assert screen.setting_row_is_visible("timelapse")
    assert screen.setting_row_is_visible("timelapse_objects")


def test_a_file_that_says_nothing_about_a_dimension_leaves_it_alone(
        qtbot, qt_theme_applied):
    """Absence is not a request to hide what the user has just opened.

    A settings CSV written before the volumetric keys existed names none of
    them; taking that as "no z axis" would close the panel under a user who
    had opened it on purpose.
    """
    screen = _screen(qtbot, "mask")
    screen.dimension_switch("z").setChecked(True)

    screen.apply_settings_dict({"src": "/tmp/plate", "cell_channel": 1})

    assert screen.dimension_is_on("z")
    assert screen.setting_row_is_visible("z_segmentation_mode")


def test_a_file_that_turns_the_dimension_off_puts_the_settings_away(
        qtbot, qt_theme_applied):
    """``z_stack=False`` is an answer too, and the switch follows it.

    The switch has to track the file in both directions or a second import
    would leave the form describing the first one.
    """
    screen = _screen(qtbot, "mask")
    screen.dimension_switch("z").setChecked(True)

    screen.apply_settings_dict({"z_stack": False})

    assert not screen.dimension_is_on("z")
    assert not screen.setting_row_is_visible("z_segmentation_mode")


# ---------------------------------------------------------------------------
# which settings are which
# ---------------------------------------------------------------------------

def test_the_two_sets_are_read_off_the_settings_module(qt_theme_applied):
    """The membership is derived, so it cannot drift from spacr.settings.

    Asserted against the category lists themselves rather than against keys
    named here: a volumetric setting added to ``3D Settings (Beta)``
    tomorrow is one the 3D switch reveals without anybody editing the GUI.
    """
    from spacr.settings import categories, timelapse_settings

    sets = dimension_settings()

    assert set(categories["3D Settings (Beta)"]) <= sets["z"]
    assert set(categories["4D Settings (Beta)"]) <= sets["t"]
    assert set(timelapse_settings) <= sets["t"]
    assert not sets["z"] & sets["t"], (
        "a setting in both sets would appear and vanish under two switches")


def test_the_folded_timelapse_module_keeps_its_own_switch(
        qtbot, qt_theme_applied):
    """Mask's tracking categories are the FOLD's, and Time does not take them.

    The Timelapse module folded into Mask Generation as two settings
    categories behind a masthead switch, and that switch does a second thing
    the dimension switches deliberately do not: it sets ``timelapse``, the
    gate the mask pipeline reads to group a plate into time stacks. Making
    Time a second owner of those cards' visibility would put two controls on
    one state, which is the failure the fold was written to avoid.

    So the boundary is stated here rather than left to be discovered: Time
    owns the time settings on the host's own form, the fold owns the folded
    module's.
    """
    screen = _screen(qtbot, "mask")
    folds = screen._category_folds
    tracking = folds.folds["timelapse"]
    assert "fps" in tracking.settings_keys

    screen.dimension_switch("t").setChecked(True)

    assert all(not card.isVisible() for card in tracking.sections), (
        "Time revealed the folded module's cards, so two switches now own "
        "them")
    assert not folds.is_active("timelapse")
    screen._fold_strip.button_for("timelapse").setChecked(True)
    assert all(card.isVisible() for card in tracking.sections)


def test_a_setting_with_no_dimension_is_never_hidden(qt_theme_applied):
    """Most settings mean the same thing whatever axes the plate has.

    ``src`` and the cell diameter are the same question on a single plane
    and on a volume, so neither switch may ever be able to take them away.
    """
    assert setting_dimension("z_stack") == "z"
    assert setting_dimension("anisotropy") == "z"
    assert setting_dimension("t_axis") == "t"
    assert setting_dimension("timelapse") == "t"
    for key in ("src", "cell_diameter", "channels", "save_png"):
        assert setting_dimension(key) == "", (
            f"{key} would be hidden by a dimension switch")
