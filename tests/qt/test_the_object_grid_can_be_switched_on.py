"""The per-object grid replaces the flat rows without changing the file.

78 of Mask's 201 settings are the same twenty-odd questions asked once per
object type, so the form asks 203 questions before anything is segmented. The
grid asks each question once, with a column per object.

IT IS OFF UNLESS CHOSEN. This is the most-used screen in the application, so
the grid arrives as an offer rather than as a change to what everyone already
knows -- and the first test here is that a user who has chosen nothing sees
exactly the form they saw yesterday.

THE PROPERTY THAT MAKES IT SAFE, and what every test below is about: the grid
edits the same widgets the flat rows do. `collect()` is unchanged, so a
settings file written with the grid on is the same file written with it off,
and nothing downstream of the panel learns the grid exists.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def grid_preference():
    """Restore whatever the machine had, whatever a test does to it."""
    from spacr.qt import preferences as prefs

    was = prefs.get_object_grid_enabled()
    yield prefs
    prefs.set_object_grid_enabled(was)


def _screen(qtbot, prefs, on):
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.app_screen import AppScreen

    prefs.set_object_grid_enabled(on)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    QApplication.processEvents()
    return screen


def test_the_default_is_the_form_that_was_always_there(qtbot, qt_theme_applied, grid_preference):
    """Nothing changes for a user who has not asked for the grid."""
    from spacr.qt import preferences as prefs

    assert prefs.DEFAULT_OBJECT_GRID is False

    screen = _screen(qtbot, grid_preference, False)

    assert not hasattr(screen, "_object_grid"), "the grid mounted uninvited"
    assert screen.setting_row_is_visible("cell_diameter") is True


def test_switching_it_on_puts_the_grid_where_the_rows_were(qtbot, qt_theme_applied,
                                                           grid_preference):
    """The rows go, the grid arrives, the widgets stay."""
    screen = _screen(qtbot, grid_preference, True)

    assert hasattr(screen, "_object_grid"), "the grid did not mount"
    assert screen.setting_row_is_visible("cell_diameter") is False
    # HIDDEN, NOT DROPPED. The widget is what `collect()` reads and what the
    # grid writes through to, and the settings search still indexes it.
    assert "cell_diameter" in screen._settings_model._widgets


def test_the_settings_are_the_same_either_way(qtbot, qt_theme_applied, grid_preference):
    """The whole safety argument, asked directly."""
    off = _screen(qtbot, grid_preference, False)._settings_model.collect()
    on = _screen(qtbot, grid_preference, True)._settings_model.collect()

    assert set(on) == set(off), "the grid changed which settings exist"
    for key, value in off.items():
        assert on[key] == value, f"{key} differs with the grid on"


def test_editing_a_cell_reaches_the_settings_the_run_reads(qtbot, qt_theme_applied,
                                                           grid_preference):
    """Through `setData`, which is what typing in a cell calls."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    screen = _screen(qtbot, grid_preference, True)
    model = screen._settings_model
    table = screen._object_grid._model
    row = list(table.table()).index("min_area")
    before = int(model.collect()["cell_min_area"])

    assert table.setData(table.index(row, 0), str(before + 7), Qt.EditRole)
    QApplication.processEvents()

    assert int(model.collect()["cell_min_area"]) == before + 7


def test_a_row_the_grid_speaks_for_stays_hidden_across_a_refresh(
        qtbot, qt_theme_applied, grid_preference):
    """`refresh_object_visibility` recomputes its hidden set from scratch.

    So the grid's keys are kept in a set of their own and unioned in. Putting
    them into `_hidden_by_the_run` would show every one of them again the
    first time a channel was typed.
    """
    screen = _screen(qtbot, grid_preference, True)

    screen._settings_model.refresh_object_visibility()

    assert screen.setting_row_is_visible("cell_diameter") is False


def test_the_grid_is_not_a_labelled_setting_row(qtbot, qt_theme_applied, grid_preference):
    """It goes in through `add_prose_row`, and that matters.

    Every entry in `Section._row_widgets` is taken to BE a labelled setting
    by the module smoke test, which asserts each field carries a `settingKey`
    and that its label holds linked API help. A grid is neither.
    """
    screen = _screen(qtbot, grid_preference, True)
    section = next(s for s in screen._settings_sections
                   if screen._object_grid in s.findChildren(type(screen._object_grid)))

    assert screen._object_grid not in [w for _l, w in section._row_widgets]


# ---------------------------------------------------------------------------
# Reachable without editing QSettings by hand
# ---------------------------------------------------------------------------

def test_preferences_offers_the_choice_and_saves_it(qtbot, qt_theme_applied, grid_preference):
    """A preference nobody can find is a preference nobody has.

    The row is on the Appearance tab beside the two tooltip switches, and
    accepting the dialog is what writes it -- so this drives the dialog
    rather than calling the setter, which would test nothing about the row.
    """
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = grid_preference.PreferencesDialog()
    qtbot.addWidget(dialog)
    toggles = [w for w in dialog.findChildren(object)
               if getattr(w, "objectName", lambda: "")() == "ObjectSettingsGrid"]

    assert len(toggles) == 1, "the Appearance tab offers no such row"
    assert toggles[0].isChecked() is grid_preference.get_object_grid_enabled()

    toggles[0].setChecked(True)
    for box in dialog.findChildren(QDialogButtonBox):
        box.accepted.emit()

    assert grid_preference.get_object_grid_enabled() is True


# ---------------------------------------------------------------------------
# Live, in both directions
# ---------------------------------------------------------------------------
#
# The switch used to be read exactly once, while a settings panel was being
# built, so it did nothing at all to a module already open -- reported as
# "turn it on, it doesn't go on; turn it off, it doesn't go off". Both
# directions are here because only one of them was ever likely to be tested,
# and the off direction is the one that leaves settings on NO screen if it
# goes wrong: the flat rows are hidden while the grid speaks for them.

def test_turning_it_on_reaches_a_module_already_open(qtbot, qt_theme_applied,
                                                     grid_preference):
    """Opened with the switch off, then switched on: the table appears."""
    screen = _screen(qtbot, grid_preference, False)
    assert getattr(screen, "_object_grid", None) is None

    grid_preference.set_object_grid_enabled(True)

    assert screen.apply_object_grid_preference() is True
    assert screen._object_grid is not None
    assert screen.setting_row_is_visible("cell_diameter") is False


def test_turning_it_off_reaches_a_module_already_open(qtbot, qt_theme_applied,
                                                      grid_preference):
    """Opened with the switch on, then switched off: the rows come back.

    THE ROWS ARE THE POINT. They are hidden rather than dropped while the
    grid speaks for them, so a table that went away without releasing them
    would leave those settings in no table and no form -- still collected,
    still saved, and impossible to edit.
    """
    screen = _screen(qtbot, grid_preference, True)
    assert screen._object_grid is not None
    assert screen.setting_row_is_visible("cell_diameter") is False

    grid_preference.set_object_grid_enabled(False)

    assert screen.apply_object_grid_preference() is True
    assert getattr(screen, "_object_grid", None) is None
    assert screen.setting_row_is_visible("cell_diameter") is True


def test_asking_twice_changes_nothing_the_second_time(qtbot, qt_theme_applied,
                                                      grid_preference):
    """Idempotent, so the broadcast can be sent to every screen blindly."""
    screen = _screen(qtbot, grid_preference, False)
    grid_preference.set_object_grid_enabled(True)

    assert screen.apply_object_grid_preference() is True
    assert screen.apply_object_grid_preference() is False


def test_the_table_lands_above_the_trailing_stretch(qtbot, qt_theme_applied,
                                                    grid_preference):
    """Mounted late, it must not be appended below the spring.

    The first build adds the stretch AFTER the grid; a later mount finds one
    already there, and appending would put the table under a spring that
    pushes it off the bottom of a scrolling panel.
    """
    screen = _screen(qtbot, grid_preference, False)
    grid_preference.set_object_grid_enabled(True)
    screen.apply_object_grid_preference()

    layout = screen._settings_layout
    section = screen._object_grid.parent()
    while section is not None and not hasattr(section, "add_prose_row"):
        section = section.parent()
    index = layout.indexOf(section)

    assert index >= 0, "the table was not put in the settings column at all"
    for below in range(index + 1, layout.count()):
        item = layout.itemAt(below)
        if item is not None and item.spacerItem() is not None:
            break
    else:
        raise AssertionError("the table was appended below the stretch")


# ---------------------------------------------------------------------------
# Two-way, not one
# ---------------------------------------------------------------------------
#
# The grid always wrote THROUGH to the widgets -- that is what makes it safe,
# and it is tested above. What was missing is the return path: nothing told
# the table when a widget moved, so a value changed anywhere else left the
# table showing the old answer while `collect()` returned the new one. Two
# answers to the same question is the exact failure the binding exists to
# prevent, so it is worth a test in each direction.

def test_a_cell_reaches_the_widget_behind_it(qtbot, qt_theme_applied,
                                             grid_preference):
    """Table to form: the direction that always worked."""
    screen = _screen(qtbot, grid_preference, True)

    screen._object_grid.set_value("diameter", "cell", "42")

    assert screen._settings_model.collect()["cell_diameter"] == 42


def test_a_widget_reaches_the_cell_in_front_of_it(qtbot, qt_theme_applied,
                                                  grid_preference):
    """Form to table: the direction that did not.

    Driven through `set_value_for_key` rather than by poking the widget,
    because that is the path a settings file, a preset and the Live Preview
    all take -- and each of them used to leave the table stale.
    """
    screen = _screen(qtbot, grid_preference, True)

    screen._settings_model.set_value_for_key("cell_diameter", 77)

    assert screen._object_grid.table()["diameter"]["cell"] == 77


def test_a_channel_typed_into_the_form_reaches_the_table(
        qtbot, qt_theme_applied, grid_preference):
    """The channel is the setting that gates every other one.

    A channel showing the old value in the table is worse than an ordinary
    stale cell: it is the switch that says whether the run has that object
    at all, so the table would be claiming an object the form has turned off.
    """
    screen = _screen(qtbot, grid_preference, True)

    screen._settings_model.set_value_for_key("nucleus_channel", 2)

    assert screen._object_grid.table()["channel"]["nucleus"] == 2


def test_the_round_trip_does_not_oscillate(qtbot, qt_theme_applied,
                                           grid_preference):
    """Each direction triggers the other, so the guard has to hold.

    Writing a widget makes it emit, which now writes the table, which emits,
    which writes the widget. Without the reentrancy guard this is a loop --
    and one that would run inside a keystroke.
    """
    screen = _screen(qtbot, grid_preference, True)
    binding = screen._object_grid_binding

    screen._object_grid.set_value("diameter", "cell", "13")

    assert binding._busy is False, "the guard was left on"
    assert screen._settings_model.collect()["cell_diameter"] == 13
    assert screen._object_grid.table()["diameter"]["cell"] == 13


def test_following_the_form_twice_connects_nothing_twice(
        qtbot, qt_theme_applied, grid_preference):
    """It is called again whenever the table widens, so it must not double up."""
    screen = _screen(qtbot, grid_preference, True)
    binding = screen._object_grid_binding
    before = len(binding._followed)

    assert before > 0, "the binding is following no widgets at all"
    assert binding.follow_the_form() == 0
    assert len(binding._followed) == before


def test_it_can_be_switched_back_and_forth(qtbot, qt_theme_applied,
                                           grid_preference):
    """"As they please" is the actual request, so a full cycle is the test.

    One direction working is not the property wanted. Each pass mounts a NEW
    table over a form whose rows the previous one hid, so anything left
    behind -- a stale grid reference, rows still hidden, a section still in
    `_settings_sections` -- shows up on the second lap rather than the first.
    """
    screen = _screen(qtbot, grid_preference, False)
    seen = []
    for wanted in (True, False, True, False):
        grid_preference.set_object_grid_enabled(wanted)
        screen.apply_object_grid_preference()
        seen.append((getattr(screen, "_object_grid", None) is not None,
                     screen.setting_row_is_visible("cell_diameter")))

    assert seen == [(True, False), (False, True), (True, False), (False, True)]
