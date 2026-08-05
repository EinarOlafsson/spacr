"""Preferences is tabbed, and every control is still reachable.

The dialog had grown to one scrollable column of thirty-odd rows: Module
visibility and the figure format sat below five animation sliders, and the
only way to find out whether a setting existed was to scroll past
everything else. Tabs are not decoration here — they are what makes "does
spaCR have a setting for this?" a question with an answer.

The risk a rewrite like this carries is a control that quietly stops being
built, which no screenshot catches and no other test notices until somebody
looks for the setting. So the inventory below is explicit: every named
control, and which tab it must be on.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QLabel,
                               QPushButton, QScrollArea, QSlider, QTabWidget,
                               QWidget)


#: object name -> the tab it belongs on. Everything the dialog builds with
#: a name, because a name is what the rest of the suite finds it by.
CONTROLS = {
    "LanguagePreference": "General",
    "AmbientTheme": "Appearance",
    "AmbientPalette": "Appearance",
    "AmbientDriftDirection": "Appearance",
    "AmbientResolution": "Appearance",
    "AmbientBlur": "Appearance",
    "AmbientSpeed": "Appearance",
    "AmbientSize": "Appearance",
    "AmbientDensity": "Appearance",
    "SettingAnimationsEnabled": "Appearance",
    "SpinnerDelay": "Appearance",
    "PaneOpacity": "Appearance",
    "FieldFadeEnabled": "Appearance",
    "FontScale": "General",
    "SpacrMode": "Performance",
    "SpacrModeNote": "Performance",
    "ClearRamButton": "Performance",
    "ClearVramButton": "Performance",
    "ClearCpuButton": "Performance",
    "CheckDiskButton": "Performance",
    "ShowAlphaFeatures": "Modules",
    "ShowBetaFeatures": "Modules",
}

EXPECTED_TABS = ("General", "Appearance", "Performance", "Modules", "Figures")


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    from spacr.qt import preferences as prefs
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    return store


@pytest.fixture
def dialog(qtbot, qt_theme_applied):
    from spacr.qt.preferences import PreferencesDialog
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    return dlg


def _tabs(dialog) -> QTabWidget:
    widget = dialog.findChild(QTabWidget, "PreferencesTabs")
    assert widget is not None, "Preferences is not tabbed"
    return widget


def _tab_of(dialog, object_name):
    """Which tab holds the control called ``object_name``."""
    tabs = _tabs(dialog)
    control = dialog.findChild(QWidget, object_name)
    assert control is not None, f"no control named {object_name}"
    for index in range(tabs.count()):
        page = tabs.widget(index)
        if control is page or page.isAncestorOf(control):
            return tabs.tabText(index)
    raise AssertionError(f"{object_name} is not on any tab")


def test_the_dialog_has_the_five_subject_tabs(dialog):
    tabs = _tabs(dialog)
    assert tuple(tabs.tabText(i) for i in range(tabs.count())) == EXPECTED_TABS


def test_general_is_first_because_language_is_in_it(dialog):
    """A reader who cannot read the interface must find that one first."""
    tabs = _tabs(dialog)
    assert tabs.tabText(0) == "General"
    assert tabs.currentIndex() == 0
    assert _tab_of(dialog, "LanguagePreference") == "General"


@pytest.mark.parametrize("object_name,tab", sorted(CONTROLS.items()))
def test_every_control_is_still_there_and_on_the_right_tab(
        dialog, object_name, tab):
    assert _tab_of(dialog, object_name) == tab


def test_no_control_is_on_two_tabs(dialog):
    tabs = _tabs(dialog)
    for object_name in CONTROLS:
        control = dialog.findChild(QWidget, object_name)
        holders = [tabs.tabText(i) for i in range(tabs.count())
                   if tabs.widget(i).isAncestorOf(control)]
        assert len(holders) == 1, f"{object_name} appears on {holders}"


def test_the_unnamed_controls_survived_the_split(dialog):
    """Theme, dock, colour-blind mode, figure format and DPI have no object
    names of their own; they are found by what they offer."""
    values = set()
    for combo in dialog.findChildren(QComboBox):
        for index in range(combo.count()):
            values.add(combo.itemData(index))
    for expected in ("system", "glass", "locked", "deuteranopia", "png",
                     "pdf", 300):
        assert expected in values, f"the {expected!r} choice is gone"


def test_every_tab_scrolls_on_its_own(dialog):
    """A small screen shortens the tallest tab, not the whole dialog."""
    tabs = _tabs(dialog)
    for index in range(tabs.count()):
        assert isinstance(tabs.widget(index), QScrollArea)


def test_save_and_cancel_are_outside_the_tabs(dialog):
    """They act on the whole dialog, so they must not look like one tab's."""
    tabs = _tabs(dialog)
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None
    for index in range(tabs.count()):
        assert not tabs.widget(index).isAncestorOf(buttons)


def test_saving_from_one_tab_still_writes_the_others(dialog, qtbot):
    """The rows moved; the single Save that writes all of them did not."""
    from spacr.qt import preferences as prefs

    tabs = _tabs(dialog)
    tabs.setCurrentIndex(EXPECTED_TABS.index("Modules"))
    dialog.findChild(QWidget, "ShowAlphaFeatures").setChecked(False)
    dialog.findChild(QSlider, "FontScale").setValue(125)
    dialog.findChild(QSlider, "AmbientSpeed").setValue(150)

    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert prefs.get_show_alpha() is False
    assert prefs.get_font_scale() == pytest.approx(1.25)
    assert prefs.get_ambient_speed() == pytest.approx(1.5)


def test_the_performance_tab_reads_as_one_subject(dialog):
    """The mode and the four buttons are together on purpose: a mode that
    says "cleanup runs at launch" is only readable next to the buttons that
    say what a cleanup is."""
    tab = EXPECTED_TABS.index("Performance")
    page = _tabs(dialog).widget(tab)
    buttons = [b.objectName() for b in page.findChildren(QPushButton)
               if b.objectName()]
    assert set(buttons) == {"ClearRamButton", "ClearVramButton",
                            "ClearCpuButton", "CheckDiskButton"}
    assert page.findChild(QComboBox, "SpacrMode") is not None
    note = page.findChild(QLabel, "SpacrModeNote")
    assert note is not None and note.wordWrap()


def test_the_dialog_still_fits_a_small_screen(dialog):
    """The point of the split: no tab is taller than a laptop panel."""
    dialog.resize(dialog.minimumWidth(), 600)
    assert dialog.sizeHint().height() <= 900, (
        f"the dialog wants {dialog.sizeHint().height()} px of height")
