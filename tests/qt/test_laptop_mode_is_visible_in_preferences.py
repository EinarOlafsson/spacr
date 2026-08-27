"""Laptop mode is a row in Preferences, not just an environment variable.

The mode shipped decided-and-applied but reachable only through
SPACR_LAPTOP_MODE, which is to say it could not be found by anybody who did
not already know it existed. These tests hold it on the Performance tab.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QLabel, QTabWidget  # noqa: E402

from spacr.qt import preferences as P  # noqa: E402


@pytest.fixture
def dialog(qtbot, tmp_path, monkeypatch):
    """A real Preferences dialog, writing to a sandboxed config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    dlg = P.PreferencesDialog()
    qtbot.addWidget(dlg)
    return dlg


def test_the_row_exists(dialog):
    assert dialog.findChild(QComboBox, "LaptopMode") is not None


def test_it_is_on_the_performance_tab(dialog):
    """"how much of my machine does this use" has one place to go."""
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    found = [tabs.tabText(i) for i in range(tabs.count())
             if tabs.widget(i).findChild(QComboBox, "LaptopMode")]
    assert found == ["Performance"]


def test_it_offers_automatic_on_and_off(dialog):
    combo = dialog.findChild(QComboBox, "LaptopMode")
    assert [combo.itemData(i) for i in range(combo.count())] == \
        list(P.LAPTOP_MODE_CHOICES)


def test_it_starts_on_the_stored_value(dialog):
    combo = dialog.findChild(QComboBox, "LaptopMode")
    assert combo.currentData() == P.get_laptop_mode()


def test_the_note_says_what_automatic_decided_on_this_machine(dialog):
    """The label cannot state the outcome; the outcome depends on the
    machine reading it, so the note has to."""
    combo = dialog.findChild(QComboBox, "LaptopMode")
    note = dialog.findChild(QLabel, "LaptopModeNote")
    combo.setCurrentIndex(list(P.LAPTOP_MODE_CHOICES).index("automatic"))
    assert "laptop mode:" in note.text()
    assert "core(s)" in note.text()


def test_the_note_changes_with_the_choice(dialog):
    combo = dialog.findChild(QComboBox, "LaptopMode")
    note = dialog.findChild(QLabel, "LaptopModeNote")
    seen = set()
    for i in range(combo.count()):
        combo.setCurrentIndex(i)
        seen.add(note.text())
    assert len(seen) == combo.count()


def test_choosing_on_says_only_the_drawing_changes(dialog):
    """The promise that makes turning things down acceptable at all."""
    combo = dialog.findChild(QComboBox, "LaptopMode")
    note = dialog.findChild(QLabel, "LaptopModeNote")
    combo.setCurrentIndex(list(P.LAPTOP_MODE_CHOICES).index("on"))
    assert "same answer either way" in note.text()


def test_the_row_has_a_caption(dialog):
    """A caption ships with its row, or the row explains nothing."""
    assert "Laptop mode" in P.PREFERENCE_TIPS


# --- storage ---------------------------------------------------------------


def test_it_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    for choice in P.LAPTOP_MODE_CHOICES:
        P.set_laptop_mode(choice)
        assert P.get_laptop_mode() == choice


def test_an_unreadable_stored_value_reads_as_automatic(tmp_path,
                                                       monkeypatch):
    """A setting nobody can interpret behaves as though never set."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    P._settings().setValue(P._KEY_LAPTOP_MODE, "sideways")
    assert P.get_laptop_mode() == "automatic"


def test_an_unknown_choice_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    with pytest.raises(ValueError, match="unknown laptop mode"):
        P.set_laptop_mode("sideways")


def test_automatic_is_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert P.get_laptop_mode() == "automatic"
