"""Instruction 169 C: the folds and the divider positions come back.

"Collapsed is a real state and it persists: a reader who folds four
categories away does not want them back on the next run."

Every test here writes into a QSettings pointed at a temporary file. Reading
the user's own settings would make the suite depend on how the machine
running it happens to be configured, and writing to them would rearrange
their panel.
"""
import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A QSettings of our own, so the suite never touches the real one."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    path = tmp_path / "spacr.ini"
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_nothing_stored_is_an_empty_dict_not_a_default(store):
    """EMPTY, not a made-up layout: the panel's own first run is the right one."""
    assert store.get_section_layout("measurements") == {}


def test_a_layout_round_trips(store):
    store.set_section_layout("measurements", folded=["Regression"],
                             sizes=[100, 200, 300])
    got = store.get_section_layout("measurements")
    assert got["folded"] == ["Regression"]
    assert got["sizes"] == [100, 200, 300]


def test_two_panels_do_not_share_one_layout(store):
    store.set_section_layout("measurements", folded=["Regression"])
    store.set_section_layout("figures", folded=["Live"])
    assert store.get_section_layout("measurements")["folded"] == ["Regression"]
    assert store.get_section_layout("figures")["folded"] == ["Live"]


def test_a_corrupt_store_is_not_an_error(store):
    """A settings file edited by hand must not stop the panel from opening."""
    store._settings().setValue(store._KEY_SECTION_LAYOUT, "{not json")
    assert store.get_section_layout("measurements") == {}


def test_the_panel_puts_its_folds_back(qtbot, store):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._scan_panel
    # OPENED FIRST, so folding it is a real change. Every section starts
    # closed as of 2026-08-20 ("measurment sections should all start
    # closed"), and asking an already-folded section to fold emits nothing
    # and stores nothing -- which is what this test used to rely on
    # `OPENS_EXPANDED` to avoid.
    title = panel.section_titles()[0]
    panel.set_section_expanded(title, True)

    panel.set_section_expanded(title, False)      # folding stores it
    stored = store.get_section_layout(panel.LAYOUT_KEY)
    assert title in stored["folded"], stored

    again = AppScreen("regression")
    qtbot.addWidget(again)
    assert not again._scan_panel.is_section_expanded(title), (
        "the fold did not survive re-opening the screen")


def test_a_section_the_stored_layout_never_heard_of_opens(qtbot, store):
    """A layout from an older version must not fold -- or crash -- a new section."""
    from spacr.qt.screens.app_screen import AppScreen

    store.set_section_layout("measurements",
                            folded=["A section that no longer exists"],
                            sizes=[1, 2])          # and the wrong count
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._scan_panel
    for title in panel.section_titles():
        assert panel.is_section_expanded(title), title


def test_restoring_does_not_write_the_half_restored_state_back(qtbot, store):
    """The folds go back one at a time; a save mid-way would store a mixture."""
    from spacr.qt.screens.app_screen import AppScreen

    first = AppScreen("regression")
    qtbot.addWidget(first)
    panel = first._scan_panel
    titles = panel.section_titles()
    # OPEN everything, then fold everything, so the stored set is every
    # section. Since 2026-08-20 they all START folded, and folding an
    # already-folded section emits nothing and stores nothing -- so without
    # the opening pass this stored no layout at all.
    for title in titles:
        panel.set_section_expanded(title, True)
    for title in titles:
        panel.set_section_expanded(title, False)
    wanted = set(store.get_section_layout("measurements")["folded"])
    assert wanted == set(titles)

    second = AppScreen("regression")
    qtbot.addWidget(second)
    assert set(store.get_section_layout("measurements")["folded"]) == wanted
