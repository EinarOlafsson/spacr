"""The Mask panel showing every object's settings with every channel None.

"it was in the mask modual that i saw the object settings eaven when object
channels were all none"

A freshly built panel does not do this -- one row per object, its own channel,
and nothing else until a channel names a plane. The disagreement WAS the
finding, and this is what differed: the settings-search strip above the form
shows every row it indexed whenever nothing is narrowing, and pressing "All
settings" is exactly that. It runs after the object rule has hidden its rows
and nothing ran afterwards, so forty rows per object came back with every
channel still empty. The level is remembered per module, so from the first
press onwards the panel opened that way every time -- which is why a
maintainer sees it and a test that builds a bare screen does not.

The rows the rule hides now watch themselves, so any route that puts one back
is answered by another pass. These tests drive the real strip.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: One setting per object, none of which is that object's own switch.
OBJECT_ROWS = ("cell_diameter", "nucleus_diameter", "pathogen_diameter",
               "organelle_diameter", "cell_CP_prob")


def _screen(qtbot, app_key: str):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    qtbot.wait(1)
    return screen, screen._settings_model


def _strip(qtbot, screen):
    """The real search strip, built the way the window builds it."""
    from spacr.qt.settings_search import SettingsSearchBar

    bar = SettingsSearchBar(screen)
    qtbot.addWidget(bar)
    qtbot.wait(1)
    return bar


def _shown(screen, keys=OBJECT_ROWS) -> dict:
    return {key: screen.setting_row_is_visible(key) for key in keys}


# ---------------------------------------------------------------------------
# The reproduction
# ---------------------------------------------------------------------------

def test_all_settings_does_not_put_the_absent_objects_back(qtbot):
    """The press that used to fill the Mask form with rows for objects that
    are not in the run."""
    from spacr.qt.settings_search import ALL

    screen, model = _screen(qtbot, "mask")
    assert all(value is None for value in
               (model.collect()[f"{role}_channel"]
                for role in ("cell", "nucleus", "pathogen", "organelle")))
    assert _shown(screen) == dict.fromkeys(OBJECT_ROWS, False)

    bar = _strip(qtbot, screen)
    bar.set_level(ALL)
    qtbot.wait(1)
    assert _shown(screen) == dict.fromkeys(OBJECT_ROWS, False)


def test_a_module_remembered_at_all_settings_still_opens_clean(qtbot):
    """The state the maintainer's session was in: the level is persisted, so
    the strip is built at ALL rather than switched to it."""
    from spacr.qt.settings_search import ALL, remember_disclosure

    remember_disclosure("mask", ALL)
    screen, _model = _screen(qtbot, "mask")
    bar = _strip(qtbot, screen)
    assert bar.level() == ALL
    qtbot.wait(1)
    assert _shown(screen) == dict.fromkeys(OBJECT_ROWS, False)


def test_releasing_a_search_query_does_not_put_them_back(qtbot):
    """Typing and clearing is the other way to reach an unnarrowed strip."""
    from spacr.qt.settings_search import ALL

    screen, _model = _screen(qtbot, "mask")
    bar = _strip(qtbot, screen)
    bar.set_level(ALL)
    bar.set_query("diameter")
    qtbot.wait(1)
    bar.set_query("")
    qtbot.wait(1)
    assert _shown(screen) == dict.fromkeys(OBJECT_ROWS, False)


def test_all_settings_does_not_put_the_absent_slots_headings_back(qtbot):
    """The headings go with the rows, or the wall comes back as captions."""
    from spacr.qt.settings_search import ALL

    screen, _model = _screen(qtbot, "mask")

    def slot_headings():
        numbers = set()
        for section in screen._settings_sections:
            title = str(section.property("settingsCategorySource")
                        or section.title())
            rest = title[len("Organelle "):] \
                if title.startswith("Organelle ") else ""
            if rest.isdigit() and not section.isHidden():
                numbers.add(int(rest))
        return sorted(numbers)

    assert slot_headings() == [1, 2, 3, 4]
    bar = _strip(qtbot, screen)
    bar.set_level(ALL)
    qtbot.wait(1)
    assert slot_headings() == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# And the rule still lets a run's own objects through
# ---------------------------------------------------------------------------

def test_an_object_the_run_has_is_shown_even_under_all_settings(qtbot):
    """The guard puts back what the rule hid; it must not hide anything the
    rule did not."""
    from spacr.qt.settings_search import ALL

    screen, model = _screen(qtbot, "mask")
    bar = _strip(qtbot, screen)
    bar.set_level(ALL)
    qtbot.wait(1)

    model._widgets["cell_channel"].setText("1")
    qtbot.wait(1)
    assert screen.setting_row_is_visible("cell_diameter") is True
    assert screen.setting_row_is_visible("cell_CP_prob") is True
    # ONE OBJECT AT A TIME, still.
    assert screen.setting_row_is_visible("nucleus_diameter") is False

    model._widgets["cell_channel"].clear()
    qtbot.wait(1)
    assert screen.setting_row_is_visible("cell_diameter") is False


def test_the_panel_says_which_rows_the_run_has_no_object_for(qtbot):
    """The seam a filter subtracts so its count matches what it shows."""
    screen, model = _screen(qtbot, "mask")

    hidden = set(model.keys_hidden_by_the_run())
    assert "cell_diameter" in hidden
    assert "cell_channel" not in hidden, "the switch is never one of them"

    model._widgets["cell_channel"].setText("1")
    qtbot.wait(1)
    assert "cell_diameter" not in set(model.keys_hidden_by_the_run())
