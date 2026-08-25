"""The Make Masks toolbar: one row of tools, and a settings toggle.

Every assertion here reads the built layout back rather than trusting the
code that built it — "in one row" is a fact about a QHBoxLayout, and "the
canvas keeps the space" is a width in pixels measured before and after
the toggle, on a screen that has been shown and laid out.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtWidgets import QHBoxLayout, QPushButton

import spacr.qt.screens.make_masks as mm
from spacr.qt.screens.make_masks import (
    MODE_BRUSH,
    MODE_NONE,
    MODE_ZOOM,
    MakeMasksScreen,
    tool_row_entries,
)


@pytest.fixture
def folder_2(tmp_path: Path) -> Path:
    folder = tmp_path / "toolbar"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for i in range(2):
        imageio.imwrite(folder / f"img_{i:02d}.tif",
                        rng.integers(0, 65535, (48, 48), dtype=np.uint16))
    return folder


def _row_widgets(screen) -> list:
    row = screen._tool_row_layout
    return [row.itemAt(i).widget() for i in range(row.count())]


# ===========================================================================
# The row
# ===========================================================================

def test_every_tool_is_in_one_horizontal_row(qtbot, qt_theme_applied):
    """"the tools can be in one row on the top"; read back off the layout."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)

    row = screen._tool_row_layout
    assert isinstance(row, QHBoxLayout)
    assert screen._mode_buttons, "the screen built no tools at all"
    for mode, button in screen._mode_buttons.items():
        assert row.indexOf(button) >= 0, f"{mode} is not in the tool row"
    # ...and one row means one: no tool has a second home in the settings.
    for mode, button in screen._mode_buttons.items():
        assert not screen._settings_scroll.isAncestorOf(button), (
            f"{mode} is still inside the settings panel")


def test_the_tools_come_out_in_the_table_order(qtbot, qt_theme_applied):
    """The row reads left to right in TOOL_MODES order, so a tool stays put."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    row = screen._tool_row_layout
    positions = [row.indexOf(screen._mode_buttons[m])
                 for m, _label, _icon in mm.TOOL_MODES]
    assert positions == sorted(positions)
    assert positions[0] == 0, "the first tool is not the first thing in the row"


def test_a_new_mode_constant_reaches_the_row_by_existing(
        qtbot, qt_theme_applied, monkeypatch):
    """A MODE_* added with no table entry still gets a button.

    Two other tools are being added to this screen. The row is built from
    :func:`tool_row_entries`, so their authors do not have to find the
    layout code for their tool to appear — which is the whole reason the
    row is not a list of literals.
    """
    monkeypatch.setattr(mm, "MODE_LASSO", "lasso", raising=False)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)

    assert "lasso" in screen._mode_buttons
    button = screen._mode_buttons["lasso"]
    assert screen._tool_row_layout.indexOf(button) >= 0
    assert button.text() == "Lasso"
    assert button.isCheckable()
    # and it inherits the rule every other tool obeys
    assert not button.isEnabled(), "a new tool was live before a folder opened"


def test_mode_none_is_not_a_tool(qtbot, qt_theme_applied):
    """MODE_NONE is the canvas with nothing held, so it gets no button."""
    assert MODE_NONE not in {mode for mode, _l, _i in tool_row_entries()}
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert MODE_NONE not in screen._mode_buttons


def test_a_new_tool_is_enabled_with_the_rest_when_a_folder_opens(
        qtbot, qt_theme_applied, monkeypatch, folder_2: Path):
    """_sync_button_states reads the row, so it cannot miss a new tool."""
    monkeypatch.setattr(mm, "MODE_LASSO", "lasso", raising=False)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))
    assert screen._mode_buttons["lasso"].isEnabled()
    assert screen._mode_buttons[MODE_BRUSH].isEnabled()


def test_an_action_button_joins_the_same_row(qtbot, qt_theme_applied):
    """A non-mode button lands in the one row, left of the settings toggle."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    button = QPushButton("Cellpose-SAM detect")
    assert screen.add_toolbar_action(button) is button

    row = screen._tool_row_layout
    where = row.indexOf(button)
    assert where >= 0, "the action did not land in the tool row"
    assert where < row.indexOf(screen._btn_settings), (
        "the action pushed past the settings toggle, which stays at the end")
    assert row.indexOf(screen._btn_settings) == row.count() - 1


def test_the_row_cannot_force_the_window_wide(qtbot, qt_theme_applied):
    """One row of everything must not become a window that will not narrow.

    Measured: with the tools, undo/redo and the detect button in it the
    row wants well over 1300px. If that were the screen's layout minimum,
    the window could not be made narrower than the row and the canvas
    would sit off the right edge of a 1366px laptop. The row scrolls
    instead, so it keeps its natural width without imposing it.
    """
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.layout().activate()

    bar = screen._tool_row.widget()
    natural = bar.sizeHint().width()
    assert natural > 600, "no real row of tools here to measure"
    assert screen._tool_row.minimumSizeHint().width() < natural, (
        "the row imposes its own width on everything above it")
    assert screen.layout().minimumSize().width() < natural, (
        f"the screen cannot be narrowed below {natural}px because of the row")


def test_the_row_is_hidden_until_there_is_something_to_edit(
        qtbot, qt_theme_applied, folder_2: Path):
    """A row of dead buttons over the empty state reads as a broken screen."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._body_stack.currentWidget() is screen._empty_state
    assert not screen._tool_row.isVisibleTo(screen)

    screen._open_folder(str(folder_2))
    assert screen._body_stack.currentWidget() is screen._body_splitter
    assert screen._tool_row.isVisibleTo(screen)


# ===========================================================================
# The settings toggle
# ===========================================================================

def test_the_settings_button_is_a_state_and_starts_lit(qtbot, qt_theme_applied):
    """"the toggle is a STATE" — checkable, and lit while the settings show."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._btn_settings.isCheckable()
    assert screen._btn_settings.isChecked()
    assert screen.settings_shown()
    assert screen._settings_scroll.isVisibleTo(screen._body_splitter)


def test_the_toggle_hides_and_shows_the_settings_as_one_group(
        qtbot, qt_theme_applied, folder_2: Path):
    """One press takes every settings card away, and one brings them back."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))
    # a card from each corner of the panel, so "as a group" means the group
    settings_widgets = (screen._brush_slider, screen._wand_pct,
                        screen._norm_hi, screen._filter_min_area,
                        screen._btn_otsu)
    for w in settings_widgets:
        assert screen._settings_scroll.isAncestorOf(w)

    screen._btn_settings.setChecked(False)
    assert not screen.settings_shown()
    assert not screen._settings_scroll.isVisibleTo(screen._body_splitter)
    for w in settings_widgets:
        assert not w.isVisibleTo(screen._body_splitter)

    screen._btn_settings.setChecked(True)
    assert screen.settings_shown()
    assert screen._settings_scroll.isVisibleTo(screen._body_splitter)
    for w in settings_widgets:
        assert w.isVisibleTo(screen._body_splitter)


def test_the_canvas_takes_the_space_the_settings_give_up(
        qtbot, qt_theme_applied, folder_2: Path):
    """Measured in pixels off a laid-out screen, not inferred from the code.

    Hiding the panel has to WIDEN THE CANVAS; a hide that left a gap where
    the settings were would pass every visibility assertion above and
    still be the wrong screen.
    """
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(1280, 800)
    screen.show()
    qtbot.waitExposed(screen)
    screen._open_folder(str(folder_2))
    qtbot.waitUntil(lambda: screen._canvas.width() > 1)

    wide_open = screen._body_splitter.sizes()
    assert wide_open[1] > 0
    before = screen._canvas.width()

    screen._btn_settings.setChecked(False)
    qtbot.waitUntil(lambda: screen._canvas.width() > before)
    assert screen._body_splitter.sizes()[1] == 0
    assert screen._canvas.width() >= before + wide_open[1] - 8, (
        "the canvas did not grow into the space the settings vacated")

    screen._btn_settings.setChecked(True)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[1] > 0)
    # back to the width it had, not to some default
    assert abs(screen._body_splitter.sizes()[1] - wide_open[1]) <= 8


def test_a_dragged_settings_width_survives_a_hide(
        qtbot, qt_theme_applied, folder_2: Path):
    """The panel returns where the user last put it, not at the default."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(1280, 800)
    screen.show()
    qtbot.waitExposed(screen)
    screen._open_folder(str(folder_2))
    qtbot.waitUntil(lambda: screen._canvas.width() > 1)

    total = sum(screen._body_splitter.sizes())
    screen._body_splitter.setSizes([total - 250, 250])
    qtbot.waitUntil(lambda: abs(screen._body_splitter.sizes()[1] - 250) <= 8)

    screen._btn_settings.setChecked(False)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[1] == 0)
    screen._btn_settings.setChecked(True)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[1] > 0)
    assert abs(screen._body_splitter.sizes()[1] - 250) <= 8, (
        f"came back at {screen._body_splitter.sizes()[1]}px, "
        f"not the 250px it was dragged to")


def test_hiding_the_settings_leaves_every_tool_reachable(
        qtbot, qt_theme_applied, folder_2: Path):
    """The tools, undo/redo and the toggle itself outlive the hide.

    The settings button has to stay visible when the settings are away or
    there is no way back, and the tools are in the row precisely so that
    putting the panel away does not put them away too.
    """
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))
    screen._btn_settings.setChecked(False)

    for mode, button in screen._mode_buttons.items():
        assert button.isVisibleTo(screen), f"{mode} went away with the settings"
        assert button.isEnabled()
    for button in (screen._btn_undo, screen._btn_redo,
                   screen._btn_reset_zoom, screen._btn_settings):
        assert button.isVisibleTo(screen)


def test_the_tools_still_drive_the_canvas_from_the_row(
        qtbot, qt_theme_applied, folder_2: Path):
    """Moving the buttons rearranged the screen; it did not unwire them."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))

    screen._mode_buttons[MODE_ZOOM].click()
    assert screen._canvas.mode == MODE_ZOOM
    assert screen._mode_buttons[MODE_ZOOM].isChecked()
    screen._mode_buttons[MODE_BRUSH].click()
    assert screen._canvas.mode == MODE_BRUSH
    assert not screen._mode_buttons[MODE_ZOOM].isChecked()
