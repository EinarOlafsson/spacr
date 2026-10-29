"""The Make Masks TOOLS group, and the settings behind their collapse arrow.

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
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from spacr.qt.widgets.flow import FlowLayout

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
    assert isinstance(row, FlowLayout)
    assert screen._mode_buttons, "the screen built no tools at all"
    for mode, button in screen._mode_buttons.items():
        if mode == mm.MODE_RULER:
            controls = screen.findChild(QWidget, "MakeMasksViewControls")
            assert controls.layout().indexOf(button) >= 0
            continue
        if mode == mm.MODE_LABEL_TIME:
            assert screen._label_tracking_controls.layout().indexOf(button) >= 0
            assert row.indexOf(screen._label_tool_group) >= 0
            continue
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
                 for m, _label, _icon in mm.TOOL_MODES if m != mm.MODE_RULER]
    assert positions == sorted(positions)
    assert positions[0] == 0, "the first tool does not open the TOOLS group"


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


def test_wand_has_one_button_and_its_modifier_help_follows_the_selected_tool(qtbot, qt_theme_applied):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    wand = screen._mode_buttons[mm.MODE_WAND_ADD]
    assert mm.MODE_WAND_ERASE not in screen._mode_buttons
    assert [button for button in screen._mode_buttons.values()
            if button.text().startswith('Wand')] == [wand]
    assert wand.text() == 'Wand'
    assert 'Hold Ctrl' in wand.toolTip()
    screen._set_mode(mm.MODE_WAND_ADD)
    assert wand.isChecked()
    assert screen._shortcut_rows['Ctrl + left click'][1].text() == 'Remove the intensity region'
    screen._set_mode(MODE_BRUSH)
    assert screen._shortcut_rows['Ctrl + left click'][1].text() == 'Split the object at its waist'


def test_a_new_tool_is_enabled_with_the_rest_when_a_folder_opens(
        qtbot, qt_theme_applied, monkeypatch, folder_2: Path):
    """_sync_button_states reads the row, so it cannot miss a new tool."""
    monkeypatch.setattr(mm, "MODE_LASSO", "lasso", raising=False)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    assert screen._mode_buttons["lasso"].isEnabled()
    assert screen._mode_buttons[MODE_BRUSH].isEnabled()


def test_an_action_button_joins_the_same_row(qtbot, qt_theme_applied):
    """A non-mode button lands at the end of TOOLS."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    button = QPushButton("Cellpose-SAM detect")
    assert screen.add_toolbar_action(button) is button

    row = screen._tool_row_layout
    assert row.indexOf(button) == row.count() - 1
    assert screen._tools_group.isAncestorOf(button)


def test_the_row_cannot_force_the_window_wide(qtbot, qt_theme_applied):
    """The TOOLS group wraps, so its minimum is one button, not the row."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.layout().activate()

    natural = sum(b.sizeHint().width() for b in screen._mode_buttons.values())
    assert natural > 600, "no real row of tools here to measure"
    assert screen._tool_row.minimumSizeHint().width() < natural
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
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    assert screen._body_stack.currentWidget() is screen._image_page
    assert screen._tool_row.isVisibleTo(screen)


# ===========================================================================
# The settings toggle
# ===========================================================================

def test_the_settings_start_shown_and_have_no_button(qtbot, qt_theme_applied):
    """The collapse arrow hides the settings; there is no Settings button."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert not hasattr(screen, "_btn_settings")
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
                        screen._norm_hi, screen._filter_add,
                        screen._btn_otsu)
    for w in settings_widgets:
        assert screen._settings_scroll.isAncestorOf(w)

    screen._set_settings_shown(False)
    assert not screen.settings_shown()

    screen._set_settings_shown(True)
    assert screen.settings_shown()
    assert screen._settings_scroll.isVisibleTo(screen._body_splitter)
    for w in settings_widgets:
        assert w.isVisibleTo(screen._body_splitter)


#: How many pixels of the width the settings give up go to the splitter's
#: grips and margins rather than to the canvas. Measured, and a CONSTANT
#: rather than a share of the panel: see the note inside the test.
_SPLITTER_OVERHEAD = 60


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
    assert wide_open[0] > 0, "the settings are the left pane (item 419)"
    before = screen._canvas.width()

    screen._set_settings_shown(False)
    qtbot.waitUntil(lambda: screen._canvas.width() > before)
    assert screen._body_splitter.sizes()[0] == 0
    # MOST of it, not all: the shortcut list became a pane of its own on
    # 2026-09-22 so it can be hidden too, and its splitter's grips take a
    # FIXED few dozen pixels of the width the settings gave up. This used to
    # allow 25 % of the panel's width for that, which was a proportion
    # standing in for a constant and broke the day the panel got narrower
    # (item 473 folded three settings categories into one and the panel went
    # from 245 px to 194 px; the overhead stayed at 52). The shortcut list
    # itself must not grow -- the room is the canvas's.
    shortcuts = screen._view_pane.pane("Shortcuts")
    # Measured once the splitters settle: the side column (with its AI chat
    # box and System, 707) is laid out a turn after the canvas grows.
    qtbot.waitUntil(lambda: screen._shortcut_panel.width() <= shortcuts.extent,
                    timeout=3000)
    assert screen._canvas.width() >= (
        before + wide_open[0] - _SPLITTER_OVERHEAD), (
        "the canvas did not grow into the space the settings vacated")
    assert screen._shortcut_panel.width() <= shortcuts.extent, (
        "the shortcut list took the room instead of the canvas")

    screen._set_settings_shown(True)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[0] > 0)
    # back to the width it had, not to some default
    assert abs(screen._body_splitter.sizes()[0] - wide_open[0]) <= 8


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
    screen._body_splitter.setSizes([250, total - 250])
    qtbot.waitUntil(lambda: abs(screen._body_splitter.sizes()[0] - 250) <= 8)

    screen._set_settings_shown(False)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[0] == 0)
    screen._set_settings_shown(True)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[0] > 0)
    assert abs(screen._body_splitter.sizes()[0] - 250) <= 8, (
        f"came back at {screen._body_splitter.sizes()[0]}px, "
        f"not the 250px it was dragged to")


def test_hiding_the_settings_leaves_every_tool_reachable(
        qtbot, qt_theme_applied, monkeypatch, folder_2: Path):
    """The tools and Reset zoom outlive the hide.

    The tools are in the row precisely so that putting the panel away does
    not put them away too.
    """
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    screen._set_settings_shown(False)
    assert not screen._mode_buttons[mm.MODE_LABEL_TIME].isEnabled()
    monkeypatch.setattr(screen, "_confirm", lambda *args: True)
    screen._btn_label_time.click()

    for mode, button in screen._mode_buttons.items():
        if mode == mm.MODE_PICK:
            assert not button.isVisibleTo(screen), "Pick showed with no mask layers"
            continue
        assert button.isVisibleTo(screen), f"{mode} went away with the settings"
        assert button.isEnabled()
    assert screen._btn_reset_zoom.isVisibleTo(screen)


def test_the_tools_still_drive_the_canvas_from_the_row(
        qtbot, qt_theme_applied, folder_2: Path):
    """Moving the buttons rearranged the screen; it did not unwire them."""
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(folder_2))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)

    screen._mode_buttons[MODE_ZOOM].click()
    assert screen._canvas.mode == MODE_ZOOM
    assert screen._mode_buttons[MODE_ZOOM].isChecked()
    screen._mode_buttons[MODE_BRUSH].click()
    assert screen._canvas.mode == MODE_BRUSH
    assert not screen._mode_buttons[MODE_ZOOM].isChecked()


@pytest.mark.parametrize("size", [(2560, 1440), (1920, 1080), (1366, 768)])
def test_every_group_is_whole_on_screen(
        qtbot, qt_theme_applied, folder_2: Path, size):
    """No group overlaps another, and every button is drawn whole."""
    from PySide6.QtCore import QRect

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(*size)
    screen.show()
    qtbot.waitExposed(screen)
    screen._open_folder(str(folder_2))
    qtbot.waitUntil(lambda: screen._tool_row.isVisible())
    screen.layout().activate()
    QApplication.processEvents()

    def where(widget):
        return QRect(widget.mapTo(screen, QPoint(0, 0)), widget.size())

    groups = (screen._tools_group, screen._annotation_group,
              screen._io_group, screen._navigation_group)
    for i, group in enumerate(groups):
        for other in groups[i + 1:]:
            assert not where(group).intersects(where(other))
    for button in (screen._btn_prev, screen._btn_next, screen._btn_keep,
                   screen._btn_magnifier):
        assert button.isVisible()
        assert button.visibleRegion().boundingRect() == button.rect(), (
            f"{button.text()} is clipped at {size}")


def test_a_fixed_side_pane_uses_its_extent_instead_of_its_wrapping_size_hint(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.collapsible_splitter import CollapsibleSplitter, EDGE

    side = QLabel("The shortcut list and console can wrap their text. " * 8)
    side.setWordWrap(True)
    minimum = side.minimumSizeHint().width()
    natural = side.sizeHint().width()
    extent = (minimum + natural) // 2
    assert minimum < extent < natural
    split = CollapsibleSplitter(Qt.Horizontal)
    qtbot.addWidget(split)
    split.add_pane(QLabel("Canvas"), "Canvas", stretch=1)
    pane = split.add_pane(side, "Shortcuts", mode=EDGE, stretch=0,
                          extent=extent, minimum=minimum)
    split.resize(natural * 3, 300)
    split.show()
    qtbot.waitExposed(split)
    QApplication.processEvents()
    assert side.width() == pane.extent == extent
    split.resize(natural * 4, 300)
    QApplication.processEvents()
    assert side.width() == extent
