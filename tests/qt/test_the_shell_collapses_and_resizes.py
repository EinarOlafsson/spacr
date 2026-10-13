"""Item 471, slice B: the app shell collapses, and every edge drags.

The maintainer, 2026-09-22: "it wuld be nice if live preview wuld ocupy the
entire screen (vertically) so if console, system and the buttons section below
system were all auto collapsed ... it would also be cool if we could collapse
the settings to the left. same should be true of the figures ... then if the
user wants they should be able to collapse the figures pannel and the live
preview pannel and uncollapse anything at any time ... whenever anything is
collapsed it should auto loch to the bottom of the container it is in (now for
example when i collapse the console it collapses to the middle of the
container.)" -- and later the same day: "whenever possible make the container
expandable or shrinkable by draging the edges."

One mechanism serves both: :mod:`spacr.qt.widgets.collapsible_splitter`.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication                # noqa: E402

from spacr.qt.widgets.collapsible_splitter import (       # noqa: E402
    EDGE, CollapsibleSplitter)


def _pump(n: int = 6) -> None:
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def mask_screen(qtbot):
    """Mask at a laptop's size, every fold it may remember opened first."""
    from spacr.qt.preferences import set_folded_panel
    from spacr.qt.screens.app_screen import AppScreen

    for name in ("Console", "System", "Actions", "Settings"):
        set_folded_panel(f"mask/{name}", False)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    _pump(10)
    yield screen
    for name in ("Console", "System", "Actions", "Settings"):
        set_folded_panel(f"mask/{name}", False)


class TestTheModuleScreen:

    def test_the_button_section_has_a_collapse_control(self, mask_screen):
        screen = mask_screen
        assert screen._actions_folder is not None
        assert screen._actions_section.isAncestorOf(screen._btn_run)
        screen._actions_folder.toggle()
        assert not screen._btn_run.isVisibleTo(screen)
        screen._actions_folder.toggle()
        assert screen._btn_run.isVisibleTo(screen)

    def test_console_system_and_buttons_each_resize_by_an_edge(
            self, mask_screen):
        split = mask_screen._runtime_splitter
        assert isinstance(split, CollapsibleSplitter)
        for widget in (mask_screen._console_wrap, mask_screen._usage_card,
                       mask_screen._actions_section,
                       mask_screen._live_preview_card,
                       mask_screen._figures_card):
            assert split.indexOf(widget) >= 0
        assert isinstance(mask_screen._body_splitter, CollapsibleSplitter)
        assert mask_screen._body_splitter.pane("Settings").mode == EDGE

    def test_the_live_preview_takes_the_height(self, mask_screen):
        screen = mask_screen
        split, body = screen._runtime_splitter, screen._body_splitter
        screen._preview_switch.setChecked(True)
        _pump()
        for name in ("Console", "System", "Actions"):
            assert split.is_collapsed(name), name
        assert body.is_collapsed("Settings")
        preview = split.sizes()[split.indexOf(screen._live_preview_card)]
        assert preview > 0.7 * split.height()
        screen._preview_switch.setChecked(False)
        _pump()
        for name in ("Console", "System", "Actions"):
            assert not split.is_collapsed(name), name
        assert not body.is_collapsed("Settings")

    def test_the_user_can_reopen_anything_and_it_stays(self, mask_screen):
        screen = mask_screen
        split = screen._runtime_splitter
        screen._preview_switch.setChecked(True)
        _pump()
        screen._console_folder.toggle()
        screen._figures_card.show()
        _pump()
        assert not split.is_collapsed("Console")
        assert screen.reveal_settings()
        assert not screen._body_splitter.is_collapsed("Settings")

    def test_the_figures_collapse_the_shell_too(self, mask_screen):
        screen = mask_screen
        screen._figures_card.show()
        _pump()
        assert screen._runtime_splitter.is_collapsed("Console")
        assert screen._body_splitter.is_collapsed("Settings")
        screen._figures_card.folder.toggle()
        assert screen._runtime_splitter.is_collapsed("Figures"), (
            "the user can collapse the figures panel itself")

    def test_registering_a_panel_does_not_build_it(self, qtbot):
        """Items 284/380: a hidden lazy panel is built when it is shown."""
        from spacr.qt.screens.app_screen import _LIVE_PREVIEW, AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        assert screen._part_is_owed(_LIVE_PREVIEW)
        screen._figures_card.show()
        _pump()
        assert screen._part_is_owed(_LIVE_PREVIEW), (
            "the figures took the height; the hidden live preview must "
            "still be unbuilt")

    def test_leaving_the_module_and_coming_back_is_a_new_view(
            self, mask_screen):
        """The view is one visit: hide (leave the module), show (return)."""
        screen = mask_screen
        split = screen._runtime_splitter
        screen._preview_switch.setChecked(True)
        _pump()
        screen._console_folder.toggle()
        assert not split.is_collapsed("Console")
        screen.hide()
        screen.show()
        _pump()
        assert split.is_collapsed("Console"), (
            "a pin lasts for the visit it was made in")

    def test_ctrl_f_opens_a_collapsed_settings_column(self, mask_screen):
        from spacr.qt.shortcuts import _focus_settings_search
        from spacr.qt.settings_search import install

        screen = mask_screen
        assert install(screen) is not None
        screen._body_splitter.set_collapsed("Settings", True, by_user=True)

        class _Window:
            class _stack:
                @staticmethod
                def currentWidget():
                    return screen

        _focus_settings_search(_Window())
        assert not screen._body_splitter.is_collapsed("Settings")


def test_a_registry_preview_takes_the_height_too(qtbot):
    """A preview mounted by :mod:`spacr.qt.preview_registry` is adopted."""
    from spacr.qt import preview_registry as pr
    from spacr.qt.preferences import set_folded_panel
    from spacr.qt.screens.app_screen import AppScreen

    for name in ("Console", "System", "Actions", "Settings"):
        set_folded_panel(f"cellpose_masks/{name}", False)
    screen = AppScreen("cellpose_masks")
    qtbot.addWidget(screen)
    host = pr.install(screen)
    if host is None:
        pytest.skip("cellpose_masks declares no registry preview here")
    split = screen._runtime_splitter
    pane = next((p for p in split.panes() if p.widget is host.card), None)
    assert pane is not None and pane.focus
    host.card.show()
    _pump()
    assert split.is_collapsed("Console")
    host.card.hide()
    _pump()
    assert not split.is_collapsed("Console")
