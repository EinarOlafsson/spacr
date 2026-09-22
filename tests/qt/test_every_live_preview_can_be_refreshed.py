"""Item 464 and item 452 reopened: the Refresh button, and Plaque's Live switch.

2026-09-21, the maintainer: "there should also be a refresh button in every
live preview that the user can pres and if pressed it checks the path and
reloads", and for Plaque Assay "the button should be to the left of the AI
button and work like the other live previews".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QToolButton

sys.path.insert(0, str(Path(__file__).parent))

import test_all_module_smoke as smoke  # noqa: E402

from spacr.qt.app import MainWindow  # noqa: E402
from spacr.qt import preview_registry  # noqa: E402
from spacr.qt.widgets import preview_refresh  # noqa: E402


def _screen(qtbot, key):
    screen = MainWindow._build_screen(smoke._FactoryHost(), key)
    qtbot.addWidget(screen)
    return screen


def _refresh_buttons(screen):
    return [b for b in screen.findChildren(QToolButton)
            if b.objectName() == "PreviewRefreshButton"]


@pytest.mark.parametrize("key", ["mask", "analyze_plaques", "measure",
                                 "timelapse", "motility"])
def test_every_screen_built_preview_has_one_refresh_button(qtbot, key):
    assert len(_refresh_buttons(_screen(qtbot, key))) == 1


def test_a_registry_mounted_preview_has_one_too(qtbot):
    screen = _screen(qtbot, "cellpose_masks")
    assert preview_registry.install(screen) is not None
    assert len(_refresh_buttons(screen)) == 1


def test_plaques_live_switch_rides_on_the_preview_card(qtbot):
    """It sat just left of AI in the Run row until 2026-09-22. A preview now
    takes the whole height and folds that row under it (item 471), so the
    switch that turns the preview off moved onto the card it controls,
    beside Refresh."""
    screen = _screen(qtbot, "analyze_plaques")
    switch = screen._preview_switch
    card = getattr(screen, screen._preview_card_attr)
    parents, node = [], switch
    while node is not None:
        parents.append(node)
        node = node.parent()
    assert card in parents
    assert switch.text().strip() == "Live"
    assert _refresh_buttons(screen), "Refresh is on the same card"


def test_plaque_is_not_mounted_a_second_time_by_the_registry(qtbot):
    screen = _screen(qtbot, "analyze_plaques")
    assert preview_registry.install(screen) is None
    assert len(_refresh_buttons(screen)) == 1


def test_refresh_reports_a_missing_path_and_starts_nothing(qtbot):
    screen = _screen(qtbot, "analyze_plaques")
    said = []
    screen._console.append_stdout = said.append
    screen._settings_model.set_value_for_key("src", "/no/such/folder")
    assert preview_refresh.reload_from_src(screen, screen._live_preview) is False
    assert "does not exist" in said[-1]


def test_refresh_reloads_through_the_panels_own_loader(qtbot, tmp_path):
    screen = _screen(qtbot, "analyze_plaques")
    called = []
    screen._live_preview.load_source_async = lambda src, **kw: called.append(src) or True
    screen._settings_model.set_value_for_key("src", str(tmp_path))
    _refresh_buttons(screen)[0].click()
    assert called == [str(tmp_path)]


def test_the_plaque_preview_translates_the_names_it_pushes_back(qtbot):
    screen = _screen(qtbot, "analyze_plaques")
    written = {}
    screen._settings_model.set_value_for_key = lambda k, v: written.__setitem__(k, v)
    screen._propagate_live_settings({"cell_diameter": 42, "cell_channel": 1})
    assert written == {"diameter": 42}
