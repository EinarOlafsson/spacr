"""The Live switch sits on the preview card it controls.

2026-09-22, the maintainer: while a preview takes the whole height, item
471's auto-collapse folds the Actions section -- and the switch that turns
the preview off was inside it, behind the thing it controls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("key", ["mask", "measure", "analyze_plaques",
                                 "timelapse", "motility"])
def test_the_switch_is_on_the_card_not_in_the_run_row(qtbot, key):
    from PySide6.QtCore import Qt
    from spacr.qt.app import MainWindow

    from .test_all_module_smoke import _FactoryHost

    screen = MainWindow._build_screen(_FactoryHost(), key)
    qtbot.addWidget(screen)
    switch = getattr(screen, "_preview_switch", None)
    if switch is None:
        pytest.skip(f"{key} offers no live preview switch")
    card = getattr(screen, getattr(screen, "_preview_card_attr"), None)
    assert card is not None
    assert switch.text().strip() == "Live"
    screen.resize(1366, 768)
    screen.show()
    qtbot.waitExposed(screen)
    screen._actions_folder.set_shut(True, by_user=False)
    for _ in range(2):
        assert not switch.isChecked()
        assert card.isHidden()
        assert switch.isVisible(), "a closed preview must leave Live reachable"
        assert not screen._actions_body.isAncestorOf(switch)
        qtbot.mouseClick(switch, Qt.LeftButton)
        qtbot.waitUntil(card.isVisible)
        assert switch.isChecked()
        assert switch.isVisible()
        assert card.isAncestorOf(switch), "Live travels with the open preview"
        qtbot.mouseClick(switch, Qt.LeftButton)
        qtbot.waitUntil(card.isHidden)
    assert switch.isVisible(), "Live must still be available after closing"
