"""Semantic Run/Propagate and Stop/Close button behavior."""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from spacr.qt.button_roles import (
    action_role, install_button_roles, set_button_busy,
)


def test_action_role_classifies_action_verbs():
    assert action_role("Run") == "positive"
    assert action_role("Run search") == "positive"
    assert action_role("Propagate settings") == "positive"
    assert action_role("Stop") == "negative"
    assert action_role("Close") == "negative"
    assert action_role("Remove selected") == "negative"
    assert action_role("Choose folder…") is None


def test_buttons_created_after_install_are_tagged(qapp, qtbot):
    install_button_roles(qapp)
    run = QPushButton("Run preview")
    close = QPushButton("Close")
    neutral = QPushButton("Choose folder…")
    for button in (run, close, neutral):
        qtbot.addWidget(button)
        button.show()
    qtbot.wait(1)

    assert run.property("buttonActionRole") == "positive"
    assert close.property("buttonActionRole") == "negative"
    assert neutral.property("buttonActionRole") is None


def test_disabled_run_stays_busy_until_reenabled(qapp, qtbot):
    install_button_roles(qapp)
    run = QPushButton("Run")
    qtbot.addWidget(run)
    run.show()
    qtbot.wait(1)
    run.clicked.connect(lambda: run.setEnabled(False))

    run.click()
    qtbot.wait(1)
    assert run.property("buttonActionBusy") is True

    run.setEnabled(True)
    qtbot.wait(1)
    assert run.property("buttonActionBusy") is False


def test_non_run_actions_clear_after_their_handler(qapp, qtbot):
    install_button_roles(qapp)
    propagate = QPushButton("Propagate settings")
    stop = QPushButton("Stop")
    for button in (propagate, stop):
        qtbot.addWidget(button)
        button.show()
        button.click()
    qtbot.wait(1)
    assert propagate.property("buttonActionBusy") is False
    assert stop.property("buttonActionBusy") is False

    set_button_busy(stop, True)
    assert stop.property("buttonActionBusy") is True
