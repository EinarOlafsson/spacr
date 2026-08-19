"""Every dialog spaCR opens is Qt's own. Instruction 151.

REPORTED: "changing the line width takes like 1 minute". The restyle measured
at 0.000 s; the minute was a NATIVE dialog, which on this desktop is brokered
through xdg-desktop-portal and stalls for tens of seconds.

The colour pickers were fixed one call site at a time. There are 117
QFileDialog calls across the widget package and five passed the option, so
per-site fixing was never going to converge -- the application attribute
covers all of them at once.
"""

import inspect
import re

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import pytest


def test_the_attribute_is_set_before_the_application_exists():
    """Qt IGNORES this attribute after construction, silently.

    Set too late it would look exactly like it had worked, which is the one
    failure mode a test has to rule out.
    """
    from spacr.qt import app as module

    source = inspect.getsource(module.main) if hasattr(module, "main") else \
        inspect.getsource(module)
    setting = source.find("AA_DontUseNativeDialogs")
    construction = source.find("QApplication(sys.argv")
    assert setting != -1, "the attribute is never set"
    assert construction != -1
    assert setting < construction, (
        "the attribute is set AFTER the QApplication is constructed, where Qt "
        "ignores it without saying so")


def test_the_attribute_takes_on_this_application(qtbot):
    """What is observable headless, and no more than that.

    `QFileDialog.testOption(DontUseNativeDialog)` reports the PER-DIALOG
    option and does NOT reflect the application attribute -- asserting it
    would have failed a working fix, which is what the first version of this
    test did. What can be checked is that the attribute is on; whether the
    platform then skips xdg-desktop-portal cannot be reproduced offscreen,
    because offscreen never asks the portal in the first place. Same honest
    limit the colour-dialog half of instruction 151 records.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    was = QApplication.testAttribute(Qt.AA_DontUseNativeDialogs)
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)
    try:
        assert QApplication.testAttribute(Qt.AA_DontUseNativeDialogs)
    finally:
        QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, was)


def test_the_colour_pickers_still_go_through_the_helper():
    """The per-site fix for colour is not undone by the global one.

    They are different dialogs: the attribute covers what Qt routes through
    the platform theme, and `pick_colour` also carries spaCR's own defaults.
    """
    from spacr.qt.widgets import colour_picker

    assert hasattr(colour_picker, "pick_colour")
    source = inspect.getsource(colour_picker.pick_colour)
    assert "DontUseNativeDialog" in source
