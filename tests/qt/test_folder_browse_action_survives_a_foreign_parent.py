"""A field's folder button keeps working when PySide re-parents the field's wrapper.

The Overview's object-mask folder button was cached as a Python wrapper on
its line edit. The line edit creates that action in C++, so the wrapper is
only as valid as the Python parent PySide's ownership glue gives the field:
in the mounted Overview form the field's wrapper hung under a settings-row
label, and when that label was rebuilt PySide invalidated the action's
wrapper while the action was still on the field. Triggering the cached
wrapper raised "Internal C++ object (QAction) already deleted".
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QLabel, QLineEdit, QWidget
from shiboken6 import isValid

from spacr.qt.screens.settings_model import _add_folder_browse_action, _folder_browse_action

pytestmark = pytest.mark.qt


def test_the_folder_action_is_found_alive_after_the_fields_python_parent_goes(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    edit = QLineEdit(host)
    chosen = []
    first = _add_folder_browse_action(edit, lambda: chosen.append(True))
    label = QLabel("Objects", host)
    label.setBuddy(edit)
    label.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()

    assert not isValid(first)
    action = _folder_browse_action(edit)
    assert action is not None and isValid(action)
    assert _add_folder_browse_action(edit, lambda: chosen.append(False)) is action
    assert [a.objectName() for a in edit.actions()].count("SourceFolderBrowse") == 1
    action.trigger()
    assert chosen == [True]
