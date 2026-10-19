"""The uninstall dialog's scope and tick edges, and a close while its popup is open."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from spacr import install_cleanup
from spacr.qt import install_consent as consent

pytestmark = pytest.mark.qt


@pytest.fixture
def preview(tmp_path):
    rows = []
    for key, current, selectable, kind in [("current", True, True, "installer"),
                                            ("shared", False, True, "environment"),
                                            ("checkout", False, False, "checkout")]:
        path = str(tmp_path / key)
        rows.append({"key": key, "current": current, "selectable": selectable,
                     "record": {"root": path, "version": "1.5.0.7", "kind": kind},
                     "paths": [path, path + "/spacr"], "reason": ""})
    return {"operation_id": "op", "snapshot_id": "snap",
            "categories": list(install_cleanup._UNINSTALL_CATEGORIES),
            "installations": rows, "cleanup": []}


@pytest.fixture
def dialog(qtbot, preview):
    box = consent._UninstallDialog(preview)
    qtbot.addWidget(box)
    return box


def test_choosing_the_ticked_scope_keeps_the_ticks_the_user_made(dialog):
    _, shared = dialog._installation_items["shared"]
    dialog._scope.setCurrentIndex(dialog._scope.findData("all"))
    shared.setCheckState(0, Qt.Unchecked)
    assert dialog._scope.currentData() == "custom"
    before = dialog._ticked_keys()
    dialog._show_scope()
    assert dialog._ticked_keys() == before == ["current"]


def test_a_tick_change_in_a_child_row_or_another_column_is_ignored(dialog):
    dialog._scope.setCurrentIndex(dialog._scope.findData("all"))
    _, current = dialog._installation_items["current"]
    dialog._ticked(current, 2)
    assert dialog._scope.currentData() == "all"
    child = SimpleNamespace(parent=lambda: object())
    dialog._ticked(child, 0)
    assert dialog._scope.currentData() == "all"


def test_a_tick_that_still_matches_the_named_scope_leaves_the_scope_alone(dialog):
    dialog._scope.setCurrentIndex(dialog._scope.findData("all"))
    _, current = dialog._installation_items["current"]
    dialog._ticked(current, 0)
    assert dialog._scope.currentData() == "all"
    assert dialog._ticked_keys() == ["current", "shared"]


def test_closing_while_the_confirmation_popup_is_open_rejects_it_and_cancels(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    owner = consent._UninstallCoordinator(window)
    rejected = []
    owner._open_dialog = SimpleNamespace(reject=lambda: rejected.append(True))
    assert owner.approve_close() is False
    assert rejected == [True]
    assert owner.close_after_cancel is True
    owner.finished = True
    owner.progress.close()
    owner.timer.stop()
    owner.setParent(None)
    owner.deleteLater()
    window.close()
