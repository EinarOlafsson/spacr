"""Shared preview settings survive all dialog exits and native disposal."""

import pytest

pytest.importorskip('PySide6')

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QDialog
from shiboken6 import isValid

from spacr.qt.widgets.live_preview import LivePreviewPanel

pytestmark = pytest.mark.qt


def _exit_dialog(dialog, qtbot, method):
    if method == 'escape':
        qtbot.keyClick(dialog, Qt.Key_Escape)
    elif method == 'done':
        dialog.done(QDialog.Accepted)
    else:
        getattr(dialog, method)()


@pytest.mark.parametrize('method', ['close', 'escape', 'accept', 'reject', 'done'])
def test_shared_controls_survive_dialog_disposal_and_keep_values(qtbot, method):
    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    tracked = list(dialog._managed_widgets()) + list(panel._organelle_widgets.values())
    diameter = panel._diameter.value() + 3
    minimum = panel._organelle_widgets['min_size'].value() + 2
    panel._diameter.setValue(diameter)
    panel._organelle_widgets['min_size'].setValue(minimum)
    _exit_dialog(dialog, qtbot, method)
    assert panel._live_settings_dialog is None
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(dialog)
    assert isValid(panel)
    assert all(isValid(widget) for widget in tracked), 'Dialog destroyed shared preview controls'
    assert panel._diameter.value() == diameter
    assert panel._organelle_widgets['min_size'].value() == minimum
    panel.open_live_settings()
    reopened = panel._live_settings_dialog
    try:
        assert reopened.isVisible()
        assert panel._diameter.value() == diameter
        assert panel._organelle_widgets['min_size'].value() == minimum
    finally:
        reopened.close()


def test_a_completed_dialog_cannot_take_controls_from_its_replacement(qtbot):
    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.open_live_settings()
    completed = panel._live_settings_dialog
    completed.close()
    panel.open_live_settings()
    replacement = panel._live_settings_dialog
    try:
        completed.done(QDialog.Accepted)
        assert panel._live_settings_dialog is replacement
        assert replacement.isVisible()
        assert replacement.isAncestorOf(panel._diameter)
        assert replacement.isAncestorOf(panel._organelle_widgets['min_size'])
    finally:
        replacement.close()


@pytest.mark.parametrize('method', ['close', 'escape', 'accept', 'reject', 'done'])
def test_closed_dialog_does_not_propagate_through_an_unchecked_new_dialog(
        qtbot, monkeypatch, method):
    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    calls = []
    monkeypatch.setattr(panel, 'propagate_settings', lambda *_: calls.append('propagated'))
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    dialog._propagate_btn.setChecked(True)
    assert calls == ['propagated']
    calls.clear()
    _exit_dialog(dialog, qtbot, method)
    panel._diameter.setValue(panel._diameter.value() + 1)
    assert calls == [], 'Closed dialog left its propagation subscriptions active'
    panel.open_live_settings()
    reopened = panel._live_settings_dialog
    try:
        assert not reopened._propagate_btn.isChecked()
        panel._diameter.setValue(panel._diameter.value() + 1)
        assert calls == []
        reopened._propagate_btn.setChecked(True)
        assert calls == ['propagated']
        calls.clear()
        panel._diameter.setValue(panel._diameter.value() + 1)
        assert calls == ['propagated']
    finally:
        reopened.close()
