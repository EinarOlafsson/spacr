"""The real chooser may appear after human pointer travel exceeds 1.2 seconds."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

import capture_refresh
from capture_refresh import schedule_test_data_choice
from spacr.qt.widgets.test_data_chooser import TestDataChooser as ActualChooser


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('route', ['load', 'stream'])
def test_real_chooser_waits_for_delayed_opening_click(app, route):
    choice, frames = {}, []
    def capture(frame):
        frames.append(frame)
    capture.host = object()  # Use the existing host modal-wait branch, no host/GL created.
    button = QPushButton('Load test data')
    dialog = ActualChooser(button)
    button.clicked.connect(dialog.exec)
    button.show()
    app.processEvents()
    watchdog = schedule_test_data_choice(app, capture, app.processEvents, route, choice, timeout=5)
    try:
        QTest.qWait(1600)  # The old unconditional timer has already fired here.
        assert choice == {} and frames == []
        button.click()  # The genuine clicked signal opens the real modal nested event loop.
        assert choice == {'selected_route': route}
        assert dialog.chosen == route
        assert frames == ['02_data_choice_load', '02_data_choice_stream']
    finally:
        watchdog.stop()
        watchdog.deleteLater()
        dialog.close()
        button.close()
        button.deleteLater()
        app.processEvents()


def test_wrong_modal_fails_closed_without_selecting_a_route(app):
    choice = {}
    def capture(_):
        pytest.fail('unexpected modal must not be captured or selected')
    capture.host = object()
    dialog = QDialog()
    watchdog = schedule_test_data_choice(app, capture, app.processEvents, 'load', choice, timeout=3)
    try:
        dialog.exec()
        assert choice == {'error': 'Expected exactly one active test-data chooser'}
    finally:
        watchdog.stop()
        watchdog.deleteLater()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_timed_out_absent_chooser_still_rejects_a_later_modal(app):
    choice = {}
    def capture(_):
        pytest.fail('expired chooser must not select or capture')
    capture.host = object()
    watchdog = schedule_test_data_choice(app, capture, app.processEvents, 'load', choice, timeout=0.05)
    dialog = ActualChooser()
    try:
        QTest.qWait(200)
        assert 'Timed out' in choice['error']
        dialog.exec()  # The repeating watchdog survives the earlier no-modal timeout.
        assert dialog.chosen == '' and 'selected_route' not in choice
    finally:
        watchdog.stop()
        watchdog.deleteLater()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_cancelled_wait_cannot_reject_a_later_unrelated_modal(app):
    choice = {}
    capture = lambda _: None
    capture.host = object()
    watchdog = schedule_test_data_choice(app, capture, app.processEvents, 'load', choice, timeout=3)
    watchdog.stop()
    watchdog.deleteLater()
    QTest.qWait(20)
    dialog = QDialog()
    QTimer.singleShot(1500, dialog.accept)
    try:
        assert dialog.exec() == QDialog.Accepted
        assert choice == {}
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_real_clip_duration_does_not_expire_the_selection_budget(app, monkeypatch):
    clock, choice, frames = [100.0], {}, []
    monkeypatch.setattr(capture_refresh, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    def capture(frame):
        frames.append(frame)
        clock[0] += 16.167 if frame.endswith('load') else 20.7
        QTest.qWait(180)  # Deliver the real watchdog while simulated recording time advances.
    capture.host = object()
    dialog = ActualChooser()
    watchdog = schedule_test_data_choice(app, capture, app.processEvents, 'stream', choice)
    try:
        dialog.exec()
        assert clock[0] > 130  # Both actual clip lengths exceed the old global30s budget.
        assert choice == {'selected_route': 'stream'}
        assert frames == ['02_data_choice_load', '02_data_choice_stream']
    finally:
        watchdog.stop()
        watchdog.deleteLater()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
