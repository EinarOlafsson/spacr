"""Example annotations must preserve both pixels and existing database labels."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'capture_annotate', Path(__file__).resolve().parents[1] / 'capture_annotate.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def test_unchanged_originals_are_accepted():
    recorder.require_preserved({'rows': 'same', 'pixels': ['a', 'b']},
                               {'pixels': ['a', 'b'], 'rows': 'same'})


@pytest.mark.parametrize('after', [
    {'rows': 'changed', 'pixels': ['a', 'b']},
    {'rows': 'same', 'pixels': ['a', 'changed']},
    {'rows': 'same'},
])
def test_changed_originals_are_rejected(after):
    with pytest.raises(RuntimeError, match='changed original'):
        recorder.require_preserved({'rows': 'same', 'pixels': ['a', 'b']}, after)


@pytest.mark.parametrize('delete_on_close', [True, False])
def test_actual_report_dismissal_handles_qt_delete_on_close(monkeypatch, delete_on_close):
    from types import SimpleNamespace
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout
    from shiboken6 import isValid

    app = QApplication.instance() or QApplication([])
    report = QDialog()
    report.setAttribute(Qt.WA_DeleteOnClose, delete_on_close)
    button = QPushButton('Close', report)
    QVBoxLayout(report).addWidget(button)
    button.clicked.connect(report.close)
    report.show()
    app.processEvents()
    clicks = []

    def click(target):
        assert target is button and target.isVisible()
        clicks.append(target.text())
        QTest.mouseClick(target, Qt.LeftButton)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def wait_until(capture, predicate, timeout, message):
        assert predicate(), message

    monkeypatch.setattr(recorder.capture_ui, '_h', lambda: SimpleNamespace(
        click=click, wait_until=wait_until))
    capture = lambda _: None
    capture.host = object()  # Actual host dismissal branch with a real Qt button; no native host.
    try:
        recorder.close_coverage_report(capture, report)
        assert clicks == ['Close']
        assert not isValid(report) if delete_on_close else not report.isVisible()
    finally:
        if isValid(report):
            report.close()
            report.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('already_deleted', [True, False])
def test_report_timeout_fallback_never_reads_a_deleted_wrapper(monkeypatch, already_deleted):
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtWidgets import QApplication, QDialog
    from shiboken6 import isValid

    app = QApplication.instance() or QApplication([])
    report = QDialog()
    report.setAttribute(Qt.WA_DeleteOnClose, already_deleted)
    report.show()
    app.processEvents()

    def timeout(capture, target):
        assert target is report
        if already_deleted:
            target.close()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        raise TimeoutError('actual dismissal timed out')

    monkeypatch.setattr(recorder.capture_ui, 'close_dialog', timeout)
    try:
        recorder.close_coverage_report(None, report)
        assert not isValid(report) if already_deleted else not report.isVisible()
    finally:
        if isValid(report):
            report.close()
            report.deleteLater()
        app.processEvents()
