"""Actual Qt result delivery before, during and after a physical Run click."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from capture_refresh import run_mask_preview


@pytest.mark.parametrize('mode', ['fast', 'deferred', 'failed', 'stale', 'empty'])
def test_public_result_subscription_precedes_physical_click_and_is_removed(mode):
    app = QApplication.instance() or QApplication([])
    events = []

    class Panel(QObject):
        preview_ready = Signal(object)
        def __init__(self):
            super().__init__()
            self._raw_masks = {'old': object()} if mode == 'stale' else {}
            self.running = False
            self._status = SimpleNamespace(text=lambda: 'actual preview status')
        def preview_running(self):
            return self.running
        @property
        def _worker(self):
            raise AssertionError('The recorder must not attach to a lazy worker')

    panel, button = Panel(), QPushButton('Run preview')
    def completed():
        panel.running = False
        panel._raw_masks = {} if mode in ('failed', 'empty') else {'cell': object()}
        panel.preview_ready.emit(None if mode == 'failed' else panel._raw_masks)
        events.append('result delivered')
    def requested():
        events.append('actual Run click')
        assert panel.receivers('2preview_ready(PyObject)') == 1
        if mode == 'stale':
            return
        if mode == 'deferred':
            panel.running = True
            QTimer.singleShot(10, completed)
        else:
            completed()
    button.clicked.connect(requested)
    try:
        operation = lambda: run_mask_preview(panel,
            lambda: QTest.mouseClick(button, Qt.LeftButton),
            lambda seconds: QTest.qWait(max(1, int(seconds * 1000))),
            0.05, lambda: events.append('settings clip ended'))
        if mode == 'stale':
            with pytest.raises(TimeoutError, match='Preview not finished'):
                operation()
        elif mode in ('failed', 'empty'):
            with pytest.raises(RuntimeError, match='Preview failed'):
                operation()
        else:
            operation()
            assert panel._raw_masks and not panel.preview_running()
        assert events[0] == 'actual Run click'
        assert 'settings clip ended' in events
        assert panel.receivers('2preview_ready(PyObject)') == 0
    finally:
        button.close()
        button.deleteLater()
        panel.deleteLater()
        app.processEvents()
