"""Recorder completion must happen outside the live search's Qt signal stack."""
from pathlib import Path
import gc
import sys
from types import SimpleNamespace
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton
import capture_ui
from capture_barcode_search import record_live_search


@pytest.mark.parametrize('queued', [False, True])
def test_first_partial_ends_once_after_real_qt_signal_delivery(monkeypatch, queued):
    app = QApplication.instance() or QApplication([])
    state = dict(in_signal=False, ends=0, searching=False, unsafe_ends=[], counts_in_signal=[])
    class Panel(QObject):
        search_updated = Signal(object)
        search_finished = Signal(object)
        def is_searching(self): return state['searching']
    panel = Panel()
    panel.search_button = QPushButton('Find barcodes')
    panel.cancel_button = QPushButton('Cancel')
    panel.search_button.show()
    panel.cancel_button.show()
    first = SimpleNamespace(reads=1000, complete=False)
    last = SimpleNamespace(reads=20000, complete=False)
    def deliver():
        state['in_signal'] = True
        try:
            panel.search_updated.emit(first)
            panel.search_updated.emit(last)
            panel.search_finished.emit(last)
            state['counts_in_signal'].append(state['ends'])
        finally:
            state['in_signal'] = False
            state['searching'] = False
    def clicked():
        state['searching'] = True
        if queued: QTimer.singleShot(1, deliver)
        else: deliver()
    panel.search_button.clicked.connect(clicked)
    class Capture:
        host = object()
        def begin(self, name):
            assert name == '06_live_search_progress'
            return True
        def end(self):
            if state['in_signal']:
                state['unsafe_ends'].append(True)
                return  # Never exercise the crashing nested collection path in this test.
            state['ends'] += 1
            gc.collect()  # Exercise collection only after signal delivery, without unsafe nested teardown.
    monkeypatch.setattr(capture_ui, 'wait', lambda capture, seconds, settle: None)
    try:
        reports, partial, finished = record_live_search(panel, Capture(), lambda _: QTest.qWait(5), 2)
        assert not state['unsafe_ends'] and state['counts_in_signal'] == [0]
        assert state['ends'] == 1 and partial == [True]
        assert reports == [{'reads': 1000, 'complete': False}, {'reads': 20000, 'complete': False}]
        assert finished == [last]
        # Cleanup removes the recorder callbacks: subsequent signals cannot end it again.
        panel.search_updated.emit(last)
        panel.search_finished.emit(last)
        assert state['ends'] == 1 and len(reports) == 2 and finished == [last]
    finally:
        for button in (panel.search_button, panel.cancel_button):
            button.close()
            button.deleteLater()
        app.processEvents()
