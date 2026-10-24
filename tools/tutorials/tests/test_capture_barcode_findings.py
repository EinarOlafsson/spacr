"""A coarse physical table scroll must not leave the final verdict rows clipped."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QAbstractItemView
import capture_ui
from capture_ui_drv_a import table_row_to_top
from capture_barcode_search import show_findings_rows


def test_real_table_coarse_scroll_is_finished_by_genuine_wheel_input(monkeypatch):
    app = QApplication.instance() or QApplication([])
    table = QTableWidget(12, 2)
    table.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
    table.resize(500, 190)
    for row in range(12):
        for col in range(2):
            table.setItem(row, col, QTableWidgetItem(f'verdict {row} {col}'))
        table.setRowHeight(row, 30)
    table.show()
    QTest.qWait(30)
    table.verticalScrollBar().setSingleStep(3)
    wheels = []
    class QtInput:
        def wheel(self, capture, port, notches, pos=None):
            wheels.append(notches)
            point = pos or port.rect().center()
            event = QWheelEvent(QPointF(point), QPointF(port.mapToGlobal(point)),
                QPoint(), QPoint(0, -120 * notches), Qt.NoButton, Qt.NoModifier,
                Qt.NoScrollPhase, False)
            QApplication.sendEvent(port, event)
            QTest.qWait(5)
        def scroll_bar_wheel(self, capture, scroll, direction):
            self.wheel(capture, scroll.viewport(), direction)
        def drag_scroll_bar(self, capture, bar, value):
            raise AssertionError('This regression must exercise real wheel delivery')
    monkeypatch.setattr(capture_ui, '_h', lambda: QtInput())
    capture = SimpleNamespace(host=object())
    rows = [8, 9, 10, 11]
    values = [[table.item(row, col).text() for col in range(2)] for row in range(12)]
    contained = lambda: all(table.viewport().rect().contains(
        table.visualItemRect(table.item(row, 0))) for row in rows)
    try:
        table_row_to_top(capture, table, rows[0])
        assert not contained()  # The unchanged approximate helper stops up to three rows early.
        show_findings_rows(capture, table, rows, lambda _: QTest.qWait(5))
        assert contained() and wheels
        assert values == [[table.item(row, col).text() for col in range(2)] for row in range(12)]
    finally:
        table.close()
        table.deleteLater()
        app.processEvents()


def test_selected_proposal_scene_contains_real_apply_and_verified_readback(monkeypatch):
    import capture_barcode_search as driver
    from PySide6.QtWidgets import QPushButton
    app = QApplication.instance() or QApplication([])
    button = QPushButton('Apply these settings')
    button.show()
    settings = {'window_length': 120}
    events, active = [], []
    def apply():
        assert active == ['09_proposed_settings']
        assert settings == {'window_length': 120}
        settings['window_length'] = 89
        events.append('actual_clicked')
    button.clicked.connect(apply)
    class Scene:
        def __init__(self, capture, name, settle): self.name = name
        def __enter__(self):
            active.append(self.name)
            events.append('begin')
            return self
        def point(self, *args, **kwargs):
            assert settings == {'window_length': 120}
            events.append('review_before_apply')
        def __exit__(self, *args):
            assert settings == {'window_length': 89}
            events.append('end')
            active.pop()
    monkeypatch.setattr(driver, 'Rec', Scene)
    panel = SimpleNamespace(apply_button=button,
        proposed_changes=lambda: [('window_length', 120, 89)])
    def collect():
        assert active == ['09_proposed_settings']
        events.append('readback')
        return dict(settings)
    try:
        changes, after = driver.record_apply_settings(panel,
            SimpleNamespace(collect=collect), None, lambda _: QTest.qWait(5))
        assert changes == [('window_length', 120, 89)] and after == {'window_length': 89}
        assert events == ['begin', 'review_before_apply', 'actual_clicked', 'readback', 'end']
    finally:
        button.close()
        button.deleteLater()
        app.processEvents()
