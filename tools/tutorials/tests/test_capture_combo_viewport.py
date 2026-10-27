"""Native combo choices must be exposed before their item can be clicked."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QVBoxLayout, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_home_live as live
from capture_host_adapter import QTestAdapter


@pytest.mark.parametrize('case', ['visible', 'off_viewport', 'blocked'])
def test_native_combo_scrolls_to_reachable_item_before_click(monkeypatch, case):
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.resize(400, 200)
    window.move(100, 100)
    layout = QVBoxLayout(window)
    combo = QComboBox()
    combo.addItems([f'column_{i:03d}' for i in range(160)])
    combo.setMaxVisibleItems(8)
    layout.addWidget(combo)
    window.show()
    app.processEvents()
    wheels, clicks = [], []
    target = 2 if case == 'visible' else 140
    capture = SimpleNamespace(host=SimpleNamespace(adapter=SimpleNamespace(
        reachable=QTestAdapter.reachable, target_point=QTestAdapter.target_point)))
    def pause(_capture=None, seconds=0):
        QTest.qWait(round(seconds * 1000) if seconds >= .5 else 2)
        app.processEvents()
    def click(widget, pos=None):
        if widget is combo.view().viewport():
            item = combo.model().index(target, combo.modelColumn())
            rect = combo.view().visualRect(item)
            assert rect.top() >= widget.rect().top() and rect.bottom() <= widget.rect().bottom()
            assert widget.rect().contains(pos) and combo.view().indexAt(pos) == item
            assert QTestAdapter.reachable(widget, QTestAdapter.target_point(widget, pos)) is None
        clicks.append(widget)
        QTest.mouseClick(widget, Qt.LeftButton, pos=pos if pos is not None else widget.rect().center())
        pause()
    def wheel(_capture, widget, notches, pos=None):
        wheels.append(notches)
        assert widget is combo.view().viewport()
        if case != 'blocked':
            point = pos or widget.rect().center()
            event = QWheelEvent(QPointF(point), QPointF(widget.mapToGlobal(point)), QPoint(),
                                QPoint(0, -120 * notches), Qt.NoButton, Qt.NoModifier,
                                Qt.NoScrollPhase, False)
            QApplication.sendEvent(widget, event)
            pause()
    monkeypatch.setattr(live, 'click', click)
    monkeypatch.setattr(live, 'wheel', wheel)
    monkeypatch.setattr(live, 'pause', pause)
    try:
        if case == 'blocked':
            with pytest.raises(RuntimeError, match='Native wheel did not expose'):
                live.choose_combo(capture, combo, target)
            assert wheels and clicks == [combo]
            assert combo.currentIndex() == 0
        else:
            live.choose_combo(capture, combo, target)
            assert combo.currentIndex() == target
            assert clicks == [combo, combo.view().viewport()]
            assert bool(wheels) == (case == 'off_viewport')
    finally:
        combo.hidePopup()
        window.close()
        window.deleteLater()
        app.processEvents()
