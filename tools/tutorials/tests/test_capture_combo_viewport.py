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


@pytest.mark.parametrize('case', ['visible', 'off_viewport', 'disabled'])
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
    if case == 'disabled': combo.model().item(target).setEnabled(False)
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
        if case != 'disabled':
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
        if case == 'disabled':
            with pytest.raises(RuntimeError, match='not enabled and selectable'):
                live.choose_combo(capture, combo, target)
            assert not wheels and clicks == [combo]
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


def test_condition_extract_menu_clipped_leading_row_uses_native_keys(monkeypatch):
    import pandas as pd
    from spacr.qt.theme import apply_qpalette, stylesheet
    from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog

    app = QApplication.instance() or QApplication([])
    old_sheet, old_palette = app.styleSheet(), app.palette()
    apply_qpalette(app)
    app.setStyleSheet(stylesheet())
    frame = pd.DataFrame([dict(object_label=1, plateID='plate1', rowID='r1', columnID='c1',
        fieldID='f1', prcf='plate1_r1_c1_f1', file_name='source.png',
        **{f'cell_metric_{i}': float(i) for i in range(270)})])
    dialog = ConditionAnnotationDialog(frame, {}, threaded=False)
    dialog.resize(1200, 800)
    dialog.move(680, 320)
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('extract'))
    dialog.show()
    QTest.qWait(20)
    combo = dialog.extract_source
    target = combo.findText('rowID')
    assert target == 2 and combo.currentText() == 'file_name'
    QTest.mouseClick(combo, Qt.LeftButton, pos=combo.rect().center())
    QTest.qWait(500)
    view = combo.view()
    assert view.isVisible() and view.window().metaObject().className() == 'QComboBoxPrivateContainer'
    item = combo.model().index(target, combo.modelColumn())
    # Establish the actual clipped-leading-row menu state with ordinary wheel input.
    for _ in range(combo.count()):
        if view.visualRect(item).top() < view.viewport().rect().top(): break
        port = view.viewport()
        point = port.rect().center()
        QApplication.sendEvent(port, QWheelEvent(QPointF(point), QPointF(port.mapToGlobal(point)),
            QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        QTest.qWait(2)
    assert view.visualRect(item).top() < view.viewport().rect().top()
    capture = SimpleNamespace(host=SimpleNamespace(adapter=SimpleNamespace(
        reachable=QTestAdapter.reachable, target_point=QTestAdapter.target_point)))
    keys, wheels = [], []
    def key(widget, value, modifier=None):
        assert widget is view
        keys.append(value)
        QTest.keyClick(widget, value)
        QTest.qWait(5)
    def no_item_click(*_args, **_kwargs):
        pytest.fail('An offscreen menu item must never be clicked')
    monkeypatch.setattr(live, 'key', key)
    monkeypatch.setattr(live, 'click', no_item_click)
    monkeypatch.setattr(live, 'wheel', lambda _capture, _widget, notches, pos=None: wheels.append(notches))
    monkeypatch.setattr(live, 'pause', lambda *_: QTest.qWait(5))
    try:
        live.choose_combo(capture, combo, target)
        assert keys == [Qt.Key_Home, Qt.Key_Down, Qt.Key_Down, Qt.Key_Return]
        assert wheels == [-1] and combo.currentIndex() == target and not view.isVisible()
    finally:
        combo.hidePopup()
        dialog.close()
        dialog.deleteLater()
        app.setStyleSheet(old_sheet)
        app.setPalette(old_palette)
        app.processEvents()
