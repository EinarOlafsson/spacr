"""The actual source chooser is accepted before canonical scene18 ends."""
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog
import capture_ui
import capture_classify_existing as driver
from spacr.qt.screens.app_screen import AppScreen


@pytest.mark.parametrize('cancel', [False, True])
def test_actual_classify_picker_finishes_scene_only_after_modal_and_readback(tmp_path, monkeypatch, cancel):
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs)
    app = QApplication.instance() or QApplication([])
    screen = AppScreen('classify_merged')
    screen.resize(1500, 1000)
    screen.show()
    QTest.qWait(60)
    source = tmp_path / 'project'
    source.mkdir()
    state = dict(callback=False, ends=[], legacy=[], unsafe=False)
    original_wait = driver.modal_wait
    def watched_wait(capture, delay, handler):
        def select():
            state['callback'] = True
            try:
                assert isinstance(app.activeModalWidget(), QFileDialog)
                if cancel: app.activeModalWidget().reject()
                else: handler()
            finally:
                state['callback'] = False
        return original_wait(capture, delay, select)
    monkeypatch.setattr(driver, 'modal_wait', watched_wait)
    class QtInput:
        def click(self, widget, pos=None):
            QTest.mouseClick(widget, Qt.LeftButton, pos=pos or widget.rect().center())
        def key(self, widget, key, modifiers=Qt.NoModifier): QTest.keyClick(widget, key, modifiers)
        def type_text(self, widget, text): QTest.keyClicks(widget, str(text))
        def pause(self, capture, seconds): QTest.qWait(5)
    monkeypatch.setattr(capture_ui, '_h', lambda: QtInput())
    monkeypatch.setattr(capture_ui, 'wait', lambda *args: None)
    class Capture:
        host = object()
        def begin(self, name):
            assert name == '18_source_folder_choice'
            return True
        def end(self):
            if state['callback']: state['unsafe'] = True
            state['ends'].append(dict(modal=app.activeModalWidget(),
                source=screen._settings_src_path(), empty=screen._empty_state_card.isVisible()))
        def __call__(self, name): state['legacy'].append(name)
    try:
        assert screen._empty_state_card.isVisible()
        if cancel:
            with pytest.raises(RuntimeError, match='did not set the source'):
                driver.choose_existing_folder(app, screen, source, Capture(), lambda *args: QTest.qWait(5))
        else:
            driver.choose_existing_folder(app, screen, source, Capture(), lambda *args: QTest.qWait(5))
        assert not state['unsafe'] and len(state['ends']) == 1
        assert state['ends'][0]['modal'] is None
        if cancel:
            assert state['legacy'] == [] and state['ends'][0]['empty']
        else:
            assert state['ends'][0]['source'] == str(source.resolve())
            assert not state['ends'][0]['empty']
            assert state['legacy'] == ['18_source_folder_selected']
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
