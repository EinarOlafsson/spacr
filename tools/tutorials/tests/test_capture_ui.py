"""capture_ui: outside the capture host each action is the old direct call; inside, a real input."""
from pathlib import Path
import os
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('PySide6.QtWidgets')
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QLineEdit, QSpinBox,  # noqa: E402
                               QSplitter, QTabWidget, QWidget)
import capture_ui  # noqa: E402


def plain_capture(*_a, **_k):
    return True


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_outside_the_host_actions_are_the_old_direct_calls(app):
    check, spin, edit, combo = QCheckBox(), QSpinBox(), QLineEdit(), QComboBox()
    combo.addItems(['a', 'b'])
    tabs = QTabWidget()
    tabs.addTab(QWidget(), 'one')
    tabs.addTab(QWidget(), 'two')
    split = QSplitter()
    split.addWidget(QWidget())
    split.addWidget(QWidget())
    split.resize(400, 100)
    assert not capture_ui.live(plain_capture)
    capture_ui.set_checked(plain_capture, check, True)
    capture_ui.set_number(plain_capture, spin, 7)
    capture_ui.set_text(plain_capture, edit, 'x')
    capture_ui.select_combo(plain_capture, combo, 1)
    capture_ui.select_tab(plain_capture, tabs, 1)
    capture_ui.splitter_sizes(plain_capture, split, [100, 300])
    assert check.isChecked() and spin.value() == 7 and edit.text() == 'x'
    assert combo.currentIndex() == 1 and tabs.currentIndex() == 1


def test_in_the_host_a_checkbox_is_clicked_not_set(app, monkeypatch):
    clicks = []

    class FakeHost:
        pump = type('P', (), {'run': staticmethod(lambda s: None)})()

    def capture(*_a, **_k):
        return True
    capture.host = FakeHost()
    check = QCheckBox()
    import capture_home_live
    monkeypatch.setattr(capture_home_live, 'click', lambda w, pos=None: (clicks.append(w), w.toggle()))
    capture_ui.set_checked(capture, check, True)
    assert clicks == [check] and check.isChecked()
    capture_ui.set_checked(capture, check, True)          # already on: no click
    assert clicks == [check]


def test_dialog_geometry_is_left_to_the_app_in_the_host(app):
    def capture(*_a, **_k):
        return True
    capture.host = object()
    w = QWidget()
    w.resize(300, 200)
    kept = capture_ui.dialog_geometry(capture, w, (1200, 1200))
    assert (w.width(), w.height()) == (300, 200) and kept['kept_application_geometry'][2:] == [300, 200]


def test_clip_outside_the_host_captures_the_finished_state_after_the_body(app):
    order = []

    def capture(name, **_k):
        order.append(('capture', name))
        return True

    with capture_ui.clip(capture, 'scene') as recording:
        order.append(('action', recording))
    assert order == [('action', False), ('capture', 'scene')]
    assert capture_ui.point_at(capture, QWidget()) == []
    assert not capture_ui.clip_due(capture)


def test_clip_in_the_host_records_around_the_actions(app, monkeypatch):
    order = []

    class Pump:
        def run(self, seconds):
            order.append(('pause', seconds))

    class Host:
        pump = Pump()
        current = None

    def capture(name, **_k):
        raise AssertionError('a host clip never captures a finished state')

    capture.host = Host()
    capture.begin = lambda name: order.append(('begin', name)) or True
    capture.end = lambda: order.append(('end',))
    with capture_ui.clip(capture, 'scene', lead=0.5) as recording:
        order.append(('action', recording))
    assert order == [('begin', 'scene'), ('pause', 0.5), ('action', True), ('end',)]
    # A scene that is not recorded runs its actions without a clip.
    order.clear()
    capture.begin = lambda name: False
    with capture_ui.clip(capture, 'other') as recording:
        order.append(('action', recording))
    assert order == [('action', False)]
    # The recording stops even when the scene fails.
    order.clear()
    capture.begin = lambda name: True
    with pytest.raises(RuntimeError):
        with capture_ui.clip(capture, 'broken', lead=0):
            raise RuntimeError('boom')
    assert order[-1] == ('end',)
