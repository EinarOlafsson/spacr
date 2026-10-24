"""The QTest -> human pointer adapter of the GNOME capture host (offscreen Qt, fake human)."""
from pathlib import Path
import os
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_host_adapter as adapter_module  # noqa: E402
from capture_host_adapter import (QTestAdapter, audit_scene_drivers, keysym_name,  # noqa: E402
                                  modifier_key_names, qt_key_to_keysym)

QtCore = pytest.importorskip('PySide6.QtCore')
QtWidgets = pytest.importorskip('PySide6.QtWidgets')
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget  # noqa: E402


class FakeHuman:
    """The asynchronous HumanProcess API, answering at once and logging every request."""

    def __init__(self):
        self.ops = []
        self.pointer = [0.0, 0.0]
        self._reply = None

    def submit(self, op, *args, **kwargs):
        self.ops.append((op, args))
        if op == 'move_to':
            self.pointer = [float(args[0]), float(args[1])]
        self._reply = list(self.pointer) if op == 'position' else None

    def reply_ready(self, timeout=0.0):
        return True

    def collect(self, op='request'):
        return self._reply


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    root = QWidget()
    layout = QVBoxLayout(root)
    button = QPushButton('Go', root)
    field = QLineEdit(root)
    hidden = QPushButton('Hidden', root)
    layout.addWidget(button)
    layout.addWidget(field)
    layout.addWidget(hidden)
    hidden.hide()
    root.resize(400, 300)
    root.show()
    QTest.qWaitForWindowExposed(root)
    yield root, button, field, hidden
    root.close()


@pytest.fixture
def adapted():
    human = FakeHuman()
    adapter = QTestAdapter(human, settle_after_click=0.0).install()
    yield human, adapter
    adapter.uninstall()


def test_keysyms_for_qt_keys():
    assert qt_key_to_keysym(Qt.Key_A) == ord('a')            # letters: the lower-case key
    assert qt_key_to_keysym(Qt.Key_5) == ord('5')
    assert qt_key_to_keysym(Qt.Key_Return) == 0xff0d
    assert qt_key_to_keysym(Qt.Key_Enter) == 0xff8d
    assert qt_key_to_keysym(Qt.Key_Escape) == 0xff1b
    assert qt_key_to_keysym(Qt.Key_End) == 0xff57
    assert qt_key_to_keysym(Qt.Key_F5) == 0xffc2
    assert qt_key_to_keysym(Qt.Key_Space) == 0x20
    assert qt_key_to_keysym('x') == ord('x')
    assert qt_key_to_keysym(Qt.Key_unknown) is None
    assert keysym_name(0xff0d) == 'Return' and keysym_name(ord('a')) == 'a'


def test_modifiers_become_held_keys():
    both = Qt.ControlModifier | Qt.ShiftModifier
    assert modifier_key_names(both) == ['Shift_L', 'Control_L']
    assert modifier_key_names(Qt.KeyboardModifier.NoModifier) == []
    assert modifier_key_names(None) == []


def test_install_replaces_and_uninstall_restores_qtest():
    original = QTest.mouseClick
    adapter = QTestAdapter(FakeHuman()).install()
    try:
        assert QTest.mouseClick is not original
        with pytest.raises(RuntimeError):
            adapter.install()
    finally:
        adapter.uninstall()
    assert QTest.mouseClick is original


def test_a_click_glides_the_real_pointer_then_presses(window, adapted):
    root, button, _field, _hidden = window
    human, adapter = adapted
    QTest.mouseClick(button, Qt.LeftButton)
    centre = button.mapToGlobal(QPoint(button.width() // 2, button.height() // 2))
    ops = [op for op, _ in human.ops]
    assert ops[0] == 'move_to' and human.ops[0][1] == (centre.x(), centre.y())
    assert ops[-1] == 'click' and human.ops[-1][1] == ('left',)
    assert not adapter.bypasses


def test_a_modified_click_holds_the_modifier_around_the_press(window, adapted):
    _root, button, _field, _hidden = window
    human, _adapter = adapted
    QTest.mouseClick(button, Qt.RightButton, Qt.ControlModifier, QPoint(3, 4))
    ops = [(op, args) for op, args in human.ops if op != 'position']
    target = button.mapToGlobal(QPoint(3, 4))
    assert ops[0] == ('move_to', (target.x(), target.y()))
    # One request: the modifiers are released before Qt can start a modal loop.
    assert ops[-1] == ('modified_click', (['Control_L'], 'right'))


def test_typing_goes_through_the_virtual_keyboard(window, adapted):
    _root, _button, field, _hidden = window
    human, _adapter = adapted
    QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(field, 'Meas')
    QTest.keyClick(field, Qt.Key_Return)
    ops = [(op, args) for op, args in human.ops]
    assert ops == [('chord', (['Control_L'], 'a')), ('type_text', ('Meas',)), ('press', ('Return',))]


def test_drag_keeps_the_button_down_while_moving(window, adapted):
    _root, button, _field, _hidden = window
    human, _adapter = adapted
    QTest.mousePress(button, Qt.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(2, 2))
    QTest.mouseMove(button, QPoint(20, 5))
    QTest.mouseRelease(button, Qt.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(30, 6))
    ops = [op for op, _ in human.ops if op != 'position']
    assert ops == ['move_to', 'button_down', 'move_to', 'move_to', 'button_up']


def test_an_unreachable_target_is_a_recorded_bypass_not_a_silent_teleport(window, adapted):
    _root, _button, _field, hidden = window
    human, adapter = adapted
    clicked = []
    hidden.clicked.connect(lambda: clicked.append(True))
    QTest.mouseClick(hidden, Qt.LeftButton)
    assert [b['qtest'] for b in adapter.bypasses] == ['mouseClick']
    assert adapter.bypasses[0]['reason'] == 'target is hidden'
    assert not [op for op, _ in human.ops if op == 'click']


def test_strict_mode_refuses_a_bypass(window):
    _root, _button, _field, hidden = window
    adapter = QTestAdapter(FakeHuman(), strict=True, settle_after_click=0.0).install()
    try:
        with pytest.raises(RuntimeError, match='hidden'):
            QTest.mouseClick(hidden, Qt.LeftButton)
    finally:
        adapter.uninstall()


def test_audit_lists_scene_code_that_changes_the_ui_without_qtest():
    report = audit_scene_drivers()
    assert 'capture_home.py' in report
    home = report['capture_home.py']
    assert home['qtest_calls_adapted'] > 0
    assert home['not_through_adapter'].get('button.click()', 0) > 0
    assert 'capture_host_adapter.py' not in report
    assert adapter_module.main(['--audit', '--json']) == 0


class SlowHuman(FakeHuman):
    """Replies arrive in request order; the first poll finds nothing yet (the glide is running)."""

    def __init__(self):
        super().__init__()
        self.pending = []
        self.polls = 0

    def submit(self, op, *args, **kwargs):
        self.ops.append((op, args))
        self.pending.append(op)

    def reply_ready(self, timeout=0.0):
        self.polls += 1
        return self.polls > 1 and bool(self.pending)

    def collect(self, op='request'):
        return self.pending.pop(0)


def test_a_call_made_while_another_is_waiting_gets_its_own_reply(app):
    human = SlowHuman()
    adapter = QTestAdapter(human, settle_after_click=0.0)
    seen = {}

    class Pump:
        def once(self):
            if 'inner' not in seen:        # e.g. a clip starting from a timer during a click
                seen['inner'] = None
                seen['inner'] = adapter.call('trace_mark')

        def run(self, seconds):
            pass
    adapter.pump = Pump()
    assert adapter.call('click', 'left') == 'click'
    assert seen['inner'] == 'trace_mark'


def test_start_and_finished_do_not_pump_and_keep_reply_order():
    """start/finished: a step inside a dialog loop sends one request and returns."""
    import capture_host_adapter as a

    class Human:
        def __init__(self):
            self.sent, self.ready = [], []

        def submit(self, op, *args, **kwargs):
            self.sent.append(op)

        def reply_ready(self, timeout=0.0):
            return bool(self.ready)

        def collect(self, op='request'):
            return self.ready.pop(0)

    class Pump:
        def once(self):
            raise AssertionError('start/finished must not pump the event loop')

    human = Human()
    adapter = a.QTestAdapter(human, pump=Pump())
    first = adapter.start('press', 'M')
    second = adapter.start('press', 'e')
    assert not adapter.finished(second)
    human.ready += ['one', 'two']
    assert adapter.finished(second) and adapter.finished(first)
    assert human.sent == ['press', 'press']


def test_note_never_blocks_the_gui_thread_on_the_log(app):
    """30 typed keys must not block the GUI for more than 20 ms each, even with a slow log file."""
    import io
    import time as _time

    class SlowLog(io.StringIO):
        def write(self, text):
            _time.sleep(0.05)               # a disk stalled under load
            return super().write(text)

        def flush(self):
            _time.sleep(0.05)

    log = SlowLog()
    adapter = QTestAdapter(FakeHuman(), log=log)
    worst = 0.0
    for _ in range(30):
        started = _time.perf_counter()
        adapter.note('key', key='a', op='press', modifiers=[])
        worst = max(worst, _time.perf_counter() - started)
    assert worst < 0.020, f'note() blocked the GUI thread for {worst * 1000:.0f} ms'
    assert adapter.flush_notes(timeout=10)
    assert log.getvalue().count('"kind": "key"') == 30
