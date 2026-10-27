"""One adapter that turns the scene drivers' QTest calls into a human at the real pointer.

The scene drivers (``capture_refresh.py`` and the ``capture_*.py`` modules it
calls) click, type and move with ``PySide6.QtTest.QTest``. QTest delivers
synthetic events straight to a widget: the visible pointer never moves, so in
a screen recording a button reacts with nobody touching it -- the "teleport"
the maintainer rejected. Inside the headless GNOME capture host,
:func:`install` replaces those QTest functions, in place, so the same scene
code instead:

* glides the real compositor pointer (mutter RemoteDesktop, through
  :class:`human_input.HumanProcess`) along a minimum-jerk path to the point
  QTest would have used, then presses and releases a real button;
* types through mutter's virtual keyboard, holding real modifier keys;
* keeps the Qt event loop running the whole time (the field keeps painting
  and the app sees genuine enter/hover/press events), because the pointer is
  moved by a separate process and this thread only pumps events until it
  replies.

Nothing else in the scene drivers changes. Anything the adapter cannot do
for real (a target that is hidden or covered at its click point, a key with
no keysym) falls back to the original QTest call and is written to
:attr:`QTestAdapter.bypasses`; the clip recorder fails a clip with any
bypass inside it. Scene code that changes the UI WITHOUT QTest
(``button.click()``, ``action.trigger()``, ``sendEvent`` of mouse events,
direct ``setCurrentIndex``) is not seen here: :func:`audit_scene_drivers`
lists those call sites for the re-record batches.

    python tools/tutorials/capture_host_adapter.py --audit     # bypass inventory
"""
from __future__ import annotations

import atexit
import json
import math
import os
import queue
import re
import threading
import time
from pathlib import Path

#: Qt.Key values for non-character keys -> X keysyms (keysymdef.h).
QT_KEY_KEYSYMS = {
    0x01000000: 0xff1b,   # Key_Escape
    0x01000001: 0xff09,   # Key_Tab
    0x01000002: 0xfe20,   # Key_Backtab (ISO_Left_Tab)
    0x01000003: 0xff08,   # Key_Backspace
    0x01000004: 0xff0d,   # Key_Return
    0x01000005: 0xff8d,   # Key_Enter (keypad)
    0x01000006: 0xff63,   # Key_Insert
    0x01000007: 0xffff,   # Key_Delete
    0x01000010: 0xff50,   # Key_Home
    0x01000011: 0xff57,   # Key_End
    0x01000012: 0xff51,   # Key_Left
    0x01000013: 0xff52,   # Key_Up
    0x01000014: 0xff53,   # Key_Right
    0x01000015: 0xff54,   # Key_Down
    0x01000016: 0xff55,   # Key_PageUp
    0x01000017: 0xff56,   # Key_PageDown
    0x01000020: 0xffe1,   # Key_Shift
    0x01000021: 0xffe3,   # Key_Control
    0x01000023: 0xffe9,   # Key_Alt
    0x01000022: 0xffeb,   # Key_Meta
}
#: Key_F1 .. Key_F35 -> F1 .. F35
for _n in range(35):
    QT_KEY_KEYSYMS[0x01000030 + _n] = 0xffbe + _n

#: Qt.KeyboardModifier bits -> the modifier key a human holds.
MODIFIER_KEYS = ((0x02000000, 'Shift_L'), (0x04000000, 'Control_L'),
                 (0x08000000, 'Alt_L'), (0x10000000, 'Super_L'))

#: The QTest functions the adapter replaces.
REPLACED = ('mouseClick', 'mouseDClick', 'mousePress', 'mouseRelease', 'mouseMove',
            'keyClick', 'keyClicks', 'keyPress', 'keyRelease', 'keySequence')


def _value(flag) -> int:
    """int of a Qt enum/flag (PySide6 6.x enums are Python enums)."""
    if flag is None:
        return 0
    return int(getattr(flag, 'value', flag))


def qt_key_to_keysym(key) -> int | None:
    """X keysym for a ``Qt.Key`` (or a one-character string), lower-case for letters.

    Letters map to their lower-case keysym: a human types Ctrl+A with the
    ``a`` key, and Shift arrives as a held modifier.
    """
    if isinstance(key, str):
        if len(key) != 1:
            return None
        from human_input import _keysym
        return _keysym(key)
    code = _value(key)
    if code in QT_KEY_KEYSYMS:
        return QT_KEY_KEYSYMS[code]
    if 0x20 <= code <= 0x7e:
        return ord(chr(code).lower())
    if 0xa0 <= code <= 0xff:
        return code
    return None


def modifier_key_names(modifiers) -> list[str]:
    bits = _value(modifiers)
    return [name for bit, name in MODIFIER_KEYS if bits & bit]


def button_name(button) -> str:
    return {1: 'left', 2: 'right', 4: 'middle'}.get(_value(button), 'left')


def keysym_name(sym: int) -> str:
    """The key name :func:`human_input._keysym` understands for ``sym``."""
    from human_input import KEYSYM
    for name, value in KEYSYM.items():
        if value == sym and len(name) > 1:
            return name
    if 0x20 <= sym <= 0xff:
        return chr(sym)
    return f'#keysym:{sym:#x}'


class EventPump:
    """Run the Qt event loop in small slices while waiting for something else."""

    def __init__(self, slice_ms: int = 4):
        self.slice_ms = slice_ms

    def once(self):
        from PySide6.QtCore import QEventLoop
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents(QEventLoop.AllEvents, self.slice_ms)

    def run(self, seconds: float):
        until = time.monotonic() + max(0.0, seconds)
        while True:
            self.once()
            left = until - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.002, left))

    def until(self, ready, timeout: float, what: str = 'condition'):
        deadline = time.monotonic() + timeout
        while not ready():
            if time.monotonic() > deadline:
                raise TimeoutError(what)
            self.once()
            time.sleep(0.001)


class QTestAdapter:
    """QTest-compatible functions that drive a :class:`human_input.HumanProcess`.

    ``human`` needs ``submit``/``reply_ready``/``collect`` (asynchronous
    requests); the Qt main thread pumps events while the pointer moves.
    """

    def __init__(self, human, *, pump: EventPump | None = None, settle_after_click: float = 0.12,
                 strict: bool = False, log=None):
        self.human = human
        self.pump = pump or EventPump()
        self.settle_after_click = settle_after_click
        self.strict = strict
        self.original = {}
        self._sent = self._received = 0
        self._replies = {}
        self.bypasses = []
        self.actions = []
        self._log = log
        self._notes = queue.SimpleQueue()
        self._writer = None

    # -- plumbing -----------------------------------------------------------
    def call(self, op, *args, **kwargs):
        """Ask the human process for ``op`` and keep Qt running until it is done.

        Re-entrant: while this call pumps events, a Qt timer may run scene code
        that makes another call (a clip starting inside a click). The human
        process answers in request order, so every reply goes to the oldest
        waiting request and each caller waits for its own.
        """
        seq = self._sent
        self._sent += 1
        self.human.submit(op, *args, **kwargs)
        while seq not in self._replies:
            if self.human.reply_ready(0.0):
                try:
                    self._replies[self._received] = ('ok', self.human.collect(op))
                except RuntimeError as error:
                    self._replies[self._received] = ('error', error)
                self._received += 1
                continue
            self.pump.once()
            self.human.reply_ready(0.002)
        kind, value = self._replies.pop(seq)
        if kind == 'error':
            raise value
        return value

    def _drain(self, op='request'):
        """Store every reply that has arrived, in request order, without pumping."""
        while self.human.reply_ready(0.0):
            try:
                self._replies[self._received] = ('ok', self.human.collect(op))
            except RuntimeError as error:
                self._replies[self._received] = ('error', error)
            self._received += 1

    def start(self, op, *args, **kwargs) -> int:
        """Send ``op`` and return at once; poll :meth:`finished` from the event loop.

        For steps that must not pump events themselves (inside an application
        dialog loop each key is one short callback, as a typing user is).
        """
        seq = self._sent
        self._sent += 1
        self.human.submit(op, *args, **kwargs)
        return seq

    def finished(self, seq) -> bool:
        self._drain()
        if seq not in self._replies:
            return False
        kind, value = self._replies.pop(seq)
        if kind == 'error':
            raise value
        return True

    def note(self, kind, **details):
        """Record one action. Cheap: the GUI thread only appends in memory.

        The log file is written by a background thread. A synchronous write and flush here ran once
        per typed key and blocked the GUI thread for 0.40 s under load (lesson 39, filter text).
        """
        entry = {'t': time.monotonic(), 'kind': kind, **details}
        self.actions.append(entry)
        if self._log is not None:
            if self._writer is None:
                self._writer = threading.Thread(target=self._write_notes, name='adapter-notes', daemon=True)
                self._writer.start()
                atexit.register(self.flush_notes)
            self._notes.put(entry)

    def _write_notes(self):
        while True:
            item = self._notes.get()
            if isinstance(item, threading.Event):
                try:
                    self._log.flush()
                finally:
                    item.set()
                continue
            try:
                self._log.write(json.dumps(item, default=str) + '\n')
            except (ValueError, OSError):
                pass

    def flush_notes(self, timeout: float = 5.0) -> bool:
        """Wait until every queued note is on disk (clip end, exit); never called per key."""
        if self._writer is None or not self._writer.is_alive():
            return True
        barrier = threading.Event()
        self._notes.put(barrier)
        return barrier.wait(timeout)

    def bypass(self, function, reason, *args, **kwargs):
        target = args[0] if args else None
        entry = {'t': time.monotonic(), 'qtest': function, 'reason': reason,
                 'widget': type(target).__name__ if target is not None else None,
                 'name': getattr(target, 'objectName', lambda: '')() if target is not None else None,
                 'text': (getattr(target, 'text', lambda: '')() or '')[:40] if hasattr(target, 'text') else None}
        self.bypasses.append(entry)
        self.note('bypass', **{k: v for k, v in entry.items() if k != 't'})
        if self.strict:
            raise RuntimeError(f'QTest.{function} cannot be done by the human pointer: {reason}')
        return self.original[function](*args, **kwargs)

    # -- geometry -------------------------------------------------------------
    @staticmethod
    def target_point(widget, pos=None):
        """Global (compositor) point QTest would use: ``pos`` or the widget centre."""
        from PySide6.QtCore import QPoint
        if pos is None or (hasattr(pos, 'isNull') and pos.isNull()):
            pos = QPoint(widget.width() // 2, widget.height() // 2)
        elif not isinstance(pos, QPoint):
            pos = QPoint(int(round(pos.x())), int(round(pos.y())))
        point = widget.mapToGlobal(pos)
        return point.x(), point.y()

    @staticmethod
    def reachable(widget, point) -> str | None:
        """None when a real pointer at ``point`` lands on ``widget``; else why not."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QApplication, QWidget
        if not isinstance(widget, QWidget):
            return f'not a widget ({type(widget).__name__})'
        if not widget.isVisible():
            return 'target is hidden'
        window = widget.window()
        if window.windowHandle() is None or not window.isVisible():
            return 'target window is not mapped'
        hit = QApplication.widgetAt(QPoint(*point))
        if hit is None:
            return 'nothing of the app under the point'
        if hit is widget or widget.isAncestorOf(hit) or hit.isAncestorOf(widget):
            return None
        if hit.window() is widget.window():
            # A child that is not a descendant (e.g. an overlay sibling) takes the press.
            text = getattr(hit, 'text', lambda: '')() if hasattr(hit, 'text') else ''
            parent = hit.parentWidget()
            return (f'covered by {type(hit).__name__} {hit.objectName()!r} {str(text)[:30]!r} '
                    f'(parent {type(parent).__name__ if parent else None} '
                    f'{parent.objectName() if parent else ""!r}, visible={hit.isVisible()})')
        return f'covered by another window ({type(hit.window()).__name__})'

    def _focus(self, widget):
        """Keyboard events go to the focused widget; a QTest key targets one directly."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QWidget
        if not isinstance(widget, QWidget):
            return
        window = widget.window()
        if QApplication.activeWindow() is not window:
            window.activateWindow()
            window.raise_()
        if QApplication.focusWidget() is not widget and widget.focusPolicy() != Qt.NoFocus \
                and not (widget.isWindow() and QApplication.focusWidget() is not None
                         and widget.isAncestorOf(QApplication.focusWidget())):
            widget.setFocus(Qt.OtherFocusReason)
        self.pump.run(0.03)

    def _close_unrelated_popup(self, widget):
        """A popup (completer list, menu) eats the next click outside it: close it with Escape first.

        A user typing a path sees the completer's suggestions and presses
        Escape (or clicks twice); a synthetic click would have gone straight
        to its target, so the scene code does not expect the popup.
        """
        from PySide6.QtWidgets import QApplication, QWidget
        popup = QApplication.activePopupWidget()
        if popup is None or not popup.isVisible() or not isinstance(widget, QWidget):
            return
        if popup is widget or popup.isAncestorOf(widget) or widget.window() is popup:
            return
        self.dismiss_popup(popup)

    def dismiss_popup(self, popup):
        """Close a popup as a user would.

        A completer's suggestion list: click its suggestion (for a fully typed
        path that is the same text) -- Escape could reach the dialog and close
        it, since a popup's keyboard grab is not reliable under Xwayland. A
        menu: Escape.
        """
        from PySide6.QtWidgets import QAbstractItemView, QMenu
        view = popup if isinstance(popup, QAbstractItemView) else popup.findChild(QAbstractItemView)
        # The file dialog completer popup is a private Qt class whose model PySide wraps as the wrong type
        # (a QWidgetItem, "no attribute rowCount", lesson 37), so look at the view, not at its model.
        from PySide6.QtCore import QPoint
        first = view.indexAt(QPoint(4, 4)) if view is not None else None
        if not isinstance(popup, QMenu) and view is not None and first is not None and (
                view.currentIndex().isValid() or first.isValid()):
            index = view.currentIndex() if view.currentIndex().isValid() else first
            rect = view.visualRect(index)
            point = view.viewport().mapToGlobal(rect.center())
            self.call('move_to', point.x(), point.y())
            self.call('click', 'left')
            self.note('pick-suggestion', popup=type(popup).__name__)
        else:
            self.call('press', 'Escape')
            self.note('escape-popup', popup=type(popup).__name__)
        self.pump.run(0.25)

    def _observed_pointer(self, point):
        """Query the existing xcb app connection; this is not a video acknowledgement."""
        from PySide6.QtCore import QPoint, QThread
        from PySide6.QtGui import QCursor, QGuiApplication
        app = QGuiApplication.instance()
        if app is None or app.platformName() != 'xcb' or not os.environ.get('SPACR_CAPTURE_HOST'):
            raise RuntimeError('Independent pointer observation requires the owned capture app on xcb')
        if QThread.currentThread() != app.thread():
            raise RuntimeError('Independent pointer observation must run on the GUI thread')
        screens = app.screens()
        if len(screens) != 1 or not screens[0].geometry().contains(QPoint(*point)):
            raise RuntimeError('Independent pointer observation requires the owned single target screen')
        try:
            observed = QCursor.pos(screens[0])
        except Exception as exc:
            raise RuntimeError('Independent X server pointer query failed') from exc
        if observed.isNull() or not screens[0].geometry().contains(observed):
            raise RuntimeError('Independent X server pointer observation is unavailable or outside the target screen')
        return observed.x(), observed.y()

    def _require_observed_pointer(self, point):
        """Refuse target acceptance without independent X server coordinates."""
        observed = self._observed_pointer(point)
        if (not isinstance(observed, (tuple, list)) or len(observed) != 2
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       or not math.isfinite(v) for v in observed)):
            raise RuntimeError('Independent pointer observation is unavailable or invalid')
        if abs(observed[0] - point[0]) > 1 or abs(observed[1] - point[1]) > 1:
            raise RuntimeError(f'Observed X server pointer {observed} did not reach target {point}')
        self.note('observed-pointer', point=list(observed), source='Qt xcb server query; not video acknowledgement')

    def _glide(self, function, widget, pos):
        self._close_unrelated_popup(widget)
        point = self.target_point(widget, pos)
        why = self.reachable(widget, point)
        if why is not None:
            return point, why
        commanded_point = point
        self.call('move_to', *point)
        self.pump.run(0.02)
        # Moving can reveal or re-lay-out things (hover effects); check again.
        point = self.target_point(widget, pos)
        why = self.reachable(widget, point)
        if why is None:
            if point != commanded_point:
                self.call('move_to', *point)
                self.pump.run(0.02)
            self._require_observed_pointer(point)
        return point, why

    def _with_modifiers(self, modifiers, action):
        held = modifier_key_names(modifiers)
        for key in held:
            self.call('key_down', key)
        try:
            return action()
        finally:
            for key in reversed(held):
                self.call('key_up', key)

    # -- QTest replacements ----------------------------------------------------
    def mouseClick(self, widget, button=None, modifier=None, pos=None, delay=-1):
        point, why = self._glide('mouseClick', widget, pos)
        if why:
            return self.bypass('mouseClick', why, *self._qtest_args(widget, button, modifier, pos))
        self.note('click', point=point, widget=type(widget).__name__, name=widget.objectName())
        held = modifier_key_names(modifier)
        if held:
            # One request: modifiers are released before Qt can start a modal loop.
            self.call('modified_click', held, button_name(button))
        else:
            self.call('click', button_name(button))
        self.pump.run(self.settle_after_click)

    def mouseDClick(self, widget, button=None, modifier=None, pos=None, delay=-1):
        point, why = self._glide('mouseDClick', widget, pos)
        if why:
            return self.bypass('mouseDClick', why, *self._qtest_args(widget, button, modifier, pos))
        self.note('double-click', point=point, widget=type(widget).__name__)
        self._with_modifiers(modifier, lambda: self.call('double_click', button_name(button)))
        self.pump.run(self.settle_after_click)

    def mousePress(self, widget, button=None, modifier=None, pos=None, delay=-1):
        point, why = self._glide('mousePress', widget, pos)
        if why:
            return self.bypass('mousePress', why, *self._qtest_args(widget, button, modifier, pos))
        self.note('press', point=point, widget=type(widget).__name__)
        for key in modifier_key_names(modifier):
            self.call('key_down', key)
        self.call('button_down', button_name(button))
        self.pump.run(0.08)

    def mouseRelease(self, widget, button=None, modifier=None, pos=None, delay=-1):
        # A drag: the button is already down, so glide (carrying it) to the
        # release point; the target may legitimately be under the drag.
        point = self.target_point(widget, pos)
        if not widget.isVisible():
            return self.bypass('mouseRelease', 'target is hidden',
                               *self._qtest_args(widget, button, modifier, pos))
        self.call('move_to', *point)
        self.call('button_up', button_name(button))
        for key in reversed(modifier_key_names(modifier)):
            self.call('key_up', key)
        self.note('release', point=point, widget=type(widget).__name__)
        self.pump.run(self.settle_after_click)

    def mouseMove(self, widget, pos=None, delay=-1):
        point = self.target_point(widget, pos)
        if not widget.isVisible():
            return self.bypass('mouseMove', 'target is hidden', widget,
                               *(() if pos is None else (pos,)))
        self.call('move_to', *point)
        self.note('move', point=point, widget=type(widget).__name__)
        self.pump.run(0.02)

    def _key(self, function, widget, key, modifier, op):
        sym = qt_key_to_keysym(key)
        if sym is None:
            return self.bypass(function, f'no keysym for {key!r}', widget, key,
                               *(() if modifier is None else (modifier,)))
        self._focus(widget)
        name = keysym_name(sym)
        if op == 'press':
            held = modifier_key_names(modifier)
            if held:
                # One request (see Human.chord): Ctrl+K opens a modal palette.
                self.call('chord', held, name)
            else:
                self.call('press', name)
        elif op == 'down':
            for held in modifier_key_names(modifier):
                self.call('key_down', held)
            self.call('key_down', name)
        else:
            self.call('key_up', name)
            for held in reversed(modifier_key_names(modifier)):
                self.call('key_up', held)
        self.note('key', key=name, op=op, modifiers=modifier_key_names(modifier))
        self.pump.run(0.03)

    def keyClick(self, widget, key, modifier=None, delay=-1):
        self._key('keyClick', widget, key, modifier, 'press')

    def keyPress(self, widget, key, modifier=None, delay=-1):
        self._key('keyPress', widget, key, modifier, 'down')

    def keyRelease(self, widget, key, modifier=None, delay=-1):
        self._key('keyRelease', widget, key, modifier, 'up')

    def keyClicks(self, widget, sequence, modifier=None, delay=-1):
        self._focus(widget)
        self._with_modifiers(modifier, lambda: self.call('type_text', str(sequence)))
        self.note('type', text=str(sequence))
        self.pump.run(0.05)

    def keySequence(self, widget, sequence):
        """``QKeySequence`` -> each chord pressed with its modifiers held."""
        from PySide6.QtGui import QKeySequence
        sequence = QKeySequence(sequence)
        for index in range(sequence.count()):
            combo = sequence[index]
            self.keyClick(widget, combo.key(), combo.keyboardModifiers())

    @staticmethod
    def _qtest_args(widget, button, modifier, pos):
        from PySide6.QtCore import QPoint, Qt
        return (widget, button if button is not None else Qt.LeftButton,
                modifier if modifier is not None else Qt.KeyboardModifier.NoModifier,
                pos if pos is not None else QPoint())

    # -- install ---------------------------------------------------------------
    def install(self):
        from PySide6.QtTest import QTest
        if self.original:
            raise RuntimeError('QTest adapter already installed')
        for name in REPLACED:
            self.original[name] = getattr(QTest, name)
            setattr(QTest, name, staticmethod(getattr(self, name)))
        return self

    def uninstall(self):
        from PySide6.QtTest import QTest
        for name, function in self.original.items():
            setattr(QTest, name, function)
        self.original = {}


# -- inventory of scene code that changes the UI without QTest ---------------
BYPASS_PATTERNS = {
    'button.click()': re.compile(r'\.click\(\)'),
    'action.trigger()': re.compile(r'\.trigger\(\)'),
    'sendEvent/postEvent of input events': re.compile(
        r'(sendEvent|postEvent)\([^)]*Q(Mouse|Wheel|Key|Hover|Enter)Event'
        r'|Q(Mouse|Wheel|Key)Event\('),
    'setCurrentIndex/setCurrentWidget': re.compile(r'\.setCurrent(Index|Widget|Row|Text)\('),
    'direct navigation (_on_nav_selected/_open_*)': re.compile(r'\._on_nav_selected\(|\.show_preferences_on\('),
    'dialog built and shown by code': re.compile(r'(PreferencesDialog|QMessageBox|Dialog)\(\s*window'),
    'menu.popup()/exec()': re.compile(r'\.popup\(|menu\.exec\('),
    'setChecked/setValue/setText on a control': re.compile(r'\.set(Checked|Value|Text|PlainText)\('),
    'layout by code (setSizes/resize/move/ensureWidgetVisible)': re.compile(
        r'\.setSizes\(|\bdialog\.(resize|move)\(|\.ensureWidgetVisible\('),
    'state by code (show_index/set_value_for_key/selectFile/set_level/set_query)': re.compile(
        r'\.show_index\(|\.set_value_for_key\(|\.selectFile\(|\.set_level\(|\.set_query\('),
}


def audit_scene_drivers(root: Path | None = None) -> dict:
    """Per scene driver: QTest calls (adapted) and call sites the adapter cannot see."""
    root = Path(root or Path(__file__).resolve().parent)
    report = {}
    for path in sorted(root.glob('capture_*.py')):
        if path.name in {'capture_host_adapter.py', 'capture_host_mode.py', 'capture_host_systemd_stub.py',
                         'capture_ui.py', 'capture_home_live.py', 'capture_host_bench.py',
                         'capture_clip_gate.py', 'capture_live_video.py', 'capture_policy.py',
                         'capture_geometry.py', 'capture_acceptance.py'}:
            continue
        text = path.read_text(errors='replace')
        qtest = len(re.findall(r'\bQTest\.(?:' + '|'.join(REPLACED) + r')\(', text))
        other = {label: len(pattern.findall(text)) for label, pattern in BYPASS_PATTERNS.items()}
        other = {label: n for label, n in other.items() if n}
        if qtest or other:
            report[path.name] = {'qtest_calls_adapted': qtest, 'not_through_adapter': other}
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audit', action='store_true', required=True,
                        help='List per scene driver the QTest calls (adapted) and the UI changes made without QTest')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    report = audit_scene_drivers()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for name, row in report.items():
        other = ', '.join(f'{k} x{v}' for k, v in row['not_through_adapter'].items()) or '-'
        print(f"{name:42s} QTest adapted: {row['qtest_calls_adapted']:3d}   bypass: {other}")
    return 0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
