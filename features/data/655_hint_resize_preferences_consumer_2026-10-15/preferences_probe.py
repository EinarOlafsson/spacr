"""One actual Preferences consumer probe with private configuration."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

started = time.monotonic()
root = Path.cwd()
expected = '58a19d0775b6337d184c8d4caffdadff643ebf93'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == expected
paths = ['spacr/qt/widgets/hint_bar.py', 'spacr/qt/preferences.py']
source_hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
network_attempts = []
def audit(event, args):
    if event == 'socket.connect' and args[0].family in (socket.AF_INET, socket.AF_INET6):
        network_attempts.append(repr(args[1]))
        raise AssertionError('Network connection forbidden in Preferences probe')
sys.addaudithook(audit)

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from spacr.qt import preferences as P, theme
from spacr.qt.widgets.hint_bar import HintBar

app = QApplication([])
theme.apply_qpalette(app)
app.setStyleSheet(theme.stylesheet())
assert P._SAFE_MODE is False
settings_path = Path(P._settings().fileName()).resolve()
private_config = Path(os.environ['XDG_CONFIG_HOME']).resolve()
assert settings_path.is_relative_to(private_config), (settings_path, private_config)
assert not P._settings().contains(P._KEY_PREFERENCES_HELP_HEIGHT)

dialog = P.PreferencesDialog(None)
dialog.show()
dialog.activateWindow()
for _ in range(3):
    app.processEvents()
bar = dialog.findChild(HintBar)
handle = bar._resize_handle
start = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel)
start.setFocus(Qt.TabFocusReason)
assert start.hasFocus()
traversed = []
for count in range(1, 257):
    assert time.monotonic() - started < 40
    focused = app.focusWidget()
    assert focused is not None
    QTest.keyClick(focused, Qt.Key_Tab)
    app.processEvents()
    focused = app.focusWidget()
    traversed.append({'class': type(focused).__name__, 'object_name': focused.objectName()})
    if focused is handle:
        break
else:
    raise AssertionError('Real Preferences Tab chain did not reach the help grip')
assert handle.hasFocus()
caption = 'Drag this edge or use Up and Down to resize the help area.'
assert bar.explains(handle) == caption
interface = QAccessible.queryAccessibleInterface(handle)
assert interface is not None
assert interface.text(QAccessible.Name) == caption
assert interface.text(QAccessible.Description) == caption
QTest.mouseMove(handle, handle.rect().center())
app.processEvents()
assert bar.text() == caption, bar.text()
before = bar.height()
maximum = bar._maximum_help_height()
step = max(1, bar.fontMetrics().lineSpacing())
assert maximum > before
committed = []
bar.helpHeightCommitted.connect(committed.append)
QTest.keyClick(handle, Qt.Key_Up)
app.processEvents()
after = bar.height()
assert after == min(before + step, maximum)
assert after > before
assert committed == [after]
P._settings().sync()
saved = int(P._settings().value(P._KEY_PREFERENCES_HELP_HEIGHT))
assert saved == after
QTest.keyClick(handle, Qt.Key_Tab)
assert bar._view.hasFocus()
dialog.close()
app.processEvents()
reopened = P.PreferencesDialog(None)
reopened.show()
for _ in range(4):
    app.processEvents()
restored = reopened.findChild(HintBar)
assert restored.height() == saved, (restored.height(), saved)
reopened.close()
app.processEvents()
assert not network_attempts
assert source_hashes == {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
print(json.dumps({'status': 'PASS', 'source_commit': expected, 'source_hashes': source_hashes,
    'configuration_file': str(settings_path), 'tab_presses': count, 'tab_path': traversed,
    'accessible_name': interface.text(QAccessible.Name),
    'accessible_description': caption, 'hover_help': caption,
    'height_before': before, 'font_step': step, 'maximum': maximum,
    'height_after': after, 'committed_heights': committed, 'saved_key': P._KEY_PREFERENCES_HELP_HEIGHT,
    'saved_height': saved, 'reopened_height': restored.height(),
    'tab_leaves_to_help_text': True, 'network_attempts': network_attempts,
    'elapsed_seconds': round(time.monotonic() - started, 3),
    'scope': 'Actual Preferences consumer execution, offscreen. Source author; not an independent source review or native-platform claim.'}, indent=2))
