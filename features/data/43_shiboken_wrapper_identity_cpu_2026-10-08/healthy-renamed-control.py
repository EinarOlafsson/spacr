from pathlib import Path
import subprocess
from PySide6.QtWidgets import QApplication

ROOT = Path.cwd()
TEST = 'tests/qt/test_the_pyside_slot_warning_is_fixed_not_filtered.py'
app = QApplication.instance() or QApplication([])
class Bot:
    def __init__(self):
        self.widgets = []
    def addWidget(self, widget):
        self.widgets.append(widget)

old = subprocess.check_output(['git', 'show', f'HEAD:{TEST}'], text=True)
new = (ROOT / TEST).read_text()
for label, source in [('original', old), ('repaired', new)]:
    assert source.count('class Child(QWidget):') == 1
    assert source.count('return host, watcher, Child(host)') == 1
    source = source.replace('class Child(QWidget):', 'class HealthyChild(QWidget):').replace('return host, watcher, Child(host)', 'return host, watcher, HealthyChild(host)')
    scope = {'__file__': str(ROOT / TEST), '__name__': f'healthy_control_{label}'}
    exec(compile(source, TEST, 'exec'), scope)
    bot = Bot()
    host, watcher, child = scope['_poisonable'](bot)
    print(label, 'metaobject=', child.metaObject().className(), 'own_signal=', child.metaObject().indexOfSignal('poked(int)'), 'parent_count=', len(host.findChildren(type(child))), flush=True)
    try:
        scope['test_a_child_parented_into_a_watched_host_keeps_its_wrapper'](bot)
    except AssertionError as exc:
        if label != 'original':
            raise
        print('EXPECTED original rejects healthy renamed class:', str(exc), flush=True)
    else:
        assert label == 'repaired', 'original unexpectedly passed'
        print('PASS repaired preserves wrapper identity and original signal/parent guards', flush=True)
    for widget in bot.widgets:
        widget.close()
        widget.deleteLater()
    app.processEvents()
print('PASS healthy-name before/after control; historical macOS poisoning not reproduced or certified', flush=True)
