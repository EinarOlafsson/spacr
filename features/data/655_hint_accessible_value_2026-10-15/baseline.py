import hashlib
import json
from pathlib import Path
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QAccessibleActionInterface
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QSplitter, QVBoxLayout, QWidget
from spacr.qt import i18n
from spacr.qt.widgets.hint_bar import HintBar

app = QApplication.instance() or QApplication([])
host = QWidget()
layout = QVBoxLayout(host)
page = QWidget()
page_layout = QVBoxLayout(page)
page_layout.addWidget(QPushButton('Before'))
page_layout.addStretch()
bar = HintBar('Help text.')
layout.addWidget(page)
layout.addWidget(bar)
layout.addWidget(QPushButton('After'))
host.resize(420, 700)
host.show()
host.activateWindow()
i18n.retranslate_widget_tree(host, 'en')
app.processEvents()
handle = bar._resize_handle
handle.setFocus(Qt.TabFocusReason)
interface = QAccessible.queryAccessibleInterface(handle)
actions = interface.actionInterface()
record = {'source_sha256': hashlib.sha256(Path('spacr/qt/widgets/hint_bar.py').read_bytes()).hexdigest(), 'role': interface.role().name, 'name': interface.text(QAccessible.Name), 'description': interface.text(QAccessible.Description), 'focusable': bool(interface.state().focusable), 'actions': actions.actionNames() if actions else [], 'value_interface': interface.valueInterface() is not None, 'value_text_before': interface.text(QAccessible.Value), 'height_before': bar.height()}
QTest.keyClick(handle, Qt.Key_Up)
app.processEvents()
record.update(height_after=bar.height(), value_text_after=interface.text(QAccessible.Value))
splitter = QSplitter(Qt.Vertical)
splitter.addWidget(QWidget())
splitter.addWidget(QWidget())
splitter.resize(400, 600)
splitter.show()
app.processEvents()
native = QAccessible.queryAccessibleInterface(splitter.handle(1))
record['native_splitter'] = {'role': native.role().name, 'value_interface': native.valueInterface() is not None, 'actions': native.actionInterface().actionNames() if native.actionInterface() else []}
Path('/mnt/wd4tb/scratch/home-b-n655-hint-value-20261015/baseline.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
assert record['height_after'] > record['height_before']
assert QAccessibleActionInterface.increaseAction() in record['actions'], 'Accessible client cannot invoke the existing grow action'
assert QAccessibleActionInterface.decreaseAction() in record['actions'], 'Accessible client cannot invoke the existing shrink action'
