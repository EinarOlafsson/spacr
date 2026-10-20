import json, os
from PySide6 import __version__
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
app=QApplication([])
app.setApplicationName('spaCR N655 Qt control')
window=QWidget(); window.setWindowTitle('spaCR accessibility control')
layout=QVBoxLayout(window)
button=QPushButton('External reader control button'); layout.addWidget(button)
button.clicked.connect(lambda: print('ACTIVATED',flush=True))
window.show()
QAccessible.setActive(True)
QTimer.singleShot(1500,lambda:print('READY '+json.dumps({'pid':os.getpid(),'qt':__version__,'internal':QAccessible.queryAccessibleInterface(app).childCount()}),flush=True))
QTimer.singleShot(18000,app.quit)
app.exec()
