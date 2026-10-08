import os
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
app = QApplication([])
QAccessible.setActive(True)
window = QWidget()
window.setWindowTitle('n655-qt-positive-control')
layout = QVBoxLayout(window)
button = QPushButton('Positive control button', window)
button.setAccessibleName('Positive control button')
layout.addWidget(button)
window.show()
QAccessible.setRootObject(app)
print('INTERNAL', QAccessible.queryAccessibleInterface(app).childCount(), flush=True)
print('READY', os.getpid(), flush=True)
QTimer.singleShot(8000, app.quit)
app.exec()
