"""Closing a dialog must not invalidate the application's shared display."""

import os
import subprocess
import sys

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget


def test_closing_annotate_settings_preserves_the_shared_screen(tmp_path):
    code = """
import gc
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog
import shiboken6
from spacr.qt.annotate_engine import AnnotateSettings
from spacr.qt.screens.annotate import _SettingsDialog

app = QApplication([])
screen = app.primaryScreen()
geometry = screen.geometry()
for _ in range(3):
    dialog = _SettingsDialog(AnnotateSettings())
    dialog.loop = dialog
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    del dialog
    gc.collect(0)
    gc.collect()
    assert shiboken6.isValid(screen), 'Closing Annotate invalidated the shared screen'
    assert screen.geometry() == geometry
    assert app.primaryScreen() is screen
following = QDialog()
print('display survived three dialogs and collection')
"""
    env = dict(os.environ)
    for key, name in (("HOME", "home"), ("XDG_CONFIG_HOME", "config")):
        folder = tmp_path / name
        folder.mkdir()
        env[key] = str(folder)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=90, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "display survived" in result.stdout


@pytest.mark.parametrize("visible", [False, True])
def test_screen_lookup_uses_the_visible_anchor_center(qtbot, monkeypatch, visible):
    from spacr.qt.hidpi import screen_for_widget

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setGeometry(120, 90, 640, 480)
    parent.show()
    child = QWidget(parent)
    child.setGeometry(20, 30, 180, 100)
    child.setVisible(visible)
    anchor = child if visible else parent
    expected = anchor.mapToGlobal(QPoint(anchor.width() // 2, anchor.height() // 2))
    found = object()
    points = []

    def at(point):
        points.append(point)
        return found

    monkeypatch.setattr(QGuiApplication, "screenAt", at)
    assert screen_for_widget(child) is found
    assert points == [expected]


def test_offscreen_position_falls_back_to_primary(qtbot, monkeypatch):
    from spacr.qt.hidpi import screen_for_widget

    widget = QWidget()
    qtbot.addWidget(widget)
    primary = QGuiApplication.primaryScreen()
    monkeypatch.setattr(QGuiApplication, "screenAt", lambda point: None)
    assert screen_for_widget(widget) is primary
    assert screen_for_widget() is primary
