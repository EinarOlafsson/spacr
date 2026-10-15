"""Identify a refused startup socket, without replaying GUI locale checks."""
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
OUT = Path('/mnt/wd4tb/scratch/f564-gui-consumer-20261015/startup-trace.json')
sys.path.insert(0, str(ROOT))
phase = 'imports'
attempts = []

def audit(event, args):
    if event == 'socket.connect':
        attempts.append({'phase': phase, 'endpoint': repr(args), 'stack': traceback.format_stack()})
        OUT.write_text(json.dumps(attempts, indent=2) + '\n')
        raise AssertionError('Startup trace refuses socket before connect')

sys.addaudithook(audit)
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication
from spacr.qt import preferences
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
app = QApplication.instance() or QApplication([])
if not attempts:
    phase = 'construction'
    with patch.object(preferences, '_settings', lambda: QSettings(str(OUT.with_suffix('.ini')), QSettings.IniFormat)):
        screen = AppScreen('activation')
        if not attempts:
            phase = 'lazy_counterfactual_heading'
            screen._open_the_heading_of('counterfactuals')
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
OUT.write_text(json.dumps(attempts, indent=2) + '\n')
print(json.dumps({'trace_only': True, 'blocked_attempts': len(attempts), 'phases': [a['phase'] for a in attempts]}))
