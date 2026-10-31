"""The composer's AI and Protocols controls stay 36 px squares in a launched app.

A real launch installs the GUI scaling layer before the first widget. Under
it, the controls' own style sheet ``min-width: 0; min-height: 0`` replaced the
fixed 36 px minimum, and the two controls were drawn at 27 x 17 px. Run in a
child process, so the layer is not installed in the test worker itself.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]

PROBE = """
from PySide6.QtWidgets import QApplication
app = QApplication([])
from spacr.qt import gui_scale
gui_scale.install_scaling_layer()
from spacr.qt.widgets.home import _EasyHomePage
page = _EasyHomePage()
page.resize(1000, 800)
page.show()
app.processEvents()
edit = page.path_edit
print("SIZES", edit.ai.size().toTuple(), edit.protocols.size().toTuple())
"""


def test_the_controls_keep_their_size_under_the_scaling_layer(tmp_path):
    env = dict(os.environ)
    env.update(QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="",
               HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "config"),
               SPACR_HOME=str(tmp_path / "spacr"),
               PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=240)
    lines = [line for line in result.stdout.splitlines()
             if line.startswith("SIZES")]
    assert lines, result.stdout + result.stderr
    assert lines[-1] == "SIZES (36, 36) (36, 36)"
