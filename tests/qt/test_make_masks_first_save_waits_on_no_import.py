"""The first mask save of a fresh process waits on no deep-learning import.

A save asked :func:`spacr.qt.mask_engine.pixel_size_nm` for the source's
calibration, and for a TIFF that imported :mod:`spacr.io`, which imports
torch and torchvision. The mask writer thread then waited on that import,
and Next or Keep, which wait for pending writes before the next field is
read, held the GUI thread as long. A cold first import is seconds; with
the interpreter lock busy it was minutes. A TIFF's calibration is now read
with tifffile alone.

The run is a fresh process in which importing :mod:`spacr.io` never
finishes, a stand-in for the slowest cold import. The save must still land
within a few seconds, and the window must keep pumping events through the
save and the move to the next field. When the save and the move take more
than two seconds together, the child dumps every thread's stack to its
standard error, which a failure shows. A calibrated TIFF is opened and read
for its pixel size through tifffile alone too, so neither the field's load
nor its save imports torch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

REPO = Path(__file__).resolve().parents[2]

CHILD = r'''
import faulthandler, json, os, sys, threading, time

faulthandler.dump_traceback_later(60, exit=True)

held = threading.Event()


class NeverFinishes:
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        held.set()
        threading.Event().wait()


class HoldSpacrIo:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name != "spacr.io":
            return None
        for finder in sys.meta_path:
            if finder is HoldSpacrIo or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                spec.loader = NeverFinishes(spec.loader)
                return spec
        return None


sys.meta_path.insert(0, HoldSpacrIo)

import numpy as np
import tifffile
from PySide6.QtCore import QEventLoop, QSettings, QTimer
from PySide6.QtWidgets import QApplication

app = QApplication([])
from spacr.qt import prefs
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from tests.qt.test_gui_responsiveness import LoopWatchdog

tmp = sys.argv[1]
calibrated = sys.argv[2] == "calibrated"
prefs._s = lambda: QSettings(os.path.join(tmp, "editor.ini"), QSettings.IniFormat)
folder = os.path.join(tmp, "fields")
os.makedirs(folder)
for index in range(2):
    tifffile.imwrite(os.path.join(folder, f"field_{index}.tif"),
                     np.full((64, 64), index + 1, dtype=np.uint16),
                     **({"resolution": (1e6, 1e6), "resolutionunit": "CENTIMETER"}
                        if calibrated else {}))


def pump_until(done, limit):
    deadline = time.monotonic() + limit
    while not done() and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()
    return done()


screen = mm.MakeMasksScreen()
screen.resize(900, 700)
screen.show()
assert screen._open_folder(folder)
assert pump_until(lambda: screen._canvas.mask is not None, 30)
dog = LoopWatchdog()
faulthandler.dump_traceback_later(2, exit=False)
dog.start()
screen._canvas.mask[12:20, 12:20] = 7
began = time.monotonic()
screen._on_save()
saved = pump_until(lambda: not mm._mask_writes_pending(), 5)
seconds = time.monotonic() - began
screen._btn_next.click()
moved = pump_until(lambda: screen._current_index == 1 and not screen._loading, 5)
dog.stop()
faulthandler.dump_traceback_later(60, exit=True)
path = engine.mask_save_path(folder, "field_0.tif")
print(json.dumps({
    "saved": saved, "seconds": seconds, "moved": moved, "worst": dog.worst,
    "exists": os.path.isfile(path),
    "spacr_io_held": held.is_set(),
    "pixel_size_nm": engine.pixel_size_nm(os.path.join(folder, "field_0.tif")),
    "torch_imported": "torch" in sys.modules,
}), flush=True)
os._exit(0)
'''


@pytest.mark.parametrize("calibration", ["uncalibrated", "calibrated"])
def test_a_fresh_process_saves_and_moves_on_while_spacr_io_cannot_import(
        tmp_path, calibration):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="",
               SPACR_HOME=str(tmp_path / "home"), SPACR_NO_NETWORK="1",
               PYTHONPATH=os.pathsep.join(
                   [str(REPO)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    out = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path), calibration],
                         capture_output=True, text=True, timeout=180,
                         cwd=str(REPO), env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    said = (result, out.stderr[-12000:])
    assert result["saved"] and result["exists"], said
    assert result["seconds"] < 3.0, said
    assert result["moved"], said
    assert result["worst"] < 1.0, said
    assert result["spacr_io_held"], said
    assert not result["torch_imported"], said
    expected = 10.0 if calibration == "calibrated" else None
    assert result["pixel_size_nm"] == (pytest.approx(expected) if expected else None), said
