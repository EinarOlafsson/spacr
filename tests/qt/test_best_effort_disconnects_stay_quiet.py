"""A best-effort disconnect stays silent on PySide6 6.11, as it was on 6.10.

6.11 answers a disconnect of a slot that is not connected with
``RuntimeWarning: libpyside: Failed to disconnect`` instead of raising, so
spaCR's ``try/except`` disconnects printed it (the Data Manager on every
close). PySide6 6.10 and older raise instead, which the ``try`` below takes,
as spaCR's own disconnects do. Checked in a fresh interpreter, because pytest
resets warning filters around each test.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CODE = r"""
import warnings
warnings.simplefilter("default")
import spacr.qt
from PySide6.QtCore import QObject, Signal
class Source(QObject):
    changed = Signal()
source = Source()
try:
    source.changed.disconnect(lambda: None)
except (RuntimeError, TypeError):
    pass
warnings.warn("an unrelated runtime warning", RuntimeWarning)
"""


def test_a_refused_disconnect_prints_nothing_but_other_warnings_still_do():
    done = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True,
                          timeout=120, env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
                                            "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0, done.stderr
    assert "Failed to disconnect" not in done.stderr, done.stderr
    assert "an unrelated runtime warning" in done.stderr, done.stderr
