"""The magnifier's startup timer does not load stroke-only dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_magnifier_timer_import_defers_scipy_until_a_frame_is_delivered():
    script = """
import sys
import numpy as np
from spacr.qt._magnifier_drag import _DragStroke, _PREVIEW_MS
assert _PREVIEW_MS == 40
assert 'scipy' not in sys.modules
stroke = _DragStroke((4, 4), (1, 1), step=0)
stroke.expect('frame')
labels = np.zeros((4, 4), dtype=np.int32)
labels[1:3, 1:3] = 1
assert stroke.deliver('frame', labels, (0, 0, 4, 4))
assert stroke.outcome().objects == 1
assert 'scipy' in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=environment,
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
