"""Heavy segmentation dependencies stay behind their feature boundary."""
from __future__ import annotations

import os
import subprocess
import sys


def test_importing_utils_does_not_import_cellpose():
    code = """
import sys
import spacr.utils as utils
loaded = [
    name for name in sys.modules
    if name == 'cellpose' or name.startswith('cellpose.')
]
assert loaded == [], loaded
assert 'not yet imported' in repr(utils.cp_models)
"""
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
