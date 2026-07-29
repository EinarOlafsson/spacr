"""SHAP is imported only by explanation features, not general analysis."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ("spacr.ml", "spacr.sim", "spacr.submodules"))
def test_analysis_module_import_does_not_import_shap(module):
    code = f"""
import sys
import {module}
loaded = [
    name for name in sys.modules
    if name == 'shap' or name.startswith('shap.')
]
assert loaded == [], loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
