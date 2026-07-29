"""Regression tests for third-party warnings emitted during Qt startup."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]


def _core_dependency_names():
    """Return normalized distribution names declared by ``setup.py``."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dependencies"
            for target in node.targets
        ):
            dependencies = ast.literal_eval(node.value)
            return {
                Requirement(value).name.lower().replace("_", "-")
                for value in dependencies
            }
    raise AssertionError("setup.py has no module-level dependencies list")


def test_spacr_declares_the_maintained_nvml_distribution():
    names = _core_dependency_names()
    assert "nvidia-ml-py" in names
    assert "pynvml" not in names


def test_startup_suppresses_only_the_known_third_party_future_notices():
    """A fresh process mirrors the warning order of the installed CLI."""
    code = r'''
import warnings
import spacr

# The deprecated pynvml compatibility distribution installs a .pth hook, so
# existing environments can still emit this even after spaCR's dependency is
# corrected. The filter keeps an upgrade quiet until that wrapper is removed.
warnings.warn(
    "The pynvml package is deprecated. Please install nvidia-ml-py instead.",
    FutureWarning,
)
warnings.warn(
    "You are using a Python version (3.10.19) which Google will stop "
    "supporting in new releases of google.api_core.",
    FutureWarning,
)

# Exercise the real heavy-import paths too. They are optional in packaging
# metadata tests, hence the guarded imports.
try:
    import torch
except ImportError:
    pass
try:
    import spacr.ml
except ImportError:
    pass
print("startup-imports-complete")
'''
    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "startup-imports-complete" in proc.stdout
    assert "pynvml package is deprecated" not in proc.stderr
    assert "logit link alias is deprecated" not in proc.stderr
    assert "Google will stop supporting" not in proc.stderr


def test_unrelated_future_warnings_are_not_hidden():
    code = r'''
import warnings
import spacr
warnings.warn("spaCR test sentinel", FutureWarning)
'''
    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert "spaCR test sentinel" in proc.stderr
