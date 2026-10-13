"""Every screen module imports in a fresh interpreter, alone.

WHY A SUBPROCESS. Inside the test suite `spacr.qt` has already imported
PySide6 by the time any screen is reached, so an import-ORDER fault in a
screen is invisible here -- the damage is already prevented by whatever ran
first. Only a fresh interpreter that imports the one module can see it.

WHAT IT CAUGHT, 2026-09-04. `spacr/qt/screens/annotate.py` imported
`PIL.ImageQt` before PySide6. `PIL.ImageQt` resolves a Qt6 of its own at
import time, and when it wins the race PySide6 fails to load against it:

    from PIL.ImageQt import ImageQt      # first
    from PySide6.QtCore import Qt        # ImportError: undefined symbol
                                         # _ZN14QObjectPrivateC2E16QtPrivate_6_11_2

Reversing the two fixes it. The running application never hit this, because
`spacr.qt` imports PySide6 long before it reaches any screen -- which is
exactly what let it sit there. What it broke was importing the module on its
own: every test and tool that did failed at the import line, on an error
that names a Qt symbol and says nothing about ordering.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

SCREENS = Path(__file__).resolve().parents[2] / "spacr" / "qt" / "screens"

#: Screens whose import genuinely needs something heavier than a test run
#: should pay for. Empty on purpose -- add with a reason, never to silence.
SKIP: set = set()


def _modules():
    for path in sorted(SCREENS.glob("*.py")):
        if path.stem.startswith("_") or path.stem in SKIP:
            continue
        yield f"spacr.qt.screens.{path.stem}"


@pytest.mark.parametrize("module", list(_modules()))
def test_the_module_imports_in_a_fresh_interpreter(module):
    """Alone, with nothing else having loaded a Qt first."""
    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, timeout=300,
        env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin",
             "HOME": str(Path.home())},
    )
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-4:]
        pytest.fail(f"{module} does not import on its own:\n" +
                    "\n".join(tail))
