"""The min-deps CI profile must satisfy the floors setup.py declares.

It did not: setup.py required cellpose>=4.0.7 while
`.github/constraints/minimum-py39.txt` pinned 4.0.1, so the "Minimum
dependencies" job installed a version the package says it does not support
and failed ~23 tests that pass on every other job. The constraint's comment
still explained the 4.0.1 choice by reasoning that predated the raised
floor.

That is a bad failure to debug from the outside: the tests look broken,
the code is fine, and the answer is in a file nobody reads.
"""

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = ROOT / ".github" / "constraints" / "minimum-py39.txt"


def _declared_floors():
    """Lower bounds from setup.py, as {name: SpecifierSet}."""
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    floors = {}
    for match in re.finditer(
            r"""['"]([A-Za-z0-9_.\-]+)\s*([<>=!,\d.\s]*)['"]""", text):
        name, spec = match.group(1), match.group(2).strip()
        if not spec.startswith(">="):
            continue
        try:
            floors[name.lower()] = Requirement(f"{name}{spec}")
        except Exception:
            continue
    return floors


def _pins():
    """Exact pins from the constraints file, as {name: version str}."""
    pins = {}
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        pins[req.name.lower()] = str(req.specifier).lstrip("=")
    return pins


def test_the_constraints_file_exists():
    assert CONSTRAINTS.is_file(), f"{CONSTRAINTS} is missing"


def test_no_pin_is_below_its_declared_floor():
    """A pin under setup.py's floor makes the min-deps job test a
    configuration the project does not claim to support."""
    floors, violations = _declared_floors(), []
    for name, pinned in _pins().items():
        want = floors.get(name)
        if want is None:
            continue
        if Version(pinned) not in want.specifier:
            violations.append(
                f"{name}: constraints pin {pinned}, setup.py requires "
                f"{want.specifier}")
    assert not violations, "\n".join(violations)


def test_cellpose_specifically_matches(such_that=None):
    """Pinned because this is the one that bit, and because the cellpose
    signature genuinely differs across 4.0.x -- use_bfloat16 is absent in
    4.0.1 and `resample` defaults differently, which
    tests/test_cellpose_api_contract.py checks against the INSTALLED
    version."""
    floors = _declared_floors()
    assert "cellpose" in floors, "setup.py no longer declares a cellpose floor"
    pinned = _pins().get("cellpose")
    assert pinned, "constraints file no longer pins cellpose"
    assert Version(pinned) in floors["cellpose"].specifier
