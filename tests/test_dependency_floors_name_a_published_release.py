"""A declared lower bound has to name a release somebody can install.

Every floor in ``setup.py`` is a promise the ``Minimum dependencies`` job
keeps: that job pins each floor from ``.github/constraints/minimum-py39.txt``
and runs the suite against it, which is the only thing that turns a lower
bound from advice into a tested claim.

A floor no published release equals breaks that silently. It cannot be
pinned, so it becomes the one bound in ``setup.py`` that nothing installs,
and nothing reports it: pip resolves such a bound happily to the nearest real
release above, the install succeeds, and the version CI skipped is the one a
pinned user lands on.

``nvidia-ml-py`` is the package where that hides, because its numbering makes
a boundary look like a floor. Each release is named after the NVIDIA driver
it binds, so the middle field is a driver branch rather than a minor version
and ``11.450.51`` sorts ABOVE a plausible-looking ``11.5``. Nothing is
published between ``10.418.84`` and ``11.450.51``, so the two spellings admit
the same releases and only one of them can be pinned.

Nothing here imports :mod:`spacr`: these are text and AST reads, so the file
runs in a cell with no scientific stack installed.
"""

from __future__ import annotations

import ast
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
CONSTRAINTS = ROOT / ".github" / "constraints" / "minimum-py39.txt"

#: The driver-numbered sequence ``nvidia-ml-py`` publishes, oldest first.
#:
#: Recorded rather than fetched so the rule holds in a cell with no network.
#: Later releases only extend this list, and every assertion below asks
#: whether a specific version is IN it, so appending never invalidates them.
#: The two entries that carry the argument are ``10.418.84`` and
#: ``11.450.51``: consecutive, and a round ``11.5`` falls between them.
NVML_PUBLISHED = (
    "2.285.01", "3.295.00", "4.304.01", "4.304.02", "4.304.03", "4.304.04",
    "6.340.0", "7.346.0", "7.352.0", "10.418.84",
    "11.450.51", "11.450.129", "11.460.79", "11.470.66", "11.495.46",
    "11.510.69", "11.515.0", "11.515.48", "11.515.75", "11.525.84",
    "11.525.112", "11.525.131", "11.525.150",
    "12.535.77", "12.535.108", "12.535.133", "12.535.161", "12.550.52",
    "12.550.89", "12.555.43", "12.560.30", "12.570.86", "12.570.172",
    "12.575.51",
    "13.580.65", "13.580.82", "13.580.126", "13.590.44", "13.590.48",
    "13.595.45", "13.610.43",
)

#: The first release in the supported band, and the floor setup.py declares.
NVML_FLOOR = "11.450.51"
#: What a reader who mistakes the driver branch for a minor version writes
#: instead: it admits the same releases and can never be pinned.
NVML_ROUND_BOUNDARY = "11.5"


def _declared(distribution: str) -> Requirement:
    """Return ``setup.py``'s core requirement for one distribution."""
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dependencies"
            for target in node.targets
        ):
            for value in ast.literal_eval(node.value):
                requirement = Requirement(value)
                if canonicalize_name(requirement.name) == canonicalize_name(
                    distribution
                ):
                    return requirement
    raise AssertionError(f"setup.py declares no core {distribution!r}")


def _floor(requirement: Requirement) -> Version:
    lows = [
        Version(clause.version)
        for clause in requirement.specifier
        if clause.operator in {">=", "==", "===", "~="}
    ]
    assert lows, f"{requirement.name} declares no lower bound"
    return min(lows)


def _pinned(distribution: str) -> Version:
    """Return the version the minimum-dependency profile installs."""
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) != canonicalize_name(distribution):
            continue
        exact = [
            clause.version
            for clause in requirement.specifier
            if clause.operator in {"==", "==="}
        ]
        assert len(exact) == 1, f"{line!r} must pin exactly one version"
        return Version(exact[0])
    raise AssertionError(f"the minimum profile pins no {distribution!r}")


def test_the_nvml_floor_names_a_release_that_exists():
    """A bound the distribution never publishes is a bound nothing can install."""
    floor = _floor(_declared("nvidia-ml-py"))
    assert str(floor) in NVML_PUBLISHED, (
        f"setup.py declares nvidia-ml-py>={floor}, which this distribution has "
        "never published. Its releases are named after the NVIDIA driver, so a "
        f"round bound such as {NVML_ROUND_BOUNDARY} is a boundary and not a "
        f"version; declare the first release that satisfies it instead "
        f"({NVML_FLOOR})."
    )


def test_the_minimum_profile_pins_the_nvml_floor_itself():
    """The floor is only a tested claim while the floor job installs it."""
    floor = _floor(_declared("nvidia-ml-py"))
    assert _pinned("nvidia-ml-py") == floor, (
        "the minimum-dependency profile installs a different nvidia-ml-py from "
        "the floor setup.py declares, so that floor is the one bound nothing "
        "runs the suite against"
    )


def test_naming_the_release_denies_a_user_nothing():
    """The exact release is free: it admits what the round boundary admits.

    This is what makes the choice one-sided. If declaring a pinnable floor
    narrowed the supported set, there would be a trade to argue about; it does
    not, because the distribution publishes nothing between ``10.418.84`` and
    ``11.450.51``.
    """
    declared = _declared("nvidia-ml-py")
    cap = SpecifierSet(
        ",".join(
            str(clause)
            for clause in declared.specifier
            if clause.operator in {"<", "<="}
        )
    )
    rounded = SpecifierSet(f">={NVML_ROUND_BOUNDARY}") & cap
    assert set(declared.specifier.filter(NVML_PUBLISHED)) == set(
        rounded.filter(NVML_PUBLISHED)
    )
    assert NVML_ROUND_BOUNDARY not in NVML_PUBLISHED
    assert NVML_FLOOR in NVML_PUBLISHED


def test_the_round_boundary_sorts_below_the_release_that_replaces_it():
    """The one fact a reader has to accept before the floor stops looking wrong.

    ``11.450.51`` is not a version below ``11.5``. The middle field is a
    driver branch, so 450 is a larger number than 5 and the release sorts
    ABOVE the boundary it satisfies. Everything else here follows from that,
    and it is exactly the step someone "correcting" the floor gets wrong.
    """
    assert Version(NVML_ROUND_BOUNDARY) < Version(NVML_FLOOR)
    assert Version(NVML_FLOOR).release[1] > 100, (
        "the floor no longer carries a driver-sized middle field, so the "
        "reason it is spelled this way has gone"
    )
    below = [v for v in NVML_PUBLISHED if Version(v) < Version(NVML_ROUND_BOUNDARY)]
    assert below[-1] == "10.418.84", (
        "the release immediately below the boundary is what proves nothing "
        f"sits between it and the floor; found {below[-1]}"
    )


@pytest.mark.network
def test_pypi_still_publishes_the_declared_nvml_floor():
    """The recorded sequence is checked against the index that serves it.

    The offline rule can only be as right as the list beside it. This is the
    same assertion made against live metadata, so a floor yanked from PyPI --
    which would make every minimum-dependency install fail at resolution --
    surfaces here rather than in CI's install step.
    """
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/nvidia-ml-py/json", timeout=30
        ) as response:
            published = set(json.load(response)["releases"])
    except (urllib.error.URLError, TimeoutError, OSError) as unreachable:
        pytest.skip(f"PyPI unreachable: {unreachable}")

    floor = _floor(_declared("nvidia-ml-py"))
    assert str(floor) in published, (
        f"nvidia-ml-py {floor} is no longer published; the declared floor "
        "cannot be installed"
    )
    assert set(NVML_PUBLISHED) <= published, (
        "releases recorded here are no longer on PyPI: "
        f"{sorted(set(NVML_PUBLISHED) - published)}"
    )
