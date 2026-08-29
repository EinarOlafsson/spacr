"""The aggregate optional install remains resolvable on Python 3.14.

The two native readers/descriptors covered here publish no CPython-3.14
wheel.  A prose note does not influence pip, so this test evaluates the
actual PEP 508 markers in both their named extras and ``spacr[all]``.
"""
from __future__ import annotations

import ast
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _extras() -> dict[str, list[str]]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "extras_require":
                return ast.literal_eval(keyword.value)
    raise AssertionError("setup.py does not declare extras_require")


def _requirement(extra: str, distribution: str) -> Requirement:
    wanted = canonicalize_name(distribution)
    matches = [
        Requirement(value)
        for value in _extras()[extra]
        if canonicalize_name(Requirement(value).name) == wanted
    ]
    assert len(matches) == 1, (extra, distribution, matches)
    return matches[0]


def _applies(requirement: Requirement, version: str) -> bool:
    environment = default_environment()
    environment.update(
        python_version=version,
        python_full_version=f"{version}.0",
    )
    return requirement.marker is None or requirement.marker.evaluate(environment)


def test_native_features_follow_their_last_wheel_interpreter():
    czi = _requirement("czi", "pylibCZIrw")
    assert _applies(czi, "3.13")
    assert not _applies(czi, "3.14")

    zernike = _requirement("zernike", "mahotas")
    assert _applies(zernike, "3.12")
    assert not _applies(zernike, "3.13")
    assert not _applies(zernike, "3.14")


def test_all_reuses_the_named_extras_native_platform_policy():
    for extra, distribution in (("czi", "pylibCZIrw"),
                                ("zernike", "mahotas")):
        named = _requirement(extra, distribution)
        aggregate = _requirement("all", distribution)
        assert str(aggregate.specifier) == str(named.specifier)
        assert str(aggregate.marker) == str(named.marker)


def test_tracker_markers_cover_real_transitive_python_and_torch_limits():
    trackastra = _requirement("trackastra", "trackastra")
    ultrack = _requirement("ultrack", "ultrack")

    assert not _applies(trackastra, "3.9")
    assert _applies(trackastra, "3.10")
    assert not _applies(ultrack, "3.9")
    assert _applies(ultrack, "3.10")
    assert _applies(ultrack, "3.13")
    assert not _applies(ultrack, "3.14")


def test_intel_macos_omits_extras_whose_transitive_wheels_end_early():
    environment = default_environment()
    environment.update(
        python_version="3.14",
        python_full_version="3.14.0",
        sys_platform="darwin",
        platform_system="Darwin",
        platform_machine="x86_64",
    )
    for extra, distribution in (("trackastra", "trackastra"),
                                ("ultrack", "ultrack"),
                                ("tutorial", "piper-tts")):
        requirement = _requirement(extra, distribution)
        assert requirement.marker is not None
        assert not requirement.marker.evaluate(environment)
