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
    assert _applies(zernike, "3.13")
    assert _applies(zernike, "3.14")


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


def test_default_native_backends_follow_actual_mac_wheel_boundaries():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    values = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(getattr(target, "id", "") == "dependencies"
                          for target in node.targets))
    requirements = {canonicalize_name(Requirement(value).name): Requirement(value)
                    for value in values}
    environment = default_environment()
    environment.update(sys_platform="darwin", platform_system="Darwin",
                       platform_machine="arm64", python_version="3.10",
                       python_full_version="3.10.22", platform_release="22.6.0")
    assert not requirements["pyfixest"].marker.evaluate(environment)
    environment.update(python_version="3.12", python_full_version="3.12.14")
    assert requirements["pyfixest"].marker.evaluate(environment)
    assert requirements["faiss-cpu"].marker.evaluate(environment)
    assert requirements["btrack"].marker.evaluate(environment)
    environment.update(python_version="3.14", python_full_version="3.14.7")
    assert not requirements["faiss-cpu"].marker.evaluate(environment)
    assert not requirements["btrack"].marker.evaluate(environment)
    assert not requirements["piper-tts"].marker.evaluate(environment)
    environment["platform_release"] = "23.6.0"
    assert requirements["faiss-cpu"].marker.evaluate(environment)
    assert requirements["piper-tts"].marker.evaluate(environment)
    assert not requirements["btrack"].marker.evaluate(environment)
    environment["platform_release"] = "24.0.0"
    assert requirements["faiss-cpu"].marker.evaluate(environment)
    assert requirements["btrack"].marker.evaluate(environment)
    environment.update(sys_platform="win32", platform_system="Windows",
                       platform_machine="AMD64", platform_release="10")
    for name in ("pyfixest", "faiss-cpu", "btrack"):
        assert requirements[name].marker.evaluate(environment)


def test_ultrack_profiles_follow_higra_native_platform_availability():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    values = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(getattr(target, "id", "") == "dependencies"
                          for target in node.targets))
    default = next(Requirement(value) for value in values
                   if canonicalize_name(Requirement(value).name) == "ultrack")
    profiles = (default, _requirement("ultrack", "ultrack"),
                _requirement("all", "ultrack"))
    environment = default_environment()
    environment.update(python_version="3.11", python_full_version="3.11.17")
    for system, platform, machine, available in (
            ("Linux", "linux", "aarch64", False),
            ("Linux", "linux", "arm64", False),
            ("Linux", "linux", "x86_64", True),
            ("Linux", "linux", "AMD64", True),
            ("Darwin", "darwin", "arm64", True),
            ("Darwin", "darwin", "x86_64", True),
            ("Windows", "win32", "AMD64", True)):
        environment.update(platform_system=system, sys_platform=platform,
                           platform_machine=machine)
        for requirement in profiles:
            assert requirement.marker is not None
            assert requirement.marker.evaluate(environment) is available, (
                str(requirement), environment)
