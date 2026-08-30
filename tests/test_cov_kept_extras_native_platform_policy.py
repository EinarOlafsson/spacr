"""The 'all' extra must reuse the named extras' native-platform policy.

Preserved from tests/test_optional_dependency_python_markers.py, which was deleted while the behaviour it pins is
still live. Sixteen of that file's nineteen tests genuinely stopped
holding and were rightly dropped; this one still passes against the
current tree, so deleting it would have retired a real contract rather
than a stale one. Kept verbatim -- only the tests that no longer hold
were removed.
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

def test_all_reuses_the_named_extras_native_platform_policy():
    for extra, distribution in (("czi", "pylibCZIrw"),
                                ("zernike", "mahotas")):
        named = _requirement(extra, distribution)
        aggregate = _requirement("all", distribution)
        assert str(aggregate.specifier) == str(named.specifier)
        assert str(aggregate.marker) == str(named.marker)
