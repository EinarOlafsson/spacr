"""Contracts for the dependency profiles exercised by GitHub Actions."""

from __future__ import annotations

import ast
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
MINIMUM = ROOT / ".github" / "constraints" / "minimum-py39.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
MINIMUM_EXTRAS = ("qt", "dev", "zernike", "attribution")


def _declared_requirements() -> list[Requirement]:
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    core: list[str] | None = None
    extras: dict[str, list[str]] | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "dependencies"
                for target in node.targets
            ):
                core = ast.literal_eval(node.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        else:
            call = None
        if call and isinstance(call.func, ast.Name) and call.func.id == "setup":
            for keyword in call.keywords:
                if keyword.arg == "extras_require":
                    extras = ast.literal_eval(keyword.value)

    assert core is not None
    assert extras is not None
    selected = list(core)
    for name in MINIMUM_EXTRAS:
        selected.extend(extras[name])
    return [Requirement(value) for value in selected]


def _minimum_constraints() -> dict[str, Requirement]:
    constraints = {}
    for raw_line in MINIMUM.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        assert name not in constraints, f"duplicate constraint for {name}"
        constraints[name] = requirement
    return constraints


def _applies_to_minimum_job(requirement: Requirement) -> bool:
    if requirement.marker is None:
        return True
    environment = default_environment()
    environment.update(
        python_version="3.9",
        python_full_version="3.9.25",
        platform_system="Linux",
        sys_platform="linux",
    )
    return requirement.marker.evaluate(environment)


def _has_lower_bound(requirement: Requirement) -> bool:
    return any(
        specifier.operator in {"==", "===", ">=", ">", "~="}
        for specifier in requirement.specifier
    )


def test_minimum_profile_covers_every_lower_bounded_direct_requirement():
    declared = [
        requirement
        for requirement in _declared_requirements()
        if _applies_to_minimum_job(requirement) and _has_lower_bound(requirement)
    ]
    constraints = _minimum_constraints()

    missing = sorted(
        requirement.name
        for requirement in declared
        if canonicalize_name(requirement.name) not in constraints
    )
    assert not missing, f"minimum profile is missing direct requirements: {missing}"


def test_minimum_profile_versions_obey_declared_ranges():
    declared = {
        canonicalize_name(requirement.name): requirement
        for requirement in _declared_requirements()
        if _applies_to_minimum_job(requirement)
    }

    for name, constraint in _minimum_constraints().items():
        assert name in declared, f"constraint {name!r} is not installed by the CI job"
        exact = [
            specifier.version
            for specifier in constraint.specifier
            if specifier.operator in {"==", "==="}
        ]
        assert len(exact) == 1, f"{constraint} must contain one exact version"
        assert Version(exact[0]) in declared[name].specifier, (
            f"{constraint} is outside declared range {declared[name].specifier}"
        )


def test_ci_exercises_minimum_and_newest_dependency_profiles():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "Newest compatible (ubuntu-24.04" in workflow
    assert 'python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]' in workflow
    assert "minimum-dependencies:" in workflow
    assert "-c .github/constraints/minimum-py39.txt" in workflow
    assert "python -m pip check" in workflow
