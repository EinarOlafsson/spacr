"""Contracts for the dependency profiles exercised by GitHub Actions."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
MINIMUM = ROOT / ".github" / "constraints" / "minimum-py39.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
COMPAT_WORKFLOW = ROOT / ".github" / "workflows" / "compat-matrix.yml"
MINIMUM_EXTRAS = (
    "qt", "dev", "zernike", "attribution", "btrack", "czi",
)


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
    assert "Fast / Full suite control (ubuntu-24.04, py3.12)" in workflow
    assert 'python-version: "3.12"' in workflow
    compat = COMPAT_WORKFLOW.read_text(encoding="utf-8")
    assert (
        'python-version: ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]'
        in compat
    )
    assert 'label: "Linux x86-64 / py3.14"' in compat
    assert 'python-version: "3.14"' in compat
    assert "  python315-linux:" in compat
    assert 'python-version: "3.15"' in compat
    assert "conda install --override-channels -c conda-forge cvxopt tabmat h5py -y" in compat
    assert "minimum-dependencies:" in workflow
    assert "-c .github/constraints/minimum-py39.txt" in workflow
    assert "python -m pip check" in workflow


def test_compatibility_metadata_job_stays_independent_of_scientific_stack():
    workflow = COMPAT_WORKFLOW.read_text(encoding="utf-8")
    metadata_job = workflow[
        workflow.index("  metadata:"):workflow.index("  current-python-installs:")
    ]
    assert "pytest --noconftest" in metadata_job


def test_macos_compatibility_installs_xgboost_openmp_runtime():
    workflow = COMPAT_WORKFLOW.read_text(encoding="utf-8")
    assert "Install OpenMP runtime (macOS only)" in workflow
    assert "if: runner.os == 'macOS'" in workflow
    assert "brew install libomp" in workflow


def test_project_harness_blocks_only_the_unrelated_napari_plugin(tmp_path):
    metadata = tmp_path / "spacr_harness_plugin_probe-0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: spacr-harness-plugin-probe\nVersion: 0.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[pytest11]\nnapari = unrelated_napari_plugin\n"
        "spacr_harness_probe = required_harness_plugin\n"
    )
    (tmp_path / "unrelated_napari_plugin.py").write_text(
        "raise AssertionError('the unrelated napari plugin must not load')\n"
    )
    (tmp_path / "required_harness_plugin.py").write_text(
        "from pathlib import Path\nimport pytest\n"
        "def pytest_collect_directory(path, parent):\n"
        "    assert Path(str(path)).resolve().is_relative_to(Path(__file__).resolve().parent), 'collection escaped the probe directory'\n"
        "@pytest.fixture\ndef required_plugin_fixture():\n"
        "    return 'other plugins remain available'\n"
    )
    probe = tmp_path / "test_logging_probe.py"
    probe.write_text(
        "import logging\n"
        "def test_mapping_log_and_other_plugin(required_plugin_fixture, caplog):\n"
        "    assert required_plugin_fixture == 'other plugins remain available'\n"
        "    options = {'tm_lim': 2500, 'mip_gap': 0.05}\n"
        "    with caplog.at_level(logging.INFO):\n"
        "        logging.getLogger('spacr.timelapse').info('Options: %s', options)\n"
        "    assert caplog.records[-1].args == options\n"
        "    assert caplog.records[-1].getMessage() == f'Options: {options}'\n"
    )
    environment = os.environ.copy()
    for name in (
        "PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_XDIST_TESTRUNUID", "SPACR_TEST_COUNT_OWNER_PID",
        "SPACR_TEST_COUNT_SESSION_ID",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--noconftest", "-q",
         "-c", str(ROOT / "pytest.ini"), "--rootdir", str(tmp_path),
         "--confcutdir", str(tmp_path), str(probe)],
        env=environment, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
