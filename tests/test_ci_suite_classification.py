"""Contracts for the mutually exclusive GitHub Actions pytest suites."""

import ast
from pathlib import Path

from tests.conftest import _automatic_ci_markers

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
REUSABLE = ROOT / ".github" / "workflows" / "_pytest-suite.yml"
TIMELAPSE = ROOT / "spacr" / "timelapse.py"


def test_qt_modules_are_classified_automatically():
    markers = _automatic_ci_markers(
        Path("/workspace/tests/qt/test_home.py"))

    assert markers == {"qt"}


def test_end_to_end_qt_module_has_both_structural_markers():
    markers = _automatic_ci_markers(
        Path("/workspace/tests/qt/test_e2e_pipeline.py"))

    assert markers == {"qt", "integration"}


def test_pipeline_and_real_data_modules_are_integrations():
    for name in (
        "test_pipeline_training_analysis.py",
        "test_hf_e2e_integration.py",
        "test_real_data_image_modules.py",
        "test_e2e_real_dataset.py",
    ):
        assert _automatic_ci_markers(
            Path("/workspace/tests") / name) == {"integration"}


def test_ordinary_unit_module_has_no_structural_marker():
    assert _automatic_ci_markers(
        Path("/workspace/tests/test_schema.py")) == set()


def test_workflow_has_one_job_per_declared_suite():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for job in ("fast", "integration", "slow", "qt", "gpu", "network", "nas"):
        assert f"\n  {job}:" in workflow
    assert workflow.count("uses: ./.github/workflows/_pytest-suite.yml") == 6


def test_marker_expressions_partition_resource_and_structural_suites():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    expected = (
        "not integration and not slow and not heavy and not qt and not gpu "
        "and not network and not nas and not gui",
        "integration and not slow and not heavy and not qt and not gpu and "
        "not network and not nas and not gui",
        "(slow or heavy) and not gpu and not network and not nas and not gui",
        "qt and not slow and not heavy and not gpu and not network and not nas "
        "and not gui",
        'marker_expression: "gpu and not gui"',
        "network and not nas and not gpu and not gui",
        "nas and not gpu and not gui",
    )
    for expression in expected:
        assert expression in workflow


def test_reusable_suite_auto_detects_resources_and_current_actions():
    workflow = REUSABLE.read_text(encoding="utf-8")

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v7" in workflow
    assert "nvidia-smi" in workflow
    assert "cuda_available" in workflow
    assert "endpoint_available" in workflow
    assert "NUMBA_CACHE_DIR" in workflow


def test_timelapse_does_not_download_btrack_data_during_collection():
    """The dataset registry belongs inside tracking, never at module import."""
    tree = ast.parse(TIMELAPSE.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "btrack"
        and any(alias.name == "datasets" for alias in node.names)
        for node in top_level_imports
    )
