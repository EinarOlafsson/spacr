"""Contracts for the mutually exclusive GitHub Actions pytest suites."""

import ast
from pathlib import Path

from tests.conftest import _automatic_ci_markers, _ci_file_shard

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
REUSABLE = ROOT / ".github" / "workflows" / "_pytest-suite.yml"
TIMELAPSE = ROOT / "spacr" / "timelapse.py"

PARALLEL_MEMORY_AMPLIFIERS = {
    "test_no_tensorflow_guard.py": {
        "test_importing_spacr_does_not_load_tensorflow",
    },
    "test_group_lasso_keeps_the_gene_together.py": {
        "test_lasso_splits_a_gene_where_group_lasso_cannot",
    },
    "test_cov_submodules_vision_model.py": {
        "test_shap_sample_explains_one_percent_of_the_objects",
    },
    "test_plate_position_is_a_setting.py": {
        "test_regression_levels_passes_the_setting_through_to_both_fits",
    },
    "test_core_umap_graphs.py": {
        "test_generate_image_umap_embedding_by_controls",
    },
    "test_core_umap_validation.py": {
        "test_screen_graphs_over_two_sources_writes_three_result_sets",
    },
    "test_core_mask_orchestration.py": {
        "test_test_mode_plots_every_merged_field",
        "test_missing_merged_folder_reports_and_continues",
    },
    "test_all_plotting_functions.py": {
        "test_plot_image_mask_overlay",
    },
    "test_object_tstack_wiring.py": {
        "test_verbose_reports_what_the_4d_run_actually_did",
    },
}

SERIAL_TIMING_NODES = {
    "test_fast_plots.py": {
        "test_the_plain_volcano_is_immediate",
        "test_colouring_does_not_cost_a_brush_per_point",
    },
}


def _is_heavy_marker(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "heavy"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _is_slow_marker(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "slow"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


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


def test_parallel_memory_amplifiers_are_assigned_to_the_serial_suite():
    """Measured high-RSS nodes must not overlap in the parallel fast job."""
    missing = []
    for filename, expected_names in PARALLEL_MEMORY_AMPLIFIERS.items():
        tree = ast.parse((ROOT / "tests" / filename).read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in expected_names:
            function = functions.get(name)
            if function is None or not any(
                    _is_heavy_marker(marker)
                    for marker in function.decorator_list):
                missing.append(f"{filename}::{name}")

    assert not missing, "heavy marker missing from: " + ", ".join(missing)


def test_fast_suites_fit_on_a_standard_hosted_runner():
    """Small worker batches release accumulated scientific-library memory."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("tools/run_pytest_batches.py") == 2
    assert workflow.count("--batch-size 32 --workers 2") == 2
    assert "-n 3 --dist loadfile" not in workflow


def test_wall_clock_measurements_run_in_the_serial_suite():
    """Timing thresholds must not measure contention from xdist siblings."""
    missing = []
    for filename, expected_names in SERIAL_TIMING_NODES.items():
        tree = ast.parse((ROOT / "tests" / filename).read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in expected_names:
            function = functions.get(name)
            if function is None or not any(
                    _is_slow_marker(marker)
                    for marker in function.decorator_list):
                missing.append(f"{filename}::{name}")

    assert not missing, "slow marker missing from: " + ", ".join(missing)


def test_reusable_suite_auto_detects_resources_and_current_actions():
    workflow = REUSABLE.read_text(encoding="utf-8")

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v7" in workflow
    assert "nvidia-smi" in workflow
    assert "cuda_available" in workflow
    assert "endpoint_available" in workflow
    assert "NUMBA_CACHE_DIR" in workflow
    assert "SPACR_HF_E2E_STUB" in workflow
    # Qt collects the external-catalog fixed-point contract. Chinese source
    # normalization is deliberately OpenCC-backed, so the reusable runner
    # must provide the same audited normalizer as docs and release jobs.
    assert "libopencc1.1" in workflow
    assert "libopencc-data" in workflow


def test_qt_measurement_suites_run_after_xdist_workers_exit():
    """Cursor, event-loop, and RSS measurements need an idle process.

    The offscreen Qt platform exposes one synthetic cursor across workers,
    and event-loop/RSS budgets become measurements of sibling-worker load
    when these files run inside xdist.  The reusable suite must exclude each
    file from the parallel pass and run it explicitly in the serial tail.
    """
    workflow = (ROOT / ".github" / "workflows" /
                "_pytest-suite.yml").read_text(encoding="utf-8")
    for path in (
        "tests/test_perf_guard.py",
        "tests/qt/test_gui_responsiveness.py",
        "tests/qt/test_home_stage_and_dock.py",
        "tests/qt/test_figure_queue.py",
        "tests/qt/test_pca.py",
    ):
        assert f"--ignore={path}" in workflow
        assert workflow.count(path) >= 2


def test_qt_suite_has_room_for_its_measured_runtime():
    """Qt files are split so neither job reaches the runner's hour boundary."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    qt_block = workflow.split("\n  qt:", 1)[1].split("\n  gpu:", 1)[0]

    # A single four-worker job repeatedly reached about 91% before the hosted
    # runner sent a shutdown signal at one hour. File-level sharding retains
    # the complete marker-selected suite while bounding each job well below
    # that service window.
    assert 'timeout_minutes: 120' in qt_block
    assert "shard: [0, 1]" in qt_block
    assert "file_shard_count: 2" in qt_block
    assert "run_serial_tail: ${{ matrix.shard == 0 }}" in qt_block


def test_file_shards_are_stable_disjoint_and_cover_every_test_module():
    files = sorted((ROOT / "tests").rglob("test_*.py"))
    first = {path for path in files if _ci_file_shard(path, 2) == 0}
    second = {path for path in files if _ci_file_shard(path, 2) == 1}

    assert first
    assert second
    assert not first & second
    assert first | second == set(files)
    assert all(
        _ci_file_shard(path, 2)
        == _ci_file_shard(path.relative_to(ROOT), 2)
        for path in files
    )


def test_reusable_suite_resets_sharding_for_the_serial_qt_tail():
    workflow = REUSABLE.read_text(encoding="utf-8")

    assert "SPACR_PYTEST_FILE_SHARD_INDEX" in workflow
    assert "SPACR_PYTEST_FILE_SHARD_COUNT" in workflow
    assert 'if [ "${{ inputs.run_serial_tail }}" = "true" ]' in workflow
    assert "SPACR_PYTEST_FILE_SHARD_COUNT=1 python -m pytest" in workflow


def test_informational_windows_sweep_cannot_cancel_the_matrix():
    """Expected Windows failures must finish before the job-level timeout."""
    workflow = (ROOT / ".github" / "workflows" /
                "compat-matrix.yml").read_text(encoding="utf-8")
    assert "Full suite (informational, never decides)" in workflow
    assert "--maxfail=25" in workflow
    assert "continue-on-error: true" in workflow


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
