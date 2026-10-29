"""Contracts for the mutually exclusive GitHub Actions pytest suites."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_the_real_nas_pipeline_is_owned_by_the_gpu_suite():
    """A mounted NAS must not make hosted CPU CI run Cellpose SAM."""
    path = ROOT / "tests" / "test_e2e_real_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "pytestmark"
                for target in node.targets)
    )
    assert isinstance(assignment.value, (ast.List, ast.Tuple))
    markers = {
        element.attr
        for element in assignment.value.elts
        if isinstance(element, ast.Attribute)
    }

    assert {"slow", "nas", "gpu"} <= markers


def test_the_nas_suite_keeps_a_bounded_non_gpu_availability_node():
    """The NAS lane must remain non-empty without owning GPU inference."""
    path = ROOT / "tests" / "test_nas_resource_availability.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "pytestmark"
                for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Attribute)

    assert assignment.value.attr == "nas"
    assert "pytest.mark.gpu" not in source
    assert "paths_available" in source
    assert "timeout=5.0" in source


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
    """Cursor, event-loop, frame-time, and RSS measurements need an idle process.

    The offscreen Qt platform exposes one synthetic cursor across workers,
    and event-loop/RSS budgets become measurements of sibling-worker load
    when these files run inside xdist.  The reusable suite must exclude each
    file from the parallel pass and run it explicitly in the serial tail.
    """
    workflow = (ROOT / ".github" / "workflows" /
                "_pytest-suite.yml").read_text(encoding="utf-8")
    for path in (
        "tests/test_perf_guard.py",
        "tests/qt/test_641_module_first_open_timing.py",
        "tests/qt/test_gui_responsiveness.py",
        "tests/qt/test_home_stage_and_dock.py",
        "tests/qt/test_figure_queue.py",
        "tests/qt/test_pca.py",
        "tests/qt/test_spaceout_fractals_move_and_stay_in_budget.py",
        # The backdrop's per-frame cost is asserted against the frame
        # budget while a Python worker runs. Beside three sibling xdist
        # workers that measures the runner, and it did: the assertion
        # passes alone and serially in its own batch, and fails under -n 4.
        "tests/qt/test_the_backdrop_survives_a_run.py",
    ):
        assert f"--ignore={path}" in workflow
        assert workflow.count(path) >= 2

    timing = "tests/qt/test_641_module_first_open_timing.py"
    run = workflow.split('if [ "${{ inputs.run_serial_tail }}" = "true" ]', 1)
    assert run[0].count(f"--ignore={timing}") == 2
    assert run[1].count(timing) == 1
    coverage = WORKFLOW.read_text(encoding="utf-8")
    assert f"--exclude-file {timing}" in coverage
    structural = "tests/qt/test_641_module_first_open_budgets.py"
    assert f"--ignore={structural}" not in workflow
    assert f"--exclude-file {structural}" not in coverage


def test_fractal_cpu_budget_runs_strictly_without_coverage_tracing():
    """Keep the strict CPU measurement in Qt's mandatory untraced tail."""
    import yaml

    filename = "tests/qt/test_spaceout_fractals_move_and_stay_in_budget.py"
    node = f"{filename}::test_no_setting_can_ask_for_more_than_half_a_frame"
    coverage = yaml.safe_load(WORKFLOW.read_text())["jobs"]["coverage-shards"]
    steps = [step["run"] for step in coverage["steps"] if "run" in step]
    assert sum(step.count(f"--deselect={node}") for step in steps) == 1
    assert not any(f"--exclude-file {filename}" in step for step in steps)



def _assert_console_timing_coverage_route(jobs, filename, node):
    commands = "\n".join(step.get("run", "") for step in jobs["coverage-shards"]["steps"])
    assert commands.count(f"--deselect={node}") == 1
    assert f"--exclude-file {filename}" not in commands


def _console_timing_markers(filename) -> set[str]:
    tree = ast.parse((ROOT / filename).read_text())
    marked = next(item for item in tree.body if isinstance(item, ast.Assign)
                  and ast.unparse(item.targets[0]) == "pytestmark")
    value = marked.value
    assert isinstance(value, (ast.List, ast.Tuple))
    assert all(isinstance(item, ast.Attribute) for item in value.elts)
    return {item.attr for item in value.elts if isinstance(item, ast.Attribute)}


def _console_timing_slow_marker(slow, filename):
    from _pytest.mark.expression import Expression

    assert not slow.get("continue-on-error", False)
    marker = slow["with"]["marker_expression"]
    assert marker == "(slow or heavy) and not gpu and not network and not nas and not gui"
    markers = _console_timing_markers(filename)
    assert markers == {"qt", "heavy"}
    assert Expression.compile(marker).evaluate(lambda name: name in markers)
    return marker


def _console_timing_slow_script(slow, node):
    import yaml

    reusable = yaml.safe_load(REUSABLE.read_text())
    inputs = reusable.get("on", reusable.get(True))["workflow_call"]["inputs"]
    script = _torch_routing_step()["run"]
    for name, definition in inputs.items():
        value = slow["with"].get(name, definition.get("default", ""))
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        script = script.replace("${{ inputs." + name + " }}", rendered)
    assert "${{" not in script
    assert node not in REUSABLE.read_text()
    return script


def _run_console_timing_slow_route(tmp_path, slow, node, batch_exit):
    script = _console_timing_slow_script(slow, node)
    _write_torch_routing_shim(tmp_path)
    log = tmp_path / "slow-routing.jsonl"
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
               PYTEST_ADDOPTS="", SPACR_QT_NATIVE_EVIDENCE_DIR="",
               ROUTING_LOG=str(log), ROUTING_NODE="tools/run_pytest_batches.py",
               ROUTING_FAIL_TAIL="1" if batch_exit else "0")
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == batch_exit, result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == 1
    return commands[0]


def _assert_console_timing_serial_batch(command, marker):
    argv = command["argv"]
    assert argv[:2] == ["tools/run_pytest_batches.py", "tests/"]
    assert argv[argv.index("--marker") + 1] == marker
    assert argv[argv.index("--batch-size") + 1] == "8"
    assert argv[argv.index("--workers") + 1] == "1"
    assert "--collect-selected-files" in argv and "--skip-qt-only-batches" not in argv


@pytest.mark.parametrize("batch_exit", [0, 7])
def test_console_timing_keeps_the_required_serial_untraced_slow_gate(tmp_path, batch_exit):
    import yaml

    filename = "tests/qt/test_perf_guard_qt.py"
    node = f"{filename}::test_appending_one_console_line_costs_a_fraction_of_a_millisecond"
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    _assert_console_timing_coverage_route(jobs, filename, node)
    slow = jobs["slow"]
    marker = _console_timing_slow_marker(slow, filename)
    command = _run_console_timing_slow_route(tmp_path, slow, node, batch_exit)
    _assert_console_timing_serial_batch(command, marker)
    argv = command["argv"]
    assert f"--ignore={filename}" not in argv
    assert f"--deselect={node}" not in command["addopts"]
    assert not any(arg.startswith("--cov") for arg in argv)
    assert "--cov" not in command["addopts"]


def test_fractal_cpu_budget_stays_in_the_mandatory_untraced_qt_tail():
    """Retain the strict measurement and its complete Qt routing contract."""
    import yaml

    filename = "tests/qt/test_spaceout_fractals_move_and_stay_in_budget.py"
    reusable = REUSABLE.read_text()
    tail = reusable.split('if [ "${{ inputs.run_serial_tail }}" = "true" ]; then', 1)[1].split("\n          fi", 1)[0]
    assert tail.count(filename) == 1
    assert "python -m pytest" in tail and "--cov" not in tail
    assert "SPACR_PYTEST_FILE_SHARD_INDEX=0" in tail
    assert "SPACR_PYTEST_FILE_SHARD_COUNT=1" in tail
    assert "--deselect" not in tail and "--ignore" not in tail
    qt = yaml.safe_load(WORKFLOW.read_text())["jobs"]["qt"]
    assert qt["with"]["run_serial_tail"] == "${{ matrix.shard == 2 }}"
    assert 2 in qt["strategy"]["matrix"]["shard"]
    assert qt["with"]["marker_expression"] == "qt and not slow and not heavy and not gpu and not network and not nas and not gui"


def _torch_routing_nodes(filename, node):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    original_nodes = {
        f"{filename}::{item.name}" for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")
    }
    assert len(original_nodes) == 9
    assert node in original_nodes
    return original_nodes


def _torch_routing_step():
    import yaml

    workflow = yaml.safe_load(REUSABLE.read_text(encoding="utf-8"))
    job = workflow["jobs"]["pytest"]
    collection_input = workflow.get("on", workflow.get(True))["workflow_call"]["inputs"]["collect_selected_files"]
    assert collection_input["type"] == "boolean" and collection_input["default"] is False
    assert not job.get("continue-on-error", False)
    step = next(
        step for step in job["steps"]
        if step.get("name", "").startswith("Run ${{ inputs.suite_name }}")
    )
    assert not step.get("continue-on-error", False)
    return step


def _write_torch_routing_shim(tmp_path):
    shim = tmp_path / "python"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "record = {'argv':sys.argv[1:], 'addopts':os.environ.get('PYTEST_ADDOPTS', ''),"
        "'index':os.environ.get('SPACR_PYTEST_FILE_SHARD_INDEX'),"
        "'count':os.environ.get('SPACR_PYTEST_FILE_SHARD_COUNT')}\n"
        "with open(os.environ['ROUTING_LOG'], 'a', encoding='utf-8') as stream:\n"
        " stream.write(json.dumps(record)+'\\n')\n"
        "sys.exit(7 if os.environ['ROUTING_FAIL_TAIL']=='1' and "
        "os.environ['ROUTING_NODE'] in sys.argv[1:] else 0)\n",
        encoding="utf-8",
    )
    shim.chmod(0o700)


def _torch_routing_script(step, batched, serial_tail):
    inputs = {
        "memory_telemetry_seconds": "0", "xdist_workers": "2",
        "per_test_timeout_seconds": "600",
        "batch_size": "32" if batched else "0",
        "marker_expression": "qt and not slow and not heavy and not gpu",
        "run_serial_tail": "true" if serial_tail else "false",
        "collect_selected_files": "false",
    }
    script = step["run"]
    for name, value in inputs.items():
        script = script.replace("${{ inputs." + name + " }}", value)
    assert "${{" not in script
    return script


def _run_torch_routing_shard(tmp_path, step, node, batched, fail_tail, shard_index):
    serial_tail = shard_index == 2
    script = _torch_routing_script(step, batched, serial_tail)
    log = tmp_path / f"routing-{batched}-{shard_index}-{fail_tail}.jsonl"
    original_addopts = "-p tools.pytest_plugins.qt_serial_rss_journal"
    env = dict(
        os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        PYTEST_ADDOPTS=original_addopts, SPACR_QT_NATIVE_EVIDENCE_DIR="",
        SPACR_PYTEST_FILE_SHARD_INDEX=str(shard_index),
        SPACR_PYTEST_FILE_SHARD_COUNT=str(QT_SHARD_COUNT),
        ROUTING_LOG=str(log), ROUTING_NODE=node,
        ROUTING_FAIL_TAIL="1" if fail_tail else "0",
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=10,
    )
    expected_exit = 7 if fail_tail and serial_tail else 0
    assert result.returncode == expected_exit, result.stderr
    commands = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(commands) == (2 if serial_tail else 1)
    return commands, original_addopts, serial_tail


def _assert_torch_parallel_routing(parallel, node, filename, batched):
    assert parallel["addopts"].split().count(f"--deselect={node}") == 1
    assert f"--ignore={filename}" not in parallel["argv"]
    assert "--cov" not in parallel["addopts"]
    expected_command = (["tools/run_pytest_batches.py", "tests/"]
                        if batched else ["-m", "pytest"])
    assert parallel["argv"][:2] == expected_command
    assert "tests/" in parallel["argv"]


def _assert_torch_serial_routing(tail, node, filename, original_addopts, allocated):
    assert tail["argv"].count(node) == 1
    assert filename not in tail["argv"]
    assert tail["addopts"] == original_addopts
    assert "-n" not in tail["argv"]
    assert not any(arg.startswith("--cov") for arg in tail["argv"])
    assert (tail["index"], tail["count"]) == ("0", "1")
    assert node not in allocated


def _assert_torch_routing_assignment(tmp_path, step, node, filename, original_nodes, batched, fail_tail):
    assigned: list[str] = []
    for shard_index in range(QT_SHARD_COUNT):
        commands, original_addopts, serial_tail = _run_torch_routing_shard(
            tmp_path, step, node, batched, fail_tail, shard_index)
        parallel = commands[0]
        _assert_torch_parallel_routing(parallel, node, filename, batched)
        allocated = original_nodes - {node}
        assert len(allocated) == 8
        if _ci_file_shard(ROOT / filename, QT_SHARD_COUNT) == shard_index:
            assigned.extend(allocated)
        if serial_tail:
            tail = commands[1]
            _assert_torch_serial_routing(tail, node, filename, original_addopts, allocated)
            assigned.append(node)
    assert set(assigned) == original_nodes
    assert all(assigned.count(item) == 1 for item in original_nodes)


@pytest.mark.posix_only
def test_torch_heartbeat_runs_once_serially_and_its_failure_blocks_ci(tmp_path):
    filename = (
        "tests/qt/test_torch_libraries_map_without_the_interpreter_lock_"
        "20261013.py"
    )
    timing = "test_importing_torch_after_the_map_blocks_other_threads_far_less"
    node = f"{filename}::{timing}"
    original_nodes = _torch_routing_nodes(filename, node)
    step = _torch_routing_step()
    _write_torch_routing_shim(tmp_path)
    for batched in (False, True):
        for fail_tail in (False, True):
            _assert_torch_routing_assignment(
                tmp_path, step, node, filename, original_nodes, batched, fail_tail)

    coverage = WORKFLOW.read_text(encoding="utf-8")
    assert coverage.count(f"--deselect={node}") == 1
    assert f"--exclude-file {filename}" not in coverage
    assert "run_serial_tail: ${{ matrix.shard == 2 }}" in coverage


def test_qt_suite_has_room_for_its_measured_runtime():
    """The budget must outlive shard 0, and still be able to name a hang."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    qt_block = workflow.split("\n  qt:", 1)[1].split("\n  gpu:", 1)[0]

    # 120 -> 180, AND THE HALF THAT NEEDED IT WAS MEASURED RATHER THAN
    # GUESSED. Dispatch 34639565357 is the first run that lived long enough
    # to show it: shard 1 finished and reported its failures while shard 0
    # was killed by the 120-minute budget. GitHub reports a budget kill as
    # "cancelled", which reads exactly like a superseded push, so a blocking
    # job had been failing to finish in silence -- a required check that
    # could never pass whatever the tests said.
    #
    # Shard 0 is the expensive half by construction, not by luck, and this
    # tree says by how much. Bucketing the Qt modules through
    # `_ci_file_shard` gives shard 0 522 parallel files carrying 6,995 test
    # functions against shard 1's 524 files and 6,699 -- near enough even --
    # and then `run_serial_tail`, true for shard 0 only, hands shard 0 a
    # further seven measurement files and 115 tests that run one at a time
    # with the four xdist workers gone. Two hours covered the parallel half
    # on both shards. It did not cover that serial tail.
    #
    # Six shards retain this budget; the matrix and single serial tail are
    # checked by test_the_qt_suite_is_six_shards_and_exactly_one_runs_the_serial_tail.
    assert "timeout_minutes: 180" in qt_block

    # A BIGGER BUDGET ON ITS OWN ONLY BUYS A HANG MORE ROOM TO HIDE IN. A job
    # killed by `timeout_minutes` prints no summary at all, so the extra
    # hour of silence would buy nothing but a later kill; the per-test
    # ceiling is what turns that into one named failure. It is asserted here,
    # beside the budget it makes safe, and asserted to be the smaller of the
    # two, because a ceiling at or above the job budget can never fire.
    budget = re.search(r"^\s*timeout_minutes:\s*(\d+)\s*$", qt_block, re.M)
    ceiling = re.search(
        r"^\s*per_test_timeout_seconds:\s*(\d+)\s*$", qt_block, re.M)
    assert budget is not None
    assert ceiling is not None
    assert 0 < int(ceiling.group(1)) < int(budget.group(1)) * 60


#: Qt shards in tests.yml. Two became three on 2026-09-15: on dispatched run
#: 35012948690 Qt shard 0 of 2 climbed to used=14,939 MiB of the runner's
#: 15,989 with 2,035 MiB swapped, and twice that day shard 0 lost its runner.
#: Six preserve that memory shape after be67 exhausted the 180-minute budget.
QT_SHARD_COUNT = 6


def _qt_job():
    import yaml

    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["qt"]


def test_the_qt_suite_is_six_shards_and_exactly_one_runs_the_serial_tail():
    """The matrix, the file split and the name agree, and the serial tail runs
    once. Two shards with the tail would run it twice; none would drop it."""
    job = _qt_job()
    given = job["with"]
    shards = job["strategy"]["matrix"]["shard"]

    assert given["file_shard_count"] == QT_SHARD_COUNT
    assert (given["batch_size"], given["xdist_workers"], given["timeout_minutes"],
            given["per_test_timeout_seconds"]) == (16, 2, 180, 1200)
    assert shards == list(range(QT_SHARD_COUNT))
    assert given["file_shard_index"] == "${{ matrix.shard }}"
    assert given["suite_name"] == (
        "Qt shard ${{ matrix.shard }} of " + str(QT_SHARD_COUNT)
    )

    tail = re.fullmatch(
        r"\$\{\{ matrix\.shard == (\d+) \}\}", str(given["run_serial_tail"]))
    assert tail, given["run_serial_tail"]
    assert [shard for shard in shards if shard == int(tail.group(1))] == [2]


def _file_shard_counts():
    return sorted({2, 3, QT_SHARD_COUNT, int(_qt_job()["with"]["file_shard_count"])})


def _files_in_shards(files, count):
    shards: dict[int, set[Path]] = {index: set() for index in range(count)}
    for path in files:
        shards[_ci_file_shard(path, count)].add(path)
    return list(shards.values())


def test_file_shards_are_disjoint_and_cover_every_test_module():
    files = sorted((ROOT / "tests").rglob("test_*.py"))
    qt_files = {path for path in files if "qt" in _automatic_ci_markers(path)}
    assert qt_files

    for count in _file_shard_counts():
        shards = _files_in_shards(files, count)
        # Every file in exactly one shard: none twice, none dropped.
        assert sum(len(shard) for shard in shards) == len(files)
        assert set().union(*shards) == set(files)
        # And every shard carries Qt files, so none of the matrix is idle.
        assert all(shard & qt_files for shard in shards)


def test_file_shards_are_stable_across_absolute_and_relative_paths():
    files = sorted((ROOT / "tests").rglob("test_*.py"))
    assert files
    for count in _file_shard_counts():
        assert all(
            _ci_file_shard(path, count)
            == _ci_file_shard(path.relative_to(ROOT), count)
            for path in files
        )


def test_reusable_suite_resets_sharding_for_the_serial_qt_tail():
    workflow = REUSABLE.read_text(encoding="utf-8")

    assert "SPACR_PYTEST_FILE_SHARD_INDEX" in workflow
    assert "SPACR_PYTEST_FILE_SHARD_COUNT" in workflow
    assert 'if [ "${{ inputs.run_serial_tail }}" = "true" ]' in workflow
    assert "SPACR_PYTEST_FILE_SHARD_COUNT=1 python -m pytest" in workflow


def test_informational_windows_sweep_cannot_cancel_the_matrix():
    """Slow advisory audits retain their own margin and skip ordinary pushes."""
    import yaml

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" /
                               "compat-matrix.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["install"]
    steps = {step.get("name"): step for step in job["steps"]}
    sweep = steps["Full suite (informational, never decides)"]
    assert sweep["continue-on-error"] is True
    assert sweep["timeout-minutes"] < job["timeout-minutes"]
    assert "--maxfail=25" in sweep["run"]
    assert "python -m pytest tests/" in sweep["run"]
    assert "python -m pytest tests/test_perf_guard.py" in sweep["run"]
    for name in ("Install and smoke the exact wheel and sdist",
                 "Packaging + smoke subset (decides this cell)"):
        step = steps[name]
        assert not step.get("if")
        assert not step.get("continue-on-error", False)


def test_informational_windows_sweep_runs_only_in_advisory_cells():
    """The slow full suite cannot run on ordinary pushes or Linux."""
    import yaml

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" /
                               "compat-matrix.yml").read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert "schedule" in triggers and "workflow_dispatch" in triggers
    sweep = next(step for step in workflow["jobs"]["install"]["steps"]
                 if step.get("name") == "Full suite (informational, never decides)")
    expression = sweep["if"].replace("runner.os", "platform").replace(
        "github.event_name", "event").replace("&&", " and ").replace(
        "||", " or ")
    for platform in ("Linux", "Windows", "macOS"):
        for event in ("push", "pull_request", "schedule", "workflow_dispatch"):
            allowed = eval(expression, {"__builtins__": {}},
                           {"platform": platform, "event": event})
            assert allowed == (platform != "Linux" and event in (
                "schedule", "workflow_dispatch"))


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
