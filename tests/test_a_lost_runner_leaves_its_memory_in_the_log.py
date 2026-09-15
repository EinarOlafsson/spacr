"""A Qt shard that loses its runner leaves its memory in the live log.

Items 43 and 288, 2026-09-15. Qt shard 0 lost its GitHub runner twice in one
day: "The runner has received a shutdown signal" at [83%] after 71 minutes
(run 34961482728) and at [96%] after 89 (run 34989909231), inside a
180-minute budget, so neither was a timeout. A lost runner runs no later step
and uploads nothing, and a machine that runs out of memory cannot log that it
did. The hypothesis to TEST is that the runner ran out of memory; the
reusable suite therefore samples memory into the step's live log, which
GitHub keeps up to the moment the runner vanished, and uploads the fuller
samples when the run survives.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
REUSABLE = ROOT / ".github" / "workflows" / "_pytest-suite.yml"
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
SAMPLER = ROOT / "tools" / "ci_memory_telemetry.sh"


def _reusable():
    return yaml.safe_load(REUSABLE.read_text(encoding="utf-8"))


def _inputs():
    document = _reusable()
    # ``on`` is a YAML 1.1 boolean, so a safe load files it under True.
    trigger = document.get("on", document.get(True))
    return trigger["workflow_call"]["inputs"]


def _steps():
    return _reusable()["jobs"]["pytest"]["steps"]


def _test_step():
    return next(
        step for step in _steps() if "python -m pytest" in step.get("run", "")
    )


def test_telemetry_is_off_unless_a_suite_asks_for_it():
    declared = _inputs()["memory_telemetry_seconds"]

    assert declared["type"] == "number"
    assert declared["default"] == 0


def test_the_qt_shards_ask_for_it_and_no_other_suite_pays_for_it():
    jobs = yaml.safe_load(TESTS_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    interval = jobs["qt"]["with"]["memory_telemetry_seconds"]
    others = [
        name for name, job in jobs.items()
        if name != "qt" and (job.get("with") or {}).get("memory_telemetry_seconds")
    ]

    # Often enough that the last line before a lost runner is recent.
    assert 0 < interval <= 60
    assert others == []


def test_the_sampler_starts_before_pytest_and_stops_with_the_script():
    script = _test_step()["run"]

    assert 'if [ "${{ inputs.memory_telemetry_seconds }}" -gt 0 ]' in script
    start = script.index("bash tools/ci_memory_telemetry.sh")
    assert start < script.index("python -m pytest")
    # Backgrounded, and killed on EXIT, which also runs when `bash -e` stops
    # the script at a failing pytest.
    assert "telemetry_pid=$!" in script
    assert "trap 'kill \"$telemetry_pid\" 2>/dev/null || true' EXIT" in script


def test_a_surviving_run_uploads_the_samples_and_a_failed_one_prints_them():
    steps = _steps()
    uploads = [
        step for step in steps
        if step.get("uses") == "actions/upload-artifact@v7"
        and "spacr-memory-telemetry" in step["with"]["path"]
    ]
    printed = [
        step for step in steps
        if "python -m pytest" not in step.get("run", "")
        and "spacr-memory-telemetry/samples.log" in step.get("run", "")
    ]

    assert len(uploads) == 1
    assert uploads[0]["if"] == "always() && inputs.memory_telemetry_seconds > 0"
    # Unique per Qt shard, or the second shard's upload would collide.
    assert "${{ inputs.suite_name }}" in uploads[0]["with"]["name"]
    assert len(printed) == 1
    assert printed[0]["if"] == "failure() && inputs.memory_telemetry_seconds > 0"


@pytest.mark.skipif(
    not Path("/proc/meminfo").exists(), reason="the sampler reads /proc",
)
def test_the_sampler_prints_one_memory_line_per_interval_and_stops_on_term(
    tmp_path,
):
    samples = tmp_path / "telemetry" / "samples.log"
    process = subprocess.Popen(
        ["bash", str(SAMPLER), "1", str(samples)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        time.sleep(2.5)
    finally:
        process.terminate()
        output, _ = process.communicate(timeout=10)

    lines = [line for line in output.splitlines() if line.startswith("MEMORY ")]
    assert len(lines) >= 2, output
    for field in (
        "used=", "available=", "swap_used=", "psi_full_avg10=", "oom_kills=",
        "disk_free=", "pytest_rss=[", "guard=", "top: ",
    ):
        assert field in lines[0], lines[0]
    assert samples.read_text(encoding="utf-8").count("=== ") >= 2
    # SIGTERM ends it cleanly (exit 0), not by the default signal death.
    assert process.returncode == 0


@pytest.mark.skipif(
    not Path("/proc/meminfo").exists(), reason="the sampler reads /proc",
)
def test_the_sampler_names_a_pytest_process_that_ended_and_its_last_rss(tmp_path):
    """The memory guard in tests/conftest.py ends a pytest process with exit 3
    at SPACR_TEST_MEMORY_GB; the sampler cannot see an exit status, so it
    logs each pytest process that disappears with the RSS it last had, next
    to the guard's ceiling."""
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time  # a pytest worker\ntime.sleep(2.2)"],
    )
    environment = {**os.environ, "SPACR_TEST_MEMORY_GB": "6"}
    sampler = subprocess.Popen(
        ["bash", str(SAMPLER), "1", str(tmp_path / "samples.log")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=environment,
    )
    try:
        time.sleep(4.5)
    finally:
        sampler.terminate()
        output, _ = sampler.communicate(timeout=10)
        worker.wait(timeout=10)

    assert f"pytest process {worker.pid} ended; last seen at " in output, output
    assert "guard=6GB" in output
