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
import re
import shlex
import shutil
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


#: Rows shaped like ``ps -ww -eo pid=,rss=,comm=,args=`` (RSS in KiB). The
#: worker's interpreter path is the GitHub runner's, 54 characters long.
_CONTROLLER = "100 297000 python python -m pytest tests/ -n 4 --dist loadfile"
_WORKER = (
    "4242 4085760 python /opt/hostedtoolcache/Python/3.12.14/x64/bin/python"
    " -c import sys;exec(eval(sys.stdin.readline()))"
)
_NOT_PYTEST = "777 51200 python python -m http.server 8000"
_NOT_PYTHON = "888 4096 bash bash -c echo pytest"


def _process_table(tmp_path, *snapshots):
    """An executable standing in for ``ps``: call N prints snapshot N, and
    every call after the last prints the last one again."""
    for number, rows in enumerate(snapshots, start=1):
        (tmp_path / f"snapshot_{number}.txt").write_text(
            "".join(f"{row}\n" for row in rows), encoding="utf-8",
        )
    calls = shlex.quote(str(tmp_path / "calls"))
    last = len(snapshots)
    script = tmp_path / "process_table"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"count=$(( $(cat {calls} 2>/dev/null || echo 0) + 1 ))\n"
        f'echo "$count" > {calls}\n'
        f'[ "$count" -le {last} ] || count={last}\n'
        f'cat {shlex.quote(str(tmp_path))}/snapshot_"$count".txt\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _rss_list(line):
    listed = re.search(r"pytest_rss=\[([^\]]*)\]MiB", line).group(1)
    return sorted(int(value) for value in listed.split(",") if value)


@pytest.mark.skipif(
    not Path("/proc/meminfo").exists(), reason="the sampler reads /proc",
)
def test_the_sampler_names_a_pytest_process_that_ended_and_its_last_rss(tmp_path):
    """The memory guard in tests/conftest.py ends a pytest process with exit 3
    at SPACR_TEST_MEMORY_GB; the sampler cannot see an exit status, so it
    logs each pytest process that disappears with the RSS it last had, next
    to the guard's ceiling.

    THE PROCESS LIST AND THE CLOCK ARE INJECTED. This test used to start a
    real 2.2-second python and sample the real ``ps`` every second for 4.5
    seconds. It failed on CI in "Fast / Full suite control" and coverage shard
    11 of run 35012948690 with pytest_rss=[] in every sample: under xdist,
    COLUMNS=80 made ps cut the runner's long interpreter path before the word
    the filter looks for, so the worker was never seen and could not be seen
    to end. It fails the same way locally under ``-n 2`` and passes serially.
    Here the table is two fixed snapshots and the sampler stops itself after
    two samples, so nothing depends on how long a real process lives, when
    a sample lands, or how wide ps thinks the terminal is.
    """
    table = _process_table(
        tmp_path,
        [_CONTROLLER, _WORKER, _NOT_PYTEST, _NOT_PYTHON],
        [_CONTROLLER],
    )
    environment = {
        **os.environ,
        "SPACR_TEST_MEMORY_GB": "6",
        "SPACR_TELEMETRY_PROCESS_TABLE": str(table),
        "SPACR_TELEMETRY_MAX_SAMPLES": "2",
    }
    result = subprocess.run(
        ["bash", str(SAMPLER), "0", str(tmp_path / "telemetry" / "samples.log")],
        capture_output=True, text=True, env=environment, timeout=60,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert (tmp_path / "calls").read_text(encoding="utf-8").strip() == "2"
    lines = result.stdout.splitlines()
    samples = [index for index, line in enumerate(lines) if " used=" in line]
    ended = [index for index, line in enumerate(lines) if " ended; " in line]
    assert len(samples) == 2, output

    # Both pytest processes are listed; the http server and the shell whose
    # arguments merely mention pytest are not.
    assert _rss_list(lines[samples[0]]) == [290, 3990]
    assert _rss_list(lines[samples[1]]) == [290]
    assert "guard=6GB" in lines[samples[0]]

    # Exactly one process ended, named with its last RSS, reported by the
    # sample that no longer saw it; the ones that were never pytest are not.
    assert len(ended) == 1, output
    assert samples[0] < ended[0] < samples[1]
    assert re.fullmatch(
        r"MEMORY \d\d:\d\d:\d\d pytest process 4242 ended; last seen at "
        r"3990MiB RSS \(tests/conftest\.py's memory guard ends a pytest "
        r"process at 6 GB with exit 3 and says so on stderr\)",
        lines[ended[0]],
    ), lines[ended[0]]


@pytest.mark.skipif(
    shutil.which("ps") is None or not Path("/proc").is_dir(),
    reason="the sampler lists processes with procps",
)
def test_a_command_line_wider_than_the_terminal_is_still_read_in_full():
    """The cause of the CI failure above, against the real ``ps``.

    A live process keeps "pytest" 200 characters into its command line, and
    the sampler runs with COLUMNS=80, as it does under xdist. Popen returns
    only after the child has exec'd, and the child outlives the listing, so
    this does not race it.
    """
    padding = "x" * 200
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", padding, "pytest"],
        stdin=subprocess.DEVNULL,
    )
    try:
        listing = subprocess.run(
            ["bash", str(SAMPLER), "--list-pytest-processes"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "COLUMNS": "80"},
        )
    finally:
        child.kill()
        child.wait(timeout=10)

    assert listing.returncode == 0, listing.stderr
    listed = {line.split()[0] for line in listing.stdout.splitlines()}
    assert str(child.pid) in listed, listing.stdout
