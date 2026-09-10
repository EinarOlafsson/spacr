"""A wedged test names itself instead of taking the job down in silence.

When a hosted job is killed by its own ``timeout-minutes`` there is no pytest
summary in the log: no last test, no stack, no counts. A single test that never
returns therefore reads exactly like a lost runner, and the whole job budget --
two hours for the Qt shards -- buys nothing that says which test to look at.

The reusable suite workflow passes pytest's faulthandler timeout so a test that
has been running longer than the threshold dumps every thread's traceback and
then carries on. That is the important half of the trade: a hang is named while
it is still hanging, and a test that is merely slow stays green rather than
being converted into a new failure.

Two things have to hold, and this file checks each against the thing itself
rather than against prose. The mechanism is driven for real, in a subprocess,
with the same option the workflow passes; and the workflow is read to confirm
every suite invocation carries it with a threshold that lands well before any
caller's job timeout.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SUITE = WORKFLOWS / "_pytest-suite.yml"
TESTS_WORKFLOW = WORKFLOWS / "tests.yml"

#: The option the suite workflow passes to pytest.
OPTION = "faulthandler_timeout"


def _suite_scripts() -> list[str]:
    document = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    return [
        step["run"]
        for job in document["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step.get("run"), str)
    ]


def _pytest_invocations() -> list[str]:
    """Every ``python -m pytest ...`` command the suite workflow runs."""
    joined = "\n".join(_suite_scripts())
    # Line continuations make one command span many lines; fold them first so a
    # command is one string and its options can be read off it.
    folded = joined.replace("\\\n", " ")
    return [
        line for line in folded.splitlines() if "python -m pytest" in line
    ]


def _thresholds() -> list[int]:
    return [
        int(value)
        for value in re.findall(rf"{OPTION}=(\d+)", "\n".join(_suite_scripts()))
    ]


def _caller_timeouts_in_seconds() -> list[int]:
    """Job timeouts of every caller of the reusable suite workflow."""
    document = yaml.safe_load(TESTS_WORKFLOW.read_text(encoding="utf-8"))
    return [
        int(job["with"]["timeout_minutes"]) * 60
        for job in document["jobs"].values()
        if str(job.get("uses", "")).endswith("_pytest-suite.yml")
        and "timeout_minutes" in job.get("with", {})
    ]


def test_a_test_that_overruns_prints_a_stack_naming_itself(tmp_path):
    """The dump has to identify the wedged test, not just report a timeout.

    A bare "Timeout!" line would leave the reader exactly where the silent job
    kill left them. The value is the frame naming the test function, which is
    what makes a hang actionable from the log alone.
    """
    test_file = tmp_path / "test_overruns.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import time


            def test_takes_longer_than_the_threshold():
                time.sleep(2)
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(test_file),
            "-o", f"{OPTION}=1",
            "-p", "no:cacheprovider", "-p", "no:randomly",
            "-q", "--no-header",
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )
    output = result.stdout + result.stderr

    assert "Timeout (0:00:01)!" in output, output[-2000:]
    assert "test_takes_longer_than_the_threshold" in output, output[-2000:]


def test_the_overrunning_test_still_passes(tmp_path):
    """A slow test must not become a failure just because it was reported.

    The threshold has to be safe to set on a suite whose slowest legitimate
    tests are minutes long. If the dump also failed the test, the only usable
    threshold would be one no real hang ever reaches.
    """
    test_file = tmp_path / "test_overruns.py"
    test_file.write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(2)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(test_file),
            "-o", f"{OPTION}=1",
            "-p", "no:cacheprovider", "-p", "no:randomly",
            "-q", "--no-header",
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_every_suite_invocation_asks_for_the_dump():
    """A shard without it is a shard that can still die saying nothing.

    The Qt suite runs a parallel pass and a serial measurement tail as two
    separate pytest commands, and either can be the one that wedges.
    """
    invocations = _pytest_invocations()
    assert invocations, "the suite workflow no longer runs pytest"
    missing = [command for command in invocations if OPTION not in command]
    assert not missing, (
        f"pytest invocations with no {OPTION}: {missing}"
    )


def test_the_threshold_lands_before_the_job_is_taken_away():
    """A dump that arrives after the runner is gone is not a dump.

    The threshold has to be shorter than the tightest job timeout any caller
    passes, or the wedged shard is killed before pytest reaches it.
    """
    thresholds = _thresholds()
    assert thresholds, f"no {OPTION} threshold is set"
    timeouts = _caller_timeouts_in_seconds()
    assert timeouts, "no caller of the reusable suite declares a job timeout"
    assert max(thresholds) < min(timeouts), (
        f"{OPTION} of {max(thresholds)}s is not reached before the tightest "
        f"job timeout of {min(timeouts)}s"
    )
