"""A wedged test must name itself instead of consuming the whole job budget.

A Qt shard reached 99%, wedged inside a single test, and then printed nothing
at all for the remaining hour and a quarter until the runner enforced the job
timeout. The log named no test, carried no stack, and looked exactly like a
lost runner, so the whole 120-minute budget bought no information.

The suite already asks pytest for a faulthandler dump, and that is why the
silence was surprising. It is not enough under ``-n``: the dump is written by
the WORKER process, and xdist forwards a worker's output only when the worker
reports a result. A worker that never reports never delivers its dump, so the
one mechanism meant to make a hang loud is precisely the one a hang disables.

pytest-timeout's thread method does not need the wedged test to come back to
the interpreter. A watchdog thread dumps every stack and ends the worker, and
xdist then reports the crash together with the test id that worker was
running -- which is the fact the silent job never produced.

These are assertions about the CI definition rather than about a run, because
the behaviour under test only exists on a hosted runner. What can be held here
is that the ceiling is wired to both pytest invocations, that it is ordered
against the other two limits so the sequence is dump-then-die-then-report, and
that the suite which actually wedged asks for it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
REUSABLE = ROOT / ".github" / "workflows" / "_pytest-suite.yml"
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

FAULTHANDLER_SECONDS = 900


def _workflow_call_inputs():
    """The reusable workflow's declared inputs.

    ``on`` is a YAML 1.1 boolean, so a safe load files the whole trigger block
    under ``True`` rather than under the string GitHub actually wrote.
    """
    reusable = yaml.safe_load(REUSABLE.read_text(encoding="utf-8"))
    trigger = reusable.get("on", reusable.get(True))
    return trigger["workflow_call"]["inputs"]


def _qt_inputs():
    """The ``with:`` block the Qt shards hand to the reusable suite."""
    workflow = yaml.safe_load(TESTS_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["qt"]["with"]


def test_the_reusable_suite_accepts_a_per_test_ceiling():
    """The ceiling is an input, so a suite that needs none is unaffected."""
    inputs = _workflow_call_inputs()

    assert "per_test_timeout_seconds" in inputs
    declared = inputs["per_test_timeout_seconds"]
    assert declared["type"] == "number"
    # Off by default: only a suite that asks pays the install and the risk.
    assert declared["default"] == 0


def test_the_ceiling_reaches_both_pytest_invocations():
    """The serial tail can wedge exactly like the parallel pass."""
    text = REUSABLE.read_text(encoding="utf-8")

    assert '--timeout "${{ inputs.per_test_timeout_seconds }}"' in text
    # The signal method needs the test to return to the interpreter, which is
    # what a wedged test does not do; only the thread method can end it.
    assert "--timeout-method thread" in text
    # Once for the sharded parallel pass, once for the serial measurement tail.
    assert text.count('"${timeout_args[@]}"') == 2
    # Without this the cure costs more than the disease: xdist replaces the
    # ended worker and gives it the same test, so a deterministic hang would
    # be paid once per restart rather than once.
    assert "--max-worker-restart=0" in text
    # pytest-timeout is not a project dependency; a suite with no ceiling must
    # not install it, so the install sits inside the guard rather than beside
    # it.
    assert (
        'if [ "${{ inputs.per_test_timeout_seconds }}" -gt 0 ]; then\n'
        '            python -m pip install "pytest-timeout'
    ) in text


def test_the_qt_shards_ask_for_a_ceiling_and_it_is_ordered_against_the_others():
    """Dump, then die, then report -- the three limits must be in that order."""
    qt = _qt_inputs()
    ceiling = qt["per_test_timeout_seconds"]

    assert ceiling > 0, "the suite that wedged is the one that must be bounded"
    # Above the faulthandler threshold, so the in-process dump is attempted
    # first and is there to read on a serial suite.
    assert ceiling > FAULTHANDLER_SECONDS
    # Comfortably below the job timeout, so the shard survives its own
    # wedged test and still prints a summary of everything else.
    assert ceiling * 2 < qt["timeout_minutes"] * 60


def test_only_a_parallel_suite_is_left_relying_on_the_faulthandler_dump():
    """A suite running under xdist cannot deliver a worker's dump.

    The ceiling is what replaces it, so any suite that hands the reusable
    workflow a non-zero ``xdist_workers`` must also hand it a ceiling.
    """
    workflow = yaml.safe_load(TESTS_WORKFLOW.read_text(encoding="utf-8"))
    unbounded = []
    for name, job in workflow["jobs"].items():
        given = job.get("with") or {}
        if not given.get("xdist_workers"):
            continue
        if not given.get("per_test_timeout_seconds"):
            unbounded.append(name)

    assert not unbounded, (
        "these suites run under xdist, where a wedged worker's faulthandler "
        f"dump never reaches the log, and have no ceiling: {unbounded}"
    )
