"""Contracts for the memory-bounded CI pytest runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_pytest_batches as runner


def test_test_files_are_recursive_sorted_and_deduplicated(tmp_path):
    first = tmp_path / "test_a.py"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "test_b.py"
    ignored = nested / "helper.py"
    for path in (first, second, ignored):
        path.write_text("", encoding="utf-8")

    assert runner._test_files([str(tmp_path), str(first)]) == sorted([
        str(first), str(second),
    ])


def test_main_recycles_workers_and_accepts_an_empty_marker_batch(
    tmp_path, monkeypatch,
):
    for index in range(3):
        (tmp_path / f"test_{index}.py").write_text("", encoding="utf-8")

    commands = []
    statuses = iter((0, runner.NO_TESTS_COLLECTED))

    def run(command, check):
        commands.append(command)
        assert check is False
        return SimpleNamespace(returncode=next(statuses))

    monkeypatch.setattr(runner.subprocess, "run", run)

    assert runner.main([
        str(tmp_path), "--marker", "not slow",
        "--batch-size", "2", "--workers", "2",
    ]) == 0
    assert [len(command[3:command.index("-m", 3)]) for command in commands] == [
        2, 1,
    ]
    assert all(command[-6:] == [
        "-n", "2", "--dist", "loadfile", "-v", "--tb=short",
    ] for command in commands)


def test_main_stops_at_the_first_real_failure(tmp_path, monkeypatch):
    (tmp_path / "test_one.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda _command, check: SimpleNamespace(returncode=2),
    )

    assert runner.main([str(tmp_path), "--marker", "not slow"]) == 2


@pytest.mark.parametrize("option,value", [
    ("--batch-size", "0"),
    ("--workers", "0"),
])
def test_main_rejects_non_positive_limits(tmp_path, option, value):
    (tmp_path / "test_one.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.main([str(tmp_path), "--marker", "not slow", option, value])


def test_every_batch_runs_even_after_one_fails(monkeypatch, tmp_path):
    """A job that stops at the first failing batch reports a PREFIX.

    The batches partition the suite, so returning early means the batches
    after the failure never execute. Measured on one commit: the run
    stopped at batch 19 of 54, thirty-five batches never ran, and the job
    reported "one failure" -- while a file in batch 39 had three real
    failures that had gone unreported for days because no run reached it.
    """
    import subprocess

    from tools import run_pytest_batches as runner

    for name in ("a", "b", "c", "d"):
        (tmp_path / f"test_{name}.py").write_text("def test_x():\n    pass\n")

    ran = []

    class _Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, check=False):
        batch = [c for c in command if c.endswith(".py")]
        ran.append(tuple(sorted(batch)))
        # The second batch fails; the rest must still be attempted.
        return _Result(1 if len(ran) == 2 else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = runner.main([str(tmp_path), "--batch-size", "1", "--marker", "not slow"])

    assert len(ran) == 4, f"only {len(ran)} of 4 batches ran"
    assert status == 1, "the failing status must still be what the job exits with"


def test_the_summary_names_every_failing_batch(monkeypatch, tmp_path, capsys):
    """One line a reader can act on, rather than a count to go hunting for."""
    import subprocess

    from tools import run_pytest_batches as runner

    for name in ("a", "b", "c"):
        (tmp_path / f"test_{name}.py").write_text("def test_x():\n    pass\n")

    seen = []

    class _Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, check=False):
        seen.append(1)
        return _Result(0 if len(seen) == 1 else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.main([str(tmp_path), "--batch-size", "1", "--marker", "not slow"])

    out = capsys.readouterr().out
    assert "2 of 3 batches failed" in out
    assert "batch 2" in out and "batch 3" in out
