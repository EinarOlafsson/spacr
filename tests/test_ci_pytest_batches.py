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
