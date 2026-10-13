"""A failing batch must name the files it was running.

The batch runner splits the suite across fresh pytest processes, and a
batch that dies from a segfault or is killed by the job timeout prints no
pytest summary at all -- the log ends mid-run. A summary that says only
"batch 39 failed" then forces the reader to re-derive the sorted file list
and slice it by the batch size to learn what batch 39 even contained.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

from tools import run_pytest_batches as runner


def _files(tmp_path, names):
    for name in names:
        (tmp_path / f"test_{name}.py").write_text(
            "def test_x():\n    pass\n", encoding="utf-8",
        )


def test_a_failing_batch_lists_the_files_it_ran(tmp_path, monkeypatch, capsys):
    _files(tmp_path, ("a", "b", "c", "d"))
    calls = []

    def fake_run(command, check=False):
        calls.append(command)
        # Batch 2 holds test_c.py; a hard crash returns a negative status.
        return SimpleNamespace(returncode=-11 if len(calls) == 2 else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = runner.main([
        str(tmp_path), "--marker", "not slow", "--batch-size", "2",
    ])

    out = capsys.readouterr().out
    assert status == -11, "the crashing status must reach the caller"
    assert str(tmp_path / "test_c.py") in out
    assert str(tmp_path / "test_d.py") in out
    # The batches that passed are not worth naming.
    assert str(tmp_path / "test_a.py") not in out.split("batch 2")[-1]


def test_every_failing_batch_is_named_not_only_the_first(
    tmp_path, monkeypatch, capsys,
):
    _files(tmp_path, ("a", "b", "c"))
    calls = []

    def fake_run(command, check=False):
        calls.append(command)
        return SimpleNamespace(returncode=0 if len(calls) == 2 else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.main([
        str(tmp_path), "--marker", "not slow", "--batch-size", "1",
    ])

    out = capsys.readouterr().out
    assert "2 of 3 batches failed" in out
    for name in ("a", "c"):
        assert str(tmp_path / f"test_{name}.py") in out
