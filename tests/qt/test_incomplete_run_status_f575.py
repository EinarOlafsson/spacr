"""Finalized item failures must survive GUI, CLI and workflow success handling."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import cli, cli_repro, run_journal
from spacr.cancellation import PipelineCancelled
from spacr.errors import PartialRunError, RunLedger
from spacr.qt.bridge import PipelineWorker


@pytest.fixture
def private_runs(monkeypatch, tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    monkeypatch.setattr(run_journal, "_notify_run_finished", lambda run: None)
    return root


def _ledger(failed=1):
    ledger = RunLedger("measure")
    ledger.record_success("good")
    for index in range(failed):
        ledger.record_failure(f"bad-{index}", exc="unreadable field")
    ledger.finalize()


@pytest.mark.parametrize("ending", ["return", "exit", "cancel"])
@pytest.mark.parametrize("failed", [0, 1, 2])
def test_gui_never_publishes_incomplete_result(
        qtbot, private_runs, tmp_path, ending, failed):
    artifact = tmp_path / "partial.csv"

    def pipeline(settings):
        artifact.write_text("measured,42\n")
        _ledger(failed)
        if ending == "exit":
            raise SystemExit(0)
        if ending == "cancel":
            raise PipelineCancelled("user stopped")
        return {"rows": 42}

    worker = PipelineWorker(pipeline, {}, app_key="measure")
    finished, errors, results = [], [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.result_ready.connect(results.append)
    worker.run()
    expected = "cancelled" if ending == "cancel" else (
        "failed" if failed else "success")
    manifest = json.loads(next(private_runs.glob("*/manifest.json")).read_text())
    assert manifest["status"] == expected
    assert finished == [expected == "success"]
    assert results == ([{"rows": 42}] if ending == "return" and not failed else [])
    assert bool(errors) == (expected == "failed")
    if errors:
        assert "PartialRunError" in errors[0]
    assert artifact.read_text() == "measured,42\n"


def test_later_clean_stage_and_explicit_success_cannot_hide_failure(private_runs):
    with pytest.raises(PartialRunError, match="1 of 2 items failed"):
        with run_journal.open_run("measure", {}) as run:
            _ledger()
            _ledger(0)
            run.set_status("success")
    assert json.loads((run.dir / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("failed", [0, 1])
def test_cli_exit_status_tracks_finalized_ledger(monkeypatch, private_runs, tmp_path, failed):
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    monkeypatch.setattr(cli, "resolve_settings", lambda *args: {})
    monkeypatch.setattr(cli, "import_entry", lambda module: lambda settings: _ledger(failed))
    code = cli.main(["measure", "--settings", str(settings), "--no-preflight"])
    assert code == (cli.EXIT_RUNTIME if failed else cli.EXIT_OK)
    manifest = json.loads(next(private_runs.glob("*/manifest.json")).read_text())
    assert manifest["status"] == ("failed" if failed else "success")


def test_real_snakemake_does_not_touch_done_after_partial_run(private_runs, tmp_path):
    engine = os.environ.get("SPACR_TEST_SNAKEMAKE")
    if not engine:
        pytest.skip("Set SPACR_TEST_SNAKEMAKE to an isolated Snakemake executable")
    with run_journal.open_run("measure", {"src": str(tmp_path / "plate")}) as run:
        pass
    # A deterministic failing item exercises the real CLI/journal exit path;
    # the actual generated rule and Snakemake executor remain unmodified.
    entry = tmp_path / "partial_entry.py"
    entry.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from spacr import cli, run_journal\n"
        "from spacr.errors import RunLedger\n"
        f"root = Path({str(tmp_path)!r})\n"
        "runs = root / 'child-runs'; runs.mkdir()\n"
        "run_journal.runs_root = lambda: runs\n"
        "run_journal._notify_run_finished = lambda run: None\n"
        "def pipeline(settings):\n"
        "    (root / 'partial.csv').write_text('measured,42\\n')\n"
        "    ledger = RunLedger('measure')\n"
        "    ledger.record_success('good')\n"
        "    ledger.record_failure('bad', exc='unreadable field')\n"
        "    ledger.finalize()\n"
        "cli.resolve_settings = lambda *args: {}\n"
        "cli.import_entry = lambda module: pipeline\n"
        "raise SystemExit(cli.main(sys.argv[1:] + ['--no-preflight']))\n"
    )
    workflow = tmp_path / "workflow"
    snakefile = cli_repro._export_workflow(
        run.dir, workflow, "snakemake",
        spacr_run=f"{shlex.quote(sys.executable)} {shlex.quote(str(entry))}",
    )
    result = subprocess.run(
        [engine, "--cores", "1", "--snakefile", str(snakefile),
         "--directory", str(workflow)],
        text=True, capture_output=True, timeout=90,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    (tmp_path / "snakemake.log").write_text(result.stdout + result.stderr)
    assert result.returncode != 0, result.stdout + result.stderr
    job_logs = "\n".join(p.read_text() for p in (workflow / "logs").glob("*.log"))
    assert "PartialRunError" in job_logs, job_logs
    assert not list((workflow / "done").glob("*.ok"))
    assert (tmp_path / "partial.csv").read_text() == "measured,42\n"
    manifest = json.loads(next((tmp_path / "child-runs").glob("*/manifest.json")).read_text())
    assert manifest["status"] == "failed"
