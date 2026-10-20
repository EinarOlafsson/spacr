"""A run whose process died shows as interrupted, never as running forever.

A Run that was SIGKILLed or crashed leaves a manifest with status "running"
and nothing to close it. The manifest now records its owner (pid, start
stamp, host and pid namespace, as the installer leases do); Run History
shows such a run as interrupted only when the owner is provably gone on this
host. A live owner, another host and a manifest with no owner stay running.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from spacr import run_journal
from spacr.install_cleanup import _process_started


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """An empty run journal under ``tmp_path``."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


def _manifest(root, name, owner=None, status="running"):
    """Write a run folder whose manifest says ``status``."""
    folder = root / f"2026-10-19_1200{len(list(root.iterdir())):02d}_0000abcd__{name}"
    folder.mkdir()
    (folder / "settings.json").write_text("{}", encoding="utf-8")
    manifest = {"schema_version": 3, "app_key": name, "status": status,
                "start_utc": datetime.now(timezone.utc).isoformat()}
    if owner is not None:
        manifest["owner"] = owner
    (folder / "manifest.json").write_text(json.dumps(manifest),
                                          encoding="utf-8")
    return folder


def _dead_owner():
    """The owner record of a process that has exited."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        record = dict(run_journal._run_owner())
        record["pid"] = str(child.pid)
        record["started"] = _process_started(child.pid)
    finally:
        child.kill()
        child.wait()
    assert record["started"]
    return record


def _statuses():
    """``{app_key: status}`` as Run History and the recent list show them."""
    searched = {r["app_key"]: r["status"] for r in run_journal.search_runs()}
    recent = {r["app_key"]: r["status"]
              for r in run_journal.recent_runs(limit=None)}
    assert searched == recent
    return searched


def test_a_new_run_records_its_owner(journal, monkeypatch):
    """The manifest written at open names this process on this host."""
    monkeypatch.setattr(run_journal, "_new_run_dir",
                        lambda app: journal / "2026-10-19_120000_0000abcd__x")
    (journal / "2026-10-19_120000_0000abcd__x").mkdir()
    with run_journal.open_run("x", {}) as run:
        manifest = json.loads((run.dir / "manifest.json").read_text())
        owner = manifest["owner"]
        assert owner["pid"] == str(os.getpid())
        assert owner["started"] == _process_started(os.getpid())
        assert manifest["status"] == "running"
        assert _statuses() == {"x": "running"}


def test_a_dead_owner_shows_interrupted(journal):
    """Only the run whose process is gone changes."""
    _manifest(journal, "dead", owner=_dead_owner())
    _manifest(journal, "alive", owner=run_journal._run_owner())
    _manifest(journal, "legacy")
    elsewhere = _dead_owner()
    elsewhere["host"] = elsewhere["host"] + "-other"
    _manifest(journal, "elsewhere", owner=elsewhere)
    _manifest(journal, "finished", owner=_dead_owner(), status="success")
    assert _statuses() == {"dead": "interrupted", "alive": "running",
                           "legacy": "running", "elsewhere": "running",
                           "finished": "success"}
    assert [r["app_key"] for r in run_journal.search_runs(
        status="interrupted")] == ["dead"]


def test_a_reused_pid_with_another_start_stamp_is_gone(journal):
    """Our own pid under an older start stamp is a different process."""
    owner = dict(run_journal._run_owner())
    owner["pid"] = str(os.getppid())
    kind = owner["started"].partition(":")[0]
    owner["started"] = f"{kind}:1"
    _manifest(journal, "reused", owner=owner)
    live = dict(run_journal._run_owner())
    live["pid"] = str(os.getppid())
    live["started"] = _process_started(os.getppid())
    _manifest(journal, "parent", owner=live)
    assert _statuses() == {"reused": "interrupted", "parent": "running"}


def test_an_unreadable_or_partial_owner_is_never_judged_dead():
    """No pid, no start stamp, or another pid namespace: still running."""
    base = _dead_owner()
    for broken in ({}, {"pid": "x"}, dict(base, started=""),
                   dict(base, pid="0"), dict(base, pidns="pid:[1]"),
                   "not a record"):
        if os.name == "nt" and isinstance(broken, dict) and "pidns" in broken:
            continue
        assert run_journal._run_owner_gone(broken) is False, broken
    assert run_journal._run_owner_gone(base) is True
