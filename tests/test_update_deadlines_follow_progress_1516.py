"""An update times out when it stops making progress, not when it has run long; abandoned stages go.

Reported from a slow Windows PC (1.5.1.5): a 48-minute preparation ended with "frozen update preparation
or shutdown timed out; nothing was changed". The preparation deadline was a fixed hour from the request,
and the Restart prompt had ten minutes in all, however long the user took to answer it. The helper now
records progress while it or its children use the CPU or move bytes, the desktop records it while its
Restart prompt is open, and each deadline is the original one or the last progress plus a stall allowance,
whichever is later, capped a day past the plan.

A helper killed mid-staging left its stage (.spacr-windows-update-*) behind. A stage whose owner is
provably gone, and that never touched the live runtime, is now pruned.
"""
import json
import os
import secrets
import subprocess
import time

import pytest

from spacr import install_cleanup as ic

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX ownership fixtures")


def _handshake(tmp_path, expires):
    workdir = tmp_path / "helper"
    workdir.mkdir(mode=0o700)
    plan = {"adapter": "pip-private-posix-v1", "version": "1.5.1.6", "pid": os.getpid(), "records": [],
            "ticked": [], "workdir": str(workdir),
            "handshake": {"schema": 1, "token": secrets.token_hex(32), "expires": expires}}
    return ic._FrozenUpdateHandshake(plan)


def test_preparation_past_its_deadline_lives_on_while_it_makes_progress(tmp_path):
    channel = _handshake(tmp_path, time.time() - 1)
    with pytest.raises(RuntimeError, match="timed out"):
        channel.check()
    channel.progress()
    channel.check()


def test_preparation_that_stops_making_progress_times_out(tmp_path, monkeypatch):
    channel = _handshake(tmp_path, time.time() - 1)
    channel.progress()
    later = time.time() + ic._PREPARATION_STALL_SECONDS + 5
    monkeypatch.setattr(ic.time, "time", lambda: later)
    with pytest.raises(RuntimeError, match="timed out"):
        channel.check()


def test_a_restart_prompt_left_open_stays_valid_while_the_desktop_reports_in(tmp_path):
    channel = _handshake(tmp_path, time.time() + 600)
    channel._write("readiness.json", "ready", expires=time.time() - 1)
    with pytest.raises(RuntimeError, match="timed out"):
        channel.check()
    channel.progress()
    channel.check()


def test_progress_never_extends_a_plan_past_its_cap(tmp_path, monkeypatch):
    channel = _handshake(tmp_path, time.time() - 1)
    beyond = channel.expires + ic._PROGRESS_CAP_SECONDS + 1
    monkeypatch.setattr(ic.time, "time", lambda: beyond)
    channel.progress()
    with pytest.raises(RuntimeError, match="timed out"):
        channel.check()


def test_a_cancelled_plan_stays_cancelled_whatever_its_progress(tmp_path):
    channel = _handshake(tmp_path, time.time() + 600)
    channel.progress()
    channel.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        channel.check()


class _Channel:
    def __init__(self):
        self.beats = 0

    def progress(self):
        self.beats += 1


def test_the_heartbeat_records_progress_only_while_work_moves():
    moving, still = _Channel(), _Channel()
    counter = iter(range(10_000))
    with ic._preparation_heartbeat(moving, interval=0.01, activity=lambda: next(counter)):
        time.sleep(0.2)
    with ic._preparation_heartbeat(still, interval=0.01, activity=lambda: 5.0):
        time.sleep(0.2)
    with ic._preparation_heartbeat(still, interval=0.01, activity=lambda: None):
        time.sleep(0.1)
    assert moving.beats > 0 and still.beats == 0


def test_the_helper_and_the_restart_prompt_both_report_progress():
    source = open(ic.__file__, encoding="utf-8").read()
    assert source.count("with _preparation_heartbeat(") == 2
    app = open(os.path.join(os.path.dirname(ic.__file__), "qt", "app.py"), encoding="utf-8").read()
    poll = app[app.index("def _poll_frozen_update"):app.index("def _confirm_update_restart")]
    assert "keep_alive.timeout.connect(handshake.progress)" in poll
    assert poll.index("keep_alive.start()") < poll.index("self._confirm_update_restart(message)")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "spaCR"
    prefix = root / "venv"
    (prefix / "bin").mkdir(parents=True)
    return root, str(prefix.resolve())


def _stage(root, prefix, name, state, *, family="windows-online-uv", owner=None, old_identity=None,
           retired=False):
    marker, schema, child = ic._RUNTIME_TRANSACTIONS[family]
    folder = root / (marker + name)
    (folder / "staged").mkdir(parents=True)
    if retired:
        (folder / child).mkdir()
    details = os.stat(prefix)
    data = {"schema": schema, "family": family, "prefix": prefix, "rollback_runtime": str(folder / child),
            "state": state, "old_identity": old_identity or [details.st_dev, details.st_ino]}
    if owner is not None:
        data["owner"] = owner
    (folder / "transaction.json").write_text(json.dumps(data))
    return folder


def _dead_owner():
    child = subprocess.Popen(["sleep", "0"])
    started = ic._process_started(child.pid)
    child.wait()
    return {"pid": child.pid, "started": started or "ticks:1"}


@pytest.mark.parametrize("state", ["staging", "ready"])
def test_a_stage_whose_owner_died_is_removed(runtime, state):
    root, prefix = runtime
    stage = _stage(root, prefix, "killed", state, owner=_dead_owner())
    assert [path for path, _ in ic._prune_runtime_transactions(prefix, "windows-online-uv")] == [str(stage)]


def test_a_stage_from_a_version_that_recorded_no_owner_is_removed_under_the_lease(runtime):
    root, prefix = runtime
    stage = _stage(root, prefix, "_zik1n0b", "staging", family="windows-private-pip")
    assert [path for path, _ in ic._prune_runtime_transactions(prefix, "windows-private-pip")] == [str(stage)]


def test_a_stage_whose_owner_lives_is_kept(runtime):
    root, prefix = runtime
    holder = subprocess.Popen(["sleep", "30"])
    try:
        stage = _stage(root, prefix, "live", "staging",
                       owner={"pid": holder.pid, "started": ic._process_started(holder.pid)})
        assert ic._prune_runtime_transactions(prefix, "windows-online-uv") == []
        assert stage.is_dir()
    finally:
        holder.kill()
        holder.wait()


def test_a_stage_that_touched_the_live_runtime_is_kept(runtime):
    root, prefix = runtime
    retired = _stage(root, prefix, "retired", "ready", owner=_dead_owner(), retired=True)
    swapped = _stage(root, prefix, "swapped", "staging", family="posix-runtime", owner=_dead_owner(),
                     old_identity=[1, 2])
    assert ic._prune_runtime_transactions(prefix, "windows-online-uv") == []
    assert ic._prune_runtime_transactions(prefix, "posix-runtime") == []
    assert retired.is_dir() and swapped.is_dir()


def test_new_transactions_record_their_owner():
    source = open(ic.__file__, encoding="utf-8").read()
    assert source.count('"owner": _transaction_owner()') == 2
    assert ic._transaction_owner()["pid"] == os.getpid()
