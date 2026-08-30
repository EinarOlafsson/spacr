"""A normal run persists and registers the process-tree account it samples."""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from spacr import artifacts, runctx


def _sampler_threads():
    return {
        thread.ident for thread in threading.enumerate()
        if thread.name.startswith("spacr-resource-sampler-")
    }


def test_a_run_persists_summarises_and_registers_its_resource_record(
        tmp_path, monkeypatch):
    log_root = tmp_path / "logs"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("SPACR_LOG_DIR", str(log_root))
    profile_before = sys.getprofile()

    child = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(60)"
    ])
    try:
        with runctx.run_context(
                "mask",
                {"src": str(project), "performance_logging": "detailed"},
                run_id="resource-pass") as run:
            identity = run.register_worker(
                "test-worker", "child", pid=child.pid)
            assert identity.startswith(f"{child.pid}:")
            assert run.resource_log_path.endswith(
                "resource-pass.resources.json")
    finally:
        child.terminate()
        child.wait(timeout=30)

    document = runctx.read_run_resources("resource-pass")
    assert document["mode"] == "detailed"
    assert document["stop_reason"] == "completed"
    assert document["configuration"]["profile_hook_installed"] is False
    assert document["summary"]["samples_recorded"] >= 1
    assert document["summary"]["peak_tree_memory_bytes"] > 0
    assert any(
        process.get("worker") == {"kind": "test-worker", "id": "child"}
        for sample in document["samples"]
        for process in sample["processes"]
    )
    assert sys.getprofile() is profile_before

    records = artifacts.by_kind("resource-log", project=project)
    assert len(records) == 1
    assert records[0].run_id == "resource-pass"
    assert records[0].path == run.resource_log_path
    assert records[0].status == artifacts.STATUS_COMPLETE
    assert run.resource_artifact_id == records[0].artifact_id

    messages = [row["message"] for row in runctx.read_run_log(
        "resource-pass")]
    assert any("resources — peak tree=" in message for message in messages)


def test_off_mode_leaves_no_sampler_thread_or_resource_file(
        tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    before = _sampler_threads()

    with runctx.run_context(
            "mask", {"performance_logging": "off"},
            run_id="resource-off") as run:
        assert run.resource_log_path == ""
        assert _sampler_threads() == before

    assert _sampler_threads() == before
    assert runctx.read_run_resources("resource-off") == {}
    assert not os.path.exists(runctx.run_resource_path("resource-off"))


def test_a_failed_run_still_leaves_a_valid_failed_resource_artifact(
        tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))

    with pytest.raises(RuntimeError, match="deliberate failure"):
        with runctx.run_context(
                "mask",
                {"src": str(project), "performance_logging": "summary"},
                run_id="resource-failed"):
            raise RuntimeError("deliberate failure")

    document = runctx.read_run_resources("resource-failed")
    assert document["stop_reason"] == "failed"
    assert document["summary"]["samples_recorded"] >= 1
    records = artifacts.by_kind("resource-log", project=project)
    assert len(records) == 1
    assert records[0].run_id == "resource-failed"
    assert records[0].status == artifacts.STATUS_FAILED
