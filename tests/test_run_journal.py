"""Tests for spacr.run_journal — reproducibility record."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_runs(monkeypatch, tmp_path):
    """Point runs_root at tmp so tests don't touch ~/.spacr/runs."""
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield


def test_open_run_creates_folder_with_settings(tmp_path):
    from spacr.run_journal import open_run
    with open_run("mask", {"src": "/tmp/x", "n_epochs": 3}) as run:
        pass
    assert run.dir.exists()
    assert (run.dir / "settings.json").exists()
    assert (run.dir / "settings.csv").exists()
    assert (run.dir / "manifest.json").exists()
    settings = json.loads((run.dir / "settings.json").read_text())
    assert settings["src"] == "/tmp/x"
    assert settings["n_epochs"] == 3


def test_manifest_records_env_and_success(tmp_path):
    from spacr.run_journal import open_run
    with open_run("measure", {"foo": 1}) as run:
        pass
    m = json.loads((run.dir / "manifest.json").read_text())
    assert m["app_key"] == "measure"
    assert m["status"] == "success"
    assert m["elapsed_s"] is not None
    assert "spacr" in m["env"]
    assert "python" in m["env"]


def test_failed_run_records_traceback(tmp_path):
    from spacr.run_journal import open_run
    with pytest.raises(RuntimeError):
        with open_run("mask", {}) as run:
            raise RuntimeError("boom")
    m = json.loads((run.dir / "manifest.json").read_text())
    assert m["status"] == "failed"
    assert "RuntimeError" in (m.get("traceback") or "")


def test_record_model_hashes_file(tmp_path):
    from spacr.run_journal import open_run, hash_file
    ckpt = tmp_path / "cyto.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    with open_run("mask", {}) as run:
        run.record_model("cyto", ckpt)
    m = json.loads((run.dir / "manifest.json").read_text())
    assert "cyto" in m["model_hashes"]
    assert m["model_hashes"]["cyto"].startswith("cyto.pt:")
    # Fingerprint is deterministic
    assert hash_file(ckpt) in m["model_hashes"]["cyto"]


def test_attach_output_copies_into_outputs_dir(tmp_path):
    from spacr.run_journal import open_run
    src = tmp_path / "some_result.csv"
    src.write_text("a,b,c\n1,2,3\n")
    with open_run("measure", {}) as run:
        dst = run.attach_output(src)
    assert dst is not None
    assert (run.dir / "outputs" / "some_result.csv").exists()


def test_recent_runs_lists_newest_first(tmp_path):
    from spacr.run_journal import open_run, recent_runs
    for i in range(3):
        with open_run("mask", {"i": i}) as _:
            time.sleep(0.02)   # ensure timestamps differ
    listing = recent_runs(limit=5)
    assert len(listing) == 3
    starts = [r["start_utc"] for r in listing]
    assert starts == sorted(starts, reverse=True)
    assert all(r["app_key"] == "mask" for r in listing)


def test_recent_runs_skips_corrupt_folders(tmp_path):
    from spacr.run_journal import open_run, recent_runs
    with open_run("mask", {"i": 0}):
        pass
    # Drop a broken folder — no manifest.json — should be ignored
    (tmp_path / "2026-01-01_000000_deadbeef__bad").mkdir()
    listing = recent_runs()
    assert len(listing) == 1


def test_load_run_settings_roundtrip(tmp_path):
    from spacr.run_journal import open_run, load_run_settings
    src_settings = {"src": "/foo", "n": 42, "flag": True}
    with open_run("mask", src_settings) as run:
        pass
    loaded = load_run_settings(run.dir)
    assert loaded["src"] == "/foo"
    assert loaded["n"] == 42
    assert loaded["flag"] is True
