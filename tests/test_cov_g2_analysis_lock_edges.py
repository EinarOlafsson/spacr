"""The analysis lock's tolerance of unreadable records and odd inputs."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from spacr import run_journal as rj


def test_who_falls_back_to_the_environment(monkeypatch):
    import getpass

    def broken():
        raise OSError("no passwd entry")

    monkeypatch.setattr(getpass, "getuser", broken)
    monkeypatch.setenv("USER", "lab-user")
    assert rj._who().startswith("lab-user")


def test_empty_scopes_never_overlap():
    assert rj._paths_overlap("", "/data") is False
    assert rj._paths_overlap("/data", "/data") is True


def test_blinding_logs_skip_unreadable_lines():
    assert rj._blinding_events("never-started") == []
    log = rj._blinding_root() / "k1.log.jsonl"
    log.write_text('{"event": "blinded"}\nnot json\n[1]\n')
    assert rj._blinding_events("k1") == [{"event": "blinded"}]
    rj._close_blinding("no-such-key")
    assert rj._unblinding_times("") == []


def test_a_path_that_cannot_be_inspected_is_not_a_file(monkeypatch):
    def broken(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "is_file", broken)
    assert rj._existing_file("/data/gates.json") is None


def test_gate_payloads_reject_what_is_not_a_gate_set(tmp_path, monkeypatch):
    class _Broken:
        def to_dict(self):
            raise ValueError("half built")

    assert rj._gate_payload(_Broken()) is None
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"gates": []}))
    monkeypatch.setattr(rj, "_GATE_FILE_MAX_BYTES", 1)
    assert rj._gate_payload(big) is None
    monkeypatch.setattr(rj, "_GATE_FILE_MAX_BYTES", 1 << 20)
    bad = tmp_path / "bad.json"
    bad.write_text("{broken")
    assert rj._gate_payload(bad) is None
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]")
    assert rj._gate_payload(listed) is None


def test_several_unnamed_gate_sets_are_numbered():
    gates = [{"gates": []}, {"gates": []}]
    assert [label for label, _ in rj._gate_items(gates)] == ["gates[0]",
                                                             "gates[1]"]


def test_an_added_gate_is_named():
    assert rj._gate_changes({"gates": {"a": "1"}},
                            {"gates": {"a": "1", "b": "2"}}) == ["added b"]


def test_unreadable_lock_files_are_skipped(tmp_path):
    root = rj._locks_root()
    (root / "broken.json").write_text("{not json")
    (root / "list.json").write_text("[1]")
    assert rj._find_lock("measure", str(tmp_path)) is None
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"gates": []}))
    (root / "other.json").write_text(json.dumps({"gates": {}}))
    assert rj._gate_file_lock_notes(gates) == []
    assert rj._gate_file_lock_notes(tmp_path / "none.json") == []


def test_a_lock_without_pipeline_entries_compares_its_own_settings():
    record = {"settings": {"threshold": 1}}
    changed = rj._lock_deviations(record, {"threshold": 2}, app_key="measure")
    assert changed


def test_a_failed_lock_check_is_a_warning_not_a_failure(monkeypatch):
    def broken(app_key, src):
        raise RuntimeError("lock store unreadable")

    monkeypatch.setattr(rj, "_find_lock", broken)
    run = types.SimpleNamespace(app_key="measure", settings={"src": "/d"},
                                provenance_warnings=[])
    rj._check_lock_for_run(run)
    assert run.provenance_warnings == [
        "analysis lock check failed: lock store unreadable"]


def test_recorded_models_skip_unreadable_runs(monkeypatch, tmp_path):
    root = rj.runs_root()
    for name, manifest in (("r1", "{bad"), ("r2", json.dumps({"app_key": "mask"}))):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(manifest)
    assert rj._recorded_models("mask", str(tmp_path)) == {}
