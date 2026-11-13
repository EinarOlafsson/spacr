"""Regression tests: Cellpose 4 string checkpoints and naive start times."""
import json
import types

from spacr import pipeline_v2 as PV
from spacr import run_journal as RJ


def _root(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(RJ, "runs_root", lambda: root)
    return root


def test_string_pretrained_model_is_recorded(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    ckpt = tmp_path / "cpsam_test"
    ckpt.write_bytes(b"weights" * 10)
    with RJ.open_run("mask", {}) as run:
        PV._record_cellpose_hash(
            types.SimpleNamespace(pretrained_model=str(ckpt)), "cyto")
    assert list(run.model_hashes) == ["cyto"]
    assert run.model_hashes["cyto"].startswith("cpsam_test:")


def test_string_matches_list_form(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    ckpt = tmp_path / "w.bin"
    ckpt.write_bytes(b"abc")
    with RJ.open_run("mask", {}) as a:
        PV._record_cellpose_hash(
            types.SimpleNamespace(pretrained_model=[str(ckpt)]), "m")
    with RJ.open_run("mask", {}) as b:
        PV._record_cellpose_hash(
            types.SimpleNamespace(pretrained_model=str(ckpt)), "m")
    assert a.model_hashes == b.model_hashes != {}


def test_nested_string_and_tuple_forms(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    ckpt = tmp_path / "w.bin"
    ckpt.write_bytes(b"abc")
    cp = types.SimpleNamespace(pretrained_model=str(ckpt))
    with RJ.open_run("mask", {}) as a:
        PV._record_cellpose_hash(types.SimpleNamespace(cp=cp), "m")
    with RJ.open_run("mask", {}) as b:
        PV._record_cellpose_hash(
            types.SimpleNamespace(pretrained_model=(str(ckpt),)), "m")
    assert a.model_hashes == b.model_hashes != {}


def test_missing_string_checkpoint_records_nothing(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    with RJ.open_run("mask", {}) as run:
        PV._record_cellpose_hash(
            types.SimpleNamespace(pretrained_model=str(tmp_path / "no")), "m")
    assert run.model_hashes == {}


def _make(root, name, start):
    d = root / name
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"run_id": name, "app_key": "mask", "start_utc": start}))


def test_recent_runs_mixed_naive_and_unreadable(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    _make(root, "2026-08-31_010000_aaaaaaaa__mask", "2026-08-31T01:00:00")
    _make(root, "2026-08-31_020000_bbbbbbbb__mask", "not a date")
    assert len(RJ.recent_runs(5)) == 2


def test_recent_runs_naive_and_aware_order(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    _make(root, "2026-08-31_010000_aaaaaaaa__mask", "2026-08-31T01:00:00")
    _make(root, "2026-08-31_030000_bbbbbbbb__mask",
          "2026-08-31T03:00:00+00:00")
    out = RJ.recent_runs(5)
    assert [e["dir"].name[:17] for e in out] == [
        "2026-08-31_030000", "2026-08-31_010000"]
