"""Coverage-fill for spacr.run_journal (Run lifecycle + totals + settings)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacr import run_journal as RJ


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(RJ, "runs_root", lambda: d)
    return d


# ---------------------------------------------------------------------------
# hash_file / _pkg_version
# ---------------------------------------------------------------------------

def test_hash_file(tmp_path):
    f = tmp_path / "m.pt"; f.write_bytes(b"weights")
    digest = RJ.hash_file(f)
    assert isinstance(digest, str) and len(digest) == 16
    # nonexistent → None
    assert RJ.hash_file(tmp_path / "nope") is None


def test_pkg_version():
    assert isinstance(RJ._pkg_version("numpy"), str)
    assert RJ._pkg_version("definitely-not-a-package-xyz") == "not installed"


# ---------------------------------------------------------------------------
# open_run lifecycle
# ---------------------------------------------------------------------------

def test_open_run_success(runs_dir, tmp_path):
    ckpt = tmp_path / "cyto.pth"; ckpt.write_bytes(b"model-weights")
    out_file = tmp_path / "result.csv"; out_file.write_text("a,b\n1,2\n")
    with RJ.open_run("mask", {"src": "/data", "diameter": 30}) as run:
        run.record_model("cellpose_cyto", str(ckpt))
        dst = run.attach_output(str(out_file))
        assert dst is not None and dst.exists()
        assert RJ.current_run() is run
        run.set_status("success")
    # manifest written on exit
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert "cellpose_cyto" in manifest["model_hashes"]


def test_open_run_failure_reraises(runs_dir):
    with pytest.raises(ValueError):
        with RJ.open_run("measure", {}) as run:
            raise ValueError("boom")
    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "boom" in run.error_traceback   # captured on the Run object


def test_record_model_unreadable(runs_dir):
    with RJ.open_run("mask", {}) as run:
        run.record_model("missing", "/nonexistent/path.pth")  # no-op, no raise
    assert "missing" not in run.model_hashes


def test_attach_output_dir(runs_dir, tmp_path):
    srcdir = tmp_path / "figs"; srcdir.mkdir()
    (srcdir / "a.png").write_bytes(b"x")
    with RJ.open_run("classify", {}) as run:
        dst = run.attach_output(str(srcdir))
        assert dst is not None and (dst / "a.png").exists()


# ---------------------------------------------------------------------------
# journal_totals
# ---------------------------------------------------------------------------

def _write_manifest(runs_dir, name, **fields):
    d = runs_dir / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(fields))
    return d


def test_journal_totals(runs_dir):
    _write_manifest(runs_dir, "r1__mask", app_key="mask",
                    model_hashes={"cyto": "cyto.pth:abc123"})
    _write_manifest(runs_dir, "r2__mask", app_key="mask",
                    model_hashes={"cyto": "cyto.pth:abc123",   # dup digest
                                  "nuc": "nuc.pth:def456"})
    _write_manifest(runs_dir, "r3__measure", app_key="measure")
    _write_manifest(runs_dir, "r4__classify", app_key="classify",
                    models=[{"sha256": "legacy999"}])   # back-compat list
    # a non-dir and a bad manifest are ignored
    (runs_dir / "loose.txt").write_text("x")
    bad = runs_dir / "r5__mask"; bad.mkdir()
    (bad / "manifest.json").write_text("{not json")

    totals = RJ.journal_totals()
    assert totals["total_runs"] == 4
    assert totals["mask_runs"] == 2
    assert totals["measure_runs"] == 1
    assert totals["classify_runs"] == 1
    # abc123, def456, legacy999 → 3 distinct
    assert totals["models_recorded"] == 3


def test_journal_totals_no_root(tmp_path, monkeypatch):
    monkeypatch.setattr(RJ, "runs_root", lambda: tmp_path / "does_not_exist")
    totals = RJ.journal_totals()
    assert totals["total_runs"] == 0


# ---------------------------------------------------------------------------
# recent_runs + load_run_settings
# ---------------------------------------------------------------------------

def test_recent_runs(runs_dir):
    _write_manifest(runs_dir, "r1__mask", app_key="mask", status="success")
    _write_manifest(runs_dir, "r2__measure", app_key="measure",
                    status="failed")
    out = RJ.recent_runs(limit=5)
    assert isinstance(out, list) and len(out) == 2


def test_load_run_settings_json(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "settings.json").write_text(json.dumps({"src": "/data", "diam": 30}))
    out = RJ.load_run_settings(d)
    assert out["src"] == "/data"


def test_load_run_settings_csv_fallback(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "settings.csv").write_text("Key,Value\nsrc,/data\ndiam,30\n")
    out = RJ.load_run_settings(d)
    assert out["src"] == "/data" and out["diam"] == "30"


def test_load_run_settings_missing(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    with pytest.raises(FileNotFoundError):
        RJ.load_run_settings(d)
