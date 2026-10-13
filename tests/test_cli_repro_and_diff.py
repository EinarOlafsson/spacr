"""Tests for spacr.cli_repro + spacr.qt.settings_diff (headless)."""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_runs(monkeypatch, tmp_path):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield


# ---------------------------------------------------------------------------
# spacr repro CLI
# ---------------------------------------------------------------------------

def _make_dummy_run(tmp_path: Path, app_key: str = "mask") -> Path:
    from spacr.run_journal import open_run
    with open_run(app_key, {"src": "/tmp/x", "n": 5}) as run:
        pass
    return run.dir


def test_cli_repro_show_prints_manifest(tmp_path):
    from spacr.cli_repro import main
    d = _make_dummy_run(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([str(d), "--show"])
    assert rc == 0
    out = buf.getvalue()
    assert "mask" in out
    assert "src" in out


def test_cli_repro_dry_prints_would_run(tmp_path, monkeypatch):
    """--dry resolves the pipeline entry then prints — doesn't invoke it."""
    from spacr import cli_repro
    d = _make_dummy_run(tmp_path, app_key="mask")

    # Stub the resolver so we don't require heavy pipeline imports
    def _fake_entry(app_key):
        def _run(settings): pass
        _run.__module__ = "spacr.core"
        _run.__name__ = "preprocess_generate_masks"
        return _run
    monkeypatch.setattr(cli_repro, "_resolve_pipeline", _fake_entry)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_repro.main([str(d), "--dry"])
    assert rc == 0
    out = buf.getvalue()
    assert "would run:" in out
    assert "spacr.core.preprocess_generate_masks" in out


def test_cli_repro_missing_folder_returns_2(tmp_path):
    from spacr.cli_repro import main
    rc = main([str(tmp_path / "does_not_exist")])
    assert rc == 2


def test_cli_repro_bad_folder_no_manifest_returns_2(tmp_path):
    from spacr.cli_repro import main
    bad = tmp_path / "empty_folder"
    bad.mkdir()
    rc = main([str(bad)])
    assert rc == 2


def test_cli_repro_replays_settings_via_stubbed_entry(tmp_path, monkeypatch):
    """End-to-end (dry side): --show → resolves + runs → returns 0."""
    from spacr import cli_repro
    d = _make_dummy_run(tmp_path)
    called = {}
    def _fake_entry(app_key):
        def _run(settings):
            called["settings"] = dict(settings)
        return _run
    monkeypatch.setattr(cli_repro, "_resolve_pipeline", _fake_entry)
    rc = cli_repro.main([str(d)])
    assert rc == 0
    assert called["settings"]["src"] == "/tmp/x"


# ---------------------------------------------------------------------------
# Settings diff
# ---------------------------------------------------------------------------

def test_diff_settings_marks_changed():
    from spacr.qt.settings_diff import diff_settings
    a = {"k": 1, "same": "x"}
    b = {"k": 2, "same": "x"}
    d = diff_settings(a, b)
    assert len(d) == 1
    assert d[0].key == "k"
    assert d[0].kind == "changed"
    assert d[0].a_val == 1 and d[0].b_val == 2


def test_diff_settings_marks_added_and_removed():
    from spacr.qt.settings_diff import diff_settings
    a = {"only_a": 1}
    b = {"only_b": 2}
    d = diff_settings(a, b)
    kinds = {r.kind for r in d}
    assert kinds == {"added", "removed"}


def test_diff_settings_normalises_string_vs_native():
    """'1' and 1 should compare equal — CSV round-trips as strings."""
    from spacr.qt.settings_diff import diff_settings
    a = {"n": 1}
    b = {"n": "1"}
    assert diff_settings(a, b) == []


def test_diff_settings_load_from_run_folder(tmp_path):
    """The Qt-layer _load helper should accept a run folder Path."""
    from spacr.qt import settings_diff
    from spacr.run_journal import open_run
    import spacr.run_journal as rj
    # tmp_path fixture already redirected runs_root
    with open_run("mask", {"src": "/a"}) as r1:
        pass
    with open_run("mask", {"src": "/b"}) as r2:
        pass
    diff = settings_diff.diff_settings(
        settings_diff._load(r1.dir),
        settings_diff._load(r2.dir),
    )
    changed = [r for r in diff if r.key == "src"]
    assert len(changed) == 1
    assert changed[0].kind == "changed"
