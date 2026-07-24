"""Coverage-fill for cli_repro, __init__, __main__, logging_util."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# spacr/__init__.py — lazy attribute access
# ---------------------------------------------------------------------------

def test_download_models_lazy_attr():
    # spacr.download_models resolves via __getattr__ (lines 84-85).
    import spacr
    fn = spacr.download_models
    assert callable(fn)


def test_dir_includes_submodules():
    # __dir__ merges the lazy submodule names (line 95).
    import spacr
    names = dir(spacr)
    assert "core" in names and "download_models" in names


def test_unknown_attr_raises():
    import spacr
    with pytest.raises(AttributeError):
        spacr.definitely_not_a_real_attribute


# ---------------------------------------------------------------------------
# spacr/__main__.py — command dispatch
# ---------------------------------------------------------------------------

def test_main_make_masks_command(monkeypatch):
    # make-masks command path (lines 90-91).
    import spacr.__main__ as M
    called = {}
    import spacr.app_make_masks as AMM
    monkeypatch.setattr(AMM, "start_make_mask_app",
                        lambda: called.setdefault("ran", True))
    rc = M.main(["make-masks"])
    assert rc == 0 and called.get("ran")


def test_main_unknown_command_errors(monkeypatch):
    # parser.error path for an unrecognised command (line 95).
    import spacr.__main__ as M
    # argparse subparsers reject unknown commands with SystemExit(2).
    with pytest.raises(SystemExit):
        M.main(["not-a-command"])


def test_main_version_command(capsys):
    import spacr.__main__ as M
    rc = M.main(["version"])
    assert rc == 0


# ---------------------------------------------------------------------------
# spacr/cli_repro.py — replay branches
# ---------------------------------------------------------------------------

def _make_run(tmp_path, with_hashes=False):
    from spacr.run_journal import open_run
    import spacr.run_journal as rj
    # Point runs_root at tmp so the run lands here.
    with open_run("mask", {"src": str(tmp_path / "data")}) as run:
        pass
    if with_hashes:
        mp = run.dir / "manifest.json"
        m = json.loads(mp.read_text())
        m["model_hashes"] = {"cyto": "abc123"}
        mp.write_text(json.dumps(m))
    return run.dir


def test_cli_show_prints_model_hashes(tmp_path, monkeypatch, capsys):
    # --show with model_hashes present (lines 45-49).
    import spacr.run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    from spacr.cli_repro import main
    d = _make_run(tmp_path, with_hashes=True)
    rc = main([str(d), "--show"])
    assert rc == 0
    assert "models:" in capsys.readouterr().out


def test_cli_resolve_pipeline_exception_returns_none(monkeypatch):
    # _resolve_pipeline swallows import errors → None (lines 60-64).
    import spacr.cli_repro as C
    monkeypatch.setattr(
        "spacr.qt.bridge.resolve_pipeline_entry",
        lambda k: (_ for _ in ()).throw(RuntimeError("boom")))
    # Even if resolve raises, _resolve_pipeline returns None.
    assert C._resolve_pipeline("mask") is None


def test_cli_no_pipeline_entry_returns_2(tmp_path, monkeypatch):
    # entry is None → error + return 2 (lines 116-118).
    import spacr.run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    import spacr.cli_repro as C
    monkeypatch.setattr(C, "_resolve_pipeline", lambda k: None)
    d = _make_run(tmp_path)
    rc = C.main([str(d)])
    assert rc == 2


def test_cli_replay_raises_returns_1(tmp_path, monkeypatch):
    # entry raises during replay → status failed + return 1 (130-134).
    import spacr.run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    import spacr.cli_repro as C
    def _boom(settings):
        raise ValueError("replay failed")
    monkeypatch.setattr(C, "_resolve_pipeline", lambda k: _boom)
    d = _make_run(tmp_path)
    rc = C.main([str(d)])
    assert rc == 1


def test_cli_replay_success_returns_0(tmp_path, monkeypatch):
    # Successful replay → "done" + return 0 (line 140 area).
    import spacr.run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    import spacr.cli_repro as C
    monkeypatch.setattr(C, "_resolve_pipeline",
                        lambda k: (lambda settings: None))
    d = _make_run(tmp_path)
    rc = C.main([str(d)])
    assert rc == 0


# ---------------------------------------------------------------------------
# spacr/logging_util.py — Timer + timed-function scan
# ---------------------------------------------------------------------------

def test_timer_records_elapsed_when_enabled(monkeypatch):
    # __exit__ elapsed calc + _TIMING_ENABLED branch (line 378).
    import spacr.logging_util as LU
    monkeypatch.setattr(LU, "_TIMING_ENABLED", True, raising=False)
    with LU.Timer("test-op") as t:
        sum(range(1000))
    assert t.elapsed_ms is not None and t.elapsed_ms >= 0


def test_timer_noop_when_not_started():
    import spacr.logging_util as LU
    t = LU.Timer("x")
    # Calling __exit__ without __enter__ should be a no-op (early return).
    t.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# __main__ — every command branch (stubbed launchers)
# ---------------------------------------------------------------------------

import pytest as _pytest

@_pytest.mark.parametrize("command,module,fn", [
    ("gui", "spacr.gui", "gui_app"),
    ("mask", "spacr.app_mask", "start_mask_app"),
    ("measure", "spacr.app_measure", "start_measure_app"),
    ("classify", "spacr.app_classify", "start_classify_app"),
    ("annotate", "spacr.app_annotate", "start_annotate_app"),
    ("sequencing", "spacr.app_sequencing", "start_seq_app"),
    ("umap", "spacr.app_umap", "start_umap_app"),
])
def test_main_each_command_dispatches(command, module, fn, monkeypatch):
    import importlib
    import spacr.__main__ as M
    mod = importlib.import_module(module)
    called = {}
    monkeypatch.setattr(mod, fn, lambda *a, **k: called.setdefault("x", 1))
    rc = M.main([command])
    assert rc == 0 and called.get("x") == 1


def test_init_lazy_submodule_import():
    # Accessing a declared submodule goes through __getattr__'s
    # import_module path (__init__ line 88).
    import spacr
    mod = spacr.run_journal   # light submodule
    assert hasattr(mod, "open_run")
