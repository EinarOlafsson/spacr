"""The native installer workflow's clean-environment smoke run."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_installed_smoke_run_completes(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "packaging" / "online" / "smoke_installed.py"
    spec = importlib.util.spec_from_file_location("spacr_installed_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert module.main() == 0
