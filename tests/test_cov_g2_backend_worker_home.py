"""The backend file loaded as a plain script still finds spaCR's home folder."""
from __future__ import annotations

import runpy
from pathlib import Path

import spacr._segmentation_backends as backends


def test_the_worker_script_reads_spacr_home_or_the_default(monkeypatch, tmp_path):
    namespace = runpy.run_path(backends.__file__, run_name="spacr_backend_worker")
    home = namespace["_spacr_home"]
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "portable"))
    assert home() == tmp_path / "portable"
    monkeypatch.delenv("SPACR_HOME")
    assert home() == Path.home() / ".spacr"
