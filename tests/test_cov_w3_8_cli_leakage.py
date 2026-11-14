"""``python -m spacr.cli_leakage``: the ``__main__`` guard carries the exit code.

The exit codes and reports of ``main`` itself are table-driven in
``tests/test_cli_audit_commands.py``.
"""
from __future__ import annotations

import json

import pytest


def _crop(root, split, cls, name, payload=b"crop"):
    """Write one crop file at ``root/split/cls/name``."""
    path = root / split / cls / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_the_module_runs_as_a_script(tmp_path, capsys, monkeypatch):
    """``python -m spacr.cli_leakage`` raises SystemExit carrying the code.

    The ``__main__`` guard is the shape the console-script entry point and a
    ``-m`` invocation both take, so it is exercised the way a shell does:
    by executing the module under the name ``__main__``.
    """
    import runpy
    import sys

    _crop(tmp_path, "train", "neg", "plate1_A01_f1_o1.png", b"one")
    _crop(tmp_path, "test", "neg", "plate1_A01_f1_o2.png", b"two")
    monkeypatch.setattr(
        sys, "argv", ["spacr-leakage", str(tmp_path), "--group-by", "well"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.cli_leakage", run_name="__main__")

    assert excinfo.value.code == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
