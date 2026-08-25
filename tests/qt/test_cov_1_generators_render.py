"""The variant renderer is a script, and its exit status is its result.

The reviewer runs ``render.py`` from a shell and reads ``$?``. Without the
``__main__`` guard turning ``main()``'s return value into ``SystemExit``, a
run that found every render missing would still exit 0 and a CI step wrapped
around it would report success.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PIL")

pytestmark = pytest.mark.qt

GENERATORS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "spacr", "resources", "home", "versions", "_generators"))


def _load(name, module_name):
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandboxed_common(qapp, monkeypatch, tmp_path):
    """``common`` with its output directories pointed at ``tmp_path``.

    Installed under the plain name ``common`` because that is the name
    ``render`` imports; without the redirect the self-check would read -- and
    the renderer's other paths would write over -- the checked-in renders.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    saved = {name: sys.modules.get(name)
             for name in ("common", "parts", "variants")}
    versions = tmp_path / "versions"
    here = versions / "_generators"
    here.mkdir(parents=True)
    try:
        module = _load("common", "common")
        monkeypatch.setattr(module, "versions_dir", lambda: str(versions))
        monkeypatch.setattr(module, "here", lambda: str(here))
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_running_the_renderer_as_a_script_exits_with_its_own_result(
        sandboxed_common, monkeypatch, tmp_path, capsys):
    """``--check`` over a directory with no renders exits 0 and says so.

    The self-check reports rather than raises -- a missing PNG is a finding,
    not a crash -- and the script's status is whatever ``main()`` returned.
    """
    monkeypatch.setattr(sys, "argv", ["render.py", "--check"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(os.path.join(GENERATORS, "render.py"),
                       run_name="__main__")

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "self-check" in out
    checked = next(line for line in out.splitlines() if "PNGs:" in line)
    assert "0 ok" in checked, (
        f"nothing was rendered into the sandbox, so nothing can be ok: "
        f"{checked}")
    assert "contact sheet:   MISSING" in out
    assert str(tmp_path) in out, "the sandbox is what was audited"
