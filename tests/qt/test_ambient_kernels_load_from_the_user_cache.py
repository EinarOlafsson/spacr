"""The ambient grain kernels compile once per machine, not once per launch.

Compiling them held the GIL for about 2.7 s on every start, over the startup
animation. They are now cached in a spaCR user folder (never beside the
package source), so a second process loads them without compiling.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("numba")

SCRIPT = textwrap.dedent("""
    import json
    import numba.core.dispatcher as dispatcher
    compiled = []
    real = dispatcher._FunctionCompiler._compile_core

    def counting(self, args, return_type):
        compiled.append(self.py_func.__name__)
        return real(self, args, return_type)

    dispatcher._FunctionCompiler._compile_core = counting
    from spacr.qt.widgets import ambient
    ambient._warm_packed_scatter()
    print(json.dumps({"compiled": compiled,
                      "ready": ambient._PACKED_SCATTER is not None,
                      "dir": str(ambient._numba_cache_directory())}))
""")


def _run(home, extra=None):
    env = dict(os.environ, SPACR_HOME=str(home), CUDA_VISIBLE_DEVICES="",
               QT_QPA_PLATFORM="offscreen")
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBA_DISABLE_JIT", None)
    env.update(extra or {})
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=300)
    lines = [x for x in done.stdout.splitlines() if x.startswith("{")]
    assert lines, done.stderr[-2000:]
    return json.loads(lines[-1])


def _package_files():
    import spacr
    root = Path(spacr.__file__).resolve().parent
    return {p for p in root.rglob("*") if p.suffix in (".nbi", ".nbc")}


def test_a_second_process_loads_the_kernels_from_the_user_cache(tmp_path):
    before = _package_files()
    first = _run(tmp_path)
    assert first["ready"] and first["compiled"]
    assert Path(first["dir"]) == tmp_path / "cache" / "numba"
    assert any(Path(first["dir"]).rglob("*.nbi"))

    second = _run(tmp_path)
    assert second["ready"]
    assert second["compiled"] == []
    assert _package_files() == before


def test_an_unwritable_cache_folder_falls_back_to_compiling(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    result = _run(tmp_path, {"NUMBA_CACHE_DIR": str(blocker / "sub")})
    assert result["ready"] and result["compiled"]
    assert result["dir"] == "None"
