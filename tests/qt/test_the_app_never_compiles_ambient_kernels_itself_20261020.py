"""The application process loads the ambient kernels; a helper compiles them.

Lesson 26 froze for 1-2 s while a worker thread compiled the spaCR field
backdrop's numba kernels: the compiler holds the interpreter lock, so the
GUI thread (in ``ambient._on_tick``) waited. On a cold cache the compile
thread now runs a child interpreter that fills the on-disk cache, and the
application only loads from it. A frozen build or a smoke run keeps the
old in-process path.
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

_TESTED_ROOT = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent("""
    import json, os, threading
    import spacr
    assert os.path.realpath(spacr.__file__).startswith(
        os.path.realpath(os.environ["SPACR_TESTED_ROOT"]) + os.sep), spacr.__file__
    import numba.core.dispatcher as dispatcher
    compiled = []
    real = dispatcher._FunctionCompiler._compile_core

    def counting(self, args, return_type):
        compiled.append(self.py_func.__name__)
        return real(self, args, return_type)

    dispatcher._FunctionCompiler._compile_core = counting
    from spacr.qt.widgets import ambient
    calls = []
    real_prime = ambient._prime_in_helper_process

    def watching():
        calls.append(threading.current_thread().name)
        return real_prime()

    ambient._prime_in_helper_process = watching
    worker = threading.Thread(target=ambient._warm_packed_scatter,
                              name="spacr-grain-compile")
    worker.start()
    worker.join(600)
    satin = threading.Thread(target=ambient._SATIN_COMPILER._warm,
                             name="spacr-satin-compile")
    satin.start()
    satin.join(600)
    folder = ambient._numba_cache_directory()
    print(json.dumps({
        "compiled": [c for c in compiled if not c.startswith("impl")],
        "ready": ambient._PACKED_SCATTER is not None,
        "satin": ambient._SATIN_COMPILER.kernel is not None,
        "primed": ambient._primed_marker(folder).exists(),
        "calls": calls}))
""")


def _run(home, extra=None):
    env = dict(os.environ, SPACR_HOME=str(home), CUDA_VISIBLE_DEVICES="",
               QT_QPA_PLATFORM="offscreen", SPACR_TESTED_ROOT=str(_TESTED_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_TESTED_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for key in ("NUMBA_CACHE_DIR", "NUMBA_DISABLE_JIT", "SPACR_DISTRIBUTION_SMOKE"):
        env.pop(key, None)
    env.update(extra or {})
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=900)
    lines = [x for x in done.stdout.splitlines() if x.startswith("{")]
    assert lines, done.stderr[-3000:]
    return json.loads(lines[-1])


def test_a_cold_cache_is_filled_by_a_helper_and_the_app_compiles_nothing(tmp_path):
    first = _run(tmp_path)
    assert first["ready"] and first["satin"], first
    assert first["primed"], first
    assert first["calls"], "the compile thread never asked the helper"
    assert first["compiled"] == [], (
        f"the application process compiled {first['compiled']} itself")
    second = _run(tmp_path)
    assert second["ready"] and second["satin"]
    assert second["compiled"] == []


def test_a_smoke_run_compiles_in_process_as_before(tmp_path):
    smoke = _run(tmp_path, {"SPACR_DISTRIBUTION_SMOKE": "1"})
    assert smoke["ready"]
    assert not smoke["primed"]
    assert "_scatter_packed_grains" in smoke["compiled"]


def test_priming_is_refused_in_a_frozen_build(monkeypatch):
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not ambient._priming_allowed()
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    assert ambient._prime_in_helper_process() is False
    assert ran == []
