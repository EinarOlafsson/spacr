"""Every cached numba kernel in spaCR is precompiled by the helper process.

The ambient grain kernels were moved to a child interpreter on 2026-10-20
(lesson 26). The night-theme fractal kernels then still compiled in the
application, and could cache to numba's default folder because they were
decorated before spaCR's cache folder was set. This file keeps the list
honest: a new ``cache=True`` kernel must be precompiled or excluded with a
reason, and the helper must leave every listed kernel in spaCR's folder so
that a second process compiles nothing.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _decorator_caches(node) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("njit", "jit"):
            continue
        for keyword in decorator.keywords:
            if (keyword.arg == "cache" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True):
                return True
    return False


def _cached_kernels() -> set:
    found = set()
    for path in (ROOT / "spacr").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        defined = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.setdefault(node.name, node)
                if _decorator_caches(node):
                    found.add(f"{module}.{node.name}")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", "")
                    == "_compiled_kernel" and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in defined):
                found.add(f"{module}.{node.args[0].id}")
    return found


def test_every_cached_kernel_is_precompiled_or_excluded_with_a_reason():
    from spacr.qt.widgets import ambient

    kernels = _cached_kernels()
    assert len(kernels) >= 10, sorted(kernels)
    listed = set(ambient._PRECOMPILED_KERNELS) | set(ambient._PRECOMPILE_EXCLUDED)
    assert kernels - listed == set(), (
        "cache=True kernels nobody precompiles: "
        f"{sorted(kernels - listed)}; add them to _PRECOMPILED_KERNELS or to "
        "_PRECOMPILE_EXCLUDED with a reason")
    assert listed - kernels == set(), f"listed but gone: {sorted(listed - kernels)}"
    assert all(str(r).strip() for r in ambient._PRECOMPILE_EXCLUDED.values())


def test_kernel_modules_set_spacrs_cache_folder_before_numba():
    for name in ("fractal_cascade", "fractal_space", "fractal_travel"):
        source = (ROOT / "spacr/qt/widgets" / f"{name}.py").read_text(encoding="utf-8")
        assert source.index("_use_spacrs_kernel_cache()\n\n") < source.index("from numba import"), name


SCRIPT = textwrap.dedent("""
    import json, os, sys
    import spacr
    assert os.path.realpath(spacr.__file__).startswith(
        os.path.realpath(os.environ["SPACR_TESTED_ROOT"]) + os.sep), spacr.__file__
    import numba.core.dispatcher as dispatcher
    compiled = []
    real = dispatcher._FunctionCompiler._compile_core

    def counting(self, args, return_type):
        compiled.append(self.py_func.__module__ + "." + self.py_func.__name__)
        return real(self, args, return_type)

    dispatcher._FunctionCompiler._compile_core = counting
    from spacr.qt.widgets import ambient
    status = ambient._prime_kernel_cache()
    folder = ambient._numba_cache_directory()
    import numba.core.config as config
    print(json.dumps({"status": status, "folder": str(folder),
                      "config": str(config.CACHE_DIR),
                      "compiled": [c for c in compiled if c.startswith("spacr.")],
                      "files": sorted(p.name for p in folder.rglob("*.nbi"))}))
""")


def _run(home):
    pytest.importorskip("numba")
    env = dict(os.environ, SPACR_HOME=str(home), CUDA_VISIBLE_DEVICES="",
               QT_QPA_PLATFORM="offscreen", SPACR_TESTED_ROOT=str(ROOT))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for key in ("NUMBA_CACHE_DIR", "NUMBA_DISABLE_JIT", "SPACR_DISTRIBUTION_SMOKE"):
        env.pop(key, None)
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=1200)
    lines = [x for x in done.stdout.splitlines() if x.startswith("{")]
    assert lines, done.stderr[-3000:]
    return json.loads(lines[-1])


def test_the_helper_caches_every_listed_kernel_in_spacrs_folder(tmp_path):
    from spacr.qt.widgets import ambient

    first = _run(tmp_path)
    assert first["status"] == 0, first
    assert first["config"] == first["folder"]
    assert Path(first["folder"]).is_relative_to(tmp_path)
    for kernel in ambient._PRECOMPILED_KERNELS:
        module, function = kernel.rsplit(".", 1)
        stem = module.rsplit(".", 1)[1]
        assert any(f.startswith(f"{stem}.{function}-") for f in first["files"]), (
            f"{kernel} left no cache entry in spaCR's folder")
    second = _run(tmp_path)
    assert second["status"] == 0
    assert second["compiled"] == [], (
        f"a second process compiled {second['compiled']} instead of loading them")
