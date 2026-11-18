"""No test module may change ``CUDA_VISIBLE_DEVICES`` at import.

A module-level ``os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")`` runs at
collection and hides the GPU from every later test of a whole-suite or
``-m gpu`` run. Hide it inside a fixture with ``monkeypatch.setenv``, or in
a subprocess environment. ``conftest.py`` files are exempt.
"""
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()


def _touches_cuda_env(stmt):
    """True when a module-level statement mentions CUDA_VISIBLE_DEVICES outside a def or class."""
    stack = [stmt]
    while stack:
        part = stack.pop()
        if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(part, ast.Constant) and part.value == "CUDA_VISIBLE_DEVICES":
            return True
        stack.extend(ast.iter_child_nodes(part))
    return False


def offenders(paths=None):
    found = []
    for path in sorted(TESTS.rglob("*.py") if paths is None else paths):
        if path.name == "conftest.py" or path.resolve() == SELF:
            continue
        for stmt in ast.parse(path.read_text(encoding="utf-8")).body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)) \
                    or not _touches_cuda_env(stmt):
                continue
            found.append(f"{path.name}:{stmt.lineno}")
    return found


def test_no_test_module_hides_the_gpu_at_import():
    assert not offenders(), "module-level CUDA_VISIBLE_DEVICES in:\n  " + "\n  ".join(offenders())


def test_the_scan_flags_an_import_time_write_and_allows_a_fixture(tmp_path):
    bad = tmp_path / "test_bad.py"
    bad.write_text('import os\nos.environ.setdefault("CUDA_VISIBLE_DEVICES", "")\n')
    good = tmp_path / "test_good.py"
    good.write_text('import pytest\n\n@pytest.fixture(autouse=True)\ndef f(monkeypatch):\n'
                    '    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")\n')
    assert offenders([bad]) == ["test_bad.py:2"]
    assert offenders([good]) == []
