"""spaCR code collects garbage only through :func:`spacr._gc.collect`.

A direct standard-library ``gc.collect()`` on a pipeline worker walks the
whole GUI heap while holding the interpreter lock, and can run Qt
destructors on the wrong thread. ``spacr.io._read_db`` once did exactly that
through a local ``import gc`` that shadowed the module's ``_gc`` alias, and
stalled the window for 0.39 s during the Invasion run in lesson 26.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "spacr"

ALLOWED = {
    "spacr/_gc.py":
        "the wrapper itself: collects synchronously headless or on the GUI "
        "thread, and defers a worker's request to the GUI timer",
    "spacr/qt/gc_policy.py":
        "the GUI-thread QTimer that services deferred requests; it must "
        "collect synchronously, on the GUI thread",
}


def _direct_collect_calls(tree):
    """Yield the line of every call that reaches the standard-library collector."""
    module_names, function_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gc":
                    module_names.add(alias.asname or "gc")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "gc":
                for alias in node.names:
                    if alias.name == "collect":
                        function_names.add(alias.asname or "collect")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in function_names:
            yield node.lineno
        elif isinstance(func, ast.Attribute) and func.attr == "collect":
            target = func.value
            if isinstance(target, ast.Name) and target.id in module_names:
                yield node.lineno
            elif (isinstance(target, ast.Call)
                  and isinstance(target.func, (ast.Name, ast.Attribute))
                  and getattr(target.func, "id",
                              getattr(target.func, "attr", None))
                  in ("__import__", "import_module")
                  and target.args
                  and isinstance(target.args[0], ast.Constant)
                  and target.args[0].value == "gc"):
                yield node.lineno


def _offenders():
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        lines = list(_direct_collect_calls(tree))
        if lines:
            found[relative] = lines
    return found


def test_no_spacr_module_calls_the_standard_library_collector_directly():
    offenders = {path: lines for path, lines in _offenders().items()
                 if path not in ALLOWED}
    assert offenders == {}, (
        "Call spacr._gc.collect instead of gc.collect (a local `import gc` "
        f"shadows the module's `_gc` alias): {offenders}")


def test_the_allow_list_names_only_files_that_still_need_it():
    found = _offenders()
    stale = sorted(path for path in ALLOWED if path not in found)
    assert stale == [], f"remove from ALLOWED: {stale}"


def test_the_scan_sees_every_spelling_of_a_direct_call():
    source = (
        "import gc\n"
        "import gc as collector\n"
        "from gc import collect\n"
        "from gc import collect as sweep\n"
        "import importlib\n"
        "from . import _gc as tidy\n"
        "def f():\n"
        "    import gc\n"
        "    gc.collect()\n"
        "collector.collect(2)\n"
        "collect()\n"
        "sweep()\n"
        "__import__('gc').collect()\n"
        "importlib.import_module('gc').collect()\n"
        "tidy.collect()\n"
    )
    lines = sorted(_direct_collect_calls(ast.parse(source)))
    assert lines == [9, 10, 11, 12, 13, 14]
