"""Structural contract for spaCR's generated API reference."""
from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "spacr"


def _api_modules():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "resources" in path.parts or "tutorial" in path.parts:
            continue
        yield path


def test_every_api_module_has_a_module_docstring():
    missing = []
    for path in _api_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(str(path.relative_to(PACKAGE_ROOT.parent)))
    assert not missing, "modules missing API documentation:\n" + "\n".join(missing)


def test_every_public_api_function_and_class_has_a_docstring():
    """Top-level public symbols are exactly what AutoAPI exposes as API."""
    missing = []
    node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for path in _api_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, node_types) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    relative = path.relative_to(PACKAGE_ROOT.parent)
                    missing.append(f"{relative}:{node.lineno} {node.name}")
    assert not missing, (
        "public API symbols missing documentation:\n" + "\n".join(missing)
    )


def test_sphinx_autoapi_covers_the_package_and_undocumented_members():
    conf = (PACKAGE_ROOT.parent / "docs" / "source" / "conf.py").read_text(
        encoding="utf-8"
    )
    assert "'autoapi.extension'" in conf
    assert "autoapi_dirs" in conf
    assert "'members'" in conf
    assert "'undoc-members'" in conf
