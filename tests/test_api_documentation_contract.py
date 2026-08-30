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


def test_sphinx_autoapi_has_a_curated_public_landing_page():
    conf = (PACKAGE_ROOT.parent / "docs" / "source" / "conf.py").read_text(
        encoding="utf-8"
    )
    assert "'autoapi.extension'" in conf
    assert "autoapi_dirs" in conf
    assert "'members'" in conf
    assert "'undoc-members'" not in conf
    assert "autoapi_template_dir" in conf
    template = (
        PACKAGE_ROOT.parent / "docs" / "source" / "_autoapi_templates" /
        "index.rst"
    ).read_text(encoding="utf-8")
    assert "Start with the workflow you want to run" in template
    assert "Complete module reference" in template


def test_generated_module_pages_are_intentional_orphans():
    """Sphinx 8 must not warn about pages outside the curated navigation.

    Sphinx 8.1 emits its not-in-any-toctree warning without a warning type,
    so ``suppress_warnings = ['toc.not_included']`` cannot match it.  AutoAPI
    still generates the complete contributor directory; marking each own
    module page as an intentional orphan keeps those pages linkable without
    adding every implementation module to the visible navigation.
    """
    template = (
        PACKAGE_ROOT.parent / "docs" / "source" / "_autoapi_templates" /
        "python" / "module.rst"
    ).read_text(encoding="utf-8")

    own_page_prefix = template.split("{{ obj.id }}", 1)[0]
    assert "{% if is_own_page %}" in own_page_prefix
    assert ":orphan:" in own_page_prefix
    assert template.index(":orphan:") < template.index("{{ obj.id }}")
    assert ".. dropdown:: Complete contributor module directory" in template
