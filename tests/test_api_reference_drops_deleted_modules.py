"""A module that no longer exists must not survive on the API reference.

The API pages are generated, so deleting a module deletes its page -- but
only in a build that starts from an empty tree. AutoAPI writes the pages it
can generate and never removes one it no longer generates, and Sphinx leaves
old HTML in the output directory, so an incremental build keeps publishing
the page of a module that was deleted months ago.

The localization catalogs below ``docs/source/_static/i18n/api`` are the
other half of the same page. They are checked in rather than generated at
build time, so a deleted module's translated docstrings stay in the payload
every reader downloads until something removes them. That payload is what
the language picker swaps in, so a stale entry is a page in nine languages
for a module the product does not have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "spacr"
CATALOG_DIR = ROOT / "docs" / "source" / "_static" / "i18n" / "api"
DOCS_WORKFLOW = ROOT / ".github" / "workflows" / "docs.yml"

#: Directories the documentation extractor never walks, so a name under one
#: of them is not a module the API reference can publish.
UNPUBLISHED = frozenset({"tests", "__pycache__", "backup_icons"})


def live_module_names(package: Path = PACKAGE) -> set[str]:
    """Every dotted module name/prefix and package-level public definition.

    A package directory counts through the ``.py`` files inside it, because
    ``spacr.resources`` carries documented modules without an ``__init__``.
    A public function or class defined by the package ``__init__`` also owns a
    two-part catalog key. It must not be mistaken for a deleted module merely
    because its name occupies the same dotted position as one.
    """
    import ast

    names: set[str] = set()
    for path in package.rglob("*.py"):
        if any(part in UNPUBLISHED for part in path.parts):
            continue
        parts = list(path.relative_to(package.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        for end in range(1, len(parts) + 1):
            names.add(".".join(parts[:end]))
    init_path = package / "__init__.py"
    if init_path.is_file():
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef))
                    and not node.name.startswith("_")):
                names.add(f"{package.name}.{node.name}")
    return names


def symbols_naming_absent_modules(symbols, live: set[str]) -> list[str]:
    """The catalog keys whose owning top-level module is not in the tree."""
    absent = []
    for key in symbols:
        parts = key.split(".")
        if len(parts) < 2:
            continue
        if ".".join(parts[:2]) not in live:
            absent.append(key)
    return absent


def catalog_paths() -> list[Path]:
    return sorted(CATALOG_DIR.glob("*.json"))


def test_the_api_catalogs_cover_english_and_the_translated_languages():
    """The rule below is only worth anything if it reads the real payload."""
    names = {path.stem for path in catalog_paths()}
    assert "en" in names
    assert len(names) >= 10, f"only found catalogs for {sorted(names)}"


@pytest.mark.parametrize("path", catalog_paths(), ids=lambda p: p.stem)
def test_no_api_catalog_documents_a_module_that_was_deleted(path):
    """A deleted module's docstrings must leave the localization payload.

    The Tkinter interface -- ``gui``, ``gui_core``, ``gui_elements``,
    ``gui_utils``, the ``app_*`` launchers and the ``legacy_tk`` package that
    briefly held them -- was removed from spaCR, and every one of them was
    still described in all ten catalogs. Those entries are reachable from the
    API page's language picker, so they are documentation of a product that
    no longer exists rather than dead weight nobody sees.
    """
    live = live_module_names()
    payload = json.loads(path.read_text(encoding="utf-8"))
    absent = symbols_naming_absent_modules(payload["symbols"], live)
    del payload
    assert not absent, (
        f"{path.name} documents {len(absent)} symbols of deleted modules, "
        f"starting with {sorted(absent)[:5]}"
    )


def test_the_deleted_module_rule_notices_a_module_that_disappears(tmp_path):
    """Prove the rule discriminates instead of passing on an empty tree.

    A check of this shape passes trivially if ``live_module_names`` is broad
    enough to accept anything, and nothing else in the suite would notice.
    """
    package = tmp_path / "spacr"
    (package / "qt").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text("", encoding="utf-8")
    (package / "qt" / "app.py").write_text("", encoding="utf-8")

    live = live_module_names(package)
    assert live == {"spacr", "spacr.core", "spacr.qt", "spacr.qt.app"}

    kept = ["spacr", "spacr.core.run", "spacr.qt.app.APPS"]
    gone = ["spacr.gui_elements.spacrButton", "spacr.legacy_tk"]
    assert symbols_naming_absent_modules(kept + gone, live) == gone


def test_the_deleted_module_rule_keeps_a_package_without_an_init(tmp_path):
    """``spacr.resources`` has documented modules and no ``__init__.py``.

    Requiring an ``__init__`` would delete every catalog entry under it, so
    the rule reads the files rather than the package marker.
    """
    package = tmp_path / "spacr"
    (package / "resources" / "home").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "resources" / "home" / "versions.py").write_text(
        "", encoding="utf-8")

    live = live_module_names(package)
    assert "spacr.resources" in live
    assert symbols_naming_absent_modules(
        ["spacr.resources.home.versions.app_map"], live) == []


def test_the_documentation_build_starts_from_an_empty_generated_tree():
    """Sphinx and AutoAPI both keep the output of a module that is gone.

    Neither tool removes a page it no longer generates, so without this
    cleanup the published reference keeps a page -- and a search-index entry
    -- for every module ever documented. Deleting the two generated trees
    before the build is the only thing that retires one.
    """
    workflow = DOCS_WORKFLOW.read_text(encoding="utf-8")
    cleanup = [
        line for line in workflow.splitlines()
        if "rm -rf" in line
        and "docs/_build" in line
        and "docs/source/api" in line
    ]
    assert cleanup, (
        "the docs workflow no longer clears docs/_build and docs/source/api "
        "before sphinx-build, so pages of deleted modules stay published"
    )
