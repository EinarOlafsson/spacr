"""Repo invariants for the two things a new user touches first.

1. The tutorial link. The lesson library publishes at
   ``https://einarolafsson.github.io/spacr/tutorials/`` because
   ``docs/source/conf.py`` copies ``docs/source/_extra/`` to the site root.
   Both GUIs shipped the singular ``/tutorial/``, which 404s. Before this
   test, ``grep -rn "tutorials/" --include=*.py spacr/`` returned nothing:
   the Python package did not link to the library at all.

2. The install recipe. ``docs/source/index.rst`` printed ``pip install spacr``
   followed by ``spacr``, but PySide6 lives in the ``qt`` extra, so that
   recipe ends in a bare ``ModuleNotFoundError``.

Deliberately free of Qt and Tk imports so it runs on a core-only install.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "spacr"
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
INDEX_RST = DOCS_SOURCE / "index.rst"

DEAD_URL = "einarolafsson.github.io/spacr/tutorial/"
LIVE_URL = "https://einarolafsson.github.io/spacr/tutorials/"


def _scanned_sources():
    """Package sources plus hand-written docs.

    ``docs/source/_extra`` is the 723 MB published tutorial bundle — content
    the maintainer owns and this test has no business reading.
    """
    yield from sorted(PACKAGE.rglob("*.py"))
    yield from sorted(DOCS_SOURCE.glob("*.rst"))
    yield REPO_ROOT / "README.rst"


def _module_constant(path: Path, name: str):
    """Read a module-level string constant without importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} has no module-level {name}")


# --- the link ------------------------------------------------------------

def test_nothing_links_to_the_404_singular_tutorial_path():
    offenders = []
    for path in _scanned_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if DEAD_URL in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "these link to a GitHub Pages 404 (the published path is "
        f"{LIVE_URL}):\n" + "\n".join(offenders)
    )


def test_the_tk_gui_logo_button_opens_the_lesson_library():
    """`spacr/gui.py` — the legacy Tk startup screen's logo button."""
    assert _module_constant(PACKAGE / "gui.py", "TUTORIALS_URL") == LIVE_URL


def test_the_qt_gui_help_menu_targets_the_lesson_library():
    """`spacr/qt/app.py` — asserted from source so no PySide6 is needed."""
    text = (PACKAGE / "qt" / "app.py").read_text(encoding="utf-8")
    base = re.search(r'^DOCS_BASE_URL = "([^"]+)"', text, re.M)
    assert base, "spacr/qt/app.py no longer defines DOCS_BASE_URL"
    assert re.search(
        r'^TUTORIALS_URL = f"\{DOCS_BASE_URL\}/tutorials/"', text, re.M
    )
    assert base.group(1) + "/tutorials/" == LIVE_URL


def test_the_python_package_links_to_the_tutorial_library_at_all():
    """The regression that started this: zero hits for ``tutorials/``."""
    linking = [
        path.relative_to(REPO_ROOT)
        for path in sorted(PACKAGE.rglob("*.py"))
        if "tutorials/" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert linking, "no module in spacr/ links to the tutorial library"


def test_the_landing_page_links_to_the_tutorial_library():
    index = INDEX_RST.read_text(encoding="utf-8")
    assert "tutorials/" in index, (
        "docs/source/index.rst never links to the tutorial library it ships"
    )


# --- the launch path -----------------------------------------------------

_BARE_RECIPE = re.compile(
    r"^\s*(?:python -m )?pip install spacr\s*$\n\s*spacr\s*(?:#.*)?$", re.M
)


def test_the_docs_do_not_print_a_recipe_that_ends_in_an_importerror():
    """``pip install spacr`` then ``spacr`` — PySide6 is extras-only."""
    index = INDEX_RST.read_text(encoding="utf-8")
    match = _BARE_RECIPE.search(index)
    assert match is None, (
        "docs/source/index.rst tells the user to launch the GUI from a "
        f"core-only install:\n{match.group(0) if match else ''}"
    )


def test_the_docs_install_the_qt_extra_for_the_gui():
    index = INDEX_RST.read_text(encoding="utf-8")
    assert 'pip install "spacr[qt]"' in index


def test_the_readme_and_the_docs_agree_on_the_gui_install():
    """README.rst was already right; index.rst was not."""
    readme = (REPO_ROOT / "README.rst").read_text(encoding="utf-8")
    index = INDEX_RST.read_text(encoding="utf-8")
    recipe = 'python -m pip install "spacr[qt]"'
    assert recipe in readme
    assert recipe in index


def test_pyside6_is_an_extra_and_not_a_core_requirement():
    """The premise of the guard. If this ever changes, the guard is dead code."""
    setup = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    qt_extra = re.search(r"'qt': \[(.*?)\]", setup, re.S)
    assert qt_extra and "PySide6" in qt_extra.group(1)

    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    core = [
        line for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any(line.lower().startswith("pyside6") for line in core), (
        "PySide6 became a core requirement; the spacr.qt install guard is "
        "now unreachable and should be removed"
    )
