"""Repo invariants for the two things a new user touches first.

1. The tutorial link. The lesson library publishes at
   ``https://einarolafsson.github.io/spacr/tutorials/`` because
   ``docs/source/conf.py`` copies ``docs/source/_extra/`` to the site root.
   Both GUIs shipped the singular ``/tutorial/``, which 404s. Before this
   test, ``grep -rn "tutorials/" --include=*.py spacr/`` returned nothing:
   the Python package did not link to the library at all.

2. The install recipe. The standard package includes the Qt desktop stack,
   so the landing page and README should both use ``pip install spacr``.

Deliberately free of Qt imports so it runs without a display.

ONE TEST IS GONE. The Tk startup screen's logo button opened the same
library from a ``TUTORIALS_URL`` constant in ``legacy_tk/gui.py``, and that
file is deleted -- the Qt help menu, asserted below from the source of
``spacr/qt/app.py``, is the only place the link is now published from.
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


def test_the_docs_install_and_launch_the_standard_package():
    """The documented standard install is followed by the desktop command."""
    index = INDEX_RST.read_text(encoding="utf-8")
    match = _BARE_RECIPE.search(index)
    assert match is not None, (
        "docs/source/index.rst should install spacr and launch its desktop "
        "interface, which is included in the standard package"
    )


def test_the_docs_do_not_require_the_redundant_qt_extra():
    index = INDEX_RST.read_text(encoding="utf-8")
    assert 'pip install spacr' in index
    assert 'spacr[qt]' not in index


def test_the_readme_and_the_docs_agree_on_the_gui_install():
    """Both entry points document the same standard installation."""
    readme = (REPO_ROOT / "README.rst").read_text(encoding="utf-8")
    index = INDEX_RST.read_text(encoding="utf-8")
    recipe = 'python -m pip install spacr'
    assert recipe in readme
    assert recipe in index


def test_the_documented_standard_install_declares_the_qt_stack():
    """Read actual package dependencies rather than an editable install shim."""
    setup = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    dependencies = next(
        ast.literal_eval(node.value)
        for node in setup.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "dependencies"
                for target in node.targets)
    )
    core = {re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower()
            for requirement in dependencies}
    assert {"pyside6", "qtawesome", "pyqtgraph"} <= core
