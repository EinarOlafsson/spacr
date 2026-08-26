"""A docstring field marker has to start its own line, or it is lost.

Sphinx builds ``:param x:`` into a field list only when the marker begins a
line. Written a second time part-way through an existing field --

    :param bottom: smaller plateau. :param top: larger plateau.

-- the second marker is ordinary text inside the first parameter's
description. Two things go wrong at once, and neither is loud: ``top`` is
never documented at all, and the reader sees raw ``:param top:`` on the
published page. Nothing warns, because the line is valid reST; it just does
not mean what it looks like it means.

The same holds for ``:returns:``, ``:raises X:``, ``:ivar:`` and ``:yields:``
appended to a one-line summary.
"""

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "spacr"

#: A field marker with something other than indentation in front of it.
STRAY = re.compile(r"\S[ \t]+:(param|returns?|raises?|ivar|yields?)\b")

#: Inline literals may legitimately quote a marker while talking about one.
LITERAL = re.compile(r"``[^`]*``")


def _docstrings(path):
    """(line, docstring) for every module, class and function in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:                      # not ours to police here
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            text = ast.get_docstring(node, clean=False)
            if text:
                yield getattr(node, "lineno", 1), text


def _stray_markers():
    """Every field marker in the package that does not start its line."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for lineno, text in _docstrings(path):
            for line in text.splitlines():
                bare = LITERAL.sub(" ", line)
                if STRAY.search(bare):
                    found.append(
                        f"{path.relative_to(PACKAGE.parent)}:~{lineno}: "
                        f"{line.strip()}")
    return found


def test_no_field_marker_hides_inside_another_field():
    stray = _stray_markers()
    assert not stray, (
        "these field markers do not start their own line, so Sphinx renders "
        "them as text and the parameter they name is undocumented:\n  "
        + "\n  ".join(stray))


@pytest.mark.parametrize("text, expected", [
    (":param a: one.\n:param b: two.", 0),
    (":param a: one. :param b: two.", 1),
    ("Summary. :raises Boom: when it does.", 1),
    ("Write ``:param x:`` to document x.", 0),
    ("    :returns: the value.", 0),
])
def test_the_detector_reads_the_shapes_it_claims_to(text, expected):
    """The guard is only worth having if it separates these five cases."""
    hits = sum(1 for line in text.splitlines()
               if STRAY.search(LITERAL.sub(" ", line)))
    assert hits == expected
