"""A NumPy-style parameter type must not carry an array shape.

The API reference is where every in-app tooltip link lands, and its build is
held at zero warnings so a real fault is visible the day it arrives. One
docstring form quietly costs that, and it looks entirely reasonable:

    contributions : array-like, shape (n_samples, n_features)
        Each sample's contribution for each feature.

Napoleon turns everything after the colon into a ``:type:`` field, and Sphinx
resolves each token of a type field as a Python object. ``shape`` is an
attribute on many classes, so the reference is ambiguous and the build reports
"more than one target found for cross-reference 'shape'".

The shape belongs in the description, where it is text rather than a type:

    contributions : array-like
        Each sample's contribution for each feature, shaped
        ``(n_samples, n_features)``.

This is worth a rule rather than a re-read because the warning is close to
undebuggable from the log: Sphinx reports it against the document's TITLE
line, not the field, because the reference is resolved after the field's own
source position is gone. The line number sends a reader to the top of a
module page that has nothing wrong with it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "spacr"

#: NumPy-style sections whose entries Napoleon converts into ``:type:`` fields.
_SECTION = re.compile(
    r"^\s*(Parameters|Other Parameters|Returns|Yields|Attributes|Raises)\s*$"
)
_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
#: ``name : type`` -- the entry line, whose right-hand side becomes the type.
_FIELD = re.compile(r"^\s*[\w*][\w*, ]*\s:\s(?P<type>.+?)\s*$")
#: A size descriptor has no business in a type: every word of one is resolved.
_SIZE_IN_TYPE = re.compile(r"\bshape\b|\bof length\b")


def _type_fields(docstring: str):
    """Yield the type half of every NumPy-style entry in one docstring."""
    lines = docstring.splitlines()
    inside = False
    underline = False
    for index, line in enumerate(lines):
        if underline:
            # The dashes under a heading are not an entry, and reading them as
            # one ends the section before its first parameter is seen.
            underline = False
            continue
        if (
            _SECTION.match(line)
            and index + 1 < len(lines)
            and _UNDERLINE.match(lines[index + 1])
        ):
            inside = True
            underline = True
            continue
        if not inside:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if line[:1].isspace():
            # Indented: the description under an entry, where a shape is
            # ordinary prose and exactly where this rule wants it.
            continue
        entry = _FIELD.match(line)
        if entry is None:
            inside = False
            continue
        yield stripped, entry.group("type")


def _label(path: Path) -> str:
    """Name a file the way a reader will look for it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _offenders(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        docstring = ast.get_docstring(node, clean=True)
        if not docstring:
            continue
        for entry, declared in _type_fields(docstring):
            if _SIZE_IN_TYPE.search(declared):
                yield f"{_label(path)}: {entry}"


def test_no_documented_type_names_a_shape():
    """One such field is one warning, and the log will not say where it is."""
    found = sorted(
        offender
        for path in sorted(PACKAGE.rglob("*.py"))
        for offender in _offenders(path)
    )
    assert not found, (
        "a NumPy-style type field carries a size descriptor, which Sphinx "
        "resolves as a Python object and reports against the wrong line. Move "
        "the shape into the description as inline literal text:\n"
        + "\n".join(found)
    )


def test_the_rule_recognises_the_form_that_costs_a_warning(tmp_path):
    """The detector has to keep its teeth, and both halves are load-bearing.

    A rule that reads only the first section, or that stops at the first
    indented continuation line, passes the offending file and reports nothing
    -- which is indistinguishable from a clean tree.
    """
    module = tmp_path / "offender.py"
    module.write_text(
        '"""A module with two NumPy sections."""\n'
        "\n"
        "\n"
        "def draw(values, contributions):\n"
        '    """Draw a beeswarm.\n'
        "\n"
        "    Parameters\n"
        "    ----------\n"
        "    values : array-like\n"
        "        The raw measurements, already ordered.\n"
        "    contributions : array-like, shape (n_samples, n_features)\n"
        "        Each sample's contribution for each feature.\n"
        "\n"
        "    Returns\n"
        "    -------\n"
        "    handle : object, shape (n_features,)\n"
        "        The drawn artist.\n"
        '    """\n',
        encoding="utf-8",
    )
    found = list(_offenders(module))
    assert len(found) == 2, found
    assert "contributions : array-like, shape (n_samples, n_features)" in found[0]
    assert "handle : object, shape (n_features,)" in found[1]


def test_the_rule_leaves_the_repaired_form_alone():
    """The description is where a shape belongs, and must not be flagged.

    The word is the same one the rule bans; only the indentation differs, and
    that is the whole distinction Napoleon makes between a type and prose.
    """
    repaired = (
        "Draw a beeswarm.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "contributions : array-like\n"
        "    Each sample's contribution for each feature, of\n"
        "    shape ``(n_samples, n_features)``.\n"
    )
    flagged = [
        declared
        for _entry, declared in _type_fields(repaired)
        if _SIZE_IN_TYPE.search(declared)
    ]
    assert not flagged
