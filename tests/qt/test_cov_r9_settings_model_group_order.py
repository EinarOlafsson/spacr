"""The classifier screen's group order, and four guards that cannot fail.

The screen rebuilds its category headings from five tuples of group
names, each look-up guarded with ``if name in ordered``. None of the
four can fail: ``ordered`` is a literal a hundred lines above plus the
branch's own additions, and the five tuples name only entries of it.

That is worth holding rather than shrugging at, because the failure is
silent in the worst way. A tuple naming a group that does not exist does
not raise -- the guard swallows it -- and the settings under that
heading simply do not appear on the screen. The user sees a shorter
form and no error.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import settings_model as SM

pytestmark = pytest.mark.qt

#: The five tuples the rebuild walks, by the name they are bound to.
TUPLES = ("cv_groups", "ml_groups", "shared_first", "shared_last")


MODULE_PATH = inspect.getsourcefile(SM)
MODULE_TREE = ast.parse(pathlib.Path(MODULE_PATH).read_text())


def _rebuild_span():
    """The line range of the classifier screen's group rebuild."""
    source = pathlib.Path(MODULE_PATH).read_text().splitlines()
    start = next(i for i, line in enumerate(source, 1)
                 if 'rebuilt = {"Classifier": ["classifier_family"]}' in line)
    ordered = max(i for i, line in enumerate(source, 1)
                  if i < start and line.strip() == "ordered = {")
    end = next(i for i, line in enumerate(source, 1)
               if i > start and "for name in shared_last:" in line) + 6
    return ordered, end


SPAN = _rebuild_span()


def _assignment(variable):
    """The AST node assigning ``variable`` inside the rebuild block."""
    lo, hi = SPAN
    for node in ast.walk(MODULE_TREE):
        if (isinstance(node, ast.Assign) and lo <= node.lineno <= hi
                and any(getattr(t, "id", None) == variable
                        for t in node.targets)):
            return node.value
    raise AssertionError(f"{variable} is no longer assigned in the rebuild")


def _literal_names(variable):
    node = _assignment(variable)
    assert isinstance(node, (ast.Tuple, ast.List)), (
        f"{variable} is no longer a tuple of literals")
    return [e.value for e in node.elts if isinstance(e, ast.Constant)]


def _ordered_keys():
    """Every group `ordered` carries by the time the rebuild reads it.

    THREE SOURCES, and missing one is how this test first produced a
    false alarm: the dict literal, the `ordered.update({...})` that adds
    the ML-only groups, and any `ordered["..."] = ...` assignment.
    """
    lo, hi = SPAN
    node = _assignment("ordered")
    assert isinstance(node, ast.Dict), (
        "the `ordered` literal is no longer a dict literal")
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]

    for inner in ast.walk(MODULE_TREE):
        if not (lo <= getattr(inner, "lineno", -1) <= hi):
            continue
        if (isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "update"
                and getattr(inner.func.value, "id", None) == "ordered"):
            for argument in inner.args:
                if isinstance(argument, ast.Dict):
                    keys += [k.value for k in argument.keys
                             if isinstance(k, ast.Constant)]
        if isinstance(inner, ast.Assign):
            for target in inner.targets:
                if (isinstance(target, ast.Subscript)
                        and getattr(target.value, "id", None) == "ordered"
                        and isinstance(target.slice, ast.Constant)):
                    keys.append(target.slice.value)
    return keys


def _rebuild_source():
    lines = pathlib.Path(MODULE_PATH).read_text().splitlines()
    lo, hi = SPAN
    return "\n".join(lines[lo - 1:hi])


def _loop_body(variable):
    """One rebuild loop's body, up to the next loop or the end."""
    source = _rebuild_source()
    start = source.index(f"for name in {variable}:")
    rest = source[start + 1:]
    nxt = rest.find("for name in ")
    return rest[:nxt] if nxt != -1 else rest


@pytest.mark.parametrize("variable", TUPLES)
def test_every_named_group_exists(variable):
    """THE PIN, for all four ``if name in ordered`` guards.

    A name that is not a key is silently dropped and its settings never
    reach the screen. Checked against the literal itself rather than a
    copy, so a heading renamed in one place and not the other fails
    here instead of on someone's screen.
    """
    keys = set(_ordered_keys())
    missing = [n for n in _literal_names(variable) if n not in keys]

    assert missing == [], (
        f"{variable} names {missing}, which `ordered` does not carry -- "
        f"those settings are dropped from the screen with no error")


def test_the_four_tuples_between_them_name_every_group():
    """The other half, and the reason the catch-all above them was
    removed: a group in `ordered` that no tuple names would never be
    rebuilt, so it would vanish just as quietly."""
    keys = set(_ordered_keys())
    named = {n for variable in TUPLES
             for n in _literal_names(variable)}
    # The two the branch adds itself, which are handled outside the loops.
    named |= {"Classifier"}

    unnamed = sorted(keys - named)

    assert unnamed == [], (
        f"{unnamed} are in `ordered` and named by none of the four tuples, "
        f"so they are dropped from the rebuilt order")


def test_evaluation_comes_last_and_is_not_prefixed():
    """Stated in the source as a decision, and worth a test: evaluation
    applies to both families, so prefixing it onto one would be a lie
    about who it belongs to, and putting it first would bury the
    settings that decide what is being trained."""
    source = _rebuild_source()

    assert _literal_names("shared_last") == ["Evaluation & Results"]

    last = source.index("for name in shared_last:")
    for earlier in ("for name in shared_first:", "for name in cv_groups:",
                    "for name in ml_groups:"):
        assert source.index(earlier) < last

    assert "_family_heading" not in _loop_body("shared_last"), (
        "the shared evaluation group is being prefixed with a family name")


def test_the_family_groups_are_prefixed_and_the_shared_ones_are_not():
    for variable, prefixed in (("cv_groups", True), ("ml_groups", True),
                               ("shared_first", False),
                               ("shared_last", False)):
        assert ("_family_heading" in _loop_body(variable)) is prefixed, (
            f"{variable} is {'not ' if prefixed else ''}being prefixed with "
            f"its family heading, which changes what the user reads")


def test_the_classifier_family_heading_leads():
    """The one group that is not conditional: it decides which family the
    rest of the form belongs to, so it cannot come after them."""
    source = _rebuild_source()

    assert 'rebuilt = {"Classifier": ["classifier_family"]}' in source
    assert source.index('rebuilt = {"Classifier"') < \
        source.index("for name in shared_first:")
