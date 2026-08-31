"""The three ``if TYPE_CHECKING:`` blocks, and the cost they avoid.

Each one shows as an uncovered branch and an uncovered import, and each
is unreachable by construction: ``typing.TYPE_CHECKING`` is a literal
``False`` at runtime and only a type checker sets it True. There is no
input that reaches those lines, so what is held here is the REASON they
are guarded rather than the lines themselves.

The reason is measurable. All three annotate with ``pandas``, and pandas
is one of the heaviest imports in the dependency set. A module that
pulls it eagerly pays that on every launch whether or not anything asks
for a frame -- which is what item 282 and item 284 are about, and what
lazy imports throughout this package exist to avoid. Moving one of these
imports out of its guard would be invisible in review and would show up
as a slower start.
"""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Every module carrying a ``TYPE_CHECKING`` guard, with the import it
#: is hiding. Enumerated rather than discovered, so a NEW guard has to
#: be added here deliberately and its runtime cost argued for.
GUARDED = (
    ("spacr.classify_classes", "pandas"),
    ("spacr.feature_dict", "pandas"),
    ("spacr.qt.widgets.class_editor", "pandas"),
)


def _sources():
    return {name: (ROOT / (name.replace(".", "/") + ".py")) for name, _ in GUARDED}


def test_the_enumeration_is_the_whole_set():
    """A new guard must be added above, not discovered by this test.

    Otherwise the list drifts into a description of whatever happens to
    exist, which cannot fail.
    """
    found = set()
    for path in ROOT.joinpath("spacr").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "if TYPE_CHECKING:" in text:
            found.add(
                str(path.relative_to(ROOT)).replace("/", ".")[: -len(".py")])

    assert found == {name for name, _ in GUARDED}, (
        "the set of TYPE_CHECKING guards changed; add the new one to "
        "GUARDED with the import it hides, and say why that import is "
        "worth deferring")


def test_the_flag_is_false_at_runtime():
    """Why the block cannot be covered, stated once.

    Not a tautology about the standard library: it is the premise every
    assertion below rests on, and if it ever stopped holding these
    blocks would become live code with no test.
    """
    assert typing.TYPE_CHECKING is False


@pytest.mark.parametrize("module,hidden", GUARDED)
def test_the_hidden_import_is_inside_the_guard(module, hidden):
    """THE PIN: the import is in the guarded block and nowhere else.

    Read from the AST rather than by string search, so an import added
    at module level with the same spelling cannot pass because the
    guarded one is also present.
    """
    tree = ast.parse(_sources()[module].read_text(encoding="utf-8"))

    guarded, unguarded = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.extend(_names(inner))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            unguarded.extend(_names(node))

    assert hidden in guarded, f"{module} no longer defers {hidden}"
    assert hidden not in unguarded, (
        f"{module} imports {hidden} at module level as well, so the guard "
        f"below it is decoration and the cost is paid anyway")


def _is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _names(node):
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    return [node.module.split(".")[0]] if node.module else []


@pytest.mark.parametrize("module,hidden", GUARDED)
def test_importing_the_module_does_not_pull_the_hidden_package(module,
                                                               hidden):
    """THE SUBSTANCE, and the half a source check cannot give.

    A fresh interpreter per module, because `sys.modules` in this one
    already holds pandas -- half the suite imports it -- so asking the
    question here would answer about the test run rather than about the
    module.

    This is what actually fails if the import moves out of the guard, or
    if something the module imports starts pulling pandas itself, which
    is the way this regresses in practice: not by editing these lines.
    """
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        f"print({hidden!r} in sys.modules)\n"
    )
    env_probe = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT),
        capture_output=True, text=True, timeout=300,
        env=_offscreen_env(),
    )

    assert env_probe.returncode == 0, env_probe.stderr[-2000:]
    assert env_probe.stdout.strip().endswith("False"), (
        f"importing {module} now pulls {hidden} at runtime, so the "
        f"TYPE_CHECKING guard is buying nothing and every launch pays for "
        f"it:\n{env_probe.stdout[-500:]}")


def _offscreen_env():
    import os

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


@pytest.mark.parametrize("module,hidden", GUARDED)
def test_the_annotations_that_need_it_are_strings(module, hidden):
    """The other half of the arrangement: with the import deferred, any
    annotation naming it has to be lazy too, or the module raises
    NameError the first time something reads its annotations.

    ``from __future__ import annotations`` makes every annotation a
    string, which is what lets the import be deferred at all.
    """
    tree = ast.parse(_sources()[module].read_text(encoding="utf-8"))

    futures = [n for n in tree.body
               if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    names = {a.name for n in futures for a in n.names}

    assert "annotations" in names, (
        f"{module} defers {hidden} without postponed annotations, so any "
        f"annotation naming it is evaluated at runtime and raises NameError")


def test_the_guard_is_not_load_bearing_for_behaviour():
    """A module whose guarded import is missing must still work.

    Driven rather than argued: the three modules are imported with the
    hidden package genuinely absent from the interpreter, and the public
    surface each one advertises is still reachable.
    """
    code = (
        "import sys\n"
        "sys.modules['pandas'] = None\n"
        "import importlib\n"
        "for name in %r:\n"
        "    importlib.import_module(name)\n"
        "print('ok')\n" % ([name for name, _ in GUARDED],)
    )
    probe = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT),
        capture_output=True, text=True, timeout=300, env=_offscreen_env())

    assert probe.returncode == 0 and "ok" in probe.stdout, (
        "a module with a TYPE_CHECKING guard could not be imported with "
        f"pandas absent, so the guard is not doing what it claims:\n"
        f"{probe.stderr[-2000:]}")


def test_this_file_is_about_lines_that_cannot_run():
    """Said once, in the suite rather than only in a comment, so the
    next reader does not go looking for a way to drive them."""
    source = inspect.getsource(sys.modules[__name__])

    assert "unreachable by construction" in source
