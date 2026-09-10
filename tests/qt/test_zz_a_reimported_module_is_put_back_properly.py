"""A re-imported submodule lives in TWO places, and tests restore one.

THE DEFECT, MEASURED 2026-09-10. A test that wants to exercise an import
guard -- "what happens when numba is missing", "what happens when the
optional dependency is not installed" -- drops the module from
``sys.modules`` and imports it again with the dependency refused. Putting
``sys.modules`` back afterwards looks like a complete cleanup and is not:
``importlib.import_module`` ALSO rebinds the module as an ATTRIBUTE of its
parent package, and nothing restores that.

The two then disagree, which is worse than either being wrong alone,
because they are reached by different syntax:

    from .settings_model import build_sections        -> the ATTRIBUTE
    monkeypatch.setattr("spacr.qt.screens.settings_model.build_sections",
                        spy)                          -> sys.modules

A spy installed through one is invisible to code that reaches the other,
and it fails SILENTLY: the code under test simply runs the unpatched
original and the assertion about the spy is what goes red, somewhere else,
in a file that did nothing wrong.

FOUND TWICE, BOTH TIMES BY A BASELINE RATHER THAN BY READING.
``test_cov_r8_fractal_travel_tails.py`` left ``fractal_travel`` split and
three tests in ``test_the_spaceout_fractal.py`` failed in combination while
passing alone. ``test_cov_r8_settings_model_import_guards.py`` left
``settings_model`` split -- and its own fixture docstring already records an
earlier casualty of the same family,
``test_a_module_screen_is_built_once_not_twice``, which is how thoroughly
this hides.

SO THERE ARE TWO TESTS HERE AND THEY CATCH IT DIFFERENTLY. The first reads
the source and does not depend on what ran before it. The second reads the
live interpreter and only fires when it happens to run after an offender --
which is exactly the ordering that makes the defect expensive, so it is
worth having even though it cannot be relied on alone.
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

_TESTS = pathlib.Path(__file__).resolve().parents[1]

_DROPS = re.compile(r"(?:monkeypatch\.delitem\(\s*sys\.modules"
                    r"|del\s+sys\.modules\[)")

#: ONLY `import_module` REBINDS THE ATTRIBUTE, and the two near neighbours
#: that look identical do not -- which is worth stating, because a detector
#: that flagged them would be ignored within a week.
#:
#:   * `importlib.reload(m)` re-executes IN PLACE. The module OBJECT is the
#:     same one, so every reference to it stays correct and the package
#:     attribute still points at it. (It has its own hazard -- every name
#:     the module defines is rebound, so an exception class imported
#:     elsewhere stops matching -- and `test_anndata_export.py` already
#:     guards that by preserving `__dict__`. Different bug, handled.)
#:   * `runpy.run_module(name, run_name="__main__")` executes the source in
#:     a throwaway namespace and never touches the package at all, which is
#:     what `test_spaceout_is_the_only_way_in.py` does after dropping
#:     `spacr.qt.spaceout`.
_REIMPORTS = re.compile(r"importlib\.import_module\(")
#: What "it puts the attribute back" looks like, in either style.
#:
#: MATCHED ON BEHAVIOUR, NOT ON A VARIABLE NAME, and the first version was
#: not: it looked for `setattr(package|parent|pkg, ...)` and therefore did
#: not recognise `monkeypatch.setattr(_widgets_package, "fractal_travel",
#: getattr(...))`, which is the same repair written the other way. A
#: detector that calls a fixed file broken is one somebody turns off.
#:
#: So: any `setattr` whose target is a module-ish name and whose attribute
#: is a quoted string, plus the `getattr(...)` or saved-value form that
#: makes it a RESTORE rather than an assignment.
_RESTORES = re.compile(
    r"setattr\(\s*[A-Za-z_][\w.]*\s*,\s*(?:[\"']|leaf\b)")

#: Files that drop a spacr SUBMODULE and import it again, and are known to
#: put the package attribute back. A file that starts doing this and is not
#: listed fails the first test below; a file listed here that stops doing it
#: fails it too, so the list cannot rot in either direction.
KNOWN_REIMPORTERS = {
    "qt/test_cov_r8_fractal_travel_tails.py",
    "qt/test_cov_r8_settings_model_import_guards.py",
    # ALREADY CORRECT WHEN THIS TEST WAS WRITTEN, and its `_rebind` helper
    # is the model the other two were fixed against -- including the part
    # that is easy to miss: the restore has to run UNCONDITIONALLY, because
    # by the second test in a file the two are already split and a restore
    # that only repaired what it saw agreeing at setup would decline to
    # repair exactly the case it exists for.
    "test_cov_r8_measure_probes.py",
}


def _submodule_names(text: str) -> set:
    """Dotted spacr names with a parent package, mentioned as strings."""
    quoted = re.findall(r"[\"']((?:spacr(?:\.[A-Za-z_]\w*)+))[\"']", text)
    return {name for name in quoted if name.count(".") >= 2}


def _reimporting_files():
    """Every test file that drops a spacr submodule and imports it again.

    PER FUNCTION, NOT PER FILE, and the difference is a false positive that
    would have got this test disabled. `test_spaceout_is_the_only_way_in.py`
    calls `importlib.import_module` in one test and drops
    `spacr.qt.spaceout` in a completely different one, which is harmless
    and reads as the defect if the file is searched as a single string.
    Both have to be in the same body to be the same event.
    """
    import ast

    found = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not (_DROPS.search(text) and _REIMPORTS.search(text)):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(text, node) or ""
            if not (_DROPS.search(body) and _REIMPORTS.search(body)):
                continue
            names = _submodule_names(body) or _submodule_names(text)
            if not names:
                continue
            found[str(path.relative_to(_TESTS))] = (text, sorted(names))
            break
    return found


def test_every_reimporting_test_restores_the_package_attribute():
    """The source-level check, which does not depend on ordering.

    A file that re-imports a spacr submodule must put the parent package's
    attribute back as well as ``sys.modules``. Both known offenders now do;
    a third would fail here rather than making some unrelated file red a
    week later.
    """
    files = _reimporting_files()
    assert files, "the detector found nothing at all; it has stopped working"

    missing = [name for name, (text, _names) in files.items()
               if not _RESTORES.search(text)]
    assert not missing, (
        "these tests re-import a spacr submodule and restore only "
        "sys.modules, leaving the parent package's attribute pointing at "
        "the crippled copy:\n  " + "\n  ".join(missing) +
        "\nSave `getattr(parent, leaf)` before the re-import and set it "
        "back in the fixture's finally.")

    unlisted = sorted(set(files) - KNOWN_REIMPORTERS)
    gone = sorted(KNOWN_REIMPORTERS - set(files))
    assert not unlisted and not gone, (
        f"KNOWN_REIMPORTERS has drifted: new {unlisted}, stale {gone}. "
        "Update it in the same change, so what re-imports a module stays "
        "written down.")


def test_no_spacr_submodule_is_split_from_its_package():
    """The live check, on whatever this process has imported.

    It can only fire when it runs after an offender, which is why the
    source check above exists as well. When it does fire it names the
    module, which the source check cannot.
    """
    split = []
    for dotted, module in sorted(sys.modules.items()):
        if not dotted.startswith("spacr.") or dotted.count(".") < 2:
            continue
        if module is None:
            continue
        parent_name, _, leaf = dotted.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None or not hasattr(parent, leaf):
            continue
        if getattr(parent, leaf) is not module:
            split.append(dotted)
    assert not split, (
        "these modules are a different object as a package attribute than "
        "in sys.modules, so `from .x import y` and a monkeypatch through "
        f"sys.modules reach different code: {split}")
