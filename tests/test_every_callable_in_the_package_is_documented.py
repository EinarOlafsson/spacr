"""Every class, function and method in ``spacr`` carries a docstring.

THE ZERO WAS REACHED ONCE AND NOTHING HELD IT. The docstring campaign
measured zero undocumented callables package-wide on 2026-09-09, wrote that
down, and closed. By 2026-09-14 there were eight again, all of them in code
written after that date::

    spacr/ops_stitch.py     _measure_residuals   2026-09-09
    spacr/embeddings.py     __post_init__        2026-09-10
    spacr/ops_store.py      _parquet_rows        2026-09-10
    spacr/ops_store.py      __bool__             2026-09-10
    spacr/point_patterns.py _as_radii            2026-09-10
    spacr/qt/theme.py       eventFilter          2026-09-11
    spacr/barcode_search.py __init__ (x2)        2026-09-12

NEITHER EXISTING RATCHET COULD HAVE CAUGHT ONE OF THEM, which is the whole
reason this file exists rather than a line added to one of those:

* ``test_nested_functions_are_documented`` counts only functions defined
  INSIDE another function. Not one of the eight was nested.
* ``test_completed_modules_keep_every_function_documented`` counts every
  callable, but only in the modules its registry names, and a module that
  did not exist when the registry was last extended is not in it. Five of
  the six modules above are in neither list.

A registry of finished modules cannot notice a module that is not in it, so
"finished" has to be the default and the exception has to be written down.
There is no exception list here, deliberately: the corpus is every
``spacr/**/*.py`` with nothing excluded, and the coverage test below makes
the count add up to the corpus so a silently skipped file cannot read as a
clean scan.

The two existing ratchets are left alone. They fail with a per-module
message that is more useful while a module is being worked through, and
they cost nothing now that this one holds the whole package.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "spacr"


def _scan(package: Path = PACKAGE) -> "tuple[list[str], int, list[str], int]":
    """Every undocumented callable in the package, and what was scanned.

    :returns: the undocumented callables as ``path:line kind name``, the
        number of callables examined, the files that could not be parsed,
        and the number of files parsed. The last three are what makes the
        first trustworthy: a scan that quietly skipped a file would
        otherwise report the same clean answer as one that read it.
    """
    missing: "list[str]" = []
    unparsable: "list[str]" = []
    callables = 0
    parsed = 0
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            unparsable.append(relative)
            continue
        parsed += 1
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            callables += 1
            if not ast.get_docstring(node):
                kind = ("class" if isinstance(node, ast.ClassDef)
                        else "function")
                missing.append(
                    f"{relative}:{node.lineno} {kind} {node.name}")
    return missing, callables, unparsable, parsed


@pytest.fixture(scope="module")
def scan():
    """One walk of the package, shared by the checks that read it."""
    return _scan()


def test_no_callable_in_the_package_is_undocumented(scan):
    """The ratchet. It is at zero, and zero is the only value it may have.

    ``ast.walk`` reaches every nesting depth, so a method on a class inside
    a function counts exactly like a module-level function. A docstring
    here is not a formality: the eight that went missing between 2026-09-09
    and 2026-09-14 included a residual measurement whose empty result means
    something specific, a cache row count that reads a file footer rather
    than the file, and a Qt event filter whose return value decides whether
    a window is ever shown.
    """
    missing, callables, _unparsable, _parsed = scan
    assert not missing, (
        f"{len(missing)} of {callables} callables have no docstring:\n  "
        + "\n  ".join(missing[:20])
        + ("\n  ..." if len(missing) > 20 else ""))


def test_the_scan_reaches_every_file_in_the_package(scan):
    """A measurement that does not add up to its corpus is not a measurement.

    A file that fails to parse is skipped by the scan above, and a skipped
    file reports as no debt at all. The parts have to sum to the whole and
    the exclusions have to be named -- here there are none.
    """
    _missing, callables, unparsable, parsed = scan
    total = len(list(PACKAGE.rglob("*.py")))
    assert parsed + len(unparsable) == total, (
        f"{parsed} parsed and {len(unparsable)} unparsable do not sum to "
        f"the {total} python files under {PACKAGE.name}")
    assert not unparsable, f"unparsable files were skipped: {unparsable}"
    assert callables > 15_000, (
        f"only {callables} callables were found in {parsed} files, which "
        "means the scan stopped reaching the package rather than that the "
        "package shrank")


def test_the_scanner_names_a_missing_docstring_at_every_depth(tmp_path):
    """Proof the scan can fail, at each depth it claims to cover.

    A gate whose scanner cannot see a class member, or cannot see inside a
    function body, passes for the same reason an empty list does.
    """
    source = tmp_path / "specimen.py"
    source.write_text(
        '''"""Module."""


def documented():
    """Documented."""


def bare_function():
    pass


class BareClass:
    def bare_method(self):
        pass


class DocumentedClass:
    """Documented."""

    def documented_method(self):
        """Documented."""

        def bare_nested():
            pass
''', encoding="utf-8")
    missing, callables, unparsable, parsed = _scan(tmp_path)
    assert (parsed, unparsable) == (1, [])
    assert callables == 7, callables
    assert [entry.split(" ", 1)[1] for entry in missing] == [
        "function bare_function",
        "class BareClass",
        "function bare_method",
        "function bare_nested",
    ], missing
    assert all(entry.startswith(f"{tmp_path.name}/specimen.py:")
               for entry in missing), missing
