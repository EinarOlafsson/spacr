"""A docstring that is not valid RST fails the docs build, and only there.

`docs.yml` runs `sphinx-build -W`, so ONE malformed docstring anywhere in
the package turns the whole documentation job red -- and nothing else in the
repository notices. Two did, and had since 2026-09-08:

    api/spacr/ops_layout/index.rst:28   ERROR: Unexpected indentation.
    api/spacr/ops_register/index.rst:170 ERROR: Unexpected indentation.

Both were a continuation line indented deeper than the line above it, inside
what docutils reads as a blockquote rather than a literal block. Both were
fixed by ending the lead-in with `::`, which makes the block literal and
costs nothing: RST renders a trailing `::` as a single colon.

WHY THIS TEST AND NOT THE DOCS BUILD. The docs build takes twenty minutes,
needs sphinx, furo, sphinx-design and sphinx-autoapi installed, and reports
the error against a GENERATED file whose line numbers mean nothing to
anybody reading the source. This takes a couple of seconds, needs only
docutils, and names the function.

WHAT IT DOES NOT CHECK, on purpose. Plain docutils does not know Sphinx's
roles -- `:func:`, `:class:`, `:param:` -- and flags every one of them as an
unknown role. Those are not errors and a test that reported them would be
turned off within a day. So this looks only for the STRUCTURAL failures,
which are exactly the ones a reader cannot see and `-W` makes fatal.
"""
from __future__ import annotations

import ast
import io
import pathlib
import re

import pytest

docutils = pytest.importorskip("docutils.core")
napoleon = pytest.importorskip("sphinx.ext.napoleon.docstring")
_napoleon_config = pytest.importorskip("sphinx.ext.napoleon").Config(
    napoleon_use_param=True, napoleon_use_rtype=True)

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "spacr"

#: The docutils messages that mean the MARKUP is broken rather than that a
#: Sphinx extension is missing. Each one is a real rendering failure and
#: each one is fatal under `-W`.
STRUCTURAL = re.compile(
    r"(Unexpected indentation"
    r"|Unexpected unindent"
    r"|Inconsistent literal block quoting"
    r"|Block quote ends without a blank line"
    r"|Bullet list ends without a blank line"
    r"|Enumerated list ends without a blank line"
    r"|Definition list ends without a blank line"
    r"|Field list ends without a blank line"
    r"|Option list ends without a blank line"
    r"|Malformed table"
    r"|Title underline too short"
    r"|Explicit markup ends without a blank line)")


def _is_public(dotted: str) -> bool:
    """Whether autoapi documents this name, and therefore whether -W sees it.

    A private name is not rendered, so a malformed docstring on one cannot
    fail the docs build -- and reporting it here would bury the ones that
    can. Measured: the unrestricted sweep found 60-odd, of which sphinx
    reported exactly two, and every difference was a leading underscore.
    """
    return not any(part.startswith("_") and part != "__init__"
                   for part in dotted.split("."))


def _docstrings():
    """``(path, qualified name, text)`` for every docstring under spacr/."""
    for path in sorted(_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                                  # pragma: no cover
            continue
        module = str(path.relative_to(_ROOT.parent)).replace("/", ".")[:-3]
        stack = [(tree, module)]
        while stack:
            node, name = stack.pop()
            text = ast.get_docstring(node, clean=True)
            if text:
                yield path, name, text
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef,
                                      ast.AsyncFunctionDef)):
                    stack.append((child, f"{name}.{child.name}"))


def _as_sphinx_sees_it(text: str) -> str:
    """The docstring after napoleon, which is what docutils is handed.

    NOT THE RAW DOCSTRING, and the difference is the whole accuracy of this
    test. A Google- or NumPy-style ``Args:`` block is indented prose to
    docutils and a field list to Sphinx, so parsing the raw text reports
    "Unexpected indentation" on docstrings that render perfectly. Measured:
    the raw sweep flagged four public functions that ``sphinx-build -W``
    passes, and all four came back clean once napoleon had run.
    """
    google = napoleon.GoogleDocstring(text, _napoleon_config)
    return str(napoleon.NumpyDocstring(str(google), _napoleon_config))


def _structural_errors(text: str):
    """Every structural docutils complaint about ``text``."""
    try:
        text = _as_sphinx_sees_it(text)
    except Exception:                                        # noqa: BLE001
        pass
    stream = io.StringIO()
    try:
        docutils.publish_doctree(text, settings_overrides={
            "report_level": 2, "halt_level": 5,
            "warning_stream": stream, "file_insertion_enabled": False,
            "raw_enabled": False})
    except Exception as exc:                                 # noqa: BLE001
        return [f"docutils refused it entirely: {exc}"]
    return [line for line in stream.getvalue().splitlines()
            if STRUCTURAL.search(line)]


def test_no_docstring_breaks_the_docs_build():
    """Every docstring under spacr/ parses as RST without a structural error.

    A failure here names the FUNCTION. The docs build names a generated
    file and a line number in it, twenty minutes later.
    """
    broken = []
    for path, name, text in _docstrings():
        if not _is_public(name):
            continue
        for message in _structural_errors(text):
            if "(ERROR" not in message:
                # Sphinx surfaces docutils ERRORs from a docstring and lets
                # its WARNINGs through, so a warning here is not a red docs
                # build and reporting it would bury what is.
                continue
            broken.append(f"{name}  ({path.name}): {message.strip()}")
    assert not broken, (
        f"{len(broken)} docstring(s) are not valid RST, so "
        "`sphinx-build -W` fails on them:\n  " + "\n  ".join(broken[:40]) +
        ("\n  ..." if len(broken) > 40 else "") +
        "\n\nAn indented block under a line ending in a single colon is a "
        "BLOCKQUOTE, and a continuation line indented deeper inside one is "
        "an error. End the lead-in with `::` and the block becomes literal; "
        "RST renders the doubled colon as a single one, so nothing changes "
        "for a reader.")


def test_the_detector_would_catch_the_two_that_got_through():
    """Pinned to the actual defect, so the filter cannot quietly stop working.

    A regex allowlist is exactly the kind of thing that gets narrowed until
    it matches nothing, and a green test that checks nothing is worse than
    no test.
    """
    bad = ("every tile at once:\n"
           "\n"
           "    columns   21, snaked down\n"
           "    heights   5, 9, 13,\n"
           "              19, 19, 17\n")
    assert _structural_errors(bad), "the detector no longer sees the defect"
    assert not _structural_errors(bad.replace("at once:", "at once::")), (
        "the `::` fix no longer satisfies the detector")
