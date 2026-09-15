"""Inline markup in a docstring renders as markup, not as its own punctuation.

reST does not nest inline markup and does not open or close it just anywhere,
and a docstring that asks for either publishes the markers themselves. Both
shapes were on the API pages, counted by an AST walk on 2026-08-26 (item 63):

    nested-in-bold     a ``literal`` or :role:`x` inside a **strong** (or
                       *emphasis*) span. The strong span wins and its content
                       is plain text, so
                           **Never ``SELECT *`` the whole table.**
                       renders as  Never ``SELECT *`` the whole table.
    bad-literal-edge   a literal glued to a character reST will not open or
                       close on -- ``plate1``..``plate4``, ``lru_cache``s,
                       ``factor``× -- or padded with whitespace inside its own
                       markers. The markers render, or one literal swallows
                       the text up to the next one.

HOW IT DECIDES, and why it is docutils rather than a regex: each docstring is
parsed by docutils itself, so the start-string and end-string punctuation
rules, ``::`` literal blocks and ``>>>`` doctest blocks are exactly what the
documentation build applies. Unknown roles (Sphinx's ``:class:``, ``:func:``
and the ``py:obj`` default role) are read as literals, which is how they
render. A backtick left in rendered text outside a literal is a fault; so is
one inside a strong or emphasis span, and so is a literal whose own content
holds a second pair of backticks.

It walks every module, class and function docstring under ``spacr/``, private
ones included: a private docstring is read by whoever opens the file, and it
becomes public the day its function does.
"""
from __future__ import annotations

import ast
import io
import pathlib
from dataclasses import dataclass

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Files whose docstrings were being edited elsewhere when the sweep ran
#: (2026-09-14), so their faults were left for that work to fix. They are
#: named here so the count below says where its remainder lives.
EXCLUDED_FILES = (
    "spacr/object.py",
    "spacr/settings.py",
    "spacr/qt/theme.py",
    "spacr/qt/widgets/measure_preview.py",
    "spacr/qt/screens/app_screen.py",
    "spacr/qt/dnd_handlers.py",
    "spacr/qt/widgets/section.py",
)

#: Faults outside the excluded files that no markup-only edit can repair,
#: with the reason. A docstring is only ever changed here in its markup.
#:
#: Empty since 2026-09-15. It held curation_queue's `_normalise_state`,
#: whose padded status value was written as a literal with whitespace at its
#: edges, which reST cannot render. Quoting the value inside the literal
#: keeps the padding visible and renders.
LEFT_AS_IS: set = set()

#: Measured 2026-09-14 after the sweep: 1 fault in the excluded files
#: (measure_preview.py) and 1 then named in LEFT_AS_IS. 2 -> 1 on 2026-09-15
#: when that one was repaired. It may fall; it may not rise.
REMAINING_FAULTS = 1


@dataclass(frozen=True)
class Fault:
    path: str
    line: int
    symbol: str
    kind: str
    text: str


def _parser():
    from docutils import nodes, utils
    from docutils.frontend import get_default_settings
    from docutils.parsers.rst import Parser, roles

    def generic(name, rawtext, text, lineno, inliner, options=None,
                content=None):
        return [nodes.literal(rawtext, utils.unescape(text))], []

    original = roles.role

    def role(name, language_module, lineno, reporter):
        found, _messages = original(name, language_module, lineno, reporter)
        return (found or generic), []

    parser = Parser()
    settings = get_default_settings(Parser)
    settings.report_level = 5
    settings.halt_level = 5
    settings.warning_stream = io.StringIO()
    return parser, settings, role, roles


def _faults_in(text: str, parse) -> list[tuple[int, str, str]]:
    from docutils import nodes, utils

    parser, settings, role, roles = parse
    document = utils.new_document("<docstring>", settings)
    original = roles.role
    roles.role = role
    try:
        parser.parse(text, document)
    finally:
        roles.role = original

    skipped = (nodes.literal_block, nodes.doctest_block, nodes.system_message,
               nodes.comment, nodes.raw)
    found = []
    seen = set()
    for node in document.findall(nodes.Text):
        inline = None
        ancestor = node.parent
        line = None
        skip = False
        while ancestor is not None:
            if isinstance(ancestor, skipped):
                skip = True
                break
            if inline is None and isinstance(
                    ancestor, (nodes.strong, nodes.emphasis, nodes.literal)):
                inline = ancestor
            if line is None and getattr(ancestor, "line", None):
                line = ancestor.line
            ancestor = ancestor.parent
        if skip:
            continue
        value = str(node)
        if isinstance(inline, (nodes.strong, nodes.emphasis)):
            if "`" in value and id(inline) not in seen:
                seen.add(id(inline))
                found.append((line or 1, "nested-in-bold", inline.astext()))
        elif isinstance(inline, nodes.literal):
            if "``" in value and id(inline) not in seen:
                seen.add(id(inline))
                found.append((line or 1, "bad-literal-edge", inline.astext()))
        elif "`" in value:
            found.append((line or 1, "bad-literal-edge", value))
    return found


def docstring_markup_faults(root: pathlib.Path = ROOT) -> list[Fault]:
    """Every inline-markup fault in every docstring under ``root/spacr``."""
    parse = _parser()
    faults = []
    for path in sorted((root / "spacr").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        pending = [(tree, "")]
        while pending:
            node, symbol = pending.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef,
                                      ast.AsyncFunctionDef)):
                    pending.append(
                        (child, f"{symbol}.{child.name}" if symbol
                         else child.name))
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if not doc or "`" not in doc:
                continue
            start = node.body[0].lineno
            for line, kind, text in _faults_in(doc, parse):
                faults.append(Fault(relative, start + line - 1, symbol, kind,
                                    " ".join(text.split())[:80]))
    return sorted(faults, key=lambda f: (f.path, f.line, f.kind))


@pytest.fixture(scope="module")
def faults():
    pytest.importorskip("docutils")
    return docstring_markup_faults()


def test_the_detector_sees_both_shapes_and_skips_code():
    """Guards the guard: a detector that finds nothing passes the count too."""
    pytest.importorskip("docutils")
    parse = _parser()
    kinds = lambda text: [k for _line, k, _t in _faults_in(text, parse)]
    assert kinds("**Never ``SELECT *`` the whole table.**") == [
        "nested-in-bold"]
    assert kinds("**Defined in :mod:`spacr.schema`.**") == ["nested-in-bold"]
    assert kinds("rows ``plate1``..``plate4`` here") == ["bad-literal-edge"]
    assert kinds("the icon ``lru_cache``s, and ``more`` text") == [
        "bad-literal-edge"]
    # The repaired spellings are clean.
    assert kinds("**Never** ``SELECT *`` the whole table.") == []
    assert kinds("rows ``plate1..plate4`` here") == []
    assert kinds("the icon ``lru_cache``\\ s, and ``more`` text") == []
    # Code is not prose.
    assert kinds("Example::\n\n    **x ``y``** ``a``s\n") == []
    assert kinds(">>> f('``a``s')\n'**``b``**'\n") == []


def test_no_fault_outside_the_named_files(faults):
    """A new fault anywhere else fails here, naming the docstring."""
    stray = [f for f in faults
             if f.path not in EXCLUDED_FILES
             and (f.path, f.symbol) not in LEFT_AS_IS]
    assert not stray, "\n".join(
        f"{f.path}:{f.line} {f.symbol or '<module>'} [{f.kind}] {f.text}"
        for f in stray)


def test_the_remaining_count_does_not_rise(faults):
    """The excluded files and the named leftovers hold the rest."""
    assert len(faults) <= REMAINING_FAULTS, "\n".join(
        f"{f.path}:{f.line} {f.symbol or '<module>'} [{f.kind}] {f.text}"
        for f in faults)
