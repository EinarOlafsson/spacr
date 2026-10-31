"""A font size written as a literal cannot follow the interface scale.

:func:`spacr.qt.theme.font_px` states the rule in its own docstring -- "Route
every such number through here instead of writing a literal" -- because a
per-widget ``setStyleSheet`` beats the application sheet whatever the selector
says. A widget that hard-codes ``font-size: 10px`` therefore stays 10 px at
200 %, while the label beside it doubles.

Reported 2026-09-05 as "the scroll while holding z works! but it dosnt change
everything, like some text in the widgets on the right of the screen" -- which
was the gene panel, the gene tiles, the figure captions and the glass hint,
each carrying its own literal.

This is a RATCHET, not a style opinion: it fails on a new literal, and the
allowed list below is the set of places where a literal is genuinely correct.
"""
from __future__ import annotations

import ast
import pathlib
import re

#: ``font-size: <n>px`` or ``pt`` written as a literal digit, in Python source.
LITERAL = re.compile(r"font-size:\s*\d+(px|pt)")

#: Where a literal font size is CORRECT, with the reason it is exempt.
#:
#: These emit a document for somewhere else -- a browser, a print stylesheet --
#: and a document that changed size because the reader had once zoomed the
#: application that produced it would be a bug, not a feature.
ALLOWED = {
    "spacr/flowview/export.py":
        "standalone HTML export, read outside spaCR",
}


def _gui_sources():
    """Every Python file that styles a widget in the running application."""
    root = pathlib.Path(__file__).resolve().parents[2]
    for sub in ("spacr/qt", "spacr/flowview"):
        for path in (root / sub).rglob("*.py"):
            yield path, path.relative_to(root).as_posix()


def _pdf_document_style_positions(text, relative):
    """Exempt only the fixed print stylesheet argument, never widget styles.

    The Protocols export uses physical points in a standalone QTextDocument
    printed by QPdfWriter. Its typography must not inherit application zoom.
    AST argument positions keep every other literal in this module guarded.
    """
    if relative != "spacr/qt/widgets/sample_project.py":
        return set()
    positions = set()
    for owner in ast.parse(text).body:
        if not isinstance(owner, ast.ClassDef) or owner.name != "_ProtocolBrowserDialog":
            continue
        for method in owner.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "_export_recipe_pdf":
                continue
            for call in ast.walk(method):
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "document"
                        and call.func.attr == "setDefaultStyleSheet"
                        and call.args and isinstance(call.args[0], ast.Constant)
                        and isinstance(call.args[0].value, str)):
                    continue
                value = call.args[0]
                for offset, line in enumerate(ast.get_source_segment(text, value).splitlines()):
                    column = value.col_offset if offset == 0 else 0
                    positions.update((value.lineno + offset,
                                      column + len(line[:match.start()].encode("utf-8")))
                                     for match in LITERAL.finditer(line))
    return positions


def test_no_widget_hard_codes_a_font_size():
    """Every font size in the GUI goes through ``font_px``."""
    offenders = []
    for path, rel in _gui_sources():
        if rel in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        document_styles = _pdf_document_style_positions(text, rel)
        for number, line in enumerate(text.splitlines(), start=1):
            for match in LITERAL.finditer(line):
                position = number, len(line[:match.start()].encode("utf-8"))
                if position not in document_styles:
                    offenders.append(f"{rel}:{number}: {line.strip()}")
                    break

    assert not offenders, (
        "these font sizes cannot follow the interface scale, so they stay "
        "put while the text beside them grows. Route each through "
        "spacr.qt.theme.font_px, or add it to ALLOWED with its reason:\n  "
        + "\n  ".join(offenders)
    )


def test_the_allowed_list_has_not_gone_stale():
    """An exemption for a file that no longer exists hides a real offender."""
    root = pathlib.Path(__file__).resolve().parents[2]
    missing = [rel for rel in ALLOWED if not (root / rel).exists()]
    assert not missing, f"exempted files that are gone: {missing}"
