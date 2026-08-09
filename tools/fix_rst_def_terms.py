"""Join reStructuredText definition-list terms that wrap onto two lines.

rST requires a definition term to be ONE line. A wrapped term becomes an
ordinary paragraph, so the indented definition under it is "Unexpected
indentation" and the entry renders as body text rather than a definition
list -- a visible defect on the API page.

Two safety properties, both learned the hard way on this codebase:

* edits are made by LINE NUMBER inside the docstring's own span, taken
  from `ast`. An earlier version replaced `ast.get_docstring()` text, which
  is the CLEANED docstring -- dedented, escapes processed -- so it matched
  the source only for module docstrings at column 0 and silently no-opped
  everywhere else.
* every change is VERIFIED: the docstring is re-parsed and the edit is
  discarded unless the "Unexpected indentation" count actually drops and no
  new error appears. A line-based auto-fix without this corrupted 203 files.
"""
import ast
import io
import sys
from pathlib import Path

import docutils.core

NAPOLEON = ("Args:", "Returns:", "Yields:", "Raises:", "Attributes:")
CONTINUES = (",", "/", "``", "-", "|")


def _errors(text):
    """(indentation errors, total messages) for one rST fragment."""
    err = io.StringIO()
    try:
        docutils.core.publish_doctree(text, settings_overrides={
            "report_level": 2, "halt_level": 5, "warning_stream": err,
            "file_insertion_enabled": False})
    except Exception:
        pass
    lines = err.getvalue().splitlines()
    return sum("Unexpected indentation" in l for l in lines), len(lines)


def _join_wrapped(lines):
    """Join wrapped definition terms in a list of source lines."""
    out, i, changed = [], 0, 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt2 = lines[i + 2] if i + 2 < len(lines) else ""
        base = len(cur) - len(cur.lstrip())
        wrapped = (
            cur.strip() and nxt.strip() and nxt2.strip()
            and len(nxt) - len(nxt.lstrip()) == base       # term continues
            and len(nxt2) - len(nxt2.lstrip()) > base      # then indents
            and cur.rstrip().endswith(CONTINUES)
        )
        if wrapped:
            out.append(cur.rstrip() + " " + nxt.strip())
            i += 2
            changed += 1
            continue
        out.append(cur)
        i += 1
    return out, changed


def fix(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    lines = src.splitlines(keepends=True)
    total = 0
    # Deepest-last, so earlier edits cannot move later line numbers.
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef))]
    spans = []
    for node in nodes:
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        doc = first.value.value
        if any(h in doc for h in NAPOLEON):
            continue
        if _errors(doc)[0] == 0:
            continue
        spans.append((first.lineno, first.end_lineno))
    for start, end in sorted(spans, reverse=True):
        body_lines = lines[start:end - 1]          # strictly inside quotes
        joined, n = _join_wrapped([l.rstrip("\n") for l in body_lines])
        if not n:
            continue
        before = _errors("".join(body_lines))
        after = _errors("\n".join(joined) + "\n")
        if after[0] >= before[0] or after[1] > before[1]:
            continue
        lines[start:end - 1] = [l + "\n" for l in joined]
        total += n
    if not total:
        return 0
    new = "".join(lines)
    try:
        ast.parse(new)
    except SyntaxError:
        return 0
    path.write_text(new, encoding="utf-8")
    return total


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        n = fix(Path(arg))
        if n:
            print(f"{arg}: joined {n} wrapped term(s)")
