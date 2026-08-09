"""Lengthen short rST underlines, INSIDE docstrings only.

The unsafe version of this walked raw lines. `"` is a legal rST underline
character, so a docstring's closing triple-quote matched the underline
pattern and was "lengthened" -- corrupting 203 files. This one asks `ast`
where each docstring starts and ends and never touches a line outside that
span, so the quotes cannot be mistaken for content.

Every file is re-parsed before it is written back. A file that would not
compile is left alone and reported.
"""
import ast, re, sys
from pathlib import Path

UNDERLINE = re.compile(r'^([-=~^#*+`_]){3,}\s*$')   # NOTE: no quote chars


def docstring_spans(tree):
    """(start, end) 1-based line numbers of every docstring body."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        # Strictly INSIDE the quotes: never the opening or closing line.
        spans.append((first.lineno + 1, first.end_lineno - 1))
    return spans


def fix(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0, "unparseable before edit"
    lines = text.splitlines(keepends=True)
    inside = set()
    for start, end in docstring_spans(tree):
        inside.update(range(start, end + 1))

    changed = 0
    for i in range(1, len(lines)):
        lineno = i + 1
        if lineno not in inside or lineno - 1 not in inside:
            continue
        under = lines[i].rstrip("\n")
        title = lines[i - 1].rstrip("\n")
        if not UNDERLINE.match(under) or not title.strip():
            continue
        t_ind = len(title) - len(title.lstrip())
        u_ind = len(under) - len(under.lstrip())
        if t_ind != u_ind or len(under.rstrip()) >= len(title.rstrip()):
            continue
        char = under.lstrip()[0]
        lines[i] = " " * u_ind + char * (len(title.rstrip()) - t_ind) + "\n"
        changed += 1

    if not changed:
        return 0, "no change"
    new = "".join(lines)
    try:
        ast.parse(new)                      # gate the write, not audit it
    except SyntaxError as exc:
        return 0, f"REFUSED: edit would break the file ({exc.lineno})"
    path.write_text(new, encoding="utf-8")
    return changed, "ok"


for arg in sys.argv[1:]:
    n, why = fix(Path(arg))
    print(f"{arg}: {n} underline(s) {why}")
