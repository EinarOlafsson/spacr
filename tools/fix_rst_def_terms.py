"""Join reStructuredText definition-list terms that wrap onto two lines.

rST requires a definition term to be ONE line. A wrapped term becomes an
ordinary paragraph, so the indented definition under it is "Unexpected
indentation" and the whole entry renders as body text instead of a
definition list.

Every candidate edit is VERIFIED before it is kept: the docstring is
re-parsed and the change is discarded unless the error count actually
drops and no new error appears. An unverified auto-fix over docstrings
corrupted 203 files earlier today.
"""
import ast, io, sys
from pathlib import Path

import docutils.core

NAPOLEON = ("Args:", "Returns:", "Yields:", "Raises:", "Attributes:")


def errors(text):
    err = io.StringIO()
    try:
        docutils.core.publish_doctree(text, settings_overrides={
            "report_level": 2, "halt_level": 5, "warning_stream": err,
            "file_insertion_enabled": False})
    except Exception:
        pass
    lines = err.getvalue().splitlines()
    return (sum("Unexpected indentation" in l for l in lines), len(lines))


def join_wrapped_terms(doc):
    """Return doc with wrapped definition terms joined, or None."""
    lines = doc.splitlines()
    out, i, changed = [], 0, 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt2 = lines[i + 2] if i + 2 < len(lines) else ""
        wrapped = (
            cur.strip() and not cur[:1].isspace()          # term line 1
            and nxt.strip() and not nxt[:1].isspace()      # term line 2
            and nxt2.strip() and nxt2[:1].isspace()        # the definition
            # A wrapped term continues a list, so it ends mid-item. Comma
            # was too narrow -- artifacts.py wraps on "/". The verification
            # gate below is what makes loosening this safe: a join that does
            # not reduce the error count is discarded.
            and cur.rstrip().endswith((",", "/", "``", "-"))
        )
        if wrapped:
            out.append(cur.rstrip() + " " + nxt.strip())
            i += 2
            changed += 1
            continue
        out.append(cur)
        i += 1
    return ("\n".join(out), changed) if changed else (None, 0)


for path in sys.argv[1:]:
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        print(f"{path}: unparseable, skipped"); continue
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or any(h in doc for h in NAPOLEON):
            continue
        before = errors(doc)
        if before[0] == 0:
            continue
        new_doc, n = join_wrapped_terms(doc)
        if not new_doc:
            continue
        after = errors(new_doc)
        # keep ONLY if indentation errors dropped and nothing new appeared
        if after[0] >= before[0] or after[1] > before[1]:
            continue
        if doc not in src:
            continue
        src = src.replace(doc, new_doc, 1)
        total += n
    if total:
        try:
            ast.parse(src)
        except SyntaxError as exc:
            print(f"{path}: REFUSED, edit would break it ({exc.lineno})")
            continue
        p.write_text(src, encoding="utf-8")
        print(f"{path}: joined {total} wrapped term(s)")
