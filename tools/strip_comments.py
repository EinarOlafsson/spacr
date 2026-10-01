#!/usr/bin/env python3
"""Remove ordinary ``#`` comments from Python files, leaving the code byte-identical.

Comments are found with :mod:`tokenize`, so a ``#`` inside a string, an
f-string or a docstring is never touched. A comment that stood alone on its
line takes the line with it; a comment after code is cut together with the
whitespace before it. Blank lines that were already there stay.

Kept, whatever they say:

* the shebang on line 1 and a PEP 263 encoding declaration on line 1 or 2;
* ``#:`` Sphinx attribute documentation, which renders into the API pages;
* tool directives (``noqa``, ``type:``, ``pragma``, ``fmt:``, ``isort:``,
  ``yapf:``, ``nosec``, ``pylint:``/``mypy:``/``ruff:``...), the same set
  :mod:`extract_source_notes` protects;
* a licence header: comments before the first code token that mention a
  copyright, licence or SPDX identifier.

Before a file is written its AST is compared with the original and every
kept comment is checked to survive in order; a file that fails either check
is refused, not written.

Usage::

    python tools/strip_comments.py spacr            # same as --check
    python tools/strip_comments.py spacr --check    # exit 1 if comments remain
    python tools/strip_comments.py spacr --apply
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_source_notes import CODING_RE, DIRECTIVE_PATTERNS  # noqa: E402

LICENCE_RE = re.compile(r"copyright|licen[cs]e|spdx-", re.IGNORECASE)
SKIP_DIRS = {"__pycache__", ".git", ".tox", ".venv", "build", "dist"}


@dataclass
class Result:
    path: Path
    removed: int
    kept: int
    new_text: str
    changed: bool


def _is_protected(text: str, row: int, before_code: bool) -> bool:
    if row == 1 and text.startswith("#!"):
        return True
    if row <= 2 and CODING_RE.match(text):
        return True
    if any(pat.search(text) for _, pat in DIRECTIVE_PATTERNS):
        return True
    return before_code and bool(LICENCE_RE.search(text))


def _comments(text: str) -> Iterator[Tuple[tokenize.TokenInfo, bool]]:
    before_code = True
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            yield tok, before_code
        elif tok.type not in skip:
            before_code = False


def strip_text(text: str) -> Tuple[str, int, int]:
    """Return ``(new_text, removed, kept)`` for one module's source text."""
    lines = io.StringIO(text).readlines()
    drop_line = set()
    cut_at = {}
    removed = kept = 0
    for tok, before_code in _comments(text):
        row, col = tok.start
        if _is_protected(tok.string, row, before_code):
            kept += 1
            continue
        removed += 1
        line = lines[row - 1]
        if line[:col].strip() == "":
            drop_line.add(row - 1)
        else:
            cut_at[row - 1] = col
    out: List[str] = []
    for i, line in enumerate(lines):
        if i in drop_line:
            continue
        if i in cut_at:
            body = line.rstrip("\r\n")
            ending = line[len(body):]
            line = body[:cut_at[i]].rstrip(" \t") + ending
        out.append(line)
    return "".join(out), removed, kept


def _kept_comments(text: str) -> List[str]:
    return [t.string for t, b in _comments(text) if _is_protected(t.string, t.start[0], b)]


def process(path: Path) -> Result:
    raw = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    text = raw.decode(encoding)
    new_text, removed, kept = strip_text(text)
    changed = new_text != text
    if changed:
        if ast.dump(ast.parse(text)) != ast.dump(ast.parse(new_text)):
            raise RuntimeError(f"{path}: AST changed, refusing to write")
        if _kept_comments(text) != _kept_comments(new_text):
            raise RuntimeError(f"{path}: a protected comment moved, refusing to write")
    return Result(path, removed, kept, new_text, changed)


def iter_py(paths: Sequence[str]) -> Iterator[Path]:
    for p in map(Path, paths):
        if p.is_file():
            yield p
            continue
        for f in sorted(p.rglob("*.py")):
            if not SKIP_DIRS.intersection(f.parts):
                yield f


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report only; exit 1 if comments remain (default)")
    mode.add_argument("--apply", action="store_true", help="rewrite the files")
    args = ap.parse_args(argv)
    files = removed = kept = 0
    for path in iter_py(args.paths):
        res = process(path)
        kept += res.kept
        if not res.removed:
            continue
        files += 1
        removed += res.removed
        print(f"{res.removed:6d}  {path}")
        if args.apply:
            raw = path.read_bytes()
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            path.write_bytes(res.new_text.encode(encoding))
    verb = "removed" if args.apply else "removable"
    print(f"{removed} comments {verb} in {files} files; {kept} protected comments kept")
    return 0 if (args.apply or removed == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
