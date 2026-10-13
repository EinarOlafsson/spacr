"""tools/strip_comments.py removes ordinary comments and nothing else."""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import strip_comments as sc  # noqa: E402


def _strip(src: str) -> str:
    src = textwrap.dedent(src)
    out, _, _ = sc.strip_text(src)
    assert ast.dump(ast.parse(src)) == ast.dump(ast.parse(out))
    return out


def test_hashes_inside_strings_and_fstrings_survive():
    src = '''\
    a = "# not a comment"
    b = f"{a} #{1 + 1} # still string"
    c = 'http://x.org/#frag'
    d = """
    # docstring hash
    """
    '''
    assert _strip(src) == textwrap.dedent(src)


def test_trailing_comment_is_cut_with_its_whitespace():
    assert _strip("x = 1   # why\ny = '#'  # also\n") == "x = 1\ny = '#'\n"


def test_comment_only_lines_go_and_blank_lines_stay():
    src = """\
    import os
    # heading


    # one
    # two
    def f():
        # inner
        if True:
            # deeper
            return [
                1,  # item
                # between
                2,
            ]
    """
    assert _strip(src) == textwrap.dedent("""\
    import os


    def f():
        if True:
            return [
                1,
                2,
            ]
    """)


@pytest.mark.parametrize("line", [
    "import os  # noqa: F401",
    "x = []  # type: list",
    "if x: pass  # pragma: no cover",
    "y = 1  # fmt: skip",
    "import z  # isort: skip",
    "q = eval('1')  # nosec",
    "A = 1  #: Sphinx attribute doc",
])
def test_directives_are_kept(line):
    src = f"{line}\nb = 2  # plain\n"
    assert _strip(src) == f"{line}\nb = 2\n"


def test_shebang_encoding_and_licence_header_kept():
    src = ("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"
           "# Copyright 2026 spaCR authors\n# a plain header note\nx = 1\n"
           "# Copyright mention after code is ordinary\n")
    assert _strip(src) == ("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"
                           "# Copyright 2026 spaCR authors\nx = 1\n")


def test_crlf_endings_preserved():
    assert _strip("x = 1  # c\r\n# only\r\ny = 2\r\n") == "x = 1\r\ny = 2\r\n"


def test_check_and_apply(tmp_path, capsys):
    f = tmp_path / "m.py"
    f.write_text("x = 1  # c\n# d\n")
    assert sc.main([str(tmp_path), "--check"]) == 1
    assert f.read_text() == "x = 1  # c\n# d\n"
    assert sc.main([str(tmp_path), "--apply"]) == 0
    assert f.read_text() == "x = 1\n"
    assert sc.main([str(tmp_path), "--check"]) == 0
    assert "0 comments removable in 0 files" in capsys.readouterr().out


def test_the_package_carries_no_ordinary_comment():
    """Every reason belongs in a docstring; spacr/ keeps only tool directives."""
    root = Path(__file__).resolve().parents[1]
    left = [f"{r.path.relative_to(root)}: {r.removed}"
            for r in map(sc.process, sc.iter_py([str(root / "spacr")]))
            if r.removed]
    assert left == [], "ordinary # comments remain in spacr/:\n" + "\n".join(left)
