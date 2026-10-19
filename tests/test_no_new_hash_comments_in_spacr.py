"""No ``#`` comment may land in ``spacr/``; item 400 keeps the package bare.

``tools/strip_comments.py`` removes every comment that is not protected
(shebang, coding line, ``#:`` Sphinx attribute docs, tool directives such
as ``noqa`` and ``type:``, a licence header). This asks it, over the real
package, whether anything removable has crept back in. The fix for a
failure is to move the note into the docstring or the feature file, or to
run ``python tools/strip_comments.py spacr --apply``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "strip_comments_for_the_package_guard",
        ROOT / "tools" / "strip_comments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_removable_hash_comment_is_left_in_the_package():
    tool = _tool()
    offenders = []
    for path in tool.iter_py([str(ROOT / "spacr")]):
        _text, removed, _kept = tool.strip_text(
            path.read_text(encoding="utf-8"))
        if removed:
            offenders.append(f"{path.relative_to(ROOT)}: {removed}")
    assert not offenders, "new # comments in spacr/:\n  " + "\n  ".join(
        offenders[:20])
