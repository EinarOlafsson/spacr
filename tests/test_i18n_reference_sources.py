"""The reference-source reader isolates provably, and says which fault it hit.

Instruction 310 entry 30 records that an editable install puts ``_EditableFinder``
on ``sys.meta_path`` and can answer ``import spacr`` with a different checkout
than the one being read.  ``tools/i18n_reference_sources.py`` exists so item
306's caption review can read an older tree without that happening.

The claim worth testing is not "it runs".  It is:

  * the finder is actually removed, and a decoy checkout does not win;
  * a module that resolves outside the target tree is REPORTED rather than
    silently accepted -- an unverified isolation is the original bug;
  * a tree that cannot import itself is diagnosed as such, and NOT as the
    module mixing this runner prevents.  Those two faults have different
    fixes, and entry 36 spent an afternoon on the wrong one.

Each test builds a miniature tree rather than the real package, so the
assertions are about the runner and run in under a second.
"""

from __future__ import annotations

import json
import sys
import textwrap
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    tools = str(ROOT / "tools")
    sys.path.insert(0, tools)
    try:
        return import_module("i18n_reference_sources")
    finally:
        sys.path.remove(tools)


def _make_tree(base: Path, *, sources: dict | None = None,
               builder_body: str | None = None) -> Path:
    """A minimal tree with a ``spacr`` package and a builder that reads it."""
    tree = base / "tree"
    (tree / "spacr" / "qt").mkdir(parents=True)
    (tree / "tools").mkdir(parents=True)
    (tree / "spacr" / "__init__.py").write_text("ORIGIN = 'target'\n")
    (tree / "spacr" / "qt" / "__init__.py").write_text("")
    if builder_body is None:
        payload = sources or {
            "setting_labels": {"alpha": "Alpha"},
            "setting_tooltips": {"alpha": "Explains alpha."},
            "categories": ("General",),
            "ui": ("Run", "Stop"),
            "installer": {},
            "module_summaries": {"mask": "Segment."},
        }
        builder_body = textwrap.dedent(f"""
            import spacr, spacr.qt  # noqa: F401  - proves the package imports

            def canonical_sources():
                return {payload!r}
        """)
    (tree / "tools" / "build_i18n_catalogs.py").write_text(builder_body)
    return tree


def _decoy(base: Path) -> Path:
    """A second checkout that a meta_path finder would otherwise serve."""
    decoy = base / "decoy"
    (decoy / "spacr").mkdir(parents=True)
    (decoy / "spacr" / "__init__.py").write_text("ORIGIN = 'decoy'\n")
    return decoy


def test_an_inherited_PYTHONPATH_cannot_choose_the_tree(runner, tmp_path):
    """``PYTHONPATH`` is inherited silently, so the parent drops it.

    This covers the launcher's half only.  The child's own sys.path scrub is a
    separate mechanism with its own test below -- proven separate by mutation:
    disabling the scrub leaves this test green.
    """
    tree = _make_tree(tmp_path)
    decoy = _decoy(tmp_path)

    import os
    env_backup = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(decoy)
    try:
        result = runner._run_child(tree, sys.executable)
    finally:
        if env_backup is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = env_backup

    assert result["ok"], result
    assert not result["foreign_modules"]
    # The decoy must not appear as the origin of anything that loaded.
    assert str(decoy) not in json.dumps(result["sources"])


def test_the_child_strips_a_foreign_checkout_it_is_handed_on_sys_path(
        runner, tmp_path):
    """The scrub inside the child, exercised without the parent's help.

    ``_run_child`` drops ``PYTHONPATH`` before starting the child, which means
    the decoy in the test above never reaches the child's ``sys.path`` at all.
    A real editable install arrives by a ``.pth`` file instead, which no
    environment scrub can prevent -- so the child has to remove it itself.
    Here the child is launched DIRECTLY with the decoy inherited, which is the
    only way to put that code on the hook.
    """
    import os
    import subprocess

    tree = _make_tree(tmp_path)
    decoy = _decoy(tmp_path)
    child = tmp_path / "read.py"
    child.write_text(runner._CHILD)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(decoy)          # deliberately NOT scrubbed
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, str(child), str(tree), runner._BEGIN, runner._END],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    payload = completed.stdout.split(runner._BEGIN, 1)[1]
    result = json.loads(payload.split(runner._END, 1)[0].strip())

    assert str(decoy) in result["removed_paths"], (
        "the child must remove a checkout it inherits, not rely on the "
        "launcher having scrubbed the environment"
    )
    assert result["ok"], result
    assert not result["foreign_modules"]


def test_a_module_from_outside_the_tree_is_reported_not_accepted(
        runner, tmp_path):
    """The isolation is CHECKED, not asserted.

    A runner that merely scrubbed sys.path would pass its own inspection while
    a stray module sat in sys.modules.  Here the builder plants exactly that,
    and the reader has to notice.
    """
    foreign = tmp_path / "elsewhere" / "spacr_qt_theme.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("")
    tree = _make_tree(tmp_path, builder_body=textwrap.dedent(f"""
        import sys, types
        import spacr  # noqa: F401

        def canonical_sources():
            planted = types.ModuleType("spacr.qt.theme")
            planted.__file__ = {str(foreign)!r}
            sys.modules["spacr.qt.theme"] = planted
            return {{"setting_labels": {{}}, "setting_tooltips": {{}},
                    "categories": (), "ui": (), "installer": {{}},
                    "module_summaries": {{}}}}
    """))

    result = runner._run_child(tree, sys.executable)

    assert result["ok"] is False
    assert "spacr.qt.theme" in result["foreign_modules"]
    assert result["foreign_modules"]["spacr.qt.theme"] == str(foreign)


def test_a_tree_that_cannot_import_itself_is_not_called_a_path_problem(
        runner, tmp_path):
    """Entry 36's actual fault, and the one it was mistaken for.

    ``d0c1a633c`` imports a name from its own ``theme.py`` that its own
    ``theme.py`` does not define.  No foreign module is involved, and no amount
    of sys.path isolation helps.  The reader must say so, because the two
    faults lead to different work.
    """
    tree = _make_tree(tmp_path, builder_body=textwrap.dedent("""
        import spacr  # noqa: F401
        from spacr.qt.missing import absent_symbol  # noqa: F401

        def canonical_sources():
            return {}
    """))
    (tree / "spacr" / "qt" / "missing.py").write_text(
        "# the symbol this revision's importer expects is not here\n")

    result = runner._run_child(tree, sys.executable)

    assert result["ok"] is False
    assert result["error_type"] == "ImportError"
    assert result["frames_all_inside_tree"] is True, (
        "every frame is inside the tree, so this is an internally "
        "inconsistent revision and must not be reported as module mixing"
    )
    # And the runner's own file must never be what makes that verdict false.
    assert all("i18n_reference_sources" not in f
               and "read_canonical_sources" not in f
               for f in result["frame_files"])


def test_the_digest_matches_the_ratchet_that_pins_it(runner):
    """The capture is only useful if it computes the ratchet's own number.

    ``EXTERNAL_SOURCE_KEY_SHA256`` is built from sorted ``(table, key)`` pairs
    joined by NULs.  If either side changes shape, a capture silently stops
    being comparable to the constant it exists to be compared against.
    """
    import hashlib

    sources = {
        "setting_labels": ["b", "a"],
        "setting_tooltips": ["a"],
        "categories": ["General"],
        "ui": ["Run"],
        "module_summaries": ["mask"],
    }
    identities = runner._identities(sources)
    assert identities == sorted(identities), "identities must be sorted"
    expected = hashlib.sha256(
        "\0".join(f"{table}\0{key}" for table, key in identities
                  ).encode("utf-8")
    ).hexdigest()
    assert runner._digest(identities) == expected


def test_the_capture_covers_exactly_the_ratchet_s_tables(runner):
    """A table added to the ratchet and not here would go unreviewed.

    The reverse -- a table here that the ratchet does not pin -- would make a
    capture disagree with the constant for a reason nobody could see.
    """
    tools = str(ROOT / "tests" / "qt")
    sys.path.insert(0, tools)
    try:
        ratchet = import_module("test_i18n_caption_ratchet")
    finally:
        sys.path.remove(tools)

    assert set(runner.EXTERNAL_TABLES) == set(ratchet.EXTERNAL_SOURCE_COUNTS)


def test_the_reader_refuses_to_patch_the_repository_itself(runner, tmp_path):
    """A repair patch belongs in a scratch checkout, never in the tree.

    The patch exists to make a broken historical revision importable.  Applied
    to the working repository it would be an undeclared source change that a
    later capture would silently bake into its digest.
    """
    patch = tmp_path / "any.patch"
    patch.write_text("")
    args = runner.argparse.Namespace(
        rev=None, tree=str(runner.ROOT), out=str(tmp_path / "out.json"),
        python=sys.executable, workdir=str(tmp_path), patch=str(patch),
    )
    with pytest.raises(SystemExit, match="refusing to patch the repository"):
        runner.cmd_capture(args)
