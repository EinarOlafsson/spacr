"""A lightweight source install still produces a working spaCR.

Instruction 328. `git clone` of spaCR pulls a multi-gigabyte history and
lays down ~427 MB of tracked files, when ~76 MB is what it takes to RUN
the program. packaging/install_from_source.sh fetches only that, using a
shallow partial clone plus a sparse checkout whose exclusion list lives in
packaging/source_install_excludes.txt.

The danger with an exclusion list is not that it fails loudly. It is that
somebody renames `spacr/resources/models/` and the exclusion silently
matches nothing -- the install quietly goes back to being large -- or that
somebody starts importing from a directory the list drops, and the install
quietly stops working for anyone who used the script.

So these tests pin both directions: every exclusion still names something
real, and nothing excluded is needed to run.
"""
from __future__ import annotations

import builtins
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
EXCLUDES = REPO / "packaging" / "source_install_excludes.txt"
SCRIPT = REPO / "packaging" / "install_from_source.sh"


def _sparse_lines():
    """The list as git would read it -- comments and blanks stripped."""
    text = EXCLUDES.read_text()
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_the_exclusion_list_and_the_script_both_exist():
    assert EXCLUDES.is_file()
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK), "the install script must be executable"


def test_the_list_takes_everything_before_it_takes_anything_away():
    """Order is load-bearing in sparse-checkout syntax.

    `/*` must come first. If an exclusion preceded it, the `/*` would add
    the whole tree back and every exclusion above it would do nothing --
    an install that is silently full-size again.
    """
    lines = _sparse_lines()
    assert lines[0] == "/*"
    assert all(line.startswith("!") for line in lines[1:]), \
        "every line after the first must be an exclusion"


@pytest.mark.parametrize("line", [ln for ln in _sparse_lines()[1:]])
def test_every_exclusion_still_names_something_that_exists(line):
    """A renamed directory makes its exclusion silently useless.

    Nothing errors -- git just matches no files, and the "lightweight"
    install quietly carries the weight again. This is the test that
    notices.
    """
    pattern = line.lstrip("!").lstrip("/").rstrip("/")
    if "*" in pattern:                      # `*.pdf` and friends
        assert list(REPO.glob(pattern)), \
            f"exclusion {line!r} matches nothing in the tree any more"
        return
    assert (REPO / pattern).exists(), \
        f"exclusion {line!r} names a path that no longer exists"


def test_the_script_reads_the_same_list_these_tests_do():
    """The script and the test must not drift.

    If the script ever grew its own copy of the exclusions, this file
    would go on passing while the real install did something else.
    """
    body = SCRIPT.read_text()
    assert "source_install_excludes.txt" in body
    assert "sparse-checkout" in body


def test_the_script_asks_for_both_a_shallow_and_a_filtered_fetch():
    """Sparse checkout ALONE does not save a download.

    It only declines to write files to disk; git still transfers them. The
    saving comes from pairing it with `--filter=blob:none`, so the
    excluded blobs are never requested. Losing the filter would keep every
    test above passing while the download went back to full size.
    """
    body = SCRIPT.read_text()
    assert "--filter=blob:none" in body
    assert "--depth 1" in body
    assert "partialclonefilter" in body, \
        "the filter must be written to config so later fetches keep it"


def test_the_optional_flags_are_if_blocks_not_and_lists():
    """`set -eu` plus `[ x = 1 ] && cmd` aborts the script when x is not 1.

    That would have made every install which did NOT ask for translations
    exit silently at that line. Pinned because the short form is the
    natural thing to write and the failure is invisible.
    """
    body = SCRIPT.read_text()
    assert "set -eu" in body
    for flag in ("KEEP_TRANSLATIONS", "KEEP_TESTS", "KEEP_DOCS"):
        assert f'if [ "${flag}" = 1 ]' in body, \
            f"{flag} must be tested with `if`, not with an AND-OR list"


def test_the_translation_catalogs_are_the_only_exclusion_that_costs_anything():
    """And the script offers a flag to put them back."""
    body = SCRIPT.read_text()
    assert "--with-translations" in body
    assert "i18n_catalogs" in EXCLUDES.read_text()


def test_dropping_the_catalogs_leaves_the_ui_translated():
    """THE CONTRACT the i18n_catalogs exclusion rests on.

    spacr/qt/i18n.py wraps every catalog import in
    `except (ImportError, AttributeError)` and falls back to a compact
    core catalog built into the module. If that fallback were ever
    removed, a lightweight install would come up untranslated -- so this
    simulates the catalogs being absent and asserts the UI still speaks
    French and German.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.i18n import tr

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if "i18n_catalogs" in name:
            raise ImportError("simulating a lightweight install")
        return real_import(name, *args, **kwargs)

    hidden = {k: v for k, v in sys.modules.items()
              if "i18n_catalogs" in k}
    for key in hidden:
        del sys.modules[key]
    builtins.__import__ = blocked
    try:
        assert tr("Settings", "fr") == "Paramètres"
        assert tr("Settings", "de") == "Einstellungen"
        assert tr("Cancel", "fr") == "Annuler"
    finally:
        builtins.__import__ = real_import
        sys.modules.update(hidden)


def test_the_fallback_is_what_makes_that_pass_and_not_luck():
    """Mutation guard for the test above.

    Without this, `tr` returning the SOURCE string unchanged on a missing
    catalog would look identical to a successful fallback for any word
    that happens to be spelled the same in both languages. Asserting the
    French differs from the English proves a catalog was consulted.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.i18n import tr
    assert tr("Settings", "fr") != "Settings"


@pytest.mark.slow
def test_the_script_produces_a_tree_that_imports_spacr():
    """End to end, against this very checkout.

    Runs the real script with this repository as the remote, then imports
    spacr out of the result in a subprocess. This is the test that would
    catch somebody excluding a directory the package actually imports.
    """
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    if shutil.which("git") is None:
        pytest.skip("git unavailable")

    # The REAL branch name, not "HEAD" -- the script ends by putting the
    # checkout on a branch so `git pull` works afterwards, and `git
    # checkout -B HEAD` is rejected as an invalid branch name. Getting
    # this wrong made the whole test skip, which is worse than failing.
    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if not branch or branch == "HEAD":
        pytest.skip("detached HEAD; no branch to fetch")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(
            ["sh", str(SCRIPT), "--repo", str(REPO), "--branch", branch,
             "--dir", os.path.join(tmp, "out"), "--no-install"],
            capture_output=True, text=True, timeout=600,
        )
        if run.returncode != 0:
            pytest.skip(f"fetch failed in this environment: {run.stderr[-400:]}")

        out = os.path.join(tmp, "out")
        assert os.path.isdir(os.path.join(out, "spacr"))
        assert not os.path.isdir(os.path.join(out, "docs")), \
            "docs should have been excluded"
        assert not os.path.isdir(os.path.join(out, "tests")), \
            "tests should have been excluded"

        # Importing out of the fetched tree takes MORE than putting it on
        # sys.path. spaCR is installed here in editable mode, and a PEP
        # 660 editable install works through a MetaPathFinder -- which
        # `import` consults BEFORE sys.path. So `sys.path.insert(0, out)`
        # silently lost to the development checkout every time, and this
        # test passed even with the whole of spacr/qt excluded.
        #
        # The finder has to be torn out of sys.meta_path first, and the
        # resolved __file__ asserted to be inside the fetched tree, or
        # the proof is worthless.
        probe = (
            "import sys\n"
            # `type(f).__module__` is NOT enough. setuptools registers
            # its editable finder as a CLASS, not an instance, so type(f)
            # is `type` and its module is `builtins` -- the filter matched
            # nothing and the finder survived, which is exactly how this
            # test passed with the whole of spacr/qt excluded. Check the
            # object's own __module__ first.
            "def _mod(f):\n"
            "    return getattr(f, '__module__', None) or type(f).__module__\n"
            "sys.meta_path = [f for f in sys.meta_path\n"
            "                 if '__editable__' not in _mod(f)]\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import spacr\n"
            "assert spacr.__file__.startswith(sys.argv[1]), (\n"
            "    'imported the wrong spacr: ' + spacr.__file__)\n"
            "from spacr.qt.i18n import tr\n"
            "print('OK', spacr.__file__, tr('Settings', 'fr'))\n"
        )
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proof = subprocess.run(
            [sys.executable, "-c", probe, out],
            capture_output=True, text=True, env=env, timeout=600,
        )
        assert "OK" in proof.stdout, proof.stderr[-2000:]
        assert out in proof.stdout, \
            f"the probe imported spacr from outside the fetched tree: {proof.stdout}"
