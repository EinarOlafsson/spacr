"""Promotion keeps the working folders off ``main`` and on ``nightly``.

``instructions/``, ``skill/``, ``proposals/`` and ``.claude/`` are tracked on
``nightly`` on purpose. Merging carries every tracked path, so each promotion
brings them back across and each promotion has to drop them again -- the
property that makes this a tool rather than a remembered step, and the one
worth a test, because a promotion that forgets once publishes them.

Every test here builds its own throwaway repository. Nothing reads or writes
the repository the suite lives in.
"""
from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        return import_module("promote_to_main")
    finally:
        sys.path.remove(tools_dir)


promote_to_main = _tool()


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo``, failing the test loudly on a non-zero exit."""
    done = subprocess.run(("git", *args), cwd=str(repo),
                          capture_output=True, text=True)
    assert done.returncode == 0, (
        f"git {' '.join(args)} failed: {done.stderr or done.stdout}")
    return done.stdout.strip()


def tracked(repo: Path, ref: str) -> set:
    """Every path ``ref`` tracks."""
    listing = git(repo, "ls-tree", "-r", "--name-only", ref)
    return {line for line in listing.splitlines() if line}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo shaped like spaCR: a product plus working folders.

    ``main`` is branched before the working folders exist, so the first
    promotion is the one that would introduce them -- exactly the shape the
    real repository is in.
    """
    root = tmp_path / "shaped-like-spacr"
    root.mkdir()
    git(root, "init", "--quiet", "--initial-branch=main")
    git(root, "config", "user.name", "Einar Olafsson")
    git(root, "config", "user.email", "einar.olafsson@gmail.com")
    git(root, "config", "commit.gpgsign", "false")

    write(root / "README.rst", "the product\n")
    write(root / "spacr" / "__init__.py", "VERSION = '1.0'\n")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "the product")

    git(root, "checkout", "--quiet", "-b", "nightly")
    write(root / "instructions" / "open" / "250_public.txt", "an instruction\n")
    write(root / "skill" / "engineer.md", "a skill\n")
    write(root / "proposals" / "a_proposal.md", "a proposal\n")
    write(root / ".claude" / "settings.json", "{}\n")
    write(root / "spacr" / "feature.py", "def feature():\n    return 1\n")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "the work, and how it is driven")
    return root


WORKING_PATHS = {
    "instructions/open/250_public.txt",
    "skill/engineer.md",
    "proposals/a_proposal.md",
    ".claude/settings.json",
}


class TestTheDryRunIsTheDefault:
    def test_a_bare_run_changes_nothing_at_all(self, repo, capsys):
        before_main = git(repo, "rev-parse", "main")
        before_nightly = git(repo, "rev-parse", "nightly")
        before_branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")

        assert promote_to_main.main(["--repo", str(repo)]) == 0

        assert git(repo, "rev-parse", "main") == before_main, (
            "the dry run moved main")
        assert git(repo, "rev-parse", "nightly") == before_nightly
        assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == before_branch
        assert tracked(repo, "main") == {"README.rst", "spacr/__init__.py"}, (
            "the dry run merged")

    def test_it_prints_the_folders_and_the_commands(self, repo, capsys):
        promote_to_main.main(["--repo", str(repo)])
        printed = capsys.readouterr().out

        assert "DRY RUN" in printed
        for folder in ("instructions", "skill", "proposals", ".claude"):
            assert f"{folder}/" in printed, f"{folder} was not named"
        assert "git merge --no-ff" in printed
        assert "git rm -r --cached" in printed
        assert "--execute" in printed, "it did not say how to do it for real"


class TestARealRun:
    def test_main_gets_the_work_and_none_of_the_working_folders(self, repo):
        assert promote_to_main.main(["--repo", str(repo), "--execute"]) == 0

        on_main = tracked(repo, "main")
        assert "spacr/feature.py" in on_main, (
            "the promotion did not carry the work across")
        leaked = on_main & WORKING_PATHS
        assert not leaked, f"the working folders reached main: {sorted(leaked)}"

    def test_nightly_still_tracks_them(self, repo):
        promote_to_main.main(["--repo", str(repo), "--execute"])
        assert WORKING_PATHS <= tracked(repo, "nightly"), (
            "the promotion removed them from nightly, where they belong")

    def test_the_files_are_still_on_disk(self, repo):
        promote_to_main.main(["--repo", str(repo), "--execute"])
        for relative in sorted(WORKING_PATHS):
            assert (repo / relative).exists(), (
                f"{relative} left the disk; the removal was not --cached")

    def test_it_returns_to_the_branch_it_started_on(self, repo):
        assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "nightly"
        promote_to_main.main(["--repo", str(repo), "--execute"])
        assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "nightly"

    def test_nothing_was_pushed(self, repo):
        promote_to_main.main(["--repo", str(repo), "--execute"])
        assert git(repo, "remote") == "", (
            "the throwaway repo grew a remote; the tool must never push")


class TestEveryPromotionDropsThemAgain:
    """The reason this is a tool. A merge re-adds every tracked path."""

    def test_a_second_promotion_drops_the_folders_the_merge_brought_back(
            self, repo):
        promote_to_main.main(["--repo", str(repo), "--execute"])
        assert not tracked(repo, "main") & WORKING_PATHS

        # More work on nightly, touching a working folder and the product.
        write(repo / "instructions" / "open" / "251_next.txt", "the next one\n")
        write(repo / "spacr" / "later.py", "def later():\n    return 2\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "more work")

        assert promote_to_main.main(["--repo", str(repo), "--execute"]) == 0

        on_main = tracked(repo, "main")
        assert "spacr/later.py" in on_main, "the second promotion carried nothing"
        leaked = {p for p in on_main if p.split("/")[0] in
                  {"instructions", "skill", "proposals", ".claude"}}
        assert not leaked, (
            "the second promotion published the working folders the merge "
            f"brought back: {sorted(leaked)}")


class TestItRefusesRatherThanGuess:
    def test_a_dirty_tree_stops_a_real_run(self, repo, capsys):
        write(repo / "spacr" / "feature.py", "def feature():\n    return 99\n")
        before = git(repo, "rev-parse", "main")

        assert promote_to_main.main(["--repo", str(repo), "--execute"]) == 1
        assert "uncommitted changes" in capsys.readouterr().err
        assert git(repo, "rev-parse", "main") == before, (
            "it changed main despite refusing")

    def test_a_missing_branch_stops_it(self, repo, capsys):
        assert promote_to_main.main(
            ["--repo", str(repo), "--target", "nope", "--execute"]) == 1
        assert "does not exist" in capsys.readouterr().err

    def test_a_folder_nightly_does_not_track_is_not_an_error(self, repo):
        """A working folder can be retired between promotions."""
        assert promote_to_main.main(
            ["--repo", str(repo), "--execute",
             "--folder", "instructions", "--folder", "never_existed"]) == 0
        assert "spacr/feature.py" in tracked(repo, "main")
        assert not {p for p in tracked(repo, "main")
                    if p.startswith("instructions/")}
