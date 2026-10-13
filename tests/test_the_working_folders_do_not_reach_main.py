"""Promotion keeps the working folders off ``main`` and on ``nightly``.

``features/``, ``skill/``, ``proposals/`` and ``.claude/`` are tracked on
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
    write(root / "features" / "future" / "250_public.txt", "an instruction\n")
    write(root / "skill" / "engineer.md", "a skill\n")
    write(root / "proposals" / "a_proposal.md", "a proposal\n")
    write(root / ".claude" / "settings.json", "{}\n")
    write(root / "spacr" / "feature.py", "def feature():\n    return 1\n")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "the work, and how it is driven")
    return root


WORKING_PATHS = {
    "features/future/250_public.txt",
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
        for folder in ("features", "skill", "proposals", ".claude"):
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
        write(repo / "features" / "future" / "251_next.txt", "the next one\n")
        write(repo / "spacr" / "later.py", "def later():\n    return 2\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "more work")

        assert promote_to_main.main(["--repo", str(repo), "--execute"]) == 0

        on_main = tracked(repo, "main")
        assert "spacr/later.py" in on_main, "the second promotion carried nothing"
        leaked = {p for p in on_main if p.split("/")[0] in
                  {"features", "skill", "proposals", ".claude"}}
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
             "--folder", "features", "--folder", "never_existed"]) == 0
        assert "spacr/feature.py" in tracked(repo, "main")
        # `features/`, NOT `instructions/`. This assertion named the old
        # folder until 2026-09-13 and the fixture has never created one, so
        # it matched nothing and could not fail -- a guard that passes because
        # it is looking in an empty place. The folder this test promotes is
        # `features`, so that is the one that must not reach main.
        assert not {p for p in tracked(repo, "main")
                    if p.startswith("features/")}


class TestAStaleLocalBranchIsRefused:
    """A promotion merges the LOCAL branch, so a stale one must not run.

    The failure this guards is not a crash. On 2026-09-13 the local
    ``nightly`` in the working checkout was 147 commits behind
    ``origin/nightly``, and the dry run reported "not tracked on nightly,
    nothing to drop: features/" -- true of that old ref and false of the
    branch anyone meant. A real run would have merged stale work into
    ``main``, dropped nothing because there was nothing there to drop, and
    printed a clean report while doing it.
    """

    @staticmethod
    def _with_upstream(root: Path, tmp_path: Path) -> Path:
        """Give ``root`` an origin that is one commit ahead on nightly."""
        origin = tmp_path / "origin.git"
        git(root, "clone", "--quiet", "--bare", str(root), str(origin))
        git(root, "remote", "add", "origin", str(origin))
        git(root, "fetch", "--quiet", "origin")
        git(root, "branch", "--set-upstream-to=origin/nightly", "nightly")

        # Move the upstream on, without moving the local branch: a second
        # clone commits and pushes, which is what another session does.
        other = tmp_path / "other"
        git(root, "clone", "--quiet", str(origin), str(other))
        git(other, "config", "user.name", "Einar Olafsson")
        git(other, "config", "user.email", "einar.olafsson@gmail.com")
        git(other, "config", "commit.gpgsign", "false")
        git(other, "checkout", "--quiet", "nightly")
        write(other / "spacr" / "newer.py", "def newer():\n    return 3\n")
        git(other, "add", "-A")
        git(other, "commit", "--quiet", "-m", "work the local branch lacks")
        git(other, "push", "--quiet", "origin", "nightly")
        git(root, "fetch", "--quiet", "origin")
        return origin

    def test_a_branch_behind_its_upstream_stops_even_the_dry_run(
            self, repo, tmp_path, capsys):
        self._with_upstream(repo, tmp_path)
        status = promote_to_main.main(["--repo", str(repo)])
        assert status != 0
        captured = capsys.readouterr()
        printed = captured.out + captured.err
        assert "behind origin/nightly" in printed, printed
        assert "Nothing was changed." in printed, printed

    def test_it_names_the_remedy(self, repo, tmp_path, capsys):
        self._with_upstream(repo, tmp_path)
        promote_to_main.main(["--repo", str(repo)])
        captured = capsys.readouterr()
        printed = captured.out + captured.err
        assert "git fetch origin" in printed, printed
        assert "git branch -f nightly origin/nightly" in printed, printed

    def test_main_is_untouched_by_the_refusal(self, repo, tmp_path):
        self._with_upstream(repo, tmp_path)
        before = git(repo, "rev-parse", "main")
        promote_to_main.main(["--repo", str(repo), "--execute"])
        assert git(repo, "rev-parse", "main") == before

    def test_a_branch_level_with_its_upstream_is_allowed_through(
            self, repo, tmp_path):
        self._with_upstream(repo, tmp_path)
        git(repo, "checkout", "--quiet", "nightly")
        git(repo, "merge", "--quiet", "--ff-only", "origin/nightly")
        assert promote_to_main.main(["--repo", str(repo), "--execute"]) == 0
        assert not (tracked(repo, "main") & WORKING_PATHS)

    def test_no_upstream_at_all_is_not_an_error(self, repo):
        """A fresh clone with no remote must still be promotable."""
        assert promote_to_main.upstream_gap(repo, "nightly") is None
        assert promote_to_main.main(["--repo", str(repo)]) == 0


class TestASecondPromotionAfterTheFoldersMovedOn:
    """The collision every promotion after the first one actually hits.

    `main` deleted the working folders in its own drop commit. `nightly`
    kept editing them, because that is where the work is written. Git then
    reports modify/delete on files the very next step removes from `main`
    again -- eight of them on 2026-09-13, which stopped a promotion whose
    product changes were entirely clean.
    """

    @staticmethod
    def _promote(repo: Path) -> int:
        return promote_to_main.main(["--repo", str(repo), "--execute"])

    def test_a_modified_working_file_does_not_stop_the_promotion(self, repo):
        assert self._promote(repo) == 0
        git(repo, "checkout", "--quiet", "nightly")
        write(repo / "features" / "future" / "250_public.txt",
              "the instruction, edited after the first promotion\n")
        write(repo / "spacr" / "second.py", "def second():\n    return 2\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "more work, and more notes")

        assert self._promote(repo) == 0
        assert not (tracked(repo, "main") & WORKING_PATHS)
        assert "spacr/second.py" in tracked(repo, "main")

    def test_the_source_still_tracks_them_afterwards(self, repo):
        self._promote(repo)
        git(repo, "checkout", "--quiet", "nightly")
        write(repo / "features" / "future" / "250_public.txt", "edited\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "edit the notes")
        self._promote(repo)
        assert WORKING_PATHS <= tracked(repo, "nightly")

    def test_a_conflict_in_the_product_still_stops_it(self, repo, capsys):
        """A real collision must not be swept up with the bookkeeping ones."""
        self._promote(repo)
        git(repo, "checkout", "--quiet", "main")
        write(repo / "spacr" / "feature.py", "def feature():\n    return 'main'\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "main edits the product")
        git(repo, "checkout", "--quiet", "nightly")
        write(repo / "spacr" / "feature.py", "def feature():\n    return 'nightly'\n")
        git(repo, "add", "-A")
        git(repo, "commit", "--quiet", "-m", "nightly edits the product")

        before = git(repo, "rev-parse", "main")
        assert self._promote(repo) != 0
        captured = capsys.readouterr()
        printed = captured.out + captured.err
        assert "outside the folders" in printed, printed
        assert "spacr/feature.py" in printed, printed
        assert git(repo, "rev-parse", "main") == before
