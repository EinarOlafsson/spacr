"""No commit on this branch credits Claude, and the guard that says so exists.

Instruction 58. The repository is authored by Einar Olafsson alone: no AI
author, no committer, no ``Co-Authored-By`` trailer.

WHY A TEST AND NOT ONLY A HOOK. The history has now been cleaned twice. The
second time, 70 trailers had arrived from a lane whose checkout had no hook,
because ``.git/hooks/`` does not clone, sync or travel -- instruction 58
recorded a guard as installed and it simply was not there. A hook protects
the checkout it sits in and nothing else. CI runs on every branch regardless
of what any developer's ``.git`` happens to contain, so this is the backstop
that actually catches a third round.

The hook is still worth having: it refuses the trailer at the moment it is
written, which is cheaper than finding it after a push. Both are checked
here -- that the tracked hook exists and really refuses, and that no commit
already on the branch carries a trailer.

READING THE COUNT CORRECTLY MATTERS, and getting it wrong once nearly caused
a 3,748-commit rewrite instead of a 364-commit one. ``git log --grep`` matches
anywhere in a message, so it also matches commits that merely QUOTE the
phrase -- instruction 58's own filing commit does, and so does every commit
recording this work. ``%(trailers:key=...)`` asks git to parse trailers, which
is the only honest question.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "tools" / "hooks" / "commit-msg"

#: Messages the guard must refuse, and why each shape has been seen.
REFUSED = [
    # what the harness re-issues at the start of a session
    "Subject\n\nBody.\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n",
    # the same, older model name; all 38 of the first round looked like this
    "Subject\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n",
    # case and spacing are not a way around it
    "Subject\n\nco-authored-by:   CLAUDE Opus 4.8 <x@y>\n",
    # the pull-request footer, which is the same claim in prose
    "Subject\n\n\U0001F916 Generated with [Claude Code](https://claude.com/claude-code)\n",
]

#: Messages that must pass. The second is the trap: it TALKS about the
#: trailer, which is what every commit recording instruction 58 does.
ALLOWED = [
    "A plain subject\n\nAn ordinary body.\n",
    "Record the trailer cleanup\n\nRemoved every Co-Authored-By: Claude "
    "trailer from nightly; the count is now zero.\n",
]


def _git(*args: str) -> str:
    return subprocess.run(("git", "-C", str(REPO)) + args,
                          capture_output=True, text=True, check=False).stdout


def _run_hook(message: str) -> int:
    """Feed one message to the tracked hook and return its exit status."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(message)
        path = handle.name
    try:
        return subprocess.run(["/bin/sh", str(HOOK), path],
                              capture_output=True, text=True).returncode
    finally:
        os.unlink(path)


# -- the guard exists and works -------------------------------------------
def test_the_hook_is_tracked_in_the_tree():
    """A hook in .git/hooks protects one checkout. This one is in the tree,
    so it is at least present in every clone."""
    assert HOOK.is_file(), f"{HOOK} is missing"
    tracked = _git("ls-files", "--", "tools/hooks/commit-msg").strip()
    assert tracked, "the hook exists but is not tracked, so it will not clone"


@pytest.mark.parametrize("message", REFUSED,
                         ids=["opus5", "bare", "shouty", "pr-footer"])
def test_the_hook_refuses_an_attribution(message):
    assert _run_hook(message) == 1, f"the guard accepted: {message!r}"


@pytest.mark.parametrize("message", ALLOWED, ids=["plain", "talks-about-it"])
def test_the_hook_allows_an_ordinary_message(message):
    assert _run_hook(message) == 0, f"the guard refused: {message!r}"


# -- and the branch is actually clean -------------------------------------
def test_no_commit_on_this_branch_carries_a_claude_trailer():
    """The backstop. Asks git to PARSE trailers rather than grepping the
    message, because a grep also matches commits that quote the phrase --
    every commit recording instruction 58 does, this file included."""
    if not (REPO / ".git").exists():                     # installed copy
        pytest.skip("not a git checkout")
    values = _git("log", "--format=%(trailers:key=Co-Authored-By,valueonly)")
    offenders = [line for line in values.splitlines()
                 if "claude" in line.lower()]
    assert not offenders, (
        f"{len(offenders)} commit(s) carry a Co-Authored-By trailer naming "
        "Claude. Instruction 58 has the rewrite recipe; do not simply "
        "re-clean without also installing the hook in the lane that wrote "
        "them, or this happens a fourth time.")


def test_no_commit_is_authored_or_committed_by_claude():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    people = _git("log", "--format=%an|%cn")
    offenders = [line for line in people.splitlines()
                 if "claude" in line.lower()]
    assert not offenders, f"{len(offenders)} commit(s) name Claude as a person"


# --- the push boundary, which is the only one that does not depend on CI ---

PRE_PUSH = REPO / "tools" / "hooks" / "pre-push"


def test_the_push_guard_is_tracked_too():
    """A hook that is not in the tree protects one checkout.

    The same reasoning as the commit-msg guard above, and it applies
    harder here: a session working in a fresh worktree has never run
    `git config core.hooksPath tools/hooks`, so a guard living only in
    `.git/hooks/` was never there at all.
    """
    tracked = _git("ls-files", "--", "tools/hooks/pre-push").strip()
    assert tracked == "tools/hooks/pre-push"
    assert PRE_PUSH.exists()
    assert os.access(PRE_PUSH, os.X_OK), "the hook is not executable"


def test_the_push_guard_matches_a_trailer_and_not_a_mention():
    """Anchored, because this repository discusses the trailer in prose.

    Three commits describe the problem -- the 38 removed, the strip
    regex, the 70 that came back -- and an unanchored pattern counts them
    as instances. Not hypothetical: it produced a "3 -> 6" report on
    2026-09-10 about a branch whose real count was zero.
    """
    source = PRE_PUSH.read_text(encoding="utf-8")
    assert "^[[:space:]]*claude" in source, (
        "the trailer match is not anchored, so prose about the trailer "
        "counts as a trailer")


def test_the_push_guard_survives_a_branch_deletion():
    """`git push --delete` sends an all-zero local sha and pushes nothing.

    Without the guard clause the range becomes `<sha>..0000000` and git
    errors on a push that is entirely legitimate.
    """
    source = PRE_PUSH.read_text(encoding="utf-8")
    assert '[ "$local_sha" = "$zero" ] && continue' in source


def test_the_push_guard_bounds_a_new_branch():
    """A first push has no remote side to subtract from.

    Scanning from the root would read the whole history every time a
    topic branch is first pushed; `--not --remotes` keeps it to the
    commits the push actually introduces.
    """
    source = PRE_PUSH.read_text(encoding="utf-8")
    assert "--not --remotes" in source
