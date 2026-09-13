#!/usr/bin/env python3
"""Promote ``nightly`` to ``main`` without publishing the working folders.

``features/``, ``skill/``, ``proposals/`` and ``.claude/`` are how the
work is driven, not the product. They stay tracked on ``nightly`` -- version
control is the point of keeping them -- and they do not appear on the branch
a visitor to GitHub lands on.

MERGING CARRIES EVERY TRACKED PATH, so the removal is not a one-off. Each
promotion brings the working folders back across with everything else, and
each promotion has to drop them again. That is why this is a tool and not a
step in a checklist: a step that has to be remembered every time is a step
that is eventually forgotten, and forgetting it publishes the folders.

The removal is index-only (``git rm --cached``). The files stay on disk and
stay tracked on ``nightly``; only ``main``'s tree loses them.

DRY RUN IS THE DEFAULT. Nothing is written, no branch is switched, and every
command that a real run would issue is printed instead. A real run needs
``--execute``.

NOTHING HERE PUSHES. Promotion ends at a local ``main``; publishing it is a
separate, deliberate act.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: Tracked on ``nightly``, never on ``main``. Directory prefixes, matched
#: against ``git ls-tree`` output, so a name here covers everything under it.
WORKING_FOLDERS: Sequence[str] = ("features", "skill", "proposals", ".claude")

DEFAULT_SOURCE = "nightly"
DEFAULT_TARGET = "main"


class PromotionError(RuntimeError):
    """A promotion that must not continue, with the reason a user can act on."""


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------

def run_git(repo: Path, *args: str, check: bool = True) -> str:
    """Run one git command in ``repo`` and return its stdout, stripped.

    :param repo: working tree to run in.
    :param args: the git arguments, without the leading ``git``.
    :param check: raise :class:`PromotionError` on a non-zero exit. Pass
        False when a non-zero exit is one of the answers -- ``rev-parse``
        on a branch that may not exist, for instance.
    :raises PromotionError: when ``check`` and git failed. The message
        carries git's own stderr, which says more than a return code.
    """
    completed = subprocess.run(
        ("git", *args),
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise PromotionError(
            "git " + " ".join(args) + " failed:\n" + (
                completed.stderr.strip() or completed.stdout.strip()))
    return completed.stdout.strip()


def repo_root(start: Path) -> Path:
    """The top of the working tree containing ``start``.

    :raises PromotionError: when ``start`` is not inside a git working tree.
    """
    try:
        top = run_git(start, "rev-parse", "--show-toplevel")
    except PromotionError as exc:
        raise PromotionError(f"{start} is not inside a git repository") from exc
    return Path(top)


def branch_exists(repo: Path, branch: str) -> bool:
    """Whether ``branch`` is a local branch of ``repo``."""
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"),
        cwd=str(repo), capture_output=True, text=True,
    )
    return completed.returncode == 0


def current_branch(repo: Path) -> str:
    """The checked-out branch name, or the commit id when detached."""
    name = run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if name == "HEAD":  # detached
        return run_git(repo, "rev-parse", "HEAD")
    return name


def is_clean(repo: Path) -> bool:
    """Whether the working tree has no staged or unstaged change.

    Untracked files do not count: the working folders are untracked while
    ``main`` is checked out, which is the normal state this tool leaves
    behind, and a tool that refused to run in the state it creates could
    never run twice.
    """
    return run_git(repo, "status", "--porcelain", "--untracked-files=no") == ""


def tracked_working_folders(
    repo: Path,
    ref: str,
    folders: Sequence[str] = WORKING_FOLDERS,
) -> List[str]:
    """Which of ``folders`` ``ref`` actually tracks, in ``folders`` order.

    Asked of the ref rather than assumed, so the dry run reports what is
    really there and a real run does not try to remove a path that was
    retired between promotions.
    """
    listing = run_git(repo, "ls-tree", "-r", "--name-only", ref)
    paths = listing.splitlines()
    present = []
    for folder in folders:
        prefix = folder.rstrip("/") + "/"
        if any(path == folder or path.startswith(prefix) for path in paths):
            present.append(folder)
    return present


def file_count(repo: Path, ref: str, folder: str) -> int:
    """How many files ``ref`` tracks under ``folder``."""
    listing = run_git(repo, "ls-tree", "-r", "--name-only", ref, "--", folder)
    return len([line for line in listing.splitlines() if line])


# ---------------------------------------------------------------------------
# the promotion
# ---------------------------------------------------------------------------

def _echo(command: Sequence[str], out) -> None:
    print("    git " + " ".join(command), file=out)


def upstream_gap(repo: Path, branch: str) -> Optional[Tuple[str, int]]:
    """The branch's upstream and how many commits it is ahead of ``branch``.

    Returns None when the branch has no upstream configured, which is a
    legitimate state -- a fresh clone, a branch never pushed -- and not
    something a promotion should refuse over.

    :param repo: working tree to ask in.
    :param branch: the local branch to measure.
    :returns: ``(upstream_name, commits_behind)``, or None if no upstream.
    """
    upstream = run_git(repo, "rev-parse", "--abbrev-ref",
                       f"{branch}@{{upstream}}", check=False)
    if not upstream:
        return None
    behind = run_git(repo, "rev-list", "--count", f"{branch}..{upstream}",
                     check=False)
    return (upstream, int(behind)) if behind.isdigit() else None


def refuse_if_stale(repo: Path, branches: Sequence[str]) -> None:
    """Stop a promotion whose local branches trail their upstreams.

    THIS IS THE CHECK THAT WAS MISSING, AND ITS ABSENCE WAS WORSE THAN A
    WRONG ANSWER. A promotion merges the LOCAL branch. On 2026-09-13 the
    local ``nightly`` here was 147 commits behind ``origin/nightly``, and a
    dry run duly reported "not tracked on nightly, nothing to drop:
    features/" -- true of that stale ref, false of the branch the maintainer
    meant. A real run would have merged four-month-old work into ``main``,
    dropped no working folder because none was there to drop, and printed
    every line of a successful promotion while doing it.

    Being behind is therefore refused rather than warned about: the failure
    mode is a promotion that LOOKS right, and a warning in a wall of
    otherwise-correct output is not a defence against that.

    :param repo: working tree to check.
    :param branches: the local branches a promotion is about to use.
    :raises PromotionError: when any of them is behind its upstream.
    """
    stale = []
    for branch in branches:
        gap = upstream_gap(repo, branch)
        if gap is not None and gap[1]:
            stale.append((branch, gap[0], gap[1]))
    if not stale:
        return
    lines = [
        f"  {branch} is {count} commit{'' if count == 1 else 's'} "
        f"behind {upstream}"
        for branch, upstream, count in stale
    ]
    remedy = [f"  git branch -f {branch} {upstream}"
              for branch, upstream, _ in stale]
    raise PromotionError(
        "the local branches are behind their upstreams:\n"
        + "\n".join(lines)
        + "\n\nA promotion merges the LOCAL branch, so this would ship a "
          "stale tree and drop nothing, because the working folders may not "
          "exist on the older ref.\n\nFetch and fast-forward first:\n"
          "  git fetch origin\n"
        + "\n".join(remedy)
        + "\n\n(`git branch -f` refuses a branch that is checked out "
          "somewhere; in that worktree use `git merge --ff-only` instead.)"
          "\n\nNothing was changed.")


def promote(
    repo: Path,
    source: str = DEFAULT_SOURCE,
    target: str = DEFAULT_TARGET,
    folders: Sequence[str] = WORKING_FOLDERS,
    execute: bool = False,
    out=None,
) -> int:
    """Take ``source`` to ``target`` and drop ``folders`` from ``target``.

    :param repo: any path inside the working tree to promote.
    :param source: the branch the work is on.
    :param target: the branch the public lands on.
    :param folders: directory prefixes to keep out of ``target``.
    :param execute: perform the promotion. False -- the default -- prints
        what a real run would do and changes nothing at all.
    :param out: where the plan or the progress is written. Resolved to
        ``sys.stdout`` when None -- and resolved HERE rather than in the
        signature, because a default argument is bound once at import and
        a caller that replaces ``sys.stdout`` afterwards (a log capture, a
        wrapping script, pytest) would otherwise get half the report on
        one stream and half on the other.
    :returns: process exit status; 0 when the promotion is done or the plan
        was printed.
    :raises PromotionError: for anything that must stop a real run --
        a missing branch, a dirty tree, a failing merge.
    """
    if out is None:
        out = sys.stdout
    root = repo_root(repo)
    print(f"repository : {root}", file=out)
    print(f"promotion  : {source} -> {target}", file=out)

    for branch in (source, target):
        if not branch_exists(root, branch):
            raise PromotionError(
                f"branch {branch!r} does not exist in {root}. Nothing was "
                "changed.")

    refuse_if_stale(root, (source, target))

    started_on = current_branch(root)
    present = tracked_working_folders(root, source, folders)
    absent = [f for f in folders if f not in present]

    print("", file=out)
    print(f"working folders tracked on {source}:", file=out)
    if present:
        for folder in present:
            count = file_count(root, source, folder)
            print(f"  {folder}/  ({count} file{'' if count == 1 else 's'})",
                  file=out)
    else:
        print("  (none)", file=out)
    if absent:
        print(f"not tracked on {source}, nothing to drop: "
              + ", ".join(f"{f}/" for f in absent), file=out)

    merge_message = f"Promote {source} to {target}"
    drop_message = (
        "Drop the working folders from "
        + target
        + "\n\n"
        + "features/, skill/, proposals/ and .claude/ are how the work "
          "is driven, not\nthe product. They stay tracked on "
        + source
        + " and on disk; the branch a\nvisitor lands on does not carry them."
    )

    if not execute:
        print("", file=out)
        print("DRY RUN -- nothing was changed. A real run would:", file=out)
        _echo(("checkout", target), out)
        _echo(("merge", "--no-ff", "-m", repr(merge_message), source), out)
        if present:
            _echo(("rm", "-r", "--cached", "--quiet", "--", *present), out)
            _echo(("commit", "-m", repr(drop_message.splitlines()[0])), out)
        else:
            print("    (no working folder to drop; no second commit)",
                  file=out)
        if started_on != target:
            _echo(("checkout", "--force", started_on), out)
        print("", file=out)
        print("The files stay on disk and stay tracked on "
              f"{source}. Nothing is pushed.", file=out)
        print("Re-run with --execute to perform it.", file=out)
        return 0

    if not is_clean(root):
        raise PromotionError(
            "the working tree has uncommitted changes. Promotion switches "
            "branches, so it refuses to run over them. Commit or stash "
            "first. Nothing was changed.")

    print("", file=out)
    print(f"checking out {target}", file=out)
    run_git(root, "checkout", target)
    try:
        print(f"merging {source}", file=out)
        run_git(root, "merge", "--no-ff", "-m", merge_message, source)

        # Asked again, of the target, AFTER the merge: the merge is what
        # brings the folders across, and a folder retired on source between
        # promotions must not be passed to `git rm`, which fails on a path
        # it does not track.
        landed = tracked_working_folders(root, "HEAD", folders)
        if landed:
            print("dropping from the " + target + " tree: "
                  + ", ".join(f"{f}/" for f in landed), file=out)
            # --cached is the whole point: the index loses them, the disk
            # keeps them, and `source` still tracks them.
            run_git(root, "rm", "-r", "--cached", "--quiet", "--", *landed)
            run_git(root, "commit", "-m", drop_message)
        else:
            print("nothing to drop; the merge brought no working folder "
                  "across", file=out)
    finally:
        if started_on != target:
            print(f"returning to {started_on}", file=out)
            # --force, because the drop left the working folders on disk and
            # UNTRACKED, and `source` tracks those same paths: a plain
            # checkout refuses to overwrite an untracked file and would
            # strand the repository on `target`. The content being
            # overwritten is what the merge just took from `source`, so
            # forcing restores identical bytes -- and a real run has already
            # refused to start over an unclean tree, so there is no edit of
            # the user's here to lose.
            run_git(root, "checkout", "--force", started_on)

    print("", file=out)
    print(f"{target} is promoted. Nothing was pushed.", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote nightly to main, keeping the working folders out of "
            "main's tree. Dry run unless --execute is given; never pushes."),
    )
    parser.add_argument(
        "--repo", default=".",
        help="path inside the repository to promote (default: cwd)")
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"branch the work is on (default: {DEFAULT_SOURCE})")
    parser.add_argument(
        "--target", default=DEFAULT_TARGET,
        help=f"branch the public lands on (default: {DEFAULT_TARGET})")
    parser.add_argument(
        "--folder", action="append", dest="folders", metavar="PATH",
        help=("a directory to keep out of the target branch; repeatable. "
              "Defaults to " + ", ".join(WORKING_FOLDERS)))
    parser.add_argument(
        "--execute", action="store_true",
        help="actually perform the promotion (default is a dry run)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    folders = tuple(args.folders) if args.folders else WORKING_FOLDERS
    try:
        return promote(
            Path(args.repo).resolve(),
            source=args.source,
            target=args.target,
            folders=folders,
            execute=args.execute,
        )
    except PromotionError as exc:
        print(f"promote_to_main: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
