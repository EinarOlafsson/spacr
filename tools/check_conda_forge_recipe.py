#!/usr/bin/env python
"""Fail if the conda-forge recipe would not build the release being made.

WHY THIS EXISTS. The feedstock updates itself: `bot.version_updates.sources`
is `[pypi]` so it watches PyPI, and `bot.automerge: version` merges the
bot's pull request as soon as CI is green. A release reaches conda-forge
with no human step, and this tool does not change that.

WHAT IT REPORTS is entry-point drift. The bot's PR rewrites `version` and
`sha256` and nothing else, so a console script this package has dropped is
still declared by the recipe long after the module is gone.

AND IT NEVER BLOCKS, which was established rather than assumed. The
published package's `info/link.json` carries the recipe's entry points as
INSTALL-TIME metadata: conda writes the wrapper scripts when a user
installs, and nothing imports the module during the build. The recipe's own
test section names five commands and `spacr-run --list`, none of them the
stale ones. So a dropped module does not fail the build, conda-forge keeps
updating, and what ships is a command that raises ModuleNotFoundError when
somebody runs it.

That is worth fixing and is not worth refusing to release over -- blocking
a release to protect a channel that was never going to stall would be the
more expensive error.

    python tools/check_conda_forge_recipe.py            # against this tree
    python tools/check_conda_forge_recipe.py --sdist X  # against an sdist
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

RECIPE_URL = ("https://raw.githubusercontent.com/conda-forge/"
              "spacr-feedstock/main/recipe/recipe.yaml")
ROOT = Path(__file__).resolve().parent.parent

ENTRY = re.compile(r"-\s+([A-Za-z0-9_-]+)\s*=\s*([A-Za-z0-9_.]+):([A-Za-z0-9_]+)")


def recipe_entry_points(text: str) -> dict:
    block = re.search(r"entry_points:\s*\n((?:\s+-\s+.*\n)+)", text)
    if not block:
        return {}
    return {m.group(1): f"{m.group(2)}:{m.group(3)}"
            for m in ENTRY.finditer(block.group(1))}


def setup_entry_points(text: str) -> dict:
    block = re.search(r"'console_scripts'\s*:\s*\[(.*?)\]", text, re.S)
    if not block:
        return {}
    return dict(re.findall(
        r"'([A-Za-z0-9_-]+)\s*=\s*([A-Za-z0-9_.]+:[A-Za-z0-9_]+)'",
        block.group(1)))


def _named(dotted: str) -> set:
    """The dotted name, plus the PACKAGE name when it is an `__init__`.

    `spacr = spacr.qt:run` imports the package `spacr.qt`, which lives in
    `spacr/qt/__init__.py`. Stripping only the suffix gives
    `spacr.qt.__init__`, so every package entry point read as missing and
    the checker would have blocked releases it should have passed.
    """
    if dotted.endswith(".__init__"):
        return {dotted, dotted[: -len(".__init__")]}
    return {dotted}


def modules_in_tree(root: Path) -> set:
    found = set()
    for path in root.rglob("spacr/**/*.py"):
        found |= _named(str(path.relative_to(root)).replace("/", ".")[:-3])
    return found


def modules_in_sdist(path: Path) -> set:
    found = set()
    with tarfile.open(path) as archive:
        for name in archive.getnames():
            parts = name.split("/", 1)
            if len(parts) == 2 and parts[1].endswith(".py"):
                found |= _named(parts[1][:-3].replace("/", "."))
    return found


def setup_py_in_sdist(path: Path) -> str:
    """The RELEASE's own setup.py, not this working tree's.

    Comparing a released sdist against the tree's setup.py reports drift
    that is not real for that release -- which is how a checker teaches
    people to ignore it."""
    with tarfile.open(path) as archive:
        for name in archive.getnames():
            if name.count("/") == 1 and name.endswith("/setup.py"):
                handle = archive.extractfile(name)
                if handle is not None:
                    return handle.read().decode("utf-8", "replace")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", help="check against this sdist instead of "
                                        "the working tree")
    parser.add_argument("--recipe", help="a local recipe instead of fetching "
                                         "the one conda-forge has")
    args = parser.parse_args()

    if args.recipe:
        recipe_text = Path(args.recipe).read_text()
        where = args.recipe
    else:
        with urllib.request.urlopen(RECIPE_URL, timeout=60) as response:
            recipe_text = response.read().decode("utf-8")
        where = RECIPE_URL

    declared = recipe_entry_points(recipe_text)
    if not declared:
        print(f"the recipe at {where} declares no entry points; "
              f"nothing can drift")
        return 0

    if args.sdist:
        available = modules_in_sdist(Path(args.sdist))
        subject = args.sdist
    else:
        available = modules_in_tree(ROOT)
        subject = "the working tree"

    broken = {name: target for name, target in declared.items()
              if target.split(":")[0] not in available}
    source = (setup_py_in_sdist(Path(args.sdist)) if args.sdist
              else (ROOT / "setup.py").read_text())
    ours = setup_entry_points(source or (ROOT / "setup.py").read_text())
    absent = {k: v for k, v in ours.items() if k not in declared}
    disagree = {k: (ours[k], declared[k])
                for k in set(ours) & set(declared) if ours[k] != declared[k]}

    print(f"recipe:  {where}")
    print(f"checked against: {subject}")
    print(f"  entry points the recipe declares: {len(declared)}")
    print(f"  console scripts setup.py declares: {len(ours)}")

    if not (broken or absent or disagree):
        print("OK -- the recipe builds this release")
        return 0

    # REPORTED, NEVER BLOCKING -- see the module docstring. The build does
    # not import these modules, so none of what follows can stop a release.

    if broken:
        print("\nCONDA USERS WILL GET BROKEN COMMANDS. The recipe declares "
              "these, conda will create a wrapper for each at install time, "
              "and the module behind them is not in this release:")
        for name, target in sorted(broken.items()):
            print(f"    {name} = {target}")
    if absent:
        print("\nConda users would not get these commands:")
        for name, target in sorted(absent.items()):
            print(f"    {name} = {target}")
    if disagree:
        print("\nThe same command points at different code:")
        for name, (mine, theirs) in sorted(disagree.items()):
            print(f"    {name}: setup.py {mine}  recipe {theirs}")

    print("\nTHE FIX is a pull request to conda-forge/spacr-feedstock "
          "editing recipe/recipe.yaml. It can land before or after the "
          "version bump -- conda-forge keeps updating either way, so this "
          "is about what the commands DO rather than about the channel "
          "falling behind.")
    print("\n(Not blocking. The conda build does not import these modules, "
          "so none of the above can fail it.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
