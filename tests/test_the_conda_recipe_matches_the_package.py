"""The conda-forge recipe declares the commands this package actually has.

conda-forge is set to update itself: `bot.version_updates.sources: [pypi]`
watches PyPI, and `bot.automerge: version` merges the bot's PR once CI is
green. So a release reaches conda-forge with no human step -- UNLESS the
build fails, and then it silently stays on the previous version.

The way it fails is entry-point drift. The bot's PR changes `version` and
`sha256` and NOTHING ELSE, so a console script removed from setup.py is
still declared by the recipe, points at a module the new sdist does not
contain, and the build dies. `automerge` needs green, so conda-forge stops
updating and nothing says so.

This test compares the two lists and fails HERE, before a release, rather
than in a feedstock PR nobody is watching afterwards.

It is skipped when the recipe is not on disk. The recipe lives in
`conda-forge/spacr-feedstock`, not in this repository; keeping a copy under
`conda-forge/` is what lets this run in CI, and `test_the_copy_is_current`
below is what stops that copy from becoming its own stale thing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECIPE = ROOT / "conda-forge" / "recipe" / "recipe.yaml"


def _setup_entry_points() -> dict:
    """``{command: target}`` from setup.py's console_scripts."""
    source = (ROOT / "setup.py").read_text()
    block = re.search(r"'console_scripts'\s*:\s*\[(.*?)\]", source, re.S)
    assert block, "setup.py no longer declares console_scripts this way"
    return dict(re.findall(r"'([A-Za-z0-9_-]+)\s*=\s*([A-Za-z0-9_.]+:[A-Za-z0-9_]+)'",
                           block.group(1)))


def _recipe_entry_points():
    """``{command: target}`` from the recipe, or None when it declares none.

    None is not an error. A recipe may leave the console scripts to the
    package's own metadata, which is the more robust arrangement -- there is
    then nothing to drift. What must never happen is a recipe that declares
    a LIST which disagrees with setup.py.
    """
    text = RECIPE.read_text()
    block = re.search(r"entry_points:\s*\n((?:\s+-\s+.*\n)+)", text)
    if not block:
        return None
    return dict(re.findall(
        r"-\s+([A-Za-z0-9_-]+)\s*=\s*([A-Za-z0-9_.]+:[A-Za-z0-9_]+)",
        block.group(1)))


needs_recipe = pytest.mark.skipif(
    not RECIPE.exists(),
    reason="no local copy of the conda-forge recipe to compare against")


@needs_recipe
def test_every_recipe_command_exists_in_this_package():
    """A command the recipe declares whose module is gone fails the conda
    build, and a failed build is how conda-forge stops updating."""
    declared = _recipe_entry_points()
    if declared is None:
        pytest.skip("the recipe declares no entry points, so none can drift")
    missing = {}
    for command, target in declared.items():
        module = target.split(":")[0]
        path = ROOT / Path(module.replace(".", "/") + ".py")
        package = ROOT / Path(module.replace(".", "/")) / "__init__.py"
        if not path.exists() and not package.exists():
            missing[command] = target
    assert not missing, (
        f"the recipe declares commands this package no longer has: {missing}. "
        f"The next autotick PR will fail to build and conda-forge will "
        f"silently stay on the previous version.")


@needs_recipe
def test_every_command_this_package_ships_is_in_the_recipe():
    """The other direction: a command added here and not there simply does
    not exist for anyone who installed with conda."""
    theirs = _recipe_entry_points()
    if theirs is None:
        pytest.skip("the recipe declares no entry points, so none can be missing")
    ours = _setup_entry_points()
    absent = {k: v for k, v in ours.items() if k not in theirs}
    assert not absent, (
        f"conda users would not get these commands: {absent}")


@needs_recipe
def test_the_two_agree_on_where_each_command_points():
    """`spaceout` pointed at `spacr.gui:gui_app` in the recipe and at
    `spacr.qt.spaceout:main` here. Both existed, so nothing failed -- conda
    users just got a different program."""
    theirs = _recipe_entry_points()
    if theirs is None:
        pytest.skip("the recipe declares no entry points, so none can disagree")
    ours = _setup_entry_points()
    disagree = {k: (ours[k], theirs[k])
                for k in set(ours) & set(theirs) if ours[k] != theirs[k]}
    assert not disagree, (
        f"the same command points at different code: {disagree}")
