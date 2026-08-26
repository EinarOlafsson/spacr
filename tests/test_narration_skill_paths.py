"""The narration skill's own repository references have to resolve.

A skill is only usable by an agent that has nothing but the skill. Step 1 of
`natural-voice-narration` is "read this instruction file", so a path that has
since moved stops the workflow at its first line -- silently, because nothing
executes a skill.

Two path spaces live in that skill and only one of them is checkable here:

    repository-relative   `instructions/...`, `docs/source/_extra/tutorials/`
                          -- these must exist in this checkout.
    workspace-relative    `tools/`, `catalog/`, `production/`, `web/`, and the
                          narration `tests/` -- these belong to the separate
                          tutorial publishing workspace the skill tells the
                          agent to locate, and are deliberately absent here.

So this pins the first kind and states the second, rather than asserting that
every backticked path is a file.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / ".claude" / "skills" / "natural-voice-narration"

#: Files inside the skill that name repository paths.
SKILL_FILES = (
    SKILL / "SKILL.md",
    SKILL / "references" / "naturalness-contract.md",
    SKILL / "references" / "diagnostics-and-release.md",
)

#: Prefixes that are repository-relative and therefore checkable.
REPO_PREFIXES = ("instructions/", "docs/")


def _named_paths(text):
    """Backticked tokens that look like a repository path."""
    out = []
    for chunk in text.split("`")[1::2]:
        token = chunk.strip().rstrip(".,;:")
        if token.startswith(REPO_PREFIXES):
            out.append(token)
    return out


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda p: p.name)
def test_every_repository_path_the_skill_names_exists(skill_file):
    assert skill_file.is_file(), f"{skill_file} is missing from the skill"
    missing = [p for p in _named_paths(skill_file.read_text(encoding="utf-8"))
               if not (REPO / p).exists()]
    assert not missing, (
        f"{skill_file.name} sends an agent to paths that are not in this "
        f"checkout: {missing}. Either restore them or update the skill -- an "
        f"agent following the skill cannot tell a moved file from a typo.")


def test_the_instruction_it_opens_with_is_the_one_that_exists():
    """The whole workflow hangs off this one file."""
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "48_optimize_the_tutorials.txt" in text
    assert (REPO / "instructions" / "done"
            / "48_optimize_the_tutorials.txt").is_file()
    assert "instructions/open/48_optimize_the_tutorials.txt" not in text


def test_the_workspace_relative_paths_are_declared_as_such():
    """`tools/pronunciation.py` is not a file in this repository, and the
    source map has to keep saying which workspace it is relative to."""
    contract = (SKILL / "references"
                / "naturalness-contract.md").read_text(encoding="utf-8")
    assert "Relative to that publishing workspace" in contract
    assert not (REPO / "tools" / "pronunciation.py").exists()
