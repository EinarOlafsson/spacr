"""The README's command reference matches the commands that exist.

Added 2026-08-31 with the reference itself. The README has twice shipped a
module grid describing a layout the GUI no longer had, and a command list
rots the same way and more quietly: an entry point renamed in ``setup.py``
leaves the README telling users to type something that does not exist,
and a new one leaves it undiscoverable.

Both directions are asserted, because they fail differently:

* a command the README names but ``setup.py`` does not declare is an
  instruction that cannot work;
* an entry point declared but never mentioned is a feature nobody finds.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
SETUP = ROOT / "setup.py"


def declared_commands() -> set:
    """Console-script names from ``setup.py``, read as text.

    Text rather than import: importing setup.py runs it, and the point of
    the check is what the packaging metadata SAYS.
    """
    return set(re.findall(r"'([a-z][a-z0-9-]*)=spacr[.:]", SETUP.read_text(
        encoding="utf-8")))


def commands_named_in(text: str) -> set:
    """spaCR command names the README tells the reader to type or names."""
    found = set(re.findall(r"^\s{3}([a-z][a-z0-9-]*)\b", text, re.M))
    found |= set(re.findall(r"``(spacr[a-z0-9-]*|safespacr|spaceout)``", text))
    return {name for name in found
            if name.startswith(("spacr", "safespacr", "spaceout"))}


def test_the_readme_names_no_command_that_does_not_exist():
    """Every command in the README is a real entry point.

    This is the half that produces a user typing something and getting
    'command not found' -- the worst first impression a README can make.
    """
    invented = sorted(commands_named_in(README.read_text(encoding="utf-8"))
                      - declared_commands())
    assert not invented, (
        f"the README tells users to run commands setup.py does not "
        f"declare: {invented}")


def test_every_command_that_exists_is_in_the_readme():
    """Every entry point is documented somewhere in the README.

    Aliases count: ``spacr-qt`` and ``spacr-nightly`` start the same
    application as ``spacr``, and they are named so a reader who finds one
    in an old script can tell what it is.
    """
    text = README.read_text(encoding="utf-8")
    undocumented = sorted(name for name in declared_commands()
                          if name not in text)
    assert not undocumented, (
        f"these entry points ship but the README never mentions them: "
        f"{undocumented}")


@pytest.mark.parametrize("phrase", [
    "git clone https://github.com/EinarOlafsson/spacr.git",
    "pip install -e .",
])
def test_the_source_install_is_copy_pasteable(phrase):
    """Asked for as "so users can just copy past".

    The two lines somebody actually needs are pinned literally. A source
    install section that paraphrases the clone command is one the reader
    has to translate before using.
    """
    assert phrase in README.read_text(encoding="utf-8")
