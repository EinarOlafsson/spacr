"""A name that worked on the command line must not become a typo.

WHAT THIS IS ABOUT. Six modules answered to ``spacr-run <key>`` because they
were rows in the GUI's app table and the CLI takes names from there. When
``571b6e77c`` and ``00f166a7f`` folded them into host screens, the rows went
and the names went with them -- so ``spacr-run outliers``, which had worked,
started answering::

    unknown module 'outliers'.
      Run 'spacr-run --list' to see every module that can run headless.

That message says "you typed it wrong". The user did not type it wrong; the
name was removed. This file holds the difference.

IT DOES NOT ASSERT THAT THEY RUN. Five of the six are interactive views with
no batch equivalent, and inventing one to keep a name alive would be worse
than saying so. What it asserts is that the answer EXPLAINS, and names where
the thing went.
"""
from __future__ import annotations

import pytest

from spacr import cli


#: The six, with the host each folded into. The host key is asserted to be a
#: real module or app key, so this test fails if a host is renamed and these
#: messages start pointing at nothing.
FOLDED_OFF_THE_CLI = {
    "outliers": "qc_dashboard",
    "control_chart": "qc_dashboard",
    "trellis": "graph_builder",
    "feature_explorer": "classify_merged",
    "import_images": "foreign",
    "regression_diagnostics": "regression",
}


@pytest.mark.parametrize("key", sorted(FOLDED_OFF_THE_CLI))
def test_the_name_is_not_reported_as_unknown(key):
    """The one thing that must not happen: "unknown module"."""
    message = cli._unknown_module_message(key)
    assert "unknown module" not in message.lower(), (
        f"'{key}' worked on the command line before the fold; reporting it "
        f"as unknown tells the user they made a typo:\n{message}")


@pytest.mark.parametrize("key,host", sorted(FOLDED_OFF_THE_CLI.items()))
def test_the_message_says_where_the_thing_went(key, host):
    """Naming the host is the whole value of the message.

    "It is gone" leaves the user with nothing to do next. "It folded into
    the QC Dashboard" is a place to look.
    """
    message = cli._unknown_module_message(key)
    assert host in message, (
        f"the message for '{key}' does not name its host '{host}':\n{message}")


def test_every_named_host_still_exists():
    """The hosts named above must be real, or the advice is a dead end.

    A message that sends the user to a screen that no longer exists is worse
    than the error it replaced, and renames happen -- this file exists
    because one did.
    """
    from spacr.qt.app import APPS

    keys = set()
    for app in APPS:
        key = app[0] if isinstance(app, (list, tuple)) else getattr(
            app, "key", None)
        if key:
            keys.add(key)
    known = keys | set(cli.MODULES) | set(cli.ALIASES)
    missing = sorted({h for h in FOLDED_OFF_THE_CLI.values()
                      if h not in known})
    assert not missing, (
        f"these hosts are named in a CLI message but no longer exist: "
        f"{missing}")
