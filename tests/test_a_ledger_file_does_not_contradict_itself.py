"""A feature file's header must not say "not started" over a finished file.

THE INDEX ALREADY WARNS ABOUT THIS IN PROSE -- "THE TRAILING NOTES AT THE END
OF EACH FILE ARE THE CURRENT STATE. The header often says 'not started' when it
is done -- eight files have been wrong about themselves this week." A warning
is not a check, and on 2026-09-13 thirteen files were wrong about themselves.

READING DOES NOT SCALE AND THE LEDGER PROVES IT TWICE, in the two files that
noticed and did not fix:

  * `119_the_regression_figure_surface.txt` closes with "Filed to done/ after
    re-reading the file rather than its header, which still said 'not
    started'." Somebody hit this exact problem, wrote down that the header was
    wrong, moved the file, and left the header. The sentence describing the
    mistake sat directly above the mistake for four weeks.
  * `69_standardise_live_views_and_searches.txt` closes with "This file said
    'not started'. It was the seventh instruction this week to be wrong about
    its own state." Same shape, same outcome.

WHAT THIS CHECKS IS DELIBERATELY NARROW. Only an UNAMBIGUOUS completion
sentence counts -- "INSTRUCTION n IS COMPLETE", "STATUS: DONE", "completes
instruction n", "STILL OWED: nothing". Prose describing finished work does not,
because "we built the analyser" is compatible with three parts still open and a
test that guessed would be turned off within a week.

So this cannot catch every stale header. It catches the ones where the file
states its own completion in words chosen to be unmistakable, and those are
exactly the ones where a reader is most entitled to expect the header to agree.
"""
from __future__ import annotations

import pathlib
import re

import pytest

FEATURES = pathlib.Path(__file__).resolve().parent.parent / "features"

#: A Status line that OPENS with a not-started claim. Anchored at the start so
#: a header saying "DONE ... this said 'not started' until" does not match its
#: own history -- forty-three files mention the phrase, six were wrong.
OPENS_NOT_STARTED = re.compile(
    r"(?i)^\s*(not started|not diagnosed|not begun|unstarted"
    r"|filed[^.\n]*,\s*not started)\b")

#: Sentences a file uses to declare itself finished, and nothing weaker.
DECLARES_COMPLETE = re.compile(
    r"(?i)(INSTRUCTION \d+ IS COMPLETE|STATUS:\s*DONE"
    r"|completes instruction \d+|STILL OWED:\s*nothing"
    r"|THIS ITEM IS (?:DONE|CLOSED))")

#: How much of the end of the file counts as "the trailing notes".
TAIL_CHARS = 4000


def _ledger_files():
    for folder in ("new", "future"):
        for path in sorted((FEATURES / folder).glob("*.txt")):
            yield path


def _status(text):
    match = re.search(r"^Status:\s*(.*)$", text, re.M)
    return match.group(1) if match else None


def test_there_are_ledger_files_to_check():
    """The sweep must not pass because it swept nothing.

    Both guards this file's docstring describes were disarmed by a folder
    rename -- one globbed `instructions/`, one checked a prefix its fixture
    never created -- and both passed on the empty set for four weeks.
    """
    files = list(_ledger_files())
    assert len(files) > 300, f"only {len(files)} ledger files found; wrong path?"


@pytest.mark.parametrize("path", list(_ledger_files()), ids=lambda p: p.name)
def test_a_not_started_header_has_no_completion_sentence_below_it(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    status = _status(text)
    if status is None or not OPENS_NOT_STARTED.match(status):
        return
    declared = DECLARES_COMPLETE.search(text[-TAIL_CHARS:])
    assert declared is None, (
        f"{path.name} opens its Status with {status.strip()[:60]!r} while its "
        f"trailing notes say {declared.group(0)!r}. The trailing notes are the "
        f"current state; correct the header."
    )
