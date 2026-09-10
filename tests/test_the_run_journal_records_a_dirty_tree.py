"""What the run journal records about the code that produced a run.

Both functions here exist so a run can be traced back to the exact source that
made it. The uncovered paths are the ones that matter most for that: a working
tree with uncommitted edits, and a distribution whose metadata will not name
itself. A journal that recorded a clean commit for a dirty tree would point a
future reader at code that never ran.
"""
from __future__ import annotations

import importlib.metadata
import subprocess
import types

import pytest


# ---------------------------------------------------------------------------
# _git_hash — the dirty marker
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def _git_answers(monkeypatch, *, head, status):
    """Make the two git calls answer with ``head`` and ``status``."""
    from spacr import run_journal

    def fake_run(argv, *_args, **_kwargs):
        if "rev-parse" in argv:
            return head
        return status

    monkeypatch.setattr(run_journal.subprocess, "run", fake_run)


def test_a_working_tree_with_edits_is_recorded_as_dirty(monkeypatch):
    """Line 159 and arc 158 -> 159.

    THE POINT OF THE WHOLE FUNCTION. A journal that recorded a bare commit
    hash for a tree with uncommitted changes would send a future reader to
    code that never ran -- and this repository's own working tree is dirty
    most of the time, so the marker is the common case rather than the rare
    one.
    """
    from spacr.run_journal import _git_hash

    _git_answers(monkeypatch,
                 head=_Result("abc1234\n"),
                 status=_Result(" M spacr/measure.py\n"))

    assert _git_hash() == "abc1234+dirty"


def test_a_clean_working_tree_is_recorded_as_the_bare_hash(monkeypatch):
    """The other side, so the marker above is visibly conditional."""
    from spacr.run_journal import _git_hash

    _git_answers(monkeypatch, head=_Result("abc1234\n"), status=_Result("\n"))

    assert _git_hash() == "abc1234"


def test_a_tree_that_is_not_a_checkout_records_nothing(monkeypatch):
    """A non-zero rev-parse is None, not an empty string.

    An installed wheel is not a checkout, and "" would be written into the
    journal as though a hash had been read and had come back blank.
    """
    from spacr.run_journal import _git_hash

    _git_answers(monkeypatch,
                 head=_Result("", returncode=128), status=_Result(""))

    assert _git_hash() is None


def test_a_git_that_will_not_run_records_nothing(monkeypatch):
    """The except: no git on PATH is an ordinary state, not a failed run."""
    from spacr import run_journal

    def refuse(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(run_journal.subprocess, "run", refuse)

    assert run_journal._git_hash() is None


# ---------------------------------------------------------------------------
# _installed_packages — a distribution that will not name itself
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _uncached():
    """``_installed_packages`` is lru_cached, so every test must start cold.

    Without this the first test's answer is returned to every later one, and
    they pass or fail depending on collection order -- which is exactly the
    kind of test that goes green while measuring nothing.
    """
    from spacr.run_journal import _installed_packages

    _installed_packages.cache_clear()
    yield
    _installed_packages.cache_clear()


class _Dist:
    def __init__(self, name, version="1.0"):
        self.metadata = {"Name": name} if name is not None else {}
        self.version = version


def test_a_distribution_with_no_name_is_passed_over(monkeypatch):
    """Arc 177 -> 175: the loop goes round rather than keying on "".

    A broken or partially-installed dist-info has no Name, and it is common
    enough in a long-lived conda environment. Keying on the empty string would
    put an entry called "" in the journal, and the next such dist would
    overwrite it -- so the count of packages would silently depend on how many
    were broken.
    """
    from spacr import run_journal

    monkeypatch.setattr(
        run_journal.importlib.metadata, "distributions",
        lambda: [_Dist("numpy", "2.0.0"), _Dist(None), _Dist("  "),
                 _Dist("Pillow", "11.0.0")])

    packages = run_journal._installed_packages()

    assert packages == {"numpy": "2.0.0", "pillow": "11.0.0"}
    assert "" not in packages


def test_a_distribution_with_no_version_is_recorded_as_unknown(monkeypatch):
    """"unknown" rather than None, so the journal is JSON-safe and readable."""
    from spacr import run_journal

    monkeypatch.setattr(
        run_journal.importlib.metadata, "distributions",
        lambda: [_Dist("weird-package", None)])

    assert run_journal._installed_packages() == {"weird-package": "unknown"}


def test_names_are_normalised_and_sorted(monkeypatch):
    """The normalisation the docstring promises, which makes lookups stable."""
    from spacr import run_journal

    monkeypatch.setattr(
        run_journal.importlib.metadata, "distributions",
        lambda: [_Dist("Zope_Interface", "5.0"), _Dist("Aaa_Bbb", "1.0")])

    packages = run_journal._installed_packages()

    assert list(packages) == ["aaa-bbb", "zope-interface"]


def test_an_environment_that_cannot_be_enumerated_records_nothing(monkeypatch):
    """The except: a warning and an empty dict, never a failed run.

    The journal is written beside a run that has already finished. Losing the
    package list is a gap in the record; raising would lose the record.
    """
    from spacr import run_journal

    def refuse():
        raise RuntimeError("the metadata directory is unreadable")

    monkeypatch.setattr(run_journal.importlib.metadata, "distributions", refuse)

    assert run_journal._installed_packages() == {}
