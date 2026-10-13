"""``published_archives`` has three answers, and two of them look alike.

A set of names means "the hub answered and these are published".  An EMPTY
set means "the hub answered and nothing is published".  ``None`` means "the
hub did not answer", and the picker draws those two differently on purpose:
an empty set greys every row out, while None leaves them enabled and says the
list could not be checked.  Collapsing the second into the first gives a user
on a hotel network a picker that offers nothing and explains nothing.

The hub is never contacted here.  ``huggingface_hub`` is replaced with a
module whose ``HfApi`` does what a real one does in each of the three cases,
including the older releases whose ``list_repo_files`` has no ``timeout``
parameter -- spaCR supports those and calls again without it, which is a
retry that can itself fail.
"""
from __future__ import annotations

import sys
import types

import pytest

from spacr import screen_data


def _hub(monkeypatch, list_repo_files):
    """Install a fake ``huggingface_hub`` whose HfApi lists what we say."""
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def list_repo_files(self, repo, **kwargs):
            return list_repo_files(repo, **kwargs)

    module.HfApi = HfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def test_the_published_tars_come_back_and_nothing_else_does(monkeypatch):
    """Only ``.tar`` archives; the repo also holds a README and a .gitattributes."""
    _hub(monkeypatch, lambda repo, **kw: [
        "README.md", ".gitattributes", "plate1.tar", "crops/plate2.tar",
        "plate3.tar.sha256",
    ])

    assert screen_data.published_archives("some/repo") == {
        "plate1.tar", "crops/plate2.tar",
    }


def test_a_repository_that_publishes_nothing_is_an_empty_set(monkeypatch):
    """Empty is an ANSWER: the picker greys the rows out and is right to."""
    _hub(monkeypatch, lambda repo, **kw: ["README.md"])

    assert screen_data.published_archives("some/repo") == set()


def test_a_hub_that_does_not_answer_is_none_and_not_an_empty_set(monkeypatch):
    """Offline, rate-limited or renamed: we cannot tell, and we say so."""
    def refuse(repo, **kwargs):
        raise OSError("Network is unreachable")

    _hub(monkeypatch, refuse)

    assert screen_data.published_archives("some/repo") is None


def test_an_older_hub_without_a_timeout_is_asked_again_without_one(monkeypatch):
    """TypeError from the keyword, not from the network: retry, do not give up.

    ``list_repo_files`` gained ``timeout`` partway through huggingface_hub's
    history.  An installation on the older release would otherwise report
    every screen as unpublished, and the picker would offer nothing.
    """
    calls = []

    def older(repo, **kwargs):
        calls.append(kwargs)
        if "timeout" in kwargs:
            raise TypeError("unexpected keyword argument 'timeout'")
        return ["plate1.tar"]

    _hub(monkeypatch, older)

    assert screen_data.published_archives("some/repo") == {"plate1.tar"}
    assert [("timeout" in kw) for kw in calls] == [True, False]


def test_a_retry_that_fails_too_is_still_i_do_not_know(monkeypatch):
    """The second call can fail for the first call's real reason.

    An offline host raises TypeError from neither -- but a host that is BOTH
    on an old huggingface_hub AND offline reaches the retry and fails there,
    and the answer is still None rather than a traceback out of a dialog.
    """
    def older_and_offline(repo, **kwargs):
        if "timeout" in kwargs:
            raise TypeError("unexpected keyword argument 'timeout'")
        raise OSError("Network is unreachable")

    _hub(monkeypatch, older_and_offline)

    assert screen_data.published_archives("some/repo") is None


def test_no_huggingface_hub_at_all_is_i_do_not_know(monkeypatch):
    """The hub is an optional dependency; the picker still opens without it."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert screen_data.published_archives("some/repo") is None


def test_the_timeout_is_passed_so_a_slow_network_cannot_hold_the_dialog(
    monkeypatch,
):
    """A picker that waits on a default socket timeout looks frozen."""
    seen = {}

    def record(repo, **kwargs):
        seen.update(kwargs)
        return []

    _hub(monkeypatch, record)
    screen_data.published_archives("some/repo", timeout=0.25)

    assert seen["timeout"] == 0.25
    assert seen["repo_type"] == "dataset"
