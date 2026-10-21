"""``published_archives`` has three answers, and two of them look alike.

A set of names means "the hub answered and these are published".  An EMPTY
set means "the hub answered and nothing is published".  ``None`` means "the
hub did not answer", and the picker draws those two differently on purpose:
an empty set greys every row out, while None leaves them enabled and says the
list could not be checked.  Collapsing the second into the first gives a user
on a hotel network a picker that offers nothing and explains nothing.

The hub is never contacted here. Fake metadata responses preserve the three
outcomes, and the installed client's real request path checks that the
timeout reaches its transport rather than an unsupported API keyword.
"""
from __future__ import annotations

import sys
import types

import pytest

from spacr import screen_data


def _hub(monkeypatch, inventory):
    """Install a fake metadata endpoint with the supported timeout signature."""
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def repo_info(self, repo, *, repo_type, timeout):
            names = inventory(repo, repo_type=repo_type, timeout=timeout)
            return types.SimpleNamespace(siblings=(
                None if names is None else
                [types.SimpleNamespace(rfilename=name) for name in names]))

        def list_repo_files(self, *args, **kwargs):
            pytest.fail("inventory must not fall back to an unbounded file listing")

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


def test_a_client_error_does_not_retry_without_the_timeout(monkeypatch):
    calls = []

    def refuse(repo, **kwargs):
        calls.append(kwargs)
        raise TypeError("invalid metadata response")

    _hub(monkeypatch, refuse)

    assert screen_data.published_archives("some/repo", timeout=0.25) is None
    assert calls == [{"repo_type": "dataset", "timeout": 0.25}]


@pytest.mark.parametrize("siblings", [None, [None]])
def test_missing_or_malformed_inventory_is_unknown_not_a_known_empty_repo(monkeypatch, siblings):
    _hub(monkeypatch, lambda repo, **kwargs: siblings)

    assert screen_data.published_archives("some/repo") is None


def test_no_huggingface_hub_at_all_is_i_do_not_know(monkeypatch):
    """The hub is an optional dependency; the picker still opens without it."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert screen_data.published_archives("some/repo") is None


def test_the_timeout_is_passed_so_a_slow_network_cannot_hold_the_dialog(
    monkeypatch,
):
    from huggingface_hub import hf_api

    seen = []

    class Session:
        def get(self, url, *, headers, timeout, params):
            seen.append((url, timeout))
            raise TimeoutError("bounded metadata request expired")

    monkeypatch.setattr(hf_api, "get_session", lambda: Session())
    assert screen_data.published_archives("some/repo", timeout=0.25) is None
    assert len(seen) == 1
    assert seen[0][0].endswith("/api/datasets/some/repo")
    assert seen[0][1] == 0.25
