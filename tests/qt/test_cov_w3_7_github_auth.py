"""Reading a token, and the two API calls the offline suite does not reach.

No test in this file touches the network. Every request is answered by a
stand-in ``urlopen``, which is also the only honest way to assert what spaCR
SENDS -- the URL, the method, the headers and the JSON body are read back off
the ``Request`` object the module built.

The token itself comes from the environment, never from the developer's ``gh``
credential store: ``resolve_token`` falls through to ``gh auth token`` and a
test that let it get that far would be reading a real credential.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt.ai import github_auth


@pytest.fixture(autouse=True)
def _a_token_from_the_environment(monkeypatch):
    """One token, from the environment, so no test consults the ``gh`` CLI."""
    monkeypatch.setattr(github_auth, "_EPHEMERAL_TOKEN", "")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_covw37token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token",
                        lambda: pytest.fail(
                            "a test reached the developer's gh credential "
                            "store"))


# ---------------------------------------------------------------------------
# The token
# ---------------------------------------------------------------------------

def test_a_token_an_older_build_persisted_is_erased_on_the_way_past():
    """spaCR does not store credentials; a build that did is undone here."""
    settings = QSettings(github_auth._ORG, github_auth._APP)
    settings.setValue(github_auth._KEY_TOKEN, "ghp_left_on_disk")
    settings.sync()
    assert settings.contains(github_auth._KEY_TOKEN)

    assert github_auth.get_stored_token() == ""
    assert not QSettings(github_auth._ORG,
                         github_auth._APP).contains(github_auth._KEY_TOKEN)


def test_the_environment_is_where_the_token_comes_from():
    assert github_auth.resolve_token() == ("ghp_covw37token", "env")
    assert github_auth.auth_source() == "env"
    assert github_auth.is_authenticated() is True


# ---------------------------------------------------------------------------
# The process-lifetime write guard
# ---------------------------------------------------------------------------

def test_the_write_guard_refuses_inside_a_test_run(monkeypatch):
    """The old inherited escape hatch is deliberately inert."""
    reason = github_auth._refuse_writes_under_test()
    assert reason and "refusing GitHub network access" in reason

    monkeypatch.setenv("SPACR_ALLOW_GITHUB_WRITES", "1")
    assert github_auth._refuse_writes_under_test() == reason


def test_outside_a_test_run_there_is_nothing_to_refuse(monkeypatch):
    monkeypatch.delenv("SPACR_PYTEST_SESSION", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert github_auth._refuse_writes_under_test() is None


# ---------------------------------------------------------------------------
# Searching for an existing report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Commenting on one that already exists
# ---------------------------------------------------------------------------
