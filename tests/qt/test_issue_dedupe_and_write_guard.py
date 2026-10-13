"""One crash must file one issue, and no test may reach the live tracker.

On 2026-08-11 the same `ml_analyze` traceback was auto-filed TEN times --
issues #79, #80, #81 and #84 through #90 -- every one carrying the identical
fingerprint `500e6c`. `_traceback_hash` exists precisely so the same bug
hashes the same across runs and machines, and nothing consumed it: the hash
was written into the body and never read back.

Separately, `[auto 54a0e8] [mask] Error: boom` (#75) is a TEST FIXTURE that
reached the public tracker. spaCR posts whenever a token is resolvable, and
on a developer machine the `gh` CLI supplies one, so any test reaching a
write path without mocking files a real issue.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import github_auth, issue_report


@pytest.fixture
def offline_transport(monkeypatch):
    """Admit the report flow only through an in-process dead transport.

    Unlike the former environment escape hatch, this replacement cannot be
    inherited by a subprocess.  If it survives until fixture teardown, it is
    still a function that fails instead of a socket.
    """
    def _no_network(*args, **kwargs):
        pytest.fail("an offline issue-report test reached HTTP")

    monkeypatch.setattr(github_auth, "_HTTP_OPEN", _no_network)



TB = ("Traceback (most recent call last):\n"
      '  File "spacr/ml.py", line 4346, in ml_analysis\n'
      "    X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
      "ValueError: With n_samples=0, test_size=0.2 the train set will be empty")


# ---------------------------------------------------------------------------
# the fingerprint is now returned, so it can be searched on
# ---------------------------------------------------------------------------

def test_build_report_returns_the_fingerprint():
    report = issue_report.build_report(TB, active_app="ml_analyze")
    assert report["fingerprint"], "the hash is written and never handed back"
    assert report["fingerprint"] in report["body"]


def test_the_same_bug_hashes_the_same_after_lines_move():
    """An unrelated edit above the crash must not fork the fingerprint."""
    shifted = TB.replace("line 4346", "line 4400")
    assert (issue_report.build_report(shifted)["fingerprint"]
            == issue_report.build_report(TB)["fingerprint"])


def test_a_different_exception_from_the_same_frame_hashes_differently():
    other = TB.replace("ValueError", "KeyError")
    assert (issue_report.build_report(other)["fingerprint"]
            != issue_report.build_report(TB)["fingerprint"])


# ---------------------------------------------------------------------------
# a second occurrence comments instead of filing again
# ---------------------------------------------------------------------------

def test_a_known_fingerprint_comments_rather_than_opening_a_new_issue(
        monkeypatch, offline_transport):
    created, commented = [], []

    monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                        lambda repo, fp: (True, {"number": 79,
                                                 "html_url": "u/79"}))
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: created.append(a) or (True, "NEW"))
    monkeypatch.setattr(github_auth, "comment_on_issue",
                        lambda repo, n, body: commented.append(n) or (True, "c"))

    url = issue_report.file_issue(TB, active_app="ml_analyze")

    assert commented == [79], "the second occurrence did not comment"
    assert not created, "a duplicate issue was opened anyway"
    assert url == "u/79", "the caller was not pointed at the existing issue"


def test_an_unknown_fingerprint_still_opens_an_issue(
        monkeypatch, offline_transport):
    created = []
    monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                        lambda repo, fp: (True, None))
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: created.append(a) or (True, "NEW"))

    assert issue_report.file_issue(TB) == "NEW"
    assert len(created) == 1


def test_a_search_that_could_not_run_still_files(
        monkeypatch, offline_transport):
    """Losing a crash report is worse than filing a duplicate, so a failed
    SEARCH must not be read as "no match"."""
    created = []
    monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                        lambda repo, fp: (False, None))
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: created.append(a) or (True, "NEW"))

    assert issue_report.file_issue(TB) == "NEW"
    assert len(created) == 1


# ---------------------------------------------------------------------------
# the backstop
# ---------------------------------------------------------------------------

def test_file_issue_refuses_from_inside_a_test_run(monkeypatch):
    """#75 is a fixture's exception on the PUBLIC tracker. Mocking is the
    caller's job; this is the backstop for the callers that forget, because
    the cost of forgetting cannot be undone by fixing the test afterwards.

    Deliberately WITHOUT the ``offline_transport`` fixture, and it asserts that
    nothing downstream was even reached -- an assertion on the message alone
    would pass while still posting.
    """
    reached = []
    # The old hole was process-wide and inherited.  It must stay inert even if
    # a stale test or subprocess still sets it.
    monkeypatch.setenv("SPACR_ALLOW_GITHUB_WRITES", "1")
    monkeypatch.setattr(github_auth, "is_authenticated",
                        lambda: reached.append("auth") or True)
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: reached.append("create") or (True, "X"))

    result = issue_report.file_issue(TB, active_app="mask")

    assert "test run" in result
    assert not reached, f"the guard let execution through to {reached}"


def test_a_fake_transport_lets_a_mocked_test_through(
        monkeypatch, offline_transport):
    """A test that has replaced transport can still exercise the path."""
    monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                        lambda repo, fp: (True, None))
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: (True, "NEW"))
    assert issue_report.file_issue(TB) == "NEW"


def test_the_guard_rearms_when_a_fake_transport_fixture_tears_down():
    """Restoring the real opener must restore the refusal immediately."""
    assert github_auth._transport_refusal()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_auth, "_HTTP_OPEN", lambda *a, **k: None)
        assert github_auth._transport_refusal() is None

    assert github_auth._HTTP_OPEN is github_auth._REAL_HTTP_OPEN
    assert github_auth._transport_refusal()


def test_a_subprocess_inherits_the_session_lifetime_refusal(tmp_path):
    """A child has no pytest phase variable and no parent monkeypatches.

    It still inherits ``SPACR_PYTEST_SESSION``.  ``resolve_token`` is replaced
    with a function that raises, proving the refusal happens before a child
    can consult ``gh`` or construct a request.  No network call exists in the
    probe even if the assertion regresses.
    """
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    env["SPACR_ALLOW_GITHUB_WRITES"] = "1"
    code = """
from spacr.qt.ai import github_auth
github_auth.resolve_token = lambda: (_ for _ in ()).throw(
    AssertionError('credential resolution was reached'))
ok, message = github_auth.comment_on_issue('owner/name', 114, 'Seen again.')
assert not ok and 'test run' in message, (ok, message)
print(message)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "refusing GitHub network access" in completed.stdout
