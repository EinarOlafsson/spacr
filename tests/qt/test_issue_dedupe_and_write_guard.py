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

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import github_auth, issue_report


@pytest.fixture
def allow_writes(monkeypatch):
    """`file_issue` refuses to post from a test run unless this is set.

    That backstop exists because `[auto 54a0e8] [mask] Error: boom` (#75)
    reached the PUBLIC tracker from a fixture's exception -- spaCR posts
    whenever a token is resolvable and `gh` supplies one on a dev machine.

    Every test in this file replaces `github_auth` wholesale, so nothing here
    can reach the network; the flag says that deliberately rather than
    leaving the guard to be discovered as five confusing failures.
    """
    monkeypatch.setenv("SPACR_ALLOW_GITHUB_WRITES", "1")



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
        monkeypatch, allow_writes):
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


def test_an_unknown_fingerprint_still_opens_an_issue(monkeypatch, allow_writes):
    created = []
    monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                        lambda repo, fp: (True, None))
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: created.append(a) or (True, "NEW"))

    assert issue_report.file_issue(TB) == "NEW"
    assert len(created) == 1


def test_a_search_that_could_not_run_still_files(monkeypatch, allow_writes):
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

    Deliberately WITHOUT the `allow_writes` fixture, and it asserts that
    nothing downstream was even reached -- an assertion on the message alone
    would pass while still posting.
    """
    reached = []
    monkeypatch.setattr(github_auth, "is_authenticated",
                        lambda: reached.append("auth") or True)
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: reached.append("create") or (True, "X"))

    result = issue_report.file_issue(TB, active_app="mask")

    assert "test run" in result
    assert not reached, f"the guard let execution through to {reached}"


def test_the_escape_hatch_lets_a_mocked_test_through(monkeypatch,
                                                     allow_writes):
    """A test that HAS mocked everything can still exercise the path."""
    monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                        lambda repo, fp: (True, None))
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: (True, "NEW"))
    assert issue_report.file_issue(TB) == "NEW"
