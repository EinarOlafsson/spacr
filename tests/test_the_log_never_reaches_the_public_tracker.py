"""A bug report is public; the log it was built from is not.

`build_issue` composes a body that is URL-encoded onto
``github.com/<repo>/issues/new`` and opened in a browser, so everything in
it becomes world-readable and permanent the moment the issue is filed.

Log lines carry whatever the run was about -- a gene name, a plate
barcode, a collaborator's folder, the name of an unpublished screen. None
of that is credential-shaped, so the redaction pass that catches tokens
and database paths does not touch it, and the person filing the bug has
no way to know it is in there. So the log is saved to the reporter's own
disk and the issue names the path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt.ai import issue_report


SECRETS = [
    "TGGT1_239010 knockdown finished",          # an unpublished gene
    "plate BARCODE-7781 loaded",                # a plate barcode
    "reading /srv/collab/hammond_lab/screen3",  # a collaborator's folder
]


@pytest.fixture()
def a_log(tmp_path):
    """A log holding the kinds of thing no redaction pass can recognise."""
    path = tmp_path / "spacr.log"
    path.write_text("\n".join(SECRETS) + "\n", encoding="utf-8")
    return path


def _body(monkeypatch, a_log, tmp_home):
    monkeypatch.setattr(issue_report, "log_tail",
                        lambda n_lines=50, log_path=None:
                        a_log.read_text(encoding="utf-8"))
    monkeypatch.setattr(issue_report, "log_bundle_dir",
                        lambda: tmp_home / "reports")
    return issue_report.build_report(
        "Traceback (most recent call last):\n"
        "  File \"run.py\", line 1, in <module>\n"
        "ValueError: something went wrong\n",
        include_log_tail=True)


def test_no_log_line_reaches_the_issue_body(monkeypatch, a_log, tmp_path):
    issue = _body(monkeypatch, a_log, tmp_path)
    for secret in SECRETS:
        assert secret not in issue["body"], (
            f"{secret!r} would have been published on the issue tracker")


def test_the_body_says_where_the_log_is_instead(monkeypatch, a_log, tmp_path):
    issue = _body(monkeypatch, a_log, tmp_path)
    assert "log-" in issue["body"]
    # And it says WHY, so the reader does not think the log was forgotten.
    assert "public" in issue["body"].lower()


def test_the_log_is_written_where_the_body_says_it_is(monkeypatch, a_log,
                                                      tmp_path):
    issue = _body(monkeypatch, a_log, tmp_path)
    written = list((tmp_path / "reports").glob("log-*.txt"))
    assert len(written) == 1, "the log was not saved for the user to send"
    assert written[0].name in issue["body"]
    # It is the WHOLE log, not the keyhole an issue body could have held.
    kept = written[0].read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret in kept


def test_a_read_only_home_still_files_the_report(monkeypatch, a_log,
                                                 tmp_path):
    """Losing the log copy must not cost the user the bug report."""
    monkeypatch.setattr(issue_report, "log_tail",
                        lambda n_lines=50, log_path=None: "a line\n")
    unwritable = tmp_path / "nope"
    unwritable.write_text("I am a file, not a directory", encoding="utf-8")
    monkeypatch.setattr(issue_report, "log_bundle_dir", lambda: unwritable)
    issue = issue_report.build_report(
        "Traceback (most recent call last):\nValueError: boom\n",
        include_log_tail=True)
    assert issue["title"]
    assert "boom" in issue["body"]


def test_an_empty_log_saves_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(issue_report, "log_tail",
                        lambda n_lines=50, log_path=None: "   \n")
    monkeypatch.setattr(issue_report, "log_bundle_dir",
                        lambda: tmp_path / "reports")
    assert issue_report.save_log_bundle("abc123") is None


def test_the_saved_log_is_named_for_the_issue_it_belongs_to(monkeypatch,
                                                            a_log, tmp_path):
    """One machine files several reports; each log has to match its issue."""
    monkeypatch.setattr(issue_report, "log_tail",
                        lambda n_lines=50, log_path=None: "a line\n")
    monkeypatch.setattr(issue_report, "log_bundle_dir",
                        lambda: tmp_path / "reports")
    saved = issue_report.save_log_bundle("deadbeef")
    assert saved is not None and "deadbeef" in saved.name


def test_more_of_the_log_is_kept_than_an_issue_could_have_carried():
    """The URL length limit was the only reason for fifty lines."""
    assert issue_report.LOG_BUNDLE_LINES > issue_report.LOG_TAIL_LINES
