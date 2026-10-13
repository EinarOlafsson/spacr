"""A filed issue carries spaCR AI's analysis of the same error.

When the AI is switched on it has usually already diagnosed the crash by the
time the user files -- the console offers to explain a traceback the moment it
arrives -- and that analysis is the most useful thing in the report after the
traceback itself.

TWO THINGS IT MUST NOT DO. It must not carry an analysis of a DIFFERENT error
(the console holds one conversation for a whole session), and it must not push
the traceback out of the report: `issue_url` trims the tail of the body to fit
GitHub's URL limit, so an unbounded analysis would cost the environment, the
settings and the log-bundle path.
"""
from __future__ import annotations

import pytest

from spacr.qt.ai.issue_report import AI_ANALYSIS_MAX_CHARS, build_report

TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "/home/someone/Documents/repo/spacr/spacr/object.py", line 891\n'
    "TypeError: '>' not supported between instances of 'str' and 'int'\n"
)
ANALYSIS = ("Root cause: object.py passes the raw setting to Cellpose's "
            "diameter kwarg, so a string reaches a `> 0` comparison.")


def test_the_analysis_appears_in_the_body():
    body = build_report(TRACEBACK, active_app="mask",
                        include_log_tail=False, ai_response=ANALYSIS)["body"]
    assert ANALYSIS in body


def test_no_analysis_means_no_section():
    """An empty section header in every report filed with the AI switched off
    would be noise on the tracker."""
    body = build_report(TRACEBACK, active_app="mask",
                        include_log_tail=False)["body"]
    assert "spaCR AI's analysis" not in body


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", None])
def test_a_blank_analysis_is_not_a_section(blank):
    body = build_report(TRACEBACK, include_log_tail=False,
                        ai_response=blank)["body"]
    assert "spaCR AI's analysis" not in body


def test_it_is_marked_as_machine_generated():
    """A maintainer must be able to tell a generated diagnosis from a human
    one. The analysis in the session this was written for was right about the
    cause and wrong about the fix."""
    body = build_report(TRACEBACK, include_log_tail=False,
                        ai_response=ANALYSIS)["body"]
    assert "unreviewed" in body
    assert "lead rather than a diagnosis" in body


def test_the_traceback_still_comes_first():
    """`issue_url` trims the TAIL, so anything above the traceback can cost
    the report the one thing it exists to carry."""
    body = build_report(TRACEBACK, include_log_tail=False,
                        ai_response=ANALYSIS)["body"]
    assert body.index("### Traceback") < body.index("spaCR AI's analysis")
    assert body.index("spaCR AI's analysis") < body.index("### Environment")


def _analysis_block(body: str) -> str:
    """Just the folded analysis section.

    Counted here rather than over the whole body: the environment block
    contains "x86_64", so a naive character count over the report measures
    the platform string as well as the analysis.
    """
    start = body.index("spaCR AI's analysis")
    return body[start:body.index("</details>", start)]


def test_a_runaway_analysis_is_capped():
    body = build_report(TRACEBACK, include_log_tail=False,
                        ai_response="x" * (AI_ANALYSIS_MAX_CHARS * 3))["body"]
    assert "analysis truncated" in body
    assert _analysis_block(body).count("x") <= AI_ANALYSIS_MAX_CHARS


def test_the_capped_report_still_holds_everything_after_it():
    """The cap exists so the sections BELOW the analysis survive."""
    body = build_report(TRACEBACK, include_log_tail=False,
                        ai_response="x" * (AI_ANALYSIS_MAX_CHARS * 3))["body"]
    assert "### Environment" in body


def test_the_analysis_goes_through_the_same_sanitiser_as_the_traceback():
    """It quotes the traceback back at you, so it carries whatever the
    traceback carried -- and an issue on the tracker is world-readable.

    `sanitize_path` abbreviates the reporter's OWN home, which is the pass
    the traceback gets; the stricter `strip_report_paths` is applied to the
    whole body by the preview dialog, and is covered below.
    """
    import os

    from spacr.qt.ai.issue_report import sanitize_path

    home_file = os.path.join(os.path.expanduser("~"), "screens", "secret.tif")
    body = build_report(TRACEBACK, include_log_tail=False,
                        ai_response=f"Look at {home_file}")["body"]

    assert home_file not in body, "the reporter's home path went out verbatim"
    assert sanitize_path(home_file) in body


def test_the_public_strip_reaches_into_the_analysis():
    """The preview's stricter pass runs over the whole body, so a path the
    analysis quotes is removed with the rest."""
    from spacr.qt.ai.issue_report import strip_report_paths

    body = build_report(
        TRACEBACK, include_log_tail=False,
        ai_response="Open /home/someone/repo/spacr/object.py")["body"]

    assert "/home/someone" not in strip_report_paths(body)


def test_the_fingerprint_is_unchanged_by_the_analysis():
    """The fingerprint identifies the CRASH. If an AI answer moved it, two
    reports of one bug would no longer match each other."""
    without = build_report(TRACEBACK, include_log_tail=False)["fingerprint"]
    with_it = build_report(TRACEBACK, include_log_tail=False,
                           ai_response=ANALYSIS)["fingerprint"]
    assert without == with_it
