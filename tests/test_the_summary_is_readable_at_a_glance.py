"""The answer first, and one explanation per reason.

Instruction 168, having read one in full: "not very accessable make it more
acessable and easier to read and overview".

Nothing in the summary was inaccurate and very little was unnecessary -- the
"not applicable" lines each say WHY, which is what makes a permutation run
legible at all. The problem was that a reader opening it could not find the
three numbers they came for.
"""
import pandas as pd
import pytest

from spacr.regression_summary import (COMPUTED, HEADLINE, RunSummary,
                                      SummaryField, SummarySection,
                                      _split_not_applicable,
                                      format_run_summary, headline)


def _answered(label, value):
    """A field carrying a number."""
    return SummaryField(name=label, label=label, value=value, kind=COMPUTED)


def _absent(label, reason):
    """A field that says why it does not apply."""
    return SummaryField(name=label, label=label, reason=reason)


def _summary():
    return RunSummary(sections=[
        SummarySection(name="fitted", title="WHAT WAS FITTED", fields=[
            _answered("regression type", "ols"),
            _answered("inference", "parametric"),
        ]),
        SummarySection(name="design", title="THE DESIGN", fields=[
            _answered("wells", "606 distinct well(s)"),
            _absent("rank of the design", "no design matrix here"),
            _absent("parameters estimated", "no design matrix here"),
        ]),
        SummarySection(name="call", title="THE CALL", fields=[
            _answered("coefficients called", "14"),
            _answered("alpha", "0.05"),
            _answered("positive control rank", "3 of 789"),
        ]),
    ])


def test_the_headline_is_the_six_things_a_reader_came_for():
    top = headline(_summary())

    labels = [one.label for one in top]
    assert "coefficients called" in labels
    assert "positive control rank" in labels
    assert len(top) <= len(HEADLINE)


def test_every_headline_line_is_QUOTED_from_below():
    """Not a summary of the summary: there must be no second place for a
    number to be wrong."""
    summary = _summary()

    top = headline(summary)

    everything = [one for section in summary.sections for one in section.fields]
    for one in top:
        assert any(other.label == one.label and other.text == one.text
                   for other in everything)


def test_the_answer_comes_before_the_working():
    text = format_run_summary(_summary())

    assert text.index("THE ANSWER") < text.index("WHAT WAS FITTED")
    assert "Everything below is how that was arrived at." in text


def test_a_reader_can_stop_after_the_first_screen():
    text = format_run_summary(_summary())
    lines = text.splitlines()

    answer = lines[:lines.index("Everything below is how that was arrived at.")]

    assert len(answer) < 25, f"the first screen is {len(answer)} lines"
    assert any("coefficients called" in line for line in answer)


# ------------------------------------------------- the not-applicable block


def test_the_not_applicable_fields_leave_one_line_where_they_stood():
    text = format_run_summary(_summary())
    body = text[text.index("THE DESIGN"):text.index("THE CALL")]

    assert "not applicable here" in body
    assert "2 field(s)" in body
    assert "rank of the design" in body, "name them, do not just count them"


def test_nothing_is_deleted():
    """The reasoning is what makes a permutation run legible; a summary that
    dropped it to be shorter would have traded away the thing worth reading."""
    text = format_run_summary(_summary())

    assert "NOT APPLICABLE, AND WHY" in text
    assert "no design matrix here" in text


def test_fields_sharing_a_reason_share_its_explanation():
    """Eleven deferred fields carried SIX distinct explanations on the real
    run, two of them printed three times each."""
    text = format_run_summary(_summary())
    appendix = text[text.index("NOT APPLICABLE, AND WHY"):]

    assert appendix.count("no design matrix here") == 1
    assert "rank of the design, parameters estimated" in appendix


def test_a_long_joined_label_does_not_squeeze_the_text():
    """A label longer than the column left the explanation one word wide."""
    summary = RunSummary(sections=[
        SummarySection(name="design", title="THE DESIGN", fields=[
            _absent(name, "a shared reason")
            for name in ("a very long field name indeed",
                         "another quite long field name",
                         "and a third long one here")])])

    text = format_run_summary(summary)

    widest = max(len(line) for line in text.splitlines())
    assert widest <= 100

    # Only the EXPLANATION lines -- the ones indented under a long label.
    # A section title or a wrapped label continuation is legitimately short.
    body = [line for line in text.splitlines()
            if line.startswith("      ") and line.strip()]
    assert body, "the explanation was not written under the label at all"
    assert min(len(line.strip()) for line in body) > 20, (
        f"the explanation came out one word wide: {body}")


def test_a_run_with_nothing_deferred_gets_no_appendix():
    summary = RunSummary(sections=[
        SummarySection(name="call", title="THE CALL", fields=[
            _answered("coefficients called", "14")])])

    text = format_run_summary(summary)

    assert "NOT APPLICABLE, AND WHY" not in text
    assert "not applicable here" not in text


def test_the_split_is_by_how_a_field_says_it_does_not_apply():
    fields = [_answered("a", "12"),
              _absent("b", "because")]

    shown, deferred = _split_not_applicable(fields)

    assert [one.label for one in shown] == ["a"]
    assert [one.label for one in deferred] == ["b"]
