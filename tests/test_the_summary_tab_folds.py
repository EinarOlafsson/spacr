"""Instruction 168 D: the Summary tab folds, the file on disk does not.

    "The Summary tab shows the verdict expanded and each section collapsed,
    with the section headings as the outline. The file on disk stays plain
    text and stays readable in a terminal, because it is a run artefact
    before it is a widget."

The parser is the risky half. It works off the text rather than off
`RunSummary` -- deliberately, because the tab may be showing a file written
by another version of spaCR, or the statsmodels summary, which has no spaCR
sections at all -- so every test here is about what it must NOT treat as a
heading.
"""
import pytest

from spacr.qt.widgets.folding_summary import (ANSWER_HEADING, split_sections)


SUMMARY = """spaCR RUN SUMMARY
=================

A warning that goes above everything else.

THE ANSWER
----------
  verdict        it worked

Everything below is how that was arrived at.

WHAT WAS FITTED
---------------
  model          ols

THE DESIGN
----------
  wells          1536
"""


# ------------------------------------------------------------- the parsing


def test_the_sections_are_found_in_order():
    _preamble, sections = split_sections(SUMMARY)
    assert [h for h, _ in sections] == [
        "THE ANSWER", "WHAT WAS FITTED", "THE DESIGN"]


def test_the_document_title_is_not_a_section():
    """`spaCR RUN SUMMARY` names the file; it is not one of its parts."""
    _preamble, sections = split_sections(SUMMARY)
    assert "spaCR RUN SUMMARY" not in [h for h, _ in sections]


def test_the_warning_stays_above_everything():
    """A rank-deficiency warning under a fold is a warning nobody reads."""
    preamble, _sections = split_sections(SUMMARY)
    assert "goes above everything else" in preamble


def test_each_body_holds_its_own_lines():
    _preamble, sections = split_sections(SUMMARY)
    bodies = dict(sections)
    assert "ols" in bodies["WHAT WAS FITTED"]
    assert "1536" in bodies["THE DESIGN"]
    assert "ols" not in bodies["THE DESIGN"]


def test_a_rule_of_the_wrong_length_is_not_a_heading():
    """THE FAILURE MODE. statsmodels draws rows of dashes all over its
    summary; if any line above one became a heading the tab would fold
    itself into nonsense."""
    text = "Dep. Variable:   pred\n" + "=" * 78 + "\ncoef   std err\n"
    _preamble, sections = split_sections(text)
    assert sections == []


def test_text_with_no_headings_comes_back_whole():
    text = "No summary: this backend is not a statsmodels fit."
    preamble, sections = split_sections(text)
    assert preamble == text
    assert sections == []


def test_an_empty_summary_is_not_an_error():
    assert split_sections("") == ("", [])
    assert split_sections(None) == ("", [])


def test_a_heading_with_no_body_is_still_a_section():
    """An empty section is a real answer -- "nothing was excluded"."""
    text = "THE ANSWER\n----------\n  verdict  ok\n\nWHAT WAS EXCLUDED\n" \
           "-----------------\n"
    _preamble, sections = split_sections(text)
    assert [h for h, _ in sections] == ["THE ANSWER", "WHAT WAS EXCLUDED"]


# --------------------------------------------------------------- the widget


@pytest.fixture()
def view(qtbot):
    from spacr.qt.widgets.folding_summary import FoldingSummaryView

    widget = FoldingSummaryView()
    qtbot.addWidget(widget)
    return widget


def test_the_verdict_is_open_and_the_rest_are_folded(view):
    view.setPlainText(SUMMARY)

    assert view.is_section_expanded(ANSWER_HEADING)
    for title in view.section_titles():
        if title != ANSWER_HEADING:
            assert not view.is_section_expanded(title), title


def test_every_sentence_is_still_reachable(view):
    """168's own bar: "Every sentence that is in today's summary is still
    reachable." Folded is not deleted."""
    view.setPlainText(SUMMARY)

    assert view.toPlainText() == SUMMARY
    for title in view.section_titles():
        view.set_section_expanded(title, True)
        assert view.is_section_expanded(title)


def test_refilling_replaces_the_sections(view):
    """A second run must not leave the first run's headings behind."""
    view.setPlainText(SUMMARY)
    view.setPlainText("THE ANSWER\n----------\n  verdict  a different run\n")

    assert view.section_titles() == ("THE ANSWER",)


def test_a_statsmodels_summary_is_shown_whole(view):
    """Not chopped up by a guess."""
    text = "OLS Regression Results\n" + "=" * 78 + "\ncoef  std err\n"
    view.setPlainText(text)

    assert view.section_titles() == ()
    assert view.toPlainText() == text


# ------------------------------------------------------ mounted in the panel


def test_the_panel_uses_it(qtbot):
    from spacr.qt.widgets.folding_summary import FoldingSummaryView
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    assert isinstance(panel._summary, FoldingSummaryView)
    assert panel.tabs.indexOf(panel._summary) >= 0


def test_the_panel_still_fills_it_the_same_way(qtbot):
    """`setPlainText` is the whole contract with the panel."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel._summary.setPlainText(SUMMARY)

    assert "1536" in panel._summary.toPlainText()
    assert panel._summary.is_section_expanded(ANSWER_HEADING)


# =========================================================================== #
#  Instruction 178 B -- the summary is a table, and it can be taken away      #
# =========================================================================== #

BODY = """  regression type              ols
  inference                    parametric — the model was fitted and its
                               standard errors are the model's own
  wells                        1536
"""


def test_a_section_body_becomes_aligned_rows():
    """"the left title in bold and then the text so they are all alligned"."""
    from spacr.qt.widgets.folding_summary import split_rows

    rows = split_rows(BODY)

    assert [label for label, _ in rows] == [
        "regression type", "inference", "wells"]
    assert dict(rows)["regression type"] == "ols"


def test_a_wrapped_value_is_joined_back_up():
    """The FILE wraps at 88 characters whatever the window is, which is why
    widening the tab gained nothing. The panel re-wraps, so it has to undo
    the file's wrapping first."""
    from spacr.qt.widgets.folding_summary import split_rows

    value = dict(split_rows(BODY))["inference"]

    assert "standard errors are the model's own" in value
    assert "\n" not in value


def test_text_that_is_not_rows_is_left_alone():
    """A statsmodels block is column-aligned ASCII that carries its own
    alignment; re-laying it out would destroy it."""
    from spacr.qt.widgets.folding_summary import split_rows

    assert split_rows("OLS Regression Results\n" + "=" * 78) == []
    assert split_rows("") == []


def test_the_lead_width_is_measured_not_assumed():
    """A summary written by another version of spaCR has a different label
    column, and the panel still has to read it."""
    from spacr.qt.widgets.folding_summary import split_rows

    narrow = "  type    ols\n  wells   1536\n"
    rows = dict(split_rows(narrow))

    assert rows["type"] == "ols"
    assert rows["wells"] == "1536"


def test_the_rows_are_drawn_as_a_table(view):
    from PySide6.QtWidgets import QTextBrowser

    view.setPlainText(SUMMARY)
    tables = view.findChildren(QTextBrowser)

    assert tables, "no section was laid out as a table"
    html = tables[0].toHtml()
    assert "<table" in html
    # Qt normalises <b> into a weight on the span, so the tag itself is gone
    # by the time the document is read back. 700 is bold; 400 is the body.
    assert "font-weight:700" in html or "<b>" in html, "the label is not bold"


def test_there_is_a_copy_and_a_save_button(view):
    """"i should be able to click a button to save them and also copy them
    with the overlapping squares icon"."""
    view.setPlainText(SUMMARY)

    assert view.copy_button is not None
    assert view.save_button is not None
    assert "⧉" in view.copy_button.text(), "no overlapping-squares icon"


def test_copy_puts_the_whole_summary_on_the_clipboard(view, qtbot):
    from PySide6.QtWidgets import QApplication

    view.setPlainText(SUMMARY)
    assert view.copy_to_clipboard() is True

    assert QApplication.clipboard().text() == SUMMARY


def test_copying_nothing_is_refused_rather_than_clearing_the_clipboard(view):
    view.setPlainText("   ")
    assert view.copy_to_clipboard() is False


def test_save_writes_the_run_s_own_text(view, tmp_path):
    """A COPY, NOT A RE-RENDER: rendering again would differ in the
    statsmodels `Time:` header alone and invite the reader to wonder which
    is authoritative."""
    view.setPlainText(SUMMARY)
    out = tmp_path / "summary.txt"

    written = view.save_to_file(str(out))

    assert written == str(out)
    assert out.read_text() == SUMMARY


def test_saving_nothing_writes_nothing(view, tmp_path):
    view.setPlainText("")
    assert view.save_to_file(str(tmp_path / "x.txt")) == ""
    assert not (tmp_path / "x.txt").exists()
