"""Taking a summary away, and rendering one the theme cannot colour.

The Save button, the lines a summary carries that are not label/value rows,
and the two places the panel asks the theme for its error colour: all of them
have to work on a summary saved by another spaCR version, on a machine whose
theme cannot be resolved, and when the file cannot be written.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QPlainTextEdit, QTextBrowser

from spacr.qt.widgets.folding_summary import FoldingSummaryView, split_rows

pytestmark = pytest.mark.qt


SUMMARY = """THE ANSWER
----------
  effect                 0.42 (95% CI 0.31 to 0.53)
  D'Agostino K2 p        4.96e-157 (REJECTED at 0.05)
short

STATSMODELS
-----------
                 coef    std err          t
const          0.4200      0.010     42.000
"""


@pytest.fixture
def panel(qtbot):
    """A summary panel holding a two-section summary."""
    widget = FoldingSummaryView()
    qtbot.addWidget(widget)
    widget.setPlainText(SUMMARY)
    return widget


def _browser_html(widget):
    return "\n".join(view.toHtml()
                     for view in widget.findChildren(QTextBrowser))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_a_line_too_short_to_be_a_row_is_kept_as_its_own_line():
    """A sentence between rows survives as an unlabelled row, not dropped.

    Summaries written by other spaCR versions mix prose in with the rows.
    Dropping what does not parse would silently delete part of the answer.
    """
    rows = split_rows("  effect        0.42\nshort\n")

    assert rows == [("effect", "0.42"), ("", "short")]


def test_an_unlabelled_line_spans_both_columns(panel):
    """The rendered table gives such a line the full width.

    Laid out in the label column it would be truncated by the value column's
    left edge; it is a sentence, not a label.
    """
    html = _browser_html(panel)

    assert "colspan=\"2\"" in html or "colspan='2'" in html
    assert "short" in html


def test_a_rejected_assumption_is_tinted_in_the_table(panel):
    """The row carrying REJECTED is coloured, not just the block bodies.

    Most of a summary arrives as rows; a marker highlighted only in the plain
    block would leave the broken assumption grey where it actually appears.
    """
    html = _browser_html(panel)

    row = next(line for line in html.splitlines() if "REJECTED at" in line)
    assert "color:" in row.lower()


# ---------------------------------------------------------------------------
# Taking it away
# ---------------------------------------------------------------------------

def test_saving_without_a_path_asks_where(panel, tmp_path, monkeypatch):
    """The Save button opens a file dialog and writes what it is told.

    The panel writes a COPY of the run's own text; re-rendering it would
    differ from the artefact in the run folder.
    """
    target = tmp_path / "model_summary.txt"
    asked = []

    def _ask(parent, caption, name, filters):
        asked.append((caption, name, filters))
        return str(target), filters

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _ask)

    written = panel.save_to_file()

    assert written == str(target)
    assert len(asked) == 1
    assert asked[0][1] == "model_summary.txt"
    assert target.read_text(encoding="utf-8") == panel.toPlainText()


def test_a_cancelled_save_writes_nothing(panel, monkeypatch, tmp_path):
    """Dismissing the dialog returns an empty path and touches no file."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: ("", ""))
    monkeypatch.chdir(tmp_path)

    assert panel.save_to_file() == ""
    assert list(tmp_path.iterdir()) == []


def test_a_save_that_cannot_be_written_says_so_by_returning_nothing(panel,
                                                                    tmp_path):
    """An unwritable destination comes back empty rather than raising.

    The caller shows "saved to ..." on a non-empty answer; a traceback out of
    a button handler would take the results window with it.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")

    assert panel.save_to_file(str(blocker / "summary.txt")) == ""


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

def test_asking_about_a_section_that_is_not_there_is_not_an_error(panel):
    """An unknown heading answers False instead of raising.

    Callers restore a remembered fold state by title, and the summary they
    are looking at may be from a version with different headings.
    """
    assert panel.section_titles() == ("THE ANSWER", "STATSMODELS")
    assert panel.is_section_expanded("THE ANSWER") is True
    assert panel.is_section_expanded("A HEADING FROM ANOTHER VERSION") is False

    panel.set_section_expanded("A HEADING FROM ANOTHER VERSION", True)
    assert panel.section_titles() == ("THE ANSWER", "STATSMODELS")


# ---------------------------------------------------------------------------
# A theme that cannot be read
# ---------------------------------------------------------------------------

def test_a_summary_still_renders_when_the_theme_cannot_be_read(qtbot,
                                                               monkeypatch):
    """No palette means no colour, not a blank Summary tab.

    Both bodies ask the theme for its error colour. If that ask were fatal the
    whole summary would be missing -- a worse failure than an uncoloured
    rejection.
    """
    from spacr.qt import theme

    def _no_palette(*args, **kwargs):
        raise KeyError("error")

    monkeypatch.setattr(theme, "active_palette", _no_palette)

    widget = FoldingSummaryView()
    qtbot.addWidget(widget)
    widget.setPlainText(SUMMARY)

    html = _browser_html(widget)
    assert "REJECTED at" in html
    assert "color:" not in html.split("REJECTED at")[0].split("<tr>")[-1], (
        "with no palette there is no colour to tint with")
    blocks = widget.findChildren(QPlainTextEdit)
    assert any("const" in block.toPlainText() for block in blocks)
    assert all(getattr(block, "_spacr_highlighter", None) is None
               for block in blocks)


def test_the_reading_surface_is_a_surface(qtbot):
    """Summary text sits on a surface, not straight on the moving backdrop.

    ``_reading_surface`` composites the ``surface_alt`` colour at the pane
    opacity. The marker came off once the alpha was asked for under the role
    the palette actually defines: asking for a role with no palette entry
    raised, and every summary body was painted fully transparent instead.
    """
    widget = FoldingSummaryView()
    qtbot.addWidget(widget)

    assert widget._reading_surface().startswith("rgba(")
