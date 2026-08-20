"""Show a run summary with its sections folded.

The Summary tab shows the verdict expanded and each later section collapsed,
using the headings as an outline. The file on disk remains plain text and
readable in a terminal because it is a run artefact before it is a widget.

PARSED FROM THE TEXT, NOT BUILT FROM `RunSummary`. That looks like the wrong
way round -- the structure is right there -- but the tab does not always have
it. The summary it shows may be the run's OWN file read back from beside a
results table loaded from disk, written by a different version of spaCR, or
the statsmodels summary, which has no spaCR sections at all. Parsing the text
is the one route that works for all three, and it keeps the file as the
source of truth rather than making the widget a second renderer that can
disagree with it.

A heading is a line whose successor is a rule of ``-`` or ``=`` the same
length -- which is what :func:`spacr.regression_summary.format_run_summary`
writes. Text with no headings is shown as it arrived; the statsmodels
summary must not be chopped up by a guess.
"""
import re
import html
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from .collapsible_section import CollapsibleSection

#: The section that answers the question, and so the one left open.
#: `format_run_summary` writes it first and quotes every line of it from the
#: sections below, so a reader who never unfolds anything still has the answer.
ANSWER_HEADING = "THE ANSWER"

#: The document title, which is not a section and gets no fold of its own.
DOCUMENT_HEADING = "spaCR RUN SUMMARY"


def split_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split ``text`` into the preamble and its ``(heading, body)`` sections.

    :returns: everything before the first section, then the sections. A text
        with no headings comes back as ``(text, [])`` -- which is the signal
        to show it unchanged.
    """
    lines = str(text or "").splitlines()
    marks: List[int] = []
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        rule = lines[i + 1].strip()
        if not title or not rule:
            continue
        # THE RULE MUST MATCH THE TITLE'S LENGTH. Without that a row of
        # dashes drawn by statsmodels turns the line above it into a heading,
        # and the summary folds itself into nonsense.
        if len(rule) != len(title):
            continue
        if set(rule) not in ({"-"}, {"="}):
            continue
        if title == DOCUMENT_HEADING:
            continue
        marks.append(i)

    if not marks:
        return str(text or ""), []

    preamble = "\n".join(lines[:marks[0]]).strip("\n")
    sections: List[Tuple[str, str]] = []
    for index, start in enumerate(marks):
        end = marks[index + 1] if index + 1 < len(marks) else len(lines)
        heading = lines[start].strip()
        body = "\n".join(lines[start + 2:end]).strip("\n")
        sections.append((heading, body))
    return preamble, sections


def split_rows(body: str) -> list:
    """A section body as ``[(label, value)]``, or ``[]`` if it is not rows.

    `format_run_summary` lays each field out as a fixed-width label column
    followed by wrapped text, with continuation lines indented to match. That
    is right for the FILE -- 168 D keeps it readable in a terminal -- and
    wrong for a panel: the column is 88 characters whatever the window is, so
    widening the tab gains nothing and the explanation stays a narrow ribbon.
    Reported as "its realluy hard to read".

    So the panel re-reads the rows and lays them out itself. THE LEAD WIDTH
    IS MEASURED, not assumed: it is taken from the first line that has one,
    so a summary written by another version of spaCR with a different label
    column still parses.

    :returns: the rows, or ``[]`` when the body is not laid out this way --
        a paragraph, a statsmodels block -- which the caller shows as it is.
    """
    lines = [line for line in str(body or "").splitlines() if line.strip()]
    if not lines:
        return []
    lead = None
    for line in lines:
        if not line.startswith("  "):
            continue
        stripped = line[2:]
        gap = len(stripped) - len(stripped.lstrip())
        # A row is "  label" then at least two spaces then the text.
        match = _ROW.match(line)
        if match:
            lead = len(match.group(1))
            break
    if lead is None:
        return []
    rows: list = []
    for line in lines:
        if len(line) > lead and line[:lead].strip():
            rows.append([line[:lead].strip(), line[lead:].strip()])
        elif rows and len(line) > lead:
            # A continuation of the value above: joined, so the panel can
            # re-wrap it to whatever width it actually has.
            rows[-1][1] = (rows[-1][1] + " " + line[lead:].strip()).strip()
        elif line.strip():
            rows.append(["", line.strip()])
    return [(label, value) for label, value in rows]


#: A summary row: two spaces, a label, then at least two more spaces.
_ROW = re.compile(r"^(  \S[^\s]*(?:[ \t]\S+)*?  +)\S")


class FoldingSummaryView(QScrollArea):
    """A drop-in for the Summary tab's ``QPlainTextEdit``.

    Keeps ``setPlainText`` and ``toPlainText`` so the panel that fills it does
    not have to know which it got, and so the text a test reads is the text
    the file holds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._body = QWidget(self)
        self._layout = QVBoxLayout(self._body)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self.setWidget(self._body)

        # SAVE AND COPY, because a summary a reader cannot take with them is
        # a summary they retype. Asked for 2026-08-19: "i should be able to
        # click a button to save them and also copy them with the
        # overlapping squares icon".
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.copy_button = QPushButton("\u29c9  Copy")
        self.copy_button.setToolTip(
            "Copy the whole summary to the clipboard, exactly as it is "
            "written to the run folder.")
        self.copy_button.setFlat(True)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        actions.addWidget(self.copy_button)
        self.save_button = QPushButton("Save\u2026")
        self.save_button.setToolTip(
            "Write the summary to a text file. It is the run's own summary, "
            "so this is a copy rather than a new rendering.")
        self.save_button.setFlat(True)
        self.save_button.clicked.connect(self.save_to_file)
        actions.addWidget(self.save_button)
        actions.addStretch(1)
        self._layout.addLayout(actions)

        self._text = ""
        self._sections: List[CollapsibleSection] = []
        self._mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)

    # ------------------------------------------------ the QPlainTextEdit API

    def setPlainText(self, text: str) -> None:      # noqa: N802 - Qt naming
        self._text = str(text or "")
        self._rebuild()

    def toPlainText(self) -> str:                   # noqa: N802 - Qt naming
        return self._text

    def isReadOnly(self) -> bool:                   # noqa: N802 - Qt naming
        """Always. A summary is a run artefact, not a document to edit."""
        return True

    def setReadOnly(self, read_only: bool) -> None:  # noqa: N802 - Qt naming
        """Accepted and ignored -- see :meth:`isReadOnly`.

        Kept so this stays a drop-in for the QPlainTextEdit it replaced: a
        caller that sets what is already true should not have to know which
        widget it got.
        """

    def font(self):
        """The fixed-width font the bodies are laid out in.

        The summary is aligned with spaces, so the font is part of whether it
        is readable at all -- and a test asserts the widget was GIVEN a fixed
        font rather than trusting a style hint.
        """
        return self._mono

    # ------------------------------------------------------------- taking it

    def copy_to_clipboard(self) -> bool:
        """The whole summary, as the run wrote it. Returns whether it went."""
        from PySide6.QtWidgets import QApplication

        text = self.toPlainText()
        if not text.strip():
            return False
        clipboard = QApplication.clipboard()
        if clipboard is None:                       # pragma: no cover
            return False
        clipboard.setText(text)
        return True

    def save_to_file(self, path: str = "") -> str:
        """Write the summary out. Returns the path written, or ``""``.

        A COPY, NOT A RE-RENDER. The run wrote this text when it was fitted;
        rendering it again here would differ in the statsmodels `Time:`
        header alone and invite the reader to wonder which is authoritative.
        """
        from PySide6.QtWidgets import QFileDialog

        text = self.toPlainText()
        if not text.strip():
            return ""
        chosen = str(path or "")
        if not chosen:
            chosen, _filter = QFileDialog.getSaveFileName(
                self, "Save the summary", "model_summary.txt",
                "Text (*.txt)")
        if not chosen:
            return ""
        try:
            with open(chosen, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            return ""
        return chosen

    # ------------------------------------------------------------- folding

    def section_titles(self) -> tuple:
        return tuple(s.title() for s in self._sections)

    def is_section_expanded(self, title: str) -> bool:
        for section in self._sections:
            if section.title() == str(title):
                return section.is_expanded()
        return False

    def set_section_expanded(self, title: str, expanded: bool) -> None:
        for section in self._sections:
            if section.title() == str(title):
                section.set_expanded(bool(expanded))

    # ------------------------------------------------------------ internals

    def _clear(self) -> None:
        self._sections = []
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _table(self, rows: list) -> QWidget:
        """One section as an aligned two-column table.

        THE LABEL IS BOLD AND THE COLUMNS LINE UP, which is what was asked
        for -- "the left title in bold and then the text so they are all
        alligned". Built as HTML in a QTextBrowser rather than a
        QTableWidget: the value column has to WRAP to whatever width the
        panel has ("the text lines in these summaries should be able to be
        as wide as the container"), and a table widget would give it a fixed
        column and a scroll bar instead.
        """
        from PySide6.QtWidgets import QTextBrowser

        view = QTextBrowser(self._body)
        view.setOpenExternalLinks(True)
        view.setFrameShape(QTextBrowser.NoFrame)
        # TRANSPARENT: the panel behind it is a translucent surface, and an
        # opaque block is what made these read as black slabs.
        view.viewport().setAutoFillBackground(False)
        view.setStyleSheet("QTextBrowser { background: transparent; }")
        cells = []
        for label, value in rows:
            if label:
                cells.append(
                    f"<tr><td style='padding:1px 14px 1px 0;"
                    f"white-space:nowrap;vertical-align:top'><b>"
                    f"{html.escape(label)}</b></td>"
                    f"<td style='padding:1px 0'>{html.escape(value)}</td></tr>")
            else:
                cells.append(
                    f"<tr><td colspan='2' style='padding:4px 0'>"
                    f"{html.escape(value)}</td></tr>")
        view.setHtml("<table style='border-collapse:collapse'>"
                     + "".join(cells) + "</table>")
        view.document().setDocumentMargin(2)
        rows_shown = max(2, len(rows))
        view.setMinimumHeight(min(460, 20 * rows_shown + 12))
        return view

    def _block(self, text: str) -> QPlainTextEdit:
        """One body, in the fixed-width font the file was laid out for.

        The summary is aligned with spaces -- a label column and a wrapped
        explanation -- so a proportional font would ruin every row of it.
        """
        view = QPlainTextEdit(self._body)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.NoWrap)
        view.setFont(self._mono)
        view.setPlainText(text)
        rows = max(3, text.count("\n") + 2)
        view.setMinimumHeight(min(420, 18 * rows))
        return view

    def _rebuild(self) -> None:
        self._clear()
        preamble, sections = split_sections(self._text)

        if not sections:
            # NOTHING TO FOLD, so nothing is folded. The statsmodels summary
            # has no spaCR headings and chopping it up by a guess would be
            # worse than leaving it whole.
            self._layout.addWidget(self._block(self._text), 1)
            return

        if preamble.strip():
            label = QLabel(preamble, self._body)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._layout.addWidget(label)

        for heading, body in sections:
            rows = split_rows(body)
            # THE VERDICT OPEN, EVERYTHING ELSE FOLDED -- the headings are
            # then the outline the instruction asks for.
            expanded = heading.upper() == ANSWER_HEADING
            # A TABLE WHERE THE BODY IS ROWS, the plain block otherwise --
            # the statsmodels summary is column-aligned ASCII and re-laying
            # it out would destroy the alignment it carries itself.
            content = self._table(rows) if rows else self._block(body)
            section = CollapsibleSection(heading, content,
                                         expanded=expanded, parent=self._body)
            self._layout.addWidget(section)
            self._sections.append(section)
        self._layout.addStretch(1)
