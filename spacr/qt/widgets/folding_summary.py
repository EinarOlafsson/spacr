"""The run summary with its sections folded, for instruction 168 D.

    "THE PANEL FOLDS IT. The Summary tab shows the verdict expanded and each
    section collapsed, with the section headings as the outline. The file on
    disk stays plain text and stays readable in a terminal, because it is a
    run artefact before it is a widget."

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
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QLabel, QPlainTextEdit, QScrollArea,
                               QVBoxLayout, QWidget)

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
            # THE VERDICT OPEN, EVERYTHING ELSE FOLDED -- the headings are
            # then the outline the instruction asks for.
            expanded = heading.upper() == ANSWER_HEADING
            section = CollapsibleSection(heading, self._block(body),
                                         expanded=expanded, parent=self._body)
            self._layout.addWidget(section)
            self._sections.append(section)
        self._layout.addStretch(1)
