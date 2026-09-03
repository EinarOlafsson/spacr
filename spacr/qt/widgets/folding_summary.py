"""Render plain-text run summaries as foldable sections.

The verdict opens by default and later sections begin collapsed. Sections are
parsed from underline-style headings in the text because the input may be a
saved summary from another spaCR version or an unstructured statsmodels
summary, not a live ``RunSummary`` object. Text without recognized headings is
shown unchanged.

The source summary remains ordinary plain text suitable for terminals and run
artifacts; this widget adds navigation without becoming a second serializer.
"""
import re
import html
import logging
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import (QColor, QFontDatabase, QSyntaxHighlighter,
                           QTextCharFormat)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from .collapsible_section import CollapsibleSection

LOG = logging.getLogger(__name__)

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

    ``format_run_summary`` uses a fixed-width label column for terminal and
    file output. This parser recovers those rows so the Qt view can reflow the
    value column to the available width. The label width is inferred from the
    first labelled row for compatibility with summaries from other versions.
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


class _RejectionHighlighter(QSyntaxHighlighter):
    """Highlight rejected assumptions and blocking recommendations.

    Cautions and small diagnostic values retain the normal text colour. The
    error colour comes from the active theme so rejection markers remain
    readable in both light and dark themes.
    """

    #: What marks a line as a broken assumption. `REJECTED at` is written by
    #: `regression_summary._verdict`; the `!` prefix is what
    #: `run_recommendations` puts on a blocking recommendation.
    MARKERS = ("REJECTED at", "  ! ")

    def __init__(self, document, colour: str):
        super().__init__(document)
        self._format = QTextCharFormat()
        self._format.setForeground(QColor(colour))

    def highlightBlock(self, text: str) -> None:   # noqa: N802 - Qt naming
        line = str(text)
        if any(marker in line for marker in self.MARKERS):
            # THE WHOLE LINE, not the matched word. "REJECTED at 0.05" is
            # the verdict on the sentence it sits in, and colouring three
            # words inside a grey line reads as emphasis rather than as a
            # state.
            self.setFormat(0, len(line), self._format)


class FoldingSummaryView(QScrollArea):
    """A drop-in for the Summary tab's ``QPlainTextEdit``.

    Keeps ``setPlainText`` and ``toPlainText`` so the panel that fills it does
    not have to know which it got, and so the text a test reads is the text
    the file holds.

    :param parent: parent widget.
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
        # A WIDGET, NOT A BARE LAYOUT, and that is the whole bug.
        #
        # Reported 2026-08-20: "in the summary section there is a giant save
        # button in the background that can only be pressed on the side of
        # the summay text, presumably because the text is in front and
        # blocking."
        #
        # `_clear` empties the body layout with takeAt and deletes what it
        # finds -- but it only finds WIDGETS. A bare QHBoxLayout was taken
        # out and dropped, while the buttons inside it stayed children of
        # `_body` with nothing laying them out: still visible, still
        # clickable, stuck at whatever geometry they last had, and painted
        # UNDER the sections added afterwards. Hence a button in the
        # background reachable only where no text covered it.
        #
        # Held as one widget, it is taken out and put back like everything
        # else, and there is nothing left behind to strand.
        self._actions = QWidget(self._body)
        actions = QHBoxLayout(self._actions)
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
        self._layout.addWidget(self._actions)

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
        if clipboard is None:
            # No clipboard on this platform or in this session. Declining
            # is the whole behaviour: a copy button that raises is worse
            # than one that does nothing.
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
        """Empty the body, KEEPING the action row to put back.

        It is taken out with everything else -- so a rebuild controls where
        it sits -- but never deleted, because Copy and Save belong to the
        panel rather than to whichever summary is currently in it.
        """
        self._sections = []
        keep = getattr(self, "_actions", None)
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is None or widget is keep:
                continue
            widget.setParent(None)
            widget.deleteLater()
        if keep is not None:
            self._layout.addWidget(keep)

    def _reading_surface(self) -> str:
        """Return an ``rgba(...)`` surface that keeps summary text legible.

        The active ``surface_alt`` colour and the pane opacity separate text
        from the animated background without making the panel fully opaque --
        the colour and the alpha are asked for under the same role, so the
        alpha is the one solved for legibility over that colour. Fall back to
        transparency when the theme cannot be resolved.
        """
        try:
            from ..preferences import get_pane_opacity, resolve_effective_theme
            from ..theme import palette_for, panel_alpha

            theme = resolve_effective_theme()
            colour = str(palette_for(theme).get("surface_alt", "#161719"))
            alpha = panel_alpha(theme, "surface_alt", get_pane_opacity())
            r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
            return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, float(alpha))):.3f})"
        except Exception:                                   # noqa: BLE001
            LOG.debug("could not resolve the summary reading surface",
                      exc_info=True)
            return "transparent"

    def _table(self, rows: list) -> QWidget:
        """Build an aligned two-column view for one summary section.

        Labels are bold and values wrap to the available width. A text browser
        is used so the value column can reflow without a horizontal scroll
        bar.
        """
        from PySide6.QtWidgets import QTextBrowser

        view = QTextBrowser(self._body)
        view.setOpenExternalLinks(True)
        view.setFrameShape(QTextBrowser.NoFrame)
        # A SURFACE, not a slab and not a window onto the backdrop. See
        # `_reading_surface` -- fully transparent put the animated
        # background directly behind the type.
        view.viewport().setAutoFillBackground(False)
        view.setStyleSheet(
            f"QTextBrowser {{ background: {self._reading_surface()};"
            f" border-radius: 6px; }}")
        # RED FOR A REJECTED ASSUMPTION, IN THE TABLE TOO (225). Most of a
        # summary's rows arrive here rather than at `_block` -- anything
        # shaped "label: value" is a row -- so a highlighter on the block
        # path alone colours almost nothing. Found exactly that way: the
        # highlighter worked and the assumptions were still grey, because
        # they were never blocks.
        #
        # Inline here rather than another highlighter: this is already HTML
        # being built, and a second mechanism for one colour is a second
        # thing to keep in step.
        try:
            from ..theme import active_palette

            alarm = active_palette()["error"]
        except Exception:                                    # noqa: BLE001
            alarm = ""

        def _tint(label: str, value: str) -> str:
            if not alarm:
                return ""
            line = f"{label} {value}"
            if any(marker in line
                   for marker in _RejectionHighlighter.MARKERS):
                return f"color:{alarm};"
            return ""

        cells = []
        for label, value in rows:
            tint = _tint(label, value)
            if label:
                cells.append(
                    f"<tr><td style='padding:1px 14px 1px 0;"
                    f"white-space:nowrap;vertical-align:top;{tint}'><b>"
                    f"{html.escape(label)}</b></td>"
                    f"<td style='padding:1px 0;{tint}'>"
                    f"{html.escape(value)}</td></tr>")
            else:
                cells.append(
                    f"<tr><td colspan='2' style='padding:4px 0;{tint}'>"
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
        # The statsmodels summary is the longest thing in this panel and the
        # hardest to read over a moving picture. Same surface as the tables,
        # for the same reason.
        view.viewport().setAutoFillBackground(False)
        view.setStyleSheet(
            f"QPlainTextEdit {{ background: {self._reading_surface()};"
            f" border-radius: 6px; }}")
        # HELD ON THE VIEW, or it is garbage collected the moment this
        # function returns and highlights nothing -- silently, which is the
        # only way a highlighter ever fails.
        try:
            from ..theme import active_palette

            view._spacr_highlighter = _RejectionHighlighter(
                view.document(), active_palette()["error"])
        except Exception:                                    # noqa: BLE001
            pass
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
