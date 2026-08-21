"""Ask the four questions, then propose the settings (192).

    "its fine if the button triggers a popup that asks the user questions
    about their data that cant be determined by reading the data"

TWO PAGES, IN THAT ORDER. The questions first, because two of the answers
change what is proposed; the proposal second, with the CURRENT value beside
the new one and nothing written until Apply. A button that rewrites a
carefully-tuned panel with one click and no undo is a button people learn not
to press.

The arithmetic is entirely in :mod:`spacr.settings_advisor`, which is
headless and does not import Qt -- so what this window shows can be checked
from a script against the same screen.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ...settings_advisor import Advice, Reading, advise, questions_for

LOG = logging.getLogger("spacr.qt.settings_advisor")


def _muted(text: str, parent=None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label


class QuestionsPage(QWidget):
    """The questions the data cannot answer, and why each one matters."""

    def __init__(self, reading: Reading, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._fields: Dict[str, Any] = {}
        outer = QVBoxLayout(self)
        outer.addWidget(_muted(
            "These are the only questions your tables cannot answer. "
            "Everything else on the next page was measured.", self))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for question in questions_for(reading):
            widget = self._field_for(question)
            self._fields[question.key] = widget
            form.addRow(QLabel(question.prompt, self), widget)
            # WHY IT MATTERS, UNDER THE QUESTION. A user who cannot see what
            # an answer buys cannot answer it well, and the alternative is a
            # number typed to make the dialog go away.
            form.addRow("", _muted(question.why_it_matters, self))
        outer.addLayout(form)
        outer.addStretch(1)

    def _field_for(self, question):
        if question.kind == "number":
            box = QSpinBox(self)
            box.setRange(0, 1000)
            box.setValue(int(question.default or 0))
            box.setSuffix(" in 1,000")
            return box
        if question.kind == "text":
            box = QLineEdit(self)
            box.setText(str(question.default or ""))
            box.setPlaceholderText("gene or guide, comma separated")
            return box
        box = QComboBox(self)
        for value, label in question.options:
            box.addItem(label, value)
        index = box.findData(question.default)
        box.setCurrentIndex(max(index, 0))
        return box

    def answers(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QSpinBox):
                out[key] = int(widget.value())
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text().strip()
            else:
                out[key] = widget.currentData()
        return out


class ProposalPage(QWidget):
    """What would change, with the current value beside the new one."""

    #: The columns, in the order the argument runs: what is being set, what
    #: it is now, what it would become, and the measurement that decided it.
    HEADINGS = ("setting", "now", "proposed", "because")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        self.summary = _muted("", self)
        outer.addWidget(self.summary)
        self.table = QTableWidget(0, len(self.HEADINGS), self)
        self.table.setHorizontalHeaderLabels(list(self.HEADINGS))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.table, 1)
        self.undecided = QPlainTextEdit(self)
        self.undecided.setReadOnly(True)
        self.undecided.setMaximumHeight(120)
        outer.addWidget(self.undecided)

    def show_the_proposal(self, advice: Advice, current: Dict[str, Any]) -> None:
        reading = advice.reading
        if reading is not None:
            note = reading.sample_note()
            self.summary.setText(
                f"Read {reading.plates} plate(s), {reading.wells:,} well(s), "
                f"{reading.guides:,} guide(s), {reading.genes:,} gene(s) and "
                f"{reading.n_response:,} object row(s)."
                + (f" The response was read {note}." if note else ""))
        self.table.setRowCount(len(advice.chosen))
        for row, choice in enumerate(advice.chosen):
            was = current.get(choice.key, "—")
            # UNCHANGED IS SAID, NOT HIDDEN. A proposal that listed only the
            # differences would read as "everything else is wrong", when
            # most of a tuned panel is usually already right.
            same = _same(was, choice.value)
            cells = (choice.key, _text(was),
                     _text(choice.value) + ("  (unchanged)" if same else ""),
                     choice.why)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(choice.why)
                if same:
                    item.setForeground(Qt.gray)
                self.table.setItem(row, column, item)
        self.table.resizeRowsToContents()
        self.table.resizeColumnToContents(0)
        if advice.undecided:
            self.undecided.setPlainText(
                "NOT DECIDED — left exactly as they are:\n"
                + "\n".join(f"  • {u.key}: {u.why}" for u in advice.undecided))
        else:
            self.undecided.setPlainText(
                "Every setting this button decides was decided.")


def _text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _same(first: Any, second: Any) -> bool:
    """Whether the panel already holds what is being proposed."""
    if isinstance(first, bool) or isinstance(second, bool):
        return bool(first) == bool(second)
    try:
        return float(first) == float(second)
    except (TypeError, ValueError):
        return _text(first).strip().lower() == _text(second).strip().lower()


class SettingsAdvisorDialog(QDialog):
    """Questions, then proposal, then -- only on Apply -- the settings."""

    def __init__(self, reading: Reading, current: Dict[str, Any],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings for your data")
        self._reading = reading
        self._current = dict(current or {})
        self._advice: Optional[Advice] = None
        self.resize(900, 640)

        outer = QVBoxLayout(self)
        self.pages = QStackedWidget(self)
        self.questions = QuestionsPage(reading, self)
        self.proposal = ProposalPage(self)
        self.pages.addWidget(self.questions)
        self.pages.addWidget(self.proposal)
        outer.addWidget(self.pages, 1)

        row = QHBoxLayout()
        self.back = QPushButton("Back", self)
        self.back.clicked.connect(self.show_the_questions)
        self.back.setVisible(False)
        row.addWidget(self.back)
        row.addStretch(1)
        self.buttons = QDialogButtonBox(self)
        self.next = self.buttons.addButton("See what it would change",
                                           QDialogButtonBox.ActionRole)
        self.next.clicked.connect(self.show_the_proposal)
        self.apply = self.buttons.addButton("Apply",
                                            QDialogButtonBox.AcceptRole)
        self.apply.clicked.connect(self.accept)
        self.apply.setVisible(False)
        cancel = self.buttons.addButton(QDialogButtonBox.Cancel)
        cancel.clicked.connect(self.reject)
        row.addWidget(self.buttons)
        outer.addLayout(row)

    # ------------------------------------------------------------- the pages

    def show_the_questions(self) -> None:
        self.pages.setCurrentWidget(self.questions)
        self.back.setVisible(False)
        self.next.setVisible(True)
        self.apply.setVisible(False)

    def show_the_proposal(self) -> Advice:
        """Compute the advice from the answers and show it."""
        self._advice = advise(self._reading, self.questions.answers())
        self.proposal.show_the_proposal(self._advice, self._current)
        self.pages.setCurrentWidget(self.proposal)
        self.back.setVisible(True)
        self.next.setVisible(False)
        self.apply.setVisible(True)
        return self._advice

    # ------------------------------------------------------------ the result

    def advice(self) -> Optional[Advice]:
        return self._advice

    def accepted_settings(self) -> Dict[str, Any]:
        """What to write. Empty until the proposal has actually been seen."""
        if self._advice is None:
            return {}
        return self._advice.as_settings()
