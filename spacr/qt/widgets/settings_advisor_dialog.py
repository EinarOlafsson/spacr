"""Review data-dependent regression-setting recommendations.

The first page asks only for information that cannot be inferred from the
tables. The second compares each proposed value with the current value and
explains the evidence for the change. No setting is written until the user
selects Apply. Calculations are provided by :mod:`spacr.settings_advisor`.
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

from ...settings_advisor import (QUESTIONS, ROW_CAP, Advice, Reading, advise,
                                 questions_for)
from ..i18n import set_translatable_text, tr

LOG = logging.getLogger("spacr.qt.settings_advisor")

_ADVISOR_SUMMARY_SOURCE = (
    "Read {plates} plate(s), {wells} well(s), {guides} guide(s), {genes} "
    "gene(s) and {objects} object row(s)."
)
_ADVISOR_SUMMARY_CAPPED_SOURCE = (
    "Read {plates} plate(s), {wells} well(s), {guides} guide(s), {genes} "
    "gene(s) and {objects} object row(s). The response was read from the "
    "first {objects} object row(s); the score table is larger than the "
    "{row_cap}-row sample this reads."
)
_ADVISOR_CHROME_SOURCES = (
    "These are the only questions your tables cannot answer. Everything "
    "else on the next page was measured.",
    "in 1,000",
    _ADVISOR_SUMMARY_SOURCE,
    _ADVISOR_SUMMARY_CAPPED_SOURCE,
    "unchanged",
    "Not decided — left unchanged:",
    "Every setting this advisor can decide was decided.",
    "none",
    "on",
    "off",
    "See what it would change",
    "Apply",
)

# Exact presentation strings assembled from the headless question records are
# not visible to the Qt literal extractor. Export them as one deterministic
# private inventory for the runtime-catalog builder and its coverage tests.
_SETTINGS_ADVISOR_UI_SOURCES = tuple(dict.fromkeys((
    *_ADVISOR_CHROME_SOURCES,
    *(question.prompt for question in QUESTIONS),
    *(question.why_it_matters for question in QUESTIONS),
    *(label for question in QUESTIONS for _value, label in question.options),
)))


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
            box.setSuffix(" " + tr("in 1,000"))
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
        """Return the current answer for each displayed question."""
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
        self._advice: Optional[Advice] = None
        self._current: Dict[str, Any] = {}
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
        """Display proposed values, current values, and supporting reasons."""
        self._advice = advice
        self._current = dict(current or {})
        self._render()

    def _render(self, language: Optional[str] = None) -> None:
        """Render the current proposal using localized application chrome."""
        advice = self._advice
        current = self._current
        if advice is None:
            return
        reading = advice.reading
        if reading is not None:
            values = {
                "plates": f"{reading.plates}",
                "wells": f"{reading.wells:,}",
                "guides": f"{reading.guides:,}",
                "genes": f"{reading.genes:,}",
                "objects": f"{reading.n_response:,}",
                "row_cap": f"{ROW_CAP:,}",
            }
            if reading.capped:
                set_translatable_text(
                    self.summary, _ADVISOR_SUMMARY_CAPPED_SOURCE,
                    language=language, **values,
                )
            else:
                set_translatable_text(
                    self.summary, _ADVISOR_SUMMARY_SOURCE,
                    language=language, **values,
                )
        self.table.setRowCount(len(advice.chosen))
        for row, choice in enumerate(advice.chosen):
            was = current.get(choice.key, "—")
            # UNCHANGED IS SAID, NOT HIDDEN. A proposal that listed only the
            # differences would read as "everything else is wrong", when
            # most of a tuned panel is usually already right.
            same = _same(was, choice.value)
            cells = (choice.key, _text(was, language),
                     _text(choice.value, language)
                     + (f"  ({tr('unchanged', language)})" if same else ""),
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
                tr("Not decided — left unchanged:", language) + "\n"
                + "\n".join(f"  • {u.key}: {u.why}" for u in advice.undecided))
        else:
            self.undecided.setPlainText(
                tr("Every setting this advisor can decide was decided.",
                   language))

    def retranslate_dynamic_content(self, language: str) -> None:
        """Refresh proposal chrome after the application language changes."""
        self._render(language)


def _text(value: Any, language: Optional[str] = None) -> str:
    if value is None:
        return tr("none", language)
    if isinstance(value, bool):
        return tr("on" if value else "off", language)
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
        """Return to the question page without discarding current answers."""
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
        """Return the most recently displayed proposal, if any."""
        return self._advice

    def accepted_settings(self) -> Dict[str, Any]:
        """What to write. Empty until the proposal has actually been seen."""
        if self._advice is None:
            return {}
        return self._advice.as_settings()
