"""The aggregation chosen for every merged column, shown and overridable.

Merging four pathogens onto their cell means picking one number per column,
and which number depends on what the column measures. The rules get that right
most of the time -- and a default that is right most of the time is a wrong
answer nobody can find the rest of the time. So the whole plan is shown, with
every row changeable.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel, QLineEdit,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from ...merge_tables import AGGREGATIONS, aggregation_for, aggregation_plan
from ..theme import SPACING


class AggregationRulesDialog(QDialog):
    """One row per column: the rule, and a dropdown to change it.

    Only columns that were CHANGED become overrides. Recording every row would
    freeze today's defaults into the settings, so a later improvement to the
    rules would never reach a user who had once opened this window.
    """

    rules_changed = Signal(dict)

    def __init__(self, frame: pd.DataFrame, parent=None, *,
                 overrides: Optional[Mapping[str, str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Aggregation rules")
        self.setObjectName("AggregationRulesDialog")
        self._overrides: Dict[str, str] = dict(overrides or {})
        self._boxes: Dict[str, QComboBox] = {}

        outer = QVBoxLayout(self)
        outer.setSpacing(SPACING["sm"])

        note = QLabel(
            "How each measurement combines when several child objects roll up "
            "onto one parent. Four pathogens in a cell: their areas SUM, "
            "their minimum intensity is the MINIMUM of the four, their mean "
            "is the mean. Change any row that is wrong for your data.", self)
        note.setWordWrap(True)
        outer.addWidget(note)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("filter measurements")
        self.search.textChanged.connect(self._filter)
        outer.addWidget(self.search)

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("AggregationRules")
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Measurement", "Combine by"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        outer.addWidget(self.tree, 1)

        self._fill(frame)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

    def _fill(self, frame: pd.DataFrame) -> None:
        plan = aggregation_plan(frame, overrides=self._overrides)
        for column, how in sorted(plan.items()):
            item = QTreeWidgetItem([str(column), ""])
            self.tree.addTopLevelItem(item)
            box = QComboBox(self.tree)
            box.addItems(AGGREGATIONS)
            box.setCurrentText(how)
            box.currentTextChanged.connect(
                lambda value, c=column: self._on_changed(c, value))
            self.tree.setItemWidget(item, 1, box)
            self._boxes[str(column)] = box

    def _on_changed(self, column: str, value: str) -> None:
        """Record a change, and UNRECORD one that returns to the default.

        So the overrides hold only real decisions: a row set back to what the
        rule already said is not an override, and keeping it would pin today's
        default forever.
        """
        default = aggregation_for(
            column, numeric=True)
        if value == default:
            self._overrides.pop(column, None)
        else:
            self._overrides[column] = value
        self.rules_changed.emit(dict(self._overrides))

    def _filter(self, text: str) -> None:
        needle = str(text).strip().lower()
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            item.setHidden(bool(needle) and needle not in item.text(0).lower())

    def overrides(self) -> Dict[str, str]:
        return dict(self._overrides)
