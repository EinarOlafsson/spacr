"""Compare a measurement between the picked cells and the rest (177 F).

The window the Cells tab's "Compare a measurement…" button opens. It owns no
statistics and no grouping of its own: `spacr.gene_measurement_compare` does
the work, `spacr.sp_stats` chooses the test, and the groups are whatever the
cell picker already decided.
"""
import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLabel, QPlainTextEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ...gene_measurement_compare import (LEVELS, PLOTS, build, plot, save,
                                         with_statistics)

LOG = logging.getLogger("spacr.qt.measurement_compare")


class MeasurementCompareDialog(QDialog):
    """One measurement, the picked genes against the rest.

    :param objects: every object row behind the montage, picked or not.
    :param groups: ``{gene: index values}`` from the picker.
    :param settings: the settings that produced the run, saved beside the
        figure so the folder answers "what made this" without a second file.
    """

    def __init__(self, objects, groups: Dict[str, Any],
                 parent: Optional[QWidget] = None,
                 settings: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Compare a measurement")
        self._objects = objects
        self._groups = dict(groups or {})
        self._settings = dict(settings or {})
        self._comparison = None
        self._canvas = None

        layout = QVBoxLayout(self)
        row = QHBoxLayout()

        row.addWidget(QLabel("measurement"))
        self.measurement = QComboBox()
        # OFFERED FROM THE DATA, never typed: the same rule every other
        # chooser in spaCR follows, and the reason `object_array` stopped
        # being a text box.
        for name in self._numeric_columns():
            self.measurement.addItem(str(name), str(name))
        self.measurement.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.measurement, 1)

        row.addWidget(QLabel("level"))
        self.level = QComboBox()
        for value, why in LEVELS:
            self.level.addItem(value, value)
            self.level.setItemData(self.level.count() - 1, why, Qt.ToolTipRole)
        self.level.setCurrentIndex(1)          # well: the unit the screen randomises
        self.level.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.level)

        row.addWidget(QLabel("plot"))
        self.kind = QComboBox()
        for value, label in PLOTS:
            self.kind.addItem(label, value)
        self.kind.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.kind)

        self.save_button = QPushButton("Save…")
        self.save_button.setToolTip(
            "Write the figure, the plotted data, the statistics, the "
            "settings and the cell images into ONE folder.")
        self.save_button.clicked.connect(self.save_everything)
        row.addWidget(self.save_button)
        layout.addLayout(row)

        self._figure_holder = QVBoxLayout()
        layout.addLayout(self._figure_holder, 1)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(140)
        layout.addWidget(self.report)

        self.resize(900, 720)
        self.refresh()

    # ------------------------------------------------------------- the data

    def _numeric_columns(self) -> list:
        """Every measurement on these objects, identifiers left out."""
        try:
            import pandas as pd

            from ...gene_measurement_sweep import is_measurement

            return [c for c in self._objects.columns
                    if pd.api.types.is_numeric_dtype(self._objects[c])
                    and is_measurement(c)]
        except Exception:                                    # noqa: BLE001
            return []

    def comparison(self):
        return self._comparison

    def refresh(self, *_args):
        """Rebuild, retest and redraw. Returns the comparison, or ``None``."""
        measurement = str(self.measurement.currentData() or "")
        if not measurement:
            self.report.setPlainText(
                "These objects carry no measurement column to compare.")
            return None
        level = str(self.level.currentData() or "well")
        self._comparison = with_statistics(
            build(self._objects, measurement, groups=self._groups,
                  level=level))
        self._draw()
        self._report()
        return self._comparison

    def _draw(self):
        from .graph_builder import _canvas_class

        while self._figure_holder.count():
            item = self._figure_holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        figure = plot(self._comparison,
                      kind=str(self.kind.currentData() or "jitter_box"))
        if figure is None:
            self._canvas = None
            return
        self._canvas = _canvas_class()(figure)
        self._figure_holder.addWidget(self._canvas)

    def _report(self):
        """n, the assumption checks, the test and why -- in that order.

        THE ORDER IS THE ARGUMENT. A test name above the checks that chose it
        reads as a decision already made; below them it reads as the
        consequence it is.
        """
        comparison = self._comparison
        if comparison is None:
            return
        lines = [f"{comparison.measurement} · {comparison.level} level"]
        if comparison.note:
            lines.append(comparison.note)
        counts = comparison.counts()
        if counts:
            lines.append("n: " + ", ".join(f"{k} = {v}"
                                           for k, v in counts.items()))
        for row in comparison.statistics or []:
            for key in ("Normality", "Equal variance"):
                if row.get(key):
                    lines.append(f"{key.lower()}: {row[key]}")
            name = row.get("Test Name", "")
            p_value = row.get("p-value")
            effect = row.get("Effect Size")
            said = f"test: {name}"
            if p_value is not None:
                said += f" · p = {p_value}"
            if effect is not None:
                said += f" · effect size = {effect}"
            lines.append(said)
            if row.get("Why This Test"):
                lines.append(f"why: {row['Why This Test']}")
        if not (comparison.statistics or []):
            lines.append("no test: a comparison needs two groups with "
                         "something in them.")
        self.report.setPlainText("\n".join(lines))

    # ------------------------------------------------------------- saving

    def save_everything(self, folder: str = "") -> dict:
        """Write everything into one folder. Returns what was written."""
        if self._comparison is None:
            return {}
        chosen = str(folder or "")
        if not chosen:
            chosen = QFileDialog.getExistingDirectory(
                self, "Save the comparison into a folder")
        if not chosen:
            return {}
        try:
            written = save(self._comparison, chosen,
                           kind=str(self.kind.currentData() or "jitter_box"),
                           settings=self._settings)
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not save the comparison", exc_info=True)
            self.report.setPlainText(
                f"{self.report.toPlainText()}\n\nCould not save: {exc}")
            return {}
        self.report.setPlainText(
            f"{self.report.toPlainText()}\n\nSaved "
            f"{len(written)} item(s) to {chosen}")
        return written
