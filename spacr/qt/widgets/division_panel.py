"""Division analysis results: what tracked cells do before they divide.

The Make Masks timelapse run writes one results folder per tracked field
under ``tracks/division/`` when ``timelapse_division_analysis`` is on (see
:func:`spacr.division_analysis.write_division_results`). This panel reads
such a folder, or finds the result folders under any folder above them, and
draws the three figures from its tables: the aligned trajectories of dividing
cells against matched controls and the shuffled alignment, the forest plot
of effect sizes, and the size-control plot. Nothing is recomputed; the
tables on disk are what is drawn.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)

from ..i18n import tr

LOG = logging.getLogger(__name__)

_FIGURE_TABS = ("trajectories", "forest", "size_control")


def _result_folders(folder: str) -> List[str]:
    """Division results folders at or below ``folder``, sorted by path."""
    if os.path.isfile(os.path.join(folder, "summary.json")):
        return [folder]
    found = []
    for root, _dirs, files in os.walk(folder):
        if "summary.json" in files and "effects.csv" in files:
            found.append(root)
    return sorted(found)


class DivisionAnalysisPanel(QWidget):
    """The division analysis figures of one tracked field, read from disk."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the panel with its folder button, field list and figure tabs.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setObjectName("DivisionAnalysisPanel")
        self._folders: List[str] = []
        self._canvases = []

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel(tr("Division analysis")))
        self._open = QPushButton(tr("Open results…"))
        self._open.setObjectName("GhostButton")
        self._open.setToolTip(tr(
            "Choose a division analysis results folder, or a folder above "
            "them such as tracks/division, to draw its figures."))
        self._open.clicked.connect(lambda _checked=False: self.browse())
        row.addWidget(self._open)
        self._choice = QComboBox()
        self._choice.setToolTip(tr("The tracked field whose results are shown."))
        self._choice.currentIndexChanged.connect(self._show_choice)
        self._choice.setVisible(False)
        row.addWidget(self._choice, 1)
        row.addStretch(1)
        column.addLayout(row)

        self._status = QLabel(tr(
            "Turn on timelapse_division_analysis and run, then open "
            "tracks/division here."))
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        column.addWidget(self._status)

        self._tabs = QTabWidget()
        self._tabs.setVisible(False)
        column.addWidget(self._tabs, 1)

    def browse(self) -> bool:
        """Ask for a results folder and load it.

        :returns: whether a folder with results was chosen and loaded.
        """
        folder = QFileDialog.getExistingDirectory(
            self, tr("Choose division analysis results"))
        return bool(folder) and self.load_folder(folder)

    def load_folder(self, folder: str) -> bool:
        """Show the results in ``folder``, or list those found below it.

        :param folder: a folder written by
            :func:`spacr.division_analysis.write_division_results`, or any
            folder above such folders; with several, the first is shown and
            the others can be picked from the list.
        :returns: whether results were found and drawn.
        """
        self._folders = _result_folders(str(folder)) if os.path.isdir(str(folder)) else []
        self._choice.blockSignals(True)
        self._choice.clear()
        for path in self._folders:
            self._choice.addItem(os.path.relpath(path, str(folder))
                                 if path != str(folder) else os.path.basename(path), path)
        self._choice.blockSignals(False)
        self._choice.setVisible(len(self._folders) > 1)
        if not self._folders:
            self._status.setText(tr("No division analysis results in {folder}.").format(
                folder=folder))
            self._tabs.setVisible(False)
            return False
        return self._draw(self._folders[0])

    def _show_choice(self, index: int) -> None:
        """Draw the folder picked in the field list."""
        if 0 <= index < len(self._folders):
            self._draw(self._folders[index])

    def _draw(self, folder: str) -> bool:
        """Read one results folder and put its figures in the tabs."""
        from ...division_analysis import division_figures, read_division_results
        from ...figures.style import figure_style, theme_target
        from ..screens.train_compare import panel_canvas_class

        try:
            results = read_division_results(folder)
            with figure_style(theme_target()):
                figures = division_figures(results)
        except Exception as exc:                                  # noqa: BLE001
            LOG.warning("division results in %s could not be drawn: %s", folder, exc)
            self._status.setText(tr("Could not read {folder}: {error}").format(
                folder=folder, error=exc))
            self._tabs.setVisible(False)
            return False
        self._tabs.clear()
        self._canvases = []
        canvas_class = panel_canvas_class()
        titles = {"trajectories": tr("Trajectories"), "forest": tr("Effect sizes"),
                  "size_control": tr("Size control")}
        for key in _FIGURE_TABS:
            canvas = canvas_class(figures[key])
            self._canvases.append(canvas)
            self._tabs.addTab(canvas, titles[key])
        summary = results.get("summary", {})
        self._status.setText(tr(
            "{kept} of {events} {mode} divisions aligned, {controls} with a "
            "matched control; window {window} frames.").format(
                kept=summary.get("n_events_kept", "?"),
                events=summary.get("n_events", "?"),
                mode=summary.get("mode", ""),
                controls=summary.get("n_controls_matched", "?"),
                window=summary.get("window", "?")))
        self._tabs.setVisible(True)
        return True


def _build_division_panel(card) -> DivisionAnalysisPanel:
    """Put a division analysis panel in ``card`` behind the alpha gate.

    :param card: a card with a ``body_layout``.
    :returns: the panel, hidden unless Show alpha features is on.
    """
    from ..preferences import _apply_alpha_widgets

    panel = DivisionAnalysisPanel(card)
    card.body_layout.addWidget(panel, 1)
    _apply_alpha_widgets(panel)
    return panel
