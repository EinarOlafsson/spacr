"""The Grid search panel of the OPS Live preview's Spots & genotyping tab (item 960, alpha).

Pick a few representative tiles, enter a grid of spot and decode settings,
and score every grid point on those tiles with the decode itself
(:mod:`spacr.ops_grid_search`), off the GUI thread. The
sortable table shows each point's spots, mapping rate, cells assigned, decoy
FDR and barcode-shuffle call rate; the plot shows yield against the shuffle
rate with the current default and the Pareto front. Apply to settings
writes the selected point into the form and records the grid run.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QTableWidget, QVBoxLayout, QWidget)

from ..i18n import tr

LOG = logging.getLogger("spacr.qt.widgets.ops_grid_search_panel")

#: Grid workers still running, held here rather than by the panel so a
#: closed window never destroys a thread that is still scoring.
_RUNNING: set = set()

#: The pooled columns the table shows, in order.
_COLUMNS = ("point", "is_default", "pareto", "ops_read_mode", "ops_read_threshold",
            "ops_prefix_min_read_quality", "ops_prefix_max_distance", "spots", "reads",
            "mapping_pct", "assignment_pct", "decoy_fdr_pct", "shuffle_rate_pct")


class _GridWorker(QThread):
    """Run one grid search off the GUI thread; any exception comes back as text."""

    done = Signal(object, str)

    def __init__(self, settings: Mapping[str, Any], run: Optional[Callable[..., Any]] = None,
                 parent=None):
        """Carry one grid run onto a worker thread.

        :param settings: a copy of the form's settings with the panel's grid
            settings laid over them.
        :param run: replaces :func:`spacr.ops_grid_search._run_grid_search`,
            for tests.
        :param parent: parent object.
        """
        super().__init__(parent)
        self._settings = dict(settings)
        self._run = run

    def run(self) -> None:
        """Score the grid and hand back the result."""
        try:
            if self._run is None:
                from spacr.ops_grid_search import _run_grid_search

                result = _run_grid_search(self._settings)
            else:
                result = self._run(self._settings)
            self.done.emit(result, "")
        except Exception as error:                            # noqa: BLE001
            LOG.info("OPS grid search failed", exc_info=True)
            self.done.emit(None, f"{type(error).__name__}: {error}")


class _OpsGridSearchPanel(QWidget):
    """The Grid search panel: tiles, grid, run, table, plot and apply.

    :param collect: returns the form's settings.
    :param apply: takes the settings to write into the form.
    :param parent: parent widget.
    :param run: replaces :func:`spacr.ops_grid_search._run_grid_search`, for
        tests.
    """

    applied = Signal(str, int)

    def __init__(self, collect: Callable[[], Mapping[str, Any]],
                 apply: Callable[[Dict[str, Any]], Any], parent=None, *,
                 run: Optional[Callable[..., Any]] = None):
        """Lay out the tile and grid inputs, the buttons, the table and the plot."""
        super().__init__(parent)
        self.setObjectName("OpsGridSearchPanel")
        self._collect, self._apply, self._run = collect, apply, run
        self._worker: Optional[_GridWorker] = None
        self._result: Optional[Mapping[str, Any]] = None
        self._settings: Dict[str, Any] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        title = QLabel(tr("Grid search"), self)
        title.setObjectName("OpsGridSearchTitle")
        layout.addWidget(title)
        blurb = QLabel(tr("Score a grid of spot and decode settings on a few tiles before "
                          "the full run."), self)
        blurb.setWordWrap(True)
        layout.addWidget(blurb)
        form = QFormLayout()
        self.mode = QComboBox(self)
        self.mode.setObjectName("OpsGridTileMode")
        self.mode.addItems(["stratified", "edge_centre", "user"])
        form.addRow(tr("Tile pick"), self.mode)
        self.n_tiles = QSpinBox(self)
        self.n_tiles.setObjectName("OpsGridTiles")
        self.n_tiles.setRange(1, 96)
        self.n_tiles.setValue(8)
        form.addRow(tr("How many tiles"), self.n_tiles)
        self.seed = QSpinBox(self)
        self.seed.setObjectName("OpsGridSeed")
        self.seed.setRange(0, 1_000_000)
        form.addRow(tr("Seed"), self.seed)
        self.tiles = QLineEdit(self)
        self.tiles.setObjectName("OpsGridTileList")
        self.tiles.setPlaceholderText("W1:29, W2:100")
        form.addRow(tr("Tiles (plate:well:site)"), self.tiles)
        self.grid = QLineEdit(self)
        self.grid.setObjectName("OpsGridParams")
        self.grid.setPlaceholderText("ops_read_threshold=40,80,160; ops_read_mode=log,raw")
        form.addRow(tr("Settings grid"), self.grid)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.status = QLabel("", self)
        self.status.setObjectName("OpsGridStatus")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.run_button = QPushButton(tr("Run grid"), self)
        self.run_button.setObjectName("OpsGridRunButton")
        self.run_button.clicked.connect(self.run_grid)
        row.addWidget(self.run_button)
        self.apply_button = QPushButton(tr("Apply to settings"), self)
        self.apply_button.setObjectName("OpsGridApplyButton")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_selected)
        row.addWidget(self.apply_button)
        layout.addLayout(row)
        self.table = QTableWidget(0, len(_COLUMNS), self)
        self.table.setObjectName("OpsGridTable")
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        from .sortable_table import install_sorting

        install_sorting(self.table)
        layout.addWidget(self.table, 1)
        self.plot = QLabel(self)
        self.plot.setObjectName("OpsGridPlot")
        self.plot.setAlignment(Qt.AlignCenter)
        self.plot.setMinimumHeight(160)
        layout.addWidget(self.plot, 1)
        self._load_form(collect() or {})

    def _load_form(self, settings: Mapping[str, Any]) -> None:
        """Fill the inputs from the form's ops_grid settings."""
        mode = str(settings.get("ops_grid_tile_mode") or "stratified")
        if self.mode.findText(mode) >= 0:
            self.mode.setCurrentText(mode)
        self.n_tiles.setValue(int(settings.get("ops_grid_n_tiles") or 8))
        self.seed.setValue(int(settings.get("ops_grid_seed") or 0))
        tiles = settings.get("ops_grid_tiles")
        self.tiles.setText(",".join(tiles) if isinstance(tiles, (list, tuple)) else str(tiles or ""))
        grid = settings.get("ops_grid_params")
        if isinstance(grid, Mapping):
            grid = "; ".join(f"{k}={','.join(str(v) for v in values)}"
                             for k, values in grid.items())
        self.grid.setText(str(grid or ""))

    def grid_settings(self) -> Dict[str, Any]:
        """The panel's inputs as ops_grid settings."""
        return {"ops_grid_tile_mode": self.mode.currentText(),
                "ops_grid_n_tiles": int(self.n_tiles.value()),
                "ops_grid_seed": int(self.seed.value()),
                "ops_grid_tiles": self.tiles.text().strip() or None,
                "ops_grid_params": self.grid.text().strip() or None}

    def run_grid(self) -> bool:
        """Start a grid run with the form's settings and the panel's inputs.

        :returns: False when a run is already going or the grid does not
            parse, True when one started.
        """
        if self._worker is not None and self._worker.isRunning():
            return False
        from spacr.ops_grid_search import _parse_grid

        settings = {**dict(self._collect() or {}), **self.grid_settings()}
        try:
            _parse_grid(settings["ops_grid_params"])
        except ValueError as error:
            self.status.setText(str(error))
            return False
        self._settings = settings
        self.run_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status.setText(tr("Grid search running..."))
        worker = _GridWorker(settings, self._run)
        _RUNNING.add(worker)
        worker.finished.connect(lambda: _RUNNING.discard(worker))
        worker.done.connect(self._on_done)
        self._worker = worker
        worker.start()
        return True

    def _on_done(self, result, error: str) -> None:
        """Show a finished run's table and plot, or why it failed."""
        self.run_button.setEnabled(True)
        if result is None:
            self.status.setText(error)
            return
        self.show_result(result)

    def show_result(self, result: Mapping[str, Any]) -> None:
        """Put a grid run's pooled table and plot on the panel."""
        from .sortable_table import table_item

        self._result = result
        table = result["table"]
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(table))
        for r, (_i, values) in enumerate(table.iterrows()):
            for c, column in enumerate(_COLUMNS):
                value = values.get(column, "")
                text = f"{value:.3f}" if isinstance(value, float) else str(value)
                self.table.setItem(r, c, table_item(text, key=value if not isinstance(
                    value, str) else None))
        self.table.setSortingEnabled(sorting)
        plot = result.get("plot")
        if plot:
            from PySide6.QtGui import QPixmap

            pixmap = QPixmap(str(plot))
            if not pixmap.isNull():
                self.plot.setPixmap(pixmap.scaledToHeight(max(160, self.plot.height()),
                                                          Qt.SmoothTransformation))
        self.status.setText(tr("Grid {run_id}: {points} points on {tiles} tiles in {seconds} s.",
                               run_id=result["run_id"], points=result["points"],
                               tiles=len(result["tiles"]), seconds=result["seconds"]))
        self.apply_button.setEnabled(True)

    def selected_point(self) -> Optional[int]:
        """The grid point of the selected row, or None."""
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return int(item.text()) if item is not None else None

    def apply_selected(self) -> bool:
        """Write the selected point and the grid run into the form.

        :returns: True when a point was applied.
        """
        point = self.selected_point()
        if self._result is None or point is None:
            self.status.setText(tr("Select a row to apply."))
            return False
        from spacr.ops_grid_search import _apply_grid_choice

        settings = _apply_grid_choice(self._settings, self._result, point)
        changed = {key: value for key, value in settings.items()
                   if key.startswith("ops_") and self._settings.get(key) != value}
        self._apply(changed)
        self.status.setText(tr("Applied point {point} of grid {run_id}.", point=point,
                               run_id=self._result["run_id"]))
        self.applied.emit(str(self._result["run_id"]), int(point))
        return True


def _attach_grid_search(live_panel, host) -> _OpsGridSearchPanel:
    """Add the Grid search panel under the Spots & genotyping tab of the OPS Live preview.

    :param live_panel: the OPS Live preview panel (item 971); its
        ``page('spots')`` is the tab the panel joins.
    :param host: the OPS screen: ``_settings_model.collect()`` gives the
        form's settings and ``apply_settings_dict`` writes them back.
    :returns: the panel.
    """
    def collect() -> Mapping[str, Any]:
        """The OPS form's settings now."""
        model = getattr(host, "_settings_model", None)
        return model.collect() if model is not None else {}

    page = live_panel.page("spots")
    panel = _OpsGridSearchPanel(collect, host.apply_settings_dict, page)
    page.layout().addWidget(panel)
    return panel
