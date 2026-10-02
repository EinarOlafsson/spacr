"""Read-only preview of Measure's existing plate calibration plan."""
from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..i18n import tr
from ..job_runner import JobRunner

__all__ = ()


class _ButtonPosition(QObject):
    def __init__(self, edit, button):
        super().__init__(edit)
        self._edit, self._button = edit, button
        edit.installEventFilter(self)
        self._place()

    def _place(self):
        height = self._button.sizeHint().height()
        self._edit.setTextMargins(0, 0, 0, height + 4)
        self._button.setGeometry(2, self._edit.height() - height - 2,
                                 min(self._button.sizeHint().width(),
                                     max(1, self._edit.width() - 4)), height)

    def eventFilter(self, watched, event):
        """Keep the action aligned as the input resizes.

        :param watched: Reference-well editor.
        :param event: Qt event being observed.
        :returns: False, preserving normal editor event handling.
        """
        if event.type() in (QEvent.Resize, QEvent.FontChange, QEvent.StyleChange):
            self._place()
        return False


def _attach_preview(widget, model):
    if widget.findChild(QPushButton, 'CalibrationPreviewButton') is not None:
        return
    button = QPushButton(tr('Preview'), widget)
    button.setObjectName('CalibrationPreviewButton')
    button.setToolTip(tr('Calculate reference-well gains without running Measure or changing files.'))
    if isinstance(widget, QLineEdit):
        widget.setMinimumHeight(widget.minimumSizeHint().height()
                                + button.sizeHint().height() + 6)
        widget._calibration_button_position = _ButtonPosition(widget, button)
    else:
        widget.layout().addWidget(button)

    def open_preview():
        dialog = _CalibrationPreview(lambda: _preview_settings(model), widget.window())
        QTimer.singleShot(0, dialog._start)
        dialog.exec()
        dialog.deleteLater()

    button.clicked.connect(open_preview)


def _preview_settings(model):
    # Only keys consumed by planning; no form mutation or source reads here.
    keys = {'src', 'timelapse', 'test_mode', 'intensity_calibration',
            'intensity_calibration_wells', 'intensity_calibration_statistic',
            'intensity_calibration_offset'}
    keys.update(key for key in model._defaults if key.endswith('_mask_dim'))
    return copy.deepcopy({key: model._valid_committed_value(key) for key in keys})


def _plan_gains(settings):
    """Use production planning on private settings, without starting a run."""
    from ...crops import reconcile_merged_mask_dims
    from ...image_quality import excluded_fields
    from ...io import _listdir_visible
    from ...measure import _prepare_measurement_calibration
    from ...settings import get_measure_crop_settings
    from ...utils import format_path_for_system, normalize_src_path

    if not settings.get('intensity_calibration'):
        raise ValueError(tr('Enable intensity calibration before previewing gains.'))
    if settings.get('test_mode'):
        raise ValueError(tr('Turn off Test mode to preview gains for all source fields.'))
    sources = normalize_src_path(settings.get('src'))
    sources = [sources] if isinstance(sources, str) else sources
    if not sources or any(not isinstance(src, str) or not src.strip() for src in sources):
        raise ValueError(tr('Choose a local source folder containing merged arrays.'))
    reports = []
    for source in sources:
        if QThread.currentThread().isInterruptionRequested():
            return []
        if '://' in source:
            raise ValueError(tr('Choose a local source folder containing merged arrays.'))
        folder = Path(format_path_for_system(source))
        if not folder.name.endswith('merged'):
            folder = folder / 'merged'
        options = copy.deepcopy(settings)
        options['src'] = str(folder)
        options = reconcile_merged_mask_dims(
            options, str(folder), explicit_keys={
                key for key in options if key.endswith('_mask_dim')})
        options = get_measure_crop_settings(options)
        excluded = excluded_fields(folder.parent)
        files = sorted(name for name in _listdir_visible(str(folder))
                       if name.endswith('.npy') and name not in excluded)
        if not files:
            raise ValueError(tr('No eligible merged arrays were found.'))
        full_plan, calibration = _prepare_measurement_calibration(options, files)
        # A scan failure must not look like a complete, usable preview.
        if full_plan.get('failures'):
            raise ValueError(tr('Some merged arrays could not be read. Check the source files.'))
        reports.append({'source': str(folder), 'calibration': calibration})
    return reports


class _CalibrationPreview(QDialog):
    """An inspect-only table whose worker never owns the editable form."""

    def __init__(self, settings_getter, parent=None, *, threaded=True):
        super().__init__(parent)
        self.setWindowTitle(tr('Calibration gains'))
        self.resize(980, 480)
        self._settings_getter = settings_getter
        self._generation = 0
        self._closed = False
        self._snapshot = None
        self._reports = []
        layout = QVBoxLayout(self)
        help_label = QLabel(' '.join([
            tr('Preview only: each source folder is planned separately.'),
            tr('The first plate by name is the reference.'),
            tr('Gains use the camera offset and reference wells; images and measurements are unchanged.')]), self)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.table = QTableWidget(0, 7, self)
        self.table.setObjectName('CalibrationGainTable')
        self.table.setHorizontalHeaderLabels([
            tr('Source'), tr('Plate'), tr('Reference plate'), tr('Channel'),
            tr('Gain'), tr('Reference statistic'), tr('Reference fields')])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 200)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.refresh = QPushButton(tr('Preview gains'), self)
        self.refresh.setObjectName('CalibrationPreviewRefresh')
        self.cancel = QPushButton(tr('Cancel'), self)
        self.cancel.setToolTip(tr('Discard this preview. Any source scan already running will finish safely.'))
        close = QPushButton(tr('Close'), self)
        for button in (self.refresh, self.cancel, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._runner = JobRunner(self, threaded=threaded, app_key='measure', user_visible=False)
        self.refresh.clicked.connect(self._start)
        self.cancel.clicked.connect(self._cancel)
        close.clicked.connect(self.reject)
        self._runner.busy_changed.connect(self._busy)
        self._busy(False)
        self._watch = QTimer(self)
        self._watch.setInterval(250)
        self._watch.timeout.connect(self._check_current)
        self._watch.start()

    def _busy(self, busy):
        self.refresh.setEnabled(not busy and not self._runner.active_jobs())
        self.cancel.setEnabled(busy)

    def _clear(self):
        self.table.setRowCount(0)
        self._reports = []

    def _check_current(self):
        self._busy(self._runner.is_busy())
        if self._snapshot is None or self._closed:
            return
        try:
            unchanged = self._settings_getter() == self._snapshot
        except (ValueError, TypeError):
            unchanged = False
        if not unchanged:
            self._generation += 1
            self._runner.cancel()
            self._snapshot = None
            self._clear()
            self.status.setText(tr('Settings changed. Preview the gains again.'))

    def _start(self):
        if self._runner.active_jobs() or self._closed:
            return
        self._generation += 1
        token = self._generation
        self._clear()
        try:
            snapshot = copy.deepcopy(self._settings_getter())
        except (ValueError, TypeError) as exc:
            self._snapshot = None
            self.status.setText(tr('Finish valid settings before previewing gains.') + ' ' + str(exc))
            return
        self._snapshot = snapshot
        self.status.setText(tr('Calculating calibration gains…'))

        def work():
            try:
                return _plan_gains(snapshot), None
            except Exception as exc:
                return None, str(exc)

        def done(result):
            self._check_current()
            if self._closed or token != self._generation:
                return
            reports, error = result
            if error:
                self.status.setText(error)
                return
            self._reports = reports
            for report in reports:
                plan = report['calibration']
                for plate, values in sorted(plan['plates'].items()):
                    for channel, gain in sorted(values['gain'].items(), key=lambda pair: int(pair[0])):
                        row = self.table.rowCount()
                        self.table.insertRow(row)
                        cells = [report['source'], plate, plan['reference_plate'], channel,
                                 f'{gain:.8g}', f"{values['reference_statistic'][channel]:.8g}",
                                 str(values['n_reference_fields'])]
                        for column, value in enumerate(cells):
                            item = QTableWidgetItem(value)
                            item.setToolTip(value)
                            self.table.setItem(row, column, item)
            self.status.setText(tr('Preview complete. No images or measurements were changed.'))

        self._runner.submit(work, done)

    def _cancel(self):
        self._generation += 1
        self._runner.cancel()
        self._snapshot = None
        self._clear()
        self.status.setText(tr('Preview cancelled.'))

    def done(self, result):
        """Detach outstanding read-only work when this dialog closes.

        :param result: Standard Qt dialog result code.
        """
        self._closed = True
        self._watch.stop()
        self._generation += 1
        self._runner.shutdown(timeout_ms=0)
        super().done(result)
