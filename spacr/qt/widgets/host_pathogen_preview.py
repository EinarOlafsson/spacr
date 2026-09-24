"""Interactive per-vacuole results beside the measured field's image."""

from __future__ import annotations

from copy import deepcopy
import html
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QVBoxLayout, QWidget,
)

from ..i18n import tr
from ..job_runner import JobRunner
from .live_preview import _ZoomView, numpy_to_qpixmap
from .preview_contract import LivePreviewContract, PREVIEW_RUN_TEXT, PREVIEW_CANCEL_TEXT
from .sortable_table import install_sorting, table_item


class HostPathogenPreviewPanel(LivePreviewContract, QWidget):
    """Preview current Host–Pathogen settings on one field without saving.

    Image and table selection share vacuole labels. Unknown marker states are
    retained; displayed infection fractions use this field's measured hosts.
    """

    preview_ready = Signal(dict)

    def __init__(self, parent=None, *, threaded=True, settings_reader=None):
        """Build controls without reading images, databases or the settings form.

        :param parent: optional parent widget.
        :param threaded: run preview work on background threads when true.
        :param settings_reader: optional callable returning current form settings.
        """
        super().__init__(parent)
        self._reader = settings_reader
        self._settings = {}
        self._fields = []
        self._result = None
        self._hover = None
        self._selected = None
        self._run_token = 0
        self._started = False
        self._pending_refresh = False
        self._jobs = JobRunner(self, threaded=threaded, app_key='host_pathogen_preview')
        self._jobs.job_failed.connect(self._failed)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self._field = QComboBox(self)
        self._field.setMinimumContentsLength(12)
        self._field.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._field.setToolTip(tr('Choose a measured field. Counts and infection fractions refer only to this field.'))
        controls.addWidget(self._field, 1)
        self._run_btn = QPushButton(tr(PREVIEW_RUN_TEXT), self)
        self._run_btn.clicked.connect(self.run_preview)
        controls.addWidget(self._run_btn)
        self._cancel_btn = QPushButton(tr(PREVIEW_CANCEL_TEXT), self)
        self._cancel_btn.clicked.connect(self.cancel_preview)
        self._cancel_btn.setEnabled(False)
        controls.addWidget(self._cancel_btn)
        layout.addLayout(controls)
        options = QHBoxLayout()
        options.addWidget(QLabel(tr('Image channel'), self))
        self._channel = QSpinBox(self)
        self._channel.setRange(0, 999)
        options.addWidget(self._channel)
        self._planes, self._overlays = {}, {}
        for role, caption in [('host', 'Hosts'), ('vacuole', 'Vacuoles'), ('parasite', 'Parasites')]:
            check = QCheckBox(tr(caption), self)
            check.setChecked(True)
            check.toggled.connect(self._paint)
            options.addWidget(check)
            spin = QSpinBox(self)
            spin.setRange(-2, 999)
            spin.setValue(-2)
            spin.setSpecialValueText(tr('Auto'))
            spin.setToolTip(tr('Merged mask plane: Auto uses recorded plane metadata; -1 hides this mask. Choose a plane explicitly for older projects.'))
            options.addWidget(spin)
            self._planes[role], self._overlays[role] = spin, check
            spin.valueChanged.connect(self._display_changed)
        self._channel.valueChanged.connect(self._display_changed)
        options.addStretch()
        layout.addLayout(options)
        self._splitter = QSplitter(Qt.Vertical, self)
        self._splitter.setHandleWidth(1)
        self._view = _ZoomView(self)
        self._view.setMinimumHeight(180)
        self._view.hover_pixel.connect(self._hover_pixel)
        self._view.clicked.connect(self._image_clicked)
        self._splitter.addWidget(self._view)
        self._table = QTableWidget(self)
        install_sorting(self._table)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._table_selected)
        self._table.setMinimumHeight(100)
        self._splitter.addWidget(self._table)
        self._splitter.setSizes([420, 160])
        layout.addWidget(self._splitter, 1)
        self._details = QLabel(self)
        self._details.setWordWrap(True)
        self._details.setMinimumHeight(65)
        self._details.setTextFormat(Qt.RichText)
        self._details.setOpenExternalLinks(True)
        layout.addWidget(self._details)
        self._summary = QLabel(self)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._status = QLabel(tr('Load a measured project, then Run preview.'), self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._field.currentIndexChanged.connect(self._field_changed)
        self._settings_timer = QTimer(self)
        self._settings_timer.setInterval(400)
        self._settings_timer.timeout.connect(self._sync_settings)
        self._settings_timer.start()
        from ..screens.settings_model import attach_api_tooltip, retarget_field_tooltips

        for widget in (self._field, self._channel, self._run_btn, *self._planes.values()):
            attach_api_tooltip(widget, 'host_pathogen', '', widget.toolTip())
        retarget_field_tooltips(self)

    def apply_settings(self, settings):
        """Replace preview settings; stale results are discarded on a change.

        :param settings: Host–Pathogen settings to copy into the preview.
        """
        values = deepcopy(settings)
        if values != self._settings:
            self.cancel_preview()
            self._clear()
            if values.get('src') != self._settings.get('src'):
                self._fields = []
        self._settings = values

    def _sync_settings(self):
        """Refresh changed analysis settings only after this visible preview has been started."""
        if not self._started or not self.isVisible():
            return
        if self._reader:
            settings = deepcopy(self._reader())
            if settings != self._settings:
                self.apply_settings(settings)
                self._pending_refresh = True
        if self._pending_refresh and self._jobs.active_jobs() == 0:
            self._pending_refresh = False
            self.run_preview()

    def load_source_async(self, source):
        """Refresh the field list and preview from the current form and source.

        :param source: project or database source used for the refreshed preview.
        :returns: whether a preview was started or queued.
        """
        settings = deepcopy(self._reader() if self._reader else self._settings)
        settings['src'] = source
        self.apply_settings(settings)
        self._fields = []
        return self.run_preview()

    def preview_running(self):
        """Whether the preview runner still owns work whose result is current."""
        return self._jobs.is_busy()

    def _extra_work_in_flight(self):
        """Report whether the background field reader still has a job in flight."""
        return self._jobs.is_busy()

    def _cancel_extra_work(self):
        """Discard a queued refresh and invalidate the current background job."""
        self._pending_refresh = False
        self._jobs.cancel()

    def run_preview(self):
        """Read a bounded field and calculate the same ratios/counts as Run."""
        from ...host_pathogen_preview import preview_fields, preview_field

        settings = deepcopy(self._reader() if self._reader else self._settings)
        if settings != self._settings:
            self.apply_settings(settings)
        if not settings.get('src') or settings['src'] == 'path':
            self.set_preview_status(tr('Choose a measured project in the source setting first.'))
            return False
        if not self.begin_preview():
            return False
        if self._jobs.active_jobs():
            self._pending_refresh = True
            self.set_preview_status(tr('Waiting for the previous read to finish…'))
            return True
        self._started = True
        self._run_token += 1
        token = self.preview_token()
        chosen = self._field.currentData() if self._fields else None
        planes = {role: spin.value() for role, spin in self._planes.items() if spin.value() != -2}
        channel = self._channel.value()
        self._settings = settings
        self.set_preview_status(tr('Reading field measurements and image…'))

        def work():
            """Read available fields and analyze the captured field and plane selection off the GUI thread."""
            try:
                fields, limited = preview_fields(settings)
                if not fields:
                    raise ValueError('No measured host fields were found')
                selected = chosen if chosen in fields else fields[0]
                return fields, limited, preview_field(settings, selected, planes=planes, image_channel=channel)
            except Exception as exc:
                return {'error': str(exc)}

        self._jobs.submit(work, lambda result: self._received(token, result))
        return True

    def _received(self, token, payload):
        """Ignore stale results, then populate field choices, vacuole rows and image overlays."""
        if self.preview_stale(token):
            return
        if isinstance(payload, dict) and 'error' in payload:
            self._failed(payload['error'])
            return
        self._fields, limited, self._result = payload
        self._field.blockSignals(True)
        self._field.clear()
        for field in self._fields:
            database = Path(field['database'])
            project = database.parent.parent.name if database.parent.name == 'measurements' else database.parent.name
            self._field.addItem(project + ': ' + ' / '.join(str(value) for value in field['identity'].values()), field)
            self._field.setItemData(self._field.count() - 1, str(database), Qt.ToolTipRole)
        self._field.setCurrentIndex(self._fields.index(self._result['field']))
        self._field.blockSignals(False)
        vacuoles = self._result['results']['vacuoles']
        channels = self._settings.get('hp_marker_channels', [0])
        columns = [('vacuole_id', tr('Vacuole')), ('cell_id', tr('Host')),
                   ('parasite_count', tr('Parasites'))]
        for channel in channels:
            columns.extend([(f'channel_{channel}_recruitment_ratio', tr('Channel {n} ratio', n=channel)),
                            (f'channel_{channel}_state', tr('Channel {n} state', n=channel))])
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels([caption for _, caption in columns])
        self._table.setRowCount(min(500, len(vacuoles)))
        for row, (_, values) in enumerate(vacuoles.iloc[:500].iterrows()):
            for column, (key, _) in enumerate(columns):
                item = table_item(_text(values[key]))
                item.setData(Qt.UserRole, int(values['vacuole_id']))
                self._table.setItem(row, column, item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)
        self._table.blockSignals(False)
        cells = self._result['results']['cells']
        infected = int(cells['infected'].sum())
        self._summary.setText(tr(
            'This field: {hosts} hosts, {infected} infected, {vacuoles} vacuoles, {orphans} orphan parasites. '
            'The host denominator includes measured uninfected cells; this is not a whole-plate rate.',
            hosts=len(cells), infected=infected, vacuoles=len(vacuoles),
            orphans=len(self._result['results']['orphan_parasites'])))
        note = self._result['image_note']
        if limited:
            note += ' ' + tr('Showing the first 50 measured fields; choose a narrower source for other fields.')
        if len(vacuoles) > 500:
            note += ' ' + tr('The table shows the first 500 vacuoles; field totals include every measured vacuole.')
        self.set_preview_status(note)
        self._selected = None
        self._paint(reset=True)
        if len(vacuoles):
            self._table.selectRow(0)
        else:
            self._details.setText(tr('No vacuoles measured in this field.'))
        self.set_preview_busy(False)
        self.preview_ready.emit(self._result)

    def _failed(self, message):
        """Clear outdated results and present the worker error in the preview status."""
        self._clear()
        self.set_preview_busy(False)
        self.set_preview_status(tr('Preview failed: {error}', error=message))

    def _clear(self):
        """Discard results, table selection and image overlays without changing the analysis settings."""
        self._result = None
        self._selected = None
        self._table.setRowCount(0)
        self._view.scene().clear()
        self._view._pixmap_item = None
        self._details.clear()
        self._summary.clear()

    def _field_changed(self, index):
        """Invalidate the current result and preview a newly selected measured field."""
        if self._fields and index >= 0:
            self.cancel_preview()
            self._clear()
            self.run_preview()

    def _display_changed(self, *args):
        """Invalidate the preview after display-plane changes and request an explicit rerun."""
        self.cancel_preview()
        self._clear()
        self.set_preview_status(tr('Display plane changed; Run preview to read it.'))

    def _hover_pixel(self, x, y):
        """Remember the last hovered image pixel for subsequent vacuole selection."""
        self._hover = (x, y)

    def _image_clicked(self):
        """Select the table row matching the vacuole label under the pointer."""
        if self._result is None or self._hover is None:
            return
        mask = self._result['masks'].get('vacuole')
        x, y = self._hover
        if mask is None or not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            return
        label = int(mask[y, x])
        for row in range(self._table.rowCount()):
            if self._table.item(row, 0).data(Qt.UserRole) == label:
                self._table.selectRow(row)
                self._table.scrollToItem(self._table.item(row, 0))
                return

    def _table_selected(self):
        """Synchronize the selected vacuole, linked measurement explanation and image outline."""
        from ..screens.settings_model import api_docs_url

        items = self._table.selectedItems()
        if not items or self._result is None:
            return
        self._selected = items[0].data(Qt.UserRole)
        rows = self._result['results']['vacuoles']
        row = rows.loc[rows.vacuole_id == self._selected].iloc[0]
        details = tr('Vacuole {vacuole}; host {host}; parasites {count}.',
                     vacuole=self._selected, host=_text(row.cell_id), count=_text(row.parasite_count))
        for channel in self._settings.get('hp_marker_channels', [0]):
            details += ' ' + tr('Channel {channel}: ratio {ratio}, {state}.', channel=channel,
                ratio=_text(row[f'channel_{channel}_recruitment_ratio']),
                state=_text(row[f'channel_{channel}_state']))
        details += ' ' + tr('Unknown measurements are not negative calls.')
        self._details.setText(html.escape(details) + f' <a href="{html.escape(api_docs_url("host_pathogen"))}">API</a>')
        self._paint()

    def _paint(self, *args, reset=False):
        """Render enabled host, vacuole and parasite contours with a selected-vacuole highlight."""
        if self._result is None or self._result['image'] is None:
            return
        rgb = np.repeat(self._result['image'][..., None], 3, axis=-1)
        for role, color in [('host', (60, 170, 255)), ('vacuole', (255, 210, 70)), ('parasite', (255, 80, 180))]:
            mask = self._result['masks'].get(role)
            if mask is None or not self._overlays[role].isChecked():
                continue
            edge = np.zeros(mask.shape, dtype=bool)
            edge[1:] |= mask[1:] != mask[:-1]
            edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
            rgb[edge & (mask > 0)] = color
        vacuole = self._result['masks'].get('vacuole')
        if vacuole is not None and self._selected is not None:
            chosen = vacuole == self._selected
            rgb[chosen] = np.uint8(.6 * rgb[chosen] + .4 * np.array([40, 160, 255]))
        pixmap = numpy_to_qpixmap(rgb, normalise=False)
        if reset or self._view._pixmap_item is None:
            self._view.set_pixmap(pixmap)
        else:
            self._view._pixmap_item.setPixmap(pixmap)

    def shutdown(self):
        """Retire background reads before the preview's widgets disappear."""
        self._settings_timer.stop()
        self._jobs.shutdown()

    def closeEvent(self, event):
        """Cancel pending work when the panel closes.

        :param event: Qt close event forwarded to the parent implementation.
        """
        self.shutdown()
        super().closeEvent(event)


def _text(value):
    """Format a measured value without converting unknowns to zero."""
    import pandas as pd

    if pd.isna(value):
        return tr('unknown')
    if isinstance(value, (float, np.floating)):
        return f'{value:.3g}'
    return tr(str(value))


def build_host_pathogen_preview_card(host, *, panel_later=False):
    """Declare a lazily built preview using the shared Live toggle/card.

    :param host: application screen providing the current settings model.
    :param panel_later: defer panel creation when true.
    :returns: optional preview panel and its containing card.
    """
    from .card import Card

    card = Card(title=tr('Host–Pathogen live preview'))
    card.setMinimumHeight(380)
    return (None if panel_later else fill_host_pathogen_preview_card(host, card)), card


def fill_host_pathogen_preview_card(host, card):
    """Connect the preview to the form so Run preview uses current settings.

    :param host: application screen providing the current settings model.
    :param card: card whose body receives the new preview panel.
    :returns: the attached preview panel.
    """
    panel = HostPathogenPreviewPanel(card, settings_reader=host._settings_model.collect)
    card.body_layout.addWidget(panel)
    return panel
