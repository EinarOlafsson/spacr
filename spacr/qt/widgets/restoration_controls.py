"""Asynchronous Cellpose 3 restoration controls for Make Masks enhancement."""
from __future__ import annotations

import threading
from dataclasses import replace

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..._segmentation_backends import _restoration_plan
from ..i18n import tr
from ..job_runner import JobRunner


class _RestorationControls(QWidget):
    """Capture model identity off-thread and invalidate obsolete loading results."""

    changed = Signal()

    def __init__(self, parent=None):
        """Build opt-in controls without loading a model or starting a worker."""
        super().__init__(parent)
        self._plan = None
        self._error = ''
        self._closed = False
        self._generation = 0
        self._cancel = threading.Event()
        self._jobs = JobRunner(self, app_key='make_masks_restoration', user_visible=False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._load)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.operation = QComboBox()
        for label, value in ((tr('Off'), 'none'), (tr('Denoise'), 'denoise'),
                             (tr('Deblur'), 'deblur'), (tr('One-click restoration'), 'oneclick')):
            self.operation.addItem(label, value)
        self.operation.setToolTip(tr(
            'Restore the selected intensity channel with Cellpose 3 in its own '
            'environment. Runs after PSF processing and before classical denoising. '
            'Compare previews the result; Apply enables it for subsequent detection. '
            'Source pixels and measurement intensities remain unchanged.'))
        form.addRow(tr('Deep image enhancement'), self.operation)
        self.structure = QComboBox()
        for label, value in ((tr('Cells (cyto3)'), 'cyto3'),
                             (tr('Cells (cyto2)'), 'cyto2'), (tr('Nuclei'), 'nuclei')):
            self.structure.addItem(label, value)
        self.structure.setToolTip(tr(
            'Choose weights trained for cells or nuclei. These models are not '
            'validated for every organelle or acquisition. Restored values are '
            'normalized model output, not calibrated fluorescence. Inspect '
            'the comparison before accepting new masks. Upsampling is excluded '
            'so image and mask coordinates stay aligned.'))
        form.addRow(tr('Restoration model'), self.structure)
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(1, 2000)
        self.diameter.setDecimals(1)
        self.diameter.setValue(30)
        self.diameter.setSuffix(' px')
        self.diameter.setToolTip(tr(
            'Approximate diameter of the selected cells or nuclei in pixels. '
            'Controls model rescaling; output dimensions remain unchanged. '
            'This does not estimate microscope calibration.'))
        form.addRow(tr('Restoration diameter'), self.diameter)
        self.reload = QPushButton(tr('Load / retry model'))
        self.reload.setToolTip(tr(
            'Load the selected Cellpose 3 restoration weights on the CPU. '
            'First use may download weights. The captured package version, '
            'checkpoint hash and diameter are recorded with applied enhancement.'))
        form.addRow(self.reload)
        self.install = QPushButton(tr('Install Cellpose 3…'))
        self.install.setToolTip(tr(
            'Open the Model Zoo installer for the isolated Cellpose 3 environment. '
            'The Cellpose version used by spaCR itself is unchanged.'))
        form.addRow(self.install)
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.operation.currentIndexChanged.connect(self._invalidate)
        self.structure.currentIndexChanged.connect(self._invalidate)
        self.diameter.valueChanged.connect(self._diameter_changed)
        self.reload.clicked.connect(self._invalidate)
        self.install.clicked.connect(self._install)
        self._sync_controls()
        from ..screens.settings_model import attach_api_tooltip, retarget_field_tooltips

        for name in ('operation', 'structure', 'diameter', 'reload', 'install'):
            widget = getattr(self, name)
            widget.setObjectName('restoration_' + name)
            attach_api_tooltip(widget, 'make_masks', 'make_masks_enh_restoration_' + name,
                               widget.toolTip(), _descriptions={})
        retarget_field_tooltips(self)

    def _sync_controls(self):
        """Keep inactive model controls visible but disabled."""
        active = self.operation.currentData() != 'none'
        for widget in (self.structure, self.diameter, self.reload):
            widget.setEnabled(active)

    def _diameter_changed(self):
        """Change the immutable request without reloading an existing model."""
        if self._plan is not None:
            self._plan = replace(self._plan, diameter=self.diameter.value())
            self.changed.emit()
        elif self.operation.currentData() != 'none':
            self._invalidate()

    def _invalidate(self, *_args):
        """Cancel old loading and debounce the current model selection."""
        self._generation += 1
        self._cancel.set()
        self._jobs.cancel()
        self._timer.stop()
        self._plan = None
        self._error = tr('Restoration model is not ready. Load the model before applying it.')
        self._sync_controls()
        if self.operation.currentData() == 'none':
            self.status.clear()
        elif not self._closed:
            self.status.setText(self._error)
            self._timer.start()
        self.changed.emit()

    def _load(self):
        """Capture widget values and load the model entirely off-thread."""
        model = self.operation.currentData() + '_' + self.structure.currentData()
        diameter = self.diameter.value()
        generation = self._generation
        cancel = self._cancel = threading.Event()

        def work():
            """Return captured model identity or the original loading error."""
            try:
                plan = _restoration_plan(model, diameter, device='cpu',
                                         should_cancel=cancel.is_set)
                return generation, plan, ''
            except Exception as exc:
                return generation, None, str(exc)

        self.status.setText(tr('Loading restoration model on CPU…'))
        self._jobs.submit(work, self._loaded)

    def _loaded(self, result):
        """Publish only a result matching the current model generation."""
        generation, plan, error = result
        if self._closed or generation != self._generation:
            return
        self._plan, self._error = plan, error
        self.status.setText(
            tr('Restoration ready: {model}. CPU processing; original intensities retained for measurements.',
               model=plan.model) if plan is not None else
            tr('Restoration unavailable: {error}', error=error))
        self.changed.emit()

    def _chain_fields(self):
        """Return the requested chain fields, including a pending-model error."""
        if self.operation.currentData() == 'none':
            return {}
        return dict(restoration=True, restoration_plan=self._plan,
                    restoration_error=self._error)

    def _install(self):
        """Delegate explicit installation to the existing Model Zoo dialog."""
        from .model_zoo_picker import install_backend

        if install_backend(self, 'cellpose3'):
            self._invalidate()

    def _shutdown(self):
        """Invalidate and cancel pending work without blocking window closure."""
        self._closed = True
        self._generation += 1
        self._timer.stop()
        self._cancel.set()
        self._jobs.shutdown(timeout_ms=0)
