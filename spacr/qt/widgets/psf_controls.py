"""Calibrated, asynchronous PSF configuration for Make Masks enhancement."""
from __future__ import annotations

import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ...point_spread import gaussian_psf, load_psf
from ..i18n import tr
from ..job_runner import JobRunner


class _PSFControls(QWidget):
    """Capture kernel bytes off-thread and expose immutable chain settings."""

    changed = Signal()

    def __init__(self, parent=None):
        """Construct calibrated PSF controls and a debounced background kernel loader."""
        super().__init__(parent)
        self._kernel = None
        self._error = ''
        self._cancel = threading.Event()
        self._closed = False
        self._jobs = JobRunner(self, app_key='make_masks_psf', user_visible=False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._load)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.operation = QComboBox()
        self.operation.setMinimumContentsLength(26)
        self.operation.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for title, value in ((tr('Off'), 'none'), (tr('Convolve (blur)'), 'convolve'),
                             (tr('Deconvolve (Richardson–Lucy)'), 'deconvolve')):
            self.operation.addItem(title, value)
        self.operation.setToolTip(tr(
            'Apply a calibrated point spread function after background subtraction. '
            'Convolution adds blur; Richardson–Lucy deconvolution can amplify noise. '
            'A Gaussian is an approximation, not a measured microscope PSF.'))
        form.addRow(tr('Point spread function'), self.operation)
        self.source = QComboBox()
        self.source.setMinimumContentsLength(26)
        self.source.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source.addItem(tr('Gaussian approximation'), 'gaussian')
        self.source.addItem(tr('Measured kernel (TIFF/NPY)'), 'measured')
        self.source.setToolTip(tr(
            'Calculate an explicitly sampled Gaussian, or load a measured 2D kernel '
            'with odd dimensions and finite nonnegative values. The centre pixel '
            'is the origin. The loaded kernel is normalized to sum to one.'))
        form.addRow(tr('PSF source'), self.source)
        self.image_y = self._length()
        self.image_x = self._length()
        for axis, widget in ((tr('Image pixel height'), self.image_y),
                             (tr('Image pixel width'), self.image_x)):
            widget.setToolTip(tr('Enter the actual image pixel spacing in micrometers. No calibration is inferred.'))
            form.addRow(axis, widget)
        self.kernel_y = self._length()
        self.kernel_x = self._length()
        for axis, widget in ((tr('Measured PSF pixel height'), self.kernel_y),
                             (tr('Measured PSF pixel width'), self.kernel_x)):
            widget.setToolTip(tr('Enter the measured kernel pixel spacing. It must match the image; no resampling is performed.'))
            form.addRow(axis, widget)
        self.fwhm_y = self._length()
        self.fwhm_x = self._length()
        for axis, widget in ((tr('Gaussian FWHM, Y'), self.fwhm_y),
                             (tr('Gaussian FWHM, X'), self.fwhm_x)):
            widget.setToolTip(tr('Full width at half maximum in micrometers. The Gaussian is sampled at image spacing and truncated at four sigma.'))
            form.addRow(axis, widget)
        self.path = QLineEdit()
        self.path.setPlaceholderText(tr('Measured PSF file'))
        self.path.setToolTip(tr('Load a 2D TIFF or NPY kernel, up to 64 MiB. File and normalized-kernel hashes identify the captured data. Reload to read changed file contents.'))
        self.browse = QPushButton(tr('Browse…'))
        self.browse.clicked.connect(self._browse)
        self.reload = QPushButton(tr('Reload'))
        self.reload.setToolTip(tr('Read the kernel file again. A file changed on disk does not silently alter an already loaded kernel.'))
        self.reload.clicked.connect(self._invalidate)
        file_row = QHBoxLayout()
        file_row.addWidget(self.path, 1)
        file_row.addWidget(self.browse)
        file_row.addWidget(self.reload)
        form.addRow(QLabel(tr('Measured kernel')))
        form.addRow(file_row)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 200)
        self.iterations.setValue(20)
        self.iterations.setToolTip(tr('Richardson–Lucy iterations. More iterations may amplify noise. This method has no regularization.'))
        form.addRow(tr('Deconvolution iterations'), self.iterations)
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for name in ('operation', 'source', 'image_y', 'image_x', 'kernel_y',
                     'kernel_x', 'fwhm_y', 'fwhm_x', 'path', 'reload', 'iterations'):
            widget = getattr(self, name)
            widget.setObjectName('psf_' + name)
        self.operation.currentIndexChanged.connect(self._operation_changed)
        self.source.currentIndexChanged.connect(self._invalidate)
        for widget in (self.image_y, self.image_x, self.kernel_y, self.kernel_x,
                       self.fwhm_y, self.fwhm_x):
            widget.valueChanged.connect(self._invalidate)
        self.path.textChanged.connect(self._invalidate)
        self.iterations.valueChanged.connect(self.changed)
        self._sync_controls()
        from ..screens.settings_model import attach_api_tooltip, retarget_field_tooltips

        for name in ('operation', 'source', 'image_y', 'image_x', 'kernel_y',
                     'kernel_x', 'fwhm_y', 'fwhm_x', 'path', 'reload', 'iterations'):
            widget = getattr(self, name)
            attach_api_tooltip(widget, 'make_masks', 'make_masks_psf_' + name,
                               widget.toolTip(), _descriptions={})
        retarget_field_tooltips(self)

    @staticmethod
    def _length():
        """Create a micrometer-valued editor whose zero state requests explicit calibration."""
        widget = QDoubleSpinBox()
        widget.setDecimals(4)
        widget.setRange(0, 10000)
        widget.setSingleStep(.1)
        widget.setSuffix(' µm')
        widget.setSpecialValueText(tr('Set calibration'))
        return widget

    def _sync_controls(self):
        """Enable only controls needed by the selected operation and measured or Gaussian source."""
        active = self.operation.currentData() != 'none'
        measured = self.source.currentData() == 'measured'
        for widget in (self.source, self.image_y, self.image_x):
            widget.setEnabled(active)
        for widget in (self.kernel_y, self.kernel_x, self.path, self.browse, self.reload):
            widget.setEnabled(active and measured)
        for widget in (self.fwhm_y, self.fwhm_x):
            widget.setEnabled(active and not measured)
        self.iterations.setEnabled(self.operation.currentData() == 'deconvolve')

    def _operation_changed(self):
        """Cancel disabled PSF work or invalidate the kernel when the operation changes."""
        self._sync_controls()
        if self.operation.currentData() == 'none':
            self._cancel.set()
            self._timer.stop()
            self._jobs.cancel()
            self.status.clear()
        elif self._kernel is None:
            self._invalidate()
        self.changed.emit()

    def _invalidate(self, *_args):
        """Cancel stale kernel work, clear its snapshot and schedule loading for the latest calibration."""
        self._cancel.set()
        self._jobs.cancel()
        self._timer.stop()
        self._kernel = None
        self._error = tr('PSF is not ready. Enter calibration and wait for the kernel to load.')
        self._sync_controls()
        if self.operation.currentData() != 'none' and not self._closed:
            self.status.setText(self._error)
            self._timer.start()
        self.changed.emit()

    def _browse(self):
        """Choose a measured TIFF or NPY kernel and update its source path."""
        path, _ = QFileDialog.getOpenFileName(
            self, tr('Choose a measured PSF'), '', tr('PSF kernels (*.tif *.tiff *.npy)'))
        if path:
            self.path.setText(path)

    def _load(self):
        """Capture source and calibration values before submitting background kernel construction."""
        source = self.source.currentData()
        image_spacing = (self.image_y.value(), self.image_x.value())
        kernel_spacing = (self.kernel_y.value(), self.kernel_x.value())
        fwhm = (self.fwhm_y.value(), self.fwhm_x.value())
        path = self.path.text().strip()
        cancel = self._cancel = threading.Event()

        def work():
            """Validate the captured calibration and return a loaded or generated kernel with its error."""
            try:
                if not all(value > 0 for value in image_spacing):
                    raise ValueError(tr('Enter positive image pixel spacing in both axes.'))
                if source == 'measured':
                    if not path:
                        raise ValueError(tr('Choose a measured PSF file.'))
                    kernel = load_psf(path, sampling_um=kernel_spacing, cancel=cancel)
                    if len(kernel.shape) != 2:
                        raise ValueError(tr('Make Masks requires a two-dimensional PSF.'))
                    import numpy as np
                    if not np.allclose(kernel.sampling_um, image_spacing, rtol=1e-6, atol=0):
                        raise ValueError(tr('Image and PSF pixel spacing must match. No resampling is performed.'))
                else:
                    kernel = gaussian_psf(fwhm_um=fwhm, sampling_um=image_spacing)
                return kernel, ''
            except Exception as exc:
                return None, str(exc)

        self.status.setText(tr('Preparing calibrated PSF…'))
        self._jobs.submit(work, self._loaded)

    def _loaded(self, result):
        """Publish a completed kernel or its error unless the controls have already closed."""
        if self._closed:
            return
        self._kernel, self._error = result
        if self._kernel is None:
            self.status.setText(tr('PSF unavailable: {error}', error=self._error))
        else:
            self.status.setText(tr('PSF ready: {height} × {width} pixels. Source image and measurement intensities remain unchanged.',
                                   height=self._kernel.shape[0], width=self._kernel.shape[1]))
        self.changed.emit()

    def _chain_fields(self):
        """Return enhancement-chain PSF fields, or an empty mapping when processing is disabled."""
        if self.operation.currentData() == 'none':
            return {}
        return dict(psf_operation=self.operation.currentData(), psf=self._kernel,
                    psf_sampling_um=(self.image_y.value(), self.image_x.value()),
                    psf_iterations=self.iterations.value(), psf_error=self._error)

    def mask_settings(self):
        """These controls as the Mask module's ``psf_*`` settings.

        What :func:`spacr.psf_pipeline.prepare_psf` reads, so the kernel a
        curator tuned here is the kernel a plate run captures. A length left
        at its "Set calibration" zero is None, which that reader refuses
        rather than guesses at.
        """
        def pair(y, x):
            """A [Y, X] length pair, or None while either side is unset."""
            if y.value() > 0 and x.value() > 0:
                return [float(y.value()), float(x.value())]
            return None

        return {
            'psf_operation': str(self.operation.currentData()),
            'psf_source': str(self.source.currentData()),
            'psf_path': self.path.text().strip() or None,
            'psf_image_sampling_um': pair(self.image_y, self.image_x),
            'psf_kernel_sampling_um': pair(self.kernel_y, self.kernel_x),
            'psf_fwhm_um': pair(self.fwhm_y, self.fwhm_x),
            'psf_iterations': int(self.iterations.value()),
        }

    def _shutdown(self):
        """Stop pending debounce and kernel work without blocking the GUI on worker completion."""
        self._closed = True
        self._timer.stop()
        self._cancel.set()
        self._jobs.shutdown(timeout_ms=0)
