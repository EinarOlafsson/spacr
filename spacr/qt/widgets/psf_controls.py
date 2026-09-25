"""Calibrated, asynchronous PSF configuration for Make Masks enhancement.

Item 509: the calibration fills itself. A microscope objective, camera and
fluorophore chooser, or "Infer from images…", fills magnification, numerical
aperture, refractive index, emission wavelength, camera pixel, image pixel
size and Gaussian FWHM, each field saying where its value came from. Only the
operation and the objective row are shown; the rest sits in a fold that starts
collapsed and remembers being opened.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from ...point_spread import (
    CAMERAS, DEFAULT_CAMERA_PIXEL_UM, DEFAULT_EMISSION_NM, DEFAULT_OBJECTIVE,
    FLUOROPHORES, IMMERSION_INDEX, OBJECTIVES, gaussian_psf, infer_optics,
    lateral_fwhm_um, load_psf, objective, pixel_size_um,
)
from ..i18n import tr
from ..job_runner import JobRunner

FOLD_KEY = 'make_masks/psf_details'
_OPTICS = ('magnification', 'numerical_aperture', 'refractive_index',
           'emission_nm', 'camera_pixel_um')
_MEASURED_PIXEL = ('metadata', 'imagej', 'tiff_resolution', 'entered', 'chosen')


def _source_text(source, detail=''):
    """Translated "where this value came from" for one :class:`OpticalValue` source."""
    texts = {
        'metadata': tr('from OME metadata ({detail})', detail=detail),
        'imagej': tr('from ImageJ calibration ({detail})', detail=detail),
        'tiff_resolution': tr('from TIFF resolution tags ({detail})', detail=detail),
        'file_name': tr('from the file name ({detail})', detail=detail),
        'image': tr('from the image ({detail})', detail=detail),
        'objective': tr('from objective {detail}', detail=detail),
        'fluorophore': tr('from {detail}', detail=detail),
        'camera': tr('from {detail}', detail=detail),
        'default': tr('default ({detail})', detail=detail) if detail else tr('default'),
        'calculated': tr('calculated: {detail}', detail=detail),
        'chosen': tr('chosen'),
        'entered': tr('entered by you'),
    }
    return texts.get(source, source)


class _PSFControls(QWidget):
    """Capture kernel bytes off-thread and expose immutable chain settings."""

    changed = Signal()

    def __init__(self, parent=None, image_paths=None):
        """Construct PSF controls with common-value defaults and a debounced background kernel loader.

        :param image_paths: optional callable returning the image files
            "Infer from images…" reads first; with none, it asks for a file.
        """
        super().__init__(parent)
        self._kernel = None
        self._error = ''
        self._cancel = threading.Event()
        self._closed = False
        self._filling = False
        self._origins = {}
        self.image_paths = image_paths
        self._jobs = JobRunner(self, app_key='make_masks_psf', user_visible=False)
        self._infer_jobs = JobRunner(self, app_key='make_masks_psf_infer', user_visible=False)
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
        self.objective = QComboBox()
        for row in OBJECTIVES:
            self.objective.addItem(tr('{magnification}× / NA {aperture} {immersion}',
                                      magnification=f'{row.magnification:g}',
                                      aperture=f'{row.numerical_aperture:.2f}',
                                      immersion=tr(row.immersion)), row.name)
        self.objective.setToolTip(tr(
            'Choose the microscope objective the images were taken with. Its '
            'magnification, numerical aperture and immersion fill the optics '
            'below, which then give the image pixel size (camera pixel divided '
            'by magnification) and the Gaussian width (0.51 × emission '
            'wavelength / NA). Default 20× / NA 0.75 air.'))
        self.infer = QPushButton(tr('Infer from images…'))
        self.infer.setToolTip(tr(
            'Read the current image file for OME or ImageJ calibration: pixel '
            'size, objective magnification, numerical aperture, immersion and '
            'emission wavelength. Whatever the file does not state comes from '
            'the objective chosen here and common defaults. Every value stays '
            'editable and says where it came from.'))
        self.infer.clicked.connect(self._infer_clicked)
        chooser = QHBoxLayout()
        chooser.addWidget(self.objective, 1)
        chooser.addWidget(self.infer)
        form.addRow(tr('Objective'), chooser)
        layout.addLayout(form)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName('Muted')
        layout.addWidget(self.summary)
        details = QWidget()
        more = QFormLayout(details)
        more.setContentsMargins(0, 0, 0, 0)
        more.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.camera = QComboBox()
        for title, pitch in CAMERAS:
            self.camera.addItem(title, pitch)
        self.camera.setToolTip(tr(
            'Choose the camera, to fill its physical pixel pitch. Common '
            'sCMOS cameras have 6.5 µm pixels; the image pixel size is this '
            'pitch divided by the magnification. Default 6.5 µm.'))
        more.addRow(tr('Camera'), self.camera)
        self.fluorophore = QComboBox()
        for title, emission in FLUOROPHORES:
            self.fluorophore.addItem(title, emission)
        self.fluorophore.setCurrentIndex(self.fluorophore.findData(DEFAULT_EMISSION_NM))
        self.fluorophore.setToolTip(tr(
            'Choose the fluorophore, to fill a typical emission wavelength: '
            'DAPI 461 nm, GFP 520 nm, Cy3 600 nm, Cy5 670 nm. A longer '
            'wavelength gives a wider PSF. Default GFP, 520 nm.'))
        more.addRow(tr('Fluorophore'), self.fluorophore)
        self.magnification = self._number(1, 250, 1, 1, '×')
        self.numerical_aperture = self._number(0.01, 1.7, 2, 0.05, '')
        self.refractive_index = self._number(1.0, 2.0, 3, 0.01, '')
        self.emission_nm = self._number(300, 1000, 0, 10, ' nm')
        self.camera_pixel_um = self._number(0.5, 30, 2, 0.1, ' µm')
        self._source_labels = {}
        for name, title, tip in (
                ('magnification', tr('Magnification'), tr(
                    'Total magnification between specimen and camera, including any '
                    'camera adapter. Changing it recalculates the image pixel size '
                    'unless the pixel size was read from the file or entered.')),
                ('numerical_aperture', tr('Numerical aperture'), tr(
                    'The objective NA, engraved on its barrel. Changing it '
                    'recalculates the Gaussian FWHM as 0.51 × emission wavelength / NA.')),
                ('refractive_index', tr('Immersion refractive index'), tr(
                    'Refractive index of the immersion medium: air 1.0, water 1.33, '
                    'oil 1.515. The NA cannot exceed it.')),
                ('emission_nm', tr('Emission wavelength'), tr(
                    'Emission wavelength in nanometers. Changing it recalculates '
                    'the Gaussian FWHM as 0.51 × wavelength / NA.')),
                ('camera_pixel_um', tr('Camera pixel'), tr(
                    'Physical camera pixel pitch in micrometers. Divided by the '
                    'magnification it gives the image pixel size.'))):
            widget = getattr(self, name)
            widget.setToolTip(tip)
            more.addRow(title, self._with_source(name, widget))
        self.dimensions = QLabel(tr('Not read yet; use Infer from images…'))
        self.dimensions.setWordWrap(True)
        more.addRow(tr('Image dimensions'), self.dimensions)
        self.source = QComboBox()
        self.source.setMinimumContentsLength(26)
        self.source.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source.addItem(tr('Gaussian approximation'), 'gaussian')
        self.source.addItem(tr('Measured kernel (TIFF/NPY)'), 'measured')
        self.source.setToolTip(tr(
            'Calculate an explicitly sampled Gaussian, or load a measured 2D kernel '
            'with odd dimensions and finite nonnegative values. The centre pixel '
            'is the origin. The loaded kernel is normalized to sum to one.'))
        more.addRow(tr('PSF source'), self.source)
        self.image_y = self._length()
        self.image_x = self._length()
        for axis, name, widget in ((tr('Image pixel height'), 'image_y', self.image_y),
                                   (tr('Image pixel width'), 'image_x', self.image_x)):
            widget.setToolTip(tr('Image pixel spacing in micrometers. Filled from the file\'s calibration when it has one, otherwise camera pixel / magnification; edit it to override.'))
            more.addRow(axis, self._with_source(name, widget))
        self.kernel_y = self._length()
        self.kernel_x = self._length()
        for axis, widget in ((tr('Measured PSF pixel height'), self.kernel_y),
                             (tr('Measured PSF pixel width'), self.kernel_x)):
            widget.setToolTip(tr('Enter the measured kernel pixel spacing. It must match the image; no resampling is performed.'))
            more.addRow(axis, widget)
        self.fwhm_y = self._length()
        self.fwhm_x = self._length()
        for axis, name, widget in ((tr('Gaussian FWHM, Y'), 'fwhm_y', self.fwhm_y),
                                   (tr('Gaussian FWHM, X'), 'fwhm_x', self.fwhm_x)):
            widget.setToolTip(tr('Full width at half maximum in micrometers. The Gaussian is sampled at image spacing and truncated at four sigma.'))
            more.addRow(axis, self._with_source(name, widget))
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
        more.addRow(QLabel(tr('Measured kernel')))
        more.addRow(file_row)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 200)
        self.iterations.setValue(20)
        self.iterations.setToolTip(tr('Richardson–Lucy iterations. More iterations may amplify noise. This method has no regularization.'))
        more.addRow(tr('Deconvolution iterations'), self.iterations)
        from .collapsible_splitter import FoldSection

        self.details = FoldSection(details, 'PSF optics and kernel', persist_key=FOLD_KEY,
                                   follow_body=False, stretch=0, folded=True)
        self.details.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(self.details)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for name in ('operation', 'source', 'image_y', 'image_x', 'kernel_y',
                     'kernel_x', 'fwhm_y', 'fwhm_x', 'path', 'reload', 'iterations'):
            getattr(self, name).setObjectName('psf_' + name)
        for name in ('objective', 'infer', 'camera', 'fluorophore') + _OPTICS:
            getattr(self, name).setObjectName('psf_optics_' + name)
        self._apply_objective(DEFAULT_OBJECTIVE, 'default')
        self.operation.currentIndexChanged.connect(self._operation_changed)
        self.source.currentIndexChanged.connect(self._invalidate)
        self.objective.currentIndexChanged.connect(
            lambda _index: self._apply_objective(self.objective.currentData(), 'objective'))
        self.camera.currentIndexChanged.connect(self._camera_chosen)
        self.fluorophore.currentIndexChanged.connect(self._fluorophore_chosen)
        for name in _OPTICS:
            getattr(self, name).valueChanged.connect(
                lambda _value, name=name: self._optics_edited(name))
        for name in ('image_y', 'image_x', 'fwhm_y', 'fwhm_x'):
            getattr(self, name).valueChanged.connect(
                lambda _value, name=name: self._calibration_edited(name))
        for widget in (self.kernel_y, self.kernel_x):
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
        for name in ('objective', 'infer', 'camera', 'fluorophore') + _OPTICS:
            widget = getattr(self, name)
            attach_api_tooltip(widget, 'make_masks', 'make_masks_psf_optics_' + name,
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

    @staticmethod
    def _number(low, high, decimals, step, suffix):
        """Create one optical-quantity editor."""
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setSuffix(suffix)
        return widget

    def _with_source(self, name, widget):
        """Lay ``widget`` out beside a muted label naming where its value came from."""
        label = QLabel()
        label.setObjectName('Muted')
        label.setWordWrap(True)
        self._source_labels[name] = label
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(label, 1)
        return row

    def source_of(self, name):
        """Where one field's value came from, as the (source, detail) pair shown beside it."""
        return self._origins.get(name, ('', ''))

    def _set(self, name, value, source, detail=''):
        """Fill one field programmatically and record its source."""
        self._filling = True
        try:
            getattr(self, name).setValue(float(value))
        finally:
            self._filling = False
        self._origins[name] = (source, detail)
        label = self._source_labels.get(name)
        if label is not None:
            label.setText(_source_text(source, detail))

    def _apply_objective(self, name, source):
        """Fill magnification, NA and immersion from one objective table row."""
        row = objective(name)
        if self.objective.currentData() != row.name:
            self.objective.blockSignals(True)
            self.objective.setCurrentIndex(self.objective.findData(row.name))
            self.objective.blockSignals(False)
        detail = row.name
        self._set('magnification', row.magnification, source, detail)
        self._set('numerical_aperture', row.numerical_aperture, source, detail)
        self._set('refractive_index', IMMERSION_INDEX[row.immersion], source, detail)
        if 'emission_nm' not in self._origins:
            self._set('emission_nm', DEFAULT_EMISSION_NM, 'default', 'GFP')
        if 'camera_pixel_um' not in self._origins:
            self._set('camera_pixel_um', DEFAULT_CAMERA_PIXEL_UM, 'default', CAMERAS[0][0])
        self._recalculate()

    def _camera_chosen(self, _index):
        """Fill the camera pixel pitch from the chosen camera."""
        self._set('camera_pixel_um', self.camera.currentData(), 'camera', self.camera.currentText())
        self._recalculate()

    def _fluorophore_chosen(self, _index):
        """Fill the emission wavelength from the chosen fluorophore."""
        self._set('emission_nm', self.fluorophore.currentData(), 'fluorophore',
                  self.fluorophore.currentText())
        self._recalculate()

    def _optics_edited(self, name):
        """Mark a hand-edited optical value and recalculate what depends on it."""
        if self._filling:
            return
        self._origins[name] = ('entered', '')
        self._source_labels[name].setText(_source_text('entered'))
        self._recalculate()

    def _calibration_edited(self, name):
        """Mark a hand-edited pixel size or FWHM, then reload the kernel."""
        if not self._filling:
            self._origins[name] = ('entered', '')
            self._source_labels[name].setText(_source_text('entered'))
        self._invalidate()

    def _recalculate(self):
        """Derive image pixel size and Gaussian FWHM from the optics, keeping stated or entered values."""
        if not any(self.source_of(name)[0] in _MEASURED_PIXEL for name in ('image_y', 'image_x')):
            size = pixel_size_um(self.camera_pixel_um.value(), self.magnification.value())
            for name in ('image_y', 'image_x'):
                self._set(name, round(size, 4), 'calculated', tr('camera pixel / magnification'))
        if not any(self.source_of(name)[0] == 'entered' for name in ('fwhm_y', 'fwhm_x')):
            try:
                fwhm = lateral_fwhm_um(self.emission_nm.value(), self.numerical_aperture.value(),
                                       self.refractive_index.value())
            except ValueError as exc:
                self.summary.setText(str(exc))
                return
            for name in ('fwhm_y', 'fwhm_x'):
                self._set(name, round(fwhm, 4), 'calculated', tr('0.51 × emission / NA'))
        self._summarize()
        self._invalidate()

    def _summarize(self):
        """One visible line naming the calibration in use and its sources."""
        self.summary.setText(tr(
            'Pixel {pixel} µm ({pixel_source}); FWHM {fwhm} µm ({fwhm_source}); objective {objective}.',
            pixel=f'{self.image_x.value():.4g}',
            pixel_source=_source_text(*self.source_of('image_x')),
            fwhm=f'{self.fwhm_x.value():.4g}',
            fwhm_source=_source_text(*self.source_of('fwhm_x')),
            objective=self.objective.currentText()))

    def _infer_clicked(self):
        """Infer optics from the current image, or from a file the user picks."""
        paths = []
        if callable(self.image_paths):
            try:
                paths = [p for p in (self.image_paths() or []) if p]
            except Exception:
                paths = []
        if not paths:
            path, _ = QFileDialog.getOpenFileName(
                self, tr('Choose an image to infer optics from'), '',
                tr('TIFF images (*.tif *.tiff)'))
            if not path:
                return
            paths = [path]
        self.infer_from(paths)

    def infer_from(self, paths):
        """Read ``paths``' metadata off the GUI thread and fill every field with its source."""
        chosen = self.objective.currentData()
        camera = self.camera_pixel_um.value()
        emission = self.emission_nm.value()
        camera_source = self.source_of('camera_pixel_um')[0]
        emission_source = self.source_of('emission_nm')[0]

        def work():
            """Return inferred optics, or the error that prevented reading them."""
            try:
                return infer_optics(
                    paths, objective_name=None,
                    camera_pixel_um=None if camera_source == 'default' else camera,
                    emission_nm=None if emission_source == 'default' else emission), chosen, ''
            except Exception as exc:
                return None, chosen, str(exc)

        self.summary.setText(tr('Reading image metadata…'))
        self._infer_jobs.submit(work, self._inferred)

    def _inferred(self, result):
        """Publish inferred optics into the editable fields, each with its source."""
        values, chosen, error = result
        if self._closed:
            return
        if values is None:
            self.summary.setText(tr('Could not read the image: {error}', error=error))
            return
        if values['objective'].source == 'default':
            row = objective(chosen)
            source = 'objective'
        else:
            row = objective(values['objective'].value)
            source = values['objective'].source
        self.objective.blockSignals(True)
        self.objective.setCurrentIndex(self.objective.findData(row.name))
        self.objective.blockSignals(False)
        for name in ('magnification', 'numerical_aperture', 'refractive_index'):
            item = values[name]
            if item.source in ('default', 'objective'):
                table = {'magnification': row.magnification,
                         'numerical_aperture': row.numerical_aperture,
                         'refractive_index': IMMERSION_INDEX[row.immersion]}[name]
                self._set(name, table, source, row.name)
            else:
                self._set(name, item.value, item.source, item.detail)
        for name in ('emission_nm', 'camera_pixel_um'):
            item = values[name]
            if item.source != 'chosen':
                self._set(name, item.value, item.source, item.detail)
        pixel = values['pixel_size_um']
        if pixel.source != 'calculated':
            self._set('image_y', pixel.value[0], pixel.source, pixel.detail)
            self._set('image_x', pixel.value[1], pixel.source, pixel.detail)
        else:
            self._origins.pop('image_y', None)
            self._origins.pop('image_x', None)
        self._origins.pop('fwhm_y', None)
        self._origins.pop('fwhm_x', None)
        shape = values.get('image_shape')
        if shape is not None:
            self.dimensions.setText(tr('{height} × {width} pixels ({source})',
                                       height=shape.value[0], width=shape.value[1],
                                       source=_source_text(shape.source, shape.detail)))
        else:
            self.dimensions.setText(tr('Not stated by the file'))
        self._recalculate()

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
        self.summary.setVisible(active)

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
        at its "Set calibration" zero is None, which the Mask run fills from
        the image metadata and ``psf_objective``
        (:func:`spacr.point_spread.fill_psf_settings`).
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
            'psf_objective': str(self.objective.currentData()),
        }

    def _shutdown(self):
        """Stop pending debounce and kernel work without blocking the GUI on worker completion."""
        self._closed = True
        self._timer.stop()
        self._cancel.set()
        self._jobs.shutdown(timeout_ms=0)
        self._infer_jobs.shutdown(timeout_ms=0)
