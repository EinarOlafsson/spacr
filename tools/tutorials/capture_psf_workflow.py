"""Private lesson 86: exercise real PSF controls without changing source images.

Capture callbacks receive stable scene names. This is authoring input, not a
published or narrated lesson, and it makes no optical-recovery claim.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path


def _sha(path):
    """Bind an input or screenshot to its exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source(source, stage, *, image_plane=1, mask_plane=4, side=256):
    """Copy a central region of a genuine merged example, retaining exact pixels."""
    import numpy as np
    import tifffile

    source, stage = Path(source), Path(stage)
    array = np.load(source, mmap_mode='r', allow_pickle=False)
    if array.ndim != 3 or array.shape[-1] != 7 or array.dtype != np.uint16:
        raise ValueError('Expected the genuine seven-plane uint16 Measure example')
    if side < 16 or not 0 <= image_plane < 4 or not 4 <= mask_plane < 7:
        raise ValueError('Invalid tutorial crop or explicit intensity/mask plane')
    y = max(0, (array.shape[0] - side) // 2)
    x = max(0, (array.shape[1] - side) // 2)
    crop = array[y:y + side, x:x + side]
    folder = stage / 'psf_source'
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'masks').mkdir()
    image = folder / 'actual_example_crop.tif'
    mask = folder / 'masks' / image.name
    tifffile.imwrite(image, crop[..., image_plane], metadata=None)
    tifffile.imwrite(mask, crop[..., mask_plane], metadata=None)
    if not np.array_equal(tifffile.imread(image), crop[..., image_plane]):
        raise RuntimeError('Image extraction changed source pixels')
    if not np.array_equal(tifffile.imread(mask), crop[..., mask_plane]):
        raise RuntimeError('Label extraction changed source labels')
    return folder, dict(source=str(source), source_sha256=_sha(source),
                        image=str(image), image_sha256=_sha(image),
                        mask=str(mask), mask_sha256=_sha(mask),
                        crop_yxhw=[y, x, *crop.shape[:2]], image_plane=image_plane,
                        mask_plane=mask_plane, calibration='none copied; chosen objective is illustrative')


def _wait(predicate, settle, timeout, description):
    """Pump the GUI until observable completion, with a bounded deadline."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        settle(.02)


def _actions(screen, settle):
    """Return actual keyboard and mouse operations with visibility checks."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit, QScrollArea

    def expose(widget):
        """Scroll a usable control into the recording surface."""
        for scroll in screen.findChildren(QScrollArea):
            if scroll.isAncestorOf(widget):
                scroll.ensureWidgetVisible(widget)
                # Keep field captions in view when a wide row was focused.
                scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().minimum())
        settle(.02)
        if not widget.isVisible() or not widget.isEnabled():
            ancestry = []
            parent = widget
            while parent is not None:
                ancestry.append((type(parent).__name__, parent.objectName(), parent.isHidden()))
                parent = parent.parentWidget()
            raise RuntimeError(f'Tutorial control unavailable: {widget.objectName()} '
                               f'visible={widget.isVisible()} enabled={widget.isEnabled()}; {ancestry}')
        widget.setFocus()

    def choose(widget, value):
        """Select a combo choice through ordinary keyboard events."""
        expose(widget)
        index = widget.findData(value)
        if index < 0:
            index = widget.findText(value)
        if index < 0:
            raise RuntimeError(f'Missing tutorial choice: {value}')
        QTest.keyClick(widget, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(widget, Qt.Key_Down)
        QTest.keyClick(widget, Qt.Key_Tab)
        if widget.currentIndex() != index:
            raise RuntimeError(f'Tutorial choice failed: {value}')

    def click(widget):
        """Click the visible control and allow queued Qt events."""
        expose(widget)
        QTest.mouseClick(widget, Qt.LeftButton)
        settle(.02)

    def text(widget, value):
        """Replace an editor value using its normal keyboard interaction."""
        expose(widget)
        entry = widget.findChild(QLineEdit, 'SettingChipEntry')
        if entry is not None:
            import ast
            values = ast.literal_eval(value)
            if widget.get_value() not in (None, []):
                raise RuntimeError('Start the tutorial from unset calibration list values')
            expose(entry)
            QTest.mouseClick(entry, Qt.LeftButton)
            for item in values:
                QTest.keyClicks(entry, str(item))
                QTest.keyClick(entry, Qt.Key_Return)
            if widget.get_value() != values:
                raise RuntimeError('The calibration chips did not retain the entered values')
            QTest.keyClick(entry, Qt.Key_Tab)
            return
        QTest.mouseClick(widget, Qt.LeftButton)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, value)
        if widget.text() != value:
            raise RuntimeError(f'Editor text was not entered: {type(widget).__name__}: {widget.text()!r}')
        QTest.keyClick(widget, Qt.Key_Tab)

    return choose, click, text


def record_editor(screen, capture, settle, timeout=30):
    """Capture Objective/Infer/Compare/Apply on an already loaded real image."""
    import numpy as np

    from spacr.point_spread import apply_psf
    from spacr.qt import preferences

    if preferences._get_show_alpha_features():
        raise RuntimeError('This lesson must demonstrate the stable controls with alpha off')
    if screen._canvas.image is None or not screen._current_image_paths():
        raise RuntimeError('Open the prepared actual image before recording PSF controls')
    original = screen._canvas.image.copy()
    labels = screen._canvas.mask.copy()
    paths = screen._current_image_paths()
    before = {str(path): _sha(path) for path in paths}
    choose, click, _ = _actions(screen, settle)
    if not screen._btn_settings.isChecked():
        click(screen._btn_settings)
    for title, section in screen._settings_categories:
        if title == 'Image enhancement' and not section.is_expanded():
            click(section.header())
    controls = screen._psf_controls
    choose(controls.objective, '40x/0.95 air')
    click(controls.infer)
    _wait(lambda: not controls._infer_jobs.is_busy() and bool(controls.dimensions.text()),
          settle, timeout, 'Image optics inference did not finish')
    if controls.details.shut:
        click(controls.details.folder.heading)
    capture('01_objective_and_inferred_sources')
    from PySide6.QtWidgets import QScrollArea

    for scroll in screen.findChildren(QScrollArea):
        if scroll.isAncestorOf(controls.fwhm_y):
            scroll.ensureWidgetVisible(controls.fwhm_y)
            scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().minimum())
    settle(.1)
    capture('01b_explicit_optics_source_labels')
    sources = {name: list(controls.source_of(name))
               for name in ('magnification', 'image_y', 'image_x', 'fwhm_y', 'fwhm_x')}
    choose(controls.operation, 'convolve')
    _wait(lambda: not controls._timer.isActive() and not controls._jobs.is_busy()
          and controls._kernel is not None, settle, timeout, 'PSF kernel did not become ready')
    expected = apply_psf(original, controls._kernel, operation='convolve',
                         image_sampling_um=(controls.image_y.value(), controls.image_x.value())).image
    click(screen._btn_compare)
    _wait(lambda: screen._comparison_request is None and screen._compare_dialog is not None,
          settle, timeout, 'PSF comparison did not finish')
    if screen._compare_dialog.views[1]._item is None:
        raise RuntimeError('Comparison has no computed PSF picture')
    capture('02_compare_raw_and_convolved')
    screen._compare_dialog.close()
    click(screen._btn_apply)
    _wait(lambda: screen._canvas._enhanced_picture is not None,
          settle, timeout, 'PSF display did not finish')
    np.testing.assert_array_equal(screen._canvas.detection_source(), expected)
    capture('03_apply_convolution_for_display_and_detection')
    provenance = screen._chain_provenance()['enhancement']['psf']
    click(screen._btn_apply)
    np.testing.assert_array_equal(screen._canvas.detection_source(), original)
    np.testing.assert_array_equal(screen._canvas.image, original)
    np.testing.assert_array_equal(screen._canvas.mask, labels)
    if before != {str(path): _sha(path) for path in paths}:
        raise RuntimeError('PSF tutorial changed its input image')
    capture('04_return_to_original')
    return dict(alpha=False, sources=sources, provenance=provenance,
                source_hashes=before, original_and_labels_unchanged=True,
                processed_differs_from_original=not np.array_equal(original, expected),
                algorithm_verified='convolution against apply_psf',
                optical_recovery_claimed=False)


def record_measure(screen, capture, settle):
    """Show Measure's explicit original/processed choices without running a batch."""
    from spacr.psf_measurement import prepare_measurement_psf
    from spacr.qt import preferences
    from spacr.qt.settings_search import install

    if preferences._get_show_alpha_features():
        raise RuntimeError('Record Measure PSF controls with alpha off')
    screen.reveal_settings()
    # Authoring frame: make the existing settings pane wide enough for captions.
    split = getattr(screen, '_body_splitter', None)
    if split is not None and split.count() == 2:
        width = screen.width()
        split.setSizes([int(width * .6), int(width * .4)])
    bar = install(screen) or getattr(screen, '_settings_search', None)
    if bar is None or not bar.reveal('psf_measurement_source'):
        raise RuntimeError('The real Measure PSF settings could not be revealed')
    settle(.2)
    choose, click, text = _actions(screen, settle)
    fields = screen._settings_model._widgets
    from spacr.qt.widgets.section import Section

    ancestors = []
    parent = fields['psf_measurement_source'].parentWidget()
    while parent is not None:
        if isinstance(parent, Section):
            ancestors.append(parent)
        parent = parent.parentWidget()
    for section in reversed(ancestors):
        if not section.is_expanded():
            click(section.header())
    initial_source = screen._settings_model.collect()['psf_measurement_source']
    choose(fields['psf_measurement_source'], 'original')
    if prepare_measurement_psf(screen._settings_model.collect()) is not None:
        raise RuntimeError('Original Measure choice unexpectedly enables PSF processing')
    capture('05_measure_original_intensities')
    choose(fields['psf_measurement_source'], 'processed')
    choose(fields['psf_operation'], 'convolve')
    text(fields['psf_image_sampling_um'], '[0.2, 0.2]')
    text(fields['psf_fwhm_um'], '[0.6, 0.6]')
    settings = screen._settings_model.collect()
    plan = prepare_measurement_psf(settings)
    if plan is None:
        raise RuntimeError('Explicit processed Measure choice did not construct a PSF plan')
    capture('06_measure_explicit_processed_intensities')
    choose(fields['psf_measurement_source'], 'original')
    return dict(original_is_default=initial_source == 'original', processed_plan=True,
                settings={key: value for key, value in settings.items() if key.startswith('psf_')},
                measurement_run_performed=False, source_images_modified=False,
                mask_timelapse_psf_requires_alpha=True)
