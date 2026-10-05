"""Record native parent-constrained puncta controls on an exact acquired pair."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time


def record_puncta(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    import numpy as np
    import tifffile
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QMessageBox, QPushButton, QScrollArea

    from spacr import tabular
    from spacr.qt import cpu_modes
    from spacr.qt.widgets.section import Section

    images, parents = Path(stage) / 'puncta/images', Path(stage) / 'puncta/parents'
    files = sorted(images.glob('*.tif'))
    if len(files) != 1 or not (parents / files[0].name).is_file():
        raise RuntimeError('Prepare one exact acquired puncta image and its parent-mask pair')
    source, parent = files[0], parents / files[0].name
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    expected_image, expected_parent = tifffile.imread(source), tifffile.imread(parent)
    if expected_image.shape != expected_parent.shape or not np.any(expected_parent):
        raise RuntimeError('The actual parent mask does not match the image')
    errors = []

    def picker():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise RuntimeError('Open folder did not open the real folder picker')
            dialog.setDirectory(str(images))
            dialog.setSidebarUrls([QUrl.fromLocalFile(str(images))])
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            QTest.mouseClick(edit, Qt.LeftButton)
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(images))
            QTest.keyClick(edit, Qt.Key_Return)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(700, picker)
    QTest.mouseClick(screen._btn_open, Qt.LeftButton)
    if errors:
        raise RuntimeError('; '.join(errors))

    def wait(check, message):
        deadline = time.monotonic() + timeout
        while not check():
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            settle(.1)
        settle()

    wait(lambda: not screen._loading and screen._canvas.image is not None,
         'The genuine puncta image did not load')
    canvas = screen._canvas
    if not np.array_equal(canvas.image, expected_image):
        raise RuntimeError('The canvas is not the acquired raw image')
    capture('puncta_01_raw_input')

    def expose(widget):
        sections = []
        ancestor = widget.parentWidget()
        scroll = None
        while ancestor is not None:
            if isinstance(ancestor, Section):
                sections.append(ancestor)
            if isinstance(ancestor, QScrollArea):
                scroll = ancestor
            ancestor = ancestor.parentWidget()
        for section in reversed(sections):
            if not section.is_expanded():
                if scroll is not None:
                    scroll.ensureWidgetVisible(section.header())
                settle()
                QTest.mouseClick(section.header(), Qt.LeftButton)
                settle()
        if scroll is not None:
            scroll.ensureWidgetVisible(widget)
        settle()
        centre = widget.mapToGlobal(widget.rect().center())
        if not widget.isVisible() or not widget.isEnabled() or not app.primaryScreen().availableGeometry().contains(centre):
            raise RuntimeError('A puncta control is outside the actual desktop')

    def choose(box, value):
        expose(box)
        index = box.findData(value)
        if index < 0:
            raise RuntimeError(f'The actual selector does not offer {value}')
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        settle()
        if box.currentData() != value:
            raise RuntimeError('The actual selector did not retain its choice')

    def type_into(edit, text):
        expose(edit)
        QTest.mouseClick(edit, Qt.LeftButton)
        edit.setFocus()
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, text)
        QTest.keyClick(edit, Qt.Key_Tab)
        settle()
        if edit.text() != text:
            raise RuntimeError('The native parent-source editor did not retain its value')

    choose(screen._mag_mode, cpu_modes.PUNCTA)
    for section in screen.findChildren(Section):
        if section.title() in {'DISPLAY', 'FILTER', 'OBJECT OPERATIONS'} and section.is_expanded():
            expose(section.header())
            QTest.mouseClick(section.header(), Qt.LeftButton)
            settle()
    selector = screen._primary_selector
    type_into(selector.primary_class.lineEdit(), 'cyst')
    type_into(selector.secondary_class.lineEdit(), 'punctum')
    type_into(selector.path, str(parents))
    reload_button = next(button for button in selector.findChildren(QPushButton)
                         if button.text() == 'Reload')
    expose(reload_button)
    QTest.mouseClick(reload_button, Qt.LeftButton)

    def parent_ready():
        if selector.error:
            raise RuntimeError(f'The genuine parent-mask source was refused: {selector.error}')
        return selector.snapshot is not None

    try:
        wait(parent_ready, 'The genuine parent mask did not load')
    except Exception:
        write_json(captures / 'parent_source_diagnostic.json', {
            'path': selector.path.text(), 'status': selector.status.text(),
            'error': selector.error, 'field': selector._field,
            'primary_class': selector.primary_class.currentText(),
            'secondary_class': selector.secondary_class.currentText(),
            'source_editor_has_focus': selector.path.hasFocus()})
        raise
    expose(screen._puncta_widgets['puncta_sigmas'])
    capture('puncta_02_parent_and_settings')
    choose(screen._combine_mode, 'replace')

    def reject_warning():
        dialog = app.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            errors.append(dialog.text())
            dialog.reject()

    QTimer.singleShot(700, reject_warning)
    expose(screen._btn_otsu)
    QTest.mouseClick(screen._btn_otsu, Qt.LeftButton)
    settle()
    if errors or not np.any(canvas.mask):
        raise RuntimeError('; '.join(errors) or 'Puncta detection produced no included centres')
    if np.any(canvas.mask[expected_parent == 0]):
        raise RuntimeError('Puncta labels leave the acquired parent masks')
    capture('puncta_03_detected_centres')
    QTimer.singleShot(700, reject_warning)
    QTest.mouseClick(screen._btn_save, Qt.LeftButton)
    settle()
    if errors:
        raise RuntimeError('; '.join(errors))
    csv_path = images / 'masks' / (source.stem + '.puncta.csv')
    rows = tabular.read_table(csv_path, canonicalise=False)
    mask_path = images / 'masks' / source.name
    if not np.array_equal(tifffile.imread(mask_path), canvas.mask):
        raise RuntimeError('The actual saved puncta mask differs from the canvas')
    required = {'candidate_id', 'parent_id', 'center20', 'center_corrected',
                'local_bg', 'included', 'mask_pixels'}
    if not required.issubset(rows.columns):
        raise RuntimeError('Exact centre/annulus measurements were not saved')
    capture('puncta_04_saved_mask_and_measurements')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash or hashlib.sha256(parent.read_bytes()).hexdigest() != parent_hash:
        raise RuntimeError('An acquired image or parent mask changed')
    write_json(captures / 'puncta_acceptance.json', {
        'accepted': True, 'scope': 'Native one-field whole-image puncta controls and saving; scientific reference replay is checked separately',
        'image': str(source), 'image_sha256': source_hash,
        'parent': str(parent), 'parent_sha256': parent_hash,
        'native_controls': True, 'warnings_replaced': False,
        'candidates': len(rows), 'included': int(rows.included.sum()),
        'saved_mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        'saved_csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'original_image_and_parent_unchanged': True,
        'mask_outside_parent_pixels': 0, 'biological_ground_truth_claimed': False})
