"""Record actual CPU restoration controls, comparison and Apply on a private field."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import time


def record_restoration(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    import numpy as np
    import tifffile
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QLineEdit, QScrollArea

    from spacr.qt.widgets.collapsible_splitter import FoldSection
    from spacr.qt.widgets.section import Section

    folder = Path(stage) / 'restoration/images'
    files = sorted(folder.glob('*.tif'))
    if len(files) != 1 or (folder / 'masks').exists():
        raise RuntimeError('Prepare one exact intensity field without an editable companion mask')
    source = files[0]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = tifffile.imread(source)
    errors = []

    def picker():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise RuntimeError('Open folder did not show the actual picker')
            dialog.setDirectory(str(folder))
            dialog.setSidebarUrls([QUrl.fromLocalFile(str(folder))])
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            QTest.mouseClick(edit, Qt.LeftButton)
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(folder))
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    def wait(check, message):
        deadline = time.monotonic() + timeout
        while not check():
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            settle(.1)
        settle()

    QTimer.singleShot(700, picker)
    QTest.mouseClick(screen._btn_open, Qt.LeftButton)
    if errors:
        raise RuntimeError('; '.join(errors))
    wait(lambda: not screen._loading and screen._canvas.image is not None,
         'The actual restoration field did not load')
    canvas = screen._canvas
    if not np.array_equal(canvas.image, expected) or np.any(canvas.mask):
        raise RuntimeError('The restoration input differs from the exact prepared intensity field')

    def expose(widget):
        ancestors = []
        node, scroll = widget.parentWidget(), None
        while node is not None:
            if isinstance(node, (Section, FoldSection)):
                ancestors.append(node)
            if isinstance(node, QScrollArea):
                scroll = node
            node = node.parentWidget()
        for section in reversed(ancestors):
            folded = not section.is_expanded() if isinstance(section, Section) else section.shut
            if folded:
                heading = section.header() if isinstance(section, Section) else section.heading
                if scroll is not None:
                    scroll.ensureWidgetVisible(heading)
                settle()
                QTest.mouseClick(heading, Qt.LeftButton)
                settle()
        if scroll is not None:
            scroll.ensureWidgetVisible(widget)
        settle()
        point = widget.mapToGlobal(widget.rect().center())
        if not widget.isVisible() or not widget.isEnabled() or not app.primaryScreen().availableGeometry().contains(point):
            raise RuntimeError('A restoration control is outside the actual desktop')

    for section in screen.findChildren(Section):
        if section.title() in {'DISPLAY', 'FILTER', 'OBJECT OPERATIONS'} and section.is_expanded():
            expose(section.header())
            QTest.mouseClick(section.header(), Qt.LeftButton)
            settle()
    restoration = screen._restoration_controls
    expose(restoration.operation)
    capture('restoration_01_restoration_row')
    restoration.operation.setFocus()
    QTest.keyClick(restoration.operation, Qt.Key_Home)
    QTest.keyClick(restoration.operation, Qt.Key_Down)
    if restoration.operation.currentData() != 'denoise' or restoration.structure.currentData() != 'cyto3':
        raise RuntimeError('The native Denoise choice/default model differs from the lesson')
    expose(restoration.structure)
    capture('restoration_02_denoise_settings')
    expose(restoration.reload)
    started = time.monotonic()
    QTest.mouseClick(restoration.reload, Qt.LeftButton)
    wait(lambda: restoration._plan is not None or restoration.status.text().startswith('Restoration unavailable:'),
         'The real CPU restoration model did not become ready')
    load_seconds = time.monotonic() - started
    plan = restoration._plan
    if plan is None or not plan.device.startswith('cpu'):
        raise RuntimeError('Restoration did not load on the CPU: ' + restoration.status.text())
    expose(restoration.status)
    capture('restoration_03_model_ready')
    expose(screen._btn_compare)
    started = time.monotonic()
    QTest.mouseClick(screen._btn_compare, Qt.LeftButton)
    wait(lambda: screen._compare_dialog is not None and screen._comparison_request is None,
         'The actual restoration comparison did not finish')
    dialog = screen._compare_dialog
    if not dialog.isVisible() or dialog.views[1]._item is None or dialog.views[1]._item.pixmap().isNull():
        raise RuntimeError('The real comparison has no enhanced picture: ' + dialog.caption.text())
    compare_seconds = time.monotonic() - started
    caption = dialog.caption.text()
    capture('restoration_04_compare')
    QTest.mouseClick(dialog._buttons.button(QDialogButtonBox.Close), Qt.LeftButton)
    settle()
    expose(screen._btn_apply)
    if screen._btn_apply.isChecked():
        raise RuntimeError('Apply must start off in the isolated fresh recording')
    started = time.monotonic()
    QTest.mouseClick(screen._btn_apply, Qt.LeftButton)
    wait(lambda: canvas._enhanced_picture is not None or canvas._enhance_failure is not None,
         'The actual Apply restoration did not finish')
    if canvas._enhance_failure is not None:
        raise RuntimeError(str(canvas._enhance_failure))
    enhanced = canvas._enhanced_picture[2]
    if enhanced.shape != expected.shape or not np.isfinite(enhanced).all() or np.array_equal(enhanced, expected):
        raise RuntimeError('Apply did not produce a distinct finite same-grid enhanced picture')
    capture('restoration_05_apply')
    if not np.array_equal(canvas.image, expected) or np.any(canvas.mask) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Restoration changed original intensity pixels or editable labels')
    write_json(captures / 'restoration_acceptance.json', {
        'accepted': True, 'scope': 'Actual CPU restoration loading, comparison and Apply; no accuracy claim',
        'native_controls': True, 'application_monkeypatches': False, 'source_label_rewritten': False,
        'source': str(source), 'source_sha256': digest, 'shape': list(expected.shape),
        'plan': asdict(plan), 'load_seconds': load_seconds, 'compare_seconds': compare_seconds,
        'apply_seconds': time.monotonic() - started, 'comparison_caption': caption,
        'enhanced_array_sha256': hashlib.sha256(enhanced.tobytes()).hexdigest(),
        'original_image_and_labels_unchanged': True, 'measurements_generated': False,
        'biological_ground_truth_claimed': False})
