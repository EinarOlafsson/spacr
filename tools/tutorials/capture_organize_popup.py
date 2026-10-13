"""Record item 639's organizer popups through their real buttons.

Mask Generation: "Organize images…" (images only). Import Images:
"Organize images and masks…" with "Write for" set to Mask Generation. The
popup is filled with private copies of real example fields and then
cancelled, so nothing is moved.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path


def _prepare(stage, with_masks):
    """Per-channel folders (and a cell-mask folder) from the Measure example."""
    import numpy as np
    import tifffile

    sources = sorted((Path(stage) / 'example_data/plate1/merged').glob('*.npy'))[:2]
    if len(sources) != 2:
        raise RuntimeError('Load the Measure example into the stage first')
    parent = Path(tempfile.mkdtemp(prefix='organize-639-', dir=Path(stage) / 'make_masks_runs'))
    export = parent / 'microscope_export'
    for number, source in enumerate(sources, start=1):
        merged = np.load(source, mmap_mode='r')
        field = f'B03_f{number:02d}'
        for plane, channel in ((0, 'Hoechst'), (1, 'ER'), (2, 'Toxo')):
            target = export / channel / f'{field}.tif'
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, np.asarray(merged[..., plane]))
        if with_masks:
            target = export / 'cell_masks' / f'{field}.tif'
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, np.asarray(merged[..., 4]))
    return export


def record_organize_popup(app, window, screen, stage, captures, capture, settle,
                          write_json, *, module):
    from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.widgets.organize_for_measure import OrganizeForMeasureDialog

    name = 'MaskOrganizeImagesButton' if module == 'mask' else 'ImportOrganizeButton'
    buttons = [b for b in screen.findChildren(QPushButton, name) if b.isVisible()]
    if len(buttons) != 1:
        raise RuntimeError(f'Expected one visible {name}')
    button = buttons[0]
    scroll = getattr(screen, '_settings_scroll', None)
    if scroll is not None:
        scroll.ensureWidgetVisible(button)
        settle(0.3)
    export = _prepare(stage, with_masks=module != 'mask')
    facts, errors = {}, []

    def fill():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, OrganizeForMeasureDialog):
                raise RuntimeError(f'{name} did not open the organizer')
            dialog.resize(2700, 1650)
            dialog.move(window.pos().x() + 570, window.pos().y() + 260)
            settle(0.5)
            for channel in ('Hoechst', 'ER', 'Toxo'):
                index = dialog.add_column('channel')
                dialog.add_files(index, [str(export / channel)])
            if module != 'mask':
                er = dialog._channel_of(1)
                column = dialog.add_column('mask', role='cell', of_channel=er)
                dialog.add_files(column, [str(export / 'cell_masks')])
                box = dialog.layout_box
                for i in range(box.count()):
                    if 'Mask Generation' in box.itemText(i):
                        box.setCurrentIndex(i)
                facts['write_for'] = box.currentText()
                facts['split_button_visible'] = dialog.split_button.isVisible()
            index = dialog.view_box.findData('both')
            if index >= 0:
                dialog.view_box.setCurrentIndex(index)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            settle(1.5)
            facts['rows'] = len(dialog.rows)
            facts['complete'] = all(all(row) for row in dialog.rows)
            if facts['rows'] != 2 or not facts['complete']:
                raise RuntimeError(f'The organizer did not match two complete rows: {dialog.rows}')
            capture('organize_popup')
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(600, fill)
    QTest.mouseClick(button, Qt.LeftButton)
    settle(0.5)
    if errors:
        raise RuntimeError('; '.join(errors))
    write_json(captures / 'organize_popup_acceptance.json', {
        'accepted': True, 'module': module, 'button': name, 'moved': False,
        'cancelled_after_capture': True, **facts})
