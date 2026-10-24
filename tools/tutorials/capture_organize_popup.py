"""Record item 639's organizer popups through their real buttons.

Mask Generation: "Organize images…" (images only). Import Images:
"Organize images and masks…" with "Write for" set to Mask Generation. The
popup is filled with private copies of real example fields and then
cancelled, so nothing is moved.
"""
from __future__ import annotations

import capture_ui
import capture_ui_drv_d as drv
import tempfile
import time
from pathlib import Path


#: Flat export the host fills through the regex route (no file manager drag).
FLAT_REGEX = r'(?P<wellID>[A-Z]\d+)_f(?P<fieldID>\d+)_C(?P<chanID>\d+)\.tif'


def _prepare_flat(stage):
    """One folder of channel-coded files (B03_f01_C1.tif) from the Measure example."""
    import numpy as np
    import tifffile

    sources = sorted((Path(stage) / 'example_data/plate1/merged').glob('*.npy'))[:2]
    if len(sources) != 2:
        raise RuntimeError('Load the Measure example into the stage first')
    (Path(stage) / 'make_masks_runs').mkdir(exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix='organize-639-', dir=Path(stage) / 'make_masks_runs'))
    export = parent / 'microscope_export'
    export.mkdir(parents=True)
    for number, source in enumerate(sources, start=1):
        merged = np.load(source, mmap_mode='r')
        for plane in (0, 1, 2):
            tifffile.imwrite(export / f'B03_f{number:02d}_C{plane + 1}.tif', np.asarray(merged[..., plane]))
    return export


def _prepare(stage, with_masks):
    """Per-channel folders (and a cell-mask folder) from the Measure example."""
    import numpy as np
    import tifffile

    sources = sorted((Path(stage) / 'example_data/plate1/merged').glob('*.npy'))[:2]
    if len(sources) != 2:
        raise RuntimeError('Load the Measure example into the stage first')
    (Path(stage) / 'make_masks_runs').mkdir(exist_ok=True)      # _prepare_flat made it; this route did not (FileNotFoundError)
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
    regex_route = capture_ui.live(capture) and module == 'mask'
    export = _prepare_flat(stage) if regex_route else _prepare(stage, with_masks=module != 'mask')
    facts, errors = {}, []
    # Host: the clip opens before the pointer scrolls to and clicks the button.
    clip_state = {'started': drv.begin(capture, settle, 'organize_popup')}
    if scroll is not None:
        capture_ui.scroll_to(capture, scroll, button)
        settle(0.3)

    def fill():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, OrganizeForMeasureDialog):
                raise RuntimeError(f'{name} did not open the organizer')
            capture_ui.dialog_geometry(capture, dialog, (2700, 1650))
            capture_ui.dialog_geometry(capture, dialog, None, window.pos() + __import__('PySide6.QtCore', fromlist=['QPoint']).QPoint(570, 260))
            settle(0.5)
            if regex_route:
                # The regex route, as a user does it: the source folder, the
                # 'custom' regex choice, the regex typed, then Sort by regex.
                from PySide6.QtWidgets import QComboBox
                capture_ui.set_text(capture, dialog.source_edit, str(export), commit=False)
                settle(0.4)
                kind = dialog.metadata_field.findChild(QComboBox)
                custom = kind.findData('custom')
                if custom < 0:
                    raise RuntimeError('The regex box offers no custom choice')
                capture_ui.select_combo(capture, kind, custom)
                settle(0.4)
                capture_ui.set_text(capture, dialog.custom_edit, FLAT_REGEX, commit=False)
                settle(0.4)
                capture_ui.press(capture, dialog.sort_button)
                settle(1.0)
            else:
                capture_ui.code_action(capture, 'organizer columns filled by add_column/add_files: a folder '
                                       'drop needs a file manager, not available in the capture host')
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
                        capture_ui.select_combo(capture, box, i)
                facts['write_for'] = box.currentText()
                facts['split_button_visible'] = dialog.split_button.isVisible()
            index = dialog.view_box.findData('both')
            if index >= 0:
                capture_ui.select_combo(capture, dialog.view_box, index)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            settle(1.5)
            facts['rows'] = len(dialog.rows)
            facts['complete'] = all(all(row) for row in dialog.rows)
            if facts['rows'] != 2 or not facts['complete']:
                raise RuntimeError(f'The organizer did not match two complete rows: {dialog.rows}')
            capture_ui.wait(capture, 1.5, settle)
            drv.snap(capture, 'organize_popup')
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            drv.finish(capture, clip_state['started'])
            if dialog is not None:
                capture_ui.reject_dialog(capture, dialog)

    capture_ui.when_modal(capture, 600, fill)
    QTest.mouseClick(button, Qt.LeftButton)
    settle(0.5)
    if errors:
        raise RuntimeError('; '.join(errors))
    write_json(captures / 'organize_popup_acceptance.json', {
        'accepted': True, 'module': module, 'button': name, 'moved': False,
        'cancelled_after_capture': True, **facts})
