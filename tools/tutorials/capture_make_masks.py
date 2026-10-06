"""Record genuine mask-editor gestures on isolated, real downloaded fields.

Preparing copies extracts exact image/label planes. Only the application's
visible gestures and buttons edit masks, save them or recrop them.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_fields(stage):
    import numpy as np
    import tifffile
    sources = sorted((stage / 'example_data/plate1/merged').glob('*.npy'))[:2]
    if len(sources) != 2:
        raise RuntimeError('Load the genuine Measure example before recording Make Masks')
    parent = stage / 'make_masks_runs'
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='example-', dir=parent))
    (folder / 'masks').mkdir()
    evidence = []
    for source in sources:
        merged = np.load(source, mmap_mode='r')
        if merged.ndim != 3 or merged.shape[-1] != 7 or merged.dtype != np.uint16:
            raise RuntimeError('Expected the downloaded seven-plane uint16 measurement example')
        # The actual Measure example settings name cell_mask_dim=4; its
        # corresponding ER intensity plane is 1, not a predicted crop image.
        image, labels = merged[..., 1], merged[..., 4]
        if not np.any(labels > 0):
            raise RuntimeError('The example field has no cell labels')
        name = source.stem + '_ER.tif'
        target, mask = folder / name, folder / 'masks' / name
        tifffile.imwrite(target, image)
        tifffile.imwrite(mask, labels)
        if not np.array_equal(tifffile.imread(target), image) or not np.array_equal(tifffile.imread(mask), labels):
            raise RuntimeError('Plane extraction changed the actual pixels or label identities')
        evidence.append({'source': str(source), 'source_sha256': digest(source),
                         'image': str(target), 'image_sha256': digest(target),
                         'mask': str(mask), 'mask_sha256': digest(mask),
                         'image_plane': 1, 'cell_mask_plane': 4,
                         'objects': int(np.count_nonzero(np.unique(labels))),
                         'shape': list(image.shape)})
    return folder, evidence


def record_yolo_boxes(app, window, screen, stage, captures, capture, settle,
                      write_json, timeout):
    """Record real annotation gestures and independently verify their exports.

    The boxes illustrate interaction only. They are not biological training
    truth, and all acquired image and companion mask bytes must stay intact.
    """
    import json
    import numpy as np
    import tifffile
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QDialogButtonBox, QFileDialog, QInputDialog,
                                   QLineEdit, QScrollArea)
    from spacr.qt import mask_engine as engine

    folder, inputs = prepare_fields(stage)
    write_json(captures / 'yolo_inputs.json', inputs)
    errors = []

    def expose(widget):
        for area in window.findChildren(QScrollArea):
            if area.widget() is not None and area.widget().isAncestorOf(widget):
                area.ensureWidgetVisible(widget, 30, 30)
        settle(.2)
        if not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('The requested YOLO control is not usable')

    def picker(button, destination, name, *, save=False):
        answered = []

        def choose():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('The control did not open a real file picker')
                dialog.accepted.connect(lambda: answered.append(True))
                dialog.resize(1300, 950)
                dialog.setSidebarUrls([QUrl.fromLocalFile(str(stage))])
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                if edit is None:
                    raise RuntimeError('The actual file picker has no filename field')
                QTest.mouseClick(edit, Qt.LeftButton)
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(destination))
                if edit.text() != str(destination):
                    raise RuntimeError('The picker did not take the exact destination')
                settle(.2)
                capture(name)
                box = dialog.findChild(QDialogButtonBox)
                action = QDialogButtonBox.Save if save else QDialogButtonBox.Open
                QTest.mouseClick(box.button(action), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()

        def reject_stalled():
            if answered or errors:
                return
            errors.append('The actual file picker was not answered')
            dialog = app.activeModalWidget()
            if dialog is not None:
                dialog.reject()

        expose(button)
        QTimer.singleShot(500, choose)
        QTimer.singleShot(12000, reject_stalled)
        previous = Path.cwd()
        try:
            os.chdir(stage)
            QTest.mouseClick(button, Qt.LeftButton)
        finally:
            os.chdir(previous)
        if errors or not answered:
            raise RuntimeError('; '.join(errors) or 'The picker was cancelled')
        settle()

    def ready():
        deadline = time.monotonic() + timeout
        while screen._loading or screen._canvas.image is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('The actual YOLO source image did not load')
            settle(.1)
        settle()

    picker(screen._btn_open, folder, 'yolo_02_folder_picker')
    ready()
    canvas = screen._canvas
    pixels, mask = canvas.image.copy(), canvas.mask.copy()
    filename = screen._image_files[screen._current_index]
    assert np.array_equal(pixels, tifffile.imread(folder / filename))
    assert np.array_equal(mask, tifffile.imread(folder / 'masks' / filename))
    capture('yolo_03_source_and_mask')
    expose(screen._mode_buttons['box'])
    QTest.mouseClick(screen._mode_buttons['box'], Qt.LeftButton)
    settle()
    if canvas.mode != 'box' or not screen._box_controls.isVisible():
        raise RuntimeError('The Box button did not reveal the real class controls')
    capture('yolo_04_box_controls')

    def name_class():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QInputDialog):
                raise RuntimeError('Add class did not show its actual name dialog')
            edit = dialog.findChild(QLineEdit)
            QTest.mouseClick(edit, Qt.LeftButton)
            QTest.keyClicks(edit, 'demonstration')
            capture('yolo_05_add_class')
            box = dialog.findChild(QDialogButtonBox)
            QTest.mouseClick(box.button(QDialogButtonBox.Ok), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    expose(screen._btn_add_box_class)
    QTimer.singleShot(500, name_class)
    QTest.mouseClick(screen._btn_add_box_class, Qt.LeftButton)
    settle()
    if errors or screen._box_classes != ['object', 'demonstration']:
        raise RuntimeError('; '.join(errors) or 'The real class map was not appended')

    def position(x, y):
        shown = canvas._image_to_canvas(x, y)
        if shown is None or not canvas.rect().contains(shown):
            raise RuntimeError('The intended box gesture is outside the visible image')
        return shown

    def drag(start, end, modifiers=Qt.NoModifier):
        begin, finish = position(*start), position(*end)
        QTest.mouseMove(canvas, begin)
        QTest.mousePress(canvas, Qt.LeftButton, modifiers, begin)
        settle(.08)
        QTest.mouseMove(canvas, finish, delay=40)
        settle(.08)
        QTest.mouseRelease(canvas, Qt.LeftButton, modifiers, finish)
        settle()

    height, width = pixels.shape[:2]
    drag((int(width * .35), int(height * .35)),
         (int(width * .55), int(height * .55)))
    if len(canvas.boxes) != 1 or canvas.boxes[0][0] != 1:
        raise RuntimeError('The visible gesture did not draw one class-1 box')
    drawn = list(canvas.boxes[0])
    capture('yolo_06_drawn_box')
    cx, cy = (drawn[1] + drawn[3]) / 2, (drawn[2] + drawn[4]) / 2
    drag((cx, cy), (cx + int(width * .05), cy + int(height * .05)))
    moved = list(canvas.boxes[0])
    if moved == drawn or moved[3] - moved[1] != drawn[3] - drawn[1]:
        raise RuntimeError('Dragging the box did not translate it at fixed width')
    capture('yolo_07_moved_box')
    drag((moved[3] - .75, moved[4] - .75),
         (moved[3] + int(width * .04), moved[4] + int(height * .04)))
    resized = list(canvas.boxes[0])
    if resized[1:3] != moved[1:3] or resized[3] <= moved[3] or resized[4] <= moved[4]:
        raise RuntimeError('The visible corner gesture did not enlarge the box')
    capture('yolo_08_resized_box')
    dx, dy = resized[3] - resized[1], resized[4] - resized[2]
    drag((resized[1] + .25 * dx, resized[2] + .25 * dy),
         (resized[1] + .55 * dx, resized[2] + .55 * dy), Qt.ControlModifier)
    if len(canvas.boxes) != 2 or list(canvas.boxes[0]) != resized:
        raise RuntimeError('Ctrl-drag did not add the contained demonstration box')
    saved_boxes = [list(box) for box in canvas.boxes]
    capture('yolo_09_contained_box')
    second = saved_boxes[1]
    QTest.mouseClick(canvas, Qt.RightButton,
                     pos=position((second[1] + second[3]) / 2,
                                  (second[2] + second[4]) / 2))
    settle()
    if len(canvas.boxes) != 1:
        raise RuntimeError('Right-click did not delete the box under the cursor')
    capture('yolo_10_deleted_box')
    QTest.mouseClick(screen._btn_undo, Qt.LeftButton)
    settle()
    if [list(box) for box in canvas.boxes] != saved_boxes:
        raise RuntimeError('Box Undo did not restore both exact annotations')
    capture('yolo_11_undo_boxes')
    QTest.mouseClick(screen._btn_redo, Qt.LeftButton)
    settle()
    if len(canvas.boxes) != 1:
        raise RuntimeError('Box Redo did not restore the demonstrated deletion')
    capture('yolo_12_redo_boxes')
    QTest.mouseClick(screen._btn_undo, Qt.LeftButton)
    settle()
    expose(screen._btn_save_boxes)
    QTest.mouseClick(screen._btn_save_boxes, Qt.LeftButton)
    settle()
    project = folder / engine.YOLO_ANNOTATIONS_NAME
    stored = engine.load_yolo_boxes(folder, filename, pixels.shape)
    if screen._boxes_dirty or [list(box) for box in stored['boxes']] != saved_boxes:
        raise RuntimeError('Save boxes did not persist the exact annotations')
    capture('yolo_13_saved_boxes')
    labels = folder / (Path(filename).stem + '.txt')
    picker(screen._btn_export_yolo, labels, 'yolo_14_export_picker', save=True)
    exported = [line.split() for line in labels.read_text().splitlines()]
    if len(exported) != len(saved_boxes):
        raise RuntimeError('The exported YOLO row count differs from the visible boxes')
    for fields, (class_id, x0, y0, x1, y1) in zip(exported, saved_boxes):
        expected = [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                    (x1 - x0) / width, (y1 - y0) / height]
        if len(fields) != 5 or int(fields[0]) != class_id or not np.allclose(
                list(map(float, fields[1:])), expected, rtol=0, atol=1e-6):
            raise RuntimeError('The YOLO export does not contain normalized full-image XYWH')
    classes = folder / '.classes.json'
    if 'demonstration' not in json.dumps(json.loads(classes.read_text())):
        raise RuntimeError('YOLO export omitted the stable class-name companion')
    capture('yolo_15_exported')
    QTest.mouseClick(screen._btn_next, Qt.LeftButton)
    ready()
    if canvas.boxes:
        raise RuntimeError('The untouched second field is not a negative example')
    negative_name = screen._image_files[screen._current_index]
    negative = folder / (Path(negative_name).stem + '.txt')
    picker(screen._btn_export_yolo, negative, 'yolo_16_negative_export_picker', save=True)
    if negative.read_bytes() != b'':
        raise RuntimeError('The negative image did not export an empty label file')
    capture('yolo_17_negative_exported')
    QTest.mouseClick(screen._btn_prev, Qt.LeftButton)
    ready()
    if screen._image_files[screen._current_index] != filename or [list(box) for box in canvas.boxes] != saved_boxes:
        raise RuntimeError('Navigation did not reload the exact saved annotations')
    capture('yolo_18_reloaded_boxes')
    if not np.array_equal(canvas.image, pixels) or not np.array_equal(canvas.mask, mask):
        raise RuntimeError('Box annotation changed source image or segmentation pixels')
    for row in inputs:
        for key in ('source', 'image', 'mask'):
            if digest(Path(row[key])) != row[key + '_sha256']:
                raise RuntimeError('Box annotation changed original acquired input bytes')
    write_json(captures / 'yolo_acceptance.json', {
        'accepted': True, 'demonstration_boxes_are_not_biological_ground_truth': True,
        'source_images_and_masks_unchanged': True, 'inputs': inputs,
        'classes': list(screen._box_classes), 'boxes': saved_boxes,
        'drawn': drawn, 'moved': moved, 'resized': resized,
        'independent_xywh_export_check': True, 'undo_redo_and_reload_exact': True,
        'project': str(project), 'project_sha256': digest(project),
        'labels': str(labels), 'labels_sha256': digest(labels),
        'class_names': str(classes), 'class_names_sha256': digest(classes),
        'negative_labels': str(negative), 'negative_labels_sha256': digest(negative),
    })


def record_editor(app, window, screen, stage, captures, capture, settle, write_json, timeout,
                  *, detect=False, readouts_only=False, include_readouts=False,
                  curation_organize=False):
    import numpy as np
    import tifffile
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QLineEdit, QMessageBox, QScrollArea
    from scipy.ndimage import distance_transform_edt

    from spacr.qt.screens.make_masks import FOLD_ORDER

    # Load test data is a split button: its arrow lists the test data and a
    # sample of each training dataset. Open the menu only; nothing downloads.
    from PySide6.QtCore import QPoint
    test_data = screen._btn_test_data
    QTest.mouseClick(test_data, Qt.LeftButton,
                     pos=QPoint(test_data.width() - 8, test_data.height() // 2))
    settle(0.6)
    menu = test_data.split_menu()
    if menu is None or not menu.isVisible():
        raise RuntimeError('The Load test data arrow did not open its menu')
    capture('01b_test_data_menu')
    menu.hide()
    settle()

    folder, evidence = prepare_fields(stage)
    write_json(captures / 'inputs.json', evidence)
    errors, accepted = [], []

    def choose():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise RuntimeError('Open folder did not open the actual directory picker')
            dialog.accepted.connect(lambda: accepted.append(True))
            dialog.resize(1300, 950)
            dialog.setSidebarUrls([QUrl.fromLocalFile(str(stage))])
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            settle(0.2)
            QTest.mouseClick(edit, Qt.LeftButton)
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            if edit.selectedText() != edit.text():
                raise RuntimeError('The folder picker did not select its previous entry')
            QTest.keyClicks(edit, str(folder))
            if edit.text() != str(folder):
                raise RuntimeError('The folder picker did not take the exact prepared path')
            capture('02_folder_picker')
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    def reject_stalled():
        if accepted and not errors:
            return  # the picker was answered; a later box is not this one's
        dialog = app.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            errors.append(dialog.text())
            capture('02_folder_warning')
            dialog.reject()
        elif dialog is not None and not accepted:
            errors.append('Folder picker did not accept the real path')
            dialog.reject()

    QTimer.singleShot(500, choose)
    QTimer.singleShot(12000, reject_stalled)
    previous_directory = Path.cwd()
    try:
        # The genuine picker starts at cwd when no folder is open. Give it
        # the neutral prepared workspace before showing its first frame.
        os.chdir(stage)
        QTest.mouseClick(screen._btn_open, Qt.LeftButton)
    finally:
        os.chdir(previous_directory)
    if errors or not accepted:
        raise RuntimeError('; '.join(errors) or 'Folder selection was cancelled')

    def ready():
        deadline = time.monotonic() + timeout
        while screen._loading or screen._canvas.image is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('The actual image/mask pair did not load')
            settle(0.1)
        settle()

    ready()
    canvas = screen._canvas
    original = canvas.mask.copy()
    pixels = canvas.image.copy()
    if not np.array_equal(original, tifffile.imread(evidence[0]['mask'])):
        raise RuntimeError('The editor did not load the actual companion labels')
    capture('03_real_labels')
    if readouts_only or include_readouts:
        from capture_mask_readouts import record_readouts

        # Record the readouts before editing or recropping this same field.
        # Prefix their frame names so a complete lesson cannot overwrite an
        # editor frame with the readouts tour's independently numbered scenes.
        def readout_capture(name, **kwargs):
            return capture(f'readouts_{name}', **kwargs)

        record_readouts(app, window, screen, captures,
                       readout_capture if include_readouts else capture,
                       settle, write_json, timeout)
        if not np.array_equal(canvas.mask, original) or not np.array_equal(canvas.image, pixels):
            raise RuntimeError('The readouts tour changed the field before editing')
        if readouts_only:
            return
    original_count = int(np.count_nonzero(np.unique(original)))
    steps = []

    def undo():
        QTest.mouseClick(screen._btn_undo, Qt.LeftButton)
        settle()
        if not np.array_equal(canvas.mask, original):
            raise RuntimeError('Undo did not restore every original label pixel')

    def mode(key):
        button = screen._mode_buttons[key]
        if not button.isVisible() or not button.isEnabled():
            raise RuntimeError(f'{key} is not a usable visible tool')
        QTest.mouseClick(button, Qt.LeftButton)
        settle()

    def gesture(points, *, button=Qt.LeftButton, modifiers=Qt.NoModifier):
        positions = [canvas._image_to_canvas(x, y) for x, y in points]
        if any(p is None or not canvas.rect().contains(p) or
               canvas._canvas_to_image(p.x(), p.y()) is None for p in positions):
            raise RuntimeError('The intended gesture is outside the visible image')
        QTest.mouseMove(canvas, positions[0])
        settle(0.08)
        QTest.mousePress(canvas, button, modifiers, pos=positions[0])
        settle(0.08)
        for position in positions[1:]:
            QTest.mouseMove(canvas, position, delay=40)
            settle(0.08)
        if canvas.mode == 'divide':
            name = 'merge' if button == Qt.RightButton else 'divide'
            write_json(captures / f'{name}_gesture.json', {
                'button': 'right' if button == Qt.RightButton else 'left',
                'requested_points': [[int(x), int(y)] for x, y in points],
                'observed_points': [list(canvas._canvas_to_image(p.x(), p.y()))
                                    for p in canvas._gesture_points]})
            capture(f'08_{name}_gesture')
        QTest.mouseRelease(canvas, button, modifiers, pos=positions[-1])
        settle()

    counts = np.bincount(original.ravel())
    for candidate in np.argsort(counts[1:])[::-1] + 1:
        yy, xx = np.nonzero(original == candidate)
        if len(xx) and min(xx.min(), yy.min()) > 20 and (
                xx.max() < original.shape[1]-21 and yy.max() < original.shape[0]-21):
            label = int(candidate)
            break
    else:
        raise RuntimeError('No complete interior label for reversible editing and recropping')
    center = int(xx[len(xx)//2]), int(yy[len(yy)//2])
    mode('erase_object')
    gesture([center])
    expected = original.copy()
    expected[expected == label] = 0
    if not np.array_equal(canvas.mask, expected):
        raise RuntimeError('Erase object did not remove exactly the clicked real label')
    capture('04_erase_one_object')
    steps.append({'action': 'erase_object', 'label': label, 'before': original_count,
                  'after': int(np.count_nonzero(np.unique(canvas.mask)))})
    undo()
    capture('05_undo_exactly')
    QTest.mouseClick(screen._btn_redo, Qt.LeftButton)
    settle()
    if not np.array_equal(canvas.mask, expected):
        raise RuntimeError('Redo did not restore the demonstrated deletion')
    capture('06_redo')
    undo()

    free = distance_transform_edt(original == 0)
    free[:80] = free[-80:] = 0
    free[:, :80] = free[:, -80:] = 0
    y, x = np.unravel_index(np.argmax(free), free.shape)
    radius = min(35, int(free[y, x] / 2))
    if radius < 10:
        raise RuntimeError('No clear background region for a reversible Draw demonstration')
    mode('draw')
    gesture([(x-radius, y-radius), (x+radius, y-radius),
             (x+radius, y+radius), (x-radius, y+radius), (x-radius, y-radius)])
    if np.count_nonzero(np.unique(canvas.mask)) != original_count + 1:
        raise RuntimeError('The actual Draw gesture did not create one closed object')
    capture('07_draw_demo_not_annotation')
    steps.append({'action': 'draw', 'demonstration_only': True,
                  'before': original_count, 'after': original_count + 1})
    undo()

    mode('divide')
    cut_y = int(np.median(yy))
    gesture([(int(xx.min())-3, cut_y), (int(xx.max())+3, cut_y)])
    divided_count = int(np.count_nonzero(np.unique(canvas.mask)))
    if divided_count <= original_count:
        raise RuntimeError('The actual Divide gesture did not split a label')
    capture('08_divide_demo')
    steps.append({'action': 'divide', 'before': original_count, 'after': divided_count})
    divided = canvas.mask.copy()
    pieces = [int(value) for value in np.unique(divided[original == label]) if value > 0]
    if len(pieces) < 2:
        raise RuntimeError('The chosen real object has no two divided pieces to merge')
    join_points = []
    for value in pieces[:2]:
        inside = distance_transform_edt((divided == value) & (original == label))
        py, px = np.unravel_index(np.argmax(inside), inside.shape)
        join_points.append((int(px), int(py)))
    gesture(join_points, button=Qt.RightButton)
    if not canvas.manual_ids or not np.array_equal(canvas.mask > 0, divided > 0):
        raise RuntimeError('Right-drag did not merge without painting background')
    if canvas.mask[join_points[0][1], join_points[0][0]] != canvas.mask[join_points[1][1], join_points[1][0]]:
        raise RuntimeError('The two crossed pieces did not retain one shared identity')
    if not np.array_equal(canvas.image, pixels):
        raise RuntimeError('Manual merge changed the acquired image')
    capture('08b_merge_demo')
    steps.append({'action': 'merge', 'button': 'right', 'demonstration_only': True,
                  'foreground_pixels_unchanged': True, 'acquired_image_unchanged': True,
                  'before': divided_count,
                  'after': int(np.count_nonzero(np.unique(canvas.mask)))})
    QTest.mouseClick(screen._btn_undo, Qt.LeftButton)
    settle()
    if not np.array_equal(canvas.mask, divided) or canvas.manual_ids:
        raise RuntimeError('Undo did not restore the pre-merge labels and identity policy')
    undo()

    def expose(widget):
        if not screen._btn_settings.isChecked():
            QTest.mouseClick(screen._btn_settings, Qt.LeftButton)
        for scroll in screen.findChildren(QScrollArea):
            if scroll.isAncestorOf(widget):
                scroll.ensureWidgetVisible(widget)
        settle()
        if not widget.isVisible() or not window.screen().geometry().contains(
                widget.mapToGlobal(widget.rect().center())):
            raise RuntimeError('The requested editor control is outside the recording surface')

    def number(widget, value):
        expose(widget)
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle()
        if widget.value() != value:
            raise RuntimeError('The visible editor setting did not take the requested value')

    def display_digest():
        # Keep the QImage alive while reading its owned pixel buffer.
        image = canvas.pixmap().toImage().copy()
        return hashlib.sha256(image.bits().tobytes()).hexdigest()

    mode('wand_add')
    if 'wand_erase' in screen._mode_buttons or screen._btn_wand.text() != 'Wand':
        raise RuntimeError('The current editor must offer exactly one Wand button')
    old_wand_pct, old_wand_max = screen._wand_pct.value(), screen._wand_max.value()
    number(screen._wand_pct, 0.5)
    number(screen._wand_max, 256)
    if screen._btn_settings.isChecked():
        QTest.mouseClick(screen._btn_settings, Qt.LeftButton)
        settle()
    gesture([(int(x), int(y))])
    added = canvas.mask.copy()
    changed = added != original
    if (not changed.any() or np.any(original[changed] != 0)
            or np.any(added[changed] == 0) or not np.array_equal(canvas.image, pixels)):
        raise RuntimeError('Default Wand did not add only a bounded real intensity region')
    capture('07b_wand_add')
    gesture([(int(x), int(y))], modifiers=Qt.ControlModifier)
    if not np.array_equal(canvas.mask, original) or not np.array_equal(canvas.image, pixels):
        raise RuntimeError('Ctrl+Wand did not remove exactly its added region')
    if screen._shortcut_rows['Ctrl + left click'][1].text() != 'Remove the intensity region':
        raise RuntimeError('The selected Wand shortcut help describes the wrong operation')
    capture('07c_wand_ctrl_remove')
    steps.append({'action': 'wand', 'add_button': 'left', 'remove_modifier': 'Ctrl',
                  'demonstration_only': True, 'added_pixels': int(changed.sum()),
                  'exact_original_labels_restored': True, 'acquired_image_unchanged': True,
                  'tolerance_percent': 0.5, 'max_pixels': 256, 'single_toolbar_button': True})
    QTest.mouseClick(screen._btn_undo, Qt.LeftButton)
    settle()
    if not np.array_equal(canvas.mask, added):
        raise RuntimeError('Undo did not restore the Wand addition')
    undo()
    number(screen._wand_pct, old_wand_pct)
    number(screen._wand_max, old_wand_max)

    expose(screen._norm_lo)
    before_display = display_digest()
    old_lower = screen._norm_lo.value()
    number(screen._norm_lo, 10)
    if display_digest() == before_display or not np.array_equal(canvas.mask, original) or not np.array_equal(canvas.image, pixels):
        raise RuntimeError('Display contrast must change the rendering but not labels or raw intensities')
    capture('08b_contrast_only')
    number(screen._norm_lo, old_lower)
    if display_digest() != before_display:
        raise RuntimeError('Restoring contrast did not restore the original display')
    # Item 511: the Filter category is a list of regionprop rows. Add an
    # area row through the visible property box and Add a filter, type its
    # minimum like a user, then press Filter; the ledger lists what it hid.
    threshold = int(np.median(counts[1:][counts[1:] > 0])) + 1
    filters = screen._filter_list
    expose(filters.property_box)
    index = filters.property_box.findText('area')
    if index < 0:
        raise RuntimeError('The Filter category does not offer area')
    filters.property_box.setCurrentIndex(index)
    settle()
    QTest.mouseClick(filters.add_button, Qt.LeftButton)
    settle()
    rows = filters.rows()
    if len(rows) != 1 or rows[0]['property'] != 'area':
        raise RuntimeError('Add a filter did not add one area row')
    low = rows[0]['min']
    expose(low)
    QTest.mouseClick(low, Qt.LeftButton)
    QTest.keyClicks(low, str(threshold))
    QTest.keyClick(low, Qt.Key_Tab)
    settle()
    expose(screen._btn_filter)
    QTest.mouseClick(screen._btn_filter, Qt.LeftButton)
    settle()
    filtered_count = int(np.count_nonzero(np.unique(canvas.mask)))
    if not 0 < filtered_count < original_count:
        raise RuntimeError('The area filter did not hide some real objects')
    ledger = screen._filter_log.toPlainText().strip().splitlines()
    if not ledger:
        raise RuntimeError('The filter ledger does not list the hidden objects')
    expose(screen._filter_log)
    capture('08c_filter_edits_labels')
    steps.append({'action': 'area_filter_row', 'minimum_area': threshold,
                  'before': original_count, 'after': filtered_count,
                  'ledger_rows': len(ledger), 'model_rerun': False})
    expose(rows[0]['remove'])
    QTest.mouseClick(rows[0]['remove'], Qt.LeftButton)
    settle()
    if filters.rows() or not np.array_equal(canvas.mask, original):
        raise RuntimeError('Removing the filter row did not bring every hidden object back')
    capture('08d_filter_removed')

    for key in FOLD_ORDER:
        button = screen._folds.button_for(key)
        if button is None or not button.isVisible():
            raise RuntimeError(f'Missing visible folded route: {key}')
        QTest.mouseMove(button)
        settle(0.7)
        capture('09_fold_' + key)

    QTest.mouseClick(screen._btn_save, Qt.LeftButton)
    settle()
    saved = tifffile.imread(evidence[0]['mask'])
    if not np.array_equal(saved > 0, original > 0):
        raise RuntimeError('Saving the restored mask changed its foreground geometry')
    capture('11_saved_restored_mask')

    # Recrop is a different kind of action: it writes a child immediately
    # and archives this PRIVATE copied parent when Next is pressed.
    parent_name = screen._image_files[screen._current_index]
    margin = 12
    box = (max(0, int(xx.min())-margin), max(0, int(yy.min())-margin),
           min(original.shape[1]-1, int(xx.max())+margin),
           min(original.shape[0]-1, int(yy.max())+margin))
    mode('recrop')
    gesture([(box[0], box[1]), (box[2], box[3])])
    if len(screen._recrop_children) != 1:
        raise RuntimeError('The actual Recrop gesture did not write one child field')
    child = folder / screen._recrop_children[0]
    x0, y0, x1, y1 = map(int, canvas.recrop_boxes[0][:4])
    child_image = tifffile.imread(child)
    child_mask = tifffile.imread(folder / 'masks' / child.name)
    if not np.array_equal(child_image, pixels[y0:y1, x0:x1]) or not np.any(child_mask):
        raise RuntimeError('Recrop did not preserve image pixels and complete labelled objects')
    capture('12_recrop_written')
    recrop = {'child': str(child), 'box': [x0, y0, x1, y1],
              'image_sha256': digest(child),
              'child_objects': int(np.count_nonzero(np.unique(child_mask)))}
    QTest.mouseClick(screen._btn_next, Qt.LeftButton)
    ready()
    archive = folder / 'recropped_originals' / parent_name
    if not archive.is_file() or digest(archive) != evidence[0]['image_sha256']:
        raise RuntimeError('Recrop did not preserve the parent image in its recovery archive')
    if parent_name in screen._image_files or screen._image_files[screen._current_index] != child.name:
        raise RuntimeError('The editor did not replace the copied parent with its child in the queue')
    capture('13_child_and_recovery_archive')
    recrop['parent_recovery_path'] = str(archive)

    inference = None
    if detect:
        # Cellpose's settings show once Detection method is Cellpose.
        expose(screen._mag_mode)
        chosen = screen._mag_mode.findData('cellpose')
        if chosen < 0:
            raise RuntimeError('Detection method does not offer Cellpose')
        screen._mag_mode.setCurrentIndex(chosen)
        settle()
        expose(screen._cp_diameter)
        capture('14_cellpose_settings')
        initial_child_mask = canvas.mask.copy()
        QTest.mouseClick(screen._btn_cellpose, Qt.LeftButton)
        settle()
        # Detection runs on a worker thread; wait for the real result.
        deadline = time.monotonic() + timeout
        while screen._detection_request is not None:
            if time.monotonic() >= deadline:
                raise TimeoutError('Object detection did not finish')
            settle(0.2)
        settle()
        if not screen._prob_pane.has_image() or not screen._flow_pane.has_image():
            raise RuntimeError('Real Cellpose inference did not produce its two intermediate views')
        if np.array_equal(canvas.mask, initial_child_mask):
            raise RuntimeError('Cellpose did not change the example mask; do not claim a new detection')
        inferred_objects = int(np.count_nonzero(np.unique(canvas.mask)))
        if inferred_objects == 0:
            raise RuntimeError('Cellpose did not produce a usable example detection')
        capture('15_cellpose_detected')
        for index, name in [(screen._tab_prob, '16_cell_probability'),
                            (screen._tab_flow, '17_flows'), (0, '18_mask_again')]:
            tabs = screen._view_tabs
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton, pos=tabs.tabBar().tabRect(index).center())
            settle()
            capture(name)
        inference = {'model': screen._cp_model.currentData(),
                     'objects': inferred_objects, 'status': screen._status_label.text(),
                     'input_shape': list(child_image.shape),
                     'cell_probability': screen._cp_cellprob.value(),
                     'flow_threshold': screen._cp_flow.value(),
                     'diameter': screen._cp_diameter.value(),
                     'normalize': screen._cp_normalize.isChecked(),
                     'mode': screen._combine_mode.currentData()}
        QTest.mouseClick(screen._btn_undo, Qt.LeftButton)
        settle()
        if not np.array_equal(canvas.mask, initial_child_mask):
            raise RuntimeError('Undo did not restore the pre-inference child labels')
        capture('19_detection_undone')

    if not np.array_equal(canvas.image, child_image):
        raise RuntimeError('Editing masks changed source image pixels')
    more = {}
    if curation_organize:
        import capture_make_masks_more as extra

        extra.record_curation(app, window, screen, captures, capture, settle, write_json, timeout)
        extra.record_upload(app, window, screen, captures, capture, settle, write_json)
        sources = [Path(row['source']) for row in evidence]
        _parent, by_well, export, masks, rows = extra.prepare_nested(stage, sources)
        more['consolidation'] = extra.record_consolidation(
            app, window, screen, stage, by_well, captures, capture, settle, write_json, timeout)
        extra.record_organize(app, window, screen, export, masks, captures, capture,
                              settle, write_json, timeout)
        more['organize_inputs'] = rows
    if any(digest(Path(row['source'])) != row['source_sha256'] or
           digest(Path(row['image']) if Path(row['image']).is_file() else
                  folder / 'recropped_originals' / Path(row['image']).name) != row['image_sha256']
           for row in evidence):
        raise RuntimeError('Editing the private mask copies changed original image data')
    write_json(captures / 'editor_acceptance.json', {
        'accepted': True, 'input_folder': str(folder), 'initial_objects': original_count,
        'steps': steps, 'reversible_label_edits_undone_before_recrop': True,
        'saved_original_foreground_unchanged': True, 'original_images_unchanged': True,
        'folded_routes_shown': list(FOLD_ORDER), 'model_inference_requested': detect,
        'model_inference': inference,
        'recrop_recorded': True, 'recrop': recrop, 'curation_organize': more})
