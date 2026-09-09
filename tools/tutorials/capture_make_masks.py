"""Record genuine mask-editor gestures on isolated, real downloaded fields.

Preparing copies extracts exact image/label planes. Only the application's
visible gestures and buttons edit masks, save them or recrop them.
"""
from __future__ import annotations

import hashlib
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


def record_editor(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    import numpy as np
    import tifffile
    from scipy.ndimage import distance_transform_edt
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox, QScrollArea
    from spacr.qt.screens.make_masks import FOLD_ORDER

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
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            edit.setFocus()
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(folder))
            capture('02_folder_picker')
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    def reject_stalled():
        dialog = app.activeModalWidget()
        if dialog is not None and not accepted:
            errors.append('Folder picker did not accept the real path')
            dialog.reject()

    QTimer.singleShot(500, choose)
    QTimer.singleShot(12000, reject_stalled)
    QTest.mouseClick(screen._btn_open, Qt.LeftButton)
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

    def gesture(points):
        positions = [canvas._image_to_canvas(x, y) for x, y in points]
        if any(p is None or not canvas.rect().contains(p) or
               canvas._canvas_to_image(p.x(), p.y()) is None for p in positions):
            raise RuntimeError('The intended gesture is outside the visible image')
        QTest.mouseMove(canvas, positions[0])
        settle(0.08)
        QTest.mousePress(canvas, Qt.LeftButton, pos=positions[0])
        settle(0.08)
        for position in positions[1:]:
            QTest.mouseMove(canvas, position, delay=40)
            settle(0.08)
        if canvas.mode == 'divide':
            write_json(captures / 'divide_gesture.json', {
                'requested_points': [[int(x), int(y)] for x, y in points],
                'observed_points': [list(canvas._canvas_to_image(p.x(), p.y()))
                                    for p in canvas._gesture_points]})
            capture('08_divide_gesture')
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=positions[-1])
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
    undo()

    for key in FOLD_ORDER:
        button = screen._folds.button_for(key)
        if button is None or not button.isVisible():
            raise RuntimeError(f'Missing visible folded route: {key}')
        QTest.mouseMove(button)
        settle(0.7)
        capture('09_fold_' + key)

    # Scroll the actual settings panel; no hidden control is operated.
    if not screen._btn_settings.isChecked():
        QTest.mouseClick(screen._btn_settings, Qt.LeftButton)
    for scroll in screen.findChildren(QScrollArea):
        if scroll.isAncestorOf(screen._cp_model):
            scroll.ensureWidgetVisible(screen._cp_model)
    settle()
    capture('10_model_controls_not_run')
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

    if not np.array_equal(canvas.image, child_image):
        raise RuntimeError('Editing masks changed source image pixels')
    if any(digest(Path(row['source'])) != row['source_sha256'] or
           digest(Path(row['image']) if Path(row['image']).is_file() else
                  folder / 'recropped_originals' / Path(row['image']).name) != row['image_sha256']
           for row in evidence):
        raise RuntimeError('Editing the private mask copies changed original image data')
    write_json(captures / 'editor_acceptance.json', {
        'accepted': True, 'input_folder': str(folder), 'initial_objects': original_count,
        'steps': steps, 'reversible_label_edits_undone_before_recrop': True,
        'saved_original_foreground_unchanged': True, 'original_images_unchanged': True,
        'folded_routes_shown': list(FOLD_ORDER), 'model_inference_requested': False,
        'recrop_recorded': True, 'recrop': recrop})
