"""Record Model Compare's real nested screen and field loader, without inference."""
from pathlib import Path
import time

from capture_model_zoo import FIELDS, PAIRS, _fingerprint


def record_screen(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    import numpy as np
    import tifffile
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from spacr.qt.screens.model_compare import ModelCompareScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    deadline = time.monotonic() + timeout

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('The Model Compare introduction timed out')
        settle(0)

    originals = {str(FIELDS / (name + '.tif')): _fingerprint(FIELDS / (name + '.tif'), tick)
                 for name, *_ in PAIRS}
    for name, _, _, _, expected in PAIRS:
        if originals[str(FIELDS / (name + '.tif'))]['sha256'] != expected:
            raise ValueError('The recorded tutorial field changed')
    buttons = [b for b in screen.findChildren(FoldButton)
               if b.app_key == 'model_compare' and b.isVisible() and b.isEnabled()]
    if len(buttons) != 1:
        raise ValueError('Expected the actual nested Model Compare control')
    capture('01_make_masks_host')
    QTest.mouseClick(buttons[0], Qt.LeftButton)
    settle(.5)
    screens = [w for w in window.findChildren(ModelCompareScreen) if w.isVisible()]
    if len(screens) != 1:
        raise ValueError('The actual Model Compare page did not open')
    compare = screens[0]
    if compare._segment_fn is not None or compare.report() is not None:
        raise ValueError('Expected a fresh, unmodified comparison backend')
    capture('02_model_parameters')
    edit = compare._path_edit
    if not edit.isVisible() or not edit.isEnabled():
        raise ValueError('The actual source field is unavailable')
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(edit, str(FIELDS))
    QTest.keyClick(edit, Qt.Key_Tab)
    QTest.mouseClick(compare._btn_load, Qt.LeftButton)
    while compare._busy or compare.active_jobs():
        tick(); settle(.1)
    expected_names = [name for name, *_ in PAIRS]
    if compare.field_names() != expected_names or len(compare._images) != 3:
        raise ValueError('The actual field loader did not retain the three source identities')
    for name, loaded in zip(expected_names, compare._images):
        np.testing.assert_array_equal(loaded, tifffile.imread(FIELDS / (name + '.tif')))
    if not compare._btn_compare.isEnabled() or compare.report() is not None:
        raise ValueError('Loading fields must enable Compare without generating results')
    capture('03_real_fields_loaded_not_compared')
    for path, expected in originals.items():
        if _fingerprint(Path(path), tick) != expected:
            raise ValueError('A source image changed')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Actual nested GUI route, settings and real field loading only',
        'route': ['make_masks', 'model_compare'], 'fields': expected_names,
        'loaded_pixels_equal_originals': True, 'source_hashes': originals,
        'original_inputs_unchanged': True, 'gui_workflow_completed': False,
        'compare_clicked': False, 'inference_performed': False,
        'results_injected': False, 'app_source_modified': False, 'published': False})


if __name__ == '__main__':
    from capture_barcode_saved_plots import launch
    raise SystemExit(launch('model_compare', 'model_compare_1507_gui_v2', ['--model-compare-api-introduction'], 120))
