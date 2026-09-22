"""Capture current mask readouts through real CPU-only gestures."""
import time


def record_readouts(app, window, screen, captures, capture, settle, write_json, timeout):
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QScrollArea

    from spacr.qt.screens.measure_inputs import MeasureInputsScreen

    canvas = screen._canvas
    original = canvas.mask.copy()
    pixels = canvas.image.copy()

    def until(predicate, description):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(description)
            settle(0.1)

    # These are the initial settings in the isolated preference store. Do not
    # silently select a model or invoke an invisible setting on a stale UI.
    if screen._mag_mode.currentData() != 'otsu' or screen._mag_scope.currentData() != 'region':
        raise RuntimeError('The readouts recording requires the default CPU Otsu region mode')
    for scroll in screen.findChildren(QScrollArea):
        if scroll.isAncestorOf(screen._mag_mode):
            scroll.ensureWidgetVisible(screen._mag_mode)
    settle()
    if not screen._mag_mode.isVisible():
        raise RuntimeError('The magnifier mode is not visible')
    capture('04_otsu_magnifier_settings')
    QTest.mouseClick(screen._btn_magnifier, Qt.LeftButton)
    position = canvas._image_to_canvas(pixels.shape[1] // 2, pixels.shape[0] // 2)
    if position is None or not canvas.rect().contains(position):
        raise RuntimeError('The real field center is outside the canvas')
    QTest.mouseMove(canvas, position, delay=100)
    magnifier = screen._magnifier
    until(lambda: magnifier._shown is not None and not magnifier.updating(),
          'The real Otsu magnifier did not return a current region')
    settle()
    shown = magnifier._shown
    if shown.mode != 'otsu' or shown.count < 1:
        raise RuntimeError('The real magnifier did not outline any Otsu objects')
    capture('05_live_otsu_magnifier')
    QTest.mouseClick(screen._btn_magnifier, Qt.LeftButton)
    settle()
    if not np.array_equal(canvas.mask, original) or not np.array_equal(canvas.image, pixels):
        raise RuntimeError('Previewing the magnifier changed source pixels or labels')

    QTest.mouseClick(screen._btn_features, Qt.LeftButton)
    until(lambda: any(isinstance(w, MeasureInputsScreen) and w.isVisible()
                      for w in app.topLevelWidgets()), 'FEATURES did not open')
    features = next(w for w in app.topLevelWidgets()
                    if isinstance(w, MeasureInputsScreen) and w.isVisible())
    try:
        features.resize(2400, 1500)
        features.move(window.pos().x() + 650, window.pos().y() + 350)
        until(lambda: not features.inputs.is_scanning() and len(features.inputs._known_paths) >= 2,
              'FEATURES did not load the real example folder')
        settle()
        capture('06_features_real_files', desktop=True)
        from capture_features_run import record_features_run

        measurement = record_features_run(
            app, features, captures, capture, settle, write_json, timeout)
        rows = len(features.inputs.table().rows)
        files = len(features.inputs._known_paths)
        unassigned = len(features.inputs._unassigned)
    finally:
        features.close()
        settle()
    write_json(captures / 'readouts_acceptance.json', {
        'accepted': True, 'mode': 'otsu', 'scope': 'region',
        'model_inference_on_gpu': False, 'features_rows': rows,
        'features_files': files, 'features_unassigned_files': unassigned,
        'features_scope': 'Visible assignment of ER images and companion cell masks, followed by the real Measure run',
        'features_measurement_run': True,
        'features_measured_cell_rows': measurement['measured_cell_rows'],
        'image_and_labels_unchanged': True,
        'magnifier_result_type': type(shown).__name__,
        'magnifier_objects': int(shown.count),
    })
