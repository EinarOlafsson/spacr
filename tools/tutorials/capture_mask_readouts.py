"""Capture the Live magnifier through real CPU-only gestures (Otsu, region under the mouse)."""
import time


def record_readouts(app, window, screen, captures, capture, settle, write_json, timeout):
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QScrollArea

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
        if scroll.isAncestorOf(screen._mag_scope):
            scroll.ensureWidgetVisible(screen._mag_scope, 50, 400)
    settle()
    if not screen._mag_scope.isVisible():
        raise RuntimeError('The Live magnifier settings are not visible')
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

    write_json(captures / 'readouts_acceptance.json', {
        'accepted': True, 'mode': 'otsu', 'scope': 'region',
        'model_inference_on_gpu': False,
        'image_and_labels_unchanged': True,
        'magnifier_result_type': type(shown).__name__,
        'magnifier_objects': int(shown.count),
    })
