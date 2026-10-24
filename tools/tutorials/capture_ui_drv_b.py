"""Scene-driver helpers of the drv-b lesson group (17-19, 27).

``fill_setting`` is :func:`capture_ui.fill_setting` with a safety net: a
control that has no user route in the host (an unusual editor type, a value
the form cannot type) is set by code and recorded as a bypass, outside a clip,
instead of aborting the whole recording.
"""
from __future__ import annotations

import capture_ui


def fill_setting(capture, screen, key, value):
    try:
        return capture_ui.fill_setting(capture, screen, key, value)
    except RuntimeError as error:
        if not capture_ui.live(capture):
            raise
        capture_ui.code_action(capture, f'setting {key} has no user route: {error}')
        return screen._settings_model.set_value_for_key(key, value)


def splitter_to(capture, splitter, sizes, tolerance=40):
    """Give any splitter these sizes by dragging its handles one after another.

    Host: handle ``i`` is dragged so the panes before it add up to the wanted
    total; if the sizes still differ by more than ``tolerance`` px, they are
    set by code and recorded as a bypass (the gate then fails that clip).
    Elsewhere: ``setSizes`` as before.
    """
    sizes = [int(v) for v in sizes]
    if not capture_ui.live(capture):
        splitter.setSizes(sizes)
        return
    from PySide6.QtCore import Qt
    h = capture_ui._h()
    horizontal = splitter.orientation() == Qt.Horizontal
    for index in range(1, splitter.count()):
        current = splitter.sizes()
        delta = sum(sizes[:index]) - sum(current[:index])
        if abs(delta) < 4:
            continue
        handle = splitter.handle(index)
        if handle is None or not handle.isVisible():
            continue
        h.drag(capture, handle, handle.rect().center(), (delta, 0) if horizontal else (0, delta), duration=0.9)
    if any(abs(a - b) > tolerance for a, b in zip(splitter.sizes(), sizes)):
        capture_ui.code_action(capture, f'splitter sizes {splitter.sizes()} -> {sizes} by code')
        splitter.setSizes(sizes)
