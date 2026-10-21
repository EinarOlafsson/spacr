"""The live driver's waiter rejects stale identities despite an idle canvas."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from PySide6.QtCore import QTimer

DRIVER = Path(__file__).resolve().parents[1] / 'capture_make_masks.py'


def actual_waiter(screen, settle, clock=time, timeout=1.0):
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    tour = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'record_yolo_boxes')
    ready = next(node for node in tour.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'ready')
    namespace = dict(screen=screen, settle=settle, time=clock, timeout=timeout)
    exec(compile(ast.Module(body=[ready], type_ignores=[]), str(DRIVER), 'exec'), namespace)
    return namespace['ready']


@pytest.mark.parametrize('stale_state', ['previous_index', 'filename_after_index'])
def test_expected_field_waits_for_queued_identity_even_when_canvas_is_idle(qtbot, stale_state):
    screen = SimpleNamespace(_loading=False, _canvas=SimpleNamespace(image=object()),
                             _image_files=['first.tif', 'second.tif'], _current_index=0)
    if stale_state == 'filename_after_index':
        screen._current_index = 1
        screen._image_files[1] = 'first.tif'
    delivered = []

    def deliver():
        screen._current_index = 1
        screen._image_files[1] = 'second.tif'
        delivered.append(True)

    QTimer.singleShot(30, deliver)
    actual_waiter(screen, lambda seconds=0: qtbot.wait(max(1, round(seconds * 1000))))('second.tif')
    assert delivered == [True]
    assert screen._image_files[screen._current_index] == 'second.tif'


@pytest.mark.parametrize('not_ready', ['wrong_filename', 'loading', 'missing_image'])
def test_identity_wait_preserves_loading_image_and_timeout_guards(not_ready):
    screen = SimpleNamespace(_loading=not_ready == 'loading',
                             _canvas=SimpleNamespace(image=None if not_ready == 'missing_image' else object()),
                             _image_files=['wrong.tif' if not_ready == 'wrong_filename' else 'second.tif'],
                             _current_index=0)
    now = [0.0]

    def settle(seconds=0):
        now[0] += max(seconds, 0.01)

    waiter = actual_waiter(screen, settle, SimpleNamespace(monotonic=lambda: now[0]), timeout=0.2)
    with pytest.raises(TimeoutError, match='actual YOLO source image did not load'):
        waiter('second.tif')
