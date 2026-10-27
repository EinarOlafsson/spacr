from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_saved_plots import check_saved_file_run
from barcode_qc_evidence import check_native_run


def actual_scope():
    return dict(outcome=dict(finished=True, ok=True, errors=[]), gui_figure_count=0,
                figures_card_visible=False, settings_errors=[
                    '[settings] ERROR [src]: src is missing from the settings.\n'])


def test_saved_file_scope_does_not_certify_the_broken_gui():
    run = actual_scope()
    check_saved_file_run(run)
    with pytest.raises(ValueError, match='visible in the GUI'):
        check_native_run(run)
    run.update(gui_figure_count=2, figures_card_visible=True, settings_errors=[])
    check_native_run(run)
    check_saved_file_run(run)


@pytest.mark.parametrize('key,value', [('finished', False), ('ok', False), ('errors', ['failed'])])
def test_failed_worker_is_never_accepted(key, value):
    run = actual_scope(); run['outcome'][key] = value
    with pytest.raises(ValueError, match='worker'): check_saved_file_run(run)


@pytest.mark.parametrize('errors', [['another error'], [
    '[settings] ERROR [src]: src is missing from the settings.', 'another error']])
def test_only_the_exact_diagnosed_preflight_error_is_allowed(errors):
    run = actual_scope(); run['settings_errors'] = errors
    with pytest.raises(ValueError, match='Unexpected settings'): check_saved_file_run(run)


@pytest.mark.parametrize('fault', [None, 'dimensions', 'fullscreen', 'small_width', 'small_height', 'screen'])
def test_activation_grid_pan_requires_measured_fullscreen_geometry(fault):
    from capture_saved_plots import _inspection_pan
    dimensions, screen = [4800, 600], [0, 0, 2560, 1440]
    window = dict(rect=list(screen), fullscreen=True)
    if fault == 'dimensions': dimensions = [2400, 600]
    if fault == 'fullscreen': window['fullscreen'] = False
    if fault == 'small_width': window['rect'][2] = screen[2] = 2300
    if fault == 'small_height': window['rect'][3] = screen[3] = 800
    if fault == 'screen': window['rect'][0] = 50
    if fault:
        with pytest.raises(ValueError): _inspection_pan(dimensions, window, screen)
    else:
        pan = _inspection_pan(dimensions, window, screen)
        assert pan['start'] == [1280, 1360] and pan['end'] == [683, 1360]
        assert pan['grabpoint'] == [1280, 1436]
        assert pan['kind'] == 'horizontal_scrollbar'
        assert pan['thumb_width'] == pytest.approx(2560 ** 2 / 4800)
        assert pan['expected_image_displacement'] == 1120
        large = _inspection_pan(dimensions, dict(rect=[0, 0, 3840, 2160], fullscreen=True), [0, 0, 3840, 2160])
        assert large['start'] == [1920, 2080] and large['end'] == [1536, 2080]
        assert large['grabpoint'] == [1920, 2156]
        assert large['expected_image_displacement'] == 480
        assert pan['duration'] == 6 and pan['image_viewport_verified'] is False


@pytest.mark.parametrize('fault', [None, 'pid', 'title', 'fullscreen'])
def test_activation_grid_window_queries_actual_owner_and_state(monkeypatch, fault):
    import capture_saved_plots as driver
    monkeypatch.setattr(driver, '_inspection_display', lambda stage: ':105')
    monkeypatch.setattr(driver, '_inspection_process', lambda stage, pid, record: dict(png=record['path']))
    def query(args, **kwargs):
        if args[1] == '-root': return '_NET_ACTIVE_WINDOW(WINDOW): window id # 0x123'
        if args[0] == 'xprop':
            pid = 999 if fault == 'pid' else 123
            title = 'other.png' if fault == 'title' else 'batch_0_grid.png'
            state = '' if fault == 'fullscreen' else '_NET_WM_STATE_FULLSCREEN'
            return f'_NET_WM_PID(CARDINAL) = {pid}\n_NET_WM_NAME(UTF8_STRING) = "{title}"\n_NET_WM_STATE(ATOM) = {state}\n'
        return 'Absolute upper-left X: 0\nAbsolute upper-left Y: 0\nWidth: 2560\nHeight: 1440\nMap State: IsViewable\n'
    monkeypatch.setattr(driver.subprocess, 'check_output', query)
    if fault:
        with pytest.raises(RuntimeError): driver._inspection_window(Path('/private'), 123, dict(path='/private/batch_0_grid.png'), fullscreen=True)
    else:
        actual = driver._inspection_window(Path('/private'), 123, dict(path='/private/batch_0_grid.png'), fullscreen=True)
        assert actual['rect'] == [0, 0, 2560, 1440]
        assert actual['image_viewport_verified'] is False


@pytest.mark.parametrize('fault', [None, 'pan'])
def test_activation_grid_uses_one_native_documented_zoom_and_pan(monkeypatch, fault):
    from types import SimpleNamespace
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication
    import capture_saved_plots as driver
    import capture_ui
    calls = []
    def call(*args, **kwargs):
        calls.append((args, kwargs))
        if fault == 'pan' and args == ('move_to', 683, 1360):
            raise RuntimeError('native move failed')
    adapter = SimpleNamespace(call=call, note=lambda *a, **kw: None)
    helper = SimpleNamespace(host_of=lambda capture: SimpleNamespace(adapter=adapter))
    monkeypatch.setattr(capture_ui, '_h', lambda: helper)
    monkeypatch.setattr(QApplication, 'screens', lambda: [SimpleNamespace(geometry=lambda: QRect(0, 0, 2560, 1440))])
    windows = iter([dict(window_id='0x123', rect=[50, 50, 2200, 350], fullscreen=False),
                    dict(window_id='0x123', rect=[50, 50, 2200, 350], fullscreen=False),
                    dict(window_id='0x123', rect=[0, 0, 2560, 1440], fullscreen=True),
                    dict(window_id='0x123', rect=[0, 0, 2560, 1440], fullscreen=True)])
    monkeypatch.setattr(driver, '_find_inspection_window', lambda *a, **kw: next(windows))
    monkeypatch.setattr(driver, '_inspection_window', lambda *a, **kw: next(windows))
    def inspect():
        return driver._inspect_activation_grid(Path('/private'), SimpleNamespace(host=object()), SimpleNamespace(pid=123),
                    dict(path='/private/batch_0_grid.png', dimensions=[4800, 600]),
                    lambda seconds: calls.append((('settle', seconds), {})))
    if fault:
        with pytest.raises(RuntimeError, match='native move failed'): inspect()
        assert calls[-1] == (('button_up', 'left'), {})
    else:
        proof = inspect()
        assert proof['native_frame_review_required'] is True
        assert proof['png_modified'] is False and proof['repeated_motion'] is False
    expected = [(('click_at', 1150, 66), {}), (('settle', .4), {}),
                (('press', '#keysym:0xffc8'), {}), (('settle', 1.0), {}),
                (('press', '1'), {}), (('settle', .6), {}),
                (('move_to', 1280, 1436), {}), (('settle', .6), {}),
                (('button_down', 'left'), {}), (('move_to', 1280, 1360), dict(duration=.35)),
                (('move_to', 683, 1360), dict(duration=6.0)), (('button_up', 'left'), {})]
    if not fault: expected.append((('settle', .6), {}))
    assert calls == expected
    from human_input import _keysym
    assert _keysym('#keysym:0xffc8') == 0xffc8



@pytest.mark.parametrize('fault', [None, 'runtime', 'root_display'])
def test_activation_grid_private_display_one_requires_owned_stage_runtime(tmp_path_factory, monkeypatch, fault):
    import socket
    import capture_saved_plots as driver
    stage = tmp_path_factory.mktemp('eog') / 's'
    runtime = stage / 'gnome-host' / 'run'
    runtime.mkdir(parents=True)
    wayland = socket.socket(socket.AF_UNIX)
    wayland.bind(str(runtime / 'w'))
    monkeypatch.setenv('SPACR_CAPTURE_HOST', str(stage / 'gnome-host'))
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(runtime))
    monkeypatch.setenv('WAYLAND_DISPLAY', 'w')
    monkeypatch.setenv('DISPLAY', ':0' if fault == 'root_display' else ':1')
    if fault == 'runtime':
        other = stage / 'other'
        other.mkdir()
        monkeypatch.setenv('XDG_RUNTIME_DIR', str(other))
    try:
        if fault:
            with pytest.raises(RuntimeError, match='owned private'):
                driver._inspection_display(stage)
        else:
            assert driver._inspection_display(stage) == ':1'
    finally:
        wayland.close()


@pytest.mark.parametrize('wrong_owner', [False, True])
def test_activation_grid_discovers_inactive_owned_viewer_without_waiving_focus(monkeypatch, wrong_owner):
    import capture_saved_plots as driver
    monkeypatch.setattr(driver, '_inspection_display', lambda stage: ':1')
    monkeypatch.setattr(driver, '_inspection_process', lambda stage, pid, record: dict(png=record['path']))
    def query(args, **kwargs):
        if args[1] == '-root':
            if args[2] == '_NET_ACTIVE_WINDOW':
                return '_NET_ACTIVE_WINDOW(WINDOW): window id # 0x111'
            return '_NET_CLIENT_LIST_STACKING(WINDOW): window id # 0x111, 0x222\n_NET_CLIENT_LIST(WINDOW): window id # 0x111, 0x222'
        if args[0] == 'xprop':
            viewer = args[2] == '0x222'
            pid = 123 if viewer and not wrong_owner else 999
            title = 'batch_0_grid.png' if viewer else 'spaCR'
            return f'_NET_WM_PID(CARDINAL) = {pid}\n_NET_WM_NAME(UTF8_STRING) = "{title}"\n_NET_WM_STATE(ATOM) = \n'
        return 'Absolute upper-left X: 50\nAbsolute upper-left Y: 50\nWidth: 2200\nHeight: 350\nMap State: IsViewable\n'
    monkeypatch.setattr(driver.subprocess, 'check_output', query)
    if wrong_owner:
        with pytest.raises(RuntimeError, match='expected_pid=123'):
            driver._find_inspection_window(Path('/private'), 123, dict(path='/private/batch_0_grid.png'))
    else:
        actual = driver._find_inspection_window(Path('/private'), 123, dict(path='/private/batch_0_grid.png'))
        assert actual['window_id'] == '0x222' and actual['active_window_id'] == '0x111'
        with pytest.raises(RuntimeError, match='Owned viewer is not active') as error:
            driver._inspection_window(Path('/private'), 123, dict(path='/private/batch_0_grid.png'), window_id='0x222')
        assert 'wid=0x222 active=0x111 properties=' in str(error.value)


@pytest.mark.parametrize('fault', [None, 'wrong_png', 'wrong_uid', 'wrong_x_pid', 'changed_png'])
def test_activation_grid_generic_title_requires_exact_process_and_png(tmp_path, monkeypatch, fault):
    import hashlib
    import os
    import capture_saved_plots as driver
    stage = tmp_path / 's'
    runtime = stage / 'gnome-host' / 'run'
    runtime.mkdir(parents=True)
    png = stage / 'batch_0_grid.png'
    png.write_bytes(b'original verified PNG fixture bytes')
    record = dict(path=str(png), sha256=hashlib.sha256(png.read_bytes()).hexdigest())
    if fault == 'changed_png': png.write_bytes(b'changed PNG bytes')
    proc = tmp_path / 'proc' / '123'
    proc.mkdir(parents=True)
    argument = str(stage / 'different.png') if fault == 'wrong_png' else str(png)
    (proc / 'cmdline').write_bytes(('eog\0--new-instance\0' + argument + '\0').encode())
    (proc / 'environ').write_bytes(('DISPLAY=:1\0SPACR_CAPTURE_HOST=' + str(stage / 'gnome-host')
        + '\0XDG_RUNTIME_DIR=' + str(runtime) + '\0WAYLAND_DISPLAY=w\0').encode())
    monkeypatch.setattr(driver, '_PROC', tmp_path / 'proc')
    monkeypatch.setattr(driver, '_inspection_display', lambda stage: ':1')
    monkeypatch.setenv('WAYLAND_DISPLAY', 'w')
    if fault == 'wrong_uid':
        actual_uid = os.getuid()
        monkeypatch.setattr(driver.os, 'getuid', lambda: actual_uid + 1)
    def query(args, **kwargs):
        if args[1] == '-root': return '_NET_ACTIVE_WINDOW(WINDOW): window id # 0xa00007'
        if args[0] == 'xprop':
            pid = 999 if fault == 'wrong_x_pid' else 123
            return f'_NET_WM_PID(CARDINAL) = {pid}\n_NET_WM_NAME(UTF8_STRING) = "Image Viewer"\nWM_NAME(STRING) = "Image Viewer"\n_NET_WM_STATE(ATOM) = _NET_WM_STATE_FULLSCREEN\n'
        return 'Absolute upper-left X: 0\nAbsolute upper-left Y: 0\nWidth: 2560\nHeight: 1440\nMap State: IsViewable\n'
    monkeypatch.setattr(driver.subprocess, 'check_output', query)
    if fault:
        with pytest.raises(RuntimeError): driver._inspection_window(stage, 123, record, fullscreen=True)
    else:
        actual = driver._inspection_window(stage, 123, record, fullscreen=True)
        assert actual['wm_titles'] == ['Image Viewer', 'Image Viewer']
        assert actual['expected_png'] == str(png)
        assert actual['process']['png_sha256'] == record['sha256']
        assert actual['image_viewport_verified'] is False
