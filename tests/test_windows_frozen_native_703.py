"""Only the checksum-pinned installer receives Windows administrator consent."""
import ctypes
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as cleanup


class NativeCall:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(elevated=0, token_open=True, token_read=True,
                            token_size=4, launched=True, handle=321, wait=0,
                            exit_read=True, exit_code=0, error=1223,
                            requests=[], closed=[], waits=[])

    def open_token(_process, rights, output):
        assert rights == 0x0008
        output._obj.value = 123
        return state.token_open

    def token_information(_token, kind, output, capacity, written):
        assert kind == 20 and capacity == 4
        output._obj.value = state.elevated
        written._obj.value = state.token_size
        return state.token_read

    def execute(output):
        request = output._obj
        state.requests.append(request)
        request.hProcess = state.handle
        return state.launched

    def wait(handle, timeout):
        state.waits.append((handle, timeout))
        return state.wait

    def exit_code(_handle, output):
        output._obj.value = state.exit_code
        return state.exit_read

    kernel = SimpleNamespace(
        GetCurrentProcess=NativeCall(lambda: -1),
        CloseHandle=NativeCall(lambda handle: state.closed.append(
            handle.value if isinstance(handle, ctypes.c_void_p) else handle) or 1),
        WaitForSingleObject=NativeCall(wait),
        GetExitCodeProcess=NativeCall(exit_code),
        GetProcessId=NativeCall(lambda _handle: 456))
    security = SimpleNamespace(OpenProcessToken=NativeCall(open_token),
                               GetTokenInformation=NativeCall(token_information))
    shell = SimpleNamespace(ShellExecuteExW=NativeCall(execute))
    libraries = {'kernel32': kernel, 'advapi32': security, 'shell32': shell}
    monkeypatch.setattr(cleanup, '_windows_update_library', libraries.__getitem__)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: state.error, raising=False)
    return state


def test_elevation_is_read_from_the_actual_token_and_token_handle_is_closed(native):
    assert cleanup._windows_process_elevated() is False
    native.elevated = 1
    assert cleanup._windows_process_elevated() is True
    assert native.closed == [123, 123]


@pytest.mark.parametrize('failure', ['token_open', 'token_read', 'token_size'])
def test_unknown_token_status_refuses_instead_of_assuming_unelevated(native, failure):
    setattr(native, failure, 0)
    with pytest.raises(OSError):
        cleanup._windows_process_elevated()
    assert native.closed == ([] if failure == 'token_open' else [123])
    assert native.requests == []


def test_only_the_installer_is_elevated_and_destination_is_the_final_nsis_argument(native):
    installer = r'C:\Users\Alice\private update\spaCR-1.5.1.5-setup.exe'
    destination = r'C:\Program Files\spaCR'
    assert cleanup._windows_frozen_install(installer, destination, timeout=23) == (0, '')
    request, = native.requests
    assert request.lpVerb == 'runas'
    assert request.lpFile == installer
    assert request.lpParameters == '/S /D=' + destination
    assert request.lpDirectory == r'C:\Users\Alice\private update'
    assert request.fMask & 0x40
    assert request.fMask & 0x100
    assert native.waits == [(321, 23000)]
    assert native.closed == [123, 321]


def test_an_elevated_coordinator_is_refused_without_starting_an_installer(native):
    native.elevated = 1
    with pytest.raises(ValueError, match='without administrator'):
        cleanup._windows_frozen_install(r'C:\stage\setup.exe', r'C:\Program Files\spaCR')
    assert native.requests == []
    assert native.closed == [123]


def test_cancelled_uac_keeps_its_native_error_and_does_not_wait_or_relaunch(native):
    native.launched = False
    native.handle = None
    with pytest.raises(OSError) as caught:
        cleanup._windows_frozen_install(r'C:\stage\setup.exe', r'C:\Program Files\spaCR')
    assert caught.value.errno == 1223
    assert native.waits == []
    assert native.closed == [123]


@pytest.mark.parametrize('code', [1, 1223, 1603])
def test_completed_installer_failure_is_not_reclassified_as_success(native, code):
    native.exit_code = code
    assert cleanup._windows_frozen_install(r'C:\stage\setup.exe', r'C:\spaCR') == (code, '')
    assert native.closed[-1] == 321


@pytest.mark.parametrize('change', ['timeout', 'wait_failed', 'missing_exit', 'still_active', 'no_handle'])
def test_every_uncertain_child_state_is_distinct_from_a_completed_failure(native, change):
    if change == 'timeout':
        native.wait = 258
    elif change == 'wait_failed':
        native.wait = 0xFFFFFFFF
    elif change == 'missing_exit':
        native.exit_read = False
    elif change == 'still_active':
        native.exit_code = 259
    else:
        native.handle = None
    with pytest.raises(TimeoutError) as caught:
        cleanup._windows_frozen_install(r'C:\stage\setup.exe', r'C:\spaCR')
    if change != 'no_handle':
        assert caught.value.pid == 456
        assert native.closed[-1] == 321
    else:
        assert native.closed == [123]


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), True, 0xFFFFFFFF / 1000])
def test_invalid_wait_budget_refuses_before_native_launch(native, timeout):
    with pytest.raises(ValueError):
        cleanup._windows_frozen_install(r'C:\stage\setup.exe', r'C:\spaCR', timeout=timeout)
    assert native.requests == []


@pytest.mark.parametrize('installer,destination', [
    ('setup.exe', r'C:\spaCR'),
    (r'\stage\setup.exe', r'C:\spaCR'),
    (r'C:\setup.exe', r'\\server\share\spaCR'),
    (r'C:\setup.exe', 'spaCR'),
    ('C:\\stage\\setup.exe\n/S', r'C:\spaCR'),
    (r'C:\setup.exe', 'C:\\spaCR" /S'),
    ('C:\\setup.exe\x00extra', r'C:\spaCR'),
])
def test_relative_or_command_injecting_paths_never_reach_shell_execute(native, installer, destination):
    with pytest.raises(ValueError):
        cleanup._windows_frozen_install(installer, destination)
    assert native.requests == []


def test_native_updater_libraries_are_not_loaded_on_other_hosts(monkeypatch):
    monkeypatch.setattr(cleanup, 'os', SimpleNamespace(name='posix'))

    def forbidden(*_args, **_kwargs):
        raise AssertionError('A foreign host must not enter a Windows native API')

    monkeypatch.setattr(ctypes, 'WinDLL', forbidden, raising=False)
    with pytest.raises(OSError, match='native Windows updater'):
        cleanup._windows_update_library('kernel32')


def test_native_library_preserves_win32_last_error_for_uac_cancellation(monkeypatch):
    monkeypatch.setattr(cleanup, 'os', SimpleNamespace(name='nt'))
    loaded = []
    library = object()

    def load(name, **options):
        loaded.append((name, options))
        return library

    monkeypatch.setattr(ctypes, 'WinDLL', load, raising=False)
    assert cleanup._windows_update_library('shell32') is library
    assert loaded == [('shell32', {'use_last_error': True})]


def test_prepared_update_passes_only_private_descriptor_and_pid_before_final_destination(native):
    installer = r'C:\stage\spaCR-1.5.1.5-update.exe'
    descriptor = r'C:\Users\Alice\private stage\phase-descriptor'
    assert cleanup._windows_frozen_install(installer, r'C:\Program Files\spaCR',
        phase_channel=descriptor, desktop_pid=4242) == (0, '')
    request, = native.requests
    assert request.lpParameters == '/S /SPACRUPDATE="' + descriptor + r'" /SPACRPID=4242 /D=C:\Program Files\spaCR'
    assert native.closed == [123, 321]


def test_fixed_console_helper_can_request_uac_without_showing_a_terminal(native):
    assert cleanup._windows_authorized_process(r'C:\Program Files\spaCR\_internal\spacr-update-helper.exe',
        'uninstall-windows-root --phase-channel "C:\\private\\phase"', visible=False) == (0, '')
    request, = native.requests
    assert request.lpVerb == 'runas' and request.nShow == 0
    assert native.closed == [123, 321]
