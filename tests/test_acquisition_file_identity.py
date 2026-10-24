"""Acquisition identity uses Windows change times without leaking file handles."""
import ctypes
from types import SimpleNamespace
import sys

import pytest


class NativeCall:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


@pytest.fixture
def windows_file_api(monkeypatch):
    # spacr.core imports torch, and on Windows torch's import calls the real
    # kernel32 through ctypes.WinDLL; load it before WinDLL is faked so a
    # test run on its own does not hand torch the fake.
    import spacr.core  # noqa: F401

    state = {'handle': 73, 'readable': True, 'change': 987654321}

    def information(handle, kind, pointer, size):
        assert kind == 0
        pointer._obj.change = state['change']
        return state['readable']

    api = SimpleNamespace(
        CreateFileW=NativeCall(lambda *args: state['handle']),
        GetFileInformationByHandleEx=NativeCall(information),
        CloseHandle=NativeCall(lambda handle: True))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: api, raising=False)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(get_osfhandle=lambda descriptor: descriptor + 100))
    return state, api


@pytest.mark.parametrize('readable', [True, False])
def test_windows_change_time_closes_a_handle_it_opened(tmp_path, windows_file_api, readable):
    from spacr.core import _watch_windows_change_ns

    state, api = windows_file_api
    state['readable'] = readable
    path = tmp_path / 'acquisition.tif'
    result = _watch_windows_change_ns(path=path)
    assert result == (987654321 if readable else None)
    assert api.CloseHandle.calls == [(73,)]
    opened = api.CreateFileW.calls[0]
    assert opened[0] == str(path)
    assert opened[1:5] == (0x80, 0x7, None, 3)
    assert opened[5] == 0x02000000 | 0x00200000


@pytest.mark.parametrize('handle', [None, ctypes.c_void_p(-1).value])
def test_unopenable_windows_input_has_no_change_time(tmp_path, windows_file_api, handle):
    from spacr.core import _watch_windows_change_ns

    state, api = windows_file_api
    state['handle'] = handle
    assert _watch_windows_change_ns(path=tmp_path / 'missing.tif') is None
    assert api.GetFileInformationByHandleEx.calls == []
    assert api.CloseHandle.calls == []


def test_windows_descriptor_identity_does_not_close_the_callers_handle(windows_file_api):
    from spacr.core import _watch_windows_change_ns

    state, api = windows_file_api
    assert _watch_windows_change_ns(descriptor=8) == state['change']
    assert api.GetFileInformationByHandleEx.calls[0][0] == 108
    assert not api.CreateFileW.calls
    assert not api.CloseHandle.calls


def test_unreadable_windows_descriptor_falls_back_without_a_native_call(windows_file_api, monkeypatch):
    from spacr.core import _watch_windows_change_ns

    state, api = windows_file_api

    def closed(descriptor):
        raise OSError('descriptor is closed')

    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(get_osfhandle=closed))
    assert _watch_windows_change_ns(descriptor=8) is None
    assert not api.GetFileInformationByHandleEx.calls
    assert not api.CloseHandle.calls


@pytest.mark.parametrize('native, expected', [(None, 55), (987654321, 987654321)])
def test_windows_stat_identity_uses_change_time_when_available(monkeypatch, native, expected):
    from spacr import core

    info = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=55)
    calls = []

    def change_time(**kwargs):
        calls.append(kwargs)
        return native

    monkeypatch.setattr(core, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(core, '_watch_windows_change_ns', change_time)
    assert core._watch_stat_identity(info, descriptor=8) == [1, 2, 3, 4, expected]
    assert calls == [{'path': None, 'descriptor': 8}]
