"""Checksum outages never publish executable bytes or discard a prior download."""
from dataclasses import asdict
import hashlib
import io
import json
import os
import stat
import urllib.error
import urllib.request

import pytest

from spacr import install_cleanup as cleanup


PAYLOAD = b'#!/bin/sh\necho verified installer\n'
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
NAME = 'spaCR-1.5.1.5-Linux-x86_64-Online.run'
URL = f'{cleanup._RELEASE_DOWNLOAD}/v1.5.1.5/{NAME}'


def _serve(monkeypatch, *, sums=None, empty=False, broken_read=False, on_sums=None):
    class Response(io.BytesIO):
        def read(self, size=-1):
            if broken_read:
                raise OSError('interrupted installer stream')
            return super().read(size)

    def urlopen(request, timeout):
        if request.full_url.endswith(cleanup._SUMS_NAME):
            if on_sums:
                on_sums()
            if isinstance(sums, BaseException):
                raise sums
            return io.BytesIO(sums if sums is not None else f'{DIGEST}  {NAME}\n'.encode())
        return Response(b'' if empty else PAYLOAD)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)


def _target(tmp_path, existing):
    target = tmp_path / 'installer.run'
    if existing:
        target.write_bytes(b'previous verified installer')
        target.chmod(0o750)
    return target


def _unchanged(target, existing):
    if existing:
        assert target.read_bytes() == b'previous verified installer'
        if os.name != 'nt':
            assert stat.S_IMODE(target.stat().st_mode) == 0o750
    else:
        assert not target.exists()
    assert not list(target.parent.glob('.spacr-download-*'))


@pytest.mark.parametrize('failure', ['503', '403', '429', 'timeout', 'connection'])
@pytest.mark.parametrize('existing', [False, True])
def test_checksum_transport_failure_refuses_and_preserves_prior_download(tmp_path, monkeypatch, failure, existing):
    error = (urllib.error.HTTPError(URL, int(failure), 'checksum unavailable', None, None)
             if failure.isdecimal() else TimeoutError('checksum timed out') if failure == 'timeout'
             else urllib.error.URLError('checksum connection unavailable'))
    _serve(monkeypatch, sums=error)
    target = _target(tmp_path, existing)
    with pytest.raises(type(error)):
        cleanup._download(URL, str(target))
    _unchanged(target, existing)


@pytest.mark.parametrize('failure', ['empty', 'read', 'wrong-digest'])
@pytest.mark.parametrize('existing', [False, True])
def test_incomplete_or_wrong_bytes_never_replace_the_previous_download(tmp_path, monkeypatch, failure, existing):
    _serve(monkeypatch, empty=failure == 'empty', broken_read=failure == 'read',
           sums=b'0' * 64 + f'  {NAME}\n'.encode() if failure == 'wrong-digest' else None)
    target = _target(tmp_path, existing)
    with pytest.raises(OSError):
        cleanup._download(URL, str(target))
    _unchanged(target, existing)


@pytest.mark.parametrize('phase', ['chmod', 'replace'])
def test_failed_activation_preserves_previous_download_and_cleans_private_staging(tmp_path, monkeypatch, phase):
    _serve(monkeypatch)
    target = _target(tmp_path, True)
    def refuse(*args, **kwargs):
        raise PermissionError('activation refused')
    monkeypatch.setattr(cleanup.os, phase, refuse)
    with pytest.raises(PermissionError):
        cleanup._download(URL, str(target))
    _unchanged(target, True)


def test_validated_bytes_replace_the_previous_download_only_after_checksum_read(tmp_path, monkeypatch):
    target = _target(tmp_path, True)
    def still_previous():
        assert target.read_bytes() == b'previous verified installer'
        pending = list(tmp_path.glob('.spacr-download-*'))
        assert len(pending) == 1
        if os.name != 'nt':
            assert stat.S_IMODE(pending[0].stat().st_mode) == 0o600
    _serve(monkeypatch, on_sums=still_previous)
    cleanup._download(URL, str(target))
    assert target.read_bytes() == PAYLOAD
    if os.name != 'nt':
        assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert not list(tmp_path.glob('.spacr-download-*'))


def test_real_missing_legacy_checksum_remains_supported(tmp_path, monkeypatch):
    _serve(monkeypatch, sums=urllib.error.HTTPError(URL, 404, 'not published', None, None))
    target = _target(tmp_path, True)
    cleanup._download(URL, str(target))
    assert target.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob('.spacr-download-*'))


def test_download_does_not_follow_a_target_symlink_into_unrelated_user_files(tmp_path, monkeypatch):
    outside = tmp_path / 'retained model.bin'
    outside.write_bytes(b'original user model')
    target = tmp_path / 'installer.run'
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip('native filesystem does not allow disposable file symlinks')
    _serve(monkeypatch)
    cleanup._download(URL, str(target))
    assert outside.read_bytes() == b'original user model'
    assert not target.is_symlink()
    assert target.read_bytes() == PAYLOAD


def test_checksum_outage_stops_real_plan_before_install_retirement_or_relaunch(tmp_path, monkeypatch):
    _serve(monkeypatch, sums=urllib.error.HTTPError(URL, 503, 'outage', None, None))
    target = _target(tmp_path, True)
    application = tmp_path / 'working application'
    application.mkdir()
    previous = application / 'original-runtime'
    previous.write_bytes(b'working runtime')
    record = cleanup.InstallRecord(kind='installer', platform='linux', layout='linux-online',
                                   root=str(application), python=str(application / 'venv/bin/python'), running=True)
    plan = {'version': '1.5.1.5', 'pid': 123, 'records': [asdict(record)],
            'fetch': {'url': URL, 'path': str(target)}, 'install': ['never execute'],
            'relaunch': ['never reopen'], 'log': None}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan), encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail('No installer/retirement/relaunch after checksum transport failure')
    machine = cleanup._Machine(platform='linux', fs_root=str(tmp_path), environ={'HOME': str(tmp_path / 'home')})
    assert cleanup._run_plan(str(path), wait=lambda pid: True, run=forbidden,
                             remove=forbidden, spawn=forbidden, system=machine) == 4
    assert previous.read_bytes() == b'working runtime'
    _unchanged(target, True)
