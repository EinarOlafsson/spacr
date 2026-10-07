"""Real zstd diagnostics preserve process identity and bounded scratch files."""

import os
import shutil
import struct
import subprocess
import time

import pytest

from tools import collect_qt_native_backtrace as collector


@pytest.fixture
def compressed_core(tmp_path):
    decoder = shutil.which('zstd')
    if decoder is None:
        pytest.skip('zstd is not installed on this host')
    reports = tmp_path / 'reports'
    scratch = tmp_path / 'scratch'
    reports.mkdir()
    scratch.mkdir()
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    header[16:18] = struct.pack('<H', 4)
    raw = bytes(header) + os.urandom(256 * 1024)
    packed = subprocess.run([decoder, '--compress', '--stdout', '--quiet'],
                            input=raw, capture_output=True, check=True).stdout
    path = reports / 'core.python.1001.bootid.3456.1791382744.zst'
    path.write_bytes(packed)
    return reports, scratch, path, raw


def test_real_systemd_core_is_extracted_without_changing_original(compressed_core):
    reports, scratch, path, raw = compressed_core
    before = path.read_bytes()
    lines = []
    extracted = collector._extract_systemd_core(reports, scratch, 3456, time.time_ns(), lines)
    try:
        assert extracted is not None
        assert extracted.read_bytes() == raw
        assert path.read_bytes() == before
        assert any('exit=0' in line for line in lines)
    finally:
        if extracted is not None:
            extracted.unlink()
    assert not list(scratch.iterdir())


@pytest.mark.parametrize('defect', ['wrong_pid', 'uid_matches_pid', 'stale', 'symlink',
                                   'bad_frame', 'not_elf_core', 'oversize_output'])
def test_unrelated_or_invalid_core_never_survives_in_scratch(compressed_core, monkeypatch, defect):
    reports, scratch, path, raw = compressed_core
    pid = 3456
    if defect == 'wrong_pid':
        pid = 7890
    elif defect == 'uid_matches_pid':
        pid = 1001
    elif defect == 'stale':
        stamp = time.time() - 3600
        os.utime(path, (stamp, stamp))
    elif defect == 'symlink':
        target = reports / 'hidden.zst'
        path.rename(target)
        path.symlink_to(target)
    elif defect == 'bad_frame':
        path.write_bytes(b'not a zstd frame')
    elif defect == 'not_elf_core':
        packed = subprocess.run([shutil.which('zstd'), '-cq'], input=b'plain text',
                                capture_output=True, check=True).stdout
        path.write_bytes(packed)
    elif defect == 'oversize_output':
        packed = subprocess.run([shutil.which('zstd'), '-cq'], input=raw[:64] + bytes(512 * 1024),
                                capture_output=True, check=True).stdout
        path.write_bytes(packed)
        monkeypatch.setattr(collector, 'MAX_CORE_BYTES', 128 * 1024)
    lines = []
    assert collector._extract_systemd_core(reports, scratch, pid, time.time_ns(), lines) is None
    assert not list(scratch.iterdir())
