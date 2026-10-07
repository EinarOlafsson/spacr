"""Real zstd diagnostics preserve process identity and bounded scratch files."""

import os
import shutil
import struct
import subprocess
import sys
import time

import pytest

pytest.importorskip('resource', reason='systemd core diagnostics require POSIX resource limits')

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


def _native_note_core(pid, executable, *, wide=True, order='<', leader_pid=None):
    """Build PRSTATUS/PRPSINFO/NT_FILE layouts without a large process core."""
    def note(kind, payload):
        name = b'CORE\0'
        return (struct.pack(order + 'III', len(name), len(payload), kind) + name
                + bytes((-len(name)) % 4) + payload + bytes((-len(payload)) % 4))

    word = 'Q' if wide else 'I'
    status = bytearray(112 if wide else 72)
    struct.pack_into(order + 'i', status, 32 if wide else 24, pid)
    mappings = (struct.pack(order + word * 5, 1, 4096, 4096, 8192, 0)
                + os.fsencode(executable) + b'\0')
    process = bytearray(136 if wide else 124)
    struct.pack_into(order + 'i', process, 24 if wide else 12,
                     pid if leader_pid is None else leader_pid)
    notes = note(1, status) + note(3, process) + note(0x46494C45, mappings)
    header = bytearray(64)
    header[:6] = b'\x7fELF' + bytes([2 if wide else 1, 1 if order == '<' else 2])
    struct.pack_into(order + 'H', header, 16, 4)
    struct.pack_into(order + 'H', header, 18, 62 if wide else 3)
    struct.pack_into(order + word, header, 32 if wide else 28, 64)
    stride = 56 if wide else 32
    struct.pack_into(order + 'HH', header, 54 if wide else 42, stride, 1)
    entry = bytearray(stride)
    struct.pack_into(order + 'I', entry, 0, 4)
    struct.pack_into(order + word, entry, 8 if wide else 4, 64 + stride)
    struct.pack_into(order + word, entry, 32 if wide else 16, len(notes))
    return bytes(header + entry) + notes


@pytest.mark.parametrize('wide,order', [(True, '<'), (False, '<'), (True, '>'), (False, '>')])
def test_elf_notes_bind_actual_process_leader_and_executable(tmp_path, wide, order):
    executable = tmp_path / 'python'
    core = tmp_path / 'core'
    core.write_bytes(_native_note_core(3456, executable, wide=wide, order=order))
    assert collector._core_process_ids(core, executable) == {3456}
    assert not collector._core_process_ids(core, tmp_path / 'different-python')


@pytest.mark.parametrize('defect', ['truncated', 'huge_notes', 'bad_class', 'not_core',
                                   'huge_program_table', 'wrong_owner'])
def test_malformed_native_notes_never_bind_a_worker(tmp_path, defect):
    executable = tmp_path / 'python'
    raw = bytearray(_native_note_core(3456, executable))
    if defect == 'truncated':
        del raw[-10:]
    elif defect == 'huge_notes':
        struct.pack_into('<Q', raw, 64 + 32, 5 * 1024 * 1024)
    elif defect == 'bad_class':
        raw[4] = 3
    elif defect == 'not_core':
        struct.pack_into('<H', raw, 16, 2)
    elif defect == 'huge_program_table':
        struct.pack_into('<H', raw, 56, 1025)
    elif defect == 'wrong_owner':
        raw = bytearray(raw.replace(b'CORE\0', b'BAD!\0'))
    core = tmp_path / 'core'
    core.write_bytes(raw)
    assert not collector._core_process_ids(core, executable)


def _identity(tmp_path):
    if sys.platform != 'linux':
        pytest.skip('ordinary worker provenance uses Linux process identity')
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = tmp_path / 'python'
    executable.write_bytes(b'owned executable')
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    return executable, evidence, {
        'event': 'session_start', 'mode': 'identity_only', 'pid': 3456,
        'time_ns': time.time_ns() - 1_000_000, 'source_sha': 'source',
        'root': str(tmp_path), 'executable': str(executable),
        'executable_sha256': collector._sha256(executable),
        'boot_id': collector.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'process_start_ticks': '123',
    }


@pytest.mark.parametrize('defect', ['none', 'finished', 'source', 'root', 'executable',
                                   'hash', 'boot', 'pid', 'start', 'malformed', 'symlink'])
def test_only_current_unfinished_owned_sessions_are_eligible(tmp_path, defect):
    import json

    executable, evidence, record = _identity(tmp_path)
    if defect in {'source', 'root', 'executable', 'hash', 'boot'}:
        field = {'source': 'source_sha', 'hash': 'executable_sha256', 'boot': 'boot_id'}.get(
            defect, defect)
        record[field] = 'different'
    elif defect == 'pid':
        record['pid'] = -1
    elif defect == 'start':
        record['process_start_ticks'] = '0'
    text = json.dumps(record) + '\n'
    if defect == 'finished':
        text += json.dumps({'event': 'session_finish', 'exitstatus': 1}) + '\n'
    elif defect == 'malformed':
        text = 'partial invalid json'
    journal = evidence / 'process-3456-123.jsonl'
    journal.write_text(text)
    if defect == 'symlink':
        target = tmp_path / 'external.jsonl'
        journal.rename(target)
        journal.symlink_to(target)
    sessions = collector._ordinary_sessions(evidence, tmp_path, 'source', 'source', executable, [])
    assert bool(sessions) is (defect == 'none')
    assert not collector._ordinary_sessions(evidence, tmp_path, 'other', 'source', executable, [])


@pytest.mark.parametrize('defect', ['none', 'pid', 'executable', 'stale'])
def test_ordinary_raw_core_requires_elf_owner_even_when_filename_matches(tmp_path, monkeypatch, defect):
    executable, evidence, session = _identity(tmp_path)
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(7890 if defect == 'pid' else 3456,
                                      tmp_path / 'other-python' if defect == 'executable' else executable))
    if defect == 'stale':
        os.utime(core, ns=(session['time_ns'] - 1, session['time_ns'] - 1))
    monkeypatch.setattr(collector, '_extract_systemd_core', lambda *args, **kwargs: None)
    selected, temporary = collector._ordinary_core(tmp_path, evidence, [session], executable, [])
    assert selected == (core if defect == 'none' else None)
    assert temporary is None


@pytest.mark.parametrize('second_finished', [False, True])
def test_reused_pid_is_refused_even_when_newer_session_matches(tmp_path, second_finished):
    import json

    executable, evidence, first = _identity(tmp_path)
    second = {**first, 'time_ns': first['time_ns'] + 1, 'process_start_ticks': '456'}
    (evidence / 'process-first.jsonl').write_text(json.dumps(first) + '\n')
    text = json.dumps(second) + '\n'
    if second_finished:
        text += json.dumps({'event': 'session_finish', 'exitstatus': 0}) + '\n'
    (evidence / 'process-second.jsonl').write_text(text)
    lines = []
    assert not collector._ordinary_sessions(evidence, tmp_path, 'source', 'source', executable, lines)
    assert 'ambiguous_process_pids_refused=[3456]' in lines


def test_empty_ordinary_session_does_not_enumerate_foreign_reports(tmp_path, monkeypatch):
    executable, evidence, _ = _identity(tmp_path)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setattr(collector.sys, 'executable', str(executable))

    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary collector scanned unrelated reports')

    def git_only(argv, **kwargs):
        assert argv[:2] == ['git', '-C']
        return subprocess.CompletedProcess(argv, 0, stdout='source\n')

    monkeypatch.setattr(collector.subprocess, 'run', git_only)
    monkeypatch.setattr(collector, '_apport_reports', forbidden)
    monkeypatch.setattr(collector, '_candidate_files', forbidden)
    assert collector.main(['--ordinary', '--evidence-dir', str(evidence)]) == 0
    report = (evidence / 'native-core-backtrace.txt').read_text()
    assert 'unfinished_owned_processes=0' in report
    assert 'No regular ELF core was available' in report


@pytest.mark.parametrize('timeout', [False, True])
def test_extracted_core_is_removed_after_gdb_success_or_timeout(tmp_path, monkeypatch, timeout):
    import json

    executable, evidence, session = _identity(tmp_path)
    temporary = tmp_path / 'qt-native-owned.elf'
    temporary.write_bytes(_native_note_core(3456, executable))
    (evidence / 'process-3456.jsonl').write_text(json.dumps(session) + '\n')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setattr(collector.sys, 'executable', str(executable))
    monkeypatch.setattr(collector.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(collector, '_ordinary_core', lambda *args: (temporary, temporary))

    def command(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.CompletedProcess(argv, 0, stdout='source\n')
        assert argv[:6] == ['gdb', '-q', '-nx', '-nh', '-batch', '-iex']
        assert argv[-2:] == [str(executable), str(temporary)]
        assert kwargs['timeout'] == collector.GDB_TIMEOUT_SECONDS
        assert kwargs['preexec_fn'] is collector._cap_backtrace_file
        kwargs['stdout'].write(b'owned C++ trace\n')
        if timeout:
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(collector.subprocess, 'run', command)
    assert collector.main(['--ordinary', '--evidence-dir', str(evidence)]) == 0
    assert not temporary.exists()
    report = (evidence / 'native-core-backtrace.txt').read_text()
    assert ('gdb_timeout=90s' if timeout else 'gdb_exit=0') in report
    assert 'owned C++ trace' in report
    assert len(report.encode()) <= collector.MAX_BACKTRACE_BYTES


@pytest.mark.parametrize('pattern,soft,expected', [('core.%p', 0, False),
                                                  ('core.%p', 1024, True),
                                                  ('|/missing/apport %p', 1024, False)])
def test_core_route_records_disabled_or_unsupported_production_without_claiming_capture(
    tmp_path, monkeypatch, pattern, soft, expected,
):
    import json
    from pathlib import Path

    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == '/proc/sys/kernel/core_pattern':
            return pattern
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(collector.resource, 'getrlimit', lambda kind: (soft, 1024))
    collector._record_core_route(tmp_path, evidence)
    receipt = json.loads((evidence / 'core-route.json').read_text())
    assert receipt['core_pattern'] == pattern
    assert receipt['core_limit_soft_bytes'] == soft
    assert receipt['route_verified'] is expected
    assert receipt['capture_guaranteed'] is False


@pytest.mark.parametrize('defect', ['none', 'pid_note', 'executable_note', 'boot', 'timestamp'])
def test_ordinary_systemd_core_requires_current_session_and_native_identity(
    tmp_path, monkeypatch, compressed_core, defect,
):
    from pathlib import Path

    reports, scratch, old, _ = compressed_core
    executable, evidence, session = _identity(tmp_path / 'identity')
    raw = _native_note_core(7890 if defect == 'pid_note' else 3456,
                            tmp_path / 'foreign-python' if defect == 'executable_note' else executable)
    old.unlink()
    boot = 'foreignboot' if defect == 'boot' else session['boot_id'].replace('-', '')
    stamp = session['time_ns'] // 1000 + (-100 if defect == 'timestamp' else 100)
    source = reports / f'core.python.1001.{boot}.3456.{stamp}.zst'
    packed = subprocess.run([shutil.which('zstd'), '-cq'], input=raw,
                            capture_output=True, check=True).stdout
    source.write_bytes(packed)
    path_type = Path
    monkeypatch.setattr(collector, 'Path', lambda value: reports if str(value) == (
        '/var/lib/systemd/coredump') else path_type(value))
    monkeypatch.setattr(collector, '_candidate_files', lambda *args: iter(()))
    selected, temporary = collector._ordinary_core(tmp_path, evidence, [session], executable, [])
    try:
        assert bool(selected) is (defect == 'none')
        if selected:
            assert selected == temporary
            assert selected.read_bytes() == raw
        assert source.read_bytes() == packed
    finally:
        if temporary is not None:
            temporary.unlink()
    assert not list(evidence.parent.glob('qt-native-*.elf'))


def test_foreign_process_leader_cannot_match_a_reused_thread_identifier(tmp_path, monkeypatch):
    executable, evidence, session = _identity(tmp_path)
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(3456, executable, leader_pid=7890))
    assert collector._core_process_ids(core, executable) == {7890}
    monkeypatch.setattr(collector, '_extract_systemd_core', lambda *args, **kwargs: None)
    lines = []
    assert collector._ordinary_core(tmp_path, evidence, [session], executable, lines) == (None, None)
    assert not any(line.startswith('matched_process=') for line in lines)


def test_owned_leader_matches_even_when_the_recorded_lwp_is_different(tmp_path, monkeypatch):
    executable, evidence, session = _identity(tmp_path)
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(7890, executable, leader_pid=3456))
    assert collector._core_process_ids(core, executable) == {3456}
    monkeypatch.setattr(collector, '_extract_systemd_core', lambda *args, **kwargs: None)
    lines = []
    assert collector._ordinary_core(tmp_path, evidence, [session], executable, lines) == (core, None)
    assert any(line.startswith('matched_process=') for line in lines)


@pytest.mark.parametrize('mode', ['signal', 'direct', 'bounded', 'unsupported', 'missing_original'])
def test_original_fault_diagnostic_selects_restored_frame_with_bounded_fallback(monkeypatch, mode):
    from types import SimpleNamespace

    selected = []
    commands = []
    messages = []

    class Frame:
        def __init__(self, index, kind):
            self.index, self.kind = index, kind
            self.next = None

        def type(self):
            if mode == 'unsupported':
                raise RuntimeError('unwinding unavailable')
            return self.kind

        def older(self):
            return self.next

        def select(self):
            selected.append(self.index)

        def level(self):
            return self.index

        def name(self):
            return 'original_fault' if self.index == 2 else 'signal_raise'

    frames = [Frame(index, 1 if index == (24 if mode == 'bounded' else 1) else 0)
              for index in range(26 if mode == 'bounded' else 3)]
    for first, second in zip(frames, frames[1:]):
        first.next = second
    if mode == 'direct':
        frames[1].kind = 0
    elif mode == 'missing_original':
        frames[1].next = None

    def execute(command):
        commands.append(command)
        if mode == 'unsupported' and command == 'p $_siginfo':
            raise RuntimeError('siginfo unavailable')

    gdb = SimpleNamespace(execute=execute, write=messages.append, newest_frame=lambda: frames[0],
                          SIGTRAMP_FRAME=1)
    monkeypatch.setitem(sys.modules, 'gdb', gdb)
    exec(collector._ORDINARY_FAULT_SCRIPT, {})
    assert commands[:4] == ['set print elements 32', 'set print max-depth 4',
                            'p $_siginfo', 'bt full 24']
    output = ''.join(messages)
    if mode == 'signal':
        assert selected == [2, 0]
        assert 'selection=after_sigtramp level=2 name=original_fault' in output
    elif mode in {'direct', 'bounded'}:
        assert selected == [0, 0]
        assert 'selection=current_fault_frame' in output
    else:
        assert 'unsupported' in output
        assert not selected or selected == [0]
    if mode in {'signal', 'direct', 'bounded'}:
        assert commands[-2:] == ['info registers', 'x/16i $pc']


def test_serial_gdb_command_remains_the_original_bounded_backtrace(tmp_path, monkeypatch):
    import json

    executable, evidence, session = _identity(tmp_path)
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(3456, executable))
    (evidence / 'file-rss.jsonl').write_text(json.dumps(session) + '\n')
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setattr(collector.sys, 'executable', str(executable))
    monkeypatch.setattr(collector.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(collector, '_candidate_files', lambda *args: iter([(core, core.stat().st_size)]))
    monkeypatch.setattr(collector, '_apport_reports', lambda *args: [])
    expected = ['gdb', '-q', '-nx', '-nh', '-batch',
                '-iex', 'set auto-load safe-path /dev/null',
                '-iex', 'set debuginfod enabled off', '-ex', 'set pagination off',
                '-ex', 'info proc', '-ex', 'info threads', '-ex', 'thread apply all bt 24',
                str(executable), str(core)]

    def command(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.CompletedProcess(argv, 0, stdout='source\n')
        assert argv == expected
        assert kwargs['timeout'] == 90
        assert kwargs['preexec_fn'] is collector._cap_backtrace_file
        kwargs['stdout'].write(b'unchanged serial trace\n')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(collector.subprocess, 'run', command)
    assert collector.main(['--evidence-dir', str(evidence)]) == 0
    assert core.exists()
    assert collector.MAX_BACKTRACE_BYTES == 4 * 1024 ** 2
