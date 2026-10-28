"""Real zstd diagnostics preserve process identity and bounded scratch files."""

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip('resource', reason='systemd core diagnostics require POSIX resource limits')

from tools import collect_qt_native_backtrace as collector


@pytest.mark.parametrize('workflow,job,run_name,recover_name,upload_name,condition', [
    ('.github/workflows/tests.yml', 'coverage-shards',
     'Run coverage shard in bounded batches',
     'Recover a coverage worker native backtrace',
     'Upload coverage native failure evidence',
     'failure() || cancelled()'),
    ('.github/workflows/_pytest-suite.yml', 'pytest',
     'Run ${{ inputs.suite_name }} tests',
     'Recover an ordinary Qt worker native backtrace',
     'Upload ordinary Qt native failure evidence',
     "(failure() || cancelled()) && startsWith(inputs.suite_name, 'Qt')"),
])
def test_cancelled_qt_or_coverage_run_preserves_only_bounded_native_evidence(
    workflow, job, run_name, recover_name, upload_name, condition,
):
    yaml = pytest.importorskip('yaml')
    steps = yaml.safe_load((Path(__file__).resolve().parents[1] / workflow).read_text(encoding="utf-8"))[
        'jobs'][job]['steps']
    by_name = {step['name']: (index, step) for index, step in enumerate(steps)
               if 'name' in step}
    run_index, _ = by_name[run_name]
    recover_index, recover = by_name[recover_name]
    upload_index, upload = by_name[upload_name]

    assert run_index < recover_index < upload_index
    assert recover['if'] == upload['if'] == condition
    assert recover['continue-on-error'] is True
    assert recover['timeout-minutes'] == 3
    assert '--ordinary' in recover['run']
    assert upload['uses'] == 'actions/upload-artifact@v7'
    assert upload['with']['if-no-files-found'] == 'ignore'
    assert upload['with']['retention-days'] == 3
    assert upload['with']['path'].splitlines() == [
        '${{ runner.temp }}/spacr-qt-native/process-*.jsonl',
        '${{ runner.temp }}/spacr-qt-native/native-core-backtrace.txt',
        '${{ runner.temp }}/spacr-qt-native/core-route.json',
    ]
    name = upload['with']['name']
    assert '${{ github.run_id }}' in name
    assert '${{ github.run_attempt }}' in name
    assert ('${{ matrix.shard }}' if job == 'coverage-shards'
            else '${{ inputs.file_shard_index }}') in name
    assert collector.GDB_TIMEOUT_SECONDS == 90
    assert collector.CORE_EXTRACT_TIMEOUT_SECONDS == 60
    assert collector.MAX_BACKTRACE_BYTES == 4 * 1024 * 1024


@pytest.mark.parametrize('blocking_results,expect_success', [
    ({'coverage-combine': 'success', 'qt': 'success'}, True),
    ({'coverage-combine': 'failure', 'qt': 'success'}, False),
    ({'coverage-combine': 'skipped', 'qt': 'success'}, False),
    ({'coverage-combine': 'success', 'qt': 'cancelled'}, False),
])
def test_intentional_cancel_skips_aggregates_without_waiving_real_failures(
    blocking_results, expect_success,
):
    yaml = pytest.importorskip('yaml')
    jobs = yaml.safe_load((Path(__file__).resolve().parents[1] /
                           '.github/workflows/tests.yml').read_text(encoding="utf-8"))['jobs']
    combine = jobs['coverage-combine']
    release = jobs['release-gate']
    assert combine['if'] == release['if'] == 'always() && !cancelled()'
    assert combine['needs'] == 'coverage-shards'
    assert set(blocking_results) <= set(release['needs'])

    shard_gate = next(step for step in combine['steps']
                      if step.get('name') == 'Require every coverage shard to have passed its tests')
    for shard_result in ('success', 'failure', 'cancelled'):
        env = {**os.environ, 'SPACR_COVERAGE_SHARDS_RESULT': shard_result}
        result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', shard_gate['run']],
                                env=env, capture_output=True, text=True, check=False)
        assert (result.returncode == 0) is (shard_result == 'success')

    release_gate = next(step for step in release['steps']
                        if step.get('name') == 'Require every blocking job to succeed')
    context = {name: {'result': result} for name, result in blocking_results.items()}
    env = {**os.environ, 'NEEDS_CONTEXT': json.dumps(context)}
    result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', release_gate['run']],
                            env=env, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is expect_success


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


def _native_many_header_core(pid, executable, count):
    """Put one real PT_NOTE after a large table of bounded PT_NULL entries."""
    original = _native_note_core(pid, executable)
    table = bytearray(56 * count)
    table[:56] = original[64:120]
    struct.pack_into('<Q', table, 8, 64 + len(table))
    header = bytearray(original[:64])
    struct.pack_into('<H', header, 56, count)
    return bytes(header + table) + original[120:]


def test_realistic_1091_header_core_keeps_exact_pid_and_executable_guard(tmp_path):
    """The observed 7a Qt batch needs more than 1,024 small ELF headers."""
    assert collector.MAX_PROGRAM_HEADER_TABLE_BYTES == 1024 * 256
    executable = tmp_path / 'python'
    core = tmp_path / 'core'
    core.write_bytes(_native_many_header_core(3456, executable, 1091))
    diagnostics = []
    assert collector._core_process_ids(core, executable, diagnostics) == {3456}
    detail = json.loads(diagnostics[0].removeprefix('elf_identity='))
    assert detail['reason'] == 'accepted'
    assert detail['program_header_table_bytes'] == 1091 * 56
    assert not collector._core_process_ids(core, tmp_path / 'wrong-python')


@pytest.mark.parametrize('wide,order', [(True, '<'), (False, '<'), (True, '>'), (False, '>')])
def test_elf_notes_bind_actual_process_leader_and_executable(tmp_path, wide, order):
    executable = tmp_path / 'python'
    core = tmp_path / 'core'
    core.write_bytes(_native_note_core(3456, executable, wide=wide, order=order))
    assert collector._core_process_ids(core, executable) == {3456}
    assert not collector._core_process_ids(core, tmp_path / 'different-python')


@pytest.mark.parametrize('defect', ['truncated', 'huge_notes', 'bad_class', 'not_core',
                                   'huge_program_table', 'truncated_program_table',
                                   'out_of_file_table', 'invalid_stride', 'wrong_owner'])
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
        struct.pack_into('<H', raw, 56, 5000)
    elif defect == 'truncated_program_table':
        struct.pack_into('<H', raw, 56, 1091)
    elif defect == 'out_of_file_table':
        struct.pack_into('<Q', raw, 32, len(raw) + 1)
    elif defect == 'invalid_stride':
        struct.pack_into('<H', raw, 54, 257)
    elif defect == 'wrong_owner':
        raw = bytearray(raw.replace(b'CORE\0', b'BAD!\0'))
    core = tmp_path / 'core'
    core.write_bytes(raw)
    assert not collector._core_process_ids(core, executable)


@pytest.mark.parametrize('defect,reason', [
    ('program_headers', 'program_header_table_byte_ceiling'),
    ('truncated_program_headers', 'truncated_program_headers'),
    ('note_bytes', 'note_byte_ceiling'),
    ('missing_pid', 'missing_process_pid'),
    ('conflicting_pid', 'conflicting_process_pids'),
    ('executable', 'absent_executable_path'),
])
def test_rejected_native_identity_reports_the_bounded_reason_without_accepting_it(
    tmp_path, defect, reason,
):
    executable = tmp_path / 'python'
    raw = bytearray(_native_note_core(3456, executable))
    notes_start = 64 + 56
    if defect == 'program_headers':
        struct.pack_into('<H', raw, 56, 5000)
    elif defect == 'truncated_program_headers':
        struct.pack_into('<H', raw, 56, 1091)
    elif defect == 'note_bytes':
        struct.pack_into('<Q', raw, 64 + 32, 5 * 1024 * 1024)
    elif defect in ('missing_pid', 'conflicting_pid'):
        cursor = notes_start
        while cursor < len(raw):
            names, length, kind = struct.unpack_from('<III', raw, cursor)
            payload = cursor + 12 + ((names + 3) & ~3)
            following = payload + ((length + 3) & ~3)
            if kind == 3:
                if defect == 'missing_pid':
                    struct.pack_into('<I', raw, cursor + 8, 77)
                else:
                    second = bytearray(raw[cursor:following])
                    struct.pack_into('<i', second, payload - cursor + 24, 7890)
                    raw.extend(second)
                    struct.pack_into('<Q', raw, 64 + 32, len(raw) - notes_start)
                break
            cursor = following
        else:
            pytest.fail('synthetic ELF lacks PRPSINFO')
    core = tmp_path / 'core'
    core.write_bytes(raw)
    diagnostics = []
    expected = tmp_path / 'other-python' if defect == 'executable' else executable
    assert collector._core_process_ids(core, expected, diagnostics) == set()
    assert len(diagnostics) == 1
    assert len(diagnostics[0]) < 512
    detail = json.loads(diagnostics[0].removeprefix('elf_identity='))
    assert detail['reason'] == reason
    assert detail['elf_class'] == 64
    assert detail['program_headers'] == (
        5000 if defect == 'program_headers' else
        1091 if defect == 'truncated_program_headers' else 1)
    if defect == 'conflicting_pid':
        assert detail['process_pids'] == [3456, 7890]
        assert detail['executable_path_present'] is True
    elif defect == 'executable':
        assert detail['process_pids'] == [3456]
        assert detail['executable_path_present'] is False
    elif defect == 'missing_pid':
        assert detail['process_pid_count'] == 0


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
        'boot_id': collector.Path('/proc/sys/kernel/random/boot_id').read_text(encoding="utf-8").strip(),
        'process_start_ticks': '123',
    }


@pytest.mark.parametrize('defect', ['none', 'finished', 'finished_success', 'source', 'root', 'executable',
                                   'hash', 'boot', 'pid', 'start', 'malformed', 'symlink'])
def test_only_current_owned_sessions_are_eligible_regardless_of_pytest_finish(tmp_path, defect):
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
    if defect in {'finished', 'finished_success'}:
        text += json.dumps({'event': 'session_finish',
                            'exitstatus': 0 if defect == 'finished_success' else 1}) + '\n'
    elif defect == 'malformed':
        text = 'partial invalid json'
    journal = evidence / 'process-3456-123.jsonl'
    journal.write_text(text, encoding="utf-8")
    if defect == 'symlink':
        target = tmp_path / 'external.jsonl'
        journal.rename(target)
        journal.symlink_to(target)
    lines = []
    sessions = collector._ordinary_sessions(evidence, tmp_path, 'source', 'source', executable, lines)
    eligible = defect in {'none', 'finished', 'finished_success'}
    finished = defect in {'finished', 'finished_success'}
    assert bool(sessions) is eligible
    assert f'eligible_owned_processes={int(eligible)}' in lines
    assert f'pytest_finished_owned_processes={int(finished)}' in lines
    assert f'unfinished_owned_processes={int(eligible and not finished)}' in lines
    if eligible:
        assert sessions == [{**record, 'pytest_session_finished': finished}]
        assert 'pytest_session_finished' not in record
    assert not collector._ordinary_sessions(evidence, tmp_path, 'other', 'source', executable, [])


@pytest.mark.parametrize('defect', ['none', 'pid', 'executable', 'stale'])
def test_ordinary_raw_core_requires_elf_owner_even_when_filename_matches(tmp_path, monkeypatch, defect):
    executable, evidence, session = _identity(tmp_path)
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(7890 if defect == 'pid' else 3456,
                                      tmp_path / 'other-python' if defect == 'executable' else executable))
    timestamp = session['time_ns'] - 1 if defect == 'stale' else session['time_ns'] + 1
    os.utime(core, ns=(timestamp, timestamp))
    monkeypatch.setattr(collector, '_extract_systemd_core', lambda *args, **kwargs: None)
    selected, temporary = collector._ordinary_core(tmp_path, evidence, [session], executable, [])
    assert selected == (core if defect == 'none' else None)
    assert temporary is None


@pytest.mark.parametrize('exitstatus', [0, 1])
@pytest.mark.parametrize('defect', ['none', 'pid', 'executable', 'stale'])
def test_a_pytest_finished_journal_still_requires_an_owned_recent_core(
        tmp_path, monkeypatch, exitstatus, defect):
    import json

    executable, evidence, record = _identity(tmp_path)
    journal = evidence / 'process-3456-123.jsonl'
    journal.write_text(json.dumps(record) + '\n' + json.dumps({
        'event': 'session_finish', 'exitstatus': exitstatus,
    }) + '\n', encoding='utf-8')
    lines = []
    sessions = collector._ordinary_sessions(
        evidence, tmp_path, 'source', 'source', executable, lines)
    assert sessions == [{**record, 'pytest_session_finished': True}]
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(
        7890 if defect == 'pid' else 3456,
        tmp_path / 'foreign-python' if defect == 'executable' else executable))
    timestamp = record['time_ns'] + (-1 if defect == 'stale' else 1)
    os.utime(core, ns=(timestamp, timestamp))
    monkeypatch.setattr(collector, '_candidate_files',
                        lambda *args: iter([(core, core.stat().st_size)]))
    monkeypatch.setattr(collector, '_extract_systemd_core', lambda *args, **kwargs: None)
    selected, temporary = collector._ordinary_core(
        tmp_path, evidence, sessions, executable, lines)
    assert selected == (core if defect == 'none' else None)
    assert temporary is None and core.exists()
    if defect == 'none':
        matched = next(line for line in lines if line.startswith('matched_process='))
        assert json.loads(matched.split('=', 1)[1]) == sessions[0]
    else:
        assert not any(line.startswith('matched_process=') for line in lines)


@pytest.mark.parametrize('second_finished', [False, True])
def test_reused_pid_is_refused_even_when_newer_session_matches(tmp_path, second_finished):
    import json

    executable, evidence, first = _identity(tmp_path)
    second = {**first, 'time_ns': first['time_ns'] + 1, 'process_start_ticks': '456'}
    (evidence / 'process-first.jsonl').write_text(json.dumps(first) + '\n', encoding="utf-8")
    text = json.dumps(second) + '\n'
    if second_finished:
        text += json.dumps({'event': 'session_finish', 'exitstatus': 0}) + '\n'
    (evidence / 'process-second.jsonl').write_text(text, encoding="utf-8")
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
    report = (evidence / 'native-core-backtrace.txt').read_text(encoding="utf-8")
    assert 'unfinished_owned_processes=0' in report
    assert 'No regular ELF core was available' in report


@pytest.mark.parametrize('timeout', [False, True])
def test_extracted_core_is_removed_after_gdb_success_or_timeout(tmp_path, monkeypatch, timeout):
    import json

    executable, evidence, session = _identity(tmp_path)
    temporary = tmp_path / 'qt-native-owned.elf'
    temporary.write_bytes(_native_note_core(3456, executable))
    (evidence / 'process-3456.jsonl').write_text(json.dumps(session) + '\n', encoding="utf-8")
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setattr(collector.sys, 'executable', str(executable))
    monkeypatch.setattr(collector.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(collector, '_ordinary_core', lambda *args: (temporary, temporary))
    library_path = str(tmp_path / "library 'quoted' with spaces") + ':/inferior/lib'
    monkeypatch.setenv('LD_LIBRARY_PATH', library_path)
    monkeypatch.setenv('PYTHONHOME', '/foreign/python')
    monkeypatch.setenv('PYTHONPATH', '/foreign/modules')
    parent_environment = os.environ.copy()

    def command(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.CompletedProcess(argv, 0, stdout='source\n')
        assert argv[:6] == ['gdb', '-q', '-nx', '-nh', '-batch', '-iex']
        assert argv[-2:] == [str(executable), str(temporary)]
        assert kwargs['timeout'] == collector.GDB_TIMEOUT_SECONDS
        assert kwargs['preexec_fn'] is collector._cap_backtrace_file
        assert kwargs['env'] == {key: value for key, value in parent_environment.items()
                                 if key not in {'LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH'}}
        library_command = argv[argv.index('python import gdb; gdb.set_parameter(' +
                                         "'solib-search-path', " + repr(library_path) + ')')]
        parameters = {}
        fake_gdb = type('Gdb', (), {'set_parameter': lambda _self, key, value:
                                  parameters.__setitem__(key, value)})()
        monkeypatch.setitem(sys.modules, 'gdb', fake_gdb)
        exec(library_command.removeprefix('python '), {})
        assert parameters == {'solib-search-path': library_path}
        assert os.environ == parent_environment
        kwargs['stdout'].write(b'owned C++ trace\n')
        if timeout:
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(collector.subprocess, 'run', command)
    assert collector.main(['--ordinary', '--evidence-dir', str(evidence)]) == 0
    assert not temporary.exists()
    report = (evidence / 'native-core-backtrace.txt').read_text(encoding="utf-8")
    assert ('gdb_timeout=90s' if timeout else 'gdb_exit=0') in report
    assert 'owned C++ trace' in report
    assert len(report.encode()) <= collector.MAX_BACKTRACE_BYTES


@pytest.mark.parametrize('line_break', ['\n', '\r'])
def test_gdb_library_search_refuses_command_line_breaks(monkeypatch, line_break):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/inferior/lib' + line_break + 'quit')
    with pytest.raises(ValueError, match='line break'):
        collector._gdb_environment()


def test_refused_debugger_environment_removes_extracted_core_and_finishes_report(tmp_path, monkeypatch):
    executable, evidence, session = _identity(tmp_path)
    temporary = tmp_path / 'qt-native-owned.elf'
    temporary.write_bytes(_native_note_core(3456, executable))
    (evidence / 'process-3456.jsonl').write_text(json.dumps(session) + '\n', encoding="utf-8")
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/inferior/lib\nquit')
    before = os.environ.copy()
    monkeypatch.setattr(collector.sys, 'executable', str(executable))
    monkeypatch.setattr(collector.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(collector, '_ordinary_core', lambda *args: (temporary, temporary))

    def command(argv, **kwargs):
        assert argv[0] == 'git', 'Refused debugger environment must never launch GDB'
        return subprocess.CompletedProcess(argv, 0, stdout='source\n')

    monkeypatch.setattr(collector.subprocess, 'run', command)
    assert collector.main(['--ordinary', '--evidence-dir', str(evidence)]) == 0
    assert not temporary.exists()
    assert os.environ == before
    report = (evidence / 'native-core-backtrace.txt').read_text(encoding="utf-8")
    assert report.endswith('gdb_environment_refused=GDB library search path contains a line break\n')
    assert 'source_sha=source' in report
    assert 'command=gdb' not in report
    assert len(report.encode()) <= collector.MAX_BACKTRACE_BYTES


def test_real_gdb_keeps_embedded_python_and_inferior_search_separate(tmp_path, monkeypatch):
    gdb = shutil.which('gdb')
    library = Path(sys.base_prefix) / 'lib' / f'libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0'
    if not gdb or not library.is_file():
        pytest.skip('GDB and a foreign shared Python runtime are required')
    clean_environment = {key: value for key, value in os.environ.items()
                         if key not in {'LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH'}}
    base = [gdb, '-q', '-nx', '-nh', '-batch']
    identity = subprocess.run(base + ['-ex', 'python import sys; print("ABI="+str(sys.version_info[:2]))'],
                              env=clean_environment, capture_output=True, text=True, timeout=10)
    if f'ABI={sys.version_info[:2]}' not in identity.stdout:
        pytest.skip('Foreign Python does not share the system GDB Python ABI')
    contaminated = dict(clean_environment, LD_LIBRARY_PATH=str(library.parent))
    failure = subprocess.run(base + ['-ex', 'python import math'], env=contaminated,
                             capture_output=True, text=True, timeout=10)
    if "No module named 'math'" not in failure.stderr:
        pytest.skip('Available foreign runtime does not reproduce the missing system extension')
    assert failure.returncode != 0
    search = str(library.parent) + ':' + str(tmp_path / "spaces 'quotes' \\slashes")
    monkeypatch.setenv('LD_LIBRARY_PATH', search)
    monkeypatch.setenv('PYTHONHOME', '/foreign/python')
    monkeypatch.setenv('PYTHONPATH', '/foreign/modules')
    before = os.environ.copy()
    environment, commands = collector._gdb_environment()
    probe = ('python import math, hashlib, gdb; '
             'assert gdb.parameter("solib-search-path") == ' + repr(search) + '; '
             'assert hashlib.sha256(b"probe").hexdigest() == '
             '"ba9c736f19e7f60b7f6764adb0b7908c0a2b394e09b6c09863528c7f2bc86095"; '
             'print("embedded imports and inferior search passed")')
    result = subprocess.run(base + commands + ['-ex', probe], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'embedded imports and inferior search passed' in result.stdout
    assert os.environ == before


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
    receipt = json.loads((evidence / 'core-route.json').read_text(encoding="utf-8"))
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
    lines = []
    selected, temporary = collector._ordinary_core(tmp_path, evidence, [session], executable, lines)
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
    if defect in ('none', 'pid_note', 'executable_note'):
        assert f"elf_identity_expected_pid={session['pid']}" in lines
        diagnostic = next(line for line in lines if line.startswith('elf_identity='))
        details = json.loads(diagnostic.removeprefix('elf_identity='))
        assert details['reason'] == ('absent_executable_path' if defect == 'executable_note'
                                     else 'accepted')
        assert details['process_pids'] == ([7890] if defect == 'pid_note' else [3456])
        if defect != 'none':
            assert 'ordinary extraction rejected: ELF PID or executable differs' in lines
    else:
        assert not any(line.startswith('elf_identity=') for line in lines)


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
    assert commands[:5] == ['set print elements 32', 'set print max-depth 4',
                            'set print frame-arguments none', 'p $_siginfo', 'bt 24']
    assert not any('full' in command for command in commands)
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


@pytest.mark.parametrize('case,expected', [
    ('source_worker', 'wrapper_type_diagnostic=_SourceWorker wrapper_ptr=0x10000'),
    ('wrong_library', 'wrapper_type_diagnostic=unsupported_library_hash'),
    ('wrong_python', 'wrapper_type_diagnostic=unsupported_python_abi'),
    ('wrong_pc', 'wrapper_type_diagnostic=unsupported_fault_pc'),
    ('wrong_architecture', 'wrapper_type_diagnostic=unsupported_architecture'),
    ('unreadable_type', 'wrapper_type_diagnostic=unavailable'),
])
def test_qthread_fault_reports_only_a_bounded_python_wrapper_class(
    tmp_path, monkeypatch, case, expected,
):
    """A matching 6.12 frame distinguishes workers without uploading core bytes."""
    from types import SimpleNamespace

    library = tmp_path / 'libshiboken6.abi3.so.6.12'
    library.write_bytes(b'controlled library')
    fingerprint = ('different' if case == 'wrong_library' else
                   '2b3d9767d69da0afabe4a383dc241a110fac4c0bb63cdeb3d01314e9c702241e')
    monkeypatch.setattr('hashlib.sha256',
                        lambda _bytes: SimpleNamespace(hexdigest=lambda: fingerprint))
    messages = []
    memory = {
        0x10008: struct.pack('<Q', 0x20000),
        0x20018: struct.pack('<Q', 0x30000),
        0x30000: b'_SourceWorker\0'.ljust(96, b'\0'),
    }

    def read_memory(address, size):
        if case == 'unreadable_type' and address == 0x20018:
            raise RuntimeError('unreadable target memory')
        return memory[address][:size]

    class Frame:
        def __init__(self, kind, name, older=None):
            self.kind, self.title, self.next = kind, name, older

        def type(self):
            return self.kind

        def older(self):
            return self.next

        def select(self):
            pass

        def level(self):
            return 1

        def name(self):
            return self.title

        def pc(self):
            return 0x12345

        def architecture(self):
            return SimpleNamespace(name=lambda: (
                'aarch64' if case == 'wrong_architecture' else 'i386:x86-64'
            ))

    original = Frame(0, 'Shiboken::BindingManager::unregisterWrapper(SbkObject*)')
    signal = Frame(1, 'signal_raise', original)

    def execute(command, *, to_string=False):
        if command.startswith('info symbol '):
            offset = 64 if case == 'wrong_pc' else 80
            return ('Shiboken::BindingManager::unregisterWrapper(SbkObject*) '
                    f'+ {offset} in section .text of {library}\n')
        assert not to_string

    gdb = SimpleNamespace(
        execute=execute,
        write=messages.append,
        newest_frame=lambda: signal,
        SIGTRAMP_FRAME=1,
        solib_name=lambda _pc: str(library),
        parse_and_eval=lambda expression: (
            0x30B0DF0 if case == 'wrong_python' else 0x30C0DF0
        ) if expression == 'Py_Version' else 0x10000,
        selected_inferior=lambda: SimpleNamespace(read_memory=read_memory),
    )
    monkeypatch.setitem(sys.modules, 'gdb', gdb)

    exec(collector._ORDINARY_FAULT_SCRIPT, {})

    assert expected in ''.join(messages)
    assert 'controlled library' not in ''.join(messages)


@pytest.mark.parametrize('library_path', [None, '', "/inferior/library 'quoted' with spaces:/inferior/lib"],
                         ids=['unset', 'empty', 'inherited'])
def test_serial_gdb_command_remains_the_original_bounded_backtrace(tmp_path, monkeypatch, library_path):
    import json

    executable, evidence, session = _identity(tmp_path)
    core = evidence / 'core.3456'
    core.write_bytes(_native_note_core(3456, executable))
    (evidence / 'file-rss.jsonl').write_text(json.dumps(session) + '\n', encoding="utf-8")
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))
    monkeypatch.setenv('GITHUB_SHA', 'source')
    if library_path is None:
        monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    else:
        monkeypatch.setenv('LD_LIBRARY_PATH', library_path)
    monkeypatch.setenv('PYTHONHOME', '/foreign/python')
    monkeypatch.setenv('PYTHONPATH', '/foreign/modules')
    parent_environment = os.environ.copy()
    monkeypatch.setattr(collector.sys, 'executable', str(executable))
    monkeypatch.setattr(collector.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(collector, '_candidate_files', lambda *args: iter([(core, core.stat().st_size)]))
    monkeypatch.setattr(collector, '_apport_reports', lambda *args: [])
    expected = ['gdb', '-q', '-nx', '-nh', '-batch',
                '-iex', 'set auto-load safe-path /dev/null',
                '-iex', 'set debuginfod enabled off']
    if library_path:
        expected.extend(['-iex', "python import gdb; gdb.set_parameter('solib-search-path', " +
                         repr(library_path) + ')'])
    expected.extend(['-ex', 'set pagination off', '-ex', 'info proc',
                     '-ex', 'info threads', '-ex', 'thread apply all bt 24',
                     str(executable), str(core)])

    def command(argv, **kwargs):
        if argv[0] == 'git':
            return subprocess.CompletedProcess(argv, 0, stdout='source\n')
        assert argv == expected
        assert kwargs['timeout'] == 90
        assert kwargs['preexec_fn'] is collector._cap_backtrace_file
        assert kwargs['env'] == {key: value for key, value in parent_environment.items()
                                 if key not in {'LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH'}}
        kwargs['stdout'].write(b'unchanged serial trace\n')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(collector.subprocess, 'run', command)
    assert collector.main(['--evidence-dir', str(evidence)]) == 0
    assert os.environ == parent_environment
    assert core.exists()
    assert collector.MAX_BACKTRACE_BYTES == 4 * 1024 ** 2


@pytest.mark.parametrize('defect', [
    'none', 'hash', 'size', 'wrong_executable', 'foreign_mapping', 'page_size',
    'bias_disagreement', 'core_mapping', 'architecture', 'program_headers',
    'relro', 'writable',
])
def test_stylesheet_binary_context_requires_real_elf_and_agreeing_mappings(
        tmp_path, monkeypatch, defect):
    """Real bounded notes and load headers must agree before register interpretation."""
    executable = tmp_path / 'python'
    library = tmp_path / 'libQt6Widgets.so.6'
    bias = 0x70000000
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<HH', header, 16, 3, 62)
    struct.pack_into('<Q', header, 32, 64)
    struct.pack_into('<HH', header, 54, 56, 3)
    segment = struct.pack('<IIQQQQQQ', 1, 5, 0, 0, 0, 0x280000, 0x280000, 4096)
    segment += struct.pack('<IIQQQQQQ', 1, 6 if defect in {'relro', 'writable'} else 4,
                           0x280000, 0x280000, 0,
                           0x80000, 0x80000, 4096)
    segment += struct.pack('<IIQQQQQQ', 0x6474e552 if defect in {'relro', 'writable'} else 0,
                           4, 0x280000, 0x280000, 0,
                           0x80000, 0x80000, 1)
    library.write_bytes((header + segment).ljust(0x300000, b'\0'))
    monkeypatch.setattr(collector, 'QT_WIDGETS_BYTES', library.stat().st_size)
    monkeypatch.setattr(collector, 'QT_WIDGETS_SHA256', collector._sha256(library))
    if defect in {'hash', 'architecture', 'program_headers'}:
        raw = bytearray(library.read_bytes())
        if defect == 'hash':
            raw[-1] = 1
        elif defect == 'architecture':
            struct.pack_into('<H', raw, 18, 183)
        else:
            struct.pack_into('<H', raw, 56, 6000)
        library.write_bytes(raw)
        if defect != 'hash':
            monkeypatch.setattr(collector, 'QT_WIDGETS_SHA256', collector._sha256(library))
    elif defect == 'size':
        monkeypatch.setattr(collector, 'QT_WIDGETS_BYTES', library.stat().st_size + 1)

    mappings = [(4096, 8192, 0), (bias, bias + 0x280000, 0),
                (bias + 0x280000, bias + 0x300000, 0x280)]
    names = [os.fsencode(executable), os.fsencode(library), os.fsencode(library)]
    if defect == 'wrong_executable':
        names[0] = b'/foreign/python'
    elif defect == 'foreign_mapping':
        names[-1] = os.fsencode(tmp_path / 'foreign/libQt6Widgets.so.6')
    elif defect == 'bias_disagreement':
        mappings[-1] = (bias + 0x281000, bias + 0x300000, 0x280)
    payload = struct.pack('<QQ', 3, 3 if defect == 'page_size' else 4096)
    payload += b''.join(struct.pack('<QQQ', *row) for row in mappings)
    payload += b'\0'.join(names) + b'\0'
    process = bytearray(136)
    struct.pack_into('<i', process, 24, 3456)

    def note(kind, body):
        return (struct.pack('<III', 5, len(body), kind) + b'CORE\0\0\0\0'
                + body + bytes((-len(body)) % 4))

    notes = note(3, process) + note(0x46494C45, payload)
    core_header = bytearray(header)
    struct.pack_into('<H', core_header, 16, 4)
    struct.pack_into('<H', core_header, 56, 3)
    table = struct.pack('<IIQQQQQQ', 4, 0, 232, 0, 0, len(notes), len(notes), 4)
    table += struct.pack('<IIQQQQQQ', 1, 5, 0,
                         bias + 4096 if defect == 'core_mapping' else bias,
                         0, 0, 0x27f000 if defect == 'core_mapping' else 0x280000, 4096)
    table += struct.pack('<IIQQQQQQ', 1, 6 if defect == 'writable' else 4,
                         0, bias + 0x280000, 0, 0, 0x80000, 4096)
    core = tmp_path / 'core'
    core.write_bytes(core_header + table + notes)
    result = collector._stylesheet_context(core, executable)
    if defect in {'none', 'relro', 'writable'}:
        readonly = [(bias, bias + 0x280000)]
        if defect != 'writable':
            readonly.append((bias + 0x280000, bias + 0x300000))
        assert result == {'library': str(library), 'bias': bias,
                          'sha256': collector._sha256(library),
                          'readonly': readonly}
        assert (bias + 0x280010 < readonly[-1][1]) is (defect != 'writable')
    else:
        assert result == {}
    assert core.exists() and library.exists()


@pytest.mark.parametrize('defect', [
    'none', 'no_context', 'architecture', 'fault_pc', 'caller', 'instructions',
    'address', 'unreadable', 'unresolved_vtable', 'writable_vtable',
    'foreign_vtable', 'vtable_offset', 'private_mismatch', 'iterator_mismatch',
])
def test_stylesheet_diagnostic_reads_only_gated_numeric_headers(monkeypatch, defect):
    from types import SimpleNamespace

    bias, entry, iterator, vtable = 0x70000000, 0x10000, 0x20000, 0x70004000
    library = '/verified/Qt/lib/libQt6Widgets.so.6'
    output, reads, commands = [], [], []
    instructions = {
        0x276657: b'\x49\x8b\x2e', 0x276663: b'\x48\x8b\x45\x08',
        0x276667: b'\xf6\x40\x30\x01', 0x27667e: b'\xe8\x3d\x37\xf6\xff',
        0x276c3b: b'\xe8\x30\xf8\xff\xff',
    }
    data = {entry: struct.pack('<QQ', vtable, 2),
            iterator: struct.pack('<Q', entry + (8 if defect == 'iterator_mismatch' else 0))}

    def memory(address, count):
        reads.append((address, count))
        if address in data:
            if defect == 'unreadable':
                raise RuntimeError('unreadable')
            return data[address][:count]
        code = instructions[address - bias]
        return b'\0' * count if defect == 'instructions' else code

    def execute(command, **kwargs):
        commands.append(command)
        if command.startswith('info symbol '):
            offset = 24 if defect == 'vtable_offset' else 16
            return ('sensitive object name/path' if defect == 'unresolved_vtable' else
                    f'vtable for QWidget + {offset} in section .rodata of {library}')

    class Frame:
        def name(self):
            return None

        def architecture(self):
            return SimpleNamespace(name=lambda: 'aarch64' if defect == 'architecture'
                                   else 'i386:x86-64')

        def pc(self):
            return bias + 0x276667 + (1 if defect == 'fault_pc' else 0)

        def older(self):
            return SimpleNamespace(pc=lambda: bias + 0x276c40 + (1 if defect == 'caller' else 0))

        def read_register(self, name):
            return {'rbp': 2 if defect == 'address' else entry, 'r14': iterator,
                    'rax': 3 if defect == 'private_mismatch' else 2}[name]

    gdb = SimpleNamespace(
        write=output.append, execute=execute,
        solib_name=lambda address: '/foreign/library' if (
            defect == 'foreign_vtable' and address == vtable) else library,
        selected_inferior=lambda: SimpleNamespace(read_memory=memory),
        newest_frame=lambda: None, SIGTRAMP_FRAME=1,
    )
    monkeypatch.setitem(sys.modules, 'gdb', gdb)
    context = {'library': library, 'bias': bias, 'sha256': collector.QT_WIDGETS_SHA256,
               'readonly': [] if defect == 'writable_vtable' else [(vtable, vtable + 4096)]}
    scope = {'_stylesheet_context': {} if defect == 'no_context' else context}
    exec(collector._ORDINARY_FAULT_SCRIPT, scope)
    output.clear()
    scope['stylesheet_entry_at_fault'](Frame())
    report = ''.join(output)
    gated = defect in {'none', 'unreadable', 'unresolved_vtable', 'writable_vtable',
                       'foreign_vtable', 'vtable_offset', 'private_mismatch', 'iterator_mismatch'}
    assert bool([row for row in reads if row[0] in data]) is gated
    assert sum(count for address, count in reads if address in data) <= 24
    if gated and defect != 'unreadable':
        assert f'entry={hex(entry)}' in report and f'private=0x2' in report
        assert f'entry_matches_iterator={defect != "iterator_mismatch"}' in report
        assert f'private_matches_fault_register={defect != "private_mismatch"}' in report
        expected_type = 'QWidget' if defect in {'none', 'private_mismatch', 'iterator_mismatch'} else 'unknown'
        assert f'native_vtable={expected_type}' in report
    else:
        assert 'unsupported' in report
    assert len(report.encode()) <= 2048
    assert 'sensitive' not in report and library not in report
    assert not any('call ' in command or 'full' in command for command in commands)


@pytest.mark.parametrize('case,expected', [
    ('list_item', 'wrapper_type_diagnostic=PySide6.QtWidgets.QListWidgetItem wrapper_ptr=0x10000'),
    ('wrong_hash', 'wrapper_type_diagnostic=unsupported_library_hash'),
    ('wrong_suffix', 'wrapper_type_diagnostic=unsupported_library'),
    ('wrong_pc', 'wrapper_type_diagnostic=unsupported_fault_pc'),
    ('wrong_architecture', 'wrapper_type_diagnostic=unsupported_architecture'),
    ('wrong_python', 'wrapper_type_diagnostic=unsupported_python_abi'),
    ('wrong_instruction', 'wrapper_type_diagnostic=unsupported_instructions'),
    ('short_instruction', 'wrapper_type_diagnostic=unsupported_instructions'),
    ('unaligned_wrapper', 'wrapper_type_diagnostic=unreadable_wrapper'),
    ('unreadable_type', 'wrapper_type_diagnostic=unavailable'),
    ('unsafe_type', 'wrapper_type_diagnostic=unreadable_type_name'),
])
def test_release_wrapper_fault_reads_only_the_qualified_6113_rbp_type(
    tmp_path, monkeypatch, case, expected,
):
    """The 6.11.3 fault keeps the wrapper in RBP, not the damaged R15 array."""
    from types import SimpleNamespace

    suffix = '6.12' if case == 'wrong_suffix' else '6.11'
    library = tmp_path / ('libshiboken6.abi3.so.' + suffix)
    library.write_bytes(b'controlled library with private signed-url token')
    fingerprint = ('unmatched' if case == 'wrong_hash' else
                   'a13ba52dc4ff2770fb749084a0e7fc38331ce5506d185c1a9858f73efa2edf2d')
    monkeypatch.setattr('hashlib.sha256',
                        lambda _bytes: SimpleNamespace(hexdigest=lambda: fingerprint))
    base = 0x40000
    instructions = {
        base + 0x0c: b'\x48\x89\xf5',
        base + 0x32: b'\x48\x8b\x45\x20',
        base + 0x41: b'\x4c\x8b\x38',
        base + 0x50: b'\x49\x8b\x34\xdf',
        base + 0x5f: b'\x48\x89\xea',
    }
    if case == 'wrong_instruction':
        instructions[base + 0x0c] = b'\x48\x89\xf7'
    if case == 'short_instruction':
        instructions[base + 0x50] = b'\x49'
    type_name = (b'/private/signed-url?token=secret' if case == 'unsafe_type'
                 else b'PySide6.QtWidgets.QListWidgetItem')
    memory = dict(instructions)
    memory.update({0x10008: struct.pack('<Q', 0x20000),
                   0x20018: struct.pack('<Q', 0x30000),
                   0x30000: (type_name + b'\0').ljust(96, b'\0')})
    reads, expressions, messages = [], [], []

    def read_memory(address, size):
        reads.append((address, size))
        if case == 'unreadable_type' and address == 0x20018:
            raise RuntimeError('private memory error token=secret')
        return memory[address][:size]

    def evaluate(expression):
        expressions.append(expression)
        if expression == 'Py_Version':
            return 0x30B0DF0 if case == 'wrong_python' else 0x30C0DF0
        assert expression == '$rbp'
        return 0x10001 if case == 'unaligned_wrapper' else 0x10000

    class Frame:
        def name(self):
            return 'Shiboken::BindingManager::releaseWrapper(SbkObject*)'

        def architecture(self):
            return SimpleNamespace(name=lambda: (
                'aarch64' if case == 'wrong_architecture' else 'i386:x86-64'))

        def pc(self):
            return base + 80

    def execute(command, *, to_string=False):
        assert command == 'info symbol ' + hex(base + 80) and to_string
        offset = 64 if case == 'wrong_pc' else 80
        return (Frame().name() + f' + {offset} in section .text of {library}\n')

    gdb = SimpleNamespace(execute=execute, write=messages.append,
                          solib_name=lambda _pc: str(library), parse_and_eval=evaluate,
                          selected_inferior=lambda: SimpleNamespace(read_memory=read_memory))
    monkeypatch.setitem(sys.modules, 'gdb', gdb)
    namespace = {}
    exec(collector._ORDINARY_FAULT_SCRIPT.split('\ndef diagnostic(command):', 1)[0], namespace)
    namespace['wrapper_type_at_fault'](Frame())
    output = ''.join(messages)
    assert expected in output
    assert 'private' not in output and 'secret' not in output and 'token=' not in output
    assert '$r15' not in expressions
    assert all(size <= 96 for _address, size in reads)
    if case in {'wrong_hash', 'wrong_suffix', 'wrong_pc', 'wrong_architecture', 'wrong_python'}:
        assert reads == []
    elif case in {'wrong_instruction', 'short_instruction', 'unaligned_wrapper'}:
        assert all(address in instructions for address, _size in reads)
