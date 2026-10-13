import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-fresh-home-contracts-20261008')
probe = Path('/mnt/wd4tb/scratch/native-69-parent-race-20261008')
hosted = Path('/mnt/wd4tb/scratch/ci-69ba-coverage-20261008')
dest = root / 'features/data/43_69_native_parent_race_cpu_2026-10-08'
dest.mkdir(parents=True, exist_ok=True)
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def save_json(name, value):
    (dest / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
def pack(name, raw):
    (dest / name).write_bytes(gzip.compress(raw, mtime=0))
for name in ('replay.py', 'gdb.commands', 'replay-provenance.json', 'hosted-artifacts.json'):
    shutil.copyfile(probe / name, dest / name)
shutil.copyfile(hosted / 'coverage0-job.json', dest / 'coverage0-job.json')
shutil.copyfile(hosted / 'run-terminal-cancelled.json', dest / 'run-terminal-cancelled.json')
raw_hosted = (hosted / 'coverage0.log').read_bytes()
raw_gdb = (probe / 'gdb.log').read_bytes()
pack('coverage0.log.gz', raw_hosted)
pack('gdb.log.gz', raw_gdb)
lines = raw_hosted.splitlines(keepends=True)
pack('hosted-fault-excerpt.txt.gz', b''.join(lines[5409:5570]))
provenance = json.loads((probe / 'replay-provenance.json').read_text())
source_hashes = {}
frozen_sources = {}
for number, (path, bindings) in enumerate(provenance['source_sha256'].items()):
    raw = (root / path).read_bytes()
    expected = next(iter(bindings.values()))
    assert digest(raw) == expected and len(set(bindings.values())) == 1
    name = 'frozen-' + str(number) + '-' + Path(path).name + '.gz'
    pack(name, raw)
    source_hashes[path] = expected
    frozen_sources[path] = name
workflow = subprocess.check_output(['git', 'show', '0b8c2c4120a0ede4de568d484ea3fc9d517f1de3:.github/workflows/tests.yml'], cwd=root)
pack('frozen-tests-workflow.yml.gz', workflow)
coverage_raw = (probe / 'coverage.json').read_bytes()
coverage = json.loads(coverage_raw)
keep = ('spacr/qt/screens/make_masks.py', 'spacr/qt/widgets/primary_mask_selector.py', 'spacr/qt/bridge.py')
selected = {path: record for path, record in coverage['files'].items() if any(path.endswith(k) for k in keep)}
assert len(selected) == 3
pack('focused-three-source-coverage.json.gz', (json.dumps({'meta': coverage['meta'], 'files': selected}, sort_keys=True) + '\n').encode())
job = json.loads((dest / 'coverage0-job.json').read_text())
steps = {step['name']: step['conclusion'] for step in job['steps']}
assert job['conclusion'] == 'cancelled'
assert steps['Recover a coverage worker native backtrace'] == 'skipped'
assert steps['Upload coverage native failure evidence'] == 'skipped'
artifacts = json.loads((dest / 'hosted-artifacts.json').read_text())
assert not any(row['name'].startswith('coverage-native-') for row in artifacts['artifacts'])
receipt = {
    'run_id': 37737743925, 'head_sha': '69ba4e462f2ea60924a977b192d006ce9b70f42c',
    'job_id': 113181069132, 'current_source_anchor': '0b8c2c4120a0ede4de568d484ea3fc9d517f1de3',
    'hosted_fault': {'observed_utc': '2026-10-08T07:04:00.006410Z', 'worker': 'gw1', 'signal': 'SIGSEGV',
      'node': 'tests/qt/test_make_masks_parent_source_race_guards.py::test_native_sources_changed_during_detection_cannot_publish_a_result[parent]',
      'python_location': '_load_parent line 36: Qt waitUntil event loop, before detector mutation',
      'crashed_process_pid': None, 'cpp_stack': None,
      'primary_batch_summary': '1 failed, 442 passed, 1 warning in 271.88s',
      'excerpt_original_line_range_inclusive': [5410, 5570]},
    'capture': {'job_conclusion': 'cancelled', 'native_collector_step': 'skipped', 'native_upload_step': 'skipped',
      'coverage_process_upload': steps['Upload coverage process data'], 'ordinary_native_artifacts_observed': 0,
      'coverage0_process_artifact_id': 11533764590,
      'scope': 'Artifact API snapshot after cancellation; no native journal PID or C++ trace is recoverable from these uploaded artifacts. Coverage data archive was not downloaded.'},
    'local_replay': {'actual_nodes': 12, 'passed': 12, 'pytest_seconds': 77.70, 'native_signal': False,
      'inferior_pid': 3815218, 'inferior_exited_normally': True, 'gdb_exit_code': 1,
      'gdb_exit_reason': 'Post-exit $_siginfo is void; following bt full command reports No stack. Inferior itself exited normally.',
      'started_utc': provenance['started_utc'], 'ended_utc': provenance['ended_utc'],
      'ram_cap': '8G', 'core_limit': [0, 0], 'cuda_visible_devices': '', 'spacr_device': 'cpu',
      'qt_qpa_platform': 'offscreen', 'pytest_version': '8.4.2', 'python_version': '3.12.13', 'qt_version': '6.11.2',
      'hosted_python_version': '3.12.15', 'hosted_pytest_version': '8.4.2', 'hosted_qt_version': '6.11.2',
      'planned_node_count_in_original_provenance': 13,
      'count_correction': 'Original launcher says 13-node; actual file has nine tests plus three selected source-race tests, 12 collected/passed. Original provenance is preserved unchanged.',
      'limitations': 'One serial original-order two-file replay; does not recreate whole long-order or two-worker xdist context. Negative reproduction is not a fix or whole native acceptance.'},
    'raw_logs': {
      'coverage0.log.gz': {'sha256': digest(raw_hosted), 'bytes': len(raw_hosted)},
      'gdb.log.gz': {'sha256': digest(raw_gdb), 'bytes': len(raw_gdb)}},
    'unarchived_full_focused_coverage': {'sha256': digest(coverage_raw), 'bytes': len(coverage_raw),
      'scope': 'Only three relevant source records archived compactly; no numerical ratchet or full-module acceptance claimed.'},
    'current_sha256': source_hashes, 'frozen_sources': frozen_sources,
    'workflow_sha256': digest(workflow), 'verdict': 'HOSTED NATIVE CAUSE OPEN; BOUNDED LOCAL REPLAY NEGATIVE; NO PRODUCT CHANGE'
}
save_json('receipt.json', receipt)
shutil.copyfile(probe / 'archive.py', dest / 'archive.py')
