import gzip
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
SOURCE = Path('/mnt/wd4tb/scratch/theme-growth-thore-20261006')
TARGET = ROOT / 'features/data/47_serial_native_failure_2026-10-06'
TARGET.mkdir(parents=True, exist_ok=True)


def compress(name, path):
    (TARGET / (name + '.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))


for name in ['pytest.log', 'file-rss.jsonl', 'host-memory.log']:
    compress('protected-' + name, SOURCE / 'n47' / name)
compress('protected-job.log', SOURCE / 'n47-serial-37457998285.log')
for name in ['n47-puncta-false-replay.log', 'n47-puncta-file-replay.log',
             'n47-eight-failed-replay.log', 'n47-distribution-failed-replay.log',
             'n47-tail17-pytest.log', 'n47-tail17-rss.jsonl',
             'n47-first-failure17-pytest.log', 'n47-first-failure17-rss.jsonl']:
    compress(name, SOURCE / name)
for name in ['n47-bounded-replay-manifest.json', 'faulthandler_fd_repro.py']:
    shutil.copyfile(SOURCE / name, TARGET / name)
for folder in ['first', 'second']:
    compress('handler-' + folder + '-crash.log',
             SOURCE / 'fd-repro' / folder / 'spacr-crash.log')
compress('handler-repro-stdout.log', SOURCE / 'fd-repro/stdout.log')
compress('handler-repro-stderr.log', SOURCE / 'fd-repro/stderr.log')
for name in ['handler-first-crash.log', 'handler-second-crash.log',
             'handler-repro-stdout.log', 'handler-repro-stderr.log']:
    (TARGET / name).unlink(missing_ok=True)

rows = [json.loads(line) for line in (SOURCE / 'n47/file-rss.jsonl').read_text().splitlines()]
last = rows[-1]
failures = []
for line in (SOURCE / 'n47/pytest.log').read_text().splitlines():
    match = re.match(r'(tests/qt/\S+) FAILED ', line)
    if match:
        failures.append(match.group(1))
assert len(failures) == 9
assert last['file'] == 'tests/qt/test_make_masks_center_pixel_puncta.py'
assert last['rss_bytes'] == 5432274944
receipt = {
    'run_id': 37457998285, 'job_id': 112250232405,
    'artifact_id': 11425588571,
    'source_sha': '23750631790924590490e356144c8da72ac805af',
    'conclusion': 'failure', 'exit': 139,
    'last_node': 'tests/qt/test_make_masks_center_pixel_puncta.py::test_choose_detect_save_and_undo_keeps_parents_and_exports_exact_values[False]',
    'last_boundary': last,
    'last_boundary_rss_gib': round(last['rss_bytes'] / 2**30, 6),
    'guard_gib': 10.8, 'hard_scope_gib': 12,
    'explicit_failed_nodes_before_native_death': failures,
    'limits': [
        'Old terminal log lacks failed assertion details and native traceback.',
        'All nine failures and puncta pass narrowly at the exact old source.',
        'Original-order 17-file windows pass 268 and 159 cases at lower RSS.',
        'The protected failure is not accepted as fixed.',
        'Missing traceback redirection reproduced with actual app crash helper.',
        'Diagnostic repair preserves failures and a dedicated fatal signal FD.',
        'No full local Qt run, GC change, test-order change or guard waiver.'],
}
(TARGET / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(TARGET / 'REPRODUCE.txt').write_text(
    'Protected run source and exact original-order journals are frozen here.\n'
    'Replay manifests list two bounded windows; original stdout/journal hashes\n'
    'refer to uncompressed files. Gzip artifacts are lossless, mtime=0.\n'
    'The last boundary RSS is 5180.62 MiB = 5.059 GiB, below10.8GiB guard;\n'
    'HWM is 5.179GiB. These figures must not be interchanged. Last raw job\n'
    'MEMORY sample reports5183MiB RSS,10118MiB available,swap_used0,oom_kills0.\n'
    'The raw job includes an abrupt disk-free drop; its cause is not proven.\n'
    'faulthandler_fd_repro.py intentionally SIGSEGVs a subprocess: run only\n'
    'CUDA-hidden/offscreen under4G, PYTHONPATH=. with a new scratch\n'
    'SPACR_FAULT_REPRO_ROOT. It reproduces app redirection, not the native\n'
    'product failure. Source reference for the redirection/closed-handle\n'
    'diagnostic is ab8690636d; source-scoped serial plugin changes are\n'
    '075e2c6cba and fac1d5ef28. Current tests/test_qt_serial_rss_journal.py\n'
    'checks actual redirection then SIGSEGV capture in an owned artifact FD.\n'
    'A new hosted serial run is required; do not certify current HEAD using\n'
    'the older237 result or a short replay. Do not run full Qt locally.\n')
shutil.copyfile(Path(__file__), TARGET / 'archive.py')
(TARGET / 'manifest.json').write_text(json.dumps({'sha256': {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(TARGET.iterdir()) if p.is_file() and p.name != 'manifest.json'
}}, indent=2) + '\n')
print(json.dumps({'artifacts': len(list(TARGET.iterdir())),
                  'old_failure_nodes': len(failures),
                  'protected_last_boundary_rss_gib': receipt['last_boundary_rss_gib']}))
