import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
folder = Path('/mnt/wd4tb/scratch/ci-native-faults-20261007/ordinary-reraised-proof')
env = os.environ.copy()
env['GITHUB_WORKSPACE'] = str(root)
env['GITHUB_SHA'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
env['SPACR_PROBE_EVIDENCE'] = str(folder)
env['RUNNER_TEMP'] = str(folder.parent)
command = ['gdb', '-q', '-nx', '-nh', '-batch', '-iex', 'set auto-load safe-path /dev/null',
           '-iex', 'set debuginfod enabled off', '-ex', 'run', '-ex', 'continue',
           '-ex', f'generate-core-file {folder / "core.controlled"}',
           '--args', sys.executable, str(folder / 'probe.py')]
with (folder / 'generate.log').open('w') as output:
    result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=30)
if result.returncode:
    raise RuntimeError(result.returncode)
core = folder / 'core.controlled'
core_size = core.stat().st_size
try:
    with (folder / 'collector.log').open('w') as output:
        result = subprocess.run([sys.executable, 'tools/collect_qt_native_backtrace.py',
                                 '--ordinary', '--evidence-dir', str(folder)], env=env,
                                stdout=output, stderr=subprocess.STDOUT, timeout=110)
    report = (folder / 'native-core-backtrace.txt').read_text()
    assert result.returncode == 0
    assert 'matched_process=' in report, report
    assert 'gdb_exit=0' in report, report
    assert 'Thread 1' in report and 'kill' in report, report
    assert 'ordinary_fault_frame selection=after_sigtramp' in report, report
    marker = [line for line in report.splitlines() if line.startswith('ordinary_fault_frame ')][0]
    assert 'pthread' not in marker and 'raise' not in marker, marker
    assert 'rip' in report and 'rdi' in report, report
    assert 'ordinary fault diagnostic unsupported' not in report, report
    assert report.encode().__len__() <= 4 * 1024 * 1024
    receipt = {
        'git_head': env['GITHUB_SHA'], 'collector_sha256': hashlib.sha256(
            (root / 'tools/collect_qt_native_backtrace.py').read_bytes()).hexdigest(),
        'journal_sha256': hashlib.sha256(
            (root / 'tools/pytest_plugins/qt_serial_rss_journal.py').read_bytes()).hexdigest(),
        'executable': str(Path(sys.executable).resolve()), 'core_bytes': core_size,
        'core_generated_in_owned_gdb_child': True, 'global_core_settings_changed': False,
        'core_upload': False, 'matched_owned_process': True, 'gdb_exit': 0,
        'original_fault_frame': marker, 'bounded_register_disassembly': True,
        'native_fault': 'controlled ctypes SIGSEGV reraised by CPython faulthandler; no spaCR causation or hosted-crash repair claim',
    }
    (folder / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
finally:
    core.unlink(missing_ok=True)
