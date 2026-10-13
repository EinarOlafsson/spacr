import hashlib
import json
import shutil
import subprocess
from pathlib import Path

root = Path('/mnt/wd4tb/spacr-worktrees/codex-primary-source-drain-20261007')
scratch = Path('/mnt/wd4tb/scratch/primary-source-drain-20261007')
archive = root / 'features/data/47_primary_source_reader_owner_2026-10-07'
archive.mkdir(parents=True, exist_ok=True)
module = 'spacr/qt/widgets/primary_mask_selector.py'
old_report_path = Path('/mnt/wd4tb/scratch/ci-e1f-ratchet-20261007/coverage.json')
old_document = json.loads(old_report_path.read_text())
old = old_document['files'][module]
new_document = json.loads((scratch / 'coverage.json').read_text())
new = new_document['files'][module]
old_source = (scratch / 'before_primary_mask_selector.py').read_bytes()
new_source = (root / module).read_bytes()
assert new_source == old_source + b'            if worker.isRunning():\n                worker.setParent(None)\n'
assert subprocess.check_output(['git', 'show', '28449ef0c8:' + module], cwd=root) == old_source
old_lines = set(old['executed_lines'] + old['missing_lines'])
new_lines = set(new['executed_lines'] + new['missing_lines'])
old_edges = set(map(tuple, old['executed_branches'] + old['missing_branches']))
new_edges = set(map(tuple, new['executed_branches'] + new['missing_branches']))
assert new_lines == old_lines | {227, 228}
assert new_edges == old_edges | {(227, 228), (227, -218)}
assert {227, 228} <= set(new['executed_lines'])
assert {(227, 228), (227, -218)} <= set(map(tuple, new['executed_branches']))
missing_lines = new_lines - set(old['executed_lines'] + new['executed_lines'])
missing_edges = new_edges - set(map(tuple, old['executed_branches'] + new['executed_branches']))
assert not missing_lines and not missing_edges
baseline_path = root / 'tools/coverage_baseline.json'
baseline = json.loads(baseline_path.read_text())
assert module not in baseline['modules']
old_module = {'source_sha256': hashlib.sha256(old_source).hexdigest(),
              'report_sha256': hashlib.sha256(old_report_path.read_bytes()).hexdigest(),
              'coverage_version': old_document['meta']['version'],
              'hosted_scope': 'complete e1f hosted coverage report, exact old source verified against source checkpoint',
              'file': old}
(archive / 'hosted_module_coverage.json').write_text(json.dumps(old_module, indent=2) + '\n')
for name in ('before_primary_mask_selector.py', 'native_probe.py', 'before.json', 'before.log',
             'after.json', 'after.log', 'coverage.json', 'focused-tests.log',
             'combined-sigsegv.log', 'gdb-reproducing-cohort.log'):
    shutil.copy2(scratch / name, archive / name)
shutil.copy2(root / module, archive / 'after_primary_mask_selector.py')
shutil.copy2(root / 'tests/qt/test_primary_mask_selector.py', archive / 'after_test_primary_mask_selector.py')
receipt = {
    'source_parent': '28449ef0c84c4be92cf34be18d7372e3f0d9e2c8',
    'source_commit': '6aa514f5b2',
    'before_source_sha256': hashlib.sha256(old_source).hexdigest(),
    'after_source_sha256': hashlib.sha256(new_source).hexdigest(),
    'test_sha256': hashlib.sha256((root / 'tests/qt/test_primary_mask_selector.py').read_bytes()).hexdigest(),
    'runtime_sources': {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in (
        'spacr/qt/bridge.py', 'spacr/qt/secondary_masks.py', 'tests/conftest.py', 'tests/qt/conftest.py')},
    'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    'baseline_module_entry': None,
    'baseline_policy': 'No module-specific allowance exists; unchanged numerical floor applies. The exact hosted/focused union reaches 0 statements and 0 branches uncovered.',
    'insertion_only_source_verified': True,
    'unchanged_old_universe': {'statements': len(old_lines), 'branches': len(old_edges)},
    'current_universe': {'statements': len(new_lines), 'branches': len(new_edges)},
    'new_guard_executed_lines': [227, 228],
    'new_guard_executed_branches': [[227, 228], [227, -218]],
    'focused_counts': new['summary'],
    'union_missing_lines': sorted(missing_lines), 'union_missing_branches': sorted(missing_edges),
    'focused_result': '23 passed in 19.66s, primary selector plus secondary source tests, CUDA hidden/offscreen/capped4G',
    'new_tests': ['test_timeout_parks_an_independent_reader_without_publishing_after_close',
                  'test_successful_shutdown_keeps_normal_worker_ownership',
                  'test_native_selector_deletion_preserves_reader_past_real_shutdown_timeout'],
    'before_native_exit': -6, 'after_native_exit': 0,
    'gdb_diagnostic': {'original_three_file_order_preserved': True,
        'signal': 'SIGSEGV', 'top_frame': 'Shiboken::BindingManager::releaseWrapper',
        'path': 'Object::destroy -> QtCore wrapper destructor -> QObject::event -> sendPostedEvents -> QEventLoop::exec',
        'gdb_exit': 0, 'note': 'GDB command success is not pytest success; inferior stopped on native signal, coverage incomplete.'},
    'limits': ['The independent five-second parked-thread parent deletion SIGABRT is repaired.',
               'The separate hosted/local parent-loading deferred-wrapper-deletion SIGSEGV remains unresolved.',
               'No full-suite, whole application memory, GPU, documentation/API or source-current hosted acceptance claim.'],
}
(archive / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
shutil.copy2(__file__, archive / 'archive_proof.py')
print(json.dumps({'archive': str(archive), 'receipt': receipt}, indent=2))
