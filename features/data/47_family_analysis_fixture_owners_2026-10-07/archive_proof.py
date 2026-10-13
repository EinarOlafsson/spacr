"""Archive two fixture ownership repairs and bounded unchanged-assertion proof."""
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

root = Path('/mnt/wd4tb/spacr-worktrees/codex-settings-fixture-owners-20261007')
scratch = Path(__file__).resolve().parent
archive = root / 'features/data/47_family_analysis_fixture_owners_2026-10-07'
archive.mkdir(parents=True, exist_ok=True)
base = 'e2983466b15a875b07d978b15cb19dd7c9576670'
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
tests = ['tests/qt/test_the_classifier_family_is_spelled_out.py',
         'tests/qt/test_the_analysis_unit_locks_the_panel.py']
bindings = {}
test_count = 0
for path in tests:
    before = subprocess.check_output(['git', 'show', base + ':' + path], cwd=root)
    after = (root / path).read_bytes()
    name = Path(path).name
    (archive / ('before_' + name)).write_bytes(before)
    (archive / ('after_' + name)).write_bytes(after)
    def test_nodes(data):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(data))
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    old_nodes, new_nodes = test_nodes(before), test_nodes(after)
    assert old_nodes == new_nodes
    test_count += len(old_nodes)
    bindings[path] = {'before_sha256': hashlib.sha256(before).hexdigest(),
                      'after_sha256': hashlib.sha256(after).hexdigest(),
                      'unchanged_test_functions': len(old_nodes)}
unchanged = {}
for path in ['spacr/qt/screens/settings_model.py', 'spacr/qt/widgets/ambient.py',
             'spacr/qt/theme.py', 'spacr/qt/preferences.py', 'tests/conftest.py',
             'tests/qt/conftest.py', 'tools/pytest_plugins/qt_serial_rss_journal.py',
             'tests/qt/test_field_fade.py']:
    data = (root / path).read_bytes()
    assert subprocess.check_output(['git', 'show', base + ':' + path], cwd=root) == data
    unchanged[path] = hashlib.sha256(data).hexdigest()
for phase in ('before', 'after', 'final'):
    for suffix in ('.jsonl', '.log'):
        shutil.copyfile(scratch / (phase + suffix), archive / (phase + suffix))
matched = []
for phase in ('before', 'after', 'final'):
    entries = [json.loads(line) for line in (archive / (phase + '.jsonl')).read_text().splitlines()]
    boundaries = [entry for entry in entries if entry['event'] == 'file_end']
    finish = next(entry for entry in entries if entry['event'] == 'session_finish')
    assert finish['exitstatus'] == 0 and len(boundaries) == 3
    matched.append({'phase': phase, 'boundaries': boundaries, 'session_finish': finish})
receipt = {'baseline_commit': base, 'tests_commit': source_commit,
           'app_source_changed': False, 'guard_or_gc_changes': False,
           'bindings': bindings, 'unchanged_source_sha256': unchanged,
           'unchanged_test_function_AST_count': test_count,
           'phases': matched,
           'acceptance': {'before_matched_cases': 3, 'after_matched_cases': 3, 'final_full_files_and_sentinel_cases': 27},
           'commands': {'environment': "CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
                        'runner': 'tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly -p tools.pytest_plugins.qt_serial_rss_journal',
                        'matched_nodes': [tests[0] + '::test_both_families_are_spelled_out', tests[1] + '::TestChoosingCellLocksThem::test_the_values_are_applied', 'tests/qt/test_field_fade.py::test_a_rendered_field_follows_the_cubic_across_its_width'],
                        'final_nodes': tests + ['tests/qt/test_field_fade.py::test_a_rendered_field_follows_the_cubic_across_its_width'],
                        'cwd': str(root), 'journal_environment': 'SPACR_QT_SERIAL_RSS_JOURNAL=' + str(scratch) + '/{phase}.jsonl'},
           'limitations': ['Cached fixture snapshots are not a direct fresh Qt census at every journal boundary.',
                           'Matched after run predates import-whitespace formatting; final27-case run binds final source. Fixture/test AST is unchanged by formatting.',
                           'RSS is a bounded sampled observation, not whole-suite memory acceptance or native crash causation.',
                           'No injected GC, test order, stylesheet, memory/assertion guard, or production changes.']}
(archive / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
shutil.copyfile(__file__, archive / 'archive_proof.py')
print(archive, test_count, 'unchanged test function ASTs')
