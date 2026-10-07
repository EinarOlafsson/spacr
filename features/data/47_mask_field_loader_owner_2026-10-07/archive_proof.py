import ast
import difflib
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

repo = Path('/mnt/wd4tb/spacr-worktrees/codex-mask-load-drain-20261007')
scratch = Path('/mnt/wd4tb/scratch/mask-load-drain-20261007')
out = repo / 'features/data/47_mask_field_loader_owner_2026-10-07'
out.mkdir(parents=True, exist_ok=True)
path = 'spacr/qt/screens/make_masks.py'
old = (scratch / 'before_make_masks.py').read_bytes()
current = (repo / path).read_bytes()
assert hashlib.sha256(old).hexdigest() == '8ff012eff063bbaa9915ad5853a9e07f53aa6d59d43aaf4eb9f0b1601246b933'
needle = b'            drain_thread(worker, timeout_ms=5000)\n        self._loading = False\n'
replacement = b'            if not drain_thread(worker, timeout_ms=5000):\n                worker.setParent(None)\n        self._loading = False\n'
assert old.count(needle) == 1 and current == old.replace(needle, replacement)
assert subprocess.check_output(['git', 'show', '0324166b59:' + path], cwd=repo) == old
focused = json.loads((scratch / 'coverage-final.json').read_text())['files'][path]
prior_prefix = 'features/data/43_six_module_coverage_cpu_2026-10-06/'
def git_bytes(name):
    return subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=repo)
prior_report = gzip.decompress(git_bytes(prior_prefix + 'focused222.json.gz'))
prior = json.loads(prior_report)['files'][path]
prior_source = subprocess.check_output(['git', 'show', '7a6c2c6f333a71996a972583be008e3e94a5c69b:' + path], cwd=repo)
assert prior_source == old
hosted_path = Path('/mnt/wd4tb/scratch/ci-e1f-ratchet-20261007/coverage.json')
hosted = json.loads(hosted_path.read_text())['files'][path]
hosted_sha = 'e1f54c80650f60942ba6c98d4e999b36a3fa2846'
hosted_source = subprocess.check_output(['git', 'show', hosted_sha + ':' + path], cwd=repo)
records = [('focused_now', current, focused), ('focused222', old, prior), ('hosted_e1', hosted_source, hosted)]
lines = set(focused['executed_lines'] + focused['missing_lines'])
arcs = set(map(tuple, focused['executed_branches'] + focused['missing_branches']))
executed, taken = set(), set()
mapping_receipts = {}
for label, source, record in records:
    left, right = source.decode().splitlines(), current.decode().splitlines()
    matching = difflib.SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
    mapping = {a + i + 1: b + i + 1 for a, b, size in matching for i in range(size)}
    changed = sorted(set(range(1, len(left) + 1)) - set(mapping))
    if label == 'focused_now':
        assert changed == []
    else:
        assert len(changed) == 1 and left[changed[0] - 1] == '            drain_thread(worker, timeout_ms=5000)'
    assert 16643 not in mapping.values() or label == 'focused_now', 'Never inherit the changed drain condition'
    def mapped(n):
        value = mapping.get(abs(n))
        return None if value is None else value * (1 if n > 0 else -1)
    old_lines = {mapped(n) for n in record['executed_lines'] + record['missing_lines'] if mapped(n) is not None}
    old_arcs = {tuple(mapped(n) for n in edge) for edge in record['executed_branches'] + record['missing_branches'] if all(mapped(n) is not None for n in edge)}
    assert old_lines <= lines
    ignored_arcs = sorted(old_arcs - arcs)
    executed |= {mapped(n) for n in record['executed_lines'] if mapped(n) is not None}
    taken |= {tuple(mapped(n) for n in edge) for edge in record['executed_branches'] if all(mapped(n) is not None for n in edge)} & arcs
    mapping_receipts[label] = {'source_sha256': hashlib.sha256(source).hexdigest(),
        'all_other_old_lines_identical': True, 'changed_drain_lines_not_inherited': changed, 'statements': len(old_lines), 'branches': len(old_arcs),
        'discarded_arcs_outside_current_universe': ignored_arcs}
    compressed = gzip.compress(json.dumps({'files': {path: record}}, sort_keys=True).encode(), mtime=0)
    (out / (label + '.json.gz')).write_bytes(compressed)
missing_lines, missing_arcs = sorted(lines - executed), sorted(arcs - taken)
baseline = json.loads((repo / 'tools/coverage_baseline.json').read_text())
allowance = baseline['modules'][path]
assert len(missing_lines) <= allowance['uncovered_statements']
assert len(missing_arcs) <= allowance['uncovered_branches']
assert {16643, 16644} <= set(focused['executed_lines'])
assert {(16643, 16644), (16643, 16645)} <= set(map(tuple, focused['executed_branches']))
for name in ('native_probe.py', 'run_probe.py', 'before.json', 'before.log', 'after-final.json', 'after-final.log', 'focused-final-tests.log'):
    shutil.copy2(scratch / name, out / name)
(out / 'before_make_masks.py.gz').write_bytes(gzip.compress(old, mtime=0))
(out / 'after_make_masks.py.gz').write_bytes(gzip.compress(current, mtime=0))
shutil.copy2(repo / 'tests/qt/test_make_masks_loader_shutdown.py', out / 'test_make_masks_loader_shutdown.py')
shutil.copy2(__file__, out / 'archive_proof.py')
def docs(source):
    return [(type(node).__name__, node.name, ast.get_docstring(node)) for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
assert docs(old) == docs(current)
def ruff(source):
    result = subprocess.run(['/home/olafsson/.local/bin/ruff', 'check', '--output-format', 'json', '--stdin-filename', path, '-'],
                            cwd=repo, input=source, capture_output=True)
    assert result.returncode in (0, 1)
    return json.loads(result.stdout)
old_ruff, new_ruff = ruff(old), ruff(current)
assert len(old_ruff) == len(new_ruff)
for before, after in zip(old_ruff, new_ruff):
    assert before['code'] == after['code'] and before['message'] == after['message']
    for field in ('location', 'end_location'):
        assert after[field]['column'] == before[field]['column']
        assert after[field]['row'] == before[field]['row'] + (1 if before[field]['row'] > 16643 else 0)
receipt = {'source_parent': '0324166b59', 'source_commit': 'd1cb435487', 'initial_superseded_candidate': '79cf17bd6b',
    'before_sha256': hashlib.sha256(old).hexdigest(), 'after_sha256': hashlib.sha256(current).hexdigest(),
    'test_sha256': hashlib.sha256((repo / 'tests/qt/test_make_masks_loader_shutdown.py').read_bytes()).hexdigest(),
    'runtime_sources': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in (
        'spacr/qt/bridge.py', 'spacr/qt/widgets/primary_mask_selector.py', 'tests/conftest.py', 'tests/qt/conftest.py')},
    'baseline_sha256': hashlib.sha256((repo / 'tools/coverage_baseline.json').read_bytes()).hexdigest(),
    'original_allowance': allowance,
    'hosted_sha': hosted_sha, 'original_hosted_report_sha256': hashlib.sha256(hosted_path.read_bytes()).hexdigest(),
    'prior_focused_archive': prior_prefix, 'prior_focused_git_blob_sha256': hashlib.sha256(git_bytes(prior_prefix + 'focused222.json.gz')).hexdigest(),
    'mapping': mapping_receipts, 'statements': len(lines), 'branches': len(arcs),
    'missing_lines': missing_lines, 'missing_branches': missing_arcs,
    'new_guard_lines_covered': [16643, 16644], 'new_guard_branches_covered': [[16643,16644], [16643,16645]], 'changed_drain_line_inherited': False,
    'callable_docstrings_unchanged': True, 'unchanged_existing_ruff_findings': len(old_ruff),
    'focused_result': '5 passed in25.16s; four shutdown tests including real deleted wrapper plus existing large decode GUI-responsiveness case',
    'native_before_exit': -6, 'native_after_exit': 0,
    'scope': 'Independent busy field-loader native ownership repair; no claim to resolve small-image Shiboken SIGSEGV, installed Save or full serial suite.'}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
