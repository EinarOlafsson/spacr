"""Verify lifecycle evidence using frozen source and previous denominators."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess

import coverage
from coverage.parser import PythonParser

INITIAL = 'features/data/664_665_changed_runtime_coverage_2026-10-08'
PRIOR = 'features/data/664_665_changed_runtime_coverage_supplement_2026-10-08'
LIFECYCLE = 'features/data/664_665_worker_lifecycle_supplement_2026-10-08'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--git', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ['GIT_NO_LAZY_FETCH'] = '1'
    assert coverage.__version__ == '7.16.0', coverage.__version__
    def blob(path):
        return subprocess.check_output(['git', 'show', args.git + ':' + path], cwd=args.repo)
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    def frozen(prefix, path):
        return gzip.decompress(blob(prefix + '/' + path))
    receipt_raw = blob(LIFECYCLE + '/receipt.json')
    receipt = json.loads(receipt_raw)
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.git, LIFECYCLE], cwd=args.repo).decode().splitlines()
    assert {p[len(LIFECYCLE) + 1:] for p in tree} == set(receipt['payloads']) | {'receipt.json'}
    assert len(receipt['payloads']) == 52
    raw = {}
    payloads = {}
    for path, expected in receipt['payloads'].items():
        data = blob(LIFECYCLE + '/' + path)
        assert sha(data) == expected['sha256'] and len(data) == expected['bytes'], path
        decoded = gzip.decompress(data)
        assert sha(decoded) == expected['raw_sha256'] and len(decoded) == expected['raw_bytes'], path
        raw[path] = decoded
        payloads[path] = expected
    logs = {}
    for path, data in raw.items():
        if not path.startswith('logs/'):
            continue
        text = data.decode()
        summary_rows = []
        for line in text.splitlines():
            counts = {name: int(number) for number, name in re.findall(r'(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b', line)}
            if counts and re.search(r'\bin \d+(?:\.\d+)?s\b', line):
                summary_rows.append({'text': line, 'counts': counts})
        logs[path] = {'raw_sha256': sha(data), 'summary_rows': summary_rows, 'source': re.findall(r'^Source: ([0-9a-f]{40})', text, re.M), 'terminal_exit_status': re.findall(r'^Terminal exit status: (\d+)', text, re.M)}
    r11 = logs['logs/worker-runtime-coverage-r11.log.gz']
    assert not r11['summary_rows'] and r11['terminal_exit_status'] == ['4']
    assert 'unrecognized arguments: --cov=spacr --cov-branch --cov-report=' in raw['logs/worker-runtime-coverage-r11.log.gz'].decode()
    for name, counts in [('worker-runtime-coverage-r12', {'passed': 4, 'deselected': 10}), ('worker-runtime-coverage-r13', {'passed': 7, 'deselected': 10}), ('worker-runtime-coverage-r14', {'passed': 19}), ('worker-new-lifecycle-py312-r1', {'passed': 9, 'deselected': 10})]:
        log = logs['logs/' + name + '.log.gz']
        assert log['summary_rows'][-1]['counts'] == counts
        assert log['terminal_exit_status'] == ['0']
    assert logs['logs/worker-runtime-coverage-r14.log.gz']['source'] == [receipt['source_commit']]
    assert logs['logs/worker-new-lifecycle-py312-r1.log.gz']['source'] == [receipt['source_commit']]
    report = json.loads(raw['reports/worker-changed-runtime-coverage-independent-r1345671014.json.gz'])
    previous_report_raw = frozen(PRIOR, 'reports/worker-changed-runtime-coverage-independent-r13456710.json.gz')
    previous_report = json.loads(previous_report_raw)
    initial_report_raw = frozen(INITIAL, 'reports/worker-changed-runtime-coverage-r134567.json.gz')
    initial_report = json.loads(initial_report_raw)
    assert report['totals'] == receipt['changed_coverage']
    assert report['comparison_base_commit'] == initial_report['comparison_base_commit']
    assert report['config_sha256'] == sha(frozen(INITIAL, 'source/current/.coveragerc.gz'))
    data_path = [path for path in raw if path.startswith('coverage/') and sha(raw[path]) == report['data_sha256']]
    assert len(data_path) == 1
    connection = sqlite3.connect(':memory:')
    connection.deserialize(raw[data_path[0]])
    db_files = dict(connection.execute('select path,id from file'))
    old_modules = {module['path']: module for module in initial_report['changed_modules']}
    prior_modules = {module['path']: module for module in previous_report['changed_modules']}
    totals = dict.fromkeys(report['totals'], 0)
    modules = []
    assert {m['path'] for m in report['changed_modules']} == set(old_modules)
    for module in report['changed_modules']:
        path = module['path']
        source = frozen(INITIAL, 'source/current/' + path + '.gz')
        assert sha(source) == module['source_sha256'] == old_modules[path]['source_sha256']
        for key in ('changed_executable_lines', 'new_or_changed_branch_arcs'):
            assert module[key] == old_modules[path][key], (path, key)
        lines, hits, misses = map(set, (module['changed_executable_lines'], module['hit_lines'], module['missed_lines']))
        arcs, arc_hits, arc_misses = [set(map(tuple, module[key])) for key in ('new_or_changed_branch_arcs', 'hit_arcs', 'missed_arcs')]
        assert hits.isdisjoint(misses) and hits | misses == lines
        assert arc_hits.isdisjoint(arc_misses) and arc_hits | arc_misses == arcs
        observed = set()
        for name, fid in db_files.items():
            if name == path or name.endswith('/' + path):
                observed.update(connection.execute('select fromno,tono from arc where file_id=?', (fid,)))
        source_parser = PythonParser(text=source.decode(), filename=path)
        source_parser.parse_source()
        source_parser.arcs()
        observed = source_parser.translate_arcs(observed)
        assert observed & arcs == arc_hits, path
        observed_lines = {number for pair in observed for number in pair if number > 0}
        assert observed_lines & lines == hits, path
        for number, text in module['missed_source'].items():
            assert source.decode().splitlines()[int(number) - 1] == text, (path, number)
        for key in totals:
            totals[key] += len(module[key])
        newly_hit_lines = sorted(hits - set(prior_modules[path]['hit_lines']))
        newly_hit_arcs = sorted(map(list, arc_hits - set(map(tuple, prior_modules[path]['hit_arcs']))))
        assert set(prior_modules[path]['hit_lines']) <= hits
        assert set(map(tuple, prior_modules[path]['hit_arcs'])) <= arc_hits
        modules.append({'path': path, 'source_sha256': sha(source), 'missed_lines': sorted(misses), 'missed_arcs': sorted(map(list, arc_misses)), 'newly_hit_lines': newly_hit_lines, 'newly_hit_arcs': newly_hit_arcs})
    assert totals == report['totals'] and len(modules) == 29
    assert sum(len(m['newly_hit_lines']) for m in modules) == 6
    assert sum(len(m['newly_hit_arcs']) for m in modules) == 3
    connection.close()
    final_tests = []
    for path in ['tests/test_item60_measure_coverage.py', 'tests/test_parallel_queue_lifecycle_boundaries.py', 'tests/test_sequencing_full_coverage.py', 'tests/test_worker_writer_caller_boundaries.py', 'tests/test_worker_writer_failure_boundaries.py']:
        final = raw['tests/' + path + '.gz']
        previous = frozen(PRIOR, 'tests/' + path + '.gz')
        final_tests.append({'path': path, 'final_sha256': sha(final), 'prior_sha256': sha(previous), 'changed_since_prior': final != previous})
    assert [m['path'] for m in final_tests if m['changed_since_prior']] == ['tests/test_parallel_queue_lifecycle_boundaries.py']
    preservation = []
    for path, count in [('tests/test_item60_measure_coverage.py', 25), ('tests/test_sequencing_full_coverage.py', 76)]:
        def definitions(data):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(data).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')}
        old = definitions(frozen(INITIAL, 'tests/original/' + path + '.gz'))
        final = definitions(raw['tests/' + path + '.gz'])
        assert len(old) == count and all(final.get(name) == body for name, body in old.items())
        preservation.append({'path': path, 'original_test_definitions_unchanged': count})
    lifecycle_path = 'tests/test_parallel_queue_lifecycle_boundaries.py'
    prior_tree = ast.parse(frozen(PRIOR, 'tests/' + lifecycle_path + '.gz'))
    final_tree = ast.parse(raw['tests/' + lifecycle_path + '.gz'])
    prior_functions = {node.name: ast.dump(node.body, include_attributes=False) if isinstance(node.body, ast.AST) else [ast.dump(item, include_attributes=False) for item in node.body] for node in prior_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    final_functions = {node.name: [ast.dump(item, include_attributes=False) for item in node.body] for node in final_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert len(prior_functions) == 11 and all(final_functions.get(name) == body for name, body in prior_functions.items())
    result = {'passed': True, 'review_git_commit': args.git, 'coverage_version': coverage.__version__, 'receipt_sha256': sha(receipt_raw), 'payloads_verified': payloads, 'logs_verified': logs, 'report_totals': totals, 'merged_data': data_path[0], 'production_modules': modules, 'source_and_denominators_identical_to_verified_initial_proof': True, 'initial_report_sha256': sha(initial_report_raw), 'previous_report_sha256': sha(previous_report_raw), 'final_tests': final_tests, 'original_test_definitions_preserved': preservation, 'prior_lifecycle_function_bodies_preserved': len(prior_functions), 'limitations': receipt['limitations'] + ['No application integration, GPU/native crash acceptance, whole-module ratchet or full hosted/serial acceptance.', 'Private whole-tree documentation/tool equality and patch identity inside the private candidate remain declarations, not independently reconstructed private Git-tree comparisons.', '19-case final cohort overlaps prior cohorts and the nine-case Python 3.12 subset; no unique-total sum is claimed.']}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'payloads': len(payloads), 'logs': len(logs), 'source_modules': len(modules), 'newly_hit_lines': 6, 'newly_hit_arcs': 3, 'final_totals': totals, 'r11_invocation_failure_preserved': True, 'original_test_definitions_preserved': 101, 'prior_lifecycle_function_bodies_preserved': 11}, indent=2))


if __name__ == '__main__':
    main()
