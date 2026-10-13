"""Read frozen Git evidence without importing spaCR or running tests."""
import argparse
import ast
import difflib
import gzip
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import coverage
from coverage.parser import PythonParser

P = 'features/data/'
INITIAL = P + '664_665_changed_runtime_coverage_2026-10-08'
SUPPLEMENT = P + '664_665_changed_runtime_coverage_supplement_2026-10-08'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--git', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ["GIT_NO_LAZY_FETCH"] = "1"
    def blob(path):
        return subprocess.check_output(['git', 'show', args.git + ':' + path], cwd=args.repo)
    def optional_blob(path):
        result = subprocess.run(['git', 'show', args.git + ':' + path], cwd=args.repo, capture_output=True)
        return None if result.returncode else result.stdout
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    records = {}
    raw_by_archive = {}
    receipts = {}
    for prefix in (INITIAL, SUPPLEMENT):
        receipt_raw = blob(prefix + '/receipt.json')
        receipt = json.loads(receipt_raw)
        receipts[prefix] = receipt
        manifest = receipt['payloads']
        tree = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.git, prefix], cwd=args.repo).decode().splitlines()
        actual = {p[len(prefix) + 1:] for p in tree}
        assert actual == set(manifest) | {'receipt.json'}, (prefix, actual - set(manifest))
        raw = {}
        payload_results = {}
        total_bytes = 0
        for path, expected in manifest.items():
            data = blob(prefix + '/' + path)
            assert len(data) == expected['bytes'] and sha(data) == expected['sha256'], path
            decoded = gzip.decompress(data)
            assert len(decoded) == expected['raw_bytes'] and sha(decoded) == expected['raw_sha256'], path
            raw[path] = decoded
            payload_results[path] = {'sha256': sha(data), 'raw_sha256': sha(decoded), 'bytes': len(data), 'raw_bytes': len(decoded)}
            total_bytes += len(data)
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
            logs[path] = {
                'raw_sha256': sha(data), 'summary_rows': summary_rows,
                'failed_nodes': re.findall(r'^FAILED (\S+)', text, re.M),
                'terminal_exit_status': re.findall(r'^Terminal exit status: (\d+)', text, re.M),
                'source_declarations': re.findall(r'^Source: ([0-9a-f]{40})', text, re.M),
            }
        records[prefix] = {
            'receipt_sha256': sha(receipt_raw),
            'receipt_git_blob': subprocess.check_output(['git', 'rev-parse', args.git + ':' + prefix + '/receipt.json'], cwd=args.repo).decode().strip(),
            'payload_count': len(manifest), 'payload_bytes': total_bytes,
            'logs': logs, 'payloads_verified': payload_results,
            'declared_scope': receipt['scope'], 'declared_limitations': receipt['limitations'],
        }
        raw_by_archive[prefix] = raw
    initial = receipts[INITIAL]
    supplement = receipts[SUPPLEMENT]
    iraw, sraw = raw_by_archive[INITIAL], raw_by_archive[SUPPLEMENT]
    assert len(iraw) == 203 and len(sraw) == 25
    bindings = []
    for path, expected in initial['source_bindings'].items():
        frozen_path = ('tests/current/' if path.startswith('tests/') else 'source/current/') + path + '.gz'
        assert sha(iraw[frozen_path]) == expected, path
        current = optional_blob(path)
        bindings.append({'path': path, 'frozen_sha256': expected, 'review_head_sha256': None if current is None else sha(current), 'review_head_present': current is not None, 'matches_review_head': current is not None and sha(current) == expected})
    assert len(bindings) == 36
    records[INITIAL]['source_bindings_verified'] = bindings
    records[SUPPLEMENT]['source_bindings_schema'] = 'No source_bindings map: report production hashes resolve to initial frozen source; five final tests are separately frozen.'
    final_tests = []
    for path in initial['changed_test_files']:
        final = sraw['tests/' + path + '.gz']
        prior = iraw['tests/current/' + path + '.gz']
        final_tests.append({'path': path, 'initial_sha256': sha(prior), 'final_sha256': sha(final), 'changed_since_initial': prior != final, 'matches_review_head': final == optional_blob(path)})
    assert sum(x['changed_since_initial'] for x in final_tests) == 2
    records[SUPPLEMENT]['final_tests_verified'] = final_tests
    base_bindings = []
    for path in initial['source_bindings']:
        if not path.startswith('spacr/'):
            continue
        base_path = 'source/comparison-base/' + path + '.gz'
        frozen = iraw[base_path]
        old = subprocess.run(['git', 'show', initial['comparison_base_commit'] + ':' + path], cwd=args.repo, capture_output=True)
        base_bindings.append({'path': path, 'frozen_sha256': sha(frozen), 'comparison_git_available': old.returncode == 0, 'comparison_git_matches': None if old.returncode else old.stdout == frozen})
        if not old.returncode:
            assert old.stdout == frozen, path
    records[INITIAL]['comparison_sources_verified'] = base_bindings
    for key, expected in initial['coverage_inputs'].items():
        assert sha(iraw['coverage/' + key + '.gz']) == expected, key
    records[INITIAL]['coverage_input_hashes_verified'] = len(initial['coverage_inputs'])
    reports = []
    for prefix, raw, report_path in [
        (INITIAL, iraw, 'reports/worker-changed-runtime-coverage-r134567.json.gz'),
        (SUPPLEMENT, sraw, 'reports/worker-changed-runtime-coverage-independent-r13456710.json.gz'),
    ]:
        report = json.loads(raw[report_path])
        assert report['totals'] == receipts[prefix]['changed_coverage']
        assert report['comparison_base_commit'] == initial['comparison_base_commit']
        assert report['config_sha256'] == sha(iraw['source/current/.coveragerc.gz'])
        matched_data = [p for p, b in raw.items() if p.startswith('coverage/') and sha(b) == report['data_sha256']]
        assert len(matched_data) == 1, matched_data
        sums = dict.fromkeys(report['totals'], 0)
        modules = []
        connection = sqlite3.connect(':memory:')
        connection.deserialize(raw[matched_data[0]])
        db_files = dict(connection.execute('select path,id from file'))
        for module in report['changed_modules']:
            path = module['path']
            source = iraw['source/current/' + path + '.gz']
            assert sha(source) == module['source_sha256'], path
            lines = set(module['changed_executable_lines'])
            hits, misses = set(module['hit_lines']), set(module['missed_lines'])
            arcs = set(map(tuple, module['new_or_changed_branch_arcs']))
            arc_hits, arc_misses = set(map(tuple, module['hit_arcs'])), set(map(tuple, module['missed_arcs']))
            assert hits.isdisjoint(misses) and hits | misses == lines, path
            assert arc_hits.isdisjoint(arc_misses) and arc_hits | arc_misses == arcs, path
            candidates = [fid for file_path, fid in db_files.items() if file_path.endswith('/' + path) or file_path == path]
            observed = set()
            for fid in candidates:
                observed.update(connection.execute('select fromno,tono from arc where file_id=?', (fid,)))
            parser = PythonParser(text=source.decode(), filename=path)
            parser.parse_source()
            parser.arcs()
            old_source = iraw['source/comparison-base/' + path + '.gz'].decode()
            matcher = difflib.SequenceMatcher(None, old_source.splitlines(), source.decode().splitlines(), autojunk=False)
            changed = set()
            old_to_new = {}
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    old_to_new.update((i + 1, j + 1) for i, j in zip(range(i1, i2), range(j1, j2)))
                elif tag in {'insert', 'replace'}:
                    changed.update(range(j1 + 1, j2 + 1))
            assert parser.statements & changed == lines, path
            old_parser = PythonParser(text=old_source, filename=path)
            old_parser.parse_source()
            old_branches = {line for line, count in old_parser.exit_counts().items() if count > 1}
            def translate(number):
                mapped = old_to_new.get(abs(number))
                return None if mapped is None else mapped if number > 0 else -mapped
            old_arcs = {tuple(map(translate, pair)) for pair in old_parser.arcs() if pair[0] in old_branches}
            old_arcs = {pair for pair in old_arcs if None not in pair}
            branches = {line for line, count in parser.exit_counts().items() if count > 1}
            assert {pair for pair in parser.arcs() if pair[0] in branches} - old_arcs == arcs, path
            observed = parser.translate_arcs(observed)
            assert observed & arcs == arc_hits, (path, observed & arcs ^ arc_hits)
            observed_lines = {number for pair in observed for number in pair if number > 0}
            assert observed_lines & lines == hits, (path, observed_lines & lines ^ hits)
            source_lines = source.decode().splitlines()
            for number, text in module.get('missed_source', {}).items():
                assert source_lines[int(number) - 1] == text, (path, number)
            sums['changed_executable_lines'] += len(lines)
            sums['hit_lines'] += len(hits)
            sums['missed_lines'] += len(misses)
            sums['new_or_changed_branch_arcs'] += len(arcs)
            sums['hit_arcs'] += len(arc_hits)
            sums['missed_arcs'] += len(arc_misses)
            modules.append({'path': path, 'source_sha256': sha(source), 'changed_lines': len(lines), 'missed_lines': sorted(misses), 'changed_arcs': len(arcs), 'missed_arcs': sorted(map(list, arc_misses))})
        assert sums == report['totals']
        connection.close()
        reports.append({'archive': prefix, 'report': report_path, 'source_commit': report['source_commit'], 'totals': sums, 'merged_data': matched_data[0], 'database_observed_arcs_and_lines_match': True, 'frozen_baseline_changed_denominators_recomputed': True, 'modules': modules})
    records[INITIAL]['reports'] = reports
    for cohort in initial['cohorts']:
        log = records[INITIAL]['logs'][f"logs/worker-runtime-coverage-r{cohort['run']}.log.gz"]
        counts = log['summary_rows'][-1]['counts']
        assert counts.get('passed', 0) == cohort['passed'] and counts.get('failed', 0) == cohort['failed']
        assert log['source_declarations'] == [cohort['source_commit']]
        assert log['terminal_exit_status'] == [str(int(cohort['failed'] > 0))]
    for run, expected in [(8, {'failed': 2, 'passed': 16}), (9, {'failed': 1, 'passed': 59}), (10, {'passed': 60})]:
        log = records[SUPPLEMENT]['logs'][f'logs/worker-runtime-coverage-r{run}.log.gz']
        assert log['summary_rows'][-1]['counts'] == expected
        assert log['terminal_exit_status'] == [str(int(run != 10))]
    for prefix, name in [(INITIAL, 'logs/worker-runtime-hosted-py312-new-cases-r1.log.gz'), (SUPPLEMENT, 'logs/worker-supplement-hosted-py312-r1.log.gz')]:
        expected = receipts[prefix]['local_py312_pytest842_new_cases']
        log = records[prefix]['logs'][name]
        assert log['summary_rows'][-1]['counts'].get('passed') == expected['passed']
        assert log['terminal_exit_status'] == ['0']
    preservation = []
    for path, count in initial['original_test_bodies_preserved'].items():
        def tests(data):
            tree = ast.parse(data)
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')}
        old = tests(iraw['tests/original/' + path + '.gz'])
        new = tests(iraw['tests/current/' + path + '.gz'])
        assert len(old) == count and all(new.get(key) == value for key, value in old.items())
        preservation.append({'path': path, 'original_test_definitions_unchanged': count})
    result = {
        'passed': True, 'review_git_commit': args.git, 'verification_coverage_version': coverage.__version__, 'recorded_coverage_version': '7.16.0', 'archives': records,
        'original_test_bodies_ast_verified': preservation,
        'acceptance': 'Payload integrity and internally source-bound CPU evidence only. Product remains held; no app integration, whole-module ratchet, GPU, native display, complete hosted run, or aesthetics acceptance.',
        'remaining_changed_lines': 38, 'remaining_changed_arcs': 21,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'payloads_verified': sum(len(r['payloads_verified']) for r in records.values()), 'gzip_logs_verified': sum(len(r['logs']) for r in records.values()), 'source_bindings': len(bindings), 'coverage_inputs': len(initial['coverage_inputs']), 'final_tests': len(final_tests), 'changed_modules': len(reports[-1]['modules']), 'original_tests_preserved': preservation, 'final_totals': reports[-1]['totals'], 'public_production_differences': [r['path'] for r in bindings if r['path'].startswith('spacr/') and not r['matches_review_head']]}, indent=2))

if __name__ == '__main__':
    main()
