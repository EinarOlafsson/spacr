"""Verify the compact exact-source CPU coverage union and unchanged allowances."""

import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = Path(__file__).resolve().parent
    root = archive.parents[2]
    receipt = json.loads((archive / 'receipt.json').read_text())
    manifest = json.loads((archive / 'manifest.json').read_text())
    for name, expected in manifest['payloads'].items():
        path = archive / name
        assert path.stat().st_size == expected['bytes'], name
        assert digest(path) == expected['sha256'], name
    source = root / receipt['module']
    assert digest(source) == receipt['source_sha256']
    original = subprocess.check_output(
        ['git', 'show', receipt['hosted_commit'] + ':' + receipt['module']], cwd=root)
    assert original == source.read_bytes(), 'Source changed: union is invalid.'
    for name, expected in receipt['tests'].items():
        assert digest(root / name) == expected, name
    baseline_path = root / 'tools/coverage_baseline.json'
    assert digest(baseline_path) == receipt['baseline_sha256']
    baseline = json.loads(baseline_path.read_text())['modules'][receipt['module']]
    for key, expected in receipt['unchanged_allowance'].items():
        assert baseline[key] == expected, key
    hosted = json.loads((archive / 'hosted_timelapse.json').read_text())['coverage']
    focused = json.loads((archive / 'focused_timelapse.json').read_text())['coverage']
    statements = lambda row: set(row['executed_lines'] + row['missing_lines'])
    branches = lambda row: {tuple(x) for x in row['executed_branches'] + row['missing_branches']}
    assert statements(hosted) == statements(focused)
    assert branches(hosted) == branches(focused)
    missing_lines = sorted(set(hosted['missing_lines']) - set(focused['executed_lines']))
    missing_branches = sorted({tuple(x) for x in hosted['missing_branches']}
                              - {tuple(x) for x in focused['executed_branches']})
    assert missing_lines == receipt['union_missing_statements'] == []
    assert [list(x) for x in missing_branches] == receipt['union_missing_branches'] == [[9323, 9328]]
    assert len(missing_lines) <= baseline['uncovered_statements']
    assert len(missing_branches) <= baseline['uncovered_branches']
    print(json.dumps({'verified': True, 'source_sha256': digest(source),
                      'missing_statements': missing_lines,
                      'missing_branch_destinations': missing_branches,
                      'unchanged_allowance': receipt['unchanged_allowance']}))


if __name__ == '__main__':
    main()
