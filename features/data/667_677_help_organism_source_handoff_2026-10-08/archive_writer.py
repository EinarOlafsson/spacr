"""Freeze the tested help and Organism follow-up for workstation integration."""

import gzip
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path('/mnt/wd4tb/scratch')
SOURCE_ROOT = pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-field-ripple-controls-20261008')
BASELINE = '0527ae6fa01a0b06544213169a706dfbac9b0f84'


def git(*args):
    """Read exact committed bytes without mutating the shared checkout."""
    return subprocess.check_output(['git', *args], cwd=ROOT)


def payload(name, data):
    """Write deterministic compressed bytes and both integrity hashes."""
    packed = gzip.compress(data, mtime=0)
    (OUT / name).write_bytes(packed)
    return {'payload': name, 'sha256': hashlib.sha256(packed).hexdigest(),
            'raw_sha256': hashlib.sha256(data).hexdigest()}


def main():
    """Archive immutable source, exact inventories and scoped terminal results."""
    checkpoint = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=SOURCE_ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE_ROOT)
    names = git('diff', '--name-only', BASELINE, checkpoint, '--',
                'spacr', 'tests/qt').decode().splitlines()
    sources = {name: payload(f'source-{index:02}.gz',
                            git('show', f'{checkpoint}:{name}'))
               for index, name in enumerate(names)}
    patch = payload('source-and-tests.patch.gz', git(
        'diff', '--binary', BASELINE, checkpoint, '--', 'spacr', 'tests/qt'))
    before = json.loads((SCRATCH / 'ui-integration-20261008/final-0527-inventories.json').read_text())
    after = json.loads((SCRATCH / 'ui-integration-20261008/final-help-organism-inventories.json').read_text())
    for name, expected in after['source_hashes'].items():
        assert hashlib.sha256(git('show', f'{checkpoint}:{name}')).hexdigest() == expected
    diff = {'baseline': BASELINE, 'checkpoint': checkpoint,
            'source_hashes': after['source_hashes'], 'inventories': {}}
    inventories = {'api': (before['api'], after['api'])}
    inventories.update({key: (before['runtime'][key], value)
                        for key, value in after['runtime'].items()})
    for key, (old, new) in inventories.items():
        if isinstance(old, list):
            old = {text: text for text in old}
        if isinstance(new, list):
            new = {text: text for text in new}
        old_keys, new_keys = set(old), set(new)
        diff['inventories'][key] = {
            'before': len(old), 'after': len(new),
            'added': {name: new[name] for name in sorted(new_keys - old_keys)},
            'removed': {name: old[name] for name in sorted(old_keys - new_keys)},
            'changed': {name: {'before': old[name], 'after': new[name]}
                        for name in sorted(old_keys & new_keys) if old[name] != new[name]},
        }
    inventory_row = payload('normal-inventory-diff.json.gz',
                            (json.dumps(diff, ensure_ascii=False, indent=2) + '\n').encode())
    phases = []
    for filename, tested, expected_failure in (
        ('home-ui-help-organism-final-612-20261008.log', '828307f14e4', '7 failed, 227 passed'),
        ('ui-integration-20261008/preferences-help-root-612.log', '15e2ad0bdc5', None),
        ('home-ui-help-organism-intermediate-612-20261008.log', 'b8eeef003f9', '1 failed, 279 passed'),
        ('ui-integration-20261008/preferences-help-painted-font-612.log', checkpoint, None),
        ('home-ui-help-organism-final-fixed-612-20261008.log', checkpoint, None),
    ):
        data = (SCRATCH / filename).read_bytes()
        summaries = re.findall(rb'[^\n]*(?:\d+ passed|\d+ failed)[^\n]*', data)
        assert summaries
        summary = summaries[-1].decode()
        if expected_failure is None:
            assert 'failed' not in summary
        else:
            assert expected_failure in summary
        phases.append({'tested_checkpoint': git('rev-parse', tested).decode().strip(),
                       'passing': expected_failure is None, 'terminal_summary': summary,
                       **payload(pathlib.Path(filename).name + '.gz', data)})
    receipt = {
        'schema': 1, 'baseline': BASELINE, 'checkpoint': checkpoint,
        'items': ['N667', 'N677'],
        'scope': 'Incremental UI-only source after the accepted0527 packet: scrollable resizable Preferences help, translated inline-only Actions, Organism category, required-parameter docstring fields and explicit legacy-fractal fixture. Excludes already published CI retry tools/tests.',
        'sources': sources, 'patch': patch, 'inventory_diff': inventory_row,
        'environment': {'PySide6': '6.12.0 overlay', 'pytest': '8.4.2 overlay',
                        'cap': '4G', 'CUDA_VISIBLE_DEVICES': '',
                        'QT_QPA_PLATFORM': 'offscreen', 'MPLBACKEND': 'Agg'},
        'phases': phases,
        'limits': ['Phase totals overlap; do not sum them.',
                   'Private source transfer is not application publication or current-source CI acceptance.',
                   'Workstation must remeasure combined source and regenerate API/callable/runtime translations normally.',
                   'N672/N673 geometry is separate and is not included.',
                   'No native compositor, GPU, full serial or human appearance acceptance is claimed.'],
    }
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Archived {len(sources)} exact source bindings at {checkpoint}')


if __name__ == '__main__':
    main()
