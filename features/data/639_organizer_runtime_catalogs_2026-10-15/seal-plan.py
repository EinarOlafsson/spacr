"""Bind a pending or authorized runtime plan without importing spaCR/collecting."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repository', required=True, type=Path)
parser.add_argument('--public-base', required=True)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--authorized-baseline', action='store_true', help='Only after coordinator confirms the foreign producer is published and merged')
args = parser.parse_args()
root = args.repository.resolve()
base = subprocess.check_output(['git', '-C', str(root), 'rev-parse', args.public_base], text=True).strip()
subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', base, 'HEAD'], check=True)
assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']), 'Dirty source/input tree'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
hash_path = lambda name: sha((root / name).read_bytes())
langs = ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr']
review = {}
owned = None
for lang in langs:
    name = f'docs/i18n/reviewed/runtime/{lang}/2026-10-15-n639-organizer-messages.json'
    payload = json.loads((root / name).read_bytes())
    assert payload['schema'] == 1 and payload['language'] == lang
    keys = [row['key'] for row in payload['records']]
    assert len(keys) == len(set(keys)) == 21
    assert all(row['table'] == 'ui' and row['source'] == row['key'] and row['source_sha256'] == sha(row['source'].encode()) for row in payload['records'])
    if owned is None:
        owned = keys
    assert owned == keys
    review[lang] = {'path': name, 'sha256': hash_path(name)}
source_files = {name: hash_path(name) for name in (
    'spacr/channel_sorting.py', 'spacr/qt/widgets/organize_for_measure.py', 'spacr/qt/i18n.py')}
outputs = {f'spacr/qt/i18n_catalogs/{lang}.py' for lang in ['en', *langs]}
outputs.add('tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz')
tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'], text=True).split('\0')
input_names = [name for name in tracked if name and name not in outputs and (
    (name.endswith('.py') and name.startswith(('spacr/', 'tools/'))) or
    (name.endswith('.json') and name.startswith('docs/i18n/reviewed/')) or
    name == 'spacr/qt/setting_consumers.json')]
preserved_names = [name for name in tracked if name and (
    (name.startswith('docs/source/_static/i18n/api/') and name.endswith('.json')) or
    name in ('docs/i18n/api_source_history.json', 'spacr/qt/help_api_index.py', 'spacr/qt/help_index.py'))]
input_names = sorted(set(input_names) - set(preserved_names))
assert {'docs/source/_static/i18n/api/en.json', 'docs/source/_static/i18n/api/ko.json', 'spacr/qt/help_api_index.py'} <= set(preserved_names)
driver = Path(__file__).with_name('generate.py')
ast.parse(driver.read_text())
plan = {
    'state': 'authorized_baseline_ready' if args.authorized_baseline else 'waiting_for_n702_public_baseline',
    'repository': str(root), 'public_base': base,
    'private_head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    'source_commit': 'f5a3f7f28987949794660e0a05fd5d4f7896a65a',
    'input_commit': '4ebf580197', 'driver_sha256': sha(driver.read_bytes()),
    'languages': langs, 'owned_sources': owned, 'owned_table': 'UI',
    'source_files': source_files, 'review_files': review,
    'input_pins': {name: hash_path(name) for name in input_names},
    'baseline_outputs': {name: hash_path(name) for name in sorted(outputs)},
    'preserved_artifacts': {name: hash_path(name) for name in sorted(preserved_names)},
    'execution': {'memory_cap': '6G', 'threads': 1, 'gpu': False,
                  'lock': '/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock',
                  'normal_runtime_canonical_calls': 2, 'api_collection': False, 'help_generation': False,
                  'all_nine_preflight_before_catalog_writes': True},
    'required_before_execute': [] if args.authorized_baseline else [
        'Wait for N702 producer terminal public SHA, normally merge it, then reseal this plan against that SHA.',
        'Independent driver/plan acceptance and coordinator writer-window authorization.']}
args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'plan': str(args.output), 'sha256': sha(args.output.read_bytes()),
                  'state': plan['state'], 'input_pins': len(input_names),
                  'baseline_outputs': len(outputs), 'preserved_artifacts': len(preserved_names)}))
