"""Review only proposed static artifacts; never import or execute spaCR source."""
import sys
from pathlib import Path

WORKTREE = Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013')
sys.path.insert(0, str(WORKTREE))

import ast
import copy
import csv
import hashlib
import io
import json
import re

OUT = Path(__file__).resolve().parent
schema = json.loads((OUT / 'model-metadata.schema.json').read_text())
index = json.loads((OUT / 'fixture-index.json').read_text())
base = json.loads((OUT / 'fixtures/valid_synthetic_declared_opaque_version.json').read_text())
opaque = copy.deepcopy(base)
opaque['model_metadata']['settings']['pathogen_model_name']['version'] = '001'
name = 'valid_synthetic_declared_001'
(OUT / 'fixtures' / (name + '.json')).write_text(json.dumps(opaque, indent=2) + '\n')
with (OUT / 'fixtures' / (name + '.csv')).open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['Key', 'Value'])
    writer.writerows((key, str(value)) for key, value in opaque.items())
index.extend([
    {'file': f'fixtures/{name}.json', 'expected': 'accept_structure_and_binding',
     'scope': 'structural_and_binding', 'reason': 'Requested static version 001 and all-digit hash witness; synthetic, no real model.'},
    {'file': f'fixtures/{name}.csv', 'expected': 'literal_csv_matches_sibling_metadata',
     'scope': 'stdlib_syntax_only', 'reason': 'Version 001 and 64-digit SHA256 remain strings inside already-parsed nested dictionaries.'},
])
(OUT / 'fixture-index-final.json').write_text(json.dumps(index, indent=2) + '\n')


def conforms(value, rule):
    """Small stdlib check of only keywords used in this proposed schema."""
    if '$ref' in rule:
        rule = schema['$defs'][rule['$ref'].split('/')[-1]]
    allowed = rule.get('type')
    if allowed:
        allowed = [allowed] if isinstance(allowed, str) else allowed
        types = {'object': dict, 'string': str, 'integer': int, 'null': type(None)}
        if not any(type(value) is types[item] for item in allowed):
            return False
    if 'const' in rule and (type(value) is not type(rule['const']) or value != rule['const']):
        return False
    if 'enum' in rule and value not in rule['enum']:
        return False
    if isinstance(value, str):
        if len(value) < rule.get('minLength', 0):
            return False
        if 'pattern' in rule and re.search(rule['pattern'], value) is None:
            return False
    if isinstance(value, dict):
        props = rule.get('properties', {})
        if not set(rule.get('required', [])) <= set(value):
            return False
        if rule.get('additionalProperties') is False and set(value) - set(props):
            return False
        if any(key in value and not conforms(value[key], child) for key, child in props.items()):
            return False
        if 'propertyNames' in rule and any(not conforms(key, rule['propertyNames']) for key in value):
            return False
        extra_rule = rule.get('additionalProperties')
        if isinstance(extra_rule, dict) and any(not conforms(item, extra_rule)
                                               for key, item in value.items() if key not in props):
            return False
    if 'oneOf' in rule and sum(conforms(value, child) for child in rule['oneOf']) != 1:
        return False
    return True


results = []
for entry in index:
    path = OUT / entry['file']
    if path.suffix == '.csv':
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        metadata = ast.literal_eval(next(row['Value'] for row in rows if row['Key'] == 'model_metadata'))
        sibling = json.loads(path.with_suffix('.json').read_text())
        assert metadata == sibling['model_metadata']
        assert conforms(metadata, schema)
        ref = next(iter(metadata['settings'].values()))
        assert type(ref['sha256']) is str
        assert ref['version'] is None or type(ref['version']) is str
        if '001' in path.name:
            assert ref['version'] == '001' and ref['sha256'] == '0' * 64
        results.append({'file': entry['file'], 'static_csv_json_metadata_equal': True,
                        'production_parser_executed': False})
        continue
    settings = json.loads(path.read_text())
    if 'model_metadata' not in settings:
        results.append({'file': entry['file'], 'legacy_metadata_absent': True})
        continue
    good = conforms(settings['model_metadata'], schema)
    if entry['scope'] == 'relational':
        assert good
    else:
        assert good == (entry['expected'] != 'reject'), entry['file']
    results.append({'file': entry['file'], 'structural_conformance': good,
                    'relational_acceptance_executed': False})

excerpts = json.loads((OUT / 'source-excerpts.json').read_text())
witness = []
for name, symbol in [('spacr/cli.py', '_parse_csv_value'), ('spacr/tabular.py', '_parse_setting_value')]:
    snippet = next(item for item in excerpts if item['path'] == name and item['symbol'] == symbol)
    tree = ast.parse(snippet['source'])
    assert any(isinstance(node, ast.If) and
               isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) and
               isinstance(node.test.operand, ast.Call) and
               getattr(node.test.operand.func, 'id', None) == 'isinstance'
               for node in ast.walk(tree))
    witness.append({'path': name, 'symbol': symbol, 'line': snippet['line'],
                    'source_excerpt_sha256': snippet['sha256'],
                    'static_branch_trace': [
                        'CSV text starts with { and ast.literal_eval yields outer metadata dict.',
                        'Outer dict comprehension reparses direct values schema_version=int and settings=dict.',
                        'Those direct values are already non-string objects, so the early non-string return applies.',
                        'The settings mapping and its per-setting records are never recursively reparsed.',
                        'Consequently nested version 001 and all-digit SHA256 remain strings.',
                        'Flattening version as a direct scalar value would instead reach numeric coercion.',
                    ], 'production_function_executed': False})
(OUT / 'opaque-type-source-witness.json').write_text(json.dumps(witness, indent=2) + '\n')
(OUT / 'static-fixture-review.json').write_text(json.dumps({
    'scope': 'stdlib_artifact_check_only_not_app_tests',
    'full_json_schema_implementation_used': False,
    'keywords_checked': ['type', 'required', 'properties', 'additionalProperties',
                         'propertyNames', 'const', 'enum', 'minLength', 'pattern', 'oneOf', '$ref'],
    'all_expected_structural_results_match': True,
    'checks': results,
    'relational_cases_remain_future_acceptance_proposals': True,
    'app_imports_or_source_execution': False,
}, indent=2) + '\n')
artifacts = []
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name != 'artifact-manifest-final.json':
        data = path.read_bytes()
        artifacts.append({'path': str(path.relative_to(OUT)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
(OUT / 'artifact-manifest-final.json').write_text(json.dumps({
    'state': 'source_only_preparatory', 'supersedes_manifest_for_current_artifacts': 'artifact-manifest.json',
    'artifacts': artifacts,
}, indent=2) + '\n')
print(json.dumps({'fixture_count': len(index), 'all_static_checks_match': True,
                  'manifest_final_sha256': hashlib.sha256((OUT / 'artifact-manifest-final.json').read_bytes()).hexdigest()}, indent=2))
