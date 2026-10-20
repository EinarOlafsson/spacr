"""Build static N702 proposal evidence using stdlib and immutable Git objects."""
import sys
from pathlib import Path

WORKTREE = Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013')
sys.path.insert(0, str(WORKTREE))

import ast
import copy
import csv
import hashlib
import json
import subprocess

OUT = Path(__file__).resolve().parent
REV = 'a744a9fda2b11835f3290cdaf7cf88f5a97f400f'
REPORT_REV = '37f73c05f978d202f6cfc3d2b3f005cff27dd784'


def git(*args):
    return subprocess.check_output(['git', '-C', str(WORKTREE), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(name, value):
    target = OUT / name
    encoded = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if target.exists():
        if target.read_text() == encoded:
            return
        raise RuntimeError(f'Preserving existing evidence: {target}')
    target.write_text(encoded)


wanted = {
    'spacr/cli.py': ['_parse_csv_value', '_load_settings_csv', '_load_settings_json',
                     'load_settings_file', '_settings_csv_cell', '_settings_json_value',
                     'write_settings_file', 'coerce_value', 'apply_overrides',
                     '_under_todays_names', 'resolve_settings', 'cmd_run'],
    'spacr/tabular.py': ['_parse_setting_value', '_settings_from_table'],
    'spacr/utils.py': ['save_settings', 'load_settings', '_resolve_cellpose_pretrained'],
    'spacr/qt/screens/app_screen.py': ['export_settings_csv', '_on_import_settings',
                                     '_on_export_settings', '_load_settings_csv',
                                     '_apply_each_setting'],
    'spacr/qt/screens/settings_model.py': ['collect', 'set_hidden_value'],
    'spacr/qt/screens/distributed_jobs.py': ['configure_submission', '_save_snapshot_csv'],
    'spacr/validate.py': ['_check_unknown_keys', '_check_required_paths'],
    'spacr/remote_execution.py': ['_map_path_string', 'map_settings_paths', 'submit'],
    'spacr/batch.py': ['resolve_job_settings'],
    'spacr/model_zoo.py': ['_entry_from_mapping', 'shared_catalogue', 'catalogue',
                           'resolve', '_ensure_model_file', 'versioned_path', 'fetch', 'hf_uri'],
    'spacr/submodules.py': ['_resolve_detector_weights', '_resolve_training_base', '_resolve_plaque_model'],
    'spacr/torch_artifacts.py': ['load_model_artifact'],
    'spacr/object_classifier.py': ['_real_classifier_bundles'],
    'spacr/plaque_papers.py': ['_zoo_path'],
    'spacr/run_journal.py': ['open_run'],
}
extra = ['spacr/settings.py', 'spacr/object.py', 'spacr/deep_spacr.py',
         'spacr/_segmentation_backends.py', 'docs/source/distributed_execution.rst',
         'tests/test_n693_write_settings_file.py',
         'tests/qt/test_n693_export_settings_from_every_form.py',
         'tests/qt/test_n693_distributed_jobs_saves_the_snapshot.py',
         'tests/test_missing_model_downloads.py', 'tests/test_model_zoo_shared_catalogue.py']
bindings = []
excerpts = []
raw_sources = {}
for name in list(wanted) + extra:
    data = git('show', f'{REV}:{name}')
    raw_sources[name] = data.decode()
    current = git('show', f'{REPORT_REV}:{name}')
    bindings.append({'path': name, 'revision': REV,
                     'git_blob': git('rev-parse', f'{REV}:{name}').decode().strip(),
                     'sha256': digest(data), 'bytes': len(data),
                     'report_observed_revision': REPORT_REV,
                     'matches_report_observed_revision': data == current})
    if name in wanted:
        source = data.decode()
        lines = source.splitlines(keepends=True)
        found = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted[name]:
                text = ''.join(lines[node.lineno - 1:node.end_lineno])
                excerpts.append({'path': name, 'symbol': node.name,
                                 'line': node.lineno, 'end_line': node.end_lineno,
                                 'sha256': digest(text.encode()), 'source': text})
                found.add(node.name)
        missing = set(wanted[name]) - found
        if missing:
            raise RuntimeError(f'Missing source symbols in {name}: {missing}')

historical = []
for name in ['AGENTS.md', 'features/HANDOFF.md',
             'features/HANDOFF_2026-10-13_claude_home_to_codex.md',
             'features/new/702_cluster_runs_download_a_missing_zoo_model_automatically.txt',
             'features/new/693_export_settings_from_every_form.txt',
             'features/data/702_missing_model_source_audit_2026-10-13/audit.json',
             'features/data/702_missing_model_source_audit_2026-10-13/root-source-binding-check.json']:
    data = git('show', f'{REPORT_REV}:{name}')
    historical.append({'path': name, 'revision': REPORT_REV,
                       'git_blob': git('rev-parse', f'{REPORT_REV}:{name}').decode().strip(),
                       'sha256': digest(data), 'bytes': len(data)})

write_json('source-bindings.json', {'state': 'source_only_preparatory',
                                  'candidate': REV, 'initial_report_head': REPORT_REV,
                                  'source_bindings': bindings, 'instruction_prior_audit_bindings': historical})
write_json('source-excerpts.json', excerpts)

tree = ast.parse(raw_sources['spacr/model_zoo.py'])
catalogue = None
for node in tree.body:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'BUNDLED_REMOTE_MODELS':
        catalogue = [{ast.literal_eval(key): ast.literal_eval(value)
                      for key, value in zip(row.keys, row.values)
                      if ast.literal_eval(key) in ('key', 'name', 'kind', 'sha256', 'version')}
                     for row in node.value.elts]
assert catalogue and len(catalogue) == 19
assert all('version' not in row for row in catalogue)
minimal_rows = [{k: row[k] for k in ['key', 'name', 'kind', 'sha256']} |
                {'declared_version': None, 'version_status': 'unspecified'} for row in catalogue]
write_json('catalogue-version-observation.json', {'source_revision': REV,
                                               'source_path': 'spacr/model_zoo.py',
                                               'count': len(catalogue),
                                               'explicit_version_count': 0,
                                               'normalized_ModelEntry_version': '1',
                                               'rows': minimal_rows})

nonblank = {'type': 'string', 'minLength': 1, 'pattern': r'\S'}
record = {
    'type': 'object', 'additionalProperties': False,
    'required': ['requested_value', 'key', 'version', 'version_status', 'sha256', 'kind', 'filename'],
    'properties': {
        'requested_value': nonblank,
        'key': nonblank,
        'version': {'type': ['string', 'null']},
        'version_status': {'enum': ['declared', 'unspecified']},
        'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
        'kind': nonblank,
        'filename': {'type': 'string', 'minLength': 1,
                     'pattern': r'^(?![/\\])(?![A-Za-z]:)(?!.*://)(?!.*(?:^|[/\\])\.\.(?:[/\\]|$)).*\S.*$'},
    },
    'oneOf': [
        {'properties': {'version_status': {'const': 'declared'}, 'version': nonblank}},
        {'properties': {'version_status': {'const': 'unspecified'}, 'version': {'type': 'null'}}},
    ],
}
schema = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'title': 'PROPOSED N702 model_metadata value, version 1',
    'type': 'object', 'additionalProperties': False,
    'required': ['schema_version', 'settings'],
    'properties': {'schema_version': {'type': 'integer', 'const': 1},
                   'settings': {'type': 'object', 'propertyNames': nonblank,
                                'additionalProperties': {'$ref': '#/$defs/model_reference'}}},
    '$defs': {'model_reference': record},
}
write_json('model-metadata.schema.json', schema)

selected = next(row for row in minimal_rows if row['key'] == 'toxoplasma_pv_v4')
reference = {'requested_value': '/laptop/.spacr/models/cpsam_v2_toxo_r7',
             'key': selected['key'], 'version': None, 'version_status': 'unspecified',
             'sha256': selected['sha256'], 'kind': selected['kind'], 'filename': selected['name']}
base = {'src': '/laptop/data/plate1', 'pathogen_model_name': reference['requested_value'],
        'segmentation_backend': 'cellpose', 'model_metadata': {'schema_version': 1,
        'settings': {'pathogen_model_name': reference}}}
fixtures = []


def fixture(name, value, expected, reason, category='structural_and_binding'):
    target = OUT / 'fixtures' / (name + '.json')
    if target.exists():
        raise RuntimeError(f'Preserving existing evidence: {target}')
    target.write_text(json.dumps(value, indent=2) + '\n')
    fixtures.append({'file': 'fixtures/' + target.name, 'expected': expected,
                     'scope': category, 'reason': reason})


fixture('valid_missing_builtin_unspecified_version', base, 'accept_pin',
        'Current built-in has exact key/name/kind/hash and no declared version; no real weights are present in this evidence.')
declared = copy.deepcopy(base)
declared['model_metadata']['settings']['pathogen_model_name'].update(
    key='example_synthetic_opaque_release', filename='synthetic_weights.bin',
    version='01', version_status='declared', sha256='0' * 64)
fixture('valid_synthetic_declared_opaque_version', declared, 'accept_structure_and_binding',
        'Synthetic trusted-catalogue example only, not a real downloadable model; version 01 is not version 1 and the all-digit hash must remain text.')
windows = copy.deepcopy(base)
windows['pathogen_model_name'] = r'C:\Users\Example\.spacr\models\cpsam_v2_toxo_r7'
windows['model_metadata']['settings']['pathogen_model_name']['requested_value'] = windows['pathogen_model_name']
fixture('valid_windows_request', windows, 'accept_pin', 'Exact request binding survives lexical Windows path preservation; native mapping is unproved.')
legacy = {'src': '/laptop/data/plate1', 'pathogen_model_name': '/laptop/.spacr/models/cpsam_v2_toxo_r7'}
fixture('valid_legacy_without_metadata', legacy, 'legacy_unpinned', 'No historical identity claimed; exact-basename selection remains conditional on unique trusted current catalogue.')
bad = copy.deepcopy(base)
bad['model_metadata']['settings']['pathogen_model_name']['version'] = '1'
del bad['model_metadata']['settings']['pathogen_model_name']['version_status']
fixture('invalid_ambiguous_normalized_version', bad, 'reject', 'Recorded normalized version 1 lacks declaration/default provenance; do not infer a publisher release.')
bad = copy.deepcopy(declared)
bad['model_metadata']['settings']['pathogen_model_name']['version'] = 1
fixture('invalid_numeric_version', bad, 'reject', 'Versions must remain opaque declared strings.')
bad = copy.deepcopy(base)
bad['model_metadata']['settings']['pathogen_model_name']['version'] = '1'
fixture('invalid_unspecified_nonnull_version', bad, 'reject', 'Unspecified version requires null.')
bad = copy.deepcopy(base)
bad['model_metadata']['schema_version'] = 2
fixture('invalid_unknown_schema', bad, 'reject', 'Unknown metadata schemas fail explicitly.')
bad = copy.deepcopy(base)
bad['model_metadata']['settings']['pathogen_model_name']['sha256'] = 'not-a-checksum'
fixture('invalid_malformed_hash', bad, 'reject', 'Hash must be 64 lowercase hex characters.')
bad = copy.deepcopy(base)
bad['pathogen_model_name'] = '/cluster/new/custom_weights.bin'
fixture('invalid_stale_requested_value', bad, 'reject', 'Metadata cannot stay bound to a former model after a file edit.', 'relational')
bad = copy.deepcopy(base)
bad['model_metadata']['settings']['pathogen_model_name']['kind'] = 'detector'
fixture('invalid_wrong_consumer_kind', bad, 'reject', 'Structurally valid metadata is semantically incompatible with the active Cellpose consumer.', 'relational')
bad = copy.deepcopy(base)
bad['model_metadata']['settings']['pathogen_model_name']['sha256'] = '0' * 64
fixture('invalid_conflicting_builtin_hash', bad, 'reject', 'Structurally valid pin contradicts the pinned trusted built-in row.', 'relational')
bad = copy.deepcopy(base)
bad['model_metadata']['settings']['pathogen_model_name']['filename'] = 'cpsam_v2_toxo_r7_v2'
fixture('invalid_cache_suffix_as_published_name', bad, 'reject', 'A collision suffix is not the published catalogue name or a release version.', 'relational')

for name, value in [('valid_missing_builtin_unspecified_version', base),
                    ('valid_synthetic_declared_opaque_version', declared),
                    ('valid_windows_request', windows)]:
    path = OUT / 'fixtures' / (name + '.csv')
    if path.exists():
        raise RuntimeError(f'Preserving existing evidence: {path}')
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Key', 'Value'])
        writer.writerows((key, str(value)) for key, value in value.items())
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    metadata = ast.literal_eval(next(row['Value'] for row in rows if row['Key'] == 'model_metadata'))
    assert metadata == value['model_metadata']
    assert type(metadata['schema_version']) is int
    fixtures.append({'file': 'fixtures/' + path.name, 'expected': 'literal_csv_matches_sibling_metadata',
                     'scope': 'stdlib_syntax_only', 'reason': 'CSV quoting and literal structure were checked; production parsers were not run.'})
write_json('fixture-index.json', fixtures)

compatibility = [
    ['N693 cli.write_settings_file', 'Literal str() dict cell and recursive JSON conversion; faithful values; no enrichment.',
     'Keep serializer faithful; pass metadata produced by shared explicit enrichment.'],
    ['cli.load_settings_file/_parse_csv_value', 'Reads CSV literals or JSON object; nested identity strings preserved by source branches; no schema checks.',
     'Validate reserved object/types and relational pins; reject unknown schema/ambiguity; avoid flattened opaque fields.'],
    ['tabular._parse_setting_value/_settings_from_table + utils.load_settings', 'Alternative CSV parser shares GUI path; direct dict values reparsed, deeper dictionaries returned unchanged.',
     'Match reserved validation and type-preservation contract across both CSV paths.'],
    ['utils.save_settings', 'CSV copies settings then changes test_mode/plot; JSON writes original settings, so whole-file parity is not general.',
     'Preserve model identities through run snapshots; do not describe this legacy writer as N693 parity.'],
    ['AppScreen import/apply + SettingsModel hidden/collect', 'CSV parses metadata, but undeclared hidden key is rejected and collect omits it.',
     'Declare reserved internal hidden metadata; preserve/invalidate on edits; prune missing slots, deep-copy snapshots.'],
    ['AppScreen export_settings_csv/_on_export_settings', 'Collects form and calls faithful writer; no model identity capture.',
     'Enrich without GUI hashing/scanning/download/remote refresh and keep current paths.'],
    ['DistributedJobs.configure_submission/_save_snapshot_csv', 'Keeps shallow snapshot; saves same snapshot using export_settings_csv.',
     'Enrich at Submit; freeze/deep-copy nested records; preserve Save/Submit parity and module registry meaning.'],
    ['validate._check_unknown_keys/_check_required_paths', 'Unknown nonempty model_metadata warns; existing path checks may refuse a missing file before acquisition.',
     'Reserve key and plan exact N702 resolution before eligible path refusals; hard pin validation cannot be bypassed by preflight flags.'],
    ['cli.resolve_settings/apply_overrides + batch.resolve_job_settings', 'Defaults/file/migration/overrides layering; no identity invalidation; known dict overrides accepted.',
     'Shared final identity reconciliation; invalidate explicitly overridden model records; reject generic model_metadata overrides.'],
    ['remote_execution.map_settings_paths/submit', 'Maps absolute string values recursively, including nested request binding; JSON serializes mapped settings.',
     'Validate paired request transformation; retain exact identity fields; origin payload provenance separate; no claim of Windows mapping.'],
    ['model_zoo.ModelEntry/_entry_from_mapping/catalogue/shared_catalogue', 'Default/mixed version meaning; catalogue deduplicates key/name; nonblocking shared refresh can use network.',
     'Preserve raw version declarations and conflicting rows for pure unique selection; explicit offline index, not remote=True block=False.'],
    ['model_zoo._ensure_model_file/resolve/fetch', 'Missing separated paths refuse; exact first-match/fuzzy resolution; checksum downloads; no process coalescing.',
     'Exact conjunct selector, pinned-local hashing, verified cache, strict policy and checked process lock.'],
    ['Cellpose/plaque/detector/classifier consumers', 'Multiple resolver seams and backend/sentinel/file-vs-directory meanings; no shared reserved metadata contract.',
     'Use active module-specific model input registry; preserve scientific/backend checks after byte resolution.'],
    ['run_journal.open_run', 'CLI opens journal before calling consumer; a later consumer-resolved path may not be the original settings identity.',
     'Record original request and actual resolved checkpoint/hash without disguising origin metadata as effective path.'],
]
write_json('compatibility.json', [{'route': row[0], 'current_source_behavior': row[1],
                                  'proposed_required_change': row[2],
                                  'acceptance': 'not_implemented_or_runtime_tested'} for row in compatibility])

write_json('static-verification.json', {
    'method': 'stdlib_only_git_object_reads_ast_parsing_csv_literal_json_syntax',
    'app_source_executed': False, 'app_or_Qt_imported': False,
    'app_tests_run': False, 'network_or_model_execution': False,
    'source_files_bound': len(bindings), 'source_symbol_excerpts': len(excerpts),
    'candidate_report_source_equality': all(row['matches_report_observed_revision'] for row in bindings),
    'catalogue_records_inspected': len(catalogue), 'explicit_builtin_version_fields': 0,
    'csv_literal_pairs_checked': 3,
    'json_schema_validator_run': False,
    'fixture_expected_results_are_proposals': True,
    'limitations': ['No current app parser or model resolver execution',
                    'No native/cluster/Windows/filesystem-lock/scientific acceptance',
                    'No proof of downloadable historical model availability',
                    'Relational rejection labels are proposed, not observed production results'],
})

artifacts = []
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name != 'artifact-manifest.json':
        data = path.read_bytes()
        artifacts.append({'path': str(path.relative_to(OUT)), 'sha256': digest(data), 'bytes': len(data)})
write_json('artifact-manifest.json', {'state': 'source_only_preparatory', 'artifacts': artifacts})
print(json.dumps({'out': str(OUT), 'artifact_count': len(artifacts),
                  'manifest_sha256': digest((OUT / 'artifact-manifest.json').read_bytes()),
                  'source_files_bound': len(bindings), 'source_symbols': len(excerpts)}, indent=2))
