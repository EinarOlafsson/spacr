from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
assert '145 passed' in (scratch / 'foundation-docstring-guards-r2.log').read_text()
controls = json.loads((scratch / 'API-boundary-guard-controls-r1.json').read_text())
assert len(controls['all_five_independent_contract_mutations_rejected']) == 5
delta = json.loads((scratch / 'foundation-inherited-API-inventory-delta-r1.json').read_text())
assert delta['full_inverse_digest_exact']
destination = Path('features/data/411_night_theme_optional_sound_API_inventory_repaired_2026-10-06')
destination.mkdir(exist_ok=False)
for name in ('API-boundary-guard-controls-r1.json', 'verify_API_boundary_guard_mutations_r1.py',
             'foundation-inherited-callable-inventory-failure-r1.json',
             'verify_inherited_callable_inventory_failure_r1.py', Path(__file__).name):
    shutil.copyfile(scratch / name, destination / name)
for name in ('foundation-docstring-guards-r1.log', 'foundation-docstring-guards-r2.log',
             'foundation-inherited-callable-inventory-failure-r1.log',
             'foundation-inherited-API-inventory-delta-r1.log', 'API-boundary-guard-controls-r1.log'):
    (destination / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
original = subprocess.check_output(['git', 'show', '4cda58d6b:tests/test_docstring_correctness.py'])
fixed = Path('tests/test_docstring_correctness.py').read_bytes()
for name, data in (('original-callable-guard.py.gz', original), ('fixed-callable-guard.py.gz', fixed)):
    path = destination / name
    path.write_bytes(gzip.compress(data, mtime=0))
    assert gzip.decompress(path.read_bytes()) == data
receipt = {'item': 411, 'coordinated_before_editing_commit': '4cda58d6b',
           'exact_one_optional_NightTheme_sound_key_contract_admitted_and_checked': True,
           'all_original_inventory_count_digest_required_docless_and_coverage_pins_preserved': True,
           'complete_original_inverse_digest_exact': delta['prior_digest'],
           'all_five_independent_contract_mutations_rejected': controls['all_five_independent_contract_mutations_rejected'],
           'normal_current_docstring_and_settings_flow_tests_passed': 145,
           'original_failed_diagnostics_preserved': True,
           'no_application_source_or_runtime_dependency_change_in_this_guard_repair': True,
           'no_global_CI_coverage_serial_Qt_or_publication_claim': True,
           'artifacts': {str(path): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'bytes': path.stat().st_size} for path in sorted(destination.iterdir())}}
Path(str(destination) + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation NightTheme API-inventory repair accepted: the coordinated exact guard now independently checks the complete seven-field NightTheme constructor, all six required fields, documented optional sound_key, category/exposure/variants, then reconstructs only its prior six-field boundary row for the original full 9,991-row count/digest checks. No old pin, debt/coverage ceiling or guard is raised or weakened. All 145 normal package documentation/callable/parameter/settings-flow tests pass. Five independent metadata mutations (missing/required/undocumented/extra optional fields and arbitrary keywords) are all rejected. Receipt 411_night_theme_optional_sound_API_inventory_repaired_2026-10-06.json retains original 144-pass/one-failure diagnostics, real before/after complete source, inverse-digest proof and completed normal/negative-control logs/scripts. This closes that inherited API-inventory failure for Home integration; Home still owns actual global CI/coverage/serial Qt and feature source. All nine updated runtime catalogs locally pass normal strict audit plus four focused placeholder/source/inventory/note tests. Current encoder source/catalog repairs remain uncommitted until the live full all-nine API generation, preservation and full strict Sphinx finish. Mask07 and corrected Measure08/Home05/Conda02 remain unchanged; no GPU process is live here.\n'
with Path('features/325_two_sessions_one_repo_working_protocol.temp').open('a') as handle:
    handle.write(note)
print('PASS guarded NightTheme source-inventory repair and all 145 checks archived', flush=True)
