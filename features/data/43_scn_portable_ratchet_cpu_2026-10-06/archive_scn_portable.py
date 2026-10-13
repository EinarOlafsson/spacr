import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

repo = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
scratch = Path('/mnt/wd4tb/scratch/ci-final-repairs-20261006')
out = repo / 'features/data/43_scn_portable_ratchet_cpu_2026-10-06'
out.mkdir(parents=True, exist_ok=True)
paths = ['spacr/convert.py', 'spacr/io.py', 'spacr/qt/timing.py']
old = json.loads(Path('/mnt/wd4tb/scratch/ci-392-ratchet-20261007/coverage.json').read_text())
new = json.loads((scratch / 'scn-portable-final-coverage.json').read_text())
source = {}
comparison = {}
for path in paths:
    before = subprocess.check_output(['git', 'show', '392ca4d6a2319fd88c2d352b1a0b021682123d6b:' + path], cwd=repo)
    current = (repo / path).read_bytes()
    assert before == current, path
    source[path] = hashlib.sha256(current).hexdigest()
    older, focused = old['files'][path], new['files'][path]
    remaining_lines = sorted(set(older['missing_lines']) - set(focused['executed_lines']))
    remaining_arcs = [arc for arc in older['missing_branches'] if arc not in focused['executed_branches']]
    comparison[path] = {
        'old_missing_statements': older['missing_lines'],
        'old_missing_branches': older['missing_branches'],
        'remaining_old_missing_statements': remaining_lines,
        'remaining_old_missing_branches': remaining_arcs,
    }
assert not comparison['spacr/convert.py']['remaining_old_missing_statements']
assert not comparison['spacr/convert.py']['remaining_old_missing_branches']
assert len(comparison['spacr/io.py']['remaining_old_missing_branches']) == 1
assert len(comparison['spacr/qt/timing.py']['remaining_old_missing_branches']) == 2
for filename, data in (
        ('hosted392-three-modules.json.gz', {'files': {path: old['files'][path] for path in paths}}),
        ('focused-three-modules.json.gz', {'files': {path: new['files'][path] for path in paths}})):
    raw = (json.dumps(data, sort_keys=True) + '\n').encode()
    (out / filename).write_bytes(gzip.compress(raw, mtime=0))
for filename in ('scn-portable-final.log',):
    (out / (filename + '.gz')).write_bytes(gzip.compress((scratch / filename).read_bytes(), mtime=0))
for filename in ('module-coverage-ratchet.json', 'module-coverage-ratchet.txt'):
    raw = (Path('/mnt/wd4tb/scratch/ci-392-ratchet-20261007') / filename).read_bytes()
    (out / ('hosted392-' + filename + '.gz')).write_bytes(gzip.compress(raw, mtime=0))
receipt = {
    'date_America_Detroit': '2026-10-06',
    'hosted_source': '392ca4d6a2319fd88c2d352b1a0b021682123d6b',
    'hosted_run': 37550980965,
    'hosted_aggregate_job': 112592627920,
    'source_sha256': source,
    'test_sha256': {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in (
        'tests/test_scn_reader.py', 'tests/test_native_tzyx_portable_reservation_f548.py',
        'tests/test_timing_missing_sysconfig.py')},
    'focused_passed': 42,
    'focused_seconds': 10.06,
    'hard_memory_cap': '4G',
    'CUDA_VISIBLE_DEVICES': '',
    'QT_QPA_PLATFORM': 'offscreen',
    'comparison': comparison,
    'scope': 'Exact unchanged-source union of hosted392 coverage and focused behavior tests. Not a new full-suite or hosted gate verdict.',
    'production_changed': False,
    'coverage_limits_changed': False,
}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(out / 'README.md').write_text(
    '# Source-bound SCN and portable-workspace CPU checks\n\n'
    'The 392 hosted aggregate has twelve complete shard records and six real\n'
    'numerical module regressions. This archive addresses three unchanged-source\n'
    'modules using meaningful input/error/fallback behavior checks. The combined\n'
    '42 cases pass under 4 GiB with CUDA hidden and Qt offscreen. SCN validation\n'
    'reaches all 19 old missing statements and 17 destinations. Portable workspace\n'
    'reservation reaches four old statements and four destinations; timing reaches\n'
    'two old statements and one destination. The remaining IO/timing destinations\n'
    'fit their original limits. No production code, exclusion or limit changes.\n\n'
    'Coverage inputs, full focused terminal log, original numerical report, source\n'
    'and test bindings are retained. This is an exact-source union, not a fresh\n'
    'complete-suite or hosted-success verdict. `archive_scn_portable.py` regenerates\n'
    'this archive from the original bounded scratch run.\n')
if Path(__file__).resolve() != (out / 'archive_scn_portable.py').resolve():
    shutil.copyfile(__file__, out / 'archive_scn_portable.py')
manifest = {'files': {}}
for path in sorted(out.iterdir()):
    if path.name == 'manifest.json':
        continue
    raw = path.read_bytes()
    manifest['files'][path.name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(receipt['comparison'], indent=2))
print('Generated', len(manifest['files']), 'payloads')
