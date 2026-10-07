"""Archive the rejected scratch-only stroker and admission investigation."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

worktree = Path('/mnt/wd4tb/spacr-worktrees/codex-mask-load-drain-20261007')
archive = worktree / 'features/data/663_rejected_fungal_stroker_cpu_2026-10-07'
stroker = Path('/mnt/wd4tb/scratch/fungal-stroker-parity-20261007')
admission = Path('/mnt/wd4tb/scratch/fungal-cache-admission-20261007')
archive.mkdir(parents=True, exist_ok=True)
for old, new in [('probe.py', 'unrestricted_probe.py'), ('receipt.json', 'unrestricted_receipt.json'),
                 ('cached_probe.py', 'cached_probe.py'), ('cached-receipt.json', 'cached_receipt.json'),
                 ('build_candidate.py', 'build_candidate.py')]:
    shutil.copyfile(stroker / old, archive / new)
for old, new in [('probe.py', 'admission_probe.py'), ('receipt.json', 'admission_receipt.json'),
                 ('probe.log', 'admission.log')]:
    shutil.copyfile(admission / old, archive / new)
sources = {}
for name in ('before', 'candidate'):
    data = (stroker / (name + '.py')).read_bytes()
    (archive / (name + '_ambient.py.gz')).write_bytes(gzip.compress(data, mtime=0))
    sources[name] = hashlib.sha256(data).hexdigest()
cached = json.loads((archive / 'cached_receipt.json').read_text())
receipt = {
    'status': 'complete/rejected',
    'app_source_changed': False,
    'hard_24_fps_accepted': False,
    'source_sha256': sources,
    'parity': {'unrestricted_size_2_5_different_pixels': 0,
               'unrestricted_size_1_different_pixels': 61750,
               'cached_native_pairs': 12, 'cached_different_pixels': 0,
               'stroke_eligible_pairs': 9, 'thin_bypass_control_pairs': 3},
    'shader_medians': [dict(row, change_percent=(row['shader_median_ms']['candidate'] /
                       row['shader_median_ms']['before'] - 1) * 100)
                      for row in cached['records'] if 'shader_median_ms' in row],
    'provenance': {
        'initial_probe_and_cached_probe': str(stroker),
        'initial_probe_logs': 'No separate raw log was saved; original scripts and JSON receipts are preserved unchanged.',
        'admission_probe': str(admission),
        'admission_command': "CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/fungal-cache-admission-20261007/probe.py",
        'admission_cwd': str(worktree),
        'admission_window': '72 evolving native4K frames95 to97.9583333333 at24fps offline time increments per case; counters, not live cadence.',
        'log': 'admission.log',
        'original_source_guard': 'The production sparse raster remains64 entries/8MiB; no guard changes. Candidate adds64 stroke paths whose native storage is not included in that sparse-array bound.'},
    'limitations': ['Direct shader medians are not actual widget or producer FPS.',
                    'One seed/time window and dark default/Random presets do not qualify all contexts/palettes.',
                    'Thin unrestricted fill mismatch is not attributed to the cached candidate, which bypasses those strokes.',
                    'No GPU, app/docs/API changes, new acceptance tests, or hard24FPS claim.'],
    'conclusion': 'Existing sparse/observation caches had no eviction or headroom refusal; rejected stroke cache had zero hits and12900 evictions. No distinct justified optimization remains from this admission lead.'}
(archive / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(archive)
