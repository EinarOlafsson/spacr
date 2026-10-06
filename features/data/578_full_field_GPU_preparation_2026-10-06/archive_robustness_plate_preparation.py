from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '578-robustness-plate-r1'
plan = json.loads((root / 'plan.json').read_text())
assert plan['prepared'] and plan['field_count'] == 16 and plan['planned_real_GPU_calls'] == 128
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
dest = Path('features/data/578_full_field_GPU_preparation_2026-10-06')
dest.mkdir(exist_ok=False)
for source in (root / 'plan.json', root / 'recipe.json', scratch / 'prepare_robustness_plate_gpu.py',
               scratch / 'benchmark_robustness_plate_gpu.py', Path(__file__)):
    shutil.copyfile(source, dest / source.name)
compile((scratch / 'benchmark_robustness_plate_gpu.py').read_text(), str(scratch / 'benchmark_robustness_plate_gpu.py'), 'exec')
(dest / 'robustness-plate-preparation-r1.log.gz').write_bytes(gzip.compress((scratch / 'robustness-plate-preparation-r1.log').read_bytes(), mtime=0))
model = Path.home() / '.cellpose/models/cpsam'
freeze = {'model_path': str(model), 'model_sha256': digest(model),
          'plan_sha256': digest(root / 'plan.json'),
          'benchmark_sha256': digest(scratch / 'benchmark_robustness_plate_gpu.py')}
(root / 'GPU-freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
shutil.copyfile(root / 'GPU-freeze.json', dest / 'GPU-freeze.json')
report = {'item': 578, 'CPU_preparation_accepted': True, 'actual_GPU_acceptance_pending': True,
          'full_acquired_example_fields': 16, 'shape': [1994, 1994], 'wells': plan['wells'],
          'normal_ingest_sampling_and_exact_acquired_plane_contract_passed': True,
          'planned_actual_GPU_calls': 128, 'planned_turn': '578-full-field-robustness-20261006-r1',
          'scope': plan['scope'], 'no_production_full_screen_or_independent_accuracy_claim': True,
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/578_full_field_GPU_preparation_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation robustness GPU preparation: all sixteen complete acquired 1994x1994 nucleus intensity fields from four wells of the example plate are exported with exact original pixels and ingested through normal Mask preprocessing. Normal report sampling includes every field, with no crop, simulated data or generated reference mask planes. The frozen normal eight-point grid includes baseline and the predeclared deliberately fragile cellprob threshold 6; 128 actual CUDA calls are planned. Receipt 578_full_field_GPU_preparation_2026-10-06.json archives the verified source/model/recipe/benchmark hashes, original/export provenance and terminal CPU preparation log. Planned normal turn 578-full-field-robustness-20261006-r1 uses tools/gpu_turn.sh, real HOME, a 24 GiB cap and unchanged six-minute idle/ten-minute handoff. The actual report, saved-label readback, first/last positive CUDA profiles and fragile verdict remain pending; this is all fields of the example plate, not a whole production screen or independent accuracy benchmark. Current installation media upload/readback is separate and CPU-only. Home retains CPU/Qt/CI/application-source ownership; protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ('features/future/578_segmentation_robustness_report.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: normal full-field robustness preparation and source/model freeze archived; actual GPU report remains pending.', flush=True)
