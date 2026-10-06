from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '558-jump-current-r1'
plan = json.loads((root / 'plan.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert plan['prepared'] and len(plan['training']) == 48 and len(plan['held_out']) == 20
assert len(plan['original_downloads']) == 272
for path, expected in plan['source_sha256'].items():
    assert digest(path) == expected
for row in plan['original_downloads']:
    assert digest(row['path']) == row['sha256']
for row in plan['training'] + plan['held_out']:
    assert digest(row['field_path']) == row['field_sha256']
benchmark = scratch / 'benchmark_jump_virtual_stain_gpu.py'
compile(benchmark.read_text(), str(benchmark), 'exec')
model = Path.home() / '.cellpose/models/cpsam'
freeze = {'plan_sha256': digest(root / 'plan.json'), 'benchmark_sha256': digest(benchmark),
          'cpsam_path': str(model), 'cpsam_sha256': digest(model),
          'resolved_training_defaults': {'batch_size': 16, 'lr': 0.001, 'model_type': 'unet'},
          'normal_turn_label': '558-jump-virtual-stain-cuda-20261006-r1'}
(root / 'GPU-freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
dest = Path('features/data/558_jump_GPU_preparation_2026-10-06')
dest.mkdir(exist_ok=False)
for source in (root / 'plan.json', root / 'GPU-freeze.json', scratch / 'prepare_jump_virtual_stain.py', benchmark, Path(__file__)):
    shutil.copyfile(source, dest / source.name)
for source in root.glob('*load_data.csv'):
    (dest / (source.name + '.gz')).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
(dest / 'jump-virtual-stain-preparation-r1.log.gz').write_bytes(gzip.compress((scratch / 'jump-virtual-stain-preparation-r1.log').read_bytes(), mtime=0))
report = {'item': 558, 'CPU_data_preparation_accepted': True,
          'original_acquired_images_downloaded_and_ETag_SHA_verified': 272,
          'exact_paired_four_channel_fields': 68, 'shape': [1080, 1080, 4],
          'training_fields': 48, 'held_out_fields': 20, 'plate_disjoint_split': True,
          'selection_predeclared_from_metadata_without_scores': True,
          'inputs_three_acquired_brightfield_channels_target_DNA_Hoechst': True,
          'historical_model_unavailable_new_training_explicit': True,
          'normal_frozen_training_and_Cellpose_GPU_scorecard_pending': True,
          'no_independent_human_gold_four_biological_replicates_or_new_cell_line_claim': True,
          'normal_turn_label': freeze['normal_turn_label'],
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/558_jump_GPU_preparation_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation virtual staining GPU preparation: 272 original acquired Cell Painting Gallery cpg0016-jump/source_4 images are downloaded with recorded S3 ETags/SHA256 and independently checked metadata. Every original plane is byte/pixel checked in 68 complete 1080x1080 four-channel fields: Hoechst target plus three brightfield inputs. The predeclared first twenty-four BR00117035 wells/sites 1-2 supply 48 training fields; the first ten BR00117036 wells/sites 1-2 supply 20 held-out fields, with plates disjoint and no image/score-based selection. The historical trained model is unavailable on this host, so this is explicitly fresh reproducible training, not reuse of that checkpoint. The normal U-Net twenty-epoch/seed-zero/scale-two setup and default Cellpose-SAM real/predicted/input scorecard are source/model/input frozen. Receipt 558_jump_GPU_preparation_2026-10-06.json archives raw metadata, all original/export hashes, split, benchmark and terminal preparation. Planned turn 558-jump-virtual-stain-cuda-20261006-r1 follows robustness retry r2 through normal tools/gpu_turn.sh, real HOME, 24 GiB and unchanged six-minute idle/ten-minute handoff. Training, CUDA profiles, original labels and independent item-532 rescoring remain pending. Real-stain Cellpose predictions are reference, not human gold; no four-replicate or new-cell-line accuracy claim is made. Home retains CPU/Qt/CI/source ownership and protected jobs remain untouched.\n'
for path in ('features/future/558_virtual_staining.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: exact paired JUMP data, disjoint split and normal training/Cellpose scorecard freeze archived; actual GPU execution remains pending.', flush=True)
