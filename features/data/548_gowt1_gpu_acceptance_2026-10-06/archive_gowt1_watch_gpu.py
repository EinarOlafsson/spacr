from pathlib import Path
import gzip
import hashlib
import json
import shutil
import sys
import zipfile

import numpy as np
import tifffile

sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
from spacr.scorecard import match_objects

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '548-gowt1-watch-r1'
output = root / 'gpu-benchmark-r1'
dest = Path('features/data/548_gowt1_gpu_acceptance_2026-10-06')
dest.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = read(output / 'benchmark.json')
plan = read(root / 'plan.json')
assert report['accepted'] and report['script_sha256'] == digest(scratch / 'benchmark_gowt1_watch.py')
assert report['plan_sha256'] == digest(root / 'plan.json')
for name, expected in report['source_sha256'].items():
    assert digest(name) == expected
assert all(p['cuda_kernel_events'] > 0 for p in report['profiles'][:2])
assert all(p['cuda_events'] == 0 for p in report['profiles'][2:])
assert report['collection_resume_without_inference'] and report['completed_restart_without_inference']
truth_record = plan['quality_reference']
truth_path = Path(truth_record['path'])
assert digest(truth_path) == truth_record['sha256']
truth = tifffile.imread(truth_path)
merged = output / 'batch/merged' / report['exact_merged_frames'][0]
planes = np.load(merged, allow_pickle=False)
assert planes.shape == (1024, 1024, 2)
assert read(output / 'batch/merged/.spacr_plane_layout.json')['mask_dims'] == {'nucleus': 1}
predicted = planes[..., 1]
assert np.array_equal(predicted, predicted.astype(np.uint32))
predicted = predicted.astype(np.uint32)
present = np.unique(predicted[50:-50, 50:-50])
retained = predicted.copy()
retained[~np.isin(retained, present)] = 0
scores = {}
for threshold in (0.5, 0.75):
    matched = match_objects(truth, retained, threshold)
    assert matched.n_truth == 24
    scores[str(threshold)] = {'tp': matched.true_positives, 'fp': matched.false_positives,
                             'fn': matched.false_negatives, 'f1': matched.f1,
                             'precision': matched.precision, 'recall': matched.recall,
                             'n_truth': matched.n_truth, 'n_pred': matched.n_pred}
quality = {'scoring_completed': True, 'acquired_frame': 'sequence 01 t021 only',
           'independent_gold_objects': 24, 'gold_sha256': digest(truth_path),
           'merged_sha256': digest(merged), 'mask_plane': 1, 'scores': scores,
           'policy': 'CTC 50-pixel field of interest; remove predictions wholly outside it, retain complete intersecting objects. Normal spacr.scorecard.match_objects.',
           'other_three_frames_have_not_been_quality_scored': True,
           'tracking_accuracy_against_gold_not_scored': True,
           'four_independent_biological_replicates_claimed': False}
(dest / 'single-field-quality.json').write_text(json.dumps(quality, indent=2) + '\n')
for path in sorted(output.glob('*.json')):
    shutil.copyfile(path, dest / path.name)
for path in (scratch / 'benchmark_gowt1_watch.py', Path(__file__)):
    shutil.copyfile(path, dest / path.name)
(dest / 'gowt1-watch-cuda-r1.log.gz').write_bytes(gzip.compress((scratch / 'gowt1-watch-cuda-r1.log').read_bytes(), mtime=0))
watched = root / 'partial-watch/spacr_watch'
with zipfile.ZipFile(dest / 'paired-scientific-outputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for prefix, incoming in [('batch', output / 'batch'), ('watch', watched)]:
        for folder in ('merged', 'tracks', 'measurements', 'settings'):
            for path in sorted((incoming / folder).rglob('*')):
                if path.is_file():
                    archive.write(path, prefix + '/' + str(path.relative_to(incoming)))
        for path in incoming.glob('*.json'):
            archive.write(path, prefix + '/' + path.name)
    archive.write(truth_path, 'gold/man_seg021.tif')
with zipfile.ZipFile(dest / 'source-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(report['source_sha256']) | {'spacr/scorecard.py'}):
        archive.write(path, path)
receipt = {'item': 548, 'accepted_cuda_fixed_map_batch_watch_parity': True,
           'application_source_commit': plan['application_source_commit'],
           'batch_size': 1, 'acquired_frames': 4, 'input_policy': plan['input_policy'],
           'nucleus_measurement_rows': report['exact_nucleus_measurement_rows'],
           'nucleus_measurement_columns': len(report['nucleus_measurement_columns']),
           'measurement_prcf_timepoints': report['measurement_frame_identities'],
           'flat_track_rows': report['track_rows'],
           'profiles': [{k: p[k] for k in ('phase', 'seconds', 'cuda_events', 'cuda_kernel_events', 'cuda_device')} for p in report['profiles']],
           'collection_resume_without_inference': True, 'completed_restart_without_inference': True,
           'single_field_gold_scoring': quality,
           'zernike_columns_skipped_when_optional_mahotas_unavailable': True,
           'native_vendor_T_gt_1': False, 'raw_vendor_completion_complete': False,
           'pooled_batch_size_gt_1': False, 'whole_item_complete': False,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
Path('features/data/548_gowt1_gpu_acceptance_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation real GPU watch acceptance: the acquired CTC GOWT1 four-frame fixed-map benchmark completes normally, rc=0, with actual CPSAM CUDA kernels in both batch and watch analysis. All four merged frames, every available scientific nucleus column across 91 rows and all 91 flat track rows match exactly. The three-arrival no-analysis wait, real-analysis/injected-I/O interruption, normal collection resume and completed restart pass; both resume/restart have zero CUDA events and staged scientific hashes/mtimes remain exact. Optional Mahotas is unavailable, so Zernike columns are explicitly absent in both runs. Independent normal IoU scoring is recorded on t021 only, with its 24 gold objects and the CTC 50-pixel field-of-interest policy; it is not tracking-accuracy or four-replicate acceptance. Receipt 548_gowt1_gpu_acceptance_2026-10-06.json archives exact paired arrays, databases, tracks, source/model hashes, positive CUDA profiles, gold scoring and completed raw turn log. Native vendor T>1, raw/vendor completion and pooled batch_size>1 remain OPEN. Home retains their implementation/CPU ownership; all remaining GPU work stays workstation-owned. No protected livecell/cellposeTIME job was touched.\n'
for path in ('features/future/548_watch_a_folder_and_analyse_incoming_images.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: archived real CUDA parity/recovery and separate one-field gold scoring:', scores, flush=True)
