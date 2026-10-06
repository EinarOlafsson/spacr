from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'558-jump-current-r1';target=root/'cuda-r1'
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
actual=read(target/'acceptance.json');independent=read(target/'independent-acceptance.json')
assert actual['actual_GPU_training_and_scorecard_complete']
assert independent['independent_item_532_saved_label_and_pixel_metrics_rescoring_accepted']
assert len(independent['independent_matches'])==80
assert len(list(target.glob('*-labels.npz')))==60 and len(list(target.glob('*-prediction.npy')))==20
checks=independent['all_20_normal_CPU_saved_model_replays_cross_device_diagnostic_not_equivalence_acceptance']
assert len(checks)==20 and not all(r['strict_2e_5_CPU_CUDA_pixel_parity_passed'] for r in checks)
out=Path('features/data/558_jump_GPU_execution_2026-10-06')
out.mkdir(exist_ok=False)
raw=[target/'acceptance.json',target/'independent-acceptance.json',root/'plan.json',root/'GPU-freeze.json',scratch/'benchmark_jump_virtual_stain_gpu.py',scratch/'verify_jump_saved_scorecard_r1.py',scratch/'verify_jump_saved_scorecard_r2.py',scratch/'verify_jump_saved_scorecard_r3.py',Path(__file__),target/'virtual_stain_c0.pt',target/'actual-Cellpose-SAM-scorecard.csv']
raw.extend(sorted(target.glob('*-labels.npz')))
raw.extend(sorted(target.glob('*-prediction.npy')))
raw.extend(sorted(target.glob('*-profile.json')))
raw.extend(scratch/name for name in ('jump-virtual-stain-gpu-r1.log','jump-saved-scorecard-verification-r1.log','jump-saved-scorecard-verification-r2.log','jump-saved-scorecard-verification-r3.log'))
for source in raw:
    dest=out/source.name
    if source.suffix in ('.log','.npy') or source.name.endswith('-profile.json'):
        dest=dest.with_name(dest.name+'.gz')
        dest.write_bytes(gzip.compress(source.read_bytes(),compresslevel=6,mtime=0))
        assert hashlib.sha256(gzip.decompress(dest.read_bytes())).hexdigest()==digest(source)
    else: shutil.copyfile(source,dest)
receipt={'item':558,'actual_normal_GPU_training_and_Cellpose_scorecard_with_independent_item_532_rescoring_accepted':True,'normal_turn_terminal_rc':0,'training_fields':48,'held_out_fields':20,'plate_disjoint_split':True,'epochs':20,'same_current_real_Cellpose_SAM_on_real_predicted_brightfield_arms':True,'all_40_normal_rows_independently_verified':True,'original_predictions_archived':20,'original_Cellpose_labels_archived':60,'trained_model_and_full_positive_CUDA_traces_archived':True,'summary_mean_per_field':actual['summary_mean_per_field'],'source_model_input_and_plan_hashes_verified':True,'normal_CPU_vs_CUDA_prediction_diagnostic':{'strict_2e_5_fields_passed':sum(r['strict_2e_5_CPU_CUDA_pixel_parity_passed'] for r in checks),'fields':20,'max_absolute_pixel_error':max(r['normal_CPU_saved_model_replay_max_abs_error'] for r in checks),'strict_parity_not_claimed_failures_retained':True},'real_stain_Cellpose_predictions_are_reference_not_human_gold':True,'fresh_training_not_unavailable_historical_checkpoint':True,'no_second_cell_line_four_biological_replicates_or_independent_biological_accuracy_claim':True,'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}}
(out.with_suffix('.json')).write_text(json.dumps(receipt,indent=2)+'\n')
note='''
2026-10-06 workstation virtual staining real GPU acceptance: normal turn 558-jump-virtual-stain-cuda-20261006-r1 is terminal rc=0. Fresh normal U-Net training completes all twenty predeclared epochs on 48 acquired brightfield/Hoechst paired fields from BR00117035; all twenty BR00117036 fields are held out by plate. The same actual default Cellpose-SAM CUDA segmenter runs on real, predicted and first-brightfield arms, with positive training and first/last segmentation CUDA profiles. Mean predicted F1 is 0.930977 at IoU 0.5 and 0.787011 at 0.75, versus brightfield 0.328579/0.141889; Pearson/SSIM are 0.929126/0.864450 versus 0.169027/0.072088. Separate CPU item-532 matching and pixel rescoring reproduce all forty actual report rows and the canonical CSV from all sixty original GPU masks and twenty predictions. Source/model/input/plan hashes remain exact. The saved normal model also runs on all twenty CPU fields, but strict 2e-5 CPU/CUDA pixel parity fails (maximum absolute difference 0.000235394); this is preserved as a separate diagnostic, not waived or claimed equivalent. The first private verifier additionally expected crop metadata that the normal checkpoint does not store; the original failed logs are retained and frozen invocation/epoch losses verify training parameters separately from actual checkpoint metadata. Receipt 558_jump_GPU_execution_2026-10-06.json archives the full trained checkpoint, all original predictions/masks, complete compressed CUDA traces, terminal execution and independent scoring. This closes the outstanding normal Cellpose GPU scorecard on genuine held-out label-free paired data. Real-stain Cellpose labels are algorithmic reference, not human gold; this is fresh training, not reuse of the unavailable historical checkpoint. Second cell line/published weights remain optional as recorded on 2026-10-03. DINOv2 human-labelled retrieval remains queued next; Home retains CPU CI/Qt/source and protected jobs stay untouched.
'''
for path in ('features/future/558_virtual_staining.txt','features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt','features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:stream.write(note)
print('PASS: complete real GPU virtual-stain training/model/60 masks/20 predictions/Cellpose scorecard and independent item-532 rescoring archived; strict cross-device diagnostic failure retained.',flush=True)
