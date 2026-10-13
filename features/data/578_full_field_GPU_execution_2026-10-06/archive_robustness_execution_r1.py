from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'578-robustness-plate-r1'
target=root/'cuda-r2'
record=json.loads((target/'independent-acceptance.json').read_text())
assert record['all_128_actual_normal_GPU_calls_accepted_after_independent_CPU_saved_label_replay']
assert len(record['calls'])==128 and len(record['CUDA_profiles'])==2
out=Path('features/data/578_full_field_GPU_execution_2026-10-06')
out.mkdir(exist_ok=False)
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=[target/'independent-acceptance.json',root/'plan.json',root/'recipe.json',root/'GPU-freeze.json',scratch/'benchmark_robustness_plate_gpu_r2.py',scratch/'verify_robustness_saved_labels_r1.py',Path(__file__),scratch/'robustness-fullfield-report-r2.png']
raw.extend(sorted((root/'plate/qc').glob('segmentation_robustness_nucleus*')))
raw.extend(sorted(target.glob('call-*-labels.npz')))
raw.extend(sorted(target.glob('call-*-profile.json')))
raw.extend(scratch/p for p in ('robustness-plate-gpu-r2.log','robustness-saved-label-verification-r1.log'))
for source in raw:
    dest=out/source.name
    if source.suffix=='.log' or source.name.endswith('-profile.json'):
        dest=dest.with_name(dest.name+'.gz')
        dest.write_bytes(gzip.compress(source.read_bytes(),compresslevel=6,mtime=0))
        assert hashlib.sha256(gzip.decompress(dest.read_bytes())).hexdigest()==digest(source)
    else: shutil.copyfile(source,dest)
review={'reviewed_native_normal_PDF_render':str(scratch/'robustness-fullfield-report-r2.png'),'render_sha256':digest(scratch/'robustness-fullfield-report-r2.png'),'actual_visual_review_complete':True,'all_eight_grid_rows_four_metrics_and_threshold_mark_readable':True,'baseline_zero_and_threshold_6_27_percent_count_29_percent_lost_visible':True,'normal_report_colours_not_altered':True}
(out/'actual-normal-PDF-visual-review.json').write_text(json.dumps(review,indent=2)+'\n')
receipt={'item':578,'actual_128_normal_CUDA_full_field_calls_and_independent_saved_label_report_replay_accepted':True,'fields':16,'full_field_shape':[1994,1994],'original_GPU_turn_rc':1,'private_postflight_failure_closed_by_normal_field_to_fieldID_table_schema_verification':True,'no_GPU_rerun_or_application_source_change':True,'original_labels_archived':128,'first_last_positive_CUDA_profiles_archived':True,'normal_CSV_PDF_and_visual_review_complete':True,'source_model_plan_verified_before_and_after':True,'baseline_stable':True,'threshold_6_fragile':True,'scope':'All sixteen complete acquired fields of four wells of the example plate; not an entire production screen','no_independent_biological_accuracy_or_other_backend_claim':True,'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}}
(out.with_suffix('.json')).write_text(json.dumps(receipt,indent=2)+'\n')
note='''
2026-10-06 workstation full-field robustness GPU acceptance: normal retry 578-full-field-robustness-20261006-r2 completed all 128 genuine CUDA segmentations across every one of the sixteen acquired 1994x1994 fields and the unchanged eight-point grid. Both original first/last profiles contain actual CUDA kernels; all original masks are retained. The terminal turn rc=1 is preserved: its private final CSV comparator expected field, while normal spacr.tabular writes canonical fieldID. Separate CPU verification applies the normal schema, reconstructs every metric/summary from saved original GPU labels, and matches both normal CSVs exactly without another inference. Source/model/plan hashes remain exact. The normal PDF was visually reviewed and is readable. Baseline is stable; deliberately fragile cellprob_threshold 6 loses 29% of baseline objects and changes counts by 27%, and is flagged. Receipt 578_full_field_GPU_execution_2026-10-06.json archives all 128 original masks, full lossless-compressed CUDA traces, terminal failed log, independent replay and real PDF/render. This closes the acquired full-field robustness GPU/report gap for all example fields, not full production-screen throughput or biological accuracy. There is no application source change or waived guard. Queued 558 virtual staining and 565 human-labelled DINOv2 follow normal idle/handoff; Home retains CPU CI/coverage/Qt/source, and protected jobs remain untouched.
'''
for p in ('features/future/578_segmentation_robustness_report.txt','features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt','features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(p).open('a') as stream: stream.write(note)
print('PASS: 128 genuine CUDA full-field outputs, full profiles, independent normal report replay and actual readable PDF archived; terminal private comparator failure preserved.',flush=True)
