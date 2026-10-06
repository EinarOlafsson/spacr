from pathlib import Path
import ast,gzip,hashlib,json,shutil,subprocess
s=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
proof=json.loads((s/'cell-dino-provenance-source-and-coverage-r1.json').read_text())
replay=json.loads((s/'cell-dino-provenance-full-cohort-replay-r1.json').read_text())
assert proof['passed'] and proof['cohort_passed']==147 and not proof['missing_added_executable_lines']
assert replay['passed'] and len(replay['comparisons'])==12
assert proof['source_sha256']==replay['source_sha256']==hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest()
assert '147 passed' in (s/'cell-dino-provenance-focused-compatible-r1.log').read_text()
out=Path('features/data/560_cell_dino_local_provenance_2026-10-06')
out.mkdir(exist_ok=False)
for name in ('cell-dino-provenance-source-and-coverage-r1.json','cell-dino-provenance-full-cohort-replay-r1.json','verify_cell_dino_provenance_full_cohorts_r1.py','run_foundation_api_tool_r2.py'):
    shutil.copyfile(s/name,out/name)
for name in ('cell-dino-provenance-coverage-r1.json','cell-dino-provenance-focused-r1.log','cell-dino-provenance-integration-r1.log','cell-dino-provenance-focused-compatible-r1.log','cell-dino-provenance-full-cohort-replay-r1.log'):
    (out/(name+'.gz')).write_bytes(gzip.compress((s/name).read_bytes(),mtime=0))
(out/'embeddings.py.gz').write_bytes(gzip.compress(Path('spacr/embeddings.py').read_bytes(),mtime=0))
(out/'prior-embeddings.py.gz').write_bytes(gzip.compress(subprocess.check_output(['git','show','HEAD:spacr/embeddings.py']),mtime=0))
shutil.copyfile(__file__,out/Path(__file__).name)
report={'item':560,'local_provenance_fix_passed':True,'source_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':proof['source_sha256'],'only_two_existing_provenance_functions_changed':True,'all_inference_spec_and_scientific_scorecard_functions_AST_exact':True,'focused_behavior_and_Qt_cases_passed':147,'missing_added_executable_lines':[],'actual_declared_local_file_bytes_hashed_without_model_load_or_download':True,'stable_regular_file_required':True,'missing_mismatched_modified_replaced_deleted_fifo_directory_unreadable_refused':True,'actual_storage_cross_plate_query_passes_with_verified_identity_and_refuses_changed_checkpoint':True,'all_twelve_original_full_human_fluorescence_scorecards_and_six_actual_checkpoint_digests_reverified':True,'all_measured_metrics_and_exact_random_AP_preserved_atol_1e_12':True,'no_pretrained_Cell_DINO_or_new_GPU_inference_claim':True,'normal_source_current_API_runtime_help_settings_guides_and_translation_refresh_pending':True,'original_base_environment_optional_skips_resolved_by_complete_compatible_environment_cohort':True,'artifacts':{str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir())}}
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
note='''
2026-10-06 workstation Cell-DINO local checkpoint provenance fixed:
_weights_on_disk now accepts explicit checkpoint_path/checkpoint_sha256 for
Cell-DINO, resolves the real path and hashes stable regular-file bytes without
loading/downloading a model or importing torch/timm/transformers. Actual bytes
must match the declared SHA-256; before/after/final path identity checks refuse
changed, replaced or removed files. Missing, malformed, mismatched, unreadable,
directory and FIFO inputs retain empty provenance. encoder_entry passes these
fields only for Cell-DINO; all existing backbone calls and provenance remain
unchanged. UNKNOWN provider/training provenance and verified=False remain honest.
Actual synthetic-vector database storage/cross-plate search accepts the real
verified file identity, then correctly refuses a changed checkpoint. This is
provenance/storage acceptance, not pretrained scientific model acceptance.

The combined compatible-environment CPU/Qt cohort passes 147 with no skips.
All 26 added executable lines are exercised; no whole-module/global coverage
claim. Only these two helpers change; all inference, spec and scientific
scorecard functions are AST-exact to the integrated Home source. All twelve
original full human-labelled fluorescence scorecards replay on this actual
source at the unchanged 1e-12 guard; all six real checkpoint digests agree
with the original frozen GPU plan. Receipt:
data/560_cell_dino_local_provenance_2026-10-06.json.

Home: the empty ModelEntry SHA gap is closed for a valid supplied local
Cell-DINO checkpoint. Do not overwrite these two coordinated helper bodies.
The workstation's normal API/runtime/Help/settings/guide and all-nine-language
refresh is next. Actual official pretrained Cell-DINO weights are still
unobtained; scientific/GPU acceptance remains OPEN. User theme revisions and
the preference-save native crash remain Home priority under N663. Protected
serial Qt 37457998285 is still running at older 237506317; leave it alone.
'''
for name in ('features/325_two_sessions_one_repo_working_protocol.temp','features/future/560_foundation_model_embeddings.txt','features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'):
    with Path(name).open('a') as f:f.write(note)
p=Path('features/HANDOFF.md')
t=p.read_text()
header='''## 2026-10-06 workstation Cell-DINO actual local provenance repaired
The coordinated _weights_on_disk/encoder_entry path now verifies supplied
Cell-DINO checkpoint_path/checkpoint_sha256 against stable regular-file bytes,
without loading/downloading a model. Invalid, changed/replaced/deleted or
nonregular files leave no digest. Valid identity enables actual stored-vector
cross-plate search; a changed checkpoint correctly refuses it. The combined
CPU/Qt cohort passes 147 with no skips and exercises all added executable lines.
Every inference/spec/scorecard function remains AST-exact. All twelve original
full human-fluorescence scorecards and six actual weight digests replay exactly
at the unchanged 1e-12 guard. Receipt:
data/560_cell_dino_local_provenance_2026-10-06.json. Home: preserve both helpers.
Normal API/runtime/Help/settings/guides and all-nine-language refresh is next;
o official pretrained Cell-DINO acquisition/scientific/GPU acceptance claimed.

'''
t=t.replace('## 2026-10-06 workstation publication accepted; local installation recovered\n',header+'## 2026-10-06 workstation publication accepted; local installation recovered\n',1)
t=t.replace('The newer Home source is integrated. Next workstation work is precise: make\nencoder_entry verify Cell-DINO\'s actual checkpoint_path/checkpoint_sha256,\nthen normally refresh API/runtime/Help/settings/guides and all nine languages\n','The newer Home source is integrated and Cell-DINO actual local provenance is\nrepaired above. Next workstation work is normal API/runtime/Help/settings/guide\nrefresh and all nine languages\n',1)
p.write_text(t)
print('PASS stable actual local provenance, complete bounded CPU/Qt evidence and exact scientific scorecards archived.',flush=True)
