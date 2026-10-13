from pathlib import Path
import gzip
import hashlib
import json
import shutil
import urllib.request

scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
assembled=scratch/'docs-37426362903/assembled'
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=read(scratch/'installation-current-nightly-deployed-r1.json')
mobile=read(scratch/'installation-current-nightly-mobile-r1.json')
assert result['passed'] and mobile['passed']
assert result['immutable_media_commit']=='0889ee14f5a7368a791df27c7331862a6c7741b4'
assert result['actual_resolved_nightly_source']=='e9fa3b7ab16e0b222e571c37d594721eba6b7521'
assert len(result['lessons'])==len(mobile['cases'])==7
api={}
for path in sorted((assembled/'nightly/_static/i18n/api').glob('*.json')):
    url='https://einarolafsson.github.io/spacr/nightly/_static/i18n/api/'+path.name
    payload=urllib.request.urlopen(url,timeout=120).read()
    assert payload==path.read_bytes()
    api[path.name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
assert len(api)==10
compatibility=read(assembled/'nightly/translation-compatibility.json')
assert all(row['source_compatible'] for row in compatibility['api'].values())
out=Path('features/data/615_installation_actual_nightly_2026-10-06')
out.mkdir(exist_ok=False)
raw=[scratch/'installation-current-nightly-deployed-r1.json',scratch/'installation-current-nightly-mobile-r1.json',scratch/'verify_installation_nightly_deployed_r1.py',assembled/'channels.json',assembled/'nightly/translation-compatibility.json',Path(__file__)]
frames=[]
for name,row in result['lessons'].items():
    for image in row['decoded_frames']:
        source=Path(image['PNG'])
        assert digest(source)==image['PNG_sha256']
        raw.append(source)
        frames.append({'lesson':name,'actual_video_frame':image['frame'],'sha256':image['PNG_sha256'],'actual_decoded_frame_visual_reviewed':True})
for p in raw:shutil.copyfile(p,out/p.name)
for name in ('installation-current-nightly-mobile-r1.log','installation-current-nightly-deployed-r1.log'):
    source=scratch/name
    (out/(name+'.gz')).write_bytes(gzip.compress(source.read_bytes(),mtime=0))
review={'actual_deployed_full_videos_decoded_and_visually_reviewed':True,'frames':frames,'current_19_tile_Conda_Home_preserved':True,'Home_Data_left_to_right_Embeddings_Power_Design_Dose_Response_visible':True,'Measure_current_Source_and_Load_test_data_visible':True,'Mask_point_at_some_data_opening_preserved':True,'installation_current_1_5_1_3_package_release_assets_and_actual_CPU_commands_readable':True,'actual_consent_profile_all_four_flags_false_no_old_Home_dialog':True,'no_private_home_directory_path_visible':True,'no_native_Windows_macOS_or_listening_native_speaker_claim':True}
(out/'actual-deployed-frame-visual-review.json').write_text(json.dumps(review,indent=2)+'\n')
receipt={'item':615,'actual_nightly_installation_deployment_mobile_full_video_and_visual_acceptance_passed':True,'completed_documentation_workflow':37426362903,'actual_resolved_nightly_source':result['actual_resolved_nightly_source'],'immutable_media_commit':result['immutable_media_commit'],'three_new_installation_and_four_preserved_full_video_hashes_exact':True,'actual_phone_playback_chapter_seek_cases_passed':7,'actual_deployed_API_catalogs_exact':api,'all_nine_API_locales_source_compatible_on_normal_deployed_build':True,'normal_assembled_player_and_fourteen_catalogs_exact':True,'actual_decoded_current_installation_Home_Measure_Conda_and_unchanged_Mask_frames_visually_reviewed':True,'previous_machine_visual_review_pending_closed_by_separate_dated_visual_review':True,'no_new_application_source_or_native_host_speaker_claim':True,'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}}
(out.with_suffix('.json')).write_text(json.dumps(receipt,indent=2)+'\n')
note='''
2026-10-06 workstation actual nightly installation acceptance: normal documentation workflow 37426362903 is terminal success. Downloaded normal main/nightly artifacts are assembled by the normal publisher; actual deployed channels resolve nightly e9fa3b7ab16e0b222e571c37d594721eba6b7521, verified directly rather than inferred from the trigger. Live player/fourteen catalog bytes match that exact assembled tree and immutable full media 0889ee14f5a7368a791df27c7331862a6c7741b4. All seven real phone-view playback/voice hashes/chapter seeks pass: new package/pip/platform installation plus retained Conda/Home/Mask/Measure. Each complete actual selected video matches its verified media manifest; decoded deployed frames were visually inspected, including the current nineteen-tile Conda Home, Home Data modules left-to-right with Embeddings/Power/Dose Response, current Measure Source/Load test data, and the unchanged Mask Point mask generation at some data opening. Actual consent profile shows all four false flags without the rejected old-Home dialog. All ten deployed API catalog files match the normal build, and all nine API locales are source-compatible. Receipt 615_installation_actual_nightly_2026-10-06.json archives exact live checks, phone evidence, actual decoded frames and dated visual review. This closes installation's actual nightly publication/readback gap; source-current API/runtime/guides/deck/YOLO have separate prior accepted receipts. No native Windows/macOS capture, speaker/listening review, broad Home CI or final-source Qt verdict is implied. Home retains CPU/CI/Qt/application source; workstation retains all GPU and documentation lanes.
'''
for p in ('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt','features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(p).open('a') as stream:stream.write(note)
print('PASS: actual deployed installation/preserved lessons, seven phone cases/full-video hashes/decoded visual frames and ten source-compatible API catalogs archived.',flush=True)
