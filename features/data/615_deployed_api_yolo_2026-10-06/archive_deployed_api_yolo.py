from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
api = read(scratch / 'deployed-data-art-api-r2/acceptance.json')
media = read(scratch / 'yolo-preserved-tutorials-deployed-r1.json')
mobile = read(scratch / 'yolo-api-nightly-mobile-r1.json')
assert api['passed'] and len(api['cases']) == 108 and all(row['passed'] for row in api['cases'])
assert media['passed'] and mobile['passed'] and len(mobile['cases']) == 4
assert api['source_commit'] == media['actual_documentation_source'] == '7d24dcf7760e63f38a85c311da3bd477d0be9479'
assert media['mobile_receipt_sha256'] == digest(scratch / 'yolo-api-nightly-mobile-r1.json')
for identity, row in media['lessons'].items():
    assert digest(scratch / (identity + '-yolo-nightly-video-r1.mp4')) == row['complete_video_sha256']
    assert digest(row['decoded_frame_png']) == row['decoded_frame_png_sha256']
    assert row['candidate_record_exact'] and row['deployed_phone_playback_and_seek_passed']
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
source = read(Path('tools/tutorials/lessons/14_make_masks.json'))
assert source['scenes'][17]['visual'] == 'yolo_06_drawn_box'
fidelity = read(stage / 'current-frame-fidelity.json')['lessons']['14_make_masks']['scenes'][17]
assert fidelity['video_frame'] == 7475
box = scratch / '14_make_masks-yolo-nightly-box-frame-r1.png'
assert box.is_file()
dest = Path('features/data/615_deployed_api_yolo_2026-10-06')
dest.mkdir(exist_ok=False)
for path in [scratch / 'deployed-data-art-api-r2/acceptance.json',
             scratch / 'yolo-preserved-tutorials-deployed-r1.json',
             scratch / 'yolo-api-nightly-mobile-r1.json',
             scratch / 'docs-37419061863/assembled/channels.json',
             scratch / 'verify_deployed_data_art_api_r2.py',
             scratch / 'verify_deployed_yolo_and_preserved_tutorials_r1.py',
             box, Path(__file__),
             *[Path(row['decoded_frame_png']) for row in media['lessons'].values()]]:
    shutil.copyfile(path, dest / path.name)
for path in sorted((scratch / 'deployed-data-art-api-r2').glob('*.png')):
    shutil.copyfile(path, dest / path.name)
for name in ('deployed-data-art-api-r1.log', 'deployed-data-art-api-r2.log',
             'yolo-api-nightly-mobile-r1.log', 'yolo-preserved-tutorials-deployed-r1.log',
             'yolo-preserved-tutorials-deployed-r2.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
review = {'passed': True, 'review_date': '2026-10-06',
          'review_scope': 'AI visual review of actual decoded deployed videos; no independent listening or native-speaker signoff',
          '08_measure': 'Source and Load test data controls, loaded example, obsolete Point measure at some data card absent',
          '05_home': 'Current Data row, including Embeddings, Power and Dose Response in left-to-right order',
          '02_conda_install': 'Current alpha-off nineteen-tile Home; only the previously requested Home scene was replaced',
          '14_make_masks': {'new_YOLO_scene_index_zero_based': 17, 'visual': 'yolo_06_drawn_box',
                            'decoded_web_frame': 7475, 'frame_sha256': digest(box),
                            'review': 'Blue drawn class-labelled box and normal Save boxes/Export YOLO controls visible',
                            'earlier_scene_45_frame_18137_is_puncta_not_new_YOLO_visual': True},
          'Mask_07_unchanged': True}
(dest / 'actual-deployed-frame-visual-review.json').write_text(json.dumps(review, indent=2) + '\n')
report = {'item': 615, 'actual_deployed_acceptance_passed': True,
          'completed_documentation_workflow': 37419061863,
          'actual_resolved_nightly_documentation_source': api['source_commit'],
          'actual_source_not_inferred_from_workflow_trigger': True,
          'immutable_media_commit': media['immutable_media_commit'],
          'actual_API_panels_passed': 108, 'changed_symbols': 12, 'translated_languages': 9,
          'actual_deployed_API_catalogs_exact': 10,
          'actual_tutorial_player_and_fourteen_catalog_files_exact': True,
          'four_actual_deployed_complete_video_hashes_exact': True,
          'four_actual_phone_playback_and_chapter_seek_cases_passed': True,
          'decoded_current_Measure_Home_Conda_and_new_YOLO_box_visually_reviewed': True,
          'Mask_07_unchanged': True,
          'historical_machine_report_visual_review_pending_closed_by_separate_dated_visual_receipt': True,
          'private_failed_source_inference_and_missing_pending_phone_receipt_attempts_retained': True,
          'installation_01_03_04_new_publication_still_pending': True,
          'native_speaker_or_independent_biological_training_accuracy_claimed': False,
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_deployed_api_yolo_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation actual deployed API/YOLO and maintainer tutorial acceptance: completed docs workflow 37419061863 resolves nightly source 7d24dcf7760e63f38a85c311da3bd477d0be9479, verified from its normal downloaded/assembled channels artifact rather than inferred from the workflow trigger. Ten actual API catalogs and three HTML pages are byte-exact; all 108 actual API panels (twelve changed symbols by nine locales) pass current-source/protected-literal/visible-language checks. Actual deployed tutorial player and all fourteen catalogs are exact at immutable media f3fe5e321a9734c103c8c83d3ca0b9753e7df79f. Four actual phone playback/seek cases pass, complete downloaded videos match their release hashes, and decoded current Measure Source/Load test data, left-to-right Home/Data, corrected Conda Home and new YOLO drawn-box frames are visually reviewed. The new YOLO scene is zero-based index 17/frame 7475; the separately retained index 45/frame 18137 is a puncta scene and is not claimed as a new YOLO demonstration. Receipt 615_deployed_api_yolo_2026-10-06.json archives source provenance, full hash checks, actual browser receipts, reviewed decoded frames and failed private preflights. Mask 07 remains unchanged. This closes the current deployed API/YOLO and requested Measure/Home/Conda readback scopes; new installation 01/03/04 publication and wider scientific tasks remain open. Home retains CPU CI/coverage/serial Qt and application-source ownership; all GPU tasks stay workstation-owned.\n'
for path in ('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/new/662_make_masks_yolo_bounding_box_annotations.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: actual deployed API, YOLO, Measure, Home and Conda acceptance archived; Mask 07 unchanged.', flush=True)
