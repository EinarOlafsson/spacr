from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
candidate = Path((stage / 'current-candidate-path.txt').read_text().strip())
dest = Path('features/data/662_yolo_media_acceptance_2026-10-06')
dest.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
audio = read(stage / 'current-complete-audio-acceptance.json')
frames = read(stage / 'current-frame-fidelity.json')
video = read(stage / 'production/14_make_masks/current-video-acceptance.json')
paths = read(scratch / 'yolo-native-path-audit-r2/14_make_masks.detect.json')
preservation = read(candidate / 'checks/yolo-preservation.json')
browser = read(candidate / 'checks/candidate-browser-checks.json')
mutations = read(candidate / 'checks/placeholder-mutation-checks.json')
assert audio['tracks_verified'] == 50 and audio['complete_50_voice_matrix']
assert frames['passed'] and len(frames['lessons']['14_make_masks']['scenes']) == 58
assert video['master_full_decode_passed'] and video['web_rendition']['accepted']
assert len(paths['images']) == 56 and all(not row['regions'] for row in paths['images'])
assert all(digest(row['image']) == row['sha256'] for row in paths['images'])
assert preservation['passed'] and len(preservation['unchanged_complete_lesson_objects']) == 14
assert set(preservation['unchanged_complete_lesson_objects'].values()) == {84}
assert browser['passed'] and len(browser['ready_playback_cases']) == 85
assert mutations['passed'] and all(row['observed_red'] for row in mutations['mutations'])
for source in [stage / 'current-complete-audio-acceptance.json', stage / 'current-frame-fidelity.json',
               stage / 'production/14_make_masks/current-video-acceptance.json',
               scratch / 'yolo-native-path-audit-r2/14_make_masks.detect.json',
               *[candidate / 'checks' / name for name in ('yolo-preservation.json', 'candidate-browser-checks.json', 'placeholder-mutation-checks.json')],
               *[scratch / name for name in ('verify_yolo_complete_audio.py', 'check_yolo_video_frames.py', 'prepare_yolo_candidate.py', 'finish_yolo_reference_video.py')],
               Path(__file__)]:
    shutil.copyfile(source, dest / source.name)
cases = sorted((stage / 'browser-web/14_make_masks').glob('*/playback-checks.json'))
assert len(cases) == 14 and all(read(p)['passed'] for p in cases)
for source in cases:
    shutil.copyfile(source, dest / (source.parent.name + '-playback.json'))
for name in ('yolo-narration-reference-r1.log', 'yolo-narration-complete-r1.log',
             'yolo-complete-audio-verification-r1.log', 'yolo-complete-audio-verification-r2.log',
             'yolo-reference-video-r1.log', 'yolo-video-frame-fidelity-r1.log',
             'yolo-native-path-audit-r1.log', 'yolo-native-path-audit-r2.log',
             'yolo-current-candidate-r1.log', 'yolo-current-candidate-r2.log',
             'yolo-current-candidate-preservation-r3.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
shutil.copyfile(stage / 'frame-review/redo-web.png', dest / 'visually-reviewed-redo-web.png')
receipt = {'accepted_prepublication_media': True, 'lesson': '14_make_masks', 'scenes': 58,
           'source_bound_voices_verified': 50, 'spoken_languages': 8, 'reviewed_translation_languages': 13,
           'native_unique_images_path_checked': 56, 'private_path_regions_found': 0,
           'exact_4k_scene_start_frames': 58, 'desktop_mobile_language_cases': 14,
           'candidate_player_routes_passed': 85, 'placeholder_mutation_guards_observed_red': 2,
           'catalog_languages_preserving_84_other_lessons': 14,
           'unchanged_other_media_records': preservation['unchanged_other_media_records'],
           'candidate_media_files': preservation['after_media_files'],
           'actual_decoded_web_box_and_class_label_visually_inspected': True,
           'Mask_07_unchanged': True, 'Home_05_Measure_08_and_Conda_02_preserved': True,
           'native_speaker_or_independent_listening_review_claimed': False,
           'biological_training_truth_claimed_for_demonstration_boxes': False,
           'deck_acceptance_receipt': 'features/data/662_yolo_deck_refresh_2026-10-06.json',
           'candidate': str(candidate), 'hosted_and_deployed_acceptance_recorded_separately': True,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
Path('features/data/662_yolo_media_acceptance_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation current YOLO media acceptance: all 50 narration tracks across eight spoken languages pass normal full decode/source/runtime/fingerprint/timing/activity/dead-air gates, with zero renderer peak failures. All 58 4K scene-start decoded pixel arrays match their independently composed native captures under the approved codec; the current 1440p video decodes with exact frame timing. All 14 desktop/mobile narration/caption cases, all 85 candidate routes and both placeholder mutation guards pass. Full native-size overlapping/brightness-lifted OCR on all 56 unique native scene images finds zero private path regions, with every source hash current. The actual decoded web Redo scene was inspected: the blue box outline and class label remain visible. All fourteen catalogs preserve every other 84 complete lesson object; 4,796 other media records remain exact. Receipt 662_yolo_media_acceptance_2026-10-06.json archives full normal evidence and failed private configuration checks. Page 32 has separate accepted deck evidence. Mask 07 and the deployed Measure/Home/Conda corrections stay intact. Immutable hosted playback and actual nightly deployed readback are separate acceptance steps; demonstration boxes are not biological training truth and no native-speaker/listening signoff is asserted.\n'
for path in ('features/new/662_make_masks_yolo_bounding_box_annotations.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: archived complete current YOLO audio/video/path/browser/preservation evidence.', flush=True)
