from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-installation-completion-r2'
candidate = Path((stage / 'current-candidate-path.txt').read_text().strip())
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
identities = ('01_pypi_github', '03_pip_install', '04_platform_installers')
audio = read(stage / 'current-complete-audio-acceptance.json')
frames = read(stage / 'current-frame-fidelity.json')
visual = read(stage / 'current-original-frame-visual-review.json')
native = read(stage / 'current-native-frame-path-and-alpha-acceptance.json')
preservation = read(candidate / 'checks/installation-preservation.json')
browser = read(candidate / 'checks/candidate-browser-checks.json')
mutations = read(candidate / 'checks/placeholder-mutation-checks.json')
assert audio['passed'] and audio['all_150_tracks_verified']
assert frames['passed'] and sum(len(row['scenes']) for row in frames['lessons'].values()) == 30
assert all(row['all_decoded_pixels_identical'] for lesson in frames['lessons'].values() for row in lesson['scenes'])
assert visual['passed'] and native['passed'] and preservation['passed']
assert browser['passed'] and len(browser['ready_playback_cases']) == 85
assert mutations['passed'] and all(row['observed_red'] for row in mutations['mutations'])
assert len(preservation['unchanged_complete_lesson_objects']) == 14
assert set(preservation['unchanged_complete_lesson_objects'].values()) == {82}
for row in visual['all_30_frame_hashes']:
    assert digest(row['source']) == row['sha256']
dest = Path('features/data/615_installation_media_2026-10-06')
dest.mkdir(exist_ok=False)
for name in ('current-complete-audio-acceptance.json', 'current-frame-fidelity.json',
             'current-native-frame-path-and-alpha-acceptance.json', 'current-original-frame-visual-review.json',
             'privacy-correction-preservation.json'):
    shutil.copyfile(stage / name, dest / name)
for name in ('installation-preservation.json', 'candidate-browser-checks.json', 'placeholder-mutation-checks.json'):
    shutil.copyfile(candidate / 'checks' / name, dest / name)
frozen = {}
for identity in identities:
    folder = stage / 'production' / identity
    video = read(folder / 'current-video-acceptance.json')
    report = read(folder / 'current-browser-acceptance.json')
    assert video['master_full_decode_passed'] and video['web_rendition']['accepted']
    assert report['passed'] and len(report['checks']) == 14
    assert report['checks'][0]['case'] == 'en-af_heart-sentence-cues'
    for entry in report['checks']:
        path = stage / entry['report']
        checked = read(path)
        assert checked['passed'] and checked['checked_web_rendition']['sha256'] == video['web_rendition']['rendition_sha256']
        shutil.copyfile(path, dest / (identity + '-' + entry['case'] + '-playback.json'))
    for name in ('current-video-acceptance.json', 'current-browser-acceptance.json'):
        shutil.copyfile(folder / name, dest / (identity + '-' + name))
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name not in ('historical-extra-English-browser-acceptance.json',):
            frozen[str(path.relative_to(stage))] = digest(path)
for directory in ('catalog', 'captures', 'web-renditions'):
    for path in sorted((stage / directory).rglob('*')):
        if path.is_file():
            frozen[str(path.relative_to(stage))] = digest(path)
for path in stage.glob('*-focus.json'):
    frozen[str(path.relative_to(stage))] = digest(path)
(stage / 'accepted-r2-source-media-freeze.json').write_text(json.dumps(frozen, indent=2) + '\n')
shutil.copyfile(stage / 'accepted-r2-source-media-freeze.json', dest / 'accepted-r2-source-media-freeze.json')
with zipfile.ZipFile(dest / 'current-native-frames-provenance-source-reviews-and-fidelity.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((stage / 'captures').rglob('*')):
        if path.is_file():
            archive.write(path, str(path.relative_to(stage)))
    for identity in identities:
        for path in sorted((stage / 'production' / identity).glob('*.json')):
            archive.write(path, str(path.relative_to(stage)))
        archive.write(stage / (identity + '-focus.json'), identity + '-focus.json')
    for path in sorted((stage / 'frame-review').rglob('*.png')):
        archive.write(path, str(path.relative_to(stage)))
for name in ('render_and_finish_installation_video_r2.py', 'verify_installation_native_frames_r2.py',
             'check_installation_video_frames_r2.py', 'verify_installation_browsers_r2.py',
             'record_installation_visual_review_r2.py', 'prepare_installation_candidate_r2.py',
             'update_installation_default_browser_receipts.py', Path(__file__).name):
    shutil.copyfile(scratch / name, dest / name)
for path in sorted(scratch.glob('installation-*.log')):
    if any(word in path.name for word in ('candidate', 'video-r2', 'video-r3', 'native', 'fidelity', 'browser', 'visual-review')):
        (dest / (path.name + '.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
report = {'item': 615, 'current_r2_installation_prepublication_media_accepted': True,
          'lessons': list(identities), 'source_bound_tracks_verified': 150,
          'spoken_languages': 8, 'voices_per_lesson': 50, 'caption_and_narration_catalog_languages': 14,
          'exact_source_bound_four_k_scene_start_frames': 30,
          'native_originals_visually_and_privacy_alpha_checked': 30,
          'desktop_mobile_narration_caption_cases_passed': 42,
          'normal_default_English_sentence_cues_verified': True,
          'candidate_routes_passed': 85, 'placeholder_mutation_guards_observed_red': 2,
          'all_82_other_complete_lesson_objects_preserved_per_catalog': True,
          'other_media_records_preserved': preservation['unchanged_other_media_records'],
          'Mask_07_Conda_02_Home_05_Measure_08_YOLO_14_preserved': True,
          'prior_r1_old_Home_privacy_visual_rejected': True,
          'r2_privacy_scene_uses_original_unmodified_all_false_consent_profile': True,
          'recorded_public_1513_and_conda_1508_distinction_preserved': True,
          'Windows_macOS_guidance_not_claimed_as_native_capture': True,
          'no_native_speaker_or_independent_listening_review_claimed': True,
          'private_candidate_missing_default_voice_and_missing_directory_attempts_retained': True,
          'immutable_upload_hosted_and_actual_nightly_deployment_pending': True,
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_installation_media_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation corrected installation media acceptance: r2 lessons 01/03/04 pass all 150 source-bound narration tracks, all thirty exact decoded 4K scene-start frames, full master/web decode/timing and all forty-two desktop/mobile narration/caption cases, including normal default English sentence cues. Native originals are visually reviewed and full-resolution path/alpha checked. The old Home privacy-dialog scene stays rejected; its replacement is the unmodified actual all-false recorded consent profile. Normal candidate acceptance passes all eighty-five routes and both red mutation guards. Every other eighty-two complete lesson object is exact in all fourteen catalogs and its media records are retained, including unchanged Mask 07 and accepted Conda/Home/Measure/YOLO. Receipt 615_installation_media_2026-10-06.json archives current source/media freeze, native capture/provenance, reviews, exact frame/browser/preservation receipts and completed failed private configuration attempts. Public 1.5.1.3/current-nightly references and conda 1.5.0.8 stay explicitly distinguished; Windows/macOS reference guidance does not claim native capture. Immutable upload, hosted playback and actual nightly deployed readback remain open. Home retains CPU/Qt/CI/source ownership and all GPU work remains workstation-owned.\n'
for path in ('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: all corrected installation media, exact original frames and normal candidate gates archived.', flush=True)
