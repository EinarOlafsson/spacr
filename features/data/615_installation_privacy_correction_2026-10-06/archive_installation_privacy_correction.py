from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
prior = scratch / 'tutorial-installation-completion-r1'
stage = scratch / 'tutorial-installation-completion-r2'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
audio = read(prior / 'current-complete-audio-acceptance.json')
assert audio['passed'] and audio['all_150_tracks_verified']
assert sum(len(row['tracks']) for row in audio['lessons'].values()) == 150
freeze = read(prior / 'frozen-narration-inputs.json')
assert all(digest(path) == expected for path, expected in freeze.items())
correction = read(stage / 'privacy-correction-preservation.json')
assert correction['passed']
for relative in correction['preserved_files']:
    assert digest(stage / relative) == digest(prior / relative)
log = (scratch / 'installation-current-narration-r1.log').read_text()
assert 'complete rendered=150 skipped=0 peak_failures=0' in log
assert '[615-current-installation-narration-20261006-r1] FINISH rc=0' in log
dest = Path('features/data/615_installation_privacy_correction_2026-10-06')
dest.mkdir(exist_ok=False)
shutil.copyfile(prior / 'current-complete-audio-acceptance.json', dest / 'r1-complete-audio-acceptance.json')
shutil.copyfile(stage / 'privacy-correction-preservation.json', dest / 'r2-privacy-preservation.json')
for name in ['installation-current-narration-r1.log', 'installation-audio-verification-r1.log',
             'installation-privacy-restage-r2.log', 'installation-installer-video-r1.log']:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
for name in ['restage_installation_privacy_r2.py', 'run_installation_narration.py',
             'verify_installation_audio.py', Path(__file__).name]:
    shutil.copyfile(scratch / name, dest / name)
capture_name = 'linux_public_plus_current_nightly_ui_r1'
with zipfile.ZipFile(dest / 'original-and-corrected-native-privacy-proof.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for root, tag in [(prior, 'rejected-private-r1'), (stage, 'corrected-private-r2')]:
        for relative in ['captures/' + capture_name + '/' + name for name in
                         ['07_privacy_keep_off.png', '02_installer_backend.png', 'frames.json', 'provenance.json']] + [
                         '04_platform_installers-focus.json', 'production/04_platform_installers/visual.json']:
            archive.write(root / relative, tag + '/' + relative)
report = {'item': 615, 'corrects_receipt': 'features/data/615_current_installation_stage_2026-10-06.json',
    'prior_no_21_tile_public_home_pixels_claim_retracted': True,
    'reason': 'The old public Home and alpha assay cards were visible behind its privacy dialog',
    'original_private_r1_04_video_rejected_for_publication': True,
    'replacement_is_original_native_recorded_all_off_consent_profile': True,
    'replacement_source_sha256': correction['replacement']['sha256'],
    'no_mock_dialog_or_pixel_repainting': True,
    'terminal_GPU_narration_returncode': 0, 'all_150_source_bound_audio_tracks_verified': True,
    'all_14_catalogs_39_reviews_and_audio_caption_bytes_preserved': True,
    '01_and_03_production_bytes_preserved': True,
    'r2_four_k_web_video_browser_candidate_publication_pending': True,
    'Mask_07_Conda_02_Home_05_Measure_08_unchanged': True,
    'next_normal_GPU_turn_planned': '552-instanseg-brightfield-cuda-20261006-r1',
    'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_installation_privacy_correction_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation installation visual correction and GPU handoff: the earlier stage receipt claim that no old twenty-one-tile public Home pixels entered lesson 04 was too broad and is retracted. Its public privacy dialog had old/alpha assay cards behind it. That private r1 scene/video is rejected for publication and its original evidence is retained. Separate r2 replaces only that visual with the actual unmodified recorded install-profile consent verification (all four options false), explicitly identified as settings verification rather than a dialog. All catalogs, thirty-nine reviews, 150 audio tracks/captions and lesson 01/03 production bytes remain exact. The normal narration GPU turn is terminal rc=0, renders all 150 with zero peak failures, and normal full-decode/source/runtime/timing/activity/dead-air checks pass all tracks. Receipt 615_installation_privacy_correction_2026-10-06.json records the correction and preservation; r2 video/frame/browser/candidate/publication remains open. Next normal GPU turn is 552-instanseg-brightfield-cuda-20261006-r1 for the original acquired full-field CPU-paired benchmark, with real HOME, 24 GiB cap and unchanged six-minute idle/ten-minute handoff. No other new GPU turn is queued. Mask 07 and accepted Conda/Home/Measure media stay unchanged. Home retains CPU/Qt/CI/source ownership and protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ['features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp']:
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: explicit privacy rejection/correction, preserved source/media and 150 complete verified narration tracks archived.', flush=True)
