from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-installation-completion-r2'
candidate = Path((stage / 'current-candidate-path.txt').read_text().strip())
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = read(candidate / 'publication-receipt.json')
browser = read(candidate / 'checks/published-media-browser-checks.json')
assert receipt['tag'] and receipt['readback']['passed']
assert receipt['readback']['downloaded_sha256_matched'] == receipt['media_files'] == 5036
assert receipt['readback']['metadata_matched'] == 5036
assert not receipt['readback']['metadata_failures'] and not receipt['readback']['download_failures']
assert browser['passed'] and len(browser['ready_playback_cases']) == 85
assert browser['media_root'] == receipt['media_root']
assert receipt['commit'] == '0889ee14f5a7368a791df27c7331862a6c7741b4'
for path, expected in read(stage / 'accepted-r2-source-media-freeze.json').items():
    assert digest(stage / path) == expected
preservation = read(candidate / 'checks/installation-preservation.json')
assert preservation['passed'] and set(preservation['unchanged_complete_lesson_objects'].values()) == {82}
for filename in preservation['unchanged_complete_lesson_objects']:
    assert digest(candidate / 'web/catalog' / filename) == digest(Path('docs/source/_extra/tutorials/catalog') / filename)
subprocess.run([sys.executable, 'tools/tutorials/publish_release_candidate.py', 'record', str(candidate)], check=True)
dest = Path('features/data/615_installation_publication_2026-10-06')
dest.mkdir(exist_ok=False)
for source in (candidate / 'publication-receipt.json', candidate / 'checks/published-media-browser-checks.json', Path(__file__)):
    shutil.copyfile(source, dest / source.name)
for name in ('installation-current-upload-r2.log', 'installation-current-pages-r2.log',
             'installation-hosted-playback-r2.log', 'installation-hosted-playback-r3.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
report = {'item': 615, 'immutable_hosted_publication_accepted': True,
          'lessons': ['01_pypi_github', '03_pip_install', '04_platform_installers'],
          'media_files_fully_downloaded_and_hash_verified': 5036,
          'media_bytes_downloaded_and_verified': receipt['readback']['bytes_read_back'],
          'immutable_media_commit': receipt['commit'], 'hosted_routes_passed': 85,
          'current_150_audio_tracks_and_30_native_scene_source_media_freeze_verified': True,
          'all_82_other_complete_lessons_per_fourteen_catalogs_preserved': True,
          'unchanged_other_media_records': preservation['unchanged_other_media_records'],
          'Mask_07_Home_05_Measure_08_Conda_02_YOLO_14_preserved': True,
          'prior_privacy_dialog_old_Home_r1_visual_not_published': True,
          'native_Windows_macOS_or_native_speaker_review_not_claimed': True,
          'actual_nightly_deployment_and_phone_readback_pending': True,
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/615_installation_publication_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation corrected installation immutable publication: all 5,036 declared media files are fully downloaded and hash-verified (5,573,313,724 bytes) at new immutable commit 0889ee14f5a7368a791df27c7331862a6c7741b4. All eighty-five actual hosted player routes pass. Normal Pages publication and hold release are recorded; the accepted 150-track/thirty-scene current source/media freeze remains exact. Every other eighty-two complete lesson object in all fourteen catalogs and all 4,730 other media records are preserved, including unchanged Mask 07 and corrected Home/Measure/Conda/YOLO. The rejected old Home privacy-dialog visual is absent. Receipt 615_installation_publication_2026-10-06.json archives complete immutable readback, hosted browser checks and normal publication; the failed private CLI attempt is retained separately. Actual nightly deployment/full-video/phone readback remains pending after push. No native Windows/macOS capture or native-speaker review is claimed. The next robustness GPU turn is queued normally and no protected job is touched. Home retains CPU/Qt/CI/application-source ownership; all GPU work remains workstation-owned.\n'
for path in ('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: normal immutable installation publication, all-file hash readback and all 85 hosted routes recorded; actual deployed nightly readback remains separate.', flush=True)
