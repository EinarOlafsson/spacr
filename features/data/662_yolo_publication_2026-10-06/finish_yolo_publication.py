from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
candidate = Path((stage / 'current-candidate-path.txt').read_text().strip())
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
publication = read(candidate / 'publication-receipt.json')
browser = read(candidate / 'checks/published-media-browser-checks.json')
assert publication['tag'] and publication['readback']['passed']
assert publication['readback']['downloaded_sha256_matched'] == publication['media_files'] == 4898
assert browser['passed'] and len(browser['ready_playback_cases']) == 85
assert browser['media_root'] == publication['media_root']
compatibility = read(candidate / 'web/translation-compatibility.json')['entries']
selected = [row for row in compatibility if row['lesson'] == '14_make_masks']
assert len(selected) == 13 and all(row['status'] == 'source_bound_review' and not row['reason'] for row in selected)
preservation = read(candidate / 'checks/yolo-preservation.json')
assert len(preservation['unchanged_complete_lesson_objects']) == 14
for filename in preservation['unchanged_complete_lesson_objects']:
    assert digest(candidate / 'web/catalog' / filename) == digest(Path('docs/source/_extra/tutorials/catalog') / filename)
subprocess.run([sys.executable, 'tools/tutorials/publish_release_candidate.py', 'record', str(candidate)], check=True)
dest = Path('features/data/662_yolo_publication_2026-10-06')
dest.mkdir(exist_ok=True)
for source in (candidate / 'publication-receipt.json', candidate / 'checks/published-media-browser-checks.json', Path(__file__)):
    shutil.copyfile(source, dest / source.name)
for name in ('yolo-current-upload-r1.log', 'yolo-current-pages-r1.log', 'yolo-hosted-playback-r1.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
receipt = {'hosted_publication_verified': True, 'nightly_deployment_verified': False,
           'lesson': '14_make_masks', 'source_bound_translated_entries': 13,
           'complete_narration_matrix': 50, 'scenes': 58,
           'all_hosted_file_hashes_verified': 4898, 'hosted_player_routes_passed': 85,
           'hosted_bytes_read_back': publication['media_bytes'],
           'immutable_media_revision': publication['commit'], 'publication': publication,
           'Mask_07_unchanged': True, 'Home_05_Measure_08_and_Conda_02_preserved': True,
           'unchanged_complete_other_lessons_per_catalog': 84, 'catalog_languages': 14,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
Path('features/data/662_yolo_publication_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation YOLO immutable hosted publication: normal upload and complete 5.52 GB readback pass for all 4,898 declared files, pinned to f3fe5e321a9734c103c8c83d3ca0b9753e7df79f. All 85 hosted player routes pass against that exact revision; lesson 14 has all 58 scenes, 50 verified voices and 13 source-bound translated entries. Normal Pages publication and hold release are recorded; fourteen catalogs retain all other 84 whole lesson objects and 4,796 other media records remain exact. Mask 07, the current Measure opening/Home navigation and accepted Conda frame remain unchanged. Receipt 662_yolo_publication_2026-10-06.json archives immutable upload/full-byte readback, exact hosted playback and normal publication. Actual nightly deployed/mobile/full-video readback remains OPEN after push. No biological truth is asserted for demonstration boxes. The source-current all-nine API strict audit is still live; its ten-language whole-record preservation, twelve catalog tests and full strict multilingual Sphinx already pass. Next scientific GPU label 552-instanseg-current-cuda-20261006-r1 remains queued under the normal scheduler. Home keeps CPU CI/coverage/Qt/source ownership.\n'
for path in ('features/new/662_make_masks_yolo_bounding_box_annotations.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: normal immutable hosted YOLO publication and all 85 routes; actual nightly deployment is separate.', flush=True)
