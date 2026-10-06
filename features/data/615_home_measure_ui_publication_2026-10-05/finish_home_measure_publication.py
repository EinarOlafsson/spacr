from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
candidate = Path((scratch / 'tutorial-home-measure-ui-current-r1/current-candidate-path.txt').read_text().strip())
read = lambda path: json.loads(path.read_text())
publication = read(candidate / 'publication-receipt.json')
assert publication['tag'] and publication['readback']['passed']
assert publication['readback']['downloaded_sha256_matched'] == publication['media_files'] == 4852
compatibility = read(candidate / 'web/translation-compatibility.json')['entries']
selected = [row for row in compatibility if row['lesson'] in {'05_home', '08_measure'}]
assert len(selected) == 26 and all(row['status'] == 'source_bound_review' and not row['reason'] for row in selected)
commands = [
    ['publish_release_candidate.py', 'pages', str(candidate), '--cache-key', 'home-measure-ui-20261005-ke3jjmi6'],
    ['verify_release_candidate.py', str(candidate), '--published', 'docs/source/_extra/tutorials'],
    ['publish_release_candidate.py', 'record', str(candidate)],
]
for command in commands:
    subprocess.run([sys.executable, 'tools/tutorials/' + command[0], *command[1:]], check=True)
publication = read(candidate / 'publication-receipt.json')
browser = read(candidate / 'checks/published-media-browser-checks.json')
assert browser['passed'] and len(browser['ready_playback_cases']) == 85
assert browser['media_root'] == publication['media_root']
destination = Path('features/data/615_home_measure_ui_publication_2026-10-05')
destination.mkdir(exist_ok=True)
for source in [candidate / 'publication-receipt.json', candidate / 'checks/published-media-browser-checks.json', Path(__file__)]:
    shutil.copyfile(source, destination / source.name)
(destination / 'home-measure-ui-upload-r1.log.gz').write_bytes(gzip.compress((scratch / 'home-measure-ui-upload-r1.log').read_bytes(), mtime=0))
artifacts = {str(target): {'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size} for target in sorted(destination.iterdir())}
receipt = {'hosted_publication_verified': True, 'nightly_deployment_verified': False, 'lessons': ['05_home', '08_measure'], 'all_hosted_file_hashes_verified': 4852, 'hosted_player_routes_passed': 85, 'translated_source_bound_reviews': 26, 'Mask_07_unchanged': True, 'Conda_02_accepted_scene_preserved': True, 'publication': publication, 'artifacts': artifacts}
Path('features/data/615_home_measure_ui_publication_2026-10-05.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-05 workstation Home/Measure hosted publication: normal immutable upload and complete byte readback pass for all 4,852 files. All 85 hosted player routes pass against the exact pinned revision; all 26 translated Home/Measure entries are source-bound reviews. Normal Pages/checkpoint publication is ready for nightly. Mask 07, other 83 lessons and the accepted Conda scene are retained. Actual nightly deployment and mobile/video readback remain separate and pending. Receipt 615_home_measure_ui_publication_2026-10-05.json.\n'
for path in ['features/325_two_sessions_one_repo_working_protocol.temp', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt']:
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: immutable hosted media, all 85 player routes and normal publication record; nightly deployment still pending', flush=True)
