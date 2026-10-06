from pathlib import Path
import json
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-installation-completion-r2'
baseline = Path((scratch / 'tutorial-make-masks-yolo-current-r1/current-candidate-path.txt').read_text().strip())
identities = ['01_pypi_github', '03_pip_install', '04_platform_installers']
read = lambda p: json.loads(p.read_text())
assert read(stage / 'current-frame-fidelity.json')['passed']
assert read(stage / 'current-native-frame-path-and-alpha-acceptance.json')['passed']
assert read(stage / 'current-complete-audio-acceptance.json')['all_150_tracks_verified']
assert read(stage / 'privacy-correction-preservation.json')['passed']
assert read(stage / 'current-original-frame-visual-review.json')['passed']
for identity in identities:
    folder = stage / 'production' / identity
    assert read(folder / 'current-video-acceptance.json')['master_full_decode_passed']
    assert read(folder / 'current-browser-acceptance.json')['passed']
sys.path.insert(0, 'tools/tutorials')
from build_appended_candidate import build
from validate_candidate import validate
pointer = stage / 'current-candidate-path.txt'
if pointer.exists():
    root = Path(pointer.read_text().strip())
    validate(root, include_hosted_media=True)
else:
    root = build(stage, baseline, identities, replace=True, host_web=True)
pointer.write_text(str(root) + '\n')
preserved = {}
for old_path in sorted((baseline / 'web/catalog').glob('*.json')):
    if not old_path.name.startswith(('lessons_', 'captions_')):
        continue
    before = {row['id']: row for row in read(old_path)['lessons']}
    after = {row['id']: row for row in read(root / 'web/catalog' / old_path.name)['lessons']}
    assert before.keys() == after.keys() and len(before) == 85
    for identity, lesson in before.items():
        if identity not in identities:
            assert after[identity] == lesson, (old_path.name, identity)
    preserved[old_path.name] = 82
assert len(preserved) == 14
before_media = {row['path']: row for row in read(baseline / 'release-manifest.json')['files'] if row['path'].startswith('media_host/')}
after_media = {row['path']: row for row in read(root / 'release-manifest.json')['files'] if row['path'].startswith('media_host/')}
retained = []
for path, record in before_media.items():
    if Path(path).parts[1] not in identities:
        assert after_media[path] == record, path
        retained.append(path)
proof = {'passed': True, 'baseline': str(baseline), 'baseline_revision': read(baseline / 'publication-receipt.json')['commit'],
    'selected_lessons': identities, 'unchanged_complete_lesson_objects': preserved,
    'unchanged_other_media_records': len(retained), 'Mask_07_unchanged': True,
    'Measure_08_Home_05_Conda_02_and_YOLO_14_complete_current_objects_and_media_retained': True,
    'before_media_files': len(before_media), 'after_media_files': len(after_media)}
(root / 'checks').mkdir(exist_ok=True)
(root / 'checks/installation-preservation.json').write_text(json.dumps(proof, indent=2) + '\n')
for script in ['verify_release_candidate.py', 'check_placeholder_mutations.py', 'checkpoint_release_candidate.py']:
    subprocess.run([sys.executable, 'tools/tutorials/' + script, str(root)], check=True)
print('PASS: current installation candidate, all 85 player routes, mutation guards and normal checkpoint; not yet uploaded.', flush=True)
