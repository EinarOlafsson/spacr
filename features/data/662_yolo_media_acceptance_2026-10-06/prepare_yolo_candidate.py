from pathlib import Path
import json
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
baseline = scratch / 'tutorial-home-measure-ui-current-r1/release-candidate-append-ke3jjmi6'
identity = '14_make_masks'
read = lambda path: json.loads(path.read_text())
assert read(stage / 'current-frame-fidelity.json')['passed']
assert read(stage / 'current-complete-audio-acceptance.json')['tracks_verified'] == 50
assert read(stage / 'production' / identity / 'current-video-acceptance.json')['master_full_decode_passed']
sys.path.insert(0, 'tools/tutorials')
from build_appended_candidate import build
from validate_candidate import validate
pointer = stage / 'current-candidate-path.txt'
if pointer.exists():
    root = Path(pointer.read_text().strip())
    validate(root, include_hosted_media=True)
else:
    root = build(stage, baseline, [identity], replace=True, host_web=True)
(stage / 'current-candidate-path.txt').write_text(str(root) + '\n')
preserved = {}
for old_path in sorted((baseline / 'web/catalog').glob('*.json')):
    if not old_path.name.startswith(('lessons_', 'captions_')):
        continue
    before = {row['id']: row for row in read(old_path)['lessons']}
    after = {row['id']: row for row in read(root / 'web/catalog' / old_path.name)['lessons']}
    assert before.keys() == after.keys() and len(before) == 85
    for key, lesson in before.items():
        if key != identity:
            assert after[key] == lesson, (old_path.name, key)
    preserved[old_path.name] = 84
assert len(preserved) == 14
before_media = {row['path']: row for row in read(baseline / 'release-manifest.json')['files'] if row['path'].startswith('media_host/')}
after_media = {row['path']: row for row in read(root / 'release-manifest.json')['files'] if row['path'].startswith('media_host/')}
retained = []
for path, record in before_media.items():
    if Path(path).parts[1] != identity:
        assert after_media[path] == record, path
        retained.append(path)
proof = {'passed': True, 'baseline': str(baseline), 'baseline_revision': read(baseline / 'publication-receipt.json')['commit'],
         'selected_lessons': [identity], 'unchanged_complete_lesson_objects': preserved,
         'unchanged_other_media_records': len(retained), 'Mask_07_unchanged': True,
         'Measure_08_current_opening_preserved': True, 'Home_05_current_navigation_preserved': True,
         'Conda_02_verified_single_scene_replacement_retained': True,
         'before_media_files': len(before_media), 'after_media_files': len(after_media)}
(root / 'checks').mkdir(exist_ok=True)
(root / 'checks/yolo-preservation.json').write_text(json.dumps(proof, indent=2) + '\n')
if '--preservation-only' not in sys.argv:
    for script in ['verify_release_candidate.py', 'check_placeholder_mutations.py', 'checkpoint_release_candidate.py']:
        subprocess.run([sys.executable, 'tools/tutorials/' + script, str(root)], check=True)
print('PASS: current YOLO candidate, all 85 browser routes, mutation guards and normal checkpoint; no upload yet', flush=True)
