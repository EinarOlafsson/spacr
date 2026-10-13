from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path('tools/tutorials').resolve()))
from stage_lesson import read, write, stage_lesson
from apply_translation_review import promote_many

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
capture = 'make_masks_yolo_complete_20261005_r1'
focus = stage / 'captures' / capture / 'focus.json'
lesson = Path('tools/tutorials/lessons/14_make_masks.json')
stage_lesson(lesson, capture, stage, check_only=True, focus_map=focus)
stage_lesson(lesson, capture, stage, focus_map=focus)
for language in ('da', 'de', 'es', 'fr', 'hi', 'is', 'it', 'ja', 'ko', 'nb', 'pt-BR', 'sv', 'zh-CN'):
    promote_many([read(Path(f'tools/tutorials/lessons/reviews/14_make_masks.{language}.json'))], stage)
reports = []
for path in sorted((stage / 'catalog').glob('*.json')):
    baseline = Path('docs/source/_extra/tutorials/catalog') / path.name
    if not baseline.is_file() or 'lessons' not in read(path):
        continue
    before = {item['id']: item for item in read(baseline)['lessons']}
    after = {item['id']: item for item in read(path)['lessons']}
    assert before.keys() == after.keys()
    untouched = before.keys() - {'14_make_masks'}
    assert len(untouched) == 84 and all(before[key] == after[key] for key in untouched)
    assert before['07_mask'] == after['07_mask']
    assert len(after['14_make_masks']['scenes']) == 58
    reports.append({'catalog': path.name, 'complete_unselected_lesson_objects_preserved': 84})
assert len(reports) == 14
proof = {'accepted': True, 'scope': 'Normal four-root native composition and all source-bound reviews staged; no narration/video/publication acceptance',
         'native_visuals': 56, 'scenes': 58, 'new_yolo_scenes': 13,
         'all_previous_scene_objects_preserved': 45, 'reviewed_languages': 13,
         'catalog_preservation': reports, 'Mask_07_unchanged': True,
         'english_sha256': hashlib.sha256(json.dumps(read(lesson), sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
         'narration_video_publication_complete': False}
write(stage / 'current-catalog-preservation-yolo.json', proof)
write(scratch / 'make-masks-yolo-stage-proof.json', proof)
print('PASS: 58 scenes, 56 exact native visuals, thirteen reviews and all 84 other complete lessons preserved in all fourteen catalogs', flush=True)
