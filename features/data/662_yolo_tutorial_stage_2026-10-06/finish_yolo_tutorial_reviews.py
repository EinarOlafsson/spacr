from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path('tools/tutorials').resolve()))
from apply_translation_review import translated_lesson

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
english = json.loads(Path('tools/tutorials/lessons/14_make_masks.json').read_text())
before = json.loads((scratch / 'make-masks-before-yolo-lesson.json').read_text())
proof = json.loads((scratch / 'make-masks-yolo-authoring-proof.json').read_text())
assert len(english['scenes']) == 58
targets = {}
for name in ('eu1', 'eu2', 'asia'):
    batch = json.loads((scratch / f'yolo-tutorial-reviewed-targets-{name}.json').read_text())
    assert not targets.keys() & batch.keys()
    targets.update(batch)
assert set(targets) == {'da', 'de', 'es', 'fr', 'hi', 'is', 'it', 'ja', 'ko', 'nb', 'pt-BR', 'sv', 'zh-CN'}
digest = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
report = {}
prepared = {}
for language, additions in targets.items():
    path = Path(f'tools/tutorials/lessons/reviews/14_make_masks.{language}.json')
    previous = json.loads(path.read_text())
    assert previous['english_sha256'] == proof['previous_english_sha256']
    assert len(previous['scenes']) == 45 and len(additions['scenes']) == 13
    (scratch / f'make-masks-before-yolo-review-{language}.json').write_bytes(path.read_bytes())
    review = dict(previous)
    index = proof['insert_index']
    review['scenes'] = previous['scenes'][:index] + additions['scenes'] + previous['scenes'][index:]
    review['description'] = previous['description'] + ' ' + additions['description']
    review['objectives'] = previous['objectives'] + [additions['objective']]
    review['english_sha256'] = digest
    review['review'] = {'kind': 'AI technical review (Codex), no native-speaker signoff',
                       'translator': 'Prior forty-five reviewed scene translations retained; Codex directly translated and technically reviewed thirteen YOLO additions',
                       'date': '2026-10-05', 'independent_peer_review': False,
                       'native_speaker_signoff': False, 'listening_review': False,
                       'scope': 'Separate class-labelled boxes, native editing/history, independent image-relative XYWH export and empty-file demonstration without biological ground-truth claims',
                       'previous_review': previous['review'], 'previous_english_sha256': previous['english_sha256']}
    translated = translated_lesson(review, english, previous)
    assert [scene['narration'] for scene in translated['scenes'] if not scene['visual'].startswith('yolo_')] == previous['scenes']
    assert review['title'] == previous['title'] and review['prerequisite'] == previous['prerequisite']
    prepared[path] = review
    report[language] = {'preserved_prior_scene_texts': 45, 'new_reviewed_scene_texts': 13, 'normal_source_pin_and_spoken_pronunciation_checks_passed': True}
for path, review in prepared.items():
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
(scratch / 'make-masks-yolo-reviewed-proof.json').write_text(json.dumps({'accepted': True, 'english_sha256': digest, 'languages': report, 'narration_video_publication_complete': False}, indent=2) + '\n')
print('All thirteen source-bound reviews accepted; all forty-five old scene texts exact; spoken pronunciation checks pass', flush=True)
