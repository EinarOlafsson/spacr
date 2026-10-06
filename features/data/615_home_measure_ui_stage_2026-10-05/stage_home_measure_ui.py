import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, 'tools/tutorials')
from stage_lesson import read, write, stage_lesson, union
from apply_translation_review import promote_many

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-home-measure-ui-current-r1'
home = scratch / 'tutorial-home-left-to-right-current-r2/captures/home_left_to_right_20261005_r2'
destination = stage / 'captures/home_left_to_right_20261005_r2'
shutil.copytree(home, destination)
identities = ('05_home', '08_measure')
captures = ('home_left_to_right_20261005_r2', 'measure_no_banner_complete_20261005_r1')
for identity, capture in zip(identities, captures, strict=True):
    folder = stage / 'captures' / capture
    frames = read(folder / 'frames.json')
    provenance = read(folder / 'provenance.json')
    assert provenance['completed_capture'] and not provenance['app_source_modified']
    lesson_path = Path('tools/tutorials/lessons') / (identity + '.json')
    lesson = read(lesson_path)
    focus = {}
    for scene in lesson['scenes']:
        name = scene['visual']
        frame = frames[name]
        source = (folder / frame['image']).resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == frame['sha256']
        if 'focus_modules' in scene:
            rectangles = []
            positions = []
            for key in scene['focus_modules']:
                matches = [b for b in frame['buttons'] if b.get('module_key') == key or b.get('nav_key') == key]
                chosen = max(matches, key=lambda b: b['rect'][2] * b['rect'][3])
                rectangles.append(chosen['rect'])
                positions.append(chosen['rect'][0])
            assert positions == sorted(positions), (name, positions)
            focus[name] = union(rectangles)
        elif name.startswith('13'):
            rectangles = [item['rect'] for item in frame.get('dialogs', []) if item.get('rect')]
            assert rectangles, name
            focus[name] = union(rectangles)
        elif name == 'preview_03_data_ready':
            focus[name] = [345, 45, 1180, 630]
        elif name.startswith('tpl_') or name in ('09a_qc_popup', 'batch_21_ram_guard'):
            rectangles = [item['rect'] for item in frame.get('dialogs', []) if item.get('rect')]
            focus[name] = union(rectangles) if rectangles else None
        else:
            focus[name] = None
    if identity == '08_measure':
        assert 'preview_01_module' not in {s['visual'] for s in lesson['scenes']}
        acceptance = read(folder / 'scientific_acceptance.json')
        assert acceptance['accepted'] and acceptance['complete_batch_output_preserved']
        assert acceptance['independent_output_checks']['run_status']['n_succeeded'] == 16
    focus_path = stage / (identity + '-current-focus.json')
    write(focus_path, {'english_sha256': hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      'scenes': focus, 'recapture': {'capture_module': capture,
                          'frames': {name: frame['sha256'] for name, frame in frames.items()}}})
    stage_lesson(lesson_path, capture, stage, check_only=True, focus_map=focus_path)
    stage_lesson(lesson_path, capture, stage, focus_map=focus_path)
for language in ('da', 'de', 'es', 'fr', 'hi', 'is', 'it', 'ja', 'ko', 'nb', 'pt-BR', 'sv', 'zh-CN'):
    reviews = [read(Path('tools/tutorials/lessons/reviews') / f'{identity}.{language}.json') for identity in identities]
    promote_many(reviews, stage)
baseline = Path('docs/source/_extra/tutorials/catalog')
for path in (stage / 'catalog').glob('*.json'):
    before = {item['id']: item for item in read(baseline / path.name)['lessons']}
    after = {item['id']: item for item in read(path)['lessons']}
    assert before.keys() == after.keys()
    assert all(before[key] == after[key] for key in before if key not in identities)
print('Home and Measure staged: 39 scenes, 26 source-pinned reviews, 83 unrelated complete lesson objects retained in every catalog.')
