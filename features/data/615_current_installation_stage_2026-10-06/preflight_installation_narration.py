from pathlib import Path
import hashlib
import json
import os
import sys

sys.path.insert(0, 'tools/tutorials')
sys.path.insert(0, 'tools/tutorials/authoring/tools')
from stage_lesson import read, write
from render_all_voices import LANGUAGES, prepare_scene_plans
from narration_audio import narration_dialect, resolve_voice_speed

stage = Path(os.environ['SPACR_TUTORIAL_WORKSPACE'])
report = []
for language, (code, voices) in LANGUAGES.items():
    catalog = read(stage / 'catalog' / f'lessons_{language}.json')
    for identity in ['01_pypi_github', '03_pip_install', '04_platform_installers']:
        lesson = next(row for row in catalog['lessons'] if row['id'] == identity)
        for voice in voices:
            speed = resolve_voice_speed(voice)
            voice_code = 'b' if language == 'en' and voice.startswith('b') else code
            plans = prepare_scene_plans(lesson, language, narration_dialect(language, voice_code, voice), speed, voice=voice)
            assert len(plans) == len(lesson['scenes'])
            report.append({'language': language, 'lesson': identity, 'voice': voice,
                           'sentences': sum(len(row['sentences']) for row in plans),
                           'plans_sha256': hashlib.sha256(json.dumps(plans, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
assert len(report) == 150
write(stage / 'narration-preflight.json', {'passed': True, 'all_150_registry_track_sentence_plans': report})
print('PASS: all 150 current narration sentence plans and pronunciation guards', flush=True)
