from pathlib import Path
import sys

sys.path.insert(0, 'tools/tutorials')
sys.path.insert(0, 'tools/tutorials/authoring/tools')
from build_appended_candidate import verify_tracks
from stage_lesson import read, write
from render_all_voices import LANGUAGES

stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r1')
catalogs = {p.name: read(p) for p in (stage / 'catalog').glob('lessons_*.json')}
results = {}
for identity in ['01_pypi_github', '03_pip_install', '04_platform_installers']:
    lesson = read(stage / 'production' / identity / 'lesson.en.json')
    declared, tracks = verify_tracks(stage, lesson, catalogs)
    expected = {language: sorted(spec[1]) for language, spec in LANGUAGES.items()}
    assert {language: sorted(value) for language, value in declared.items()} == expected
    assert len(tracks) == 50
    results[identity] = {'voices': declared, 'tracks': tracks, 'scenes': len(lesson['scenes'])}
    print('PASS:', identity, 'all 50 source/runtime-bound tracks, full decode, timing, activity and dead-air gates', flush=True)
write(stage / 'current-complete-audio-acceptance.json', {'passed': True, 'all_150_tracks_verified': True,
      'lessons': results, 'normal_full_decode_source_runtime_timing_activity_and_dead_air_verification': True,
      'published': False})
