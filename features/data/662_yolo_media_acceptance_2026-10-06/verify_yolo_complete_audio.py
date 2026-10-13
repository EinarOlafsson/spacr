from pathlib import Path
import sys

sys.path.insert(0, 'tools/tutorials')
sys.path.insert(0, 'tools/tutorials/authoring/tools')
from build_appended_candidate import verify_tracks
from stage_lesson import read, write
from render_all_voices import LANGUAGES

stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-make-masks-yolo-current-r1')
lesson = read(stage / 'production/14_make_masks/lesson.en.json')
catalogs = {p.name: read(p) for p in (stage / 'catalog').glob('lessons_*.json')}
voices, tracks = verify_tracks(stage, lesson, catalogs)
expected = {language: sorted(spec[1]) for language, spec in LANGUAGES.items()}
assert {language: sorted(value) for language, value in voices.items()} == expected
assert len(tracks) == sum(map(len, expected.values())) == 50
assert len(lesson['scenes']) == 58
write(stage / 'current-complete-audio-acceptance.json', {
    'lesson': lesson['id'], 'accepted': True, 'complete_50_voice_matrix': True,
    'scenes': 58, 'voices': voices, 'tracks_verified': len(tracks), 'tracks': tracks,
    'scope': 'Exact normal registry matrix; normal full decode, source/runtime/render fingerprint, timing, activity and dead-air gates.',
    'published': False,
})
print('PASS: all 50 current source-bound tracks across all eight spoken languages; every normal audio verifier gate passes.', flush=True)
