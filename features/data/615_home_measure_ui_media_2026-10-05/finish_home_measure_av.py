import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-home-measure-ui-current-r1'
identities = ('05_home', '08_measure')
sys.path.insert(0, 'tools/tutorials')
sys.path.insert(0, 'tools/tutorials/authoring/tools')
from build_appended_candidate import verify_tracks
from render_all_voices import LANGUAGES
from stage_lesson import read, write

deadline = time.monotonic() + 5400
expected_tracks = sum(len(voices) for _, voices in LANGUAGES.values())
assert expected_tracks == 50
while any(len(list((stage / 'production' / identity / 'audio').glob('*/*.json'))) != expected_tracks
          or len(list((stage / 'production' / identity / 'audio').glob('*/*.m4a'))) != expected_tracks
          for identity in identities):
    if time.monotonic() >= deadline:
        raise TimeoutError('The complete narration matrix did not arrive; nothing published')
    time.sleep(15)
catalogs = {path.name: read(path) for path in (stage / 'catalog').glob('lessons_*.json')}
results = {}
for identity in identities:
    lesson = read(stage / 'production' / identity / 'lesson.en.json')
    declared, tracks = verify_tracks(stage, lesson, catalogs)
    assert len(tracks) == expected_tracks and len(declared) == 8
    results[identity] = {'voices': declared, 'tracks': tracks}
write(stage / 'current-audio-acceptance.json', {'accepted': True, 'lessons': results,
      'tracks_verified': 100, 'scope': 'Full decode, source/render identity, exact timings, scene activity and dead-air gates', 'published': False})
cpu_environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
for identity in identities:
    folder = stage / 'production' / identity
    subprocess.run([sys.executable, 'tools/tutorials/authoring/tools/render_visual_master.py',
                    str(folder / 'visual.json'), '--timings', str(folder / 'audio/en/af_heart.json'),
                    '--output', str(folder / 'video' / (identity + '_silent.mp4'))], check=True, env=cpu_environment)
    source = (scratch / 'finish_final_wave_video.py').read_text()
    source = source.replace("choices=['08_measure', '14_make_masks', '37_batch', '38_distributed_jobs']", "choices=['05_home', '08_measure']")
    source = source.replace("scratch/docs-completion-20261005/tutorial-final-wave", "scratch/docs-completion-20261005/tutorial-home-measure-ui-current-r1")
    helper = stage / (identity + '-finish-video.py')
    helper.write_text(source)
    subprocess.run([sys.executable, str(helper), identity], check=True, env=cpu_environment)
    for language, voices in results[identity]['voices'].items():
        command = [sys.executable, 'tools/tutorials/verify_staged_lesson.py', '--stage', str(stage),
                   '--lesson', identity, '--web-rendition', '--language', language, '--voice', voices[0]]
        if language == 'en':
            command.append('--sentence-cues')
        subprocess.run(command, check=True, env=cpu_environment)
    for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
        subprocess.run([sys.executable, 'tools/tutorials/verify_staged_lesson.py', '--stage', str(stage),
                        '--lesson', identity, '--web-rendition', '--caption-language', language],
                       check=True, env=cpu_environment)
    write(folder / 'current-browser-acceptance.json', {'passed': True, 'cases': 14,
          'scope': 'Eight narration and six caption languages; desktop/mobile playback and exact current media', 'published': False})
print('PASS: both current tutorials, 100 narration tracks, decoded 4K/1440p videos and 28 desktop/mobile language cases. Hosted publication remains separate.', flush=True)
