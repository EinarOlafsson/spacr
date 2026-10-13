import os
from pathlib import Path
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
stage = scratch / 'tutorial-make-masks-yolo-current-r1'
identity = '14_make_masks'
sys.path.insert(0, 'tools/tutorials')
sys.path.insert(0, 'tools/tutorials/authoring/tools')
from build_appended_candidate import verify_tracks
from stage_lesson import read, write

catalogs = {path.name: read(path) for path in (stage / 'catalog').glob('lessons_*.json')}
lesson = read(stage / 'production' / identity / 'lesson.en.json')
assert len(lesson['scenes']) == 58
declared, tracks = verify_tracks(stage, lesson, catalogs)
expected = {'en': 'af_heart', 'es': 'ef_dora', 'fr': 'ff_siwis', 'hi': 'hf_alpha',
            'it': 'if_sara', 'pt-BR': 'pf_dora', 'ja': 'jf_alpha', 'zh-CN': 'zf_xiaobei'}
assert declared == {language: [voice] for language, voice in expected.items()}
assert len(tracks) == 8
write(stage / 'current-reference-audio-acceptance.json',
      {'accepted_reference_tracks': True, 'lesson': identity, 'tracks': tracks,
       'voices': declared, 'tracks_verified': 8, 'scenes': 58,
       'scope': 'Full decode, source/render identity, exact timings, scene activity and dead-air gates for eight reference voices only.',
       'complete_50_voice_matrix': False, 'published': False})
print('PASS: all eight complete reference tracks pass the normal source/runtime/decode/activity verifier; forty-two other voices remain.', flush=True)
cpu_environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
folder = stage / 'production' / identity
subprocess.run([sys.executable, 'tools/tutorials/authoring/tools/render_visual_master.py',
                str(folder / 'visual.json'), '--timings', str(folder / 'audio/en/af_heart.json'),
                '--output', str(folder / 'video' / (identity + '_silent.mp4'))],
               check=True, env=cpu_environment)
source = (scratch / 'finish_final_wave_video.py').read_text()
source = source.replace('scratch/docs-completion-20261005/tutorial-final-wave',
                        'scratch/docs-completion-20261005/tutorial-make-masks-yolo-current-r1')
helper = stage / 'finish-current-yolo-video.py'
helper.write_text(source)
subprocess.run([sys.executable, str(helper), identity], check=True, env=cpu_environment)
for language, voice in expected.items():
    command = [sys.executable, 'tools/tutorials/verify_staged_lesson.py', '--stage', str(stage),
               '--lesson', identity, '--web-rendition', '--language', language, '--voice', voice]
    if language == 'en':
        command.append('--sentence-cues')
    subprocess.run(command, check=True, env=cpu_environment)
for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
    subprocess.run([sys.executable, 'tools/tutorials/verify_staged_lesson.py', '--stage', str(stage),
                    '--lesson', identity, '--web-rendition', '--caption-language', language],
                   check=True, env=cpu_environment)
print('PASS: current 4K/web videos and all fourteen reference narration/caption browser cases; complete audio matrix, visual/editorial/deck and publication acceptance remain separate.', flush=True)
