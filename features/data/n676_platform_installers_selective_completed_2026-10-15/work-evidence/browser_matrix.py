"""04 retained repair: normal72 cases plus dedicated preserve-interval boundary/EOF case.
Preparation only; root runs after actual accepted native capture and sealed render.
Dedicated explicit English captions yield a distinct report path without changing narration."""
import json, subprocess, sys, time
from pathlib import Path
root = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0] = [str(root), str(root/'tools'), str(root/'tools/tutorials'), str(root/'tools/tutorials/authoring/tools')]
from render_all_voices import LANGUAGES
N = Path('/media/carruthers/mnt3/claude/n676_stages'); tool = str(N / 'work/lesson04-selective-20261015/tool.sh')
import os; stage = os.environ['MATRIX_STAGE']; captions = ['da', 'de', 'is', 'ko', 'nb', 'sv']
records = []
def run(name, *args):
    name = f'{name}-{sys.argv[1]}'
    began = time.monotonic()
    code = subprocess.run([tool, name, *args], stdout=subprocess.DEVNULL).returncode
    records.append({'run': name, 'returncode': code, 'seconds': round(time.monotonic() - began, 1),
                    'log': str(N / 'logs' / f'{os.environ["TAG"]}-{name}.log')})
    (N / 'logs' / f'browser-matrix-{sys.argv[1]}.json').write_text(json.dumps(records, indent=2) + '\n')
    print(name, code, flush=True)
    if code:
        raise SystemExit(code)
assert not (N/'logs'/f'browser-matrix-{sys.argv[1]}.json').exists(), 'Preserve existing matrix receipt'
assert sys.argv[2:] == ['04_platform_installers'], 'Lesson04 only'
for lesson in sys.argv[2:]:
    base = ['tools/tutorials/verify_staged_lesson.py', '--stage', stage, '--lesson', lesson, '--candidate-player','--retain-visual-from',os.environ['BASELINE_CANDIDATE']]
    run(f'{lesson}-cues', *base, '--language', 'en', '--voice', 'af_heart', '--sentence-cues')
    for language, (_, voices) in LANGUAGES.items():
        run(f'{lesson}-master-{language}-{voices[0]}', *base, '--language', language, '--voice', voices[0])
    for caption in captions:
        run(f'{lesson}-master-en-af_heart-captions-{caption}', *base, '--language', 'en', '--voice', 'af_heart', '--caption-language', caption)
    for language, (_, voices) in LANGUAGES.items():
        for voice in voices:
            run(f'{lesson}-web-{language}-{voice}', *base, '--web-rendition', '--language', language, '--voice', voice)
    for caption in captions:
        run(f'{lesson}-web-en-af_heart-captions-{caption}', *base, '--web-rendition', '--language', 'en', '--voice', 'af_heart', '--caption-language', caption)
    run(f'{lesson}-preserve-interval-boundary-eof', *base, '--language', 'en', '--voice', 'af_heart', '--caption-language', 'en', '--native-boundary-clocks')
    run(f'{lesson}-completed-matrix', 'tools/tutorials/check_completed_matrix.py', '--stage', stage, '--lesson', lesson, '--retained-narration', '--retained-from', os.environ['ORIGINAL'])
    assert len(records) == 73 and all(row['returncode'] == 0 for row in records)
    print(lesson, 'PASS 72 normal cases + 1 dedicated boundary/static-final EOF', flush=True)
