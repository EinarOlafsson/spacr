import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('identity', choices=['01_pypi_github', '03_pip_install', '04_platform_installers'])
args = parser.parse_args()
stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r2')
accepted = json.loads((stage / 'current-complete-audio-acceptance.json').read_text())['lessons'][args.identity]
video = json.loads((stage / 'production' / args.identity / 'current-video-acceptance.json').read_text())
assert video['master_full_decode_passed'] and video['web_rendition']['accepted']
checks = []
for language, voices in accepted['voices'].items():
    selected_voice = 'af_heart' if language == 'en' else voices[0]
    command = [sys.executable, 'tools/tutorials/verify_staged_lesson.py', '--stage', str(stage),
               '--lesson', args.identity, '--web-rendition', '--language', language, '--voice', selected_voice]
    if language == 'en':
        command.append('--sentence-cues')
    subprocess.run(command, check=True)
    tag = language + '-' + selected_voice + ('-sentence-cues' if language == 'en' else '')
    path = stage / 'browser-web' / args.identity / tag / 'playback-checks.json'
    report = json.loads(path.read_text())
    assert report['passed'] and report['checked_web_rendition']['sha256'] == video['web_rendition']['rendition_sha256']
    checks.append({'case': tag, 'report': str(path.relative_to(stage))})
    print(args.identity, tag, 'desktop/mobile playback passed', flush=True)
for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
    subprocess.run([sys.executable, 'tools/tutorials/verify_staged_lesson.py', '--stage', str(stage),
                    '--lesson', args.identity, '--web-rendition', '--caption-language', language], check=True)
    tag = 'en-af_heart-captions-' + language
    path = stage / 'browser-web' / args.identity / tag / 'playback-checks.json'
    report = json.loads(path.read_text())
    assert report['passed'] and report['caption_scenes_match_staging']
    assert report['checked_web_rendition']['sha256'] == video['web_rendition']['rendition_sha256']
    checks.append({'case': tag, 'report': str(path.relative_to(stage))})
    print(args.identity, tag, 'caption/desktop/mobile checks passed', flush=True)
assert len(checks) == 14
receipt = {'schema': 1, 'lesson': args.identity, 'passed': True, 'checks': checks,
           'scope': 'Eight narration languages and six independent caption languages, current-byte 1440p playback, desktop/mobile layout, exact narration loading, chapter links and English sentence cues.',
           'independent_native_speaker_review': False, 'published': False}
(stage / 'production' / args.identity / 'current-browser-acceptance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
