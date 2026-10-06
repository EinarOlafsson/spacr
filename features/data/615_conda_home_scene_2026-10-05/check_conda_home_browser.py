"""Verify the changed shared Conda video with all retained language routes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005')
STAGE = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-conda-home-current-r2')
IDENTITY = '02_conda_install'
sys.path.insert(0, str(ROOT / 'tools/tutorials'))
from retain_narration import digest
from stage_lesson import read, write

receipt_path = STAGE / 'production' / IDENTITY / 'single-scene-refresh.json'
receipt = read(receipt_path)
if receipt.get('accepted') is not True:
    raise ValueError('The single-scene frame-preservation checks must pass first')
lesson = next(row for row in read(STAGE / 'catalog/lessons_en.json')['lessons'] if row['id'] == IDENTITY)
web = STAGE / 'web-renditions' / IDENTITY / 'video' / (IDENTITY + '_silent.mp4')
if digest(web) != receipt['video']['web']['replacement_sha256']:
    raise ValueError('The checked replacement video changed')
checks = []
for language, voices in lesson['narration_voices'].items():
    command = [sys.executable, str(ROOT / 'tools/tutorials/verify_staged_lesson.py'),
               '--stage', str(STAGE), '--lesson', IDENTITY, '--web-rendition',
               '--language', language, '--voice', voices[0]]
    if language == 'en':
        command.append('--sentence-cues')
    subprocess.run(command, cwd=ROOT, check=True)
    tag = language + '-' + voices[0] + ('-sentence-cues' if language == 'en' else '')
    report_path = STAGE / 'browser-web' / IDENTITY / tag / 'playback-checks.json'
    report = read(report_path)
    assert report['passed'] and report['checked_web_rendition']['sha256'] == digest(web)
    checks.append({'case': tag, 'sha256': digest(report_path), 'path': str(report_path.relative_to(STAGE))})
for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
    subprocess.run([sys.executable, str(ROOT / 'tools/tutorials/verify_staged_lesson.py'),
                    '--stage', str(STAGE), '--lesson', IDENTITY, '--web-rendition',
                    '--caption-language', language], cwd=ROOT, check=True)
    tag = 'en-af_heart-captions-' + language
    report_path = STAGE / 'browser-web' / IDENTITY / tag / 'playback-checks.json'
    report = read(report_path)
    assert report['passed'] and report['caption_scenes_match_staging']
    assert report['checked_web_rendition']['sha256'] == digest(web)
    checks.append({'case': tag, 'sha256': digest(report_path), 'path': str(report_path.relative_to(STAGE))})
assert len(checks) == 14
receipt.update(browser_verified=True, browser_checks=checks)
write(receipt_path, receipt)
print('Conda one-scene replacement: all fourteen language/caption routes pass desktop/mobile checks; publication remains.', flush=True)
