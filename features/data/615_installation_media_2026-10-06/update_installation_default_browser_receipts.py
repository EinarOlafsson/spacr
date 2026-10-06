from pathlib import Path
import json

stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r2')
for identity in ('01_pypi_github', '03_pip_install', '04_platform_installers'):
    path = stage / 'production' / identity / 'current-browser-acceptance.json'
    before = json.loads(path.read_text())
    assert before['passed'] and len(before['checks']) == 14
    report = stage / 'browser-web' / identity / 'en-af_heart-sentence-cues/playback-checks.json'
    checked = json.loads(report.read_text())
    video = json.loads((path.parent / 'current-video-acceptance.json').read_text())
    assert checked['passed'] and checked['checked_web_rendition']['sha256'] == video['web_rendition']['rendition_sha256']
    old = path.parent / 'historical-extra-English-browser-acceptance.json'
    if not old.exists():
        old.write_bytes(path.read_bytes())
    assert before['checks'][0]['case'].startswith('en-')
    before['checks'][0] = {'case': 'en-af_heart-sentence-cues', 'report': str(report.relative_to(stage))}
    for entry in before['checks']:
        actual = json.loads((stage / entry['report']).read_text())
        assert actual['passed'] and actual['checked_web_rendition']['sha256'] == video['web_rendition']['rendition_sha256']
    path.write_text(json.dumps(before, ensure_ascii=False, indent=2) + '\n')
    print(identity, 'all 14 current-byte cases passed, including the normal default English voice')
