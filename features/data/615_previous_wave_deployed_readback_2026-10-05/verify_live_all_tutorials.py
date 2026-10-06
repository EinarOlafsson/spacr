import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path[:0] = [str(Path.cwd() / 'tools'), str(Path.cwd() / 'tools/tutorials')]
from verify_tutorial_live import static_audit
from verify_release_candidate import verify_live_mobile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
url = 'https://einarolafsson.github.io/spacr/nightly/tutorials/'
local = Path('docs/source/_extra/tutorials')
report = static_audit(url, timeout=90)
paths = [*sorted((local / 'catalog').glob('*.json')), local / 'module_navigation.js',
         local / 'styles.css', local / 'translation-compatibility.json']
assert len(paths) == 17
catalogs = []
for path in paths:
    relative = path.relative_to(local).as_posix()
    payload = urllib.request.urlopen(url + relative, timeout=90).read()
    assert payload == path.read_bytes(), relative
    catalogs.append({'path': relative, 'sha256': hashlib.sha256(payload).hexdigest(), 'passed': True})
    print('Live tutorial asset hash PASS:', relative, flush=True)
english = json.loads((local / 'catalog/lessons_en.json').read_text())
lessons = tuple(lesson['id'] for lesson in english['lessons'] if lesson.get('status') != 'coming_soon')
assert len(lessons) == 85
mobile = verify_live_mobile(url, lessons=lessons, tap_timeout_ms=1000)
assert mobile['passed'] and len(mobile['cases']) == 85
receipt = {'schema': 1, 'passed': True, 'static': report,
           'deployed_catalog_and_navigation_readback': catalogs,
           'mobile_all_85_lessons': mobile}
(scratch / 'live-final-wave-all-tutorials.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('All 85 deployed routes and 14 catalog variants PASS.', flush=True)
