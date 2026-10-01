"""Web copies can live on the immutable media revision instead of the Pages tree.

Posters and catalogs stay on Pages. A lesson whose web copy is hosted says so in
``lesson_catalog.js`` (``web``); every other lesson keeps its local copy, so a
candidate can move lessons one at a time without breaking the rest.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_appended_candidate import (copy_preserved_web, ensure_web_root, hosted_web_path,
                                      mark_hosted_web, migrate_web_copy)
from publish_release_candidate import drop_superseded_web_copies
from validate_candidate import require_consistent_web_hosting

PLAYER = Path(__file__).resolve().parents[1] / 'authoring/web/app_v2.js'
ROOT = 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + 'a' * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_player_video_source(dataset, lesson, quality='1440p'):
    """Evaluate the player's own videoSource() and root constants in Node."""
    if not shutil.which('node'):
        pytest.skip('Node is required to run the player source')
    text = PLAYER.read_text()
    start = text.index('function videoSource(')
    body = text[start:text.index('\n}\n', start) + 3]
    roots = [line for line in text.splitlines()
             if line.startswith(('const PRODUCTION_ROOT', 'const AUDIO_ROOT', 'const VIDEO_4K_ROOT',
                                 'const WEB_VIDEO_ROOT'))]
    script = ('const document = {documentElement: {dataset: %s}};\n' % json.dumps(dataset)
              + '\n'.join(roots) + '\n'
              + 'const elements = {quality: {value: %s}};\n' % json.dumps(quality)
              + 'let activeLesson = null;\nfunction isPlayable(l) { return Boolean(l && l.silent); }\n'
              + 'function fourKAvailable() { return Boolean(VIDEO_4K_ROOT); }\n'
              + body + 'console.log(videoSource(%s));\n' % json.dumps(lesson))
    return subprocess.run(['node', '-e', script], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def test_player_plays_a_hosted_web_copy_and_keeps_local_ones_for_other_lessons():
    dataset = {'productionRoot': 'production', 'audioRoot': ROOT, 'video4kRoot': ROOT, 'webRoot': ROOT}
    hosted = {'id': '79_x', 'silent': '79_x/video/79_x_silent.mp4', 'web': '79_x/web/79_x_silent.mp4'}
    local = {'id': '07_y', 'silent': '07_y/video/07_y_silent.mp4'}
    assert run_player_video_source(dataset, hosted) == ROOT + '/79_x/web/79_x_silent.mp4'
    assert run_player_video_source(dataset, local) == 'production/07_y/video/07_y_silent.mp4'
    # The 4K choice still plays the master, hosted web copy or not.
    assert run_player_video_source(dataset, hosted, '4k') == ROOT + '/79_x/video/79_x_silent.mp4'
    # A page without a web root (an older index) keeps the local path.
    del dataset['webRoot']
    assert run_player_video_source(dataset, local) == 'production/07_y/video/07_y_silent.mp4'


def test_hosting_moves_the_verified_web_copy_to_the_media_host_and_keeps_the_poster(tmp_path):
    published, baseline, candidate = [tmp_path / name for name in ('pages', 'baseline', 'candidate')]
    video = Path('production/79_x/video/79_x_silent.mp4')
    poster = Path('production/79_x/poster.jpg')
    for base in (published, baseline / 'web'):
        for relative, content in ((video, b'verified web copy'), (poster, b'poster')):
            (base / relative).parent.mkdir(parents=True, exist_ok=True)
            (base / relative).write_bytes(content)
    manifest = {'files': [{'path': 'web/' + p.as_posix(), 'bytes': len(c), 'sha256': sha(c)}
                          for p, c in ((video, b'verified web copy'), (poster, b'poster'))]}
    records = copy_preserved_web(published, baseline, candidate, manifest, hosted=['79_x'])
    assert not (candidate / 'web' / video).exists()
    assert (candidate / 'web' / poster).read_bytes() == b'poster'
    migrate_web_copy(baseline, candidate, manifest, '79_x', records)
    target = candidate / 'media_host' / hosted_web_path('79_x')
    assert target.read_bytes() == b'verified web copy'
    assert {'path': 'media_host/' + hosted_web_path('79_x'), 'sha256': sha(b'verified web copy'),
            'bytes': 17} in records
    # A changed baseline copy is refused rather than hosted.
    (baseline / 'web' / video).write_bytes(b'changed after verification')
    with pytest.raises(ValueError):
        migrate_web_copy(baseline, tmp_path / 'other', manifest, '79_x', [])


def test_javascript_catalog_marks_only_selected_lessons_and_refuses_unknown_ones():
    catalog = {'lessons': [{'id': '07_y', 'silent': '07_y/video/07_y_silent.mp4'},
                           {'id': '79_x', 'silent': '79_x/video/79_x_silent.mp4'}]}
    result = mark_hosted_web(catalog, ['79_x'])
    assert result['lessons'][1]['web'] == '79_x/web/79_x_silent.mp4'
    assert 'web' not in result['lessons'][0] and 'web' not in catalog['lessons'][1]
    with pytest.raises(ValueError):
        mark_hosted_web(catalog, ['99_missing'])


def records_for(*paths):
    return [{'path': p, 'sha256': sha(p.encode()), 'bytes': 1} for p in paths]


@pytest.mark.parametrize('case', ['both copies', 'no hosted file', 'hosted file without catalog flag',
                                  'wrong catalog path'])
def test_validation_requires_exactly_one_web_copy_per_hosted_lesson(case):
    good = records_for('web/production/79_x/poster.jpg', 'media_host/79_x/video/79_x_silent.mp4',
                       'media_host/79_x/web/79_x_silent.mp4', 'web/production/07_y/video/07_y_silent.mp4')
    lessons = [{'id': '79_x', 'web': '79_x/web/79_x_silent.mp4'}, {'id': '07_y'}]
    assert require_consistent_web_hosting(good, lessons) == ['79_x']
    records, js = list(good), [dict(lesson) for lesson in lessons]
    if case == 'both copies':
        records += records_for('web/production/79_x/video/79_x_silent.mp4')
    elif case == 'no hosted file':
        records = [r for r in records if r['path'] != 'media_host/79_x/web/79_x_silent.mp4']
    elif case == 'hosted file without catalog flag':
        js[0].pop('web')
    else:
        js[0]['web'] = '79_x/video/79_x_silent.mp4'
    with pytest.raises(ValueError):
        require_consistent_web_hosting(records, js)


def test_candidate_index_gains_exactly_one_local_web_root():
    index = '<html data-production-root="production"\n      data-audio-root="X"\n      data-video4k-root="X">'
    result = ensure_web_root(index)
    assert result.count('data-web-root="X"') == 1
    assert ensure_web_root(result) == result


def test_pages_removes_only_the_superseded_local_copy_of_a_hosted_lesson(tmp_path):
    pages = tmp_path / 'pages'
    old = pages / 'production/79_x/video/79_x_silent.mp4'
    kept = pages / 'production/07_y/video/07_y_silent.mp4'
    poster = pages / 'production/79_x/poster.jpg'
    for path in (old, kept, poster):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x')
    manifest = {'files': records_for('media_host/79_x/web/79_x_silent.mp4',
                                     'media_host/07_y/video/07_y_silent.mp4')}
    assert drop_superseded_web_copies(pages, manifest) == ['production/79_x/video/79_x_silent.mp4']
    assert not old.exists() and kept.exists() and poster.exists()


def published_index(tmp_path, web_root):
    pages = tmp_path / 'pages'
    pages.mkdir()
    (pages / 'index.html').write_text(
        f'<html data-production-root="production" data-audio-root="{ROOT}" '
        f'data-video4k-root="{ROOT}" data-web-root="{web_root}">')
    root = tmp_path / 'candidate'
    root.mkdir()
    (root / 'release-manifest.json').write_text(json.dumps({'files': []}))
    return root, pages


def test_published_tree_requires_the_web_root_on_the_same_pinned_revision(tmp_path):
    pytest.importorskip('playwright')  # the browser verifier imports it at module level
    from verify_release_candidate import published_tree
    root, pages = published_index(tmp_path, ROOT)
    assert published_tree(root, pages) == ROOT
    other = tmp_path / 'other'
    other.mkdir()
    root, pages = published_index(other, ROOT[:-1] + 'b')
    with pytest.raises(ValueError):
        published_tree(root, pages)
