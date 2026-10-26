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
    text = PLAYER.read_text(encoding="utf-8")
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
        f'data-video4k-root="{ROOT}" data-web-root="{web_root}">', encoding="utf-8")
    root = tmp_path / 'candidate'
    root.mkdir()
    manifest = {'files': [dict(path='media_host/fixture.m4a', bytes=1, sha256=sha(b'x'))]}
    (root / 'release-manifest.json').write_text(json.dumps(manifest), encoding="utf-8")
    (root / 'publication-receipt.json').write_text(json.dumps(dict(commit='a' * 40,
        media_root=ROOT, tag='synthetic-fixture', media_files=1,
        manifest_sha256=sha((root / 'release-manifest.json').read_bytes()),
        readback=dict(commit='a' * 40, passed=True, files_expected=1, metadata_matched=1,
                      downloaded_sha256_matched=1, metadata_failures=[], download_failures=[]),
        pages=dict(index_sha256=sha((pages / 'index.html').read_bytes())))), encoding='utf-8')
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


@pytest.fixture
def archived_publication(tmp_path, monkeypatch):
    """Real84 catalog structures, synthetic media: no network or browser work."""
    import archive_lessons
    import publish_release_candidate as publication
    from stage_lesson import write
    source = Path(__file__).resolve().parents[1] / 'release_candidate/web'
    full = archive_lessons.load_tree(source)
    root, pages = published_index(tmp_path, ROOT)
    repo = tmp_path / 'repo'
    pages = repo / 'docs/source/_extra/tutorials'
    pages.mkdir(parents=True)
    # Only the normal publisher's index substitutions matter in this fixture.
    index = ('<html data-production-root="production" data-audio-root="../media_host" '
             'data-video4k-root="../media_host" data-web-root="../media_host">'
             '<b id="available-count">84</b><b id="total-count">84</b>0 of 84 complete Lesson 1 of 84'
             + ''.join(f'<script src="{name}"></script>' for name in publication.VERSIONED))
    files = {name: archive_lessons.render(name, data).encode() for name, data in full.items()}
    files.update({'index.html': index.encode(), 'app_v2.js': b'unchanged player',
                  'styles.css': b'unchanged styles', 'voice_catalog.js': b'unchanged voices',
                  'production/06_api/poster.jpg': b'visible poster',
                  'production/18_motility/poster.jpg': b'archived poster'})
    records = []
    for name, content in files.items():
        path = root / 'web' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append(dict(path='web/' + name, bytes=len(content), sha256=sha(content)))
    records.append(dict(path='media_host/fixture.m4a', bytes=1, sha256=sha(b'x')))
    identities = [row['id'] for row in full['catalog/lessons_en.json']['lessons']]
    assert len(identities) == 84
    records.extend(dict(path=f'media_host/{identity}/web/{identity}_silent.mp4', bytes=1, sha256=sha(b'x'))
                   for identity in identities)
    write(root / 'release-manifest.json', dict(files=records, routes=84, ready_lessons=84,
                                             hosted_web_lessons=identities))
    receipt = publication.read(root / publication.RECEIPT)
    receipt['manifest_sha256'] = publication.digest(root / 'release-manifest.json')
    receipt.update(repository='datasets/einarolafsson/spacr-tutorials', branch='synthetic-fixture')
    receipt['media_files'] = 85
    receipt['readback'].update(files_expected=85, metadata_matched=85, downloaded_sha256_matched=85)
    write(root / publication.RECEIPT, receipt)
    (pages / 'index.html').write_text(index)
    # The production count/index mutation is real; unrelated index exporting is isolated.
    (repo / 'spacr/resources').mkdir(parents=True)
    monkeypatch.setattr(publication, 'REPO', repo)
    monkeypatch.setattr(publication, 'PAGES', pages)
    monkeypatch.setattr(publication, 'validate', lambda *args, **kwargs: None)
    monkeypatch.setattr(publication.build_tutorial_index, 'build', lambda path: {'fixture': True})
    publication.pages(root, 'archive-fixture')
    receipt = publication.read(root / publication.RECEIPT)
    shown = publication.publication_view(root)['catalog/lessons_en.json']
    visible = [row['id'] for row in shown['lessons']]
    assert len(visible) == 59
    checks = dict(passed=True, published=False, placeholder_cases=[],
                  manifest_sha256=receipt['manifest_sha256'], media_root=ROOT,
                  index_sha256=publication.digest(pages / 'index.html'),
                  ready_playback_cases=[dict(lesson=identity, passed=True) for identity in visible])
    write(root / 'checks/published-media-browser-checks.json', checks)
    write(repo / 'tools/tutorials/release_candidate/checkpoint.json',
          dict(manifest_sha256=receipt['manifest_sha256'], release_hold=True))
    return root, pages, publication, full


def test_archived_publication_keeps84_candidate_and_records_only_exact59(archived_publication):
    from verify_release_candidate import published_tree
    root, pages, publication, full = archived_publication
    assert published_tree(root, pages) == ROOT
    receipt = publication.read(root / publication.RECEIPT)
    assert {key: receipt['pages'][key] for key in ('ready', 'routes', 'hosted_web_copies')} == {
        'ready': 59, 'routes': 59, 'hosted_web_copies': 59}
    assert not (pages / 'production/18_motility/poster.jpg').exists()
    assert (pages / 'production/06_api/poster.jpg').read_bytes() == b'visible poster'
    publication.record(root)
    assert publication.read(root / 'web/catalog/lessons_en.json') == full['catalog/lessons_en.json']
    assert len(publication.read(root / 'checks/published-media-browser-checks.json')['ready_playback_cases']) == 59


@pytest.mark.parametrize('change', ['caption', 'runtime', 'navigation', 'player',
                                   'receipt_root', 'dropped_visible', 'candidate_drift'])
def test_archived_publication_rejects_projection_asset_or_revision_drift(archived_publication, change):
    from verify_release_candidate import published_tree
    import archive_lessons
    from stage_lesson import write
    root, pages, publication, _ = archived_publication
    if change == 'receipt_root':
        receipt = publication.read(root / publication.RECEIPT)
        receipt['media_root'] = ROOT[:-1] + 'b'
        write(root / publication.RECEIPT, receipt)
    elif change == 'player':
        (pages / 'app_v2.js').write_bytes(b'foreign player')
    elif change in ('caption', 'dropped_visible', 'candidate_drift'):
        name = 'catalog/captions_da.json' if change == 'caption' else 'catalog/lessons_en.json'
        data = publication.read((root / 'web' if change == 'candidate_drift' else pages) / name)
        if change == 'dropped_visible':
            data['lessons'].pop(0)
        else:
            data['lessons'][0]['scenes'][0]['narration'] += ' Unreviewed change.'
        if change == 'candidate_drift':
            write(root / 'web' / name, data)
            data = publication.publication_view(root)[name]
        write(pages / name, data)
    else:
        name = 'lesson_catalog.js' if change == 'runtime' else 'module_navigation.js'
        data = archive_lessons.load_tree(pages)[name]
        if change == 'runtime':
            data['lessons'][0]['title'] = 'Unreviewed route'
        else:
            data['routes'].pop(next(iter(data['routes'])))
        (pages / name).write_text(archive_lessons.render(name, data), encoding='utf-8')
    with pytest.raises(ValueError):
        published_tree(root, pages)


@pytest.mark.parametrize('change', ['wrong_lesson_same_count', 'duplicate_pass', 'wrong_domain_counts',
                                   'wrong_report_counts'])
def test_archived_publication_record_rejects_arbitrary59_case_claim(archived_publication, change):
    from stage_lesson import write
    root, _, publication, _ = archived_publication
    checks = publication.read(root / 'checks/published-media-browser-checks.json')
    if change == 'wrong_domain_counts':
        receipt = publication.read(root / publication.RECEIPT)
        receipt['pages'].update(ready=84, routes=84, hosted_web_copies=84)
        write(root / publication.RECEIPT, receipt)
    elif change == 'wrong_report_counts':
        checks.update(routes=84, ready_lessons=84)
    elif change == 'wrong_lesson_same_count':
        checks['ready_playback_cases'][0]['lesson'] = '18_motility'
    else:
        checks['ready_playback_cases'][-1] = dict(checks['ready_playback_cases'][0])
    write(root / 'checks/published-media-browser-checks.json', checks)
    with pytest.raises(SystemExit, match='incomplete'):
        publication.record(root)
