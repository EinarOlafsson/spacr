"""Publication preserves branch boundaries and records translation debt."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from publish_docs_channels import assemble, prepare
from report_translation_compatibility import api_issues, audit_record, runtime_issues


def hosted_checkpoint(tmp_path, content, commit='a' * 40, hosted_content=None):
    import hashlib
    from publish_docs_channels import MEDIA_ROOT
    checkpoint = tmp_path / ('checkpoint-' + commit)
    checkpoint.mkdir()
    hosted_content = content if hosted_content is None else hosted_content
    manifest = checkpoint / 'release-manifest.json'
    manifest.write_text(json.dumps({'files': [
        {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
         'path': prefix + 'lesson/video.mp4'}
        for prefix, data in (('web/production/', content), ('media_host/', hosted_content))]}))
    (checkpoint / 'publication-receipt.json').write_text(json.dumps({
        'commit': commit, 'media_root': MEDIA_ROOT + commit,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'readback': {'passed': True, 'commit': commit, 'files_expected': 1,
                     'downloaded_sha256_matched': 1, 'metadata_matched': 1}}))
    return checkpoint


@pytest.mark.parametrize('hosted_content', [None, b'full-resolution recording'])
def test_verified_hosted_videos_reduce_size_and_preserve_each_channel_revision(tmp_path, hosted_content):
    from publish_docs_channels import MEDIA_ROOT
    main = site(tmp_path, 'main', b'old video')
    nightly = site(tmp_path, 'nightly', b'new video')
    report = tmp_path / 'report.json'
    report.write_text('{"api": {}}')
    for branch, root, content, revision in (
            ('main', main, b'old video', 'a' * 40),
            ('nightly', nightly, b'new video', 'b' * 40)):
        prepare(root, report, branch, branch + '-sha',
                hosted_checkpoint(tmp_path, content, revision, hosted_content))
    output = tmp_path / 'pages'
    receipt = assemble(main, nightly, output)
    assert receipt['media_files'] == 1
    for root, revision in ((output, 'a' * 40), (output / 'nightly', 'b' * 40)):
        manifest = json.loads((root / 'tutorials/published-media.json').read_text())
        assert manifest['lesson/video.mp4'] == MEDIA_ROOT + revision + '/lesson/video.mp4'
        assert not (root / 'tutorials/production/lesson/video.mp4').exists()
        assert (root / 'tutorials' / manifest['lesson/poster.jpg']).read_bytes() == b'poster'


def test_unverified_or_changed_video_stays_local(tmp_path):
    from publish_docs_channels import verified_video_hosts
    root = site(tmp_path, 'main', b'changed video')
    checkpoint = hosted_checkpoint(tmp_path, b'original video')
    assert verified_video_hosts(root, checkpoint) == {}
    video = root / 'tutorials/production/lesson/video.mp4'
    video.write_bytes(b'original video')
    assert verified_video_hosts(root, checkpoint)
    receipt_path = checkpoint / 'publication-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['readback']['downloaded_sha256_matched'] = 0
    receipt_path.write_text(json.dumps(receipt))
    assert verified_video_hosts(root, checkpoint) == {}
    assert video.exists()


def test_tampered_host_proof_cannot_redirect_a_video(tmp_path):
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly')
    proof = {'lesson/video.mp4': {'sha256': 'wrong', 'url': 'https://wrong.example/video'}}
    (nightly / 'tutorials/verified-video-hosts.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='hosted video proof differs'):
        assemble(main, nightly, tmp_path / 'pages')


def site(tmp_path, branch, video=b'same recording'):
    root = tmp_path / branch
    root.mkdir()
    (root / 'index.html').write_text(f'<body>{branch}</body>')
    (root / 'publication.json').write_text(json.dumps({'branch': branch, 'commit': branch + '-sha'}))
    tutorial = root / 'tutorials'
    media = tutorial / 'production/lesson'
    media.mkdir(parents=True)
    (media / 'video.mp4').write_bytes(video)
    (media / 'poster.jpg').write_bytes(b'poster')
    (tutorial / 'index.html').write_text('<body><header>Fixed toolbar</header><main id="lesson-content">Player</main>'
                                       '<script src="app_v2.js?v=unchanged-source"></script></body>')
    (tutorial / 'app_v2.js').write_text('"use strict";\n'
        'const video = `${PRODUCTION_ROOT}/${lesson.silent}`;\n'
        'const poster = `${PRODUCTION_ROOT}/${activeLesson.poster}`;\n'
        f'const AUDIO_ROOT = "https://media.example/{branch}-revision";')
    (tutorial / 'lesson_catalog.js').write_text(branch + ' lessons')
    return root


def test_both_channels_keep_own_content_and_share_only_identical_media(tmp_path):
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly')
    output = tmp_path / 'pages'
    receipt = assemble(main, nightly, output)
    assert receipt['channels']['main']['commit'] == 'main-sha'
    assert receipt['channels']['nightly']['commit'] == 'nightly-sha'
    assert receipt['media_files'] == 2
    for branch, path in [('main', output), ('nightly', output / 'nightly')]:
        assert f'>{branch}</body>' in (path / 'index.html').read_text()
        assert (path / 'tutorials/lesson_catalog.js').read_text() == branch + ' lessons'
        player = (path / 'tutorials/app_v2.js').read_text()
        assert f'https://media.example/{branch}-revision' in player
        assert 'publishedMedia(lesson.silent)' in player
        manifest = json.loads((path / 'tutorials/published-media.json').read_text())
        assert (path / 'tutorials' / manifest['lesson/video.mp4']).read_bytes() == b'same recording'
        assert '/spacr/nightly/' in (path / 'index.html').read_text()
        assert '<main id="lesson-content"><div class="spacr-publication-channel"' in (path / 'tutorials/index.html').read_text()
    assert (main / 'tutorials/production/lesson/video.mp4').exists()


def test_changed_nightly_video_cannot_change_main_video(tmp_path):
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly', b'new recording')
    output = tmp_path / 'pages'
    receipt = assemble(main, nightly, output)
    assert receipt['media_files'] == 3
    for path, expected in [(output, b'same recording'), (output / 'nightly', b'new recording')]:
        manifest = json.loads((path / 'tutorials/published-media.json').read_text())
        assert (path / 'tutorials' / manifest['lesson/video.mp4']).read_bytes() == expected


def test_media_only_update_changes_the_player_cache_key(tmp_path):
    import hashlib
    import re
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly', b'old recording')
    original_player = (nightly / 'tutorials/app_v2.js').read_bytes()
    before, after = tmp_path / 'before', tmp_path / 'after'
    assemble(main, nightly, before)
    (nightly / 'tutorials/production/lesson/video.mp4').write_bytes(b'new recording')
    assemble(main, nightly, after)
    versions = []
    for output in (before, after):
        tutorial = output / 'nightly/tutorials'
        version = re.search(r'app_v2\.js\?v=([0-9a-f]{64})', (tutorial / 'index.html').read_text())[1]
        assert version == hashlib.sha256((tutorial / 'app_v2.js').read_bytes()).hexdigest()
        versions.append(version)
    assert versions[0] != versions[1]
    assert (nightly / 'tutorials/app_v2.js').read_bytes() == original_player
    assert (before / 'tutorials/app_v2.js').read_bytes() == (after / 'tutorials/app_v2.js').read_bytes()


def test_live_audit_recognizes_only_the_publishers_asset_changes(tmp_path, monkeypatch):
    import verify_tutorial_live as live
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly')
    output = tmp_path / 'published'
    assemble(main, nightly, output)
    receipt = tmp_path / 'tools/tutorials/release_candidate/publication-receipt.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'media_root': 'https://example.invalid/immutable'}))
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    monkeypatch.setattr(live, 'LOCAL', nightly / 'tutorials')
    monkeypatch.setattr(live, 'EXPECTED_APP_KEY', 'unchanged-source')
    names = ('index.html', 'app_v2.js')
    remote = {name: (output / 'nightly/tutorials' / name).read_bytes() for name in names}
    source = {name: (nightly / 'tutorials' / name).read_bytes() for name in names}
    assert live.source_equivalent_assets(remote) == source
    changed = dict(remote, **{'app_v2.js': remote['app_v2.js'] + b'\nalert("changed")'})
    assert live.source_equivalent_assets(changed) != source
    wrong_media = dict(remote, **{'app_v2.js': remote['app_v2.js'].replace(b'../../_media/', b'../wrong-media/')})
    with pytest.raises(AssertionError):
        live.source_equivalent_assets(wrong_media)


def test_missing_main_or_oversized_site_cannot_replace_public_site(tmp_path):
    main, nightly = site(tmp_path, 'main'), site(tmp_path, 'nightly')
    with pytest.raises(ValueError, match='budget'):
        assemble(main, nightly, tmp_path / 'oversized', limit=1)
    (main / 'index.html').unlink()
    with pytest.raises(ValueError, match='main'):
        assemble(main, nightly, tmp_path / 'missing')
    assert not (tmp_path / 'missing').exists()


def test_incompatible_catalog_is_registered_and_english_is_publishable(tmp_path):
    root = site(tmp_path, 'main')
    catalogs = root / '_static/i18n/api'
    catalogs.mkdir(parents=True)
    for language in ('en', 'es', 'de'):
        (catalogs / f'{language}.json').write_text('{}')
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'api': {'es': {'source_compatible': False, 'issues': ['stale']},
                                           'de': {'source_compatible': True, 'issues': []}}}))
    prepare(root, report, 'main', 'current-sha')
    assert json.loads((catalogs / 'es.json').read_text())['unavailable'] is True
    assert (catalogs / 'en.json').exists()
    assert (catalogs / 'de.json').exists()
    assert json.loads((root / 'translation-compatibility.json').read_text()) == {
        **json.loads(report.read_text()), 'source_commit': 'current-sha'}


def test_report_names_every_missing_stale_and_obsolete_symbol():
    english = {'symbols': {'a': {'source_sha256': 'new', 'source_blocks_sha256': ['one']}, 'b': {}}}
    translated = {'schema': 2, 'symbols': {'a': {'text': 'Translation', 'source_sha256': 'old',
        'source_blocks_sha256': ['one'], 'translation_source_blocks_sha256': ['one']}, 'c': {}}}
    assert api_issues(english, translated) == [
        {'kind': 'stale', 'symbol': 'a', 'field': 'source_sha256'},
        {'kind': 'missing', 'symbol': 'b'}, {'kind': 'obsolete', 'symbol': 'c'}]


def test_runtime_report_registers_missing_rows_and_changed_placeholders():
    issues = runtime_issues({'UI': {'Hello {name}': 'Hello {name}', 'Next': 'Next'}},
        {'UI': {'Hello {name}': 'Hallo {wrong}'}, 'SOURCE_HASHES': {}},
        {('UI', 'Hello {name}'): 'new'})
    assert {row['kind'] for row in issues} == {'stale', 'placeholders', 'missing'}


def test_failed_or_crashed_locale_audit_is_never_called_compatible():
    class Audit:
        def audit(self, sources, languages):
            print('es/missing-symbol')
            return 1
    assert audit_record(Audit(), {}, ('es',)) == {
        'status': 'incompatible', 'exit_code': 1, 'diagnostics': 'es/missing-symbol\n'}
    class Broken:
        def audit(self, sources, languages):
            raise ValueError('broken catalog')
    assert audit_record(Broken(), {}, ('es',))['status'] == 'audit_error'


def test_report_only_pytest_policy_preserves_english_and_code_failures(tmp_path):
    import os
    import subprocess
    from tools.pytest_translation_compatibility import advisory

    assert advisory('tests/test_documentation_i18n.py',
                    'test_documentation_api_catalog_inventory_and_hashes_are_current', {'language': 'es'})
    assert not advisory('tests/test_documentation_i18n.py',
                        'test_documentation_api_catalog_inventory_and_hashes_are_current', {'language': 'en'})
    assert not advisory('tests/test_api_i18n_frontend.py', 'test_stale_translation_falls_back')
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'test_external_i18n_catalogs.py').write_text(
        'def test_external_runtime_catalogs_have_exact_current_source_keys():\n'
        '    raise KeyError("missing translated Next caption")\n')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'tools.pytest_translation_compatibility']
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 xfailed' in result.stdout
    records = list((tmp_path / '.translation-reports').glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record['blocking'] is False
    assert 'missing translated Next caption' in record['diagnostics']
    (tmp_path / 'test_english.py').write_text('def test_english_build():\n    assert False, "broken English"\n')
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 1
    assert '1 failed' in result.stdout


@pytest.mark.parametrize('english_status,expected_exit', [(0, 0), (1, 1)])
def test_cli_allows_bad_translations_but_requires_good_english(tmp_path, monkeypatch, english_status, expected_exit):
    from types import SimpleNamespace
    import report_translation_compatibility as reporter

    api_dir = tmp_path / 'docs/source/_static/i18n/api'
    api_dir.mkdir(parents=True)
    (api_dir / 'en.json').write_text(json.dumps({'symbols': {'current': {'text': 'Current'}}}))
    (api_dir / 'es.json').write_text(json.dumps({'schema': 2, 'symbols': {}}))
    sources = {'setting_labels': {}, 'setting_tooltips': {}, 'categories': {},
               'ui': {'Next': 'Next'}, 'module_summaries': {}}
    docs = SimpleNamespace(MODEL_SPECS={'es': None}, public_docstrings=lambda: {},
                           audit=lambda *_: english_status)
    runtime = SimpleNamespace(MODEL_SPECS={'es': None}, canonical_sources=lambda: sources,
                              audit=lambda *_: english_status, _source_hashes=lambda _: {})
    monkeypatch.setattr(reporter.importlib, 'import_module',
                        lambda name: docs if name == 'build_documentation_i18n' else runtime)
    report = tmp_path / 'report.json'
    assert reporter.main(['--root', str(tmp_path), '--output', str(report), '--english-required']) == expected_exit
    data = json.loads(report.read_text())
    assert data['api']['es']['issues'] == [{'kind': 'missing', 'symbol': 'current'}]
    assert data['runtime']['es']['issues'][0]['kind'] == 'unreadable_catalog'


def test_older_main_table_gets_only_its_known_whitespace_correction():
    from docs_publication_compat import normalize_ambient_table
    from docutils.core import publish_doctree
    from docutils.utils import SystemMessage

    border = '=========  =====================  ====================='
    rows = [' theme      shading (moved)        soften + blit (stays)', border]
    rows += [f' {name:<9}  1.072 -> see below     0.842 -> 1.115'
             for name in ('blobs', 'aurora', 'ripple', 'bokeh', 'cells', 'drift', 'resonance')]
    broken = border + '\n' + '\n'.join(rows) + '\n' + border
    with pytest.raises(SystemMessage):
        publish_doctree(broken, settings_overrides={'halt_level': 2})
    fixed = normalize_ambient_table(broken)
    assert fixed.split() == broken.split()
    publish_doctree(fixed, settings_overrides={'halt_level': 2})
    assert normalize_ambient_table(fixed) == fixed
    assert normalize_ambient_table('Other docstring') == 'Other docstring'


def test_configure_loads_the_compat_extension_without_shadowing_branch_tools(tmp_path, monkeypatch):
    """An older pinned branch must be checked by its own tools, not the publisher's."""
    from publish_docs_channels import main
    config = tmp_path / 'docs/source/conf.py'
    config.parent.mkdir(parents=True)
    config.write_text('extensions = []\n')
    before = list(sys.path)
    monkeypatch.delitem(sys.modules, 'docs_publication_compat', raising=False)
    assert main(['configure', '--root', str(tmp_path)]) == 0
    namespace = {}
    exec(compile(config.read_text(), str(config), 'exec'), namespace)
    publisher_tools = str(Path(__file__).resolve().parents[1] / 'tools')
    assert namespace['extensions'] == ['docs_publication_compat']
    assert sys.path == before
    assert sys.modules['docs_publication_compat'].__file__ == publisher_tools + '/docs_publication_compat.py'
