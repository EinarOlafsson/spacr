"""Publication preserves branch boundaries and records translation debt."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from publish_docs_channels import assemble, prepare
from report_translation_compatibility import api_issues, audit_record, runtime_issues


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
    (tutorial / 'index.html').write_text('<body><header>Fixed toolbar</header><main id="lesson-content">Player</main></body>')
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
