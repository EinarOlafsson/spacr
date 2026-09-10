"""Pure browser-evidence guards only: fixtures are not real generated reports.

No Playwright/Chrome/Qt/application launch or report generation occurs here.
"""
from copy import deepcopy
import hashlib
import html
import importlib.util
import math
from pathlib import Path
import struct

import pytest


_PATH = Path(__file__).resolve().parents[1] / 'capture_report_browser.py'
_SPEC = importlib.util.spec_from_file_location('report_browser_guards', _PATH)
browser = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(browser)


@pytest.fixture
def profile():
    facts = {
        'vector_figures': [f'results/plate1_A01_{i}/{name}.pdf' for i in (1, 2)
                           for name in ('after_filtration', 'before_filtration', 'pngs')],
        'database_counts': {'artifacts.db': {'artifacts': 1},
                            'measurements/measurements.db': {'cell': 103, 'cytoplasm': 103,
                                                            'png_list': 103, 'run_status': 1}}}
    sections = []
    for key in browser._REPORT.CORE_SECTIONS:
        sections.append({'key': key, 'status': 'missing' if key in
                         ('segmentation_qc', 'plate_qc') else 'ok'})
    summary = {'title': 'Pure fixture, not a captured report', 'src': '/private/example/project',
               'status_detail': 'Complete — every stamped step processed every item.',
               'sections': sections}
    tables = {'run_status': [{'caption': 'Recorded steps', 'rows': [
        ['artifact', 'name', 'status', 'attempted', 'succeeded', 'failed'],
        ['measurements.db', 'measure_crop', 'complete', '2', '2', '0']]}],
        'figures': [{'caption': 'Six fixture identities', 'rows':
                    [['figure', 'why it is not shown']] +
                    [[name, 'vector — not embeddable'] for name in facts['vector_figures']]}],
        'statistics': [{'caption': relative + ' — tables and row counts',
                        'rows': [['table', 'rows']] + [[name, str(count)] for name, count in counts.items()]}
                       for relative, counts in facts['database_counts'].items()]}
    parts = ['<!doctype html><html><head><style>body { font-size: 15px; }</style></head><body>']
    for section in sections:
        key = section['key']
        classes = 'chapter missing' if section['status'] == 'missing' else 'chapter'
        parts.append(f'<section id="{key}" class="{classes}"><h2>{key}</h2>')
        if section['status'] == 'missing':
            parts.append('<p>not available</p>')
        for table in tables.get(key, []):
            parts.append('<table><caption>' + html.escape(table['caption']) + '</caption>')
            for index, row in enumerate(table['rows']):
                tag = 'th' if index == 0 else 'td'
                parts.append('<tr>' + ''.join(f'<{tag}>' + html.escape(cell) + f'</{tag}>' for cell in row) + '</tr>')
            parts.append('</table>')
        parts.append('</section>')
    text = ''.join(parts) + '</body></html>'
    expected = browser._expected_sections(text)
    dom = {'title': summary['title'], 'source': summary['src'],
           'banner_classes': ['banner', 'complete'],
           'banner_text': summary['status_detail'] + ' 0 of 6 figure(s) are embedded in this file.',
           'toc': ['#' + key for key in browser._REPORT.CORE_SECTIONS],
           'images': 0, 'sections': deepcopy(expected)}
    return text, expected, dom, summary, facts


def test_exact_full_dom_profile_passes(profile):
    _, expected, dom, summary, facts = profile
    assert browser._verify_dom(dom, expected, summary, facts)


@pytest.mark.parametrize('change', ['section_missing', 'section_extra', 'section_reordered',
    'qc_missing_removed', 'source', 'title', 'completion', 'figure_embedded', 'image',
    'toc', 'text', 'row_missing', 'row_extra', 'column_missing', 'column_extra',
    'wrong_count', 'wrong_pdf', 'wrong_stamp', 'caption'])
def test_changed_dom_is_rejected(profile, change):
    _, expected, dom, summary, facts = profile
    if change == 'section_missing':
        dom['sections'].pop()
    elif change == 'section_extra':
        dom['sections'].append(deepcopy(dom['sections'][0]))
    elif change == 'section_reordered':
        dom['sections'].reverse()
    elif change == 'qc_missing_removed':
        dom['sections'][2]['classes'].remove('missing')
    elif change in ('source', 'title'):
        dom[change] = 'wrong'
    elif change == 'completion':
        dom['banner_classes'] = ['banner', 'partial']
    elif change == 'figure_embedded':
        dom['banner_text'] = dom['banner_text'].replace('0 of 6', '6 of 6')
    elif change == 'image':
        dom['images'] = 1
    elif change == 'toc':
        dom['toc'].pop()
    elif change == 'text':
        dom['sections'][2]['text'] = 'QC passed'
    elif change == 'row_missing':
        dom['sections'][5]['tables'][1]['rows'].pop()
    elif change == 'row_extra':
        dom['sections'][5]['tables'][1]['rows'].append(['fake', '103'])
    elif change == 'column_missing':
        dom['sections'][5]['tables'][1]['rows'][1].pop()
    elif change == 'column_extra':
        dom['sections'][5]['tables'][1]['rows'][1].append('extra')
    elif change == 'wrong_count':
        dom['sections'][5]['tables'][1]['rows'][1][1] = '206'
    elif change == 'wrong_pdf':
        dom['sections'][4]['tables'][0]['rows'][1][0] = 'other.pdf'
    elif change == 'wrong_stamp':
        dom['sections'][0]['tables'][0]['rows'][1][4] = '1'
    else:
        dom['sections'][5]['tables'][0]['caption'] = 'other.db — tables and row counts'
    with pytest.raises(RuntimeError):
        browser._verify_dom(dom, expected, summary, facts)


@pytest.mark.parametrize('change', ['wrong_count', 'missing_table', 'extra_pdf', 'wrong_pdf'])
def test_source_binding_is_not_only_dom_html_equality(profile, change):
    _, expected, dom, summary, facts = profile
    if change == 'wrong_count':
        facts['database_counts']['measurements/measurements.db']['cell'] = 206
    elif change == 'missing_table':
        facts['database_counts']['measurements/measurements.db']['new_table'] = 1
    elif change == 'extra_pdf':
        facts['vector_figures'].append('extra.pdf')
    else:
        facts['vector_figures'][0] = 'wrong.pdf'
    with pytest.raises(RuntimeError):
        browser._verify_dom(dom, expected, summary, facts)


@pytest.mark.parametrize('injection', [
    '<script>alert(1)</script>', '<img src="file:///private/image.png">',
    '<iframe src="https://example.invalid"></iframe>', '<a href="https://example.invalid">remote</a>',
    '<style>@import "https://example.invalid";</style>',
    '<style>body { background: url(https://example.invalid/a); }</style>',
    '<div style="background: url(file:///private/a)"></div>',
    '<div onclick="alert(1)"></div>', '<meta http-equiv="refresh" content="0;url=bad">',
    '<link rel="stylesheet" href="file:///private/style.css">', '<base href="file:///other/">'])
def test_active_or_external_html_is_rejected(profile, injection):
    text, *_ = profile
    with pytest.raises(RuntimeError, match='active/resource'):
        browser._expected_sections(text.replace('</body>', injection + '</body>'))


def test_native_fragment_link_is_allowed(profile):
    text, expected, *_ = profile
    assert browser._expected_sections(text.replace('<body>', '<body><a href="#run_status">Contents</a>')) == expected


@pytest.mark.parametrize('url,allowed', [
    ('file:///private/report.html', True), ('file:///private/report.html#figures', True),
    ('https://example.invalid/report.html', False), ('http://127.0.0.1/report.html', False),
    ('data:text/html,hello', False), ('file:///private/other.html', False),
    ('file://remote/private/report.html', False), ('file:///private/report.html?changed=1', False)])
def test_request_allowlist_is_exact_local_file_only(url, allowed):
    assert browser._allowed_request(url, Path('/private/report.html')) is allowed


def test_hash_bound_file_and_changes(tmp_path):
    path = tmp_path / 'receipt.json'
    raw = b'{"fixture": true}'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert browser._read_bound(path, digest) == raw
    path.write_bytes(b'{"fixture": false}')
    with pytest.raises(RuntimeError, match='SHA256 differs'):
        browser._read_bound(path, digest)


def test_bound_file_rejects_symlink_and_size(tmp_path):
    path = tmp_path / 'file'
    path.write_bytes(b'abc')
    digest = hashlib.sha256(b'abc').hexdigest()
    with pytest.raises(RuntimeError, match='bounded'):
        browser._read_bound(path, digest, maximum=2)
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(RuntimeError, match='bounded'):
        browser._read_bound(link, digest)


@pytest.mark.parametrize('width,height,accepted', [(3840, 2160, True), (1920, 1080, False), (3840, 2100, False)])
def test_screenshot_header_has_exact_hidpi_geometry(tmp_path, width, height, accepted):
    path = tmp_path / 'header-only-fixture.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', width, height))
    if accepted:
        assert browser._png_geometry(path) == [3840, 2160]
    else:
        with pytest.raises(RuntimeError, match='3840'):
            browser._png_geometry(path)


@pytest.fixture
def accepted_flags():
    return {'accepted': True, 'lesson': '29_report', 'original_preserved': True,
            'private_material_files_unchanged': True,
            'html': {'accepted': True, 'section_count': 8, 'embedded_images': 0,
                     'external_dependencies': False},
            'synthetic_status_created': False, 'source_pipeline_rerun': False,
            'active_jobs': 0, 'job_results': [True, True]}


def test_accepted_receipt_contract(accepted_flags):
    browser._require_accepted_receipt(accepted_flags)


@pytest.mark.parametrize('field,value', [
    ('accepted', False), ('lesson', '31_external_masks'), ('original_preserved', False),
    ('private_material_files_unchanged', False), ('synthetic_status_created', True),
    ('source_pipeline_rerun', True), ('active_jobs', ['still running']),
    ('active_jobs', []), ('active_jobs', False), ('active_jobs', True),
    ('active_jobs', 0.0), ('active_jobs', None), ('active_jobs', '0'),
    ('active_jobs', 1), ('active_jobs', -1),
    ('job_results', []), ('job_results', [False]), ('job_results', ['True']),
    ('html', {'accepted': False}), ('html', {'accepted': True, 'section_count': 7})])
def test_unaccepted_or_incomplete_receipt_rejected(accepted_flags, field, value):
    accepted_flags[field] = value
    with pytest.raises(RuntimeError, match='accepted real GUI'):
        browser._require_accepted_receipt(accepted_flags)


@pytest.mark.parametrize('change', [None, 'original_bytes', 'private_bytes', 'private_added',
                                  'private_sidecar_changed', 'initial_copy', 'change_record'])
def test_exact_post_generation_and_original_snapshots(change):
    original = {'artifacts.db': {'sha256': 'a' * 64, 'bytes': 4096}}
    private = {**deepcopy(original), 'artifacts.db-shm': {'sha256': 'b' * 64, 'bytes': 32768}}
    changes = [{'path': 'artifacts.db-shm', 'change': 'added'}]
    receipt = {'source_files_before': deepcopy(original),
               'source_files_after': deepcopy(private),
               'original_source_files_before': deepcopy(original),
               'private_sqlite_sidecar_changes': deepcopy(changes)}
    if change == 'original_bytes':
        original['artifacts.db']['sha256'] = 'c' * 64
    elif change == 'private_bytes':
        private['artifacts.db']['sha256'] = 'c' * 64
    elif change == 'private_added':
        private['unexpected'] = {'sha256': 'c' * 64, 'bytes': 1}
    elif change == 'private_sidecar_changed':
        private['artifacts.db-shm']['sha256'] = 'c' * 64
    elif change == 'initial_copy':
        receipt['source_files_before'] = {}
    elif change == 'change_record':
        changes = []
    if change is None:
        browser._verify_snapshots(receipt, private, original, changes)
    else:
        with pytest.raises(RuntimeError):
            browser._verify_snapshots(receipt, private, original, changes)


@pytest.mark.parametrize('timeout', [0, -1, math.nan, math.inf])
def test_invalid_timeout_rejected_without_loading_or_launching(tmp_path, timeout):
    with pytest.raises(ValueError, match='positive and finite'):
        browser.capture_report_browser(tmp_path / 'not_read', 'not_read', tmp_path / 'not_created', timeout=timeout)
    assert not (tmp_path / 'not_created').exists()


def test_truncated_png_header_rejected(tmp_path):
    path = tmp_path / 'truncated.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n')
    with pytest.raises(RuntimeError, match='not a PNG'):
        browser._png_geometry(path)
