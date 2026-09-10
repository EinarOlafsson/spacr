"""Temporary synthetic fixtures for pure manifest composition; no app/browser."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


_SPEC = importlib.util.spec_from_file_location('compose_report',
    Path(__file__).resolve().parents[1] / 'compose_report_capture.py')
compose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compose)


def write(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def captures(tmp_path):
    root = tmp_path / 'captures'
    root.mkdir()
    gui, browser, output = [root / name for name in ('gui', 'browser', 'report_composed')]
    gui.mkdir(); browser.mkdir()
    html = tmp_path / 'fixture.html'
    html.write_text('Fixture only; never loaded into a browser.')
    provenance = {'completed_capture': True, 'module': 'report', 'commit': 'a' * 40, 'version': 'fixture'}
    report = {'accepted': True, 'lesson': '29_report', 'original_preserved': True,
              'private_material_files_unchanged': True, 'output': str(html),
              'output_sha256': compose._sha(html), 'source': str(tmp_path / 'private_source'),
              'original_source': str(tmp_path / 'original_source')}
    write(gui / 'provenance.json', provenance)
    write(gui / 'scientific_acceptance.json', report)
    write(gui / 'report_output_evidence.json', report)
    for path, color in [(gui / '01_controls.png', '#102030'), (browser / '01_report_top.png', '#403020')]:
        Image.new('RGB', (3840, 2160), color).save(path)
    gui_frames = {'01_controls': {'image': '01_controls.png', 'sha256': compose._sha(gui / '01_controls.png'),
                                  'buttons': [{'name': 'Run', 'rect': [1, 2, 30, 40]}], 'dialogs': []}}
    write(gui / 'frames.json', gui_frames)
    png = browser / '01_report_top.png'
    browser_frames = {'kind': 'standalone_chrome_report', 'frames': [
        {'name': png.name, 'path': str(png), 'sha256': compose._sha(png), 'bytes': png.stat().st_size,
         'dimensions': [3840, 2160], 'viewport_css': {'width': 1920, 'height': 1080},
         'device_scale_factor': 2, 'scroll_css': [0, 0]}]}
    write(browser / 'frames.json', browser_frames)
    write(browser / 'browser_dom.json', {'fixture': 'not browser observations'})
    review = {'accepted': True, 'browser_reviewed': True, 'dom_verified': True,
              'original_preserved': True, 'private_source_exact_post_generation_state_preserved': True,
              'receipt_unchanged': True, 'html_unchanged': True, 'native_human_publish': False,
              'published': False, 'open_control_clicked': False, 'html_css_dom_injected': False,
              'report_recomputed': False, 'horizontal_overflow': False, 'lesson': '29_report',
              'receipt': str(gui / 'report_output_evidence.json'),
              'receipt_sha256': compose._sha(gui / 'report_output_evidence.json'),
              'html': str(html), 'html_sha256': compose._sha(html), 'source': report['source'],
              'frames_manifest_sha256': compose._sha(browser / 'frames.json'),
              'browser_dom_sha256': compose._sha(browser / 'browser_dom.json'),
              'frames': browser_frames['frames'], 'viewport_css': {'width': 1920, 'height': 1080},
              'device_scale_factor': 2, 'browser_version': 'fixture', 'chrome_executable': '/fixture/chrome',
              'verification_code_sha256': {'fixture.py': 'b' * 64},
              **{key: [] for key in ('blocked_requests', 'javascript_errors', 'console_errors', 'request_failures', 'downloads')}}
    write(browser / 'browser_review.json', review)
    return gui, browser, output


def test_positive_composition_writes_only_three_manifests_and_preserves_sources(captures):
    gui, browser, output = captures
    before = {str(path): compose._sha(path) for root in (gui, browser) for path in root.iterdir()}
    result = compose.compose_report_capture(gui, browser, output)
    assert result['total_frames'] == 2
    assert sorted(path.name for path in output.iterdir()) == ['frames.json', 'provenance.json', 'scientific_acceptance.json']
    frames = json.loads((output / 'frames.json').read_text())
    assert list(frames) == ['01_controls', 'browser_01_report_top']
    assert frames['01_controls']['image'] == '../gui/01_controls.png'
    assert frames['01_controls']['buttons'][0]['rect'] == [1, 2, 30, 40]
    assert frames['browser_01_report_top']['image'] == '../browser/01_report_top.png'
    assert frames['browser_01_report_top']['buttons'] == []
    for frame in frames.values():
        assert compose._sha(output / frame['image']) == frame['sha256']
        with Image.open(output / frame['image']) as image:
            assert image.size == (3840, 2160)
    provenance = json.loads((output / 'provenance.json').read_text())
    acceptance = json.loads((output / 'scientific_acceptance.json').read_text())
    assert provenance['completed_capture'] is True
    assert provenance['commit'] == 'a' * 40
    assert acceptance['accepted'] is True and acceptance['composition_verified'] is True
    assert acceptance['scientific_validation_performed'] is False
    assert acceptance['native_human_publish'] is False and acceptance['published'] is False
    assert before == {str(path): compose._sha(path) for root in (gui, browser) for path in root.iterdir()}


@pytest.mark.parametrize('change', [
    'gui_incomplete', 'gui_unaccepted', 'gui_report_disagreement', 'browser_unaccepted',
    'receipt_sha', 'receipt_path', 'html_sha', 'html_path', 'html_bytes', 'source_path',
    'browser_manifest_sha', 'browser_inventory', 'dom_hash', 'gui_image_hash', 'browser_image_hash',
    'geometry', 'declared_geometry', 'key_collision', 'duplicate_browser_key', 'network_error',
    'overflow', 'human_publish', 'missing_version'])
def test_rejects_broken_binding_or_acceptance_without_creating_destination(captures, change):
    gui, browser, output = captures
    p = json.loads((gui / 'provenance.json').read_text())
    a = json.loads((gui / 'scientific_acceptance.json').read_text())
    g = json.loads((gui / 'frames.json').read_text())
    r = json.loads((browser / 'browser_review.json').read_text())
    f = json.loads((browser / 'frames.json').read_text())
    if change == 'gui_incomplete': p['completed_capture'] = False
    elif change == 'gui_unaccepted': a['accepted'] = False
    elif change == 'gui_report_disagreement': a['source'] += '_wrong'
    elif change == 'browser_unaccepted': r['accepted'] = False
    elif change == 'receipt_sha': r['receipt_sha256'] = 'c' * 64
    elif change == 'receipt_path': r['receipt'] = str(gui / 'other.json')
    elif change == 'html_sha': r['html_sha256'] = 'c' * 64
    elif change == 'html_path': r['html'] += '_wrong'
    elif change == 'html_bytes': Path(r['html']).write_text('Changed fixture')
    elif change == 'source_path': r['source'] += '_wrong'
    elif change == 'browser_manifest_sha': r['frames_manifest_sha256'] = 'c' * 64
    elif change == 'browser_inventory': r['frames'] = []
    elif change == 'dom_hash': r['browser_dom_sha256'] = 'c' * 64
    elif change == 'gui_image_hash': g['01_controls']['sha256'] = 'c' * 64
    elif change == 'browser_image_hash': f['frames'][0]['sha256'] = 'c' * 64
    elif change == 'geometry':
        Image.new('RGB', (1920, 1080)).save(gui / '01_controls.png')
        g['01_controls']['sha256'] = compose._sha(gui / '01_controls.png')
    elif change == 'declared_geometry': f['frames'][0]['dimensions'] = [1920, 1080]
    elif change == 'key_collision': g['browser_01_report_top'] = deepcopy(g['01_controls'])
    elif change == 'duplicate_browser_key': f['frames'].append(deepcopy(f['frames'][0]))
    elif change == 'network_error': r['blocked_requests'] = ['https://example.invalid']
    elif change == 'overflow': r['horizontal_overflow'] = True
    elif change == 'human_publish': r['native_human_publish'] = True
    elif change == 'missing_version': p.pop('version')
    write(gui / 'provenance.json', p); write(gui / 'scientific_acceptance.json', a)
    write(gui / 'frames.json', g)
    if change in ('browser_image_hash', 'declared_geometry', 'duplicate_browser_key'):
        write(browser / 'frames.json', f)
        r['frames_manifest_sha256'] = compose._sha(browser / 'frames.json')
        r['frames'] = f['frames']
    write(browser / 'browser_review.json', r)
    with pytest.raises(ValueError):
        compose.compose_report_capture(gui, browser, output)
    assert not output.exists()


def test_existing_destination_is_never_overwritten(captures):
    gui, browser, output = captures
    output.mkdir(); (output / 'keep').write_text('user content')
    with pytest.raises(ValueError, match='new sibling'):
        compose.compose_report_capture(gui, browser, output)
    assert (output / 'keep').read_text() == 'user content'
    assert list(output.iterdir()) == [output / 'keep']


def test_frame_cannot_escape_its_source_capture(captures):
    gui, browser, output = captures
    g = json.loads((gui / 'frames.json').read_text())
    g['01_controls']['image'] = '../browser/01_report_top.png'
    g['01_controls']['sha256'] = compose._sha(browser / '01_report_top.png')
    write(gui / 'frames.json', g)
    with pytest.raises(ValueError, match='inside its capture'):
        compose.compose_report_capture(gui, browser, output)
    assert not output.exists()


def test_read_only_build_creates_nothing(captures):
    gui, browser, output = captures
    _, frames, provenance, acceptance, _ = compose._build(gui, browser, output)
    assert len(frames) == 2
    assert provenance['completed_capture'] is False and acceptance['accepted'] is False
    assert not output.exists()
