"""Anonymous diagnostics retain useful facts without retaining private input."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import crashreport as cr


def test_problem_lines_are_bounded_and_deduplicated_after_truncation():
    assert cr.public_problem_lines([
        'threshold too high', 'threshold too small', '', '   '], limit=12) == [
            'threshold t…']


def test_public_settings_keep_numeric_facts_but_not_names_paths_or_nested_contents(tmp_path):
    existing = tmp_path / 'private-experiment'
    existing.mkdir()
    settings = {
        'enabled': True, 'unset': None, 'count': 3, 'threshold': 1.25,
        'nan': float('nan'), 'inf': float('inf'), 'negative_inf': -float('inf'),
        'mode': 'otsu', 'blank': '   ', 'scrubbed': '<text>',
        'plate_name': 'private-experiment', 'unsafe': 'private prose',
        'src': str(existing), 'output_folder': str(tmp_path / 'missing'),
        'channels': [1, 2], 'nested': [[1, 2], {'private': 'value'}],
        'many': list(range(17)), 'mapping': {'private': 'value'},
        'unknown': object(), 'invalid/key': 'private',
    }

    public = cr.public_settings(settings)

    assert public == {
        'enabled': True, 'unset': None, 'count': 3, 'threshold': 1.25,
        'nan': 'nan', 'inf': 'inf', 'negative_inf': '-inf',
        'mode': 'otsu', 'blank': '', 'scrubbed': '<text>',
        'plate_name': '<text>', 'unsafe': '<text>',
        'src': '<path: exists=yes>', 'output_folder': '<path: exists=no>',
        'channels': [1, 2], 'nested': ['<list: n=2>', '<mapping: n=1>'],
        'many': '<list: n=17>', 'mapping': '<mapping: n=1>',
        'unknown': '<object>',
    }
    assert 'private' not in json.dumps(public)
    assert str(tmp_path) not in json.dumps(public)
    assert settings['nested'] == [[1, 2], {'private': 'value'}]


def test_public_settings_cap_counts_only_valid_names_in_sorted_order(monkeypatch):
    monkeypatch.setattr(cr, 'MAX_SETTINGS', 2)
    assert cr.public_settings({'zeta': 3, 'bad/key': 4, 'beta': 2, 'alpha': 1}) == {
        'alpha': 1, 'beta': 2}


@pytest.mark.parametrize('kind,shape,dtype,channels', [
    ('tif', (5, 7), 'uint16', 1),
    ('npy', (2, 5, 7), 'uint16', 2),
    ('npy', (17, 18, 19), 'uint16', 1),
    ('npy', (5,), 'uint16', None),
    ('png-rgb', (5, 7, 3), 'RGB', 3),
    ('png-grey', (5, 7), 'L', 1),
])
def test_input_facts_read_real_headers_without_publishing_filenames(
        tmp_path, kind, shape, dtype, channels):
    suffix = 'png' if kind.startswith('png') else kind
    path = tmp_path / ('private-sample.' + suffix)
    data = np.zeros(shape, dtype=np.uint8 if suffix == 'png' else np.uint16)
    if suffix == 'tif':
        import tifffile
        tifffile.imwrite(path, data)
    elif suffix == 'npy':
        np.save(path, data)
    else:
        from PIL import Image
        Image.fromarray(data).save(path)
    (tmp_path / 'not-an-image.txt').write_text('private')
    (tmp_path / 'directory.png').mkdir()

    facts = cr.public_input_facts({'src': [], 'source': 17, 'images': [str(tmp_path)]})

    expected = {'images': 1, 'extensions': [suffix], 'shape': list(shape), 'dtype': dtype}
    if channels is not None:
        expected['channels'] = channels
    assert facts == expected
    assert 'private' not in json.dumps(facts)
    assert path.exists()


def test_empty_or_missing_sources_have_no_invented_image_facts(tmp_path):
    assert cr.public_input_facts({'src': [], 'images': []}) == {}
    assert cr.public_input_facts({'src': str(tmp_path / 'missing')}) == {}
    assert cr.public_input_facts({'src': str(tmp_path)}) == {'images': 0, 'extensions': []}


def test_input_fact_scan_respects_the_entry_budget(tmp_path):
    np.save(tmp_path / 'first.npy', np.zeros((2, 3), dtype=np.uint8))
    np.save(tmp_path / 'second.npy', np.zeros((2, 3), dtype=np.uint8))
    assert cr.public_input_facts({'src': str(tmp_path)}, limit=1) == {
        'images': 1, 'extensions': ['npy'], 'shape': [2, 3], 'dtype': 'uint8', 'channels': 1}
    assert cr.public_input_facts({'src': str(tmp_path)}, limit=0) == {
        'images': 0, 'extensions': []}


def test_an_unreadable_source_directory_does_not_break_error_reporting(tmp_path, monkeypatch):
    def denied(_path):
        raise PermissionError('private folder cannot be listed')

    with monkeypatch.context() as patch:
        patch.setattr(cr.os, 'scandir', denied)
        facts = cr.public_input_facts({'src': str(tmp_path)})

    assert facts == {}


def test_an_unreadable_image_still_contributes_only_its_count_and_extension(tmp_path):
    (tmp_path / 'private-broken.png').write_bytes(b'not an image')
    assert cr.public_input_facts({'src': str(tmp_path)}) == {
        'images': 1, 'extensions': ['png']}


@pytest.mark.parametrize('installation', ['installer', 'editable', 'conda', 'pip', 'unknown'])
def test_install_type_uses_available_packaging_evidence(monkeypatch, installation):
    from importlib import metadata

    def distribution(name):
        assert name == 'spacr'
        if installation == 'unknown':
            raise metadata.PackageNotFoundError(name)
        assert installation != 'installer', 'frozen builds must not query distribution metadata'
        return SimpleNamespace(read_text=lambda name: (
            json.dumps({'dir_info': {'editable': True}}) if installation == 'editable' else None))

    with monkeypatch.context() as patch:
        patch.setattr(sys, 'frozen', installation == 'installer', raising=False)
        patch.setattr(metadata, 'distribution', distribution)
        patch.setattr(cr.os.path, 'isdir', lambda path: (
            installation == 'conda' and Path(path).name == 'conda-meta'))
        actual = cr._install_type()

    assert actual == installation


@pytest.mark.parametrize('failure', [None, 'memory', 'cuda', 'qt', 'no-qt'])
def test_environment_facts_survive_unavailable_optional_probes_without_loading_a_gpu(
        monkeypatch, failure):
    import psutil

    def unavailable():
        raise RuntimeError('optional probe unavailable')

    cuda = SimpleNamespace(
        is_available=unavailable if failure == 'cuda' else lambda: True,
        get_device_properties=lambda index: SimpleNamespace(
            name='Synthetic GPU', total_memory=8 * 2 ** 30))
    torch = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)))
    with monkeypatch.context() as patch:
        patch.setattr(cr, '_install_type', lambda: 'pip')
        patch.setattr(cr.platform, 'release', lambda: 'synthetic-os')
        patch.setattr(cr.platform, 'mac_ver', lambda: ('synthetic-os', (), ''))
        patch.setattr(cr.platform, 'machine', lambda: 'test-architecture')
        patch.setattr(cr.os, 'cpu_count', lambda: 4)
        patch.setattr(cr, '_distribution_version', lambda name: '2.0' if name == 'numpy' else '')
        patch.setattr(psutil, 'virtual_memory', unavailable if failure == 'memory' else
                      lambda: SimpleNamespace(total=16 * 2 ** 30))
        patch.setitem(sys.modules, 'torch', torch)
        patch.setitem(sys.modules, 'PySide6.QtCore', None if failure == 'no-qt' else
                      SimpleNamespace(qVersion=unavailable if failure == 'qt' else lambda: '6.11.2'))
        facts = cr.public_environment()

    expected = {'os_version': 'synthetic-os', 'arch': 'test-architecture',
                'cores': 4, 'install': 'pip', 'versions': {'numpy': '2.0'}}
    if failure != 'memory':
        expected['ram_gb'] = 16
    if failure != 'cuda':
        expected.update(cuda=True, gpu_model='Synthetic GPU', vram_gb=8, mps=False)
    if failure not in ('qt', 'no-qt'):
        expected['versions']['qt'] = '6.11.2'
    assert facts == expected


def test_exception_chains_stop_at_cycles_and_omit_nonexception_preamble():
    first = ValueError('unreadable /private/source')
    last = RuntimeError('processing stopped')
    last.__cause__ = first
    first.__context__ = last
    assert cr._exception_chain(last) == [{'type': 'ValueError', 'message': 'unreadable <PATH>'}]
    assert cr._exception_chain() == []
    assert cr._exception_chain(traceback_text=(
        'plain diagnostic preamble\n'
        'During handling of the above exception, another exception occurred:\n'
        'ValueError: current failure')) == []


def test_log_tail_truncation_keeps_complete_recent_lines():
    assert cr.public_log_tail('private first line\nsecond line\nthird', lines=2, limit=15) == '…third'


@pytest.mark.parametrize('elapsed,expected', [(-1, {}), ('unknown', {}), (2.34, {'elapsed_s': 2.3})])
def test_run_facts_do_not_invent_elapsed_time_or_a_preview_flag(elapsed, expected):
    assert cr._public_run({'elapsed_s': elapsed}) == expected


@pytest.mark.parametrize('field', ['environment', 'inputs', 'settings'])
def test_revalidated_summary_drops_malformed_optional_blocks(field):
    out = cr._validated_summary({field: ['not a mapping']})
    assert field not in out


def test_revalidated_summary_reduces_untrusted_metadata_to_its_public_schema(monkeypatch):
    monkeypatch.setattr(cr, 'MAX_SETTINGS', 2)
    raw = {
        'chain': [None, {'type': '', 'message': 'private text'}],
        'settings': {'invalid/key': 'private', 'first': 1, 'second': 2, 'third': 3},
        'inputs': {'images': -4, 'channels': True, 'extensions': ['PNG', 'private', 'tif'],
                   'shape': [4, True, 2.5, 6, 7, 8, 9], 'dtype': 'uint16', 'filename': 'private'},
        'environment': {'arch': 'x86_64!', 'cores': True, 'ram_gb': -2, 'vram_gb': 100001,
                        'cuda': 'yes', 'mps': True, 'install': 'untrusted',
                        'versions': {'numpy': '2.0', 'private': 'secret'}, 'host': 'private'},
    }
    original = deepcopy(raw)

    out = cr._validated_summary(raw)

    assert 'chain' not in out
    assert out['settings'] == {'first': 1, 'second': 2}
    assert out['inputs'] == {'images': 0, 'extensions': ['png', 'tif'],
                             'shape': [4, 6, 7, 8], 'dtype': 'uint16'}
    assert out['environment'] == {'arch': 'x86_64', 'ram_gb': 0, 'vram_gb': 100000,
                                  'mps': True, 'versions': {'numpy': '2.0'}}
    assert raw == original
    assert 'private' not in json.dumps(out)


def test_revalidated_summary_does_not_read_strings_as_lists_or_version_mappings():
    out = cr._validated_summary({
        'inputs': {'channels': 1, 'extensions': 'png', 'shape': 'private'},
        'environment': {'install': 'untrusted', 'versions': 'private'},
    })
    assert out['inputs'] == {'channels': 1}
    assert 'environment' not in out


@pytest.mark.parametrize('advice,expected', [
    ('Read /private/source then retry', 'Read <PATH> then retry'), ('   \n', None)])
def test_a_relay_summary_is_scrubbed_and_omitted_if_it_has_no_text(monkeypatch, advice, expected):
    monkeypatch.setattr(cr, '_post_json', lambda path, summary, timeout: (
        200, {'issue_number': 12, 'ai_summary_source': 'relay', 'ai_summary': advice}))

    outcome = cr._send_one({'message': 'failed'}, timeout=1)

    assert outcome['status'] == cr.REPORT_SENT and outcome['issue_number'] == 12
    assert outcome['ai_summary_source'] == 'relay'
    if expected is None:
        assert 'ai_summary' not in outcome
    else:
        assert outcome['ai_summary'] == expected


def test_a_problem_containing_only_private_paths_and_counts_adds_no_signature_content():
    base = {'kind': cr.KIND_USER_ERROR, 'exception_type': 'ConfigurationError',
            'module': 'mask', 'frames': []}
    empty = cr.explanation_signature(dict(base, message=''))
    assert cr.explanation_signature(dict(base, message='- /private/source 123 <PATH>')) == empty
    assert cr.explanation_signature(dict(base, message='- missing channel')) != empty
