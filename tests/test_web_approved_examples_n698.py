"""Public sample transport contracts; no network or scientific dataset approval."""
import hashlib
import io

import numpy as np
import pytest
import requests
import tifffile

from tools.web_mask_generator.examples import ExampleStore


@pytest.fixture
def source():
    output = io.BytesIO()
    tifffile.imwrite(output, np.arange(35, dtype=np.uint16).reshape(5, 7), metadata={'axes': 'YX'})
    data = output.getvalue()
    record = {'family': 'live_cell', 'repo_id': 'owner/public-tests', 'revision': 'a' * 40,
              'filename': 'test/image.tif', 'sha256': hashlib.sha256(data).hexdigest(),
              'bytes': len(data), 'credit': 'Example attribution', 'split': 'test'}
    return record, data


@pytest.fixture
def transport(monkeypatch):
    called = []
    class Response:
        chunks = []
        error = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            called.append('response closed')
        def raise_for_status(self):
            if self.error:
                raise self.error
        def iter_content(self, **kwargs):
            yield from self.chunks
    class Session:
        trust_env = True
        def __enter__(self):
            return self
        def __exit__(self, *args):
            called.append('session closed')
        def get(self, url, **kwargs):
            called.append((url, kwargs, self.trust_env))
            return Response()
    monkeypatch.setattr(requests, 'Session', Session)
    return Response, called


def test_random_approved_image_checks_bytes_headers_credit_and_cleanup(source, transport, tmp_path):
    record, data = source
    response, called = transport
    response.chunks = [data[:50], b'', data[50:]]
    store = ExampleStore([record], job_parent=tmp_path)
    record['credit'] = 'mutated caller data'
    assert store.available('live_cell') and not store.available('toxoplasma_pv')
    with store.random_image('live_cell') as example:
        path = example['image']
        assert path.read_bytes() == data
        assert example['source']['credit'] == 'Example attribution'
        assert example['source']['source_url'].startswith('https://huggingface.co/datasets/owner/public-tests/blob/' + 'a' * 40)
        assert example['description']['series'][0]['shape'] == [5, 7]
        assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not path.exists() and not list(tmp_path.iterdir())
    assert called[0] == ('https://huggingface.co/datasets/owner/public-tests/resolve/' + 'a' * 40 + '/test/image.tif', {'stream': True, 'timeout': (10, 30)}, False)
    assert called[-2:] == ['response closed', 'session closed']


@pytest.mark.parametrize('failure', ['oversize', 'short', 'checksum', 'http', 'consumer'])
def test_download_failures_remove_private_example(source, transport, tmp_path, failure):
    record, data = source
    response, _ = transport
    response.chunks = [data]
    if failure == 'oversize':
        response.chunks = [data + b'oversize']
    elif failure == 'short':
        response.chunks = [data[:-1]]
    elif failure == 'checksum':
        record['sha256'] = '00' * 32
    elif failure == 'http':
        response.error = requests.HTTPError('failed')
    store = ExampleStore([record], job_parent=tmp_path)
    with pytest.raises((ValueError, requests.HTTPError, RuntimeError)):
        with store.random_image('live_cell'):
            if failure == 'consumer':
                raise RuntimeError('client failed')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', [
    {'family': 'unknown'}, {'family': []}, {'filename': '.'}, {'repo_id': '../repo'}, {'split': 'train'}, {'revision': 'main'}, {'sha256': 'bad'},
    {'bytes': True}, {'bytes': 2**40}, {'credit': ''}, {'repo_id': 'https://evil.invalid'},
    {'filename': '../outside.tif'}, {'filename': '/absolute.tif'}, {'filename': 'a\\b.tif'},
    {'filename': 'a//b.tif'}, {'extra': 'value'},
])
def test_invalid_operator_manifest_refuses_before_network(source, transport, change):
    record, _ = source
    record.update(change)
    with pytest.raises(ValueError):
        ExampleStore([record])
    assert transport[1] == []


def test_unconfigured_family_has_no_network_or_guessed_dataset(transport):
    store = ExampleStore([])
    with pytest.raises(ValueError, match='No approved'):
        with store.random_image('live_cell'):
            pytest.fail('unconfigured example served')
    assert transport[1] == []


def test_example_filename_nul_refuses_before_any_network_or_file_creation(source, monkeypatch, tmp_path):
    record, _ = source
    record['filename'] = 'test/image\x00.tif'

    def no_network():
        pytest.fail('NUL example filename reached network setup')

    monkeypatch.setattr(requests, 'Session', no_network)
    with pytest.raises(ValueError, match='immutable HF dataset revision'):
        store = ExampleStore([record], job_parent=tmp_path)
        with store.random_image('live_cell'):
            pytest.fail('NUL example filename became an available image')
    assert not list(tmp_path.iterdir())
