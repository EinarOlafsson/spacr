"""Readback authenticates, respects resolver cooldowns and still checks bytes."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import huggingface_hub
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish_release_candidate as publication


@pytest.fixture
def readback_case(tmp_path, monkeypatch):
    data = b'original media bytes'
    media = tmp_path / 'media_host/lesson/audio/en/voice.m4a'
    media.parent.mkdir(parents=True)
    media.write_bytes(data)
    relative = media.relative_to(tmp_path / 'media_host').as_posix()
    record = dict(path='media_host/' + relative, bytes=len(data),
                  sha256=hashlib.sha256(data).hexdigest())
    (tmp_path / 'release-manifest.json').write_text(json.dumps(dict(files=[record])))
    entry = type('RepoFile', (), {} )()
    entry.path, entry.size, entry.lfs = relative, len(data), None
    entry.blob_id = publication.git_blob_id(media)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        list_repo_tree=lambda *args, **kwargs: [entry]))
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: 'test-token')
    calls, waits = [], []
    monkeypatch.setattr(publication.time, 'sleep', waits.append)

    def install(statuses, body=data, headers=None):
        def get(url, **kwargs):
            calls.append((url, kwargs))
            response = requests.Response()
            response.status_code = statuses.pop(0)
            response.headers.update(headers or {})
            response._content = body
            response._content_consumed = True
            response.url = url
            return response
        monkeypatch.setattr(requests, 'get', get)
    return tmp_path, install, calls, waits


def test_rate_limited_readback_waits_for_reset_and_hashes_authenticated_bytes(readback_case):
    root, install, calls, waits = readback_case
    install([429, 200], headers={'RateLimit': '"resolvers";r=0;t=45', 'Retry-After': '30'})
    result = publication.readback(root, 'a' * 40, workers=1)
    assert result['passed'] and result['downloaded_sha256_matched'] == 1
    assert waits == [46]
    assert all(call[1]['headers'] == {'Authorization': 'Bearer test-token'} for call in calls)
    assert all('/resolve/' + 'a' * 40 + '/' in call[0] for call in calls)


def test_authenticated_download_still_rejects_wrong_media(readback_case):
    root, install, _, _ = readback_case
    install([200], body=b'corrupted media')
    result = publication.readback(root, 'a' * 40, workers=1)
    assert result['passed'] is False
    assert result['downloaded_sha256_matched'] == 0
    assert result['download_failures'] == ['lesson/audio/en/voice.m4a']


def test_persistent_host_failure_cannot_produce_a_passed_receipt(readback_case):
    root, install, _, waits = readback_case
    install([429] * 4, headers={'Retry-After': '60'})
    with pytest.raises(requests.HTTPError):
        publication.readback(root, 'a' * 40, workers=1)
    assert waits == [61] * 3
