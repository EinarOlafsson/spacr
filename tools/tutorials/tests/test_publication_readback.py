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
    (tmp_path / 'release-manifest.json').write_text(json.dumps(dict(files=[record])), encoding="utf-8")
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


@pytest.fixture
def reuse_media_case(tmp_path, monkeypatch):
    baseline, root = tmp_path / 'baseline', tmp_path / 'player-only'
    records, entries = [], []
    for relative, data, lfs in [('lesson/video/master.mp4', b'original master', True),
                                ('lesson/audio/en/voice.json', b'{"original":true}', False)]:
        for folder in (baseline, root):
            path = folder / 'media_host' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        records.append(dict(path='media_host/' + relative, bytes=len(data), sha256=sha))
        entry = type('RepoFile', (), {})()
        entry.path, entry.size = relative, len(data)
        entry.lfs = SimpleNamespace(sha256=sha) if lfs else None
        entry.blob_id = publication.git_blob_id(root / 'media_host' / relative)
        entries.append(entry)
    dump = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text(json.dumps(value)))
    manifest = dict(routes=1, ready_lessons=1, hosted_web_lessons=[],
                    media_host_bytes=sum(r['bytes'] for r in records), files=records)
    for folder in (baseline, root):dump(folder / 'release-manifest.json', manifest)
    checked = dict(commit='a' * 40, passed=True, files_expected=2, metadata_matched=2,
                   downloaded_sha256_matched=2, metadata_failures=[], download_failures=[],
                   bytes_read_back=manifest['media_host_bytes'], files_on_revision_not_in_candidate=[])
    receipt = dict(repository=f'datasets/{publication.HF_DATASET}', branch='original-branch', tag='verified-tag',
                   commit='a' * 40, base_commit='b' * 40, media_root=publication.HOST + 'a' * 40,
                   manifest_sha256=publication.digest(baseline / 'release-manifest.json'),
                   media_files=2, media_bytes=manifest['media_host_bytes'], readback=checked,
                   pages=dict(ready=1, routes=1, hosted_web_copies=0, index_sha256='d' * 64))
    dump(baseline / publication.RECEIPT, receipt)
    dump(baseline / 'web/catalog/lessons_en.json', dict(lessons=[dict(id='01_test', status='ready')]))
    dump(baseline / 'checks/published-media-browser-checks.json', dict(passed=True, published=False,
         manifest_sha256=receipt['manifest_sha256'], media_root=receipt['media_root'],
         index_sha256='d' * 64, placeholder_cases=[], ready_playback_cases=[dict(lesson='01_test', passed=True)]))
    refs = {'original-branch': 'a' * 40, 'verified-tag': 'a' * 40, 'main': 'c' * 40}
    queries, validations = [], []
    def info(*args, revision='main', **kwargs):
        queries.append(revision)
        return SimpleNamespace(sha=refs[revision])
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        dataset_info=info, list_repo_tree=lambda *args, **kwargs: entries,
        list_repo_commits=lambda *args, **kwargs: [SimpleNamespace(commit_id='a' * 40), SimpleNamespace(commit_id='b' * 40)]))
    monkeypatch.setattr(publication, 'require_verified_candidate', lambda folder: (validations.append(('current', folder)), publication.read(folder / 'release-manifest.json'))[1])
    monkeypatch.setattr(publication, 'validate', lambda folder, **kwargs: validations.append(('baseline', folder, kwargs)))
    monkeypatch.setattr(requests, 'get', lambda *a, **k: pytest.fail('Media reuse must not download hosted bytes'))
    return SimpleNamespace(root=root, baseline=baseline, receipt=receipt, refs=refs, entries=entries,
                           queries=queries, validations=validations, dump=dump)


def test_reuse_media_keeps_historical_base_and_readback_when_main_advanced(reuse_media_case):
    case = reuse_media_case
    publication.reuse_media_receipt(case.root, case.baseline)
    result = publication.read(case.root / publication.RECEIPT)
    assert result['base_commit'] == 'b' * 40 and case.refs['main'] == 'c' * 40
    assert result['commit'] == 'a' * 40 and result['readback'] == case.receipt['readback']
    assert result['readback_source'] == 'unchanged media reused from verified publication'
    assert result['manifest_sha256'] == publication.digest(case.root / 'release-manifest.json')
    assert result['media_reuse']['baseline_receipt_sha256'] == publication.digest(case.baseline / publication.RECEIPT)
    assert 'pages' not in result and case.queries == ['original-branch', 'verified-tag']
    assert case.validations == [('current', case.root), ('baseline', case.baseline, {'require_browser': True})]
    assert publication.read(case.baseline / publication.RECEIPT) == case.receipt


@pytest.mark.parametrize('mutation', ['changed_media', 'branch', 'tag', 'missing_file', 'size', 'lfs_sha',
    'plain_blob', 'extra_file', 'readback_incomplete', 'readback_bytes', 'historical_base', 'published_failed',
    'published_wrong_lesson', 'published_wrong_manifest', 'published_wrong_index'])
def test_reuse_media_rejects_changed_identity_or_incomplete_original_proof(reuse_media_case, mutation):
    c = reuse_media_case
    if mutation == 'changed_media':
        value = publication.read(c.root / 'release-manifest.json');value['files'][0]['sha256'] = 'f' * 64
        c.dump(c.root / 'release-manifest.json', value)
    elif mutation in ('branch', 'tag'):c.refs['original-branch' if mutation == 'branch' else 'verified-tag'] = 'e' * 40
    elif mutation == 'missing_file':c.entries.pop()
    elif mutation == 'size':c.entries[0].size += 1
    elif mutation == 'lfs_sha':c.entries[0].lfs.sha256 = 'f' * 64
    elif mutation == 'plain_blob':c.entries[1].blob_id = 'f' * 40
    elif mutation == 'extra_file':
        extra = type('RepoFile', (), {})();extra.path = 'not-in-original-readback';c.entries.append(extra)
    elif mutation == 'historical_base':
        value = publication.read(c.baseline / publication.RECEIPT);value['base_commit'] = 'f' * 40
        c.dump(c.baseline / publication.RECEIPT, value)
    elif mutation.startswith('readback_'):
        value = publication.read(c.baseline / publication.RECEIPT)
        value['readback']['downloaded_sha256_matched' if mutation == 'readback_incomplete' else 'bytes_read_back'] = 0
        c.dump(c.baseline / publication.RECEIPT, value)
    else:
        path = c.baseline / 'checks/published-media-browser-checks.json';value = publication.read(path)
        if mutation == 'published_failed':value['passed'] = False
        elif mutation == 'published_wrong_lesson':value['ready_playback_cases'][0]['lesson'] = 'not-the-lesson'
        elif mutation == 'published_wrong_manifest':value['manifest_sha256'] = 'f' * 64
        else:value['index_sha256'] = 'f' * 64
        c.dump(path, value)
    with pytest.raises(SystemExit):publication.reuse_media_receipt(c.root, c.baseline)
    assert not (c.root / publication.RECEIPT).exists()


def test_reuse_media_preserves_an_existing_receipt(reuse_media_case):
    c = reuse_media_case
    c.dump(c.root / publication.RECEIPT, {'preserve': True})
    with pytest.raises(SystemExit):publication.reuse_media_receipt(c.root, c.baseline)
    assert publication.read(c.root / publication.RECEIPT) == {'preserve': True}


def test_original_resume_still_rejects_advanced_main(reuse_media_case):
    c = reuse_media_case
    with pytest.raises(SystemExit, match='not exactly one commit'):
        publication.resume_receipt(c.root, c.receipt['branch'], c.receipt['tag'], c.receipt['readback'])
    assert not (c.root / publication.RECEIPT).exists()


@pytest.mark.parametrize('projected', [False, True])
def test_reuse_media_accepts_exact_historical_or_archived_publication_domain(reuse_media_case, projected):
    c = reuse_media_case
    lessons = [dict(id='01_test', status='ready'), dict(id='18_motility', status='ready')]
    c.dump(c.baseline / 'web/catalog/lessons_en.json', dict(lessons=lessons))
    for root in (c.baseline, c.root):
        manifest = publication.read(root / 'release-manifest.json')
        manifest.update(routes=2, ready_lessons=2)
        c.dump(root / 'release-manifest.json', manifest)
    receipt = publication.read(c.baseline / publication.RECEIPT)
    receipt['manifest_sha256'] = publication.digest(c.baseline / 'release-manifest.json')
    receipt['pages'].update(ready=1 if projected else 2, routes=1 if projected else 2)
    c.dump(c.baseline / publication.RECEIPT, receipt)
    checks = publication.read(c.baseline / 'checks/published-media-browser-checks.json')
    checks['manifest_sha256'] = receipt['manifest_sha256']
    selected = lessons[:1] if projected else lessons
    checks['ready_playback_cases'] = [dict(lesson=row['id'], passed=True) for row in selected]
    c.dump(c.baseline / 'checks/published-media-browser-checks.json', checks)
    publication.reuse_media_receipt(c.root, c.baseline)
    assert publication.read(c.root / publication.RECEIPT)['readback'] == receipt['readback']


def test_reuse_media_rejects_arbitrary_subset_even_with_consistent_small_counts(reuse_media_case):
    c = reuse_media_case
    receipt = publication.read(c.baseline / publication.RECEIPT)
    receipt['pages'].update(ready=0, routes=0)
    c.dump(c.baseline / publication.RECEIPT, receipt)
    checks = publication.read(c.baseline / 'checks/published-media-browser-checks.json')
    checks['ready_playback_cases'] = []
    c.dump(c.baseline / 'checks/published-media-browser-checks.json', checks)
    with pytest.raises(SystemExit, match='incomplete'):
        publication.reuse_media_receipt(c.root, c.baseline)
    assert c.queries == []
