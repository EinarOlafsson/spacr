"""Immutable Space metadata must cover the whole assembled site."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import huggingface_hub
from huggingface_hub.hf_api import RepoFile, RepoFolder
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from publish_docs_channels import main, verify_space_tree


@pytest.fixture
def uploaded(tmp_path, monkeypatch):
    files = {'index.html': b'hello\n', 'README.md': b'',
             'channels.json': json.dumps({'channels': {'nightly': {'commit': 'source'}}}).encode(),
             'api/page.html': b'API body'}
    entries = []
    for name, raw in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        oid = subprocess.check_output(['git', 'hash-object', '--stdin'], input=raw).decode().strip()
        entries.append(RepoFile(path=name, size=len(raw), oid=oid))
    assert entries[0].blob_id == 'ce013625030ba8dba906f756967f9e9ca394464a'
    entries.append(RepoFolder(path='api', oid='1' * 40))
    entries.append(RepoFile(path='.gitattributes', size=1, oid='2' * 40))
    calls = []

    class Api:
        def list_repo_tree(self, space, **kwargs):
            calls.append((space, kwargs))
            return iter(entries)

    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    return tmp_path, files, entries, calls


def test_complete_git_tree_and_cli_use_exact_immutable_revision(uploaded, capsys):
    site, files, entries, calls = uploaded
    assert main(['verify-space-tree', '--site', str(site), '--space', 'owner/docs',
                 '--revision', 'a' * 40]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {'space': 'owner/docs', 'revision': 'a' * 40,
                       'source_commit': 'source', 'files_verified': len(files),
                       'bytes_verified': sum(map(len, files.values()))}
    assert calls == [('owner/docs', {'repo_type': 'space', 'revision': 'a' * 40,
                                    'recursive': True})]


def test_lfs_content_identity_is_checked_instead_of_pointer_blob(uploaded):
    site, files, entries, calls = uploaded
    raw = files['index.html']
    entries[0] = RepoFile(path='index.html', size=len(raw), oid='0' * 40,
                          lfs={'size': len(raw), 'oid': hashlib.sha256(raw).hexdigest(),
                               'pointerSize': 130})
    assert verify_space_tree(site, 'owner/docs', 'a' * 40)['files_verified'] == len(files)
    entries[0].lfs = entries[0].lfs.__class__(size=len(raw), sha256='0' * 64, pointer_size=130)
    with pytest.raises(ValueError, match='content differs: index.html'):
        verify_space_tree(site, 'owner/docs', 'a' * 40)


@pytest.mark.parametrize('change', ['missing', 'extra', 'same-size-content', 'size', 'duplicate'])
def test_mixed_or_incomplete_tree_is_refused(uploaded, change):
    site, files, entries, calls = uploaded
    if change == 'missing':
        entries.pop(3)
    elif change == 'extra':
        entries.append(RepoFile(path='obsolete.html', size=0, oid='0' * 40))
    elif change == 'same-size-content':
        entries[3].blob_id = '0' * 40
    elif change == 'size':
        entries[3].size += 1
    else:
        entries.append(entries[3])
    with pytest.raises(ValueError):
        verify_space_tree(site, 'owner/docs', 'a' * 40)


@pytest.mark.parametrize('revision', ['main', 'abc123', 'A' * 40])
def test_mutable_or_invalid_revision_is_refused_before_network(uploaded, revision):
    site, files, entries, calls = uploaded
    with pytest.raises(ValueError, match='immutable full Space commit'):
        verify_space_tree(site, 'owner/docs', revision)
    assert calls == []


def test_locally_supplied_gitattributes_must_match_too(uploaded):
    site, files, entries, calls = uploaded
    (site / '.gitattributes').write_text('different', encoding="utf-8")
    with pytest.raises(ValueError, match='Space file size differs: .gitattributes'):
        verify_space_tree(site, 'owner/docs', 'a' * 40)
