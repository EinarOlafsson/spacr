"""A broken cached environment spec cannot silently replace its lockfiles."""
import pytest

from spacr import run_journal as journal


@pytest.mark.parametrize('broken', ['pip', 'conda'])
@pytest.mark.parametrize('bad_bytes', [b'', b'\xff'])
def test_corrupt_cache_is_repaired_once_then_reused(tmp_path, monkeypatch, broken, bad_bytes):
    root = tmp_path / 'runs'
    monkeypatch.setattr(journal, 'runs_root', lambda: root)
    monkeypatch.setattr(journal, '_ENV_LOCK_MEMO', {})
    packages = {'pkg': '2'}
    digest = journal._environment_digest(packages)
    store = root.parent / 'env_locks' / digest[:32]
    store.mkdir(parents=True)
    texts = {'pip': 'pkg==2\n',
             'conda': '@EXPLICIT\nhttps://example.invalid/pkg-2.conda\n'}
    names = {'pip': journal._PIP_LOCK_NAME, 'conda': journal._CONDA_LOCK_NAME}
    for key, name in names.items():
        (store / name).write_bytes(bad_bytes if key == broken else texts[key].encode())
    calls = []

    def pip(current):
        assert current == packages
        calls.append('pip')
        return texts['pip']

    def conda():
        calls.append('conda')
        return texts['conda']

    monkeypatch.setattr(journal, '_pip_lock_text', pip)
    monkeypatch.setattr(journal, '_conda_lock_text', conda)
    for attempt in range(3):
        if attempt == 2:
            monkeypatch.setattr(journal, '_ENV_LOCK_MEMO', {})
        folder = tmp_path / f'run-{attempt}'
        record = journal._write_environment_lock(folder, packages)
        assert record['sha256'] == digest
        for key, name in names.items():
            assert record[key] == f'environment/{name}'
            assert (folder / record[key]).read_text(encoding="utf-8") == texts[key]
            assert (store / name).read_text(encoding="utf-8") == texts[key]
        assert calls == ['pip', 'conda']


def test_valid_pip_cache_without_conda_is_reused(tmp_path, monkeypatch):
    root = tmp_path / 'runs'
    monkeypatch.setattr(journal, 'runs_root', lambda: root)
    monkeypatch.setattr(journal, '_ENV_LOCK_MEMO', {})
    packages = {'pkg': '2'}
    digest = journal._environment_digest(packages)
    store = root.parent / 'env_locks' / digest[:32]
    store.mkdir(parents=True)
    (store / journal._PIP_LOCK_NAME).write_text('pkg==2\n')

    def unexpected(*args):
        raise AssertionError('a valid cache must not rerun an environment command')

    monkeypatch.setattr(journal, '_pip_lock_text', unexpected)
    monkeypatch.setattr(journal, '_conda_lock_text', unexpected)
    folder = tmp_path / 'run'
    record = journal._write_environment_lock(folder, packages)
    assert (folder / record['pip']).read_text(encoding="utf-8") == 'pkg==2\n'
    assert record['conda'] is None
