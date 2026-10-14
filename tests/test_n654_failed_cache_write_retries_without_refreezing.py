"""A recovered cache store receives the same already generated lock texts."""
import pytest

from spacr import run_journal as journal


@pytest.mark.parametrize('blocked_name', [journal._PIP_LOCK_NAME, journal._CONDA_LOCK_NAME])
def test_recovered_cache_write_reuses_generated_texts_and_keeps_blocking_bytes(tmp_path, monkeypatch, blocked_name):
    root = tmp_path / 'runs'
    monkeypatch.setattr(journal, 'runs_root', lambda: root)
    monkeypatch.setattr(journal, '_ENV_LOCK_MEMO', {})
    packages = {'pkg': '2'}
    digest = journal._environment_digest(packages)
    store = root.parent / 'env_locks' / digest[:32]
    blocked = store / blocked_name
    blocked.mkdir(parents=True)
    (blocked / 'keep.txt').write_bytes(b'preserve the pre-existing directory')
    texts = {'pip': 'pkg==2\n', 'conda': '@EXPLICIT\nhttps://example.invalid/pkg-2.conda\n'}
    names = {'pip': journal._PIP_LOCK_NAME, 'conda': journal._CONDA_LOCK_NAME}
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
    for attempt in range(2):
        folder = tmp_path / f'run-{attempt}'
        record = journal._write_environment_lock(folder, packages)
        for key in names:
            assert (folder / record[key]).read_text() == texts[key]
        assert (blocked / 'keep.txt').read_bytes() == b'preserve the pre-existing directory'
        assert calls == ['pip', 'conda']
    preserved = store / 'preserved-blocker'
    blocked.rename(preserved)
    folder = tmp_path / 'recovered-run'
    record = journal._write_environment_lock(folder, packages)
    for key, name in names.items():
        assert (folder / record[key]).read_text() == texts[key]
        assert (store / name).read_text() == texts[key]
    assert (preserved / 'keep.txt').read_bytes() == b'preserve the pre-existing directory'
    assert calls == ['pip', 'conda']
    monkeypatch.setattr(journal, '_ENV_LOCK_MEMO', {})
    again = journal._write_environment_lock(tmp_path / 'next-process', packages)
    assert again == record
    assert calls == ['pip', 'conda']
