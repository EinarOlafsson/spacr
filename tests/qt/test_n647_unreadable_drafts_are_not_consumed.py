"""Draft recovery keeps native INI bytes until they can actually be read."""
import errno
import json
import os

import pytest
from PySide6.QtCore import QSettings

from spacr import restart_state


@pytest.fixture
def native_store(tmp_path, monkeypatch):
    path = tmp_path / 'settings.ini'
    store = QSettings(str(path), QSettings.IniFormat)
    monkeypatch.setattr(restart_state, '_store', lambda: store)
    return store


@pytest.mark.parametrize('failure', ['factory', 'value'])
def test_a_transient_read_failure_keeps_drafts_for_the_next_attempt(native_store, monkeypatch, failure):
    drafts = {'regression': {'src': '/a temporary draft', 'fdr_alpha': 0.08}}
    assert restart_state._save_drafts(drafts)
    original = native_store.value(restart_state._DRAFTS_KEY)
    calls = []

    class InterruptedRead:
        def value(self, key, default):
            raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN), native_store.fileName())

    def store():
        calls.append(True)
        if len(calls) == 1:
            if failure == 'factory':
                raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN), native_store.fileName())
            return InterruptedRead()
        return native_store

    monkeypatch.setattr(restart_state, '_store', store)
    assert restart_state._take_drafts() == {}
    durable = QSettings(native_store.fileName(), QSettings.IniFormat)
    assert durable.value(restart_state._DRAFTS_KEY) == original
    assert len(calls) == 1
    assert restart_state._take_drafts() == drafts
    assert restart_state._take_drafts() == {}


@pytest.mark.parametrize('raw', ['{', '[]', json.dumps({'version': 1, 'modules': []})])
def test_unrecognized_native_draft_records_remain_available_for_recovery(native_store, raw):
    native_store.setValue(restart_state._DRAFTS_KEY, raw)
    native_store.sync()
    assert native_store.status() == QSettings.NoError
    assert restart_state._take_drafts() == {}
    durable = QSettings(native_store.fileName(), QSettings.IniFormat)
    assert durable.value(restart_state._DRAFTS_KEY) == raw
