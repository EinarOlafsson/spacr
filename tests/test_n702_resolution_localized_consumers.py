"""Real refusal and native acquisition consumers use installed reviewed messages."""
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pytest

from spacr import model_zoo as zoo
from spacr import utils

from tests import locale_cases

LANGUAGES = ('sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is')
ROOT = Path(__file__).parents[1]


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    payload = b'controlled localized publisher checkpoint'
    source = tmp_path / 'publisher.CP_model'
    source.write_bytes(payload)
    raw = {'key': 'localized_cells', 'name': source.name, 'kind': 'cellpose',
           'version': '001', 'sha256': hashlib.sha256(payload).hexdigest(),
           'uri': source.as_uri()}
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'on')
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'node')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw,))
    monkeypatch.setattr(zoo, '_SHARED_CATALOGUE_CACHE', {'fetched_at': 0.0, 'entries': ()})
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [zoo._entry_from_mapping(raw)])
    original = urllib.request.urlopen

    def local_only(url, *args, **kwargs):
        value = getattr(url, 'full_url', str(url))
        assert value.startswith('file:'), value
        return original(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, 'urlopen', local_only)
    return raw, source, payload


def records(language):
    path = ROOT / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-n702-resolution-context.json'
    return json.loads(path.read_text(encoding="utf-8"))['records']


def _actual_unknown_request_uses_localized_context_case(publisher, monkeypatch, language):
    """One case of :func:`test_actual_unknown_request_uses_localized_context`."""
    raw, source, payload = publisher
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    monkeypatch.setattr(utils, '_installed_cellpose_models', lambda: ('cpsam',))
    monkeypatch.setattr(utils, 'cp_models', SimpleNamespace(
        CellposeModel=lambda **kwargs: pytest.fail('constructor reached')))
    monkeypatch.setattr(zoo, 'fetch', lambda *args, **kwargs: pytest.fail('unknown request fetched'))
    requested = 'unknown-localized-model.CP_model'
    row, = [r for r in records(language) if r['source'].startswith('Model setting ')]
    candidate = f"{raw['key']}@{raw['version']!r} [sha256={raw['sha256']}]"
    expected = row['translation'].format(setting='cell_model_name', requested=repr(requested), candidates=candidate)
    prior_path = ROOT / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-cellpose-unknown-model-refusal.json'
    prior, = json.loads(prior_path.read_text(encoding="utf-8"))['records']
    full_expected = prior['translation'].format(name=repr(requested)) + '\n' + expected
    with pytest.raises(zoo.ModelZooError) as error:
        utils._choose_model(requested, device='cpu', object_type='cell')
    assert str(error.value) == full_expected
    assert expected != row['source'].format(setting='cell_model_name', requested=repr(requested), candidates=candidate)
    assert source.read_bytes() == payload


def test_actual_unknown_request_uses_localized_context(subtests, publisher, monkeypatch):
    for label, case in locale_cases.cases(('language', LANGUAGES)):
        with subtests.test(**label), locale_cases.monkeypatch_scope() as case_monkeypatch:
            _actual_unknown_request_uses_localized_context_case(publisher=publisher, monkeypatch=case_monkeypatch, **case)


def _actual_native_download_logs_localized_context_case(publisher, monkeypatch, language, caplog):
    """One case of :func:`test_actual_native_download_logs_localized_context`."""
    raw, source, payload = publisher
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    requested = '/old-laptop/' + source.name
    row, = [r for r in records(language) if r['source'].startswith('{setting}:')]
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        result = zoo._ensure_model_file(requested, kinds=('cellpose',), setting='plaque_model')
    expected = row['translation'].format(setting='plaque_model', requested=requested, key=raw['key'],
                                         version=repr(raw['version']), sha256=raw['sha256'], path=str(result))
    assert [record.getMessage() for record in caplog.records].count(expected) == 1
    assert result.read_bytes() == source.read_bytes() == payload
    assert result != source


def test_actual_native_download_logs_localized_context(subtests, publisher, monkeypatch, caplog, tmp_path):
    for label, case in locale_cases.cases(('language', LANGUAGES)):
        with subtests.test(**label), locale_cases.monkeypatch_scope() as case_monkeypatch:
            # every case downloads into an empty model home of its own, as a
            # separate test did; a model a previous case fetched is not logged
            case_home = locale_cases.case_dir(tmp_path, label)
            case_monkeypatch.setattr(zoo, '_spacr_home', lambda: case_home)
            caplog.clear()
            _actual_native_download_logs_localized_context_case(publisher=publisher, monkeypatch=case_monkeypatch, caplog=caplog, **case)


@pytest.mark.parametrize('language', ('en', 'ko'))
def test_actual_train_model_help_retains_precise_failure_contract(language, api_catalogs):
    from spacr.qt.help_index import build_index, search
    from spacr.qt.help_search import api_url, local_docstring

    symbol = 'spacr.deep_spacr.train_model'
    doc = local_docstring(symbol)
    assert 'custom_model_path' in doc and 'resume_checkpoint' in doc
    assert 'exact compatible model-zoo identity' in doc
    rows = search(build_index(), 'train_model')
    assert rows and rows[0].kind == 'api' and rows[0].title == symbol
    url = api_url(symbol, language)
    assert url.endswith('#' + symbol)
    if language == 'ko':
        assert '?lang=ko' in url
    else:
        assert '?lang=ko' not in url
    payload = json.loads((api_catalogs / (language + '.json')).read_text(encoding="utf-8"))
    assert 'custom_model_path' in payload['symbols'][symbol]['text']
    assert 'resume_checkpoint' in payload['symbols'][symbol]['text']
