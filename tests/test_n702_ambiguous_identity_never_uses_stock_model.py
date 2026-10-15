"""An ambiguous catalogue identity cannot silently select stock segmentation."""
from dataclasses import replace

import pytest

from spacr import model_zoo as zoo


@pytest.fixture
def ambiguous_models(monkeypatch):
    first = zoo.ModelEntry(key='cells_r7', name='shared_weights.CP_model', kind='cellpose',
                           version='7', sha256='7' * 64, uri='file:///unused-first')
    second = replace(first, key='cells_r8', version='8', sha256='8' * 64,
                     uri='file:///unused-second')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [first, second])
    monkeypatch.setattr(zoo, 'fetch', lambda *args: pytest.fail('ambiguous model fetched'))
    return first, second


def test_ambiguous_missing_laptop_path_names_every_exact_candidate(ambiguous_models):
    first, second = ambiguous_models
    requested = '/laptop/' + first.name
    with pytest.raises(zoo.ModelZooError) as error:
        zoo._ensure_model_file(requested)
    message = str(error.value)
    assert requested in message
    for entry in (first, second):
        assert entry.key in message
        assert entry.version in message
        assert entry.sha256 in message


def test_actual_cellpose_resolver_refuses_ambiguous_stock_fallback(ambiguous_models, monkeypatch):
    from spacr import utils
    first, _second = ambiguous_models
    monkeypatch.setattr(utils, '_installed_cellpose_models', lambda: ())
    with pytest.raises(zoo.ModelZooError):
        utils._resolve_cellpose_pretrained(first.name, object_type='cell')


def test_existing_local_weights_still_win_over_ambiguous_catalogue(ambiguous_models, tmp_path):
    first, _second = ambiguous_models
    local = tmp_path / first.name
    local.write_bytes(b'locally selected weights')
    assert zoo._ensure_model_file(local) == local
    assert local.read_bytes() == b'locally selected weights'


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'])
def test_ambiguity_error_uses_real_existing_locale_row(ambiguous_models, monkeypatch, language):
    import importlib
    from spacr.qt import i18n
    first, second = ambiguous_models
    caption = '{name} · {n} matches'
    translated = importlib.import_module('spacr.qt.i18n_catalogs.' + language).UI[caption]
    assert translated != caption
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    with pytest.raises(zoo.ModelZooError) as error:
        zoo._ensure_model_file(first.name)
    message = str(error.value)
    assert message.startswith(translated.format(name=repr(first.name), n=2) + ':\n')
    for entry in (first, second):
        assert f'{entry.key}@{entry.version} [sha256={entry.sha256}]' in message
