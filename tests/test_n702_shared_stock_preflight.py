"""Direct segmentation and training consumers share the stock offline boundary."""
from pathlib import Path
import json

import pytest

from spacr import model_zoo as zoo
from spacr import utils
from spacr import submodules
from tests.test_n702_unknown_cellpose_model_refusal import isolated


@pytest.fixture
def offline(isolated, monkeypatch, tmp_path):
    cache = tmp_path / 'native-stock-cache'
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
    monkeypatch.setattr(utils.cp_models, 'MODEL_DIR', cache)
    monkeypatch.setattr('cellpose.models.cache_model_path', lambda *args, **kwargs:
                        pytest.fail('Offline preflight must not call native download/cache helper'),
                        raising=False)
    monkeypatch.setattr(utils.cp_models, 'CellposeModel', lambda **kwargs:
                        pytest.fail('Uncached offline model reached constructor'))
    return cache


@pytest.mark.parametrize('requested', ['cpsam', None, 'cyto3', 'sam'])
def test_central_stock_resolution_refuses_uncached_offline(offline, requested):
    with pytest.raises(zoo.ModelUnreadable):
        utils._resolve_cellpose_pretrained(requested, object_type='cell')
    assert not offline.exists()


@pytest.mark.parametrize('route', ['robustness', 'generator'])
def test_actual_object_consumer_refuses_before_constructor(offline, tmp_path, route):
    from spacr import object as objects
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks({
        'cell_channel': 0, 'nucleus_channel': None, 'pathogen_channel': None,
        'channels': [0], 'cell_model_name': 'cpsam', 'verbose': False})
    with pytest.raises(zoo.ModelUnreadable):
        if route == 'robustness':
            objects._robustness_segmenter(settings, 'cell')
        else:
            batches = tmp_path / 'batches'
            batches.mkdir()
            objects.generate_cellpose_masks_sam(str(batches), settings, 'cell')
    assert not offline.exists()
    assert not (tmp_path / 'measurements').exists()


@pytest.mark.parametrize('requested', ['cpsam', None, 'cyto3', 'sam'])
def test_training_stock_base_refuses_uncached_offline(offline, requested):
    with pytest.raises(zoo.ModelUnreadable):
        submodules._resolve_training_base(requested)
    assert not offline.exists()


def test_training_unknown_bare_base_refuses_instead_of_using_stock(offline):
    with pytest.raises(zoo.ModelZooError) as error:
        submodules._resolve_training_base('unpublished_checkpoint_742')
    assert "'unpublished_checkpoint_742'" in str(error.value)
    assert not offline.exists()


@pytest.mark.parametrize('consumer', ['central', 'training'])
@pytest.mark.parametrize('kind', ['cached', 'relative', 'catalogue', 'online'])
def test_exact_existing_and_online_routes_are_preserved(offline, monkeypatch, tmp_path, consumer, kind):
    resolve = utils._resolve_cellpose_pretrained if consumer == 'central' else submodules._resolve_training_base
    requested = 'cpsam'
    expected = 'cpsam'
    checkpoint = None
    if kind == 'cached':
        offline.mkdir()
        checkpoint = offline / 'cpsam'
        checkpoint.write_bytes(b'private native cache; not decoded')
    elif kind == 'relative':
        checkpoint = Path('cpsam')
        checkpoint.write_bytes(b'private explicit same-name checkpoint')
    elif kind == 'catalogue':
        requested = 'approved_local_key'
        checkpoint = tmp_path / 'published.CP_model'
        checkpoint.write_bytes(b'private exact catalogue checkpoint')
        entry = zoo.ModelEntry(key=requested, name=checkpoint.name,
                               kind='cellpose', path=str(checkpoint))
        catalogue = tmp_path / 'explicit-catalogue.json'
        catalogue.write_text(json.dumps([{
            'key': entry.key, 'name': entry.name, 'kind': entry.kind,
            'path': str(checkpoint)}]))
        monkeypatch.setenv(zoo.CATALOGUE_ENV_VAR, str(catalogue))
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs:
                            pytest.fail('Offline local declaration consulted providers'))
        expected = str(checkpoint)
    else:
        monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '1')
    before = checkpoint.read_bytes() if checkpoint is not None else None
    assert resolve(requested) == expected
    if checkpoint is not None:
        assert checkpoint.read_bytes() == before
    else:
        assert not offline.exists()


def test_training_missing_explicit_path_retains_its_error(offline, tmp_path):
    missing = str(tmp_path / 'missing.CP_model')
    with pytest.raises(zoo.ModelZooError) as error:
        submodules._resolve_training_base(missing)
    assert missing in str(error.value) and 'base_model' in str(error.value)
    assert not offline.exists()
