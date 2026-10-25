"""The Overview Measurements selection and its Advanced image operations."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.measure import (
    _IMAGE_FEATURE_GROUPS,
    _image_advanced_features,
    _image_advanced_options,
    _image_advanced_prepare,
    _image_embedding_components,
    _image_embedding_text,
    _image_embedding_vector,
    _image_feature_groups,
    _image_level_features,
)
from spacr.settings import _overview_measurement_declarations
from spacr.settings_spec import convert_settings_dict_for_gui


def _planes(seed=0, size=96):
    rng = np.random.default_rng(seed)
    first = rng.poisson(20, (size, size)).astype(np.uint16)
    first[20:50, 30:60] += 400
    second = rng.poisson(30, (size, size)).astype(np.uint16)
    second[25:55, 35:65] += 300
    return {0: first, 1: second}


_GROUP_MARKERS = {
    'intensity': 'channel_0_intensity_mean',
    'saturation': 'channel_0_saturation_fraction',
    'focus': 'channel_0_focus_laplacian_variance',
    'texture': 'channel_0_texture_glcm_contrast',
    'coverage': 'channel_0_otsu_area_fraction',
    'background': 'channel_0_snr',
    'objects': 'channel_0_objects_count',
    'cross_channel': 'channels_0_1_pearson',
}


@pytest.mark.parametrize('group', _IMAGE_FEATURE_GROUPS)
def test_each_group_alone_produces_only_its_own_columns(group):
    record = _image_level_features(_planes(), groups=(group,), confluency=False)
    assert _GROUP_MARKERS[group] in record
    for other, marker in _GROUP_MARKERS.items():
        if other != group:
            assert marker not in record, (group, marker)


def test_settings_switches_choose_the_groups_and_defaults_select_all():
    defaults = _overview_measurement_declarations()['defaults']
    assert _image_feature_groups(defaults) == _IMAGE_FEATURE_GROUPS
    assert _image_feature_groups(None) == _IMAGE_FEATURE_GROUPS
    chosen = dict(defaults, overview_measure_texture=False, overview_measure_cross_channel=False)
    record = _image_level_features(_planes(), groups=chosen, confluency=False)
    assert not any('texture_glcm' in key for key in record)
    assert not any(key.startswith('channels_') for key in record)
    assert 'channel_1_focus_brenner' in record
    full = _image_level_features(_planes(), confluency=False)
    assert all(full[key] == value or (np.isnan(value) and np.isnan(full[key]))
               for key, value in record.items())
    with pytest.raises(ValueError, match='Unknown'):
        _image_feature_groups(['sharpness'])


def test_declarations_are_complete_typed_and_render_as_settings_rows():
    declared = _overview_measurement_declarations()
    keys = set(declared['defaults'])
    assert keys == set(declared['expected_types']) == set(declared['tooltips'])
    assert sorted(sum(declared['categories'].values(), [])) == sorted(keys)
    assert declared['parents'] == {'Advanced Measurements': 'Measurements'}
    assert {name.upper() for name in declared['categories']} == set(declared['category_tooltips'])
    advanced = declared['categories']['Advanced Measurements']
    assert all(declared['defaults'][key] is False for key in advanced
               if declared['expected_types'][key] is bool)
    for key, text in declared['tooltips'].items():
        assert text.startswith('(') and 'Default' in text, key
    spec = convert_settings_dict_for_gui(declared['defaults'])
    assert spec['overview_measure_focus'][0] == 'check'
    assert spec['overview_measure_n_jobs'][0] == 'entry'
    assert spec['overview_embedding_backbone'] == ('combo', ['resnet18', 'openphenom', 'chada_vit'], 'resnet18')


def test_advanced_options_are_off_by_default():
    options = _image_advanced_options(_overview_measurement_declarations()['defaults'])
    assert not (options['embedding'] or options['deep_qc'] or options['filter_bank'])
    assert _image_advanced_features(_planes(), dict(options, unavailable={})) == {}


def test_filter_bank_responds_to_texture():
    rng = np.random.default_rng(1)
    flat = np.full((128, 128), 100, np.uint16)
    stripes = (100 + 80 * (np.indices((128, 128))[1] // 2 % 2)).astype(np.uint16)
    noisy = rng.normal(100, 30, (128, 128)).clip(0).astype(np.uint16)
    options = dict(filter_bank=True, unavailable={})
    record = _image_advanced_features({0: flat, 1: stripes, 2: noisy}, options)
    assert record['filter_bank_status'] == 'ok'
    assert record['channel_0_filter_gabor_f0p25_mean'] == pytest.approx(0, abs=1e-6)
    assert record['channel_1_filter_gabor_f0p25_mean'] > record['channel_0_filter_gabor_f0p25_mean']
    assert record['channel_2_filter_log_s1_energy'] > record['channel_0_filter_log_s1_energy']


def test_embedding_vectors_are_compact_and_summarised_with_an_injected_encoder():
    def encoder(stack):
        return np.stack([stack.mean(axis=(1, 2, 3)), stack.std(axis=(1, 2, 3)),
                         stack.max(axis=(1, 2, 3))], axis=1).astype(np.float32)
    options = dict(embedding=True, embedding_backbone='injected', embedding_size=32, unavailable={})
    record = _image_advanced_features(_planes(size=200), options, encoder=encoder)
    assert record['embedding_status'] == 'ok'
    assert record['embedding_dims'] == 6
    vector = _image_embedding_vector(record['embedding_vector'])
    assert vector.shape == (6,)
    assert record['embedding_norm'] == pytest.approx(float(np.linalg.norm(vector)), rel=1e-3)
    assert len(record['embedding_vector']) < 6 * 4 * 2
    rows = {index: _image_embedding_text(np.arange(6) * (index + 1) + index ** 2) for index in range(5)}
    rows['missing'] = None
    components = _image_embedding_components(rows, count=2)
    assert set(components) == set(range(5))
    assert set(components[0]) == {'embedding_pc1', 'embedding_pc2'}
    assert _image_embedding_components({0: rows[0]}) == {}


def test_unavailable_models_are_reported_without_failing_the_image(monkeypatch):
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '0')
    options = _image_advanced_prepare(dict(embedding=True, deep_qc=True, filter_bank=False,
                                           embedding_backbone='no_such_backbone_xyz',
                                           embedding_size=64))
    assert 'embedding' in options['unavailable']
    record = _image_advanced_features(_planes(), options)
    assert record['embedding_status'].startswith('unavailable')
    if 'deep_qc' in options['unavailable']:
        assert record['deep_qc_status'].startswith('unavailable')
    else:
        assert record['deep_qc_status'] == 'ok'


def test_deep_qc_scores_each_channel_with_the_image_quality_network():
    pytest.importorskip('torch')
    from spacr.image_quality import _QC_CLASSES, _qc_network
    model = _qc_network().eval()
    record = _image_advanced_features(_planes(), dict(deep_qc=True, unavailable={}), qc_model=model)
    assert record['deep_qc_status'] == 'ok'
    for name in _QC_CLASSES:
        values = [record[f'channel_{channel}_deep_qc_p_{name}'] for channel in (0, 1)]
        assert all(0 <= value <= 1 for value in values)
        assert record[f'deep_qc_max_{name}'] == pytest.approx(max(values))


def test_cached_deep_qc_model_scores_synthetic_images(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '0')
    options = _image_advanced_prepare(dict(deep_qc=True, embedding=False, filter_bank=False))
    if options['deep_qc_model'] is None:
        pytest.skip('image-quality classifier is not in the local model cache')
    record = _image_advanced_features(_planes(), options)
    assert record['deep_qc_status'] == 'ok'


def test_cached_resnet18_embeds_synthetic_images(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('timm')
    from spacr.embeddings import _weights_on_disk
    if not _weights_on_disk('resnet18')[0]:
        pytest.skip('resnet18 weights are not in the local cache')
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '0')
    options = _image_advanced_prepare(dict(embedding=True, deep_qc=False, filter_bank=False,
                                           embedding_backbone='resnet18', embedding_size=64))
    record = _image_advanced_features(_planes(), options)
    assert record['embedding_status'] == 'ok'
    assert record['embedding_dims'] == 2 * 512


def test_deep_qc_model_is_acquired_once_through_the_model_cache(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from spacr import image_quality, measure
    cached = tmp_path / image_quality._builtin_qc_model_path().name
    monkeypatch.setattr(image_quality, '_builtin_qc_model_path', lambda: cached)
    monkeypatch.delenv('SPACR_MODEL_DOWNLOAD', raising=False)
    monkeypatch.delenv('SPACR_NO_NETWORK', raising=False)
    offline = _image_advanced_prepare(dict(deep_qc=True), download=False)
    assert offline['deep_qc_model'] is None and 'deep_qc' in offline['unavailable']
    calls = []

    def acquire():
        calls.append(cached)
        image_quality._save_qc_model(image_quality._qc_network(), cached, 'test')
        return cached

    monkeypatch.setattr(image_quality, '_acquire_builtin_qc_model', acquire)
    monkeypatch.setattr(measure, '_IMAGE_ADVANCED_CACHE', {})
    options = _image_advanced_prepare(dict(deep_qc=True))
    assert options['deep_qc_model'] == str(cached) and calls == [cached]
    record = _image_advanced_features(_planes(), options)
    assert record['deep_qc_status'] == 'ok'
    model = measure._IMAGE_ADVANCED_CACHE[('deep_qc', str(cached))]
    assert model.qc_cutoffs == image_quality._QC_BUILTIN_CUTOFFS
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '0')
    assert _image_advanced_prepare(dict(deep_qc=True))['deep_qc_model'] == str(cached)


def test_embedding_weights_are_fetched_once_only_when_downloads_are_allowed(monkeypatch):
    pytest.importorskip('torch')
    from spacr import embeddings, measure
    built = []

    def encoder(stack):
        return stack.mean(axis=(1, 2))

    def backbone_encoder(spec):
        built.append(spec.backbone)
        return encoder

    monkeypatch.setattr(embeddings, '_backbone_encoder', backbone_encoder)
    monkeypatch.setattr(measure, '_IMAGE_ADVANCED_CACHE', {})
    monkeypatch.delenv('SPACR_MODEL_DOWNLOAD', raising=False)
    monkeypatch.delenv('SPACR_NO_NETWORK', raising=False)
    monkeypatch.setattr(embeddings, '_weights_on_disk', lambda backbone, **_: ('', '', 0))
    options = _image_advanced_prepare(dict(embedding=True, embedding_backbone='openphenom',
                                           embedding_size=32))
    assert not options['unavailable'] and built == ['openphenom']
    timm_choice = _image_advanced_prepare(dict(embedding=True, embedding_backbone='resnet18',
                                               embedding_size=32))
    assert not timm_choice['unavailable'] or 'timm' in timm_choice['unavailable']['embedding']
    record = _image_advanced_features(_planes(), options)
    assert record['embedding_status'] == 'ok' and record['embedding_dims'] == 6
    assert built.count('openphenom') == 1
    monkeypatch.setattr(embeddings, '_weights_on_disk', lambda backbone, **_: ('/cached', 'x', 1))
    assert not _image_advanced_prepare(dict(embedding=True, embedding_backbone='chada_vit',
                                            embedding_size=32))['unavailable']
    assert 'chada_vit' not in built
    monkeypatch.setattr(embeddings, '_weights_on_disk', lambda backbone, **_: ('', '', 0))
    monkeypatch.setenv('SPACR_NO_NETWORK', '1')
    refused = _image_advanced_prepare(dict(embedding=True, embedding_backbone='chada_vit',
                                           embedding_size=32))
    assert 'not downloaded' in refused['unavailable']['embedding']


def test_a_failing_advanced_operation_is_recorded_and_the_others_still_run(monkeypatch):
    from spacr import measure

    def broken(*args, **kwargs):
        raise RuntimeError('bank failed')

    def broken_model(*args):
        raise RuntimeError('model failed')

    def broken_encoder(stack):
        raise RuntimeError('encoder failed')

    monkeypatch.setattr(measure, '_image_filter_bank', broken)
    options = dict(filter_bank=True, deep_qc=True, embedding=True, embedding_size=32, unavailable={})
    record = _image_advanced_features(_planes(), options, qc_model=broken_model, encoder=broken_encoder)
    assert record['filter_bank_status'] == 'error: bank failed'
    assert record['deep_qc_status'].startswith('error')
    assert record['embedding_status'] == 'error: encoder failed'


def test_embedding_components_use_the_common_vector_length():
    assert _image_embedding_vector('not an embedding') is None
    rows = {0: np.ones(4), 1: np.arange(4.0), 2: np.zeros(6), 3: np.arange(6.0)}
    assert _image_embedding_components(rows) == {}
    rows[4] = np.arange(4.0) ** 2
    assert set(_image_embedding_components(rows)) == {0, 1, 4}


def test_a_user_classifier_is_loaded_once_without_builtin_cutoffs(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from spacr import image_quality, measure
    path = tmp_path / 'image_qc_model.pt'
    image_quality._save_qc_model(image_quality._qc_network(), path, 'test')
    monkeypatch.setattr(measure, '_IMAGE_ADVANCED_CACHE', {})
    model = measure._image_qc_model(path)
    assert not getattr(model, 'qc_cutoffs', None)
    assert measure._image_qc_model(path) is model
