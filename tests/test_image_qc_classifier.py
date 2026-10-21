"""The learned image-quality classifier flags defects and feeds exclusion."""
import json

import numpy as np
import pandas as pd
import pytest

from spacr import image_quality as iq
from spacr.image_quality import excluded_fields, quality_policy, screen_fields


def test_policy_validates_the_classifier_settings():
    policy = quality_policy(dict(image_qc_classifier=True, image_qc_classifier_model='  ',
                                 image_qc_classifier_threshold=0.7))
    assert policy['image_qc_classifier'] is True
    assert policy['image_qc_classifier_model'] is None
    assert policy['image_qc_classifier_threshold'] == 0.7
    for bad in (dict(image_qc_classifier_threshold=1.0), dict(image_qc_classifier='yes'),
                dict(image_qc_classifier_labels=3)):
        with pytest.raises(ValueError):
            quality_policy(bad)


def test_labels_accept_good_several_classes_and_aliases():
    assert iq._parse_qc_labels('good') == set()
    assert iq._parse_qc_labels('Blur; saturated') == {'out_of_focus', 'saturated'}
    assert iq._parse_qc_labels('bubble,empty') == {'bubble', 'empty'}
    with pytest.raises(ValueError, match='Unknown image-quality label'):
        iq._parse_qc_labels('smudge')


def test_inputs_keep_noise_near_zero_and_mark_saturation():
    rng = np.random.default_rng(0)
    empty = iq._synthetic_qc_field(rng, ('empty',), 128, 4095)
    saturated = iq._synthetic_qc_field(rng, ('saturated',), 128, 4095)
    tiles, thumbnail = iq._qc_inputs(empty, 4095)
    assert tiles.shape == (16, 3, 64, 64) and thumbnail.shape == (3, 64, 64)
    assert abs(float(thumbnail[0].mean())) < 0.05
    assert float(thumbnail[2].sum()) == 0
    assert float(iq._qc_inputs(saturated, 4095)[1][2].mean()) > 0.005
    small, _ = iq._qc_inputs(np.ones((20, 30), np.uint16), None)
    assert small.shape == (16, 3, 64, 64)


def test_a_saved_model_reloads_as_tensors_and_predicts(tmp_path):
    model = iq._train_builtin_qc_model(fields=12, epochs=1)
    path = iq._save_qc_model(model, tmp_path / 'm.pt', 'test')
    again = iq._load_qc_model(path)
    rng = np.random.default_rng(3)
    tiles, thumbnail = iq._qc_inputs(iq._synthetic_qc_field(rng, (), 128), 65535)
    first = iq._predict_qc(model, tiles[None], thumbnail[None])
    np.testing.assert_allclose(first, iq._predict_qc(again, tiles[None], thumbnail[None]), rtol=1e-5)
    assert first.shape == (1, 5) and ((first >= 0) & (first <= 1)).all()
    import torch
    torch.save({'version': 1, 'classes': ['x'], 'state': {}}, tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='not an image-quality classifier'):
        iq._load_qc_model(tmp_path / 'other.pt')


def _stub_predictions(monkeypatch):
    """Call a field empty when its normalised intensity is near zero."""
    monkeypatch.setattr(iq, '_base_qc_model', lambda policy: iq._qc_network())

    def predict(model, tiles, thumbnails):
        signal = np.asarray(thumbnails)[:, 0].mean(axis=(1, 2))
        out = np.full((len(signal), len(iq._QC_CLASSES)), 0.1, np.float32)
        out[:, iq._QC_CLASSES.index('empty')] = np.where(signal < 0.01, 0.9, 0.1)
        return out
    monkeypatch.setattr(iq, '_predict_qc', predict)


def test_classifier_flags_exclude_fields_without_touching_pixels(tmp_path, monkeypatch):
    _stub_predictions(monkeypatch)
    rng = np.random.default_rng(1)
    stack = tmp_path / 'stack'
    stack.mkdir()
    np.save(stack / 'cells.npy', iq._synthetic_qc_field(rng, (), 128)[..., None])
    np.save(stack / 'blank.npy', iq._synthetic_qc_field(rng, ('empty',), 128)[..., None])
    before = {path.name: path.read_bytes() for path in stack.glob('*.npy')}
    policy = dict(image_qc_mode='exclude', image_qc_classifier=True)
    assert screen_fields(tmp_path, policy) == ['blank.npy']
    assert excluded_fields(tmp_path) == {'blank.npy'}
    report = json.loads((tmp_path / 'qc/image_quality.json').read_text(encoding="utf-8"))
    blank = next(row for row in report['fields'] if row['field'] == 'blank.npy')
    assert blank['reasons'] == ['channel 0: classifier_empty']
    assert blank['channels'][0]['p_empty'] == pytest.approx(0.9)
    table = pd.read_csv(tmp_path / 'qc/image_quality.csv')
    assert {f'p_{name}' for name in iq._QC_CLASSES} <= set(table.columns)
    assert all((stack / name).read_bytes() == data for name, data in before.items())
    assert screen_fields(tmp_path, dict(policy, image_qc_mode='report')) == []


def test_labelled_fields_are_benchmarked_and_fine_tune_a_saved_model(tmp_path, monkeypatch):
    monkeypatch.setattr(iq, '_base_qc_model', lambda policy: iq._qc_network())
    monkeypatch.setattr(iq, '_QC_FINE_TUNE_EPOCHS', 2)
    rng = np.random.default_rng(2)
    stack = tmp_path / 'stack'
    stack.mkdir()
    rows = []
    for index in range(12):
        labels = () if index % 3 else ('out_of_focus',)
        np.save(stack / f'f{index}.npy', iq._synthetic_qc_field(rng, labels, 96))
        rows.append(dict(field=f'f{index}', label=';'.join(labels) or 'good'))
    pd.DataFrame(rows).to_csv(tmp_path / 'labels.csv', index=False)
    policy = dict(image_qc_mode='report', image_qc_classifier=True,
                  image_qc_classifier_labels=str(tmp_path / 'labels.csv'),
                  image_qc_min_focus={0: 1.0})
    assert screen_fields(tmp_path, policy) == []
    benchmark = pd.read_csv(tmp_path / 'qc/image_qc_benchmark.csv')
    assert set(benchmark['method']) == {'classifier', 'rule_saved_policy', 'rule_tuned'}
    assert set(benchmark['defect']) == {'any_defect', *iq._QC_CLASSES}
    assert (benchmark['fields'] == 12).all()
    assert benchmark.set_index(['method', 'defect']).loc[('classifier', 'out_of_focus'), 'positives'] == 4
    assert iq._load_qc_model(tmp_path / 'qc/image_qc_model.pt') is not None


def test_tuned_rule_separates_blur_by_focus():
    focus = np.array([1., 2., 3., 50., 60., 70.])
    saturation = np.zeros(6)
    low, high = iq._tuned_rule(focus, saturation, focus < 10)
    assert 3 < low <= 50 and high == np.inf


def test_builtin_cutoffs_move_to_the_threshold_and_fine_tuning_drops_them(
        tmp_path, monkeypatch):
    raw = np.array([[0.89, 0.5, 0.94, 0.76, 0.2],
                    [0.91, 0.6, 0.96, 0.70, 0.5]], np.float32)
    plain = iq._qc_network()
    assert iq._calibrated_qc(plain, raw) is raw
    monkeypatch.setattr(iq, '_acquire_builtin_qc_model', lambda: tmp_path / 'm.pt')
    iq._save_qc_model(iq._qc_network(), tmp_path / 'm.pt', 'test')
    builtin = iq._base_qc_model(iq.quality_policy({'image_qc_classifier': True}))
    assert builtin.qc_cutoffs == iq._QC_BUILTIN_CUTOFFS
    called = iq._calibrated_qc(builtin, raw) >= 0.5
    np.testing.assert_array_equal(called, [[False, True, False, True, False],
                                           [True, True, True, False, True]])
    at_cut = np.array([[iq._QC_BUILTIN_CUTOFFS[c] for c in iq._QC_CLASSES]],
                      np.float32)
    np.testing.assert_allclose(iq._calibrated_qc(builtin, at_cut), 0.5, atol=1e-5)


def test_explicit_checkpoint_skips_builtin_download(tmp_path, monkeypatch):
    path = iq._save_qc_model(iq._qc_network(), tmp_path / 'custom.pt', 'user')
    monkeypatch.setattr(iq, '_acquire_builtin_qc_model', lambda: pytest.fail('built-in selected'))
    model = iq._base_qc_model(iq.quality_policy({'image_qc_classifier_model': str(path)}))
    assert not hasattr(model, 'qc_cutoffs')


@pytest.mark.parametrize('version,classes', [(2, iq._QC_CLASSES), (1, iq._QC_CLASSES[::-1])])
def test_checkpoint_refuses_version_and_class_order(tmp_path, version, classes):
    import torch

    path = tmp_path / 'wrong.pt'
    torch.save(dict(version=version, classes=list(classes), state={}), path)
    with pytest.raises(ValueError, match='not an image-quality classifier'):
        iq._load_qc_model(path)


def test_model_save_preserves_destination_and_cleans_partial_on_failure(tmp_path, monkeypatch):
    import torch

    path = tmp_path / 'model.pt'
    path.write_bytes(b'previous checkpoint')

    def fail(saved, temporary):
        temporary.write_bytes(b'partial')
        raise OSError('disk full')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        iq._save_qc_model(iq._qc_network(), path, 'test')
    assert path.read_bytes() == b'previous checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def _pinned_fixture(tmp_path, monkeypatch):
    import hashlib
    from spacr import model_zoo as zoo

    path = iq._save_qc_model(iq._qc_network(), tmp_path / 'fixture.pt', 'fixture')
    payload = path.read_bytes()
    monkeypatch.setattr(iq, '_QC_WEIGHTS_SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(tmp_path / 'cache'))
    monkeypatch.delenv('SPACR_MODEL_DOWNLOAD', raising=False)
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    return zoo, payload


def test_pinned_default_download_is_immutable_verified_and_cached_offline(tmp_path, monkeypatch):
    zoo, payload = _pinned_fixture(tmp_path, monkeypatch)
    seen = []

    def stream(uri, **kwargs):
        seen.append(uri)
        return [payload], len(payload)

    monkeypatch.setattr(zoo, 'open_uri', stream)
    model = iq._base_qc_model(iq.quality_policy({}))
    assert model.qc_cutoffs == iq._QC_BUILTIN_CUTOFFS
    assert seen == [('https://huggingface.co/datasets/einarolafsson/models/resolve/'
                     f'{iq._QC_WEIGHTS_REVISION}/{iq._QC_WEIGHTS_FILE}')]
    cached = iq._builtin_qc_model_path()
    assert cached.read_bytes() == payload
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '0')
    assert iq._acquire_builtin_qc_model() == cached
    assert len(seen) == 1
    cached.write_bytes(b'corrupt')
    with pytest.raises(zoo.ModelUnreadable, match='disabled'):
        iq._acquire_builtin_qc_model()


def test_pinned_default_rejects_wrong_hash_without_legacy_cache_fallback(tmp_path, monkeypatch):
    zoo, payload = _pinned_fixture(tmp_path, monkeypatch)
    cached = iq._builtin_qc_model_path()
    cached.parent.mkdir()
    legacy = cached.parent / 'image_qc_classifier_v1.pt'
    legacy.write_bytes(payload)
    monkeypatch.setattr(zoo, 'open_uri', lambda uri, **kwargs: ([b'wrong'], 5))
    with pytest.raises(zoo.ChecksumMismatch):
        iq._acquire_builtin_qc_model()
    assert not cached.exists()
    assert legacy.read_bytes() == payload
    assert not list(cached.parent.glob('*.part'))


def test_concurrent_pinned_downloads_install_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    zoo, payload = _pinned_fixture(tmp_path, monkeypatch)
    arrivals = threading.Barrier(2)
    seen = []

    def stream(uri, **kwargs):
        seen.append(uri)
        return [payload], len(payload)

    def acquire():
        arrivals.wait(timeout=10)
        return iq._acquire_builtin_qc_model()

    monkeypatch.setattr(zoo, 'open_uri', stream)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: acquire(), range(2)))
    assert first == second == iq._builtin_qc_model_path()
    assert first.read_bytes() == payload
    assert len(seen) == 1
