"""Own-DINO classifier software contract with tiny synthetic encoders only."""
import hashlib
import sys
import types

import pytest
import torch
from torch import nn

from spacr import embeddings as emb
from spacr import torch_artifacts as artifacts


class TinyEncoder(nn.Module):
    num_features = 3

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 3, 1)
        self.norm = nn.BatchNorm2d(3)
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return self.norm(self.conv(x)).mean((2, 3))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    calls = []

    def create_model(arch, **kwargs):
        calls.append((arch, kwargs))
        assert kwargs['pretrained'] is False
        assert arch == 'tiny_local'
        return TinyEncoder(kwargs['in_chans'])

    monkeypatch.setitem(sys.modules, 'timm', types.SimpleNamespace(
        list_models=lambda: ['tiny_local'], create_model=create_model))

    def make(policy='project', scale=(10., 20.), channels=2):
        source = TinyEncoder(channels)
        path = tmp_path / f'{policy}.pt'
        torch.save({'setup': {'arch': 'tiny_local', 'in_chans': channels,
                              'size': 4, 'channel_policy': policy},
                    'scale': scale, 'teacher': source.state_dict()}, path)
        return path, source, calls

    return make


@pytest.mark.parametrize('freeze', [True, False])
@pytest.mark.parametrize('policy', ['project', 'per_channel'])
def test_raw_planes_gradients_and_reload(checkpoint, tmp_path, freeze, policy):
    path, teacher, calls = checkpoint(policy, channels=2 if policy == 'project' else 1)
    original = path.read_bytes()
    model = emb._dino_classifier_from_checkpoint(
        path, num_classes=2, source_channels=[3, 0], freeze_encoder=freeze)
    assert model._own_dino_config['source_sha256'] == hashlib.sha256(original).hexdigest()
    images = torch.arange(2*2*3*5, dtype=torch.float32).reshape(2, 2, 3, 5)
    before = images.clone()
    model.train()
    assert model.encoder.training is (not freeze)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    initial = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    loss = nn.CrossEntropyLoss()(model(images), torch.tensor([0, 1]))
    loss.backward()
    assert model.head.weight.grad is not None
    assert all((p.grad is None) is freeze for p in model.encoder.parameters())
    optimizer.step()
    assert torch.equal(images, before)
    if freeze:
        assert all(torch.equal(v, initial[k]) for k, v in model.encoder.state_dict().items())
    else:
        assert any(not torch.equal(v, initial[k]) for k, v in model.encoder.state_dict().items())
    scaled = (images / torch.tensor([10., 20.]).reshape(1, 2, 1, 1)).clamp(0, 1)
    if policy == 'per_channel':
        scaled = scaled.reshape(4, 1, 3, 5)
    expected = nn.functional.interpolate(scaled, size=(4, 4), mode='bilinear', align_corners=False)
    assert torch.equal(model.encoder.seen, expected)
    model.eval()
    expected_logits = model(images).detach()
    artifact_path = tmp_path / 'classifier.pt'
    artifacts.atomic_torch_save(artifacts.make_model_artifact(
        model, include_rng=False, classes=['a', 'b'], channels=['3', '0']), artifact_path)
    restored, metadata = artifacts.load_model_artifact(artifact_path, weights_only=True)
    restored.eval()
    assert torch.equal(restored(images), expected_logits)
    assert metadata['legacy'] is False
    assert restored._own_dino_config == model._own_dino_config
    assert path.read_bytes() == original
    assert len(calls) == 2


@pytest.mark.parametrize('change', [
    {'num_classes': True}, {'source_channels': [0, 0]},
    {'source_channels': [0]}, {'freeze_encoder': 'yes'},
])
def test_invalid_constructor_refuses_before_backbone(checkpoint, change):
    path, _, calls = checkpoint()
    kwargs = dict(num_classes=2, source_channels=[0, 1])
    kwargs.update(change)
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_checkpoint(path, **kwargs)
    assert calls == []


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_bad_scale_refuses(checkpoint, bad):
    path, _, calls = checkpoint(scale=(10., bad))
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    assert calls == []


def test_blank_scale_and_invalid_inputs(checkpoint):
    path, _, _ = checkpoint(scale=(0., 20.))
    model = emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    model(torch.ones(1, 2, 4, 4))
    assert torch.count_nonzero(model.encoder.seen[:, 0]) == 0
    for images in (torch.ones(1, 3, 4, 4), torch.full((1, 2, 4, 4), float('inf')),
                   torch.ones(1, 2, 4, 4, dtype=torch.bool), torch.ones(1, 2, 4)):
        with pytest.raises(emb.EmbeddingError):
            model(images)


def test_unknown_arch_never_constructed(checkpoint):
    path, _, calls = checkpoint()
    payload = torch.load(path, weights_only=True)
    payload['setup']['arch'] = 'hf-hub:remote/model'
    torch.save(payload, path)
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    assert not calls


def test_unsafe_checkpoint_has_no_retry(checkpoint, tmp_path):
    _, _, calls = checkpoint()
    path = tmp_path / 'unsafe.pt'
    torch.save(TinyEncoder(2), path)
    with pytest.raises(Exception, match='Weights only load failed'):
        emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    assert not calls


def test_reconstruction_rejects_scale_or_config_changes(checkpoint, tmp_path):
    path, _, _ = checkpoint()
    model = emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    payload = artifacts.make_model_artifact(model, include_rng=False)
    payload['model_state_dict']['channel_scale'] = torch.ones(1, 2, 1, 1)
    path = tmp_path / 'wrong_scale.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        artifacts.load_model_artifact(path, weights_only=True)


def test_actual_grouped_training_produces_safe_classifier(checkpoint, tmp_path):
    import numpy as np
    path, _, _ = checkpoint()
    rng = np.random.default_rng(8)
    crops = rng.uniform(0, 30, (16, 5, 6, 4)).astype(np.float32)
    before = crops.copy()
    labels = ['a', 'b'] * 8
    groups = [f'plateA/well{i // 2}' for i in range(16)]
    output = tmp_path / 'trained.pt'
    reports = []
    result = emb._dino_finetune_classifier(
        crops, path, output, labels=labels, groups=groups, source_channels=[3, 0],
        epochs=2, batch_size=4, holdout=.25, progress=lambda *args: reports.append(args))
    model, payload = artifacts.load_model_artifact(output, weights_only=True)
    assert set(payload['train_groups']).isdisjoint(payload['validation_groups'])
    assert payload['grouped_split']['group_by'] == 'well'
    assert result['classes'] == ['a', 'b']
    assert len(reports) == len(result['loss']) == 2
    assert all(len(report) == 4 for report in reports)
    assert model._own_dino_config['source_channels'] == [3, 0]
    assert np.array_equal(crops, before)
    assert payload['preprocessing']['source_channel_order_is_caller_declared'] is True
    assert payload['rng_state'] is None
    model.eval()
    assert model(torch.from_numpy(crops[..., [3, 0]].transpose(0, 3, 1, 2))).shape == (16, 2)


def test_inadequate_groups_refuse_before_model_and_output(checkpoint, tmp_path):
    import numpy as np
    path, _, calls = checkpoint()
    output = tmp_path / 'not-created.pt'
    with pytest.raises(ValueError):
        emb._dino_finetune_classifier(
            np.ones((4, 4, 4, 2)), path, output, labels=['a', 'b']*2,
            groups=['only-well']*4, source_channels=[0, 1], epochs=1)
    assert not calls
    assert not output.exists()


def test_cancelled_training_has_no_artifact(checkpoint, tmp_path):
    import numpy as np
    path, _, _ = checkpoint()
    output = tmp_path / 'cancelled.pt'
    with pytest.raises(InterruptedError):
        emb._dino_finetune_classifier(
            np.ones((8, 4, 4, 2)), path, output, labels=['a', 'b']*4,
            groups=[f'w{i//2}' for i in range(8)], source_channels=[0, 1],
            cancelled=lambda: True, epochs=1)
    assert not output.exists()


def test_checkpoint_and_safe_reload_without_python311_file_digest(checkpoint, monkeypatch, tmp_path):
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    path, _, _ = checkpoint()
    before = path.read_bytes()
    model = emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    model.eval()
    images = torch.ones(2, 2, 4, 4)
    expected = model(images).detach()
    output = tmp_path / 'portable.pt'
    artifacts.save_model_artifact(model, output, include_rng=False)
    restored, _ = artifacts.load_model_artifact(output, weights_only=True)
    restored.eval()
    assert torch.equal(restored(images), expected)
    assert path.read_bytes() == before


def test_last_validation_cancellation_does_not_publish(checkpoint, monkeypatch, tmp_path):
    import numpy as np
    path, _, _ = checkpoint()
    original_factory = emb._dino_classifier_from_checkpoint
    state = {'cancelled': False}

    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        original_forward = model.forward

        def forward(images):
            result = original_forward(images)
            if not model.training:
                state['cancelled'] = True
            return result

        model.forward = forward
        return model

    monkeypatch.setattr(emb, '_dino_classifier_from_checkpoint', factory)
    output = tmp_path / 'late-cancel.pt'
    with pytest.raises(InterruptedError):
        emb._dino_finetune_classifier(
            np.ones((8, 4, 4, 2)), path, output, labels=['a', 'b']*4,
            groups=[f'w{i//2}' for i in range(8)], source_channels=[0, 1],
            cancelled=lambda: state['cancelled'], epochs=1, batch_size=64)
    assert not output.exists()


def test_foreign_output_appearing_after_preflight_is_preserved(checkpoint, monkeypatch, tmp_path):
    import numpy as np
    path, _, _ = checkpoint()
    original_factory = emb._dino_classifier_from_checkpoint
    output = tmp_path / 'foreign.pt'
    foreign = b'another user application wrote this file'

    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        output.write_bytes(foreign)
        return model

    monkeypatch.setattr(emb, '_dino_classifier_from_checkpoint', factory)
    with pytest.raises(FileExistsError):
        emb._dino_finetune_classifier(
            np.ones((8, 4, 4, 2)), path, output, labels=['a', 'b']*4,
            groups=[f'w{i//2}' for i in range(8)], source_channels=[0, 1],
            epochs=1, batch_size=64)
    assert output.read_bytes() == foreign


@pytest.mark.parametrize('change', [
    {'normalize_features': 'yes'}, {'channel_policy': 'unknown'},
    {'in_chans': 3}, {'source_sha256': 'A' * 64},
    {'source_sha256': None}, {'source_sha256': 'a' * 63},
])
def test_saved_constructor_contract_refuses_before_creating_an_encoder(checkpoint, change):
    path, _, calls = checkpoint()
    config = dict(arch='tiny_local', in_chans=2, size=4, channel_policy='project',
                  scale=[10., 20.], source_channels=[3, 0], num_classes=2,
                  freeze_encoder=True, normalize_features=True, source_sha256='a' * 64)
    config.update(change)
    original = path.read_bytes()
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_config(config)
    assert calls == [] and path.read_bytes() == original


def test_incomplete_constructor_contract_is_a_domain_refusal_not_a_partial_model(checkpoint):
    path, _, calls = checkpoint()
    before = path.read_bytes()
    with pytest.raises(emb.EmbeddingError) as caught:
        emb._dino_classifier_from_config({'in_chans': 2})
    assert isinstance(caught.value.__cause__, KeyError)
    assert calls == [] and path.read_bytes() == before


@pytest.mark.parametrize('payload', [[], {}, {'setup': [], 'teacher': {}},
                                     {'setup': {}, 'teacher': []}])
def test_teacher_structure_refuses_before_encoder_construction(checkpoint, payload):
    path, _, calls = checkpoint()
    torch.save(payload, path)
    before = path.read_bytes()
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    assert calls == [] and path.read_bytes() == before


@pytest.mark.parametrize('weight', [1, torch.tensor(float('nan')),
                                   torch.tensor(float('inf')), torch.tensor(1 + 2j)])
def test_teacher_rejects_non_tensor_nonfinite_or_complex_weights(checkpoint, weight):
    path, _, calls = checkpoint()
    payload = torch.load(path, weights_only=True)
    payload['teacher']['conv.weight'] = weight
    torch.save(payload, path)
    before = path.read_bytes()
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    assert calls == [] and path.read_bytes() == before


@pytest.mark.parametrize('restore_original', [
    False, pytest.param(True, marks=pytest.mark.posix_only)])
def test_teacher_decode_has_no_retry_when_held_bytes_change(checkpoint, monkeypatch, restore_original):
    import os

    path, _, calls = checkpoint()
    original_load = torch.load
    attempts = []
    original = path.read_bytes()
    original_status = path.stat()
    def load_and_change(stream, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'weights_only': True}
        attempts.append(stream.fileno())
        payload = original_load(stream, **kwargs)
        changed = original[:-1] + bytes([original[-1] ^ 1])
        path.write_bytes(changed)
        if restore_original:
            path.write_bytes(original)
            os.utime(path, ns=(original_status.st_atime_ns, original_status.st_mtime_ns))
            assert path.stat().st_ctime_ns != original_status.st_ctime_ns
        return payload
    monkeypatch.setattr(torch, 'load', load_and_change)
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_from_checkpoint(path, num_classes=2, source_channels=[0, 1])
    assert len(attempts) == 1 and calls == []
    expected = original if restore_original else original[:-1] + bytes([original[-1] ^ 1])
    assert path.read_bytes() == expected


@pytest.mark.parametrize('fault', [
    'zero-epochs', 'boolean-epochs', 'zero-batch', 'negative-seed', 'boolean-seed',
    'zero-rate', 'nonfinite-rate', 'boolean-rate', 'empty-crops', 'nan-crops',
    'boolean-crops', 'complex-crops', 'empty-channels', 'duplicate-channels',
    'missing-channel', 'boolean-channel', 'missing-label', 'unknown-label',
    'missing-group', 'unknown-group', 'one-class', 'existing-output',
    'teacher-output', 'float32-overflow',
])
def test_training_preflight_preserves_teacher_crops_and_existing_outputs(checkpoint, tmp_path, fault):
    import numpy as np

    path, _, calls = checkpoint()
    original_teacher = path.read_bytes()
    crops = np.ones((8, 4, 4, 2), dtype=np.float32)
    options = dict(labels=['a', 'b'] * 4, groups=[f'well{i // 2}' for i in range(8)],
                   source_channels=[0, 1], epochs=1, batch_size=4)
    output = tmp_path / 'should-not-be-created.pt'
    scalar_faults = {'zero-epochs': ('epochs', 0), 'boolean-epochs': ('epochs', True),
        'zero-batch': ('batch_size', 0), 'negative-seed': ('seed', -1),
        'boolean-seed': ('seed', True), 'zero-rate': ('lr', 0),
        'nonfinite-rate': ('lr', float('inf')), 'boolean-rate': ('lr', True)}
    if fault in scalar_faults:
        key, value = scalar_faults[fault]
        options[key] = value
    elif fault == 'empty-crops':
        crops = crops[:0]
    elif fault == 'nan-crops':
        crops[0, 0, 0, 0] = np.nan
    elif fault == 'boolean-crops':
        crops = crops.astype(bool)
    elif fault == 'complex-crops':
        crops = crops.astype(complex)
    elif fault.endswith('-channel') or fault.endswith('-channels'):
        options['source_channels'] = {'empty-channels': [], 'duplicate-channels': [0, 0],
            'missing-channel': [0, 2], 'boolean-channel': [True, 1]}[fault]
    elif fault == 'missing-label':
        options['labels'] = options['labels'][:-1]
    elif fault == 'unknown-label':
        options['labels'][0] = None
    elif fault == 'missing-group':
        options['groups'] = options['groups'][:-1]
    elif fault == 'unknown-group':
        options['groups'][0] = 'unknown'
    elif fault == 'one-class':
        options['labels'] = ['a'] * len(crops)
    elif fault == 'existing-output':
        output.write_bytes(b'previous unrelated classifier artifact')
    elif fault == 'teacher-output':
        output = path
    else:
        assert fault == 'float32-overflow'
        crops = np.full(crops.shape, np.finfo(np.float64).max)
    before = crops.copy()
    existing = output.read_bytes() if output.exists() else None
    with np.errstate(over='ignore'):
        with pytest.raises(emb.EmbeddingError):
            emb._dino_finetune_classifier(crops, path, output, **options)
    np.testing.assert_array_equal(crops, before)
    assert path.read_bytes() == original_teacher and calls == []
    assert (output.read_bytes() if output.exists() else None) == existing
    assert not list(tmp_path.glob('.spacr-own-dino-*'))
