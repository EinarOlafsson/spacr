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
