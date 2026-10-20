"""Apply saved own-DINO artifacts to raw crops without changing preprocessing."""
import hashlib

import numpy as np
import pytest
import torch

from spacr import embeddings as emb
from spacr import torch_artifacts as artifacts
from tests.test_own_dino_classifier import checkpoint


@pytest.mark.parametrize('policy', ['project', 'per_channel'])
@pytest.mark.parametrize('normalize_features', [False, True])
def test_saved_classifier_applies_raw_physical_channels_once(
        checkpoint, tmp_path, policy, normalize_features):
    teacher, _, _ = checkpoint(policy, channels=2 if policy == 'project' else 1)
    model = emb._dino_classifier_from_checkpoint(
        teacher, num_classes=2, source_channels=[3, 0],
        normalize_features=normalize_features)
    model.eval()
    raw = np.arange(5 * 3 * 5 * 4, dtype=np.uint16).reshape(5, 3, 5, 4) % 23
    original = raw.copy()
    selected = torch.from_numpy(raw[..., [3, 0]].transpose(0, 3, 1, 2).astype(np.float32))
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        expected = model(selected).detach().numpy()
    hook.remove()
    path = tmp_path / 'classifier.pt'
    artifacts.atomic_torch_save(artifacts.make_model_artifact(
        model, include_rng=False, classes=['negative', 'positive'],
        channels=['3', '0']), path)
    original_artifact = path.read_bytes()

    result = emb._dino_classifier_apply(raw, path, batch_size=2)

    np.testing.assert_allclose(result['logits'], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result['features'], captured[0].numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result['probabilities'], torch.softmax(
        torch.from_numpy(expected), dim=1).numpy(), rtol=1e-6, atol=1e-7)
    assert result['predictions'] == [
        ['negative', 'positive'][i] for i in expected.argmax(axis=1)]
    assert result['classes'] == ['negative', 'positive']
    assert result['source_channels'] == [3, 0]
    assert result['checkpoint_sha256'] == hashlib.sha256(original_artifact).hexdigest()
    assert result['features'].dtype == result['logits'].dtype == np.float32
    np.testing.assert_array_equal(raw, original)
    assert path.read_bytes() == original_artifact


def saved_classifier(checkpoint, tmp_path, classes=('negative', 'positive')):
    teacher, _, _ = checkpoint()
    model = emb._dino_classifier_from_checkpoint(
        teacher, num_classes=2, source_channels=[3, 0])
    path = tmp_path / 'apply.pt'
    payload = artifacts.make_model_artifact(
        model, include_rng=False, classes=classes, channels=['3', '0'])
    artifacts.atomic_torch_save(payload, path)
    return path, model, payload


@pytest.mark.parametrize('crops', [
    np.empty((0, 2, 2, 4)), np.ones((2, 2, 4)),
    np.ones((1, 2, 2, 4), dtype=bool), np.ones((1, 2, 2, 4), dtype=complex),
    np.full((1, 2, 2, 4), np.nan), np.full((1, 2, 2, 4), np.inf),
])
def test_invalid_raw_crops_refuse_before_loading(crops, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid raw crops must not load any artifact')
    monkeypatch.setattr(artifacts, 'load_model_artifact', forbidden)
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_apply(crops, 'unread.pt')


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_invalid_batch_size_refuses_before_loading(batch_size, monkeypatch):
    monkeypatch.setattr(artifacts, 'load_model_artifact', lambda *args, **kwargs:
                        pytest.fail('invalid batching must not load any artifact'))
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_apply(np.ones((1, 2, 2, 4)), 'unread.pt',
                                   batch_size=batch_size)


@pytest.mark.parametrize('classes', [None, ['one'], ['same', 'same'], ['', 'two']])
def test_saved_classes_must_describe_each_output(checkpoint, tmp_path, classes):
    path, _, _ = saved_classifier(checkpoint, tmp_path, classes)
    before = path.read_bytes()
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_apply(np.ones((1, 2, 2, 4)), path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('raw', [np.ones((1, 2, 2, 3)),
                                np.full((1, 2, 2, 4), np.finfo(np.float64).max)])
def test_missing_physical_plane_and_float32_overflow_refuse(checkpoint, tmp_path, raw):
    path, _, _ = saved_classifier(checkpoint, tmp_path)
    with np.errstate(over='ignore'):
        with pytest.raises(emb.EmbeddingError):
            emb._dino_classifier_apply(raw, path)


def test_restricted_stream_load_matches_path_and_remains_open(checkpoint, tmp_path):
    path, _, _ = saved_classifier(checkpoint, tmp_path)
    by_path, path_metadata = artifacts.load_model_artifact(path, weights_only=True)
    with path.open('rb') as stream:
        by_stream, stream_metadata = artifacts.load_model_artifact(stream, weights_only=True)
        assert not stream.closed
        assert stream.seek(0) == 0
    images = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    by_path.eval()
    by_stream.eval()
    assert torch.equal(by_path(images), by_stream(images))
    assert path_metadata['model_config'] == stream_metadata['model_config']


def test_stream_without_explicit_restricted_loading_never_decodes(checkpoint, tmp_path, monkeypatch):
    path, _, _ = saved_classifier(checkpoint, tmp_path)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs:
                        pytest.fail('unrestricted stream must not decode'))
    with path.open('rb') as stream:
        with pytest.raises(TypeError):
            artifacts.load_model_artifact(stream)
        assert not stream.closed


def test_application_hashes_the_same_held_stream_it_loads(checkpoint, tmp_path, monkeypatch):
    path, _, _ = saved_classifier(checkpoint, tmp_path)
    original = artifacts.load_model_artifact
    seen = []

    def inspect_stream(stream, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'strict': True, 'weights_only': True}
        seen.append((stream.fileno(), hashlib.sha256(stream.read()).hexdigest()))
        stream.seek(0)
        return original(stream, **kwargs)

    monkeypatch.setattr(artifacts, 'load_model_artifact', inspect_stream)
    result = emb._dino_classifier_apply(np.ones((1, 2, 2, 4)), path)
    assert len(seen) == 1
    assert result['checkpoint_sha256'] == seen[0][1]


def test_application_rejects_changed_bytes_after_decode(checkpoint, tmp_path, monkeypatch):
    path, _, _ = saved_classifier(checkpoint, tmp_path)
    original = artifacts.load_model_artifact

    def mutate_after_decode(stream, **kwargs):
        result = original(stream, **kwargs)
        with path.open('r+b') as changed:
            changed.seek(-1, 2)
            byte = changed.read(1)
            changed.seek(-1, 2)
            changed.write(bytes([byte[0] ^ 1]))
        return result

    monkeypatch.setattr(artifacts, 'load_model_artifact', mutate_after_decode)
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_apply(np.ones((1, 2, 2, 4)), path)


def test_teacher_and_unsafe_module_are_not_classifier_artifacts(checkpoint, tmp_path):
    teacher, module, _ = checkpoint()
    raw = np.ones((1, 2, 2, 4))
    with pytest.raises(ValueError, match='no model state dictionary'):
        emb._dino_classifier_apply(raw, teacher)
    unsafe = tmp_path / 'unsafe-module.pt'
    torch.save(module, unsafe)
    with pytest.raises(Exception, match='Weights only load failed'):
        emb._dino_classifier_apply(raw, unsafe)


@pytest.mark.parametrize('kind', ['directory', 'fifo'])
def test_nonregular_classifier_is_refused_before_deserialization(tmp_path, monkeypatch, kind):
    import os

    path = tmp_path / 'not-a-checkpoint'
    if kind == 'fifo':
        if not hasattr(os, 'mkfifo'):
            pytest.skip('requires the POSIX named-pipe fixture')
        os.mkfifo(path)
    else:
        path.mkdir()
    monkeypatch.setattr(artifacts, 'load_model_artifact', lambda *args, **kwargs:
                        pytest.fail('nonregular file must never be deserialized'))
    before = path.lstat()
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_apply(np.ones((1, 2, 2, 4)), path)
    after = path.lstat()
    assert (before.st_dev, before.st_ino, before.st_mode) == (after.st_dev, after.st_ino, after.st_mode)


def test_missing_classifier_has_a_domain_error_without_a_loader_retry(tmp_path, monkeypatch):
    path = tmp_path / 'absent.pt'
    monkeypatch.setattr(artifacts, 'load_model_artifact', lambda *args, **kwargs:
                        pytest.fail('missing file must never reach the loader'))
    with pytest.raises(emb.EmbeddingError) as caught:
        emb._dino_classifier_apply(np.ones((1, 2, 2, 4)), path)
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert not path.exists()


@pytest.mark.parametrize('overflow', ['features', 'logits'])
def test_finite_weights_that_overflow_inference_refuse_without_predictions(checkpoint, tmp_path, overflow):
    path, model, payload = saved_classifier(checkpoint, tmp_path)
    largest = torch.finfo(torch.float32).max
    with torch.no_grad():
        model.encoder.conv.weight.fill_(largest if overflow == 'features' else 1)
        model.encoder.conv.bias.zero_()
        model.encoder.norm.running_mean.zero_()
        model.encoder.norm.running_var.fill_(1)
        model.encoder.norm.weight.fill_(1)
        model.encoder.norm.bias.zero_()
        model.head.weight.fill_(largest)
        model.head.bias.fill_(largest)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    artifacts.atomic_torch_save(artifacts.make_model_artifact(
        model, include_rng=False, classes=['negative', 'positive'], channels=['3', '0']), path)
    before_artifact = path.read_bytes()
    raw = np.full((1, 2, 2, 4), 20, dtype=np.uint16)
    before_raw = raw.copy()
    with pytest.raises(emb.EmbeddingError):
        emb._dino_classifier_apply(raw, path)
    np.testing.assert_array_equal(raw, before_raw)
    assert path.read_bytes() == before_artifact
