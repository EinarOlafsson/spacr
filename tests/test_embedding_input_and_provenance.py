"""Embedding refusal, CPU batching and local weight provenance need no downloads."""
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from spacr import embeddings as emb


@pytest.mark.parametrize('kwargs', [
    {'batch_size': 0}, {'channel_scale': ('not-a-number',)},
])
def test_invalid_spec_is_refused_before_loading_a_model(kwargs):
    with pytest.raises(emb.EmbeddingError):
        emb.EmbeddingSpec(**kwargs)


@pytest.mark.parametrize('array,kwargs,match', [
    (np.zeros((2, 3)), {}, 'shape'),
    (np.zeros((0, 2, 2, 1)), {}, 'no crops'),
    (np.zeros((2, 2, 2, 1)), {'sample_size': 0}, 'at least 1'),
    (np.zeros((2, 2, 2, 1)), {'read_batch': 0}, 'at least 1'),
    (np.zeros((2, 2, 2, 1)), {'channels': (1,)}, 'channel 1'),
])
def test_scale_estimation_rejects_invalid_inputs(array, kwargs, match):
    with pytest.raises(emb.EmbeddingError, match=match):
        emb._estimate_channel_scale(array, **kwargs)


def test_a_short_storage_read_cannot_be_used_to_estimate_a_plate_scale():
    class TruncatedStorage:
        shape = (4, 2, 2, 1)

        def __getitem__(self, index):
            return np.zeros((len(index)-1, 2, 2, 1), dtype=np.float32)

    with pytest.raises(emb.EmbeddingError, match='reading 2 crops returned shape'):
        emb._estimate_channel_scale(TruncatedStorage(), sample_size=4, read_batch=2)


def test_bad_channels_and_plate_shape_are_rejected_before_encoder_calls():
    encoder = Mock()
    with pytest.raises(emb.EmbeddingError, match='channel 2'):
        emb.embed_array(np.zeros((2, 2, 2, 1)), emb.EmbeddingSpec(channels=(2,)), encoder=encoder)
    with pytest.raises(emb.EmbeddingError, match='shape'):
        emb._embed_plate(np.zeros((2, 2)), encoder=encoder)
    encoder.assert_not_called()


def test_single_projected_channel_is_padded_with_zeros_without_changing_values():
    crops = np.arange(12, dtype=np.float32).reshape(3, 2, 2, 1)/12
    seen = []

    def encode(stack):
        seen.append(stack.copy())
        return stack.mean(axis=(1, 2))

    result = emb._embed_plate(crops, emb.EmbeddingSpec(
        normalize=False, channel_policy=emb.CHANNEL_PROJECT), encoder=encode, record={})
    np.testing.assert_array_equal(seen[0][..., 0], crops[..., 0])
    np.testing.assert_array_equal(seen[0][..., 1:], 0)
    np.testing.assert_allclose(result.values[:, 0], crops.mean(axis=(1, 2, 3)))
    assert result.columns == ('emb_rgb_000', 'emb_rgb_001', 'emb_rgb_002')
    assert emb.channel_of_column('cell_area') is None


def test_inconsistent_encoder_widths_are_not_published_as_aligned_features():
    sizes = iter([2, 3])
    with pytest.raises(emb.EmbeddingError, match='different widths per channel'):
        emb.embed_array(np.ones((3, 2, 2, 2)), emb.EmbeddingSpec(normalize=False),
                        encoder=lambda stack: np.ones((len(stack), next(sizes))))


def test_cpu_adapter_batches_real_tensors_in_eval_mode_without_gradients(monkeypatch):
    torch = pytest.importorskip('torch')
    batches = []

    class TinyEncoder(torch.nn.Module):
        def forward(self, tensor):
            batches.append((tuple(tensor.shape), self.training,
                            torch.is_grad_enabled(), tensor.device.type))
            return tensor.mean(dim=(2, 3))

    model = TinyEncoder()
    create = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'timm', SimpleNamespace(create_model=create))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    crops = np.arange(5*2*3*3, dtype=np.float32).reshape(5, 2, 3, 3)/100
    run = emb._timm_encoder(emb.EmbeddingSpec(backbone='test-encoder', batch_size=2))
    values = run(crops)
    np.testing.assert_allclose(values, crops.mean(axis=(1, 2)), rtol=1e-6)
    assert batches == [((2, 3, 2, 3), False, False, 'cpu'),
                       ((2, 3, 2, 3), False, False, 'cpu'),
                       ((1, 3, 2, 3), False, False, 'cpu')]
    create.assert_called_once_with('test-encoder', pretrained=True, num_classes=0)


@pytest.mark.parametrize('filename', [None, 'weights.bin'])
def test_weight_provenance_hashes_actual_cached_bytes(tmp_path, monkeypatch, filename):
    path = tmp_path/'cached-weights'
    content = b'weights\x00'*(300_000)  # Larger than the hash reader's one-MiB chunk.
    path.write_bytes(content)
    lookup = Mock(return_value=str(path))
    config = SimpleNamespace(hf_hub_id='test/encoder', hf_hub_filename=filename)
    monkeypatch.setitem(sys.modules, 'timm', SimpleNamespace(get_pretrained_cfg=lambda _: config))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(try_to_load_from_cache=lookup))
    digest = hashlib.sha256(content).hexdigest()
    assert emb._weights_on_disk('test-encoder') == (str(path), digest, len(content))
    lookup.assert_called_once_with('test/encoder', filename or 'model.safetensors')
    entry = emb.encoder_entry(emb.EmbeddingSpec(backbone='test-encoder'), scorecard={'accuracy': 0.5})
    assert entry.path == str(path) and entry.sha256 == digest and entry.size_bytes == len(content)
    assert entry.source == 'local' and entry.metrics == {'accuracy': 0.5}
    assert not entry.verified
    assert not any('No checksum' in note or 'No scorecard' in note for note in entry.notes)


@pytest.mark.parametrize('outcome', ['no_repository', 'config_error', 'cache_error', 'sentinel', 'missing', 'empty'])
def test_missing_or_unavailable_cache_never_claims_a_checksum(tmp_path, monkeypatch, outcome):
    config = SimpleNamespace(hf_hub_id=None if outcome == 'no_repository' else 'test/encoder',
                             hf_hub_filename='model.safetensors')
    get_config = Mock(return_value=config, side_effect=ValueError('unknown') if outcome == 'config_error' else None)
    result = object() if outcome == 'sentinel' else (None if outcome == 'empty' else str(tmp_path/'absent'))
    lookup = Mock(return_value=result, side_effect=OSError('cache unavailable') if outcome == 'cache_error' else None)
    monkeypatch.setitem(sys.modules, 'timm', SimpleNamespace(get_pretrained_cfg=get_config))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(try_to_load_from_cache=lookup))
    assert emb._weights_on_disk('test-encoder') == ('', '', 0)
    if outcome in ('no_repository', 'config_error'):
        lookup.assert_not_called()


def test_missing_optional_model_library_has_no_cached_provenance(monkeypatch):
    monkeypatch.setitem(sys.modules, 'timm', None)
    assert emb._weights_on_disk('missing-library') == ('', '', 0)
