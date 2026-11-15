"""Table-driven tests of the foundation-model loaders and embedding preflight, ``spacr.embeddings``.

OpenPhenom and SubCell are fetched from the Hugging Face hub or an S3 bucket
the first time they are chosen, which a test run must not do.  The tests
stand in for exactly the two seams the loaders call -- ``transformers``
(``AutoModel``, ``ViTConfig``, ``ViTModel``) and ``torch.hub`` -- and check
what spaCR does around them: the pinned revision, the resize or native
geometry, the batches, the checkpoint download and min-max scaling, and the
sentence a missing package gets.  ``embed_array`` and ``_embed_parallel`` must
refuse a bad request before any model is loaded, and a chunked embedding must
equal the single-call one.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

import spacr.embeddings as emb

torch = pytest.importorskip("torch")

RYBG_CHANNELS = "subcell_rybg needs the projection policy and four explicit, " \
                "distinct channel indices in r, y, b, g order"
SMALL_CROP = "subcell_rybg needs crops at least 16 pixels high and wide for " \
             "its ViT-B/16 patch embedding"


def _spec(backbone, batch_size=2, **kwargs):
    return emb.EmbeddingSpec(backbone=backbone, device="cpu",
                             batch_size=batch_size, **kwargs)


def _rybg(**kwargs):
    base = dict(backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
                channels=(0, 1, 2, 3), normalize=False)
    base.update(kwargs)
    return emb.EmbeddingSpec(**base)


def _transformers(monkeypatch, net=None, vit=None):
    """Install a ``transformers`` stand-in; returns the pretrained requests."""
    asked = []
    module = types.ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(repo, revision, trust_remote_code):
            asked.append((repo, revision, trust_remote_code))
            return net

    module.AutoModel = AutoModel
    if vit is not None:
        module.ViTConfig = lambda **config: types.SimpleNamespace(**config)
        module.ViTModel = vit
    monkeypatch.setitem(sys.modules, "transformers", module)
    return asked


class _OpenPhenom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def predict(self, x):
        self.seen.append(tuple(x.shape))
        return x.mean(dim=(2, 3)) / 255.0


class _ViT(torch.nn.Module):
    """A token encoder without weights that records every input it sees."""

    seen: list = []

    def __init__(self, config, add_pooling_layer):
        super().__init__()
        assert add_pooling_layer is False
        self.config = config

    def forward(self, x, interpolate_pos_encoding):
        type(self).seen.append(x.detach().cpu().numpy().copy())
        n = x.shape[0]
        tokens = x.mean(dim=(1, 2, 3)).reshape(n, 1, 1).expand(n, 4, 768)
        return types.SimpleNamespace(last_hidden_state=tokens.contiguous())


def _subcell_checkpoint():
    """SubCell's published pooler keys, before spaCR renames them."""
    generator = torch.Generator().manual_seed(0)

    def weights(*shape):
        return torch.randn(*shape, generator=generator) * 0.01

    return {"pool_model.attention_v.1.weight": weights(512, 768),
            "pool_model.attention_v.1.bias": weights(512),
            "pool_model.attention_u.1.weight": weights(512, 768),
            "pool_model.attention_u.1.bias": weights(512),
            "pool_model.attention.weight": weights(2, 512),
            "pool_model.attention.bias": weights(2)}


def _subcell_hub(monkeypatch, tmp_path):
    """Stand in for ``torch.hub``; returns the list of downloaded URLs."""
    downloads = []
    _ViT.seen = []
    _transformers(monkeypatch, vit=_ViT)

    def download(url, target):
        downloads.append(url)
        torch.save(_subcell_checkpoint(), target)

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    return downloads


def test_foundation_encoder_loads_each_model_through_its_two_seams(
        monkeypatch, tmp_path, subtests):
    """Pinned revision, own input size, one download, and named missing packages."""
    with subtests.test(msg="openphenom_by_revision_at_its_own_size"):
        net = _OpenPhenom()
        asked = _transformers(monkeypatch, net)
        encode = emb._foundation_encoder(_spec("openphenom"))
        info = emb._FOUNDATION_MODELS["openphenom"]
        assert asked == [(info["repo"], info["revision"], True)]
        assert net.return_channelwise_embeddings is False
        assert encode.in_channels is None
        features = encode(np.full((3, 10, 12, 2), 0.5, dtype=np.float32))
        np.testing.assert_allclose(features, np.full((3, 2), 0.5), atol=1e-6)
        assert net.seen == [(2, 2, 256, 256), (1, 2, 256, 256)]

    with subtests.test(msg="subcell_downloaded_once_and_min_max_scaled"):
        downloads = _subcell_hub(monkeypatch, tmp_path / "subcell")
        encode = emb._foundation_encoder(_spec("subcell"))
        assert encode.in_channels == 2
        assert downloads == [emb._FOUNDATION_MODELS["subcell"]["url"]]
        stack = np.random.default_rng(1).random((1, 20, 20, 2)) \
            .astype(np.float32)
        features = encode(stack)
        assert features.shape == (1, 1536)
        assert np.isfinite(features).all()
        # Each plane reaches the network min-max scaled to exactly [0, 1].
        (seen,) = _ViT.seen
        assert seen.shape[:2] == (1, 2)
        np.testing.assert_allclose(seen.min(axis=(2, 3)), [[0.0, 0.0]],
                                   atol=1e-6)
        np.testing.assert_allclose(seen.max(axis=(2, 3)), [[1.0, 1.0]],
                                   atol=1e-6)
        emb._foundation_encoder(_spec("subcell"))
        assert len(downloads) == 1

    for shape in ((15, 20), (20, 15)):
        with subtests.test(msg=f"subcell_rybg_crop_{shape}_smaller_than_a_patch"):
            _subcell_hub(monkeypatch, tmp_path / f"small{shape[0]}")
            encoder = emb._foundation_encoder(_rybg(device="cpu"))
            with pytest.raises(emb.EmbeddingError) as excinfo:
                encoder(np.zeros((1, *shape, 4), dtype=np.float32))
            assert str(excinfo.value) == SMALL_CROP
            assert _ViT.seen == []

    with subtests.test(msg="zip_archive_named_safetensors_still_loads"):
        weights = tmp_path / "model.safetensors"
        torch.save({"state_dict": {"w": torch.ones(2)}}, weights)
        real_load = torch.load

        def routed(f, *args, **kwargs):
            # Recent torch sends every *.safetensors path to that reader.
            if isinstance(f, (str, os.PathLike)) and \
                    os.fspath(f).endswith(".safetensors"):
                raise RuntimeError("header too large")
            return real_load(f, *args, **kwargs)

        seen = {}

        class AutoModel:
            @staticmethod
            def from_pretrained(repo, revision, trust_remote_code):
                seen["state"] = torch.load(str(weights), map_location="cpu")
                return _OpenPhenom()

        module = types.ModuleType("transformers")
        module.AutoModel = AutoModel
        monkeypatch.setitem(sys.modules, "transformers", module)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "load", routed)
            emb._foundation_encoder(_spec("openphenom"))
            assert torch.equal(seen["state"]["state_dict"]["w"], torch.ones(2))
            assert torch.load is routed

    # (case, backbone, missing module, exact message)
    missing = [
        ("openphenom_without_transformers", "openphenom", "transformers",
         "openphenom needs the transformers package; pip install transformers"),
        ("subcell_without_transformers", "subcell", "transformers",
         "subcell needs the transformers package; pip install transformers"),
        ("no_torch_names_the_extra", "openphenom", "torch",
         "foundation-model embeddings need torch; install the "
         "`spacr[embeddings]` extra"),
    ]
    for case_id, backbone, module_name, message in missing:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, module_name, None)
                with pytest.raises(emb.EmbeddingError) as excinfo:
                    emb._foundation_encoder(_spec(backbone))
            assert str(excinfo.value) == message


def test_embed_array_refuses_a_bad_request_before_any_model_load(
        monkeypatch, tmp_path, subtests):
    """Plane, width, size and column problems fail first; a large crop stays native."""
    def unexpected(*_args, **_kwargs):
        raise AssertionError("no scale estimate or model load may begin")

    # (case, policy, channels, exact message)
    planes = [
        ("no_channels", emb.CHANNEL_PROJECT, None, RYBG_CHANNELS),
        ("three_channels", emb.CHANNEL_PROJECT, (0, 1, 2), RYBG_CHANNELS),
        ("repeated_channel", emb.CHANNEL_PROJECT, (0, 1, 2, 2), RYBG_CHANNELS),
        ("channel_not_in_the_crops", emb.CHANNEL_PROJECT, (0, 1, 2, 4),
         "channel 4 is not in the crops, which have 4"),
        ("negative_channel", emb.CHANNEL_PROJECT, (0, 1, 2, -1), RYBG_CHANNELS),
        ("boolean_channel", emb.CHANNEL_PROJECT, (0, 1, 2, True),
         RYBG_CHANNELS),
        ("float_channel", emb.CHANNEL_PROJECT, (0, 1, 2, 3.0), RYBG_CHANNELS),
        ("text_channel", emb.CHANNEL_PROJECT, (0, 1, 2, "3"), RYBG_CHANNELS),
        ("per_channel_policy", emb.CHANNEL_PER_CHANNEL, (0, 1, 2, 3),
         RYBG_CHANNELS),
    ]
    for case_id, policy, channels, message in planes:
        with subtests.test(msg=f"planes_{case_id}"):
            with monkeypatch.context() as patch:
                patch.setattr(emb, "_backbone_encoder", unexpected)
                with pytest.raises(emb.EmbeddingError) as excinfo:
                    emb.embed_array(np.zeros((1, 20, 20, 4), dtype=np.float32),
                                    _rybg(channel_policy=policy,
                                          channels=channels))
            assert str(excinfo.value) == message

    for width in (3, None, "missing"):
        with subtests.test(msg=f"encoder_width_{width}"):
            calls = []

            def wrong_encoder(stack):
                calls.append(stack.shape)
                raise AssertionError("a wrong-width encoder must never run")

            if width != "missing":
                wrong_encoder.in_channels = width
            with pytest.raises(emb.EmbeddingError) as excinfo:
                emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32),
                                _rybg(), encoder=wrong_encoder)
            assert str(excinfo.value) == ("subcell_rybg encoder must accept "
                                          "exactly four planes in r, y, b, g "
                                          "order")
            assert calls == []

    for shape in ((15, 20), (20, 15)):
        for entry in ("array", "plate"):
            with subtests.test(msg=f"small_crop_{shape}_via_{entry}"):
                record = {}
                with monkeypatch.context() as patch:
                    patch.setattr(emb, "_estimate_channel_scale", unexpected)
                    patch.setattr(emb, "_backbone_encoder", unexpected)
                    crops = np.zeros((2, *shape, 4), dtype=np.float32)
                    with pytest.raises(emb.EmbeddingError) as excinfo:
                        if entry == "plate":
                            emb._embed_plate(crops, _rybg(normalize=True),
                                             record=record)
                        else:
                            emb.embed_array(crops, _rybg(normalize=True))
                assert str(excinfo.value) == SMALL_CROP
                assert record == {}

    with subtests.test(msg="encoder_width_disagrees_with_the_column_names"):
        def encoder(array):
            return np.zeros((len(array), 4), dtype=np.float32)

        encoder.in_channels = 3
        with monkeypatch.context() as patch:
            patch.setattr(emb, "embedding_column_names",
                          lambda *a, **k: ("emb_0",))
            with pytest.raises(emb.EmbeddingError) as excinfo:
                emb.embed_array(
                    np.random.default_rng(0).random((2, 4, 4, 3))
                    .astype(np.float32),
                    emb.EmbeddingSpec(normalize=False), encoder=encoder)
        # Three channels, each encoded alone into four dimensions.
        assert str(excinfo.value) == "12 embedding dimensions but 1 column names"

    with subtests.test(msg="large_crop_kept_native_with_the_authors_scale"):
        _subcell_hub(monkeypatch, tmp_path / "large")
        crops = np.full((1, 512, 544, 4), 100.0, dtype=np.float32)
        crops[0, 2, 3, 0] = 0.0
        crops[0, 500, 501, 3] = 1000.0
        result = emb.embed_array(crops, _rybg(device="cpu"))
        original = crops[0].transpose(2, 0, 1)
        low, high = original.min(), original.max()
        assert result.values.shape == (1, 1536)
        assert result.n_channels == 4
        (seen,) = _ViT.seen
        assert seen.shape == (1, 4, 512, 544)
        np.testing.assert_array_equal(seen[0],
                                      (original - low) / (high - low + 1e-8))
        assert seen[0, 0, 2, 3] == 0.0
        assert seen[0, 3, 500, 501] == 1.0


def test_embed_parallel_matches_one_call_and_refuses_bad_chunks(
        monkeypatch, subtests):
    """Chunked embedding equals a single embed_array call; bad crops and chunks refused."""
    spec = emb.EmbeddingSpec(normalize=False)
    crops = np.random.default_rng(0).random((5, 4, 4, 1)).astype(np.float32)

    def encoder(array):
        return array.reshape(len(array), -1)[:, :3]

    # (case, crops, kwargs, exact message)
    refusals = [
        ("no_crops", np.zeros((0, 4, 4, 1), dtype=np.float32), {},
         "crops must be a non-empty (n, height, width, channels); got shape "
         "(0, 4, 4, 1)"),
        ("not_four_dimensional", np.zeros((4, 4), dtype=np.float32), {},
         "crops must be a non-empty (n, height, width, channels); got shape "
         "(4, 4)"),
        ("chunk_size_zero", np.zeros((2, 4, 4, 1), dtype=np.float32),
         dict(chunk_size=0, encoder=encoder),
         "chunk_size must be at least 1"),
    ]
    for case_id, array, kwargs, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(emb.EmbeddingError) as excinfo:
                emb._embed_parallel(array, spec, **kwargs)
            assert str(excinfo.value) == message

    single = emb.embed_array(crops, spec, encoder=encoder)
    np.testing.assert_array_equal(single.values,
                                  np.repeat(crops[:, 0, 0, :1], 3, axis=1))
    # (case, device patch, workers, chunk size, expected spec device)
    runs = [
        ("cpu_one_worker_in_chunks_of_two", None, 1, 2, "cpu"),
        ("gpu_device_runs_in_one_process", "cuda:0", 4, 2, "cuda"),
    ]
    for case_id, device, workers, chunk, expected_device in runs:
        with subtests.test(msg=case_id):
            run_spec = spec
            with monkeypatch.context() as patch:
                if device is not None:
                    run_spec = emb.EmbeddingSpec(normalize=False,
                                                 device="cuda")
                    patch.setattr(emb, "_embedding_device",
                                  lambda requested=None: device)
                result = emb._embed_parallel(crops, run_spec, workers=workers,
                                             chunk_size=chunk, encoder=encoder)
            np.testing.assert_array_equal(result.values, single.values)
            assert result.columns == single.columns
            assert result.n_channels == 1
            assert result.spec.device == expected_device

    with subtests.test(msg="chunks_with_different_columns"):
        calls = {"n": 0}

        def uneven(array):
            calls["n"] += 1
            return array.reshape(len(array), -1)[:, :2 + calls["n"] % 2]

        with pytest.raises(emb.EmbeddingError) as excinfo:
            emb._embed_parallel(crops, spec, workers=1, chunk_size=2,
                                encoder=uneven)
        assert str(excinfo.value) == ("the chunks were embedded with "
                                      "different columns")
