"""Real Metal execution of encoder, pretraining, and TorchScript adapters."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available(),
    reason="A working Metal device is required",
)
from spacr import embeddings as emb  # noqa: E402


def test_timm_embedding_uses_radeon_with_real_resnet(monkeypatch):
    timm = pytest.importorskip("timm")
    original = timm.create_model

    def model(*args, **kwargs):
        kwargs["pretrained"] = False
        net = original(*args, **kwargs)
        model.net = net
        return net

    monkeypatch.setattr(timm, "create_model", model)
    spec = emb.EmbeddingSpec(backbone="resnet18", batch_size=2)
    run = emb._timm_encoder(spec)
    data = np.random.default_rng(12).random((3, 64, 64, 3), dtype=np.float32)
    output = run(data)
    assert output.shape == (3, 512) and np.isfinite(output).all()
    assert next(model.net.parameters()).device.type == "mps"


def test_dino_training_checkpoint_resume_and_encoder_on_radeon(tmp_path):
    pytest.importorskip("timm")
    path = str(tmp_path / "dino.pt")
    data = np.random.default_rng(8).random((4, 32, 32, 1), dtype=np.float32)
    options = dict(
        arch="resnet18", size=32, batch_size=2, out_dim=8, n_local=0, pretrained=False, seed=2
    )
    first = emb._dino_pretrain(data, path, epochs=1, **options)
    assert first['epochs'] == 1 and np.isfinite(first['loss']).all()
    second = emb._dino_pretrain(data, path, epochs=2, **options)
    assert len(second["loss"]) == 2 and np.isfinite(second["loss"]).all()
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["epoch"] == 2
    run = emb._dino_encoder(emb.EmbeddingSpec(backbone="dino:" + path, batch_size=2))
    output = run(data)
    assert output.shape == (4, 512) and np.isfinite(output).all()


def test_mitonet_torchscript_runs_heads_and_tiling_on_radeon(tmp_path):
    from spacr import model_zoo as zoo

    class Heads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)

        def forward(self, x):
            out = self.conv(x)
            return {
                "sem_logits": out[:, :1],
                "ctr_hmp": out[:, 1:2].sigmoid(),
                "offsets": out[:, 2:4],
            }

    net = torch.jit.trace(Heads().eval(), torch.ones(1, 1, 128, 128), strict=False)
    path = tmp_path / "heads.pt"
    net.save(str(path))
    labels = zoo.segment_mitonet(
        np.ones((160, 160), dtype=np.float32), weights=path, tile=128, halo=16, min_size=1
    )
    assert labels.shape == (160, 160)
    assert any(
        next(model.parameters()).device.type == "mps" for model in zoo._MITONET_MODELS.values()
    )


def test_timepose_default_uses_shared_radeon():
    from spacr.model_zoo import _timepose_device

    assert _timepose_device() == "mps"
    assert _timepose_device("cpu") == "cpu"


def test_transformers_vit_and_segformer_support_current_torch():
    transformers = pytest.importorskip("transformers")
    assert transformers.is_torch_available()
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, ViTConfig, ViTModel

    config = ViTConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    model = ViTModel(config).to("mps")
    out = model(torch.randn(2, 3, 32, 32, device="mps")).last_hidden_state
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    config = SegformerConfig(
        num_labels=2,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 2, 4, 8],
        decoder_hidden_size=32,
    )
    model = SegformerForSemanticSegmentation(config).to("mps").eval()
    out = model(torch.randn(2, 3, 64, 64, device="mps")).logits
    assert torch.isfinite(out).all()
    out.square().mean().backward()


def test_trackastra_attention_on_radeon():
    pytest.importorskip("trackastra")
    from trackastra.model import TrackingTransformer

    model = (
        TrackingTransformer(
            coord_dim=2,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            pos_embed_per_dim=8,
            window=2,
        )
        .to("mps")
        .eval()
    )
    coords = torch.tensor(
        [[[0, 10, 10], [0, 20, 20], [1, 11, 11], [1, 21, 21]]], dtype=torch.float32, device="mps"
    )
    out = model(coords)
    assert out.shape == (1, 4, 4) and torch.isfinite(out).all()
    out.square().mean().backward()
