"""Merging classifier checkpoints in ``spacr.deep_spacr``: fusion and distillation.

``model_fusion`` averages (or otherwise aggregates) the weights of several
checkpoints and ``model_knowledge_transfer`` distils several teachers into one
student.  Both read the pre-versioned ``{'model': state_dict, ...}`` mapping,
take the architecture from the checkpoint rather than the call, and write a
versioned artifact that ``spacr.torch_artifacts.load_model_artifact`` rebuilds
without being told the architecture.  The fusion checkpoints are constant
valued (every weight 2.0 in one, 8.0 in the other), so each aggregator has a
single exact answer.  CPU only, ``pretrained=False``, no downloads, on
``shufflenet_v2_x0_5`` (~350k parameters).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

_MODEL = "shufflenet_v2_x0_5"


@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a figure window accumulate during these tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _torchmodel(model_name=_MODEL):
    """A small untrained classifier with two classes."""
    from spacr.utils import TorchModel
    return TorchModel(model_name=model_name, pretrained=False, num_classes=2)


def _const_state_dict(model, value):
    """A state dict with the model's exact keys and shapes, every entry ``value``."""
    return {k: torch.full_like(v, float(value) if v.is_floating_point()
                               else int(value))
            for k, v in model.state_dict().items()}


def _save_dict_ckpt(path, value, model_name=_MODEL):
    """Save a pre-versioned ``{'model': state_dict, ...metadata}`` checkpoint."""
    torch.save({"model": _const_state_dict(_torchmodel(model_name), value),
                "model_name": model_name, "pretrained": False,
                "dropout_rate": None, "use_checkpoint": False}, str(path))
    return str(path)


def _tiny_loader(n=4, size=64):
    """Four random 64 px images with binary labels, two per batch."""
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, size, size, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_model_fusion_aggregates_dict_checkpoints_exactly(tmp_path, subtests):
    """Every aggregator gives its exact fused weight and a reloadable artifact."""
    from spacr.deep_spacr import model_fusion
    from spacr.torch_artifacts import load_model_artifact
    from spacr.utils import TorchModel

    p1 = _save_dict_ckpt(tmp_path / "m1.pth", 2.0)
    p2 = _save_dict_ckpt(tmp_path / "m2.pth", 8.0)
    cases = [("mean", 5.0), ("geomean", 4.0), ("median", 2.0),
             ("sum", 10.0), ("max", 8.0), ("min", 2.0)]
    for aggregator, expected in cases:
        with subtests.test(msg=aggregator):
            fused = model_fusion(
                [p1, p2], str(tmp_path / "fused"), device="cpu",
                model_name="resnet18",       # overridden by the checkpoints
                pretrained=True,             # overridden by the checkpoints
                aggregator=aggregator)

            assert isinstance(fused, TorchModel)
            assert fused.model_name == _MODEL
            sd = fused.state_dict()
            float_keys = [k for k, v in sd.items() if v.is_floating_point()]
            assert float_keys
            for k in float_keys:
                assert torch.allclose(sd[k], torch.full_like(sd[k], expected)), k

            out = tmp_path / f"fused_{aggregator}.pth"
            reloaded, payload = load_model_artifact(out, map_location="cpu")
            assert (type(reloaded), reloaded.model_name) == (TorchModel, _MODEL)
            assert payload["artifact_role"] == "model_fusion"


def test_model_knowledge_transfer_distils_dict_checkpoint_teachers(
        tmp_path, capsys):
    """Dict-checkpoint teachers are rebuilt, distilled, and saved as ``_KD.pth``."""
    from spacr.deep_spacr import model_knowledge_transfer
    from spacr.torch_artifacts import load_model_artifact
    from spacr.utils import TorchModel

    t1 = _save_dict_ckpt(tmp_path / "t1.pth", 0.01)
    t2 = _save_dict_ckpt(tmp_path / "t2.pth", 0.02)

    student = model_knowledge_transfer(
        teacher_paths=[t1, t2],
        student_save_path=str(tmp_path / "student"),   # no suffix: kept as base
        data_loader=_tiny_loader(), device="cpu",
        student_model_name=_MODEL, pretrained=False, epochs=1, lr=1e-3)

    assert isinstance(student, TorchModel)
    assert student.model_name == _MODEL
    printed = capsys.readouterr().out
    assert "Epoch [1/1]" in printed
    assert np.isfinite(float(printed.split("Loss: ")[1].split()[0]))

    out_path = tmp_path / "student_KD.pth"
    reloaded, payload = load_model_artifact(out_path, map_location="cpu")
    assert (type(reloaded), reloaded.model_name) == (TorchModel, _MODEL)
    assert payload["artifact_role"] == "knowledge_distillation"
    assert payload["optimizer_state_dict"] is not None
    with torch.no_grad():
        logits = reloaded(torch.rand(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, 2)
