"""``spacr.deep_spacr.model_knowledge_transfer``: whole-module teachers and refusals.

A teacher saved as a whole ``TorchModel`` object is used as-is, and a
checkpoint that is neither a module nor a mapping is refused before a student
file is written.  The dict-checkpoint teacher branch and ``model_fusion`` live
in ``tests/test_deep_spacr_checkpoint_merging.py``.  CPU only,
``pretrained=False`` (no downloads), on ``shufflenet_v2_x0_5``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a figure window accumulate (or block) during these tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# shared tiny-checkpoint helpers
# ---------------------------------------------------------------------------

_MODEL = "shufflenet_v2_x0_5"      # ~350k params, no download with pretrained=False


def _torchmodel(model_name=_MODEL):
    from spacr.utils import TorchModel
    return TorchModel(model_name=model_name, pretrained=False, num_classes=2)


def _const_state_dict(model, value):
    """A state dict with the model's exact keys/shapes, every entry == value."""
    out = {}
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            out[k] = torch.full_like(v, float(value))
        else:
            out[k] = torch.full_like(v, int(value))
    return out


def _tiny_loader(n=4, size=64):
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, size, size, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=2)


# ---------------------------------------------------------------------------
# model_knowledge_transfer
# ---------------------------------------------------------------------------

def test_model_knowledge_transfer_from_torchmodel_checkpoint(tmp_path, capsys):
    """A teacher saved as a whole TorchModel object is used as-is."""
    from spacr.deep_spacr import model_knowledge_transfer
    from spacr.utils import TorchModel

    teacher = _torchmodel()
    teacher.load_state_dict(_const_state_dict(teacher, 0.05))
    torch.save(teacher, str(tmp_path / "t.pth"))

    student = model_knowledge_transfer(
        teacher_paths=[str(tmp_path / "t.pth")],
        student_save_path=str(tmp_path / "student.pth"),   # '.pth' -> stem reused
        data_loader=_tiny_loader(n=2),
        device="cpu",
        student_model_name=_MODEL,
        pretrained=False,
        epochs=1,
        lr=1e-3,
    )

    assert isinstance(student, TorchModel)
    assert (tmp_path / "student_KD.pth").exists()
    assert not (tmp_path / "student.pth.pth").exists()
    assert "Loading teacher" in capsys.readouterr().out


def test_model_knowledge_transfer_rejects_unsupported_checkpoint(tmp_path):
    """A checkpoint that is neither a TorchModel nor a dict is rejected."""
    from spacr.deep_spacr import model_knowledge_transfer

    bad = tmp_path / "bad.pth"
    torch.save([1, 2, 3], str(bad))

    with pytest.raises(ValueError, match="Unsupported checkpoint type"):
        model_knowledge_transfer(
            teacher_paths=[str(bad)],
            student_save_path=str(tmp_path / "s.pth"),
            data_loader=_tiny_loader(n=2),
            device="cpu",
            student_model_name=_MODEL,
            pretrained=False,
            epochs=1,
        )
    assert not (tmp_path / "s_KD.pth").exists()
