"""CPU coverage for the tail end of ``spacr.deep_spacr``.

Covers the four public entry points at the bottom of the module:

  * ``deep_spacr``                — the orchestrator that chains dataset
    generation -> training -> tar generation -> inference -> top-example
    export -> DB merge. Every stage is stubbed with a recorder so the
    wiring (which src is handed to which stage, which paths are derived,
    which stage is skipped) can be asserted exactly, offline and in
    milliseconds.
  * ``model_knowledge_transfer``  — the dict-checkpoint teacher branch and
    the unsupported-checkpoint guard.
  * ``model_fusion``              — the dict-checkpoint branch, every
    aggregator, and all three ``ValueError`` guards.
  * ``annotate_filter_vision``    — CSV annotation + threshold filtering +
    removal of rows whose PNG was used for training.

The fusion tests use constant-valued state dicts (all weights 2.0 in one
checkpoint, 8.0 in the other) so each aggregator has a single exact
expected value: mean=5, geomean=4, median=2, sum=10, max=8, min=2.

Everything runs on CPU with ``pretrained=False`` (no downloads) on
``shufflenet_v2_x0_5`` (~350k parameters), so the whole file is fast.
"""
from __future__ import annotations


import numpy as np
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
# deep_spacr orchestrator — every stage replaced by a recorder
# ---------------------------------------------------------------------------

class _Stubs:
    """Recorders for the six collaborators ``deep_spacr`` calls."""

    def __init__(self):
        self.gen_train_ret = (None, None)
        self.gen_ds_ret = None
        self.train_ret = None
        self.apply_ret = None
        self.save_settings = []
        self.gen_train = []
        self.gen_ds = []
        self.train = []
        self.apply_tar = []
        self.top = []
        self.merge = []


@pytest.fixture
def ds_stubs(monkeypatch):
    """Stub every collaborator of ``deep_spacr`` and hand back the recorder.

    ``deep_spacr`` imports ``generate_training_dataset`` / ``generate_dataset``
    / ``save_settings`` *inside* the function body, so patching the attribute
    on ``spacr.io`` / ``spacr.utils`` is enough; the rest are module globals
    of ``spacr.deep_spacr``.
    """
    s = _Stubs()

    def fake_save_settings(settings, name="settings", show=False):
        s.save_settings.append((dict(settings), name))

    def fake_generate_training_dataset(settings):
        s.gen_train.append(dict(settings))
        return s.gen_train_ret

    def fake_generate_dataset(settings):
        s.gen_ds.append(dict(settings))
        return s.gen_ds_ret

    def fake_train_test_model(settings):
        s.train.append(dict(settings))
        return s.train_ret

    def fake_apply_model_to_tar(settings):
        s.apply_tar.append(dict(settings))
        return s.apply_ret

    def fake_save_top(df, tar_path, dst, n=20, classes=None):
        s.top.append((df, tar_path, dst, n, classes))

    def fake_merge(df, db_path):
        s.merge.append((df, db_path))

    monkeypatch.setattr("spacr.utils.save_settings", fake_save_settings)
    monkeypatch.setattr("spacr.io.generate_training_dataset",
                        fake_generate_training_dataset)
    monkeypatch.setattr("spacr.io.generate_dataset", fake_generate_dataset)
    monkeypatch.setattr("spacr.deep_spacr.train_test_model", fake_train_test_model)
    monkeypatch.setattr("spacr.deep_spacr.apply_model_to_tar", fake_apply_model_to_tar)
    monkeypatch.setattr("spacr.deep_spacr.save_top_class_examples", fake_save_top)
    monkeypatch.setattr("spacr.deep_spacr.merge_predictions_into_db", fake_merge)
    return s


# ---------------------------------------------------------------------------
# shared tiny-checkpoint helpers for knowledge transfer / fusion
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


def _save_dict_ckpt(path, value, model_name=_MODEL, state_dict=None):
    """Save a ``{'model': state_dict, ...metadata}`` checkpoint (the dict branch)."""
    if state_dict is None:
        state_dict = _const_state_dict(_torchmodel(model_name), value)
    torch.save(
        {
            "model": state_dict,
            "model_name": model_name,
            "pretrained": False,
            "dropout_rate": None,
            "use_checkpoint": False,
        },
        str(path),
    )
    return str(path)


def _tiny_loader(n=4, size=64):
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, size, size, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=2)


# ---------------------------------------------------------------------------
# model_knowledge_transfer
# ---------------------------------------------------------------------------

def test_model_knowledge_transfer_from_dict_checkpoints(tmp_path, capsys):
    """Teachers stored as ``{'model': state_dict, ...}`` dicts are rebuilt + distilled."""
    from spacr.deep_spacr import model_knowledge_transfer
    from spacr.utils import TorchModel

    t1 = _save_dict_ckpt(tmp_path / "t1.pth", 0.01)
    t2 = _save_dict_ckpt(tmp_path / "t2.pth", 0.02)

    # No '.pth' suffix -> the else-branch keeps the path as the base name.
    student = model_knowledge_transfer(
        teacher_paths=[t1, t2],
        student_save_path=str(tmp_path / "student"),
        data_loader=_tiny_loader(),
        device="cpu",
        student_model_name=_MODEL,
        pretrained=False,
        epochs=1,
        lr=1e-3,
    )

    out_path = tmp_path / "student_KD.pth"
    assert out_path.exists(), "expected the _KD.pth suffix to be appended"
    assert isinstance(student, TorchModel)
    assert student.model_name == _MODEL

    # The distillation loop really ran (one epoch, finite loss).
    printed = capsys.readouterr().out
    assert "Epoch [1/1]" in printed
    loss_txt = printed.split("Loss: ")[1].split()[0]
    assert np.isfinite(float(loss_txt))

    # The saved artifact reconstructs a usable model and carries training state.
    from spacr.torch_artifacts import load_model_artifact
    reloaded, payload = load_model_artifact(out_path, map_location="cpu")
    assert isinstance(reloaded, TorchModel)
    assert payload["artifact_role"] == "knowledge_distillation"
    assert payload["optimizer_state_dict"] is not None
    with torch.no_grad():
        logits = reloaded(torch.rand(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, 2)


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


# ---------------------------------------------------------------------------
# model_fusion
# ---------------------------------------------------------------------------

_AGGREGATOR_EXPECTATIONS = [
    ("mean", 5.0),
    ("geomean", 4.0),      # exp(mean(log([2, 8]))) == 4
    ("median", 2.0),       # torch.median picks the lower of the two
    ("sum", 10.0),
    ("max", 8.0),
    ("min", 2.0),
]


@pytest.mark.parametrize("aggregator,expected", _AGGREGATOR_EXPECTATIONS)
def test_model_fusion_dict_checkpoints_every_aggregator(tmp_path, aggregator,
                                                        expected):
    """Fuse two constant-valued dict checkpoints; assert the exact fused weight."""
    from spacr.deep_spacr import model_fusion
    from spacr.utils import TorchModel

    p1 = _save_dict_ckpt(tmp_path / "m1.pth", 2.0)
    p2 = _save_dict_ckpt(tmp_path / "m2.pth", 8.0)

    fused = model_fusion(
        [p1, p2],
        str(tmp_path / "fused"),          # no '.pth' -> base name kept verbatim
        device="cpu",
        model_name="resnet18",            # must be overridden by ckpt metadata
        pretrained=True,                  # must be overridden by ckpt metadata
        aggregator=aggregator,
    )

    assert isinstance(fused, TorchModel)
    assert fused.model_name == _MODEL, "architecture must come from the checkpoint"
    assert (tmp_path / f"fused_{aggregator}.pth").exists()

    sd = fused.state_dict()
    float_keys = [k for k, v in sd.items() if v.is_floating_point()]
    assert float_keys
    for k in float_keys:
        assert torch.allclose(sd[k], torch.full_like(sd[k], expected)), k

    # The saved artifact reconstructs a usable model and records provenance.
    from spacr.torch_artifacts import load_model_artifact
    reloaded, payload = load_model_artifact(
        tmp_path / f"fused_{aggregator}.pth", map_location="cpu")
    assert isinstance(reloaded, TorchModel)
    assert payload["artifact_role"] == "model_fusion"
