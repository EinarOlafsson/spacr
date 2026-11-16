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


class _Stubs:
    """Recorders for the six collaborators ``deep_spacr`` calls.

    The ``ds_stubs`` fixture below is shared: the orchestrator tests in
    ``tests/test_cov_r7_deep_spacr.py`` and
    ``tests/test_n702_pinned_classifier_consumers.py`` import it from here.
    """

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
