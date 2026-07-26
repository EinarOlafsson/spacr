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

import os

import numpy as np
import pandas as pd
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

    def fake_save_top(df, tar_path, dst, n=20):
        s.top.append((df, tar_path, dst, n))

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


def test_deep_spacr_none_settings_aborts_when_dataset_generation_fails(ds_stubs):
    """``settings=None`` -> defaults; a failed train-split aborts before training."""
    from spacr.deep_spacr import deep_spacr

    ds_stubs.gen_train_ret = (None, None)

    assert deep_spacr() is None

    # The None -> {} branch means deep_spacr_defaults supplied everything.
    (saved, name), = ds_stubs.save_settings
    assert name == "DL_model"
    assert saved["src"] == "path"                     # defaults placeholder
    assert saved["generate_training_dataset"] is True
    assert saved["train"] is True

    # generate_training_dataset returned no train path -> hard stop.
    assert len(ds_stubs.gen_train) == 1
    assert ds_stubs.train == []
    assert ds_stubs.gen_ds == []
    assert ds_stubs.apply_tar == []


def test_deep_spacr_full_pipeline_wires_every_stage(tmp_path, ds_stubs):
    """Train -> tar -> inference -> top examples -> DB merge, with derived paths."""
    from spacr.deep_spacr import deep_spacr

    train_dir = tmp_path / "datasets" / "training" / "train"
    test_dir = tmp_path / "datasets" / "training" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    ds_stubs.gen_train_ret = (str(train_dir), str(test_dir))

    model = tmp_path / "model" / "clf.pth"
    model.parent.mkdir()
    model.write_bytes(b"not-really-a-model")
    ds_stubs.train_ret = str(model)

    tar = tmp_path / "datasets" / "ds.tar"
    tar.write_bytes(b"not-really-a-tar")
    ds_stubs.gen_ds_ret = str(tar)

    df = pd.DataFrame({"path": ["a.png", "b.png"], "pred": [0.1, 0.9]})
    ds_stubs.apply_ret = df

    settings = {
        "src": str(tmp_path),
        "train": True, "test": True,
        "generate_training_dataset": True,
        "apply_model_to_dataset": True,
        "tar_path": None,
        "n_top_examples": 3,
    }
    deep_spacr(settings)

    # Training was pointed at the parent of the generated train/ folder.
    (train_settings,) = ds_stubs.train
    assert train_settings["src"] == str(train_dir.parent)

    # ...and src was restored afterwards for the apply stage.
    assert settings["src"] == str(tmp_path)
    assert settings["model_path"] == str(model)

    # tar_path was None -> regenerated and written back into settings.
    assert len(ds_stubs.gen_ds) == 1
    assert settings["tar_path"] == str(tar)

    # Inference ran once, on the settings carrying the fresh tar + model.
    (apply_settings,) = ds_stubs.apply_tar
    assert apply_settings["tar_path"] == str(tar)
    assert apply_settings["model_path"] == str(model)

    # Top examples land next to the tar; n_top_examples is passed through.
    (top_df, top_tar, top_dst, top_n), = ds_stubs.top
    assert top_df is df
    assert top_tar == str(tar)
    assert top_dst == os.path.join(str(tar.parent), "top_examples")
    assert top_n == 3

    # Predictions merged into the measurements DB of the single src.
    (merge_df, merge_db), = ds_stubs.merge
    assert merge_df is df
    assert merge_db == os.path.join(str(tmp_path), "measurements", "measurements.db")


def test_deep_spacr_apply_only_reuses_existing_tar_for_every_src(tmp_path, ds_stubs):
    """train/test off + a valid absolute tar -> no training, no tar regeneration."""
    from spacr.deep_spacr import deep_spacr

    tar = tmp_path / "ds.tar"
    tar.write_bytes(b"tar")
    model = tmp_path / "m.pth"
    model.write_bytes(b"model")
    src_a = tmp_path / "plateA"
    src_b = tmp_path / "plateB"
    src_a.mkdir()
    src_b.mkdir()

    ds_stubs.apply_ret = pd.DataFrame({"path": ["a.png"], "pred": [0.4]})

    settings = {
        "src": [str(src_a), str(src_b)],
        "train": False, "test": False,
        "generate_training_dataset": False,
        "apply_model_to_dataset": True,
        "tar_path": str(tar),
        "model_path": str(model),
    }
    deep_spacr(settings)

    assert ds_stubs.gen_train == []
    assert ds_stubs.train == []
    assert ds_stubs.gen_ds == []                 # existing absolute tar reused
    assert len(ds_stubs.apply_tar) == 1

    # One merge per src entry, in order.
    assert [db for _df, db in ds_stubs.merge] == [
        os.path.join(str(src_a), "measurements", "measurements.db"),
        os.path.join(str(src_b), "measurements", "measurements.db"),
    ]


def test_deep_spacr_relative_tar_regenerated_and_missing_model_skips(tmp_path,
                                                                     ds_stubs,
                                                                     capsys):
    """A non-absolute tar_path is regenerated; a missing model skips inference."""
    from spacr.deep_spacr import deep_spacr

    tar = tmp_path / "ds.tar"
    tar.write_bytes(b"tar")
    ds_stubs.gen_ds_ret = str(tar)
    missing_model = tmp_path / "nope.pth"

    settings = {
        "src": str(tmp_path),
        "train": False, "test": False,
        "generate_training_dataset": False,
        "apply_model_to_dataset": True,
        "tar_path": os.path.join("relative", "ds.tar"),
        "model_path": str(missing_model),
    }
    deep_spacr(settings)

    assert len(ds_stubs.gen_ds) == 1             # relative -> regenerated
    assert settings["tar_path"] == str(tar)
    assert ds_stubs.apply_tar == []              # model missing -> skipped
    assert ds_stubs.top == []
    assert ds_stubs.merge == []
    out = capsys.readouterr().out
    assert "not found; skipping model application" in out


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

    # The saved object is a usable model, not a bare state dict.
    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    assert isinstance(reloaded, TorchModel)
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
            data_loader=None,
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

    # The saved file is the full model object, and it still runs.
    reloaded = torch.load(tmp_path / f"fused_{aggregator}.pth",
                          map_location="cpu", weights_only=False)
    assert isinstance(reloaded, TorchModel)


def test_model_fusion_of_whole_torchmodel_checkpoints(tmp_path):
    """Checkpoints saved as TorchModel objects fuse through the non-dict branch."""
    from spacr.deep_spacr import model_fusion
    from spacr.utils import TorchModel

    for name, value in (("a.pth", 2.0), ("b.pth", 8.0)):
        m = _torchmodel()
        m.load_state_dict(_const_state_dict(m, value))
        torch.save(m, str(tmp_path / name))

    fused = model_fusion([str(tmp_path / "a.pth"), str(tmp_path / "b.pth")],
                         str(tmp_path / "fused.pth"), device="cpu",
                         model_name=_MODEL, pretrained=False, aggregator="mean")

    assert isinstance(fused, TorchModel)
    assert (tmp_path / "fused_mean.pth").exists()
    sd = fused.state_dict()
    float_keys = [k for k, v in sd.items() if v.is_floating_point()]
    assert float_keys
    for k in float_keys:
        assert torch.allclose(sd[k], torch.full_like(sd[k], 5.0)), k


def test_model_fusion_rejects_invalid_aggregator_name(tmp_path):
    """The aggregator whitelist rejects unknown names before any model is loaded."""
    from spacr.deep_spacr import model_fusion

    p1 = _save_dict_ckpt(tmp_path / "m1.pth", 1.0)
    with pytest.raises(ValueError, match="Invalid aggregator 'bogus'"):
        model_fusion([p1], str(tmp_path / "f.pth"), device="cpu",
                     aggregator="bogus")
    assert not (tmp_path / "f_bogus.pth").exists()


def test_model_fusion_rejects_unsupported_first_checkpoint(tmp_path):
    from spacr.deep_spacr import model_fusion

    bad = tmp_path / "bad.pth"
    torch.save("definitely-not-a-model", str(bad))

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        model_fusion([str(bad)], str(tmp_path / "f.pth"), device="cpu",
                     model_name=_MODEL, pretrained=False, aggregator="mean")
    assert not (tmp_path / "f_mean.pth").exists()


def test_model_fusion_rejects_unsupported_later_checkpoint(tmp_path):
    from spacr.deep_spacr import model_fusion

    good = _save_dict_ckpt(tmp_path / "m1.pth", 1.0)
    bad = tmp_path / "bad.pth"
    torch.save(("tuple", "checkpoint"), str(bad))

    with pytest.raises(ValueError, match="must be dict or TorchModel"):
        model_fusion([good, str(bad)], str(tmp_path / "f.pth"), device="cpu",
                     aggregator="mean")
    assert not (tmp_path / "f_mean.pth").exists()


def test_model_fusion_rejects_mismatched_state_dict_keys(tmp_path):
    from spacr.deep_spacr import model_fusion

    p1 = _save_dict_ckpt(tmp_path / "m1.pth", 2.0)
    truncated = _const_state_dict(_torchmodel(), 3.0)
    dropped = next(iter(truncated))
    truncated.pop(dropped)
    p2 = _save_dict_ckpt(tmp_path / "m2.pth", 3.0, state_dict=truncated)

    with pytest.raises(ValueError, match="identical architecture"):
        model_fusion([p1, p2], str(tmp_path / "f.pth"), device="cpu",
                     aggregator="mean")
    assert not (tmp_path / "f_mean.pth").exists()


class _AggregatorThatChangesItsMind(str):
    """A str subclass that compares equal exactly once.

    ``model_fusion`` whitelists the aggregator up front, which makes the
    defensive ``else: raise ValueError`` inside its nested
    ``combine_tensors`` unreachable for any plain string. This object
    passes the whitelist membership test (first ``__eq__``) and then
    stops matching, so the fall-through guard actually executes.
    """

    def __new__(cls, value="mean"):
        return super().__new__(cls, value)

    def __init__(self, value="mean"):
        self.n_eq = 0

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        self.n_eq += 1
        return self.n_eq == 1 and str(self) == other


def test_model_fusion_combine_tensors_guards_unknown_mode(tmp_path):
    """The fall-through guard in combine_tensors raises rather than returning None."""
    from spacr.deep_spacr import model_fusion

    p1 = _save_dict_ckpt(tmp_path / "m1.pth", 2.0)
    sneaky = _AggregatorThatChangesItsMind("mean")

    with pytest.raises(ValueError, match="Unsupported aggregator"):
        model_fusion([p1], str(tmp_path / "f.pth"), device="cpu",
                     aggregator=sneaky)
    # It got past the whitelist (proving the guard, not the whitelist, fired).
    assert sneaky.n_eq > 1


# ---------------------------------------------------------------------------
# annotate_filter_vision
# ---------------------------------------------------------------------------

def _write_scores_csv(path, preds, names=None):
    """A vision-score CSV with the metadata columns annotate_conditions needs."""
    n = len(preds)
    if names is None:
        names = [f"plate1_r{i % 2 + 1}_c{i % 2 + 1}_f1_o{i}.png" for i in range(n)]
    df = pd.DataFrame({
        "path": names,
        "plateID": ["plate1"] * n,
        "rowID": [f"r{i % 2 + 1}" for i in range(n)],
        "columnID": [f"c{i % 2 + 1}" for i in range(n)],
        "fieldID": [1] * n,
        "pred": preds,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _make_training_pngs(datasets_dir, nc_names, pc_names, extra_txt=True):
    train = datasets_dir / "training" / "train"
    for sub, names in (("nc", nc_names), ("pc", pc_names)):
        d = train / sub
        d.mkdir(parents=True, exist_ok=True)
        for nm in names:
            (d / nm).write_bytes(b"")
    if extra_txt:
        # a non-PNG in the folder must be ignored by the extension filter
        (train / "nc" / "notes.txt").write_bytes(b"ignore me")
    return train


def test_annotate_filter_vision_filters_thresholds_and_training_pngs(tmp_path):
    """str src -> list, condition annotation, threshold filter, train-PNG removal."""
    from spacr.deep_spacr import annotate_filter_vision

    datasets = tmp_path / "datasets"
    csv = datasets / "scores.csv"
    preds = [0.05, 0.10, 0.50, 0.60, 0.90, 0.95]
    df_in = _write_scores_csv(csv, preds)
    names = list(df_in["path"])

    # Two of the four rows that survive the threshold filter were training imgs.
    _make_training_pngs(datasets, nc_names=[names[0]], pc_names=[names[4]])

    settings = {
        "src": str(csv),
        "cells": ["HeLa"], "cell_loc": [["c1", "c2"]],
        "pathogens": ["wt", "dgra"], "pathogen_loc": [["c1"], ["c2"]],
        "treatments": ["untreated", "drug"], "treatment_loc": [["r1"], ["r2"]],
        "filter_column": "pred", "upper_threshold": 0.8, "lower_threshold": 0.2,
        "remove_train": True,
    }
    annotate_filter_vision(settings)

    # The string src was normalised to a list in place.
    assert settings["src"] == [str(csv)]

    out_csv = datasets / "scores_annotated_filtered.csv"
    assert out_csv.exists()
    out = pd.read_csv(out_csv)

    # threshold filter kept 4 rows; remove_train dropped the 2 training PNGs.
    assert sorted(out["path"]) == sorted([names[1], names[5]])

    # annotate_conditions ran: host cell / pathogen / treatment / condition.
    assert set(out["host_cells"]) == {"HeLa"}
    assert out.loc[out["path"] == names[1], "pathogen"].iloc[0] == "dgra"
    assert out.loc[out["path"] == names[1], "treatment"].iloc[0] == "drug"
    assert out.loc[out["path"] == names[1], "condition"].iloc[0] == "HeLa_dgra_drug"
    assert out.loc[out["path"] == names[5], "condition"].iloc[0] == "HeLa_dgra_drug"


def test_annotate_filter_vision_missing_filter_column_keeps_all_rows(tmp_path,
                                                                     capsys):
    """An unknown filter_column is reported and leaves the frame untouched."""
    from spacr.deep_spacr import annotate_filter_vision

    csv = tmp_path / "datasets" / "scores.csv"
    _write_scores_csv(csv, [0.1, 0.5, 0.9])

    settings = {
        "src": [str(csv)],                      # already a list -> no wrapping
        "cells": "HeLa", "cell_loc": None,
        "pathogens": None, "pathogen_loc": None,
        "treatments": None, "treatment_loc": None,
        "filter_column": "does_not_exist",
        "upper_threshold": 0.8, "lower_threshold": 0.2,
        "remove_train": False,
    }
    annotate_filter_vision(settings)

    out = pd.read_csv(tmp_path / "datasets" / "scores_annotated_filtered.csv")
    assert len(out) == 3
    assert set(out["condition"]) == {"HeLa"}
    assert "does_not_exist not in DataFrame columns" in capsys.readouterr().out


def test_annotate_filter_vision_no_filter_multiple_sources(tmp_path):
    """filter_column=None keeps every row; every src in the list is processed."""
    from spacr.deep_spacr import annotate_filter_vision

    datasets = tmp_path / "datasets"
    csv_a = datasets / "a.csv"
    csv_b = datasets / "b.csv"
    _write_scores_csv(csv_a, [0.2, 0.4])
    _write_scores_csv(csv_b, [0.6, 0.8, 0.99])

    settings = {
        "src": [str(csv_a), str(csv_b)],
        "cells": ["HeLa"], "cell_loc": [["c1", "c2"]],
        "pathogens": ["wt"], "pathogen_loc": [["c1", "c2"]],
        "treatments": ["untreated"], "treatment_loc": [["r1", "r2"]],
        "filter_column": None,
        "upper_threshold": 0.8, "lower_threshold": 0.2,
        # remove_train with no training folder at all: nothing to remove.
        "remove_train": True,
    }
    annotate_filter_vision(settings)

    out_a = pd.read_csv(datasets / "a_annotated_filtered.csv")
    out_b = pd.read_csv(datasets / "b_annotated_filtered.csv")
    assert len(out_a) == 2 and len(out_b) == 3
    assert set(out_a["condition"]) == {"HeLa_wt_untreated"}
    assert "pred" in out_b.columns
