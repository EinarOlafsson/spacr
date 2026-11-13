"""Inference, merging and pipeline entry points of :mod:`spacr.deep_spacr`.

One table-driven test per unit, each row run as a subtest so a failing case
names itself.  Units: ``apply_model``, ``apply_model_to_tar``,
``merge_predictions_into_db``, ``deep_spacr`` (the orchestrator),
``model_fusion`` and ``annotate_filter_vision``.

The inference tests use a deterministic model whose logit is
``LOGIT_SCALE * mean(pixels)`` and crops whose pixels are constant, so every
probability is re-derived here from the torchvision transform maths
(``x/255`` without normalisation, ``(x/255 - 0.5)/0.5`` with it): a dropped
or an added ``Normalize`` step changes the numbers.  Everything runs on CPU,
offline, with ``num_workers=0``.

The helpers ``_save_model``, ``_tar_settings``, ``_expected_prob``,
``_scalar`` and the fixtures ``const_png_dir`` / ``const_png_tar`` /
``_force_cpu`` are imported by other test modules.
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import deep_spacr as D  # noqa: E402
from spacr.resource_log import _guard_workers as _REAL_GUARD  # noqa: E402


# ===========================================================================
# deterministic model + synthetic crops
# ===========================================================================

#: multiplier applied to the mean pixel value to turn it into a logit
LOGIT_SCALE = 3.0

#: what ``pick_device`` says when a busy card sends the run to the CPU
_NOTE = ("The GPU has 300 MiB free of 24576 and this run needs about 2000 "
         "MiB, so it is running on the CPU instead.")


class _MeanLogitModel(torch.nn.Module):
    """Deterministic classifier: logit = ``LOGIT_SCALE * mean(pixels)``.

    ``n_out=1`` -> ``(N, 1)``; ``n_out=2`` -> ``(-l, +l)``;
    ``n_out=3`` -> ``(-l, 0, +l)``.  Module level so ``torch.save`` pickles
    it by reference, the way spaCR pickles its own ``TorchModel``.
    """

    def __init__(self, n_out: int = 1):
        super().__init__()
        self.n_out = int(n_out)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):  # noqa: D102 - trivial
        logit = x.flatten(1).mean(dim=1, keepdim=True) * LOGIT_SCALE + self.bias
        if self.n_out == 2:
            return torch.cat([-logit, logit], dim=1)
        if self.n_out == 3:
            return torch.cat([-logit, torch.zeros_like(logit), logit], dim=1)
        return logit


def _save_model(path, n_out=1):
    torch.save(_MeanLogitModel(n_out=n_out), str(path))
    return str(path)


def _const_png(path, value, size=32):
    """Write a ``size x size`` RGB PNG whose every pixel equals ``value``."""
    Image.fromarray(np.full((size, size, 3), int(value), dtype=np.uint8)).save(str(path))
    return str(path)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.float64(x)))


def _scaled(value, normalize):
    """The tensor value ``ToTensor`` (and optionally ``Normalize(.5, .5)``) gives."""
    v = np.float32(value) / np.float32(255.0)
    if normalize:
        v = (v - np.float32(0.5)) / np.float32(0.5)
    return v


def _expected_prob(value, normalize):
    """Probability a single-logit head must emit for a constant crop."""
    return _sigmoid(np.float32(LOGIT_SCALE) * _scaled(value, normalize))


def _expected_head(value, normalize, n_out):
    """``(pred, predicted_label, class probabilities)`` for each head size."""
    logit = float(np.float32(LOGIT_SCALE) * _scaled(value, normalize))
    if n_out == 1:
        p = _sigmoid(logit)
        return p, None, None
    logits = np.array([-logit, logit] if n_out == 2 else [-logit, 0.0, logit])
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    if n_out == 2:
        return probs[1], None, None
    return probs.max(), int(np.argmax(probs)), probs


def _scalar(pred):
    """``apply_model`` keeps the trailing head axis, so preds may be ``[p]``."""
    if isinstance(pred, (list, tuple, np.ndarray)):
        assert len(pred) == 1, f"unexpected multi-element pred {pred!r}"
        return float(pred[0])
    return float(pred)


PIXEL_VALUES = (0, 51, 128, 204, 255)


@pytest.fixture
def const_png_dir(tmp_path):
    """Directory of constant-valued PNGs; returns ``(dir, {path: value})``."""
    d = tmp_path / "crops"
    d.mkdir()
    mapping = {}
    for i, v in enumerate(PIXEL_VALUES):
        mapping[_const_png(d / f"plate1_A01_f1_o{i}.png", v)] = v
    return str(d), mapping


@pytest.fixture
def const_png_tar(tmp_path):
    """Tar of constant-valued PNGs; returns ``(tar_path, {member: value})``."""
    stage = tmp_path / "stage"
    stage.mkdir()
    tar_path = tmp_path / "dataset.tar"
    mapping = {}
    with tarfile.open(tar_path, "w") as tar:
        for i, v in enumerate(PIXEL_VALUES):
            name = f"plate1_A01_f1_o{i}.png"
            _const_png(stage / name, v)
            tar.add(str(stage / name), arcname=name)
            mapping[name] = v
    return str(tar_path), mapping


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Pin inference to the CPU even on a CUDA-capable developer box."""
    import spacr.accelerator as accelerator

    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _tar_settings(tar_path, model_path, **over):
    settings = {
        "tar_path": tar_path, "model_path": model_path, "image_size": 32,
        "batch_size": 2, "normalize": True, "n_jobs": 0, "verbose": False,
        "score_threshold": 0.5,
    }
    settings.update(over)
    return settings


@pytest.fixture
def guarded(monkeypatch):
    """Record every RAM-guard request for loader workers; grant none."""
    import spacr.resource_log as resource_log

    calls = []

    def guard(task, workers, unit, **kwargs):
        calls.append((task, workers))
        return 0

    monkeypatch.setattr(resource_log, "_guard_workers", guard)
    return calls


def _busy_card(monkeypatch):
    """Every ``pick_device`` call reports a CPU fallback with a note."""
    monkeypatch.setattr(D, "pick_device",
                        lambda *a, **k: (torch.device("cpu"), _NOTE))


# ===========================================================================
# apply_model
# ===========================================================================

def test_apply_model_scores_every_crop_through_the_requested_transform(
        const_png_dir, tmp_path, monkeypatch, capsys, guarded, subtests):
    """Each normalisation setting gives the exactly re-derived probability for
    every crop, the frame round-trips through the dated CSV beside the model,
    workers go through the RAM guard and a CPU fallback is announced first."""
    src, values = const_png_dir
    model_path = _save_model(tmp_path / "m.pth", n_out=1)
    csv = (os.path.splitext(model_path)[0]
           + datetime.date.today().strftime("%y%m%d") + "_.pth_test_result.csv")

    cases = [
        # (id, kwargs, normalised?, expected guard calls)
        ("raw-tensor-scale", dict(normalize=False, batch_size=2, n_jobs=0), False, []),
        ("symmetric-normalize", dict(normalize=True, batch_size=8, n_jobs=0), True, []),
        ("statistics-none-is-raw", dict(normalize=True, batch_size=8, n_jobs=0,
                                        input_statistics="none"), False, []),
        ("workers-are-guarded", dict(normalize=False, batch_size=3, n_jobs=3),
         False, [("classify", 3)]),
    ]
    for cid, kwargs, normalised, guard_calls in cases:
        with subtests.test(case=cid):
            guarded.clear()
            df = D.apply_model(src, model_path, image_size=32, **kwargs)
            assert list(df.columns) == ["path", "pred"]
            assert sorted(df["path"]) == sorted(values)
            got = {p: _scalar(v) for p, v in zip(df["path"], df["pred"])}
            for path, value in values.items():
                assert got[path] == pytest.approx(
                    _expected_prob(value, normalised), abs=1e-6), path
            assert guarded == guard_calls
            on_disk = pd.read_csv(csv, index_col=0)
            assert list(on_disk["path"]) == list(df["path"])

    with subtests.test(case="black-is-exactly-even-only-without-normalize"):
        black = next(p for p, v in values.items() if v == 0)
        raw = D.apply_model(src, model_path, image_size=32, normalize=False, n_jobs=0)
        norm = D.apply_model(src, model_path, image_size=32, normalize=True, n_jobs=0)
        assert _scalar(raw.set_index("path").loc[black, "pred"]) == pytest.approx(0.5, abs=1e-7)
        assert _scalar(norm.set_index("path").loc[black, "pred"]) == pytest.approx(
            _sigmoid(-LOGIT_SCALE), abs=1e-6)

    with subtests.test(case="cpu-fallback-is-announced-first"):
        _busy_card(monkeypatch)
        capsys.readouterr()
        D.apply_model(src, model_path, image_size=32, normalize=False, n_jobs=0)
        assert capsys.readouterr().out.splitlines()[0] == _NOTE

    with subtests.test(case="empty-folder-is-refused"):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match=r"No images to score in .*empty"):
            D.apply_model(str(empty), model_path, image_size=32, n_jobs=0)


# ===========================================================================
# apply_model_to_tar
# ===========================================================================

def test_apply_model_to_tar_scores_each_head_through_the_requested_transform(
        const_png_tar, tmp_path, monkeypatch, capsys, guarded, subtests):
    """Single-logit (sigmoid, squeezed), two-class (softmax column 1) and
    three-class (winning probability + label) heads give the re-derived value
    for every member under each normalisation setting, enriched with plate and
    well and thresholded, and the result CSV lands beside the tar."""
    tar_path, values = const_png_tar
    date = datetime.date.today().strftime("%y%m%d")

    for cid, settings in (("settings-none", None), ("settings-empty", {})):
        with subtests.test(case=cid):
            with pytest.raises(KeyError) as excinfo:
                D.apply_model_to_tar(settings)
            assert excinfo.value.args == ("tar_path",)

    cases = [
        # (id, n_out, overrides, normalised?, guard calls)
        ("one-logit-symmetric", 1, dict(normalize=True, verbose=True), True, []),
        ("one-logit-batch-of-one-raw", 1, dict(normalize=False, batch_size=1), False, []),
        ("one-logit-statistics-none", 1, dict(normalize=True, input_statistics="none"),
         False, []),
        ("two-class-raw-threshold-0.6", 2, dict(normalize=False, score_threshold=0.6),
         False, []),
        ("two-class-symmetric", 2, dict(normalize=True), True, []),
        ("three-class-raw", 3, dict(normalize=False), False, []),
        ("workers-are-guarded", 1, dict(normalize=False, n_jobs=2), False,
         [("classify", 2)]),
    ]
    for cid, n_out, over, normalised, guard_calls in cases:
        with subtests.test(case=cid):
            guarded.clear()
            model_path = _save_model(tmp_path / f"head{n_out}.pth", n_out=n_out)
            settings = _tar_settings(tar_path, model_path, **over)
            df = D.apply_model_to_tar(settings)

            assert sorted(df["path"]) == sorted(values)
            assert all(isinstance(p, float) for p in df["pred"])
            rows = df.set_index("path")
            for member, value in values.items():
                pred, label, probs = _expected_head(value, normalised, n_out)
                assert rows.loc[member, "pred"] == pytest.approx(pred, abs=1e-6)
                if n_out == 3:
                    assert rows.loc[member, "predicted_label"] == label
                    assert rows.loc[member, "cv_predictions"] == label
                    np.testing.assert_allclose(
                        rows.loc[member, ["prob_class_0", "prob_class_1",
                                          "prob_class_2"]].to_numpy(float),
                        probs, atol=1e-6)
            if n_out < 3:
                threshold = settings["score_threshold"]
                assert list(df["cv_predictions"]) == [int(p >= threshold) for p in df["pred"]]
                assert "prob_class_0" not in df.columns
            assert set(df["plateID"]) == {"plate1"}
            assert set(df["prc"]) == {"plate1_r1_c1"}
            assert guarded == guard_calls
            csv = os.path.join(os.path.dirname(tar_path),
                               f"{date}_dataset_head{n_out}_result.csv")
            assert list(pd.read_csv(csv, index_col=0)["path"]) == list(df["path"])

    with subtests.test(case="ram-guard-false-in-settings-keeps-every-worker"), \
            pytest.MonkeyPatch.context() as mp:
        import spacr.resource_log as resource_log

        granted = []

        def real_guard(module, n_jobs, unit, **kwargs):
            granted.append((module, n_jobs, kwargs["settings"]["ram_guard"],
                            _REAL_GUARD(module, n_jobs, unit, **kwargs)))
            return 0  # keep the run in-process; the answer is what is checked

        mp.setattr(resource_log, "_guard_workers", real_guard)
        model_path = _save_model(tmp_path / "head1.pth", n_out=1)
        df = D.apply_model_to_tar(_tar_settings(tar_path, model_path, normalize=False,
                                                n_jobs=5, ram_guard=False))
        assert granted == [("classify", 5, False, 5)]
        assert sorted(df["path"]) == sorted(values)

    with subtests.test(case="normalised-and-raw-runs-disagree"):
        model_path = _save_model(tmp_path / "head2.pth", n_out=2)
        raw = D.apply_model_to_tar(_tar_settings(tar_path, model_path, normalize=False))
        norm = D.apply_model_to_tar(_tar_settings(tar_path, model_path, normalize=True))
        moved = (raw.set_index("path")["pred"] - norm.set_index("path")["pred"]).abs()
        assert int((moved > 1e-3).sum()) == len(values) - 1  # only mid-grey ~fixed

    with subtests.test(case="cpu-fallback-is-announced-first"):
        _busy_card(monkeypatch)
        capsys.readouterr()
        D.apply_model_to_tar(_tar_settings(tar_path, _save_model(tmp_path / "n.pth")))
        assert capsys.readouterr().out.splitlines()[0] == _NOTE


# ===========================================================================
# merge_predictions_into_db
# ===========================================================================

def _make_db(path, rows, table="png_list", extra=""):
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE {table} (png_path TEXT, prcfo TEXT{extra})")
    width = 2 + extra.count(",")
    con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * width)})", rows)
    con.commit()
    con.close()
    return str(path)


def _columns(db, table="png_list"):
    with sqlite3.connect(db) as con:
        return [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({table})")]


def _rows(db, cols, table="png_list"):
    with sqlite3.connect(db) as con:
        return list(con.execute(f"SELECT {cols} FROM {table} ORDER BY rowid"))


def test_merge_predictions_into_db_writes_matching_rows_only(tmp_path, capsys,
                                                             subtests):
    """Scores land on the rows whose png_path matches, existing columns are
    reused, NULL/empty paths and unmatched rows stay NULL, custom table and
    column names are honoured and a missing database is reported as None."""
    cases = [
        # (id, db rows, extra cols, frame, kwargs, matched, columns, values, printed)
        ("updates-matching-rows",
         [("/d/a.png", "p1_A01_1_o1"), ("/d/b.png", "p1_A01_1_o2"),
          ("/d/unseen.png", "p1_A01_1_o3")], "",
         dict(path=["/d/a.png", "/d/b.png", "/d/not_in_db.png"],
              pred=[0.1, 0.85, 0.5], cv_predictions=[0, 1, 1]), {},
         2, [("png_path", "TEXT"), ("prcfo", "TEXT"), ("pred", "REAL"),
             ("cv_predictions", "INTEGER")],
         [(0.1, 0), (0.85, 1), (None, None)], "2/3 rows matched"),
        ("reuses-existing-column",
         [("/x/a.png", "p1", 0.999), ("/x/b.png", "p2", 0.111)], ", pred REAL",
         dict(path=["/x/a.png", "/x/b.png"], pred=[0.2, 0.7], cv_predictions=[0, 1]),
         {}, 2, [("png_path", "TEXT"), ("prcfo", "TEXT"), ("pred", "REAL"),
                 ("cv_predictions", "INTEGER")],
         [(0.2, 0), (0.7, 1)], "2/2 rows matched"),
        ("skips-null-and-empty-paths",
         [(None, "p1"), ("/x/a.png", "p2"), ("", "p3")], "",
         dict(path=["/x/a.png"], pred=[0.33], cv_predictions=[0]), {},
         1, None, [(None, None), (0.33, 0), (None, None)], "1/3 rows matched"),
        ("no-overlap-leaves-nulls",
         [("/x/a.png", "p1"), ("/x/b.png", "p2")], "",
         dict(path=["/x/zzz.png"], pred=[0.9], cv_predictions=[1]), {},
         0, None, [(None, None), (None, None)], "0/2 rows matched"),
    ]
    for cid, db_rows, extra, frame, kwargs, matched, columns, values, printed in cases:
        with subtests.test(case=cid):
            db = _make_db(tmp_path / f"{cid}.db", db_rows, extra=extra)
            df = pd.DataFrame(frame)
            before = df.copy()
            assert D.merge_predictions_into_db(df, db, **kwargs) == matched
            if columns is not None:
                assert _columns(db) == columns
            got = _rows(db, "pred, cv_predictions")
            assert [(None if p is None else pytest.approx(p), c) for p, c in got] == values
            assert printed in capsys.readouterr().out
            pd.testing.assert_frame_equal(df, before)

    with subtests.test(case="custom-table-and-columns"):
        db = _make_db(tmp_path / "custom.db", [("/x/a.png", "p1")], table="object_png")
        df = pd.DataFrame({"path": ["/x/a.png"], "score": [0.6], "label": [1]})
        assert D.merge_predictions_into_db(df, db, table="object_png",
                                           pred_col="score", class_col="label") == 1
        assert _columns(db, "object_png") == [
            ("png_path", "TEXT"), ("prcfo", "TEXT"), ("score", "REAL"),
            ("label", "INTEGER")]
        assert _rows(db, "score, label", "object_png") == [(pytest.approx(0.6), 1)]

    with subtests.test(case="missing-database-is-none"):
        missing = tmp_path / "nope" / "measurements.db"
        df = pd.DataFrame({"path": ["a.png"], "pred": [0.9], "cv_predictions": [1]})
        assert D.merge_predictions_into_db(df, str(missing)) is None
        assert "Database not found" in capsys.readouterr().out
        assert not missing.exists()

    with subtests.test(case="missing-table-raises"):
        db = tmp_path / "other.db"
        with sqlite3.connect(db) as con:
            con.execute("CREATE TABLE something_else (x INTEGER)")
        df = pd.DataFrame({"path": ["a.png"], "pred": [0.1], "cv_predictions": [0]})
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            D.merge_predictions_into_db(df, str(db))


# ===========================================================================
# deep_spacr (the orchestrator)
# ===========================================================================

class _Stages:
    """Recorders for the collaborators ``deep_spacr`` calls."""

    def __init__(self):
        self.gen_train_ret = (None, None)
        self.gen_ds_ret = None
        self.train_ret = None
        self.apply_ret = None
        self.save_settings, self.gen_train, self.gen_ds = [], [], []
        self.train, self.apply_tar, self.top, self.merge = [], [], [], []


def _stub_stages(monkeypatch):
    """Replace every stage of ``deep_spacr`` with a recorder; return it."""
    s = _Stages()
    monkeypatch.setattr("spacr.utils.save_settings",
                        lambda settings, name="settings", show=False:
                        s.save_settings.append((dict(settings), name)))
    monkeypatch.setattr("spacr.io.generate_training_dataset",
                        lambda settings: (s.gen_train.append(dict(settings)),
                                          s.gen_train_ret)[1])
    monkeypatch.setattr("spacr.io.generate_dataset",
                        lambda settings: (s.gen_ds.append(dict(settings)),
                                          s.gen_ds_ret)[1])
    monkeypatch.setattr(D, "train_test_model",
                        lambda settings: (s.train.append(dict(settings)),
                                          s.train_ret)[1])
    monkeypatch.setattr(D, "apply_model_to_tar",
                        lambda settings: (s.apply_tar.append(dict(settings)),
                                          s.apply_ret)[1])
    monkeypatch.setattr(D, "save_top_class_examples",
                        lambda df, tar, dst, n=20, classes=None:
                        s.top.append((df, tar, dst, n, classes)))
    monkeypatch.setattr(D, "merge_predictions_into_db",
                        lambda df, db: s.merge.append((df, db)))
    return s


def test_deep_spacr_runs_exactly_the_stages_its_flags_ask_for(tmp_path, monkeypatch,
                                                              capsys, subtests):
    """Each flag combination calls exactly the stages it asks for, hands each
    the derived paths (train parent, fresh tar, top_examples beside the tar,
    one database per src) and skips or refuses when its inputs are missing."""
    from spacr.model_zoo import ModelZooError

    with subtests.test(case="defaults-abort-when-the-split-fails"):
        s = _stub_stages(monkeypatch)
        assert D.deep_spacr() is None
        (saved, name), = s.save_settings
        assert (name, saved["src"], saved["generate_training_dataset"],
                saved["train"]) == ("DL_model", "path", True, True)
        assert len(s.gen_train) == 1
        assert s.train == s.gen_ds == s.apply_tar == []
        assert ("Training dataset generation failed; skipping model training step."
                in capsys.readouterr().out)

    with subtests.test(case="full-pipeline-wires-every-stage"):
        root = tmp_path / "full"
        train_dir = root / "datasets" / "training" / "train"
        train_dir.mkdir(parents=True)
        (root / "datasets" / "training" / "test").mkdir()
        model = root / "model" / "clf.pth"
        model.parent.mkdir()
        model.write_bytes(b"not-really-a-model")
        tar = root / "datasets" / "ds.tar"
        tar.write_bytes(b"not-really-a-tar")
        s = _stub_stages(monkeypatch)
        s.gen_train_ret = (str(train_dir), str(train_dir.parent / "test"))
        s.train_ret, s.gen_ds_ret = str(model), str(tar)
        s.apply_ret = df = pd.DataFrame({"path": ["a.png"], "pred": [0.9]})
        settings = {"src": str(root), "train": True, "test": True,
                    "generate_training_dataset": True, "apply_model_to_dataset": True,
                    "tar_path": None, "n_top_examples": 3}
        D.deep_spacr(settings)
        assert [t["src"] for t in s.train] == [str(train_dir.parent)]
        assert (settings["src"], settings["model_path"], settings["tar_path"]) == (
            str(root), str(model), str(tar))
        assert len(s.gen_ds) == 1
        assert [(a["tar_path"], a["model_path"]) for a in s.apply_tar] == [
            (str(tar), str(model))]
        assert s.top == [(df, str(tar), str(tar.parent / "top_examples"), 3,
                          ["nc", "pc"])]
        assert s.merge == [(df, os.path.join(str(root), "measurements",
                                             "measurements.db"))]

    with subtests.test(case="apply-only-reuses-the-tar-for-every-src"):
        root = tmp_path / "apply"
        root.mkdir()
        tar, model = root / "ds.tar", root / "m.pth"
        tar.write_bytes(b"tar")
        model.write_bytes(b"model")
        s = _stub_stages(monkeypatch)
        s.apply_ret = pd.DataFrame({"path": ["a.png"], "pred": [0.4]})
        srcs = [str(root / "plateA"), str(root / "plateB")]
        D.deep_spacr({"src": srcs, "train": False, "test": False,
                      "generate_training_dataset": False,
                      "apply_model_to_dataset": True, "tar_path": str(tar),
                      "model_path": str(model)})
        assert s.gen_train == s.train == s.gen_ds == []
        assert len(s.apply_tar) == 1
        assert [db for _df, db in s.merge] == [
            os.path.join(p, "measurements", "measurements.db") for p in srcs]

    with subtests.test(case="relative-tar-regenerated-missing-model-refused"):
        root = tmp_path / "relative"
        root.mkdir()
        tar = root / "ds.tar"
        tar.write_bytes(b"tar")
        missing = root / "nope.pth"
        s = _stub_stages(monkeypatch)
        s.gen_ds_ret = str(tar)
        settings = {"src": str(root), "train": False, "test": False,
                    "generate_training_dataset": False, "apply_model_to_dataset": True,
                    "tar_path": os.path.join("relative", "ds.tar"),
                    "model_path": str(missing)}
        capsys.readouterr()
        with pytest.raises(ModelZooError) as refused:
            D.deep_spacr(settings)
        assert len(s.gen_ds) == 1 and settings["tar_path"] == str(tar)
        assert s.apply_tar == s.top == s.merge == []
        assert "Generating full dataset tar" in capsys.readouterr().out
        message = str(refused.value)
        assert "model_path" in message and repr(str(missing)) in message
        assert "Available compatible zoo identities: []." in message

    with subtests.test(case="full-dataset-without-applying"):
        tar = tmp_path / "only" / "datasets" / "full.tar"
        tar.parent.mkdir(parents=True)
        tar.write_bytes(b"tar")
        s = _stub_stages(monkeypatch)
        s.gen_ds_ret = str(tar)
        settings = {"src": str(tmp_path / "only"), "train": False, "test": False,
                    "generate_training_dataset": False, "generate_full_dataset": True,
                    "apply_model_to_dataset": False, "tar_path": ""}
        D.deep_spacr(settings)
        assert len(s.gen_ds) == 1 and settings["tar_path"] == str(tar)
        assert s.apply_tar == []

    with subtests.test(case="unreadable-generated-tar-is-an-error"):
        s = _stub_stages(monkeypatch)
        s.gen_ds_ret = str(tmp_path / "never_written.tar")
        with pytest.raises(RuntimeError, match="did not produce a readable tar file"):
            D.deep_spacr({"src": str(tmp_path), "train": False, "test": False,
                          "generate_training_dataset": False,
                          "generate_full_dataset": True, "tar_path": ""})

    for requested in (None, "", "   "):
        with subtests.test(case=f"no-model-selected-{requested!r}"):
            root = tmp_path / f"skip{len(requested or 'x')}{requested is None}"
            (root / "measurements").mkdir(parents=True)
            tar = root / "existing.tar"
            tar.write_bytes(b"existing reusable inference dataset")
            database = root / "measurements" / "measurements.db"
            database.write_bytes(b"existing measurement database")
            before = {p: p.read_bytes() for p in (tar, database)}
            s = _stub_stages(monkeypatch)
            settings = {"src": str(root), "train": False, "test": False,
                        "generate_training_dataset": False,
                        "generate_full_dataset": False, "apply_model_to_dataset": True,
                        "tar_path": str(tar), "model_path": requested}
            capsys.readouterr()
            assert D.deep_spacr(settings) is None
            assert len(s.save_settings) == 1
            assert s.gen_train == s.train == s.gen_ds == []
            assert s.apply_tar == s.top == s.merge == []
            assert (settings["model_path"], settings["tar_path"], settings["src"]) == (
                requested, str(tar), str(root))
            assert {p: p.read_bytes() for p in before} == before
            assert (f"Model path {requested} not found; skipping model application."
                    in capsys.readouterr().out)


# ===========================================================================
# model_fusion
# ===========================================================================

_FUSE_MODEL = "shufflenet_v2_x0_5"      # ~350k params, no download with pretrained=False


def _torchmodel():
    from spacr.utils import TorchModel
    return TorchModel(model_name=_FUSE_MODEL, pretrained=False, num_classes=2)


def _const_state_dict(model, value):
    """A state dict with the model's exact keys/shapes, every entry == value."""
    return {k: torch.full_like(v, float(value) if v.is_floating_point() else int(value))
            for k, v in model.state_dict().items()}


def _save_dict_ckpt(path, state_dict):
    torch.save({"model": state_dict, "model_name": _FUSE_MODEL, "pretrained": False,
                "dropout_rate": None, "use_checkpoint": False}, str(path))
    return str(path)


class _AggregatorThatChangesItsMind(str):
    """A str that compares equal exactly once: it passes the whitelist and
    then reaches the fall-through guard inside ``combine_tensors``."""

    def __init__(self, value="mean"):
        self.n_eq = 0

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        self.n_eq += 1
        return self.n_eq == 1 and str(self) == other


def test_model_fusion_fuses_whole_models_and_refuses_what_it_cannot_fuse(
        tmp_path, subtests):
    """Whole-TorchModel checkpoints fuse to the exact element-wise mean and
    are saved as ``<stem>_<aggregator>.pth``; every unsupported input is
    refused with its own message and nothing is written."""
    template = _torchmodel()
    for name, value in (("a.pth", 2.0), ("b.pth", 8.0)):
        model = _torchmodel()
        model.load_state_dict(_const_state_dict(template, value))
        torch.save(model, str(tmp_path / name))
    whole = [str(tmp_path / "a.pth"), str(tmp_path / "b.pth")]

    with subtests.test(case="whole-torchmodels-mean"):
        fused = D.model_fusion(whole, str(tmp_path / "fused.pth"), device="cpu",
                               model_name=_FUSE_MODEL, pretrained=False,
                               aggregator="mean")
        assert type(fused).__name__ == "TorchModel"
        assert (tmp_path / "fused_mean.pth").is_file()
        assert not (tmp_path / "fused.pth_mean.pth").exists()
        floats = {k: v for k, v in fused.state_dict().items() if v.is_floating_point()}
        assert floats
        for key, tensor in floats.items():
            assert torch.equal(tensor, torch.full_like(tensor, 5.0)), key

    bad_first = tmp_path / "bad_first.pth"
    torch.save("definitely-not-a-model", str(bad_first))
    bad_later = tmp_path / "bad_later.pth"
    torch.save(("tuple", "checkpoint"), str(bad_later))
    good = _save_dict_ckpt(tmp_path / "m1.pth", _const_state_dict(template, 1.0))
    truncated = _const_state_dict(template, 3.0)
    truncated.pop(next(iter(truncated)))
    short = _save_dict_ckpt(tmp_path / "short.pth", truncated)
    from spacr.utils import TorchModel
    torch.save(TorchModel(model_name=_FUSE_MODEL, pretrained=False, num_classes=3),
               str(tmp_path / "three_head.pth"))

    refusals = [
        ("no-checkpoints", [], "mean",
         "model_paths must contain at least one checkpoint."),
        ("invalid-aggregator", [good], "bogus", "Invalid aggregator 'bogus'"),
        ("unsupported-first", [str(bad_first)], "mean",
         "Unsupported checkpoint format. Must be a spaCR artifact, legacy state "
         "dict, or TorchModel instance."),
        ("unsupported-later", [good, str(bad_later)], "mean",
         f"Unsupported checkpoint format in {bad_later}; it must be a spaCR "
         "artifact, dict or TorchModel with an identical architecture."),
        # a dict missing a key cannot even be loaded into the architecture
        ("missing-key-dict-is-unloadable", [good, short], "mean",
         f"Unsupported checkpoint format in {short}; it must be a spaCR "
         "artifact, dict or TorchModel with an identical architecture."),
        ("mismatched-head-shape", [whole[0], str(tmp_path / "three_head.pth")], "mean",
         "All models must have identical architecture/state_dict keys."),
    ]
    for cid, paths, aggregator, message in refusals:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as excinfo:
                D.model_fusion(paths, str(tmp_path / "f.pth"), device="cpu",
                               aggregator=aggregator)
            assert str(excinfo.value).startswith(message)
            assert sorted(p.name for p in tmp_path.glob("f_*.pth")) == []

    with subtests.test(case="combine-tensors-guards-an-unknown-mode"):
        sneaky = _AggregatorThatChangesItsMind("mean")
        with pytest.raises(ValueError, match="^Unsupported aggregator: mean$"):
            D.model_fusion([good], str(tmp_path / "f.pth"), device="cpu",
                           aggregator=sneaky)
        assert sneaky.n_eq > 1  # it got past the whitelist


# ===========================================================================
# annotate_filter_vision
# ===========================================================================

def _write_scores_csv(path, preds):
    """A vision-score CSV with the metadata columns annotate_conditions needs."""
    n = len(preds)
    df = pd.DataFrame({
        "path": [f"plate1_r{i % 2 + 1}_c{i % 2 + 1}_f1_o{i}.png" for i in range(n)],
        "plateID": ["plate1"] * n,
        "rowID": [f"r{i % 2 + 1}" for i in range(n)],
        "columnID": [f"c{i % 2 + 1}" for i in range(n)],
        "fieldID": [1] * n,
        "pred": preds,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return list(df["path"])


def _annotation(**over):
    settings = {"cells": ["HeLa"], "cell_loc": [["c1", "c2"]],
                "pathogens": None, "pathogen_loc": None,
                "treatments": None, "treatment_loc": None,
                "filter_column": None, "upper_threshold": 0.8,
                "lower_threshold": 0.2, "remove_train": False}
    settings.update(over)
    return settings


def test_annotate_filter_vision_annotates_filters_and_drops_training_crops(
        tmp_path, capsys, subtests):
    """Every src CSV gets its conditions annotated, rows inside the
    (lower, upper) band dropped when the filter column exists, training PNGs
    removed on request, and is written as ``<stem>_annotated_filtered.csv``."""
    preds = [0.05, 0.10, 0.50, 0.60, 0.90, 0.95]

    with subtests.test(case="threshold-and-training-png-removal"):
        datasets = tmp_path / "a" / "datasets"
        csv = datasets / "scores.csv"
        names = _write_scores_csv(csv, preds)
        train = datasets / "training" / "train"
        for sub, name in (("nc", names[0]), ("pc", names[4])):
            (train / sub).mkdir(parents=True)
            (train / sub / name).write_bytes(b"")
        (train / "nc" / names[1].replace(".png", ".txt")).write_bytes(b"not a png")
        settings = _annotation(src=str(csv), pathogens=["wt", "dgra"],
                               pathogen_loc=[["c1"], ["c2"]],
                               treatments=["untreated", "drug"],
                               treatment_loc=[["r1"], ["r2"]],
                               filter_column="pred", remove_train=True)
        D.annotate_filter_vision(settings)
        assert settings["src"] == [str(csv)]
        out = pd.read_csv(datasets / "scores_annotated_filtered.csv")
        assert list(out["path"]) == [names[1], names[5]]
        assert list(out["condition"]) == ["HeLa_dgra_drug", "HeLa_dgra_drug"]
        assert list(out["host_cells"]) == ["HeLa", "HeLa"]
        assert "Filtered DataFrame with 6 rows to 4 rows." in capsys.readouterr().out

    with subtests.test(case="unknown-filter-column-keeps-every-row"):
        csv = tmp_path / "b" / "datasets" / "scores.csv"
        _write_scores_csv(csv, [0.1, 0.5, 0.9])
        D.annotate_filter_vision(_annotation(src=[str(csv)], cells="HeLa",
                                             cell_loc=None,
                                             filter_column="does_not_exist"))
        out = pd.read_csv(csv.parent / "scores_annotated_filtered.csv")
        assert list(out["pred"]) == [0.1, 0.5, 0.9]
        assert list(out["condition"]) == ["HeLa"] * 3
        assert "does_not_exist not in DataFrame columns." in capsys.readouterr().out

    with subtests.test(case="no-filter-every-src-no-training-folder"):
        datasets = tmp_path / "c" / "datasets"
        _write_scores_csv(datasets / "a.csv", [0.2, 0.4])
        _write_scores_csv(datasets / "b.csv", [0.6, 0.8, 0.99])
        D.annotate_filter_vision(_annotation(
            src=[str(datasets / "a.csv"), str(datasets / "b.csv")],
            pathogens=["wt"], pathogen_loc=[["c1", "c2"]],
            treatments=["untreated"], treatment_loc=[["r1", "r2"]],
            remove_train=True))
        out_a = pd.read_csv(datasets / "a_annotated_filtered.csv")
        out_b = pd.read_csv(datasets / "b_annotated_filtered.csv")
        assert list(out_a["condition"]) == ["HeLa_wt_untreated"] * 2
        assert list(out_b["pred"]) == [0.6, 0.8, 0.99]

    with subtests.test(case="remove-train-outside-a-datasets-folder"):
        csv = tmp_path / "elsewhere" / "scores.csv"
        _write_scores_csv(csv, [0.1, 0.9])
        with pytest.raises(ValueError, match="remove_train=True needs the score CSV "
                                             "to sit inside a"):
            D.annotate_filter_vision(_annotation(src=str(csv), remove_train=True))
