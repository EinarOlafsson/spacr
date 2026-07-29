"""CPU coverage for :func:`spacr.deep_spacr.apply_model` and
:func:`spacr.deep_spacr.apply_model_to_tar`.

The uncovered branches of these two functions are the *un-normalised*
transform pipelines and the single-logit head of the tar variant.  To assert
something stronger than "it ran", every test here uses a deterministic model
whose output is an exact function of the mean pixel value, and images whose
pixels are constant.  The expected probability is then re-derived in the test
with an independent implementation of the torchvision transform maths
(``x/255`` versus ``(x/255 - 0.5)/0.5``), so a silently dropped
``Normalize`` step -- or a silently added one -- fails the test.

Everything runs on CPU, offline, with ``num_workers=0``.
"""
from __future__ import annotations

import datetime
import os
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# deterministic model + synthetic data helpers
# ---------------------------------------------------------------------------

#: multiplier applied to the mean pixel value to turn it into a logit
LOGIT_SCALE = 3.0


class _MeanLogitModel(torch.nn.Module):
    """Deterministic classifier: logit = ``LOGIT_SCALE * mean(pixels)``.

    ``n_out=1``  -> shape ``(N, 1)``  (single-logit binary head)
    ``n_out=2``  -> shape ``(N, 2)``  columns ``(-logit, +logit)``

    Defined at module level so ``torch.save``/``torch.load`` can pickle it by
    reference the same way spaCR pickles its own ``TorchModel``.
    """

    def __init__(self, n_out: int = 1):
        super().__init__()
        self.n_out = int(n_out)
        # a real parameter so .to(device)/.eval() exercise the usual machinery
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):  # noqa: D102 - trivial
        logit = x.flatten(1).mean(dim=1, keepdim=True) * LOGIT_SCALE + self.bias
        if self.n_out == 2:
            return torch.cat([-logit, logit], dim=1)
        if self.n_out == 3:
            return torch.cat([-logit, torch.zeros_like(logit), logit], dim=1)
        return logit


def _save_model(path, n_out=1):
    model = _MeanLogitModel(n_out=n_out)
    torch.save(model, str(path))
    return str(path)


def _const_png(path, value, size=32):
    """Write a ``size x size`` RGB PNG whose every pixel equals ``value``."""
    arr = np.full((size, size, 3), int(value), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.float64(x)))


def _expected_prob(value, normalize):
    """Independently re-derive the probability the model must emit.

    ``ToTensor`` maps uint8 -> ``value/255`` float32; ``Normalize(0.5, 0.5)``
    then maps that to ``(v - 0.5) / 0.5``.  ``CenterCrop`` is a no-op because
    the images are written at exactly ``image_size``.
    """
    v = np.float32(value) / np.float32(255.0)
    if normalize:
        v = (v - np.float32(0.5)) / np.float32(0.5)
    return _sigmoid(np.float32(LOGIT_SCALE) * v)


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
        p = d / f"plate1_A01_f1_o{i}.png"
        mapping[_const_png(p, v)] = v
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
    """Pin inference to the CPU even on a CUDA-capable developer box.

    ``apply_model`` / ``apply_model_to_tar`` pick their device from
    ``torch.cuda.is_available()``; forcing it False keeps the numbers below
    reproducible (and the DataLoader's ``pin_memory`` off) everywhere.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _tar_settings(tar_path, model_path, **over):
    settings = {
        "tar_path": tar_path,
        "model_path": model_path,
        "image_size": 32,
        "batch_size": 2,
        "normalize": True,
        "n_jobs": 0,
        "verbose": False,
        "score_threshold": 0.5,
    }
    settings.update(over)
    return settings


# ---------------------------------------------------------------------------
# apply_model -- the ``normalize=False`` transform branch (line 62)
# ---------------------------------------------------------------------------

def test_apply_model_without_normalize_uses_raw_tensor_scale(const_png_dir, tmp_path):
    """normalize=False must feed raw ``[0, 1]`` tensors to the model."""
    from spacr.deep_spacr import apply_model

    src, values = const_png_dir
    model_path = _save_model(tmp_path / "m.pth", n_out=1)

    df = apply_model(src, model_path, image_size=32, batch_size=2,
                     normalize=False, n_jobs=0)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["path", "pred"]
    assert len(df) == len(values)
    # DataLoader shuffles, so compare as a path -> pred mapping
    assert set(df["path"]) == set(values)
    for path, pred in zip(df["path"], df["pred"]):
        expected = _expected_prob(values[path], normalize=False)
        assert _scalar(pred) == pytest.approx(expected, abs=1e-5)
    # black (0) maps to logit 0 -> exactly 0.5 without normalisation, which is
    # impossible under Normalize(0.5, 0.5) (that would give sigmoid(-3)).
    black = [p for p, v in values.items() if v == 0][0]
    assert _scalar(df.loc[df["path"] == black, "pred"].iloc[0]) == pytest.approx(0.5, abs=1e-6)


def test_apply_model_normalize_flag_changes_predictions(const_png_dir, tmp_path):
    """The two transform branches must not produce identical scores."""
    from spacr.deep_spacr import apply_model

    src, values = const_png_dir
    model_path = _save_model(tmp_path / "m.pth", n_out=1)

    raw = apply_model(src, model_path, image_size=32, batch_size=8,
                      normalize=False, n_jobs=0)
    norm = apply_model(src, model_path, image_size=32, batch_size=8,
                       normalize=True, n_jobs=0)

    raw_map = {p: _scalar(v) for p, v in zip(raw["path"], raw["pred"])}
    norm_map = {p: _scalar(v) for p, v in zip(norm["path"], norm["pred"])}
    assert set(raw_map) == set(norm_map)
    differing = [k for k in raw_map if abs(raw_map[k] - norm_map[k]) > 1e-4]
    # every value except the mid-grey 127.5 fixed point must move
    assert len(differing) >= len(values) - 1
    for path, v in values.items():
        assert norm_map[path] == pytest.approx(_expected_prob(v, normalize=True), abs=1e-5)


def test_apply_model_writes_csv_matching_returned_frame(const_png_dir, tmp_path):
    """The CSV written next to the model must round-trip the returned frame."""
    from spacr.deep_spacr import apply_model

    src, values = const_png_dir
    model_path = _save_model(tmp_path / "m.pth", n_out=1)

    df = apply_model(src, model_path, image_size=32, batch_size=3,
                     normalize=False, n_jobs=0)

    stem, ext = os.path.splitext(model_path)
    expected_csv = (stem + datetime.date.today().strftime("%y%m%d")
                    + "_" + ext + "_test_result.csv")
    assert os.path.isfile(expected_csv), f"missing {expected_csv}"

    on_disk = pd.read_csv(expected_csv, index_col=0)
    assert list(on_disk["path"]) == list(df["path"])
    assert len(on_disk) == len(values)


# ---------------------------------------------------------------------------
# apply_model_to_tar -- default settings branch (line 131)
# ---------------------------------------------------------------------------

def test_apply_model_to_tar_default_settings_raises_keyerror():
    """``settings=None`` degrades to an empty dict, so ``tar_path`` is missing."""
    from spacr.deep_spacr import apply_model_to_tar

    with pytest.raises(KeyError) as excinfo:
        apply_model_to_tar()
    assert excinfo.value.args[0] == "tar_path"


def test_apply_model_to_tar_empty_settings_raises_keyerror():
    """An explicitly empty settings dict fails the same way."""
    from spacr.deep_spacr import apply_model_to_tar

    with pytest.raises(KeyError) as excinfo:
        apply_model_to_tar(settings={})
    assert excinfo.value.args[0] == "tar_path"


# ---------------------------------------------------------------------------
# apply_model_to_tar -- ``normalize=False`` transform branch (line 147)
# ---------------------------------------------------------------------------

def test_apply_model_to_tar_without_normalize_two_class_head(const_png_tar, tmp_path):
    """normalize=False + a 2-column head -> softmax over raw [0, 1] tensors."""
    from spacr.deep_spacr import apply_model_to_tar

    tar_path, values = const_png_tar
    model_path = _save_model(tmp_path / "twoclass.pth", n_out=2)
    settings = _tar_settings(tar_path, model_path, normalize=False,
                             score_threshold=0.6, verbose=False)

    df = apply_model_to_tar(settings)

    assert isinstance(df, pd.DataFrame)
    assert set(df["path"]) == set(values)
    for path, pred in zip(df["path"], df["pred"]):
        v = np.float32(values[path]) / np.float32(255.0)
        # softmax over (-l, +l) == sigmoid(2l)
        expected = _sigmoid(2.0 * LOGIT_SCALE * v)
        assert float(pred) == pytest.approx(expected, abs=1e-5)
    # process_vision_results enrichment
    assert list(df["cv_predictions"]) == [int(p >= 0.6) for p in df["pred"]]
    assert set(df["plateID"]) == {"plate1"}
    assert set(df["prc"]) == {"plate1_r1_c1"}

    dst = os.path.dirname(tar_path)
    expected_csv = os.path.join(
        dst,
        f"{datetime.date.today().strftime('%y%m%d')}_dataset_twoclass_result.csv")
    assert os.path.isfile(expected_csv)
    assert len(pd.read_csv(expected_csv, index_col=0)) == len(values)


def test_apply_model_to_tar_normalize_flag_changes_predictions(const_png_tar, tmp_path):
    """Normalised and un-normalised tar inference must disagree."""
    from spacr.deep_spacr import apply_model_to_tar

    tar_path, values = const_png_tar
    model_path = _save_model(tmp_path / "twoclass.pth", n_out=2)

    raw = apply_model_to_tar(_tar_settings(tar_path, model_path, normalize=False))
    norm = apply_model_to_tar(_tar_settings(tar_path, model_path, normalize=True))

    raw_map = dict(zip(raw["path"], raw["pred"]))
    norm_map = dict(zip(norm["path"], norm["pred"]))
    for path, v in values.items():
        vv = np.float32(v) / np.float32(255.0)
        assert raw_map[path] == pytest.approx(_sigmoid(2.0 * LOGIT_SCALE * vv), abs=1e-5)
        nv = (vv - np.float32(0.5)) / np.float32(0.5)
        assert norm_map[path] == pytest.approx(_sigmoid(2.0 * LOGIT_SCALE * nv), abs=1e-5)
    assert any(abs(raw_map[p] - norm_map[p]) > 1e-3 for p in values)


# ---------------------------------------------------------------------------
# apply_model_to_tar -- single-logit head branch (line 200)
# ---------------------------------------------------------------------------

def test_apply_model_to_tar_single_logit_head_is_squeezed(const_png_tar, tmp_path):
    """A ``(N, 1)`` head goes through sigmoid+squeeze, yielding scalar preds."""
    from spacr.deep_spacr import apply_model_to_tar

    tar_path, values = const_png_tar
    model_path = _save_model(tmp_path / "binary.pth", n_out=1)
    settings = _tar_settings(tar_path, model_path, normalize=True,
                             batch_size=2, score_threshold=0.5, verbose=True)

    df = apply_model_to_tar(settings)

    assert set(df["path"]) == set(values)
    # squeeze(-1) means each pred is a plain float, not a 1-element list
    assert all(isinstance(p, float) for p in df["pred"]), df["pred"].tolist()
    for path, pred in zip(df["path"], df["pred"]):
        assert pred == pytest.approx(_expected_prob(values[path], normalize=True),
                                     abs=1e-5)
    # mid-grey (128) sits just above the 0.5 decision boundary, black well below
    lo = df.loc[df["path"] == "plate1_A01_f1_o0.png", "pred"].iloc[0]
    hi = df.loc[df["path"] == "plate1_A01_f1_o4.png", "pred"].iloc[0]
    assert lo < 0.5 < hi
    assert list(df["cv_predictions"]) == [int(p >= 0.5) for p in df["pred"]]


def test_apply_model_to_tar_single_logit_batch_size_one(const_png_tar, tmp_path):
    """batch_size=1 still squeezes the head axis away (no (1,) leftovers)."""
    from spacr.deep_spacr import apply_model_to_tar

    tar_path, values = const_png_tar
    model_path = _save_model(tmp_path / "binary.pth", n_out=1)
    settings = _tar_settings(tar_path, model_path, normalize=False, batch_size=1)

    df = apply_model_to_tar(settings)

    assert len(df) == len(values)
    assert all(np.isscalar(p) and 0.0 <= p <= 1.0 for p in df["pred"])
    for path, pred in zip(df["path"], df["pred"]):
        assert pred == pytest.approx(_expected_prob(values[path], normalize=False),
                                     abs=1e-5)


def test_apply_model_to_tar_multiclass_emits_class_probabilities(const_png_tar,
                                                                  tmp_path):
    """A multiclass head must use softmax and preserve the winning class."""
    from spacr.deep_spacr import apply_model_to_tar

    tar_path, values = const_png_tar
    model_path = _save_model(tmp_path / "threeclass.pth", n_out=3)
    df = apply_model_to_tar(
        _tar_settings(tar_path, model_path, normalize=False, batch_size=2))

    assert {"prob_class_0", "prob_class_1", "prob_class_2",
            "predicted_label", "cv_predictions"} <= set(df.columns)
    probabilities = df[
        ["prob_class_0", "prob_class_1", "prob_class_2"]
    ].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert df["cv_predictions"].tolist() == df["predicted_label"].tolist()
    assert np.allclose(df["pred"].to_numpy(), probabilities.max(axis=1))
