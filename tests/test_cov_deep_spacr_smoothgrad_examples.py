"""CPU coverage for spacr.deep_spacr's SmoothGrad attribution, the
``visualize_smooth_grad`` driver, ``save_top_class_examples`` and
``merge_predictions_into_db``.

Everything runs on a tiny hand-built ``nn.Sequential`` so the gradients are
analytically known (for a linear head the input gradient is exactly the
weight row), which lets the tests assert real numbers instead of shapes.
Matplotlib is forced to Agg and every figure is closed by an autouse
fixture, so ``plt.show()`` inside the product code never blocks.
"""
from __future__ import annotations

import os
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)

import torch.nn as nn  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


def _linear_model(n_in, n_out, weight=None, bias=0.0):
    """Flatten -> Linear model with fully deterministic parameters."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(n_in, n_out))
    with torch.no_grad():
        if weight is None:
            weight = torch.arange(n_out * n_in, dtype=torch.float32).reshape(n_out, n_in)
            weight = weight / float(n_out * n_in)
        model[1].weight.copy_(weight)
        model[1].bias.fill_(bias)
    return model


def _write_png(path, rng, size=8):
    arr = rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# SmoothGrad.__init__
# ---------------------------------------------------------------------------

def test_smoothgrad_init_defaults_and_overrides():
    """__init__ stores the model plus both noise knobs verbatim."""
    from spacr.deep_spacr import SmoothGrad
    model = _linear_model(12, 2)

    default = SmoothGrad(model)
    assert default.model is model
    assert default.n_samples == 50
    assert default.stdev_spread == pytest.approx(0.15)

    custom = SmoothGrad(model, n_samples=7, stdev_spread=0.4)
    assert custom.n_samples == 7
    assert custom.stdev_spread == pytest.approx(0.4)
    # the two instances must not share state
    assert default.n_samples == 50


# ---------------------------------------------------------------------------
# SmoothGrad.compute_smooth_grad
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# visualize_smooth_grad
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# save_top_class_examples
# ---------------------------------------------------------------------------

def _tar_of_pngs(tmp_path, rng, names, size=8):
    d = tmp_path / "imgs"
    d.mkdir(exist_ok=True)
    payload = {}
    for nm in names:
        p = d / os.path.basename(nm)
        _write_png(p, rng, size=size)
        payload[nm] = p.read_bytes()
    tar_path = tmp_path / "ds.tar"
    with tarfile.open(tar_path, "w") as t:
        for nm in names:
            t.add(d / os.path.basename(nm), arcname=nm)
    return str(tar_path), payload


def test_save_top_class_examples_picks_the_extremes(tmp_path, rng):
    """class_0 gets the lowest preds, class_1 the highest; bytes match the tar."""
    from spacr.deep_spacr import save_top_class_examples
    names = [f"crops/o{i}.png" for i in range(6)]
    tar_path, payload = _tar_of_pngs(tmp_path, rng, names)
    df = pd.DataFrame({"path": names, "pred": [0.01, 0.02, 0.5, 0.6, 0.98, 0.99]})

    dst = tmp_path / "top"
    out = save_top_class_examples(df, tar_path, str(dst), n=2)

    assert out == str(dst)
    assert sorted(p.name for p in (dst / "class_0").iterdir()) == ["o0.png", "o1.png"]
    assert sorted(p.name for p in (dst / "class_1").iterdir()) == ["o4.png", "o5.png"]
    # the extracted bytes are the real tar members, not placeholders
    assert (dst / "class_0" / "o0.png").read_bytes() == payload["crops/o0.png"]
    assert (dst / "class_1" / "o5.png").read_bytes() == payload["crops/o5.png"]


def test_save_top_class_examples_duplicate_extremes(tmp_path, rng, capsys):
    """One image can be both the lowest and the highest; it is copied twice."""
    from spacr.deep_spacr import save_top_class_examples
    names = ["only.png"]
    tar_path, payload = _tar_of_pngs(tmp_path, rng, names)
    df = pd.DataFrame({"path": names, "pred": [0.42]})

    dst = tmp_path / "dup"
    save_top_class_examples(df, tar_path, str(dst), n=5)

    assert (dst / "class_0" / "only.png").read_bytes() == payload["only.png"]
    assert (dst / "class_1" / "only.png").read_bytes() == payload["only.png"]
    assert "Saved 2 top-confidence example images" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# merge_predictions_into_db
# ---------------------------------------------------------------------------


