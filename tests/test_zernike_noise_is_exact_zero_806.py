"""Item 806: Zernike moments that are zero by symmetry are exactly 0.0."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mahotas")

from spacr import measure


def _disc(n=41, r=15):
    y, x = np.ogrid[:n, :n]
    return (((y - n // 2) ** 2 + (x - n // 2) ** 2) <= r * r).astype(np.int32)


def _ellipse(n=41):
    y, x = np.ogrid[:n, :n]
    return ((((y - n // 2) / 8.0) ** 2 + ((x - n // 2) / 15.0) ** 2) <= 1).astype(np.int32)


@pytest.mark.parametrize("make", [_disc, _ellipse])
def test_cpu_noise_moments_are_exactly_zero(make):
    out = measure._calculate_zernike(make(), pd.DataFrame({"label": [1]}))
    assert out["zernike_1"].iloc[0] == 0.0
    assert out["zernike_0"].iloc[0] > 0


def test_noise_below_tolerance_is_zeroed_on_the_gpu_path(monkeypatch):
    mask = _disc()
    feats = np.array([[0.3, 1e-15, 2e-3, -1e-14]])
    monkeypatch.setattr(measure, "_gpu_measurable", lambda m: True)
    monkeypatch.setattr(measure, "_torch_zernike", lambda m, d, dev: feats)
    out = measure._calculate_zernike(mask, pd.DataFrame({"label": [1]}), device="cuda")
    assert out[["zernike_0", "zernike_1", "zernike_2", "zernike_3"]].iloc[0].tolist() == [0.3, 0.0, 2e-3, 0.0]
