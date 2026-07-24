"""Coverage-fill batch 8 for spacr.utils augment / model-metrics helpers."""
from __future__ import annotations

import numpy as np
import pytest
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import utils as U


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _write_img(path):
    img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# augment_single_image / augment_images
# ---------------------------------------------------------------------------

def test_augment_single_image(tmp_path):
    src = tmp_path / "img.png"; _write_img(src)
    dst = tmp_path / "out"; dst.mkdir()
    U.augment_single_image((str(src), str(dst)))
    pngs = list(dst.glob("*.png"))
    assert len(pngs) == 6   # original + 3 rotations + 2 flips


def test_augment_images(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"img{i}.png"; _write_img(p); paths.append(str(p))
    dst = tmp_path / "aug"
    U.augment_images(paths, str(dst))
    assert len(list(dst.glob("*.png"))) == 12   # 6 per image


# ---------------------------------------------------------------------------
# model_metrics
# ---------------------------------------------------------------------------

def test_model_metrics():
    import statsmodels.api as sm
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 50)
    y = 2 * x + rng.normal(0, 0.3, 50)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    U.model_metrics(model)   # prints + shows diagnostic plots; must not raise
