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


def test_augment_single_image_preserves_rgb_semantics(tmp_path):
    from PIL import Image

    src = tmp_path / "red.png"
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 240
    Image.fromarray(rgb, mode="RGB").save(src)
    dst = tmp_path / "out_rgb"
    dst.mkdir()

    U.augment_single_image((str(src), str(dst)))

    restored = np.asarray(Image.open(dst / "red_original.png").convert("RGB"))
    assert restored[0, 0].tolist() == [240, 0, 0]


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

def _fit(noise, seed=1, n=50):
    """OLS fit of a known ``y = 2x + noise*eps`` relationship."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, noise, n)
    return sm.OLS(y, sm.add_constant(x)).fit(), len(x)


def _parse_metrics(text):
    """Pull the three numbers model_metrics prints back out of stdout."""
    got = {}
    for line in text.splitlines():
        for label, key in (("Root Mean Squared Error (RMSE):", "rmse"),
                           ("Mean Absolute Error (MAE):", "mae"),
                           ("Durbin-Watson:", "dw")):
            if line.startswith(label):
                got[key] = float(line.split(":", 1)[1])
    return got


def test_model_metrics(capsys):
    """The printed numbers are the model's, and the 2x2 diagnostic grid is drawn.

    Two fits of the *same* ``y = 2x`` relationship at different noise levels
    give the contrast: a printout that ignored the model would report the
    same RMSE/MAE twice.
    """
    from statsmodels.stats.stattools import durbin_watson

    tight, n = _fit(noise=0.3)
    loose, _ = _fit(noise=1.5)
    # The fixture really is y = 2x with a good fit.
    assert tight.params[1] == pytest.approx(2.0, abs=0.15)
    assert tight.rsquared > 0.95

    plt.close("all")
    U.model_metrics(tight)
    tight_out = _parse_metrics(capsys.readouterr().out)

    assert tight_out["rmse"] == pytest.approx(float(np.sqrt(tight.mse_resid)))
    assert tight_out["mae"] == pytest.approx(float(np.mean(np.abs(tight.resid))))
    assert tight_out["dw"] == pytest.approx(float(durbin_watson(tight.resid)))

    # The four diagnostic panels carry real data, not empty axes.
    fig = plt.gcf()
    assert [ax.get_title() for ax in fig.axes] == [
        "Residuals vs Fitted", "Histogram of Residuals",
        "QQ Plot", "Scale-Location"]
    resid_vs_fitted, hist, qq, scale_loc = fig.axes
    assert len(resid_vs_fitted.collections) == 1
    assert len(scale_loc.collections) == 1
    assert len(resid_vs_fitted.collections[0].get_offsets()) == n
    assert np.asarray(resid_vs_fitted.collections[0].get_offsets())[:, 0] \
        == pytest.approx(np.asarray(tight.fittedvalues))
    assert len(hist.patches) > 0
    assert sum(p.get_height() for p in hist.patches) == pytest.approx(n)
    assert len(qq.lines) == 2                     # sample points + 45° line
    assert len(scale_loc.collections[0].get_offsets()) == n
    plt.close("all")

    U.model_metrics(loose)
    loose_out = _parse_metrics(capsys.readouterr().out)
    assert loose_out["rmse"] == pytest.approx(float(np.sqrt(loose.mse_resid)))

    # Contrast: five times the noise, so the errors must be materially bigger.
    assert loose_out["rmse"] > 3 * tight_out["rmse"]
    assert loose_out["mae"] > 3 * tight_out["mae"]
