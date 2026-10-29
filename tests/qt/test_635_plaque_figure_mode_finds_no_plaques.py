"""Figure diameter is explicit in original crop pixels, with zero retaining auto scale."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import plaque  # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402


class _Region:
    y0, y1, x0, x1 = 0, 160, 0, 160


def _well():
    """A purple monolayer with five bright plaques, 160 px across."""
    rng = np.random.default_rng(635)
    crop = np.empty((160, 160, 3), np.uint8)
    crop[...] = (110, 60, 170)
    crop = np.clip(crop + rng.normal(0, 8, crop.shape), 0, 255)
    yy, xx = np.mgrid[:160, :160]
    for cy, cx in ((30, 30), (40, 110), (80, 70), (120, 30), (125, 125)):
        crop[(yy - cy) ** 2 + (xx - cx) ** 2 < 64] = 235
    return crop.astype(np.uint8)


class _ScaleAwareModel:
    """Finds the bright holes, unless asked for objects far larger than they
    are, as a Cellpose model rescaling the crop by 30 / diameter does."""

    def __init__(self):
        self.diameters = []

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True,
             progress=None):
        image = x
        from skimage.measure import label

        self.diameters.append(diameter)
        if diameter and diameter > 60:
            labels = np.zeros(image.shape[:2], np.int32)
        else:
            labels = label(np.asarray(image)[..., 1] > 200)
        return labels, [None, None, None], None


@pytest.fixture
def model(monkeypatch):
    fake = _ScaleAwareModel()
    monkeypatch.setattr(ppv, "resolve_plaque_model",
                        lambda settings: ("fake", "", None))
    monkeypatch.setattr(ppv, "_cellpose_model", lambda path: fake)
    monkeypatch.setattr(plaque, "plaque_flow_outputs", lambda output: {})
    return fake


@pytest.mark.parametrize('diameter,count,forwarded', [(0, 5, None), (None, 5, None),
                                                      (16, 5, 16), (180, 0, 180)])
def test_figure_diameter_uses_original_crop_pixels(model, diameter, count, forwarded):
    image = _well()
    settings = {"plaque_model": "toxoplasma_plaque_v2", "diameter": diameter,
                "flow_threshold": 0.4, "CP_prob": 0.0}
    result = ppv.segment_well(image, _Region(), settings)
    assert result["count"] == count
    assert model.diameters == [forwarded]
    assert settings["diameter"] == diameter


def test_figure_settings_preserve_diameter_and_other_tuned_values():
    settings = {"diameter": 30, "CP_prob": 1.5, "flow_threshold": 0.2}
    assert ppv._at_figure_scale(settings) == settings
    assert ppv._at_figure_scale(settings) is not settings
