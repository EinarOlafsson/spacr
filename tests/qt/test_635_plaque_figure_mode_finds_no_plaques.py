"""Figure mode segments a well crop at the model's own scale.

    "in Plaque's Figure mode the default model finds 0 plaques on
     open-access paper figures, although well finding and label reading
     work" -- spacr-d7, 2026-10-03 (PLoS Biol pbio.3002110 Fig 8)

The diameter spin box is shared with Plaque mode, where it is in the pixels
of a whole-well image. A well on a paper figure is a hundred or two pixels
across, so a Plaque-mode diameter (180 px on the synthetic plates) made
Cellpose shrink each crop to a few pixels: 0 plaques in all 8 wells, against
66 at the model's own scale. The run's Figure mode never passed a diameter.
"""
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

    def eval(self, image, channel_axis=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0):
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


def test_a_plaque_mode_diameter_does_not_empty_a_figure_well(model):
    image = _well()
    settings = {"plaque_model": "toxoplasma_plaque_v2", "diameter": 180.0,
                "flow_threshold": 0.4, "CP_prob": 0.0}
    result = ppv.segment_well(image, _Region(), settings)
    assert result["count"] == 5
    assert model.diameters == [None]
    assert settings["diameter"] == 180.0, "the caller's settings are kept"


def test_figure_scale_drops_only_the_diameter():
    settings = {"diameter": 30, "CP_prob": 1.5, "flow_threshold": 0.2}
    assert ppv._at_figure_scale(settings) == {"CP_prob": 1.5,
                                              "flow_threshold": 0.2}
