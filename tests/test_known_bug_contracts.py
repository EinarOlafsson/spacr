"""Regression tests for resolved entries in the historical known-bug ledgers."""

from collections import OrderedDict
from typing import List, get_type_hints

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_cluster_grid_uses_the_panel_color_in_its_legend(monkeypatch):
    """A non-contiguous integer label must not get a positional legend color."""
    from spacr.utils import plot_grid

    monkeypatch.setattr(plt, "show", lambda: None)
    clusters = OrderedDict([
        (0, [np.zeros((2, 2), dtype=np.uint8)]),
        (3, [np.ones((2, 2), dtype=np.uint8)]),
    ])
    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
    ]

    figure = plot_grid(
        clusters,
        colors,
        figuresize=2,
        black_background=False,
        verbose=False,
    )
    try:
        panel_colors = [axis.patches[0].get_facecolor()[:3]
                        for axis in figure.axes[:2]]
        legend_colors = [patch.get_facecolor()[:3]
                         for patch in figure.patches]
        assert legend_colors == pytest.approx(panel_colors)
        assert legend_colors[1] == pytest.approx(colors[3])
    finally:
        plt.close(figure)


def test_thumbnail_plot_visits_every_non_noise_cluster(monkeypatch):
    """A palette sized to real clusters must not let noise consume one slot."""
    from spacr import utils

    opened = []
    drawn = []
    monkeypatch.setattr(utils.Image, "open", lambda path: opened.append(path) or path)
    monkeypatch.setattr(
        utils,
        "plot_image",
        lambda _ax, _x, _y, image, *_args: drawn.append(image),
    )

    utils.plot_images_by_cluster(
        object(),
        ["noise.png", "zero.png", "one.png"],
        np.array([[0, 0], [1, 1], [2, 2]], dtype=float),
        np.array([-1, 0, 1]),
        image_nr=1,
        img_zoom=1.0,
        colors=["red", "blue"],
        cluster_indices={-1: [0], 0: [1], 1: [2]},
        remove_image_canvas=False,
        verbose=False,
    )

    assert opened == ["zero.png", "one.png"]
    assert drawn == opened


@pytest.mark.parametrize("generator_name", [
    "SaliencyMapGenerator",
    "GradCAMGenerator",
])
def test_activation_grids_hide_every_unused_panel(generator_name):
    """A one-sample grid should not leave seven framed axes behind."""
    torch = pytest.importorskip("torch")
    from spacr import utils

    generator_type = getattr(utils, generator_name)
    generator = object.__new__(generator_type)
    batch = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    maps = torch.ones((1, 4, 4), dtype=torch.float32)
    predictions = torch.tensor([2])

    figure = generator.plot_activation_grid(
        batch,
        maps,
        predictions,
        overlay=False,
        normalize=False,
    )
    try:
        assert len(figure.axes[0].images) == 1
        assert all(not axis.axison for axis in figure.axes)
    finally:
        plt.close(figure)


def test_merged_crop_batches_are_fail_loud_and_non_optional():
    """The crop-source interface must describe the behavior it implements."""
    from spacr.crops import CropError, CropSource, CropSpec, MergedCropSource

    source = MergedCropSource(CropSpec(merged_path="unused.npy"))
    with pytest.raises(CropError, match="object_label"):
        source.get_many([{}])

    assert get_type_hints(CropSource.get_many)["return"] == List[np.ndarray]
    assert get_type_hints(MergedCropSource.get_many)["return"] == List[np.ndarray]
