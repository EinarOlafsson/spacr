"""A palette sized to real clusters must not let noise consume a slot.

Preserved from tests/test_known_bug_contracts.py, which was deleted while the behaviour it pins is
still live. Sixteen of that file's nineteen tests genuinely stopped
holding and were rightly dropped; this one still passes against the
current tree, so deleting it would have retired a real contract rather
than a stale one. Kept verbatim -- only the tests that no longer hold
were removed.
"""

from collections import OrderedDict

from typing import List, get_type_hints

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pytest

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
