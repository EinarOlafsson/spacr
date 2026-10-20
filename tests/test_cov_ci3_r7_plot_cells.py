"""Print mode: heatmap cell labels are recognised from image and mesh extents only."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from spacr import plot


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def test_an_image_without_an_extent_is_skipped(ax, monkeypatch):
    good = ax.imshow(np.arange(4).reshape(2, 2), extent=(0, 2, 0, 2))
    broken = ax.imshow(np.arange(4).reshape(2, 2), extent=(10, 12, 10, 12))

    def no_extent():
        raise ValueError('no extent yet')

    monkeypatch.setattr(broken, 'get_extent', no_extent)
    assert plot._raster_extents(ax) == [(0, 2, 0, 2)]
    assert good in ax.images


def test_meshes_outside_data_coordinates_or_without_area_are_skipped(ax, monkeypatch):
    ax.pcolormesh(np.arange(6).reshape(2, 3))
    flat = PolyCollection([[(5, 5), (6, 5), (7, 5)]], transform=ax.transData)
    ax.add_collection(flat)
    overlay = PolyCollection([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]],
                             transform=ax.transAxes)
    ax.add_collection(overlay)
    broken = PolyCollection([[(20, 20), (30, 20), (30, 30)]], transform=ax.transData)
    ax.add_collection(broken)

    def no_limits(_transform):
        raise ValueError('no paths')

    monkeypatch.setattr(broken, 'get_datalim', no_limits)
    ax.scatter([1], [1])
    assert plot._raster_extents(ax) == [pytest.approx((0, 3, 0, 2))]


def test_only_data_texts_on_a_cell_label_a_cell(ax):
    ax.imshow(np.arange(4).reshape(2, 2), extent=(0, 2, 0, 2))
    on_cell = ax.text(1, 1, '3')
    off_cell = ax.text(5, 5, 'note')
    title = ax.text(0.5, 0.5, 'axes text', transform=ax.transAxes)
    unplaced = ax.text(1, 1, 'pending')
    unplaced.set_position((None, None))
    words = ax.text(1, 1, 'odd')
    words.set_position(('left', 'top'))
    assert plot._labels_a_cell(ax, on_cell) is True
    assert plot._labels_a_cell(ax, off_cell) is False
    assert plot._labels_a_cell(ax, title) is False
    assert plot._labels_a_cell(ax, unplaced) is False
    assert plot._labels_a_cell(ax, words) is False
