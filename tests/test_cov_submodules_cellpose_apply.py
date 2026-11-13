"""``spacr.submodules.plot_cellpose_batch``: the image / label preview grid.

``apply_cellpose_model`` and ``analyze_percent_positive`` are tested in
``tests/test_submodules_cellpose.py`` and ``tests/test_submodules_well_assays.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _no_blocking_show(monkeypatch):
    """Never let a product-code ``plt.show()`` reach a real backend."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    return calls


IMG_SIZE = 64


# ---------------------------------------------------------------------------
# synthetic Cellpose doubles
# ---------------------------------------------------------------------------

def _label_mask(shape=(IMG_SIZE, IMG_SIZE), include_corner=True):
    """Deterministic 4-object label mask.

    Three squares sit inside the inscribed circle of the frame, one sits in
    the top-left corner *outside* it -- so ``circularize`` provably drops
    exactly one object.

    Areas: 100, 100, 64 (inside) and 25 (corner).
    """
    m = np.zeros(shape, dtype=np.int32)
    m[10:20, 10:20] = 1      # 100 px, inside the circle
    m[28:38, 28:38] = 2      # 100 px, dead centre
    m[44:52, 20:28] = 3      # 64 px, inside
    if include_corner:
        m[0:5, 0:5] = 4      # 25 px, outside the inscribed circle
    return m


# ===========================================================================
# plot_cellpose_batch
# ===========================================================================

def test_plot_cellpose_batch_builds_two_row_grid(_no_blocking_show):
    """Two rows (images / labels) with titled, axis-less panels."""
    from spacr.submodules import plot_cellpose_batch

    images = [
        np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32),
        np.zeros((32, 32), dtype=np.float32),
    ]
    labels = [_label_mask((32, 32), include_corner=False) for _ in range(2)]

    plot_cellpose_batch(images, labels)

    assert len(_no_blocking_show) == 1
    fig = plt.gcf()
    axs = fig.axes
    assert len(axs) == 4
    assert [ax.get_title() for ax in axs] == [
        "Image 1", "Image 2", "Label 1", "Label 2",
    ]
    assert all(ax.axison is False for ax in axs)
    # the top row shows the images, the bottom row the labels
    np.testing.assert_allclose(axs[0].images[0].get_array(), images[0])
    np.testing.assert_array_equal(axs[2].images[0].get_array(), labels[0])
    # labels are drawn with the random mask colormap (1 colour per label + bg)
    cmap = axs[2].images[0].get_cmap()
    n_labels = len(np.unique(np.asarray(labels))) - 1
    assert cmap.N == n_labels + 1
    assert tuple(cmap.colors[0]) == (0.0, 0.0, 0.0, 1.0)
    # images are drawn in grayscale, interpolation nearest for the masks
    assert axs[0].images[0].get_cmap().name == "gray"
    assert axs[2].images[0].get_interpolation() == "nearest"


def test_plot_cellpose_batch_single_image(_no_blocking_show):
    """A batch of one is a legal batch and must render a 2-panel figure."""
    from spacr.submodules import plot_cellpose_batch

    images = [np.zeros((16, 16), dtype=np.float32)]
    labels = [_label_mask((16, 16), include_corner=False)]

    plot_cellpose_batch(images, labels)

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Image 1", "Label 1"]
