"""Two guards in :mod:`spacr.utils` that only an unusual input reaches.

Both are the "there is nothing here" side of a decision the ordinary run
never takes:

* ``_process_single_fov_in_memory`` counts a field's objects as
  ``len(np.unique(...)) - 1``, which is ``-1`` -- not ``0`` -- for a mask
  array with no pixels in it. Such a field slips past the "empty mask,
  skipping" return and arrives at the merge phase with no labels to merge,
  where a second count sends it straight to the filter;
* ``plot_clusters`` recolours the legend it just asked for, and skips that
  work when the axes it was handed produced none, without dropping the axis
  labels that come after it.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import utils


def _in_memory(mask, **overrides):
    """Call ``_process_single_fov_in_memory`` with the pipeline's defaults."""
    settings = dict(
        intensity_img=None, intensity_channel=None, do_split=False,
        do_perimeter_merge=True, do_intensity_merge=False,
        perimeter_fraction=0.1, area_multiplier=2.0, min_distance=10,
        min_object_area=100, intensity_threshold_method='mean',
        intensity_percentile=75, min_area=0, max_area=0,
        remove_border_objects=False, min_intensity_percentile=0,
        max_intensity_percentile=100)
    settings.update(overrides)
    return utils._process_single_fov_in_memory(
        mask, settings['intensity_img'], settings['intensity_channel'],
        settings['do_split'], settings['do_perimeter_merge'],
        settings['do_intensity_merge'], settings['perimeter_fraction'],
        settings['area_multiplier'], settings['min_distance'],
        settings['min_object_area'], settings['intensity_threshold_method'],
        settings['intensity_percentile'], settings['min_area'],
        settings['max_area'], settings['remove_border_objects'],
        settings['min_intensity_percentile'],
        settings['max_intensity_percentile'])


# ===========================================================================
# a field whose array has no pixels at all
# ===========================================================================

def test_a_mask_array_with_no_pixels_reaches_the_filter_without_merging(capsys):
    """A zero-sized field is handed back as it came, and merges nothing.

    The early return above the merge phase asks whether
    ``len(np.unique(label_img)) - 1`` is zero. For a ``(0, 0)`` array the
    unique list is empty, so that count is ``-1`` and the field carries on --
    which is why the merge phase takes its own count of the labels present.
    Without it the union-find would size its mapping from ``label_img.max()``
    on an empty array, which numpy refuses (asserted below), and the field
    would fail rather than come back empty.
    """
    empty = np.zeros((0, 0), np.uint16)

    kept = _in_memory(empty)

    assert kept.shape == (0, 0)
    assert kept.dtype == np.uint16
    # No merge pass ran: the merge phase is the only thing that reports a
    # changed object count for this field.
    assert "merge:" not in capsys.readouterr().out
    # And it is the label count, not luck, that kept it out: the union-find
    # the merge phase ends with cannot be built over an empty array.
    with pytest.raises(ValueError):
        utils._apply_union_find(empty, {})

    # The same call with labels present does run the merge phase, and says so.
    touching = np.zeros((16, 16), np.uint16)
    touching[2:6, 2:6] = 1
    touching[2:6, 6:10] = 2

    merged = _in_memory(touching)

    assert sorted(np.unique(merged).tolist()) == [0, 1]
    assert "merge: 2 → 1 objects" in capsys.readouterr().out


# ===========================================================================
# an axes that produces no legend
# ===========================================================================

def test_a_legend_that_was_not_created_does_not_cost_the_axis_labels():
    """The legend recolouring is skipped, the axis labels are not.

    ``plot_clusters`` draws into an axes the caller owns and then repaints
    the legend it asked that axes for, so the fonts and frame follow the
    theme rather than Matplotlib's defaults. A stock ``Axes.legend()`` always
    hands back a ``Legend``; an axes that hands back nothing must not take
    the axis labels, ticks and scatter down with it, which is the whole point
    of the check. The axes here is the injected collaborator the signature
    already documents -- nothing inside the module is reached around.
    """
    rng = np.random.default_rng(3)
    points = rng.normal(0.0, 1.0, (12, 2))
    labels = np.zeros(len(points), dtype=int)
    colors = [(0.3, 0.5, 0.7, 1.0)]
    centers = [points.mean(axis=0)]

    figure, (themed, legendless) = plt.subplots(1, 2)
    try:
        for axes in (themed, legendless):
            axes.set_facecolor("#202020")
            axes.xaxis.label.set_color("#ff0000")
        legendless.legend = lambda *args, **kwargs: None

        utils.plot_clusters(themed, points, labels, colors, centers,
                            plot_outlines=False, plot_points=True,
                            smooth_lines=False)
        utils.plot_clusters(legendless, points, labels, colors, centers,
                            plot_outlines=False, plot_points=True,
                            smooth_lines=False)

        # The axes that produced a legend had it repainted in the theme.
        drawn = themed.get_legend()
        assert drawn is not None
        assert drawn.get_frame().get_facecolor()[:3] == pytest.approx(
            themed.get_facecolor()[:3])
        assert [text.get_color() for text in drawn.get_texts()] == ["#ff0000"]

        # The axes that produced none kept everything that comes after.
        assert legendless.get_legend() is None
        assert legendless.get_xlabel() == "UMAP Dimension 1"
        assert legendless.get_ylabel() == "UMAP Dimension 2"
        assert len(legendless.collections) == len(themed.collections) == 1
    finally:
        plt.close(figure)
