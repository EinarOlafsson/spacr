"""Round-6 ``spacr.plot``: the documented defaults behind three dead guards.

Every branch this round put in ``spacr.plot``'s scope turned out to be one
the module's own callers have already made true, and all of them are the
same two shapes:

* a ``None`` default replaced at the TOP of the function and re-tested for
  ``None`` at the bottom -- ``plot_lorenz_curves``'s ``x_lim``/``y_lim``,
  ``read_and_plot__vision_results``'s ``y_lim``, ``find_files``'s
  ``extensions`` and ``join_measurments_and_annotation``'s ``tables``;
* a re-check of something the same function just constructed --
  ``_plot_merged_plot``'s ``outline_info``/``outline``, the magenta
  variant's ``cell/nucleus/pathogen_outlines``, ``random_color_cmap``'s
  ``seed``, ``apply_contours_on_image``'s ``image.ndim``.

None of them is silenced. Each is written up in the round report with the
line that guarantees it, and the three whose guaranteed value is worth
holding on to are pinned here from the outside, so the day one of these
defaults changes a test says so:

* ``plot_lorenz_curves`` draws [0, 1] x [0, 1] unless told otherwise;
* every intensity channel that carries a mask gets that mask drawn on it;
* ``read_and_plot__vision_results`` bands its bars at 0.8-0.9.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.plot as P  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_lorenz_curves -- the x_lim / y_lim defaults
# ---------------------------------------------------------------------------

def _counts_csv(path, counts):
    pd.DataFrame({
        "grna_name": [f"g{i}" for i in range(len(counts))],
        "count": counts,
    }).to_csv(path, index=False)
    return str(path)


def test_lorenz_curves_are_drawn_on_the_unit_square_by_default(tmp_path):
    """plot.py:3905 / 3908 -- why ``x_lim is not None`` cannot be False.

    Both limits are replaced with ``[0.0, 1]`` and ``[0, 1]`` at the top of
    the function, ~120 lines before the ``is not None`` tests that apply
    them, so by then neither can be ``None``. The unit square those tests
    can only ever apply is the observable half, and an explicit limit has to
    beat it -- otherwise the pair of statements would be indistinguishable
    from hard-coded axes.
    """
    csv = _counts_csv(tmp_path / "plate1.csv", [1, 2, 3, 40, 100])

    P.plot_lorenz_curves([csv], save=False)
    free = plt.gcf().axes[0]
    assert free.get_xlim() == (0.0, 1.0)
    assert free.get_ylim() == (0.0, 1.0)
    assert free.get_xlabel() == "Cumulative Share of Individuals"

    P.plot_lorenz_curves([csv], x_lim=(0.2, 0.8), y_lim=(0.1, 0.6),
                         save=False)
    clamped = plt.gcf().axes[0]
    assert clamped.get_xlim() == (0.2, 0.8)
    assert clamped.get_ylim() == (0.1, 0.6)


# ---------------------------------------------------------------------------
# The invariant behind _plot_merged_plot's two dead re-checks
# ---------------------------------------------------------------------------

SHAPE = (48, 48)


def _mask(slices, label):
    m = np.zeros(SHAPE, dtype=np.uint16)
    m[slices] = label
    return m


def _stack(tmp_path, masks, n_intensity=3):
    rng = np.random.default_rng(3)
    planes = [rng.random(SHAPE).astype(np.float32) for _ in range(n_intensity)]
    stack = np.dstack([*planes, *[m.astype(np.float32) for m in masks]])
    folder = tmp_path / "merged"
    folder.mkdir()
    path = folder / "fov.npy"
    np.save(path, stack)
    return str(path)


def test_every_channel_with_a_mask_gets_that_mask_drawn(tmp_path):
    """plot.py:734 / 739 -- the invariant that makes both re-checks dead.

    ``channel_to_outline`` is built in the same function that consumes it,
    one entry per object type whose channel is not ``None``, and every entry
    holds a ``'mask'`` taken straight out of the stack with ``np.take``. So
    a channel that is IN ``channels_with_outlines`` always has a non-None
    entry with a non-None mask. What that guarantees is drawn here: each of
    the three channels carries its own object's colour and no other's.
    """
    cell = _mask(np.s_[4:20, 4:20], 1)
    nucleus = _mask(np.s_[8:14, 8:14], 1)
    pathogen = _mask(np.s_[30:38, 30:38], 1)
    path = _stack(tmp_path, [cell, nucleus, pathogen])

    fig = P.plot_image_mask_overlay(
        path, [0, 1, 2], cell_channel=0, nucleus_channel=1,
        pathogen_channel=2, figuresize=2, thickness=1, save_pdf=False,
        mode="outlines")

    def panel(i):
        return np.asarray(fig.axes[i].images[0].get_array())

    def has(p, rgb):
        return bool(np.any(np.all(np.isclose(p, rgb, atol=1e-6), axis=-1)))

    red, green, blue = (1., 0., 0.), (0., .5019607843137255, 0.), (0., 0., 1.)
    assert has(panel(0), red) and not has(panel(0), blue)
    assert has(panel(1), blue) and not has(panel(1), red)
    assert has(panel(2), green) and not has(panel(2), red)
    assert [a.get_title() for a in fig.axes][:3] == [
        "cell (channel 0)", "nucleus (channel 1)", "pathogen (channel 2)"]


# ---------------------------------------------------------------------------
# The default behind read_and_plot__vision_results' dead re-check
# ---------------------------------------------------------------------------

def test_vision_results_use_the_documented_default_y_limits(tmp_path):
    """plot.py:4077 -- ``y_lim`` is never None by the time it is applied.

    ``y_lim=None`` is replaced with ``[0.8, 0.9]`` before any file is read,
    so the ``if y_lim is not None`` guard at the bottom is always true. The
    documented default is the observable half, and an explicit value has to
    win over it.
    """
    for model, acc in (("alpha", 0.81), ("beta", 0.88)):
        epoch = tmp_path / model / "epoch_1"
        epoch.mkdir(parents=True)
        pd.DataFrame({"accuracy": [acc]}).to_csv(
            epoch / f"{model}_time0_test_result.csv", index=False)

    P.read_and_plot__vision_results(str(tmp_path))
    assert plt.gcf().axes[0].get_ylim() == (0.8, 0.9)

    P.read_and_plot__vision_results(str(tmp_path), y_lim=[0.0, 1.0])
    ax = plt.gcf().axes[0]
    assert ax.get_ylim() == (0.0, 1.0)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["alpha", "beta"]
    assert os.path.isdir(os.path.join(str(tmp_path), "result"))
