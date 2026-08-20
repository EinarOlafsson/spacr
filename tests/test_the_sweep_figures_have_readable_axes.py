"""Instruction 175 — "i cannot see any of the x or y axes".

Reported 2026-08-19 of the sweep figures. The measurement, taken 2026-08-20:

* every figure that draws anything HAS x and y labels or tick labels;
* the LIVE figure's ink is the theme's, which on a dark session is #E8EDEE --
  correct on screen and invisible on paper;
* the SAVED figure is a white page with dark ink, because instruction 150's
  chrome flip runs on the way to disk. Measured on `plot_sweep`: page
  (255,255,255), 6,928 pure-black pixels and 355,788 dark ones.

So the two are different on purpose and both are right — which is precisely
the thing that is easy to break, because the screen keeps looking fine while
the file stops being readable. That is what this holds.
"""
from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import spacr.gene_measurement_sweep as sweep_module        # noqa: E402


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    n = 80
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    a = rng.random(n)
    wells = pd.DataFrame({
        "real": a * 3.0 + rng.normal(0, 0.2, n),
        "noise": rng.normal(0, 1, n),
        "area": a * 2.0 + rng.normal(0, 0.3, n),
    }, index=index)
    fractions = pd.DataFrame({"A": a, "B": rng.random(n)}, index=index)
    return sweep_module.sweep(wells, fractions,
                              blocks=[i.split("_")[0] for i in index])


def _plot_names():
    return sorted(n for n in dir(sweep_module) if n.startswith("plot_"))


def _draw(name, result):
    function = getattr(sweep_module, name)
    parameters = inspect.signature(function).parameters
    return function(result, "A") if "gene" in parameters else function(result)


def test_there_are_ten_of_them_to_pick_from():
    """175's remaining item is the maintainer's pick of which earn a place."""
    assert len(_plot_names()) == 10


@pytest.mark.parametrize("name", _plot_names())
def test_a_figure_that_draws_anything_labels_both_axes(name, result):
    figure = _draw(name, result)
    if figure is None:
        pytest.skip(f"{name} had nothing to draw on this fixture")
    axes = figure.axes[0]

    def labelled(label, ticks):
        return bool(label) or any(t.get_text() for t in ticks)

    assert labelled(axes.get_xlabel(), axes.get_xticklabels()), (
        f"{name} has no x axis a reader can identify")
    assert labelled(axes.get_ylabel(), axes.get_yticklabels()), (
        f"{name} has no y axis a reader can identify")


@pytest.mark.parametrize("name", _plot_names())
def test_the_saved_file_is_readable_on_paper(name, result, tmp_path):
    """The half that breaks silently: the screen keeps looking fine."""
    function = getattr(sweep_module, name)
    parameters = inspect.signature(function).parameters
    out = tmp_path / f"{name}.png"
    figure = (function(result, "A", path=str(out)) if "gene" in parameters
              else function(result, path=str(out)))
    if figure is None or not out.exists():
        pytest.skip(f"{name} had nothing to draw on this fixture")

    from PIL import Image

    image = Image.open(out).convert("RGB")
    colours = Counter(image.getdata())
    page = colours.most_common(1)[0][0]
    dark = sum(n for colour, n in colours.items() if sum(colour) / 3 < 100)

    assert sum(page) / 3 > 200, (
        f"{name} saved onto a dark page: {page}. A figure pasted into a "
        f"manuscript is going onto white.")
    assert dark > 500, (
        f"{name} saved with only {dark} dark pixels -- the axes and the text "
        f"are the theme's light ink on a white page, which is the "
        f"'i cannot see any of the x or y axes' report.")
