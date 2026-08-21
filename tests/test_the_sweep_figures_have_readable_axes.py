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


#: The measurement families a real screen's columns fall into. Named so the
#: family panel has something to group BY -- with two columns called `real`
#: and `noise` it grouped everything into "other" and drew one bar.
FAMILIES = ("cell_area", "nucleus_area", "pathogen_area", "cytoplasm_area",
            "cell_channel_1_mean_intensity",
            "nucleus_channel_2_mean_intensity")


@pytest.fixture(scope="module")
def result():
    """A screen-shaped sweep: 12 genes x 3 guides, 40 measurements, 4 plates.

    THE OLD FIXTURE WAS TWO GUIDES AND THREE COLUMNS, on one gene each, and
    three of the ten figures could not draw on it at all -- `plot_circularity`,
    `plot_gene_similarity` and `plot_guide_concordance` came back None and the
    tests SKIPPED. A skip is not a pass: the two most important checks in a
    pooled screen, "do a gene's guides agree" and "is this just the
    classification score", were untested for as long as they have existed.

    They need what a real library has and the old fixture did not: SEVERAL
    GUIDES PER GENE (concordance is a statement about guides of one gene),
    ENOUGH GENES TO CLUSTER, and per-well SCORES (circularity is the
    correlation between an effect and the classification score).
    """
    rng = np.random.default_rng(0)
    genes, per_gene, measures, plates, n = 12, 3, 40, 4, 240
    index = [f"plate{1 + i // (n // plates)}_r{i // 24}_c{i % 24}"
             for i in range(n)]

    fractions, drivers = {}, {}
    for gene in range(genes):
        shared = rng.random(n)
        drivers[gene] = shared
        for guide in range(per_gene):
            # THE GUIDES OF A GENE SHARE A DRIVER but are not identical, so
            # concordance has something real to measure rather than 1.0.
            fractions[f"TGGT1_{200000 + gene:06d}_{guide + 1}"] = (
                shared * 0.7 + rng.random(n) * 0.3)

    wells = {}
    for m in range(measures):
        name = f"{FAMILIES[m % len(FAMILIES)]}_{m}"
        # The first eight measurements carry a real effect; the rest are the
        # null the correction has to hold against.
        strength = 2.5 if m < 8 else 0.0
        wells[name] = drivers[m % genes] * strength + rng.normal(0, 1, n)

    return sweep_module.sweep(
        pd.DataFrame(wells, index=index),
        pd.DataFrame(fractions, index=index),
        blocks=[i.split("_")[0] for i in index],
        scores=list(rng.random(n)),
        level="both")


def _plot_names():
    return sorted(n for n in dir(sweep_module) if n.startswith("plot_"))


def _a_gene(result) -> str:
    """A name the result actually holds, for the per-gene panels.

    The hard-coded "A" was the old fixture's guide name. On a library shaped
    like a real one it names nothing, and `plot_gene_profile` returned None
    -- so the test SKIPPED rather than failed, and a per-gene panel that
    could not draw looked exactly like one that had nothing to draw.
    """
    return str(next(iter(result.effects.index)))


def _draw(name, result):
    function = getattr(sweep_module, name)
    parameters = inspect.signature(function).parameters
    if "gene" in parameters:
        return function(result, _a_gene(result))
    return function(result)


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
    figure = (function(result, _a_gene(result), path=str(out))
              if "gene" in parameters else function(result, path=str(out)))
    if figure is None or not out.exists():
        pytest.skip(f"{name} had nothing to draw on this fixture")

    from PIL import Image

    image = Image.open(out).convert("RGB")
    colours = Counter(image.getdata())
    dark = sum(n for colour, n in colours.items() if sum(colour) / 3 < 100)

    # THE PAGE IS THE MARGIN, NOT THE MODAL COLOUR. This read
    # `colours.most_common(1)`, which is the page only while the page has more
    # pixels than any single ink -- and `plot_sweep` is a heatmap on a
    # saturated diverging map, where two cell colours came to 194,087 and
    # 193,609 against white's 191,196. A three-way tie decided by layout: the
    # assertion flipped when the house style changed the label metrics by a
    # few points, while the saved file was white the whole time.
    #
    # `bbox_inches="tight"` trims to the artists, so what is left in the
    # corners IS the page.
    width, height = image.size
    corners = [image.getpixel(p) for p in
               ((0, 0), (width - 1, 0), (0, height - 1),
                (width - 1, height - 1))]
    for page in corners:
        assert sum(page) / 3 > 200, (
            f"{name} saved onto a dark page: {page}. A figure pasted into a "
            f"manuscript is going onto white.")
    assert dark > 500, (
        f"{name} saved with only {dark} dark pixels -- the axes and the text "
        f"are the theme's light ink on a white page, which is the "
        f"'i cannot see any of the x or y axes' report.")


# --------------------------------------------------------------------------- #
#  175's last item: which of the ten earn their place
# --------------------------------------------------------------------------- #

#: What each panel is FOR, in one line. Measured on a screen-shaped fixture
#: -- 12 genes x 3 guides, 40 measurements, 4 plates -- rather than argued:
#: every one of the ten draws, and no two answer the same question.
#:
#: THE CUT IS THAT THERE IS NO CUT, and the evidence is here so that dropping
#: one later is a decision rather than an archaeology exercise. The closest
#: pair is `plot_measurement_families` and `plot_measurement_hits`, and they
#: are the two MARGINS of one matrix: what kind of thing a gene moves, and
#: which measurements are moved most. A reader chasing a hit needs the first;
#: a reader asking whether one measurement is doing all the work needs the
#: second.
WHAT_EACH_ONE_ANSWERS = {
    "plot_sweep":
        "which guide/measurement associations survived the correction",
    "plot_grid_volcano":
        "how large those effects are and how significant, together",
    "plot_effect_against_representation":
        "whether a gene's hit count is explained by its statistical weight "
        "rather than by biology -- the confound check",
    "plot_measurement_families":
        "what KIND of thing a gene moves: pathogen, nucleus, shape, "
        "intensity",
    "plot_measurement_hits":
        "the other margin: which measurements are moved most, across genes",
    "plot_gene_profile":
        "one gene's effects across every measurement, in detail",
    "plot_guide_concordance":
        "whether a gene's guides agree -- the single most important QC in a "
        "pooled screen, and the one no other panel makes",
    "plot_gene_similarity":
        "which genes behave alike, which is where a pathway hypothesis "
        "comes from",
    "plot_circularity":
        "whether an association is just the classification score again",
    "plot_calibration":
        "whether the null is right, which decides if any of the above can "
        "be believed",
}


def test_every_panel_is_accounted_for():
    """A panel with no entry is one nobody has argued for."""
    assert sorted(WHAT_EACH_ONE_ANSWERS) == _plot_names()


def test_no_two_answer_the_same_question():
    said = list(WHAT_EACH_ONE_ANSWERS.values())

    assert len(set(said)) == len(said)


@pytest.mark.parametrize("name", sorted(WHAT_EACH_ONE_ANSWERS))
def test_every_one_of_them_actually_draws(name, result):
    """THE MEASUREMENT BEHIND THE PICK, and the thing the old fixture hid.

    Three of these returned None on the previous fixture and their tests
    SKIPPED -- including `plot_guide_concordance` and `plot_circularity`, the
    two checks a pooled screen most depends on. A skip is not a pass, and a
    panel that cannot draw looks exactly like one with nothing to draw.
    """
    figure = _draw(name, result)

    assert figure is not None, f"{name} drew nothing on a screen-shaped sweep"
    assert figure.axes, name
