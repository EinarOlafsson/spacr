"""``spacr.submodules`` draws its twelve figures in the house style.

The assay modules -- Cellpose diagnostics, the invasion panels, the score
heatmaps, the post-regression effect sizes -- built every one of their figures
without touching ``spacr/figures/style.py``. These tests DRAW each of them on
a small synthetic input and then look at the artists that landed on the axes,
because that is what a reader sees.

What is asserted, and why each one is a failure this repository has had:

* **Everything is grey except what the sentence is about.** A viridis ramp
  across forty gRNA bars encodes nothing the x axis does not already say; a
  crimson threshold line and a steelblue reference line are two equally loud
  marks, so the panel does not say which one the classifier used.
* **The globals come back.** These figures are drawn from a long-lived GUI, so
  one ``rcParams.update`` styles every later figure in the session.
* **The ink follows the theme**, rather than a near-black that is invisible on
  spaCR's dark ground.
* **The numbers did not move.** This is a restyle, so the effect sizes, the
  histogram counts and the correlation matrix are all re-derived here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from spacr import submodules as S  # noqa: E402
from spacr.figures.style import (ROLES, TYPE_SCALE, Palette,  # noqa: E402
                                 resolve_ink, theme_target)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


#: The hues the invasion QC panels used to draw with. None is in the palette.
OLD_INVASION_HUES = ("crimson", "steelblue")


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

def _invasion_frames(wells=2, per_well=60, seed=1):
    """A classified parasite table and its per-well thresholds.

    Two clean modes per well, so the threshold has something real to sit
    between and the histogram has a shape rather than a smear.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(wells):
        prc = f"plate1_r1_c{well + 1}"
        low = rng.normal(10, 2, per_well // 2)
        high = rng.normal(100, 5, per_well // 2)
        for value in np.concatenate([low, high]):
            rows.append({"prc": prc, "outside_intensity": float(value)})
    parasites = pd.DataFrame(rows)
    well_rows = [{
        "prc": f"plate1_r1_c{well + 1}",
        "threshold_median": 55.0,
        "reference_threshold_median": 5.0,
        "threshold_source": "otsu",
        "bimodality_coefficient": 0.71,
        "n_total": per_well,
    } for well in range(wells)]
    return parasites, pd.DataFrame(well_rows)


def _regression_csv(path):
    """The fixture ``tests/test_cov_submodules_heatmap_postreg.py`` designed.

    Built so the correlation matrix is exactly +-1 and the propagated effect
    sizes are known in closed form -- which is what makes "the restyle did not
    move a number" checkable rather than merely claimed.
    """
    v = np.linspace(0.1, 0.6, 6)
    records = []
    for index, value in enumerate(v):
        prc = f"plate1_r{index + 1}_c3"
        records += [
            {"prc": prc, "grna": "g2", "fraction": 1.0 - value},
            {"prc": prc, "grna": "g3", "fraction": 2.0 * value},
            {"prc": prc, "grna": "g4", "fraction": 3.0 * (1.0 - value)},
            {"prc": prc, "grna": "g1", "fraction": value / 4.0},
            {"prc": prc, "grna": "g1", "fraction": 3.0 * value / 4.0},
        ]
    pd.DataFrame(records).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
#  Rule 2, first: nothing leaks.
# --------------------------------------------------------------------------- #

def test_no_submodules_figure_leaves_its_style_on_the_globals(tmp_path):
    """Draw the reachable figures, then compare matplotlib's globals.

    ``figure_style`` is a context manager for exactly this reason: a global
    write here would restyle every later figure in the session, in every other
    module, until the process exits.
    """
    before = dict(matplotlib.rcParams)

    images = [np.random.default_rng(0).random((8, 8)) for _ in range(2)]
    labels = [np.zeros((8, 8), dtype=int) for _ in range(2)]
    labels[0][2:5, 2:5] = 1
    S.plot_cellpose_batch(images, labels)

    parasites, wells = _invasion_frames()
    S._invasion_threshold_panels(parasites, wells)

    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    S.post_regression_analysis(str(csv), {"g1": 1.0, "g2": 0.0},
                               ["g1", "g2", "g3", "g4"], save=False)

    after = dict(matplotlib.rcParams)
    changed = [key for key in before
               if repr(before[key]) != repr(after.get(key))]
    assert not changed, f"spacr.submodules leaked rcParams: {sorted(changed)}"


# --------------------------------------------------------------------------- #
#  The Cellpose micrograph rows.
# --------------------------------------------------------------------------- #

def test_the_cellpose_preview_is_a_styled_micrograph_row():
    """Greyscale per channel, label maps in their identity colours, no frame."""
    rng = np.random.default_rng(0)
    images = [rng.random((8, 8)) for _ in range(3)]
    labels = []
    for index in range(3):
        mask = np.zeros((8, 8), dtype=int)
        mask[1:4, 1:4] = index + 1
        labels.append(mask)

    S.plot_cellpose_batch(images, labels)
    figure = plt.gcf()
    assert len(figure.axes) == 6

    for index in range(3):
        top, bottom = figure.axes[index], figure.axes[3 + index]
        assert top.images[0].get_cmap().name == "gray"
        # A mask's colours are identities, not a quantity, so the random map
        # stays: it is the one place a many-hue map is the data.
        assert bottom.images[0].get_cmap().name != "gray"
        for axis in (top, bottom):
            assert not axis.axison, "a micrograph must carry no axes furniture"

    ink = resolve_ink(theme_target())
    assert to_hex(figure.axes[0].title.get_color()) == to_hex(ink)
    assert figure.axes[0].title.get_fontsize() == pytest.approx(
        TYPE_SCALE["label"])


# --------------------------------------------------------------------------- #
#  The invasion QC panels.
# --------------------------------------------------------------------------- #

def test_the_threshold_panels_colour_the_applied_cut_and_grey_the_reference():
    """One line is the classification, the other is what it was judged against.

    They were crimson at 1.5 pt and steelblue at 1.2 pt -- two equally loud
    marks over a saturated viridis-green histogram, so nothing on the panel
    said which cut the classifier actually applied.
    """
    parasites, wells = _invasion_frames()
    figure = S._invasion_threshold_panels(parasites, wells)
    axis = figure.axes[0]

    drawn = [line for line in axis.lines
             if len(line.get_xdata()) and np.isfinite(line.get_xdata()[0])]
    applied = [line for line in drawn
               if to_hex(line.get_color()) == to_hex(ROLES["highlight"])]
    reference = [line for line in drawn
                 if to_hex(line.get_color()) == to_hex(ROLES["reference"])]
    assert len(applied) == 1
    assert float(applied[0].get_xdata()[0]) == pytest.approx(55.0)
    assert applied[0].get_linestyle() == "-"

    assert reference, "the reference threshold did not draw"
    on_axes = [line for line in reference if len(line.get_xdata()) == 2
               and float(line.get_xdata()[0]) == pytest.approx(5.0)]
    assert on_axes, "the reference is not at the reference threshold"
    assert on_axes[0].get_linestyle() != "-"
    assert on_axes[0].get_linewidth() <= 1.0

    for gone in OLD_INVASION_HUES:
        assert all(to_hex(line.get_color()) != to_hex(gone) for line in drawn)


def test_the_threshold_histogram_takes_the_styles_fill():
    """It was the middle of viridis: a saturated green, louder than the claim."""
    parasites, wells = _invasion_frames()
    figure = S._invasion_threshold_panels(parasites, wells)
    axis = figure.axes[0]
    assert axis.patches, "the histogram did not draw"
    fill = to_rgba(axis.patches[0].get_facecolor())
    assert fill[:3] == pytest.approx(to_rgba(ROLES["fill"])[:3])
    assert fill[:3] != pytest.approx(to_rgba(plt.get_cmap("viridis")(0.5))[:3])


def test_a_caller_that_asks_for_a_colormap_still_gets_one():
    """The house fill is the DEFAULT, not a refusal to honour ``cmap``."""
    parasites, wells = _invasion_frames()
    figure = S._invasion_threshold_panels(parasites, wells, cmap="magma")
    fill = to_rgba(figure.axes[0].patches[0].get_facecolor())
    assert fill[:3] == pytest.approx(to_rgba(plt.get_cmap("magma")(0.5))[:3])


def test_the_threshold_panels_still_histogram_every_parasite():
    """A restyle must not drop a count."""
    parasites, wells = _invasion_frames(wells=1, per_well=60)
    figure = S._invasion_threshold_panels(parasites, wells)
    counted = sum(patch.get_height() for patch in figure.axes[0].patches)
    assert counted == pytest.approx(60)


def test_an_unusable_colormap_name_falls_back_to_the_house_fill():
    """It used to fall back to a bare '0.6' grey with no place in the palette."""
    parasites, wells = _invasion_frames()
    figure = S._invasion_threshold_panels(parasites, wells, cmap="not-a-cmap")
    fill = to_rgba(figure.axes[0].patches[0].get_facecolor())
    assert fill[:3] == pytest.approx(to_rgba(ROLES["fill"])[:3])


# --------------------------------------------------------------------------- #
#  Post-regression: the correlation heatmap and the effect-size bars.
# --------------------------------------------------------------------------- #

def test_the_effect_size_bars_are_grey_with_the_anchors_highlighted(tmp_path):
    """``palette='viridis'`` gave every gRNA its own hue and said nothing.

    The gRNAs whose effect was FIXED by ``grna_dict`` are the ones a reader
    has to be able to pick out from the ones propagated from them, so those
    are the coloured minority.
    """
    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    anchors = {"g1": 1.0, "g2": 0.0}
    S.post_regression_analysis(str(csv), anchors, ["g1", "g2", "g3", "g4"],
                               save=False)

    bars = plt.figure(plt.get_fignums()[-1]).axes[0]
    faces = [to_hex(patch.get_facecolor()) for patch in bars.patches]
    assert len(faces) == 4
    assert faces.count(to_hex(ROLES["highlight"])) == 2, "one per anchor"
    assert faces.count(to_hex(ROLES["data"])) == 2
    assert set(faces) == {to_hex(ROLES["highlight"]), to_hex(ROLES["data"])}


def test_the_effect_sizes_themselves_did_not_move(tmp_path):
    """The propagated values, recomputed from the fixture's closed form."""
    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    S.post_regression_analysis(str(csv), {"g1": 1.0, "g2": 0.0},
                               ["g1", "g2", "g3", "g4"], save=False)

    bars = plt.figure(plt.get_fignums()[-1]).axes[0]
    heights = [patch.get_height() for patch in bars.patches]
    assert heights == pytest.approx([1.0, 0.0, 0.5, 0.0])


def test_the_correlation_heatmap_is_centred_on_zero(tmp_path):
    """A signed quantity earns a diverging map, and it has to be centred.

    ``coolwarm`` with no ``vmin``/``vmax`` scales to the data, so an
    all-positive matrix came out red end to end and looked like a finding.
    Found while restyling; fixed rather than filed.
    """
    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    S.post_regression_analysis(str(csv), {"g1": 1.0}, ["g1", "g2", "g3", "g4"],
                               save=False)

    heatmap = plt.figure(plt.get_fignums()[0]).axes[0]
    image = heatmap.collections[0]
    # seaborn rebuilds the map under its own name, so the map is identified by
    # the colours it produces rather than by ``cmap.name``.
    diverging = plt.get_cmap("coolwarm")
    assert image.cmap(0.0) == pytest.approx(diverging(0.0)), "a correlation IS signed"
    assert image.cmap(1.0) == pytest.approx(diverging(1.0))
    assert (image.norm.vmin, image.norm.vmax) == (-1.0, 1.0)


def test_the_correlation_matrix_is_the_matrix_it_always_was(tmp_path):
    """The fixture makes every correlation exactly +-1."""
    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    S.post_regression_analysis(str(csv), {"g1": 1.0}, ["g1", "g2", "g3", "g4"],
                               save=False)

    drawn = plt.figure(plt.get_fignums()[0]).axes[0].collections[0].get_array()
    assert sorted(set(np.round(np.asarray(drawn).ravel(), 6))) == [-1.0, 1.0]


def test_the_post_regression_panels_take_their_ink_from_the_theme(tmp_path):
    """Rule 3, on both of them."""
    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    S.post_regression_analysis(str(csv), {"g1": 1.0}, ["g1", "g2", "g3", "g4"],
                               save=False)

    ink = to_hex(resolve_ink(theme_target()))
    for number in plt.get_fignums():
        axis = plt.figure(number).axes[0]
        assert to_hex(axis.xaxis.label.get_color()) == ink
        assert not axis.xaxis.get_gridlines()[0].get_visible()


def test_the_long_grna_names_rotate_forty_five_degrees(tmp_path):
    """The style rotates a long categorical axis 45 and anchors it right."""
    csv = tmp_path / "regression.csv"
    _regression_csv(csv)
    S.post_regression_analysis(str(csv), {"g1": 1.0}, ["g1", "g2", "g3", "g4"],
                               save=False)
    bars = plt.figure(plt.get_fignums()[-1]).axes[0]
    labels = bars.get_xticklabels()
    assert labels
    assert all(label.get_rotation() == pytest.approx(45) for label in labels)
    assert all(label.get_ha() == "right" for label in labels)


# --------------------------------------------------------------------------- #
#  The series palette.
# --------------------------------------------------------------------------- #

def test_the_series_colours_are_the_published_palette():
    """Every line colour a multi-series plot can reach comes from the palette.

    It used to be seaborn 'deep' expanded to a hundred colours and reordered
    by an index list -- a hundred hues is a hundred series nobody can tell
    apart, and the pale ones vanished on the dark ground.
    """
    published = {value for name, value in vars(Palette).items()
                 if name.isupper() and isinstance(value, str)
                 and value.startswith("#")}
    assert set(S.SERIES_COLOURS) <= published
    # Fixed order, so a series cannot change colour between two panels.
    assert S.SERIES_COLOURS[0] == Palette.BLUE
    assert len(set(S.SERIES_COLOURS)) == len(S.SERIES_COLOURS)
