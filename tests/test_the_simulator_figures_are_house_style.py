"""``spacr.sim`` draws its nine diagnostics in the house style.

These are a simulation's own diagnostics rather than a user's data, so the
captions are plainer than elsewhere -- but the rules are the same ones, and
this module broke every one of them:

* a ROC curve and its random-classifier diagonal were both 0.5 pt black, so
  the null read exactly as loudly as the result;
* the active and inactive score distributions were slategray and teal at full
  strength, and **the legend named them the wrong way round**;
* ``save_plot`` and ``save_shap_plot`` wrote ``format='pdf', dpi=600``
  directly -- two of the literal formats instruction 136 exists to remove;
* the correlation matrix was a diverging map with no centre, so an
  all-positive block ran to the hot end and looked like a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from spacr import sim as S  # noqa: E402
from spacr.figures.style import (ROLES, TYPE_SCALE, Palette,  # noqa: E402
                                 resolve_ink, theme_target)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


#: What this module used to draw with. Not one of them is in the palette.
OLD_SIM_HUES = ("teal", "slategray", "red", "purple", "lightgrey", "b")


def _sweep_results(n_reps=2, accuracies=(0.8, 0.95)):
    """A tidy sweep table: nr_plates varies within each accuracy condition."""
    rows = []
    for accuracy in accuracies:
        for plates in (1, 2, 4):
            for rep in range(n_reps):
                rows.append({
                    'number_of_active_genes': 10,
                    'number_of_control_genes': 5,
                    'avg_reads_per_gene': 100,
                    'classifier_accuracy': accuracy,
                    'nr_plates': plates,
                    'number_of_genes': 30,
                    'avg_genes_per_well': 5,
                    'avg_cells_per_well': 20,
                    'sequencing_error': 0.01,
                    'well_ineq_coeff': 1.2,
                    'gene_ineq_coeff': 1.2,
                    'prauc': 0.1 * plates + accuracy + 0.01 * rep,
                })
    return pd.DataFrame(rows)


def _curve():
    fpr = np.linspace(0, 1, 20)
    return pd.DataFrame({"fpr": fpr, "tpr": np.sqrt(fpr)})


# --------------------------------------------------------------------------- #
#  The shared panel helpers.
# --------------------------------------------------------------------------- #

def test_the_roc_curve_outranks_its_null_diagonal():
    """They were both 0.5 pt black: the null read as loudly as the result."""
    figure, axis = plt.subplots()
    S.plot_roc_pr(_curve(), axis, "ROC", "fpr", "tpr")

    curve, diagonal = axis.lines[0], axis.lines[1]
    assert to_hex(curve.get_color()) == to_hex(ROLES["highlight"])
    assert curve.get_linestyle() == "-"
    assert to_hex(diagonal.get_color()) == to_hex(ROLES["reference"])
    assert diagonal.get_linestyle() != "-"
    assert diagonal.get_linewidth() < curve.get_linewidth()
    assert axis.get_legend().get_frame_on() is False


def test_the_confusion_matrix_annotations_follow_the_theme():
    """They were hard-coded black, invisible on spaCR's dark ground."""
    figure, axis = plt.subplots()
    S.plot_confusion_matrix(np.array([[8, 2], [1, 9]]), axis, "Confusion")

    ink = to_hex(resolve_ink(theme_target()))
    written = [t for t in axis.texts if "True Neg" in t.get_text()
               or "True Pos" in t.get_text()]
    assert len(written) == 2
    for text in written:
        assert to_hex(text.get_color()) == ink
        assert text.get_fontsize() == pytest.approx(TYPE_SCALE["annotation"])
    # The counts themselves are unchanged.
    assert any("8" in t.get_text() for t in written)
    assert any("9" in t.get_text() for t in written)


def test_the_confusion_matrix_ramp_is_the_single_hue_one():
    figure, axis = plt.subplots()
    S.plot_confusion_matrix(np.array([[8, 2], [1, 9]]), axis, "Confusion")
    ramp = plt.get_cmap(Palette.SEQUENTIAL)
    assert axis.collections[0].cmap(0.9) == pytest.approx(ramp(0.9))


# --------------------------------------------------------------------------- #
#  The sweep figures.
# --------------------------------------------------------------------------- #

def test_the_sweep_band_takes_the_lines_own_hue():
    """It was a blue line over a separate grey band at 0.5."""
    figure = S.plot_simulations(_sweep_results(), "nr_plates")
    axis = [a for a in figure.axes if a.get_visible()][0]

    line = axis.lines[0]
    assert to_hex(line.get_color()) == to_hex(ROLES["highlight"])
    band = axis.collections[0]
    assert to_hex(band.get_facecolor()[0]) == to_hex(ROLES["highlight"])
    assert float(band.get_alpha()) == pytest.approx(0.25)
    for gone in OLD_SIM_HUES:
        assert to_hex(line.get_color()) != to_hex(gone)


def test_the_sweep_means_are_the_means_they_always_were():
    """A restyle must not move a number."""
    frame = _sweep_results()
    figure = S.plot_simulations(frame, "nr_plates")
    axis = [a for a in figure.axes if a.get_visible()][0]

    accuracy = frame["classifier_accuracy"].unique()[0]
    subset = frame[frame["classifier_accuracy"] == accuracy]
    expected = subset.groupby("nr_plates")["prauc"].mean().sort_index()
    drawn = axis.lines[0].get_ydata()
    assert np.asarray(drawn) == pytest.approx(expected.to_numpy())


def test_the_verbose_note_lost_its_white_box():
    figure = S.plot_simulations(_sweep_results(), "nr_plates", verbose=True)
    axis = [a for a in figure.axes if a.get_visible()][0]
    notes = [t for t in axis.texts if "classifier_accuracy" in t.get_text()]
    assert notes
    assert notes[0].get_bbox_patch() is None
    assert to_hex(notes[0].get_color()) == to_hex(resolve_ink(theme_target()))


def test_the_correlation_matrix_is_centred_on_zero():
    """It was diverging but unbounded, so seaborn scaled it to the data."""
    figure = S.plot_correlation_matrix(_sweep_results(), dst=None)
    image = figure.axes[0].collections[0]
    assert (image.norm.vmin, image.norm.vmax) == (-1.0, 1.0)
    # ...and no white rules between the cells.
    assert float(np.ravel(image.get_linewidths())[0]) == 0.0


def test_an_explicit_colormap_is_no_longer_thrown_away():
    """``cmap`` defaulted to 'inferno' and was overwritten two lines later,
    so a caller who passed one got the diverging map anyway."""
    figure = S.plot_correlation_matrix(_sweep_results(), cmap="magma", dst=None)
    image = figure.axes[0].collections[0]
    assert image.cmap(0.9) == pytest.approx(plt.get_cmap("magma")(0.9))


# --------------------------------------------------------------------------- #
#  Rule 4: no figure writes its own format.
# --------------------------------------------------------------------------- #

def test_save_plot_goes_through_the_format_preference(tmp_path, monkeypatch):
    """It was ``fig.savefig(..., format='pdf', dpi=600)``."""
    seen = {}
    real = S.save_figure

    def spy(fig, path, **kwargs):
        seen["path"] = path
        return real(fig, path, **kwargs)

    monkeypatch.setattr(S, "save_figure", spy)
    figure = plt.figure()
    S.save_plot(figure, str(tmp_path), "well_ineq_coeff", 12)

    assert seen["path"] == f"{tmp_path}/well_ineq_coeff/12_figure.pdf"
    written = list((tmp_path / "well_ineq_coeff").iterdir())
    assert len(written) == 1 and written[0].stem == "12_figure"


def test_save_shap_plot_goes_through_the_format_preference(tmp_path, monkeypatch):
    seen = {}
    real = S.save_figure

    def spy(fig, path, **kwargs):
        seen["path"] = path
        return real(fig, path, **kwargs)

    monkeypatch.setattr(S, "save_figure", spy)
    S.save_shap_plot(plt.figure(), str(tmp_path), "shap", 7)
    assert seen["path"] == f"{tmp_path}/shap/7_figure.pdf"
    written = list((tmp_path / "shap").iterdir())
    assert len(written) == 1 and written[0].stem == "7_figure"


def test_no_sim_figure_leaves_its_style_on_the_globals(tmp_path):
    """Rule 2, over everything reachable without a full simulation."""
    before = dict(matplotlib.rcParams)

    frame = _sweep_results()
    S.plot_simulations(frame, "nr_plates")
    S.plot_correlation_matrix(frame, dst=None)
    S.vis_dists([np.random.default_rng(i).random(30) for i in range(6)],
                str(tmp_path), "v", 1)

    after = dict(matplotlib.rcParams)
    changed = [key for key in before
               if repr(before[key]) != repr(after.get(key))]
    assert not changed, f"spacr.sim leaked rcParams: {sorted(changed)}"


def test_the_distribution_previews_use_the_styles_fill(tmp_path):
    """Six single-series panels, so none of them has a claim: teal is out."""
    captured = {}
    real = S.save_plot
    S.save_plot = lambda fig, src, variable, i: captured.setdefault("fig", fig)
    try:
        S.vis_dists([np.random.default_rng(i).random(30) for i in range(6)],
                    str(tmp_path), "v", 1)
    finally:
        S.save_plot = real

    # element="step" with a fill gives seaborn a PolyCollection rather than
    # one patch per bin, so the fill colour is read off the collection.
    axis = captured["fig"].axes[0]
    drawn = [to_hex(collection.get_facecolor()[0])
             for collection in axis.collections
             if len(collection.get_facecolor())]
    assert drawn, "the histogram did not draw"
    assert set(drawn) == {to_hex(ROLES["fill"])}
    assert to_hex("teal") not in drawn
