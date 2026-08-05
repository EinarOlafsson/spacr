"""PCA: the statistics first, then the picture, then the link.

Every assertion here is on a number that can be worked out by hand or on a
behaviour a user would notice. A PCA screen that lays out beautifully and
returns "PC1 = cell_area, 99% of the variance" because nobody standardised the
features is the failure worth catching, and it is the one that looks fine.

The exact dataset
-----------------
:func:`equicorrelated_frame` is built from Hadamard sign vectors, which are
*exactly* orthogonal and have *exactly* equal variance, so the correlation
matrix is exactly equicorrelated with rho = 1/2 and the answer is known in
closed form rather than approximately:

    f_i = a + e_i,  with a, e_1, e_2, e_3 orthogonal and of equal variance
    => var(f_i) = 2v, cov(f_i, f_j) = v, corr = 1/2
    => the correlation matrix is (1 - rho) I + rho J with p = 3, rho = 1/2
    => eigenvalues 1 + (p-1)rho = 2 (once) and 1 - rho = 1/2 (twice)
    => PC1 takes exactly 2/3 of the variance, along exactly (1, 1, 1)/sqrt(3).

There is no random seed in that and no tolerance worth arguing about. It is
also, deliberately, the "PC1 is just size" case: every feature loads the same
way, which is what :meth:`PCAResult.is_size_like` exists to say out loud.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_builder import GraphCanvas
from spacr.qt.widgets.graph_spec import SCATTER, GraphSpec
from spacr.qt.widgets.pca_model import (
    DEGENERATE_RATIO, NAN_AUTO, NAN_COMPLETE, NAN_DROP_FEATURES, NAN_MEAN,
    SCALE_NONE, SCALE_ZSCORE, PCAError, PCASpec, candidate_features,
    component_index, component_name, pca,
)
from spacr.qt.widgets.pca_view import PCAPanel, PCAScoresCanvas, arrow_scale


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Four mutually orthogonal +/-1 vectors of length 8 (Hadamard columns 2..5).
_SIGNS = np.array([
    [+1, +1, +1, +1, -1, -1, -1, -1],   # a  — the shared direction
    [+1, +1, -1, -1, +1, +1, -1, -1],   # e1
    [+1, -1, +1, -1, +1, -1, +1, -1],   # e2
    [+1, -1, -1, +1, -1, +1, +1, -1],   # e3
], dtype=float)


def equicorrelated_frame(repeats: int = 25) -> pd.DataFrame:
    """Three features sharing one exact direction; rho = 1/2 by construction."""
    a, e1, e2, e3 = (np.tile(row, repeats) for row in _SIGNS)
    n = a.size
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["r1"] * n,
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": np.arange(n),
        "f1": a + e1,
        "f2": a + e2,
        "f3": a + e3,
    })


def cluster_frame(n: int = 240, seed: int = 11) -> pd.DataFrame:
    """Two groups separated along one direction, in wildly different units.

    ``area`` is px²-sized, ``intensity`` is counts, ``eccentricity`` is a
    ratio — the spread that makes an unstandardised PCA answer a question
    about units rather than about cells.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    group = np.array(["control"] * half + ["knockdown"] * (n - half))
    shift = np.where(group == "control", -1.0, 1.0)
    return pd.DataFrame({
        "plateID": ["p1"] * (n // 2) + ["p2"] * (n - n // 2),
        "rowID": ["r1"] * n,
        "columnID": [f"c{i % 4 + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": np.arange(n),
        "area": 900.0 + 120.0 * shift + rng.normal(scale=25.0, size=n),
        "perimeter": 110.0 + 9.0 * shift + rng.normal(scale=2.0, size=n),
        "intensity": 140.0 - 6.0 * shift + rng.normal(scale=3.0, size=n),
        "eccentricity": 0.55 + rng.normal(scale=0.04, size=n),
        "gene": group,
    })


@pytest.fixture
def link() -> LinkedSelection:
    """A PRIVATE link — never the process-wide one, which every other open
    view listens to."""
    return LinkedSelection()


# ---------------------------------------------------------------------------
# The planted component
# ---------------------------------------------------------------------------

def test_pca_recovers_a_planted_direction_and_its_exact_variance_share():
    result = pca(equicorrelated_frame(),
                 PCASpec(features=("f1", "f2", "f3"), n_components=3))

    # Direction: (1, 1, 1)/sqrt(3), to the sign convention (largest |loading|
    # positive), which for an all-equal component makes every loading positive.
    assert result.features == ("f1", "f2", "f3")
    assert result.loadings[:, 0] == pytest.approx([1 / math.sqrt(3)] * 3)

    # Share: exactly 2/3, from eigenvalues 2, 1/2, 1/2 of the equicorrelated
    # matrix. The remaining two are degenerate copies of each other.
    assert result.explained_variance_ratio[0] == pytest.approx(2 / 3)
    assert result.explained_variance_ratio[1] == pytest.approx(1 / 6)
    assert result.explained_variance_ratio[2] == pytest.approx(1 / 6)
    assert result.explained_variance_ratio.sum() == pytest.approx(1.0)

    # Eigenvalues themselves, not just their shares: for a correlation PCA the
    # total is the number of features, so PC1 is 2 of 3.
    assert result.total_variance == pytest.approx(3.0)
    assert result.explained_variance[0] == pytest.approx(2.0)


def test_the_scores_of_a_planted_component_are_the_planted_variable():
    """PC1 must point at the shared direction — and only as well as it can.

    PC1 is (g1 + g2 + g3)/sqrt(3), which is proportional to 3a + e1 + e2 + e3:
    the three indicators average the shared direction up and their independent
    parts down, but not away. So the correlation with the planted ``a`` is
    exactly ``3 / sqrt(9 + 3) = sqrt(3)/2``, not 1 — a component estimated from
    three noisy measurements of a thing is not the thing, and asserting 1 here
    would be asserting something PCA does not claim.
    """
    frame = equicorrelated_frame()
    shared = np.tile(_SIGNS[0], 25)
    result = pca(frame, PCASpec(features=("f1", "f2", "f3"), n_components=1))
    assert abs(np.corrcoef(result.scores[:, 0], shared)[0, 1]) == \
        pytest.approx(math.sqrt(3) / 2)
    # And it is the *best* such estimate: better than any single feature.
    for feature in ("f1", "f2", "f3"):
        assert abs(np.corrcoef(frame[feature], shared)[0, 1]) < math.sqrt(3) / 2


def test_a_component_everything_loads_the_same_way_on_is_reported_as_size():
    """The single most common way a morphology PCA misleads its reader."""
    result = pca(equicorrelated_frame(), PCASpec(features=("f1", "f2", "f3")))
    assert result.sign_agreement(0) == pytest.approx(1.0)
    assert result.is_size_like(0)
    assert "general-magnitude" in result.headline(0)
    assert "general-magnitude" in result.report()


def test_the_feature_correlations_are_the_biplot_arrows_and_they_close():
    """r(feature, PC) is what an arrow is; the squared row sums are 1."""
    result = pca(equicorrelated_frame(),
                 PCASpec(features=("f1", "f2", "f3"), n_components=3))
    # corr(f_i, PC1) = sqrt(eigenvalue/p) * loading * sqrt(p) = sqrt(2/3).
    assert result.correlations[:, 0] == pytest.approx(
        [math.sqrt(2 / 3)] * 3, abs=1e-9)
    communality = (result.correlations ** 2).sum(axis=1)
    assert communality == pytest.approx([1.0, 1.0, 1.0])


def test_the_sign_convention_is_deterministic():
    """The same table must draw the same picture, twice and in two processes."""
    frame = cluster_frame()
    spec = PCASpec(features=("area", "perimeter", "intensity"))
    first, second = pca(frame, spec), pca(frame, spec)
    assert first.loadings == pytest.approx(second.loadings)
    for k in range(first.n_components):
        column = first.loadings[:, k]
        assert column[int(np.argmax(np.abs(column)))] > 0


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def test_not_scaling_hands_pc1_to_whichever_column_has_the_biggest_numbers():
    """The documented consequence of SCALE_NONE, asserted rather than trusted.

    ``area`` is px²-sized and ``eccentricity`` is a ratio. Unscaled, PC1 is
    area and almost nothing else; standardised, it is a combination — the same
    data, two different answers, and only one of them survives a change of
    units.
    """
    frame = cluster_frame()
    features = ("area", "perimeter", "intensity", "eccentricity")

    raw = pca(frame, PCASpec(features=features, scaling=SCALE_NONE))
    assert raw.dominant(0)[0] == "area"
    assert raw.dominant(0)[1] > 0.99          # PC1 is area, essentially alone
    assert raw.explained_variance_ratio[0] > 0.99

    scaled = pca(frame, PCASpec(features=features, scaling=SCALE_ZSCORE))
    assert scaled.dominant(0)[1] < 0.5        # no single feature owns it
    assert scaled.explained_variance_ratio[0] < raw.explained_variance_ratio[0]
    # And the three correlated features all load on it, which is the structure
    # the units were hiding.
    loadings = dict(zip(scaled.features, scaled.loadings[:, 0]))
    assert abs(loadings["area"]) > 0.4
    assert abs(loadings["perimeter"]) > 0.4
    assert abs(loadings["intensity"]) > 0.4


def test_standardising_makes_the_answer_immune_to_a_change_of_units():
    """px² to um²: the same cells, so the same components."""
    frame = cluster_frame()
    rescaled = frame.assign(area=frame["area"] * 0.1024)
    features = ("area", "perimeter", "intensity")

    a = pca(frame, PCASpec(features=features))
    b = pca(rescaled, PCASpec(features=features))
    assert a.explained_variance_ratio == pytest.approx(
        b.explained_variance_ratio)
    assert a.loadings == pytest.approx(b.loadings)

    # Whereas centring alone moves it, which is the whole argument.
    raw_a = pca(frame, PCASpec(features=features, scaling=SCALE_NONE))
    raw_b = pca(rescaled, PCASpec(features=features, scaling=SCALE_NONE))
    assert raw_a.explained_variance_ratio[0] != pytest.approx(
        raw_b.explained_variance_ratio[0], abs=1e-6)


def test_the_scaling_choice_is_named_in_the_report():
    result = pca(cluster_frame(),
                 PCASpec(features=("area", "intensity"), scaling=SCALE_NONE))
    assert "centred but not scaled" in " ".join(result.caveats())


# ---------------------------------------------------------------------------
# Constant and collinear columns
# ---------------------------------------------------------------------------

def test_a_constant_column_is_dropped_by_name_and_produces_no_nan():
    frame = cluster_frame().assign(always_seven=7.0)
    features = ("area", "perimeter", "intensity", "always_seven")
    result = pca(frame, PCASpec(features=features))

    assert "always_seven" not in result.features
    assert "always_seven" in result.dropped_features
    assert "constant" in result.dropped_features["always_seven"]
    assert np.isfinite(result.loadings).all()
    assert np.isfinite(result.scores).all()
    assert np.isfinite(result.explained_variance_ratio).all()
    assert np.isfinite(result.correlations).all()


def test_a_constant_column_adds_no_component_and_moves_no_number():
    """The answer must be identical to never having offered the column."""
    frame = cluster_frame()
    features = ("area", "perimeter", "intensity")
    without = pca(frame, PCASpec(features=features))
    with_flat = pca(frame.assign(flat=1.0),
                    PCASpec(features=features + ("flat",)))
    assert with_flat.n_components == without.n_components
    assert with_flat.explained_variance_ratio == pytest.approx(
        without.explained_variance_ratio)
    assert with_flat.loadings == pytest.approx(without.loadings)


def test_a_column_that_is_constant_only_after_a_filter_is_still_caught():
    """Constantness is a property of the analysed rows, not of the table."""
    frame = cluster_frame()
    frame = frame.assign(patchy=np.where(frame["gene"] == "control", 1.0, 2.0))
    controls = frame[frame["gene"] == "control"]
    result = pca(controls, PCASpec(features=("area", "intensity", "patchy")))
    assert "patchy" in result.dropped_features


def test_all_features_constant_is_refused_with_a_reason():
    frame = pd.DataFrame({"a": [1.0] * 20, "b": [3.0] * 20})
    with pytest.raises(PCAError, match="no variance"):
        pca(frame, PCASpec(features=("a", "b")))


def test_perfectly_collinear_columns_are_kept_reported_and_capped():
    """Collinearity must not invent a component out of rounding error."""
    frame = cluster_frame()
    frame = frame.assign(area_um2=frame["area"] * 0.1024)
    features = ("area", "perimeter", "intensity", "area_um2")
    result = pca(frame, PCASpec(features=features, n_components=4))

    assert "area_um2" in result.features          # kept, not silently dropped
    assert result.rank == 3                       # four features, three axes
    assert result.n_components == 3               # and never a fourth
    assert ("area", "area_um2") in result.collinear_groups
    assert "Perfectly correlated" in " ".join(result.caveats())


def test_a_near_degenerate_component_says_so_rather_than_posing():
    """Numerically full rank, effectively not — the case the rank cap misses.

    ``area_copy`` is ``area`` plus a jitter far below the measurement, so the
    rank test still counts four directions and the fourth is rounding error.
    A screen that drew it as a component would be drawing noise with a label.
    """
    frame = cluster_frame()
    jitter = np.random.default_rng(0).normal(scale=1e-7, size=len(frame))
    frame = frame.assign(area_copy=frame["area"] * 2.0 + jitter)
    result = pca(frame, PCASpec(
        features=("area", "perimeter", "intensity", "area_copy"),
        n_components=4))
    assert result.rank == 4                    # the rank cap does not save us
    last = result.n_components - 1
    assert result.explained_variance_ratio[last] < DEGENERATE_RATIO
    assert result.is_degenerate(last)
    assert "not identified" in result.headline(last)
    assert "rounding error" in " ".join(result.caveats())


# ---------------------------------------------------------------------------
# NaN
# ---------------------------------------------------------------------------

def structural_frame() -> pd.DataFrame:
    """A pathogen feature measured only on the infected two thirds."""
    frame = cluster_frame(n=240, seed=5)
    infected = np.arange(len(frame)) % 3 != 0
    frame["pathogen_area"] = np.where(infected, 40.0 + frame["area"] / 30.0,
                                      np.nan)
    return frame


def test_auto_drops_a_structurally_missing_feature_and_keeps_every_object():
    """A pathogen_* NaN means 'no pathogen', so the objects are the point."""
    frame = structural_frame()
    features = ("area", "perimeter", "intensity", "pathogen_area")
    result = pca(frame, PCASpec(features=features, nan_policy=NAN_AUTO))

    assert len(result) == len(frame)              # not one object lost
    assert result.dropped_rows == 0
    assert "pathogen_area" not in result.features
    assert "structurally absent" in result.dropped_features["pathogen_area"]


def test_complete_cases_keeps_the_feature_and_says_which_population_is_left():
    frame = structural_frame()
    features = ("area", "perimeter", "intensity", "pathogen_area")
    result = pca(frame, PCASpec(features=features, nan_policy=NAN_COMPLETE))

    assert "pathogen_area" in result.features
    assert len(result) == int(frame["pathogen_area"].notna().sum())
    assert result.dropped_rows == len(frame) - len(result)
    assert "population, not a random sample" in " ".join(result.caveats())


def test_dropping_features_keeps_every_object_and_names_what_went():
    frame = structural_frame()
    result = pca(frame, PCASpec(
        features=("area", "perimeter", "pathogen_area"),
        nan_policy=NAN_DROP_FEATURES))
    assert len(result) == len(frame)
    assert result.dropped_features["pathogen_area"]


def test_mean_imputation_happens_only_when_asked_and_is_recorded():
    frame = structural_frame()
    features = ("area", "perimeter", "pathogen_area")
    imputed = pca(frame, PCASpec(features=features, nan_policy=NAN_MEAN))

    assert "pathogen_area" in imputed.features
    assert len(imputed) == len(frame)
    assert any("mean" in note for note in imputed.notes)
    # And it genuinely changed the answer, which is why it is not the default.
    honest = pca(frame, PCASpec(features=features, nan_policy=NAN_COMPLETE))
    assert imputed.explained_variance_ratio[0] != pytest.approx(
        honest.explained_variance_ratio[0], abs=1e-6)


def test_a_sporadic_nan_costs_rows_rather_than_the_whole_feature():
    """One failed Zernike is not a structural absence."""
    frame = cluster_frame(n=300, seed=8)
    frame.loc[frame.index[:2], "perimeter"] = np.nan   # 0.67% of the rows
    result = pca(frame, PCASpec(
        features=("area", "perimeter", "intensity"), nan_policy=NAN_AUTO))
    assert "perimeter" in result.features
    assert result.dropped_rows == 2
    assert len(result) == len(frame) - 2


def test_an_infinity_is_missing_and_is_counted_separately():
    frame = cluster_frame(n=120, seed=4)
    frame.loc[frame.index[0], "intensity"] = np.inf
    frame.loc[frame.index[1], "intensity"] = -np.inf
    result = pca(frame, PCASpec(
        features=("area", "perimeter", "intensity"), nan_policy=NAN_COMPLETE))
    assert result.n_infinite == 2
    assert result.dropped_rows == 2
    assert np.isfinite(result.scores).all()
    assert "non-finite" in " ".join(result.caveats())


def test_a_nan_policy_that_leaves_nothing_says_how_to_get_out_of_it():
    frame = cluster_frame(n=60, seed=2)
    frame["never_measured"] = np.nan
    with pytest.raises(PCAError) as raised:
        pca(frame, PCASpec(features=("area", "never_measured"),
                           nan_policy=NAN_COMPLETE))
    assert "never_measured" in str(raised.value)
    assert "auto" in str(raised.value)


# ---------------------------------------------------------------------------
# Honesty about how much of the table this is
# ---------------------------------------------------------------------------

def test_the_report_states_the_population_the_picture_is_about():
    frame = structural_frame()
    result = pca(frame, PCASpec(
        features=("area", "perimeter", "intensity", "pathogen_area"),
        nan_policy=NAN_COMPLETE))
    report = result.report()
    assert f"{len(result):,} objects" in report
    assert result.row_share < 0.7
    assert f"{result.dropped_rows:,}" in report


def test_explained_variance_is_over_the_whole_matrix_not_the_kept_components():
    frame = cluster_frame()
    features = ("area", "perimeter", "intensity", "eccentricity")
    two = pca(frame, PCASpec(features=features, n_components=2))
    four = pca(frame, PCASpec(features=features, n_components=4))
    assert two.explained_variance_ratio == pytest.approx(
        four.explained_variance_ratio[:2])
    assert two.retained_ratio < 1.0
    assert four.retained_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_the_spec_round_trips_through_json_exactly():
    spec = PCASpec(features=("area", "intensity"), n_components=5,
                   scaling=SCALE_NONE, nan_policy=NAN_COMPLETE,
                   structural_missing=0.1)
    assert PCASpec.from_json(spec.to_json()) == spec
    assert json.loads(spec.to_json())["features"] == ["area", "intensity"]


def test_a_spec_from_another_build_still_opens():
    payload = {"features": ["area"], "n_components": 3, "future_option": 12}
    spec = PCASpec.from_dict(payload)
    assert spec.features == ("area",)
    assert spec.n_components == 3


@pytest.mark.parametrize("kwargs, message", [
    ({"scaling": "sqrt"}, "unknown scaling"),
    ({"nan_policy": "guess"}, "unknown nan_policy"),
    ({"n_components": 0}, "at least 1"),
    ({"structural_missing": 2.0}, "in \\[0, 1\\]"),
])
def test_a_meaningless_spec_is_refused_where_it_is_built(kwargs, message):
    with pytest.raises(PCAError, match=message):
        PCASpec(**kwargs)


def test_candidate_features_reads_the_one_column_classifier():
    """Keys and small integer codes are not measured quantities."""
    frame = cluster_frame()
    offered = candidate_features(frame)
    assert "area" in offered and "intensity" in offered
    for key in ("plateID", "rowID", "fieldID", "gene", "object_label"):
        assert key not in offered


def test_component_names_round_trip():
    assert component_name(0) == "PC1"
    assert component_index("PC1") == 0
    assert component_index("PC12") == 11
    assert component_index("area") is None
    assert component_index("PC0") is None


def test_the_scores_frame_keeps_every_original_column():
    """Which is what lets the scores plot colour by gene and brush by key."""
    frame = cluster_frame()
    result = pca(frame, PCASpec(features=("area", "perimeter", "intensity"),
                                n_components=2))
    scores = result.scores_frame(frame)
    assert list(frame.columns) == list(scores.columns)[:len(frame.columns)]
    assert "PC1" in scores.columns and "PC2" in scores.columns
    assert len(scores) == len(result)
    assert scores["PC1"].to_numpy() == pytest.approx(result.scores[:, 0])


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------

def test_the_panel_draws_a_scores_plot_with_arrows(qtbot, link):
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())

    result = panel.result
    assert result is not None
    assert panel.canvas.spec.x == "PC1" and panel.canvas.spec.y == "PC2"
    assert panel.canvas.plane() == (0, 1)
    assert panel.canvas.arrow_scale > 0
    # One arrow and one label per drawn feature, on the single panel.
    ax = panel.canvas.axes_at(0, 0)
    assert len(ax.texts) >= 2 * min(len(result.features), 8) - 1
    assert panel.report.text().startswith("PCA of ")


def test_the_arrows_go_when_the_plane_stops_being_a_plane_of_components(
        qtbot, link):
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    frame = cluster_frame()
    panel.set_frame(frame)
    assert panel.canvas.arrow_scale > 0

    # A component against a raw measurement is not a biplot.
    panel.canvas.set_spec(GraphSpec(x="PC1", y="area", kind=SCATTER))
    assert panel.canvas.plane() is None
    assert panel.canvas.arrow_scale == 0.0


def test_turning_the_biplot_off_removes_the_arrows(qtbot, link):
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())
    before = len(panel.canvas.axes_at(0, 0).texts)
    panel.canvas.set_biplot(False)
    assert panel.canvas.arrow_scale == 0.0
    assert len(panel.canvas.axes_at(0, 0).texts) < before


def test_the_arrow_scale_is_one_constant_for_every_panel(qtbot, link):
    """Faceted panels share axes; arrows that differed between them would
    make one PCA look like several."""
    canvas = PCAScoresCanvas(link=link)
    qtbot.addWidget(canvas)
    frame = cluster_frame()
    result = pca(frame, PCASpec(features=("area", "perimeter", "intensity")))
    canvas.set_result(result, result.scores_frame(frame))
    canvas.set_spec(GraphSpec(x="PC1", y="PC2", facet_col="plateID",
                              kind=SCATTER))
    assert canvas.grid.shape[1] == 2
    scale = canvas.arrow_scale
    assert scale > 0
    # Every panel carries the same circle radius, which is the ruler.
    radii = {round(patch.get_radius(), 9)
             for ax in canvas.panel_axes().values()
             for patch in ax.patches if hasattr(patch, "get_radius")}
    assert radii == {round(scale, 9)}


def test_arrow_scale_refuses_a_degenerate_axis():
    frame = cluster_frame()
    result = pca(frame, PCASpec(features=("area", "perimeter", "intensity")))
    assert arrow_scale(result, 0, 1, (0.0, 0.0), (0.0, 0.0)) == 0.0
    assert arrow_scale(result, 0, 1, (-3.0, 3.0), (-3.0, 3.0), count=0) == 0.0


def test_the_scree_plot_chooses_the_plane(qtbot, link):
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())
    panel.scree.component_picked.emit(2)
    assert panel.canvas.spec.x == "PC3"
    assert panel.canvas.spec.y == "PC1"     # the old X slides across


@pytest.mark.parametrize("frame, expected", [
    (pd.DataFrame({"a": [1.0] * 8, "b": [2.0] * 8}), "no continuous columns"),
    (pd.DataFrame({"area": np.linspace(1.0, 100.0, 40),
                   "gene": ["g"] * 40}), "at least two features"),
])
def test_a_refused_pca_becomes_a_message_not_a_traceback(
        qtbot, link, frame, expected):
    """Every refusal is a sentence in the panel, and every one says a way out."""
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    failures = []
    panel.failed.connect(failures.append)
    panel.set_frame(frame)
    assert panel.result is None
    assert failures and expected in failures[0]
    assert panel.report.text() == failures[0]


def test_colouring_the_scores_by_a_label_reaches_the_chart(qtbot, link):
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(cluster_frame())
    index = panel._colour.findData("gene")
    assert index >= 0
    panel._colour.setCurrentIndex(index)
    assert panel.canvas.spec.colour == "gene"
    assert panel.canvas.scales.colour_levels == ("control", "knockdown")


# ---------------------------------------------------------------------------
# The link — the point of the whole screen
# ---------------------------------------------------------------------------

def test_a_brush_in_pc_space_reaches_a_second_linked_view(qtbot, link):
    """Brushing a cluster in PC space is what the screen is for."""
    frame = cluster_frame()
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(frame)

    other = GraphCanvas(link=link, source="somewhere_else")
    qtbot.addWidget(other)
    other.set_spec(GraphSpec(x="area", y="intensity", kind=SCATTER))
    other.set_frame(frame)
    assert other.selected_count() == 0

    scores = panel.scores_frame
    # The two groups separate along PC1; sweep the negative half of it.
    midpoint = float(scores["PC1"].median())
    selection = panel.canvas.brush(
        float(scores["PC1"].min()) - 1.0, float(scores["PC2"].min()) - 1.0,
        midpoint, float(scores["PC2"].max()) + 1.0)

    assert selection is not None
    assert len(selection) == pytest.approx(len(frame) / 2, abs=2)
    assert other.selected_count() == len(selection)
    # And it is one group, not a random half — which is the actual claim.
    picked = scores.loc[
        (scores["PC1"] <= midpoint), "gene"].value_counts()
    assert picked.iloc[0] / picked.sum() > 0.95


def test_a_brush_in_pc_space_can_be_opened_as_objects(qtbot, link):
    frame = cluster_frame()
    panel = PCAPanel(link=link)
    qtbot.addWidget(panel)
    panel.set_frame(frame)

    opened = []
    link.register_object_opener("annotate", opened.append)
    scores = panel.scores_frame
    panel.canvas.brush(float(scores["PC1"].min()) - 1.0,
                       float(scores["PC2"].min()) - 1.0,
                       float(scores["PC1"].median()),
                       float(scores["PC2"].max()) + 1.0)
    panel.canvas.open_objects(link.selection.keys, reason="a test")
    assert len(opened) == 1
    assert len(opened[0].keys) == len(link.selection)


def _settled(qtbot, screen, timeout: int = 20000):
    """Wait for a PCAScreen's read and decomposition to deliver."""
    qtbot.waitUntil(
        lambda: not screen.is_busy() and screen.active_jobs() == 0,
        timeout=timeout)
    return screen


def test_the_shared_filter_narrows_the_population_and_the_pca_with_it(
        qtbot, link):
    """A filter is a new PCA, not the old one with points removed: the centre,
    the scale and the directions all belong to the population."""
    from spacr.qt.screens.pca import PCAScreen
    from spacr.selection import CategoryFilter, DataFilter

    screen = PCAScreen(link=link)
    qtbot.addWidget(screen)
    frame = cluster_frame()
    screen.set_frame(frame)
    # The screen threads its panel (see PCAPanel.recompute): the sklearn fit
    # is 1.63 s on a real table, so it runs on a worker and the result lands
    # on a later turn of the event loop rather than on the call's return.
    _settled(qtbot, screen)
    everything = screen.pca.result
    assert len(everything) == len(frame)

    link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))
    screen._recompute_filtered()
    _settled(qtbot, screen)
    controls = screen.pca.result
    assert len(controls) == len(frame[frame["gene"] == "control"])
    # The separation was the biggest axis; without it the components move.
    assert controls.explained_variance_ratio[0] != pytest.approx(
        everything.explained_variance_ratio[0], abs=1e-3)


def test_the_canvas_lets_go_of_the_link_on_close(qtbot, link):
    canvas = PCAScoresCanvas(link=link)
    qtbot.addWidget(canvas)
    assert canvas.is_linked
    canvas.close()
    assert not canvas.is_linked



@pytest.fixture
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    keyboard binding for every test that runs afterwards, so the list object is
    restored in place rather than trusting an unregister call.
    """
    from spacr.qt import app as app_mod
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    app_mod._refresh_sections()


def test_the_screen_is_registered(qtbot):
    """The one row in app.py's `_SELF_REGISTERING_APPS` is present.

    This test used to assert the opposite. The screen was finished and
    tested but deliberately switched off, because a new APPS row reddened
    the per-app inventory tests and Explore stood at
    MAX_APPS_PER_SECTION. Both reasons expired -- the ledgers were filled
    in and Explore came back down to eight -- and the row landed in
    `baa704fc`, at which point this file was the only thing still
    claiming the screen was unreachable.

    Inverted rather than deleted: "is it on the sidebar" is worth
    asserting in whichever direction is currently true, and a test that
    pins the old state is how a finished feature stays invisible.
    """
    from spacr.qt.app import APPS
    from spacr.qt.screens import pca as screen

    assert any(row[0] == screen.APP_KEY for row in APPS), (
        "pca is missing from APPS; its row in _SELF_REGISTERING_APPS "
        "(spacr/qt/app.py) is what puts it there")
    qtbot.addWidget(screen.make_pca_screen())


def test_registering_the_screen_reaches_every_reader_of_the_registry(
        registry_sandbox):
    """Driving `register()` is the same thing the one line will do."""
    from spacr.qt.screens import pca as screen
    app_mod = registry_sandbox

    # The sandbox SNAPSHOTS the registry and restores it afterwards; it does
    # not empty it. So the screen is already registered here, from app.py's
    # own call at import, and register() would correctly answer False. Take
    # it back out first, then assert the round trip -- which is also the
    # stronger test, because it exercises unregister as well.
    assert app_mod.unregister_app(screen.APP_KEY) is True
    assert not any(r[0] == screen.APP_KEY for r in app_mod.APPS)

    assert screen.register() is True
    assert screen.register() is False           # idempotent, not a raise

    row = next(r for r in app_mod.APPS if r[0] == screen.APP_KEY)
    assert row[1] == screen.APP_NAME
    assert row[3] == app_mod.SECTION_EXPLORE
    assert app_mod.APP_FACTORIES[screen.APP_KEY] is screen.make_pca_screen
    assert app_mod.APP_STAGE[screen.APP_KEY] == app_mod.STAGE_ALPHA

    # The strings fan out into the tables that used to need a hand-edit each.
    from spacr import cli
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES[screen.APP_KEY] == screen.APP_NAME
    assert APP_INTROS[screen.APP_KEY] == screen.APP_INTRO
    assert cli.INTERACTIVE_ONLY[screen.APP_KEY] == screen.APP_CLI_NOTE
