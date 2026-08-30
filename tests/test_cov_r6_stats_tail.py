"""The last branch in nine statistics / sweep modules.

Every module here is above 99%. What is left in each is a single guard,
and more than half of them turn out to be guards that cannot fail --
each one a re-check of something a line just above has already
established. Those are written up as proofs with the invariant pinned by
a test, never silenced.

Module by module:

``spacr.parameter_sweep``
    ``_named_control_rows``: a control found in the frame is always found
    in the ranked view of the same frame, so its rank is always
    reported. ``run_sweep_parallel``: the completion loop always breaks,
    which is what makes the pool top up one job at a time.
``spacr.regression_diagnostics``
    A suite whose sheets carry no verdict still writes its summary.
``spacr.batch_correction``
    The empirical-Bayes fixed point is bounded: a feature that cannot
    converge costs the iteration cap, not the run.
``spacr.response_distribution``
    A rescaled panel always draws its two-axis legend.
``spacr.gene_measurement_sweep``
    Every gene on the concordance axis has guides behind it.
``spacr.gene_measurement_compare``
    A violin comparison in which one group is entirely non-finite still
    draws the group that is not.
``spacr.surrogate``
    Three-dimensional SHAP output is reduced along the class axis, and
    the signed value kept is the predicted class's.
``spacr.sudoku``
    ``sudoku_all`` hands every ranked guide to each round, so the guide
    it is committing is always one of the columns that came back.
``spacr.model_zoo``
    ``_human_bytes`` always names a unit.

Not repeated here, because the proof already exists in this tree:

* ``spacr.power_model`` lines 2011/2013 (the two per-column guards in
  ``scan_parameters``' resume block) are proved unreachable by
  ``tests/test_cov_r5_power_model.py::
  test_a_progress_file_missing_a_result_column_is_rejected_by_its_header``.
* ``spacr.regression_qc``'s ``if edge.size and interior.size`` and
  ``if v.size`` are proved unreachable by
  ``tests/test_cov_r4_regression_qc.py``, which pins the partition that
  makes them so.
"""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# spacr.parameter_sweep -- a found control always has a rank
# ---------------------------------------------------------------------------

class TestANamedControlAlwaysGetsItsRank:
    """``if position.any():`` inside ``_named_control_rows``.

    The false side cannot happen. Ten lines above,
    ``ranked = frame.assign(_abs=...).sort_values("_abs", ...)``
    reorders ``frame`` -- ``sort_values`` is a permutation, it drops no
    rows -- and ``ranked_labels`` is that same label column read off the
    reordered frame. ``hit`` and ``position`` therefore run the identical
    ``str.contains`` over the identical multiset of labels, so
    ``hit.any() == position.any()``; and the loop has already done
    ``if not hit.any(): continue`` before reaching this line.

    That is the "defensive re-check after a call that already guarantees
    the condition" case. What is pinned instead is the property the
    re-check was defending: when a control is present, both its effect
    and its rank come back, and the rank is measured on the |effect|
    ordering rather than on the row order.
    """

    FRAME = pd.DataFrame({
        "grna": ["ctrl_weak", "other_a", "ctrl_strong", "other_b"],
        "coefficient": [0.10, -0.40, 0.90, 0.05],
        "q_value": [0.9, 0.01, 0.001, 0.8],
        "p_value": [0.8, 0.005, 0.0001, 0.7],
    })

    def test_a_present_control_reports_both_effect_and_rank(self):
        from spacr.parameter_sweep import _named_control_rows

        out = _named_control_rows(
            self.FRAME, {"pc": "ctrl_strong", "nc": "ctrl_weak"})

        assert out["pc_present"] is True and out["nc_present"] is True
        assert out["pc_effect"] == pytest.approx(0.90)
        assert out["nc_effect"] == pytest.approx(0.10)
        # 0.90 is the largest |coefficient| in the frame, so the positive
        # control ranks first even though it is the THIRD row.
        assert out["pc_rank"] == 1, (
            "the rank must be read off the |effect| ordering, not the row "
            f"order; got {out.get('pc_rank')}")
        assert out["nc_rank"] == 3
        assert out["pc_q"] == pytest.approx(0.001)

    def test_an_absent_control_reports_neither(self):
        """The contrast: no hit, so no effect and no rank at all."""
        from spacr.parameter_sweep import _named_control_rows

        out = _named_control_rows(self.FRAME, {"pc": "not_in_this_screen"})

        assert out == {"pc_present": False}


class TestTheSweepPoolTopsUpOneJobAtATime:
    """``for future in as_completed(list(futures)):`` never runs out.

    The last statement in that loop's body is an unconditional ``break``,
    so the ``for`` can only be exhausted when ``as_completed`` yields
    nothing -- and it is called with ``list(futures)`` under
    ``while futures:``, which guarantees at least one. The arc "the for
    loop ended normally" is therefore unreachable: this is the
    "loop-else where the loop always breaks" case.

    That break is not decoration. It is what makes ``_fill()`` run after
    EVERY single completion, which is what lets the memory floor hold a
    sweep back one job at a time instead of resubmitting a whole batch.
    Draining the snapshot before topping up would resubmit ``n_jobs``
    futures against a memory reading taken before any of them started.

    So the invariant is pinned on the source: the completion loop's body
    ends with a bare ``break``, and ``_fill()`` is the statement before
    it. If someone drops the ``break``, this fails -- which is the point,
    because coverage never would.
    """

    def test_the_completion_loop_refills_then_breaks(self):
        from spacr import parameter_sweep

        source = textwrap.dedent(
            inspect.getsource(parameter_sweep.run_sweep_parallel))
        tree = ast.parse(source)

        loops = [node for node in ast.walk(tree)
                 if isinstance(node, ast.For)
                 and isinstance(node.iter, ast.Call)
                 and getattr(node.iter.func, "id", "") == "as_completed"]
        assert len(loops) == 1, (
            "run_sweep_parallel should drain futures in exactly one "
            f"as_completed loop; found {len(loops)}")

        body = loops[0].body
        assert isinstance(body[-1], ast.Break), (
            "the completion loop must break after one future, so the pool is "
            "topped up against a fresh memory reading")
        refill = body[-2]
        assert (isinstance(refill, ast.Expr)
                and isinstance(refill.value, ast.Call)
                and getattr(refill.value.func, "id", "") == "_fill"), (
            "the statement before the break must be the _fill() that "
            f"resubmits, not {ast.dump(refill)[:60]}")
        assert not any(isinstance(node, ast.Break)
                       for node in ast.walk(loops[0])
                       if node is not body[-1]), \
            "a second break would make the refill conditional"


# ---------------------------------------------------------------------------
# spacr.regression_diagnostics -- a suite with no verdicts
# ---------------------------------------------------------------------------

class TestTheDiagnosticSummaryWithoutAVerdict:
    """``if levels:`` before the suite verdict rows are appended."""

    @staticmethod
    def _p_values():
        rng = np.random.default_rng(0)
        return pd.Series(rng.uniform(0.0, 1.0, 60))

    def test_a_sheet_that_scores_nothing_still_gets_a_summary(
            self, tmp_path, monkeypatch):
        """A panel returning a report with no ``verdict_level``.

        The summary is written either way -- the CSV is the manifest of
        what the suite measured -- but with nothing to rank there is no
        ``suite`` verdict row to write.
        """
        from spacr import regression_diagnostics as rd

        def _unscored(save_path, save_format=None, **kwargs):
            written = f"{save_path}.png"
            open(written, "wb").close()
            return written, {"n_tests": 60}

        monkeypatch.setattr(rd, "plot_inference_diagnostics", _unscored)

        out = rd.write_diagnostic_suite(tmp_path / "quiet",
                                        p_values=self._p_values())

        summary = pd.read_csv(out["diagnostic_summary"])
        assert "n_tests" in set(summary["metric"]), \
            "the sheet's own metrics must still reach the summary"
        assert "suite" not in set(summary["section"]), (
            "with no sheet carrying a verdict_level there is no worst "
            f"verdict to name; got {summary['section'].tolist()}")

    def test_a_scored_sheet_names_the_suite_verdict(self, tmp_path):
        """The contrast: the real panel scores, so the suite row appears."""
        from spacr import regression_diagnostics as rd

        out = rd.write_diagnostic_suite(tmp_path / "scored",
                                        p_values=self._p_values())

        summary = pd.read_csv(out["diagnostic_summary"])
        sections = set(summary["section"])
        assert "suite" in sections, (
            "a scored sheet must produce the suite's worst verdict; got "
            f"{sorted(sections)}")
        metrics = set(summary.loc[summary["section"] == "suite", "metric"])
        assert metrics == {"verdict_level", "verdict"}


# ---------------------------------------------------------------------------
# spacr.batch_correction -- the fixed point is bounded
# ---------------------------------------------------------------------------

class TestTheEmpiricalBayesFixedPointIsBounded:
    """``for _ in range(_COMBAT_MAX_ITER):`` running out without a break."""

    @staticmethod
    def _inputs(n_features=4, n_samples=6):
        rng = np.random.default_rng(3)
        standardized = rng.normal(0.0, 1.0, (n_features, n_samples))
        gamma_hat = standardized.mean(axis=1)
        delta_hat = standardized.var(axis=1) + 0.5
        return standardized, gamma_hat, delta_hat

    def test_a_feature_that_cannot_converge_costs_the_cap_not_the_run(self):
        """One non-finite feature must not hang the shrinkage.

        ``change`` is a max over relative movements; with a NaN in it the
        comparison ``change < _COMBAT_CONV`` is False forever, so the
        loop can only end by exhausting its cap. What must survive that
        is the OTHER features: a single unusable feature is not a reason
        to lose the batch.
        """
        from spacr.batch_correction import _eb_fixed_point

        standardized, gamma_hat, delta_hat = self._inputs()
        gamma_bar = float(np.mean(gamma_hat))
        tau2 = float(np.var(gamma_hat))
        broken = gamma_hat.copy()
        broken[2] = np.nan

        gamma_star, delta_star = _eb_fixed_point(
            standardized, broken, delta_hat, gamma_bar, tau2,
            a_prior=3.0, b_prior=1.0)

        assert gamma_star.shape == gamma_hat.shape
        assert np.isnan(gamma_star[2]), \
            "the unusable feature stays unusable rather than being invented"
        assert np.isfinite(gamma_star[[0, 1, 3]]).all(), (
            "the features that could converge must still carry numbers: "
            f"{gamma_star}")
        assert np.isfinite(delta_star[[0, 1, 3]]).all()

    def test_clean_inputs_reach_the_fixed_point_and_stop(self):
        """The contrast: convergence, and the answer really is the fixed point."""
        from spacr.batch_correction import _eb_fixed_point

        standardized, gamma_hat, delta_hat = self._inputs()
        gamma_bar = float(np.mean(gamma_hat))
        tau2 = float(np.var(gamma_hat))
        n = standardized.shape[1]

        gamma_star, delta_star = _eb_fixed_point(
            standardized, gamma_hat, delta_hat, gamma_bar, tau2,
            a_prior=3.0, b_prior=1.0)

        # One more turn of the crank must not move it.
        again = ((tau2 * n * gamma_hat + delta_star * gamma_bar)
                 / (tau2 * n + delta_star))
        assert np.allclose(again, gamma_star, rtol=1e-3, atol=1e-6), (
            "the loop must have stopped AT the fixed point, not at the cap: "
            f"{gamma_star} vs {again}")


# ---------------------------------------------------------------------------
# spacr.response_distribution -- the two-axis legend is always drawn
# ---------------------------------------------------------------------------

class TestTheRescaledPanelAlwaysLabelsItsTwoAxes:
    """``if handles:`` after both histograms have been drawn.

    ``handles`` is built from ``ax.patches`` and ``twin.patches``, and
    both lists were filled two lines earlier by ``ax.hist(..., bins=40)``
    and ``twin.hist(..., bins=40)``. ``Axes.hist`` creates one patch per
    bin whatever the data -- forty patches even for an empty array -- so
    ``ax.patches`` is never empty here and ``handles`` never is either.
    Another re-check of something the call above guarantees.

    The invariant pinned instead is the one the legend exists for: when
    the transform rescales the response, the panel puts the two
    incomparable scales on two axes and says in the legend which is
    which.
    """

    def test_a_rescaling_transform_labels_before_and_after(self):
        from matplotlib.figure import Figure

        from spacr.response_distribution import panel

        rng = np.random.default_rng(0)
        values = rng.uniform(0.01, 0.99, 400)
        ax = Figure(figsize=(6.0, 3.5)).add_subplot(111)

        result = panel(values, "log", ax=ax, dependent_variable="score")

        assert result["rescaled"] and result["changed"], (
            "this test only says anything on the two-axis branch; "
            f"got rescaled={result['rescaled']} changed={result['changed']}")
        legend = ax.get_legend()
        assert legend is not None, "the two-axis panel must carry a legend"
        assert [t.get_text() for t in legend.get_texts()] == [
            "before", f"after {result['transform']}"]
        assert len(ax.patches) == 40, \
            "hist fills the patch list the legend handles are taken from"

    def test_a_transform_that_changes_nothing_uses_one_axis(self):
        """The contrast: shared bins, one axis, and no twin at all."""
        from matplotlib.figure import Figure

        from spacr.response_distribution import panel

        figure = Figure(figsize=(6.0, 3.5))
        ax = figure.add_subplot(111)

        panel(np.linspace(1.0, 9.0, 200), "none", ax=ax)

        assert len(figure.axes) == 1, \
            "an unchanged response must not be split across two scales"


# ---------------------------------------------------------------------------
# spacr.gene_measurement_sweep -- every gene on the axis has guides
# ---------------------------------------------------------------------------

def _sweep_table(**columns):
    n = len(next(iter(columns.values())))
    base = {"level": ["guide"] * n, "guide": [f"g{i}" for i in range(n)],
            "measurement": ["cell_area"] * n, "effect": [0.5] * n,
            "p": [0.001] * n, "q": [0.001] * n, "circularity": [0.05] * n,
            "n_wells": [40] * n, "effective_wells": [35.0] * n,
            "share": [0.2] * n, "ubiquitous": [False] * n,
            "control": [False] * n}
    base.update(columns)
    return pd.DataFrame(base)


class TestEveryConcordanceRowHasItsGuides:
    """``if not len(values): continue`` in ``plot_guide_concordance``.

    ``summary`` is ``frame.groupby("gene").agg(...)``, so its index is
    exactly the set of genes that appear in ``frame``; the loop then asks
    ``frame.loc[frame["gene"] == gene, "agree"]`` for each of them. A
    groupby key with no rows behind it does not exist, so ``values`` is
    never empty and the ``continue`` is unreachable -- a re-check of what
    the groupby has already established.

    Pinned instead: every gene the axis lists gets its own points and its
    own mean line, so a label on that axis is never a row drawn from
    nothing.
    """

    def test_each_listed_gene_draws_its_own_points_and_mean(self):
        from spacr.gene_measurement_sweep import (SweepResult,
                                                  plot_guide_concordance)

        # Two genes, two guides each. AAA's guides agree, BBB's disagree,
        # so the two rows are drawn with different verdicts.
        table = _sweep_table(
            guide=["AAA_1", "AAA_2", "BBB_1", "BBB_2"],
            effect=[0.6, 0.8, 0.7, -0.7],
            q=[0.001, 0.001, 0.001, 0.001])
        result = SweepResult(
            table=table,
            effects=pd.DataFrame(np.full((4, 1), 0.5),
                                 index=table["guide"], columns=["cell_area"]),
            n_wells=40, n_blocks=2)

        figure = plot_guide_concordance(result)

        assert figure is not None, "two genes with two guides each is a picture"
        axes = figure.axes[0]
        labels = [t.get_text() for t in axes.get_yticklabels()]
        assert len(labels) == 2, f"expected one row per gene, got {labels}"
        assert all("(" in label for label in labels), \
            f"each row must state how many pairs it summarises: {labels}"
        assert len(axes.collections) == len(labels), (
            "every listed gene must carry its own scatter of guide "
            f"agreements; {len(axes.collections)} for {len(labels)} rows")
        # ...and its own mean bar, coloured by the verdict.
        # The dotted reference line at agreement = 1.0 is also a two-point
        # line, so the mean bars are the ones drawn solid.
        means = [line for line in axes.lines
                 if len(line.get_xdata()) == 2 and line.get_linestyle() != ":"]
        assert len(means) == len(labels), (
            "one mean bar per listed gene; got "
            f"{[l.get_linestyle() for l in axes.lines]}")
        assert len({tuple(line.get_color()) if not isinstance(
            line.get_color(), str) else line.get_color()
            for line in means}) == 2, (
            "agreeing and disagreeing genes must not be inked the same")


# ---------------------------------------------------------------------------
# spacr.gene_measurement_compare -- a violin over a partly empty comparison
# ---------------------------------------------------------------------------

class TestAViolinKeepsTheGroupThatHasValues:
    """``if alive:`` in ``render_comparison``'s violin branch.

    Twelve lines above, ``if not any(len(s) for s in series): return
    None, None`` has already refused a comparison in which every group
    is empty after the finite filter. ``alive`` is
    ``[(i, s) for i, s in enumerate(series) if len(s)]`` over that same
    ``series``, so it holds at least one entry whenever execution
    reaches it. The false side is unreachable -- the third re-check in
    this file of a condition established by the line above.

    What the guard protects is real, though, and is pinned here: a group
    that is entirely non-finite drops out of the violin while the group
    beside it is still drawn, rather than the panel refusing outright.
    """

    @staticmethod
    def _comparison(groups):
        from spacr.gene_measurement_compare import Comparison

        rows = [{"group": name, "value": value}
                for name, values in groups.items() for value in values]
        return Comparison(measurement="cell_area", level="object",
                          frame=pd.DataFrame(rows))

    def test_an_all_nan_group_drops_out_and_the_other_stays(self):
        from spacr import gene_measurement_compare as gmc

        comparison = self._comparison({"treated": [np.nan, np.inf],
                                       "control": [1.0, 2.0, 3.0]})

        figure, axes = gmc.render_comparison(
            comparison, gmc.ComparisonStyle(kind="violin"))

        assert figure is not None and axes is not None, (
            "one group with finite values is enough to draw a panel")
        assert len(axes.collections) == 1, (
            "exactly the group that had values gets a violin; got "
            f"{len(axes.collections)}")
        # Both groups keep their tick, so the empty one is visibly empty
        # rather than silently missing.
        labels = [t.get_text() for t in axes.get_xticklabels()]
        assert len(labels) == 2, f"both groups keep a slot: {labels}"

    def test_both_groups_finite_draws_both_violins(self):
        """The contrast that says the count above is a real count."""
        from spacr import gene_measurement_compare as gmc

        comparison = self._comparison({"treated": [4.0, 5.0, 6.0],
                                       "control": [1.0, 2.0, 3.0]})

        _figure, axes = gmc.render_comparison(
            comparison, gmc.ComparisonStyle(kind="violin"))

        assert len(axes.collections) == 2


# ---------------------------------------------------------------------------
# spacr.surrogate -- three-dimensional SHAP output
# ---------------------------------------------------------------------------

class _FakeExplainer:
    def __init__(self, values):
        self._values = values

    def shap_values(self, sample, check_additivity=None):
        return self._values


@pytest.fixture
def fake_shap(monkeypatch):
    """Install a ``shap`` module whose TreeExplainer returns what we choose.

    ``_shap_importance`` does ``import shap`` inside the function body and
    has no injection seam of its own, so ``sys.modules`` is the only way
    to drive the shapes it normalises. Reached through the private
    function for the same reason: ``fit_surrogate`` would need a real
    tree model AND a real shap install to get here.
    """
    holder = {"values": None}
    module = types.ModuleType("shap")
    module.TreeExplainer = lambda model: _FakeExplainer(holder["values"])
    monkeypatch.setitem(sys.modules, "shap", module)
    return holder


class TestThreeDimensionalShapIsReducedOverClasses:
    """``importance = np.abs(normalised).mean(axis=(0, 2))`` and the signed pick."""

    SAMPLE = pd.DataFrame({"area": [1.0, 2.0, 3.0],
                           "perimeter": [4.0, 5.0, 6.0]})

    @staticmethod
    def _values():
        """(rows, features, classes) with a different sign per class."""
        array = np.zeros((3, 2, 2), dtype=float)
        array[:, 0, 0] = [1.0, 2.0, 3.0]      # area, class 0
        array[:, 0, 1] = [-1.0, -2.0, -3.0]   # area, class 1
        array[:, 1, 0] = [0.1, 0.1, 0.1]      # perimeter, class 0
        array[:, 1, 1] = [-0.1, -0.1, -0.1]
        return array

    class _WithProba:
        def predict_proba(self, sample):
            # rows 0 and 2 predict class 1; row 1 predicts class 0.
            return np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]])

    class _WithoutProba:
        pass

    def test_the_signed_column_follows_the_predicted_class(self, fake_shap):
        from spacr.surrogate import _shap_importance

        fake_shap["values"] = self._values()
        warnings: list = []

        importance, details, sample = _shap_importance(
            self._WithProba(), self.SAMPLE, 10, warnings, return_details=True)

        assert warnings == [], f"nothing was wrong here: {warnings}"
        # |mean| over rows AND classes: area averages 2.0, perimeter 0.1.
        assert importance == pytest.approx([2.0, 0.1])
        assert list(details.columns) == ["area", "perimeter"]
        assert details["area"].tolist() == pytest.approx([-1.0, 2.0, -3.0]), (
            "each row must keep the SHAP value of the class the model "
            f"actually predicts for it; got {details['area'].tolist()}")
        assert len(sample) == 3

    def test_a_model_without_probabilities_keeps_class_zero(self, fake_shap):
        """The contrast: no predict_proba, so column 0 is what is signed."""
        from spacr.surrogate import _shap_importance

        fake_shap["values"] = self._values()
        warnings: list = []

        _importance, details, _sample = _shap_importance(
            self._WithoutProba(), self.SAMPLE, 10, warnings,
            return_details=True)

        assert details["area"].tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_a_shape_that_matches_neither_axis_order_is_refused(
            self, fake_shap):
        """And the shape guard above it still says so rather than guessing."""
        from spacr.surrogate import _shap_importance

        fake_shap["values"] = np.zeros((4, 7, 2))
        warnings: list = []

        assert _shap_importance(self._WithProba(), self.SAMPLE, 10,
                                warnings) is None
        assert any("unexpected SHAP output shape" in w for w in warnings)


# ---------------------------------------------------------------------------
# spacr.sudoku -- the committed guide is always one of the columns
# ---------------------------------------------------------------------------

class TestTheCommittedGuideIsAlwaysAColumn:
    """``mine = here.names.index(guide) if guide in here.names else None``

    ``sudoku_all`` builds ``names = tuple(g for g, _ in order)`` where
    ``order`` holds ``(str(g), float(c))`` pairs, iterates ``order``, and
    passes that same ``names`` tuple into every per-round ``sudoku``
    call. ``sudoku`` sets ``names = tuple(str(g) for g in guides)`` and
    returns it verbatim on both of its exits. ``guide`` is already a
    ``str`` and is an element of ``names``, so ``guide in here.names`` is
    always true and ``mine`` is never ``None``: the ``if mine is not
    None`` guard cannot be false.

    That is not an accident of the data -- it is the design decision the
    comment above the call records ("EVERY GUIDE IN THE RUN, ONE GUIDE
    COMMITTED"): each round scores every guide so the posterior is a
    comparison, and commits one. The test pins exactly that: the columns
    of the result are the ranked guides, in ranked order.
    """

    @staticmethod
    def _screen():
        rng = np.random.default_rng(0)
        # Two well-separated clusters of cells; each well is dominated by
        # one guide, so the well constraint has something to say.
        a = rng.normal(0.0, 0.25, (24, 2))
        b = rng.normal(4.0, 0.25, (24, 2))
        features = np.vstack([a, b])
        scores = np.concatenate([np.zeros(24), np.ones(24)])
        wells = ["w1"] * 24 + ["w2"] * 24
        fractions = {"w1": {"gA": 0.9, "gB": 0.1},
                     "w2": {"gA": 0.1, "gB": 0.9}}
        return features, scores, wells, fractions

    def test_the_result_columns_are_the_ranked_guides(self):
        from spacr.sudoku import sudoku_all

        features, scores, wells, fractions = self._screen()

        result = sudoku_all(features, scores, wells, fractions,
                            ranking=[("gB", 0.9), ("gA", 0.4)])

        assert result.names == ("gB", "gA"), (
            "the columns are the ranking in descending confidence, which is "
            f"what makes `mine` findable every round; got {result.names}")
        assert result.affirm.shape == (len(wells), 2)
        assert result.posterior.shape == (len(wells), 2)
        # Every ranked guide is a column, so every round can read its own.
        assert set(result.names) == {"gA", "gB"}

    def test_a_lower_ranked_guide_is_still_scored_before_it_is_committed(self):
        """Both columns carry evidence after the first round, not just one.

        This is the property the ``mine`` lookup depends on: the round
        that commits ``gB`` still computes ``gA``'s posterior, because a
        posterior over one guide is 1.0 by construction and says nothing.
        """
        from spacr.sudoku import sudoku_all

        features, scores, wells, fractions = self._screen()

        result = sudoku_all(features, scores, wells, fractions,
                            ranking=[("gB", 0.9), ("gA", 0.4)])

        assert float(np.nanmax(result.posterior[:, 0])) > 0.0
        assert float(np.nanmax(result.posterior[:, 1])) > 0.0, (
            "the guide that was not being committed must still have been "
            "scored; a one-column posterior is 1.0 everywhere")


# ---------------------------------------------------------------------------
# spacr.model_zoo -- every size gets a unit
# ---------------------------------------------------------------------------

class TestHumanBytesAlwaysNamesAUnit:
    """``for unit in ("B", "KB", "MB", "GB"):`` never runs out.

    The condition inside is ``if n < 1024 or unit == "GB"``, so the last
    pass returns unconditionally; the loop cannot be exhausted, and the
    arc "the for ended and the function fell off the bottom" is
    unreachable. The module's own comment says so, and this is the test
    that keeps it true: a size larger than any unit still comes back
    labelled rather than as ``None``.
    """

    @pytest.mark.parametrize("size,expected", [
        (0, "unknown"),
        (None, "unknown"),
        ("not a number", "unknown"),
        (512, "512 B"),
        (2048, "2.0 KB"),
        (5 * 1024 ** 2, "5.0 MB"),
        (3 * 1024 ** 3, "3.0 GB"),
        (4096 * 1024 ** 3, "4096.0 GB"),
    ])
    def test_every_magnitude_is_named(self, size, expected):
        from spacr.model_zoo import _human_bytes

        assert _human_bytes(size) == expected

    def test_a_petabyte_still_comes_back_as_a_string(self):
        """The exhaustion case, if it existed, would land here."""
        from spacr.model_zoo import _human_bytes

        answer = _human_bytes(1024 ** 6)

        assert isinstance(answer, str) and answer.endswith(" GB"), (
            "the GB pass returns whatever is left, so nothing falls through "
            f"the loop; got {answer!r}")
