"""The last turns in the statistics modules, and three guards that cannot fire.

Closed here:

* :mod:`spacr.regression_summary` -- the unidentifiability warning without a
  rank deficiency, a cell count that is not finite, and a LASSO selection
  table with no coefficient column to intersect with.

Proven unreachable, with the invariant asserted instead:

* ``surrogate`` line 785 -- every accepted SHAP shape is normalised onto the
  feature axis first, so the count can never disagree.
* ``regression_qc`` lines 2958/2988 -- the groups are the distinct values of
  the very array being indexed, so no group is empty.
* ``parameter_sweep`` line 374 -- the ranked labels are a permutation of the
  labels already known to contain the needle.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import (parameter_sweep, regression_qc, regression_summary,
                   surrogate)


# ---------------------------------------------------------------------------
# surrogate: the SHAP feature axis
# ---------------------------------------------------------------------------

class _FakeShap:
    """Stand-in for the optional ``shap`` dependency.

    ``_shap_importance`` imports ``shap`` inside the call, which is the seam
    the module documents for exactly this: the real package is optional and
    its output shape is the thing under test, not its arithmetic.
    """

    def __init__(self, values):
        self._values = values

        outer = self

        class TreeExplainer:
            def __init__(self, model):
                self.model = model

            def shap_values(self, sample, check_additivity=True):
                return outer._values(sample)

        self.TreeExplainer = TreeExplainer


def test_every_shap_shape_is_normalised_onto_the_feature_axis(monkeypatch):
    """UNREACHABLE LINE, PROVEN: ``surrogate`` line 785.

    ``importance`` is built on one of three paths and every one of them ends
    on the feature axis of ``sample``:

    * 2-D and equal to ``sample.shape`` -> ``mean(axis=0)``;
    * 3-D with ``shape[:2] == sample.shape`` -> ``mean(axis=(0, 2))``;
    * 3-D with ``shape[1:] == sample.shape`` -> moved to the same layout.

    Anything else has already returned None. ``sample`` is ``x_test`` or
    ``x_test.sample(rows)``, which has the same columns, so
    ``importance.shape[0] == x_test.shape[1]`` always and the mismatch
    warning is dead.

    Asserted over all three accepted layouts, plus the two refusals that make
    them the only ones that get here.
    """
    x_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                           "b": [4.0, 3.0, 2.0, 1.0],
                           "c": [0.5, 0.5, 1.5, 2.5]})
    model = object()

    layouts = {
        "2d": lambda s: np.arange(s.size, dtype=float).reshape(s.shape),
        "3d_classes_last": lambda s: np.arange(
            s.size * 2, dtype=float).reshape(s.shape[0], s.shape[1], 2),
        "3d_classes_first": lambda s: np.arange(
            s.size * 2, dtype=float).reshape(2, s.shape[0], s.shape[1]),
        "list_per_class": lambda s: [
            np.arange(s.size, dtype=float).reshape(s.shape),
            np.ones(s.shape)],
    }
    for name, maker in layouts.items():
        monkeypatch.setitem(sys.modules, "shap", _FakeShap(maker))
        warnings: list = []
        importance = surrogate._shap_importance(model, x_test, 100, warnings)
        assert importance is not None, name
        assert importance.shape == (x_test.shape[1],), name
        assert warnings == [], name

    # The two refusals that keep the accepted set to those layouts. Both
    # return None with a stated reason, so nothing with a mismatched feature
    # axis ever reaches the count check.
    for maker, reason in (
            (lambda s: np.zeros((s.shape[0], s.shape[1] + 1)), "shape"),
            (lambda s: [np.zeros((2, 2, 2))], "list-shaped")):
        monkeypatch.setitem(sys.modules, "shap", _FakeShap(maker))
        warnings = []
        assert surrogate._shap_importance(model, x_test, 100, warnings) is None
        assert any(reason in message for message in warnings)


# ---------------------------------------------------------------------------
# regression_summary
# ---------------------------------------------------------------------------

def _run(**over):
    base = dict(
        res_folder=None, model=None, settings={}, coef_df=None,
        regression_type="ols", nonparametric=False, penalised=False,
        data=None, data_note="no run folder was given", metrics={},
    )
    base.update(over)
    return regression_summary._Run(**base)


def test_a_wide_fit_warns_once_and_a_rank_deficient_one_warns_twice():
    """More parameters than rows is one sentence; a short rank adds another.

    The second sentence quotes a rank, so it must not be printed when the run
    recorded no rank at all -- a blank there reads as rank zero.
    """
    wide = regression_summary._warnings(
        _run(metrics={"n_parameters": 12, "n_observations": 8}))
    assert len(wide) == 1
    assert "8 analysed observations" in wide[0]

    deficient = regression_summary._warnings(
        _run(metrics={"n_parameters": 12, "n_observations": 8,
                      "design_rank": 5}))
    assert len(deficient) == 2
    assert "rank 5 against 12 columns" in deficient[1]
    assert "7 direction(s)" in deficient[1]


def test_a_cell_count_that_is_not_finite_is_reported_as_unknown():
    """``n_cells`` stays None rather than becoming a nonsense integer.

    ``int(inf)`` raises and ``int(nan)`` is undefined, so a column carrying
    either has to leave the field unset -- the summary's own rule is that a
    number it cannot compute is named, never guessed at.
    """
    infinite = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                             "cell_count": [np.inf, 3.0]})
    counts = regression_summary._design_counts(infinite)
    assert counts["n_cells"] is None
    assert counts["n_wells"] == 2
    assert counts["n_rows_prepared"] == 2
    assert "n_rows_fitted" not in counts

    finite = infinite.assign(cell_count=[5.0, 3.0])
    assert regression_summary._design_counts(finite)["n_cells"] == 8


def test_a_selection_table_with_no_coefficient_column_still_counts_hits():
    """Selection frequency alone is enough to say what was called.

    A bootstrap-LASSO table normally carries the coefficient too, and the
    called set is the intersection of "selected often" and "not shrunk to
    zero". With no coefficient column there is nothing to intersect, and the
    frequency alone has to stand -- the note says so rather than the count
    silently meaning something else.
    """
    frame = pd.DataFrame({"grna": ["a", "b", "c"],
                          "selection_frequency": [0.9, 0.1, 0.7]})
    mask, note = regression_summary._hit_mask(
        _run(coef_df=frame, penalised=True))
    assert list(mask) == [True, False, True]
    assert "NOT a P value" in note

    # With the coefficient column present the zero-coefficient row drops out,
    # which is the intersection the missing column cannot be part of.
    with_coefficients = frame.assign(coefficient=[0.4, 0.0, 0.0])
    intersected, _note = regression_summary._hit_mask(
        _run(coef_df=with_coefficients, penalised=True))
    assert list(intersected) == [True, False, False]


# ---------------------------------------------------------------------------
# regression_qc: a group is never empty
# ---------------------------------------------------------------------------

def test_no_positional_group_of_residuals_is_ever_empty():
    """UNREACHABLE ARCS, PROVEN: ``regression_qc`` lines 2958 and 2988.

    ``_grouped_residuals`` builds ``groups`` from ``pd.unique`` of the very
    ``keys`` array it then indexes ``ctx.resid`` with, so every group selects
    at least one residual. The edge/interior split and the per-group jitter
    loop both re-check ``.size`` afterwards, and neither can be false.

    Asserted over a column with a singleton group, an unbalanced group and a
    numeric-looking label -- the three that would break the argument if
    ``pd.unique`` and the boolean mask could ever disagree.
    """
    ctx = types.SimpleNamespace(
        resid=np.arange(9, dtype=float),
        metadata=pd.DataFrame({
            "rowID": ["r1", "r1", "r1", "r2", "r2", "r10", "r3", "r3", "r3"],
        }))

    groups, values = regression_qc._grouped_residuals(ctx, "rowID")

    assert groups == ["r1", "r2", "r3", "r10"], "natural, not lexical, order"
    assert [v.size for v in values] == [3, 2, 3, 1]
    assert all(v.size for v in values)
    assert sum(v.size for v in values) == ctx.resid.size

    # The two derived quantities the guards protect are non-empty for free.
    edge = np.concatenate([values[0], values[-1]])
    interior = np.concatenate(values[1:-1])
    assert edge.size == 4 and interior.size == 5

    # One distinct value is refused outright, which is why "no group" is not
    # a state the panel has to draw.
    with pytest.raises(regression_qc.PanelUnavailable, match="no between-"):
        regression_qc._grouped_residuals(
            types.SimpleNamespace(
                resid=np.zeros(3),
                metadata=pd.DataFrame({"rowID": ["r1", "r1", "r1"]})),
            "rowID")


# ---------------------------------------------------------------------------
# parameter_sweep: the ranked labels are the same labels
# ---------------------------------------------------------------------------

def test_a_control_that_is_present_always_has_a_rank():
    """UNREACHABLE ARC, PROVEN: ``parameter_sweep`` line 374's False side.

    ``ranked`` is ``frame`` sorted by ``|effect|`` -- a permutation, not a
    filter -- so ``ranked_labels`` holds exactly the strings ``labels`` holds.
    The code has already ``continue``d unless ``hit.any()``, so
    ``position.any()`` is true whenever it is evaluated.

    Asserted with NaN effects (which sorting moves but does not drop) and a
    control that is only a substring of its label.
    """
    frame = pd.DataFrame({
        "grna": ["ctrl_pos_1", "TargetA_2", "ctrl_neg_3", "TargetB_4"],
        "coefficient": [0.9, np.nan, -0.4, 0.2],
        "q_value": [0.01, 0.5, 0.2, 0.3],
    })
    out = parameter_sweep._named_control_rows(
        frame, {"positive": "ctrl_pos", "negative": "ctrl_neg",
                "absent": "ctrl_missing"})

    assert out["positive_present"] is True
    assert out["positive_rank"] == 1
    assert out["negative_present"] is True
    assert out["negative_rank"] == 2
    # A control that is not in the table gets no rank at all, which is the
    # only way a rank is ever missing.
    assert out["absent_present"] is False
    assert "absent_rank" not in out

    ranked = frame.assign(_abs=frame["coefficient"].abs()).sort_values(
        "_abs", ascending=False)
    assert sorted(ranked["grna"]) == sorted(frame["grna"])
