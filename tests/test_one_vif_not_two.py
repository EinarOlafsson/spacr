"""One variance inflation factor, computed once.

Instruction 127, finding 4: `spacr/regression_qc.py` and
`spacr/regression_diagnostics.py` share exactly one name,
``variance_inflation_factors``, and computed it twice -- "a small correctness
risk of the same shape as finding 2".

IT IS NOT THE SAME AS FINDING 2, AND THE DIFFERENCE IS WORTH RECORDING. The
two statistical engines DISAGREED, on three of five inputs, always in the
direction that overstates significance. These two AGREED: measured before
anything was changed, on 200x20, 400x60 and 96x30 designs each carrying a
near-collinear pair, the largest relative difference was 2.3e-10, and both
named the same columns ``inf`` under exact collinearity.

Which is exactly why the duplication was worth removing rather than
arbitrating. Two routes to one number that agree today are two routes that can
stop agreeing, and nothing in the package was comparing them. This file is
what compares them now.

THE SURVIVOR WAS CHOSEN ON COST, since it could not be chosen on correctness.
The auxiliary-regression form ran one least-squares fit per guide, ``O(p^4)``;
``regression_qc`` uses the identity ``VIF_j = (R^-1)_jj``, one ``O(p^3)``
decomposition. Timed on this machine at 60 guides: 130.8 ms against 13.0 ms,
and a pooled screen has hundreds of guides, where the gap grows as the fourth
power.

WHAT DID NOT MOVE is the screen's contract around the number -- the refusal on
a rank-deficient design, the widest-support-first report cap, the
``wells_with_guide`` column. That is a different thing from the statistic, and
it is why both functions still exist.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import pytest

from spacr import regression_diagnostics as RD
from spacr import regression_qc as QC

SPACR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "spacr")


def _design(n_wells=200, n_guides=20, seed=0):
    """A sparse well-by-guide matrix with one near-collinear pair.

    A design where nothing is collinear would pass every assertion below while
    proving nothing: every VIF would be about 1 and the two routes would agree
    on a number neither of them had to work for.
    """
    rng = np.random.default_rng(seed)
    values = (rng.random((n_wells, n_guides)) < 0.25) * rng.random(
        (n_wells, n_guides))
    frame = pd.DataFrame(values, columns=[f"g{i}" for i in range(n_guides)])
    frame["g1"] = frame["g0"] + 1e-3 * rng.normal(size=n_wells)
    return frame


@pytest.mark.parametrize("shape", [(200, 20), (400, 60), (96, 30)])
def test_both_doors_return_the_same_number(shape):
    """Bit-identical, not "the same family". A tolerance would pass again on
    the day someone reintroduces a second implementation that happens to agree
    on the cases anyone thought to check."""
    frame = _design(*shape)
    reported = RD.variance_inflation_factors(frame).set_index("guide")["vif"]
    engine = QC.variance_inflation_factors(frame)

    assert set(reported.index) == set(engine.index)
    for guide, value in reported.items():
        assert value == engine[guide], guide


def test_the_screen_facing_call_uses_the_engine():
    """Delegation, asserted rather than assumed. Equal numbers alone would
    still be equal if the second implementation came back."""
    frame = _design(120, 12)
    calls = []
    original = QC.variance_inflation_factors

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    QC.variance_inflation_factors = spy
    try:
        RD.variance_inflation_factors(frame)
    finally:
        QC.variance_inflation_factors = original
    assert calls, "regression_diagnostics computed a VIF of its own"


def test_exact_collinearity_names_the_same_guides_on_both_sides():
    frame = _design(200, 10)
    frame["g9"] = frame["g0"] + frame["g2"]

    reported = RD.variance_inflation_factors(frame)
    engine = QC.variance_inflation_factors(frame)

    aliased = sorted(reported.loc[np.isinf(reported["vif"]), "guide"])
    assert aliased == sorted(engine.index[np.isinf(engine)])
    assert aliased == ["g0", "g2", "g9"]


def test_a_rank_deficient_design_is_refused_not_answered():
    """The engine would answer `inf` for every guide, which is true and
    useless. This is the contract that stays in the screen-facing call."""
    with pytest.raises(ValueError, match="rank deficient"):
        RD.variance_inflation_factors(_design(20, 40))

    # And the engine really does answer rather than refuse -- so the refusal
    # is this function's, and removing it would silently change the answer.
    assert np.isinf(QC.variance_inflation_factors(_design(20, 40))).all()


def test_the_report_cap_never_limited_the_computation():
    """`max_guides` says how many rows to PRINT. Each guide's VIF is still
    taken against every other guide, which is what makes the number mean
    anything."""
    frame = _design(500, 60)
    everything = RD.variance_inflation_factors(frame).set_index("guide")["vif"]
    capped = RD.variance_inflation_factors(frame, max_guides=5)

    assert len(capped) == 5
    for guide, value in capped.set_index("guide")["vif"].items():
        assert value == everything[guide]
    # The widest-support guides are the ones kept.
    support = (frame > 0).sum(axis=0).sort_values(ascending=False)
    assert set(capped["guide"]) == set(support.index[:5])


def test_the_frame_shape_is_unchanged():
    frame = _design(150, 8)
    result = RD.variance_inflation_factors(frame)
    assert list(result.columns) == ["guide", "vif", "wells_with_guide"]
    assert result["vif"].is_monotonic_decreasing
    assert result.index.tolist() == list(range(len(result)))


def test_a_design_with_nothing_varying_is_an_empty_report():
    frame = pd.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10})
    result = RD.variance_inflation_factors(frame)
    assert list(result.columns) == ["guide", "vif"]
    assert result.empty


def test_one_varying_guide_cannot_be_collinear_with_anything():
    frame = pd.DataFrame({"a": [1.0] * 10, "b": np.arange(10.0)})
    result = RD.variance_inflation_factors(frame)
    assert result["vif"].tolist() == [1.0]


#: Every definition in spacr/ whose name says "variance inflation", with what
#: it IS. Same shape as `test_one_volcano_for_each_job.py`'s inventory and for
#: the same reason: a third implementation is not forbidden, an UNRECORDED
#: third one is, and recording one costs a sentence about which job it does.
INVENTORY = {
    ("regression_qc.py", "variance_inflation_factors"):
        "THE ENGINE: VIF_j = (R^-1)_jj, one O(p^3) decomposition",
    ("regression_diagnostics.py", "variance_inflation_factors"):
        "not an implementation: the screen's contract around the engine -- "
        "refuses a rank-deficient design, caps the report, adds support",
}


def test_nothing_else_in_the_package_computes_a_vif():
    found = {}
    for folder, _dirs, files in os.walk(SPACR):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            relative = os.path.relpath(path, SPACR).replace(os.sep, "/")
            with open(path, encoding="utf-8") as handle:
                try:
                    tree = ast.parse(handle.read())
                except SyntaxError:                              # pragma: no cover
                    continue
            for node in ast.walk(tree):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef))
                        and "variance_inflation" in node.name):
                    found[(relative, node.name)] = node.lineno

    unrecorded = sorted(set(found) - set(INVENTORY))
    assert not unrecorded, (
        "a new variance-inflation implementation appeared and nobody said "
        f"which job it does: {unrecorded}. Add it to INVENTORY with a "
        "sentence, or make it call the engine.")
    gone = sorted(set(INVENTORY) - set(found))
    assert not gone, f"the inventory names something that is not there: {gone}"
