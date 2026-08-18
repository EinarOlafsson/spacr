"""Where the significance line is drawn, and on which P, is the run's choice.

Instruction 135, asked for on 2026-08-17: "add a setting that setts what alpha
the p threshold is set at and if adjusted p or raw p is used".

THE FAILURE THIS REPLACES. The correction's alpha was also the hit cut, and
the cut was always on the ADJUSTED P -- while the volcano's own right-click
menu could switch its axis to the RAW P. So results_significant.csv and the
figure printed beside it could mean two different things by "significant",
with nothing on either saying which.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.ml import _call_level_hits


def _coefficients(n=40, seed=0):
    """A gene-level table with a spread of P values, and nothing else."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([np.geomspace(1e-8, 0.04, 8), rng.uniform(0.06, 1, n - 8)])
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{200000 + i}]" for i in range(n)],
        "coefficient": rng.normal(0, 1.0, n),
        "p_value": p,
        "std_err": np.full(n, 0.2),
        "n_grna": np.full(n, 4),
        "n_gene": np.full(n, 1),
    })


def _call(settings, frame=None):
    frame = _coefficients() if frame is None else frame
    table, hits, _threshold, _rule = _call_level_hits(
        frame, "gene", settings, "ols", pd.DataFrame(), "pred")
    return table, hits


def _settings(**over):
    base = {"multiple_testing_method": "fdr_bh", "fdr_alpha": 0.05,
            "threshold_method": "none", "threshold_multiplier": 1.0,
            "min_n": 0, "negative_control": "", "positive_control": "",
            "controls": []}
    base.update(over)
    return base


def test_the_default_is_the_adjusted_p_at_the_correction_level():
    """Unchanged behaviour when nothing is set -- this is a new control, not
    a new default."""
    _table, hits = _call(_settings())
    _table2, hits2 = _call(_settings(p_threshold_alpha=0.05,
                                     p_threshold_kind="adjusted"))
    assert len(hits) == len(hits2)


def test_the_raw_p_calls_more_hits_than_the_adjusted_one():
    """The whole reason the choice matters. Over 40 tests, BH pulls the
    borderline ones back over the line."""
    _t, adjusted = _call(_settings(p_threshold_kind="adjusted"))
    _t, raw = _call(_settings(p_threshold_kind="raw"))
    assert len(raw) > len(adjusted)


def test_a_tighter_cut_calls_fewer():
    _t, loose = _call(_settings(p_threshold_alpha=0.05))
    _t, tight = _call(_settings(p_threshold_alpha=0.001))
    assert len(tight) < len(loose)


def test_the_cut_is_separate_from_the_correction_level():
    """`fdr_alpha` is what the PROCEDURE targets; `p_threshold_alpha` is what
    a coefficient is CALLED at. Correcting at 0.05 and reporting at 0.01 is an
    ordinary thing to want, and it must not change the q-values themselves."""
    table_a, _hits = _call(_settings(fdr_alpha=0.05, p_threshold_alpha=0.05))
    table_b, _hits = _call(_settings(fdr_alpha=0.05, p_threshold_alpha=0.01))
    pd.testing.assert_series_equal(table_a["q_value"], table_b["q_value"])


def test_every_called_hit_actually_beats_the_cut():
    for kind, column in (("adjusted", "q_value"), ("raw", "p_value")):
        _table, hits = _call(_settings(p_threshold_kind=kind,
                                       p_threshold_alpha=0.02))
        assert len(hits)
        assert (hits[column] < 0.02).all(), kind


def test_a_raw_cut_is_announced(capsys):
    """A cut on the raw P over hundreds of guides is a defensible choice and
    an indefensible accident, and the only way to tell them apart is whether
    the run said so."""
    _call(_settings(p_threshold_kind="raw"))
    printed = capsys.readouterr().out
    assert "Calling hits on the raw P" in printed
    assert "NOT corrected for multiple testing" in printed


def test_a_moved_alpha_is_announced_too(capsys):
    _call(_settings(p_threshold_alpha=0.01))
    printed = capsys.readouterr().out
    assert "Calling hits on the adjusted P at 0.01" in printed


def test_the_default_says_nothing_extra(capsys):
    """A note that fires every time is a note nobody reads."""
    _call(_settings())
    assert "Calling hits on the" not in capsys.readouterr().out


@pytest.mark.parametrize("given", [None, 0, ""])
def test_an_empty_alpha_falls_back_to_the_correction_level(given):
    """A blank box is not a request for a cut at zero, which would call
    nothing and look like a screen with no hits in it."""
    _t, fallback = _call(_settings(p_threshold_alpha=given))
    _t, default = _call(_settings())
    assert len(fallback) == len(default)


def test_an_unknown_kind_is_treated_as_adjusted():
    """The safe side. `check_settings` refuses anything but the two words, so
    reaching here means a hand-edited CSV -- and defaulting to the raw P
    would silently drop the correction."""
    _t, odd = _call(_settings(p_threshold_kind="Adjusted "))
    _t, default = _call(_settings())
    assert len(odd) == len(default)
    _t, nonsense = _call(_settings(p_threshold_kind="sideways"))
    assert len(nonsense) == len(default)


def test_the_line_follows_the_correction_level_unless_it_is_moved():
    """What stops this becoming a second `score_column`.

    Two controls for one question, where the only thing the second can
    express is a disagreement with the first, is exactly what instruction
    135 A had just retired. Every existing caller moves `fdr_alpha` alone and
    means "call hits at this level"; a hard 0.05 default would silently
    ignore all of them.
    """
    from spacr.settings import get_perform_regression_default_settings as g

    assert g({})["p_threshold_alpha"] == g({})["fdr_alpha"]
    assert g({"fdr_alpha": 0.2})["p_threshold_alpha"] == 0.2
    # And an explicit choice still wins, which is the point of having it.
    assert g({"fdr_alpha": 0.2,
              "p_threshold_alpha": 0.01})["p_threshold_alpha"] == 0.01
