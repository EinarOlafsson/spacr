"""One mitotic hold-out for Measure and OPS: stage-model OR DNA-rule calls leave the DNA-phase fit."""
import numpy as np
import pandas as pd

from spacr.measure import dna_phases, mitotic_holdout
from spacr.ops_cell_cycle import _dna_phases


def _cells(n=600, seed=0):
    rng = np.random.default_rng(seed)
    g1 = rng.normal(100, 8, n // 2)
    s = rng.uniform(115, 185, n // 4)
    g2 = rng.normal(200, 14, n - n // 2 - n // 4)
    dna = np.concatenate([g1, s, g2])
    return pd.DataFrame({"_dna_integrated": dna, "well": "A1",
                         "site": np.arange(n) % 3})


def test_union_of_calls():
    a = np.array([True, False, False, False])
    b = pd.Series([False, False, True, np.nan])
    assert mitotic_holdout(a, b).tolist() == [True, False, True, False]
    assert mitotic_holdout(a, None).tolist() == a.tolist()
    assert mitotic_holdout(None, None) is None


def test_dna_rule_mitotic_cell_is_out_of_the_fit():
    cells = _cells()
    stage = np.zeros(len(cells), bool)
    rule = np.zeros(len(cells), bool)
    # 60 over-bright G2-like cells that only the DNA rule flags
    rule[-60:] = True
    with_rule = cells.assign(mitotic=rule)
    got, fits = _dna_phases(with_rule, {"cell_cycle_field_min_cells": 50}, stage)
    ref, ref_fits = dna_phases(cells.assign(mitotic=rule), mitotic="mitotic", min_cells=50)
    assert got["phase"].tolist() == ref["phase"].tolist()
    assert (got["phase"].to_numpy()[rule] == "M").all()
    assert fits["A1"]["fit"].n == int((~rule).sum())
    # without the column the stage flag alone is used and the fit sees every cell
    plain, plain_fits = _dna_phases(cells, {"cell_cycle_field_min_cells": 50}, stage)
    assert plain_fits["A1"]["fit"].n == len(cells)
    assert not (plain["phase"].to_numpy()[rule] == "M").any()

