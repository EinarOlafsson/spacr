"""Item 978: the E-test's default null is guide_null; settings saved before schema 946 keep empirical_null.

The pooled non-targeting null ignores the spread between guides, so at OPS
depth it calls most genes without an effect (item 968's simulation: 83 % at
160 cells per gene); the guide null keeps that spread and calls none.
"""
from __future__ import annotations

import numpy as np

#: The pre-946 value of the one key item 978 changes.
OLD = {"ops_etest_mode": "empirical_null"}


def test_the_new_default_and_the_schema_row():
    from spacr.ops_settings import OPS_DEFAULTS, ops_defaults
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION

    assert OPS_DEFAULTS["ops_etest_mode"] == "guide_null"
    assert ops_defaults({})["ops_etest_mode"] == "guide_null"
    assert SETTINGS_SCHEMA_VERSION >= 946
    assert DEFAULT_CHANGES[946] == OLD


def test_a_file_saved_before_946_keeps_the_empirical_null():
    from spacr.cli import _in_its_own_era
    from spacr.ops_settings import ops_defaults
    from spacr.settings import SETTINGS_VERSION_KEY

    known = ops_defaults({})
    unstamped = _in_its_own_era({"dst_root": "out"}, known)
    assert unstamped["ops_etest_mode"] == "empirical_null"
    before = _in_its_own_era({SETTINGS_VERSION_KEY: 944, "dst_root": "out"}, known)
    assert before["ops_etest_mode"] == "empirical_null"
    now = _in_its_own_era({SETTINGS_VERSION_KEY: 946, "dst_root": "out"}, known)
    assert "ops_etest_mode" not in now or now["ops_etest_mode"] == "guide_null"


def test_a_file_that_sets_the_mode_keeps_it():
    from spacr.cli import _in_its_own_era
    from spacr.ops_settings import ops_defaults

    known = ops_defaults({})
    for mode in ("empirical_null", "guide_null", "permutation"):
        assert _in_its_own_era({"ops_etest_mode": mode}, known)["ops_etest_mode"] == mode


def test_the_etest_phase_without_the_key_uses_the_guide_null(tmp_path):
    from spacr.gene_measurement_compare import etest_phase
    from spacr.tabular import write_table

    import pandas as pd

    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(20) for _ in range(30)]
    nt = [f"nontargeting_{j}" for j in range(40) for _ in range(80)]
    guides = [f"g{i}_{k % 4}" for i in range(20) for k in range(30)] + [f"nt{j}" for j in range(40) for _ in range(80)]
    labels = genes + ["nontargeting"] * len(nt)
    folder = tmp_path / "cells"
    folder.mkdir()
    np.save(folder / "cells_E.npy", rng.standard_normal((len(labels), 6)).astype(np.float32))
    write_table(pd.DataFrame({"gene_symbol_0": labels, "cell_barcode_0": guides}),
                str(folder / "cells_meta.parquet"), canonicalise=False)
    out = etest_phase({"ops_etest_cells": str(folder), "ops_power_cells": [None], "ops_power_seeds": [0],
                       "ops_etest_reference_cells": 500, "ops_etest_null": 50, "ops_etest_holdout": 20},
                      str(folder), str(tmp_path / "dst"))
    explicit = etest_phase({"ops_etest_cells": str(folder), "ops_power_cells": [None], "ops_power_seeds": [0],
                            "ops_etest_reference_cells": 500, "ops_etest_null": 50, "ops_etest_holdout": 20,
                            "ops_etest_mode": "guide_null"},
                           str(folder), str(tmp_path / "dst2"))
    assert open(out["outputs"]["power_hits"], "rb").read() == open(explicit["outputs"]["power_hits"], "rb").read()
