"""The nuisance covariates read the DNA-fit posteriors the cell-cycle phase writes."""
import numpy as np
import pandas as pd


def test_nuisance_covariates_use_the_posteriors_the_phase_writes(tmp_path):
    import pytest

    from spacr import ops_cell_cycle as cc
    from spacr import ops_engine, ops_tvn
    from spacr.ops_store import read_table
    from tests.test_ops_cell_cycle_990 import PLATE, _base, _cells

    base = _base(tmp_path)
    destination = tmp_path / "run"
    destination.mkdir()
    for well, rows in _cells().groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    settings = {"ops_cell_cycle": "DNA", "ops_cell_cycle_base_model": base,
                "nucleus_channel": "dapi"}
    db = str(destination / "measurements.db")
    cc.run_cell_cycle_phase(db, PLATE, ["A1", "A2"], settings, str(destination), permutations=20)
    out = read_table(db, "ops_cell_cycle")
    written = [ops_tvn.dna_posterior_column(p) for p in ("S", "G2")]
    assert all(c in out.columns for c in written)
    for kind in ("cell_cycle", "stage_probs"):
        z = ops_tvn.nuisance_covariates(out, [kind])
        assert {"nz_cc_posterior_S", "nz_cc_posterior_G2"} <= set(z.columns)
        assert z["nz_cc_posterior_S"].nunique() > 1
        np.testing.assert_allclose(z["nz_cc_posterior_S"], out[written[0]].to_numpy(float))
    old = out.rename(columns={"dna_phase_posterior_S": "posterior_S",
                              "dna_phase_posterior_G2": "posterior_G2"})
    pd.testing.assert_frame_equal(ops_tvn.nuisance_covariates(old, ["cell_cycle"]),
                                  ops_tvn.nuisance_covariates(out, ["cell_cycle"]))
    with pytest.raises(ValueError, match="posterior_S"):
        ops_tvn.nuisance_covariates(out.drop(columns=written), ["stage_probs"])
