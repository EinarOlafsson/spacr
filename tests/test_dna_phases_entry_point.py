"""One model-free public DNA-phase call, used by every caller (spec G5, item 995)."""
import numpy as np
import pandas as pd

from spacr import measure, ops_cell_cycle as cc
from tests.test_cell_cycle_field_centring_995 import _db, _nuclei


def _cells():
    table = _nuclei()
    content = (table["nucleus_channel_0_integrated_intensity"]
               - table["nucleus_channel_0_outside_percentile_50"] * table["nucleus_area"])
    return pd.DataFrame({"well": table["columnID"].to_numpy(), "site": table["fieldID"].to_numpy(),
                         "mitotic": table["mitotic_856"].to_numpy(),
                         "_dna_integrated": content.to_numpy()})


def _spy(monkeypatch):
    calls = []
    original = measure.dna_phases

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(measure, "dna_phases", spy)
    return calls


def test_both_callers_go_through_the_public_entry_point(tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    cells = _cells()
    cc._dna_phases(cells.assign(site=cells["site"]), {}, cells["mitotic"].to_numpy())
    assert len(calls) == 1
    db = _db(tmp_path, _nuclei())
    measure._classify_cell_cycle(db, {"channels": [0], "cell_cycle_method": "measurements",
                                      "plot": False, "cell_cycle_field_centre": True,
                                      "cell_cycle_fit_group": "well"})
    assert len(calls) == 2


def test_entry_point_needs_no_model_and_equals_the_ops_call(tmp_path):
    cells = _cells()
    ops, _ = cc._dna_phases(cells, {}, cells["mitotic"].to_numpy())
    free, fits = measure.dna_phases(cells[["_dna_integrated", "well", "site"]])
    assert set(free["phase"]) <= {"G1", "S", "G2", "subG1", ">4N", "M"}
    assert set(fits) == {"c1", "c2"}
    mitotic, _ = measure.dna_phases(cells, mitotic="mitotic")
    pd.testing.assert_frame_equal(mitotic, ops)
    path = str(tmp_path / "w.parquet")
    cells.assign(junk=1.0).to_parquet(path, index=False)
    from_path, _ = measure.dna_phases(path, mitotic="mitotic")
    pd.testing.assert_frame_equal(from_path, mitotic)
