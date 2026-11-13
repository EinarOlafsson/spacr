"""Table-driven tests of the invasion assay in ``spacr.submodules``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture:

* ``_bimodality_coefficient``: the exact coefficient, and ``NaN`` wherever a
  sample cannot support one (too few objects, overflowing moments, a kurtosis
  that leaves no ratio);
* ``_invasion_threshold``: the exact cut inside a real gap for every offered
  method, ``NaN`` for data that cannot be thresholded, and the refusal of an
  unknown method;
* ``_invasion_control_mask``: control wells named by row, column, well or
  full key, as a list or a bare string;
* ``analyze_invasion``: the per-field and per-well invaded / attached counts
  from a small ``measurements.db``.

The databases are built with the writers of ``tests/test_invasion_assay.py``.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr import submodules as S  # noqa: E402
from tests.test_invasion_assay import settings_for, split, well, write_db  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _same(got, want):
    """Equality that treats NaN as equal to NaN."""
    if isinstance(want, float) and np.isnan(want):
        return isinstance(got, float) and np.isnan(got)
    return got == pytest.approx(want)


NAN = float("nan")


def test_bimodality_coefficient_is_exact_or_unknown(subtests, monkeypatch):
    """The coefficient is exact on a two-point mixture, below 5/9 on a normal
    sample, 0 on one repeated value and NaN where a sample cannot support it."""
    two_point = np.concatenate([np.zeros(20), np.ones(20)])
    cases = [
        # (id, values, kwargs, expected)
        ("a 0/1 mixture", two_point, {}, 1.0),
        ("a 1/5 mixture", np.array([1.0] * 30 + [5.0] * 30), {}, 1.0),
        ("a normal sample", np.random.default_rng(3).normal(size=400), {}, 0.30770991385390983),
        ("one value repeated", np.full(40, 7.0), {}, 0.0),
        ("fewer objects than the default 30", np.arange(10.0), {}, NAN),
        ("an empty sample", [], {"min_objects": 4}, NAN),
        ("moments that overflow", np.array([1e300, -1e300] * 30), {}, NAN),
    ]
    for case_id, values, kw, expected in cases:
        with subtests.test(msg=case_id):
            assert _same(S._bimodality_coefficient(values, **kw), expected)

    with subtests.test(msg="kurtosis + 3 at or below zero leaves no coefficient"):
        import scipy.stats
        with monkeypatch.context() as mp:
            mp.setattr(scipy.stats, "kurtosis", lambda *_a, **_k: -3.0)
            assert np.isnan(S._bimodality_coefficient(two_point))


def test_invasion_threshold_cuts_inside_the_gap_or_answers_unknown(subtests, monkeypatch):
    """Every offered method puts the cut exactly between two populations;
    data that cannot be thresholded gives NaN, an unknown method is refused."""
    gap_1_10 = np.array([1.0] * 20 + [10.0] * 20)
    gap_1_9 = np.concatenate([np.full(20, 1.0), np.full(20, 9.0)])
    too_wide = np.array([1e308, -1e308, 0.0])
    cases = [(f"{m}: 1 | 10", gap_1_10, m, 5.5) for m in ("otsu", "triangle", "yen", "li", "mean")]
    cases += [
        ("default method: 1 | 9", gap_1_9, None, 5.0),
        ("mean: 1 | 9", gap_1_9, "mean", 5.0),
        ("one distinct value", np.full(20, 3.0), "otsu", NAN),
        ("a single value", np.array([1.0]), "otsu", NAN),
        ("no values", np.array([]), "mean", NAN),
    ]
    cases += [(f"{m}: a range too wide to bin", too_wide, m, NAN)
              for m in ("otsu", "triangle", "yen")]
    for case_id, values, method, expected in cases:
        with subtests.test(msg=case_id):
            got = (S._invasion_threshold(values) if method is None
                   else S._invasion_threshold(values, method))
            assert _same(got, expected)

    with subtests.test(msg="an unknown method is refused by setting name"):
        with pytest.raises(ValueError, match="outside_threshold_method"):
            S._invasion_threshold(np.arange(10.0), method="kittler")

    with subtests.test(msg="a method that raises on the data gives NaN"):
        from skimage import filters

        def refuse(_values):
            raise ValueError("cannot compute a threshold from this")

        with monkeypatch.context() as mp:
            mp.setattr(filters, "threshold_otsu", refuse)
            assert np.isnan(S._invasion_threshold(gap_1_9, method="otsu"))


def test_invasion_control_mask_selects_wells_by_any_name(subtests):
    """A control is named by row, column, row_column or full key, as a list or
    one bare string; naming nothing selects nothing, on the frame's index."""
    letters = pd.DataFrame({"prc": ["p1_A_c1", "p1_B_c2", "p1_C_c3"],
                            "rowID": ["A", "B", "C"], "columnID": ["c1", "c2", "c3"]},
                           index=[10, 11, 12])
    numbered = pd.DataFrame({"prc": ["plate1_r1_c1", "plate1_r1_c12", "plate1_r2_c12"],
                             "rowID": ["r1", "r1", "r2"], "columnID": ["c1", "c12", "c12"]})
    cases = [
        # (id, frame, controls, expected mask)
        ("a row", letters, ["A"], [True, False, False]),
        ("a column", letters, ["c3"], [False, False, True]),
        ("row_column", letters, ["B_c2"], [False, True, False]),
        ("a full key as a bare string", letters, "p1_B_c2", [False, True, False]),
        ("row_column as a bare string", numbered, "r1_c12", [False, True, False]),
        ("a column as a bare string", numbered, "c12", [False, True, True]),
        ("several names", numbered, ["c1", "r2_c12"], [True, False, True]),
        ("None", letters, None, [False] * 3),
        ("an empty list", numbered, [], [False] * 3),
        ("an empty tuple", letters, (), [False] * 3),
    ]
    for case_id, frame, controls, expected in cases:
        with subtests.test(msg=case_id):
            mask = S._invasion_control_mask(frame, controls)
            assert mask.tolist() == expected
            assert list(mask.index) == list(frame.index)


# ---------------------------------------------------------------------------
# analyze_invasion
# ---------------------------------------------------------------------------

def _source(root, fields, drop_area=False, parasite_columns=None):
    """An invasion ``measurements.db``, optionally reshaped after writing."""
    src = write_db(root, fields)
    if drop_area or parasite_columns:
        db = os.path.join(src, "measurements", "measurements.db")
        with sqlite3.connect(db) as con:
            frame = pd.read_sql("SELECT * FROM pathogen", con)
            if drop_area:
                frame = frame.drop(columns=["pathogen_area"])
            for column, value in (parasite_columns or {}).items():
                frame[column] = value
            frame.to_sql("pathogen", con, index=False, if_exists="replace")
    return src


CLEAN = [{"row": "r1", "column": "c1", "outside": split(36, 24)},
         {"row": "r1", "column": "c2", "outside": split(24, 36)}]


def _one_well_db(root, outside, *, host):
    """One well whose outside-stain statistic is set literally; parasites sit
    in host cells only when ``host``."""
    measurements = os.path.join(str(root), "measurements")
    os.makedirs(measurements, exist_ok=True)
    prcf = "plate1_r1_c1_f1"
    meta = {"plateID": "plate1", "rowID": "r1", "columnID": "c1", "fieldID": "f1", "prcf": prcf}
    pathogens = [dict(meta, object_label=i, cell_id=i if host else 0, pathogen_area=100.0,
                      pathogen_channel_0_mean_intensity=200.0,
                      pathogen_channel_1_percentile_95=float(v))
                 for i, v in enumerate(outside, start=1)]
    cells = ([dict(meta, object_label=i, cell_area=20000.0) for i in range(1, len(outside) + 1)]
             if host else [dict(meta, object_label=900, cell_area=20000.0)])
    with sqlite3.connect(os.path.join(measurements, "measurements.db")) as con:
        pd.DataFrame(pathogens).to_sql("pathogen", con, index=False, if_exists="replace")
        pd.DataFrame(cells).to_sql("cell", con, index=False, if_exists="replace")
    return str(root)


def test_analyze_invasion_counts_invaded_and_attached_parasites_per_well(
        subtests, tmp_path, monkeypatch, capsys):
    """Each well's invaded / attached / total counts and efficiency are exact;
    unclassifiable plates and uninvaded fields report zeros rather than
    missing columns."""
    cases = [
        # (id, source builder, settings overrides, expected per prc
        #  (n_invaded, n_attached, n_total, efficiency), expected field n_attached)
        ("a clean two-condition plate", lambda d: _source(d, CLEAN), {},
         {"plate1_r1_c1": (36, 24, 60, 0.6), "plate1_r1_c2": (24, 36, 60, 0.4)}, [24, 36]),
        ("no area column is still classified", lambda d: _source(d, CLEAN, drop_area=True), {},
         {"plate1_r1_c1": (36, 24, 60, 0.6), "plate1_r1_c2": (24, 36, 60, 0.4)}, [24, 36]),
        ("one population cannot be split",
         lambda d: _source(d, [{"row": "r1", "column": "c1", "outside": [10.0] * 40},
                               {"row": "r1", "column": "c2", "outside": [10.0] * 40}]), {},
         None, [0, 0]),
        ("nothing in a host cell is all attached",
         lambda d: _one_well_db(d, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], host=False), {},
         {"plate1_r1_c1": (0, 6, 6, 0.0)}, [6]),
        ("parasites in host cells split by stain",
         lambda d: _one_well_db(d, [10.0] * 4 + [100.0] * 6, host=True), {},
         {"plate1_r1_c1": (4, 6, 10, 0.4)}, [6]),
        ("a group column the cell table lacks seeds nothing",
         lambda d: _source(d, CLEAN, parasite_columns={"strain": "RH"}),
         {"group_column": "strain"},
         {"plate1_r1_c1": (36, 24, 60, 0.6), "plate1_r1_c2": (24, 36, 60, 0.4)}, [24, 36]),
    ]
    for case_id, build, over, expected, attached in cases:
        with subtests.test(msg=case_id):
            out = S.analyze_invasion(settings_for(build(tmp_path / case_id.replace(" ", "_")),
                                                  **over))
            assert sorted(out["fields"]["n_attached"]) == attached
            if expected is None:
                assert set(out["parasites"]["invasion_class"]) == {"unclassified"}
                assert list(out["fields"]["n_invaded"]) == [0, 0]
                assert list(out["fields"]["n_total"]) == [0, 0]
                continue
            assert len(out["wells"]) == len(expected)
            for prc, (inv, att, tot, eff) in expected.items():
                row = well(out, prc)
                assert (row["n_invaded"], row["n_attached"], row["n_total"]) == (inv, att, tot)
                assert row["invasion_efficiency"] == pytest.approx(eff)
            if "drop" in case_id or "no area" in case_id:
                assert "pathogen_area" not in out["parasites"].columns
            if "group column" in case_id:
                assert set(out["wells"]["strain"]) == {"RH"}
            if case_id == "a clean two-condition plate":
                assert "pathogen_area" in out["parasites"].columns
                assert "condition" in out["wells"].columns
                assert all(out["wells"]["qc_pass"])
            if case_id.startswith("nothing in a host"):
                assert set(out["parasites"]["invasion_class"]) == {"attached"}
                assert list(out["fields"]["n_invaded"]) == [0]

    with subtests.test(msg="verbose lists thresholds, and QC only when a well fails"):
        capsys.readouterr()
        S.analyze_invasion(settings_for(_source(tmp_path / "cleanv", CLEAN), verbose=True))
        printed = capsys.readouterr().out
        assert "Per-field thresholds:" in printed and "QC:" not in printed
        S.analyze_invasion(settings_for(_source(tmp_path / "thinv", [
            {"row": "r1", "column": "c1", "outside": split(6, 4)},
            {"row": "r1", "column": "c2", "outside": split(4, 6)}]), verbose=True))
        flagged = capsys.readouterr().out
        assert "QC:" in flagged and "low_total" in flagged

    with subtests.test(msg="the API runs before the GUI registers its defaults"):
        from spacr import settings as settings_module
        src = _source(tmp_path / "hook", CLEAN)
        with_hook = S.analyze_invasion(settings_for(src))
        with monkeypatch.context() as mp:
            mp.delattr(settings_module, "set_analyze_invasion_defaults")
            without_hook = S.analyze_invasion(settings_for(src))
        pd.testing.assert_frame_equal(without_hook["wells"], with_hook["wells"])
