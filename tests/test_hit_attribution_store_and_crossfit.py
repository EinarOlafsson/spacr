"""Storing, promoting and reverting a hit investigation, and the candidate cross-fit.

Units of :mod:`spacr.hit_attribution`, each walked over a case table:

* ``store_attribution`` writes an investigation's cells under its run context
  and refuses a missing database, duplicate object keys or a cell table
  without probabilities.
* ``promote_calls`` / ``revert_promotion`` -- the write/undo round trip on a
  user's ``png_list``: a revert restores exactly what was there (NULLs too),
  and reverting twice or reverting the unknown changes nothing.
* ``crossfit_candidate_probabilities`` refuses designs without four target and
  four control wells, and well keys that collapse into one group.
* ``fit_hit_attribution`` skips a recorded guide column the frame lost.
"""
from __future__ import annotations

import itertools
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import hit_attribution as ha
from spacr.hit_attribution import (HitAttributionError, HitInvestigationResult,
                                   HitRunContext, InsufficientDesignError,
                                   promote_calls, revert_promotion,
                                   store_attribution)

KEYS = [f"plate1_r1_c1_f1_o{i}" for i in range(1, 5)]


def _database(folder):
    """A measurements database whose png_list has two hand annotations."""
    os.makedirs(str(folder), exist_ok=True)
    path = os.path.join(str(folder), "measurements.db")
    png = pd.DataFrame({
        "prcfo": KEYS,
        "png_path": [f"/crops/{k}.png" for k in KEYS],
        "test_annotation": ["keep-me", None, "also-mine", None],
    })
    with sqlite3.connect(path) as db:
        png.to_sql("png_list", db, index=False)
    return path


def _result(run_id="run-1", cells=None):
    if cells is None:
        cells = pd.DataFrame({"prcfo": KEYS,
                              "candidate_probability": [0.9, 0.8, 0.2, 0.1]})
    return HitInvestigationResult(
        attribution_run_id=run_id,
        context=HitRunContext(
            regression_results_folder="/screens/plate1/results",
            regression_run_sha256="0" * 64, gene="TGGT1_123456",
            phenotype="infection", effect=1.25, guides=("g1", "g2"),
            fdr=0.05, direction="positive"),
        cells=cells,
        wells=pd.DataFrame({"prc": ["plate1_r1_c1"], "fraction": [0.5]}),
        enrichment={"odds_ratio": 3.0},
        feature_columns=["area"],
        split_level="well",
    )


def _rows(path, sql):
    with sqlite3.connect(path) as db:
        return db.execute(sql).fetchall()


def _annotations(path, column="test_annotation"):
    return dict(_rows(path, f'SELECT prcfo, "{column}" FROM png_list ORDER BY prcfo'))


def test_store_attribution(subtests, tmp_path):
    """Exact stored rows and context; each malformed input is refused by name."""
    with subtests.test(msg="defaults derived from the probability"):
        path = _database(tmp_path / "defaults")
        assert store_attribution(path, _result()) == 4
        assert _rows(path, "SELECT * FROM hit_investigation_cells ORDER BY prcfo") == [
            ("run-1", KEYS[0], 0.9, pytest.approx(0.2), 1, -1),
            ("run-1", KEYS[1], 0.8, pytest.approx(0.4), 1, -1),
            ("run-1", KEYS[2], 0.2, pytest.approx(0.4), 0, -1),
            ("run-1", KEYS[3], 0.1, pytest.approx(0.2), 0, -1),
        ]
        (run_id, created, context), = _rows(path, "SELECT * FROM hit_investigation_runs")
        assert run_id == "run-1" and created
        assert json.loads(context) == {
            "regression_results_folder": "/screens/plate1/results",
            "regression_run_sha256": "0" * 64, "gene": "TGGT1_123456",
            "phenotype": "infection", "effect": 1.25, "guides": ["g1", "g2"],
            "fdr": 0.05, "direction": "positive", "feature_columns": ["area"],
            "split_level": "well", "warnings": [],
            "enrichment": {"odds_ratio": 3.0}}
    with subtests.test(msg="explicit uncertainty, call and fold are kept"):
        path = _database(tmp_path / "explicit")
        cells = pd.DataFrame({"prcfo": KEYS[:2], "candidate_probability": [0.9, 0.3],
                              "candidate_uncertainty": [0.5, 0.6],
                              "candidate_call": [False, True],
                              "attribution_fold": [2, 3]})
        assert store_attribution(path, _result(cells=cells)) == 2
        assert _rows(path, "SELECT * FROM hit_investigation_cells ORDER BY prcfo") == [
            ("run-1", KEYS[0], 0.9, 0.5, 0, 2), ("run-1", KEYS[1], 0.3, 0.6, 1, 3)]

    path = _database(tmp_path / "refusals")
    missing = str(tmp_path / "nope.db")
    refusals = [
        ("database that is not there", missing, _result(),
         f"no database at {missing}"),
        ("duplicate object keys", path, _result(cells=pd.DataFrame({
            "prcfo": [KEYS[0], KEYS[0]], "candidate_probability": [0.9, 0.4]})),
         "investigation contains duplicate prcfo object keys"),
        ("no probability column", path, _result(cells=pd.DataFrame({"prcfo": KEYS})),
         r"investigation cells is missing \['candidate_probability'\]; "
         r"it has \['prcfo'\]"),
    ]
    for case_id, db, result, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(HitAttributionError, match=message):
                store_attribution(db, result)
    with subtests.test(msg="a refusal writes nothing"):
        assert not os.path.exists(missing)
        assert _rows(path, "SELECT name FROM sqlite_master WHERE type='table'") == [
            ("png_list",)]


def test_promote_calls_and_revert_promotion(subtests, tmp_path):
    """Promotion writes the calls; revert restores the exact prior values once."""
    path = _database(tmp_path)
    before = _annotations(path)
    assert before == {KEYS[0]: "keep-me", KEYS[1]: None,
                      KEYS[2]: "also-mine", KEYS[3]: None}
    with subtests.test(msg="revert before any promotion is a no-op"):
        assert revert_promotion(path, "anything") == 0
    store_attribution(path, _result())

    for bad in ("drop table png_list; --", "a b", "col-1", '"x"', "1abc", ""):
        with subtests.test(msg=f"unsafe column {bad!r}"):
            with pytest.raises(HitAttributionError,
                               match="only letters, numbers and underscores"):
                promote_calls(path, "run-1", bad)
    with subtests.test(msg="unknown run"):
        with pytest.raises(HitAttributionError,
                           match="no stored cells for attribution run 'run-9'"):
            promote_calls(path, "run-9", "test_annotation")

    with subtests.test(msg="a new column is created and filled"):
        promote_calls(path, "run-1", "brand_new_column")
        assert _annotations(path, "brand_new_column") == {
            KEYS[0]: 1, KEYS[1]: 1, KEYS[2]: 0, KEYS[3]: 0}

    promotion = promote_calls(path, "run-1", "test_annotation")
    with subtests.test(msg="promotion overwrites with the calls"):
        # the existing TEXT column stores the integer calls with text affinity
        assert _annotations(path) == {KEYS[0]: "1", KEYS[1]: "1",
                                      KEYS[2]: "0", KEYS[3]: "0"}
    with subtests.test(msg="an unknown promotion id changes nothing"):
        assert revert_promotion(path, "no-such-promotion-id") == 0
        # the existing TEXT column stores the integer calls with text affinity
        assert _annotations(path) == {KEYS[0]: "1", KEYS[1]: "1",
                                      KEYS[2]: "0", KEYS[3]: "0"}
    with subtests.test(msg="revert restores exactly, NULLs included"):
        assert revert_promotion(path, promotion) == 4
        assert _annotations(path) == before
    with subtests.test(msg="a second revert re-applies nothing"):
        assert revert_promotion(path, promotion) == 0
        assert _annotations(path) == before


def _colliding_wells(n=8):
    """Distinct (plate, row, column) keys that all join to one ``|`` group key."""
    tokens = ["a"] * 6
    return [("|".join(tokens[:i]), "|".join(tokens[i:j]), "|".join(tokens[j:]))
            for i, j in itertools.combinations(range(1, 6), 2)][:n]


def _candidate_frame(keys, *, seed=5, cells_per_well=8):
    """Cells for ``keys``, alternating target and control wells."""
    rng = np.random.default_rng(seed)
    rows = []
    for index, (plate, row, column) in enumerate(keys):
        target = index % 2 == 0
        for cell in range(cells_per_well):
            rows.append({
                "plateID": plate, "rowID": row, "columnID": column,
                "prcfo": f"{index}_{cell}", "target_well": target,
                "cell_area": rng.normal(2.0 if target else 0.0, 0.5),
                "cell_texture": rng.normal(1.0 if target else 0.0, 0.5)})
    return pd.DataFrame(rows)


def test_crossfit_candidate_probabilities(subtests):
    """Too few wells or one collapsed group is refused; a real design is cross-fit."""
    keys = _colliding_wells()
    assert len(set(keys)) == 8 and len({"|".join(k) for k in keys}) == 1
    plain = [tuple(part.replace("|", "-") for part in key) for key in keys]
    frame = _candidate_frame(plain)
    short = _candidate_frame(plain[:7])

    refusals = [
        ("one control well short", short, {},
         r"at least four independent target and four independent control wells "
         r"are required \(have 4 and 3\)"),
        ("keys collapsing into one group", _candidate_frame(keys),
         {"prefer_plate": False}, "cross-fitting needs at least two groups"),
    ]
    for case_id, data, kwargs, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(InsufficientDesignError, match=message):
                ha.crossfit_candidate_probabilities(data, **kwargs)

    with subtests.test(msg="unambiguous keys cross-fit by well"):
        scored, features, level, warnings = ha.crossfit_candidate_probabilities(
            frame, prefer_plate=False, threshold=0.5)
        assert level == "well" and warnings == []
        assert features == ["cell_area", "cell_texture"]
        assert len(scored) == len(frame)
        assert sorted(set(scored["attribution_fold"])) == [0, 1, 2, 3, 4]
        p = scored["candidate_probability"].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.testing.assert_allclose(scored["candidate_uncertainty"],
                                   1 - np.abs(2 * p - 1))
        assert (scored["candidate_call"] == (p >= 0.5)).all()
        # each well is held out whole, in exactly one fold
        per_well = scored.groupby(["plateID", "rowID", "columnID"])
        assert (per_well["attribution_fold"].nunique() == 1).all()


def _screen(seed=11, cells_per_well=10):
    """Three plates of eight wells, with a planted guide-fraction gradient."""
    rng = np.random.default_rng(seed)
    cells, fractions = [], []
    designs = [0.0, 0.2, 0.45, 0.75, 0.0, 0.3, 0.6, 0.0]
    for plate in range(3):
        for well, fraction in enumerate(designs):
            row, column = f"r{well // 4 + 1}", f"c{well % 4 + 1}"
            guide = "EAF1_1" if well % 2 == 0 else "EAF1_2"
            fractions += [
                {"plateID": f"p{plate}", "rowID": row, "columnID": column,
                 "grna": guide, "fraction": fraction},
                {"plateID": f"p{plate}", "rowID": row, "columnID": column,
                 "grna": "NTC", "fraction": 1 - fraction},
            ]
            for cell in range(cells_per_well):
                truth = int(rng.random() < fraction)
                cells.append({
                    "prcfo": f"p{plate}_{row}_{column}_f1_o{cell}",
                    "plateID": f"p{plate}", "rowID": row, "columnID": column,
                    "fieldID": "f1", "object_label": cell,
                    "cell_area": rng.normal(2.5 * truth, 0.7),
                    "cell_texture": rng.normal(1.4 * truth, 0.85),
                    "phenotype_score": rng.random(),
                })
    return ha.build_hit_cell_frame(
        pd.DataFrame(cells), pd.DataFrame(fractions),
        target_guides=["EAF1_1", "EAF1_2"], score_column="phenotype_score")


def test_fit_hit_attribution_skips_a_stale_guide_column():
    """A recorded guide column the frame no longer carries is skipped, not aggregated."""
    frame = _screen()
    columns = dict(frame.attrs["target_guide_columns"])
    live = sorted(columns.values())
    assert len(live) == 2
    columns["EAF1_gone"] = "target_guide_fraction__stale"
    frame.attrs["target_guide_columns"] = columns

    result = ha.fit_hit_attribution(
        frame, target_gene="EAF1", feature_columns=["cell_area", "cell_texture"],
        split_by="plate", threshold=0.7, n_bootstrap=20, n_permutations=20,
        random_seed=5)

    for column in live:
        assert column in result.wells.columns
    assert "target_guide_fraction__stale" not in result.wells.columns
    assert len(result.wells) == 24
