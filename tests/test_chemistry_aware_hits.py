"""Chemistry-aware hits: SMILES on wells, structure clusters and SAR tables.

A small compound screen is planted with two active chemical series and a
set of unrelated inactive compounds, one of which is an inactive analogue
of the first series. Scored by SSMD, the hits must cluster into exactly the
two series, the analogue must join its series' cluster as an inactive
member, the host toxicity of Measure's viability step must reach the SAR
row of each compound, and the tables and the structure sheet must export.
Without RDKit the SAR tables still export and the missing clustering is
explained with the install command.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import (HitScoringError, _read_compound_map,
                            _structure_activity, _write_sar_report,
                            score_arrayed_screen)
import spacr.sp_stats as sp_stats

SERIES_A = ("c1ccc2[nH]c(-c3ccccc3)nc2c1", "Cc1ccc2[nH]c(-c3ccccc3)nc2c1",
            "Clc1ccc2[nH]c(-c3ccccc3)nc2c1")
SERIES_B = ("CC(C)(C)c1ccc(cc1)C(=O)NCCN", "CC(C)(C)c1ccc(cc1)C(=O)NCCCN")
INACTIVE = ("CCO", "CCCCO", "c1ccccc1", "CC(=O)O", "OCC(O)CO")
ANALOGUE = "Fc1ccc2[nH]c(-c3ccccc3)nc2c1"
LIBRARY = SERIES_A + SERIES_B + INACTIVE + (ANALOGUE,)
TOXIC = "cpd3"


def _screen():
    """Two plates; column 1 negative, 12 positive, compounds in 2-11."""
    rng = np.random.default_rng(1)
    rows, layout, host = [], [], []
    for plate in ("P1", "P2"):
        for r in range(1, 9):
            for c in range(1, 13):
                value = 50.0 + rng.normal(0, 1.0)
                kind = "neg" if c == 1 else ("pos" if c == 12 else "sample")
                name = None
                if kind == "pos":
                    value += 20.0
                if kind == "sample":
                    k = ((r - 1) * 10 + (c - 2)) % len(LIBRARY)
                    name = f"cpd{k}"
                    if k < len(SERIES_A) + len(SERIES_B):
                        value += 12.0
                    layout.append({"plateID": plate,
                                   "well": f"{'ABCDEFGH'[r - 1]}{c:02d}",
                                   "compound": name, "SMILES": LIBRARY[k]})
                rows.append({"plateID": plate, "rowID": f"r{r}",
                             "columnID": f"c{c}", "well_type": kind,
                             "compound": name, "signal": value})
                host.append({"plateID": plate, "rowID": r, "columnID": c,
                             "viability": 20.0 if name == TOXIC else 95.0,
                             "cytotoxicity_index":
                                 80.0 if name == TOXIC else 3.0})
    return pd.DataFrame(rows), pd.DataFrame(layout), pd.DataFrame(host)


def _scored(frame, treatment=None):
    return score_arrayed_screen(
        frame, "signal", plate_column="plateID", control_column="well_type",
        negative_levels=("neg",), positive_levels=("pos",),
        treatment_column=treatment, rank_by="ssmd")


def _by_name(sar):
    return sar.set_index("compound")


def test_the_compound_table_places_wells_and_names_smiles():
    _frame, layout, _host = _screen()
    table = _read_compound_map(layout)
    assert {"compound", "smiles", "plateID", "row_index",
            "column_index"} <= set(table.columns)
    first = table.iloc[0]
    assert (first["plateID"], first["row_index"],
            first["column_index"]) == ("P1", 1, 2)
    assert first["smiles"] == LIBRARY[0]


def test_a_table_without_smiles_is_refused():
    with pytest.raises(HitScoringError, match="SMILES"):
        _read_compound_map(pd.DataFrame({"compound": ["a"], "well": ["A01"]}))


def test_without_rdkit_the_sar_table_still_links_potency_and_toxicity(
        monkeypatch, tmp_path):
    def absent():
        raise ImportError(sp_stats._RDKIT_INSTALL.format(module="rdkit"))

    monkeypatch.setattr(sp_stats, "_rdkit", absent)
    frame, layout, host = _screen()
    chem = _structure_activity(_scored(frame), layout, host=host)
    assert not chem.clustered
    assert "pip install rdkit" in chem.report()
    sar = _by_name(chem.sar)
    hits = set(sar.index[sar["hit"]])
    assert hits == {f"cpd{k}" for k in range(5)}
    assert sar.loc[TOXIC, "cytotoxicity_index"] == 80.0
    assert sar.loc["cpd0", "viability"] == 95.0
    assert sar["cluster"].isna().all()
    written = _write_sar_report(chem, tmp_path / "sar")
    assert set(written) == {"sar_table", "sar_wells"}
    with pytest.raises(ImportError, match="pip install rdkit"):
        _structure_activity(_scored(frame), layout, cluster=True)


def test_joined_by_treatment_name_when_the_table_places_no_wells():
    frame, layout, _host = _screen()
    names = layout.drop_duplicates("compound")[["compound", "SMILES"]]
    chem = _structure_activity(_scored(frame, treatment="compound"), names,
                               cluster=False)
    assert set(chem.sar["compound"]) == {f"cpd{k}"
                                         for k in range(len(LIBRARY))}
    with pytest.raises(HitScoringError, match="Treatment"):
        _structure_activity(_scored(frame), names, cluster=False)


def test_hits_cluster_into_their_series_and_export(tmp_path):
    pytest.importorskip("rdkit")
    frame, layout, host = _screen()
    selectivity = pd.DataFrame({"compound": ["cpd0"], "si": [12.5]})
    chem = _structure_activity(_scored(frame), layout, host=host,
                               selectivity=selectivity)
    assert chem.clustered
    sar = _by_name(chem.sar)
    series_a = {sar.loc[f"cpd{k}", "cluster"] for k in range(3)}
    series_b = {sar.loc[f"cpd{k}", "cluster"] for k in (3, 4)}
    assert len(series_a) == 1 and len(series_b) == 1
    assert series_a != series_b
    assert len(chem.clusters) == 2
    assert chem.clusters["n_hits"].sum() == 5
    analogue = f"cpd{len(LIBRARY) - 1}"
    assert not sar.loc[analogue, "hit"]
    assert sar.loc[analogue, "cluster"] == series_a.pop()
    assert sar.loc[analogue, "similarity_to_hit"] >= 0.6
    for k in range(5, len(LIBRARY) - 1):
        assert pd.isna(sar.loc[f"cpd{k}", "cluster"])
    assert sar.loc["cpd0", "selectivity_si"] == 12.5
    assert sar.loc[TOXIC, "cytotoxicity_index"] == 80.0
    assert (chem.sar["potency"].iloc[:5] > 3).all()

    written = _write_sar_report(chem, tmp_path / "sar")
    assert {"sar_table", "sar_wells", "sar_clusters",
            "hit_structures"} <= set(written)
    assert os.path.getsize(written["hit_structures"]) > 0
    exported = pd.read_csv(written["sar_table"])
    assert {"smiles", "cluster", "potency", "phenotype",
            "cytotoxicity_index", "viability"} <= set(exported.columns)


def test_a_stricter_similarity_splits_a_series():
    pytest.importorskip("rdkit")
    frame, layout, _host = _screen()
    loose = _structure_activity(_scored(frame), layout, similarity=0.6)
    strict = _structure_activity(_scored(frame), layout, similarity=0.95)
    assert len(strict.clusters) > len(loose.clusters)
    assert len(strict.clusters) == 5


def test_an_unreadable_smiles_is_noted_not_fatal():
    pytest.importorskip("rdkit")
    frame, layout, _host = _screen()
    layout.loc[layout["compound"] == "cpd6", "SMILES"] = "not-a-smiles"
    chem = _structure_activity(_scored(frame), layout)
    assert not bool(_by_name(chem.sar).loc["cpd6", "smiles_valid"])
    assert any("could not be read" in note for note in chem.notes)
