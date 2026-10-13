"""Coverage-fill batch 2 for spacr.sequencing save/lookup helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sequencing as SEQ


# ---------------------------------------------------------------------------
# map_sequences_to_names
# ---------------------------------------------------------------------------

def test_map_sequences_to_names(tmp_path):
    csv = tmp_path / "bc.csv"
    pd.DataFrame({"sequence": ["ACGT", "TTGG"], "name": ["b1", "b2"]}).to_csv(
        csv, index=False)
    out = SEQ.map_sequences_to_names(str(csv), ["ACGT", "ZZZZ", "TTGG"], rc=False)
    assert out[0] == "b1" and out[2] == "b2"
    assert pd.isna(out[1])   # no match → NA


def test_map_sequences_to_names_rc(tmp_path):
    csv = tmp_path / "bc.csv"
    # 'ACGT' reverse-complemented is 'ACGT'; use asymmetric seq for a real test
    pd.DataFrame({"sequence": ["AAAC"], "name": ["b1"]}).to_csv(csv, index=False)
    # rc of 'AAAC' is 'GTTT' → lookup dict keyed on 'GTTT'
    out = SEQ.map_sequences_to_names(str(csv), ["GTTT"], rc=True)
    assert out[0] == "b1"


# ---------------------------------------------------------------------------
# get_consensus_base
# ---------------------------------------------------------------------------

def test_get_consensus_base():
    assert SEQ.get_consensus_base([("N", 40), ("A", 30)]) == "A"
    assert SEQ.get_consensus_base([("C", 40), ("N", 30)]) == "C"
    # both real → highest quality wins
    assert SEQ.get_consensus_base([("G", 20), ("T", 35)]) == "T"
    assert SEQ.get_consensus_base([("G", 35), ("T", 20)]) == "G"


# ---------------------------------------------------------------------------
# HDF5 / CSV savers
# ---------------------------------------------------------------------------

def test_save_df_to_hdf5_append(tmp_path):
    h5 = tmp_path / "d.h5"
    df1 = pd.DataFrame({"a": [1, 2]})
    SEQ.save_df_to_hdf5(df1, str(h5), key="df")
    SEQ.save_df_to_hdf5(pd.DataFrame({"a": [3]}), str(h5), key="df")
    with pd.HDFStore(str(h5), "r") as store:
        out = store["df"]
    assert len(out) == 3    # appended


def test_save_unique_combinations_to_csv(tmp_path):
    csv = tmp_path / "uc.csv"
    uc = pd.DataFrame({
        "rowID": ["r1", "r2"], "columnID": ["c1", "c1"],
        "grna_name": ["g1", "g2"], "count": [5, 3],
    })
    SEQ.save_unique_combinations_to_csv(uc, str(csv))
    # append same combos → counts sum
    SEQ.save_unique_combinations_to_csv(uc, str(csv))
    out = pd.read_csv(str(csv))
    total = out[out["grna_name"] == "g1"]["count"].sum()
    assert total == 10


def test_save_qc_df_to_csv(tmp_path):
    csv = tmp_path / "qc.csv"
    qc = pd.DataFrame({"total_reads": [100], "unmatched": [10]})
    SEQ.save_qc_df_to_csv(qc, str(csv))
    SEQ.save_qc_df_to_csv(qc, str(csv))   # element-wise sum
    out = pd.read_csv(str(csv))
    assert out["total_reads"].iloc[0] == 200


# ---------------------------------------------------------------------------
# reverse_complement / extract_sequence_and_quality
# ---------------------------------------------------------------------------

def test_reverse_complement():
    assert SEQ.reverse_complement("AAAC") == "GTTT"


def test_extract_sequence_and_quality():
    seq, qual = SEQ.extract_sequence_and_quality("ACGTACGT", "IIIIIIII", 2, 5)
    assert seq == "GTA" and qual == "III"
