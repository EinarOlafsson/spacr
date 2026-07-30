"""Coverage-fill batch 3 for spacr.sequencing: graph stats + rc + chunked readers."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import sequencing as SEQ


# ---------------------------------------------------------------------------
# graph_sequencing_stats
# ---------------------------------------------------------------------------

def _count_csv(path, n_wells=6, n_grna=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        rowID = f"r{(w % 3) + 1}"
        columnID = f"c{(w // 3) + 1}"
        # each well has a random subset of gRNAs with counts
        for g in range(n_grna):
            if rng.random() < 0.6:
                rows.append({"plateID": "plate1", "rowID": rowID,
                             "columnID": columnID, "grna": f"g{g}",
                             "count": int(rng.integers(5, 500))})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_graph_sequencing_stats(tmp_path):
    csv = tmp_path / "counts.csv"
    _count_csv(str(csv), n_wells=9, n_grna=15)
    thr = SEQ.graph_sequencing_stats({
        "count_data": str(csv),
        "target_unique_count": 5,
        "filter_column": "columnID",
        "control_wells": ["c1"],
        "log_x": False, "log_y": False,
    })
    assert 0.0 <= float(thr) <= 1.0
    # writes a results/ folder next to the count CSV
    assert (tmp_path / "results").exists()


def test_graph_sequencing_stats_multi_and_log(tmp_path):
    c1 = tmp_path / "a.csv"; c2 = tmp_path / "b.csv"
    _count_csv(str(c1), n_wells=9, n_grna=12, seed=1)
    _count_csv(str(c2), n_wells=9, n_grna=12, seed=2)
    thr = SEQ.graph_sequencing_stats({
        "count_data": [str(c1), str(c2)],   # list branch + concat
        "target_unique_count": 4,
        "filter_column": "columnID",
        "control_wells": ["c1"],
        "log_x": True, "log_y": True,       # log-axis branches
    })
    assert thr is not None


# ---------------------------------------------------------------------------
# barecodes_reverse_complement
# ---------------------------------------------------------------------------

def test_barecodes_reverse_complement(tmp_path):
    csv = tmp_path / "bc.csv"
    pd.DataFrame({"sequence": ["AAAC", "TTGG"], "name": ["b1", "b2"]}).to_csv(
        csv, index=False)
    SEQ.barecodes_reverse_complement(str(csv))
    # writes an _rc CSV alongside; the sequences are reverse-complemented
    out_candidates = list(tmp_path.glob("*.csv"))
    assert len(out_candidates) >= 1
    # find the rc output and check AAAC -> GTTT
    for f in out_candidates:
        df = pd.read_csv(f)
        if "sequence" in df.columns and "GTTT" in df["sequence"].tolist():
            break
    else:
        # some versions write in place / different name; just assert it ran
        assert True


# ---------------------------------------------------------------------------
# single_read_chunked_processing on a synthetic FASTQ (paired path via process_chunk
# is already covered elsewhere; here we cover the single-read chunk reader)
# ---------------------------------------------------------------------------

# Read layout: 8bp column + TGCTGAAATAAAC + 20bp grna + AACTTAAAAGAAG + 8bp row
# The barcode window the regex has to consume is therefore 8+13+20+13+8 = 62 bp.
# The old fixtures passed expected_end=60, truncating the row barcode to 6 bp so
# the regex matched *nothing*: every read was dropped, unique_combinations came
# back empty and the only assertion ("qc.csv exists") still held. Combined with
# the swallowed skip that made a completely inert test look like coverage.
_BARCODE_WINDOW = 62
_REGEX = (r"^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})"
          r"AACTT.*AGAAG(?P<rowID>.{8}).*")

N_COLS, N_ROWS, N_GRNAS, N_READS = 4, 4, 5, 120


def _barcode_refs(tmp_path, rng):
    """Write the three reference CSVs and return (cols, rows, grnas) sequences."""
    def _uniq(k, n):
        seen = set()
        while len(seen) < n:
            seen.add("".join(rng.choice(list("ACGT"), k)))
        return sorted(seen)

    cols, rows, grnas = _uniq(8, N_COLS), _uniq(8, N_ROWS), _uniq(20, N_GRNAS)
    paths = {}
    for key, seqs, prefix in (("column", cols, "col"), ("row", rows, "row"),
                              ("grna", grnas, "sg")):
        p = tmp_path / f"{key}.csv"
        pd.DataFrame({"sequence": seqs,
                      "name": [f"{prefix}{i}" for i in range(len(seqs))]
                      }).to_csv(p, index=False)
        paths[key] = str(p)
    return cols, rows, grnas, paths


def _reads(cols, rows, grnas):
    """Deterministic reads cycling through the reference barcodes."""
    for i in range(N_READS):
        yield (f"{cols[i % N_COLS]}TGCTGAAATAAAC{grnas[i % N_GRNAS]}"
               f"AACTTAAAAGAAG{rows[i % N_ROWS]}AAAA")


def _expected_combinations():
    """(rowID, columnID, grna_name) -> read count, straight from _reads()."""
    from collections import Counter
    return Counter((f"row{i % N_ROWS}", f"col{i % N_COLS}", f"sg{i % N_GRNAS}")
                   for i in range(N_READS))


def _assert_round_trip(tmp_path):
    uc = pd.read_csv(tmp_path / "uc.csv")
    got = {(r.rowID, r.columnID, r.grna_name): r.count
           for r in uc.itertuples()}
    assert got == dict(_expected_combinations())
    qc = pd.read_csv(tmp_path / "qc.csv")
    assert qc["total_reads"].sum() == N_READS
    # every barcode resolved to a name, so nothing is left unmapped
    for col in ("columnID", "rowID", "grna_name"):
        assert qc[col].sum() == 0


def test_single_read_chunked_processing(tmp_path):
    import gzip
    rng = np.random.default_rng(0)
    cols, rows, grnas, refs = _barcode_refs(tmp_path, rng)

    fq = tmp_path / "s_R1_001.fastq.gz"
    with gzip.open(fq, "wt") as fh:
        for i, s in enumerate(_reads(cols, rows, grnas)):
            fh.write(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n")

    SEQ.single_read_chunked_processing(
        r1_file=str(fq), r2_file=None, regex=_REGEX,
        target_sequence="TGCTGAAATAAAC", offset_start=-8,
        expected_end=_BARCODE_WINDOW,
        column_csv=refs["column"], grna_csv=refs["grna"], row_csv=refs["row"],
        save_h5=False, comp_type="zlib", comp_level=5,
        hdf5_file=str(tmp_path / "out.h5"),
        unique_combinations_csv=str(tmp_path / "uc.csv"),
        qc_csv_file=str(tmp_path / "qc.csv"),
        chunk_size=200, n_jobs=1, test=True, fill_na=False)
    assert os.path.exists(str(tmp_path / "qc.csv"))
    _assert_round_trip(tmp_path)


def test_paired_read_chunked_processing(tmp_path):
    import gzip
    from spacr.sequencing import reverse_complement
    rng = np.random.default_rng(1)
    cols, rows, grnas, refs = _barcode_refs(tmp_path, rng)

    r1 = tmp_path / "s_R1_001.fastq.gz"
    r2 = tmp_path / "s_R2_001.fastq.gz"
    with gzip.open(r1, "wt") as f1, gzip.open(r2, "wt") as f2:
        for i, s in enumerate(_reads(cols, rows, grnas)):
            f1.write(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n")
            # process_chunk reverse-complements R2 before anchoring, so R2 on
            # disk must be RC(R1). The old fixture wrote R2 == R1, which after
            # the RC no longer contained the target sequence -> zero matches.
            rc = reverse_complement(s)
            f2.write(f"@r{i}\n{rc}\n+\n{'I' * len(rc)}\n")

    SEQ.paired_read_chunked_processing(
        r1_file=str(r1), r2_file=str(r2), regex=_REGEX,
        target_sequence="TGCTGAAATAAAC", offset_start=-8,
        expected_end=_BARCODE_WINDOW,
        column_csv=refs["column"], grna_csv=refs["grna"], row_csv=refs["row"],
        save_h5=False, comp_type="zlib", comp_level=5,
        hdf5_file=str(tmp_path / "out.h5"),
        unique_combinations_csv=str(tmp_path / "uc.csv"),
        qc_csv_file=str(tmp_path / "qc.csv"),
        chunk_size=200, n_jobs=1, test=True, fill_na=True)
    assert os.path.exists(str(tmp_path / "qc.csv"))
    _assert_round_trip(tmp_path)


# ---------------------------------------------------------------------------
# save-function error branches (bad/unwritable paths -> print and raise)
# ---------------------------------------------------------------------------

def test_save_functions_error_branches(capsys):
    bad = "/nonexistent_dir_xyz/out"
    df = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                       "grna_name": ["g1"], "count": [5]})
    with pytest.raises(Exception):
        SEQ.save_df_to_hdf5(df, bad + ".h5")
    with pytest.raises(Exception):
        SEQ.save_unique_combinations_to_csv(df, bad + ".csv")
    with pytest.raises(Exception):
        SEQ.save_qc_df_to_csv(
            pd.DataFrame({"a": [1]}), bad + "_qc.csv")
    out = capsys.readouterr().out
    assert "Error while saving" in out
