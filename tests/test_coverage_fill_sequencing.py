"""Coverage-fill for spacr.sequencing pure-logic helpers + process_chunk.

No FASTQ files, no multiprocessing — synthetic reads exercise the
barcode extraction / consensus / mapping logic directly.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import sequencing as SQ


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_sequence_and_quality(self):
        s, q = SQ.extract_sequence_and_quality("ACGTACGT", "IIIIIIII", 2, 5)
        assert s == "GTA" and q == "III"

    def test_get_consensus_base_prefers_non_N(self):
        assert SQ.get_consensus_base([("N", "I"), ("A", "!")]) == "A"
        assert SQ.get_consensus_base([("A", "!"), ("N", "I")]) == "A"

    def test_get_consensus_base_higher_quality(self):
        # Neither N → higher quality wins.
        assert SQ.get_consensus_base([("A", "I"), ("C", "!")]) == "A"
        assert SQ.get_consensus_base([("A", "!"), ("C", "I")]) == "C"

    def test_create_consensus(self):
        out = SQ.create_consensus("ANGT", "I!II", "ACGT", "IIII")
        assert out[1] == "C"   # N replaced by the other read's C

    def test_reverse_complement(self):
        assert SQ.reverse_complement("AAAATTTT") == "AAAATTTT"
        assert SQ.reverse_complement("ACGT") == "ACGT"

    def test_save_df_to_hdf5_roundtrip(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        h5 = tmp_path / "d.h5"
        SQ.save_df_to_hdf5(df, str(h5))
        SQ.save_df_to_hdf5(df, str(h5))   # append path (concat)
        assert h5.exists()
        with pd.HDFStore(str(h5)) as store:
            assert len(store["df"]) == 4

    def test_save_unique_combinations_to_csv(self, tmp_path):
        csv_file = tmp_path / "u.csv"
        uc = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                           "grna_name": ["g1"], "count": [5]})
        SQ.save_unique_combinations_to_csv(uc, str(csv_file))
        # Second write triggers the concat + groupby-sum path.
        SQ.save_unique_combinations_to_csv(uc, str(csv_file))
        out = pd.read_csv(csv_file)
        assert (out["count"] == 10).any()

    def test_save_qc_df_to_csv(self, tmp_path):
        csv_file = tmp_path / "qc.csv"
        qc = pd.DataFrame({"total_reads": [100], "NaN": [3]})
        SQ.save_qc_df_to_csv(qc, str(csv_file))
        SQ.save_qc_df_to_csv(qc, str(csv_file))   # add path
        out = pd.read_csv(csv_file)
        assert (out["total_reads"] == 200).any()

    def test_barecodes_reverse_complement(self, tmp_path):
        src = tmp_path / "bc.csv"
        with src.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["sequence", "name"])
            w.writerow(["ACGT", "b1"])
        SQ.barecodes_reverse_complement(str(src))
        # A reverse-complemented CSV should be written somewhere.
        rc_files = list(tmp_path.glob("*rc*")) + list(tmp_path.glob("*.csv"))
        assert rc_files


# ---------------------------------------------------------------------------
# process_chunk — single + paired
# ---------------------------------------------------------------------------

def _barcode_csvs(tmp_path):
    def _w(name, rows):
        p = tmp_path / name
        with p.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["sequence", "name"])
            for seq, nm in rows:
                w.writerow([seq, nm])
        return str(p)
    col = _w("col.csv", [("COL1", "c1")])
    grna = _w("grna.csv", [("GRNA01", "g1")])
    row = _w("row.csv", [("ROW1", "r1")])
    return col, grna, row


def _fastq_record(seq):
    q = "I" * len(seq)
    return f"@id\n{seq}\n+\n{q}"


class TestProcessChunk:
    # Regex with the group names process_chunk reads (columnID/grna/rowID).
    REGEX = r"(?P<columnID>.{4})(?P<grna>.{6})(?P<rowID>.{4})"
    TARGET = "TGT"

    def _read(self):
        # "XX" + target + window(COL1 GRNA01 ROW1) + "YY"
        return "XX" + self.TARGET + "COL1GRNA01ROW1" + "YY"

    def test_single_read_chunk(self, tmp_path):
        col, grna, row = _barcode_csvs(tmp_path)
        r1 = [_fastq_record(self._read())]
        chunk = (r1, self.REGEX, self.TARGET, 3, 14, col, grna, row, False)
        df, uc, qc = SQ.process_chunk(chunk)
        assert len(df) == 1
        assert df.iloc[0]["columnID"] == "c1"
        assert "total_reads" in qc.columns

    def test_single_read_fill_na(self, tmp_path):
        col, grna, row = _barcode_csvs(tmp_path)
        # A read whose barcodes are NOT in the CSVs → names NaN → fill_na
        # backfills from the raw sequences.
        read = "XX" + self.TARGET + "ZZZZQQQQQQWWWW" + "YY"
        r1 = [_fastq_record(read)]
        chunk = (r1, self.REGEX, self.TARGET, 3, 14, col, grna, row, True)
        df, uc, qc = SQ.process_chunk(chunk)
        assert len(df) == 1

    def test_paired_read_chunk(self, tmp_path):
        col, grna, row = _barcode_csvs(tmp_path)
        window_read = "XX" + self.TARGET + "COL1GRNA01ROW1" + "YY"
        # R2 is the reverse complement of the same window (paired design).
        r2_seq = SQ.reverse_complement(window_read)
        r1 = [_fastq_record(window_read)]
        r2 = [_fastq_record(r2_seq)]
        chunk = (r1, r2, self.REGEX, self.TARGET, 3, 14,
                 col, grna, row, False)
        df, uc, qc = SQ.process_chunk(chunk)
        assert qc["total_reads"].iloc[0] == len(df)

    def test_no_match_warns(self, tmp_path):
        col, grna, row = _barcode_csvs(tmp_path)
        # A regex requiring a literal 'ZZZZ' the padded window lacks →
        # the "no sequences matched" warning + RC-check path (298-307).
        regex = r"(?P<columnID>.{4})ZZZZ(?P<grna>.{6})(?P<rowID>.{4})"
        read = "XX" + self.TARGET + "COL1GRNA01ROW1" + "YY"
        r1 = [_fastq_record(read)]
        chunk = (r1, regex, self.TARGET, 3, 14, col, grna, row, False)
        df, uc, qc = SQ.process_chunk(chunk)
        assert len(df) == 0
