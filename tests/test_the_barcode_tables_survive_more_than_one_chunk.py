"""Map barcodes writes both its tables once per chunk. Both were wrong.

A run appends: every chunk of reads calls ``save_unique_combinations_to_csv``
and ``save_qc_df_to_csv``, each of which re-reads what is on disk, combines,
and writes the whole file back. Both combines were broken, and neither
showed up on a single chunk -- which is what the existing tests in
``test_coverage_fill_sequencing.py`` and ``test_coverage_fill_sequencing2.py``
exercise, and why they passed.

* The count table was written with ``index=True``. The frame comes out of a
  ``groupby(as_index=False)``, so that index is a bare RangeIndex; written
  out, the next chunk read it back as a data column called ``Unnamed: 0``,
  summed it with the counts, and wrote a fresh index beside it. One junk
  column per chunk. The synthetic 4,800-read demo is five chunks and its
  count table came out with five of them; a real run is hundreds.

* The QC table is ONE row of totals labelled ``NaN_Counts``, written with
  ``index=False`` -- so the copy read back from disk is labelled ``0``, and
  ``DataFrame.add`` aligned those two labels to nothing and returned their
  union. The file gained a row per chunk instead of accumulating. Five
  chunks, five partial rows, no total. The old tests missed this by building
  their QC fixture with a default RangeIndex, which is the one case where
  the alignment happens to work; this file builds it the way
  ``map_barcodes_to_annotate_reads`` really does.

Both are checked here over MORE THAN ONE chunk, because one chunk cannot
fail either way.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import sequencing as SEQ


CHUNKS = 5


def _counts(count):
    """One chunk's worth of counts, the shape the pipeline's groupby makes."""
    frame = pd.DataFrame({
        "rowID": ["r1", "r1", "r2"],
        "columnID": ["c1", "c2", "c1"],
        "grna_name": ["gRNA_0001", "gRNA_0002", "gRNA_0001"],
    })
    frame["count"] = count
    return frame


def _qc(total_reads, missing=0):
    """One chunk's QC row, built the way the pipeline builds it.

    Load-bearing: the index label. ``map_barcodes_to_annotate_reads`` writes
    ``qc_df.index = ["NaN_Counts"]``, and that label is what the alignment
    bug turned on.
    """
    frame = pd.DataFrame({"columnID": [missing], "rowID": [missing],
                          "grna_name": [missing]})
    frame.index = ["NaN_Counts"]
    frame["total_reads"] = total_reads
    return frame


def test_the_count_table_gains_no_columns_however_many_chunks(tmp_path):
    csv = tmp_path / "unique_combinations.csv"
    for _ in range(CHUNKS):
        SEQ.save_unique_combinations_to_csv(_counts(2), str(csv))

    out = pd.read_csv(csv)
    junk = [c for c in out.columns if str(c).startswith("Unnamed")]
    assert not junk, f"{len(junk)} junk column(s) after {CHUNKS} chunks: {junk}"
    assert list(out.columns) == ["rowID", "columnID", "grna_name", "count"]


def test_the_count_table_sums_the_chunks(tmp_path):
    """The counts still have to be right, not merely tidy."""
    csv = tmp_path / "unique_combinations.csv"
    for _ in range(CHUNKS):
        SEQ.save_unique_combinations_to_csv(_counts(2), str(csv))

    out = pd.read_csv(csv)
    assert len(out) == 3
    assert out["count"].sum() == 2 * 3 * CHUNKS
    one = out[(out["rowID"] == "r1") & (out["columnID"] == "c1")]
    assert one["count"].iloc[0] == 2 * CHUNKS


def test_the_qc_table_accumulates_into_one_row(tmp_path):
    csv = tmp_path / "qc.csv"
    for _ in range(CHUNKS):
        SEQ.save_qc_df_to_csv(_qc(1000, missing=2), str(csv))

    out = pd.read_csv(csv)
    assert len(out) == 1, (
        f"{len(out)} QC rows after {CHUNKS} chunks; the file is meant to hold "
        f"the run's totals, not one row per chunk")
    assert out["total_reads"].iloc[0] == 1000 * CHUNKS
    assert out["columnID"].iloc[0] == 2 * CHUNKS


def test_a_qc_row_that_reports_no_losses_still_reports_the_reads(tmp_path):
    """The all-zero case is the normal one and must not be mistaken for empty."""
    csv = tmp_path / "qc.csv"
    for _ in range(CHUNKS):
        SEQ.save_qc_df_to_csv(_qc(960, missing=0), str(csv))

    out = pd.read_csv(csv)
    assert len(out) == 1
    assert out["total_reads"].iloc[0] == 960 * CHUNKS
    assert (out.drop(columns=["total_reads"]).iloc[0] == 0).all()


@pytest.mark.parametrize("writer,frame", [
    ("save_unique_combinations_to_csv", _counts(1)),
    ("save_qc_df_to_csv", _qc(10)),
])
def test_the_first_chunk_writes_the_file(tmp_path, writer, frame):
    """Neither writer may need the file to exist already."""
    csv = tmp_path / "out.csv"
    getattr(SEQ, writer)(frame, str(csv))
    assert csv.exists()
    out = pd.read_csv(csv)
    assert len(out) == len(frame)
    assert not [c for c in out.columns if str(c).startswith("Unnamed")]
