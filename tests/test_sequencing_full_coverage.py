"""Every remaining branch of spacr.sequencing, pinned by behaviour.

F17. The rule these tests are written to: a test that runs a line without
asserting what it produced protects nothing. Each one here names the wrong
answer it would catch — a barcode silently mapped to the wrong well, a
truncated FASTQ pair that quietly halves a plate's counts, a writer process
that died leaving a half-written table the run reported as a success.
"""
from __future__ import annotations

import gzip
import importlib
import multiprocessing as mp
import os
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import sequencing as SEQ


# ---------------------------------------------------------------------------
# map_sequences_to_names — the lookup that turns reads into wells
# ---------------------------------------------------------------------------

def _write_csv(path, seqs, names):
    pd.DataFrame({"sequence": seqs, "name": names}).to_csv(path, index=False)
    return str(path)


def test_map_sequences_to_names_maps_and_leaves_unknowns_missing(tmp_path):
    csv = _write_csv(tmp_path / "bc.csv", ["ACGT", "TTGG"], ["b1", "b2"])
    got = SEQ.map_sequences_to_names(csv, ["TTGG", "ACGT", "GGGG"], rc=False)
    # Positional alignment is the whole contract: got[i] is the name of
    # sequences[i]. A shuffle here would relabel every read in the screen.
    assert got[0] == "b2"
    assert got[1] == "b1"
    assert got[2] is pd.NA


def test_map_sequences_to_names_rc_reverse_complements_the_csv_only(tmp_path):
    # ACGT -> ACGT is a palindrome under RC, so use something asymmetric.
    csv = _write_csv(tmp_path / "bc.csv", ["AAAC", "TTGN"], ["b1", "b2"])
    # rc=True complements the CSV column, so the *input* must be the RC form.
    got = SEQ.map_sequences_to_names(csv, ["GTTT", "NCAA", "AAAC"], rc=True)
    assert got[0] == "b1"          # RC('AAAC') == 'GTTT'
    assert got[1] == "b2"          # N maps to N, RC('TTGN') == 'NCAA'
    # And the un-complemented spelling no longer matches — proving the CSV
    # and not the query was transformed.
    assert got[2] is pd.NA


def test_map_sequences_to_names_rejects_a_csv_missing_its_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"seq": ["ACGT"], "label": ["b1"]}).to_csv(p, index=False)
    with pytest.raises(ValueError) as exc:
        SEQ.map_sequences_to_names(str(p), ["ACGT"], rc=False)
    msg = str(exc.value)
    # Both missing names are reported, sorted, so the user fixes the file once.
    assert "name" in msg and "sequence" in msg


def test_map_sequences_to_names_rejects_duplicate_sequences(tmp_path):
    csv = _write_csv(tmp_path / "dup.csv",
                     ["ACGT", "ACGT", "TTGG"], ["b1", "b2", "b3"])
    with pytest.raises(ValueError) as exc:
        SEQ.map_sequences_to_names(csv, ["ACGT"], rc=False)
    # One sequence resolving to two names would make the well assignment
    # depend on dict insertion order. The example is named so it can be found.
    assert "duplicate sequences" in str(exc.value)
    assert "ACGT" in str(exc.value)


def test_map_sequences_to_names_ignores_duplicate_nan_rows(tmp_path):
    # dropna() before the duplicate check: two blank rows are a sloppy CSV,
    # not an ambiguous barcode, and must not block the run.
    p = tmp_path / "nan.csv"
    pd.DataFrame({"sequence": [np.nan, np.nan, "ACGT"],
                  "name": ["x", "y", "b1"]}).to_csv(p, index=False)
    assert SEQ.map_sequences_to_names(str(p), ["ACGT"], rc=False) == ["b1"]


# ---------------------------------------------------------------------------
# consensus calling — where a wrong base becomes a wrong gRNA
# ---------------------------------------------------------------------------

def test_get_consensus_base_prefers_a_call_over_N_regardless_of_quality():
    # 'N' with the *better* quality still loses: an N is not a measurement.
    assert SEQ.get_consensus_base([("N", "I"), ("A", "!")]) == "A"
    assert SEQ.get_consensus_base([("A", "!"), ("N", "I")]) == "A"


def test_get_consensus_base_breaks_ties_toward_the_first_read():
    assert SEQ.get_consensus_base([("A", "I"), ("C", "I")]) == "A"
    assert SEQ.get_consensus_base([("A", "!"), ("C", "I")]) == "C"


def test_create_consensus_is_per_position():
    #      pos0: equal calls        -> 'A'
    #      pos1: 'N' loses to 'G'   -> 'G'   (even though 'N' has better qual)
    #      pos2: equal calls        -> 'C'
    #      pos3: 'G'@I vs 'T'@I tie -> 'G'   (first read wins a tie)
    assert SEQ.create_consensus("ANCG", "I!II", "AGCT", "!III") == "AGCG"
    # Same position, R2 now strictly better: the call flips.
    assert SEQ.create_consensus("ANCG", "I!I!", "AGCT", "!IiI") == "AGCT"


def test_extract_sequence_and_quality_slices_both_alike():
    assert SEQ.extract_sequence_and_quality("ACGTAC", "!!IIII", 2, 5) == (
        "GTA", "IIII"[0:3])


def test_reverse_complement_round_trips():
    assert SEQ.reverse_complement("AACGTT") == "AACGTT"
    assert SEQ.reverse_complement("AAAC") == "GTTT"


# ---------------------------------------------------------------------------
# process_chunk — the extraction itself
# ---------------------------------------------------------------------------

_REGEX = (r"^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})"
          r"AACTT.*AGAAG(?P<rowID>.{8}).*")
_PUBLIC_REGEX = (r"^(?P<column>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})"
                 r"AACTT.*AGAAG(?P<row>.{8}).*")
_WINDOW = 62
_TARGET = "TGCTGAAATAAAC"

_COL, _ROW, _GRNA = "AAAACCCC", "GGGGTTTT", "A" * 20


def _read(col=_COL, row=_ROW, grna=_GRNA, tail="AAAA"):
    return f"{col}TGCTGAAATAAAC{grna}AACTTAAAAGAAG{row}{tail}"


def _fastq(seq, name="r0", qual=None):
    qual = qual or "I" * len(seq)
    return f"@{name}\n{seq}\n+\n{qual}"


def _refs(tmp_path):
    return (_write_csv(tmp_path / "c.csv", [_COL], ["col0"]),
            _write_csv(tmp_path / "g.csv", [_GRNA], ["sg0"]),
            _write_csv(tmp_path / "r.csv", [_ROW], ["row0"]))


def test_process_chunk_single_end_decodes_the_barcodes(tmp_path):
    c, g, r = _refs(tmp_path)
    df, uc, qc = SEQ.process_chunk(
        ([_fastq(_read())], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))
    assert df["columnID"].tolist() == ["col0"]
    assert df["rowID"].tolist() == ["row0"]
    assert df["grna_name"].tolist() == ["sg0"]
    assert df["column_sequence"].tolist() == [_COL]
    assert df["row_sequence"].tolist() == [_ROW]
    assert uc.loc[0, "count"] == 1
    assert qc.loc["NaN_Counts", "total_reads"] == 1
    assert qc.loc["NaN_Counts", "columnID"] == 0


def test_process_chunk_accepts_the_public_column_row_group_names(tmp_path):
    # The shipped DEFAULT_BARCODE_REGEX uses columnID/rowID; the settings
    # panel offers column/row. Both must decode to the same table, or a user
    # who edited the regex loses every read without an error.
    c, g, r = _refs(tmp_path)
    df, _, _ = SEQ.process_chunk(
        ([_fastq(_read())], _PUBLIC_REGEX, _TARGET, -8, _WINDOW,
         c, g, r, False))
    assert df["columnID"].tolist() == ["col0"]
    assert df["rowID"].tolist() == ["row0"]


@pytest.mark.parametrize("regex,missing", [
    (r"^(?P<grna>.{20,21})(?P<row>.{8})", "column/columnID"),
    (r"^(?P<column>.{8})(?P<grna>.{20,21})", "row/rowID"),
    (r"^(?P<column>.{8})(?P<row>.{8})", "grna"),
])
def test_process_chunk_names_the_regex_group_that_is_missing(
        tmp_path, regex, missing):
    c, g, r = _refs(tmp_path)
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(([_fastq(_read())], regex, _TARGET, -8, _WINDOW,
                           c, g, r, False))
    assert missing in str(exc.value)


def test_process_chunk_reports_all_missing_groups_at_once(tmp_path):
    c, g, r = _refs(tmp_path)
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(([_fastq(_read())], r"^(?P<nothing>.{8})", _TARGET,
                           -8, _WINDOW, c, g, r, False))
    msg = str(exc.value)
    assert "column/columnID" in msg and "row/rowID" in msg and "grna" in msg


@pytest.mark.parametrize("n", [0, 8, 11])
def test_process_chunk_rejects_a_wrong_sized_tuple(n):
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(tuple(range(n)))
    assert "9 values" in str(exc.value) and "10 values" in str(exc.value)


def test_process_chunk_rejects_a_non_sequence_payload():
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(object())
    assert "an unknown count" in str(exc.value)


def test_process_chunk_rejects_a_non_positive_window(tmp_path):
    c, g, r = _refs(tmp_path)
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(([_fastq(_read())], _REGEX, _TARGET, -8, 0,
                           c, g, r, False))
    assert "expected_end must be a positive integer" in str(exc.value)


@pytest.mark.parametrize("record,fragment", [
    ("@r0\nACGT\n+\nIIII\nextra", "exactly four lines"),
    ("r0\nACGT\n+\nIIII", "not a valid FASTQ record"),
    ("@r0\nACGT\n-\nIIII", "not a valid FASTQ record"),
    ("@r0\nACGT\n+\nIII", "lengths differ"),
])
def test_process_chunk_rejects_malformed_fastq_records(
        tmp_path, record, fragment):
    c, g, r = _refs(tmp_path)
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(([record], _REGEX, _TARGET, -8, _WINDOW,
                           c, g, r, False))
    assert fragment in str(exc.value)
    # The label says which read, so a 40M-read file can be repaired.
    assert "R1 record 1" in str(exc.value)


def test_process_chunk_paired_labels_the_R2_record_that_is_broken(tmp_path):
    c, g, r = _refs(tmp_path)
    good = _fastq(_read())
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(([good], ["@r0\nACGT\n+\nIII"], _REGEX, _TARGET,
                           -8, _WINDOW, c, g, r, False))
    assert "R2 record 1" in str(exc.value)


def test_process_chunk_paired_rejects_unequal_chunk_lengths(tmp_path):
    c, g, r = _refs(tmp_path)
    with pytest.raises(ValueError) as exc:
        SEQ.process_chunk(([_fastq(_read())], [], _REGEX, _TARGET, -8,
                           _WINDOW, c, g, r, False))
    assert "R1=1, R2=0" in str(exc.value)


def test_process_chunk_paired_builds_a_consensus_from_both_mates(tmp_path):
    """R2 is RC'd before anchoring; a low-quality R1 base is rescued by R2."""
    c, g, r = _refs(tmp_path)
    seq = _read()
    r1_qual = list("I" * len(seq))
    # Corrupt one base of R1's column barcode and give it the worst quality.
    bad = seq[:0] + ("T" if seq[0] != "T" else "G") + seq[1:]
    r1_qual[0] = "!"
    rc = SEQ.reverse_complement(seq)
    df, _, _ = SEQ.process_chunk(
        ([_fastq(bad, qual="".join(r1_qual))],
         [_fastq(rc)], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))
    # R2's high-quality call wins, so the column still resolves.
    assert df["columnID"].tolist() == ["col0"]
    assert df["column_sequence"].tolist() == [_COL]


def test_process_chunk_pads_a_truncated_window_with_N(tmp_path):
    """A read that ends inside the barcode is padded, not silently shortened."""
    c, g, r = _refs(tmp_path)
    # Drop the tail and four bases of the row barcode.
    short = _read(tail="")[:-4]
    df, _, _ = SEQ.process_chunk(
        ([_fastq(short)], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))
    # It still matches (the regex takes 8 chars for row), but the row barcode
    # now carries the padding and therefore does NOT map to a name.
    assert df["row_sequence"].tolist() == [_ROW[:4] + "NNNN"]
    assert df["rowID"].isna().all()


def test_process_chunk_paired_pads_both_mates(tmp_path, capsys):
    c, g, r = _refs(tmp_path)
    short = _read(tail="")[:-4]
    rc = SEQ.reverse_complement(short)
    df, _, _ = SEQ.process_chunk(
        ([_fastq(short)], [_fastq(rc)], _REGEX, _TARGET, -8, _WINDOW,
         c, g, r, False))
    assert df["row_sequence"].tolist() == [_ROW[:4] + "NNNN"]


# A read that is exactly the reverse complement of the 62 bp barcode window:
# the anchor is still found, the window is still extracted, but the regex
# cannot match it — and matches its reverse complement perfectly. This is what
# a run whose barcode CSVs were entered in the wrong orientation looks like.
_RC_WINDOW = SEQ.reverse_complement(_read(tail=""))
_RC_ANCHOR = _RC_WINDOW[:13]


def test_process_chunk_warns_when_nothing_matched_and_offers_the_rc(
        tmp_path, capsys):
    """The single commonest sequencing-run mistake: barcodes entered as RC."""
    c, g, r = _refs(tmp_path)
    df, _, _ = SEQ.process_chunk(
        ([_fastq(_RC_WINDOW)], _REGEX, _RC_ANCHOR, 0, _WINDOW,
         c, g, r, False))
    out = capsys.readouterr().out
    assert df.empty
    assert "No sequences matched" in out
    assert "correct orientation" in out
    # The actionable half: spaCR tried the other orientation for the user.
    assert "Reverse complement of last sequence in chunk matched" in out


def test_process_chunk_paired_warns_and_offers_the_rc(tmp_path, capsys):
    c, g, r = _refs(tmp_path)
    df, _, _ = SEQ.process_chunk(
        ([_fastq(_RC_WINDOW)], [_fastq(SEQ.reverse_complement(_RC_WINDOW))],
         _REGEX, _RC_ANCHOR, 0, _WINDOW, c, g, r, False))
    out = capsys.readouterr().out
    assert df.empty
    assert "Reverse complement of last sequence in chunk matched" in out


def test_process_chunk_warns_when_the_anchor_is_never_found(tmp_path, capsys):
    """consensus_seq stays None — the warning must still print, not crash."""
    c, g, r = _refs(tmp_path)
    df, uc, qc = SEQ.process_chunk(
        ([_fastq(_read())], _REGEX, "GGGGGGGGGGGGGGGG", -8, _WINDOW,
         c, g, r, False))
    out = capsys.readouterr().out
    assert df.empty and uc.empty
    assert "Is None compatible with" in out
    assert qc.loc["NaN_Counts", "total_reads"] == 0


def test_process_chunk_paired_warns_when_the_anchor_is_never_found(
        tmp_path, capsys):
    c, g, r = _refs(tmp_path)
    seq = _read()
    df, _, _ = SEQ.process_chunk(
        ([_fastq(seq)], [_fastq(SEQ.reverse_complement(seq))], _REGEX,
         "GGGGGGGGGGGGGGGG", -8, _WINDOW, c, g, r, False))
    assert df.empty
    assert "Is None compatible with" in capsys.readouterr().out


def test_process_chunk_fill_na_substitutes_the_raw_barcode(tmp_path):
    """Unmapped barcodes become their own key rather than vanishing.

    Without fill_na the groupby drops NaN keys, so an unmapped gRNA silently
    disappears from the count table.
    """
    c, g, r = _refs(tmp_path)
    unknown = "C" * 20
    read = _read(grna=unknown)
    dropped, uc_dropped, _ = SEQ.process_chunk(
        ([_fastq(read)], _REGEX, _TARGET, -8, _WINDOW, c, g, r, False))
    assert uc_dropped.empty          # NaN grna_name -> row is dropped
    kept, uc_kept, qc = SEQ.process_chunk(
        ([_fastq(read)], _REGEX, _TARGET, -8, _WINDOW, c, g, r, True))
    assert uc_kept["grna_name"].tolist() == [unknown]
    assert uc_kept["count"].tolist() == [1]
    # The reads DataFrame itself is NOT rewritten — fill_na only affects the
    # aggregate, so the QC row still reports the unmapped read.
    assert kept["grna_name"].isna().all()
    assert qc.loc["NaN_Counts", "grna_name"] == 1


# ---------------------------------------------------------------------------
# worker/chunk-size validation
# ---------------------------------------------------------------------------

def test_chunk_worker_count_defaults_to_all_but_three_cpus(monkeypatch):
    monkeypatch.setattr(SEQ, "cpu_count", lambda: 16)
    assert SEQ._chunk_worker_count(None) == 13


def test_chunk_worker_count_never_returns_zero_on_a_small_box(monkeypatch):
    monkeypatch.setattr(SEQ, "cpu_count", lambda: 2)
    assert SEQ._chunk_worker_count(None) == 1


def test_chunk_worker_count_honours_an_explicit_value():
    assert SEQ._chunk_worker_count(4) == 4
    assert SEQ._chunk_worker_count("4") == 4


@pytest.mark.parametrize("bad", [0, -1])
def test_chunk_worker_count_rejects_a_non_positive_value(bad):
    with pytest.raises(ValueError) as exc:
        SEQ._chunk_worker_count(bad)
    assert "at least 1" in str(exc.value)


def test_validate_chunk_size_coerces_and_rejects():
    assert SEQ._validate_chunk_size("500") == 500
    with pytest.raises(ValueError) as exc:
        SEQ._validate_chunk_size(0)
    assert "chunk_size must be at least 1" in str(exc.value)


# ---------------------------------------------------------------------------
# the saver process and its lifecycle
# ---------------------------------------------------------------------------

class _FakeQueue:
    def __init__(self, items):
        self._items = list(items)
        self.put_items = []

    def get(self):
        return self._items.pop(0)

    def put(self, item):
        self.put_items.append(item)


def test_saver_process_writes_h5_and_csvs_then_stops(tmp_path):
    df = pd.DataFrame({"read": ["ACGT"], "columnID": ["col0"],
                       "rowID": ["row0"], "grna_name": ["sg0"]})
    uc = pd.DataFrame({"rowID": ["row0"], "columnID": ["col0"],
                       "grna_name": ["sg0"], "count": [3]})
    qc = pd.DataFrame({"total_reads": [3]})
    h5 = tmp_path / "reads.h5"
    ucsv = tmp_path / "uc.csv"
    qcsv = tmp_path / "qc.csv"
    q = _FakeQueue([(df, uc, qc), (df, uc, qc), "STOP"])
    SEQ.saver_process(q, str(h5), True, str(ucsv), str(qcsv), "zlib", 5)

    assert pd.read_hdf(h5, "df").shape[0] == 2          # appended, not replaced
    # Counts SUM across chunks — the whole reason the writer re-reads the CSV.
    assert pd.read_csv(ucsv)["count"].tolist() == [6]
    assert pd.read_csv(qcsv)["total_reads"].tolist() == [6]


def test_saver_process_skips_the_h5_when_save_h5_is_off(tmp_path):
    df = pd.DataFrame({"read": ["ACGT"]})
    uc = pd.DataFrame({"rowID": ["r"], "columnID": ["c"],
                       "grna_name": ["g"], "count": [1]})
    qc = pd.DataFrame({"total_reads": [1]})
    h5 = tmp_path / "reads.h5"
    q = _FakeQueue([(df, uc, qc), "STOP"])
    SEQ.saver_process(q, str(h5), False, str(tmp_path / "uc.csv"),
                      str(tmp_path / "qc.csv"), "zlib", 5)
    assert not h5.exists()


def _sleep_forever(seconds=300):
    import time as _t
    _t.sleep(seconds)


def _exit_nonzero():
    raise SystemExit(3)


def _return_none():
    """Pickle-safe successful process target (Python 3.14 uses forkserver)."""
    return None


def test_finish_saver_is_a_no_op_for_a_writer_that_stops():
    q = _FakeQueue([])
    proc = mp.Process(target=_return_none)
    proc.start()
    SEQ._finish_saver(q, proc, timeout=30)
    assert q.put_items == ["STOP"]
    assert proc.exitcode == 0


@pytest.mark.integration
def test_finish_saver_terminates_and_fails_a_hung_writer():
    q = _FakeQueue([])
    proc = mp.Process(target=_sleep_forever)
    proc.start()
    try:
        with pytest.raises(RuntimeError) as exc:
            SEQ._finish_saver(q, proc, timeout=1)
        assert "did not stop within 1 seconds" in str(exc.value)
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()


@pytest.mark.integration
def test_finish_saver_fails_loudly_when_the_writer_died():
    """The bug this guards: outputs half-written, run still exits 0."""
    q = _FakeQueue([])
    proc = mp.Process(target=_exit_nonzero)
    proc.start()
    with pytest.raises(RuntimeError) as exc:
        SEQ._finish_saver(q, proc, timeout=30)
    assert "exit code 3" in str(exc.value)
    assert "incomplete" in str(exc.value)


class _FakePool:
    def __init__(self):
        self.terminated = False
        self.joined = False

    def terminate(self):
        self.terminated = True

    def join(self):
        self.joined = True


class _FakeProc:
    def __init__(self, alive_answers):
        self._alive = list(alive_answers)
        self.joins = []
        self.terminated = False

    def is_alive(self):
        return self._alive.pop(0)

    def join(self, timeout=None):
        self.joins.append(timeout)

    def terminate(self):
        self.terminated = True


def test_abort_chunk_workers_stops_a_live_writer_politely_first():
    pool, q = _FakePool(), _FakeQueue([])
    proc = _FakeProc([True, False])
    SEQ._abort_chunk_workers(pool, q, proc)
    assert pool.terminated and pool.joined
    assert q.put_items == ["STOP"]     # asked nicely
    assert proc.joins == [10]
    assert not proc.terminated         # and it obeyed


def test_abort_chunk_workers_kills_a_writer_that_ignores_stop():
    pool, q = _FakePool(), _FakeQueue([])
    proc = _FakeProc([True, True])
    SEQ._abort_chunk_workers(pool, q, proc)
    assert proc.terminated
    assert proc.joins == [10, 5]


def test_abort_chunk_workers_leaves_a_dead_writer_alone():
    pool, q = _FakePool(), _FakeQueue([])
    proc = _FakeProc([False, False])
    SEQ._abort_chunk_workers(pool, q, proc)
    assert q.put_items == []
    assert not proc.terminated


# ---------------------------------------------------------------------------
# the chunked readers
# ---------------------------------------------------------------------------

def _write_fastq_gz(path, seqs):
    with gzip.open(path, "wt") as fh:
        for i, s in enumerate(seqs):
            fh.write(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n")
    return str(path)


def _run_kwargs(tmp_path, refs):
    c, g, r = refs
    return dict(regex=_REGEX, target_sequence=_TARGET, offset_start=-8,
                expected_end=_WINDOW, column_csv=c, grna_csv=g, row_csv=r,
                save_h5=False, comp_type="zlib", comp_level=5,
                hdf5_file=str(tmp_path / "out.h5"),
                unique_combinations_csv=str(tmp_path / "uc.csv"),
                qc_csv_file=str(tmp_path / "qc.csv"),
                chunk_size=2, n_jobs=1)


@pytest.mark.parametrize("missing", ["R1", "R2"])
def test_paired_reader_names_the_missing_fastq(tmp_path, missing):
    refs = _refs(tmp_path)
    real = _write_fastq_gz(tmp_path / "ok.fastq.gz", [_read()])
    kw = _run_kwargs(tmp_path, refs)
    r1 = str(tmp_path / "gone.gz") if missing == "R1" else real
    r2 = str(tmp_path / "gone.gz") if missing == "R2" else real
    with pytest.raises(FileNotFoundError) as exc:
        SEQ.paired_read_chunked_processing(r1_file=r1, r2_file=r2, **kw)
    assert f"{missing} FASTQ file does not exist" in str(exc.value)


def test_single_reader_names_a_missing_fastq(tmp_path):
    kw = _run_kwargs(tmp_path, _refs(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        SEQ.single_read_chunked_processing(
            r1_file=str(tmp_path / "gone.gz"), r2_file=None, **kw)
    assert "R1 FASTQ file does not exist" in str(exc.value)


def test_paired_reader_rejects_a_falsy_path(tmp_path):
    kw = _run_kwargs(tmp_path, _refs(tmp_path))
    with pytest.raises(FileNotFoundError):
        SEQ.paired_read_chunked_processing(r1_file="", r2_file="", **kw)


def test_single_reader_rejects_a_falsy_path(tmp_path):
    kw = _run_kwargs(tmp_path, _refs(tmp_path))
    with pytest.raises(FileNotFoundError):
        SEQ.single_read_chunked_processing(r1_file=None, r2_file=None, **kw)


@pytest.mark.integration
def test_paired_reader_refuses_truncated_pairs(tmp_path):
    """A short R2 used to silently halve the plate's counts."""
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "r1.fastq.gz", [_read()] * 4)
    rc = SEQ.reverse_complement(_read())
    r2 = _write_fastq_gz(tmp_path / "r2.fastq.gz", [rc] * 2)
    kw = _run_kwargs(tmp_path, refs)
    with pytest.raises(ValueError) as exc:
        SEQ.paired_read_chunked_processing(r1_file=r1, r2_file=r2, **kw)
    assert "different read counts" in str(exc.value)


@pytest.mark.integration
def test_paired_reader_refuses_a_truncated_R1_too(tmp_path):
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "r1.fastq.gz", [_read()] * 2)
    rc = SEQ.reverse_complement(_read())
    r2 = _write_fastq_gz(tmp_path / "r2.fastq.gz", [rc] * 4)
    kw = _run_kwargs(tmp_path, refs)
    with pytest.raises(ValueError):
        SEQ.paired_read_chunked_processing(r1_file=r1, r2_file=r2, **kw)


@pytest.mark.integration
def test_paired_reader_counts_every_read_over_several_chunks(tmp_path):
    """test=False: the real read-count + multi-chunk loop, not the preview."""
    refs = _refs(tmp_path)
    reads = [_read()] * 5
    r1 = _write_fastq_gz(tmp_path / "r1.fastq.gz", reads)
    r2 = _write_fastq_gz(
        tmp_path / "r2.fastq.gz",
        [SEQ.reverse_complement(s) for s in reads])
    kw = _run_kwargs(tmp_path, refs)
    SEQ.paired_read_chunked_processing(r1_file=r1, r2_file=r2, test=False,
                                       fill_na=False, **kw)
    # 5 reads / chunk_size 2 == 3 chunks, and the counts must SUM to 5.
    assert pd.read_csv(tmp_path / "uc.csv")["count"].sum() == 5
    assert pd.read_csv(tmp_path / "qc.csv")["total_reads"].sum() == 5


@pytest.mark.integration
def test_single_reader_counts_every_read_over_several_chunks(tmp_path):
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "r1.fastq.gz", [_read()] * 5)
    kw = _run_kwargs(tmp_path, refs)
    SEQ.single_read_chunked_processing(r1_file=r1, r2_file=None, test=False,
                                       fill_na=False, **kw)
    assert pd.read_csv(tmp_path / "uc.csv")["count"].sum() == 5
    assert pd.read_csv(tmp_path / "qc.csv")["total_reads"].sum() == 5


@pytest.mark.integration
def test_paired_reader_propagates_a_worker_failure(tmp_path):
    """A bad regex must abort the run, not leave a half-written table."""
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "r1.fastq.gz", [_read()])
    r2 = _write_fastq_gz(tmp_path / "r2.fastq.gz",
                         [SEQ.reverse_complement(_read())])
    kw = _run_kwargs(tmp_path, refs)
    kw["regex"] = r"^(?P<nothing>.{8})"
    with pytest.raises(ValueError) as exc:
        SEQ.paired_read_chunked_processing(r1_file=r1, r2_file=r2, test=True,
                                           **kw)
    assert "missing required named group" in str(exc.value)


@pytest.mark.integration
def test_single_reader_propagates_a_worker_failure(tmp_path):
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "r1.fastq.gz", [_read()])
    kw = _run_kwargs(tmp_path, refs)
    kw["regex"] = r"^(?P<nothing>.{8})"
    with pytest.raises(ValueError):
        SEQ.single_read_chunked_processing(r1_file=r1, r2_file=None,
                                           test=True, **kw)


@pytest.mark.integration
def test_single_reader_handles_an_empty_fastq(tmp_path):
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "empty.fastq.gz", [])
    kw = _run_kwargs(tmp_path, refs)
    SEQ.single_read_chunked_processing(r1_file=r1, r2_file=None, test=False,
                                       **kw)
    # Nothing to write: the loop breaks before a chunk is ever dispatched.
    assert not (tmp_path / "uc.csv").exists()


@pytest.mark.integration
def test_paired_reader_handles_an_empty_pair(tmp_path):
    refs = _refs(tmp_path)
    r1 = _write_fastq_gz(tmp_path / "e1.fastq.gz", [])
    r2 = _write_fastq_gz(tmp_path / "e2.fastq.gz", [])
    kw = _run_kwargs(tmp_path, refs)
    SEQ.paired_read_chunked_processing(r1_file=r1, r2_file=r2, test=False,
                                       **kw)
    assert not (tmp_path / "uc.csv").exists()


# ---------------------------------------------------------------------------
# _run_barcode_qc — opt-in, and never allowed to lose a finished run
# ---------------------------------------------------------------------------

def test_run_barcode_qc_is_off_by_default(tmp_path):
    assert SEQ._run_barcode_qc({}, str(tmp_path), "c.csv", "q.csv") is None


def test_run_barcode_qc_forwards_the_paths_and_prints_the_recommendation(
        tmp_path, monkeypatch, capsys):
    seen = {}

    def _fake_barcode_qc(cfg):
        seen.update(cfg)
        return {"recommendation": "threshold 0.01"}

    mod = types.ModuleType("spacr.sequencing_qc")
    mod.barcode_qc = _fake_barcode_qc
    monkeypatch.setitem(sys.modules, "spacr.sequencing_qc", mod)

    out = SEQ._run_barcode_qc(
        {"barcode_qc": True, "row_csv": "r.csv", "column_csv": "c.csv",
         "grna_csv": "g.csv", "target_grnas_per_well": 7},
        str(tmp_path), "counts.csv", "qc.csv")
    assert out == {"recommendation": "threshold 0.01"}
    assert seen["count_data"] == "counts.csv"
    assert seen["qc_data"] == "qc.csv"
    assert seen["target_grnas_per_well"] == 7
    assert seen["dst"] == os.path.join(str(tmp_path), "barcode_qc")
    assert "threshold 0.01" in capsys.readouterr().out


def test_run_barcode_qc_defaults_target_grnas_per_well_to_one(
        tmp_path, monkeypatch):
    seen = {}
    mod = types.ModuleType("spacr.sequencing_qc")
    mod.barcode_qc = lambda cfg: seen.update(cfg) or {}
    monkeypatch.setitem(sys.modules, "spacr.sequencing_qc", mod)
    SEQ._run_barcode_qc({"barcode_qc": True}, str(tmp_path), "c", "q")
    assert seen["target_grnas_per_well"] == 1


def test_run_barcode_qc_never_costs_the_run_its_counts(
        tmp_path, monkeypatch, capsys):
    mod = types.ModuleType("spacr.sequencing_qc")

    def _boom(cfg):
        raise RuntimeError("no display")

    mod.barcode_qc = _boom
    monkeypatch.setitem(sys.modules, "spacr.sequencing_qc", mod)
    assert SEQ._run_barcode_qc({"barcode_qc": True}, str(tmp_path), "c",
                               "q") is None
    out = capsys.readouterr().out
    assert "barcode QC failed" in out
    assert "were written and are unaffected" in out


# ---------------------------------------------------------------------------
# generate_barecode_mapping — the sample loop
# ---------------------------------------------------------------------------

class _Attempt:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Policy:
    def attempts_for(self, key, stage=None):
        yield _Attempt()


class _Run:
    policy = _Policy()


class _RunCtx:
    def __enter__(self):
        return _Run()

    def __exit__(self, *exc):
        return False


@pytest.fixture
def mapping_env(tmp_path, monkeypatch):
    """Stub out everything generate_barecode_mapping delegates to."""
    calls = []

    def _fake_paired(**kw):
        calls.append(("paired", kw))

    def _fake_single(**kw):
        calls.append(("single", kw))

    monkeypatch.setattr(SEQ, "paired_read_chunked_processing", _fake_paired)
    monkeypatch.setattr(SEQ, "single_read_chunked_processing", _fake_single)
    monkeypatch.setattr(SEQ, "run_context", lambda *a, **k: _RunCtx())
    monkeypatch.setattr(SEQ, "_run_barcode_qc",
                        lambda *a, **k: calls.append(("qc", a[1])))

    import spacr.io as IO
    import spacr.utils as U
    monkeypatch.setattr(U, "save_settings", lambda *a, **k: None)
    monkeypatch.setattr(
        IO, "parse_gz_files",
        lambda src: {"S1": {"R1": os.path.join(src, "S1_R1.fastq.gz"),
                            "R2": os.path.join(src, "S1_R2.fastq.gz")}})
    return calls


def test_generate_barecode_mapping_paired_dispatch(tmp_path, mapping_env):
    SEQ.generate_barecode_mapping({"src": str(tmp_path), "mode": "paired"})
    kinds = [c[0] for c in mapping_env]
    assert kinds == ["paired", "qc"]
    kw = mapping_env[0][1]
    assert kw["r1_file"].endswith("S1_R1.fastq.gz")
    assert kw["r2_file"].endswith("S1_R2.fastq.gz")
    # Output folder is named for the sample AND the mode.
    dst = os.path.join(str(tmp_path), "S1_paired")
    assert os.path.isdir(dst)
    assert kw["unique_combinations_csv"] == os.path.join(
        dst, "unique_combinations.csv")
    assert kw["hdf5_file"] == os.path.join(dst, "annotated_reads.h5")
    assert mapping_env[1][1] == dst


@pytest.mark.parametrize("direction,mate", [("R1", "R1"), ("R2", "R2")])
def test_generate_barecode_mapping_single_uses_the_named_mate(
        tmp_path, mapping_env, direction, mate):
    SEQ.generate_barecode_mapping({"src": str(tmp_path), "mode": "single",
                                   "single_direction": direction})
    kind, kw = mapping_env[0]
    assert kind == "single"
    # R2-only runs must read R2, not R1. Getting this backwards decodes the
    # wrong strand and yields an empty count table with no error.
    assert kw["r1_file"].endswith(f"S1_{mate}.fastq.gz")
    assert kw["r2_file"] is None
    assert os.path.isdir(os.path.join(str(tmp_path), f"S1_single_{direction}"))


def test_generate_barecode_mapping_accepts_no_settings_at_all(monkeypatch):
    """settings=None must be canonicalized, not dereferenced."""
    seen = {}
    monkeypatch.setattr(SEQ, "run_context", lambda *a, **k: _RunCtx())
    import spacr.io as IO
    import spacr.utils as U
    monkeypatch.setattr(U, "save_settings",
                        lambda s, **k: seen.update(s))
    monkeypatch.setattr(IO, "parse_gz_files", lambda src: {})
    SEQ.generate_barecode_mapping(None)
    assert seen["mode"] == "paired"
    assert seen["regex"]


def test_generate_barecode_mapping_skips_a_sample_with_no_reads(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(SEQ, "paired_read_chunked_processing",
                        lambda **kw: calls.append(kw))
    monkeypatch.setattr(SEQ, "run_context", lambda *a, **k: _RunCtx())
    import spacr.io as IO
    import spacr.utils as U
    monkeypatch.setattr(U, "save_settings", lambda *a, **k: None)
    monkeypatch.setattr(IO, "parse_gz_files",
                        lambda src: {"S1": {"R1": None, "R2": None}})
    SEQ.generate_barecode_mapping({"src": str(tmp_path), "mode": "paired"})
    assert calls == []
    assert not os.path.isdir(os.path.join(str(tmp_path), "S1_paired"))


# ---------------------------------------------------------------------------
# barecodes_reverse_complement
# ---------------------------------------------------------------------------

def test_barecodes_reverse_complement_writes_an_rc_sibling(tmp_path, capsys):
    csv = tmp_path / "bc.csv"
    pd.DataFrame({"sequence": ["AAAC", "TTGN"],
                  "name": ["b1", "b2"]}).to_csv(csv, index=False)
    SEQ.barecodes_reverse_complement(str(csv))
    out = tmp_path / "bc_RC.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert df["sequence"].tolist() == ["GTTT", "NCAA"]
    assert df["name"].tolist() == ["b1", "b2"]      # names are not shuffled
    assert "Reverse complement file saved as" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# graph_sequencing_stats
# ---------------------------------------------------------------------------

def _counts_frame(plate=True, n_grna=8):
    rows = []
    for w, (r, c) in enumerate([("r1", "c1"), ("r1", "c2"), ("r2", "c1")]):
        for g in range(n_grna):
            rows.append({"rowID": r, "columnID": c, "grna": f"g{g}",
                         "count": 10 * (g + 1)})
    df = pd.DataFrame(rows)
    if plate:
        df["plateID"] = "plate1"
    return df


def test_graph_sequencing_stats_invents_a_plate_id_when_absent(tmp_path):
    csv = tmp_path / "counts.csv"
    _counts_frame(plate=False).to_csv(csv, index=False)
    thr = SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 3,
        "filter_column": "columnID", "control_wells": [],
        "log_x": False, "log_y": False})
    assert 0.0 < float(thr) < 1.0


def test_graph_sequencing_stats_numbers_multiple_plates_in_order(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _counts_frame(plate=False).to_csv(a, index=False)
    _counts_frame(plate=False).to_csv(b, index=False)
    SEQ.graph_sequencing_stats({
        "count_data": [str(a), str(b)], "target_unique_count": 3,
        "filter_column": "columnID", "control_wells": [],
        "log_x": False, "log_y": False})
    # dst is derived from the FIRST count CSV, so both plates' output lands
    # in one folder rather than beside whichever file happened to be last.
    assert (tmp_path / "results" / "fraction_threshold.pdf").exists()


def test_graph_sequencing_stats_demands_the_well_key_columns(tmp_path):
    csv = tmp_path / "counts.csv"
    df = _counts_frame()
    df = df.drop(columns=["columnID"])
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError) as exc:
        SEQ.graph_sequencing_stats({
            "count_data": str(csv), "target_unique_count": 3,
            "filter_column": "rowID", "control_wells": [],
            "log_x": False, "log_y": False})
    assert "'plateID', 'rowID', and 'columnID'" in str(exc.value)


def test_graph_sequencing_stats_drops_control_wells_before_choosing(tmp_path):
    csv = tmp_path / "counts.csv"
    df = _counts_frame()
    # Give c2 a single dominant gRNA so including it would drag the mean down.
    df.loc[df["columnID"] == "c2", "count"] = 1
    df.loc[(df["columnID"] == "c2") & (df["grna"] == "g0"), "count"] = 10_000
    df.to_csv(csv, index=False)
    kept = SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 4,
        "filter_column": "columnID", "control_wells": ["c2"],
        "log_x": False, "log_y": False})
    dropped = SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 4,
        "filter_column": "columnID", "control_wells": [],
        "log_x": False, "log_y": False})
    assert kept != dropped


def test_graph_sequencing_stats_takes_the_row_after_the_last_separator(
        tmp_path):
    """rowID arrives as '<plate>_<row>' in some count CSVs; mixed is legal."""
    from spacr import schema
    csv = tmp_path / "counts.csv"
    df = _counts_frame()
    sep = schema.KEY_SEPARATOR
    df.loc[df["rowID"] == "r1", "rowID"] = f"exp1{sep}plate1{sep}r1"
    df.to_csv(csv, index=False)
    # Before the fix a plain 'r2' alongside a composite raised IndexError and
    # the caller lost the threshold it had already computed.
    thr = SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 3,
        "filter_column": "columnID", "control_wells": [],
        "log_x": False, "log_y": False})
    assert thr is not None


def test_graph_sequencing_stats_log_axes(tmp_path):
    csv = tmp_path / "counts.csv"
    _counts_frame().to_csv(csv, index=False)
    thr = SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 3,
        "filter_column": "columnID", "control_wells": [],
        "log_x": True, "log_y": True})
    assert thr is not None


def test_graph_sequencing_stats_threshold_really_filters(tmp_path):
    """The returned threshold is the one applied — not a decoration."""
    csv = tmp_path / "counts.csv"
    df = _counts_frame(n_grna=10)
    df.to_csv(csv, index=False)
    thr = SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 2,
        "filter_column": "columnID", "control_wells": [],
        "log_x": False, "log_y": False})
    # Recompute what the threshold means and check it hits the target.
    d = pd.read_csv(csv)
    d["prc"] = (d["plateID"].astype(str) + "_" + d["rowID"].astype(str)
                + "_" + d["columnID"].astype(str))
    d["fraction"] = d["count"] / d.groupby("prc")["count"].transform("sum")
    mean_unique = (d[d["fraction"] >= thr]
                   .groupby(["plateID", "rowID", "columnID"])["grna"]
                   .nunique().mean())
    assert abs(mean_unique - 2) <= 1


# ---------------------------------------------------------------------------
# import-time IPython fallback
# ---------------------------------------------------------------------------

def test_display_falls_back_to_a_no_op_when_IPython_is_unavailable(
        monkeypatch):
    """Importing this module must never block on a half-initialised IPython."""
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "IPython.display" or name.startswith("IPython"):
            raise ImportError("IPython is mid-init")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "spacr.sequencing", raising=False)
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setattr("builtins.__import__", _fake_import)
    mod = importlib.import_module("spacr.sequencing")
    try:
        # The fallback swallows the call rather than raising.
        assert mod.display("anything", extra=1) is None
    finally:
        monkeypatch.undo()
        restored = importlib.reload(importlib.import_module("spacr.sequencing"))
        # Removing and re-importing a submodule updates the package attribute
        # to the temporary module.  Restoring only ``sys.modules`` leaves two
        # live module identities, so a later test can patch one while a
        # function-level import reads the other.  Reunify both views here.
        import spacr
        spacr.sequencing = restored
        assert spacr.sequencing is sys.modules["spacr.sequencing"]


# ---------------------------------------------------------------------------
# save helpers — the append-and-sum contract
# ---------------------------------------------------------------------------

def test_save_unique_combinations_sums_across_calls(tmp_path):
    csv = tmp_path / "uc.csv"
    df = pd.DataFrame({"rowID": ["r1", "r2"], "columnID": ["c1", "c1"],
                       "grna_name": ["g1", "g1"], "count": [3, 4]})
    SEQ.save_unique_combinations_to_csv(df, str(csv))
    SEQ.save_unique_combinations_to_csv(df, str(csv))
    got = pd.read_csv(csv)
    assert sorted(got["count"].tolist()) == [6, 8]


def test_save_qc_df_adds_element_wise(tmp_path):
    csv = tmp_path / "qc.csv"
    qc = pd.DataFrame({"columnID": [2], "total_reads": [10]})
    SEQ.save_qc_df_to_csv(qc, str(csv))
    SEQ.save_qc_df_to_csv(qc, str(csv))
    got = pd.read_csv(csv)
    assert got["columnID"].tolist() == [4]
    assert got["total_reads"].tolist() == [20]


def test_save_df_to_hdf5_appends(tmp_path):
    h5 = tmp_path / "reads.h5"
    df = pd.DataFrame({"read": ["ACGT", "TTGG"]})
    SEQ.save_df_to_hdf5(df, str(h5))
    SEQ.save_df_to_hdf5(df, str(h5))
    assert pd.read_hdf(h5, "df")["read"].tolist() == [
        "ACGT", "TTGG", "ACGT", "TTGG"]
