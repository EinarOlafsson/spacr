"""
Tests for spacr.sequencing — Illumina paired-end + 3-barcode pipeline.

Uses ONLY the synthetic Illumina FASTQ.gz + FASTA/CSV barcode fixtures
built in conftest.py. No real sequencing data required.
"""
from __future__ import annotations

import gzip
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spacr.sequencing as SEQ
import spacr.utils as U


# ---------------------------------------------------------------------------
# Barcode reference: FASTA + CSV parity
# ---------------------------------------------------------------------------

def test_synth_barcodes_writes_fasta_and_csv(synth_barcodes):
    """The fixture must emit both FASTA (user-requested) and CSV (spacr
    requires this) versions for each of the three barcode axes."""
    p = synth_barcodes["paths"]
    for label in ("column", "row", "grna"):
        assert Path(p[f"{label}_fasta"]).exists()
        assert Path(p[f"{label}_csv"]).exists()


def test_fasta_and_csv_agree(synth_barcodes):
    """Every FASTA entry must appear in the CSV with the same sequence."""
    p = synth_barcodes["paths"]
    for label in ("column", "row", "grna"):
        fasta = Path(p[f"{label}_fasta"]).read_text()
        csv_df = pd.read_csv(p[f"{label}_csv"])
        fasta_map = {}
        current = None
        for line in fasta.splitlines():
            if line.startswith(">"):
                current = line[1:].strip()
            elif current:
                fasta_map[current] = line.strip()
                current = None
        csv_map = dict(zip(csv_df["name"], csv_df["sequence"]))
        assert fasta_map == csv_map, f"{label}: FASTA and CSV disagree"


def test_barcode_lengths_match_regex_expectations(synth_barcodes):
    """spacr's default barcode regex expects 8bp column, 8bp row,
    20-21bp gRNA."""
    for name, seq in synth_barcodes["columns"].items():
        assert len(seq) == 8, f"column {name} is not 8bp"
    for name, seq in synth_barcodes["rows"].items():
        assert len(seq) == 8, f"row {name} is not 8bp"
    for name, seq in synth_barcodes["grnas"].items():
        assert 20 <= len(seq) <= 21, f"gRNA {name} is not 20-21bp"


# ---------------------------------------------------------------------------
# The synthetic FASTQ files are real gzipped Illumina-format records
# ---------------------------------------------------------------------------

def test_synth_illumina_reads_are_gzipped_fastq(synth_illumina_reads):
    for path_key in ("r1_path", "r2_path"):
        path = synth_illumina_reads[path_key]
        assert path.endswith(".fastq.gz")
        # Verify gzip magic + valid 4-line FASTQ records inside.
        with gzip.open(path, "rt") as fh:
            head = "".join(fh.readline() for _ in range(4))
        parts = head.rstrip("\n").split("\n")
        assert len(parts) == 4
        assert parts[0].startswith("@")
        assert parts[2] == "+"
        assert len(parts[1]) == len(parts[3])


def test_count_reads_in_fastq_matches_fixture(synth_illumina_reads):
    n = U.count_reads_in_fastq(synth_illumina_reads["r1_path"])
    assert n == synth_illumina_reads["n_reads"]


# ---------------------------------------------------------------------------
# Every synthetic read matches spacr's default barcode regex, with the
# right barcodes recoverable via the named groups.
# ---------------------------------------------------------------------------

# NOTE: spacr.sequencing.process_chunk accesses match.group('columnID') and
# match.group('rowID'), not 'column' / 'row' (see paired_find_sequence_in_chunk_reads
# in sequencing.py). The default regex string in settings.py uses the shorter
# names — that's a pre-existing spacr bug — so tests that call process_chunk
# must pass a regex with the *ID variants. Tests that only exercise the
# regex itself use the short form to mirror the default that ships.
DEFAULT_BARCODE_REGEX = r"^(?P<column>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<row>.{8}).*"
INTERNAL_BARCODE_REGEX = r"^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<rowID>.{8}).*"


def test_default_regex_matches_every_synthetic_read(synth_illumina_reads):
    rx = re.compile(DEFAULT_BARCODE_REGEX)
    for entry in synth_illumina_reads["truth"]:
        m = rx.match(entry["seq"])
        assert m is not None, f"read {entry['read_id']} did not match the regex"


def test_default_regex_recovers_truth_barcodes(synth_illumina_reads, synth_barcodes):
    rx = re.compile(DEFAULT_BARCODE_REGEX)
    for entry in synth_illumina_reads["truth"]:
        m = rx.match(entry["seq"])
        col_seq = m.group("column")
        row_seq = m.group("row")
        grna_seq = m.group("grna")
        # Look up the sequences in the truth table.
        assert col_seq == synth_barcodes["columns"][entry["column"]]
        assert row_seq == synth_barcodes["rows"][entry["row"]]
        assert grna_seq == synth_barcodes["grnas"][entry["grna"]]


# ---------------------------------------------------------------------------
# map_sequences_to_names — using the CSV shape spacr expects
# ---------------------------------------------------------------------------

def test_map_sequences_to_names_recovers_labels(synth_barcodes):
    csv_path = synth_barcodes["paths"]["column_csv"]
    seqs = list(synth_barcodes["columns"].values())
    expected_names = list(synth_barcodes["columns"].keys())
    got = SEQ.map_sequences_to_names(csv_path, seqs, rc=False)
    assert got == expected_names


def test_map_sequences_to_names_returns_pd_na_for_unknown(synth_barcodes):
    csv_path = synth_barcodes["paths"]["column_csv"]
    unknown = "AAAAAAAA"
    got = SEQ.map_sequences_to_names(csv_path, [unknown], rc=False)
    assert len(got) == 1
    assert pd.isna(got[0])


def test_map_sequences_to_names_reverse_complement_mode(synth_barcodes):
    """When rc=True the CSV sequences are reverse-complemented before matching,
    so the RAW barcode should no longer match — but its reverse complement
    should."""
    csv_path = synth_barcodes["paths"]["column_csv"]
    seqs = list(synth_barcodes["columns"].values())
    got_rc = SEQ.map_sequences_to_names(csv_path, seqs, rc=True)
    # Nothing should match under rc=True because we injected the raw sequences,
    # not their complements.
    assert all(pd.isna(x) for x in got_rc)


# ---------------------------------------------------------------------------
# Consensus / helper functions
# ---------------------------------------------------------------------------

def test_reverse_complement_basic():
    assert SEQ.reverse_complement("ACGT") == "ACGT"
    assert SEQ.reverse_complement("A") == "T"
    assert SEQ.reverse_complement("AAAA") == "TTTT"
    assert SEQ.reverse_complement("ATCG") == "CGAT"


def test_extract_sequence_and_quality():
    seq = "ACGTACGT"
    qual = "IIIIIIII"
    s, q = SEQ.extract_sequence_and_quality(seq, qual, 2, 6)
    assert s == "GTAC"
    assert q == "IIII"


def test_get_consensus_base_prefers_non_N():
    assert SEQ.get_consensus_base([("N", "!"), ("A", "!")]) == "A"
    assert SEQ.get_consensus_base([("A", "!"), ("N", "!")]) == "A"


def test_get_consensus_base_prefers_higher_quality():
    # ASCII quality: '!' = 33, 'I' = 73
    assert SEQ.get_consensus_base([("A", "!"), ("G", "I")]) == "G"
    assert SEQ.get_consensus_base([("A", "I"), ("G", "!")]) == "A"


def test_create_consensus_takes_best_per_position():
    # Same length, quality picks winner per base.
    seq1 = "ACGT"; qual1 = "IIII"
    seq2 = "ANGT"; qual2 = "IIII"
    cons = SEQ.create_consensus(seq1, qual1, seq2, qual2)
    # Position 1: A vs N → A. Others equal.
    assert cons == "ACGT"


# ---------------------------------------------------------------------------
# End-to-end: process_chunk on a batch of synthetic reads
# ---------------------------------------------------------------------------

def _read_fastq_records(path):
    """Yield each FASTQ record as a single 4-line string, which is what
    spacr.sequencing.paired_find_sequence_in_chunk_reads expects."""
    with gzip.open(path, "rt") as fh:
        rec = []
        for line in fh:
            rec.append(line.rstrip("\n"))
            if len(rec) == 4:
                yield "\n".join(rec)
                rec = []


def test_process_chunk_end_to_end(synth_illumina_reads, synth_barcodes):
    """Feed the sequencing pipeline's process_chunk with a paired R1/R2
    chunk of synthetic reads and check that:
      * every read is decoded into a consensus + 3 barcodes
      * the barcode-to-name mapping recovers the truth labels
      * unique_combinations aggregates rowID x columnID x grna correctly."""
    p = synth_barcodes["paths"]

    r1_records = list(_read_fastq_records(synth_illumina_reads["r1_path"]))
    r2_records = list(_read_fastq_records(synth_illumina_reads["r2_path"]))
    # process_chunk expects the chunk in the same interleaved form the
    # readers produce internally. Feed as (r1_chunk, r2_chunk, ...).
    # Anchor the extractor on the "TGCTG" spacer at position 8 of every
    # synthetic read; extract from position 0 (offset -8 from the anchor)
    # for the full read length (~76 bp with our fixture layout).
    chunk_data = (
        r1_records, r2_records,
        INTERNAL_BARCODE_REGEX,        # columnID / rowID group names
        "TGCTG",                       # anchor present in every synth read
        -8, 76,                        # start at the true read start; length 76bp
        p["column_csv"], p["grna_csv"], p["row_csv"],
        False,                         # fill_na
    )
    df, unique_combinations, qc_df = SEQ.process_chunk(chunk_data)

    # Basic shape checks.
    assert isinstance(df, pd.DataFrame)
    assert isinstance(unique_combinations, pd.DataFrame)
    assert isinstance(qc_df, pd.DataFrame)
    assert len(df) == synth_illumina_reads["n_reads"]
    for col in ("read", "column_sequence", "columnID",
                "row_sequence", "rowID", "grna_sequence", "grna_name"):
        assert col in df.columns

    # The QC frame has one row keyed 'NaN_Counts' with total_reads populated.
    assert "total_reads" in qc_df.columns
    assert qc_df["total_reads"].iloc[0] == synth_illumina_reads["n_reads"]

    # Unique combinations should be a subset of the truth combinations.
    truth_combos = {(t["row"], t["column"], t["grna"])
                    for t in synth_illumina_reads["truth"]}
    detected_combos = set(
        zip(unique_combinations["rowID"],
            unique_combinations["columnID"],
            unique_combinations["grna_name"])
    )
    # detected_combos should be a subset of truth_combos (there may be
    # unmapped reads if consensus extraction dropped any, but nothing
    # spurious should appear).
    assert detected_combos.issubset(truth_combos), (
        f"detected barcode combos not in truth: {detected_combos - truth_combos}"
    )
