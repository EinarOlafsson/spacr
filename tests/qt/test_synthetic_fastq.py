"""Tests for the synthetic FASTQ generator in spacr.qt.synthetic.

The sequencing demo used to exit 0 having written nothing. Two independent
reasons, both asserted here:

* the reads were named ``synthetic_R1.fastq.gz``, but
  ``spacr.io.parse_gz_files`` groups on ``filename.split('_')`` and reads
  ``parts[1]`` as the read direction — so it returned ``{'synthetic': {}}``
  and generate_barecode_mapping died on ``KeyError: 'R1'``;
* the reads did not carry the adapter frame the shipped ``regex`` /
  ``target_sequence`` / ``offset_start`` / ``expected_end`` defaults parse,
  so even a correctly-named file would have mapped nothing.

So the assertions here are made against spaCR's own defaults and its own
parsers rather than against a copy of the layout.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Barcodes
# ---------------------------------------------------------------------------

def test_barcode_pool_is_unique_and_reproducible():
    from spacr.qt.synthetic import barcode_pool, GRNA_LENGTH
    a = barcode_pool(12, GRNA_LENGTH, seed=3)
    b = barcode_pool(12, GRNA_LENGTH, seed=3)
    assert a == b
    assert len(set(a)) == 12
    assert all(re.fullmatch(r"[ACGT]{21}", bc) for bc in a)


def test_barcodes_avoid_the_adapter_motifs():
    """A barcode carrying an adapter motif gives the regex a second place to
    anchor, so the barcode that comes back out is not the one planted."""
    from spacr.qt.synthetic import (
        barcode_pool, GRNA_LENGTH, WELL_BARCODE_LENGTH, _FORBIDDEN_MOTIFS,
    )
    for length in (GRNA_LENGTH, WELL_BARCODE_LENGTH):
        for bc in barcode_pool(40, length, seed=1):
            for motif in _FORBIDDEN_MOTIFS:
                assert motif not in bc, (bc, motif)


def test_generate_barcode_csv_is_readable_by_map_sequences_to_names(tmp_path: Path):
    """map_sequences_to_names needs 'name' and 'sequence' columns and rejects
    duplicate sequences. The demo used to ship a FASTA, which it cannot read."""
    from spacr.qt.synthetic import (
        barcode_pool, generate_barcode_csv, WELL_BARCODE_LENGTH,
    )
    from spacr.sequencing import map_sequences_to_names
    seqs = barcode_pool(4, WELL_BARCODE_LENGTH, seed=5)
    names = [f"r{i + 1}" for i in range(len(seqs))]
    csv_path = generate_barcode_csv(tmp_path / "row.csv", names, seqs)
    assert csv_path.exists()
    assert map_sequences_to_names(str(csv_path), seqs, rc=False) == names


def test_generate_barcode_csv_rejects_mismatched_lengths(tmp_path: Path):
    from spacr.qt.synthetic import generate_barcode_csv
    with pytest.raises(ValueError, match="same length"):
        generate_barcode_csv(tmp_path / "x.csv", ["a", "b"], ["ACGT"])


# ---------------------------------------------------------------------------
# One read
# ---------------------------------------------------------------------------

def test_synthetic_read_is_parsed_by_the_shipped_barcode_defaults():
    """The whole point: anchor, slice, split — with spaCR's own settings."""
    from spacr.qt.synthetic import (
        barcode_pool, synthetic_read, FASTQ_READ_LENGTH,
        GRNA_LENGTH, WELL_BARCODE_LENGTH,
    )
    from spacr.settings import set_default_generate_barecode_mapping

    d = set_default_generate_barecode_mapping({})
    grna = barcode_pool(1, GRNA_LENGTH, seed=9)[0]
    row = barcode_pool(1, WELL_BARCODE_LENGTH, seed=10)[0]
    column = barcode_pool(1, WELL_BARCODE_LENGTH, seed=11)[0]

    read = synthetic_read(column, grna, row)
    assert len(read) == FASTQ_READ_LENGTH

    pos = read.find(d["target_sequence"])
    assert pos != -1, "the anchor sequence is not in the read"
    start = max(pos + d["offset_start"], 0)
    window = read[start:start + d["expected_end"]]
    assert len(window) == d["expected_end"]

    match = re.match(d["regex"], window)
    assert match is not None, f"shipped regex did not match {window!r}"
    assert match.group("columnID") == column
    assert match.group("grna") == grna
    assert match.group("rowID") == row


def test_synthetic_read_refuses_a_wrong_length_barcode():
    """A mis-sized barcode shifts every downstream field and maps to nothing —
    far harder to see than a stop."""
    from spacr.qt.synthetic import synthetic_read
    with pytest.raises(ValueError, match="column barcode"):
        synthetic_read("ACG", "A" * 21, "ACGTACGT")
    with pytest.raises(ValueError, match="gRNA barcode"):
        synthetic_read("ACGTACGT", "A" * 24, "ACGTACGT")
    with pytest.raises(ValueError, match="row barcode"):
        synthetic_read("ACGTACGT", "A" * 21, "ACG")


# ---------------------------------------------------------------------------
# FASTQ files
# ---------------------------------------------------------------------------

def test_generate_synthetic_fastq_matches_illumina_shape(tmp_path: Path):
    from spacr.qt.synthetic import (
        barcode_pool, generate_synthetic_fastq,
        FASTQ_READ_LENGTH, FASTQ_I7_INDEX, GRNA_LENGTH, WELL_BARCODE_LENGTH,
    )
    paths = generate_synthetic_fastq(
        tmp_path,
        grnas=barcode_pool(4, GRNA_LENGTH, seed=7),
        rows=barcode_pool(2, WELL_BARCODE_LENGTH, seed=8),
        columns=barcode_pool(2, WELL_BARCODE_LENGTH, seed=9),
        n_reads=200, seed=7,
    )
    assert [p.name for p in paths] == ["demo_R1_001.fastq.gz",
                                       "demo_R2_001.fastq.gz"]
    for path in paths:
        with gzip.open(path, "rt") as f:
            lines = f.readlines()
        assert lines and len(lines) % 4 == 0
        for i in range(0, len(lines), 4):
            h, s, plus, q = lines[i:i + 4]
            assert h.startswith("@")
            assert h.rstrip().endswith(FASTQ_I7_INDEX)
            assert len(s.rstrip()) == FASTQ_READ_LENGTH
            assert plus.rstrip() == "+"
            assert len(q.rstrip()) == FASTQ_READ_LENGTH


def test_r2_is_the_reverse_complement_of_r1(tmp_path: Path):
    """spaCR's paired path reverse-complements R2 and takes a per-base
    consensus with R1, so a perfectly overlapping mate must round-trip."""
    from spacr.qt.synthetic import (
        barcode_pool, generate_synthetic_fastq, GRNA_LENGTH,
        WELL_BARCODE_LENGTH,
    )
    from spacr.sequencing import reverse_complement
    r1_path, r2_path = generate_synthetic_fastq(
        tmp_path,
        grnas=barcode_pool(2, GRNA_LENGTH, seed=1),
        rows=barcode_pool(1, WELL_BARCODE_LENGTH, seed=2),
        columns=barcode_pool(1, WELL_BARCODE_LENGTH, seed=3),
        n_reads=20, seed=1,
    )
    with gzip.open(r1_path, "rt") as f1, gzip.open(r2_path, "rt") as f2:
        r1 = [ln.rstrip() for ln in f1]
        r2 = [ln.rstrip() for ln in f2]
    assert len(r1) == len(r2)
    for i in range(1, len(r1), 4):
        assert reverse_complement(r2[i]) == r1[i]


def test_generate_synthetic_fastq_rejects_an_empty_barcode_set(tmp_path: Path):
    from spacr.qt.synthetic import generate_synthetic_fastq
    with pytest.raises(ValueError, match="non-empty"):
        generate_synthetic_fastq(tmp_path, grnas=[], rows=["ACGTACGT"],
                                 columns=["ACGTACGT"])


# ---------------------------------------------------------------------------
# The demo folder as a whole
# ---------------------------------------------------------------------------

def test_generate_map_barcodes_demo_full_layout(tmp_path: Path):
    from spacr.qt.synthetic import generate_map_barcodes_demo, BARCODE_DIRNAME
    layout = generate_map_barcodes_demo(tmp_path / "demo", n_barcodes=5,
                                        n_reads=200, seed=1)
    barcodes = layout.src / BARCODE_DIRNAME
    for name in ("grna.csv", "row.csv", "column.csv"):
        assert (barcodes / name).exists(), name
    # Flat, not in a fastq/ subfolder: parse_gz_files does a flat listdir of
    # src, so a subfolder means zero samples found.
    assert (layout.src / "demo_R1_001.fastq.gz").exists()
    assert (layout.src / "demo_R2_001.fastq.gz").exists()
    assert layout.settings_csv is not None and layout.settings_csv.exists()
    assert layout.notes["n_barcodes"] == 5
    assert layout.notes["n_reads"] == 200


def test_demo_reads_are_grouped_by_parse_gz_files(tmp_path: Path):
    """``{'synthetic': {}}`` — no 'R1', no 'R2' — was what the old naming gave
    parse_gz_files, and generate_barecode_mapping indexes ['R1'] on it."""
    from spacr.io import parse_gz_files
    from spacr.qt.synthetic import generate_map_barcodes_demo
    layout = generate_map_barcodes_demo(tmp_path / "demo", n_barcodes=4,
                                        n_reads=100, seed=0)
    samples = parse_gz_files(str(layout.src))
    assert set(samples) == {"demo"}
    assert set(samples["demo"]) == {"R1", "R2"}
    for path in samples["demo"].values():
        assert Path(path).is_file()


def test_map_barcodes_demo_settings_pass_preflight(tmp_path: Path):
    """grna_csv / row_csv / column_csv unset is three hard errors, and
    `barcode_length` / `barcode_offset` / `processes` are not spaCR settings."""
    from spacr.qt.synthetic import generate_map_barcodes_demo
    from spacr.utils import load_settings
    from spacr.validate import validate_settings

    layout = generate_map_barcodes_demo(tmp_path / "demo", n_barcodes=4,
                                        n_reads=100, seed=0)
    settings = load_settings(str(layout.settings_csv),
                             setting_key="Key", setting_value="Value")
    problems = validate_settings(settings, "map_barcodes")
    assert not problems, [str(p) for p in problems]
    for key in ("grna_csv", "row_csv", "column_csv"):
        assert Path(settings[key]).is_file(), key
    assert settings["src"] == str(layout.src)


def test_every_read_of_the_demo_maps_back_to_a_planted_barcode(tmp_path: Path):
    """End to end through spaCR's own extractor: anchor, slice, regex, and the
    CSV lookup — every read must resolve to a (row, column, gRNA) name."""
    import pandas as pd
    from spacr.qt.synthetic import generate_map_barcodes_demo, BARCODE_DIRNAME
    from spacr.sequencing import map_sequences_to_names
    from spacr.settings import set_default_generate_barecode_mapping

    layout = generate_map_barcodes_demo(tmp_path / "demo", n_barcodes=4,
                                        n_reads=120, seed=2)
    d = set_default_generate_barecode_mapping({})
    barcodes = layout.src / BARCODE_DIRNAME

    columns, grnas, rows, n_reads = [], [], [], 0
    with gzip.open(layout.src / "demo_R1_001.fastq.gz", "rt") as f:
        for i, line in enumerate(f):
            if i % 4 != 1:
                continue
            n_reads += 1
            read = line.rstrip()
            pos = read.find(d["target_sequence"])
            assert pos != -1
            start = max(pos + d["offset_start"], 0)
            match = re.match(d["regex"], read[start:start + d["expected_end"]])
            assert match is not None, read
            columns.append(match.group("columnID"))
            grnas.append(match.group("grna"))
            rows.append(match.group("rowID"))

    assert n_reads > 0
    for csv_name, seqs in (("column.csv", columns), ("grna.csv", grnas),
                           ("row.csv", rows)):
        names = map_sequences_to_names(str(barcodes / csv_name), seqs, rc=False)
        assert not any(pd.isna(n) for n in names), (
            f"{csv_name}: unmapped barcodes in the demo reads")
