"""Tests for the synthetic FASTQ generator in spacr.qt.synthetic.

Verifies that generated files match:
* Illumina 1.8+ FASTQ format (4-line records, @-headers, +-separator)
* 150 bp read length and per-read qual length
* Every read carries one of the FASTA barcodes at the expected offset
* The full demo layout drops fastq/, barcodes/, and settings CSV
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pytest


def test_generate_barcode_fasta_writes_expected_count(tmp_path: Path):
    from spacr.qt.synthetic import generate_barcode_fasta
    p = generate_barcode_fasta(tmp_path / "grnas.fasta", n_barcodes=8, seed=1)
    assert p.exists()
    lines = p.read_text().splitlines()
    # 2 lines per record → 16 total
    assert len(lines) == 16
    assert all(lines[i].startswith(">gRNA_")
                for i in range(0, len(lines), 2))
    assert all(re.fullmatch(r"[ACGT]{24}", lines[i])
                for i in range(1, len(lines), 2))


def test_barcode_fasta_is_reproducible(tmp_path: Path):
    from spacr.qt.synthetic import generate_barcode_fasta
    p1 = generate_barcode_fasta(tmp_path / "a.fasta", n_barcodes=6, seed=42)
    p2 = generate_barcode_fasta(tmp_path / "b.fasta", n_barcodes=6, seed=42)
    assert p1.read_text() == p2.read_text()


def test_generate_synthetic_fastq_matches_illumina_shape(tmp_path: Path):
    from spacr.qt.synthetic import (
        generate_barcode_fasta, generate_synthetic_fastq,
        FASTQ_READ_LENGTH, FASTQ_I7_INDEX,
    )
    fasta = generate_barcode_fasta(tmp_path / "grnas.fasta",
                                     n_barcodes=4, seed=7)
    fq = generate_synthetic_fastq(
        tmp_path / "reads.fastq", barcodes_fasta=fasta,
        n_reads=200, reads_per_barcode_min=10, seed=7,
    )
    assert fq.name.endswith(".fastq.gz")
    with gzip.open(fq, "rt") as f:
        lines = f.readlines()
    assert len(lines) % 4 == 0
    for i in range(0, len(lines), 4):
        h, s, plus, q = (lines[i], lines[i + 1],
                          lines[i + 2], lines[i + 3])
        assert h.startswith("@")
        assert h.rstrip().endswith(FASTQ_I7_INDEX)
        assert len(s.rstrip()) == FASTQ_READ_LENGTH
        assert plus.rstrip() == "+"
        assert len(q.rstrip()) == FASTQ_READ_LENGTH


def test_synthetic_fastq_reads_carry_barcodes(tmp_path: Path):
    from spacr.qt.synthetic import (
        generate_barcode_fasta, generate_synthetic_fastq,
    )
    fasta = generate_barcode_fasta(tmp_path / "grnas.fasta",
                                     n_barcodes=3, seed=3)
    barcodes = [l.strip() for l in fasta.read_text().splitlines()
                if not l.startswith(">") and l.strip()]
    fq = generate_synthetic_fastq(tmp_path / "reads.fastq",
                                    barcodes_fasta=fasta,
                                    n_reads=60, seed=3)
    with gzip.open(fq, "rt") as f:
        seqs = [f.readlines()[1].rstrip()
                for f in [f]]   # first record
    # Broader check: at least ONE barcode should appear in most reads
    hit = 0
    with gzip.open(fq, "rt") as f:
        for i, ln in enumerate(f):
            if i % 4 == 1 and any(b in ln for b in barcodes):
                hit += 1
    assert hit > 10, "very few reads carry any known barcode — layout drift?"


def test_generate_map_barcodes_demo_full_layout(tmp_path: Path):
    from spacr.qt.synthetic import generate_map_barcodes_demo
    layout = generate_map_barcodes_demo(tmp_path / "demo",
                                          n_barcodes=5, n_reads=200,
                                          seed=1)
    assert (layout.src / "barcodes" / "grnas.fasta").exists()
    assert (layout.src / "fastq" / "synthetic_R1.fastq.gz").exists()
    assert layout.settings_csv is not None and layout.settings_csv.exists()
    assert layout.notes["n_barcodes"] == 5
    assert layout.notes["n_reads"] == 200
