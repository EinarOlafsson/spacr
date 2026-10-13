"""FASTQ from ENA/SRA groups into R1/R2, and an odd name says so.

Reported 2026-09-01, immediately after the Map Barcodes download succeeded::

    {'SRR33531218': {}, 'SRR33531217': {}, ...}
    KeyError: 'R1'

``parse_gz_files`` split ``SRR33531217_1.fastq.gz`` on "_" and read
``1.fastq.gz`` as the read direction. It matched neither "R1" nor "R2", so
nothing was stored -- but the sample key was created anyway, as an empty dict.
The crash then happened several frames later, naming neither the file nor the
problem.

Both halves are covered here: the archive convention is understood, and a name
that genuinely cannot be read contributes nothing instead of a booby trap.
"""
from __future__ import annotations

import pytest

from spacr.io import parse_gz_files


def _touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")
    return str(folder)


def test_the_archive_convention_groups_into_pairs(tmp_path):
    """``<run>_1.fastq.gz`` is what every ENA and SRA download is called."""
    src = _touch(tmp_path, "SRR33531217_1.fastq.gz", "SRR33531217_2.fastq.gz",
                 "SRR33531218_1.fastq.gz", "SRR33531218_2.fastq.gz")
    got = parse_gz_files(src)
    assert set(got) == {"SRR33531217", "SRR33531218"}
    assert set(got["SRR33531217"]) == {"R1", "R2"}
    assert got["SRR33531217"]["R1"].endswith("SRR33531217_1.fastq.gz")
    assert got["SRR33531217"]["R2"].endswith("SRR33531217_2.fastq.gz")


def test_the_illumina_convention_still_works(tmp_path):
    """The naming this function was written for must not regress."""
    src = _touch(tmp_path, "sample_R1.fastq.gz", "sample_R2.fastq.gz")
    got = parse_gz_files(src)
    assert set(got["sample"]) == {"R1", "R2"}


def test_a_full_illumina_name_finds_the_mate_in_the_middle(tmp_path):
    """``<sample>_S1_L001_R1_001.fastq.gz`` does not end in its mate."""
    src = _touch(tmp_path, "plate1_S1_L001_R1_001.fastq.gz",
                 "plate1_S1_L001_R2_001.fastq.gz")
    got = parse_gz_files(src)
    assert list(got) == ["plate1_S1_L001"], got
    assert set(got["plate1_S1_L001"]) == {"R1", "R2"}


def test_a_sample_name_containing_underscores_survives(tmp_path):
    """Splitting on "_" and taking parts[0] truncated these to their first
    token, so two different plates collided into one sample."""
    src = _touch(tmp_path, "hilib_p1_1.fastq.gz", "hilib_p1_2.fastq.gz",
                 "hilib_p2_1.fastq.gz", "hilib_p2_2.fastq.gz")
    got = parse_gz_files(src)
    assert set(got) == {"hilib_p1", "hilib_p2"}, got


def test_an_unreadable_name_contributes_nothing(tmp_path):
    """NOT an empty entry. That was the booby trap: a sample with no reads
    looks like a sample right up until something indexes it."""
    src = _touch(tmp_path, "mystery.fastq.gz", "also_weird_xyz.fastq.gz")
    got = parse_gz_files(src)
    for sample, reads in got.items():
        assert reads, f"{sample} was created with no reads in it"


def test_a_single_ended_file_is_still_seen(tmp_path):
    src = _touch(tmp_path, "onlyone.fastq.gz")
    got = parse_gz_files(src)
    assert got and all(reads for reads in got.values())


def test_non_fastq_files_are_ignored(tmp_path):
    src = _touch(tmp_path, "SRR1_1.fastq.gz", "notes.txt", "archive.tar.gz")
    assert set(parse_gz_files(src)) == {"SRR1"}


def test_the_real_download_layout_pairs_up(tmp_path):
    """The exact eight files the Map Barcodes download writes."""
    names = [f"SRR3353121{n}_{m}.fastq.gz"
             for n in (7, 8, 9) for m in (1, 2)]
    names += [f"SRR33531220_{m}.fastq.gz" for m in (1, 2)]
    got = parse_gz_files(_touch(tmp_path, *names))
    assert len(got) == 4
    assert all(set(reads) == {"R1", "R2"} for reads in got.values())
