"""Map Barcodes reads plain and gzip FASTQ, mixed pairing and checks the sequencer.

Every case runs on small synthetic FASTQ files written here or by the shared
``synth_illumina_reads`` fixture; no network or GPU.
"""
import bz2
import gzip
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest

from spacr import sequencing as SEQ
from spacr.io import parse_gz_files
from spacr.utils import count_reads_in_fastq


def _record(name, seq):
    """One FASTQ record with a constant high quality."""
    return f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n"


def _settings(src, barcodes, **extra):
    """Mapping settings for the synthetic references, one worker, no HDF5."""
    paths = barcodes["paths"]
    settings = dict(src=str(src), column_csv=paths["column_csv"],
                    row_csv=paths["row_csv"], grna_csv=paths["grna_csv"],
                    save_h5=False, chunk_size=1000, n_jobs=1, test=False,
                    fill_na=False)
    settings.update(extra)
    return settings


def _plain_copy(source, target):
    """Write the decompressed text of a gzip FASTQ to ``target``."""
    with gzip.open(source, "rt") as handle:
        Path(target).write_text(handle.read())


def _counts(folder):
    """The per-well count table of one sample folder, sorted for comparison."""
    table = pd.read_csv(Path(folder) / "unique_combinations.csv")
    return table.sort_values(list(table.columns)).reset_index(drop=True)


def test_folder_reader_groups_every_fastq_spelling_and_leaves_out_index_reads(tmp_path, caplog):
    for name in ("a_R1.fastq", "a_R2.fq", "b_1.fq.gz", "b_2.fastq.gz",
                 "c_S1_L001_R1_001.fastq.gz", "c_S1_L001_I1_001.fastq.gz",
                 "d_R1.fastq", "d_R1.fastq.gz", "notes.txt", "e_R1.bam"):
        (tmp_path / name).write_text("")
    with caplog.at_level(logging.INFO, logger="spacr.io"):
        found = parse_gz_files(str(tmp_path))
    assert found == {
        "a": {"R1": str(tmp_path / "a_R1.fastq"), "R2": str(tmp_path / "a_R2.fq")},
        "b": {"R1": str(tmp_path / "b_1.fq.gz"), "R2": str(tmp_path / "b_2.fastq.gz")},
        "c_S1_L001": {"R1": str(tmp_path / "c_S1_L001_R1_001.fastq.gz")},
    }
    assert "I1 index read" in caplog.text
    assert "2 files claim R1 of this sample (d_R1.fastq, d_R1.fastq.gz)" in caplog.text


def test_compression_is_read_from_the_bytes_not_the_name(tmp_path, synth_illumina_reads):
    gz_named_plain = tmp_path / "x_R1.fastq.gz"
    _plain_copy(synth_illumina_reads["r1_path"], gz_named_plain)
    plain_named_gz = tmp_path / "y_R1.fastq"
    shutil.copy(synth_illumina_reads["r1_path"], plain_named_gz)
    for path, compression in ((gz_named_plain, "plain"), (plain_named_gz, "gzip")):
        assert count_reads_in_fastq(str(path)) == synth_illumina_reads["n_reads"]
        assert SEQ._fastq_head(str(path))[0] == compression


def _bam(path):
    """A gzip stream that opens with the BAM magic, as a BGZF BAM does."""
    with gzip.open(path, "wb") as handle:
        handle.write(b"BAM\x01" + b"\0" * 32)


@pytest.mark.parametrize("writer, message", [
    (_bam, "BAM file"),
    (lambda p: p.write_bytes(bz2.compress(b"@r\nACGT\n+\nIIII\n")), "bzip2, zstd or xz"),
    (lambda p: p.write_bytes(b"CRAM\x03\x00"), "CRAM file"),
    (lambda p: p.write_text(">r\nACGT\n"), "FASTA, not FASTQ"),
    (lambda p: p.write_text("@r\nACGT\n+\nII\n"), "is not FASTQ"),
    (lambda p: p.write_text(""), "holds no reads"),
])
def test_reads_that_are_not_fastq_are_refused_by_what_they_are(tmp_path, writer, message):
    path = tmp_path / "s_R1.fastq.gz"
    writer(path)
    with pytest.raises(ValueError, match=message):
        SEQ._fastq_head(str(path))


@pytest.mark.parametrize("header, convention", [
    ("@A00123:8:HV2FVDSXY:1:1101:1000:1000 1:N:0:ACGT", "illumina"),
    ("@AV224501:1:2324600342:1:10102:0193:0010 1:N:0:ACGT+TTTT", "illumina"),
    ("@V300012345L1C001R0010000001/1", "mgi"),
    ("@E100012345L2C003R0040123456/2", "mgi"),
    ("@m64011_190830_220126/1/ccs", "pacbio"),
    ("@m84011_220902_175841_s1/12345/ccs", "pacbio"),
    ("@1a2b3c4d-0000 runid=abc read=7 ch=12 start_time=2024-01-01T00:00:00Z", "ont"),
    ("@SRR1234567.1 1 length=150", "archive"),
    ("@read_1", ""),
])
def test_header_conventions_are_named_only_from_reliable_patterns(header, convention):
    assert SEQ._header_convention(header) == convention


def test_plain_fastq_decodes_to_the_same_counts_as_gzip(tmp_path, synth_illumina_reads, synth_barcodes):
    gz_src, plain_src = tmp_path / "gz", tmp_path / "plain"
    gz_src.mkdir()
    plain_src.mkdir()
    for mate in ("R1", "R2"):
        source = synth_illumina_reads[f"{mate.lower()}_path"]
        shutil.copy(source, gz_src / f"s1_{mate}_001.fastq.gz")
        _plain_copy(source, plain_src / f"s1_{mate}_001.fq")
    SEQ.generate_barecode_mapping(_settings(gz_src, synth_barcodes, mode="paired"))
    SEQ.generate_barecode_mapping(_settings(plain_src, synth_barcodes, mode="paired"))
    gz_counts, plain_counts = _counts(gz_src / "s1_paired"), _counts(plain_src / "s1_paired")
    assert gz_counts["count"].sum() == synth_illumina_reads["n_reads"]
    pd.testing.assert_frame_equal(gz_counts, plain_counts)
    record = json.loads((plain_src / "s1_paired" / "read_inputs.json").read_text())
    assert record["pairing"] == "paired"
    assert record["sequencer_declared"] == "auto"
    assert record["sequencer"] == "illumina-style"
    assert {row["header_convention"] for row in record["files"].values()} == {"illumina"}
    assert {row["compression"] for row in record["files"].values()} == {"plain"}


def test_mixed_mode_pairs_what_is_paired_and_reads_the_rest_from_one_mate(
        tmp_path, synth_illumina_reads, synth_barcodes, caplog):
    src = tmp_path / "mixed"
    src.mkdir()
    r1, r2 = synth_illumina_reads["r1_path"], synth_illumina_reads["r2_path"]
    shutil.copy(r1, src / "pair_R1.fastq.gz")
    shutil.copy(r2, src / "pair_R2.fastq.gz")
    _plain_copy(r1, src / "lone_R1.fastq")
    shutil.copy(r2, src / "other_R2.fastq.gz")
    with caplog.at_level(logging.WARNING, logger="spacr.sequencing"):
        SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, mode="mixed",
                                                single_direction="R1"))
    made = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name != "settings")
    assert made == ["lone_single_R1", "pair_paired"]
    assert "other: skipped -- mixed mode decodes an unpaired sample from its R1" in caplog.text
    n = synth_illumina_reads["n_reads"]
    assert _counts(src / "pair_paired")["count"].sum() == n
    assert _counts(src / "lone_single_R1")["count"].sum() == n
    lone = json.loads((src / "lone_single_R1" / "read_inputs.json").read_text())
    assert lone["pairing"] == "single" and list(lone["files"]) == ["R1"]


def test_mates_out_of_step_are_refused(tmp_path, synth_illumina_reads, synth_barcodes):
    src = tmp_path / "swapped"
    src.mkdir()
    shutil.copy(synth_illumina_reads["r1_path"], src / "s_R1.fastq.gz")
    with gzip.open(synth_illumina_reads["r2_path"], "rt") as handle:
        lines = handle.read().splitlines(keepends=True)
    records = [lines[i:i + 4] for i in range(0, len(lines), 4)]
    records[3], records[4] = records[4], records[3]
    with gzip.open(src / "s_R2.fastq.gz", "wt") as handle:
        handle.write("".join("".join(r) for r in records))
    with pytest.raises(ValueError, match="are not mates"):
        SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, mode="paired"))

    first = tmp_path / "first"
    first.mkdir()
    (first / "s_R1.fastq").write_text(_record("A:1:F:1:1:1:1", "ACGT"))
    (first / "s_R2.fastq").write_text(_record("A:1:F:1:1:1:2", "ACGT"))
    with pytest.raises(ValueError, match="named differently"):
        SEQ.generate_barecode_mapping(_settings(first, synth_barcodes, mode="paired"))
    assert not (first / "s_paired").exists()


def test_the_declared_sequencer_is_checked_against_the_headers(tmp_path, synth_barcodes):
    src = tmp_path / "mgi"
    src.mkdir()
    (src / "s_R1.fastq").write_text(_record("A00123:8:HV2:1:1101:1000:1000 1:N:0:A", "ACGT"))
    with pytest.raises(ValueError, match="follow the illumina convention, but sequencer is 'mgi'"):
        SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, mode="single",
                                                sequencer="mgi"))
    assert not (src / "s_single_R1").exists()
    for sequencer in ("ont", "pacbio", "ultima"):
        with pytest.raises(ValueError, match="mode must be 'single'"):
            SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, mode="paired",
                                                    sequencer=sequencer))
    with pytest.raises(ValueError, match="sequencer must be one of"):
        SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, sequencer="hiseq"))
    with pytest.raises(ValueError, match="'paired', 'single' or 'mixed'"):
        SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, mode="both"))


def test_auto_reads_a_single_read_platform_from_its_headers(tmp_path, synth_barcodes):
    src = tmp_path / "ont"
    src.mkdir()
    header = "1a2b-0000 runid=abc read=7 ch=12"
    (src / "s_R1.fastq").write_text(_record(header, "ACGT"))
    (src / "s_R2.fastq").write_text(_record(header, "ACGT"))
    with pytest.raises(ValueError, match="Oxford Nanopore, which produces single reads"):
        SEQ.generate_barecode_mapping(_settings(src, synth_barcodes, mode="paired"))
    lone = {"R1": str(src / "s_R1.fastq")}
    assert SEQ._inspect_sample_reads("s", lone, "auto", "single")["sequencer"] == "ont"
    assert SEQ._inspect_sample_reads("s", lone, "other", "single")["sequencer"] == "other"
    with pytest.raises(ValueError, match="follow the ont convention"):
        SEQ._inspect_sample_reads("s", lone, "illumina", "single")


def test_a_folder_of_unconvertible_reads_names_the_conversion(tmp_path, synth_barcodes):
    for name in ("run.bam", "run.sra", "s_R1.fastq.ora"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(ValueError) as caught:
        SEQ.generate_barecode_mapping(_settings(tmp_path, synth_barcodes))
    text = str(caught.value)
    assert "samtools fastq" in text and "fasterq-dump" in text and "orad" in text


def test_protocols_carry_sequencer_and_mixed_pairing_into_map_barcodes():
    from spacr.qt.widgets import sample_project as sample

    mixed = sample._protocol_result('sequencing', 'fastq_gz',
                                    dict(sequencer='element', read_mode='mixed',
                                         single_direction='R1'))
    assert [row[0] for row in mixed['actions']] == ['map_barcodes']
    handoff = sample._protocol_handoff(mixed, 'map_barcodes')
    assert handoff['settings']['mode'] == 'mixed'
    assert handoff['settings']['sequencer'] == 'element'
    assert handoff['settings']['single_direction'] == 'R1'
    unknown = sample._protocol_result('sequencing', 'fastq_gz',
                                      dict(sequencer='other', read_mode='paired'))
    assert sample._protocol_handoff(unknown, 'map_barcodes')['settings']['sequencer'] == 'auto'
    with pytest.raises(ValueError, match='required'):
        sample._protocol_result('sequencing', 'fastq_gz',
                                dict(sequencer='other', read_mode='mixed'))
    long_read = sample._protocol_result('sequencing', 'fastq_gz',
                                        dict(sequencer='ont', read_mode='paired'))
    assert not long_read['actions']
    assert any('single reads' in text for text in long_read['requirements'])
    single = sample._protocol_result('sequencing', 'fastq_gz',
                                     dict(sequencer='pacbio', read_mode='single',
                                          single_direction='R1'))
    assert [row[0] for row in single['actions']] == ['map_barcodes']
    other = sample._protocol_result('sequencing', 'other')
    assert not other['actions']
    assert any('convert' in text for text in other['requirements'])
