"""An empty read file is described and warned about, not refused."""

import logging

from spacr import sequencing as SEQ


def _record(header, sequence):
    return f"@{header}\n{sequence}\n+\n{'I' * len(sequence)}\n"


def test_an_empty_mate_is_described_as_holding_no_reads(tmp_path, caplog):
    r1 = tmp_path / "s_R1.fastq"
    r2 = tmp_path / "s_R2.fastq"
    r1.write_text("")
    r2.write_text(_record("A00123:8:HV2FVDSXY:1:1101:1000:1000 1:N:0:ACGT",
                          "ACGT"))
    with caplog.at_level(logging.WARNING, logger=SEQ.LOG.name):
        described = SEQ._inspect_sample_reads(
            "s", {"R1": str(r1), "R2": str(r2)}, "auto", "paired")
    assert described["files"]["R1"] == dict(
        path=str(r1), compression="", header_convention="no reads",
        first_read="")
    assert described["files"]["R2"]["compression"] == "plain"
    assert "s: R1 holds no reads, so nothing is decoded from it." in caplog.text


def test_two_empty_mates_are_not_compared_for_pairing(tmp_path, caplog):
    files = {}
    for mate in ("R1", "R2"):
        path = tmp_path / f"s_{mate}.fastq"
        path.write_text("")
        files[mate] = str(path)
    with caplog.at_level(logging.WARNING, logger=SEQ.LOG.name):
        described = SEQ._inspect_sample_reads("s", files, "auto", "paired")
    assert described["sequencer"] == "unknown"
    assert "R1, R2 holds no reads" in caplog.text
