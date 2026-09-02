"""Sequencing reads come from the archive, and the read limit really limits.

Asked for on 2026-09-01: Map Barcodes should fetch the paper's own sequencing
data, with the user saying how many reads to take from each file. The runs are
2.2-3.2 GB per mate and 19 GB in total, so the limit is what makes the feature
usable rather than a convenience -- and it only works because ENA serves plain
gzipped FASTQ that can be read as a stream and abandoned early.

Everything here runs offline through an injected opener except
``test_the_live_archive_still_answers``, which is marked and skipped by
default.
"""
from __future__ import annotations

import gzip
import io
import zlib

import pytest

from spacr.sra import (
    DEFAULT_BIOPROJECT,
    RunFile,
    estimated_bytes,
    fetch_reads,
    runs_for,
    total_bytes,
)

PORTAL_TSV = (
    "run_accession\tlibrary_name\tfastq_ftp\tfastq_bytes\tread_count\n"
    "SRR33531217\thilib_p4\t"
    "ftp.sra.ebi.ac.uk/a/SRR33531217_1.fastq.gz;"
    "ftp.sra.ebi.ac.uk/a/SRR33531217_2.fastq.gz\t"
    "2833522805;3222897932\t73698595\n"
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _fastq(reads: int) -> bytes:
    out = []
    for n in range(reads):
        out += [f"@read{n}".encode(), b"ACGT", b"+", b"IIII"]
    return b"\n".join(out) + b"\n"


def _gzipped(payload: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    return compressor.compress(payload) + compressor.flush()


def test_the_portal_rows_become_one_entry_per_mate():
    files = runs_for(DEFAULT_BIOPROJECT,
                     opener=lambda url: _Response(PORTAL_TSV.encode()))
    assert [f.mate for f in files] == [1, 2]
    assert {f.run for f in files} == {"SRR33531217"}
    assert files[0].library == "hilib_p4", "the library names the plate"


def test_a_run_label_names_the_cost_when_the_archive_reports_it():
    known = RunFile(
        run="SRR1", library="hilib_p1", url="https://x/SRR1_1.fastq.gz",
        mate=1, size_bytes=2_500_000_000, read_count=73_000_000,
    )
    unknown = RunFile(
        run="SRR2", library="hilib_p2", url="https://x/SRR2_2.fastq.gz",
        mate=2,
    )

    assert known.label() == "SRR1  ·  mate 1  ·  hilib_p1  ·  2.5 GB  ·  73M reads"
    assert unknown.label() == "SRR2  ·  mate 2  ·  hilib_p2  ·  ?"


def test_the_ftp_host_is_fetched_over_https():
    """FTP is blocked on many institutional networks, and a failure there
    would be unexplainable to the person it happens to."""
    files = runs_for(opener=lambda url: _Response(PORTAL_TSV.encode()))
    assert all(f.url.startswith("https://") for f in files)


def test_an_empty_or_broken_portal_answer_yields_nothing():
    assert runs_for(opener=lambda url: _Response(b"")) == ()
    assert runs_for(opener=lambda url: _Response(b"nonsense\n")) == ()


def test_text_metadata_and_bad_archive_counts_degrade_to_unknown():
    payload = (
        "run_accession\tlibrary_name\tfastq_ftp\tfastq_bytes\tread_count\n"
        "SRR1\tp1\thttps://x/one.fastq.gz;https://x/two.fastq.gz\t"
        "not-a-size\tnot-a-count\n"
    )
    files = runs_for(opener=lambda url: io.StringIO(payload))

    assert [(one.url, one.size_bytes, one.read_count) for one in files] == [
        ("https://x/one.fastq.gz", 0, 0),
        ("https://x/two.fastq.gz", 0, 0),
    ]


def test_the_limit_stops_the_download(tmp_path):
    """The point of the feature: ask for 10 reads from a 1000-read file."""
    body = _gzipped(_fastq(1000))
    one = RunFile(run="SRR1", library="p1", url="https://x/SRR1_1.fastq.gz",
                  mate=1, size_bytes=len(body), read_count=1000)

    written = fetch_reads(one, tmp_path, max_reads=10,
                          opener=lambda url: _Response(body))

    with gzip.open(written, "rt") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == 40, "four lines to a read, so ten reads is forty lines"
    assert lines[0] == "@read0"
    assert lines[-4] == "@read9"


def test_no_limit_fetches_the_whole_file(tmp_path):
    body = _gzipped(_fastq(25))
    one = RunFile(run="SRR1", library="p1", url="https://x/SRR1_1.fastq.gz",
                  mate=1)
    written = fetch_reads(one, tmp_path, opener=lambda url: _Response(body))
    with gzip.open(written, "rt") as handle:
        assert len(handle.read().splitlines()) == 100


def test_a_limited_download_accepts_records_arriving_across_small_chunks(tmp_path):
    class TinyChunks(_Response):
        def read(self, n=-1):
            return super().read(min(n, 7))

    body = _gzipped(_fastq(25))
    one = RunFile(run="SRR1", library="p1", url="https://x/a.fastq.gz", mate=1)
    written = fetch_reads(
        one, tmp_path, max_reads=5, opener=lambda url: TinyChunks(body),
    )

    with gzip.open(written, "rt") as handle:
        lines = handle.read().splitlines()
    assert len(lines) == 20
    assert lines[-4] == "@read4"


def test_an_unlimited_download_preserves_a_final_line_without_a_newline(tmp_path):
    body = _gzipped(b"@read0\nACGT\n+\nIIII")
    one = RunFile(run="SRR1", library="p1", url="https://x/a.fastq.gz", mate=1)

    written = fetch_reads(one, tmp_path, opener=lambda url: _Response(body))

    with gzip.open(written, "rb") as handle:
        assert handle.read() == b"@read0\nACGT\n+\nIIII"


def test_the_output_is_gzipped_fastq(tmp_path):
    """`src` documents "the folder of .fastq.gz reads", so a subset and a full
    download must be the same shape to everything downstream."""
    body = _gzipped(_fastq(8))
    one = RunFile(run="SRR1", library="p1", url="https://x/SRR1_1.fastq.gz",
                  mate=1)
    written = fetch_reads(one, tmp_path, max_reads=4,
                          opener=lambda url: _Response(body))
    assert written.name.endswith(".fastq.gz")
    with gzip.open(written, "rb") as handle:
        assert handle.read(1) == b"@"


def test_a_cancelled_download_leaves_no_partial_file(tmp_path):
    """A half file looks finished to anything that lists the folder."""
    body = _gzipped(_fastq(1000))
    one = RunFile(run="SRR1", library="p1", url="https://x/SRR1_1.fastq.gz",
                  mate=1)
    with pytest.raises(InterruptedError):
        fetch_reads(one, tmp_path, opener=lambda url: _Response(body),
                    should_stop=lambda: True)
    assert list(tmp_path.iterdir()) == []


def test_a_failure_mid_stream_leaves_no_partial_file(tmp_path):
    class Broken(_Response):
        def read(self, n=-1):
            raise OSError("connection reset")

    one = RunFile(run="SRR1", library="p1", url="https://x/SRR1_1.fastq.gz",
                  mate=1)
    with pytest.raises(OSError):
        fetch_reads(one, tmp_path, opener=lambda url: Broken(b""))
    assert list(tmp_path.iterdir()) == []


def test_a_nonsense_limit_is_refused(tmp_path):
    one = RunFile(run="SRR1", library="p1", url="https://x/a.fastq.gz", mate=1)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            fetch_reads(one, tmp_path, max_reads=bad,
                        opener=lambda url: _Response(b""))


def test_progress_is_reported_in_reads_not_lines(tmp_path):
    seen = []
    body = _gzipped(_fastq(400))
    one = RunFile(run="SRR1", library="p1", url="https://x/a.fastq.gz", mate=1)
    fetch_reads(one, tmp_path, max_reads=100,
                opener=lambda url: _Response(body),
                progress=lambda reads, byts: seen.append(reads))
    assert seen and seen[-1] == 100, seen


def test_the_estimate_scales_with_the_share_asked_for():
    files = runs_for(opener=lambda url: _Response(PORTAL_TSV.encode()))
    full = total_bytes(files)
    assert estimated_bytes(files, None) == full
    part = estimated_bytes(files, 100_000)
    assert 0 < part < full / 100, (
        "100k of 73.7M reads should be well under a hundredth of 6 GB")


def test_the_estimate_does_not_undersell_an_unknown_run():
    """Guessing low on a multi-gigabyte download is the expensive mistake."""
    one = RunFile(run="SRR1", library="p1", url="https://x/a.fastq.gz",
                  mate=1, size_bytes=2_000_000_000, read_count=0)
    assert estimated_bytes([one], 1000) == 2_000_000_000


@pytest.mark.network
def test_the_live_archive_still_answers():
    """The accessions in the paper resolve, and the mirror serves them.

    Skipped unless -m network is asked for: the rest of this file proves the
    logic, and this proves the world has not moved.
    """
    files = runs_for(DEFAULT_BIOPROJECT)
    assert len(files) == 8, "four runs, paired"
    assert {f.run for f in files} == {
        "SRR33531217", "SRR33531218", "SRR33531219", "SRR33531220"}
