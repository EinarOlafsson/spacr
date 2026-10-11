"""Map Barcodes' test reads come from the cached read set when it is there.

Tutorial capture runs with the network unshared, and Load test data streamed
its reads from ENA, so the lesson failed. The ``reads`` example set
(``einarolafsson/spacr-example-reads``) holds the first reads of every
published file and ENA's listing of the project; with it unpacked, listing
and a limited fetch need no network. ENA stays the route for anything the
cache does not hold. Every download here is mocked.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
import zlib
from pathlib import Path

import pytest

from spacr import sra
from spacr.example_archives import (EXAMPLE_ARCHIVES, READS_EXAMPLE_REPO,
                                    example_set, example_set_folder)

LISTING = (
    "run_accession\tlibrary_name\tfastq_ftp\tfastq_bytes\tread_count\n"
    "SRR33531217\thilib_p1\t"
    "ftp.sra.ebi.ac.uk/a/SRR33531217_1.fastq.gz;"
    "ftp.sra.ebi.ac.uk/a/SRR33531217_2.fastq.gz\t"
    "2833522805;3222897932\t73698595\n"
    "SRR33531218\thilib_p2\t"
    "ftp.sra.ebi.ac.uk/a/SRR33531218_1.fastq.gz;"
    "ftp.sra.ebi.ac.uk/a/SRR33531218_2.fastq.gz\t"
    "2900000000;3300000000\t70000000\n"
)

CACHED = 50


def _fastq(reads: int, tag: str = "read") -> bytes:
    lines = []
    for n in range(reads):
        lines += [f"@{tag}{n}".encode(), b"ACGT", b"+", b"IIII"]
    return b"\n".join(lines) + b"\n"


def _gzipped(payload: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    return compressor.compress(payload) + compressor.flush()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to reach ENA fails the test."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("the network was reached")

    monkeypatch.setattr("urllib.request.urlopen", _forbidden)


def _cached_set(folder: Path, reads: int = CACHED) -> Path:
    """Write a cached read set the way the builder lays one out."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / sra.CACHED_LISTING).write_text(LISTING)
    files = {}
    for one in sra.runs_for(opener=lambda _url: _Response(LISTING.encode())):
        with gzip.open(folder / one.filename, "wb") as handle:
            handle.write(_fastq(reads, tag=f"{one.run}_{one.mate}_"))
        files[one.filename] = {"run": one.run, "mate": one.mate,
                               "reads": reads}
    (folder / sra.CACHED_MANIFEST).write_text(json.dumps(
        {"bioproject": sra.DEFAULT_BIOPROJECT, "reads_per_file": reads,
         "files": files}))
    return folder


def _reads_in(path: Path) -> list:
    with gzip.open(path, "rt") as handle:
        return [line.strip() for line in handle][::4]


def test_the_reads_set_is_a_registered_example_set():
    """``spacr-download reads`` knows the set, its repo and its archive."""
    entry = example_set("reads")
    assert entry.repo == READS_EXAMPLE_REPO == "einarolafsson/spacr-example-reads"
    assert EXAMPLE_ARCHIVES[READS_EXAMPLE_REPO] == "spacr-example-reads.tar"
    assert example_set_folder("reads").name == "reads"
    assert not entry.in_default


def test_the_listing_comes_from_the_cache_without_the_network(tmp_path,
                                                               no_network):
    """The cached listing gives the runs ENA's portal reports."""
    folder = _cached_set(tmp_path / "reads")
    runs = sra._cached_runs(folder)
    online = sra.runs_for(opener=lambda _url: _Response(LISTING.encode()))
    assert runs == online
    assert sra._has_cached_reads(folder)
    assert sra._cached_reads_per_file(folder) == CACHED


def test_a_limited_fetch_is_cut_from_the_cache(tmp_path, monkeypatch,
                                               no_network):
    """Ten reads of a file the cache holds fifty of, with no network."""
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = _cached_set(example_set_folder("reads"))
    one = sra._cached_runs(folder)[0]
    written = sra.fetch_reads(one, tmp_path / "out", max_reads=10)
    assert _reads_in(written) == [f"@{one.run}_{one.mate}_{n}"
                                  for n in range(10)]


def test_more_reads_than_the_cache_holds_go_to_ena(tmp_path, monkeypatch):
    """The cache never answers for reads it does not have."""
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = _cached_set(example_set_folder("reads"))
    one = sra._cached_runs(folder)[0]
    asked = []

    def _ena(request, timeout=None):
        asked.append(request.full_url)
        return _Response(_gzipped(_fastq(200, tag="ena")))

    monkeypatch.setattr("urllib.request.urlopen", _ena)
    written = sra.fetch_reads(one, tmp_path / "out", max_reads=CACHED + 1)
    assert asked == [one.url]
    assert _reads_in(written)[0] == "@ena0"
    assert len(_reads_in(written)) == CACHED + 1


def test_a_whole_file_always_comes_from_ena(tmp_path):
    """The cache holds the first reads only, so it never serves a whole file."""
    folder = _cached_set(tmp_path / "reads")
    one = sra._cached_runs(folder)[0]
    assert sra._cached_copy(one, None, folder) is None
    assert sra._cached_copy(one, CACHED, folder) == folder / one.filename


def test_without_a_cache_nothing_is_served_from_one(tmp_path):
    """An absent or half-unpacked set is no set."""
    assert sra._cached_runs(tmp_path / "missing") == ()
    assert not sra._has_cached_reads(tmp_path / "missing")
    half = tmp_path / "half"
    half.mkdir()
    (half / sra.CACHED_MANIFEST).write_text("{}")
    assert not sra._has_cached_reads(half)


def test_the_default_cache_is_the_downloaded_example_set(tmp_path,
                                                          monkeypatch,
                                                          no_network):
    """With no folder named, fetch_reads looks in ~/.cache/.../reads."""
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = _cached_set(example_set_folder("reads"))
    assert sra._cached_read_folder() == folder
    one = sra._cached_runs()[1]
    written = sra.fetch_reads(one, tmp_path / "out", max_reads=5)
    assert len(_reads_in(written)) == 5


def test_a_stream_that_stops_early_is_an_error_not_a_short_file(tmp_path):
    """A dropped connection used to leave a short file that looked complete."""
    one = sra.RunFile(run="SRR1", library="lib", mate=1,
                      url="https://example.invalid/SRR1_1.fastq.gz")
    whole = _gzipped(_fastq(500))
    cut = whole[:len(whole) // 3]
    with pytest.raises(ConnectionError):
        sra.fetch_reads(one, tmp_path, max_reads=400,
                        opener=lambda _url: _Response(cut))
    assert not (tmp_path / one.filename).exists()


def test_the_downloaded_archive_unpacks_into_a_usable_cache(tmp_path,
                                                            monkeypatch,
                                                            no_network):
    """``spacr-download reads`` with the Hub mocked leaves a working cache."""
    from spacr import cli_download

    monkeypatch.setenv("HOME", str(tmp_path))
    staged = _cached_set(tmp_path / "staged")

    def fake_download(repo, archive, folder, *, progress=None,
                      chunk_size=None):
        assert (repo, archive) == (READS_EXAMPLE_REPO,
                                   "spacr-example-reads.tar")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / archive
        with tarfile.open(target, "w") as tar:
            for path in sorted(staged.iterdir()):
                tar.add(path, arcname=path.name)
        return target

    monkeypatch.setattr(cli_download, "download_archive", fake_download)
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda path: type("Usage", (), {"total": 10 ** 14, "used": 0,
                                        "free": 10 ** 14})())
    dest = tmp_path / ".cache" / "spacr" / "example_data"
    assert cli_download.main(["reads", "--dest", str(dest)]) == 0
    folder = dest / "reads"
    assert sra._has_cached_reads(folder)
    assert len(sra._cached_runs(folder)) == 4
