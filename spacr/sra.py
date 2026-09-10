"""Fetch sequencing reads for the published screen straight from ENA/NCBI.

Map Barcodes can fetch the paper's own sequencing data directly. Each download
accepts a read limit per file, so a small representative subset does not
require transferring the complete archive.

The reads are NCBI BioProject :data:`DEFAULT_BIOPROJECT`, runs
SRR33531217-SRR33531220 -- the four plates, paired, named ``hilib_p1`` through
``hilib_p4`` in the submission.

WHY ENA AND NOT NCBI'S OWN DOWNLOAD. The three routes are not equivalent
here. ``fasterq-dump`` needs the SRA toolkit installed, which spaCR does not
ship and cannot assume. NCBI's own FASTQ endpoint does not support fetching a
prefix. ENA mirrors every SRA submission as plain gzipped FASTQ over HTTPS,
which is what makes the read limit meaningful rather than cosmetic: the file
is read as a STREAM and the connection is dropped as soon as enough reads have
arrived.

That distinction is the whole feature. The four runs are 2.2-3.2 GB per mate,
about 19 GB in total, and 60-74 million reads each. Measured against the live
archive on 2026-09-01: 3,662 reads arrived in 0.13 MB and 1.7 seconds from a
2,833 MB file. Asking for a hundred thousand reads costs a few megabytes, not
a few gigabytes, so a laptop on hotel wifi can have a real subset of the real
screen in under a minute.
"""
from __future__ import annotations

import gzip
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

#: The paper's raw sequencing. "Pooled image-based CRISPR screening identifies
#: EAF1 as a regulator of host ESCRT recruitment by Toxoplasma gondii".
DEFAULT_BIOPROJECT = "PRJNA1261935"

#: ENA's metadata endpoint. Returns TSV; asked for the fields below.
ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api/filereport"

#: Requested from the portal, in this order.
ENA_FIELDS = ("run_accession", "library_name", "fastq_ftp", "fastq_bytes",
              "read_count")

#: Four lines to a FASTQ record. Named because the read limit is expressed in
#: READS and the stream is split on lines, and confusing the two is a factor
#: of four in what the user gets.
LINES_PER_READ = 4

#: How much of the compressed stream to pull at a time.
CHUNK_BYTES = 65536


@dataclass(frozen=True)
class RunFile:
    """One downloadable FASTQ: a run, and which mate of the pair.

    :param run: archive run accession shared by its mate files.
    :param library: experiment library name used to identify the source plate.
    :param url: HTTPS location of the compressed FASTQ stream.
    :param mate: one-based mate number within a paired-end run.
    :param size_bytes: archive-reported compressed size, or zero when unknown.
    :param read_count: archive-reported reads in the run, or zero when unknown.
    """

    run: str
    library: str
    url: str
    mate: int
    size_bytes: int = 0
    read_count: int = 0

    @property
    def filename(self) -> str:
        """What it is saved as: the archive's own name."""
        return self.url.rsplit("/", 1)[-1]

    def label(self) -> str:
        """A one-line description for a picker.

        Names the LIBRARY as well as the run, because ``hilib_p3`` says which
        plate this is and ``SRR33531218`` does not.
        """
        size = f"{self.size_bytes / 1e9:.1f} GB" if self.size_bytes else "?"
        reads = f"{self.read_count / 1e6:.0f}M reads" if self.read_count else ""
        parts = [self.run, f"mate {self.mate}", self.library, size, reads]
        return "  ·  ".join(p for p in parts if p)


def _read_url(url: str, timeout: float, opener=None):
    """Open ``url``. ``opener`` is the seam every test uses instead of a socket."""
    if opener is not None:
        return opener(url)
    request = urllib.request.Request(url, headers={"User-Agent": "spaCR"})
    return urllib.request.urlopen(request, timeout=timeout)


def runs_for(accession: str = DEFAULT_BIOPROJECT, *, timeout: float = 30.0,
             opener=None) -> tuple[RunFile, ...]:
    """Every FASTQ in ``accession``, newest ENA metadata.

    :param accession: a BioProject (``PRJNA...``) or a single run (``SRR...``).
    :param opener: replaces the network call; receives the URL and returns a
        file-like object of TSV bytes.
    :returns: one :class:`RunFile` per mate per run, ordered by run then mate.
    """
    query = (f"{ENA_PORTAL}?accession={accession}&result=read_run"
             f"&fields={','.join(ENA_FIELDS)}&format=tsv")
    with _read_url(query, timeout, opener) as response:
        payload = response.read()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", "replace")

    rows = [line for line in payload.splitlines() if line.strip()]
    if not rows:
        return ()
    header = rows[0].split("\t")
    index = {name: position for position, name in enumerate(header)}
    if "fastq_ftp" not in index:
        return ()

    files: list[RunFile] = []
    for row in rows[1:]:
        cells = row.split("\t")

        def cell(name: str, row_cells=cells) -> str:
            """Read one named value from the bound portal row.

            :param name: ENA header name to look up in the captured index.
            :param row_cells: cells bound when this row's helper is created,
                preventing later loop iterations from changing the source row.
            :returns: the indexed cell, or an empty string when the column is
                absent or the row is too short.
            """
            position = index.get(name, -1)
            return (row_cells[position]
                    if 0 <= position < len(row_cells) else "")

        urls = [u for u in cell("fastq_ftp").split(";") if u]
        sizes = [s for s in cell("fastq_bytes").split(";") if s]
        try:
            reads = int(cell("read_count") or 0)
        except ValueError:
            reads = 0
        for mate, url in enumerate(urls, start=1):
            try:
                size = int(sizes[mate - 1]) if mate - 1 < len(sizes) else 0
            except ValueError:
                size = 0
            # The portal returns a bare host/path. HTTPS rather than FTP:
            # FTP is blocked on many institutional networks and is the reason
            # a "download failed" here would be unexplainable.
            files.append(RunFile(
                run=cell("run_accession"), library=cell("library_name"),
                url=url if url.startswith("http") else f"https://{url}",
                mate=mate, size_bytes=size, read_count=reads))
    files.sort(key=lambda f: (f.run, f.mate))
    return tuple(files)


def fetch_reads(run_file: RunFile, destination, *, max_reads: Optional[int] = None,
                timeout: float = 60.0, opener=None,
                progress: Optional[Callable[[int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None) -> Path:
    """Download ``run_file`` into ``destination``, stopping after ``max_reads``.

    The stream is decompressed as it arrives and the connection is dropped the
    moment enough reads are in hand, so a limited request transfers only what
    it needs. ``max_reads`` of ``None`` fetches the whole file.

    Written back out as ``.fastq.gz`` because that is what the ``src`` setting
    documents for sequencing ("the folder of .fastq.gz reads"), so a subset
    and a full download are the same shape to everything downstream.

    :param progress: called with ``(reads_so_far, compressed_bytes_so_far)``.
    :param should_stop: polled between chunks; a truthy answer abandons the
        download and removes the partial file.
    :returns: the path written.
    :raises ValueError: for a non-positive ``max_reads``.
    """
    if max_reads is not None and max_reads <= 0:
        raise ValueError(f"max_reads must be positive, got {max_reads!r}")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / run_file.filename

    wanted_lines = None if max_reads is None else max_reads * LINES_PER_READ
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    written_lines = 0
    fetched = 0

    # A PARTIAL FILE IS WORSE THAN NO FILE: it looks like a finished download
    # to everything that lists the folder. Written beside the target and moved
    # only on success.
    part = target.with_suffix(target.suffix + ".part")
    try:
        with _read_url(run_file.url, timeout, opener) as response, \
                gzip.open(part, "wb") as out:
            while True:
                if should_stop is not None and should_stop():
                    raise InterruptedError("cancelled")
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                fetched += len(chunk)
                pending += decompressor.decompress(chunk)
                *complete, pending = pending.split(b"\n")
                if wanted_lines is not None:
                    room = wanted_lines - written_lines
                    if len(complete) >= room:
                        complete = complete[:room]
                for line in complete:
                    out.write(line + b"\n")
                written_lines += len(complete)
                if progress is not None:
                    progress(written_lines // LINES_PER_READ, fetched)
                if wanted_lines is not None and written_lines >= wanted_lines:
                    break
            # The tail, only when the whole file was asked for -- a truncated
            # request must not end on half a record.
            if wanted_lines is None and pending:
                out.write(pending)
        part.replace(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return target


def total_bytes(files: Iterable[RunFile]) -> int:
    """What downloading all of ``files`` in full would cost."""
    return sum(f.size_bytes for f in files)


def estimated_bytes(files: Sequence[RunFile], max_reads: Optional[int]) -> int:
    """Roughly what ``max_reads`` from each of ``files`` will transfer.

    Scaled from each run's own read count rather than from a fixed rate, so a
    run with longer reads is not under-quoted. Returns the full size when no
    limit is set, and when a run does not report its read count -- guessing low
    there would understate a multi-gigabyte download.
    """
    if max_reads is None:
        return total_bytes(files)
    estimate = 0
    for one in files:
        if one.read_count > 0 and one.size_bytes > 0:
            share = min(1.0, max_reads / one.read_count)
            estimate += int(one.size_bytes * share)
        else:
            estimate += one.size_bytes
    return estimate


__all__ = ["DEFAULT_BIOPROJECT", "RunFile", "runs_for", "fetch_reads",
           "total_bytes", "estimated_bytes"]
