"""Stage Map Barcodes' cached read set (the first reads of the published screen).

Source: NCBI BioProject PRJNA1261935, runs SRR33531217-SRR33531220 (the four
sequenced plates ``hilib_p1``-``hilib_p4``, paired), as ENA mirrors them. The
set holds, for each of the eight FASTQ files, its first ``--reads`` reads,
taken by :func:`spacr.sra.fetch_reads` -- the same stream and the same cut
Map Barcodes' Load test data makes online -- and ENA's own listing of the
project (``filereport.tsv``), so the picker lists the same runs without the
network.

Output (``dest``): ``reads/`` (the unpacked set), ``README.md`` (the dataset
card) and ``spacr-example-reads.tar``. ``--upload`` publishes the tar and the
card to the Hugging Face dataset ``einarolafsson/spacr-example-reads``.

Usage: python tools/build_reads_example_dataset.py DEST [--reads 100000]
           [--upload]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE = "spacr-example-reads.tar"
FOLDER = "reads"
REPO = "einarolafsson/spacr-example-reads"
READS = 100_000


def _sha256(path: Path) -> str:
    """Hex SHA-256 of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_reads(path: Path) -> int:
    """Reads in a gzipped FASTQ."""
    import gzip

    with gzip.open(path, "rt") as handle:
        return sum(1 for _ in handle) // 4


def _from_ena(url: str):
    """Open ``url`` at ENA, never the cached set this script is building."""
    request = urllib.request.Request(url, headers={"User-Agent": "spaCR"})
    return urllib.request.urlopen(request, timeout=60)


def build(dest: Path, reads: int = READS) -> dict:
    """Fetch the listing and the first ``reads`` reads of every file."""
    from spacr import sra

    folder = dest / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    query = (f"{sra.ENA_PORTAL}?accession={sra.DEFAULT_BIOPROJECT}"
             f"&result=read_run&fields={','.join(sra.ENA_FIELDS)}&format=tsv")
    request = urllib.request.Request(query, headers={"User-Agent": "spaCR"})
    with urllib.request.urlopen(request, timeout=60) as response:
        (folder / sra.CACHED_LISTING).write_bytes(response.read())
    listing = folder / sra.CACHED_LISTING
    files = sra.runs_for(opener=lambda _url: listing.open("rb"))
    if len(files) != 8:
        raise RuntimeError(f"expected eight FASTQ files, ENA lists {len(files)}")
    entries = {}
    for one in files:
        for attempt in range(4):
            try:
                path = sra.fetch_reads(one, folder, max_reads=reads,
                                       opener=_from_ena)
                break
            except ConnectionError:
                if attempt == 3:
                    raise
        held = _count_reads(path)
        if held != reads:
            raise RuntimeError(f"{path.name}: {held} reads, expected {reads}")
        entries[one.filename] = {
            "run": one.run, "mate": one.mate, "library": one.library,
            "url": one.url, "reads": held, "bytes": path.stat().st_size,
            "sha256": _sha256(path)}
    manifest = {
        "bioproject": sra.DEFAULT_BIOPROJECT,
        "source": "ENA (European Nucleotide Archive) FASTQ mirror",
        "reads_per_file": reads,
        "built_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "listing": sra.CACHED_LISTING,
        "listing_sha256": _sha256(listing),
        "files": entries,
    }
    (folder / sra.CACHED_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


CARD = """\
---
license: other
license_name: insdc-unrestricted
license_link: https://www.insdc.org/policy/
pretty_name: spaCR Map Barcodes test reads (PRJNA1261935, first {reads_k}k reads per file)
tags:
- spacr
- sequencing
- crispr-screen
- toxoplasma
size_categories:
- 100K<n<1M
---

# spaCR — Map Barcodes test reads

The first {reads:,} reads of each of the eight FASTQ files of NCBI BioProject
[{bioproject}](https://www.ncbi.nlm.nih.gov/bioproject/{bioproject}), as the
European Nucleotide Archive serves them. It is the cached copy behind
**Load test data…** on spaCR's Map Barcodes screen: when this set is
downloaded (`spacr-download reads`), the picker lists the runs and takes up
to {reads:,} reads from each file from here, without reaching ENA. A larger
request, or a whole file, still streams from ENA.

{size_line}

## Source

* Study: Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
  EAF1 as a *T. gondii* modulator of ESCRT subversion.
  [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1);
  raw guide-barcode sequencing of the screen's four plates.
* BioProject: {bioproject}; runs {runs} (libraries {libraries}), paired-end.
* Each file is the first {reads:,} reads of the ENA FASTQ at the URL listed in
  `manifest.json`, cut by `spacr.sra.fetch_reads` exactly as Map Barcodes cuts
  it online, and written back out as `.fastq.gz`. Reads are not filtered,
  reordered or edited; the two mates of a run keep their pairing.
* `filereport.tsv` is ENA's portal listing of the project
  (fields {fields}), fetched {built}.

## Files

| file | run | mate | reads | SHA-256 |
|---|---|---|---|---|
{rows}

## Licence

The reads are public INSDC data. Under the INSDC policy
(https://www.insdc.org/policy/) they are available without restriction on
use or distribution; this copy adds none. Please cite the study above when
using them. Built with `tools/build_reads_example_dataset.py` from the spaCR
repository.
"""


def _card(manifest: dict, size_line: str) -> str:
    """The dataset card for ``manifest``."""
    from spacr import sra

    files = manifest["files"]
    runs = sorted({entry["run"] for entry in files.values()})
    libraries = sorted({entry["library"] for entry in files.values()})
    rows = "\n".join(
        f"| `{name}` | {entry['run']} | {entry['mate']} | "
        f"{entry['reads']:,} | `{entry['sha256']}` |"
        for name, entry in sorted(files.items()))
    return CARD.format(
        reads=manifest["reads_per_file"],
        reads_k=manifest["reads_per_file"] // 1000,
        bioproject=manifest["bioproject"], runs=", ".join(runs),
        libraries=", ".join(libraries), fields=", ".join(sra.ENA_FIELDS),
        built=manifest["built_utc"], rows=rows, size_line=size_line)


def _write_tar(dest: Path, folder: Path) -> None:
    """Pack ``folder`` with members relative to it and fixed metadata."""
    with tarfile.open(dest / ARCHIVE, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(folder.rglob("*")):
            if path.is_dir() or path.suffix == ".part":
                continue
            info = tar.gettarinfo(str(path),
                                  arcname=str(path.relative_to(folder)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                tar.addfile(info, handle)


def finish(dest: Path, manifest: dict) -> Path:
    """Write the inner card, the tar and the dataset page's card."""
    folder = dest / FOLDER
    (folder / "README.md").write_text(_card(manifest, (
        f"Downloaded as `{ARCHIVE}` from the dataset page.")), encoding="utf-8")
    _write_tar(dest, folder)
    data = (dest / ARCHIVE).read_bytes()
    line = (f"`{ARCHIVE}`: {len(data):,} bytes ({len(data) / 1e6:.1f} MB), "
            f"one uncompressed tar, SHA-256 "
            f"`{hashlib.sha256(data).hexdigest()}`.")
    (dest / "README.md").write_text(_card(manifest, line), encoding="utf-8")
    return dest / ARCHIVE


def upload(dest: Path) -> None:
    """Publish the tar and the card to :data:`REPO` (a dataset repo)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(dest / ARCHIVE), path_in_repo=ARCHIVE,
                    repo_id=REPO, repo_type="dataset",
                    commit_message="Map Barcodes cached test reads")
    api.upload_file(path_or_fileobj=str(dest / "README.md"),
                    path_in_repo="README.md", repo_id=REPO,
                    repo_type="dataset", commit_message="Dataset card")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", type=Path)
    parser.add_argument("--reads", type=int, default=READS)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    result = build(args.dest, args.reads)
    finish(args.dest, result)
    if args.upload:
        upload(args.dest)
    print(json.dumps({k: v for k, v in result.items() if k != "files"},
                     indent=2))
