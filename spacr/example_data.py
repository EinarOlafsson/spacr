"""Download, validate, and cache the optional example-screen data.

The example CSV files are distributed as release assets rather than package
data. They are downloaded only when requested, validated against the bundled
manifest, and reused from the user's cache on subsequent runs. This module has
no Qt dependency and can also be used from scripts.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .example_data_manifest import FILES

#: Stable release tag that hosts the example assets across spaCR versions.
RELEASE_TAG = "example-data"

#: Base URL for the downloadable example-screen assets.
BASE_URL = (f"https://github.com/EinarOlafsson/spacr/releases/download/"
            f"{RELEASE_TAG}")

#: User-facing descriptions of the two example input types.
KINDS = {
    "counts": "the per-well gRNA read counts",
    "scores": "the per-cell classification scores",
}


class ExampleDataError(RuntimeError):
    """Raised when the example data cannot be downloaded or validated."""


@dataclass(frozen=True)
class Fetched:
    """Paths and download status for a prepared example screen.

    Parameters
    ----------
    counts : list of str
        Cached per-well guide-count tables.
    scores : list of str
        Cached per-cell classification-score tables.
    downloaded : list of str
        Files downloaded during this fetch; other returned files were cached.
    folder : str
        Directory containing the validated files.
    """

    counts: List[str]
    scores: List[str]
    downloaded: List[str]
    folder: str

    @property
    def files(self) -> List[str]:
        """Return all validated count and score table paths."""
        return list(self.counts) + list(self.scores)

    def note(self) -> str:
        """Return a concise status message describing the fetch result."""
        if not self.files:
            return "No example data."
        cached = len(self.files) - len(self.downloaded)
        how = (f"downloaded {len(self.downloaded)}"
               + (f", {cached} already cached" if cached else "")
               if self.downloaded else f"all {cached} already cached")
        return (f"Example screen ready ({how}): "
                f"{len(self.counts)} count table(s) and "
                f"{len(self.scores)} score table(s) in {self.folder}.")


def cache_folder() -> str:
    """Return the directory used to cache example-screen files.

    ``SPACR_EXAMPLE_DATA`` overrides the location. Otherwise the function uses
    ``XDG_CACHE_HOME`` or the platform-neutral ``~/.cache`` fallback.
    """
    override = os.environ.get("SPACR_EXAMPLE_DATA")
    if override:
        return str(Path(override).expanduser())
    base = (os.environ.get("XDG_CACHE_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(base, "spacr", "example_data")


def _digest(path) -> str:
    """Return the file's hexadecimal SHA-256 digest, read in 1 MiB blocks."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def is_whole(path, entry) -> bool:
    """Return whether a file matches one manifest entry.

    :param path: file whose size and SHA-256 digest are to be checked.
    :param entry: manifest mapping containing the expected ``bytes`` and
        ``sha256`` values.

    The inexpensive size check runs before the SHA-256 digest is calculated.
    """
    try:
        if os.path.getsize(path) != int(entry["bytes"]):
            return False
    except OSError:
        return False
    return _digest(path) == entry["sha256"]


def entries_of_kind(kind: Optional[str] = None) -> List[dict]:
    """The manifest entries for one ``kind``, or all of them.

    :param kind: ``"counts"``, ``"scores"``, or ``None`` for everything.
    :raises ValueError: for a kind the manifest does not contain, rather than
        returning an empty list -- a typo would otherwise download nothing and
        report success.
    """
    if kind is None:
        return list(FILES)
    known = {entry["kind"] for entry in FILES}
    if kind not in known:
        raise ValueError(
            f"no example files of kind {kind!r}; the manifest has "
            f"{sorted(known)}")
    return [entry for entry in FILES if entry["kind"] == kind]


def missing(folder=None, kind: Optional[str] = None) -> List[dict]:
    """Return manifest entries absent or invalid in ``folder``.

    :param kind: restrict to one kind. Regression can fetch its counts and its
        scores separately, because a user checking one of them should not wait
        for the other.
    """
    where = folder or cache_folder()
    return [entry for entry in entries_of_kind(kind)
            if not is_whole(os.path.join(where, entry["name"]), entry)]


def total_bytes(entries: Optional[Sequence[dict]] = None) -> int:
    """Return the total expected size of the selected manifest entries."""
    return sum(int(e["bytes"]) for e in (FILES if entries is None else entries))


def _download(entry, folder, progress=None, cancelled=None) -> str:
    """Download and validate one asset, then return its final path.

    Data is written to a sibling ``.part`` file and atomically renamed only
    after its size and digest match the manifest. Interrupted or invalid
    downloads are removed.
    """
    from urllib.error import URLError
    from urllib.request import urlopen

    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, entry["name"])
    partial = target + ".part"
    url = f"{BASE_URL}/{entry['name']}"
    seen = 0
    try:
        with urlopen(url, timeout=60) as response, \
                open(partial, "wb") as handle:
            while True:
                if cancelled is not None and cancelled():
                    raise ExampleDataError(
                        f"the download of {entry['name']} was cancelled")
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                seen += len(block)
                if progress is not None:
                    progress(entry["name"], seen, int(entry["bytes"]))
    except ExampleDataError:
        _forget(partial)
        raise
    except (URLError, OSError) as error:
        _forget(partial)
        raise ExampleDataError(
            f"could not download {entry['name']} from {url}: {error}. The "
            f"example screen is a release asset, so this needs a network "
            f"connection the first time.") from error

    if not is_whole(partial, entry):
        got = os.path.getsize(partial) if os.path.exists(partial) else 0
        _forget(partial)
        raise ExampleDataError(
            f"{entry['name']} arrived incomplete or corrupt: expected "
            f"{entry['bytes']} bytes, got {got}. Nothing was kept, so "
            f"pressing again is safe.")
    os.replace(partial, target)
    return target


def _forget(path) -> None:
    """Best-effort remove ``path``, including when it is already absent."""
    try:
        os.remove(path)
    except OSError:
        pass


def fetch(folder=None, *, progress: Optional[Callable] = None,
          cancelled: Optional[Callable] = None,
          download: bool = True,
          kind: Optional[str] = None) -> Fetched:
    """Prepare the example screen, downloading only missing files.

    Parameters
    ----------
    folder : path-like, optional
        Cache directory. The standard example cache is used when omitted.
    progress : callable, optional
        Called as ``progress(name, received_bytes, total_bytes)`` while each
        file downloads.
    cancelled : callable, optional
        Zero-argument callback. A true result cancels the active download.
    download : bool, default=True
        If false, require every file to be present in the cache and do not use
        the network.

    Returns
    -------
    Fetched
        Validated count and score paths plus download status.

    Raises
    ------
    ExampleDataError
        If a file cannot be downloaded, validation fails, the operation is
        cancelled, or downloading is disabled while files are missing.
    """
    where = folder or cache_folder()
    wanted = entries_of_kind(kind)
    absent = missing(where, kind)
    if absent and not download:
        raise ExampleDataError(
            f"{len(absent)} of the {len(wanted)} example files are not cached "
            f"in {where}, and downloading was not allowed.")
    got: List[str] = []
    for entry in absent:
        got.append(_download(entry, where, progress, cancelled))

    # REPORTED FOR THE REQUESTED KIND ONLY. Listing a path for a file that was
    # never asked for -- and so may not be on disk -- would hand the caller a
    # name it cannot open.
    by_kind: Dict[str, List[str]] = {"counts": [], "scores": []}
    for entry in wanted:
        by_kind[entry["kind"]].append(os.path.join(where, entry["name"]))
    return Fetched(counts=sorted(by_kind["counts"]),
                   scores=sorted(by_kind["scores"]),
                   downloaded=got, folder=where)
