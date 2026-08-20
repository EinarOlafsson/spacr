"""Fetch the example screen on demand, and say what arrived.

Instruction 191. Asked for on 2026-08-20 as "add my cound and dependent
variable csvs to the datafolder in spacr and add a button that auto loads
them into the correct slots", and settled -- once the numbers were on the
table -- as a download:

    the eight CSVs                      33 MB
    github.com/EinarOlafsson/spacr      PUBLIC
    setup.py package_data               ships resources/data/*

Committing them would have put 33 MB into every ``pip install spacr`` and
into public git history permanently, which deleting the files later does not
undo. A RELEASE ASSET sits outside the git objects and outside the wheel, and
is fetched only by someone who asks for it.

THE CACHE IS NOT THE PACKAGE. A pip-installed spaCR can sit in a read-only
site-packages, and writing 33 MB into someone's virtualenv is not a thing a
button should do anyway. Downloads land in the user's own cache directory and
are reused, so the second press costs nothing.

Nothing here imports Qt. The button is a caller.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .example_data_manifest import FILES

#: The release the assets hang off. A DEDICATED TAG, not a version tag:
#: re-releasing spaCR must not orphan the data, and the URL has to stay valid
#: across versions.
RELEASE_TAG = "example-data"

#: Where the assets are fetched from.
BASE_URL = (f"https://github.com/EinarOlafsson/spacr/releases/download/"
            f"{RELEASE_TAG}")

#: What the two kinds of file are FOR, which is what the button fills in.
KINDS = {
    "counts": "the per-well gRNA read counts",
    "scores": "the per-cell classification scores",
}


class ExampleDataError(RuntimeError):
    """The example screen could not be produced, and the message says why."""


@dataclass(frozen=True)
class Fetched:
    """What a fetch produced: the files, and what had to be downloaded."""

    counts: List[str]
    scores: List[str]
    downloaded: List[str]
    folder: str

    @property
    def files(self) -> List[str]:
        return list(self.counts) + list(self.scores)

    def note(self) -> str:
        """One line for the status bar."""
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
    """Where the example screen is kept between runs.

    Honours ``SPACR_EXAMPLE_DATA`` first, so a test -- or a user on a machine
    that already holds the screen -- can point this anywhere without a
    download.
    """
    override = os.environ.get("SPACR_EXAMPLE_DATA")
    if override:
        return str(Path(override).expanduser())
    base = (os.environ.get("XDG_CACHE_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(base, "spacr", "example_data")


def _digest(path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def is_whole(path, entry) -> bool:
    """Whether the file on disk is the file the manifest describes.

    SIZE FIRST, because it rejects a truncated download for the price of a
    stat, and a truncated download is the common failure. The digest is only
    read when the size already agrees.
    """
    try:
        if os.path.getsize(path) != int(entry["bytes"]):
            return False
    except OSError:
        return False
    return _digest(path) == entry["sha256"]


def missing(folder=None) -> List[dict]:
    """The manifest entries not already present and whole in ``folder``."""
    where = folder or cache_folder()
    return [entry for entry in FILES
            if not is_whole(os.path.join(where, entry["name"]), entry)]


def total_bytes(entries: Optional[Sequence[dict]] = None) -> int:
    """How much a fetch would move."""
    return sum(int(e["bytes"]) for e in (FILES if entries is None else entries))


def _download(entry, folder, progress=None, cancelled=None) -> str:
    """Fetch one asset. Returns the path written.

    WRITTEN BESIDE AND RENAMED, never in place: an interrupted download that
    left a half file under the real name would be found by `is_whole` on the
    next press, rejected, and downloaded again -- which is merely wasteful --
    but a crash between the size check and the digest would hand a caller a
    truncated CSV. A rename is atomic and cannot.
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
    try:
        os.remove(path)
    except OSError:
        pass


def fetch(folder=None, *, progress: Optional[Callable] = None,
          cancelled: Optional[Callable] = None,
          download: bool = True) -> Fetched:
    """Make the example screen available, downloading only what is absent.

    :param progress: called ``(name, seen_bytes, total_bytes)`` while a file
        is arriving. 33 MB on a slow connection is not instant, and a button
        that appears to hang is worse than one that refuses.
    :param cancelled: called with no arguments; a true answer stops the fetch.
    :param download: when False, use only what is already cached and refuse
        rather than reaching for the network.
    :raises ExampleDataError: with a message naming the file and the reason.
    """
    where = folder or cache_folder()
    absent = missing(where)
    if absent and not download:
        raise ExampleDataError(
            f"{len(absent)} of the {len(FILES)} example files are not cached "
            f"in {where}, and downloading was not allowed.")
    got: List[str] = []
    for entry in absent:
        got.append(_download(entry, where, progress, cancelled))

    by_kind: Dict[str, List[str]] = {"counts": [], "scores": []}
    for entry in FILES:
        by_kind[entry["kind"]].append(os.path.join(where, entry["name"]))
    return Fetched(counts=sorted(by_kind["counts"]),
                   scores=sorted(by_kind["scores"]),
                   downloaded=got, folder=where)
