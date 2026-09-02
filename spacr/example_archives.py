"""The published example datasets: what they are, and how to fetch them.

WHY THIS IS NOT IN :mod:`spacr.qt.hf_download`. That module owns the same
downloads with a Qt progress dialog wrapped round them, and it imports PySide6
at module scope. ``spacr-download`` has to work on a cluster login node with no
display and no Qt installed at all, so everything that is only about the DATA --
which repositories publish it, what each archive is called, how a stream is
verified, how an archive is safely unpacked, and what has to be repaired on
arrival -- lives here instead. ``hf_download`` imports these names back out, so
every existing GUI caller (and every test that patches one of them on that
module) keeps working unchanged.

NOT TO BE CONFUSED WITH :mod:`spacr.example_data`, which fetches the Regression
example screen's count and score CSVs from a GitHub release. That is a
different set of files, a different host and a different transport. This module
is about the four ``.tar`` archives of imaging data: the Mask demo plate, the
Measure example, the Annotate/Classify example, and -- through
:mod:`spacr.screen_data` -- the pieces of the published TSG101 screen.

SIZES ARE STATED BEFORE ANYTHING IS FETCHED. A user choosing between example
sets is choosing how many gigabytes to spend, and a picker (or a CLI) that
lists names without sizes makes that choice blind. :data:`EXAMPLE_SETS` carries
one size per set for the same reason :class:`spacr.screen_data.ScreenAsset`
does.
"""
from __future__ import annotations

import logging
import os
import socket
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("spacr.example_archives")

__all__ = [
    "ANNOTATE_EXAMPLE_REPO",
    "DATASET_PLACEHOLDER",
    "DATASET_REPO",
    "DATASET_SUB",
    "EXAMPLE_ARCHIVES",
    "EXAMPLE_SETS",
    "ExampleSet",
    "MEASURE_EXAMPLE_REPO",
    "SETTINGS_REPO",
    "download_archive",
    "example_plate_folder",
    "example_set",
    "expand_measure_arrays",
    "explain_download_failure",
    "extract_example_archive",
    "make_the_example_paths_absolute",
]


# Match the classic Tk GUI's demo endpoints so users see the same
# dataset here they'd have seen in the Tk build.
DATASET_REPO  = "einarolafsson/toxo_mito"
DATASET_SUB   = "plate1"
SETTINGS_REPO = "einarolafsson/spacr_settings"

#: Measure's own example data: the merged arrays a Mask run produces, so
#: Measure can be exercised without segmenting anything first.
#:
#: A SEPARATE REPO from the Mask demo because it is a different artefact at a
#: different stage -- `toxo_mito` is raw acquisition, this is that plate after
#: `preprocess_generate_masks`. Sixteen fields across four wells; the wells are
#: all kept because well-level aggregation and between-condition comparison
#: are most of what Measure does after the per-object step.
MEASURE_EXAMPLE_REPO = "einarolafsson/spacr-example-measure"

#: Annotate and Classify share one example set: the crops a Measure run cut,
#: the measurements database that indexes them, and 88 real labels.
#:
#: ONE REPO, TWO SETTINGS FILES. Both modules read the same 282 MB of crops and
#: the same database; only the settings differ. Publishing it twice would
#: double the download and let the two copies drift.
ANNOTATE_EXAMPLE_REPO = "einarolafsson/spacr-example-annotate"

#: The token a published settings file uses for "wherever this was unpacked".
DATASET_PLACEHOLDER = "<dataset>"


def example_plate_folder() -> Path:
    """The ONE folder every example dataset unpacks into.

    ``~/.cache/spacr/example_data/plate1`` -- a real spaCR plate directory,
    holding whichever of ``merged/``, ``data/``, ``measurements/`` and
    ``settings/`` have been downloaded.

    ONE FOLDER BECAUSE THE SETS ARE USED TOGETHER. `data/` is the crops and
    `measurements/measurements.db` is what indexes them; downloading them into
    separate trees meant the two halves of one plate could not be opened at
    once, and the user had to know which download had put what where. Each
    archive's members are relative to this folder, so the three unpack into it
    side by side and compose into a plate that Measure, Annotate and Classify
    can all be pointed at.
    """
    return Path.home() / ".cache" / "spacr" / "example_data" / "plate1"


#: The single archive each example repo ships, keyed by repo.
#:
#: ONE REQUEST INSTEAD OF THOUSANDS. The annotate set is 2,365 files; fetching
#: it a file at a time spent most of its wall clock on HTTP round trips, and
#: `snapshot_download` -- the obvious alternative -- cannot be interrupted, so
#: Cancel did nothing and quitting mid-download aborted the process.
#:
#: A tar fixes all three: one stream that can be stopped between chunks, one
#: progress figure that means something, and no per-file overhead at either
#: end. It is NOT compressed: the payloads are PNGs and .npz arrays, already
#: compressed, so gzip would cost minutes of CPU on every download to save
#: almost nothing.
EXAMPLE_ARCHIVES: Dict[str, str] = {
    DATASET_REPO: "spacr-example-mask.tar",
    MEASURE_EXAMPLE_REPO: "spacr-example-measure.tar",
    ANNOTATE_EXAMPLE_REPO: "spacr-example-annotate.tar",
}


@dataclass(frozen=True)
class ExampleSet:
    """One published example dataset, described before it is downloaded.

    :param key: what a user types to ask for this one.
    :param repo: the Hugging Face dataset repository it is published in.
    :param summary: one line saying what the set is and what it is for.
    :param bytes: the archive's size. APPROXIMATE, and deliberately so: these
        are the figures the GUI buttons already state ("about 280 MB"), taken
        from the archives at publication. They exist to let someone decide
        whether to spend the download, not to be checked against. The exact
        length is verified during the transfer, against what the server
        declares.
    :param markers: what the unpacked set LEAVES BEHIND, as globs relative to
        the folder it unpacks into. Presence is tested on these rather than on
        the folder, because all three sets share one plate directory: the
        directory existing says nothing about which of them is in it.
    :param expands_npz: whether ``.npz`` arrays have to be written back out as
        the ``.npy`` Measure reads. See :func:`expand_measure_arrays`.
    """

    key: str
    repo: str
    summary: str
    bytes: int
    markers: Tuple[str, ...]
    expands_npz: bool = False

    @property
    def archive(self) -> str:
        """The ``.tar`` this set ships as."""
        return EXAMPLE_ARCHIVES[self.repo]

    def is_present(self, folder) -> bool:
        """Whether this set is already unpacked under ``folder``.

        Every marker has to match. A half-unpacked set -- a transfer that
        died between the images and the database -- is therefore reported as
        absent, which is the answer that gets it repaired.

        :param folder: the plate directory the set unpacks into.
        """
        folder = Path(folder)
        return all(any(folder.glob(pattern)) for pattern in self.markers)


#: Every example set, in the order the pipeline runs them.
#:
#: THE ORDER IS THE PIPELINE'S. Mask segments raw images into merged stacks,
#: Measure cuts crops out of those stacks and measures them, Annotate and
#: Classify label the crops. Someone downloading all three is following that
#: sequence, and a list in any other order would have to be re-sorted in the
#: reader's head.
EXAMPLE_SETS: Tuple[ExampleSet, ...] = (
    ExampleSet(
        key="mask",
        repo=DATASET_REPO,
        summary="Mask demo: one raw toxo_mito plate, plus the settings to "
                "segment it.",
        bytes=400_000_000,
        markers=("*.tif",),
    ),
    ExampleSet(
        key="measure",
        repo=MEASURE_EXAMPLE_REPO,
        summary="Measure example: sixteen merged fields across four wells, "
                "as a Mask run leaves them.",
        bytes=390_000_000,
        markers=("merged/*.npy",),
        expands_npz=True,
    ),
    ExampleSet(
        key="annotate",
        repo=ANNOTATE_EXAMPLE_REPO,
        summary="Annotate / Classify example: 2,341 single-cell crops, the "
                "database that indexes them, and 88 real labels.",
        bytes=280_000_000,
        markers=("measurements/measurements.db", "data"),
    ),
)


def example_set(key: str) -> ExampleSet:
    """The set called ``key``.

    :param key: ``mask``, ``measure`` or ``annotate``.
    :raises KeyError: naming the keys that do exist. A typo that returned
        ``None`` would download nothing and report success, which is the one
        outcome a download command must never produce.
    """
    for candidate in EXAMPLE_SETS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"no example set named {key!r}; there is "
                   f"{', '.join(s.key for s in EXAMPLE_SETS)}")


def explain_download_failure(exc: BaseException) -> str:
    """Turn a download exception into something a user can act on.

    This is the only demo in the Demos menu that needs the network — the six
    synthetic generators are entirely offline — so it is the only one that can
    fail for a reason outside spaCR. What the user saw before was
    ``str(exc)``, which for the ordinary offline case is a nested urllib3
    dump::

        (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443):
        Max retries exceeded with url: /api/datasets/... (Caused by
        NewConnectionError('<urllib3.connection.HTTPSConnection object at
        0x7e8d...>: Failed to establish a new connection: [Errno 101] Network
        is unreachable'))"), '(Request ID: 73ac20ed-...)')

    — 300 characters that never say "you are offline" and never say what to do
    instead. The three conditions this actually fails on are: no network, the
    ``huggingface_hub`` extra not installed, and a truncated transfer. Each
    gets a sentence naming the cause and the way out; anything else keeps its
    own message with the same closing advice attached.

    :param exc: the exception raised inside the download worker.
    :returns: a multi-line message for the failure dialog.
    """
    offline_hint = (
        "Every other entry in the Demos menu is synthetic and runs with no "
        "network at all — use one of those to try the pipelines offline.")

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            "The real-dataset demo needs the 'huggingface_hub' package to "
            "list the demo repository, and it is not installed in this "
            f"environment ({exc}).\n\n"
            "Install it with:  pip install huggingface_hub\n\n"
            + offline_hint)

    # The truncation check comes first: `IOError` IS `OSError`, and the
    # builtin ConnectionError below is an OSError subclass, so ordering these
    # the other way round would let a half-finished transfer be reported as
    # "check your internet connection" — true but useless, because the
    # connection was fine right up to the point it was not.
    if isinstance(exc, OSError) and "Truncated download" in str(exc):
        return (
            f"{exc}\n\n"
            "The connection dropped part-way through. Nothing partial was "
            "kept, so re-running the demo starts the file again.\n\n"
            + offline_hint)

    # requests is an install-time dependency of huggingface_hub, but the
    # import is kept local so a broken environment reports the missing
    # package above rather than dying here. The builtins are in the tuple
    # too: `requests.exceptions.ConnectionError` descends from OSError, not
    # from the builtin ConnectionError, and a DNS failure raised by anything
    # other than requests (urllib, socket, huggingface_hub's own client)
    # arrives as one of these instead.
    network_errors: tuple = (ConnectionError, TimeoutError, socket.gaierror)
    try:
        import requests
        network_errors += (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
    except Exception:
        pass

    if isinstance(exc, network_errors):
        return (
            "Could not reach huggingface.co, so the real demo dataset could "
            "not be downloaded. Check your internet connection (or your "
            "proxy settings) and try again.\n\n"
            + offline_hint)

    return f"{exc}\n\n{offline_hint}"


def _list_files(repo_id: str, subfolder: str) -> List[str]:
    """Return every file path in ``repo_id`` matching ``subfolder``.

    Empty subfolder means "top-level CSVs only" (mirrors the Tk
    downloader's behaviour for the settings pack).

    :raises ImportError: when ``huggingface_hub`` is not installed. Re-raised
        with the package named rather than letting the bare
        ``ModuleNotFoundError`` text stand on its own, because
        :func:`explain_download_failure` turns it into install instructions
        and the message is what the user reads.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError(f"huggingface_hub is not installed: {exc}") from exc
    files = list_repo_files(repo_id, repo_type="dataset")
    if subfolder:
        return [f for f in files if f.startswith(subfolder)]
    return [f for f in files if f.endswith(".csv")]


def _content_length(resp) -> Optional[int]:
    """Declared body size from the response, or None when unusable.

    Hugging Face always sends ``Content-Length`` for a resolved LFS
    object, so this doubles as the integrity check for
    :func:`_download_one`: fewer bytes on disk than advertised means the
    stream was cut short.
    """
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def download_archive(repo_id: str, file_name: str, dest_dir: Path, *,
                     progress: Optional[Callable[[int, Optional[int]], None]]
                     = None,
                     chunk_size: int = 1 << 15) -> Path:
    """Stream one file from the HF repo to ``dest_dir/basename``.

    Uses plain HTTP + streaming so we don't need the full ``hf_hub``
    download machinery (and its cache dir) for a one-shot demo pull.

    The body lands in a sibling ``.part`` file and is only moved onto
    the final path once every advertised byte has arrived. Writing
    straight to the destination meant a dropped connection left a
    truncated image behind that was indistinguishable from a good
    download — the next pipeline run then failed deep inside the mask
    stage instead of at the download.

    :param repo_id: the Hugging Face dataset repository.
    :param file_name: the path within it; only the basename is kept on disk.
    :param dest_dir: the folder the file lands in.
    :param progress: called as ``progress(written, expected)`` after each
        chunk, where ``expected`` is ``None`` when the server declared no
        length. A CLI draws a percentage from it; the GUI does not use it,
        because its workers emit Qt signals from inside their own loops.
    :param chunk_size: 32 KB by default, which is the size the per-file demo
        pull has always used. An archive of gigabytes passes a bigger one.
    """
    import requests
    url = (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
             f"{file_name}?download=true")
    dst = Path(dest_dir) / Path(file_name).name
    part = dst.with_name(dst.name + ".part")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    expected = _content_length(resp)
    written = 0
    try:
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, expected)
        if expected is not None and written != expected:
            raise IOError(
                f"Truncated download for {file_name}: wrote {written} "
                f"bytes but the server declared {expected}."
            )
        os.replace(part, dst)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return dst


#: The name the Qt downloader has always called it by.
#:
#: Kept as an alias rather than renamed at every call site because
#: ``monkeypatch.setattr(hf_download, "_download_one", ...)`` is how several
#: tests stop the demo flow reaching the network. Renaming it would leave
#: those patches naming an attribute nothing calls, and the tests would go
#: to the network for real.
_download_one = download_archive


def extract_example_archive(archive, dest) -> int:
    """Unpack a downloaded example archive under ``dest``.

    EXTRACTED WITH ``filter="data"``, which is the whole reason this is a
    function rather than two lines at the call site. A tar can name
    ``../../etc/something`` or an absolute path, and a plain ``extractall``
    will happily write there -- so unpacking downloaded content without a
    filter hands whoever can publish to the repo a write anywhere the user can
    write. The filter rejects those members, along with device nodes, setuid
    bits and symlinks pointing outside the tree.

    Python 3.12 and later have it built in. Older interpreters get an explicit
    check instead of a silent unfiltered unpack.

    :param archive: the ``.tar`` on disk.
    :param dest: the folder to unpack into.
    :returns: how many members were written.
    """
    import tarfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive)) as tar:
        members = tar.getmembers()
        if hasattr(tarfile, "data_filter"):
            tar.extractall(str(dest), filter="data")
        else:
            # No filter available: refuse anything that leaves the tree rather
            # than trusting the archive.
            for member in members:
                name = member.name
                if name.startswith("/") or ".." in name.split("/"):
                    raise ValueError(
                        f"refusing to unpack {name!r}: it escapes the "
                        f"destination folder")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(
                        f"refusing to unpack {name!r}: it is not a plain file "
                        f"or directory")
            tar.extractall(str(dest))
    return len(members)


def expand_measure_arrays(merged: Path) -> None:
    """Write each ``.npz`` back out as the ``.npy`` Measure reads.

    The compression is a TRANSPORT detail -- it halves a 700 MB download -- and
    Measure loads `.npy`. Converting on arrival keeps that entirely inside the
    downloader rather than teaching every reader about a second format.

    The ``.npz`` is removed afterwards: keeping both doubles the disk cost of
    an example dataset for a file nothing will open again.

    A MODULE FUNCTION, NOT A METHOD, and that is the point. `after_extract`
    runs on the download thread, and reaching this code through
    ``_MeasureExampleWorker(dest)._expand_arrays(...)`` CONSTRUCTED a QObject
    there purely to borrow a helper. `thread_guard` reported it exactly as it
    should have: the object then lived on 'Dummy-6' and every later touch from
    the GUI thread was illegal. Nothing in here ever read ``self``, so there
    was never an object to need.

    :param merged: the folder the ``.npz`` arrays were unpacked into.
    """
    import numpy as np


    if not merged.is_dir():
        return
    for archive in sorted(merged.glob("*.npz")):
        target = archive.with_suffix(".npy")
        if target.is_file():
            archive.unlink(missing_ok=True)
            continue
        try:
            with np.load(archive) as bundle:
                # Written by the publisher under `image`; the first key is
                # the fallback so a hand-made archive still loads.
                key = "image" if "image" in bundle else bundle.files[0]
                np.save(target, bundle[key])
            archive.unlink(missing_ok=True)
        except Exception:                                # noqa: BLE001
            # One bad archive must not cost the other fifteen. It is left
            # on disk, so what failed is visible rather than merely absent.
            LOG.warning("could not unpack %s", archive, exc_info=True)


def make_the_example_paths_absolute(root) -> int:
    """Point a downloaded example at where it actually landed.

    A measurements database stores ABSOLUTE paths to its crops, which name the
    machine that made it and resolve nowhere else. The published copy stores
    them relative to the dataset root instead, so it is portable and carries no
    account name -- and this is what turns them back into paths that open.

    The settings files are rewritten the same way: they carry
    :data:`DATASET_PLACEHOLDER` where the unpack location goes, so a user can
    press Run without first editing a path.

    Idempotent. A path that is already absolute is left alone, so running this
    twice -- a re-download over an existing copy -- does not produce
    ``/home/me/data//home/me/data/...``.

    :param root: the folder the dataset was unpacked into.
    :returns: how many values were rewritten.
    """
    from .database_concurrency import connect

    root = Path(root)
    prefix = str(root).rstrip("/") + "/"
    rewritten = 0

    # WHEREVER THE DATABASE IS. spaCR keeps it at `measurements/measurements.db`
    # inside a plate; the published archive used to carry it at the top. Both
    # are checked so an already-unpacked older copy is still repaired.
    for database in (root / "measurements" / "measurements.db",
                     root / "measurements.db"):
        if database.is_file():
            break
    if database.is_file():
        # THE HOUSE CONNECT, for its busy timeout. This ran without one for as
        # long as it lived in `spacr/qt/hf_download.py`, where the
        # concurrency audit does not look. Moving it here put it in scope and
        # the audit caught it immediately: an untimed connection raises
        # "database is locked" the instant a Measure writer holds the file,
        # rather than waiting for it -- and this runs right after an example
        # unpacks, which is exactly when something else may be opening the
        # same database.
        connection = connect(database)
        try:
            tables = [r[0] for r in connection.execute(
                "select name from sqlite_master where type='table'")]
            for table in tables:
                columns = [r[1] for r in connection.execute(
                    f'PRAGMA table_info("{table}")')]
                for column in columns:
                    try:
                        # Only the values that look like OUR relative paths.
                        # A column holding prose is untouched, and one already
                        # absolute is skipped by the same test.
                        cursor = connection.execute(
                            f'update "{table}" set "{column}" = ? || "{column}" '
                            f'where cast("{column}" as text) like \'data/%\' '
                            f'or cast("{column}" as text) like \'measurements/%\'',
                            (prefix,))
                        rewritten += cursor.rowcount or 0
                    except sqlite3.Error:
                        # A column that cannot be updated -- a generated one,
                        # or a type that will not concatenate -- is not a
                        # reason to abandon the other forty.
                        continue
            connection.commit()
        finally:
            connection.close()

    for settings_file in sorted((root / "settings").glob("*.csv")):
        try:
            text = settings_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if DATASET_PLACEHOLDER not in text:
            continue
        settings_file.write_text(
            text.replace(DATASET_PLACEHOLDER, str(root).rstrip("/")),
            encoding="utf-8")
        rewritten += 1

    LOG.info("example dataset at %s: %d paths made absolute", root, rewritten)
    return rewritten
