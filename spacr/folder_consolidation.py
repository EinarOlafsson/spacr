"""Copy a folder tree into ONE folder, naming each copy after its folder path.

A tree like::

    exp/
      nucleus/  a.tif  b.tif
      cell/     a.tif  b.tif

becomes ``exp_renamed/`` holding ``exp_nucleus.tif``, ``exp_nucleus_2.tif``,
``exp_cell.tif`` and ``exp_cell_2.tif``: every copy is named after the folders
it sat in, joined by ``_``, and a second file from the same folder is
numbered. A ``rename_manifest.csv`` beside the copies maps each original path
to its new name, so nothing about where a file came from is lost.

This is a port of the maintainer's stand-alone ``rename_by_folders.py``
(standard library only) with its behaviour kept: the originals are COPIED,
never moved; symbolic links, to files or folders, are skipped and listed;
names are made safe for Windows; a name too long for a typical filesystem is
shortened with a hash of the full name so two long names cannot collide; and
the output folder must not exist yet. Two options were added for Make Masks,
which offers this on a dropped folder: ``extensions`` limits the copy to
image files and ``skip_dirs`` leaves out folders such as ``masks`` whose
contents are not images to edit. With neither, every file is copied, as the
script did.

Run as a program::

    python -m spacr.folder_consolidation /path/to/source [/path/to/new_output]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

#: Multi-part extensions kept whole, rather than only their final part.
COMPOUND_EXTENSIONS = (
    ".ome.tiff", ".ome.tif", ".nii.gz", ".tar.gz", ".tar.bz2", ".tar.xz",
    ".tar.zst",
)

#: The manifest written beside the copies.
MANIFEST_NAME = "rename_manifest.csv"

#: The suffix of the default output folder, ``<parent>/<name>_renamed``.
OUTPUT_SUFFIX = "_renamed"


def safe_part(name: str) -> str:
    """Replace characters Windows forbids in a filename; keep spaces.

    :param name: one folder name.
    :returns: the name with ``<>:"/\\|?*`` and control characters replaced
        by ``_`` and trailing spaces and dots removed; ``folder`` when
        nothing is left.
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(" .")
    return name or "folder"


def file_extension(path: Path) -> str:
    """Return the extension of ``path``, keeping a compound one whole.

    :param path: a file path.
    :returns: e.g. ``.ome.tif`` for ``x.ome.tif`` and ``.tif`` for ``x.tif``.
    """
    for extension in COMPOUND_EXTENSIONS:
        if path.name.lower().endswith(extension):
            return path.name[-len(extension):]
    return path.suffix


def available_filename(output: Path, stem: str, extension: str, used: set,
                       counters: dict) -> Path:
    """Pick the next free name for ``stem`` in ``output``.

    Collisions are numbered ``_2``, ``_3`` and so on; names are compared
    case-insensitively so the folder survives a move to Windows or macOS. A
    name longer than 240 bytes is cut and ends in a 12-character hash of the
    full stem. Windows' reserved device names are prefixed with ``_``.

    :param output: the output folder.
    :param stem: the name wanted, without extension.
    :param extension: the extension, with its dot.
    :param used: case-folded names already handed out; updated.
    :param counters: the next number per ``(stem, extension)``; updated.
    :returns: the path to copy to.
    :raises ValueError: when the extension alone leaves no room for a name.
    """
    if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])",
                    stem.split(".")[0]):
        stem = "_" + stem
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]
    key = (stem.casefold(), extension.casefold())
    number = counters.get(key, 1)
    while True:
        suffix = "" if number == 1 else f"_{number}"
        budget = 240 - len((extension + suffix).encode("utf-8"))
        fitted_stem = stem
        if len(stem.encode("utf-8")) > budget:
            if budget < 16:
                raise ValueError(
                    "File extension is too long for a safe output filename.")
            fitted_stem = (
                stem.encode("utf-8")[:budget - 13].decode(
                    "utf-8", errors="ignore") + "_" + digest)
        candidate = output / f"{fitted_stem}{suffix}{extension}"
        if candidate.name.casefold() not in used and not candidate.exists():
            used.add(candidate.name.casefold())
            counters[key] = number + 1
            return candidate
        number += 1


def unused_output_folder(parent: Path, name: str) -> Path:
    """Return ``parent/name``, or ``name_2``, ``name_3``... when it exists.

    :param parent: the folder to create the output in.
    :param name: the name wanted.
    :returns: a path that does not exist yet.
    """
    parent = Path(parent)
    candidate = parent / name
    number = 2
    while candidate.exists():
        candidate = parent / f"{name}_{number}"
        number += 1
    return candidate


def default_output_folder(source) -> Path:
    """Return the unused ``<parent>/<name>_renamed`` beside ``source``.

    :param source: the folder to consolidate.
    """
    source = Path(source).expanduser().resolve()
    return unused_output_folder(source.parent, source.name + OUTPUT_SUFFIX)


def _wanted(name: str, extensions: Optional[Tuple[str, ...]]) -> bool:
    """Whether a file called ``name`` is copied under ``extensions``.

    :param name: a file name.
    :param extensions: lower-case extensions to keep, or None for all.
    """
    return extensions is None or name.lower().endswith(extensions)


def _skipped(name: str, skip: set) -> bool:
    """Whether folder ``name`` is left out under ``skip``.

    A name matches with or without a numbered suffix, so skipping
    ``sorted_channels`` also skips ``sorted_channels_2``.

    :param name: a folder name.
    :param skip: case-folded folder names to leave out.
    """
    folded = name.casefold()
    return folded in skip or folded.rstrip("0123456789").rstrip("_") in skip


def _normalise_extensions(extensions) -> Optional[Tuple[str, ...]]:
    """Return ``extensions`` as a lower-case tuple, or None for every file.

    :param extensions: an iterable of extensions such as ``.tif``, or None.
    """
    if extensions is None:
        return None
    return tuple(str(ext).lower() for ext in extensions)


def nested_file_count(source, extensions: Optional[Iterable[str]] = None,
                      skip_dirs: Iterable[str] = ()) -> Tuple[int, int]:
    """Count the files that sit in SUBFOLDERS of ``source``, recursively.

    Make Masks asks whether to consolidate only when this is not zero:
    files directly in ``source`` open as they are.

    :param source: the folder.
    :param extensions: count only these extensions; None counts every file.
    :param skip_dirs: folder names not descended into (case-insensitive),
        such as ``masks``; hidden folders are never descended into.
    :returns: ``(files, folders)`` -- how many files lie below the top level
        and how many distinct subfolders hold them. Symbolic links are not
        counted or followed.
    """
    source = Path(source)
    exts = _normalise_extensions(extensions)
    skip = {str(name).casefold() for name in skip_dirs}
    files = 0
    folders = set()
    if not source.is_dir():
        return 0, 0
    for current, directories, filenames in os.walk(source, followlinks=False):
        directories[:] = [
            name for name in directories
            if not name.startswith(".") and not _skipped(name, skip)
            and not os.path.islink(os.path.join(current, name))]
        if Path(current) == source:
            continue
        found = [name for name in filenames
                 if _wanted(name, exts)
                 and not os.path.islink(os.path.join(current, name))]
        if found:
            files += len(found)
            folders.add(current)
    return files, len(folders)


@dataclass
class ConsolidationResult:
    """What :func:`consolidate_folder` did.

    :ivar output: the new folder holding the copies.
    :ivar manifest: the ``rename_manifest.csv`` inside it.
    :ivar copied: files copied.
    :ivar failed: files or folders that could not be read or copied.
    :ivar skipped_links: symbolic links left out.
    :ivar rows: the manifest rows, ``(original, new name, status, error)``.
    """

    output: Path
    manifest: Path
    copied: int = 0
    failed: int = 0
    skipped_links: int = 0
    rows: List[Tuple[str, str, str, str]] = field(default_factory=list)


def consolidate_folder(source, output=None, *,
                       extensions: Optional[Iterable[str]] = None,
                       skip_dirs: Iterable[str] = (),
                       log: Optional[Callable[[str], None]] = None
                       ) -> ConsolidationResult:
    """Copy every file under ``source`` into one new folder, named by folder path.

    Each copy is named after the folder path it came from, from ``source``'s
    own name down, parts joined by ``_``; a file directly in ``source`` is
    named after ``source``. Several files in one folder are numbered
    ``_2``, ``_3``..., in sorted file-name order. The originals are never
    touched.

    :param source: the folder to consolidate.
    :param output: the NEW folder to create; default
        :func:`default_output_folder`.
    :param extensions: copy only these extensions (e.g. image types); None,
        the script's behaviour, copies every file.
    :param skip_dirs: folder names not descended into, case-insensitively
        and with or without a numbered ``_2`` suffix.
    :param log: called with progress lines; default prints them.
    :returns: a :class:`ConsolidationResult`.
    :raises ValueError: when ``source`` is not a folder or ``output`` exists.
    """
    say = log or (lambda text: print(text, flush=True))
    source = Path(source).expanduser().resolve()
    output = (default_output_folder(source) if output is None
              else Path(output).expanduser().resolve())
    exts = _normalise_extensions(extensions)
    skip = {str(name).casefold() for name in skip_dirs}
    if not source.is_dir():
        raise ValueError(f"Source is not a directory: {source}")
    if output.exists():
        raise ValueError(f"Output already exists. Choose a NEW directory: {output}")

    output.mkdir(parents=True, exist_ok=False)
    result = ConsolidationResult(output=output, manifest=output / MANIFEST_NAME)
    used = {MANIFEST_NAME}
    counters: dict = {}

    with result.manifest.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_path", "new_filename", "status", "error"])

        def record(row) -> None:
            """Write one manifest row and keep it on the result.

            :param row: ``(original, new name, status, error)``.
            """
            writer.writerow(row)
            result.rows.append(tuple(row))

        def walk_error(error) -> None:
            """Record a folder ``os.walk`` could not read.

            :param error: the OSError it raised.
            """
            result.failed += 1
            record((str(error.filename), "", "directory_error", str(error)))
            say(f"Cannot read directory: {error}")

        for current, directories, filenames in os.walk(
                source, topdown=True, followlinks=False, onerror=walk_error):
            current = Path(current)
            retained = []
            for name in sorted(directories):
                directory = current / name
                if directory.is_symlink():
                    result.skipped_links += 1
                    record((str(directory), "", "skipped_symlink", ""))
                elif _skipped(name, skip):
                    continue
                elif directory.resolve() != output:
                    retained.append(name)
            directories[:] = retained

            folder_names = ((source.name or "root",)
                            + current.relative_to(source).parts)
            stem = "_".join(safe_part(part) for part in folder_names)
            for name in sorted(filenames):
                original = current / name
                if original.is_symlink():
                    result.skipped_links += 1
                    record((str(original), "", "skipped_symlink", ""))
                    continue
                if not _wanted(name, exts):
                    continue
                destination = None
                try:
                    destination = available_filename(
                        output, stem, file_extension(original), used, counters)
                    shutil.copy2(original, destination)
                except (OSError, ValueError) as exc:
                    result.failed += 1
                    if destination is not None and destination.exists():
                        try:
                            destination.unlink()
                        except OSError:
                            pass
                    record((str(original), "", "error", str(exc)))
                    say(f"Could not copy {original}: {exc}")
                else:
                    result.copied += 1
                    record((str(original), destination.name, "copied", ""))
                    if result.copied % 100 == 0:
                        say(f"Copied {result.copied} files...")

    say(f"Copied: {result.copied} | Errors: {result.failed} | "
        f"Symlinks skipped: {result.skipped_links}")
    say(f"Output: {output}  File mapping: {result.manifest}")
    return result


def copy_and_rename(source: Path, output: Path) -> int:
    """The script's entry point: consolidate and return an exit status.

    :param source: the folder to consolidate.
    :param output: the NEW output folder.
    :returns: 1 when any file failed, else 0.
    """
    result = consolidate_folder(source, output)
    return 1 if result.failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Command line: ``python -m spacr.folder_consolidation SOURCE [OUTPUT]``.

    :param argv: the arguments; default ``sys.argv[1:]``.
    :returns: the exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="Source directory")
    parser.add_argument("output", nargs="?", type=Path,
                        help="NEW output directory")
    args = parser.parse_args(argv)
    try:
        output = args.output or default_output_folder(args.source)
        return copy_and_rename(args.source, output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Original files were not changed; completed copies "
              "remain.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
