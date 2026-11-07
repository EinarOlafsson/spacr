#!/usr/bin/env python
"""Fetch the documentation's large media from a pinned Hugging Face revision.

Media that only the documentation site and the tutorial tooling read is kept
out of git. It lives in the public dataset ``einarolafsson/spacr-tutorials``;
``docs/media_revision.json`` pins, per group, the dataset revision and the
sha256 of every file, and this tool puts exactly those bytes back where the
site and the tools expect them.

Groups:

``deck``
    The info deck's published PDF and its animated GIFs, under ``deck/``. The
    pictures, thumbnails, GitHub pages, viewer and manifest stay in git: the
    README and its slide-by-slide GitHub pages show the pictures directly.
``api_memory``
    The translated API catalogs used as translation memory, under ``i18n/``.
``tutorials``
    What the tutorial library on the site serves from its own tree: every
    lesson poster and every example download. The player code, the lesson
    and caption catalogs, fonts and logo stay in git.
``tutorial_archive``
    Tutorial media nothing on the site serves: the narration copies that
    predate the hosted narration, the release-candidate snapshot's posters
    and examples, archived lessons' posters and repaired narration. Tutorial
    tools read them through :func:`media_path` when they need one.

The two tutorial groups keep each file at ``tree/<its path in spaCR>`` in
the dataset, so a file's place in the dataset says where it belongs here.

Commands::

    python tools/fetch_docs_media.py fetch      # into docs/source/_static/deck
    python tools/fetch_docs_media.py check      # exit 1 if a fetched file differs
    python tools/fetch_docs_media.py status     # how many pinned files are here
    python tools/fetch_docs_media.py publish    # upload changed files, bump the pin

Every command takes ``--group`` (default ``deck``).

``fetch`` downloads into the Hugging Face cache, checks every hash, and copies
the files into the tree. A file already in the tree with the pinned hash is
left alone. When the deck's ``slides.json`` names a newer version than the
pinned PDF shows (a release bump restamps the title picture in git), it puts
the stamped title picture on the PDF's first page, as
``tools/build_readme_deck.py --stamp`` does for a local PDF.

``publish`` is the last step of a rebuild (``tools/build_readme_deck.py
--publish`` and ``tools/refresh_alpha_deck.py`` call it for the deck): it
uploads the files whose hash changed to the dataset in one commit and writes
the new revision and hashes into the pin. ``--source`` reads the files from
another folder laid out like the repository, such as the tutorial staging
folder the tutorial tools write to. A group that has never been published
(its revision is empty) uploads every file. It needs a Hugging Face token
with write access to the dataset.

Tests read the media through :func:`media_path`, which uses the same pin and
cache and raises when the file cannot be fetched. :func:`available` asks the
same question without raising, for code that must keep working offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from email.utils import parsedate_to_datetime
from datetime import timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / "docs" / "media_revision.json"

#: Folders whose every tracked file is tutorial media, kept out of git.
TUTORIAL_MEDIA_FOLDERS = (
    "docs/source/_extra/tutorials/production/",
    "docs/source/_extra/tutorials/examples/",
    "tools/tutorials/release_candidate/web/production/",
    "tools/tutorials/release_candidate/web/examples/",
)

#: Media file types that are tutorial media anywhere under the tutorial trees.
TUTORIAL_MEDIA_SUFFIXES = (".m4a", ".mp3", ".wav", ".mp4", ".webm", ".mov",
                           ".jpg", ".jpeg", ".zip")

#: The two trees tutorial work happens in.
TUTORIAL_TREES = ("docs/source/_extra/tutorials/", "tools/tutorials/")

#: Where the tutorial groups keep a file in the dataset: this prefix plus the
#: file's path in the repository.
TUTORIAL_PREFIX = "tree/"


def is_tutorial_media(tree_path: str) -> bool:
    """Whether a repository path is tutorial media that stays out of git.

    :param tree_path: a path relative to the repository root, with ``/``.
    :returns: True for every file under :data:`TUTORIAL_MEDIA_FOLDERS` and
        for audio, video, picture and archive files anywhere in
        :data:`TUTORIAL_TREES`.
    """
    path = str(tree_path).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith(TUTORIAL_MEDIA_FOLDERS):
        return True
    return path.startswith(TUTORIAL_TREES) and path.lower().endswith(TUTORIAL_MEDIA_SUFFIXES)


def load_pin(pin: Path = PIN) -> dict:
    """Return the media pin.

    :param pin: the pin file.
    :returns: the parsed pin.
    """
    return json.loads(pin.read_text(encoding="utf-8"))


def write_pin(data: dict, pin: Path = PIN) -> None:
    """Write the media pin in its stable form.

    :param data: the pin.
    :param pin: the pin file.
    """
    pin.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file.

    :param path: the file.
    :returns: its digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(data: dict, name: str) -> dict:
    groups = data.get("groups", {})
    if name not in groups:
        raise KeyError(f"media pin has no group {name!r}")
    return groups[name]


def _retry_delay(response, attempt):
    """Honor Retry-After dates or seconds, otherwise use jittered backoff."""
    header = response.headers.get("Retry-After", "") if response is not None else ""
    try:
        seconds = float(header)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(header)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = date.timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            seconds = -1.0
    if math.isfinite(seconds) and seconds >= 0:
        return seconds
    return 2.0 ** attempt + random.uniform(0.0, 1.0)


def media_path(tree_path: str, group: str = "deck", pin: Path = PIN) -> Path:
    """Return a verified local copy of one pinned file, from the cache.

    :param tree_path: the file's path in the repository tree, as the pin
        lists it, for example ``docs/source/_static/deck/spacr_deck.pdf``.
    :param group: the pin group.
    :param pin: the pin file.
    :returns: the cached file, whose sha256 matches the pin.
    :raises RuntimeError: when the file cannot be fetched or does not match.
    """
    spec = _group(load_pin(pin), group)
    record = spec["files"][tree_path]
    if not spec.get("revision"):
        raise RuntimeError(f"the {group} media group has not been published to {spec['repo']} yet")
    from huggingface_hub import hf_hub_download

    deadline = time.monotonic() + 120.0
    for attempt in range(4):
        try:
            local = Path(hf_hub_download(spec["repo"], record["path_in_repo"],
                                         repo_type=spec["repo_type"],
                                         revision=spec["revision"]))
            break
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None) if response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < 3:
                delay = _retry_delay(response, attempt)
                remaining = deadline - time.monotonic()
                if remaining > 0 and delay <= remaining:
                    time.sleep(delay)
                    if time.monotonic() < deadline:
                        continue
            detail = f"HTTP {status}" if type(status) is int else type(error).__name__
            raise RuntimeError(
                f"could not fetch {record['path_in_repo']} from {spec['repo']} at "
                f"{spec['revision']} ({detail})") from None
    if sha256(local) != record["sha256"]:
        raise RuntimeError(f"{record['path_in_repo']} at {spec['revision']} does not match the pin")
    return local


def available(tree_path: str, group: str, root: Path = ROOT) -> Optional[Path]:
    """Return a verified local copy of a pinned file, or None, never raising.

    The file in the tree is used when its hash matches the pin; otherwise the
    pinned revision is fetched. Offline, unpublished or unpinned, the answer
    is None, so a caller can show that the media is not downloaded.

    :param tree_path: the file's path in the repository tree.
    :param group: the pin group.
    :param root: the repository root.
    :returns: the verified file, or None.
    """
    pin = root / "docs" / "media_revision.json"
    try:
        record = _group(load_pin(pin), group)["files"][tree_path]
    except (OSError, ValueError, KeyError):
        return None
    target = root / tree_path
    try:
        if target.is_file() and sha256(target) == record["sha256"]:
            return target
    except OSError:
        pass
    try:
        return media_path(tree_path, group, pin)
    except Exception:
        return None


def status(group: str = "deck", root: Path = ROOT) -> Dict[str, int]:
    """Count the group's pinned files that are in the tree with their hash.

    :param group: the pin group.
    :param root: the repository root.
    :returns: ``{"pinned": n, "present": n, "missing": n, "different": n}``.
    """
    spec = _group(load_pin(root / "docs" / "media_revision.json"), group)
    counts = {"pinned": len(spec["files"]), "present": 0, "missing": 0, "different": 0}
    for tree_path, record in spec["files"].items():
        target = root / tree_path
        if not target.is_file():
            counts["missing"] += 1
        elif sha256(target) == record["sha256"]:
            counts["present"] += 1
        else:
            counts["different"] += 1
    return counts


def _deck_version(root: Path) -> Optional[str]:
    manifest = root / "docs" / "source" / "_static" / "deck" / "slides.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("version")


def _restamp_pdf(root: Path, pdf: Path) -> bool:
    import importlib.util

    from PIL import Image

    spec = importlib.util.spec_from_file_location("build_readme_deck", root / "tools" / "build_readme_deck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deck = root / "docs" / "source" / "_static" / "deck"
    first = json.loads((deck / "slides.json").read_text(encoding="utf-8"))["slides"][0]
    with Image.open(deck / first["image"]) as picture:
        return module.replace_first_page(pdf, picture.convert("RGB"))


def fetch(group: str = "deck", root: Path = ROOT) -> Dict[str, Path]:
    """Copy every pinned file of a group into the tree, verified.

    :param group: the pin group.
    :param root: the repository root.
    :returns: tree path -> written file.
    """
    pin = root / "docs" / "media_revision.json"
    spec = _group(load_pin(pin), group)
    written = {}
    for tree_path in spec["files"]:
        target = root / tree_path
        if not (target.is_file() and sha256(target) == spec["files"][tree_path]["sha256"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(media_path(tree_path, group, pin), target)
        written[tree_path] = target
    shown = _deck_version(root)
    if group == "deck" and shown and spec.get("version") and shown != spec["version"]:
        pdf = root / "docs" / "source" / "_static" / "deck" / "spacr_deck.pdf"
        if pdf.relative_to(root).as_posix() in spec["files"] and _restamp_pdf(root, pdf):
            print(f"deck PDF title page restamped for {shown} (pinned PDF shows {spec['version']})")
    return written


def check(group: str = "deck", root: Path = ROOT) -> Iterable[str]:
    """Check raw pins and the deck's verified release-title projection.

    :param group: the pin group.
    :param root: the repository root.
    :returns: the problems found.
    """
    import tempfile

    pin = root / "docs" / "media_revision.json"
    spec = _group(load_pin(pin), group)
    shown = _deck_version(root) if group == "deck" else None
    projected_pdf = (shown and spec.get("version") and shown != spec["version"])
    for tree_path, record in spec["files"].items():
        target = root / tree_path
        if not target.is_file():
            yield f"{tree_path}: not fetched"
            continue
        expected = record["sha256"]
        provenance = "the pin"
        if projected_pdf and tree_path == "docs/source/_static/deck/spacr_deck.pdf":
            with tempfile.TemporaryDirectory(prefix="spacr-deck-check-") as directory:
                pdf = Path(directory) / "spacr_deck.pdf"
                shutil.copyfile(media_path(tree_path, group, pin), pdf)
                if sha256(pdf) != record["sha256"]:
                    raise RuntimeError(f"{tree_path}: cached source differs from the pin")
                if not _restamp_pdf(root, pdf):
                    yield f"{tree_path}: could not verify the release-title projection"
                    continue
                expected = sha256(pdf)
                provenance = f"the verified {shown} deck projection"
        if sha256(target) != expected:
            yield f"{tree_path}: differs from {provenance}"


def publish(group: str = "deck", root: Path = ROOT, message: Optional[str] = None,
            source: Optional[Path] = None) -> Optional[str]:
    """Upload the group's changed files and move the pin to the new revision.

    :param group: the pin group.
    :param root: the repository root, whose pin is updated.
    :param message: the dataset commit message.
    :param source: a folder laid out like the repository to read the files
        from; the repository root when omitted.
    :returns: the new revision, or None when nothing changed.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    pin = root / "docs" / "media_revision.json"
    data = load_pin(pin)
    spec = _group(data, group)
    source = Path(source) if source is not None else root
    unpublished = not spec.get("revision")
    changed = []
    for tree_path, record in spec["files"].items():
        target = source / tree_path
        if not target.is_file():
            raise FileNotFoundError(f"{tree_path} is missing; fetch or rebuild it before publishing")
        digest = sha256(target)
        if unpublished or digest != record["sha256"]:
            changed.append((tree_path, record, digest, target))
    shown = _deck_version(root) if group == "deck" else None
    if not changed:
        if shown and spec.get("version") != shown and any(
                p.endswith("spacr_deck.pdf") for p in spec["files"]):
            print("deck: version changed but the PDF did not; nothing uploaded")
        return None
    operations = [CommitOperationAdd(path_in_repo=record["path_in_repo"], path_or_fileobj=str(target))
                  for _, record, _, target in changed]
    info = HfApi().create_commit(
        spec["repo"], repo_type=spec["repo_type"], operations=operations,
        commit_message=message or f"{group}: {len(operations)} files from the spaCR tree")
    for _, record, digest, target in changed:
        record["sha256"] = digest
        record["size"] = target.stat().st_size
    spec["revision"] = info.oid
    if shown:
        spec["version"] = shown
    write_pin(data, pin)
    return info.oid


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Command-line entry point.

    :param argv: the arguments.
    :returns: the exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "check", "status", "publish"))
    parser.add_argument("--group", default="deck")
    parser.add_argument("--message")
    parser.add_argument("--source", type=Path,
                        help="publish: read the files from this folder, laid out like the repository")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        written = fetch(args.group)
        print(f"{args.group}: {len(written)} pinned files in the tree")
        return 0
    if args.command == "check":
        problems = list(check(args.group))
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    if args.command == "status":
        counts = status(args.group)
        print(f"{args.group}: {counts['present']} of {counts['pinned']} pinned files here, "
              f"{counts['missing']} not downloaded, {counts['different']} different")
        return 0
    revision = publish(args.group, message=args.message, source=args.source)
    print(f"{args.group}: pinned at {revision}" if revision else f"{args.group}: nothing changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
