#!/usr/bin/env python
"""Fetch the documentation's large media from a pinned Hugging Face revision.

Item 710, phase 3 (the slide deck). The info deck's published PDF and its
animated GIFs are ~70 MB that every clone used to download, though only the
documentation site serves them. They live in the public dataset
``einarolafsson/spacr-tutorials`` under ``deck/``; ``docs/media_revision.json``
pins the revision and the sha256 of every file, and this tool puts exactly
those bytes back where the site expects them.

The pictures, thumbnails, GitHub pages, viewer and manifest stay in git: the
README and its slide-by-slide GitHub pages show the pictures directly.

Commands::

    python tools/fetch_docs_media.py fetch      # into docs/source/_static/deck
    python tools/fetch_docs_media.py check      # exit 1 if a fetched file differs
    python tools/fetch_docs_media.py publish    # upload changed files, bump the pin

``fetch`` downloads into the Hugging Face cache, checks every hash, and copies
the files into the tree. When the deck's ``slides.json`` names a newer version
than the pinned PDF shows (a release bump restamps the title picture in git),
it puts the stamped title picture on the PDF's first page, as
``tools/build_readme_deck.py --stamp`` does for a local PDF.

``publish`` is the last step of a deck rebuild (``tools/build_readme_deck.py
--publish`` and ``tools/refresh_alpha_deck.py`` call it): it uploads the
files whose hash changed to the dataset in one commit and writes the new
revision and hashes into the pin. It needs a Hugging Face token with write
access to the dataset.

Tests read the media through :func:`media_path`, which uses the same pin and
cache and raises when the file cannot be fetched; nothing skips.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / "docs" / "media_revision.json"


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


def media_path(tree_path: str, group: str = "deck", pin: Path = PIN) -> Path:
    """Return a verified local copy of one pinned file, from the cache.

    :param tree_path: the file's path in the repository tree, as the pin
        lists it, for example ``docs/source/_static/deck/spacr_deck.pdf``.
    :param group: the pin group.
    :param pin: the pin file.
    :returns: the cached file, whose sha256 matches the pin.
    :raises RuntimeError: when the file cannot be fetched or does not match.
    """
    from huggingface_hub import hf_hub_download

    spec = _group(load_pin(pin), group)
    record = spec["files"][tree_path]
    try:
        local = Path(hf_hub_download(spec["repo"], record["path_in_repo"],
                                     repo_type=spec["repo_type"],
                                     revision=spec["revision"]))
    except Exception as error:
        raise RuntimeError(
            f"could not fetch {record['path_in_repo']} from {spec['repo']} at "
            f"{spec['revision']}: {error}") from error
    if sha256(local) != record["sha256"]:
        raise RuntimeError(f"{record['path_in_repo']} at {spec['revision']} does not match the pin")
    return local


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
        if str(pdf.relative_to(root)) in spec["files"] and _restamp_pdf(root, pdf):
            print(f"deck PDF title page restamped for {shown} (pinned PDF shows {spec['version']})")
    return written


def check(group: str = "deck", root: Path = ROOT) -> Iterable[str]:
    """Name every pinned file that is missing or differs in the tree.

    :param group: the pin group.
    :param root: the repository root.
    :returns: the problems found.
    """
    spec = _group(load_pin(root / "docs" / "media_revision.json"), group)
    for tree_path, record in spec["files"].items():
        target = root / tree_path
        if not target.is_file():
            yield f"{tree_path}: not fetched"
        elif sha256(target) != record["sha256"]:
            yield f"{tree_path}: differs from the pin"


def publish(group: str = "deck", root: Path = ROOT, message: Optional[str] = None) -> Optional[str]:
    """Upload the group's changed files and move the pin to the new revision.

    :param group: the pin group.
    :param root: the repository root.
    :param message: the dataset commit message.
    :returns: the new revision, or None when nothing changed.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    pin = root / "docs" / "media_revision.json"
    data = load_pin(pin)
    spec = _group(data, group)
    changed = []
    for tree_path, record in spec["files"].items():
        target = root / tree_path
        if not target.is_file():
            raise FileNotFoundError(f"{tree_path} is missing; fetch or rebuild it before publishing")
        digest = sha256(target)
        if digest != record["sha256"]:
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
    parser.add_argument("command", choices=("fetch", "check", "publish"))
    parser.add_argument("--group", default="deck")
    parser.add_argument("--message")
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
    revision = publish(args.group, message=args.message)
    print(f"{args.group}: pinned at {revision}" if revision else f"{args.group}: nothing changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
