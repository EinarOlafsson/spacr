"""Community training data through the upload Space, as pull requests.

spaCR's "Contribute training data" buttons (Plaque Assay's figures and
plaques, Make Masks' images and masks) write a contribution folder:

    <id>/contribution.json
    <id>/images/<stem>.<png|jpg|jpeg|tif|tiff|bmp>
    <id>/masks/<stem>.tif        (masks layout: plaques and community_<name>)
    <id>/labels/<stem>.txt       (boxes layout: figures, YOLO lines)
    <id>/meta/<stem>.json

A contributor with a Hugging Face login sends it with their own token. One
without sends it here as a tar of that folder, and this module checks it and
opens a PULL REQUEST on the target dataset with the Space's token. Nothing is
ever committed to a dataset's main branch from here (a brand-new dataset's
README is the one exception, and only when creating datasets is allowed):
the maintainer reviews every pull request before it becomes training data.

Everything here is plain Python with the Hugging Face client passed in, so
it runs, and is tested, without Gradio and without a network.
"""
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time

OWNER = "einarolafsson"
FIGURES_REPO = "einarolafsson/community_toxoplasma_plaque_figures"
PLAQUES_REPO = "einarolafsson/community_toxoplasma_plaques"
LICENCE = "CC BY 4.0"

MAX_ARCHIVE_BYTES = 2_000_000_000
MAX_TOTAL_BYTES = 2_000_000_000
MAX_FILE_BYTES = 500_000_000
MAX_FILES = 4000

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
FOLDER_SUFFIXES = {
    "images": IMAGE_SUFFIXES,
    "masks": (".tif",),
    "labels": (".txt",),
    "meta": (".json",),
}
MAGIC = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".tif": (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"),
    ".tiff": (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"),
    ".bmp": (b"BM",),
}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
NAME_PATTERN = re.compile(r"^community_[a-z0-9]+(?:_[a-z0-9]+)*$")
YOLO_LINE = re.compile(
    r"^0 (?:0|1|0?\.\d+|1\.0+)(?: (?:0|1|0?\.\d+|1\.0+)){3}$")


class Refused(ValueError):
    """A contribution this endpoint will not forward; the text says why."""


def resolve_target(target):
    """The dataset a target names, and its layout; Refused for anything else.

    Accepted: ``figures`` and ``plaques`` (Plaque Assay's two datasets, by
    kind or by full id), and ``community_<name>`` with or without the
    ``einarolafsson/`` prefix. Every other repository -- another owner's,
    the model collection, a name with odd characters -- is unknown and
    refused, so this endpoint can only ever open pull requests on community
    training datasets.

    :returns: ``(repo_id, layout)`` with layout ``"boxes"`` or ``"masks"``.
    """
    text = str(target or "").strip()
    if text in ("figures", FIGURES_REPO):
        return FIGURES_REPO, "boxes"
    if text in ("plaques", PLAQUES_REPO):
        return PLAQUES_REPO, "masks"
    name = text[len(OWNER) + 1:] if text.startswith(OWNER + "/") else text
    if len(name) <= 96 and NAME_PATTERN.match(name):
        return f"{OWNER}/{name}", "masks"
    raise Refused(f"unknown target {text!r}: only figures, plaques and "
                  f"{OWNER}/community_<name> datasets are accepted")


def _members(archive):
    """The archive's regular files as ``{relative path: member}``; Refused if unsafe."""
    files, top, total = {}, None, 0
    for member in archive.getmembers():
        name = member.name
        while name.startswith("./"):
            name = name[2:]
        name = name.rstrip("/")
        if not name:
            continue
        parts = name.split("/")
        if name.startswith("/") or "\\" in name or any(
                p in ("", ".", "..") for p in parts):
            raise Refused(f"unsafe path in the archive: {member.name!r}")
        if top is None:
            top = parts[0]
        if parts[0] != top:
            raise Refused("the archive must hold one contribution folder")
        if member.isdir():
            if len(parts) > 2:
                raise Refused(f"unexpected folder {name!r}")
            continue
        if not member.isfile():
            raise Refused(f"{name!r} is not a regular file (links and "
                          "devices are refused)")
        if member.size > MAX_FILE_BYTES:
            raise Refused(f"{name} is {member.size} bytes, over the "
                          f"{MAX_FILE_BYTES} per-file limit")
        total += member.size
        if total > MAX_TOTAL_BYTES:
            raise Refused(f"the contribution is over {MAX_TOTAL_BYTES} bytes")
        files["/".join(parts[1:])] = member
        if len(files) > MAX_FILES:
            raise Refused(f"the contribution holds over {MAX_FILES} files")
    if top is None or not files:
        raise Refused("the archive is empty")
    if not ID_PATTERN.match(top):
        raise Refused(f"{top!r} is not an acceptable contribution folder name")
    return top, files


def _check_layout(files, layout):
    """Refused unless the files are exactly a contribution of ``layout``.

    :returns: ``{stem: image path}``.
    """
    marks = "labels" if layout == "boxes" else "masks"
    allowed = {"images", marks, "meta"}
    by_folder = {"images": {}, marks: {}, "meta": {}}
    for path in files:
        if path == "contribution.json":
            continue
        parts = path.split("/")
        if len(parts) != 2 or parts[0] not in allowed:
            raise Refused(f"{path} is not part of a {layout} contribution "
                          f"(expected contribution.json, images/, {marks}/, "
                          "meta/)")
        folder, filename = parts
        stem, dot, suffix = filename.rpartition(".")
        suffix = "." + suffix.lower() if dot else ""
        if not stem or suffix not in FOLDER_SUFFIXES[folder]:
            raise Refused(f"{path}: only "
                          f"{', '.join(FOLDER_SUFFIXES[folder])} files are "
                          f"accepted in {folder}/")
        if not STEM_PATTERN.match(stem):
            raise Refused(f"{path}: the file name has unaccepted characters")
        if stem in by_folder[folder]:
            raise Refused(f"{folder}/ holds more than one file named {stem}")
        by_folder[folder][stem] = path
    if "contribution.json" not in files:
        raise Refused("contribution.json is missing")
    images = by_folder["images"]
    if not images:
        raise Refused("the contribution holds no images")
    for folder in (marks, "meta"):
        missing = sorted(set(images) - set(by_folder[folder]))
        orphan = sorted(set(by_folder[folder]) - set(images))
        if missing:
            raise Refused(f"no {folder}/ file for: " + ", ".join(missing[:20]))
        if orphan:
            raise Refused(f"no image for {folder}/: " + ", ".join(orphan[:20]))
    return images, by_folder[marks], by_folder["meta"]


def _read(archive, member, limit=None):
    """A member's bytes (the first ``limit`` of them when given)."""
    handle = archive.extractfile(member)
    try:
        return handle.read() if limit is None else handle.read(limit)
    finally:
        handle.close()


def _check_contents(archive, files, layout, images, marks, metas):
    """Refused unless every file is what its name says; returns the record."""
    try:
        record = json.loads(_read(archive, files["contribution.json"])
                            .decode("utf-8"))
    except Exception:
        raise Refused("contribution.json is not valid JSON") from None
    if not isinstance(record, dict):
        raise Refused("contribution.json must hold one object")
    if record.get("layout") != layout:
        raise Refused(f"contribution.json says layout {record.get('layout')!r}"
                      f", but this target takes {layout!r}")
    if record.get("licence") != LICENCE:
        raise Refused(f"contribution.json must record the {LICENCE} licence")
    if not isinstance(record.get("consent"), dict) or not record["consent"]:
        raise Refused("contribution.json must record the contributor's consent")
    if record.get("images") != len(images):
        raise Refused(f"contribution.json counts {record.get('images')} "
                      f"images, but the folder holds {len(images)}")
    for stem, path in list(images.items()) + list(
            marks.items() if layout == "masks" else []):
        suffix = "." + path.rsplit(".", 1)[-1].lower()
        head = _read(archive, files[path], 8)
        if not any(head.startswith(sig) for sig in MAGIC[suffix]):
            raise Refused(f"{path} is not a {suffix} file")
    for path in metas.values():
        try:
            if not isinstance(json.loads(_read(archive, files[path])
                                         .decode("utf-8")), dict):
                raise ValueError
        except Exception:
            raise Refused(f"{path} is not a JSON object") from None
    if layout == "boxes":
        for path in marks.values():
            try:
                text = _read(archive, files[path]).decode("ascii")
            except Exception:
                raise Refused(f"{path} is not plain text") from None
            rows = [row.strip() for row in text.splitlines() if row.strip()]
            if not rows:
                raise Refused(f"{path} holds no boxes")
            bad = [row for row in rows if not YOLO_LINE.match(row)]
            if bad:
                raise Refused(f"{path}: not a YOLO box line: {bad[0][:60]!r}")
    return record


def readme(repo_id):
    """A new masks dataset's README; the same terms spaCR's own one states."""
    return f"""---
license: cc-by-4.0
tags: [spacr, community, image-segmentation]
---
# {repo_id.split('/')[-1]}

Community training data contributed through spaCR.

## Purpose

Training data for the next spaCR model of this kind.

## Licence

Shared under **{LICENCE}**. By uploading, a contributor confirms
that they have the right to share the images and agrees that they and the
annotations are redistributed under {LICENCE}.

## File layout

```
contributions/<contribution-id>/
  contribution.json        spaCR version, date, licence agreed, consent, counts
  images/<stem>.<ext>      the original image file, unchanged
  masks/<stem>.tif         uint16 label mask, 0 = background
  meta/<stem>.json         original file name, size, and which of spaCR's
                           proposals the contributor kept, edited or removed
```

An image is only accepted with at least one annotation.

## Review

Contributions arrive as pull requests and are **reviewed by the maintainer**
before they are merged.

spaCR: https://github.com/EinarOlafsson/spacr
"""


def _ensure_dataset(api, repo_id, create_new):
    """Refused unless the dataset exists or (a community_ one) may be made."""
    if api.repo_exists(repo_id, repo_type="dataset"):
        return
    if repo_id in (FIGURES_REPO, PLAQUES_REPO) or not create_new:
        raise Refused(f"{repo_id} does not exist; ask the spaCR maintainer "
                      "to create it")
    api.create_repo(repo_id, repo_type="dataset", private=False,
                    exist_ok=True)
    api.upload_file(path_or_fileobj=readme(repo_id).encode(),
                    path_in_repo="README.md", repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="README: licence, layout, purpose, review")


def receive(archive_path, target, api, *, who="unknown", create_new=True,
            work_dir=None):
    """Check one contribution archive and open a pull request with it.

    :param archive_path: a ``.tar`` (or ``.tar.gz``) of ONE contribution
        folder as spaCR's ``write_contribution`` lays it out.
    :param target: ``figures``, ``plaques`` or ``community_<name>``; see
        :func:`resolve_target`. Checked before the archive is opened.
    :param api: a ``huggingface_hub.HfApi`` holding the Space's token.
    :param who: the sender's address; only its hash is recorded.
    :param create_new: whether a ``community_<name>`` dataset that does not
        exist yet may be created (with its README) before the pull request.
    :param work_dir: where the archive is unpacked; a temporary folder,
        removed afterwards, when None.
    :returns: ``"ok: <pull request URL> ..."`` or ``"error: <why>"``.
    """
    try:
        repo_id, layout = resolve_target(target)
        if archive_path is None:
            raise Refused("no file")
        path = str(archive_path)
        if not path.lower().endswith((".tar", ".tar.gz", ".tgz")):
            raise Refused("send the contribution folder as one .tar file")
        size = os.path.getsize(path)
        if size > MAX_ARCHIVE_BYTES:
            raise Refused(f"{size} bytes is over the {MAX_ARCHIVE_BYTES} limit")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        try:
            archive = tarfile.open(path, "r:*")
        except Exception:
            raise Refused("the file is not a tar archive") from None
        scratch = tempfile.mkdtemp(prefix="contribution-", dir=work_dir)
        try:
            with archive:
                ident, files = _members(archive)
                images, marks, metas = _check_layout(files, layout)
                record = _check_contents(archive, files, layout, images,
                                         marks, metas)
                root = os.path.join(scratch, ident)
                for rel, member in files.items():
                    out = os.path.join(root, *rel.split("/"))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    source = archive.extractfile(member)
                    with source, open(out, "wb") as sink:
                        shutil.copyfileobj(source, sink, 1 << 20)
            _ensure_dataset(api, repo_id, create_new)
            sender = hashlib.sha256(str(who).encode()).hexdigest()[:16]
            info = api.upload_folder(
                folder_path=root, path_in_repo=f"contributions/{ident}",
                repo_id=repo_id, repo_type="dataset", create_pr=True,
                commit_message=f"Community contribution {ident}",
                commit_description=(
                    f"Sent through the spaCR upload Space without a Hugging "
                    f"Face login.\n\n- images: {len(images)}\n- layout: "
                    f"{layout}\n- spaCR: {record.get('spacr_version', '')}\n"
                    f"- archive sha256: {digest.hexdigest()}\n- sender "
                    f"(hashed): {sender}\n- received: "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\nReview before "
                    "merging."))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Refused as exc:
        return f"error: {exc}"
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
    url = str(getattr(info, "pr_url", "") or
              f"https://huggingface.co/datasets/{repo_id}/discussions")
    return (f"ok: {url} -- a pull request on {repo_id}, reviewed by the "
            "maintainer before it becomes training data.")
