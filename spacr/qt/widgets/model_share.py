"""Add a local model to the zoo, and optionally share it on Hugging Face.

WHY THERE IS NO BUILT-IN TOKEN. Uploading to one account from every user's
machine would mean shipping a write token inside spaCR. A write token is not
an upload permit: it can also rewrite and DELETE every model in that account,
and anyone who installs spaCR can read it out of the package. So the upload
uses the token belonging to whoever is running it:

  - `huggingface-cli login`, or the HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    environment variable.

It publishes to :data:`SHARE_REPO` when that token may write there -- the
owner can grant that per person in the repository's settings -- and otherwise
to the uploader's own namespace, which always works and is never destructive
to somebody else. Either way the model card carries the same table, so a
shared model is readable in the same terms as a bundled one.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Where shared models go when the uploader may write there.
SHARE_REPO = "einarolafsson/user-models"

#: The central upload endpoint: a Hugging Face Space that holds the token, so
#: a contributor needs no Hugging Face account of their own. See
#: tools/model_upload_space/. Empty until it is deployed; set
#: SPACR_MODEL_UPLOAD_URL to point spaCR at one.
CENTRAL_ENDPOINT = os.environ.get(
    "SPACR_MODEL_UPLOAD_URL",
    "https://einarolafsson-spacr-model-upload.hf.space").strip()


#: About how much training data the endpoint accepts, packed.
MAX_TRAIN_BYTES = 25_000_000_000


def pack_training_data(folder: str, out_dir: Optional[str] = None) -> str:
    """Pack a training-data folder into ONE uncompressed tar. Returns its path.

    Uncompressed on purpose: microscopy TIFFs are already poorly compressible
    and gzipping tens of gigabytes costs far more time than it saves bytes.
    Raises if the result would be larger than the endpoint accepts, BEFORE the
    upload is attempted, so a user does not wait out a transfer that was never
    going to be accepted.

    :param folder: training-data folder to pack; it must exist, and its files
        must total no more than ``MAX_TRAIN_BYTES``.
    """
    import tarfile
    import tempfile

    source = Path(folder)
    if not source.is_dir():
        raise RuntimeError(f"{folder} is not a folder")
    total = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
    if total > MAX_TRAIN_BYTES:
        raise RuntimeError(
            f"that folder holds {total / 1e9:.1f} GB, and the limit is about "
            f"{MAX_TRAIN_BYTES / 1e9:.0f} GB")
    handle = tempfile.NamedTemporaryFile(
        suffix=".tar", delete=False,
        dir=out_dir or tempfile.gettempdir())
    handle.close()
    with tarfile.open(handle.name, "w") as tar:
        tar.add(str(source), arcname=source.name)
    return handle.name


#: The scorecard, in the order the model cards print it.
SHARE_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("display_name", "Model name", ""),
    ("kind", "Kind (cellpose / classifier / detector)", "cellpose"),
    ("trained_on", "Trained on (what images, what objects)", ""),
    ("n_train", "Train (images)", ""),
    ("train_objects", "Train obj. (objects)", ""),
    ("n_test", "Test (held-out images)", ""),
    ("test_objects", "Test obj. (held-out objects)", ""),
    ("cv", "CV (e.g. 'no' or '5-fold')", "no"),
    ("f1", "F1 @ IoU 0.5", ""),
    ("aji", "AJI", ""),
    ("dice", "Dice", ""),
    ("stock_f1", "Stock model F1 @ IoU 0.5", ""),
    ("stock_aji", "Stock model AJI", ""),
    ("stock_dice", "Stock model Dice", ""),
    ("train_loss", "Final train loss", ""),
    ("val_loss", "Final validation loss", ""),
    ("best_epoch", "Best epoch / total", ""),
    ("contact", "Contact (optional)", ""),
    ("notes", "Anything a reader should know (limitations)", ""),
)


def _gradio_upload(base: str, path: str, timeout: int) -> Dict[str, Any]:
    """Send one file to a Gradio app's upload route; returns its file handle.

    :param base: the app's URL without a trailing slash.
    :param path: the local file.
    :param timeout: seconds to wait for the transfer.
    """
    import json
    import urllib.request
    import uuid

    boundary = uuid.uuid4().hex
    with open(path, "rb") as handle:
        payload = handle.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
            f"filename=\"{os.path.basename(path)}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        base + "/gradio_api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    sent = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
    return {"path": sent[0], "meta": {"_type": "gradio.FileData"}}


def _gradio_call(base: str, api_name: str, data: list,
                 wait: float = 600) -> str:
    """Call one endpoint of a Gradio app and wait for its text reply.

    :param base: the app's URL without a trailing slash.
    :param api_name: the endpoint, e.g. ``"upload"``.
    :param data: the endpoint's arguments, in order.
    :param wait: seconds to wait for the reply.
    :returns: the reply with its ``ok:`` kept.
    :raises RuntimeError: with the endpoint's own words when it replies
        ``error: ...``, or when it does not answer in time.
    """
    import json
    import time
    import urllib.request

    request = urllib.request.Request(
        base + f"/gradio_api/call/{api_name}",
        data=json.dumps(dict(data=data)).encode(),
        headers={"Content-Type": "application/json"})
    event = json.loads(urllib.request.urlopen(request, timeout=120).read())["event_id"]
    deadline = time.time() + wait
    while time.time() < deadline:
        with urllib.request.urlopen(
                base + f"/gradio_api/call/{api_name}/{event}",
                timeout=120) as reply:
            text = reply.read().decode("utf-8", "replace")
        if "event: complete" in text:
            answer = json.loads(text.rsplit("data:", 1)[1].strip())[0]
            if str(answer).startswith("error:"):
                raise RuntimeError(str(answer)[6:].strip())
            return str(answer)
        if "event: error" in text:
            raise RuntimeError("the upload endpoint failed while handling it")
        time.sleep(3)
    raise RuntimeError("the upload endpoint did not answer in time")


def central_upload(path: str, fields: Dict[str, Any]) -> str:
    """Publish through the central endpoint. Returns its reply.

    Speaks the Gradio HTTP API directly with urllib rather than through
    gradio_client, because that package is not a spaCR dependency: when it was
    missing this raised, the caller fell back to the uploader's own token, and
    the model went somewhere nobody was looking for it. A publish path that
    depends on an optional import is a publish path that silently does
    something else.

    :param path: local checkpoint file to upload.
    :param fields: the share form's values; ``display_name``, ``kind``,
        ``trained_on`` and ``contact`` are sent as their own arguments, the
        whole mapping as JSON, and a non-empty ``train_data_dir`` is packed
        with :func:`pack_training_data` and uploaded too.
    """
    import json

    if not CENTRAL_ENDPOINT:
        raise RuntimeError("no central upload endpoint is configured")
    base = CENTRAL_ENDPOINT.rstrip("/")
    uploaded = _gradio_upload(base, path, 600)

    train_handle = None
    train_dir = str(fields.get("train_data_dir") or "")
    if train_dir:
        tarball = pack_training_data(train_dir)
        try:
            train_handle = _gradio_upload(base, tarball, 7200)
        finally:
            try:
                os.unlink(tarball)
            except OSError:
                pass

    return _gradio_call(base, "upload", [
        uploaded, str(fields.get("display_name") or ""),
        str(fields.get("kind") or "cellpose"),
        str(fields.get("trained_on") or ""), json.dumps(fields),
        str(fields.get("contact") or ""), train_handle])


def slugify(text: str) -> str:
    """A repository-safe folder name.

    :param text: display name or file name to convert; it is lower-cased and
        every run of characters other than ``a-z`` and ``0-9`` becomes one
        hyphen. An empty result becomes ``"model"``.
    """
    out = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return out or "model"


def find_token() -> Optional[str]:
    """The uploader's own Hugging Face token, or ``None``."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:                                        # noqa: BLE001
        return None


def _num(value: Any) -> str:
    """A card value as text, or "not recorded" when blank."""
    text = str(value or "").strip()
    return text if text else "not recorded"


def card(fields: Dict[str, Any], filename: str, sha256: str, repo_id: str,
         folder: str) -> str:
    """The model card, carrying the same table every spaCR model card uses.

    :param fields: the share form's values: ``display_name``, ``kind``,
        ``trained_on``, ``notes`` and the metrics (``f1``, ``aji``, ``dice``,
        ``train_loss``, ``val_loss`` and the rest); a missing value is shown as
        unstated.
    :param filename: file name of the checkpoint; also the heading when no
        ``display_name`` is given.
    :param sha256: hex SHA-256 digest of the checkpoint, printed on the card.
    :param repo_id: Hugging Face repository the model is published to; accepted
        but not written into the card.
    :param folder: folder inside the repository that holds the checkpoint.
    """
    name = fields.get("display_name") or filename
    gap = ""
    try:
        gap = f"{float(fields['val_loss']) - float(fields['train_loss']):+.4f}"
    except Exception:                                        # noqa: BLE001
        gap = "—"
    return f"""# {name}

Shared through spaCR's Model Zoo. Uploaded by a spaCR user, not validated by
the spaCR maintainers -- read the table and the limitations before using it.

- **Kind:** {fields.get('kind') or 'cellpose'}
- **Checkpoint:** `{folder}/{filename}`
- **SHA-256:** `{sha256}`

## Performance

| model | train | train obj. | test | test obj. | CV | F1 @ IoU 0.5 | AJI | Dice | final train loss | final val loss | val - train | best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock (no fine-tuning) | — | — | {_num(fields.get('n_test'))} | {_num(fields.get('test_objects'))} | — | {_num(fields.get('stock_f1'))} | {_num(fields.get('stock_aji'))} | {_num(fields.get('stock_dice'))} | — | — | — | — |
| **this model** | {_num(fields.get('n_train'))} | {_num(fields.get('train_objects'))} | {_num(fields.get('n_test'))} | {_num(fields.get('test_objects'))} | {_num(fields.get('cv'))} | **{_num(fields.get('f1'))}** | {_num(fields.get('aji'))} | {_num(fields.get('dice'))} | {_num(fields.get('train_loss'))} | {_num(fields.get('val_loss'))} | {gap} | {_num(fields.get('best_epoch'))} |

## Trained on

{fields.get('trained_on') or 'Not stated by the uploader.'}

## Limitations

{fields.get('notes') or 'None stated by the uploader.'}

## Use it in spaCR

```bash
pip install spacr
```

Open the Model Zoo, find this model, press Download. Or:

```python
from spacr import model_zoo
entry = next(e for e in model_zoo.catalogue() if e.name == "{filename}")
path = model_zoo.install(entry, dest="~/spacr_models")
```

- spaCR on GitHub: https://github.com/EinarOlafsson/spacr
- Model Zoo API: `spacr.model_zoo`
- Mask generation: `spacr.core.preprocess_generate_masks`
"""


def target_repo(token: str) -> Tuple[str, bool]:
    """Where this token may publish: the shared repo, or its own namespace.

    :param token: Hugging Face access token; it decides whether the shared
        repository is reachable and, if not, whose namespace is used.
    :returns: ``(repo_id, is_shared)``.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.repo_info(SHARE_REPO, repo_type="model")
        me = api.whoami()
        owner = SHARE_REPO.split("/")[0]
        if me.get("name") == owner:
            return SHARE_REPO, True
        for org in me.get("orgs", []) or []:
            if org.get("name") == owner:
                return SHARE_REPO, True
        # A collaborator's write access cannot be read back reliably; try it
        # and let the upload itself be the test.
        return SHARE_REPO, True
    except Exception:                                        # noqa: BLE001
        pass
    try:
        return f"{HfApi(token=token).whoami()['name']}/spacr-models", False
    except Exception:                                        # noqa: BLE001
        return "", False


def share(path: str, fields: Dict[str, Any], token: str) -> str:
    """Upload a checkpoint and its card. Returns the model page URL.

    :param path: local checkpoint file to upload.
    :param fields: the share form's values; ``display_name`` names the staging
        folder and the whole mapping fills the model card.
    :param token: Hugging Face access token used for every call.
    """
    from huggingface_hub import HfApi

    from ... import model_zoo

    api = HfApi(token=token)
    repo_id, _shared = target_repo(token)
    if not repo_id:
        raise RuntimeError("Could not work out where to publish: check the token.")
    filename = os.path.basename(path)
    # Into staging/ as well: unvetted is unvetted however it arrived, and
    # the community listing looks in exactly one place.
    folder = "staging/" + slugify(fields.get("display_name") or filename)
    digest = model_zoo.sha256_file(path)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"{folder}/{filename}",
                    repo_id=repo_id, repo_type="model")
    api.upload_file(
        path_or_fileobj=card(fields, filename, digest, repo_id, folder).encode(),
        path_in_repo=f"{folder}/README.md", repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}/tree/main/{folder}"


COMMUNITY_OWNER = "einarolafsson"
COMMUNITY_FIGURES_REPO = "einarolafsson/community_toxoplasma_plaque_figures"
COMMUNITY_PLAQUES_REPO = "einarolafsson/community_toxoplasma_plaques"
COMMUNITY_LICENCE = "CC BY 4.0"
COMMUNITY_CONSENT_KEY = "community_training_data/consent"
FIGURES_KIND = "figures"
PLAQUES_KIND = "plaques"
BOXES_LAYOUT = "boxes"
MASKS_LAYOUT = "masks"

FIGURE_CONSCIENCE = (
    "Every box you send becomes a lesson for the next well detector. A well "
    "you skip teaches it that wells can be ignored; a box that cuts a well "
    "in half teaches it that wells are halves. Someone has to find and fix "
    "each one before the next model can ship, so the next release waits. "
    "Take the extra minute: every well, edge to edge.")

PLAQUE_CONSCIENCE = (
    "Every mask you send becomes a lesson for the next plaque model. A loose "
    "outline teaches it to be loose, a missed plaque teaches it to miss "
    "plaques, and two plaques painted as one teach it that they are one. "
    "Someone has to fix each of those before the next model can ship, so "
    "the next release waits and may come out worse than it could have. Take "
    "the extra minute: every plaque, hugging its edge.")

MASK_CONSCIENCE = (
    "Every mask you send becomes a lesson for the next model. A loose "
    "outline teaches it to be loose, a missed object teaches it to miss "
    "objects, and two objects painted as one teach it that they are one. "
    "Someone has to fix each of those before the next model can ship, so "
    "the next release waits and may come out worse than it could have. Take "
    "the extra minute: every object, hugging its edge.")


def community_name(name: str) -> str:
    """A community target name as it appears in a repository id.

    :param name: what the user or the caller calls the collection, e.g.
        ``"Toxoplasma PV"``; lower-cased, and every run of characters other
        than ``a-z`` and ``0-9`` becomes one underscore.
    """
    out = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not out:
        raise ValueError("a community collection needs a name")
    return out


def community_repo(target: str) -> str:
    """The dataset repository community training data for ``target`` goes to.

    One dataset repository per collection, not folders of one repository:
    every spaCR training set on Hugging Face is its own dataset repository
    beside the model it trains, and each upload arrives on it as a pull
    request. ``"figures"`` and ``"plaques"`` are Plaque Assay's two; any
    other name is ``einarolafsson/community_<name>``.

    :param target: ``"figures"``, ``"plaques"``, a name, or a full
        ``owner/repo`` id (used as it is).
    """
    if target == FIGURES_KIND:
        return COMMUNITY_FIGURES_REPO
    if target == PLAQUES_KIND:
        return COMMUNITY_PLAQUES_REPO
    if "/" in str(target):
        return str(target)
    name = community_name(target)
    if not name.startswith("community_"):
        name = "community_" + name
    return f"{COMMUNITY_OWNER}/{name}"


def community_layout(target: str) -> str:
    """``"boxes"`` for figure pages with well boxes, ``"masks"`` for everything else.

    :param target: as for :func:`community_repo`.
    """
    return BOXES_LAYOUT if target == FIGURES_KIND else MASKS_LAYOUT


def conscience_for(target: str) -> str:
    """The short text shown beside Upload, asking for careful annotations.

    Callers translate it with ``tr()``.

    :param target: as for :func:`community_repo`.
    """
    if target == FIGURES_KIND:
        return FIGURE_CONSCIENCE
    if target == PLAQUES_KIND:
        return PLAQUE_CONSCIENCE
    return MASK_CONSCIENCE


def community_readme(target: str, *, purpose: str = "") -> str:
    """A README for a new community dataset, in the layout this module writes.

    :param target: as for :func:`community_repo`.
    :param purpose: what the data will train, one sentence; a general
        sentence when empty.
    """
    repo_id = community_repo(target)
    boxes = community_layout(target) == BOXES_LAYOUT
    layout = ("  images/<page>.png        the page the boxes were drawn on\n"
              "  labels/<page>.txt        YOLO format: `0 cx cy w h`, normalised\n"
              if boxes else
              "  images/<stem>.<ext>      the original image file, unchanged\n"
              "  masks/<stem>.tif         uint16 label mask, 0 = background\n")
    return f"""---
license: cc-by-4.0
tags: [spacr, community, {'object-detection' if boxes else 'image-segmentation'}]
---
# {repo_id.split('/')[-1]}

Community training data contributed through spaCR.

## Purpose

{purpose or 'Training data for the next spaCR model of this kind.'}

## Licence

Shared under **{COMMUNITY_LICENCE}**. By uploading, a contributor confirms
that they have the right to share the images and agrees that they and the
annotations are redistributed under {COMMUNITY_LICENCE}.

## File layout

```
contributions/<contribution-id>/
  contribution.json        spaCR version, date, licence agreed, consent, counts
{layout}  meta/<stem>.json         original file name, size, and which of spaCR's
                           proposals the contributor kept, edited or removed
```

An image is only accepted with at least one annotation.

## Review

Contributions arrive as pull requests and are **reviewed by the maintainer**
before they are merged.

spaCR: https://github.com/EinarOlafsson/spacr
"""


def ensure_community_repo(target: str, token: str, *, purpose: str = "") -> str:
    """Make sure the community dataset exists; create it with a README if not.

    Only the owner of the ``einarolafsson`` namespace can create one; for
    anyone else a missing repository is an error that says who to ask.

    :param target: as for :func:`community_repo`.
    :param token: a Hugging Face token.
    :param purpose: passed to :func:`community_readme` for a new repository.
    :returns: the dataset's URL.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = community_repo(target)
    url = f"https://huggingface.co/datasets/{repo_id}"
    if api.repo_exists(repo_id, repo_type="dataset"):
        return url
    try:
        api.create_repo(repo_id, repo_type="dataset", private=False,
                        exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"{repo_id} does not exist yet and this account cannot create "
            f"it; ask the spaCR maintainer to create it ({exc})") from exc
    api.upload_file(path_or_fileobj=community_readme(
        target, purpose=purpose).encode(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset",
        commit_message="README: licence, layout, purpose, review")
    return url


def yolo_lines(boxes: Any, width: int, height: int) -> list:
    """Well boxes as YOLO label lines, ``0 cx cy w h`` normalised to 0-1.

    :param boxes: ``(x0, y0, x1, y1)`` pixel boxes; corners in either order,
        clipped to the image.
    :param width: image width in pixels.
    :param height: image height in pixels.
    :returns: one line per box with a positive area, class 0 ("plaque well").
    """
    out = []
    for box in boxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = sorted((min(max(x0, 0.0), width), min(max(x1, 0.0), width)))
        y0, y1 = sorted((min(max(y0, 0.0), height), min(max(y1, 0.0), height)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append(f"0 {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                   f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    return out


def seed_changes(seed: Any, final: Any) -> Dict[str, Any]:
    """Which of spaCR's proposed plaques the contributor kept, edited or removed.

    :param seed: the label mask spaCR proposed, or None when it proposed
        nothing.
    :param final: the label mask being contributed, same shape.
    :returns: ``{"kept", "edited", "removed"}`` as lists of seed label ids,
        and ``"added"``: how many final labels overlap no seeded plaque.
    """
    import numpy as np

    final = np.asarray(final)
    if seed is None:
        ids = [int(v) for v in np.unique(final) if v]
        return {"kept": [], "edited": [], "removed": [], "added": len(ids)}
    seed = np.asarray(seed)
    kept, edited, removed = [], [], []
    for label in (int(v) for v in np.unique(seed) if v):
        where = seed == label
        now = final == label
        if not now.any():
            removed.append(label)
        elif np.array_equal(where, now):
            kept.append(label)
        else:
            edited.append(label)
    added = sum(1 for v in np.unique(final)
                if v and not (seed[final == v] > 0).any())
    return {"kept": kept, "edited": edited, "removed": removed,
            "added": int(added)}


def _has_annotation(layout: str, item: Dict[str, Any]) -> bool:
    """Whether one image carries at least one box or one masked object."""
    import numpy as np

    if layout == BOXES_LAYOUT:
        image = np.asarray(item["image"])
        return bool(yolo_lines(item.get("boxes") or (), image.shape[1],
                               image.shape[0]))
    labels = item.get("labels")
    return labels is not None and bool(np.asarray(labels).any())


PAIR_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PAIR_MASK_EXTS = PAIR_IMAGE_EXTS + (".npy",)


def masks_dataset_target(name: str) -> str:
    """The community target a user-named image-and-mask dataset goes to.

    Always ``community_<name>``, so a dataset the user calls "figures" or
    "plaques" never lands in Plaque Assay's two collections by accident.

    :param name: what the user calls it, e.g. ``"Toxoplasma vacuoles GFP"``.
    :returns: a target :func:`community_repo` maps to
        ``einarolafsson/community_<name>``.
    :raises ValueError: when ``name`` has no letters or digits.
    """
    slug = community_name(name)
    return slug if slug.startswith("community_") else "community_" + slug


def _files_by_stem(folder: Any, exts: Tuple[str, ...]) -> Dict[str, list]:
    """Every file in ``folder`` ending in one of ``exts``, grouped by stem."""
    out: Dict[str, list] = {}
    for entry in sorted(Path(folder).iterdir()):
        if entry.is_file() and entry.name.lower().endswith(exts):
            out.setdefault(entry.stem, []).append(entry)
    return out


def pair_images_and_masks(images_dir: Any, masks_dir: Any) -> Dict[str, Any]:
    """Match an images folder to a masks folder by file name.

    An image and a mask belong together when their names match without the
    extension, so ``a.png`` pairs with ``a.tif``. Nothing is read but the
    folder listings.

    :param images_dir: the folder of images.
    :param masks_dir: the folder of label masks.
    :returns: ``pairs`` (``(image, mask)`` paths, in name order),
        ``images`` and ``masks`` (the counts), ``no_mask`` (images with no
        mask of the same name), ``no_image`` (masks with no image),
        ``duplicates`` (names that match more than one file on one side),
        and ``problem`` (why the folders cannot be read, else empty). The
        folders match when ``problem``, ``no_mask``, ``no_image`` and
        ``duplicates`` are all empty and there is at least one pair.
    """
    report: Dict[str, Any] = {"pairs": [], "images": 0, "masks": 0,
                              "no_mask": [], "no_image": [],
                              "duplicates": [], "problem": ""}
    images_dir = str(images_dir or "").strip()
    masks_dir = str(masks_dir or "").strip()
    if not images_dir or not masks_dir:
        report["problem"] = "choose both folders"
        return report
    for folder in (images_dir, masks_dir):
        if not Path(folder).is_dir():
            report["problem"] = f"{folder} is not a folder"
            return report
    if Path(images_dir).resolve() == Path(masks_dir).resolve():
        report["problem"] = "the images and the masks are the same folder"
        return report
    images = _files_by_stem(images_dir, PAIR_IMAGE_EXTS)
    masks = _files_by_stem(masks_dir, PAIR_MASK_EXTS)
    report["images"] = sum(len(v) for v in images.values())
    report["masks"] = sum(len(v) for v in masks.values())
    for side in (images, masks):
        report["duplicates"].extend(
            p.name for v in side.values() if len(v) > 1 for p in v)
    report["no_mask"] = [p.name for stem, v in images.items()
                         if stem not in masks for p in v]
    report["no_image"] = [p.name for stem, v in masks.items()
                          if stem not in images for p in v]
    report["pairs"] = [(images[stem][0], masks[stem][0])
                       for stem in images
                       if stem in masks and len(images[stem]) == 1
                       and len(masks[stem]) == 1]
    return report


def pairs_match(report: Dict[str, Any]) -> bool:
    """Whether a :func:`pair_images_and_masks` report may be uploaded.

    :param report: the report.
    :returns: True when every image has exactly one mask of the same name,
        every mask one image, and there is at least one pair.
    """
    return bool(report["pairs"]) and not (
        report["problem"] or report["no_mask"] or report["no_image"]
        or report["duplicates"]) and report["images"] == report["masks"]


def read_label_mask(path: Any) -> Any:
    """A label mask file as a 2-D array.

    :param path: a ``.tif``/``.tiff``, ``.npy`` or ordinary image file.
    :returns: the labels; a colour image keeps its first channel.
    """
    import numpy as np

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif suffix in (".tif", ".tiff"):
        import tifffile

        labels = tifffile.imread(str(path))
    else:
        import imageio.v2 as imageio

        labels = imageio.imread(str(path))
    labels = np.squeeze(np.asarray(labels))
    if labels.ndim == 3 and labels.shape[-1] in (3, 4):
        labels = labels[..., 0]
    return labels


def _image_size(path: Any) -> Optional[Tuple[int, int]]:
    """An image file's ``(height, width)`` from its header, or None.

    A TIFF is read through its series axes, so a channel or z axis in front
    of Y and X does not count as a size; anything else goes through PIL.
    None when the file cannot be read this way, so a format only a reader
    plug-in knows is left for the reviewer rather than refused.
    """
    path = Path(str(path))
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                axes, shape = str(series.axes), tuple(series.shape)
            if "Y" in axes and "X" in axes:
                return int(shape[axes.index("Y")]), int(shape[axes.index("X")])
            return None
        from PIL import Image

        with Image.open(path) as handle:
            width, height = handle.size
        return int(height), int(width)
    except Exception:
        return None


def _size_mismatches(items: Any) -> list:
    """Masks-layout items whose source image and mask differ in size.

    :returns: one ``"name: image W x H, mask W x H"`` line per mismatch.
    """
    import numpy as np

    out = []
    for item in items:
        labels = item.get("labels")
        if labels is None or not item.get("source"):
            continue
        shape = np.asarray(labels).shape
        if len(shape) < 2:
            continue
        size = _image_size(item["source"])
        if size is None or size == (int(shape[0]), int(shape[1])):
            continue
        out.append(f"{item.get('name') or Path(str(item['source'])).name}: "
                   f"image {size[1]} x {size[0]}, "
                   f"mask {int(shape[1])} x {int(shape[0])}")
    return out


def write_contribution(target: str, items: Any, dest: Any, *,
                       consent: Dict[str, Any],
                       contribution_id: str = "", notes: str = "") -> Path:
    """Lay a contribution out on disk exactly as the dataset README describes.

    Refuses the whole contribution when any image has no annotation: an
    image without boxes or masks teaches a model that it holds nothing,
    which is almost never true of an image somebody chose to send. In the
    masks layout it also refuses, before writing anything, any image whose
    mask is not the image's own size, naming each one with both sizes: a
    mask of another size does not lie on its image.

    :param target: ``"figures"`` (boxes layout), ``"plaques"`` or any other
        community name (masks layout); see :func:`community_repo`.
    :param items: one mapping per image. Boxes layout: ``name``, ``image``
        (the ``H x W x 3`` page the boxes were drawn on), ``boxes`` (pixel
        ``(x0, y0, x1, y1)``), and optionally ``source``, ``paper`` (DOI,
        citation, PMCID) and ``provenance`` (which proposed boxes were kept,
        moved or deleted). Masks layout: ``name``, ``source`` (the original
        file, copied unchanged), ``labels`` (the label mask) and optionally
        ``seed`` (spaCR's proposed mask) and ``extra`` (anything else to
        record in the image's meta file).
    :param dest: the folder the contribution folder is made in.
    :param consent: what the contributor agreed to; recorded verbatim in
        ``contribution.json``.
    :param contribution_id: the folder name; a date and a random suffix when
        empty.
    :param notes: the contributor's own words about the images, recorded in
        ``contribution.json`` when given.
    :returns: the contribution folder.
    """
    import datetime
    import json
    import shutil
    import uuid

    import numpy as np

    import spacr

    items = list(items)
    if not items:
        raise ValueError("there is nothing to contribute")
    layout = community_layout(target)
    bare = [str(item.get("name") or "?") for item in items
            if not _has_annotation(layout, item)]
    if bare:
        raise ValueError("these images have no annotations and cannot be "
                         "sent: " + ", ".join(bare))
    if layout != BOXES_LAYOUT:
        mismatched = _size_mismatches(items)
        if mismatched:
            raise ValueError(
                "these images and their masks are not the same size (width x "
                "height, in pixels), so the masks would not lie on the "
                "images: " + "; ".join(mismatched))
    repo_id = community_repo(target)
    stamp = datetime.datetime.now(datetime.timezone.utc)
    ident = contribution_id or (stamp.strftime("%Y%m%d-%H%M%S-")
                                + uuid.uuid4().hex[:8])
    root = Path(dest) / ident
    marks = "labels" if layout == BOXES_LAYOUT else "masks"
    for sub in ("images", marks, "meta"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    version = str(getattr(spacr, "__version__", ""))
    seen: Dict[str, int] = {}
    count = 0
    for item in items:
        stem = slugify(Path(str(item.get("name") or "image")).stem)
        seen[stem] = seen.get(stem, 0) + 1
        if seen[stem] > 1:
            stem = f"{stem}-{seen[stem]}"
        meta: Dict[str, Any] = {"original_name": str(item.get("name") or ""),
                                "source": str(item.get("source") or "")}
        if layout == BOXES_LAYOUT:
            from PIL import Image

            image = np.asarray(item["image"])
            height, width = image.shape[:2]
            Image.fromarray(image.astype(np.uint8)).save(
                root / "images" / f"{stem}.png")
            lines = yolo_lines(item["boxes"], width, height)
            (root / marks / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            meta.update(width=int(width), height=int(height),
                        boxes_px=[[int(round(float(v))) for v in box]
                                  for box in item["boxes"]],
                        paper=dict(item.get("paper") or {}),
                        provenance=dict(item.get("provenance") or {}))
            count += len(lines)
        else:
            import tifffile

            source = Path(str(item["source"]))
            suffix = source.suffix.lower() or ".tif"
            shutil.copyfile(source, root / "images" / f"{stem}{suffix}")
            labels = np.asarray(item["labels"])
            tifffile.imwrite(str(root / marks / f"{stem}.tif"),
                             labels.astype(np.uint16))
            objects = int(len([v for v in np.unique(labels) if v]))
            meta.update(height=int(labels.shape[0]), width=int(labels.shape[1]),
                        objects=objects,
                        provenance=seed_changes(item.get("seed"), labels),
                        **dict(item.get("extra") or {}))
            count += objects
        (root / "meta" / f"{stem}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
    record = {
        "id": ident, "target": str(target), "layout": layout, "repo": repo_id,
        "licence": COMMUNITY_LICENCE, "consent": dict(consent),
        "created": stamp.isoformat(), "spacr_version": version,
        "images": len(items),
        ("boxes" if layout == BOXES_LAYOUT else "objects"): count,
    }
    if str(notes or "").strip():
        record["notes"] = str(notes).strip()
    (root / "contribution.json").write_text(json.dumps(record, indent=2),
                                            encoding="utf-8")
    return root


MAX_CONTRIBUTION_BYTES = 2_000_000_000


def central_contribute(folder: Any, target: str) -> str:
    """Send a contribution folder through the central upload Space.

    For a contributor with no Hugging Face login: the folder goes to the
    Space (see ``tools/model_upload_space/``) as one uncompressed tar, and
    the Space checks it and opens the pull request with its own token. The
    Space accepts only ``figures``, ``plaques`` and ``community_<name>``
    targets, about :data:`MAX_CONTRIBUTION_BYTES`, the file types
    :func:`write_contribution` writes, and every image paired with its mask
    or labels and its meta file.

    :param folder: a folder :func:`write_contribution` made.
    :param target: as for :func:`community_repo`; sent as the name the Space
        resolves, ``figures``, ``plaques`` or ``community_<name>``.
    :returns: the pull request's URL.
    :raises RuntimeError: when no endpoint is configured, the folder is over
        the limit, or the Space refuses it (with the Space's reason).
    """
    import shutil
    import tarfile
    import tempfile

    if not CENTRAL_ENDPOINT:
        raise RuntimeError("no central upload endpoint is configured")
    folder = Path(folder)
    if not (folder / "contribution.json").is_file():
        raise RuntimeError(f"{folder} is not a contribution folder")
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    if total > MAX_CONTRIBUTION_BYTES:
        raise RuntimeError(
            f"the contribution holds {total / 1e9:.1f} GB and the upload "
            f"service takes about {MAX_CONTRIBUTION_BYTES / 1e9:.0f} GB; send "
            "it in parts, or with your own Hugging Face login")
    if target in (FIGURES_KIND, PLAQUES_KIND):
        name = str(target)
    else:
        name = community_repo(target)
    work = tempfile.mkdtemp(prefix="spacr-contribution-")
    tarball = os.path.join(work, f"{folder.name}.tar")
    try:
        with tarfile.open(tarball, "w") as tar:
            tar.add(str(folder), arcname=folder.name)
        base = CENTRAL_ENDPOINT.rstrip("/")
        handle = _gradio_upload(base, tarball, 3600)
        reply = _gradio_call(base, "contribute", [handle, name], wait=1800)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    found = re.search(r"https://huggingface\.co/\S+", reply)
    if not found:
        raise RuntimeError(f"the upload service replied: {reply}")
    return found.group(0)


def contribute(folder: Any, target: str, token: Optional[str] = None) -> str:
    """Send a contribution folder as a pull request. Returns its URL.

    A pull request rather than a commit, whoever sends it: every
    contribution is reviewed before it becomes training data, and any
    logged-in Hugging Face user may open one on a public dataset, so the
    uploader's own token is enough and no write token ships with spaCR.
    A dataset that does not exist yet is created first by
    :func:`ensure_community_repo`, which only its owner can do.

    The contributor's own login is preferred. With no login, or when the
    login cannot create a dataset that does not exist yet, the folder goes
    through the central upload Space instead (:func:`central_contribute`),
    which opens the same pull request with its own token.

    :param folder: a folder :func:`write_contribution` made.
    :param target: as for :func:`community_repo`; picks the repository.
    :param token: the uploader's Hugging Face token; :func:`find_token`'s
        when None.
    """
    from huggingface_hub import HfApi

    folder = Path(folder)
    repo_id = community_repo(target)
    token = token or find_token()
    if not token:
        return central_contribute(folder, target)
    try:
        ensure_community_repo(target, token)
    except RuntimeError:
        if not CENTRAL_ENDPOINT:
            raise
        return central_contribute(folder, target)
    info = HfApi(token=token).upload_folder(
        folder_path=str(folder), path_in_repo=f"contributions/{folder.name}",
        repo_id=repo_id, repo_type="dataset", create_pr=True,
        commit_message=f"Community contribution {folder.name}")
    return str(getattr(info, "pr_url", "") or
               f"https://huggingface.co/datasets/{repo_id}/discussions")

