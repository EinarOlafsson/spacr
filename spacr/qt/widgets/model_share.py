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
    import os as _os
    import time
    import urllib.request
    import uuid

    if not CENTRAL_ENDPOINT:
        raise RuntimeError("no central upload endpoint is configured")
    base = CENTRAL_ENDPOINT.rstrip("/")

    boundary = uuid.uuid4().hex
    with open(path, "rb") as handle:
        payload = handle.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
            f"filename=\"{_os.path.basename(path)}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        base + "/gradio_api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    uploaded = json.loads(urllib.request.urlopen(request, timeout=600).read())

    train_handle = None
    train_dir = str(fields.get("train_data_dir") or "")
    if train_dir:
        tarball = pack_training_data(train_dir)
        boundary = uuid.uuid4().hex
        with open(tarball, "rb") as handle:
            blob = handle.read()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                f"filename=\"{_os.path.basename(tarball)}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n").encode()
        body += blob + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            base + "/gradio_api/upload", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        sent = json.loads(urllib.request.urlopen(request, timeout=7200).read())
        train_handle = {"path": sent[0], "meta": {"_type": "gradio.FileData"}}
        try:
            _os.unlink(tarball)
        except OSError:
            pass

    call = dict(data=[{"path": uploaded[0], "meta": {"_type": "gradio.FileData"}},
                      str(fields.get("display_name") or ""),
                      str(fields.get("kind") or "cellpose"),
                      str(fields.get("trained_on") or ""),
                      json.dumps(fields),
                      str(fields.get("contact") or ""),
                      train_handle])
    request = urllib.request.Request(
        base + "/gradio_api/call/upload", data=json.dumps(call).encode(),
        headers={"Content-Type": "application/json"})
    event = json.loads(urllib.request.urlopen(request, timeout=120).read())["event_id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        with urllib.request.urlopen(
                base + f"/gradio_api/call/upload/{event}", timeout=120) as reply:
            text = reply.read().decode("utf-8", "replace")
        if "event: complete" in text:
            answer = json.loads(text.rsplit("data:", 1)[1].strip())[0]
            if str(answer).startswith("error:"):
                raise RuntimeError(str(answer)[6:].strip())
            return str(answer)
        time.sleep(3)
    raise RuntimeError("the upload endpoint did not answer in time")


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


COMMUNITY_FIGURES_REPO = "einarolafsson/community_toxoplasma_plaque_figures"
COMMUNITY_PLAQUES_REPO = "einarolafsson/community_toxoplasma_plaques"
COMMUNITY_LICENCE = "CC BY 4.0"
FIGURES_KIND = "figures"
PLAQUES_KIND = "plaques"


def community_repo(kind: str) -> str:
    """The dataset repository community training data of ``kind`` goes to.

    Two repositories, not two folders of one: every spaCR training set on
    Hugging Face is its own dataset repository beside the model it trains,
    and a contributor's upload arrives as a pull request on it.

    :param kind: ``"figures"`` (figure pages with well boxes, for the YOLO
        well detector) or ``"plaques"`` (plaque images with plaque masks, for
        the next plaque model).
    """
    if kind == FIGURES_KIND:
        return COMMUNITY_FIGURES_REPO
    if kind == PLAQUES_KIND:
        return COMMUNITY_PLAQUES_REPO
    raise ValueError(f"no community repository for {kind!r}")


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


def _has_annotation(kind: str, item: Dict[str, Any]) -> bool:
    """Whether one image carries at least one box or one plaque."""
    import numpy as np

    if kind == FIGURES_KIND:
        image = np.asarray(item["image"])
        return bool(yolo_lines(item.get("boxes") or (), image.shape[1],
                               image.shape[0]))
    labels = item.get("labels")
    return labels is not None and bool(np.asarray(labels).any())


def write_contribution(kind: str, items: Any, dest: Any, *,
                       consent: Dict[str, Any],
                       contribution_id: str = "") -> Path:
    """Lay a contribution out on disk exactly as the dataset README describes.

    Refuses the whole contribution when any image has no annotation: an
    image without boxes or masks teaches a detector that it holds nothing,
    which is almost never true of an image somebody chose to send.

    :param kind: ``"figures"`` or ``"plaques"``.
    :param items: one mapping per image. Figures: ``name``, ``image`` (the
        ``H x W x 3`` page the boxes were drawn on), ``boxes`` (pixel
        ``(x0, y0, x1, y1)``), and optionally ``source``, ``paper`` (DOI,
        citation, PMCID) and ``provenance`` (which proposed boxes were kept,
        moved or deleted). Plaques: ``name``, ``source`` (the original file,
        copied unchanged), ``labels`` (the label mask) and optionally
        ``seed`` (spaCR's proposed mask).
    :param dest: the folder the contribution folder is made in.
    :param consent: what the contributor agreed to; recorded verbatim in
        ``contribution.json``.
    :param contribution_id: the folder name; a date and a random suffix when
        empty.
    :returns: the contribution folder.
    """
    import datetime
    import json
    import shutil
    import uuid

    import numpy as np

    items = list(items)
    if not items:
        raise ValueError("there is nothing to contribute")
    bare = [str(item.get("name") or "?") for item in items
            if not _has_annotation(kind, item)]
    if bare:
        raise ValueError("these images have no annotations and cannot be "
                         "sent: " + ", ".join(bare))
    repo_id = community_repo(kind)
    stamp = datetime.datetime.now(datetime.timezone.utc)
    ident = contribution_id or (stamp.strftime("%Y%m%d-%H%M%S-")
                                + uuid.uuid4().hex[:8])
    root = Path(dest) / ident
    marks = "labels" if kind == FIGURES_KIND else "masks"
    for sub in ("images", marks, "meta"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    import spacr

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
        if kind == FIGURES_KIND:
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
            plaques = int(len([v for v in np.unique(labels) if v]))
            meta.update(height=int(labels.shape[0]), width=int(labels.shape[1]),
                        plaques=plaques,
                        provenance=seed_changes(item.get("seed"), labels))
            count += plaques
        (root / "meta" / f"{stem}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
    (root / "contribution.json").write_text(json.dumps({
        "id": ident, "kind": kind, "repo": repo_id,
        "licence": COMMUNITY_LICENCE, "consent": dict(consent),
        "created": stamp.isoformat(), "spacr_version": version,
        "images": len(items),
        ("boxes" if kind == FIGURES_KIND else "plaques"): count,
    }, indent=2), encoding="utf-8")
    return root


def contribute(folder: Any, kind: str, token: str) -> str:
    """Send a contribution folder as a pull request. Returns its URL.

    A pull request rather than a commit, whoever sends it: every
    contribution is reviewed before it becomes training data, and any
    logged-in Hugging Face user may open one on a public dataset, so the
    uploader's own token is enough and no write token ships with spaCR.

    :param folder: a folder :func:`write_contribution` made.
    :param kind: ``"figures"`` or ``"plaques"``; picks the repository.
    :param token: the uploader's Hugging Face token (:func:`find_token`).
    """
    from huggingface_hub import HfApi

    folder = Path(folder)
    repo_id = community_repo(kind)
    info = HfApi(token=token).upload_folder(
        folder_path=str(folder), path_in_repo=f"contributions/{folder.name}",
        repo_id=repo_id, repo_type="dataset", create_pr=True,
        commit_message=f"Community contribution {folder.name}")
    return str(getattr(info, "pr_url", "") or
               f"https://huggingface.co/datasets/{repo_id}/discussions")
