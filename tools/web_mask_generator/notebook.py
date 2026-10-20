"""Helpers for the Google Colab mask generator notebook, and its builder.

The notebook runs the same spaCR pipeline as the operator command
(``run.main``) and writes the same ZIP as the site and the Space: image, mask,
flows, cell probability, objects.csv, README and manifest. Weights come from
the bundled catalogue's public Hugging Face files and are checked against the
published SHA-256 before use. No account token is used and nothing is uploaded.

Regenerate the notebooks after editing the template below::

    python -m tools.web_mask_generator.notebook --write
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from spacr.qt.i18n import tr

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent / "colab"
EXAMPLES_FILE = NOTEBOOK_DIR / "examples.json"
REPOSITORY = "EinarOlafsson/spacr"
BRANCH = "nightly"
FAMILIES = (
    "toxoplasma_pv", "toxoplasma_plaque", "toxoplasma_well_detector", "cell_from_hoechst",
    "nuclei_from_cellmask", "toxoplasma_from_cellmask", "toxoplasma_from_hoechst", "live_cell",
)
DEFAULT_DIAMETER = {"toxoplasma_plaque": "auto"}


def colab_url(family=None, *, branch=BRANCH):
    """The Open in Colab link for the generic notebook or one model class."""
    name = "spacr_mask_generator.ipynb" if family is None else f"by_class/{family}.ipynb"
    return f"https://colab.research.google.com/github/{REPOSITORY}/blob/{branch}/tools/web_mask_generator/colab/{name}"


def device_report():
    """Say whether this runtime has a GPU, in one line."""
    try:
        import torch
        if torch.cuda.is_available():
            return tr("GPU: {name}. Runs take seconds to a minute.", name=torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return tr("No GPU on this runtime: spaCR runs on the CPU, which can take several minutes per image. "
              "For a GPU choose Runtime > Change runtime type > T4 GPU, then run all cells again.")


def model_versions(family):
    """The published versions of one model class and its default key."""
    from .registry import model_families
    for entry in model_families():
        if entry["family"] == family:
            return entry["default"], [version["key"] for version in entry["versions"]]
    raise ValueError(tr("Unknown web model class: {family}", family=family))


def fetch_model(family, model_key=None, *, cache_dir="spacr-models"):
    """Download one published checkpoint once and verify its SHA-256.

    :returns: ``{"key", "version", "sha256", "path"}`` for ``generate``.
    """
    from spacr import model_zoo
    from tools.build_model_cards import published_models

    from .registry import resolve_trusted_model

    default, versions = model_versions(family)
    key = default if model_key in (None, "", "default") else model_key
    if key not in versions:
        raise ValueError(tr("Model {key} is not a version of {family}. Choose one of: {versions}",
                            key=key, family=family, versions=", ".join(versions)))
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    index_file = cache / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.is_file() else {}
    path = Path(index[key]) if key in index else None
    if path is None or not path.is_file():
        entry = next(model_zoo._entry_from_mapping(record, source="bundled") for record in published_models()
                     if model_zoo._entry_from_mapping(record, source="bundled").key == key)
        shown = [-1]

        def progress(done, total):
            step = int(done * 10 / total) if total else done // 100000000
            if step != shown[0]:
                shown[0] = step
                print(tr("Downloading {key}: {done} MB of {total}", key=key, done=done // 1000000,
                         total=f"{total // 1000000} MB" if total else "?"), flush=True)

        path = Path(model_zoo.fetch(entry, cache / key, progress=progress))
        index[key] = str(path.resolve())
        index_file.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    resolved = resolve_trusted_model(family, {key: path}, model_key=key)
    metadata = resolved["metadata"]
    return {"key": key, "version": metadata["version"], "sha256": metadata["sha256"], "path": str(resolved["checkpoint"])}


def example_records():
    """Pinned held-out test images, the same list the site uses."""
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))["records"]


def random_example(family, *, folder="."):
    """Download one random held-out test image for ``family``; returns (path, credit)."""
    from .examples import ExampleStore
    store = ExampleStore(example_records())
    with store.random_image(family) as example:
        target = Path(folder) / example["image"].name
        shutil.copyfile(example["image"], target)
        credit = tr("Test image {name} from {url} ({credit}).", name=target.name,
                    url=example["source"]["source_url"], credit=example["source"]["credit"])
    return target, credit


def _channel_count(description):
    if description["format"] == "PDF":
        return 3
    series = description["series"][0]
    axes = series["axes"].replace("S", "C") if "S" in series["axes"] and "C" not in series["axes"] else series["axes"]
    return dict(zip(axes, series["shape"])).get("C", 1)


def _number(text, name):
    try:
        return float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(tr("{name} must be a number, or left blank.", name=name)) from error


def generate(image, family, model, *, output_dir=".", diameter="", flow_threshold=0.4,
             cellprob_threshold="", channels="", magnification=20, series=0, page=0, pdf_dpi=150,
             detection_confidence=0.25, minimum_axis_ratio=0.7):
    """Run the spaCR pipeline on one image and write the result ZIP.

    ``diameter``: blank uses the magnification rule (not for plaques), ``auto``
    lets the model choose (native scale for plaques), a number is pixels.
    ``channels``: blank uses channel 0, or all three of an RGB image for
    plaques and wells; otherwise zero-based numbers such as ``"0"`` or ``"0,1,2"``.
    """
    from .inputs import describe_upload
    from .run import main

    image = Path(image)
    if family not in FAMILIES:
        raise ValueError(tr("Unknown web model class: {family}", family=family))
    description = describe_upload(image)
    if str(channels).strip():
        chosen = [int(part) for part in str(channels).replace(" ", "").split(",") if part != ""]
    elif family in ("toxoplasma_plaque", "toxoplasma_well_detector") and _channel_count(description) == 3:
        chosen = [0, 1, 2]
    else:
        chosen = [0]
    detector = family == "toxoplasma_well_detector"
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in image.stem)[:80] or "image"
    output = Path(output_dir) / f"{stem}-spacr-{'wells' if detector else 'masks'}.zip"
    argv = [str(image), "--family", family, "--output", str(output), "--checkpoint", model["path"],
            "--sha256", model["sha256"], "--model-id", model["key"], "--model-version", model["version"],
            "--series", str(int(series)), "--page", str(int(page)), "--pdf-dpi", str(int(pdf_dpi))]
    for channel in chosen:
        argv += ["--channel", str(channel)]
    if detector:
        argv += ["--confidence", str(float(detection_confidence)), "--min-axis-ratio", str(float(minimum_axis_ratio))]
    else:
        text = str(diameter).strip().lower()
        if text == "auto":
            argv.append("--native-scale")
        elif text:
            argv += ["--diameter", str(_number(text, tr("Diameter")))]
        elif family == "toxoplasma_plaque":
            raise ValueError(tr("Plaques need a diameter in pixels, or auto."))
        argv += ["--flow-threshold", str(float(flow_threshold))]
        if str(cellprob_threshold).strip():
            argv += ["--cellprob-threshold", str(_number(cellprob_threshold, tr("Cell probability threshold")))]
        if family != "toxoplasma_plaque":
            argv += ["--magnification", str(float(magnification))]
    main(argv)
    return output


def summary(zip_path):
    """Object or well count and the file list of a result ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("manifest.json")) if "manifest.json" in names else {}
    count = manifest.get("object_count", manifest.get("detection_count"))
    return {"count": count, "files": names}


def show(zip_path):
    """Draw the image with mask outlines, or with well boxes, from a result ZIP."""
    import io

    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile
    from skimage.segmentation import find_boundaries

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        image = tifffile.imread(io.BytesIO(archive.read("image.tif"))) if "image.tif" in names else None
        mask = tifffile.imread(io.BytesIO(archive.read("mask.tif"))) if "mask.tif" in names else None
        boxes = json.loads(archive.read("boxes.json")) if "boxes.json" in names else None
    if image is None:
        return None
    image = np.asarray(image, dtype=float)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    low, high = np.percentile(image, (1, 99.5))
    image = np.clip((image - low) / (high - low or 1), 0, 1)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
    if mask is not None:
        edges = np.ma.masked_where(~find_boundaries(np.asarray(mask), mode="thick"), np.ones(mask.shape))
        axis.imshow(edges, cmap="autumn", alpha=0.9, interpolation="nearest")
    for box in boxes or []:
        x0, y0, x1, y1 = (box[k] for k in ("x0", "y0", "x1", "y1"))
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", linewidth=1.5))
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def offer_download(path):
    """Start a browser download in Colab; elsewhere print the path."""
    try:
        from google.colab import files
    except ImportError:
        print(tr("Result ZIP: {path}", path=Path(path).resolve()))
        return
    files.download(str(path))


def _markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text, **metadata):
    return {"cell_type": "code", "execution_count": None, "metadata": metadata, "outputs": [], "source": _lines(text)}


def _lines(text):
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def notebook(family=None):
    """The notebook as a JSON-ready dict; ``family`` preselects a model class."""
    chosen = family or FAMILIES[0]
    options = json.dumps(list(FAMILIES))
    diameter = DEFAULT_DIAMETER.get(chosen, "")
    cells = [
        _markdown("""
# spaCR mask generator (Google Colab)

Masks from a spaCR model, run by the real spaCR pipeline on Google's free GPU under your own Google account.
The result is the same ZIP as the [spaCR mask generator page](https://einarolafsson.github.io/projects/spacr/#mask-generator):
the image, the label mask, the flows, the cell probability (when the model returns them), `objects.csv`, a README and a manifest.

1. For a GPU, choose **Runtime > Change runtime type > T4 GPU** (free). Without one the notebook still runs on the CPU, slowly.
2. Choose **Runtime > Run all**, or run the cells one by one.
3. Upload an image when asked, or leave `USE_TEST_IMAGE` ticked for a held-out test image from Hugging Face.

Your image stays in this Colab runtime; it is not sent to spaCR or its authors. Model weights are downloaded from Hugging Face and checked against their published SHA-256.
"""),
        _markdown("""
## 1. Install spaCR

Downloads spaCR from GitHub and installs it with its dependencies (a few minutes). If Colab asks to restart the session, restart it and run the cells from step 2 on.
"""),
        _code(f"""
SPACR_BRANCH = "{BRANCH}"  #@param {{type:"string"}}
import os, sys
if not os.path.isdir("/content/spacr"):
    !git clone --quiet --depth 1 --branch {{SPACR_BRANCH}} https://github.com/{REPOSITORY}.git /content/spacr
%pip install --quiet "/content/spacr[plaque]" "pypdfium2>=5.13.0,<6"
sys.path.insert(0, "/content/spacr")
""", cellView="form"),
        _markdown("""
## 2. Choose the model and the settings

- **MODEL_CLASS**: what to segment. `toxoplasma_well_detector` returns one box per well instead of a mask.
- **MODEL_VERSION**: `default` is the version the spaCR desktop app uses.
- **DIAMETER**: object diameter in pixels; blank uses the magnification rule (`MAGNIFICATION`); `auto` lets the model choose (native scale for plaques).
- **FLOW_THRESHOLD** and **CELLPROB_THRESHOLD**: Cellpose thresholds; a blank cell probability threshold keeps the model's default.
- **CHANNELS**: zero-based channel numbers, for example `0` or `0,1,2`; blank uses channel 0 (all three of an RGB image for plaques and wells).
"""),
        _code(f"""
MODEL_CLASS = "{chosen}"  #@param {options}
MODEL_VERSION = "default"  #@param {{type:"string"}}
DIAMETER = "{diameter}"  #@param {{type:"string"}}
MAGNIFICATION = 20  #@param {{type:"number"}}
FLOW_THRESHOLD = 0.4  #@param {{type:"number"}}
CELLPROB_THRESHOLD = ""  #@param {{type:"string"}}
CHANNELS = ""  #@param {{type:"string"}}
DETECTION_CONFIDENCE = 0.25  #@param {{type:"number"}}
USE_TEST_IMAGE = True  #@param {{type:"boolean"}}

import os, sys
sys.path.insert(0, "/content/spacr")
from tools.web_mask_generator import notebook as spacr_web
print(spacr_web.device_report())
default_key, versions = spacr_web.model_versions(MODEL_CLASS)
print("Model class:", MODEL_CLASS, "| versions:", ", ".join(versions), "| default:", default_key)
""", cellView="form"),
        _markdown("""
## 3. Choose an image

TIFF and OME-TIFF (multi-channel), PNG, JPEG, BMP and PDF; at most 64 MB.
"""),
        _code("""
from pathlib import Path
os.makedirs("/content/input", exist_ok=True)
if USE_TEST_IMAGE:
    IMAGE, credit = spacr_web.random_example(MODEL_CLASS, folder="/content/input")
    print(credit)
else:
    from google.colab import files
    uploaded = files.upload()
    name = next(iter(uploaded))
    IMAGE = Path("/content/input") / name
    IMAGE.write_bytes(uploaded[name])
print("Image:", IMAGE)
"""),
        _markdown("""
## 4. Run

Downloads the model once (about 1.2 GB for segmentation models, 5 MB for the well detector), runs spaCR and draws the result.
"""),
        _code("""
model = spacr_web.fetch_model(MODEL_CLASS, MODEL_VERSION, cache_dir="/content/spacr-models")
os.makedirs("/content/output", exist_ok=True)
RESULT = spacr_web.generate(IMAGE, MODEL_CLASS, model, output_dir="/content/output", diameter=DIAMETER,
                            flow_threshold=FLOW_THRESHOLD, cellprob_threshold=CELLPROB_THRESHOLD,
                            channels=CHANNELS, magnification=MAGNIFICATION,
                            detection_confidence=DETECTION_CONFIDENCE)
info = spacr_web.summary(RESULT)
print("Found:", info["count"], "| model:", model["key"])
print("ZIP contents:", ", ".join(info["files"]))
spacr_web.show(RESULT)
"""),
        _markdown("""
## 5. Download the ZIP

To try other settings, change them in step 2 and run steps 2 and 4 again (the model is not downloaded again).
"""),
        _code("""
spacr_web.offer_download(RESULT)
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"spacr_mask_generator_{chosen}.ipynb" if family else "spacr_mask_generator.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def rendered():
    """Every notebook file this module owns, as {relative path: text}."""
    def text(value):
        return json.dumps(value, indent=1, ensure_ascii=False) + "\n"

    files = {"spacr_mask_generator.ipynb": text(notebook())}
    files.update({f"by_class/{family}.ipynb": text(notebook(family)) for family in FAMILIES})
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=tr("Write or check the Colab mask generator notebooks."))
    parser.add_argument("--write", action="store_true", help=tr("Write the notebooks."))
    arguments = parser.parse_args(argv)
    stale = []
    for name, text in rendered().items():
        path = NOTEBOOK_DIR / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(name)
            if arguments.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    if stale and not arguments.write:
        print(tr("Out of date: {names}. Run with --write.", names=", ".join(stale)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
