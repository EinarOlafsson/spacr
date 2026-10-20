"""Operator-only export of web model classes to ONNX for the in-browser route.

The browser runs these graphs with onnxruntime-web (WebGPU, WASM fallback).
Weights are verified against space/models.json before loading. Nothing is
downloaded unless --download is given, and nothing is ever uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
MODELS_JSON = HERE / "space" / "models.json"
GENERATED_BY = "tools/web_mask_generator/export_onnx.py"
SCHEMA = 1
OPSET = 17
BSIZE = 256
DETECTOR_IMGSZ = 640
FLOWS_TOLERANCE = 1e-3
DETECTOR_TOLERANCE = {"boxes": 1e-2, "scores": 1e-3}
CHECK_SEED = 698
PATHOGEN_FAMILIES = ("toxoplasma_pv", "toxoplasma_from_cellmask", "toxoplasma_from_hoechst")
PUBLISH_HINT = "https://huggingface.co/datasets/einarolafsson/models/resolve/main/web-onnx/"


class ExportError(RuntimeError):
    """A plain operator diagnostic."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalogue(path=MODELS_JSON):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("models"), list):
        raise ExportError(f"unexpected models.json schema: {path}")
    rows = {}
    for row in data["models"]:
        if row["key"] in rows:
            raise ExportError(f"duplicate model key in models.json: {row['key']}")
        rows[row["key"]] = row
    return rows


def select_keys(catalogue, keys, everything):
    if everything and keys:
        raise ExportError("use either --key or --all, not both")
    if everything:
        return sorted(catalogue)
    if not keys:
        raise ExportError("choose at least one --key, or --all")
    unknown = [key for key in keys if key not in catalogue]
    if unknown:
        raise ExportError("unknown model key(s): " + ", ".join(unknown))
    return list(dict.fromkeys(keys))


def source_url(row):
    from .provision import _url
    return _url(row)


def parse_weights(pairs):
    result = {}
    for pair in pairs or ():
        key, separator, path = pair.partition("=")
        if not separator or not key or not path:
            raise ExportError(f"--weights expects KEY=PATH, got {pair!r}")
        result[key] = Path(path).expanduser()
    return result


def resolve_weights(row, *, explicit, weights_dir, download, download_dir):
    """Explicit path first, then a basename match in --weights-dir, then opt-in download."""
    key = row["key"]
    if key in explicit:
        path = explicit[key]
        if not path.is_file():
            raise ExportError(f"{key}: weights file not found: {path}")
        return path, "explicit"
    if weights_dir is not None:
        candidate = Path(weights_dir) / PurePosixPath(row["filename"]).name
        if candidate.is_file():
            return candidate, "weights-dir"
    if download:
        return download_weights(row, download_dir), "download"
    raise ExportError(f"{key}: no local weights; pass --weights {key}=PATH, --weights-dir DIR or --download")


def verify_weights(path, row):
    size = Path(path).stat().st_size
    if size != row["bytes"]:
        raise ExportError(f"{row['key']}: {path} has {size} bytes, models.json says {row['bytes']}")
    digest = sha256_file(path)
    if digest != row["sha256"]:
        raise ExportError(f"{row['key']}: SHA-256 mismatch for {path}: {digest} != {row['sha256']}")
    return digest


def download_weights(row, directory):
    """Fetch the public, revision-pinned resolve URL without credentials, then verify."""
    from .provision import _public_url, _transport
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (row["key"] + "_" + row["sha256"][:16] + PurePosixPath(row["filename"]).suffix)
    if target.is_file() and target.stat().st_size == row["bytes"] and sha256_file(target) == row["sha256"]:
        return target
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=directory)
    pending = Path(temporary)
    try:
        digest, total = hashlib.sha256(), 0
        with os.fdopen(descriptor, "wb") as handle, _transport(source_url(row), 60) as response:
            _public_url(response.geturl())
            if response.status != 200:
                raise ExportError(f"{row['key']}: download returned HTTP {response.status}")
            while True:
                chunk = response.read(1024 ** 2)
                if not chunk:
                    break
                total += len(chunk)
                if total > row["bytes"]:
                    raise ExportError(f"{row['key']}: download is larger than models.json says")
                digest.update(chunk)
                handle.write(chunk)
        if total != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ExportError(f"{row['key']}: downloaded file does not match models.json")
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    return target


def _diff(reference, candidate):
    import numpy as np
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ExportError(f"output shape mismatch: {reference.shape} != {candidate.shape}")
    absolute = float(np.max(np.abs(reference - candidate))) if reference.size else 0.0
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    return {"max_abs_diff": absolute, "max_rel_diff": absolute / max(scale, 1e-12), "reference_max_abs": scale}


def _consolidate(produced, destination):
    """Write one self-contained .onnx file; the browser fetches a single URL."""
    produced, destination = Path(produced), Path(destination)
    siblings = [path for path in produced.parent.iterdir() if path != produced]
    if siblings:
        import onnx
        model = onnx.load(str(produced), load_external_data=True)
        onnx.save_model(model, str(destination), save_as_external_data=False)
    else:
        shutil.move(str(produced), str(destination))


def _flows_module(net):
    import torch

    class FlowsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            return self.inner(image)[0]

    return FlowsOnly(net).eval()


def load_cellpose_net(weights):
    import torch
    from cellpose import models
    model = models.CellposeModel(gpu=False, pretrained_model=str(weights), use_bfloat16=False,
                                 device=torch.device("cpu"))
    model.net.eval()
    return model.net


def export_cellpose(net, destination, *, opset=OPSET):
    import numpy as np
    import torch
    wrapper = _flows_module(net)
    example = torch.from_numpy(np.random.default_rng(CHECK_SEED).random((1, 3, BSIZE, BSIZE), dtype=np.float32))
    with tempfile.TemporaryDirectory(prefix=".export-", dir=Path(destination).parent) as directory:
        produced = Path(directory) / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(wrapper, (example,), str(produced), input_names=["image"], output_names=["flows"],
                              dynamic_axes={"image": {0: "N"}, "flows": {0: "N"}}, opset_version=opset,
                              dynamo=False, do_constant_folding=True)
        _consolidate(produced, destination)
    return destination


def _ort_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def check_cellpose(net, onnx_path, *, tiles=2, tolerance=FLOWS_TOLERANCE):
    import numpy as np
    import torch
    rng = np.random.default_rng(CHECK_SEED)
    image = rng.random((tiles, 3, BSIZE, BSIZE), dtype=np.float32)
    with torch.no_grad():
        reference = _flows_module(net)(torch.from_numpy(image)).float().numpy()
    candidate = _ort_session(onnx_path).run(["flows"], {"image": image})[0]
    flows = _diff(reference, candidate)
    channels = {name: _diff(reference[:, index], candidate[:, index])
                for index, name in enumerate(("dY", "dX", "cellprob"))}
    passed = flows["max_abs_diff"] <= tolerance
    return {"runtime": "onnxruntime CPUExecutionProvider", "reference": "torch float32 CPU",
            "inputs": f"{tiles} tiles, numpy default_rng({CHECK_SEED}).random float32 in [0, 1)",
            "rel_definition": "max_abs_diff / max(|torch output|)",
            "outputs": {"flows": {**flows, "channels": channels}},
            "tolerance": {"flows_max_abs_diff": tolerance}, "passed": passed}


def _detector_info(model):
    head = model.model.model[-1]
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    return {"num_classes": int(head.nc), "class_names": [names[index] for index in sorted(names)],
            "end2end": bool(getattr(head, "end2end", False)),
            "one2one_head_present": getattr(head, "one2one_cv2", None) is not None, "max_det": int(getattr(head, "max_det", 300)),
            "stride": int(max(float(value) for value in model.model.stride)), "head": type(head).__name__}


def export_detector(weights, destination, *, opset=OPSET, imgsz=DETECTOR_IMGSZ, dynamic=True):
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory(prefix=".export-", dir=Path(destination).parent) as directory:
        local = Path(directory) / "detector.pt"
        shutil.copyfile(weights, local)
        model = YOLO(str(local))
        info = _detector_info(model)
        produced = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=dynamic, simplify=False,
                                device="cpu")
        shutil.move(str(produced), str(destination))
    return info


def check_detector(weights, onnx_path, *, imgsz=DETECTOR_IMGSZ, tolerance=DETECTOR_TOLERANCE):
    import numpy as np
    import torch
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO
    image = np.random.default_rng(CHECK_SEED).random((1, 3, imgsz, imgsz), dtype=np.float32)
    network = YOLO(str(weights)).model.float().eval()
    with torch.no_grad():
        reference = network(torch.from_numpy(image))
    reference = (reference[0] if isinstance(reference, (list, tuple)) else reference).float().numpy()
    session = _ort_session(onnx_path)
    candidate = session.run(None, {session.get_inputs()[0].name: image})[0]
    whole = _diff(reference, candidate)
    result = {"runtime": "onnxruntime CPUExecutionProvider", "reference": "ultralytics torch float32 CPU (unfused)",
              "inputs": f"1 image, numpy default_rng({CHECK_SEED}).random float32 in [0, 1), {imgsz}x{imgsz}",
              "rel_definition": "max_abs_diff / max(|torch output|)", "tolerance": dict(tolerance),
              "outputs": {"output0": whole}}
    if reference.ndim == 3 and reference.shape[1] > 4:
        boxes, scores = _diff(reference[:, :4], candidate[:, :4]), _diff(reference[:, 4:], candidate[:, 4:])
        result["outputs"]["output0"].update({"boxes": boxes, "scores": scores})
        result["passed"] = boxes["max_abs_diff"] <= tolerance["boxes"] and scores["max_abs_diff"] <= tolerance["scores"]
    else:
        result["passed"] = whole["max_abs_diff"] <= tolerance["boxes"]
    result["output_shape"] = list(candidate.shape)
    return result


def spacr_defaults(family):
    """Desktop and web-app thresholds for one family, read from spaCR where possible."""
    web = {"flow_threshold": 0.4, "cellprob_threshold": -1.0 if family in PATHOGEN_FAMILIES else 0.0,
           "source": "tools/web_mask_generator/app.py defaults"}
    if family == "toxoplasma_plaque":
        return {"web_app": {**web, "diameter": "required; resolve it as the desktop plaque run does"},
                "desktop": {"flow_threshold": 0.4, "cellprob_threshold": 0.0,
                            "source": "tools/web_mask_generator/plaque.py run_plaque_download"}}
    try:
        from .masks import OBJECT_ROLES
        from spacr.settings import set_default_settings_preprocess_generate_masks
        from spacr.utils import _get_diam
    except ImportError as error:
        return {"web_app": web, "desktop": {"unavailable": str(error)}}
    role = OBJECT_ROLES[family]
    settings = set_default_settings_preprocess_generate_masks({"verbose": False})
    desktop = {"object_role": role,
               f"{role}_flow_threshold": settings[f"{role}_flow_threshold"],
               f"{role}_cellprob_threshold": settings[f"{role}_cellprob_threshold"],
               f"{role}_diameter": settings[f"{role}_diameter"],
               "magnification": settings["magnification"],
               "diameter_at_default_magnification": _get_diam(settings["magnification"], obj=role),
               "diameter_rule": f"blank {role}_diameter resolves from magnification via spacr.utils._get_diam(magnification, obj='{role}')",
               "normalization_note": "the spaCR desktop route normalizes with spacr.io._normalize_img_batch and calls Cellpose with normalize=False; the preprocess block here describes Cellpose's own normalize=True path",
               "source": "spacr.settings.set_default_settings_preprocess_generate_masks"}
    return {"web_app": {**web, "magnification": 20}, "desktop": desktop}


def detector_defaults():
    try:
        from spacr.plaque import DEFAULT_CONFIDENCE
    except ImportError:
        DEFAULT_CONFIDENCE = 0.25
    return {"confidence": DEFAULT_CONFIDENCE, "imgsz": DETECTOR_IMGSZ, "min_axis_ratio": 0.7,
            "source": "spacr.plaque.detect_wells via tools/web_mask_generator/detector.py"}


def _source(row):
    return {"repo_id": row["repo_id"], "repo_type": row["repo_type"], "revision": row["revision"],
            "filename": row["filename"], "sha256": row["sha256"], "url": source_url(row)}


def cellpose_entry(row, *, file, size, digest, check):
    return {
        "family": row["family"], "kind": row["kind"], "file": file, "bytes": size, "sha256": digest,
        "source": _source(row), "opset": OPSET,
        "input": {"name": "image", "dtype": "float32", "shape": ["N", 3, BSIZE, BSIZE], "layout": "NCHW",
                  "channels": "first three image channels; missing channels are zero (Cellpose slices the patch-embedding weights to the given channel count, which equals zero-filling)"},
        "output": {"name": "flows", "dtype": "float32", "shape": ["N", 3, BSIZE, BSIZE], "layout": "NCHW",
                   "channels": ["dY", "dX", "cellprob"], "cellprob": "logits (no sigmoid)",
                   "consumed_as": "cellpose.core.run_net: tiles averaged, then dP = y[:2], cellprob = y[2]"},
        "preprocess": {
            "normalize": {"method": "cellpose.transforms.normalize99 per channel", "percentiles": [1.0, 99.0],
                          "formula": "(x - p1) / (p99 - p1) as float32",
                          "flat_channel": "channels with zero range, or p99 - p1 <= 1e-3, become all zeros",
                          "downsample": "percentiles use a strided subsample only when a channel has more than 224**3 pixels"},
            "diam_mean": 30.0, "rescale": "30 / diameter when diameter > 0, else 1",
            "resize": "cv2.INTER_LINEAR to (int(Ly * rescale), int(Lx * rescale)) when rescale != 1",
            "pad": "cellpose.transforms.get_pad_yx(div=16, extra=1, min_size=(256, 256)), zero padding, split as evenly as possible",
            "bsize": BSIZE, "tile_overlap": 0.1, "batch_size": 8,
            "tiles": "ny = 1 if Ly <= 256 else ceil(1.2 * Ly / 256); starts = linspace(0, Ly - 256, ny).astype(int); same for x",
            "blend": "cellpose.transforms.average_tiles with _taper_mask(sig=7.5): weight 1 / (1 + exp((|i - 127.5| - 108) / 7.5)) per axis, outer product",
            "resample": "crop padding, then resize the flows back to the original Ly x Lx with cv2.INTER_LINEAR (resample=True)",
            "augment": False},
        "postprocess": {
            "function": "cellpose.dynamics.resize_and_compute_masks", "niter": "int(200 / rescale) when resample is True, else 200",
            "flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": 15, "max_size_fraction": 0.4,
            "dynamics": "follow_flows on dP * (cellprob > cellprob_threshold) / 5 for niter steps; get_masks_torch; remove_bad_flow_masks(threshold=flow_threshold); fill_holes_and_remove_small_masks(min_size)",
            "spacr_defaults": spacr_defaults(row["family"])},
        "check": check,
    }


def detector_entry(row, *, file, size, digest, check, info, dynamic=True):
    rows = 4 + info["num_classes"]
    if dynamic:
        shape, output_shape = ["N", 3, "H", "W"], ["N", rows, "H*W/64 + H*W/256 + H*W/1024"]
        padding = "rect: dw = (640 - round(W * r)) % 32, dh = (640 - round(H * r)) % 32, so the padded image is only rounded up to a multiple of 32 (ultralytics predict with PyTorch weights, rect=True, auto=True)"
    else:
        shape = [1, 3, DETECTOR_IMGSZ, DETECTOR_IMGSZ]
        output_shape = [1, rows, sum((DETECTOR_IMGSZ // stride) ** 2 for stride in (8, 16, 32))]
        padding = "square: dw = 640 - round(W * r), dh = 640 - round(H * r); this differs from desktop rect letterboxing and can change detections"
    return {
        "family": row["family"], "kind": row["kind"], "file": file, "bytes": size, "sha256": digest,
        "source": _source(row), "opset": OPSET,
        "input": {"name": "images", "dtype": "float32", "shape": shape, "layout": "NCHW", "color": "RGB", "range": [0.0, 1.0]},
        "output": {"name": "output0", "dtype": "float32", "shape": output_shape,
                   "layout": "rows 0-3 are box cx, cy, w, h in letterboxed input pixels; rows 4.. are per-class scores with sigmoid applied; anchors ordered stride 8, 16, 32",
                   "nms_included": False,
                   "head_note": "one-to-many head, exactly what desktop predict runs (ultralytics nms=None keeps NMS outside the graph)",
                   **info},
        "preprocess": {
            "channel_order": "RGB into the network. spacr.plaque.detect_wells flips an RGB HxWx3 array to BGR and ultralytics flips it back, so the browser feeds RGB directly",
            "grayscale": "a single channel is repeated to three",
            "scale": "uint8 pixels / 255",
            "letterbox": {"size": DETECTOR_IMGSZ, "ratio": "r = min(640 / H, 640 / W) (scale up allowed)",
                          "resize": "cv2.INTER_LINEAR to (round(W * r), round(H * r)); skipped when the size is unchanged",
                          "padding": padding, "pad_value": 114, "center": True,
                          "pad_split": "dw /= 2, dh /= 2; top = round(dh - 0.1), bottom = round(dh + 0.1), left = round(dw - 0.1), right = round(dw + 0.1)",
                          "stride": info["stride"], "dynamic_graph": dynamic}},
        "postprocess": {
            "nms": {"conf": detector_defaults()["confidence"], "iou": 0.7, "max_det": info["max_det"], "agnostic": False,
                    "function": "ultralytics.utils.nms.non_max_suppression"},
            "unletterbox": "subtract the left/top padding, divide by r, clip to the image",
            "wells": "x0, y0, x1, y1 rounded to integers; drop boxes with width or height <= 0; drop boxes with min(w, h) / max(w, h) < min_axis_ratio; sort by (y0, x0)",
            "desktop_settings": detector_defaults()},
        "check": check,
    }


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        return {"schema": SCHEMA, "generated_by": GENERATED_BY, "base_url": None, "models": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or not isinstance(data.get("models"), dict):
        raise ExportError(f"refusing to merge into an unrecognised manifest: {path}")
    return data


def merge_manifest(path, entries):
    """Update the listed keys in place, keeping other keys and the maintainer's base_url."""
    data = read_manifest(path)
    data["schema"], data["generated_by"] = SCHEMA, GENERATED_BY
    data.setdefault("base_url", None)
    data["models"].update(entries)
    data["models"] = dict(sorted(data["models"].items()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    return data


def _publish_file(staged, out_dir, key):
    """Name the graph by content so browser caches never serve a stale export."""
    digest = sha256_file(staged)
    name = f"{key}-{digest[:12]}.onnx"
    target = Path(out_dir) / name
    Path(staged).replace(target)
    for older in Path(out_dir).glob(f"{key}-*.onnx"):
        if older.name != name:
            older.unlink()
    return name, target.stat().st_size, digest


def export_one(row, weights, out_dir, *, check=True, detector_dynamic=True, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / f".{row['key']}.onnx.partial"
    timings = {}
    started = time.monotonic()
    try:
        if row["kind"] == "cellpose":
            net = load_cellpose_net(weights)
            timings["load_s"] = round(time.monotonic() - started, 1)
            export_cellpose(net, staged)
            timings["export_s"] = round(time.monotonic() - started - timings["load_s"], 1)
            result = check_cellpose(net, staged) if check else None
            info = None
        elif row["kind"] == "detector":
            info = export_detector(weights, staged, dynamic=detector_dynamic)
            timings["export_s"] = round(time.monotonic() - started, 1)
            result = check_detector(weights, staged) if check else None
        else:
            raise ExportError(f"{row['key']}: unsupported model kind {row['kind']!r}")
        timings["total_s"] = round(time.monotonic() - started, 1)
        if result is not None and not result["passed"]:
            raise ExportError(f"{row['key']}: torch and ONNX outputs differ beyond tolerance: {json.dumps(result['outputs'])}")
        name, size, digest = _publish_file(staged, out_dir, row["key"])
    finally:
        staged.unlink(missing_ok=True)
    log(json.dumps({"key": row["key"], "file": name, "bytes": size, "sha256": digest, "timings": timings,
                    "check_passed": None if result is None else result["passed"]}, sort_keys=True))
    if info is None:
        return cellpose_entry(row, file=name, size=size, digest=digest, check=result)
    return detector_entry(row, file=name, size=size, digest=digest, check=result, info=info, dynamic=detector_dynamic)


def _iou_summary(first, second):
    import numpy as np
    labels = [value for value in np.unique(first) if value]
    scores = []
    for value in labels:
        region = first == value
        overlap = second[region]
        overlap = overlap[overlap > 0]
        if not overlap.size:
            scores.append(0.0)
            continue
        match = np.bincount(overlap).argmax()
        other = second == match
        scores.append(float((region & other).sum() / (region | other).sum()))
    return {"labels_first": len(labels), "labels_second": int(len(np.unique(second)) - (0 in second)),
            "identical": bool(np.array_equal(first, second)),
            "min_iou": min(scores) if scores else None, "mean_iou": float(np.mean(scores)) if scores else None}


def reference_fixture(weights, onnx_path, image_path, out_dir, *, diameter=30.0, flow_threshold=0.4, cellprob_threshold=0.0):
    """Save an end-to-end Cellpose reference (torch and ORT) for the browser dynamics port."""
    import numpy as np
    import tifffile
    import torch
    from cellpose import core, dynamics, models, transforms

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(tifffile.imread(str(image_path)))
    if raw.ndim != 2:
        raise ExportError("the reference fixture expects one 2-D plane")
    model = models.CellposeModel(gpu=False, pretrained_model=str(weights), use_bfloat16=False, device=torch.device("cpu"))
    masks, flows, _ = model.eval(raw, diameter=diameter, flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold)
    rescale = 30.0 / diameter
    image = transforms.convert_image(raw, channel_axis=None, z_axis=None)[np.newaxis]
    params = {**models.normalize_default, "normalize": True}
    normalized = transforms.normalize_img(image.copy(), **params)
    if rescale != 1.0:
        normalized = transforms.resize_image(normalized, rsz=rescale)
    net_input = np.zeros((*normalized.shape[:-1], 3), np.float32)
    net_input[..., :normalized.shape[-1]] = normalized
    session = _ort_session(onnx_path)

    class OrtNet:
        device, dtype = torch.device("cpu"), torch.float32

        def eval(self):
            return self

        def __call__(self, tensor):
            flows_out = session.run(["flows"], {"image": tensor.numpy().astype(np.float32)})[0]
            return torch.from_numpy(flows_out), torch.zeros((tensor.shape[0], 256))

    torch_y, _ = core.run_net(model.net, net_input, bsize=BSIZE, tile_overlap=0.1, batch_size=8)
    ort_y, _ = core.run_net(OrtNet(), net_input, bsize=BSIZE, tile_overlap=0.1, batch_size=8)
    niter = int(200 / rescale)

    def masks_from(y):
        dP, prob = y[0, ..., :2].transpose(2, 0, 1), y[0, ..., 2]
        return dP, prob, dynamics.resize_and_compute_masks(dP, prob, niter=niter, cellprob_threshold=cellprob_threshold,
                                                          flow_threshold=flow_threshold, min_size=15, max_size_fraction=0.4)

    torch_dP, torch_prob, torch_masks = masks_from(torch_y)
    ort_dP, ort_prob, ort_masks = masks_from(ort_y)
    arrays = {"raw_image": raw, "normalized_input": net_input[0], "desktop_dP": np.asarray(flows[1], np.float32),
              "desktop_cellprob": np.asarray(flows[2], np.float32), "desktop_masks": masks,
              "torch_dP": torch_dP, "torch_cellprob": torch_prob, "torch_masks": torch_masks,
              "ort_dP": ort_dP, "ort_cellprob": ort_prob, "ort_masks": ort_masks}
    for name, value in arrays.items():
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(value))
    meta = {"image": str(image_path), "image_sha256": sha256_file(image_path), "weights_sha256": sha256_file(weights),
            "onnx_sha256": sha256_file(onnx_path),
            "settings": {"diameter": diameter, "rescale": rescale, "flow_threshold": flow_threshold,
                         "cellprob_threshold": cellprob_threshold, "min_size": 15, "max_size_fraction": 0.4,
                         "niter": niter, "bsize": BSIZE, "tile_overlap": 0.1, "normalize": "cellpose normalize_default, percentiles (1, 99)",
                         "model_dtype": "float32 (use_bfloat16=False), CPU"},
            "arrays": {name: {"file": f"{name}.npy", "shape": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
                       for name, value in arrays.items()},
            "layout": {"normalized_input": "H x W x 3 float32 fed to cellpose.core.run_net (channels 2-3 zero)",
                       "dP": "2 x H x W, [dY, dX]", "cellprob": "H x W logits", "masks": "H x W labels"},
            "comparisons": {"torch_vs_ort_dP": _diff(torch_dP, ort_dP), "torch_vs_ort_cellprob": _diff(torch_prob, ort_prob),
                            "desktop_vs_torch_dP": _diff(flows[1], torch_dP),
                            "desktop_vs_torch_cellprob": _diff(flows[2], torch_prob),
                            "desktop_vs_torch_masks": _iou_summary(masks, torch_masks),
                            "desktop_vs_ort_masks": _iou_summary(masks, ort_masks),
                            "torch_vs_ort_masks": _iou_summary(torch_masks, ort_masks)}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return meta


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m tools.web_mask_generator.export_onnx", description=__doc__.splitlines()[0])
    parser.add_argument("--key", action="append", default=[], help="model key from space/models.json (repeatable)")
    parser.add_argument("--all", action="store_true", help="export every advertised model")
    parser.add_argument("--weights", action="append", default=[], metavar="KEY=PATH", help="local weights for one key (repeatable)")
    parser.add_argument("--weights-dir", type=Path, help="directory searched by the basename of each models.json filename")
    parser.add_argument("--download", action="store_true", help="allow fetching the public, revision-pinned weights without a token")
    parser.add_argument("--download-dir", type=Path, help="where --download stores weights (default OUT/weights)")
    parser.add_argument("--out", type=Path, required=True, help="output directory for .onnx files")
    parser.add_argument("--manifest", type=Path, help="browser manifest to create or update (default OUT/browser-manifest.json)")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True, help="compare torch and onnxruntime outputs")
    parser.add_argument("--detector-static", action="store_true", help="export detectors at a fixed 640 x 640 (differs from desktop letterboxing)")
    parser.add_argument("--reference-fixture", type=Path, metavar="IMAGE", help="also save a Cellpose reference fixture for one 2-D image")
    parser.add_argument("--fixture-out", type=Path, help="fixture directory (default OUT/fixture)")
    parser.add_argument("--catalogue", type=Path, default=MODELS_JSON, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    options = build_parser().parse_args(argv)
    try:
        catalogue = load_catalogue(options.catalogue)
        keys = select_keys(catalogue, options.key, options.all)
        explicit = parse_weights(options.weights)
        stray = sorted(set(explicit) - set(catalogue))
        if stray:
            raise ExportError("--weights names unknown key(s): " + ", ".join(stray))
        out_dir = options.out
        manifest = options.manifest or out_dir / "browser-manifest.json"
        download_dir = options.download_dir or out_dir / "weights"
        resolved = {}
        for key in keys:
            path, how = resolve_weights(catalogue[key], explicit=explicit, weights_dir=options.weights_dir,
                                        download=options.download, download_dir=download_dir)
            verify_weights(path, catalogue[key])
            resolved[key] = path
            print(f"{key}: weights verified ({how}): {path}", file=sys.stderr)
        if options.reference_fixture is not None and not any(catalogue[key]["kind"] == "cellpose" for key in keys):
            raise ExportError("--reference-fixture needs a Cellpose key")
        for key in keys:
            entry = export_one(catalogue[key], resolved[key], out_dir, check=options.check,
                               detector_dynamic=not options.detector_static)
            merge_manifest(manifest, {key: entry})
            if options.reference_fixture is not None and catalogue[key]["kind"] == "cellpose":
                meta = reference_fixture(resolved[key], out_dir / entry["file"], options.reference_fixture,
                                         options.fixture_out or out_dir / "fixture")
                print(json.dumps({"key": key, "fixture": meta["comparisons"]}, sort_keys=True))
                options.reference_fixture = None
        print(f"manifest: {manifest}", file=sys.stderr)
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
