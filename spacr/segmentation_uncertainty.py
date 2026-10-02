"""Headless segmentation-uncertainty scoring and lossless map export.

Four test-time transforms of the primary model define reference objects.
An optional second model contributes four equally weighted label sets;
near-miss probabilities and flow diagnostics remain those of the primary
model, whose threshold and object identities the user selected. Scores rank
review effort; they are not calibrated probabilities of biological error.
"""
from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["compute_uncertainty", "save_uncertainty_map", "compute_queue_uncertainty"]


def compute_uncertainty(image, segment, *, second_segment=None, probability_threshold=0.0):
    """Score aligned TTA passes, optionally adding a second-model ensemble.

    :param image: one source field, unchanged by this function.
    :param segment: primary callable returning labels or labels/probability/flows.
    :param second_segment: optional second callable with the same contract.
    :param probability_threshold: the primary model's cell-probability threshold.
    :returns: the existing uncertainty score dictionary, including a float32 map.
    """
    from .active_learning import _segmentation_uncertainty, _tta_passes

    passes = _tta_passes(image, segment)
    labels = list(passes["labels"])
    if second_segment is not None:
        labels.extend(_tta_passes(image, second_segment)["labels"])
    return _segmentation_uncertainty(
        labels, probabilities=passes["probabilities"], vectors=passes["vectors"],
        probability_threshold=probability_threshold)


def save_uncertainty_map(path, result, *, provenance=None, protected_paths=()):
    """Atomically save a float32 TIFF map with embedded JSON provenance.

    :param path: destination TIFF; parent directories are created if necessary.
    :param result: uncertainty score dictionary containing a finite 2-D map.
    :param provenance: source/model/settings metadata to embed in the TIFF.
    :param protected_paths: scientific source images or masks never overwritten.
    :returns: the written Path; failed writes retain the previous destination.
    :raises ValueError: for invalid maps or a protected destination.
    """
    import tifffile
    from .tiff_io import write_tiff

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("An uncertainty map must be saved as .tif or .tiff.")
    for source in protected_paths:
        if source is not None and destination == Path(source).expanduser().resolve():
            raise ValueError("An uncertainty map cannot overwrite a source image or mask.")
    if destination.exists():
        try:
            with tifffile.TiffFile(destination) as existing:
                previous = json.loads(existing.pages[0].description)
        except (OSError, ValueError, KeyError, TypeError):
            previous = {}
        if not isinstance(previous, dict) or "spacr_uncertainty" not in previous:
            raise ValueError("Choose a new filename; this file is not an uncertainty map.")
    array = np.asarray(result["map"], dtype=np.float32)
    if (array.ndim != 2 or not array.size or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 1)):
        raise ValueError("An uncertainty map must be a finite 2-D array in [0, 1].")
    metadata = dict(provenance or {})
    metadata["scores"] = {key: value for key, value in result.items() if key != "map"}
    # Validate metadata before opening a file; int-keyed object scores are JSON-safe.
    metadata = json.loads(json.dumps(metadata, allow_nan=False, default=str))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".uncertainty-", suffix=".tif", dir=destination.parent)
    os.close(fd)
    try:
        write_tiff(temporary, array, photometric="minisblack", compression="zlib",
                   metadata={"axes": "YX", "spacr_uncertainty": metadata})
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _make_segmenter(model_name, device, parameters):
    """Load a repository-supported model without importing the Qt interface.

    :param model_name: stock model, local checkpoint or supported backend prefix.
    :param device: explicit torch device, normally cpu.
    :param parameters: Cellpose inference keyword arguments.
    :returns: a callable returning labels, cell-probability logits and flow vectors.
    """
    from . import _segmentation_backends as backends
    from .spacr_cellpose import cellpose_channel_axis, parse_cellpose4_output

    prefixed = (backends._cellpose3_choice(model_name) is not None
                or backends._cellpose_dino_choice(model_name) is not None
                or backends._prefixed_backend(model_name) is not None)
    if prefixed:
        backend = str(model_name).split(":", 1)[0]
        model = backends._load_backend(backend, model_name=model_name, device=device)
    else:
        import torch
        from cellpose.models import CellposeModel
        from .utils import _resolve_cellpose_pretrained

        target = torch.device(device)
        kwargs = dict(pretrained_model=_resolve_cellpose_pretrained(model_name),
                      device=target, gpu=target.type != "cpu")
        if "use_bfloat16" in inspect.signature(CellposeModel).parameters:
            kwargs["use_bfloat16"] = False
        model = CellposeModel(**kwargs)

    def segment(image):
        """Infer one transformed field through the shared output parser.

        :param image: 2-D field supplied by the TTA scorer.
        :returns: labels, logits and vector flows in the input orientation.
        """
        kwargs = dict(parameters, channel_axis=cellpose_channel_axis(image), batch_size=1)
        signature = inspect.signature(model.eval).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in signature}
        masks, _rgb, vectors, probabilities, _other = parse_cellpose4_output(model.eval([image], **kwargs))
        labels = np.asarray(masks[0] if np.asarray(masks).ndim > 2 else masks, dtype=np.int32)
        return (labels, probabilities[0] if probabilities else None,
                vectors[0] if vectors else None)

    return segment


def compute_queue_uncertainty(queue, *, model="cpsam", second_model=None, device="cpu",
                              map_folder=None, parameters=None, progress=None,
                              segmenter_factory=None):
    """Score a bounded curation queue headlessly and persist its ranking.

    :param queue: an existing CurationQueue; only its pending selected fields run.
    :param model: primary model name or checkpoint, default cpsam.
    :param second_model: optional distinct second model; absent means four passes.
    :param device: Device used for model predictions. The default is ``cpu``.
    :param map_folder: optional folder for lossless maps and embedded provenance.
    :param parameters: inference overrides for diameter, normalize and thresholds.
    :param progress: optional callback receiving one completed field's stem.
    :param segmenter_factory: injectable model/device/parameters loader for tests.
    :returns: Dictionary of score summaries keyed by image filename without
        its extension. Each summary excludes the pixel map.
    :raises ValueError: for duplicate ensemble models or a field that cannot run.
    """
    from .curation_queue import _write_uncertainty
    # Pure image I/O: mask_engine imports no PySide6 and creates no Qt objects.
    from .qt.mask_engine import load_image_and_mask

    if second_model and str(second_model) == str(model):
        raise ValueError("The ensemble model must differ from the primary model.")
    if not queue.items:
        return {}
    protected = [path for item in queue.items
                 for path in (item.image, item.mask, item.bundle) if path is not None]
    options = dict(diameter=None, normalize=True, flow_threshold=0.4,
                   cellprob_threshold=0.0, min_size=0)
    options.update(parameters or {})
    factory = segmenter_factory or _make_segmenter
    primary = factory(model, device, options)
    secondary = factory(second_model, device, options) if second_model else None
    scores = {}
    for item in queue.items:
        source = item.bundle if item.bundle is not None else item.image
        layout = {"masks_dir": str(item.mask.parent)} if item.mask is not None and item.bundle is None else {}
        image, _mask = load_image_and_mask(str(source.parent), source.name, **layout)
        result = compute_uncertainty(image, primary, second_segment=secondary,
                                     probability_threshold=options["cellprob_threshold"])
        provenance = dict(source=str(source.resolve()), primary_model=str(model),
                          second_model=second_model, device=device, parameters=options,
                          transforms=["identity", "flip_lr", "flip_ud", "rot90"])
        if map_folder is not None:
            save_uncertainty_map(Path(map_folder) / f"{item.stem}_uncertainty.tif", result,
                                 provenance=provenance, protected_paths=protected)
        summary = {key: value for key, value in result.items() if key != "map"}
        scores[item.stem] = summary
        # Each completed field is recoverable if a later field or model fails.
        _write_uncertainty(queue.folder, {item.stem: dict(
            uncertainty=result["field"], n_objects=result["n_objects"],
            passes=result["n_passes"], model=" + ".join(filter(None, (model, second_model))))})
        if progress is not None:
            progress(item.stem)
    return scores
