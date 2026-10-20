"""N698 plaque inference adapter using the desktop loader and segmenter.

Only trusted, operator-provisioned local checkpoints are accepted. This is not
an HTTP endpoint and never downloads weights. Hosting must select a checkpoint
from an approved registry rather than exposing arbitrary paths to visitors.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

from spacr.plaque import _host_array, segment_plaque_image
from spacr.qt.i18n import tr

from .inputs import read_upload_plane
from .results import _sha256, build_result_zip


@lru_cache(maxsize=2)
def _cached_model(path, weights_sha256, context):
    """Construct a desktop model for the already resolved web execution context."""
    from spacr.submodules import _plaque_cellpose_model
    return _plaque_cellpose_model(str(path))


def _load_model(path, weights_sha256=None):
    """Refresh this web process's context before selecting cached checkpoint weights."""
    from spacr import accelerator
    resolved = accelerator.resolve(refresh=True)
    context = (resolved.kind, resolved.device, resolved.is_gpu,
               resolved.bfloat16 if resolved.is_gpu else False)
    return _cached_model(path, weights_sha256, context)


_load_model.cache_clear = _cached_model.cache_clear
_load_model.cache_info = _cached_model.cache_info
_load_model.cache_parameters = _cached_model.cache_parameters


class _CapturedModel:
    def __init__(self, model):
        self.model = model
        self.output = None

    def eval(self, *args, **kwargs):
        self.output = self.model.eval(*args, **kwargs)
        return self.output


def run_plaque_download(
    destination, *, input_file, checkpoint, expected_sha256,
    model_id, model_version, effective_settings, selection=None,
):
    """Read one plane, use the actual desktop plaque path, and package results.

    The caller must resolve the desired diameter before entering this adapter;
    ``None`` deliberately selects the desktop's native-scale behavior. No model
    default, magnification rule, normalization or postprocessing is duplicated
    here. CPU contract tests do not establish real checkpoint/GPU parity.
    """
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError(tr("Provide a trusted local checkpoint and its published SHA-256."))
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(tr("The checkpoint does not match the expected SHA-256."))
    if not isinstance(effective_settings, dict) or "diameter" not in effective_settings:
        raise ValueError(tr("Resolve the effective plaque diameter before inference."))
    if set(effective_settings) - {"diameter", "flow_threshold", "CP_prob"}:
        raise ValueError(tr("Unsupported plaque inference settings were supplied."))
    settings = {"flow_threshold": 0.4, "CP_prob": 0.0, **effective_settings}
    for name, value in settings.items():
        if name == "diameter" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(tr("Plaque setting {name} must be a finite number.", name=name))
        if (name == "diameter" and value <= 0) or (name == "flow_threshold" and value < 0):
            raise ValueError(tr("Plaque setting {name} is outside its valid range.", name=name))
    plane = read_upload_plane(input_file, **dict(selection or {}))
    image = plane["image"]
    if plane["image_axes"] == "CYX":
        if image.shape[0] > 3:
            raise ValueError(tr("Choose at most three channels for the plaque model."))
        image = np.moveaxis(image, 0, -1)
    model = _CapturedModel(_load_model(checkpoint, actual_sha256))
    masks = segment_plaque_image(model, image, settings)
    flow_list = model.output[1] if isinstance(model.output, (tuple, list)) and len(model.output) > 1 else None
    flows = cell_probability = None
    if isinstance(flow_list, (tuple, list)):
        if len(flow_list) > 1 and flow_list[1] is not None:
            flows = _host_array(flow_list[1])
        if len(flow_list) > 2 and flow_list[2] is not None:
            cell_probability = _host_array(flow_list[2])
    unavailable = {}
    if flows is None:
        unavailable["flows"] = tr("The desktop backend returned no vector field.")
    if cell_probability is None:
        unavailable["cell_probability"] = tr("The desktop backend returned no score map.")
    provenance = {**settings, "input_selection": plane["selection"],
                  "pipeline": "spacr.plaque.segment_plaque_image"}
    return build_result_zip(
        destination, input_file=input_file, image=plane["image"], image_axes=plane["image_axes"],
        masks=masks, model_id=model_id, model_version=model_version,
        weights_sha256=actual_sha256, settings=provenance, flows=flows,
        cell_probability=cell_probability,
        probability_kind="logits" if cell_probability is not None else None,
        unavailable=unavailable)
