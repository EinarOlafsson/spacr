"""N698 single-channel model families through shared desktop Mask functions.

Requires the canonical N696 diameter backend. No alternative normalization,
segmentation or mask-filter algorithm is implemented here. Real scientific/GPU
parity must still be checked with each actual approved checkpoint and dataset.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re

import numpy as np

from spacr.plaque import _host_array
from spacr.qt.i18n import tr

from .inputs import read_upload_plane
from .plaque import _CapturedModel
from .results import _sha256, build_result_zip


OBJECT_ROLES = {
    "toxoplasma_pv": "pathogen", "live_cell": "cell",
    "nuclei_from_cellmask": "nucleus", "cell_from_hoechst": "cell",
    "toxoplasma_from_cellmask": "pathogen", "toxoplasma_from_hoechst": "pathogen",
}
ALLOWED_SETTINGS = {"diameter", "magnification", "flow_threshold", "CP_prob",
                    "background", "signal_to_noise", "remove_background",
                    "lower_percentile", "perimeter_fraction", "remove_border_objects", "object_filters"}


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(tr("Mask setting {name} must be a finite number.", name=name))
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(tr("Mask setting {name} is outside its valid range.", name=name))
    return value


def _settings(role, overrides, checkpoint):
    try:
        from spacr.object_settings_table import _diameter_setting, _resolved_diameter, _diameter_for_eval
    except ImportError as error:
        raise RuntimeError(tr("This model route requires the updated N696 diameter backend.")) from error
    from spacr.settings import set_default_settings_preprocess_generate_masks, _get_object_settings

    if not isinstance(overrides, dict) or set(overrides) - ALLOWED_SETTINGS:
        raise ValueError(tr("Unsupported Mask inference settings were supplied."))
    raw_diameter = _diameter_setting(overrides.get("diameter"))
    for name in ("remove_background", "remove_border_objects"):
        if name in overrides and not isinstance(overrides[name], bool):
            raise ValueError(tr("Mask setting {name} must be true or false.", name=name))
    if "object_filters" in overrides and not isinstance(overrides["object_filters"], list):
        raise ValueError(tr("Object filters must be a list for this model's target."))
    settings = set_default_settings_preprocess_generate_masks({"verbose": False})
    for target in ("cell", "nucleus", "pathogen"):
        settings[f"{target}_channel"] = 0 if target == role else None
    settings["channels"] = [0]
    settings[f"{role}_model_name"] = str(checkpoint)
    settings[f"{role}_diameter"] = raw_diameter
    magnification = _number(overrides.get("magnification", settings["magnification"]), "magnification", minimum=0.000001)
    settings["magnification"] = magnification
    mapping = {"flow_threshold": f"{role}_flow_threshold", "CP_prob": f"{role}_cellprob_threshold",
               "background": f"{role}_background", "signal_to_noise": f"{role}_signal_to_noise",
               "perimeter_fraction": f"{role}_perimeter_fraction"}
    for name, target in mapping.items():
        if name in overrides:
            settings[target] = _number(overrides[name], name, minimum=None if name == "CP_prob" else 0,
                                       maximum=1 if name == "perimeter_fraction" else None)
    if "lower_percentile" in overrides:
        settings["lower_percentile"] = _number(overrides["lower_percentile"], "lower_percentile", minimum=0, maximum=100)
    if "remove_background" in overrides:
        settings[f"remove_background_{role}"] = overrides["remove_background"]
    if "remove_border_objects" in overrides:
        settings[f"{role}_remove_border_objects"] = overrides["remove_border_objects"]
    if "object_filters" in overrides:
        settings["object_filters"] = {role: overrides["object_filters"]}
    object_settings = _get_object_settings(role, settings)
    resolved = _resolved_diameter(raw_diameter, magnification, role)
    diameter = _diameter_for_eval(resolved, estimates=False)
    point = {"diameter": diameter, "flow_threshold": settings[f"{role}_flow_threshold"],
             "cellprob_threshold": settings[f"{role}_cellprob_threshold"], "enhance": False}
    return settings, object_settings, point, resolved


@lru_cache(maxsize=2)
def _cached_mask_model(checkpoint, digest, role, context):
    """Construct one role model for the already resolved web execution context."""
    from spacr.object import _robustness_segmenter
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks({"verbose": False, f"{role}_model_name": str(checkpoint), f"{role}_channel": 0})
    return _robustness_segmenter(settings, role).args[0]


def _load_mask_model(checkpoint, digest, role):
    """Refresh context before reusing the same trusted checkpoint and object role."""
    from spacr import accelerator
    resolved = accelerator.resolve(refresh=True)
    context = (resolved.kind, resolved.device, resolved.is_gpu,
               resolved.bfloat16 if resolved.is_gpu else False)
    return _cached_mask_model(checkpoint, digest, role, context)


_load_mask_model.cache_clear = _cached_mask_model.cache_clear
_load_mask_model.cache_info = _cached_mask_model.cache_info
_load_mask_model.cache_parameters = _cached_mask_model.cache_parameters


def run_mask_download(destination, *, family, input_file, checkpoint, expected_sha256,
                      model_id, model_version, effective_settings, selection=None):
    """Use the existing desktop normalization/evaluation/filtering for one plane.

    A single explicit input channel is intentional: it prevents inventing a
    second nuclear/reference input for cross-channel model classes. Original
    intensity pixels remain available for measurement and filtering.
    """
    if family not in OBJECT_ROLES or not str(model_id).startswith(family + "_v"):
        raise ValueError(tr("This checkpoint does not belong to a supported Mask model class."))
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError(tr("Provide a trusted Mask checkpoint and its published SHA-256."))
    digest = _sha256(checkpoint)
    if digest != expected_sha256.lower():
        raise ValueError(tr("The Mask checkpoint does not match the published SHA-256."))
    role = OBJECT_ROLES[family]
    settings, object_settings, point, resolved = _settings(role, effective_settings, checkpoint)
    plane = read_upload_plane(input_file, **dict(selection or {}))
    raw = plane["image"]
    if plane["image_axes"] == "CYX":
        if raw.shape[0] != 1:
            raise ValueError(tr("Choose exactly one input channel for this model class."))
        raw = raw[0]
    from spacr.io import _normalize_img_batch
    from spacr.object import _robustness_segment, merge_split_filter_masks, _mask_at_field_size
    from spacr.spacr_cellpose import parse_cellpose4_output
    from spacr.utils import prepare_batch_for_segmentation

    batch = raw[None, ..., None].copy()
    normalized = _normalize_img_batch(batch, [0], np.float32, settings)
    prepared = prepare_batch_for_segmentation(normalized)[0]
    model = _CapturedModel(_load_mask_model(checkpoint, digest, role))
    labels = _robustness_segment(model, object_settings, role, prepared, point)
    labels = merge_split_filter_masks([labels], [raw], settings, role)[0]
    labels = _mask_at_field_size(labels, raw.shape, role)
    parsed = parse_cellpose4_output(model.output)
    flow = None if parsed[2][0] is None else _host_array(parsed[2][0])
    score = None if parsed[3][0] is None else _host_array(parsed[3][0])
    unavailable = {}
    model_scale = tr("computed at the model scale (resample off), not the image size")
    if flow is None:
        unavailable["flows"] = tr("The desktop Mask backend returned no vector field.")
    elif np.shape(flow) != (2, *np.shape(labels)):
        flow = None
        unavailable["flows"] = model_scale
    if score is None:
        unavailable["cell_probability"] = tr("The desktop Mask backend returned no score map.")
    elif np.shape(score) != np.shape(labels):
        score = None
        unavailable["cell_probability"] = model_scale
    provenance = {"input_selection": plane["selection"], "family": family, "object_role": role,
                  "diameter_requested": settings[f"{role}_diameter"], "diameter_resolved": resolved,
                  "diameter_used": point["diameter"], "magnification": settings["magnification"],
                  "normalization": {"lower_percentile": settings["lower_percentile"], "background": settings[f"{role}_background"],
                                    "signal_to_noise": settings[f"{role}_signal_to_noise"], "remove_background": settings[f"remove_background_{role}"]},
                  "evaluation": {"normalize": False, "min_size": object_settings["min_size"], "resample": object_settings["resample"],
                                 "flow_threshold": point["flow_threshold"], "cellprob_threshold": point["cellprob_threshold"]},
                  "postprocessing": {"perimeter_fraction": settings.get(f"{role}_perimeter_fraction", 0),
                                     "remove_border_objects": settings.get(f"{role}_remove_border_objects", False),
                                     "object_filters": settings.get("object_filters", {}).get(role, [])},
                  "pipeline": ["spacr.io._normalize_img_batch", "spacr.object._robustness_segment", "spacr.object.merge_split_filter_masks", "spacr.object._mask_at_field_size"]}
    return build_result_zip(destination, input_file=input_file, image=plane["image"], image_axes=plane["image_axes"], masks=labels,
                            model_id=model_id, model_version=model_version, weights_sha256=digest, settings=provenance,
                            flows=flow, cell_probability=score, probability_kind="logits" if score is not None else None,
                            unavailable=unavailable)
