"""Operator CLI for the N698 model upload/inference/download backend.

Run ``python -m tools.web_mask_generator.run --help`` from the checkout.
This starts no server and never downloads models. Real inference is performed
only when explicitly invoked with a trusted local checkpoint and expected hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacr.qt.i18n import tr

from .inputs import describe_upload
from .plaque import run_plaque_download
from .detector import run_well_download
from .masks import OBJECT_ROLES, run_mask_download


def _axis_index(value):
    try:
        axis, number = value.split("=", 1)
        if len(axis) != 1 or not axis.isalpha() or not axis.isupper():
            raise ValueError
        number = int(number)
        if number < 0:
            raise ValueError
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(tr("Use an uppercase axis and nonnegative index, such as T=0.")) from None
    return axis, number


def main(argv=None):
    parser = argparse.ArgumentParser(description=tr("Run the experimental model download backend with operator-provided local weights."))
    parser.add_argument("input", type=Path, help=tr("Local uploaded image or PDF."))
    parser.add_argument("--describe", action="store_true", help=tr("Print image/page choices without decoding pixels or loading a model."))
    parser.add_argument("--family", choices=["toxoplasma_plaque", "toxoplasma_well_detector", *OBJECT_ROLES], default="toxoplasma_plaque")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--model-id")
    parser.add_argument("--model-version")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--diameter", type=float, help=tr("Resolved diameter in pixels; resolve magnification/Auto before this call."))
    scale.add_argument("--native-scale", action="store_true", help=tr("Explicitly use native scale for plaques or Auto diameter for Mask families."))
    parser.add_argument("--magnification", type=float, default=20)
    parser.add_argument("--settings-json", default="{}", help=tr("Additional desktop normalization and object filters as JSON."))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-axis-ratio", type=float, default=0.7)
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=None)
    parser.add_argument("--series", type=int, default=0)
    parser.add_argument("--index", type=_axis_index, action="append", default=[], metavar="AXIS=N")
    parser.add_argument("--channel", type=int, action="append")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--pdf-dpi", type=int, default=150)
    arguments = parser.parse_args(argv)
    input_file = arguments.input.expanduser()
    if arguments.describe:
        print(json.dumps(describe_upload(input_file), sort_keys=True))
        return 0
    required = [name for name in ("output", "checkpoint", "sha256", "model_id", "model_version") if getattr(arguments, name) is None]
    if required:
        parser.error(tr("Inference requires: {arguments}", arguments=", ".join("--" + name.replace("_", "-") for name in required)))
    if arguments.family == "toxoplasma_plaque" and arguments.diameter is None and not arguments.native_scale:
        parser.error(tr("Choose --diameter or explicitly choose --native-scale."))
    indices = dict(arguments.index)
    if len(indices) != len(arguments.index):
        parser.error(tr("Select each image axis only once."))
    if arguments.family not in OBJECT_ROLES and arguments.settings_json != "{}":
        parser.error(tr("Additional normalization and filter settings apply to Mask model classes only."))
    if arguments.family == "toxoplasma_well_detector":
        runner = run_well_download
        settings = {"confidence": arguments.confidence, "min_axis_ratio": arguments.min_axis_ratio, "imgsz": arguments.detector_image_size}
    else:
        runner = run_plaque_download if arguments.family == "toxoplasma_plaque" else run_mask_download
        threshold = arguments.cellprob_threshold
        if threshold is None:
            threshold = -1 if OBJECT_ROLES.get(arguments.family) == "pathogen" else 0
        settings = {"diameter": None if arguments.native_scale else arguments.diameter,
                    "flow_threshold": arguments.flow_threshold, "CP_prob": threshold}
        if arguments.family in OBJECT_ROLES:
            try:
                extra = json.loads(arguments.settings_json)
            except ValueError:
                parser.error(tr("Desktop settings must be valid JSON."))
            if not isinstance(extra, dict) or set(extra) - {"background", "signal_to_noise", "remove_background", "lower_percentile", "perimeter_fraction", "remove_border_objects", "object_filters"}:
                parser.error(tr("Unsupported desktop normalization or filter settings."))
            settings.update(extra, magnification=arguments.magnification,
                            diameter="auto" if arguments.native_scale else arguments.diameter)
    family_argument = {"family": arguments.family} if arguments.family in OBJECT_ROLES else {}
    output = runner(
        arguments.output.expanduser(), input_file=input_file,
        checkpoint=arguments.checkpoint.expanduser(), expected_sha256=arguments.sha256,
        model_id=arguments.model_id, model_version=arguments.model_version,
        effective_settings=settings, **family_argument,
        selection={"series": arguments.series, "indices": indices, "channels": arguments.channel,
                   "page": arguments.page, "pdf_dpi": arguments.pdf_dpi})
    print(json.dumps({"result_zip": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
