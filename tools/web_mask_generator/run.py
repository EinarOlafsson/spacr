"""Operator CLI for the N698 plaque upload/inference/download backend.

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
    parser = argparse.ArgumentParser(description=tr("Run the experimental plaque download backend with operator-provided local weights."))
    parser.add_argument("input", type=Path, help=tr("Local uploaded image or PDF."))
    parser.add_argument("--describe", action="store_true", help=tr("Print image/page choices without decoding pixels or loading a model."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--model-id")
    parser.add_argument("--model-version")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--diameter", type=float, help=tr("Resolved diameter in pixels; resolve magnification/Auto before this call."))
    scale.add_argument("--native-scale", action="store_true", help=tr("Explicitly use the desktop native-scale behavior."))
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
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
    if arguments.diameter is None and not arguments.native_scale:
        parser.error(tr("Choose --diameter or explicitly choose --native-scale."))
    indices = dict(arguments.index)
    if len(indices) != len(arguments.index):
        parser.error(tr("Select each image axis only once."))
    output = run_plaque_download(
        arguments.output.expanduser(), input_file=input_file,
        checkpoint=arguments.checkpoint.expanduser(), expected_sha256=arguments.sha256,
        model_id=arguments.model_id, model_version=arguments.model_version,
        effective_settings={"diameter": None if arguments.native_scale else arguments.diameter,
                            "flow_threshold": arguments.flow_threshold, "CP_prob": arguments.cellprob_threshold},
        selection={"series": arguments.series, "indices": indices, "channels": arguments.channel,
                   "page": arguments.page, "pdf_dpi": arguments.pdf_dpi})
    print(json.dumps({"result_zip": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
