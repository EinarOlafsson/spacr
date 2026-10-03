#!/usr/bin/env python
"""Train and score the real / not-real object classifiers (item 470).

The maintainer annotates ground-truth crops in Annotate with a ``real``
column (1 = real, 2 = not real); see
tools/build_ground_truth_annotation_sample.py. This tool turns those calls
into one classifier per object type and scores it on wells held out of
training, so the score never sees a well the classifier learnt from.

    train  --type pathogen --db <sample>/measurements/measurements.db
           --out <models>/pathogen.joblib
           Trains on every annotated crop, scores on held-out wells, and
           writes the classifier plus <out>.scorecard.json.
    score  --model <models>/pathogen.joblib --db <new round>.db
           Scores a saved classifier on annotated crops it was not trained
           on (a later annotation round), printing and optionally writing
           the scorecard.

A folder holding cell.joblib, nucleus.joblib and pathogen.joblib is what
Mask generation's alpha ``real_object_classifier`` setting takes; it then
erases the objects a classifier calls not real after detection. CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacr.object_classifier import (  # noqa: E402
    _REAL_TYPES, _RealObjectHead, _annotated_real_crops, _read_real_crop,
    _real_scorecard, _train_real_classifier)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train(args) -> dict:
    import joblib

    frame = _annotated_real_crops(args.db, column=args.column)
    bundle = _train_real_classifier(frame, args.type,
                                    test_fraction=args.test_fraction,
                                    seed=args.seed, threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    scorecard = dict(bundle["scorecard"], object_type=args.type,
                     n_annotated=int(len(frame)),
                     databases=[str(db) for db in args.db])
    _write_json(out.with_suffix(".scorecard.json"), scorecard)
    return scorecard


def score(args) -> dict:
    import joblib

    bundle = joblib.load(args.model)
    frame = _annotated_real_crops(args.db, column=args.column)
    head = _RealObjectHead(bundle, args.threshold)
    crops = [_read_real_crop(path) for path in frame["png_path"]]
    probability = head.real_probability(crops) if crops else []
    scorecard = dict(_real_scorecard(frame["real"].to_numpy(), probability,
                                     args.threshold),
                     object_type=bundle["object_type"], model=str(args.model),
                     databases=[str(db) for db in args.db])
    if args.out:
        _write_json(Path(args.out), scorecard)
    return scorecard


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "score"):
        command = commands.add_parser(name)
        command.add_argument("--db", nargs="+", required=True,
                             help="measurements.db files with annotated png_list")
        command.add_argument("--column", default="real")
        command.add_argument("--threshold", type=float, default=0.5)
        if name == "train":
            command.add_argument("--type", required=True, choices=_REAL_TYPES)
            command.add_argument("--out", required=True)
            command.add_argument("--test-fraction", type=float, default=0.2)
            command.add_argument("--seed", type=int, default=0)
        else:
            command.add_argument("--model", required=True)
            command.add_argument("--out")
    args = parser.parse_args(argv)
    scorecard = train(args) if args.command == "train" else score(args)
    print(json.dumps(scorecard, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
