"""Browse, verify, fetch and benchmark spaCR segmentation and classification models.

Why this exists
---------------
spaCR can run a stock Cellpose model, a Cellpose model somebody on the team
fine-tuned last year, a classifier checkpoint from a run three folders up, or a
checkpoint downloaded from Hugging Face. Today the only way to find out which
of those exist on a machine is ``find / -name '*.pth'``, the only way to know
what one of them was trained on is to remember, and the only way to know
whether the copy on disk is the file the author published is to hope.

This module answers those three questions and nothing else:

* **what is here** — :func:`discover_local` walks folders and returns the
  checkpoints, classified as Cellpose or classifier, with whatever provenance
  is recoverable from the settings snapshots spaCR already writes;
* **is it the right bytes** — :func:`sha256_file` / :func:`verify` /
  :func:`fetch`, which downloads atomically, checksums what arrived, and
  refuses to install a mismatch;
* **what does it do on my data** — :func:`benchmark`, which is
  :mod:`spacr.model_compare`'s "three fields" harness pointed at one model
  instead of two.

Nothing here imports torch or cellpose at module import time. Browsing the zoo,
reading provenance, checksumming a file and rendering the table all work on a
machine with neither installed; only :func:`benchmark` (through
:func:`spacr.model_compare.segment_with_cellpose`) needs them, and only when it
is called. ``tests/test_model_zoo.py`` asserts that.

The four things that make this trustworthy rather than merely convenient
------------------------------------------------------------------------
**Every download is checksummed, and verification happens before use.**
    A checkpoint truncated by a dropped connection, or swapped for a different
    one at the same URL, still loads. It does not raise; it produces silently
    different masks, and the run that used it looks exactly like the run that
    did not. So :func:`fetch` hashes what arrived and compares it to the hash
    the catalogue published: a mismatch deletes the file and raises
    :class:`ChecksumMismatch` — the entry is never registered. The hash that
    was actually computed is stored on the returned entry, so "verified"
    means "these bytes", not "this filename".

    A catalogue entry with no published hash cannot be verified at all, and
    that is refused by default (``require_checksum=True``) rather than quietly
    treated as fine. Callers that knowingly accept an unverifiable source pass
    ``require_checksum=False``; the resulting entry carries
    ``verified=False`` and says so in :func:`format_zoo`.

**Every download is atomic, and never overwrites.**
    Bytes stream into a temporary file *in the destination directory* (so the
    final ``os.replace`` is a same-filesystem rename, which is atomic) and the
    rename happens only after the checksum passes. An interrupted or cancelled
    download therefore leaves nothing behind that looks like a model — the
    failure mode where half a checkpoint sits at the real filename, loads, and
    segments badly, cannot happen.

    The destination is versioned rather than overwritten
    (:func:`versioned_path`: ``foo.CP_model``, ``foo_v2.CP_model``, …). Two
    models with the same filename are a normal thing to have; losing the first
    one to the second is not.

**Provenance is recorded, and "unknown" is written out in full.**
    A Cellpose model fine-tuned on 60x confluent HeLa is not interchangeable
    with one trained on 20x sparse fibroblasts, and no amount of benchmark
    score makes it so. What a model was trained on is the single most useful
    thing the zoo can show, so :class:`ModelEntry` carries
    :attr:`~ModelEntry.trained_on` and :attr:`~ModelEntry.trained_by`, both
    recovered from the settings snapshots spaCR already writes beside its
    models (``<file>_settings.csv`` for a Cellpose model,
    ``<dst>/settings.csv`` for a classifier — read through
    :func:`spacr.train_compare.load_run`, which already knows where to look).

    Where it could not be recovered the field reads ``'unknown'``, never ``''``.
    A blank cell in a provenance table reads as "no constraints"; that is the
    opposite of what it means.

**Benchmarks are only comparable inside one field set.**
    A model's score on your three fields says nothing whatsoever about its
    score on somebody else's, and a table that sorts the two together invents a
    ranking out of two unrelated numbers. So every :class:`BenchmarkResult`
    records a :func:`fieldset_id` — a hash of the actual pixels, not the folder
    name — and :func:`rank` **raises** :class:`IncomparableBenchmarks` when
    handed results from more than one field set. :func:`rank_groups` and
    :func:`format_benchmarks` are the supported alternative: they group by
    field set, rank within each group, and label the groups.

And a fifth, smaller one: a model file that is missing, empty, or not a torch
checkpoint at all fails in :func:`inspect_checkpoint` with a message naming the
file, before anything tries to load it. The default failure — a ``KeyError`` on
a state-dict key from deep inside torch — names nothing the user chose.

What the benchmark can and cannot say
-------------------------------------
There is no ground truth here. :func:`benchmark` runs one model over N fields
and reports what came out: object counts, timings and the
:mod:`spacr.seg_qc` verdict per field (fused? shattered? empty? all on the
border?). That is a *quality-control* score, not an accuracy — it can tell you
a model collapsed on your data, it cannot tell you which of two plausible
segmentations is right. :data:`RANK_KEYS` therefore offers exactly two keys,
``'qc'`` and ``'seconds'``, and no key that would read as accuracy. To compare
two models against each other, use :func:`compare_entries`, which hands both to
:func:`spacr.model_compare.compare_models` — the A/B harness that is explicit
about neither side being the truth.

Cellpose 4 accepts and ignores ``model_type``, ``diam_mean``, ``nchan``,
``channels`` and ``rescale``; only ``diameter`` at ``eval`` still changes the
masks. :class:`BenchmarkResult` carries the resolved ``honoured`` and
``ignored`` parameter dicts straight from
:class:`spacr.model_compare.ModelConfig` so a benchmark cannot silently be a
benchmark of settings nothing read.

Example::

    from spacr import model_zoo as zoo

    entries = zoo.catalogue() + zoo.discover_local('/data/screen1')
    print(zoo.format_zoo(entries))

    entry = zoo.resolve('cpsam_plaque_r3', entries)
    result = zoo.benchmark(entry, source='/data/screen1/plate1/1', n_fields=3)
    print(zoo.format_benchmarks([result]))

See Also:
    :mod:`spacr.model_compare` — the A/B harness this module reuses for
    segmentation and for the two-model comparison.
    :mod:`spacr.train_compare` — run discovery and settings recovery, reused
    wholesale for the classifier half of the zoo.
    :func:`spacr.utils.download_models` — the legacy bulk pull of the bundled
    Hugging Face model pack, wrapped by :func:`download_bundled_models`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from dataclasses import field as _dc_field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ._segmentation_backends import _SPECS as _BACKEND_SPECS

LOG = logging.getLogger(__name__)

__all__ = [
    "BUNDLED_REMOTE_MODELS",
    "BenchmarkResult",
    "CATALOGUE_ENV_VAR",
    "RETIRED_MODEL_NAMES",
    "REMOTE_CATALOGUE_URI",
    "shared_catalogue",
    "shared_catalogue_is_stale",
    "publish_model",
    "CLASSIFIER_SUFFIXES",
    "CELLPOSE_SUFFIXES",
    "ChecksumMismatch",
    "ZOO_SOURCES",
    "DEFAULT_ZOO_SOURCES",
    "source_of",
    "group_by_source",
    "entries_from_sources",
    "DEFAULT_N_FIELDS",
    "DEFAULT_SCAN_DEPTH",
    "DownloadCancelled",
    "FieldBenchmark",
    "HF_MODELS_REPO",
    "IncomparableBenchmarks",
    "KINDS",
    "ModelEntry",
    "ModelUnreadable",
    "ModelZooError",
    "RANK_KEYS",
    "UNKNOWN",
    "benchmark",
    "catalogue",
    "classify_kind",
    "compare_entries",
    "default_local_roots",
    "discover_local",
    "download_bundled_models",
    "entry_from_file",
    "fetch",
    "fieldset_id",
    "format_benchmarks",
    "format_zoo",
    "group_by_fieldset",
    "hf_uri",
    "inspect_checkpoint",
    "install",
    "load_catalogue_file",
    "open_uri",
    "package_model_root",
    "rank",
    "rank_groups",
    "resolve",
    "sha256_file",
    "verify",
    "versioned_path",
]



#: What an unrecoverable provenance field says. Never ``''``: a blank cell in a
#: provenance table reads as "no constraints", which is the opposite of "we do
#: not know what this model was trained on".
UNKNOWN = "unknown"

#: The kinds of model spaCR runs.
#:
#: ``detector`` was added for the YOLO well detector, which is neither of the
#: first two: it does not segment and it does not classify a crop, it locates
#: regions so something else can. :class:`ModelEntry` VALIDATES against this
#: tuple and raises on anything else, so an entry naming a kind that is not
#: here fails at construction rather than being quietly filed as a Cellpose
#: model and handed to CellposeModel later.
#:
#: ``cellpose3`` is a model that runs through the Cellpose 3 backend -- its
#: cyto3, cyto2, cyto and nuclei models, and Cellpose-format checkpoints from
#: bioimage.io. It is a kind of its own because spaCR's Cellpose 4 loads such
#: a checkpoint without complaint and then segments nonsense with it.
KINDS = ("cellpose", "classifier", "detector", "encoder", "backend",
         "cellpose3")
#: "backend" is not a checkpoint: it is a segmentation PACKAGE the zoo
#: lists so a user learns it exists and can install it from inside spaCR.

#: ``encoder`` is 386's kind: a self-supervised backbone that produces an
#: EMBEDDING rather than a mask or a label. It belongs in the zoo for the
#: reason 386 gives -- "an embedding that ships without one is a black box
#: twice over" -- and it is a separate kind because nothing that consumes a
#: classifier can consume one. Its weights are resolved by `timm` from the
#: HuggingFace hub rather than shipped, so its entry describes a download
#: that already happened; see :func:`spacr.embeddings.encoder_entry`.

#: Filename endings that mark a Cellpose checkpoint. ``.CP_model`` is what
#: :func:`spacr.submodules.train_cellpose` names its output.
CELLPOSE_SUFFIXES = (".cp_model", ".cpmodel")

#: Filename endings that mark a torch checkpoint. A ``.pth`` inside a Cellpose
#: folder is still a Cellpose model — see :func:`classify_kind`.
CLASSIFIER_SUFFIXES = (".pth", ".pt")

#: Directory names that mean "the files in here are Cellpose checkpoints".
#: ``cellpose_model`` is :func:`spacr.submodules.train_cellpose`'s output
#: folder; ``cp`` is where :func:`spacr.utils.download_models` lands the
#: bundled pack; ``models`` is what ``cellpose.train.train_seg`` creates under
#: whatever ``save_path`` it is given.
CELLPOSE_DIR_NAMES = ("cellpose_model", "cp", "models")

#: How deep :func:`discover_local` walks below each root.
DEFAULT_SCAN_DEPTH = 6

#: Ceiling on how many files one :func:`discover_local` call will *examine*, so
#: pointing it at ``/`` is slow rather than fatal.
DEFAULT_SCAN_LIMIT = 20000

#: Fields a benchmark uses by default — the number a human actually looks at.
DEFAULT_N_FIELDS = 3

#: Seconds before a download gives up on the server.
DEFAULT_TIMEOUT = 30

#: Bytes per chunk while streaming a download.
DEFAULT_CHUNK = 1 << 16

#: The Hugging Face dataset repo :func:`spacr.utils.download_models` pulls the
#: bundled model pack from. Same repo, same URL scheme — this module adds the
#: checksum, the atomic write and the versioned destination that one lacks.
HF_MODELS_REPO = "einarolafsson/models"

#: Environment variable naming a JSON catalogue of remote models. See
#: :func:`load_catalogue_file` for the format.
CATALOGUE_ENV_VAR = "SPACR_MODEL_CATALOGUE"

#: The SHARED catalogue, fetched at runtime rather than shipped.
#:
#: WHY THIS IS REMOTE. Until this existed, a model reached other spaCR users
#: only by someone editing :data:`BUNDLED_REMOTE_MODELS` in this file and
#: cutting a release -- so contributing a model meant contributing to spaCR,
#: waiting for a version, and every user upgrading. That is a high price for a
#: row of metadata, and it is why the zoo had exactly one entry.
#:
#: A contributor now uploads to THEIR OWN Hugging Face account -- they keep
#: ownership, and nobody has to hand out write access to anyone else's -- and
#: adds one row here. See :func:`publish_model`, which does the upload and
#: prints the row.
REMOTE_CATALOGUE_URI = (
    "https://huggingface.co/datasets/einarolafsson/models/resolve/main/"
    "catalogue.json"
)

#: How long a fetched shared catalogue is reused before being re-fetched.
#: Long enough that opening a module repeatedly is not a repeated request,
#: short enough that a newly contributed model appears the same day.
CATALOGUE_CACHE_SECONDS = 3600

#: Remote entries spaCR knows about out of the box.
#:
#: EVERY ENTRY HERE CARRIES A REAL sha256, and that is now the rule rather
#: than an aspiration. The retired ``toxo_plaque_cyto`` entry published none,
#: so :func:`fetch` refused to install it -- correctly, since a truncated or
#: substituted checkpoint could not be told from the real one -- which meant
#: it appeared in the model zoo as a row whose Download button could never
#: succeed. An entry without a hash is not a conservative entry; it is one
#: nobody can install.
BUNDLED_REMOTE_MODELS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "toxoplasma_pv_v1",
        "name": "cpsam_v2_toxo_r2",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-pv-segmentation-cpsam",
        "repo_type": "model",
        "uri": None,
        "sha256":
            "182d8cf6b32c7b9ef2917c85870d188486e5e119f05e9c5c1f07652f6859f2d0",
        "metrics": {'n_train': '229', 'train_objects': 'not recorded', 'n_test': '11 wells', 'test_objects': 'not recorded', 'cv': 'no', 'f1': '0.8640', 'aji': '0.8090', 'dice': 'not recorded', 'stock_f1': '0.7130', 'stock_aji': '0.4260', 'stock_dice': 'not recorded', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': '100 / 100'},
        "display_name": "Toxoplasma PV v1",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        # ROUND 2, CORRECTED 2026-09-15 (item 370). Until then this row
        # quoted ROUND 1 -- 115 images, 104 train / 11 test, F1 0.867 --
        # while the sha256 above has always been round 2's checkpoint. Every
        # figure below is from round 2's own run: round2.log for the split and
        # the stock baseline, round2_vs_round1.csv for the scores.
        "dataset": "anti-Toxoplasma-biotin and DsRed PV lumen; 229 images "
                   "from 2 datasets, 104 round-1 and 125 newly curated",
        "versus_stock": "F1 0.864 against 0.713 for stock cpsam on 11 "
                        "held-out in-house wells, at IoU 0.5; literature "
                        "hold-out pending",
        "trained_on": (
            "Toxoplasma tachyzoite parasitophorous vacuoles stained with goat "
            "anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the "
            "PV lumen. Round 2: 229 training images (round 1's 104 plus 125 "
            "newly curated RH and ME49 fields), 100 epochs, base cpsam_v2"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "F1 0.864 at IoU 0.5 against 0.713 for stock cpsam on the 11 "
            "wells round 1 also held out (round 1 scored 0.867); AJI 0.809 "
            "against 0.426",
            "accuracy falls sharply above IoU 0.8 -- suited to counting and "
            "area rather than precise morphometry",
            "the held-out literature scorecard is pending a stock-seeded "
            "re-curation; on the current literature set, whose truth leans "
            "toward this model's lineage, it ties stock Cellpose-SAM on "
            "detection (F1 0.403 against 0.400)",
        ),
    },
    {
        "key": "toxoplasma_plaque_v1",
        "name": "cpsam_plaque_r3",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-plaque-segmentation-cpsam",
        "repo_type": "model",
        "uri": None,
        "sha256":
            "eeecd2d6cd5cbb4dddee71564d5f460d26bb07ac125e0b494b7502fea4292d5d",
        "metrics": {'n_train': '184 wells', 'train_objects': 'not recorded', 'n_test': 'per fold', 'test_objects': 'not recorded', 'cv': '3-fold', 'f1': '0.8560 in-domain / 0.8060 literature', 'aji': 'not recorded', 'dice': 'not recorded', 'stock_f1': 'not recorded', 'stock_aji': 'not recorded', 'stock_dice': 'not recorded', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': 'not recorded'},
        "display_name": "Toxoplasma Plaque v1",
        "architecture": "Cellpose-SAM (cpsam)",
        "dataset": "crystal violet plaque wells; 184 wells from 3 datasets, "
                   "95 in-house and 89 literature",
        "versus_stock": "F1 0.856 in-domain; 0.806 on literature "
                        "(3-fold cross-validated, SD 0.020)",
        "trained_on": (
            "Toxoplasma gondii plaque assays; round 3, evaluated in-domain "
            "(NAS) and against a literature generalisation set"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "F1 0.856 in-domain and 0.806 on the literature set (3-fold "
            "cross-validated, SD 0.020), against 0.718 for round 1",
            "round 3 trades precision (0.939 down to 0.858) for recall "
            "(0.631 up to 0.811) on the literature set, which is the right "
            "direction for a counting assay",
            "PREFER THIS ONE FOR MICROSCOPE-ONLY WORK. On the round-5 test "
            "split it scores 0.836 on PFA-fixed wells against round 5's 0.808, "
            "at precision 0.93 against 0.81. For mixed sources, or any phone-"
            "camera image, use toxoplasma_plaque_v2, which round 3 cannot "
            "handle at all (0.249 there)",
        ),
    },
    {
        "key": "toxoplasma_plaque_v2",
        "name": "cpsam_plaque_r5",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5",
        "repo_type": "model",
        # THE WEIGHT IS UNDER weights/ IN THIS REPO, unlike the older plaque and PV
        # repos which put it at the root, so the URL is given rather than built from
        # `name`: hf_uri(repo_id, "cpsam_plaque_r5") would 404 on a repo that has it
        # one directory down, and a 404 here reads as "the model is gone".
        "uri": "https://huggingface.co/einarolafsson/"
               "toxoplasma-plaque-segmentation-cpsam-r5/resolve/main/"
               "weights/cpsam_plaque_r5?download=true",
        "sha256":
            "0927023a745ac6a19bae0ec72c89b7b864a4ff8d047a41f3f1e9767e1a4d0600",
        "metrics": {'n_train': '332 fields, 4 domains', 'train_objects': '18532', 'n_test': '81 fields', 'test_objects': '4294', 'cv': 'no (grouped train/valid/test)', 'f1': '0.819 literature / 0.876 bigbean / 0.808 patrick / 0.415 malnio', 'aji': 'not recorded', 'dice': 'not recorded', 'stock_f1': 'not measured; scored against round 3 (0.820 literature) and round 4 (0.819)', 'stock_aji': 'not recorded', 'stock_dice': 'not recorded', 'train_loss': '0.1822', 'val_loss': '0.2069', 'best_epoch': '100 / 100'},
        "display_name": "Toxoplasma Plaque v2 (round 5)",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "488 curated fields across four domains -- 298 wells cropped "
                   "from published figures, 96 phone-camera wells, 67 PFA and 27 "
                   "methanol-fixed whole-well microscope scans; 27,582 plaques",
        "versus_stock": "not scored against stock; on 81 held-out fields it ties "
                        "round 3 on literature (0.819 vs 0.820) and beats it by "
                        "0.166 on phone-camera wells (0.415 vs 0.249)",
        "trained_on": (
            "Toxoplasma plaque assays stained with crystal violet, from three "
            "microscopes and from published figures. Round 5: 332 training fields "
            "grouped by figure and by plate so none straddles the split, 100 "
            "epochs, base cpsam_v2, empty wells kept as negatives"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "COMPLEMENTS toxoplasma_plaque_v1 rather than replacing it: prefer v1 "
            "(round 3) for microscope-only work, where it scores 0.836 against "
            "this model's 0.808 on PFA-fixed wells and is far more precise; "
            "prefer this one for mixed or unknown sources",
            "the only plaque model trained on phone-camera wells -- F1 0.415 "
            "against round 3's 0.249, though recall there is 0.296, so it still "
            "misses most plaques on phone images and is not yet a counting tool",
            "it did NOT clear the promotion bar of 0.02 literature F1 fixed before "
            "the run (it came in at -0.001), so round 3 remains production",
            "balanced precision/recall (0.81/0.83) where round 3 is lopsided "
            "(0.93/0.73): round 3's low recall systematically UNDERCOUNTS, which "
            "matters more than F1 for a counting assay",
            "hallucinates 2 objects across 6 blank-lawn wells where round 3 "
            "hallucinates 19",
            "first plaque model on cpsam_v2; rounds 1-4 used cpsam v1, so base and "
            "data changed together and the gap to round 3 is not attributable to "
            "the extra curation alone",
            "training data: https://huggingface.co/datasets/einarolafsson/"
            "toxoplasma-plaque-dataset",
        ),
    },
    {
        "key": "toxoplasma_well_detector_v1",
        "name": "yolo_welldetect_v3.pt",
        "kind": "detector",
        "repo_id": "einarolafsson/toxoplasma-plaque-well-detector-yolo11",
        "repo_type": "model",
        "uri": None,
        "sha256":
            "b826058754fb5d4df36c3a7283aac049015cbb044b5ef096c55d19f37172a50c",
        "metrics": {'n_train': '562 images', 'train_objects': 'not recorded', 'n_test': 'held-out split', 'test_objects': 'not recorded', 'cv': 'no', 'f1': 'mAP50 0.9930 on v3\'s own split; 0.8838 on the shared test set', 'aji': 'mAP50-95 0.8860 own split; 0.7630 shared', 'dice': 'P/R 0.9870 own split; P 0.8613 / R 0.9085 shared', 'stock_f1': 'not recorded', 'stock_aji': 'not recorded', 'stock_dice': 'not recorded', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': '150 / 150'},
        "display_name": "Toxoplasma Plaque Well Detector v1",
        "architecture": "YOLO11n",
        "dataset": "whole-plate and multi-well crystal violet images; 562 "
                   "images from 1 dataset, 190 of them with no well in them",
        "versus_stock": "mAP50 0.993 on its own held-out split; on the test "
                        "set shared with v2 it scores mAP50 0.8838, against "
                        "v2's 0.9457",
        "trained_on": (
            "whole-plate and multi-well Toxoplasma plaque-assay images; "
            "yolo11n base, 150 epochs, batch 16, imgsz 640"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "the 0.993 is measured on v3's OWN split, which is easier than "
            "the set v2 is measured on; on that shared set this model scores "
            "mAP50 0.8838 against v2's 0.9457, so v2 is the better detector",
            "kept because the published plaque corpus was measured with these "
            "weights, so results in the paper trace back to this row",
            "locates WELLS, not plaques; it is the front half of a two-stage "
            "pipeline with toxoplasma_plaque_v1, and the well it finds also "
            "gives the diameter that makes areas comparable across "
            "microscopes",
        ),
    },
    {
        "key": "toxoplasma_well_detector_v2",
        "name": "yolo_welldetect_v4.pt",
        "kind": "detector",
        "repo_id": "einarolafsson/toxoplasma-plaque-well-detector-yolo26",
        "repo_type": "model",
        "uri": "https://huggingface.co/einarolafsson/"
               "toxoplasma-plaque-well-detector-yolo26/resolve/main/"
               "weights/best.pt",
        "remote_name": "best.pt",
        "sha256":
            "f2a1e1110f09b2a1d5ef5545adaba7c57f1158669d0bfc50d8fabe9f86da30c7",
        "metrics": {'n_train': '1,070 images', 'train_objects': '2,455 boxes', 'n_test': '129 images, 84 of them with no well', 'test_objects': '297 boxes', 'cv': 'no', 'f1': 'mAP50 0.9457 (shared test set)', 'aji': 'mAP50-95 0.8341', 'dice': 'P 0.8912 / R 0.9440', 'stock_f1': 'v3 scores 0.8838 on this set', 'stock_aji': 'v3 scores 0.7630', 'stock_dice': 'v3 P 0.8613 / R 0.9085', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': '28'},
        "display_name": "Toxoplasma Plaque Well Detector v2",
        "architecture": "YOLO26n (ultralytics 8.4.155)",
        "dataset": "plate images and literature figures; 1,070 train / 254 "
                   "val / 129 test, split by PMC article so no paper is in two "
                   "sets; training data at einarolafsson/"
                   "toxoplasma-plaque-well-detector-dataset",
        "versus_stock": "mAP50 0.9457 and mAP50-95 0.8341 against v3's 0.8838 "
                        "and 0.7630 on the SAME test set; stock YOLO has no "
                        "plaque-well class, so v3 is the baseline",
        "trained_on": (
            "whole-plate and multi-well Toxoplasma plaque-assay images plus 939 "
            "newly reviewed PMC figures, accepted boxes and confirmed negatives "
            "alike; yolo26n base, best validation mAP50-95 at epoch 28"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "on the shared test set it beats v1 (the v3 weights) on every "
            "measure, and cuts false boxes on no-well figures from 152 to 49",
            "84 of the 129 test images contain no well at all, which is what "
            "the false-box count is measured on",
            "locates WELLS, not plaques; the front half of a two-stage pipeline "
            "with the plaque segmentation model",
            "the repository publishes this weight as weights/best.pt; spaCR "
            "saves it under the name above so two detectors cannot both land "
            "as best.pt",
        ),
    },
    {
        "key": "toxoplasma_from_cellmask_v1",
        "name": "toxoplasma_from_cellmask_pv",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-from-cellmask-cpsam",
        "repo_type": "model",
        "uri": None,
        "sha256":
            "481dfccc1a68cc594aafcb71088efc25b5f5c6a6240e52902c0089759b3149ab",
        "metrics": {'n_train': '2567', 'train_objects': 'not recorded', 'n_test': '463', 'test_objects': '6116', 'cv': 'no', 'f1': '0.6058', 'aji': '0.4939', 'dice': '0.6096', 'stock_f1': '0.0215', 'stock_aji': '0.0080', 'stock_dice': '0.0201', 'train_loss': '0.0075', 'val_loss': '0.0100', 'best_epoch': '100 / 100'},
        "display_name": "Toxoplasma from Cell Mask (cross-channel)",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "Toxoplasma PV masks predicted from the HOST CELL MASK channel "
                   "alone; 2567 training and 463 held-out fields, split by well, "
                   "hosts HFF/HeLa/THP1",
        "versus_stock": "F1 0.606 against 0.021 for stock cpsam_v2 on 463 "
                        "well-grouped held-out fields, at IoU 0.5",
        "trained_on": (
            "cross-channel: given the host cell image, predicts where the "
            "Toxoplasma parasitophorous vacuoles are, with no parasite stain. "
            "100 epochs, base cpsam_v2, AdamW lr 1e-5, targets are "
            "PV-regenerated masks"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "F1 0.606, AJI 0.494, Dice 0.610 at IoU 0.5 against stock cpsam_v2's "
            "0.021/0.008/0.020 -- stock cannot do this task at all",
            "per host: HeLa 0.711, HFF 0.557, THP1 0.465; THP1 is the weak case",
            "the held-out split selects the checkpoint, so it is validation data "
            "rather than an independent test set",
            "accuracy falls above IoU 0.8 -- suited to counting, occupancy and "
            "area rather than precise morphometry",
        ),
    },
    {
        "key": "toxoplasma_pv_v2",
        "name": "cpsam_v2_toxo_r5",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-pv-segmentation-cpsam-r5",
        "repo_type": "model",
        "uri": None,
        "sha256":
            "17c689e3b117745561e20a885c2a2a998ed360fa97cac8c0446316ae5905c10f",
        "metrics": {'n_train': '556', 'train_objects': 'not recorded', 'n_test': '619 pairs', 'test_objects': 'not recorded', 'cv': '5-fold', 'f1': '0.8170 ± 0.036', 'aji': '0.7144 ± 0.107', 'dice': '0.8024 ± 0.118', 'stock_f1': '0.7130', 'stock_aji': '0.4260', 'stock_dice': 'not recorded', 'train_loss': '0.0476', 'val_loss': 'not recorded', 'best_epoch': '100 / 100'},
        "display_name": "Toxoplasma PV v2 (round 5)",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "anti-Toxoplasma-biotin and DsRed PV lumen; 556 curated "
                   "images accumulated over five rounds",
        "versus_stock": "F1 0.817 +/- 0.036 by 5-fold cross-validation over 619 "
                        "pairs; ~0.86 against 0.713 for stock on the 11 in-house "
                        "held-out wells",
        "trained_on": (
            "Toxoplasma tachyzoite parasitophorous vacuoles stained with goat "
            "anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV "
            "lumen (RH and ME49). Round 5: 556 images, 100 epochs, base cpsam_v2"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "supersedes toxoplasma_pv_v1 (round 2, 229 images): more than twice "
            "the training data and cross-validated rather than single-split",
            "5-fold CV over 619 pairs: F1 0.817 (SD 0.036), AJI 0.714, Dice 0.802",
            "per-dataset variance is real -- F1 ranges ~0.74 to ~0.93 by screen",
            "accuracy falls above IoU 0.8 -- suited to counting and area rather "
            "than precise morphometry",
        ),
    },
    {
        "key": "toxoplasma_pv_v3",
        "name": "cpsam_v2_toxo_r6",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-pv-segmentation-cpsam-r6",
        "repo_type": "model",
        "uri": "https://huggingface.co/einarolafsson/"
               "toxoplasma-pv-segmentation-cpsam-r6/resolve/main/"
               "weights/cpsam_v2_toxo_r6",
        "sha256":
            "146ef269979b1d1ab45c11039b0ab164f68001adaa8f73f1f8f18be6fcfd060e",
        "metrics": {'n_train': '437 fields', 'train_objects': '15,550', 'n_test': '11 anchor wells', 'test_objects': '683', 'cv': '5-fold, grouped by source', 'f1': '0.8602', 'aji': '0.8026', 'dice': '0.9059', 'stock_f1': '0.7648', 'stock_aji': '0.5050', 'stock_dice': '0.6431', 'train_loss': 'not recorded', 'val_loss': '0.0864', 'best_epoch': '20 / 100'},
        "display_name": "Toxoplasma PV v3 (round 6)",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "the 556 curated PV fields of round 5, split 437 train / "
                   "108 validation / 11 test; training data at "
                   "einarolafsson/toxoplasma-pv-segmentation-dataset",
        "versus_stock": "F1 0.860 against stock cpsam_v2's 0.765 on the 11 "
                        "anchor wells at IoU 0.5; AJI 0.803 against 0.505",
        "trained_on": (
            "Toxoplasma tachyzoite parasitophorous vacuoles stained with goat "
            "anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV "
            "lumen (RH and ME49). Round 6 retrains round 5's data with cellpose "
            "4.2.1.1, 100 epochs, base cpsam_v2"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "NEWEST IS NOT BEST HERE: round 6 does not beat round 2 on the "
            "anchor wells -- 0.8602 against 0.8648 -- and the PV project still "
            "promotes round 5",
            "it is the first PV round whose checkpoint was chosen on a held-out "
            "validation set (108 fields) instead of on the test wells",
            "5-fold cross-validation, grouped by source: F1 0.8168 +/- 0.028, "
            "AJI 0.7516, Dice 0.8424",
            "the 11 anchor wells have been held out since round 1, so they are "
            "the only fields no PV round has ever trained on",
        ),
    },
    {
        "key": "live_cell_v1",
        "name": "live_cell_v1",
        "kind": "cellpose",
        "repo_id": "einarolafsson/live-cell-segmentation-cpsam",
        "repo_type": "model",
        "uri": "https://huggingface.co/einarolafsson/"
               "live-cell-segmentation-cpsam/resolve/main/"
               "weights/live_cell_v1",
        "sha256":
            "7ade69377093fe81830ddc7c52ba8618bef1fefe7d1243c01b9c1beed7fcb090",
        "metrics": {'n_train': '6,778 fields, 14 datasets', 'train_objects': 'not recorded', 'n_test': '2,199 fields', 'test_objects': 'not recorded', 'cv': 'no (acquisition-grouped train/valid/test)', 'f1': '0.694 all / 0.960 on datasets stock never saw', 'aji': 'not recorded', 'dice': 'not recorded', 'stock_f1': '0.738 all / 0.885 on datasets stock never saw', 'stock_aji': 'not recorded', 'stock_dice': 'not recorded', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': '37 (stopped by the maintainer)'},
        "display_name": "Live cell v1 (phase, brightfield, DIC)",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "11,007 transmitted-light fields from 14 public datasets, "
                   "split by acquisition 6,778 train / 2,030 validation / "
                   "2,199 test; training data at "
                   "einarolafsson/live-cell-segmentation-dataset",
        "versus_stock": "on the datasets stock cpsam_v2 never trained on, F1 "
                        "0.960 against 0.885 at IoU 0.5; over all 2,199 test "
                        "fields, 0.694 against 0.738, because stock trained "
                        "on LIVECell and YeaZ and wins on LIVECell",
        "trained_on": (
            "unstained cells in phase contrast, brightfield and DIC: "
            "LIVECell, DeepSea, YeaZ, yeast microstructures, five Cell "
            "Tracking Challenge sets, BBBC009, BBBC030, QPI and Revvity. "
            "Base cpsam_v2, cellpose 4.2.1.1, lr 1e-5, batch 4; stopped at "
            "epoch 37 of 100"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "TWO STOCK COMPARISONS, NOT ONE: stock cpsam_v2 trained on LIVECell "
            "and YeaZ, so its score there is partly memorisation. On the "
            "datasets it never saw (DeepSea, the CTC sets, BBBC009, BBBC030, "
            "QPI, Revvity, yeast microstructures) this model scores F1 0.960 "
            "against 0.885",
            "it does NOT replace stock on LIVECell-style Incucyte phase: "
            "0.671 against 0.724 there, and it missed its own pre-registered "
            "promotion bar",
            "by modality at IoU 0.5: brightfield 0.964 (stock 0.912), DIC "
            "+0.026 over stock, phase 0.689 (stock 0.735)",
            "F1 0.865 on a train sample, 0.696 on validation and 0.694 on "
            "test; no per-epoch loss was recorded",
        ),
    },
    {
        "key": "nuclei_from_cellmask_v1",
        "name": "nuclei_from_cellmask_best",
        "kind": "cellpose",
        "repo_id": "einarolafsson/cross-channel-nuclei-from-cellmask-cpsam",
        "repo_type": "model",
        "uri": "https://huggingface.co/einarolafsson/"
               "cross-channel-nuclei-from-cellmask-cpsam/resolve/main/"
               "weights/nuclei_from_cellmask_best",
        "sha256":
            "2675553a46e97a7bc4bd2bfe3e954954194fe71ca4e94261e752a02bf0b6eb47",
        "metrics": {'n_train': 'not recorded', 'train_objects': 'not recorded', 'n_test': '453', 'test_objects': 'not recorded', 'cv': 'no', 'f1': '0.8881', 'aji': '0.7916', 'dice': '0.8774', 'stock_f1': 'not recorded', 'stock_aji': 'not recorded', 'stock_dice': 'not recorded', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': 'not recorded'},
        "display_name": "Cross-channel nuclei-from-cellmask",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "nuclei predicted from the HOST CELL MASK channel alone; "
                   "453 well-grouped held-out fields, hosts HFF/HeLa/THP1",
        "versus_stock": "F1 0.888 against 0.201 for stock cpsam_v2 on "
                        "453 well-grouped held-out fields, at IoU 0.5",
        "trained_on": (
            "cross-channel: given the cell image, predicts where the nuclei "
            "are, with no nuclear stain -- which frees the DAPI/Hoechst "
            "channel for another marker. 100 epochs, base cpsam_v2"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "F1 0.888, AJI 0.792, Dice 0.877 at IoU 0.5 against stock "
            "cpsam_v2's 0.201/0.286/0.449",
            "per host: HFF 0.932, HeLa 0.860, THP1 0.861",
            "the held-out split selects the checkpoint, so it is validation "
            "data rather than an independent test set",
            "predicts nuclei from cell morphology -- expect degraded accuracy "
            "on unusual or highly confluent morphologies",
        ),
    },
    {
        "key": "cell_from_hoechst_v1",
        "name": "cell_from_hoechst_best",
        "kind": "cellpose",
        "repo_id": "einarolafsson/cross-channel-cell-from-hoechst-cpsam",
        "repo_type": "model",
        "uri": "https://huggingface.co/einarolafsson/"
               "cross-channel-cell-from-hoechst-cpsam/resolve/main/"
               "weights/cell_from_hoechst_best",
        "sha256":
            "d1992433b4f2f291f73738830bb953198165fdc10719e54dae9b8c2bd430e0eb",
        "size_bytes": 1218647799,
        "metrics": {'n_train': '2,578 fields', 'train_objects': '237,957', 'n_test': '451 fields', 'test_objects': '45,098', 'cv': 'no, split by well', 'f1': '0.8697', 'aji': '0.7991', 'dice': '0.8948', 'stock_f1': '0.3012', 'stock_aji': '0.3506', 'stock_dice': '0.5235', 'train_loss': 'not recorded', 'val_loss': 'not recorded', 'best_epoch': '70 / 100'},
        "display_name": "Cross-channel cell-from-hoechst",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "the HOST CELL outline predicted from the Hoechst (nuclear) "
                   "channel alone; 2,578 training fields and 451 held-out test "
                   "fields, split by well so no well is on both sides",
        "versus_stock": "F1 0.870 against stock cpsam_v2's 0.301 on 451 "
                        "held-out fields at IoU 0.5 -- a delta of 0.569",
        "trained_on": (
            "Hoechst-stained nuclei paired with curated host-cell masks; "
            "fine-tuned from stock cpsam_v2, 100 epochs, best epoch 70"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "the counterpart of nuclei_from_cellmask_v1: that one predicts "
            "nuclei from the cell mask, this one predicts the cell from the "
            "nucleus",
            "precision 0.944 against recall 0.806 -- it misses cells rather "
            "than inventing them, which is the safer direction for counting",
            "quote the DELTA over stock (0.569), not the ratio: stock's mAP of "
            "0.0575 is a near-zero denominator that makes any ratio look huge",
        ),
    },
    {
        "key": "toxoplasma_from_hoechst_v1",
        "name": "toxoplasma_from_hoechst_pv",
        "kind": "cellpose",
        "repo_id": "einarolafsson/toxoplasma-from-hoechst-cpsam",
        "repo_type": "model",
        "uri": None,
        "sha256":
            "8dc05ebced3550d1a418c13d24d319e0482c742988df29a525520026cb2f0d96",
        "display_name": "Toxoplasma from Hoechst (cross-channel)",
        "architecture": "Cellpose-SAM (cpsam_v2)",
        "dataset": "Toxoplasma PV masks predicted from the HOECHST channel alone; "
                   "2567 training and 463 held-out fields, split by well, "
                   "hosts HFF/HeLa/THP1",
        "versus_stock": "F1 0.569 against 0.002 for stock cpsam_v2 on "
                        "463 well-grouped held-out fields, at IoU 0.5",
        "trained_on": (
            "cross-channel: given the Hoechst/nuclear image, predicts where the "
            "Toxoplasma parasitophorous vacuoles are, with no parasite stain. "
            "100 epochs, base cpsam_v2, AdamW lr 1e-05, targets are "
            "PV-regenerated masks"
        ),
        "trained_by": "einarolafsson",
        "notes": (
            "F1 0.569, AJI 0.421, Dice 0.546 at IoU 0.5 against "
            "stock cpsam_v2's 0.002/0.006/0.016",
            "the Hoechst route is harder than the cell-mask route -- compare "
            "toxoplasma_from_cellmask_v1",
            "the held-out split selects the checkpoint, so it is validation data "
            "rather than an independent test set",
            "accuracy falls above IoU 0.8 -- suited to counting, occupancy and "
            "area rather than precise morphometry",
        ),
    },
)

#: Models that are no longer OFFERED, by filename.
#:
#: ``toxo_plaque_cyto`` is retired: it
#: recalls 0.631 on the literature set -- it misses about a third of the
#: plaques -- against 0.811 for ``toxoplasma_plaque_v1``, and it published no
#: checksum, so its row in the picker had a Download button that could never
#: succeed.
#:
#: FILTERED FROM THE LISTING, NOT DELETED FROM DISK. The checkpoint still
#: ships, and ``plaque_model='bundled'`` still resolves to it, because a run
#: recorded against it has to stay reproducible: removing the weights would
#: silently change what re-running an old analysis produces, which is worse
#: than offering a model nobody should pick. It is simply no longer something
#: the zoo suggests.
RETIRED_MODEL_NAMES: frozenset = frozenset({
    "toxo_plaque_cyto_e25000_X1120_Y1120.CP_model",
})


#: Keys :func:`rank` will sort on, with the direction and what the number is.
#:
#: There is deliberately no accuracy key. A benchmark here has no ground truth
#: (see the module docstring), so a column called "score" that sorts models
#: would be inventing one.
RANK_KEYS: Dict[str, str] = {
    "qc": "fraction of fields spacr.seg_qc scored 'ok' — a quality-control "
          "verdict on this model's own masks, not an accuracy (higher first)",
    "seconds": "wall-clock segmentation time over the field set (lower first)",
}

#: :func:`rank`'s default key.
DEFAULT_RANK_KEY = "qc"

#: First bytes of a torch checkpoint. Everything torch has saved since 1.6 is a
#: zip; ``\\x80`` opens the legacy pickle protocol.
_TORCH_MAGICS = (b"PK\x03\x04", b"\x80")

#: ``name_v3`` -> ``('name', 3)``. See :func:`versioned_path`.
_VERSION_RE = re.compile(r"^(?P<base>.+)_v(?P<n>\d+)$")



class ModelZooError(Exception):
    """Base class for every refusal in this module."""


class ChecksumMismatch(ModelZooError):
    """What arrived is not what the catalogue published. Nothing was installed."""


class ModelUnreadable(ModelZooError):
    """A model file is missing, empty, or not a checkpoint. Names the file."""


class DownloadCancelled(ModelZooError):
    """The caller cancelled a fetch. Nothing was left at the destination."""


class IncomparableBenchmarks(ModelZooError):
    """Benchmarks from different field sets cannot be ranked against each other."""



@dataclass(frozen=True)
class ModelEntry:
    """One model the zoo knows about, wherever it lives.

    Frozen because an entry is a *record of a file at a moment* — the hash, the
    size and the provenance describe those bytes. Changing one in place would
    silently invalidate the other two; :func:`dataclasses.replace` makes the
    new record explicit.

    :param key: stable id, unique within a listing. For a local file this is
        derived from the filename; for a catalogue entry it is whatever the
        catalogue declared.
    :param name: the filename (or the published name) — what a human reads.
    :param kind: ``'cellpose'`` or ``'classifier'``; see :data:`KINDS`.
    :param source: ``'bundled'`` (ships with spaCR), ``'local'`` (found on this
        machine) or ``'remote'`` (declared in a catalogue, not yet fetched).
    :param path: absolute path on this machine, or ``''`` for a remote entry.
    :param uri: where a remote entry is fetched from, or ``''``.
    :param version: the zoo's own version number for a filename. ``'1'`` for a
        plain name, ``'2'`` for ``foo_v2.CP_model`` (see
        :func:`versioned_path`), or whatever a catalogue declared.
    :param sha256: hex digest. For a downloaded model this is the digest of the
        bytes that were actually written; for a catalogue entry it is the
        published digest to check against; ``''`` means "no checksum known",
        which :func:`fetch` treats as a refusal rather than a pass.
    :param size_bytes: file size, ``0`` when unknown.
    :param trained_on: what data produced this model, in prose, or
        :data:`UNKNOWN`. Never ``''``.
    :param trained_by: who produced it, or :data:`UNKNOWN`. Never ``''``.
    :param metrics: whatever numbers came with it — for a classifier, the
        best/last epoch metrics :func:`spacr.train_compare.load_run` recovered.
        Excluded from equality: two records of the same bytes are the same
        model whether or not somebody attached numbers to one of them.
    :param notes: everything the reader needs to know that is not a field:
        missing provenance, an unverified download, a file that does not look
        like a checkpoint.
    :param verified: True only when :attr:`sha256` was checked against a
        published digest. A downloaded file whose hash was merely *recorded* is
        not verified, and says so.
    :param settings_path: where the provenance came from, for the reader who
        wants to go and look at it.
    :param licence: the licence the model or package is published under, as
        its publisher states it (an SPDX identifier where there is one), or
        ``''`` when none is recorded.
    """

    key: str
    name: str
    kind: str = "cellpose"
    source: str = "local"
    path: str = ""
    uri: str = ""
    version: str = "1"
    sha256: str = ""
    size_bytes: int = 0
    trained_on: str = UNKNOWN
    trained_by: str = UNKNOWN
    metrics: Dict[str, Any] = _dc_field(default_factory=dict, compare=False)
    notes: Tuple[str, ...] = ()
    verified: bool = False
    settings_path: str = ""
    licence: str = ""

    def __post_init__(self):
        """Fill in the provenance fields and validate the kind.

        A blank ``trained_on`` or ``trained_by`` reads as "no constraints", so
        it is replaced with an explicit unknown -- the field has to say so out
        loud rather than by omission.

        :raises ValueError: if ``kind`` is not one of the known model kinds.
        """
        for attribute in ("trained_on", "trained_by"):
            value = str(getattr(self, attribute) or "").strip()
            object.__setattr__(self, attribute, value or UNKNOWN)
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def exists(self) -> bool:
        """True when :attr:`path` names a file that is here now."""
        return bool(self.path) and os.path.isfile(self.path)

    @property
    def provenance_known(self) -> bool:
        """True when this model says what it was trained on."""
        return self.trained_on != UNKNOWN

    @property
    def scorecard_known(self) -> bool:
        """True when this model says how accurate it is.

        The accuracy twin of :attr:`provenance_known`, which 370 asks for by
        name. A model with no numbers did not score zero, and a table of
        empty cells reads as the second -- so the absence is a state to
        report rather than a gap to render.
        """
        from .scorecard import scorecard_is_present

        return scorecard_is_present(self.metrics or {})

    @property
    def scorecard_holdout(self) -> str:
        """``name @ version`` of the hold-out set, or ``""``.

        A SCORECARD WITHOUT ITS SET IS A NUMBER WITHOUT A UNIT. Two people
        quoting an F1 for the same model have said nothing to each other
        unless they scored the same masks, so the set travels with the
        numbers into every surface that shows them.
        """
        metrics = self.metrics or {}
        name = str(metrics.get("holdout") or "").strip()
        version = str(metrics.get("holdout_version") or "").strip()
        if not name:
            return ""
        return f"{name} @ {version}" if version else name

    def scorecard_lines(self) -> List[str]:
        """The scorecard as display lines, or the sentence saying there is none.

        ONE SOURCE, FOUR RENDERINGS. The tooltip, the API page, the Zoo screen
        and the Hugging Face table all render THIS, so they cannot disagree --
        366 found six README tiles pointing at three different API pages, and
        that is what happens when a number is written down in more than one
        place.
        """
        from .scorecard import NO_SCORECARD, headline

        if not self.scorecard_known:
            return [NO_SCORECARD]
        metrics = self.metrics or {}
        finetuned = {k: v.get("finetuned") for k, v in metrics.items()
                     if isinstance(v, dict)}
        baseline = {k: v.get("vanilla") for k, v in metrics.items()
                    if isinstance(v, dict)}
        lines = list(headline(finetuned, baseline=baseline))
        holdout = self.scorecard_holdout
        if holdout:
            lines.append(f"hold-out set {holdout}")
        return lines

    @property
    def checksum_state(self) -> str:
        """What the checksum column says, in one word.

        ``'none'``
            no hash at all — nothing can be checked, and :func:`fetch` refuses
            such an entry unless the caller overrides it.
        ``'published'``
            a hash came with the entry but the bytes are not here yet, so it is
            a promise about what will arrive.
        ``'recorded'``
            the honest middle: the hash of the file on disk is known, but
            nobody published one to compare it against. It proves the file has
            not changed *since we looked*, and nothing more.
        ``'verified'``
            the bytes on disk were compared with a published digest and match.
        """
        if not self.sha256:
            return "none"
        if self.verified:
            return "verified"
        return "recorded" if self.exists else "published"

    def summary_line(self) -> str:
        """One line for a list widget."""
        bits = [self.name, self.kind, self.source, f"v{self.version}"]
        bits.append(f"trained on: {_shorten(self.trained_on, 48)}")
        bits.append(f"checksum {self.checksum_state}")
        if self.notes:
            bits.append(f"! {len(self.notes)} note"
                        f"{'s' if len(self.notes) > 1 else ''}")
        return " · ".join(bits)

    @property
    def model_card_url(self) -> str:
        """The Hugging Face page for this model, derived from its download uri.

        A checksum and a metrics table are not enough on their own: the reader
        wants the page that says what the model was trained on and shows its
        training curves. Derived rather than declared, so every Hugging Face
        entry has one without a per-entry field to forget.
        """
        uri = str(self.uri or "")
        marker = "huggingface.co/"
        if marker not in uri:
            return ""
        rest = uri.split(marker, 1)[1]
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 2 and parts[0] == "datasets":
            parts = parts[1:]
        if len(parts) < 2:
            return ""
        return f"https://huggingface.co/{parts[0]}/{parts[1]}"

    def describe(self) -> str:
        """The multi-line provenance card shown next to a selected model."""
        lines = [
            f"{self.name}  [{self.kind} · {self.source} · v{self.version}]",
            f"  path       {self.path or '(not downloaded)'}",
        ]
        if self.uri:
            lines.append(f"  uri        {self.uri}")
        lines.append(f"  size       {_human_bytes(self.size_bytes)}")
        lines.append(f"  sha256     {self.sha256 or '(none published)'} "
                     f"({self.checksum_state})")
        lines.append(f"  trained on {self.trained_on}")
        lines.append(f"  trained by {self.trained_by}")
        if self.licence:
            lines.append(f"  licence    {self.licence}")
        card = self.model_card_url
        if card:
            lines.append(f"  model card {card}")
        if self.settings_path:
            lines.append(f"  provenance {self.settings_path}")
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name:<10} {value}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        if not self.provenance_known:
            lines.append(
                "  ! this model does not say what it was trained on, so "
                "nothing here tells you whether it suits your images.")
        return "\n".join(lines)


def _shorten(text: Any, width: int) -> str:
    """Truncate text to a width, marking where it was cut.

    :param text: the text.
    :param width: the maximum length INCLUDING the ellipsis, so a column
        laid out at this width never overflows.
    :returns: the text, or its prefix with an ellipsis.
    """
    text = str(text)
    return text if len(text) <= width else text[:width - 1] + "…"


def _human_bytes(size: Any) -> str:
    """Bytes as something a person reads, ``'unknown'`` for 0/None."""
    try:
        n = float(size)
    except (TypeError, ValueError):
        return UNKNOWN
    if n <= 0:
        return UNKNOWN
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        if unit == "GB":
            continue
        n /= 1024.0
    return f"{n:.1f} GB"



def sha256_file(path: Any, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks so a 2 GB checkpoint is not RAM.

    :param path: the file.
    :param chunk_size: bytes per read.
    :returns: the lowercase hex digest.
    :raises ModelUnreadable: when the file is missing or cannot be read, with
        the path in the message.
    """
    p = Path(path)
    digest = hashlib.sha256()
    try:
        with p.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except FileNotFoundError:
        raise ModelUnreadable(f"no such model file: {p}") from None
    except OSError as e:
        raise ModelUnreadable(f"could not read {p}: {e}") from None
    return digest.hexdigest()


def verify(entry: ModelEntry, expected: Optional[str] = None) -> bool:
    """Hash the file this entry points at and compare it to a known digest.

    :param entry: the entry to check.
    :param expected: the digest to compare against; defaults to
        :attr:`ModelEntry.sha256`.
    :returns: True when the file's digest matches.
    :raises ModelUnreadable: when the entry has no local file, naming it.
    :raises ModelZooError: when there is no digest to compare against — that is
        a caller error, and returning False for it would read as "the file is
        wrong" when what happened is "nobody said what right looks like".
    """
    if not entry.path:
        raise ModelUnreadable(
            f"{entry.name} has not been downloaded, so there is nothing to "
            f"verify (source={entry.source}, uri={entry.uri or 'none'})")
    if not os.path.isfile(entry.path):
        raise ModelUnreadable(f"no such model file: {entry.path}")
    want = (expected if expected is not None else entry.sha256) or ""
    want = want.strip().lower()
    if not want:
        raise ModelZooError(
            f"no checksum recorded for {entry.name} ({entry.path}) — there is "
            f"nothing to verify it against. Compute one with sha256_file() and "
            f"put it in the catalogue.")
    return sha256_file(entry.path) == want



def _looks_like_checkpoint(path: Path) -> bool:
    """True when the first bytes are a torch save (zip or legacy pickle)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(4)
    except OSError:
        return False
    return any(head.startswith(magic) for magic in _TORCH_MAGICS)


def classify_kind(path: Any) -> Optional[str]:
    """Say whether a file is a Cellpose model, a classifier, or not a model.

    The rules, in order:

    1. ``*.CP_model`` is a Cellpose checkpoint — that is what
       :func:`spacr.submodules.train_cellpose` names its output.
    2. ``*.pth`` / ``*.pt`` is a classifier checkpoint
       (:func:`spacr.io._save_model` writes
       ``<model_type>_epoch_<n>_channels_<ch>.pth``) **unless** it sits in a
       Cellpose folder or has ``cellpose``/``cp_model`` in its name.
    3. An **extensionless** file inside a Cellpose folder is a Cellpose
       checkpoint only if its first bytes are a torch save. ``cellpose.train``
       writes ``<save_path>/models/<name>`` with no suffix, and that folder
       also holds READMEs and logs — the magic-byte check is what keeps a
       ``README`` out of the zoo.
    4. Anything else is not a model. CSVs, PNGs, ``.npy`` masks and settings
       snapshots all land here and are ignored.

    :param path: a file path.
    :returns: ``'cellpose'``, ``'classifier'`` or None.
    """
    p = Path(path)
    low = p.name.lower()
    near = {p.parent.name.lower(), p.parent.parent.name.lower()}
    in_cellpose_dir = bool(near & set(CELLPOSE_DIR_NAMES))

    if low.endswith(CELLPOSE_SUFFIXES):
        return "cellpose"
    if low.endswith(CLASSIFIER_SUFFIXES):
        if "cellpose" in low or "cp_model" in low or in_cellpose_dir:
            return "cellpose"
        return "classifier"
    if not p.suffix and in_cellpose_dir:
        return "cellpose" if _looks_like_checkpoint(p) else None
    return None


def inspect_checkpoint(path: Any, loader: Optional[Callable[[str], Any]] = None,
                       deep: bool = False) -> Dict[str, Any]:
    """Check a file is a loadable checkpoint, failing with the filename in it.

    The default failure for a wrong or corrupt checkpoint is a ``KeyError`` on
    a state-dict key raised somewhere inside torch, which names nothing the
    user chose and reads like a spaCR bug. This turns all of it —  missing,
    empty, truncated, a PNG somebody renamed, a Cellpose model handed to the
    classifier path — into one :class:`ModelUnreadable` naming the file.

    The shallow check needs no torch at all: it is a stat and four bytes.

    :param path: the checkpoint.
    :param loader: ``fn(path) -> object`` used for the deep check; defaults to
        ``torch.load(..., map_location='cpu')``, imported only if used.
    :param deep: actually load the file. Off by default because loading a 2 GB
        checkpoint to populate a list widget is not acceptable.
    :returns: ``{'path', 'size_bytes', 'format', 'loaded'}``.
    :raises ModelUnreadable: naming the file, always.
    """
    p = Path(path)
    if not p.exists():
        raise ModelUnreadable(f"no such model file: {p}")
    if p.is_dir():
        raise ModelUnreadable(
            f"{p} is a directory, not a model checkpoint — point at the file "
            f"inside it")
    size = p.stat().st_size
    if size == 0:
        raise ModelUnreadable(
            f"{p} is empty (0 bytes) — an interrupted download or copy leaves "
            f"exactly this")
    try:
        with p.open("rb") as handle:
            head = handle.read(4)
    except OSError as e:
        raise ModelUnreadable(f"could not read {p}: {e}") from None
    if not any(head.startswith(magic) for magic in _TORCH_MAGICS):
        raise ModelUnreadable(
            f"{p} is not a PyTorch checkpoint: it starts with {head!r}, and a "
            f".pth / .CP_model file starts with a zip header or a pickle "
            f"opcode. Check the path — this is usually a text file, an HTML "
            f"error page saved by a failed download, or the wrong file "
            f"entirely.")
    fmt = "zip" if head.startswith(b"PK\x03\x04") else "pickle"

    out: Dict[str, Any] = {"path": str(p), "size_bytes": int(size),
                           "format": fmt, "loaded": False}
    if not deep:
        return out
    load = loader if loader is not None else _torch_loader
    try:
        load(str(p))
    except ModelUnreadable:
        raise
    except Exception as e:
        raise ModelUnreadable(
            f"{p} could not be loaded as a model checkpoint "
            f"({type(e).__name__}: {e}). The file is a torch save but not the "
            f"architecture that was asked for — check that this is a "
            f"{'Cellpose' if p.name.lower().endswith(CELLPOSE_SUFFIXES) else 'classifier'} "
            f"model and not the other kind.") from None
    out["loaded"] = True
    return out


def _torch_loader(path: str) -> Any:
    """``torch.load`` on the CPU. Imported here so the module stays torch-free."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)



def _read_key_value_csv(path: Path) -> Dict[str, Any]:
    """Read a ``Key,Value`` settings CSV the way the run diff reads one.

    Reuses :func:`spacr.run_journal._read_settings_csv` rather than writing a
    second parser: that one already handles the JSON/CSV/live-dict round trips
    a spaCR settings dict takes, and the zoo has to agree with the run diff
    about what a settings file says.
    """
    from .run_journal import _read_settings_csv

    return _read_settings_csv(path)


def _settings_beside(path: Path) -> Tuple[Dict[str, Any], str]:
    """Find the settings snapshot beside a Cellpose checkpoint.

    :func:`spacr.submodules.train_cellpose` calls ``save_settings(settings,
    name=model_name)``, which writes ``<src>/settings/<model_name>.csv``; the
    bundled pack ships ``<model_file>_settings.csv`` next to the weights. Both
    are looked for, nearest first.

    :returns: ``(settings, where)``; ``({}, '')`` when there is none.
    """
    candidates = [
        path.with_name(path.name + "_settings.csv"),
        path.with_name(path.stem + "_settings.csv"),
        path.with_suffix(".csv"),
    ]
    node = path.parent
    for _ in range(DEFAULT_SCAN_DEPTH):
        candidates.append(node / "settings" / f"{path.name}.csv")
        candidates.append(node / "settings" / f"{path.stem}.csv")
        node = node.parent
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            settings = _read_key_value_csv(candidate)
        except Exception:
            continue
        if settings:
            return settings, str(candidate)
    return {}, ""


def _describe_cellpose_training(settings: Mapping[str, Any]) -> str:
    """One prose line saying what a Cellpose model saw, or :data:`UNKNOWN`.

    Deliberately concrete: magnification and confluence are not in the settings
    file, but the source folder, the crop size, the diameter and the epoch
    count are, and together they are enough for a reader to tell "this is not
    my data".
    """
    source = settings.get("img_src") or settings.get("src") or ""
    bits: List[str] = []
    if source:
        bits.append(str(source))
    shape = settings.get("width_height") or settings.get("target_size")
    if shape:
        bits.append(f"crops {shape}")
    if settings.get("diameter"):
        bits.append(f"diameter {settings['diameter']}")
    if settings.get("n_epochs"):
        bits.append(f"{settings['n_epochs']} epochs")
    if settings.get("grayscale") is True or str(settings.get("grayscale")).lower() == "true":
        bits.append("greyscale")
    return ", ".join(bits) if bits else UNKNOWN


def _describe_classifier_training(settings: Mapping[str, Any]) -> str:
    """One prose line saying what a classifier saw, or :data:`UNKNOWN`."""
    bits: List[str] = []
    source = settings.get("src") or ""
    if source:
        bits.append(str(source))
    if settings.get("model_type"):
        bits.append(str(settings["model_type"]))
    if settings.get("classes"):
        bits.append(f"classes {settings['classes']}")
    if settings.get("image_size"):
        bits.append(f"{settings['image_size']}px crops")
    if settings.get("epochs"):
        bits.append(f"{settings['epochs']} epochs")
    return ", ".join(bits) if bits else UNKNOWN


def _who(settings: Mapping[str, Any]) -> str:
    """Who trained it, from whatever the settings recorded, or :data:`UNKNOWN`."""
    for key in ("user", "author", "trained_by", "operator", "hostname", "host"):
        value = settings.get(key)
        if value not in (None, "", "nan"):
            return str(value)
    return UNKNOWN



def _version_of(name: str) -> str:
    """The zoo's version number for a filename (see :func:`versioned_path`)."""
    stem = Path(name).stem
    match = _VERSION_RE.match(stem)
    return match.group("n") if match else "1"


def _key_for(path: Path) -> str:
    """A short, stable key for a local file: its name, made filesystem-safe."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name)


def entry_from_file(path: Any, kind: Optional[str] = None,
                    source: str = "local", key: Optional[str] = None,
                    runs: Optional[Mapping[str, Any]] = None,
                    compute_hash: bool = False,
                    sha256: str = "", verified: bool = False,
                    extra_notes: Sequence[str] = ()) -> ModelEntry:
    """Build a :class:`ModelEntry` for a checkpoint on this machine.

    Provenance is recovered from whatever spaCR already wrote beside the model:
    a ``*_settings.csv`` or ``<src>/settings/<name>.csv`` for a Cellpose model,
    and — for a classifier — the training run the checkpoint sits in, loaded
    through :func:`spacr.train_compare.load_run` so the zoo and the training-run
    comparison agree about where settings live and what they say.

    :param path: the checkpoint.
    :param kind: override :func:`classify_kind`.
    :param source: ``'local'`` or ``'bundled'``.
    :param key: override the generated key.
    :param runs: ``{folder: TrainingRun}`` from :func:`_runs_under`, so a scan
        of 40 checkpoints in one run folder reads that folder once.
    :param compute_hash: hash the file now. Off by default: hashing every
        checkpoint on a machine to populate a list widget is minutes.
    :param sha256: a digest already known for these bytes.
    :param verified: whether ``sha256`` was checked against a published digest.
    :param extra_notes: notes to carry onto the entry.
    :returns: the entry.
    :raises ModelUnreadable: when the path is not a file.
    """
    p = Path(path)
    if not p.is_file():
        raise ModelUnreadable(f"no such model file: {p}")
    kind = kind or classify_kind(p) or "classifier"
    notes: List[str] = list(extra_notes)

    settings: Dict[str, Any] = {}
    settings_path = ""
    metrics: Dict[str, Any] = {}
    if kind == "classifier":
        run = _run_for(p, runs)
        if run is not None:
            settings = dict(run.settings)
            settings_path = run.settings_path
            metrics = _metrics_from_run(run)
        if not settings:
            settings, settings_path = _settings_beside(p)
        trained_on = _describe_classifier_training(settings)
    else:
        settings, settings_path = _settings_beside(p)
        trained_on = _describe_cellpose_training(settings)

    if trained_on == UNKNOWN:
        notes.append(
            "no settings snapshot found beside this model, so what it was "
            "trained on is unknown — treat it as untested on your images")
    if not _looks_like_checkpoint(p):
        notes.append(
            f"{p.name} does not start with a torch header; it may be a Git LFS "
            f"pointer, a failed download, or not a model at all")

    try:
        size = p.stat().st_size
    except OSError:
        size = 0

    digest = sha256
    if compute_hash and not digest:
        digest = sha256_file(p)

    return ModelEntry(
        key=key or _key_for(p),
        name=p.name,
        kind=kind,
        source=source,
        path=str(p.resolve()),
        version=_version_of(p.name),
        sha256=digest,
        size_bytes=int(size),
        trained_on=trained_on,
        trained_by=_who(settings),
        metrics=metrics,
        notes=tuple(notes),
        verified=bool(verified and digest),
        settings_path=settings_path,
    )


def _metrics_from_run(run: Any) -> Dict[str, Any]:
    """Best/last accuracy off a :class:`spacr.train_compare.TrainingRun`.

    Both, never one: the best epoch of a validation curve was chosen using that
    curve, so it is optimistically biased; the last epoch is unbiased but may
    be well past the optimum. :mod:`spacr.train_compare` makes that argument at
    length and reports both for it — the zoo shows the same pair for the same
    reason.
    """
    out: Dict[str, Any] = {}
    final = getattr(run, "final_metrics", None) or {}
    for label, entry in final.items():
        if str(entry.get("split")) != "val":
            continue
        best = (entry.get("best") or {}).get("accuracy")
        last = (entry.get("last") or {}).get("accuracy")
        if best:
            out[f"{label} best accuracy"] = f"{best['value']:.4f} @ epoch {best['epoch']}"
        if last:
            out[f"{label} last accuracy"] = f"{last['value']:.4f} @ epoch {last['epoch']}"
    if not out:
        for label, entry in final.items():
            best = (entry.get("best") or {}).get("accuracy")
            if best:
                out[f"{label} best accuracy"] = (
                    f"{best['value']:.4f} @ epoch {best['epoch']} "
                    f"(train split — not held out)")
    return out


def _run_for(path: Path, runs: Optional[Mapping[str, Any]]) -> Any:
    """The discovered training run a checkpoint belongs to, or one loaded now."""
    if runs:
        for folder in (path.parent, path.parent.parent):
            run = runs.get(str(folder.resolve()))
            if run is not None:
                return run
    from .train_compare import load_run

    for folder in (path.parent, path.parent.parent):
        try:
            return load_run(folder)
        except Exception:
            continue
    return None


def _runs_under(root: Path) -> Dict[str, Any]:
    """``{folder: TrainingRun}`` for every training run below ``root``.

    Straight reuse of :func:`spacr.train_compare.find_runs` — the classifier
    half of the zoo *is* the training-run scan, and a second discovery pass
    would drift out of step with it the first time the on-disk layout changed.
    Failures are swallowed: a zoo that cannot list local models because one
    folder was unreadable is worse than one with thinner provenance.
    """
    try:
        from .train_compare import find_runs

        return {str(Path(run.path).resolve()): run for run in find_runs(root)}
    except Exception:
        return {}



def package_model_root() -> Path:
    """``<spacr>/resources/models`` — where the bundled pack lives.

    The same folder :func:`spacr.utils.download_models` fills and
    :func:`spacr.submodules.analyze_plaques` reads from.
    """
    return Path(__file__).resolve().parent / "resources" / "models"


def default_local_roots() -> List[Path]:
    """Folders worth scanning when the caller has not named one.

    The bundled pack, the Cellpose user folder, and spaCR's own model cache.
    Only the ones that exist come back.
    """
    roots = [
        package_model_root(),
        Path.home() / ".cellpose" / "models",
        Path.home() / ".spacr" / "models",
    ]
    return [r for r in roots if r.is_dir()]


def _walk(base: Path, max_depth: int, limit: int) -> Iterator[Path]:
    """Every file at most ``max_depth`` levels below ``base``, hidden dirs skipped.

    ``limit`` caps the files *examined*, not the models found, so a folder with
    a million PNGs in it is slow rather than fatal and does not silently drop
    the models below them.
    """
    stack: List[Tuple[Path, int]] = [(base, 0)]
    seen = 0
    while stack and seen < limit:
        node, depth = stack.pop(0)
        try:
            children = sorted(node.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir():
                if depth < max_depth:
                    stack.append((child, depth + 1))
            elif child.is_file():
                seen += 1
                yield child
                if seen >= limit:
                    return


def discover_local(roots: Any = None, max_depth: int = DEFAULT_SCAN_DEPTH,
                   compute_hashes: bool = False,
                   limit: int = DEFAULT_SCAN_LIMIT) -> List[ModelEntry]:
    """Find the model checkpoints already on this machine.

    Cellpose models and classifier checkpoints are told apart by
    :func:`classify_kind`; everything else in the folders — settings CSVs, mask
    ``.npy`` files, montage PNGs, logs — is ignored.

    Nothing is downloaded and nothing is hashed unless ``compute_hashes`` is
    set: this is the function behind a list widget, and it has to be fast
    enough to run on a folder the user just typed.

    :param roots: a folder, a file, or an iterable of them; None uses
        :func:`default_local_roots`.
    :param max_depth: how deep below each root to look.
    :param compute_hashes: hash every file found (minutes on a big folder).
    :param limit: stop after examining this many files per root.
    :returns: entries, Cellpose first, then by name.
    """
    entries: List[ModelEntry] = []
    seen: set = set()
    for root in _as_paths(roots if roots is not None else default_local_roots()):
        if root.is_file():
            candidates: List[Path] = [root]
            runs: Dict[str, Any] = {}
        elif root.is_dir():
            candidates = list(_walk(root, max_depth, limit))
            runs = _runs_under(root)
        else:
            continue
        for path in candidates:
            kind = classify_kind(path)
            if kind is None:
                continue
            try:
                resolved = str(path.resolve())
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            source = ("bundled" if _under(path, package_model_root())
                      else "local")
            try:
                entries.append(entry_from_file(path, kind=kind, source=source,
                                               runs=runs,
                                               compute_hash=compute_hashes))
            except ModelUnreadable:
                continue
    entries.sort(key=lambda e: (0 if e.kind == "cellpose" else 1, e.name))
    return entries


def _under(path: Path, root: Path) -> bool:
    """Whether a path is inside a root, after resolving both.

    Resolved first, so a symlink or a ``..`` cannot escape the root while
    appearing to be under it -- this gates where a downloaded checkpoint may
    be written.

    :param path: the path to test.
    :param root: the root it must be under.
    :returns: ``True`` when it is; ``False`` when it is not, and also when
        either path cannot be resolved.
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _as_paths(roots: Any) -> List[Path]:
    """Normalise a folder / path / iterable-of-those into a list of Paths."""
    if roots is None:
        return []
    if isinstance(roots, (str, os.PathLike)):
        return [Path(roots)]
    return [Path(r) for r in roots]



def hf_uri(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """The download URL for a file in a Hugging Face repo.

    :param repo_id: Hugging Face repository identifier.
    :param filename: repository-relative name of the file to download.
    :param repo_type: ``"dataset"`` (the default, and what spaCR shipped
        first) or ``"model"``.

    Exactly the URL :func:`spacr.utils.download_models` and
    :func:`spacr.qt.hf_download._download_one` build, kept in one place so the
    zoo cannot drift away from the downloader spaCR already ships.

    THE TWO REPO KINDS HAVE DIFFERENT URLS, which is not cosmetic: a dataset
    file lives under ``/datasets/<repo>/resolve/...`` and a model file under
    ``/<repo>/resolve/...``. Asking for one at the other's URL returns a 404
    page, and a downloader that does not check the content type writes that
    HTML into the destination and leaves a "checkpoint" that fails to load
    with a torch error naming neither the URL nor the repo.

    ``dataset`` remains the default because :data:`HF_MODELS_REPO` is a
    DATASET repo -- ``einarolafsson/models`` -- and every entry written before
    this parameter existed assumes it. New model repos pass ``"model"``.
    """
    kind = str(repo_type or "dataset").lower()
    if kind not in ("dataset", "model"):
        raise ValueError(
            f"repo_type must be 'dataset' or 'model', not {repo_type!r}")
    prefix = "datasets/" if kind == "dataset" else ""
    return (f"https://huggingface.co/{prefix}{repo_id}/resolve/main/"
            f"{filename}?download=true")


def _entry_from_mapping(data: Mapping[str, Any],
                        source: str = "remote") -> ModelEntry:
    """One catalogue record -> a :class:`ModelEntry`."""
    name = str(data.get("name") or data.get("key") or "")
    if not name:
        raise ValueError("a catalogue entry needs at least a name")
    uri = data.get("uri")
    if not uri:
        uri = hf_uri(str(data.get("repo_id") or HF_MODELS_REPO), name,
                     str(data.get("repo_type") or "dataset"))
    notes = tuple(str(n) for n in (data.get("notes") or ()))
    sha = str(data.get("sha256") or "").strip().lower()
    if not sha:
        notes = notes + (
            "no published checksum — this entry cannot be verified, and fetch "
            "refuses it unless you explicitly accept that",)
    return ModelEntry(
        key=str(data.get("key") or name),
        name=name,
        kind=str(data.get("kind") or "cellpose"),
        source=str(data.get("source") or source),
        path=str(data.get("path") or ""),
        uri=str(uri),
        version=str(data.get("version") or "1"),
        sha256=sha,
        size_bytes=int(data.get("size_bytes") or 0),
        trained_on=data.get("trained_on") or UNKNOWN,
        trained_by=data.get("trained_by") or UNKNOWN,
        metrics=dict(data.get("metrics") or {}),
        notes=notes,
        licence=str(data.get("licence") or data.get("license") or ""),
    )


_SHARED_CATALOGUE_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "entries": ()}

#: Set while a background refresh is in flight, so a screen that opens twice
#: in a second starts one fetch rather than two.
_SHARED_CATALOGUE_FETCHING = threading.Event()


def _on_the_qt_gui_thread() -> bool:
    """Whether this call is running on Qt's GUI thread.

    Answers False for a process with no Qt, for a worker thread, and for
    anything that goes wrong while asking -- so the only thing this can do
    is turn a blocking fetch into a background one, never the reverse.
    """
    try:
        from PySide6.QtCore import QCoreApplication, QThread
    except Exception:                                        # noqa: BLE001
        return False
    try:
        app = QCoreApplication.instance()
        return app is not None and QThread.currentThread() is app.thread()
    except Exception:                                        # noqa: BLE001
        return False


def _refresh_shared_catalogue_in_background(uri: Optional[str],
                                            timeout: float) -> None:
    """Fetch the catalogue on a daemon thread, for the cache to serve later.

    Nothing waits on the thread and nothing is redrawn when it lands: the
    point is only that the NEXT caller answers from a warm cache instead of
    from the network.
    """
    if _SHARED_CATALOGUE_FETCHING.is_set():
        return
    _SHARED_CATALOGUE_FETCHING.set()

    def run() -> None:
        """Fetch the catalogue off the GUI thread, and always release the flag.

        The `finally` is the whole point: `_SHARED_CATALOGUE_FETCHING` is what
        stops a second refresh being started while this one is in flight, so a
        fetch that raises must still clear it or no later refresh can ever
        begin.
        """
        try:
            shared_catalogue(uri, timeout=timeout, force=True, block=True)
        finally:
            _SHARED_CATALOGUE_FETCHING.clear()

    threading.Thread(target=run, daemon=True,
                     name="spacr-model-catalogue").start()


def shared_catalogue(uri: Optional[str] = None, *,
                     timeout: float = DEFAULT_TIMEOUT,
                     force: bool = False,
                     block: Optional[bool] = None) -> Tuple["ModelEntry", ...]:
    """The community catalogue, fetched from :data:`REMOTE_CATALOGUE_URI`.

    :param uri: override the catalogue location.
    :param timeout: seconds to wait for the request.
    :param force: ignore the cache and re-fetch.
    :param block: whether to wait for the network. ``None`` -- the default,
        and what an unthinking caller gets -- waits everywhere EXCEPT Qt's
        GUI thread, where it answers from the cache and refreshes on a daemon
        thread. ``True`` waits wherever it is called, which only a caller that
        knows it is on a worker or in a CLI may ask for. ``False`` never
        waits.
    :returns: the entries, or ``()`` when the catalogue cannot be read.

    NEVER FETCHES ON THE GUI THREAD, and that is not an optimisation. This is
    reached from ``spacr.settings.downloaded_zoo_models`` while a settings
    panel is being built, so it ran inside ``MainWindow._on_nav_selected``
    with nothing able to paint or answer the compositor. Measured
    with the catalogue host non-routable (10.255.255.1, the shape of a down
    VPN or a captive portal -- the connect neither completes nor is refused):
    opening the Mask module took **32.2 s**, all of it a GUI thread stuck in
    ``urlopen``. GNOME asks a window whether it is alive after five, so what
    the user sees is spaCR's "force quit" dialog, which is how this was
    reported. With the fetch moved off the thread the same open is 2.4 s.

    The cost of not waiting is a first module open whose Cellpose dropdown
    lists the bundled and local models but not the community ones; the
    background refresh means the second one has them.

    NEVER RAISES, and that is deliberate. This runs when a user opens a module
    that offers a model list, and the list is useful without it: the bundled
    entries and any local models are still there. A laptop on a train, a lab
    behind a proxy and a Hugging Face outage all produce the same thing -- a
    shorter list and a log line -- rather than a module that will not open.

    The failure that WOULD be silent and harmful is a corrupt or hostile
    catalogue, so entries that do not parse are dropped individually, and an
    entry without a ``sha256`` still cannot be installed by :func:`fetch`
    without an explicit override. A catalogue row is a claim about where a file
    lives; the checksum is what makes it a claim about which file.
    """
    import time

    target = uri or REMOTE_CATALOGUE_URI
    now = time.time()
    if (not force and float(_SHARED_CATALOGUE_CACHE["fetched_at"]) > 0
            and now - float(_SHARED_CATALOGUE_CACHE["fetched_at"])
            < CATALOGUE_CACHE_SECONDS):
        return tuple(_SHARED_CATALOGUE_CACHE["entries"])

    wait = (not _on_the_qt_gui_thread()) if block is None else bool(block)
    if not wait:
        _refresh_shared_catalogue_in_background(uri, timeout)
        return tuple(_SHARED_CATALOGUE_CACHE["entries"])

    try:
        import urllib.request

        with urllib.request.urlopen(target, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:                                # noqa: BLE001
        _SHARED_CATALOGUE_CACHE["fetched_at"] = now
        first = not _SHARED_CATALOGUE_CACHE.get("warned")
        _SHARED_CATALOGUE_CACHE["warned"] = True
        (LOG.info if first else LOG.debug)(
            "shared model catalogue unavailable (%s): %s",
            type(exc).__name__, exc)
        return tuple(_SHARED_CATALOGUE_CACHE["entries"])

    records = payload.get("models") if isinstance(payload, Mapping) else payload
    entries: List[ModelEntry] = []
    for record in (records or ()):
        try:
            entries.append(_entry_from_mapping(record, source="shared"))
        except Exception as exc:                            # noqa: BLE001
            LOG.warning("skipping a shared catalogue entry: %s", exc)
    _SHARED_CATALOGUE_CACHE.update(fetched_at=now, entries=tuple(entries))
    return tuple(entries)


def shared_catalogue_is_stale() -> bool:
    """Whether :func:`shared_catalogue` would go to the network to answer.

    For a caller that wants to do the waiting somewhere it is allowed to --
    a worker thread -- rather than get the cached answer and not know it was
    one.
    """
    stamp = float(_SHARED_CATALOGUE_CACHE["fetched_at"])
    return stamp <= 0 or (time.time() - stamp) >= CATALOGUE_CACHE_SECONDS


def publish_model(local_path: Any, repo_id: str, *,
                  key: str,
                  kind: str = "cellpose",
                  trained_on: str = UNKNOWN,
                  trained_by: str = UNKNOWN,
                  private: bool = False,
                  notes: Sequence[str] = ()) -> Dict[str, Any]:
    """Upload a model to Hugging Face and return its catalogue row.

    :param local_path: the checkpoint to upload.
    :param repo_id: ``<user>/<repo>`` -- YOUR OWN account.
    :param key: the short name spaCR will offer the model under.
    :param kind: one of :data:`KINDS`.
    :param trained_on: what the model was trained on. Say it properly: this is
        the only thing another lab has to decide whether it applies to them.
    :param trained_by: who trained it, and roughly when.
    :param private: keep the repo private. A private model cannot be fetched by
        other spaCR users, so it is off by default.
    :param notes: caveats worth carrying next to the model.
    :returns: the catalogue row, with the sha256 filled in.
    :raises ImportError: when ``huggingface_hub`` is not installed.

    THE CHECKSUM IS COMPUTED HERE, from the file that was actually uploaded,
    which is the whole reason this exists as a function rather than as
    instructions in a README. :func:`fetch` refuses an entry it cannot verify,
    so a row written by hand without a hash produces a model nobody can install
    without disabling the check -- which is what the one pre-existing bundled
    entry does, and it is a hole rather than a precedent.

    Publishing does NOT distribute the model on its own: add the returned row
    to the shared catalogue (:data:`REMOTE_CATALOGUE_URI`) and every spaCR user
    sees it within :data:`CATALOGUE_CACHE_SECONDS`.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Publishing a model needs the 'huggingface_hub' package:\n"
            "  pip install huggingface_hub\n"
            "then log in with `huggingface-cli login`.") from exc

    path = Path(str(local_path))
    if not path.is_file():
        raise ModelZooError(f"{path} is not a file")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=bool(private),
                    exist_ok=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                    repo_id=repo_id, repo_type="model")

    row = {
        "key": key,
        "name": path.name,
        "kind": kind,
        "repo_id": repo_id,
        "repo_type": "model",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "trained_on": trained_on,
        "trained_by": trained_by,
        "notes": tuple(notes),
    }
    LOG.info("published %s to %s; add this row to the shared catalogue",
             path.name, repo_id)
    return row


def load_catalogue_file(path: Any) -> List[ModelEntry]:
    """Read a JSON catalogue of remote models.

    Format — a list, or an object with a ``models`` list::

        {"models": [
          {"key": "hela_60x",
           "name": "hela_60x_confluent.CP_model",
           "kind": "cellpose",
           "uri": "https://…/hela_60x_confluent.CP_model",
           "sha256": "9f86d0…",
           "size_bytes": 26566572,
           "trained_on": "HeLa, 60x, confluent monolayer, 512px crops",
           "trained_by": "A. Researcher, 2026-02",
           "metrics": {"note": "benchmarked on plate3 fields 1-3"}}
        ]}

    ``sha256`` is the field that decides whether the entry is usable without an
    explicit override, so a catalogue is worth exactly as much as its hashes.

    :param path: the JSON file.
    :returns: the entries.
    :raises ModelZooError: when the file cannot be read or is not a catalogue,
        naming the file.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        raise ModelZooError(f"no such catalogue file: {p}") from None
    except (OSError, ValueError) as e:
        raise ModelZooError(f"could not read the catalogue {p}: {e}") from None
    records = data.get("models") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ModelZooError(
            f"{p} is not a model catalogue — expected a list of entries, or an "
            f"object with a 'models' list, got {type(records).__name__}")
    out: List[ModelEntry] = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ModelZooError(
                f"{p}: entry {i} is a {type(record).__name__}, not an object")
        try:
            out.append(_entry_from_mapping(record))
        except ValueError as e:
            raise ModelZooError(f"{p}: entry {i} is unusable — {e}") from None
    return out


#: What each Cellpose stock model is, for the zoo listing. Cellpose publishes
#: names, not descriptions, and a row reading only "cpdino" tells nobody
#: whether it applies to their images.
STOCK_CELLPOSE_NOTES = {
    "cpsam": ("Cellpose-SAM v1, the original SAM-based generalist. Superseded "
              "by cpsam_v2 but kept for reproducing older runs."),
    "cpsam_v2": ("Cellpose-SAM v2, the current Cellpose generalist and the "
                 "base every spaCR fine-tune here starts from."),
    "cpdino": "Cellpose-DINO, a DINO-backbone generalist.",
    "cpdino-vitb": "Cellpose-DINO with the larger ViT-B backbone.",
}


def stock_cellpose_entries() -> List["ModelEntry"]:
    """Every model the installed Cellpose can fetch for itself.

    These are not spaCR's files and carry no checksum of ours: Cellpose
    downloads and verifies them, and the name IS the path -- passing "cpsam"
    to Cellpose resolves it. They are listed so that the zoo answers "what can
    I segment with" rather than "what has Einar trained", which is the
    question a new user actually has.
    """
    # Deliberately does NOT import cellpose: importing this module must stay
    # free of torch and cellpose (there is a test for it, and the GUI lists
    # models long before anything segments). If cellpose is already loaded its
    # own list is authoritative; otherwise the names above are the fallback.
    import sys

    loaded = sys.modules.get("cellpose.models")
    names = list(getattr(loaded, "MODEL_NAMES", ()) or ()) if loaded else []
    names = names or list(STOCK_CELLPOSE_NOTES)
    home = Path.home() / ".cellpose" / "models"
    out = []
    for name in names:
        local = home / name
        out.append(ModelEntry(
            key=_key_for(Path(name)), name=name,
            path=str(local) if local.is_file() else name,
            kind="cellpose", source="stock", uri="", sha256="",
            size_bytes=local.stat().st_size if local.is_file() else 0,
            trained_on=STOCK_CELLPOSE_NOTES.get(
                name, "Cellpose stock model; see the Cellpose documentation."),
            trained_by="Cellpose"))
    return out


#: bioimage.io's published collection.
BIOIMAGEIO_COLLECTION = ("https://hypha.aicell.io/bioimage-io/artifacts/"
                         "bioimage.io/children?limit=1000&pagination=false")

#: What spaCR's Cellpose can actually load. The test is COMPATIBILITY, not the
#: word "cellpose": spaCR runs Cellpose 4, so the SAM and DINO backbones load
#: and cyto3 and earlier do not -- they carry the same "cellpose" tag, load,
#: and then produce nonsense. The older collection.json does not list these
#: models at all, which is why reading it found nothing.
BIOIMAGEIO_COMPATIBLE = ("cellpose sam", "cpsam", "cellposedino",
                         "cellpose dino", "cpdino")

#: How long a fetched collection is trusted before it is fetched again.
BIOIMAGEIO_CACHE_HOURS = 24


def _looks_like_cellpose_sam(manifest: Mapping[str, Any]) -> bool:
    """Whether a bioimage.io manifest is loadable by the Cellpose spaCR runs.

    STRICTLY. spaCR loads these as drop-in replacements, and only the Cellpose
    4 backbones work that way: a cyto3 or a ResNet model carries the same
    "cellpose" tag, loads, and then produces nonsense.
    """
    if str(manifest.get("type") or "") != "model":
        return False
    parts = [str(manifest.get("name") or ""), str(manifest.get("description") or "")]
    parts += [str(t) for t in (manifest.get("tags") or ())]
    text = " ".join(parts).lower().replace("-", " ").replace("_", " ")
    return any(key in text for key in BIOIMAGEIO_COMPATIBLE)


def bioimageio_entries(timeout: float = 5.0,
                       url: Optional[str] = None,
                       allow_network: bool = False) -> List["ModelEntry"]:
    """Cellpose models published on bioimage.io, or an empty list.

    Two kinds of row. A Cellpose-SAM or Cellpose-DINO model, which spaCR's
    own Cellpose 4 loads, is a ``cellpose`` row pointing at its bioimage.io
    page. A Cellpose 3-format checkpoint is a ``cellpose3`` row that
    downloads the weights file itself, checked against the SHA-256 the
    manifest publishes, for the Cellpose 3 backend to run.

    Best effort and never raises: no network, a slow mirror or a changed
    schema all mean "no extra rows", never a zoo that fails to open. The
    response is cached so opening the dialog repeatedly is not repeatedly a
    network call.
    """
    import json as _json
    import time as _time
    import urllib.request

    cache = Path.home() / ".spacr" / "bioimageio_children.json"
    payload = None
    try:
        fresh = (cache.is_file() and
                 _time.time() - cache.stat().st_mtime
                 < BIOIMAGEIO_CACHE_HOURS * 3600)
        if fresh:
            payload = _json.loads(cache.read_text())
    except Exception:                                        # noqa: BLE001
        payload = None
    if payload is None and not allow_network:
        # Cache only. catalogue() must not touch the network -- it is called on
        # offline paths and from the GUI thread -- so the fetch happens in the
        # background warm-up and this reads what that left behind.
        return []
    if payload is None:
        try:
            with urllib.request.urlopen(url or BIOIMAGEIO_COLLECTION,
                                        timeout=timeout) as response:
                payload = _json.loads(response.read().decode("utf-8"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(_json.dumps(payload))
        except Exception:                                    # noqa: BLE001
            return []

    items = payload if isinstance(payload, list) else payload.get("items", [])
    out = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        manifest = item.get("manifest") or {}
        if not isinstance(manifest, Mapping):
            continue
        cellpose3 = _cellpose3_weights(manifest)
        if cellpose3 is None and not _looks_like_cellpose_sam(manifest):
            continue
        alias = str(item.get("alias") or "")
        if not alias:
            continue
        # Named after the model, not its bioimage.io alias: a row reading
        # "idealistic-eagle" tells the reader nothing.
        title = str(manifest.get("name") or alias)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or alias
        authors = ", ".join(
            str(a.get("name")) for a in (manifest.get("authors") or ())
            if isinstance(a, Mapping))
        if cellpose3 is not None:
            out.append(_cellpose3_download_entry(
                alias, slug, title, manifest, authors, *cellpose3))
            continue
        out.append(ModelEntry(
            key=slug, name=alias, path="", kind="cellpose",
            source="bioimage.io",
            uri=f"https://bioimage.io/#/artifacts/{alias}",
            sha256="", size_bytes=0,
            trained_on=f"{title} — {manifest.get('description') or ''}"[:300],
            trained_by=authors or "bioimage.io"))
    return out


#: The architectures a Cellpose 3-format bioimage.io model names.
#: ``CellPoseWrapper`` wraps a Cellpose 3 checkpoint and ``CPnetBioImageIO``
#: is Cellpose's own export of one; either way the ``pytorch_state_dict``
#: weights ARE the checkpoint, which Cellpose 3's ``CellposeModel`` loads.
_BIOIMAGEIO_CELLPOSE3_ARCHITECTURES = ("CellPoseWrapper", "CPnetBioImageIO")

#: Where bioimage.io serves an artifact's files.
_BIOIMAGEIO_FILES = ("https://hypha.aicell.io/bioimage-io/artifacts/{alias}/"
                     "files/{name}")

#: What a Cellpose 3-format row says about how to use it.
_CELLPOSE3_USE = (
    "runs through the Cellpose 3 backend: install that from this list, set "
    "segmentation_backend to cellpose3, and put this model's path in the "
    "object's model setting")


def _cellpose3_weights(manifest: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """The Cellpose 3 checkpoint a bioimage.io manifest publishes.

    Only the two architectures in :data:`_BIOIMAGEIO_CELLPOSE3_ARCHITECTURES`
    count -- the word "cellpose" in a tag does not, and a Cellpose-SAM model
    is the Cellpose 4 kind, listed as such.

    :returns: ``(source, sha256)`` of the weights file, or None.
    """
    if str(manifest.get("type") or "") != "model":
        return None
    if _looks_like_cellpose_sam(manifest):
        return None
    weights = manifest.get("weights")
    state = weights.get("pytorch_state_dict") if isinstance(weights, Mapping) else None
    if not isinstance(state, Mapping):
        return None
    architecture = state.get("architecture")
    called = (str(architecture.get("callable") or "")
              if isinstance(architecture, Mapping) else "")
    source = str(state.get("source") or "").strip()
    if called not in _BIOIMAGEIO_CELLPOSE3_ARCHITECTURES or not source:
        return None
    return source, str(state.get("sha256") or "").strip().lower()


def _cellpose3_download_entry(alias: str, slug: str, title: str,
                              manifest: Mapping[str, Any], authors: str,
                              source: str, sha256: str) -> "ModelEntry":
    """A bioimage.io Cellpose 3 checkpoint as a row that downloads it.

    The weights file is fetched from bioimage.io itself and checked against
    the SHA-256 its manifest publishes, like any other zoo download, and the
    licence the uploader chose travels with the row.
    """
    if source.startswith(("http://", "https://")):
        uri = source
    else:
        uri = _BIOIMAGEIO_FILES.format(alias=alias, name=source)
    suffix = Path(source).suffix if Path(source).suffix in (".pth", ".pt") else ".pth"
    return ModelEntry(
        key=f"bioimageio_{slug}", name=f"{slug}{suffix}", path="",
        kind="cellpose3", source="bioimage.io", uri=uri, sha256=sha256,
        trained_on=f"{title} — {manifest.get('description') or ''}"[:300],
        trained_by=authors or "bioimage.io",
        licence=str(manifest.get("license") or ""),
        notes=(f"bioimage.io model {alias}; {_CELLPOSE3_USE}",))


#: What each Cellpose 3 model is, for its zoo row.
_CELLPOSE3_NOTES = {
    "cyto3": "Cellpose 3's generalist whole-cell model.",
    "cyto2": "Cellpose 2's whole-cell model.",
    "cyto": "The original Cellpose whole-cell model.",
    "nuclei": "Cellpose's nucleus model.",
}


def _cellpose3_model_entries() -> List["ModelEntry"]:
    """The Cellpose 3 backend's own models, listed whether or not it is here.

    The name IS the path, as for the Cellpose 4 stock models: ``cyto3`` in
    an object's model setting, with segmentation_backend set to cellpose3,
    is what runs it. Until the backend is installed the row has no path and
    says what it needs.
    """
    from ._segmentation_backends import _CELLPOSE3, _SPECS, _backend_state

    spec = _SPECS[_CELLPOSE3]
    ready = _backend_state(_CELLPOSE3).ready
    out = []
    for model in spec.models:
        out.append(ModelEntry(
            key=f"cellpose3_{model}", name=model, kind="cellpose3",
            source="stock", path=model if ready else "",
            uri=f"backend:{_CELLPOSE3}", sha256="", size_bytes=0,
            trained_on=(f"{_CELLPOSE3_NOTES.get(model, '')} Runs through the "
                        f"Cellpose 3 backend.").strip(),
            trained_by="Cellpose", licence=spec.licence,
            notes=() if ready else (
                "needs the Cellpose 3 backend, which installs from this "
                "list into an environment of its own",)))
    return out


def _backend_for(entry: Any) -> str:
    """The optional segmentation backend a zoo row needs, or ``''``.

    A backend row names itself in its ``backend:<name>`` uri; every
    ``cellpose3`` model needs the Cellpose 3 backend.
    """
    uri = str(getattr(entry, "uri", "") or "")
    if uri.startswith("backend:"):
        return uri.split(":", 1)[1]
    if getattr(entry, "kind", "") == "cellpose3":
        return "cellpose3"
    return ""


#: ``name -> (label, install uri, import name, what it is)`` for every
#: optional segmentation backend. These are PACKAGES, not checkpoints: the zoo
#: lists them so a user learns they exist, and each installs into an
#: environment of its own, never into spaCR's.
INSTALLABLE_BACKENDS = {
    _name: (_spec.label, f"backend:{_name}", _spec.module, _spec.blurb)
    for _name, _spec in _BACKEND_SPECS.items() if _spec.segments
}


def installable_backend_entries() -> List["ModelEntry"]:
    """Every optional segmentation backend, in whatever state it is here.

    A backend absent from the zoo teaches nobody that it exists, so each one
    is listed, and its ``source`` says where it stands -- ``installed``,
    ``installable``, ``installing`` or ``not installable here`` -- with the
    reason as its first note and its licence on the row. Installing one
    builds it an environment of its own under ``~/.spacr/backends`` and
    leaves spaCR's own environment alone.
    """
    from ._segmentation_backends import _backend_state

    out = []
    for name, spec in _BACKEND_SPECS.items():
        state = _backend_state(name)
        out.append(ModelEntry(
            key=f"{name}_v1", name=spec.label,
            path=state.env if state.ready and not state.in_process else "",
            kind="backend", source=state.state, uri=f"backend:{name}",
            sha256="", size_bytes=0, trained_on=spec.blurb,
            trained_by=spec.label, licence=spec.licence,
            notes=tuple(note for note in (f"{state.state}: {state.reason}",
                                          spec.licence_note, spec.published)
                        if note)))
    return out


#: Where spaCR's Add button puts community submissions.
COMMUNITY_REPO = "einarolafsson/user-models"

#: Submissions live under this prefix until somebody promotes them.
COMMUNITY_PREFIX = "staging/"

#: The one-line warning that must travel with every community row.
COMMUNITY_WARNING = (
    "Community upload — NOT vetted. Anyone can submit through spaCR's Add "
    "button; nobody has checked what this file is, what it was trained on, or "
    "whether its reported scores are real.")


def community_entries(allow_network: bool = False,
                      repo: str = COMMUNITY_REPO) -> List["ModelEntry"]:
    """Unvetted models uploaded by spaCR users, or an empty list.

    These are shown only when the user asks for them, because an unreviewed
    checkpoint sitting beside a measured one invites the reader to treat them
    alike. Every row carries :data:`COMMUNITY_WARNING`, and the checksum comes
    from the uploader's own submission record -- it proves the file has not
    changed since it was uploaded, NOT that it is any good.

    Cache-first for the same reason as the bioimage.io listing: catalogue()
    must not reach the network.
    """
    import json as _json
    import time as _time

    cache = Path.home() / ".spacr" / "community_models.json"
    records = None
    try:
        # allow_network means the user just asked to see these, so the cache is
        # skipped: a cached empty list from before the first submission would
        # otherwise hide it for hours, which is exactly how this was found.
        if (not allow_network and cache.is_file()
                and _time.time() - cache.stat().st_mtime < 6 * 3600):
            records = _json.loads(cache.read_text())
    except Exception:                                        # noqa: BLE001
        records = None
    if records is None and not allow_network:
        return []
    if records is None:
        try:
            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi()
            files = [f for f in api.list_repo_files(repo)
                     if f.startswith(COMMUNITY_PREFIX)]
            folders = sorted({f.split("/")[1] for f in files if "/" in f[8:]})
            records = []
            for folder in folders:
                meta = {}
                sub = f"{COMMUNITY_PREFIX}{folder}/submission.json"
                if sub in files:
                    try:
                        meta = _json.loads(Path(hf_hub_download(repo, sub)).read_text())
                    except Exception:                        # noqa: BLE001
                        meta = {}
                # The checkpoint, not the README beside it: match on the
                # extensions a model actually has, or a folder with a README
                # listed first offers the README as the download.
                inside = [f for f in files
                          if f.startswith(f"{COMMUNITY_PREFIX}{folder}/")]
                weights = [f for f in inside
                           if f.endswith((".pth", ".pt", ".safetensors",
                                          ".CP_model"))]
                if not weights:
                    continue
                records.append(dict(folder=folder, path=weights[0], meta=meta))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(_json.dumps(records))
        except Exception:                                    # noqa: BLE001
            return []

    out = []
    for record in records:
        meta = record.get("meta") or {}
        folder = str(record.get("folder") or "")
        remote_path = str(record.get("path") or "")
        title = str(meta.get("name") or folder)
        out.append(ModelEntry(
            key=f"community_{folder}".replace("-", "_"),
            name=remote_path.rsplit("/", 1)[-1], path="",
            kind=str(meta.get("kind") or "cellpose"), source="community",
            uri=(f"https://huggingface.co/{repo}/resolve/main/{remote_path}"),
            sha256=str(meta.get("sha256") or ""),
            size_bytes=int(meta.get("size_bytes") or 0),
            trained_on=f"{title} — {meta.get('trained_on') or 'not stated'}. "
                       + COMMUNITY_WARNING,
            trained_by=str(meta.get("contact") or "a spaCR user")))
    return out


#: The five places a zoo row can come from, in the order they are offered.
#:
#: WHY FIVE HEADINGS RATHER THAN ONE LIST AND A BOOLEAN. The zoo used to be a
#: single list with "show unvetted community uploads" beside it. Ten models
#: trained here, four stock Cellpose-SAM backbones, bioimage.io's collection,
#: the Cellpose 3 backend's own four and a shared catalogue is more than a
#: list can carry, and a boolean that means "also show these" cannot fold away
#: the four groups a reader is not looking at.
#:
#: These names are IDENTIFIERS as well
#: as captions: :func:`source_of` returns one of them, and the picker
#: remembers which are on by this spelling. Renaming one is a migration.
ZOO_SOURCES: Tuple[str, ...] = (
    "cellposeSAM", "spaCR", "spaCR community", "bioimage.io", "cellpose3")

#: The headings a user who has never touched them sees turned on.
#:
#: The stock weights and the models trained here: the two a new user can act
#: on immediately. The other three are opt-in -- community uploads because
#: nobody has vetted them, bioimage.io and cellpose3 because they are long
#: lists of other people's models that would bury the ten this project ships.
DEFAULT_ZOO_SOURCES: Tuple[str, ...] = ("cellposeSAM", "spaCR")

#: ``ModelEntry.source`` values that mean "spaCR's own catalogue".
#:
#: ``remote`` is :data:`BUNDLED_REMOTE_MODELS` and any JSON catalogue named by
#: :data:`CATALOGUE_ENV_VAR`; ``bundled`` is what shipped in the package;
#: ``local`` is a checkpoint found on this machine or listed through the Add
#: button. All three are models this project offers, so all three are spaCR.
_SPACR_OWN_SOURCES = ("remote", "bundled", "local")

#: ``ModelEntry.source`` values that mean an unvetted upload.
#:
#: ``shared`` comes from :func:`shared_catalogue`, ``community`` from
#: :func:`community_entries`; the two are different transports for the same
#: thing, which is a file somebody uploaded that nobody has checked.
_COMMUNITY_SOURCES = ("shared", "community")


def source_of(entry: Any) -> str:
    """Which of :data:`ZOO_SOURCES` this row belongs under.

    Decided FROM THE ROW, never from a list of names kept somewhere else: a
    hand-written list goes stale the first time a model is added, and the
    failure it produces is a model that is in the catalogue and under no
    heading, which is a model nobody can see.

    The order of the tests is the rule, and it matters in one place: a
    Cellpose 3 checkpoint published on bioimage.io is a bioimage.io row, not
    a Cellpose 3 one. ``cellpose3`` means the backend's OWN models -- cyto,
    cyto2, cyto3, nuclei -- and the backend package that runs them.

    A row that matches nothing is filed under ``spaCR`` AND SAID OUT LOUD. It
    is the fallback rather than a sixth heading because a model under the
    wrong heading is a nuisance and a model under no heading is a bug the
    user experiences as a missing model.

    :param entry: any zoo row -- a :class:`ModelEntry`, or anything with
        ``kind``, ``source`` and ``uri``.
    :returns: one of :data:`ZOO_SOURCES`.
    """
    source = str(getattr(entry, "source", "") or "")
    kind = str(getattr(entry, "kind", "") or "")
    uri = str(getattr(entry, "uri", "") or "")
    if source in _COMMUNITY_SOURCES:
        return "spaCR community"
    if source == "bioimage.io" or "bioimage" in uri.lower():
        return "bioimage.io"
    if kind == "cellpose3" or _backend_for(entry) == "cellpose3":
        return "cellpose3"
    if kind == "cellpose" and source == "stock":
        return "cellposeSAM"
    if kind == "backend" or source in _SPACR_OWN_SOURCES:
        return "spaCR"
    LOG.warning(
        "model zoo: %r (kind=%r, source=%r) matches no source heading; "
        "listing it under spaCR so it stays visible",
        getattr(entry, "name", "") or getattr(entry, "key", ""), kind, source)
    return "spaCR"


def group_by_source(entries: Iterable[Any]) -> Dict[str, List[Any]]:
    """Split a listing into :data:`ZOO_SOURCES`, keeping each source's order.

    Every heading is present even when it has no rows, so a caller drawing
    the strip does not have to know which of the five happened to be empty
    this time.

    :param entries: the rows to split.
    :returns: heading -> rows, in :data:`ZOO_SOURCES` order.
    """
    out: Dict[str, List[Any]] = {name: [] for name in ZOO_SOURCES}
    for entry in entries:
        out[source_of(entry)].append(entry)
    return out


def entries_from_sources(entries: Iterable[Any],
                         sources: Iterable[str]) -> List[Any]:
    """The rows belonging to the headings that are on, in the given order.

    :param entries: the rows to filter.
    :param sources: the headings currently on.
    """
    wanted = set(sources)
    return [entry for entry in entries if source_of(entry) in wanted]


def catalogue(include_bundled: bool = True, remote: bool = True,
              catalogue_path: Any = None,
              include_plugins: bool = True,
              block: Optional[bool] = None) -> List[ModelEntry]:
    """Everything the zoo knows about without scanning the user's disks.

    That is: the models bundled with the installed package (whatever
    :func:`spacr.utils.download_models` has put in ``resources/models``), plus
    the declared remote entries — :data:`BUNDLED_REMOTE_MODELS` and, if one is
    configured, the JSON catalogue named by ``catalogue_path`` or the
    :data:`CATALOGUE_ENV_VAR` environment variable.

    Local apart from one thing, and the exception used to be undocumented:
    with ``remote`` on, this also asks :func:`shared_catalogue` for the
    community rows, which is a network call. It works offline either way --
    that fetch never raises -- and it never blocks Qt's GUI thread, which
    :func:`shared_catalogue` enforces for itself.

    :param include_bundled: list the models in the package resources folder.
    :param remote: list declared remote entries.
    :param catalogue_path: a JSON catalogue to add; defaults to
        ``$SPACR_MODEL_CATALOGUE`` when that names a file.
    :param include_plugins: include entries returned by installed spaCR model
        providers. Provider failures are recorded in plugin diagnostics and do
        not hide built-in entries.
    :param block: passed to :func:`shared_catalogue`. ``False`` takes the
        community rows from its cache rather than waiting for the network;
        ``None`` lets that function decide from the thread it is on.
    :returns: bundled entries first, then remote ones already present locally
        are dropped (a downloaded model is listed once, as the local file).
    """
    entries: List[ModelEntry] = []
    if include_bundled:
        root = package_model_root()
        if root.is_dir():
            entries.extend(discover_local(root, max_depth=2))

    entries.extend(stock_cellpose_entries())
    entries.extend(installable_backend_entries())
    entries.extend(_cellpose3_model_entries())
    if remote:
        entries.extend(bioimageio_entries())
    if remote:
        have = {(e.key, e.name) for e in entries}
        for record in BUNDLED_REMOTE_MODELS:
            entry = _entry_from_mapping(record)
            if (entry.key, entry.name) not in have:
                entries.append(entry)
                have.add((entry.key, entry.name))
        path = catalogue_path or os.environ.get(CATALOGUE_ENV_VAR, "")
        if path and os.path.isfile(str(path)):
            for entry in load_catalogue_file(path):
                if (entry.key, entry.name) not in have:
                    entries.append(entry)
                    have.add((entry.key, entry.name))
        for entry in shared_catalogue(block=block):
            if (entry.key, entry.name) not in have:
                entries.append(entry)
                have.add((entry.key, entry.name))
    entries = [e for e in entries if e.name not in RETIRED_MODEL_NAMES]
    if include_plugins:
        try:
            from .plugins import (
                load_object,
                model_providers,
                record_diagnostic,
            )
            have = {(entry.key, entry.name) for entry in entries}
            for plugin_name, contribution in model_providers():
                try:
                    provider = load_object(contribution.provider)
                    if not callable(provider):
                        raise TypeError(
                            f"{contribution.provider!r} is not callable"
                        )
                    produced = provider()
                    if isinstance(produced, (ModelEntry, Mapping)):
                        produced = (produced,)
                    for item in produced or ():
                        entry = (
                            item if isinstance(item, ModelEntry)
                            else _entry_from_mapping(item)
                        )
                        identity = (entry.key, entry.name)
                        if identity not in have:
                            entries.append(entry)
                            have.add(identity)
                except Exception as exc:
                    record_diagnostic(
                        plugin_name,
                        f"Model provider {contribution.key!r} failed",
                        exc,
                    )
        except Exception:
            LOG.exception("Could not initialise plugin model providers")
    return entries


def resolve(key_or_path: Any,
            entries: Optional[Sequence[ModelEntry]] = None) -> ModelEntry:
    """Turn a key, a name or a path into a :class:`ModelEntry`.

    A path that exists wins over a key: pointing the zoo at a file you just
    trained has to work without registering it anywhere first.

    :param key_or_path: an entry key, a model filename, or a path to a file.
    :param entries: the listing to search; defaults to :func:`catalogue`.
    :returns: the entry.
    :raises ModelUnreadable: when it looks like a path and no file is there.
    :raises ModelZooError: when no entry matches, listing the near misses.
    """
    text = str(key_or_path or "").strip()
    if not text:
        raise ModelZooError("no model given")
    if os.path.isfile(text):
        return entry_from_file(text)
    if os.sep in text or text.startswith("~"):
        raise ModelUnreadable(
            f"no such model file: {text} — and nothing in the zoo is called "
            f"that either")

    pool = list(entries) if entries is not None else catalogue()
    for entry in pool:
        if entry.key == text or entry.name == text:
            return entry
    lowered = text.lower()
    near = [e.key for e in pool if lowered in e.key.lower()
            or lowered in e.name.lower()]
    if len(near) == 1:
        return next(e for e in pool if e.key == near[0])
    raise ModelZooError(
        f"no model called {text!r} in the zoo"
        + (f" — did you mean one of: {', '.join(near[:5])}?" if near
           else f" ({len(pool)} entries known; pass a path to use a file "
                f"that is not registered)"))



def versioned_path(dest: Any, filename: str) -> Path:
    """The first free destination for ``filename`` in ``dest``.

    ``foo.CP_model`` -> ``foo.CP_model``, then ``foo_v2.CP_model``,
    ``foo_v3.CP_model``… An existing checkpoint is never overwritten: two
    models with the same filename are a normal thing to have (the same author
    retrained, or two people picked the same name), and the failure mode of
    overwriting — a run that used the old weights becoming unreproducible with
    no trace — is silent.

    An input that already carries ``_vN`` counts from there rather than
    becoming ``foo_v2_v2``.

    :param dest: destination directory.
    :param filename: the name to place there.
    :returns: a path that does not exist yet.
    """
    folder = Path(dest)
    p = Path(filename)
    suffix = p.suffix
    match = _VERSION_RE.match(p.stem)
    base = match.group("base") if match else p.stem
    n = int(match.group("n")) if match else 1

    candidate = folder / (f"{base}{suffix}" if n == 1
                          else f"{base}_v{n}{suffix}")
    while candidate.exists():
        n += 1
        candidate = folder / f"{base}_v{n}{suffix}"
    return candidate


def open_uri(uri: str, timeout: int = DEFAULT_TIMEOUT,
             chunk_size: int = DEFAULT_CHUNK
             ) -> Tuple[Iterable[bytes], int]:
    """Open a model URI for streaming. ``(chunks, total_bytes)``.

    ``http://`` and ``https://`` stream over ``requests`` — the same call
    :func:`spacr.utils.download_models` and
    :func:`spacr.qt.hf_download._download_one` make, imported here so this
    module has no hard dependency on it. ``file://`` and a plain existing path
    are read from disk, which is what a lab mirror on a NAS looks like and what
    the tests use, so the whole fetch path is exercised without a network.

    :param uri: where the model lives.
    :param timeout: seconds, HTTP only.
    :param chunk_size: bytes per chunk.
    :returns: ``(iterable of byte chunks, total size or 0 when unknown)``.
    :raises ModelZooError: for a scheme this does not speak.
    """
    text = str(uri or "")
    if text.startswith(("http://", "https://")):
        import requests

        response = requests.get(text, stream=True, timeout=timeout)
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        return response.iter_content(chunk_size=chunk_size), total

    if text.startswith("file://"):
        local = Path(text[len("file://"):])
    elif os.path.exists(text):
        local = Path(text)
    else:
        raise ModelZooError(
            f"do not know how to fetch {text!r} — expected an http(s):// URL, "
            f"a file:// URL, or a path that exists")
    if not local.is_file():
        raise ModelUnreadable(f"no such model file: {local}")
    return _read_chunks(local, chunk_size), local.stat().st_size


def _read_chunks(path: Path, chunk_size: int) -> Iterator[bytes]:
    """Read a file in fixed-size blocks.

    :param path: the file.
    :param chunk_size: the block size.
    :returns: an iterator over the blocks -- streamed rather than read
        whole, because these are multi-gigabyte checkpoints being hashed.
    """
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return
            yield block


def fetch(entry: ModelEntry, dest: Any,
          expected_sha256: Optional[str] = None,
          require_checksum: bool = True,
          opener: Optional[Callable[[str], Any]] = None,
          progress: Optional[Callable[[int, int], None]] = None,
          cancel: Optional[Callable[[], bool]] = None,
          chunk_size: int = DEFAULT_CHUNK,
          timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Download a model, verify it, and only then put it where it belongs.

    The order is the whole point:

    1. bytes stream into a temporary file **inside** ``dest``, so the rename in
       step 4 is a same-filesystem ``os.replace`` and therefore atomic;
    2. the checksum of what actually arrived is computed;
    3. if it does not match the published digest the temporary file is deleted
       and :class:`ChecksumMismatch` is raised — nothing is installed, and the
       destination still holds whatever it held before;
    4. only now is the temporary file renamed, to a
       :func:`versioned_path` that does not exist yet.

    Every failure — a dead server, a cancel, a bad hash, a full disk — leaves
    the destination directory exactly as it was. There is no window in which a
    half-written file sits at a name that looks like a model.

    :param entry: what to fetch. :attr:`ModelEntry.uri` is the source.
    :param dest: destination directory; created if missing.
    :param expected_sha256: digest to require, overriding
        :attr:`ModelEntry.sha256`.
    :param require_checksum: refuse to install when no digest is known. True by
        default — a download nobody can check is exactly the thing this module
        exists to stop being routine. Pass False to accept one knowingly; the
        entry :func:`install` returns then reports ``verified=False``.
    :param opener: ``fn(uri) -> chunks`` or ``fn(uri) -> (chunks, total)``;
        defaults to :func:`open_uri`.
    :param progress: ``fn(done_bytes, total_bytes)``; ``total`` is 0 when the
        server did not say.
    :param cancel: ``fn() -> bool``, polled between chunks. Returning True
        deletes the partial file and raises :class:`DownloadCancelled`.
    :param chunk_size: bytes per read.
    :param timeout: seconds, HTTP only.
    :returns: the path the model was written to.
    :raises ChecksumMismatch: the bytes are not the published bytes.
    :raises DownloadCancelled: ``cancel()`` returned True.
    :raises ModelZooError: no URI, or no checksum with ``require_checksum``.
    """
    if not entry.uri:
        raise ModelZooError(
            f"{entry.name} has no uri to fetch it from (source={entry.source})")
    want = (expected_sha256 if expected_sha256 is not None
            else entry.sha256) or ""
    want = want.strip().lower()
    if require_checksum and not want:
        raise ModelZooError(
            f"refusing to install {entry.name}: no sha256 was published for "
            f"it, so a truncated or substituted checkpoint could not be told "
            f"from the real one. Supply expected_sha256=…, put a hash in the "
            f"catalogue, or pass require_checksum=False to accept it "
            f"unverified.")

    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)

    import tempfile

    handle = tempfile.NamedTemporaryFile(
        dir=str(folder), prefix=f".{Path(entry.name).stem}.", suffix=".part",
        delete=False)
    temp = Path(handle.name)
    done = 0
    try:
        stream = (opener(entry.uri) if opener is not None
                  else open_uri(entry.uri, timeout=timeout,
                                chunk_size=chunk_size))
        chunks, total = stream if isinstance(stream, tuple) else (stream, 0)
        total = int(total or entry.size_bytes or 0)
        if progress is not None:
            progress(0, total)
        for block in chunks:
            if cancel is not None and cancel():
                raise DownloadCancelled(
                    f"download of {entry.name} cancelled after {done} byte(s) "
                    f"— nothing was written to {folder}")
            if not block:
                continue
            handle.write(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()

        if done == 0:
            raise ModelZooError(
                f"{entry.uri} returned no data for {entry.name} — nothing was "
                f"written to {folder}")

        got = sha256_file(temp)
        if want and got != want:
            raise ChecksumMismatch(
                f"{entry.name} does not match its published checksum and was "
                f"NOT installed in {folder}.\n"
                f"  expected sha256 {want}\n"
                f"  got      sha256 {got}\n"
                f"  ({done} bytes from {entry.uri})\n"
                f"A checkpoint that fails this still loads and still produces "
                f"masks — they are just not the masks the author's model "
                f"produces. Re-download, or get the right hash from whoever "
                f"published it.")
        return _claim(temp, folder, entry.name)
    except BaseException:
        try:
            handle.close()
        except Exception:
            pass
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass
        raise


def _claim(temp: Path, folder: Path, name: str) -> Path:
    """Move ``temp`` onto the first free version of ``name``, race-free.

    :func:`versioned_path` alone is check-then-act: two downloads of the same
    model finishing together both see ``foo.CP_model`` free, and the second
    ``os.replace`` silently destroys the first. So the name is *reserved* with
    ``O_CREAT | O_EXCL`` — which the kernel makes atomic — before the rename,
    and a lost race just tries the next version.

    The reservation is a zero-byte file that exists for microseconds and is
    then replaced. If the process is killed inside that window what remains is
    an empty file, which :func:`inspect_checkpoint` rejects by name — unlike a
    truncated checkpoint, which loads.
    """
    while True:
        target = versioned_path(folder, name)
        try:
            os.close(os.open(str(target),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
        except FileExistsError:
            continue
        os.replace(str(temp), str(target))
        return target


def install(entry: ModelEntry, dest: Any, **kwargs: Any) -> ModelEntry:
    """:func:`fetch` the model and return the registered local entry.

    The returned entry carries the digest of the bytes that were actually
    written — not the one the catalogue claimed — and
    :attr:`ModelEntry.verified` is True only when the two were compared and
    matched. Provenance from the catalogue entry is carried over, because that
    is the whole reason for having had a catalogue.

    :param entry: the remote entry.
    :param dest: destination directory.
    :param kwargs: passed to :func:`fetch`.
    :returns: a ``source='local'`` entry pointing at the new file.
    """
    expected = kwargs.get("expected_sha256")
    want = ((expected if expected is not None else entry.sha256) or "").strip()
    path = fetch(entry, dest, **kwargs)
    digest = sha256_file(path)
    notes = tuple(n for n in entry.notes if "no published checksum" not in n)
    if not want:
        notes = notes + (
            f"installed without a published checksum to check against; the "
            f"recorded sha256 {digest[:12]}… is of the bytes that arrived, "
            f"which proves nothing about where they came from",)
    return replace(
        entry,
        key=_key_for(path),
        name=path.name,
        source="local",
        path=str(path.resolve()),
        version=_version_of(path.name),
        sha256=digest,
        size_bytes=path.stat().st_size,
        verified=bool(want),
        notes=notes,
    )


def download_bundled_models(**kwargs: Any) -> str:
    """Pull the bundled Hugging Face model pack via the existing downloader.

    Thin, deliberate wrapper over :func:`spacr.utils.download_models` — the
    downloader spaCR already ships and the one
    :func:`spacr.submodules.analyze_plaques` depends on. It is *not*
    reimplemented here, so there is one code path that fills
    ``resources/models`` and one place to fix when the repo moves.

    It is also the unverified path: that function has no checksum, writes
    straight to the destination filename, and skips the whole pull when the
    folder is non-empty. Prefer a catalogue entry with a hash and
    :func:`install`; this exists so the zoo can offer the legacy pack rather
    than pretend it does not exist.

    ``spacr.utils`` imports torch, so it is imported here and not at module
    level. Nothing else in this module reaches for it.

    :param kwargs: forwarded to :func:`spacr.utils.download_models`.
    :returns: the local directory the pack landed in.
    """
    return _bulk_downloader()(**kwargs)


def _bulk_downloader() -> Callable[..., str]:
    """The legacy bulk downloader, imported late (it pulls torch in with it)."""
    from .utils import download_models

    return download_models



@dataclass
class FieldBenchmark:
    """One field's result for one model.

    :param field: the field name.
    :param n_objects: labels in the mask this model produced.
    :param severity: :mod:`spacr.seg_qc`'s verdict — ``'ok'``, ``'warn'`` or
        ``'fail'``, or ``'-'`` when QC was off.
    :param flags: the named defects seg_qc raised.
    :param note: seg_qc's verdict in prose, with its numbers in it.
    """

    field: str
    n_objects: int = 0
    severity: str = "-"
    flags: Tuple[str, ...] = ()
    note: str = ""


@dataclass
class BenchmarkResult:
    """One model over one field set. Only comparable to results on the *same* set.

    :param entry: the model that ran.
    :param fieldset: :func:`fieldset_id` of the images — a hash of the pixels,
        so two runs over the same three fields share it and two runs over
        different fields never do, whatever the folders were called.
    :param fieldset_label: the same thing for a human.
    :param rows: one :class:`FieldBenchmark` per field, in field order.
    :param seconds: wall-clock seconds the model spent segmenting.
    :param honoured: the parameters that reached the model.
    :param ignored: what was set and Cellpose 4 dropped (``diam_mean`` and
        friends) — carried so a benchmark cannot silently be a benchmark of
        settings nothing read.
    :param notes: warnings the reader must see before the numbers.
    :param masks: the label images, when kept, so a GUI can draw them.
    :param images: the source fields, likewise.
    :param object_type: what was segmented.
    """

    entry: ModelEntry
    fieldset: str = ""
    fieldset_label: str = ""
    rows: List[FieldBenchmark] = _dc_field(default_factory=list)
    seconds: float = 0.0
    honoured: Dict[str, Any] = _dc_field(default_factory=dict)
    ignored: Dict[str, Any] = _dc_field(default_factory=dict)
    notes: List[str] = _dc_field(default_factory=list)
    masks: List[Any] = _dc_field(default_factory=list)
    images: List[Any] = _dc_field(default_factory=list)
    object_type: str = "cell"

    @property
    def fields(self) -> List[str]:
        """Every field this model was benchmarked on.

        :returns: the field names.
        """
        return [r.field for r in self.rows]

    @property
    def n_fields(self) -> int:
        """How many fields were benchmarked.

        :returns: the field count.
        """
        return len(self.rows)

    @property
    def total_objects(self) -> int:
        """Every object the model found, across all fields.

        :returns: the object count.
        """
        return sum(r.n_objects for r in self.rows)

    @property
    def mean_objects(self) -> float:
        """Objects per field.

        NaN rather than zero when nothing was benchmarked: no fields is a
        different statement from a model that found nothing.

        :returns: the mean, or NaN.
        """
        return self.total_objects / self.n_fields if self.rows else float("nan")

    @property
    def n_failed(self) -> int:
        """Fields :mod:`spacr.seg_qc` scored ``'fail'``."""
        return sum(1 for r in self.rows if r.severity == "fail")

    @property
    def n_ok(self) -> int:
        """How many fields came back without a quality complaint.

        :returns: the count of fields at severity ``ok``.
        """
        return sum(1 for r in self.rows if r.severity == "ok")

    @property
    def qc_score(self) -> float:
        """Fraction of fields seg_qc scored ``'ok'``; ``nan`` without QC.

        A quality-control verdict on this model's own masks — it says the masks
        are not obviously broken, not that they are right. There is no ground
        truth in a benchmark (see the module docstring), so this is as close to
        a score as the zoo will produce.
        """
        scored = [r for r in self.rows if r.severity != "-"]
        if not scored:
            return float("nan")
        return sum(1 for r in scored if r.severity == "ok") / len(scored)

    @property
    def summary(self) -> str:
        """The model, what it found, over how many fields, and its QC score.

        :returns: a one-line summary.
        """
        score = self.qc_score
        return (f"{self.entry.name}: {self.total_objects} "
                f"{self.object_type}(s) over {self.n_fields} field(s) "
                f"({self.mean_objects:.1f}/field) in {self.seconds:.1f}s; "
                f"seg_qc scored {self.n_ok}/{self.n_fields} field(s) ok"
                + ("" if score != score else f" ({score * 100:.0f}%)")
                + f", {self.n_failed} fail.")


def fieldset_id(names: Sequence[str], images: Sequence[Any]) -> str:
    """A stable id for a set of fields, taken from the **pixels**.

    Folder names are not identity: ``plate1/1`` on two machines is two
    different sets of images, and the same three images copied to a new folder
    are the same benchmark input. So the id hashes each array's bytes, shape and
    dtype together with its name.

    This is what makes :func:`rank` able to refuse. Without it, two benchmarks
    run on different data are two numbers, and two numbers always sort.

    :param names: field names, in order.
    :param images: the arrays, in the same order.
    :returns: a 16-character hex id.
    """
    digest = hashlib.sha256()
    for name, image in zip(names, images):
        array = np.ascontiguousarray(np.asarray(image))
        digest.update(str(name).encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(hashlib.sha256(array.tobytes()).digest())
    return digest.hexdigest()[:16]


def _model_compare():
    """:mod:`spacr.model_compare`, imported late so browsing stays cheap."""
    from . import model_compare

    return model_compare


def config_for(entry: ModelEntry, overrides: Optional[Mapping[str, Any]] = None):
    """The :class:`spacr.model_compare.ModelConfig` that runs this entry.

    A local checkpoint is passed by path, which
    :func:`spacr.utils._choose_model` and
    :func:`spacr.model_compare.segment_with_cellpose` both load as
    ``pretrained_model``; anything else goes through by name and is subject to
    Cellpose 4's legacy-name remapping, which the config reports.

    :param entry: the model.
    :param overrides: eval settings (``diameter``, ``flow_threshold``, …).
    :returns: the config.
    """
    mc = _model_compare()
    settings: Dict[str, Any] = dict(overrides or {})
    settings["model"] = entry.path or entry.name
    settings.setdefault("name", entry.name)
    return mc.ModelConfig.from_mapping(settings)


def benchmark(entry: ModelEntry, images: Optional[Sequence[Any]] = None,
              source: Any = None, n_fields: int = DEFAULT_N_FIELDS,
              field_names: Optional[Sequence[str]] = None,
              segment_fn: Optional[Callable] = None,
              settings: Optional[Mapping[str, Any]] = None,
              object_type: str = "cell", qc: bool = True,
              keep_images: bool = True, channel: Optional[int] = None,
              progress: Optional[Callable[[str, int, int], None]] = None,
              ) -> BenchmarkResult:
    """Run one model over N fields and report what came out. "Test on 3 fields".

    This is :mod:`spacr.model_compare`'s harness with one model instead of two:
    the same :func:`~spacr.model_compare.load_fields` reader, the same
    :class:`~spacr.model_compare.ModelConfig` (so the same arguments are
    honoured and the same ones reported as ignored), the same
    :func:`~spacr.model_compare.segment_with_cellpose` backend, and the same
    :mod:`spacr.seg_qc` scorecards. To put two models side by side use
    :func:`compare_entries`, which calls
    :func:`~spacr.model_compare.compare_models` proper.

    The checkpoint is checked before it is loaded, so a missing or corrupt file
    fails with its own name in the message rather than a torch ``KeyError``.

    :param entry: the model to run.
    :param images: fields already in memory; None loads them from ``source``.
    :param source: a folder of fields (``.tif`` / ``.png`` / ``.npy`` /
        ``.npz``), read by :func:`spacr.model_compare.load_fields`.
    :param n_fields: how many fields to take from ``source``.
    :param field_names: names for the rows.
    :param segment_fn: ``fn(images, config) -> masks``; defaults to
        :func:`spacr.model_compare.segment_with_cellpose`. This is the seam the
        GUI and the tests use, and the reason no test here loads Cellpose.
    :param settings: eval overrides (``diameter``, ``flow_threshold``, …).
    :param object_type: what is being segmented, for the seg_qc scorecards.
    :param qc: score the masks with :mod:`spacr.seg_qc`.
    :param keep_images: keep images and masks on the result for a GUI to draw.
    :param channel: index into the last axis for multi-channel fields.
    :param progress: ``fn(message, done, total)``.
    :returns: a :class:`BenchmarkResult`.
    :raises ModelUnreadable: when the checkpoint is missing or not a checkpoint.
    :raises ValueError: when there is no field, or the model returned the wrong
        number of masks.
    """
    mc = _model_compare()

    if images is None:
        if source is None:
            raise ValueError(
                "benchmark needs either images= or source= (a folder of fields)")
        field_names, images = mc.load_fields(source, n_fields=n_fields,
                                             channel=channel)
    fields = [np.asarray(image) for image in images]
    if not fields:
        raise ValueError("no field to benchmark: pass at least one image")
    names = ([str(n) for n in field_names] if field_names is not None
             else [f"field_{i:04d}" for i in range(len(fields))])
    if len(names) != len(fields):
        raise ValueError(
            f"got {len(names)} field name(s) for {len(fields)} field(s)")

    config = config_for(entry, settings)
    notes = list(entry.notes) + list(config.notes())
    if not entry.provenance_known:
        notes.append(
            f"{entry.name} does not record what it was trained on, so a good "
            f"score here says it works on these fields and nothing more.")

    if entry.path:
        inspect_checkpoint(entry.path)

    total_steps = 2

    def _tick(message: str, done: int) -> None:
        """Report one benchmark milestone through the captured callback.

        :param message: stage description for the progress display.
        :param done: completed-step index from zero through two.
        :returns: None. When a callback was supplied it receives the message,
            completed index, and captured total of two; otherwise this is a
            no-op.
        """
        if progress is not None:
            progress(message, done, total_steps)

    _tick(f"Segmenting {len(fields)} field(s) with {entry.name}…", 0)
    run = segment_fn if segment_fn is not None else _default_segmenter(
        [entry], mc)
    started = time.perf_counter()
    produced = list(run(fields, config))
    seconds = time.perf_counter() - started
    if len(produced) != len(fields):
        raise ValueError(
            f"{entry.name} returned {len(produced)} mask(s) for "
            f"{len(fields)} field(s)")
    masks = [mc._as_labels(m) for m in produced]

    _tick("Scoring masks…", 1)
    scores = mc._score(masks, names, object_type) if qc else [None] * len(fields)

    rows = [
        FieldBenchmark(
            field=names[i],
            n_objects=int(np.unique(masks[i]).size - (1 if (masks[i] == 0).any()
                                                      else 0)),
            severity=scores[i].severity if scores[i] else "-",
            flags=tuple(scores[i].flags) if scores[i] else (),
            note=scores[i].note if scores[i] else "",
        )
        for i in range(len(fields))
    ]
    _tick("Done", 2)

    return BenchmarkResult(
        entry=entry,
        fieldset=fieldset_id(names, fields),
        fieldset_label=_fieldset_label(names, source),
        rows=rows,
        seconds=seconds,
        honoured=config.honoured_parameters(),
        ignored=config.ignored_parameters(),
        notes=notes,
        masks=masks if keep_images else [],
        images=fields if keep_images else [],
        object_type=object_type,
    )


def _cellpose3_segment(model: str, images: Sequence[Any],
                       config: Any) -> List[np.ndarray]:
    """Segment ``images`` with a Cellpose 3 model, in the Cellpose 3 backend.

    spaCR's Cellpose 4 loads a Cellpose 3 checkpoint without complaint and
    then segments nonsense with it, so a ``cellpose3`` row is never handed
    to it: the benchmark and the A/B comparison run it where Mask generation
    does, in the Cellpose 3 environment, with the config's eval settings.

    :param model: a Cellpose 3 model name or checkpoint path.
    :param images: the fields.
    :param config: the :class:`spacr.model_compare.ModelConfig` for the side.
    :returns: one integer label image per field.
    :raises ImportError: when the Cellpose 3 backend is not installed.
    """
    from ._segmentation_backends import _load_backend

    backend = _load_backend("cellpose3", model_name=model)
    masks, _flows, _styles = backend.eval(
        [np.asarray(image, dtype=np.float32) for image in images],
        **config.eval_kwargs())
    return [np.asarray(mask).astype(np.int32) for mask in masks]


def _default_segmenter(entries: Sequence[ModelEntry], mc: Any) -> Callable:
    """``fn(images, config) -> masks`` that runs each entry where it belongs.

    A ``cellpose3`` entry goes to the Cellpose 3 backend, told apart by the
    model its config names; everything else to spaCR's own Cellpose,
    exactly as before.
    """
    cellpose3 = {str(e.path or e.name) for e in entries
                 if e.kind == "cellpose3"}

    def _segment(images, config):
        """Segment ``images`` with the backend the config's model belongs to."""
        model = str(getattr(config, "model", "") or "")
        if model in cellpose3:
            return _cellpose3_segment(model, images, config)
        return mc.segment_with_cellpose(images, config)

    return _segment


def _fieldset_label(names: Sequence[str], source: Any) -> str:
    """"3 field(s) from …: a, b, c" — what a group header says."""
    where = f" from {os.fspath(source)}" if isinstance(
        source, (str, os.PathLike)) else ""
    listed = ", ".join(str(n) for n in list(names)[:4])
    if len(names) > 4:
        listed += f", … (+{len(names) - 4})"
    return f"{len(names)} field(s){where}: {listed}"


def compare_entries(entry_a: ModelEntry, entry_b: ModelEntry,
                    images: Optional[Sequence[Any]] = None,
                    source: Any = None, n_fields: int = DEFAULT_N_FIELDS,
                    field_names: Optional[Sequence[str]] = None,
                    settings_a: Optional[Mapping[str, Any]] = None,
                    settings_b: Optional[Mapping[str, Any]] = None,
                    **kwargs: Any):
    """Put two zoo entries head to head on the same fields.

    Straight delegation to :func:`spacr.model_compare.compare_models` — the
    metrics, the split/merge attribution and the "neither model is ground
    truth" wording all come from there, unchanged. This function's only job is
    turning two :class:`ModelEntry` objects into two
    :class:`~spacr.model_compare.ModelConfig` objects.

    :param entry_a: the A side.
    :param entry_b: the B side.
    :param images: fields already in memory; None loads them from ``source``.
    :param source: a folder of fields.
    :param n_fields: how many fields to take from ``source``.
    :param field_names: names for the rows.
    :param settings_a: eval overrides for A.
    :param settings_b: eval overrides for B.
    :param kwargs: forwarded to :func:`spacr.model_compare.compare_models`.
    :returns: a :class:`spacr.model_compare.ComparisonReport`.
    """
    mc = _model_compare()
    if images is None:
        if source is None:
            raise ValueError(
                "compare_entries needs either images= or source=")
        field_names, images = mc.load_fields(source, n_fields=n_fields)
    for entry in (entry_a, entry_b):
        if entry.path:
            inspect_checkpoint(entry.path)
    kwargs.setdefault("segment_fn", _default_segmenter([entry_a, entry_b], mc))
    return mc.compare_models(
        images,
        config_for(entry_a, settings_a),
        config_for(entry_b, settings_b),
        field_names=field_names,
        **kwargs)



def group_by_fieldset(results: Sequence[BenchmarkResult]
                      ) -> Dict[str, List[BenchmarkResult]]:
    """Bucket benchmarks by the field set they ran on, first-seen order.

    :param results: benchmarks.
    :returns: ``{fieldset_id: [results]}``.
    """
    groups: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault(result.fieldset, []).append(result)
    return groups


def _rank_value(result: BenchmarkResult, key: str) -> Tuple:
    """Build the sort key for one benchmark result.

    ``nan`` sorts LAST rather than first: a model nobody scored is not the
    best model.

    :param result: the benchmark result.
    :param key: what to rank by.
    :returns: the sort key.
    :raises ValueError: for an unknown key. There is deliberately no
        accuracy key -- a benchmark here has no ground truth, so a column
        sorting models by "score" would be inventing one.
    """
    if key == "qc":
        score = result.qc_score
        return (-(score if score == score else -1.0), result.seconds,
                result.entry.name)
    if key == "seconds":
        return (result.seconds, result.entry.name)
    raise ValueError(
        f"rank key must be one of {tuple(RANK_KEYS)}, got {key!r}. There is "
        f"deliberately no accuracy key: a benchmark here has no ground truth, "
        f"so a column that sorted models by 'score' would be inventing one.")


def rank(results: Sequence[BenchmarkResult],
         key: str = DEFAULT_RANK_KEY) -> List[BenchmarkResult]:
    """Order benchmarks best-first — **within one field set only**.

    A model's numbers on your three fields say nothing about its numbers on
    somebody else's: different cell density, different exposure, different
    magnification. Sorting results from two field sets into one list produces a
    ranking that looks exactly like a real one and means nothing, which is the
    failure this function exists to prevent. So it refuses.

    :param results: benchmarks, all from the same field set.
    :param key: one of :data:`RANK_KEYS`.
    :returns: the results, best first.
    :raises IncomparableBenchmarks: when the results span more than one field
        set. Use :func:`rank_groups` or :func:`format_benchmarks`, which group
        and label instead.
    :raises ValueError: on an unknown ``key``.
    """
    groups = group_by_fieldset(results)
    if len(groups) > 1:
        detail = "; ".join(
            f"{members[0].fieldset_label} "
            f"[{', '.join(m.entry.name for m in members)}]"
            for members in groups.values())
        raise IncomparableBenchmarks(
            f"refusing to rank {len(results)} benchmark(s) that ran on "
            f"{len(groups)} different field sets — a score on one set of "
            f"fields says nothing about a score on another, so sorting them "
            f"together would invent a ranking. The sets were: {detail}. Use "
            f"rank_groups() or format_benchmarks() to rank within each set.")
    return sorted(results, key=lambda r: _rank_value(r, key))


def rank_groups(results: Sequence[BenchmarkResult],
                key: str = DEFAULT_RANK_KEY
                ) -> Dict[str, List[BenchmarkResult]]:
    """Rank inside each field set, keeping the sets apart. The safe alternative.

    :param results: benchmarks from any number of field sets.
    :param key: one of :data:`RANK_KEYS`.
    :returns: ``{fieldset_id: [results, best first]}``.
    """
    return {fieldset: sorted(members, key=lambda r: _rank_value(r, key))
            for fieldset, members in group_by_fieldset(results).items()}



def _render_table(rows: Sequence[Sequence[str]],
                  header: Sequence[str]) -> List[str]:
    """A fixed-width text table.

    :func:`spacr.model_compare._render_table`, reused so the zoo's console
    output is the same shape as the comparison's rather than a second table
    style two lines apart in the same terminal.
    """
    return _model_compare()._render_table(rows, header)


_ZOO_COLUMNS = (
    ("model", lambda e: e.name),
    ("kind", lambda e: e.kind),
    ("source", lambda e: e.source),
    ("v", lambda e: e.version),
    ("size", lambda e: _human_bytes(e.size_bytes)),
    ("checksum", lambda e: e.checksum_state),
    ("trained on", lambda e: _shorten(e.trained_on, 46)),
    ("trained by", lambda e: _shorten(e.trained_by, 22)),
)


SCORECARD_ROWS = (
    ("train", "n_train"), ("train obj.", "train_objects"),
    ("test", "n_test"), ("test obj.", "test_objects"), ("CV", "cv"),
    ("F1 @ IoU 0.5", "f1"), ("AJI", "aji"), ("Dice", "dice"),
    ("final train loss", "train_loss"), ("final val loss", "val_loss"),
    ("best epoch", "best_epoch"),
)


def scorecard_html(entry) -> str:
    """The model's scorecard as an HTML table, for a tooltip.

    A paragraph of prose is what a tooltip used to show, and a reader
    comparing two models had to parse two paragraphs to find two numbers.
    The same table the model card prints answers that at a glance. Falls back
    to the prose when an entry publishes no metrics, because an empty table is
    worse than a sentence.

    Metrics that hold none of the scorecard's keys -- a free-form note, a
    training loss under a name of its own -- are not a scorecard either, and
    return nothing too: a table of eleven "not recorded" rows would replace a
    two-line note that said something.

    :param entry: a catalogue entry, or anything with ``metrics`` and a name.
    :returns: the HTML table, or ``""`` when there is no scorecard to show.
    """
    metrics = dict(getattr(entry, "metrics", None) or {})
    if not any(key in metrics for _label, key in SCORECARD_ROWS):
        return ""
    name = (getattr(entry, "display_name", "") or getattr(entry, "name", "")
            or getattr(entry, "key", ""))
    def cell(value):
        """A scorecard value as shown, or "not recorded" when blank."""
        return value if str(value).strip() else "not recorded"
    rows = []
    for label, key in SCORECARD_ROWS:
        stock = metrics.get(f"stock_{key}", "") if key in ("f1", "aji", "dice") else ""
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td align='right'><b>{cell(metrics.get(key, ''))}</b></td>"
            f"<td align='right'>{cell(stock) if stock or key in ('f1','aji','dice') else ''}</td></tr>")
    trained = getattr(entry, "trained_on", "") or ""
    card = getattr(entry, "model_card_url", "")
    return (f"<p><b>{name}</b></p>"
            "<table cellspacing='0' cellpadding='3'>"
            "<tr><th align='left'></th><th align='right'>this model</th>"
            "<th align='right'>stock</th></tr>"
            + "".join(rows) + "</table>"
            + (f"<p>{trained[:200]}</p>" if trained else "")
            + (f"<p>{card}</p>" if card else ""))


def format_zoo(entries: Sequence[ModelEntry]) -> str:
    """Render a listing a human reads before choosing a model.

    Provenance is a column, not a footnote: "trained on" is the field that
    decides whether a model is applicable to your images at all, and it is
    printed for every row — reading ``unknown`` where it is unknown, because a
    blank there would read as "no constraints".

    :param entries: what to list.
    :returns: a multi-line string.
    """
    entries = list(entries)
    if not entries:
        return ("Model zoo: nothing found.\n"
                "  Scan a folder with discover_local(), or point "
                "$SPACR_MODEL_CATALOGUE at a catalogue file.")

    lines = [f"Model zoo — {len(entries)} model(s)"]
    rows = [[str(fmt(e)) for _, fmt in _ZOO_COLUMNS] for e in entries]
    lines.extend(_render_table(rows, [name for name, _ in _ZOO_COLUMNS]))

    unknown = [e for e in entries if not e.provenance_known]
    if unknown:
        lines.append("")
        lines.append(
            f"  {len(unknown)} model(s) do not record what they were trained "
            f"on: {', '.join(e.name for e in unknown[:6])}"
            f"{'…' if len(unknown) > 6 else ''}. A Cellpose model fine-tuned "
            f"on confluent 60x cells is not interchangeable with one trained "
            f"on sparse 20x ones, and nothing here tells you which this is.")

    unchecked = [e for e in entries if e.checksum_state == "none" and e.path]
    if unchecked:
        lines.append(
            f"  {len(unchecked)} local model(s) have no checksum on record; "
            f"run verify() against a published digest before trusting one.")

    notes = [(e.name, n) for e in entries for n in e.notes]
    if notes:
        lines.append("")
        for name, note in notes:
            lines.append(f"  ! {name}: {note}")
    return "\n".join(lines)


_BENCH_COLUMNS = (
    ("field", lambda r: r.field),
    ("objects", lambda r: str(r.n_objects)),
    ("seg_qc", lambda r: r.severity),
    ("flags", lambda r: ", ".join(r.flags) if r.flags else "-"),
)


def format_benchmarks(results: Sequence[BenchmarkResult],
                      key: str = DEFAULT_RANK_KEY) -> str:
    """Render benchmarks **grouped by field set**, ranked only within a group.

    Two models benchmarked on different fields appear under two headers with a
    line saying the two blocks cannot be compared. That is the alternative to
    :func:`rank`'s refusal, and it is the only way this module will ever put
    incomparable numbers on the same page.

    :param results: benchmarks, from any number of field sets.
    :param key: one of :data:`RANK_KEYS`.
    :returns: a multi-line string.
    """
    results = list(results)
    if not results:
        return "No benchmark to show."

    groups = rank_groups(results, key=key)
    lines: List[str] = [
        f"Model benchmarks — {len(results)} run(s) over "
        f"{len(groups)} field set(s), ranked by {key} "
        f"({RANK_KEYS[key] if key in RANK_KEYS else ''})"
    ]
    if len(groups) > 1:
        lines.append(
            "  The blocks below ran on DIFFERENT fields and are not comparable "
            "with each other: a score on one set of images says nothing about "
            "a score on another. Compare within a block only.")

    for fieldset, members in groups.items():
        lines.append("")
        lines.append(f"  field set {fieldset} — {members[0].fieldset_label}")
        for rank_index, result in enumerate(members, start=1):
            lines.append("")
            lines.append(f"    {rank_index}. {result.summary}")
            entry = result.entry
            lines.append(f"       trained on: {entry.trained_on}")
            diameter = result.honoured.get("diameter")
            lines.append(
                f"       model {result.honoured.get('model', entry.name)}, "
                f"diameter {diameter if diameter is not None else 'native'}")
            if result.ignored:
                lines.append(
                    "       set but ignored by Cellpose 4: "
                    + ", ".join(f"{k}={v!r}" for k, v in result.ignored.items()))
            rows = [[str(fmt(r)) for _, fmt in _BENCH_COLUMNS]
                    for r in result.rows]
            lines.extend("    " + line for line in
                         _render_table(rows, [n for n, _ in _BENCH_COLUMNS]))
            for note in result.notes:
                lines.append(f"       ! {note}")

    lines.append("")
    lines.append("  seg_qc is a quality-control verdict on each model's own "
                 "masks — it catches a model that collapsed on your data. It "
                 "is not an accuracy: there is no ground truth here. To put "
                 "two models against each other, use compare_entries().")
    return "\n".join(lines)
