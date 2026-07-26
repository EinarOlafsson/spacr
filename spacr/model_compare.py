"""Run two segmentation models over the same fields and say what changed.

Why this exists
---------------
Choosing a segmentation model in spaCR is currently done by running a whole
plate twice and squinting at the montages. That is hours per candidate, and the
comparison it produces is "these look about the same" — which is exactly the
answer you get whether the models differ or not.

This module makes the comparison small and quantitative: three fields, two
models, one table. It is deliberately split in two halves so the Model Zoo
(TO-DO #29, "test a model on 3 fields") can reuse the whole thing:

* the **metric layer** (:func:`compare_masks` and everything under it) takes
  label arrays and nothing else. No Cellpose, no torch, no GUI, no file paths.
* the **orchestration layer** (:func:`compare_models`) runs a segmentation
  callable twice and folds the per-field results into a
  :class:`ComparisonReport`. The callable is an argument, so the zoo can hand
  it a different backend, and a test can hand it a stub.

Neither half imports torch or cellpose at module level; only
:func:`segment_with_cellpose` does, and only when it is called.

Neither model is ground truth
-----------------------------
This is an A/B comparison, not an evaluation. There is no correct mask here, so
nothing in this module calls one model right and the other wrong: every count
is reported **directionally** ("B found 12 objects A did not"), never as
precision or recall, and the wording of :func:`format_comparison` follows the
same rule. The only symmetric summary numbers are the ones that genuinely are
symmetric — the ARI and the matched fraction.

The metrics, and why each one is computed the way it is
------------------------------------------------------
``ari`` — **background is excluded, and that is the whole trick.**
    The Adjusted Rand Index over two raw label images is nearly useless: a
    1400x1400 field is ~95 % background, both models agree it is background,
    and those agreed pairs swamp everything else. Two models that disagree
    about every single object still score about 0.999 (``tests/
    test_model_compare.py`` asserts exactly that number on a field where the
    foreground agreement is 0.0).

    So the index is computed over the **union foreground** — the pixels at
    least one model assigned to an object — with pixels the *other* model left
    unassigned treated as unclustered singletons rather than as one big
    background cluster. That second half matters as much as the first: with
    background as a single cluster, a model that misses an entire object still
    scores 1.0, because "one object plus a background blob" and "one object
    plus a background blob" are the same partition. As singletons, missing
    half the objects scores 0.50, which is what it deserves.

    Computed in closed form from the object overlap matrix (see
    :func:`adjusted_rand_index`) rather than by materialising millions of
    singleton labels; ``tests/test_model_compare.py`` checks it against
    ``sklearn.metrics.adjusted_rand_score`` on the expanded arrays.

    ARI is a *pixel-pair* index, so it is sensitive to boundaries and it is
    degenerate when a field holds one object (a single cluster has no pair
    structure to agree about). It is reported next to the object-level numbers
    for that reason, never alone.

``iou_matched_fraction`` and ``mean_matched_iou`` — from an **optimal**
    assignment, not a greedy one. Matching objects between two segmentations is
    a bipartite assignment problem; picking each object's best partner
    double-assigns, and picking greedily in descending IoU can leave a pair
    stranded that an optimal assignment would have kept. Both failures are
    reproduced in the tests. :func:`match_objects` thresholds the IoU matrix and
    then runs :func:`scipy.optimize.linear_sum_assignment` over it, which is
    also what ``cellpose.metrics`` does.

    Above an IoU of 0.5 the assignment is provably unique — two objects cannot
    both overlap a third by more than half of it — so at the default threshold
    greedy would in fact get the same answer. The optimal assignment is used
    anyway because ``iou_threshold`` is a knob, and every value below 0.5 (which
    is where "roughly the same object" lives) makes greedy wrong.

``split_events`` / ``merge_events`` — **fragmentation is not discovery.**
    "Model B found 20 more objects" means completely different things if they
    are 20 new cells or 20 fragments of cells A already found, and fragmentation
    is the common Cellpose failure. So the object-count delta is decomposed:

    * a B object is a **fragment** of an A object when the majority of it (see
      ``containment``) lies inside that A object *and* it was not assigned to
      some other A object. An A object with two or more such fragments is one
      ``split_event``, and it explains ``k - 1`` of B's extra objects.
    * the mirror image gives ``merge_events`` at B objects that swallow two or
      more A objects, explaining ``k - 1`` of A's objects going missing.
    * whatever is left over — ``new_objects_b`` and ``missing_objects_a`` —
      is the genuine difference in what the two models detected.

    The "not assigned elsewhere" clause is what keeps the attribution honest: a
    B object that straddles two A objects but is *paired* with one of them is
    that object's counterpart, not a fragment of its neighbour.

``qc_a`` / ``qc_b``
    Each field's masks are additionally run through :mod:`spacr.seg_qc`, so the
    table can say *which* of the two disagreeing masks looks broken on its own
    terms (fused, shattered, empty, all on the border). Those thresholds are
    argued in that module and are not duplicated here.

Degenerate fields, and what they are defined to be
--------------------------------------------------
* **both masks empty** — ``ari = 1.0``, ``iou_matched_fraction = 1.0``,
  ``mean_matched_iou = nan``. Two models that both say "there is nothing in
  this field" have made the same statement, and that is agreement; the
  alternative (``nan``) would drop the field out of every aggregate, so a
  channel that is legitimately empty would silently shrink the sample instead
  of showing up as the unanimous verdict it is. It is counted separately in
  :attr:`ComparisonReport.n_both_empty` so it can never be mistaken for
  agreement about objects, and ``mean_matched_iou`` stays ``nan`` because there
  is no matched pair to take an IoU of.
* **one mask empty** — ``ari = 0.0`` and no matches; falls out of the
  definitions with no special case, and is the right answer: the models agree
  about nothing.
* **one object each** — matched normally, but the ARI is degenerate (a single
  cluster carries no pair information) and can even be negative for two masks
  that overlap well. The object-level numbers carry the field in that case.
* **completely disjoint labels** — ARI near zero, nothing matched, every object
  reported as new/missing.

What Cellpose 4 accepts and then ignores
----------------------------------------
A comparison that differs only in an argument the model never sees reports "no
difference" and wastes the run. On the installed Cellpose 4, these are accepted
and then dropped (see :data:`IGNORED_ARGUMENTS`): ``model_type``, ``diam_mean``
and ``nchan`` at construction, ``channels`` and ``rescale`` at ``eval``, plus
spaCR's own ``restore_type``. Every pre-SAM model name resolves to ``cpsam``
too, so "cyto3 versus nuclei" is one model against itself.

``diameter`` is *not* in that list: ``eval`` still honours it by rescaling the
image by ``30 / diameter`` before inference. It is the one size knob that does
anything, which is why :func:`format_comparison` prints it first.

:func:`compare_models` therefore records both what reached the model and what
was dropped on the floor, and raises a loud warning when the two configurations
differ *only* in arguments nothing will read.

Public API
----------
``ModelConfig``            one model's settings, plus what of it survives.
``SegComparison``          one field, two masks: every number above.
``ComparisonReport``       the whole run: configs, per-field rows, aggregates.
``compare_masks``          the pure metric entry point (label arrays in).
``compare_models``         run two models over the same fields.
``adjusted_rand_index``    background-excluded ARI on its own.
``match_objects``          the optimal object assignment on its own.
``segment_with_cellpose``  the default segmentation backend.
``load_fields``            pull N fields out of a folder for a comparison.
``format_comparison``      the printable report.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field as _dc_field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .seg_qc import FieldQC, _as_labels

__all__ = [
    "ComparisonReport",
    "DEFAULT_CONTAINMENT",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_N_FIELDS",
    "IGNORED_ARGUMENTS",
    "LEGACY_MODEL_NAMES",
    "ModelConfig",
    "SegComparison",
    "adjusted_rand_index",
    "compare_configs",
    "compare_masks",
    "compare_models",
    "format_comparison",
    "load_fields",
    "match_objects",
    "object_overlap",
    "segment_with_cellpose",
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: IoU at which two objects are called the same object. 0.5 is the COCO /
#: Cellpose convention and the point above which the assignment is unique.
DEFAULT_IOU_THRESHOLD = 0.5

#: Fraction of an object that must lie inside another for it to count as a
#: fragment of it. 0.5 means "the majority of it", which is what makes the
#: attribution unambiguous: a fragment can belong to at most one parent.
DEFAULT_CONTAINMENT = 0.5

#: Fields compared by default — the number a human will actually look at.
DEFAULT_N_FIELDS = 3

#: Pre-SAM Cellpose model names. Cellpose 4 ships only ``cpsam`` and resolves
#: anything else to it with a log line, so two configurations naming two
#: different legacy models are the same model twice. Mirrors
#: :data:`spacr.utils.LEGACY_CELLPOSE_MODELS` (``tests/test_model_compare.py``
#: asserts the two stay in step); duplicated rather than imported because
#: ``spacr.utils`` pulls in torch, and this module must not.
LEGACY_MODEL_NAMES: Tuple[str, ...] = (
    'cyto', 'cyto2', 'cyto3', 'cyto_2', 'cyto_3',
    'nuclei', 'nucleus', 'toxo_pv_lumen', 'toxo_cyto',
)

#: The model that every name above resolves to.
DEFAULT_MODEL = 'cpsam'

#: Arguments Cellpose 4 accepts and then does not use, with the reason. A
#: comparison whose two sides differ only in these is a comparison of a model
#: with itself, so :func:`compare_configs` refuses to let that pass quietly.
IGNORED_ARGUMENTS: Dict[str, str] = {
    'model_type': (
        "CellposeModel(model_type=...) logs 'not used in v4.0.1+' and drops it; "
        "Cellpose 4 has one architecture."
    ),
    'diam_mean': (
        "CellposeModel(diam_mean=...) logs 'not used in v4.0.1+' and drops it. "
        "Use diameter= at eval time — that one still rescales the image."
    ),
    'nchan': "CellposeModel(nchan=...) is deprecated in v4.0.1+ and dropped.",
    'channels': (
        "eval(channels=...) is deprecated in v4.0.1+; the first three channels "
        "of the image are used whatever you pass. Select channels before "
        "handing the image over."
    ),
    'rescale': (
        "eval(rescale=...) is deprecated in v4.0.1+; scaling is driven by "
        "diameter alone."
    ),
    'net_avg': "removed in Cellpose 3; there is one network to average.",
    'restore_type': (
        "spaCR's denoise/deblur/upsample restore models are pre-SAM checkpoints "
        "that Cellpose 4 no longer ships; spacr.utils._choose_model prints this "
        "and ignores it."
    ),
}

#: ``CellposeModel.eval`` keyword arguments that do change the masks. Anything a
#: caller puts in :attr:`ModelConfig.extra` that is neither here nor in
#: :data:`IGNORED_ARGUMENTS` is passed through and flagged as unrecognised.
HONOURED_EVAL_ARGUMENTS: Tuple[str, ...] = (
    'batch_size', 'resample', 'channel_axis', 'z_axis', 'normalize', 'invert',
    'diameter', 'flow_threshold', 'cellprob_threshold', 'do_3D', 'anisotropy',
    'flow3D_smooth', 'stitch_threshold', 'min_size', 'max_size_fraction',
    'niter', 'augment', 'tile_overlap', 'bsize',
)


# ---------------------------------------------------------------------------
# model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """One side of the comparison: which model, run how.

    Only the fields below and the honoured keys of :attr:`extra` reach
    ``CellposeModel.eval``. Anything in :attr:`extra` that Cellpose 4 ignores is
    kept, reported and *not* passed on, so it shows up in the report as the
    no-op it is instead of quietly making two runs look identical.

    :param name: what to call this side in the report. Defaults to the model.
    :param model: ``'cpsam'``, a legacy Cellpose name (resolved to ``cpsam``),
        or a path to a custom checkpoint.
    :param diameter: expected object diameter in pixels, or None to let
        Cellpose run at native scale. This is the one size argument Cellpose 4
        still acts on — it resizes the image by ``30 / diameter``.
    :param flow_threshold: flow-error cutoff.
    :param cellprob_threshold: mask-probability cutoff.
    :param normalize: per-image percentile normalisation inside Cellpose.
    :param invert: invert the image before inference.
    :param resample: run the dynamics at full resolution.
    :param min_size: objects smaller than this are dropped by Cellpose.
    :param niter: dynamics iterations, or None for Cellpose's default.
    :param augment: 8-way test-time augmentation.
    :param batch_size: tiles per forward pass — speed only, not results.
    :param extra: any other ``eval`` keyword. Honoured keys are forwarded;
        ignored ones are reported.
    """

    name: str = ""
    model: str = DEFAULT_MODEL
    diameter: Optional[float] = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    normalize: bool = True
    invert: bool = False
    resample: bool = True
    min_size: int = 15
    niter: Optional[int] = None
    augment: bool = False
    batch_size: int = 8
    extra: Dict[str, Any] = _dc_field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = os.path.basename(str(self.model)) or str(self.model)

    @classmethod
    def from_mapping(cls, source: Any) -> "ModelConfig":
        """Build a config from a dict (or pass a :class:`ModelConfig` through).

        Keys that are not fields of this class land in :attr:`extra`, which is
        how an ignored argument such as ``diam_mean`` survives long enough to be
        reported instead of silently doing nothing.

        :param source: a mapping or an existing :class:`ModelConfig`.
        :returns: a :class:`ModelConfig`.
        """
        if isinstance(source, cls):
            return source
        if not isinstance(source, Mapping):
            raise TypeError(
                f"model configuration must be a ModelConfig or a mapping, "
                f"got {type(source).__name__}"
            )
        known = {f for f in cls.__dataclass_fields__ if f != 'extra'}
        fields = {k: v for k, v in source.items() if k in known}
        extra = dict(source.get('extra') or {})
        extra.update({k: v for k, v in source.items()
                      if k not in known and k != 'extra'})
        return cls(extra=extra, **fields)

    @property
    def resolved_model(self) -> str:
        """The checkpoint Cellpose will actually load.

        Every pre-SAM name maps to ``cpsam``; a path is left alone so a custom
        checkpoint is loaded as asked.
        """
        name = str(self.model or DEFAULT_MODEL)
        if name in LEGACY_MODEL_NAMES:
            return DEFAULT_MODEL
        return name

    @property
    def model_was_remapped(self) -> bool:
        """True when the requested model is not the model that will be loaded."""
        return str(self.model or DEFAULT_MODEL) != self.resolved_model

    def honoured_parameters(self) -> Dict[str, Any]:
        """Everything that reaches the model, resolved model included.

        This is what the report displays. If two configurations produce the same
        dict here, they will produce the same masks, and the comparison has
        nothing to show.
        """
        out: Dict[str, Any] = {
            'model': self.resolved_model,
            'diameter': self.diameter,
            'flow_threshold': self.flow_threshold,
            'cellprob_threshold': self.cellprob_threshold,
            'normalize': self.normalize,
            'invert': self.invert,
            'resample': self.resample,
            'min_size': self.min_size,
            'niter': self.niter,
            'augment': self.augment,
            'batch_size': self.batch_size,
        }
        for key, value in self.extra.items():
            if key not in IGNORED_ARGUMENTS:
                out[key] = value
        return out

    def eval_kwargs(self) -> Dict[str, Any]:
        """The keyword arguments to hand ``CellposeModel.eval``.

        :attr:`honoured_parameters` minus ``model``, which is a constructor
        argument rather than an ``eval`` one.
        """
        kwargs = self.honoured_parameters()
        kwargs.pop('model', None)
        return kwargs

    def ignored_parameters(self) -> Dict[str, Any]:
        """What was set and will not be read, ``{name: value}``.

        The requested model appears here as ``model`` when it was remapped:
        asking for ``cyto3`` and getting ``cpsam`` is the same class of surprise
        as setting ``diam_mean``.
        """
        out: Dict[str, Any] = {}
        if self.model_was_remapped:
            out['model'] = self.model
        for key, value in self.extra.items():
            if key in IGNORED_ARGUMENTS:
                out[key] = value
        return out

    def notes(self) -> List[str]:
        """One line per argument of this config that will not be read."""
        lines: List[str] = []
        if self.model_was_remapped:
            lines.append(
                f"{self.name}: model {self.model!r} predates Cellpose-SAM and "
                f"resolves to {DEFAULT_MODEL!r}."
            )
        for key, value in self.extra.items():
            if key in IGNORED_ARGUMENTS:
                lines.append(f"{self.name}: {key}={value!r} is ignored — "
                             f"{IGNORED_ARGUMENTS[key]}")
            elif key not in HONOURED_EVAL_ARGUMENTS:
                lines.append(
                    f"{self.name}: {key}={value!r} is not a Cellpose 4 eval "
                    f"argument; it is passed through untouched and may raise."
                )
        return lines


def compare_configs(config_a: ModelConfig,
                    config_b: ModelConfig) -> Dict[str, Any]:
    """Diff two configurations into what matters and what cannot matter.

    :param config_a: the A side.
    :param config_b: the B side.
    :returns: ``{'honoured': {key: (a, b)}, 'ignored': {key: (a, b)},
        'identical': bool, 'warnings': [str]}``. ``identical`` is True when
        every argument that reaches the model is the same on both sides — the
        case where the run cannot show a difference and the report has to say
        so before anybody reads a number off it.
    """
    ha, hb = config_a.honoured_parameters(), config_b.honoured_parameters()
    ia, ib = config_a.ignored_parameters(), config_b.ignored_parameters()

    honoured_diff = {
        key: (ha.get(key), hb.get(key))
        for key in sorted(set(ha) | set(hb))
        if ha.get(key) != hb.get(key)
    }
    ignored_diff = {
        key: (ia.get(key), ib.get(key))
        for key in sorted(set(ia) | set(ib))
        if ia.get(key) != ib.get(key)
    }

    warnings: List[str] = list(config_a.notes()) + list(config_b.notes())
    if not honoured_diff:
        if ignored_diff:
            warnings.insert(0, (
                f"{config_a.name} and {config_b.name} differ only in arguments "
                f"Cellpose 4 ignores ({', '.join(ignored_diff)}) — they are the "
                f"same model with the same settings, so any difference below is "
                f"run-to-run noise, not a model difference."
            ))
        else:
            warnings.insert(0, (
                f"{config_a.name} and {config_b.name} resolve to identical "
                f"settings; this run compares a model with itself."
            ))
    return {
        'honoured': honoured_diff,
        'ignored': ignored_diff,
        'identical': not honoured_diff,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class SegComparison:
    """One field, two masks — every number the comparison produces.

    Directional by construction: ``*_a`` describes the first mask, ``*_b`` the
    second, and neither is treated as the truth. ``unmatched_a`` is "objects A
    found that B did not pair with", not "false negatives".

    :param field: the field's name.
    :param n_objects_a: labels in mask A.
    :param n_objects_b: labels in mask B.
    :param ari: background-excluded Adjusted Rand Index (see
        :func:`adjusted_rand_index`). 1.0 for identical masks, ~0 for unrelated
        ones, 1.0 for two empty masks, ``nan`` only when the union foreground
        holds fewer than two pixels.
    :param iou_matched_fraction: ``2 * matched / (n_a + n_b)`` — the symmetric
        share of objects that have a partner. Symmetric on purpose: it says how
        much of the two segmentations correspond without calling either right.
    :param mean_matched_iou: mean IoU over matched pairs; ``nan`` with none.
    :param unmatched_a: A objects with no partner.
    :param unmatched_b: B objects with no partner.
    :param split_events: A objects that B broke into two or more pieces.
    :param merge_events: B objects that swallowed two or more A objects.
    :param n_matched: pairs in the optimal assignment above the threshold.
    :param fragments_from_splits: extra B objects explained by ``split_events``.
    :param merged_away: A objects that disappeared into ``merge_events``.
    :param new_objects_b: B objects that are neither matched nor fragments —
        the genuinely new detections.
    :param missing_objects_a: A objects neither matched nor merged away.
    :param iou_threshold: the threshold the matching used.
    :param matches: ``[(label_a, label_b, iou), ...]``, for drawing.
    :param qc_a: :class:`spacr.seg_qc.FieldQC` for mask A, when computed.
    :param qc_b: the same for mask B.
    :param note: the field's verdict in prose, with its numbers in it.
    """

    field: str = "field"
    n_objects_a: int = 0
    n_objects_b: int = 0
    ari: float = float("nan")
    iou_matched_fraction: float = float("nan")
    mean_matched_iou: float = float("nan")
    unmatched_a: int = 0
    unmatched_b: int = 0
    split_events: int = 0
    merge_events: int = 0
    n_matched: int = 0
    fragments_from_splits: int = 0
    merged_away: int = 0
    new_objects_b: int = 0
    missing_objects_a: int = 0
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    matches: List[Tuple[int, int, float]] = _dc_field(default_factory=list)
    qc_a: Optional[FieldQC] = None
    qc_b: Optional[FieldQC] = None
    note: str = ""

    @property
    def object_count_delta(self) -> int:
        """``n_objects_b - n_objects_a``. Positive means B found more."""
        return self.n_objects_b - self.n_objects_a

    @property
    def both_empty(self) -> bool:
        """True when neither model found anything in this field."""
        return self.n_objects_a == 0 and self.n_objects_b == 0

    def __str__(self) -> str:
        return (f"{self.field}: A {self.n_objects_a} vs B {self.n_objects_b} "
                f"objects ({self.object_count_delta:+d}), ARI {self.ari:.3f}")


@dataclass
class ComparisonReport:
    """Two models over a set of fields: the configs, the rows, the aggregate.

    :param model_a: the A configuration.
    :param model_b: the B configuration.
    :param comparisons: one :class:`SegComparison` per field, in field order.
    :param config_diff: what :func:`compare_configs` found.
    :param warnings: the lines a reader must see before the numbers — ignored
        arguments, remapped model names, an A/B that cannot differ.
    :param seconds_a: wall-clock seconds model A spent segmenting.
    :param seconds_b: the same for B.
    :param masks_a: A's label images, kept so a GUI can draw them.
    :param masks_b: B's label images.
    :param images: the source images, kept for the same reason.
    :param object_type: what was segmented, for the seg_qc scorecards.
    """

    model_a: ModelConfig = _dc_field(default_factory=ModelConfig)
    model_b: ModelConfig = _dc_field(default_factory=ModelConfig)
    comparisons: List[SegComparison] = _dc_field(default_factory=list)
    config_diff: Dict[str, Any] = _dc_field(default_factory=dict)
    warnings: List[str] = _dc_field(default_factory=list)
    seconds_a: float = 0.0
    seconds_b: float = 0.0
    masks_a: List[np.ndarray] = _dc_field(default_factory=list)
    masks_b: List[np.ndarray] = _dc_field(default_factory=list)
    images: List[np.ndarray] = _dc_field(default_factory=list)
    object_type: str = "object"

    # -- aggregates --------------------------------------------------------

    @property
    def fields(self) -> List[str]:
        """The field names, in order."""
        return [c.field for c in self.comparisons]

    @property
    def n_fields(self) -> int:
        return len(self.comparisons)

    @property
    def total_objects_a(self) -> int:
        return sum(c.n_objects_a for c in self.comparisons)

    @property
    def total_objects_b(self) -> int:
        return sum(c.n_objects_b for c in self.comparisons)

    @property
    def object_count_delta(self) -> int:
        """``total_objects_b - total_objects_a``, directional."""
        return self.total_objects_b - self.total_objects_a

    @property
    def count_ratio(self) -> float:
        """B's object count as a multiple of A's; ``nan`` when A found none."""
        if not self.total_objects_a:
            return float("nan")
        return self.total_objects_b / self.total_objects_a

    @property
    def mean_ari(self) -> float:
        """Mean ARI over the fields that have one; ``nan`` when none do."""
        return _nanmean([c.ari for c in self.comparisons])

    @property
    def mean_matched_iou(self) -> float:
        """Mean of the per-field mean matched IoU."""
        return _nanmean([c.mean_matched_iou for c in self.comparisons])

    @property
    def mean_matched_fraction(self) -> float:
        return _nanmean([c.iou_matched_fraction for c in self.comparisons])

    @property
    def total_splits(self) -> int:
        return sum(c.split_events for c in self.comparisons)

    @property
    def total_merges(self) -> int:
        return sum(c.merge_events for c in self.comparisons)

    @property
    def total_fragments(self) -> int:
        """Extra B objects that are pieces of A objects, not new detections."""
        return sum(c.fragments_from_splits for c in self.comparisons)

    @property
    def total_merged_away(self) -> int:
        return sum(c.merged_away for c in self.comparisons)

    @property
    def total_new_objects_b(self) -> int:
        return sum(c.new_objects_b for c in self.comparisons)

    @property
    def total_missing_objects_a(self) -> int:
        return sum(c.missing_objects_a for c in self.comparisons)

    @property
    def n_both_empty(self) -> int:
        """Fields where neither model found anything — trivial agreement."""
        return sum(1 for c in self.comparisons if c.both_empty)

    @property
    def identical_masks(self) -> bool:
        """True when every field's masks agree object-for-object and pixel-pair."""
        if not self.comparisons:
            return False
        return all(c.ari >= 1.0 - 1e-9 and c.object_count_delta == 0
                   and c.unmatched_a == 0 and c.unmatched_b == 0
                   for c in self.comparisons)

    @property
    def summary(self) -> str:
        """One directional sentence about the whole run."""
        if not self.comparisons:
            return "No field was compared."
        a, b = self.model_a.name, self.model_b.name
        delta = self.object_count_delta
        head = (f"{b} found {abs(delta)} "
                f"{'more' if delta >= 0 else 'fewer'} {self.object_type}(s) "
                f"than {a} over {self.n_fields} field(s) "
                f"({self.total_objects_a} vs {self.total_objects_b})")
        if delta and (self.total_fragments or self.total_merged_away):
            head += (f", of which {self.total_fragments} are fragments of "
                     f"{a}'s objects and {self.total_merged_away} are "
                     f"{a} objects {b} fused")
        return (f"{head}. Mean ARI {_fmt(self.mean_ari)}, "
                f"{_fmt(self.mean_matched_fraction, pct=True)} of objects "
                f"matched at IoU>={_threshold_of(self.comparisons):g}. "
                f"Neither model is ground truth.")


def _threshold_of(comparisons: Sequence[SegComparison]) -> float:
    return comparisons[0].iou_threshold if comparisons else DEFAULT_IOU_THRESHOLD


def _nanmean(values: Sequence[float]) -> float:
    """Mean of the finite entries, ``nan`` when there are none."""
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else float("nan")


def _fmt(value: Any, pct: bool = False) -> str:
    """Format a metric, ``'-'`` when it does not exist."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:.3f}"


# ---------------------------------------------------------------------------
# the metric layer: label arrays in, numbers out
# ---------------------------------------------------------------------------

def object_overlap(mask_a: Any, mask_b: Any) -> Dict[str, Any]:
    """Contingency between the objects of two label images.

    Both masks are relabelled to ``0..n`` first, so arbitrary label values (and
    gaps in them) are handled, and the pixel-pair counting below can index
    straight into the table.

    :param mask_a: a 2-D label image.
    :param mask_b: a second label image of the same shape.
    :returns: ``{'labels_a', 'labels_b', 'areas_a', 'areas_b', 'overlap',
        'n_pixels', 'union_foreground'}``. ``overlap`` is the
        ``n_a x n_b`` object-by-object intersection count with background
        already dropped; ``labels_a`` maps row ``i`` back to the original label.
    :raises ValueError: when the two masks are not the same shape.
    """
    a = _as_labels(mask_a)
    b = _as_labels(mask_b)
    if a.shape != b.shape:
        raise ValueError(
            f"masks must cover the same field: A is {a.shape}, B is {b.shape}"
        )

    values_a, index_a = np.unique(a.ravel(), return_inverse=True)
    values_b, index_b = np.unique(b.ravel(), return_inverse=True)
    n_rows, n_cols = values_a.size, values_b.size
    table = np.bincount(index_a * n_cols + index_b,
                        minlength=n_rows * n_cols).reshape(n_rows, n_cols)

    # Drop the background row/column, keeping the marginals of what is left.
    row0 = 1 if (n_rows and values_a[0] == 0) else 0
    col0 = 1 if (n_cols and values_b[0] == 0) else 0
    labels_a = values_a[row0:]
    labels_b = values_b[col0:]
    areas_a = table[row0:, :].sum(axis=1)
    areas_b = table[:, col0:].sum(axis=0)
    overlap = table[row0:, col0:]

    return {
        'labels_a': labels_a,
        'labels_b': labels_b,
        'areas_a': areas_a,
        'areas_b': areas_b,
        'overlap': overlap,
        'n_pixels': int(a.size),
        'union_foreground': int(areas_a.sum() + areas_b.sum()
                                - overlap.sum()),
    }


def adjusted_rand_index(mask_a: Any, mask_b: Any) -> float:
    """Adjusted Rand Index over the **union foreground**, background excluded.

    The index is taken over the pixels at least one mask assigned to an object.
    Pixels the other mask left unassigned stay in the sample but belong to no
    cluster (each is its own singleton), so a model that misses an object is
    penalised for it — with background as one shared cluster it would not be.
    Pixels neither mask claimed are dropped entirely: they are the agreement
    that would otherwise be the whole answer.

    Computed in closed form from the object overlap table. Singleton clusters
    contribute nothing to any ``C(n, 2)`` term, so the expansion never has to be
    materialised; the only thing they change is the total pair count, which is
    taken over the union foreground.

    :param mask_a: a 2-D label image.
    :param mask_b: a second label image of the same shape.
    :returns: 1.0 for identical partitions, ~0 for unrelated ones, negative when
        the two masks agree less than chance. 1.0 when both masks are empty (see
        the module docstring for why), ``nan`` when the union foreground holds
        fewer than two pixels and there is no pair to score.
    """
    parts = object_overlap(mask_a, mask_b)
    n = parts['union_foreground']
    if not parts['areas_a'].size and not parts['areas_b'].size:
        # Both masks empty: the two models made the same statement.
        return 1.0
    if n < 2:
        return float("nan")

    comb2 = lambda x: x * (x - 1.0) / 2.0     # noqa: E731 - C(x, 2), vectorised
    index = float(comb2(parts['overlap'].astype(np.float64)).sum())
    sum_a = float(comb2(parts['areas_a'].astype(np.float64)).sum())
    sum_b = float(comb2(parts['areas_b'].astype(np.float64)).sum())
    total = comb2(float(n))
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        # Both partitions are structureless (every object one pixel): there is
        # nothing to agree or disagree about, so this is agreement.
        return 1.0
    return (index - expected) / (maximum - expected)


def iou_matrix(parts: Mapping[str, Any]) -> np.ndarray:
    """Object-by-object IoU from an :func:`object_overlap` table.

    :param parts: the dict :func:`object_overlap` returned.
    :returns: an ``n_a x n_b`` float array; empty when either mask has no object.
    """
    overlap = parts['overlap']
    if overlap.size == 0:
        return np.zeros(overlap.shape, dtype=np.float64)
    areas_a = parts['areas_a'].astype(np.float64)[:, None]
    areas_b = parts['areas_b'].astype(np.float64)[None, :]
    inter = overlap.astype(np.float64)
    union = areas_a + areas_b - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def match_objects(mask_a: Any, mask_b: Any,
                  iou_threshold: float = DEFAULT_IOU_THRESHOLD
                  ) -> Dict[str, Any]:
    """Pair the objects of two masks by **optimal** assignment.

    Bipartite, not greedy. The IoU matrix is thresholded first (everything below
    ``iou_threshold`` becomes 0) and :func:`scipy.optimize.linear_sum_assignment`
    then maximises the total IoU over what is left, which is the same procedure
    ``cellpose.metrics._true_positive`` uses. Taking each object's best partner
    double-assigns; taking pairs greedily in descending IoU can strand a pair
    that the optimal assignment keeps. Both are exercised in the tests.

    :param mask_a: a 2-D label image.
    :param mask_b: a second label image of the same shape.
    :param iou_threshold: minimum IoU for a pair to count as the same object.
    :returns: ``{'matches': [(label_a, label_b, iou), ...], 'iou': matrix,
        'parts': the overlap table, 'unmatched_a': [labels], 'unmatched_b':
        [labels]}``, matches sorted by descending IoU.
    """
    from scipy.optimize import linear_sum_assignment

    parts = object_overlap(mask_a, mask_b)
    iou = iou_matrix(parts)
    labels_a, labels_b = parts['labels_a'], parts['labels_b']

    matches: List[Tuple[int, int, float]] = []
    if iou.size:
        thresholded = np.where(iou >= float(iou_threshold), iou, 0.0)
        rows, cols = linear_sum_assignment(-thresholded)
        for r, c in zip(rows, cols):
            if thresholded[r, c] > 0:
                matches.append((int(labels_a[r]), int(labels_b[c]),
                                float(iou[r, c])))
    matches.sort(key=lambda m: -m[2])

    taken_a = {m[0] for m in matches}
    taken_b = {m[1] for m in matches}
    return {
        'matches': matches,
        'iou': iou,
        'parts': parts,
        'unmatched_a': [int(l) for l in labels_a if int(l) not in taken_a],
        'unmatched_b': [int(l) for l in labels_b if int(l) not in taken_b],
    }


def _split_merge(parts: Mapping[str, Any],
                 matches: Sequence[Tuple[int, int, float]],
                 containment: float) -> Dict[str, Any]:
    """Attribute B's extra objects to fragmentation and A's to fusion.

    A B object is a *fragment* of an A object when at least ``containment`` of
    its area lies inside it and it was not assigned to a different A object.
    Two or more fragments make that A object one split event, and the ``k``
    pieces stand in for ``k - 1`` extra B objects. Merges are the mirror image.

    The "not assigned elsewhere" clause is what stops a B object that straddles
    two A objects from being counted as a fragment of the neighbour it is not
    paired with — without it, an ordinary boundary shift reads as fragmentation.

    Sets rather than counts, because the counts alone get the residual wrong:
    when a split is fine enough that no piece reaches the IoU threshold, the
    parent has no partner and would otherwise be reported as an object B lost
    *and* one of its own pieces as an object B invented. Returning who was
    involved lets :func:`compare_masks` take them out of both residuals.

    :param parts: the :func:`object_overlap` table.
    :param matches: the optimal assignment.
    :param containment: fraction of an object that must lie inside another.
    :returns: ``{'split_events', 'fragments_from_splits', 'merge_events',
        'merged_away', 'split_parents', 'fragment_children', 'merge_parents',
        'merged_children'}`` — the four sets hold original label values.
    """
    overlap = parts['overlap'].astype(np.float64)
    out: Dict[str, Any] = {
        'split_events': 0, 'fragments_from_splits': 0,
        'merge_events': 0, 'merged_away': 0,
        'split_parents': set(), 'fragment_children': set(),
        'merge_parents': set(), 'merged_children': set(),
    }
    if overlap.size == 0:
        return out

    labels_a = [int(l) for l in parts['labels_a']]
    labels_b = [int(l) for l in parts['labels_b']]
    areas_a = parts['areas_a'].astype(np.float64)[:, None]
    areas_b = parts['areas_b'].astype(np.float64)[None, :]
    # in_a[i, j]: the share of B object j that lies inside A object i.
    with np.errstate(divide='ignore', invalid='ignore'):
        in_a = np.where(areas_b > 0, overlap / areas_b, 0.0)
        in_b = np.where(areas_a > 0, overlap / areas_a, 0.0)

    partner_of_b = {m[1]: m[0] for m in matches}
    partner_of_a = {m[0]: m[1] for m in matches}
    threshold = float(containment)

    for i, label_a in enumerate(labels_a):
        pieces = [labels_b[j] for j in np.flatnonzero(in_a[i] >= threshold)
                  if partner_of_b.get(labels_b[j], label_a) == label_a]
        if len(pieces) >= 2:
            out['split_events'] += 1
            out['fragments_from_splits'] += len(pieces) - 1
            out['split_parents'].add(label_a)
            out['fragment_children'].update(pieces)

    for j, label_b in enumerate(labels_b):
        pieces = [labels_a[i] for i in np.flatnonzero(in_b[:, j] >= threshold)
                  if partner_of_a.get(labels_a[i], label_b) == label_b]
        if len(pieces) >= 2:
            out['merge_events'] += 1
            out['merged_away'] += len(pieces) - 1
            out['merge_parents'].add(label_b)
            out['merged_children'].update(pieces)

    return out


def compare_masks(mask_a: Any, mask_b: Any,
                  field: str = "field",
                  iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                  containment: float = DEFAULT_CONTAINMENT,
                  qc_a: Optional[FieldQC] = None,
                  qc_b: Optional[FieldQC] = None) -> SegComparison:
    """Compare two label images of the same field. Pure: arrays in, numbers out.

    Nothing about Cellpose, models, files or the GUI reaches this function, so
    it works just as well on masks from any other source — which is the point:
    the Model Zoo reuses it unchanged.

    The comparison is **directional but not judgemental**. ``*_a`` and ``*_b``
    describe the two masks; neither is the reference, so there is no precision,
    no recall, and no "correct" column. See the module docstring for how the
    ARI excludes background, why the object assignment is optimal rather than
    greedy, and how splits and merges are told apart from genuine differences.

    :param mask_a: a 2-D label image (bool and float masks are coerced).
    :param mask_b: a second label image of the same shape.
    :param field: the field's name, carried into the row.
    :param iou_threshold: minimum IoU for two objects to be the same object.
    :param containment: fraction of an object that must lie inside another for
        it to count as a fragment of it.
    :param qc_a: an optional :class:`spacr.seg_qc.FieldQC` for mask A, attached
        so the row can say which mask looks broken on its own terms.
    :param qc_b: the same for mask B.
    :returns: a :class:`SegComparison`.
    :raises ValueError: when the masks are not the same shape, or are not label
        images.
    """
    matched = match_objects(mask_a, mask_b, iou_threshold=iou_threshold)
    parts = matched['parts']
    matches = matched['matches']
    n_a = int(parts['labels_a'].size)
    n_b = int(parts['labels_b'].size)
    n_matched = len(matches)

    events = _split_merge(parts, matches, containment)
    unmatched_a = set(matched['unmatched_a'])
    unmatched_b = set(matched['unmatched_b'])
    # An object left over by the assignment is only a genuine difference when
    # no split or merge already accounts for it: the pieces of a shattered A
    # object are not new detections, and the A object they came from is not a
    # missed one.
    explained_a = events['split_parents'] | events['merged_children']
    explained_b = events['fragment_children'] | events['merge_parents']

    total = n_a + n_b
    if total == 0:
        matched_fraction = 1.0          # both empty: trivially in agreement
    else:
        matched_fraction = 2.0 * n_matched / total

    row = SegComparison(
        field=field,
        n_objects_a=n_a,
        n_objects_b=n_b,
        ari=adjusted_rand_index(mask_a, mask_b),
        iou_matched_fraction=matched_fraction,
        mean_matched_iou=(float(np.mean([m[2] for m in matches]))
                          if matches else float("nan")),
        unmatched_a=len(unmatched_a),
        unmatched_b=len(unmatched_b),
        split_events=events['split_events'],
        merge_events=events['merge_events'],
        n_matched=n_matched,
        fragments_from_splits=events['fragments_from_splits'],
        merged_away=events['merged_away'],
        new_objects_b=len(unmatched_b - explained_b),
        missing_objects_a=len(unmatched_a - explained_a),
        iou_threshold=float(iou_threshold),
        matches=matches,
        qc_a=qc_a,
        qc_b=qc_b,
    )
    row.note = _compose_note(row)
    return row


def _compose_note(row: SegComparison) -> str:
    """The field's verdict in prose, always carrying its numbers."""
    if row.both_empty:
        return ("neither mask holds an object; the two models agree there is "
                "nothing in this field, which is agreement about the field and "
                "not about any object")
    bits = [
        f"A {row.n_objects_a} vs B {row.n_objects_b} object(s) "
        f"({row.object_count_delta:+d}), {row.n_matched} matched at "
        f"IoU>={row.iou_threshold:g} (mean {_fmt(row.mean_matched_iou)}), "
        f"ARI {_fmt(row.ari)}"
    ]
    if row.split_events:
        bits.append(f"B split {row.split_events} of A's object(s) into "
                    f"{row.split_events + row.fragments_from_splits} pieces")
    if row.merge_events:
        bits.append(f"B fused {row.merge_events + row.merged_away} of A's "
                    f"objects into {row.merge_events}")
    if row.new_objects_b:
        bits.append(f"{row.new_objects_b} object(s) only B found")
    if row.missing_objects_a:
        bits.append(f"{row.missing_objects_a} object(s) only A found")
    return "; ".join(bits) + "."


# ---------------------------------------------------------------------------
# the segmentation backend
# ---------------------------------------------------------------------------

def segment_with_cellpose(images: Sequence[np.ndarray],
                          config: ModelConfig) -> List[np.ndarray]:
    """Segment ``images`` with one Cellpose model. The default backend.

    Only :meth:`ModelConfig.eval_kwargs` is forwarded, so an argument Cellpose 4
    ignores never reaches ``eval`` — it is reported by the caller instead of
    being passed on to be silently dropped, which is the difference between a
    comparison that explains itself and one that says "no difference".

    torch and cellpose are imported here and nowhere else in this module.

    :param images: 2-D or 3-D arrays, one per field.
    :param config: the model to run.
    :returns: one integer label image per input image.
    """
    import torch
    from cellpose import models as cp_models

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cp_models.CellposeModel(
        gpu=torch.cuda.is_available(),
        device=device,
        pretrained_model=config.resolved_model,
    )
    batch = [np.asarray(image, dtype=np.float32) for image in images]
    output = model.eval(x=batch, **config.eval_kwargs())
    masks = output[0] if isinstance(output, tuple) else output
    return [np.asarray(m).astype(np.int32) for m in masks]


# ---------------------------------------------------------------------------
# loading fields to compare
# ---------------------------------------------------------------------------

#: Image extensions :func:`load_fields` will read from a folder.
FIELD_EXTENSIONS = ('.tif', '.tiff', '.png', '.npy', '.npz')


def load_fields(source: Any, n_fields: int = DEFAULT_N_FIELDS,
                channel: Optional[int] = None) -> Tuple[List[str], List[np.ndarray]]:
    """Pull the first ``n_fields`` images out of a folder (or a list).

    Handles the shapes spaCR actually leaves on disk: a folder of ``.tif`` /
    ``.png`` fields, a folder of ``.npy`` arrays, and the ``.npz`` batches the
    Mask module writes (``data`` + ``filenames``). Reading stops as soon as
    ``n_fields`` images are in hand, so pointing this at a 1536-field plate
    costs three files.

    :param source: a folder, or an already-loaded sequence of arrays.
    :param n_fields: how many fields to take.
    :param channel: index into the last axis for a multi-channel field; None
        keeps the array as it is.
    :returns: ``(names, images)``.
    :raises FileNotFoundError: when the folder does not exist.
    :raises ValueError: when it holds no readable field.
    """
    n_fields = max(1, int(n_fields))
    if not isinstance(source, (str, os.PathLike)):
        images = list(source)[:n_fields]
        if not images:
            raise ValueError("no field to compare")
        return ([f"field_{i:04d}" for i in range(len(images))],
                [_select_channel(np.asarray(im), channel) for im in images])

    folder = os.fspath(source)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no such folder: {folder}")

    names: List[str] = []
    images: List[np.ndarray] = []
    for filename in sorted(os.listdir(folder)):
        if len(images) >= n_fields:
            break
        if not filename.lower().endswith(FIELD_EXTENSIONS):
            continue
        path = os.path.join(folder, filename)
        try:
            # Only the read is forgiven: one corrupt field must not cost the
            # comparison the other two. A bad ``channel`` is a caller error and
            # is raised below, outside the guard, so it cannot be swallowed and
            # re-reported as "this folder has no images in it".
            read = list(_read_field_file(path, filename,
                                         n_fields - len(images)))
        except Exception:
            continue
        for name, array in read:
            names.append(name)
            images.append(_select_channel(array, channel))

    if not images:
        raise ValueError(
            f"found no readable field in {folder} — expected one of "
            f"{', '.join(FIELD_EXTENSIONS)}"
        )
    return names[:n_fields], images[:n_fields]


def _read_field_file(path: str, filename: str, wanted: int):
    """Yield ``(name, array)`` pairs from one file on disk."""
    lower = filename.lower()
    if lower.endswith('.npy'):
        yield os.path.splitext(filename)[0], np.load(path, allow_pickle=False)
        return
    if lower.endswith('.npz'):
        with np.load(path, allow_pickle=False) as handle:
            key = 'data' if 'data' in handle else handle.files[0]
            stack = handle[key]
            labels = (handle['filenames'] if 'filenames' in handle.files
                      else None)
            for i in range(min(int(stack.shape[0]), wanted)):
                name = (str(labels[i]) if labels is not None and i < len(labels)
                        else f"{os.path.splitext(filename)[0]}_{i}")
                yield name, stack[i]
        return
    import imageio.v2 as imageio
    yield os.path.splitext(filename)[0], np.asarray(imageio.imread(path))


def _select_channel(array: np.ndarray, channel: Optional[int]) -> np.ndarray:
    """Reduce a multi-channel field to one channel, when asked."""
    array = np.asarray(array)
    if channel is None or array.ndim < 3:
        return array
    index = int(channel)
    if not -array.shape[-1] <= index < array.shape[-1]:
        raise ValueError(
            f"channel {index} is out of range for a field with "
            f"{array.shape[-1]} channel(s)"
        )
    return array[..., index]


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def compare_models(images: Sequence[np.ndarray],
                   model_a: Any,
                   model_b: Any,
                   field_names: Optional[Sequence[str]] = None,
                   segment_fn: Optional[Callable[[Sequence[np.ndarray],
                                                  ModelConfig],
                                                 Sequence[np.ndarray]]] = None,
                   iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                   containment: float = DEFAULT_CONTAINMENT,
                   object_type: str = "cell",
                   qc: bool = True,
                   keep_images: bool = True,
                   progress: Optional[Callable[[str, int, int], None]] = None,
                   ) -> ComparisonReport:
    """Run two models over the same fields and compare what they produced.

    Each model is loaded once and run over every field, A first, then B — one
    model in memory at a time, and the per-model wall clock in the report is
    therefore a fair (if small-sample) comparison of their cost.

    The segmentation call is an argument. That is what makes this reusable: the
    Model Zoo can hand in a different backend, and a test can hand in a stub, so
    nothing about this function needs Cellpose to be exercised.

    :param images: one array per field.
    :param model_a: a :class:`ModelConfig` or a mapping (see
        :meth:`ModelConfig.from_mapping`).
    :param model_b: the other side.
    :param field_names: names for the rows; defaults to ``field_0000…``.
    :param segment_fn: ``fn(images, config) -> masks``; defaults to
        :func:`segment_with_cellpose`.
    :param iou_threshold: passed to :func:`compare_masks`.
    :param containment: passed to :func:`compare_masks`.
    :param object_type: what is being segmented, for the seg_qc scorecards.
    :param qc: score each model's masks with :mod:`spacr.seg_qc` as well.
    :param keep_images: keep the images and masks on the report so a GUI can
        draw them. Pass False for a headless sweep over many fields.
    :param progress: ``fn(message, done, total)``, called as the run proceeds.
    :returns: a :class:`ComparisonReport`.
    :raises ValueError: when there is no field, or a model returns the wrong
        number of masks.
    """
    config_a = ModelConfig.from_mapping(model_a)
    config_b = ModelConfig.from_mapping(model_b)
    if config_a.name == config_b.name:
        # Two sides called the same thing make every message ambiguous. Copies,
        # so renaming them never reaches back into the caller's own objects.
        import dataclasses

        config_a = dataclasses.replace(config_a, name=f"{config_a.name} (A)")
        config_b = dataclasses.replace(config_b, name=f"{config_b.name} (B)")

    fields = [np.asarray(image) for image in images]
    if not fields:
        raise ValueError("no field to compare: pass at least one image")
    names = ([str(n) for n in field_names] if field_names is not None
             else [f"field_{i:04d}" for i in range(len(fields))])
    if len(names) != len(fields):
        raise ValueError(
            f"got {len(names)} field name(s) for {len(fields)} field(s)"
        )

    run = segment_fn if segment_fn is not None else segment_with_cellpose
    total_steps = 3

    def _tick(message: str, done: int) -> None:
        if progress is not None:
            progress(message, done, total_steps)

    masks: Dict[str, List[np.ndarray]] = {}
    seconds: Dict[str, float] = {}
    for step, (side, config) in enumerate((('a', config_a), ('b', config_b))):
        _tick(f"Segmenting {len(fields)} field(s) with {config.name}…", step)
        started = time.perf_counter()
        produced = list(run(fields, config))
        seconds[side] = time.perf_counter() - started
        if len(produced) != len(fields):
            raise ValueError(
                f"{config.name} returned {len(produced)} mask(s) for "
                f"{len(fields)} field(s)"
            )
        masks[side] = [_as_labels(m) for m in produced]

    _tick("Comparing masks…", 2)
    qc_a = _score(masks['a'], names, object_type) if qc else [None] * len(fields)
    qc_b = _score(masks['b'], names, object_type) if qc else [None] * len(fields)

    comparisons = [
        compare_masks(masks['a'][i], masks['b'][i], field=names[i],
                      iou_threshold=iou_threshold, containment=containment,
                      qc_a=qc_a[i], qc_b=qc_b[i])
        for i in range(len(fields))
    ]

    diff = compare_configs(config_a, config_b)
    report = ComparisonReport(
        model_a=config_a,
        model_b=config_b,
        comparisons=comparisons,
        config_diff=diff,
        warnings=list(diff['warnings']),
        seconds_a=seconds['a'],
        seconds_b=seconds['b'],
        masks_a=masks['a'] if keep_images else [],
        masks_b=masks['b'] if keep_images else [],
        images=fields if keep_images else [],
        object_type=object_type,
    )
    if diff['identical'] and not report.identical_masks:
        report.warnings.append(
            "…and yet the masks differ, which cannot be a model difference: "
            "check for a non-deterministic backend or a GPU/CPU split."
        )
    _tick("Done", 3)
    return report


def _score(masks: Sequence[np.ndarray], names: Sequence[str],
           object_type: str) -> List[Optional[FieldQC]]:
    """Run :func:`spacr.seg_qc.score_masks` over one model's masks.

    Reused wholesale rather than reimplemented: whether a mask is fused,
    shattered or empty is exactly the question seg_qc already answers, with
    thresholds argued there. Scoring all fields together (rather than one at a
    time) is what gives the plate-relative flags something to compare against.
    """
    from .seg_qc import score_masks

    scored = score_masks({name: mask for name, mask in zip(names, masks)},
                         object_type=object_type)
    by_name = {qc.field: qc for qc in scored}
    return [by_name.get(name) for name in names]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

_ROW_COLUMNS = (
    ("field", lambda c: c.field),
    ("A objects", lambda c: f"{c.n_objects_a}"),
    ("B objects", lambda c: f"{c.n_objects_b}"),
    ("delta", lambda c: f"{c.object_count_delta:+d}"),
    ("ARI", lambda c: _fmt(c.ari)),
    ("matched", lambda c: _fmt(c.iou_matched_fraction, pct=True)),
    ("mean IoU", lambda c: _fmt(c.mean_matched_iou)),
    ("splits", lambda c: f"{c.split_events}"),
    ("merges", lambda c: f"{c.merge_events}"),
    ("only B", lambda c: f"{c.new_objects_b}"),
    ("only A", lambda c: f"{c.missing_objects_a}"),
    ("A qc", lambda c: c.qc_a.severity if c.qc_a else "-"),
    ("B qc", lambda c: c.qc_b.severity if c.qc_b else "-"),
)


def _render_table(rows: Sequence[Sequence[str]],
                  header: Sequence[str]) -> List[str]:
    """A fixed-width text table, header and rule included."""
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows
              else len(header[i]) for i in range(len(header))]
    out = ["  " + "  ".join(c.ljust(widths[i])
                            for i, c in enumerate(header)).rstrip(),
           "  " + "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  " + "  ".join(c.ljust(widths[i])
                                    for i, c in enumerate(row)).rstrip())
    return out


def _parameter_lines(report: ComparisonReport) -> List[str]:
    """The resolved parameters, with the differing ones marked.

    This block is the reason the module exists in the shape it does: a run whose
    two sides differ only in ``diam_mean`` looks like "the models are the same"
    unless somebody prints what actually reached the model.
    """
    a, b = report.model_a, report.model_b
    ha, hb = a.honoured_parameters(), b.honoured_parameters()
    ia, ib = a.ignored_parameters(), b.ignored_parameters()

    rows = [[('* ' if ha.get(k) != hb.get(k) else '  ') + k,
             _value(ha.get(k)), _value(hb.get(k))]
            for k in sorted(set(ha) | set(hb))]
    lines = ["  parameters that reached the model "
             "(* = the ones this run actually varies):"]
    lines.extend("  " + line for line in
                 _render_table(rows, ["  parameter", a.name, b.name]))
    if ia or ib:
        rows = [[k, _value(ia.get(k)), _value(ib.get(k))]
                for k in sorted(set(ia) | set(ib))]
        lines.append("")
        lines.append("  set but ignored by Cellpose 4 — these changed nothing:")
        lines.extend("  " + line for line in
                     _render_table(rows, ["parameter", a.name, b.name]))
    return lines


def _value(value: Any) -> str:
    """Render a parameter value, ``'-'`` when the side did not set it."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_comparison(report: ComparisonReport) -> str:
    """Render the report a human reads before picking a model.

    Order is deliberate: the warnings first (a run that cannot show a difference
    must say so before anybody reads a number off it), then the parameters that
    reached each model, then the per-field table, then the aggregate — and the
    aggregate is phrased directionally, because neither model is the truth.

    :param report: what :func:`compare_models` returned.
    :returns: a multi-line string, ready to print.
    """
    a, b = report.model_a.name, report.model_b.name
    lines = [f"Model comparison: {a} (A) vs {b} (B) — "
             f"{report.n_fields} field(s), {report.object_type}"]

    if report.warnings:
        lines.append("")
        for warning in report.warnings:
            lines.append(f"  ! {warning}")

    lines.append("")
    lines.extend(_parameter_lines(report))

    if not report.comparisons:
        lines.append("")
        lines.append("  No field was compared.")
        return "\n".join(lines)

    lines.append("")
    rows = [[fmt(c) for _, fmt in _ROW_COLUMNS] for c in report.comparisons]
    lines.extend(_render_table(rows, [name for name, _ in _ROW_COLUMNS]))

    lines.append("")
    lines.append(f"  {report.summary}")
    lines.append(
        f"  object-count difference decomposes into "
        f"{report.total_fragments} fragment(s) of A's objects, "
        f"{report.total_merged_away} A object(s) B fused, "
        f"{report.total_new_objects_b} only B found and "
        f"{report.total_missing_objects_a} only A found."
    )
    lines.append(
        f"  segmentation time: {a} {report.seconds_a:.1f}s, "
        f"{b} {report.seconds_b:.1f}s."
    )
    if report.n_both_empty:
        lines.append(
            f"  {report.n_both_empty} field(s) were empty in both models and "
            f"score a trivial ARI of 1.0; they say nothing about either model."
        )
    return "\n".join(lines)
