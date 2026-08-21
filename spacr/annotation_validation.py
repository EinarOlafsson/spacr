"""Evaluate guide-annotation strategies against measurable reference cases.

Real pooled screens do not provide cell-level guide ground truth. Validation
therefore combines four complementary checks: simulated screens with known
assignments, guide-to-well permutations that should reduce performance to
chance, held-out controls, and order-sensitivity analysis for sequential
methods.

Results report coverage and precision separately. This avoids treating a
method that labels every cell unreliably as equivalent to one that abstains
on uncertain cells and labels the remainder accurately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Screen",
    "Verdict",
    "synthesise",
    "score_annotation",
    "calibration",
    "permuted",
    "order_sensitivity",
    "benchmark",
    "baseline_majority",
    "baseline_chance",
    "BASELINES",
]


@dataclass(frozen=True)
class Screen:
    """Simulated screen with known cell-level guide assignments.

    :ivar features: ``(n_cells, n_features)``.
    :ivar scores: the classification score per cell, with the classifier's
        error already applied.
    :ivar wells: one well label per cell.
    :ivar truth: the guide carried by each cell.
    :ivar fractions: ``{well: {guide: fraction}}`` as sequencing reports
        them -- biased and thresholded, i.e. what the method actually gets.
    :ivar true_fractions: the same before the bias, kept so a test can ask
        how much of a failure was the fraction rather than the method.
    :ivar guides: every guide in the screen.
    """

    features: np.ndarray
    scores: np.ndarray
    wells: Tuple[str, ...]
    truth: Tuple[str, ...]
    fractions: Dict[str, Dict[str, float]]
    true_fractions: Dict[str, Dict[str, float]]
    guides: Tuple[str, ...]
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class Verdict:
    """How an annotation did against a known truth."""

    coverage: float          # share of cells annotated at all
    precision: float         # share of ANNOTATED cells that are right
    recall: float            # share of ALL cells annotated correctly
    per_guide: Dict[str, Tuple[float, float]]   # guide -> (precision, recall)
    confusion: Dict[Tuple[str, str], int]
    n: int

    def summary(self) -> str:
        return (f"{self.coverage:.0%} annotated, {self.precision:.0%} of "
                f"those correct ({self.recall:.0%} of all cells)")


# ---------------------------------------------------------------------------
# 1. a screen whose truth is known
# ---------------------------------------------------------------------------

def synthesise(*,
               wells: int = 24,
               guides_per_well: int = 4,
               cells_per_well: int = 60,
               features: int = 6,
               effect: float = 2.0,
               penetrance: float = 1.0,
               fraction_bias: float = 1.0,
               fraction_threshold: float = 0.0,
               classifier_accuracy: float = 1.0,
               guides: int = 8,
               seed: int = 0) -> Screen:
    """Build a simulated screen with known guide assignments.

    :param effect: how far a guide's cells sit from the origin in feature
        space, in standard deviations. At zero the simulated guides are
        indistinguishable in feature space.
    :param penetrance: the share of a guide's cells that actually show its
        phenotype. Remaining cells are drawn around the origin.
    :param fraction_bias: multiply the reported fractions by this, then
        renormalise. 1.8 reproduces what `normalised_share` documents on a
        screen after thresholding.
    :param fraction_threshold: drop guides below this share BEFORE
        renormalising.
    :param classifier_accuracy: the score is flipped toward the wrong tail
        this often. Values closer to 1 produce better-separated classes.
    :returns: the :class:`Screen`.
    """
    rng = np.random.default_rng(int(seed))
    names = [f"g{i:03d}" for i in range(int(guides))]
    centres = rng.normal(size=(len(names), int(features)))
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)
    centres *= float(effect)

    rows: List[np.ndarray] = []
    scores: List[float] = []
    well_labels: List[str] = []
    truth: List[str] = []
    reported: Dict[str, Dict[str, float]] = {}
    honest: Dict[str, Dict[str, float]] = {}

    for w in range(int(wells)):
        label = f"w{w:03d}"
        here = list(rng.choice(len(names), size=int(guides_per_well),
                               replace=False))
        weights = rng.dirichlet(np.ones(len(here)) * 2.0)
        counts = rng.multinomial(int(cells_per_well), weights)
        honest[label] = {names[g]: float(p) for g, p in zip(here, weights)}

        # SHUFFLED WITHIN THE WELL, and this is not tidiness -- it is the
        # difference between a benchmark and a lie. Emitting each guide's
        # cells as a contiguous block leaves the ROW ORDER carrying the
        # answer, and any method that hands out contiguous runs then scores
        # far above what the data can support. It was caught by the
        # `no_effect` scenario reporting 85% precision on features that
        # contain no information at all, which is the scenario's whole job.
        block: List[Tuple[np.ndarray, float, str]] = []
        for guide_index, count in zip(here, counts):
            for _ in range(int(count)):
                shows = rng.random() < float(penetrance)
                centre = centres[guide_index] if shows else np.zeros(features)
                point = centre + rng.normal(size=int(features))
                # The score tracks the phenotype, then the classifier errs.
                value = float(np.linalg.norm(centre)) + rng.normal(scale=0.5)
                if rng.random() > float(classifier_accuracy):
                    value = -value
                block.append((point, value, names[guide_index]))
        rng.shuffle(block)
        for point, value, owner in block:
            rows.append(point)
            scores.append(value)
            well_labels.append(label)
            truth.append(owner)

        # What sequencing REPORTS: thresholded, biased, renormalised.
        seen = {names[g]: float(p) for g, p in zip(here, weights)}
        kept = {g: p for g, p in seen.items() if p >= float(fraction_threshold)}
        if not kept:
            kept = dict(seen)
        total = sum(kept.values()) or 1.0
        reported[label] = {g: min(1.0, (p / total) * float(fraction_bias))
                           for g, p in kept.items()}

    return Screen(
        features=np.asarray(rows, dtype=float),
        scores=np.asarray(scores, dtype=float),
        wells=tuple(well_labels),
        truth=tuple(truth),
        fractions=reported,
        true_fractions=honest,
        guides=tuple(names),
        meta={"effect": effect, "penetrance": penetrance,
              "fraction_bias": fraction_bias,
              "classifier_accuracy": classifier_accuracy, "seed": seed},
    )


# ---------------------------------------------------------------------------
# 2. scoring
# ---------------------------------------------------------------------------

def score_annotation(truth: Sequence[str],
                     called: Sequence[str], *,
                     abstain: str = "Non_annotated",
                     guides: Optional[Sequence[str]] = None) -> Verdict:
    """Compare guide calls with known cell-level assignments.

    Coverage is the fraction of cells that receive a non-abstaining call.
    Precision is calculated only among called cells, while recall is the
    fraction of all cells called correctly. The result also contains
    per-guide precision and recall plus the complete confusion counts.
    """
    actual = [str(t) for t in truth]
    said = [str(c) for c in called]
    n = len(actual)
    if n == 0 or len(said) != n:
        return Verdict(0.0, 0.0, 0.0, {}, {}, n)

    answered = [i for i in range(n) if said[i] != abstain]
    right = [i for i in answered if said[i] == actual[i]]
    coverage = len(answered) / n
    precision = (len(right) / len(answered)) if answered else 0.0
    recall = len(right) / n

    wanted = list(guides) if guides is not None else sorted(set(actual))
    per_guide: Dict[str, Tuple[float, float]] = {}
    for guide in wanted:
        claimed = [i for i in answered if said[i] == guide]
        real = [i for i in range(n) if actual[i] == guide]
        hit = [i for i in claimed if actual[i] == guide]
        per_guide[str(guide)] = (
            (len(hit) / len(claimed)) if claimed else 0.0,
            (len(hit) / len(real)) if real else 0.0,
        )

    confusion: Dict[Tuple[str, str], int] = {}
    for i in range(n):
        key = (actual[i], said[i])
        confusion[key] = confusion.get(key, 0) + 1

    return Verdict(coverage, precision, recall, per_guide, confusion, n)


def calibration(truth: Sequence[str],
                called: Sequence[str],
                confidence: Sequence[float], *,
                bins: int = 5,
                abstain: str = "Non_annotated") -> List[Tuple[float, float, int]]:
    """Summarize confidence calibration among non-abstaining calls.

    Returns one ``(mean confidence, observed accuracy, count)`` tuple per
    populated confidence bin. This distinguishes accurate probability
    estimates from labels that happen to have high aggregate precision.
    """
    actual = [str(t) for t in truth]
    said = [str(c) for c in called]
    values = np.asarray(list(confidence), dtype=float)
    keep = [i for i in range(len(actual))
            if i < values.size and said[i] != abstain
            and np.isfinite(values[i])]
    if not keep:
        return []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    out: List[Tuple[float, float, int]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        here = [i for i in keep
                if low <= values[i] < high or (high >= 1.0 and values[i] == 1.0)]
        if not here:
            continue
        correct = sum(1 for i in here if said[i] == actual[i])
        out.append((float(np.mean(values[here])), correct / len(here),
                    len(here)))
    return out


# ---------------------------------------------------------------------------
# 3. the null -- the check that runs on REAL data
# ---------------------------------------------------------------------------

def permuted(screen: Screen, *, seed: int = 0) -> Screen:
    """Return a screen with guide-to-well assignments permuted.

    The permutation preserves cells, features, classifier scores, well sizes,
    and the distribution of guide fractions while breaking the sequencing-to-
    imaging relationship. Truth and fractions are permuted together so the
    returned object remains internally consistent for null benchmarking.
    """
    rng = np.random.default_rng(int(seed))
    labels = sorted(screen.fractions)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))
    fractions = {well: dict(screen.fractions[mapping[well]])
                 for well in labels}
    true_fractions = {well: dict(screen.true_fractions.get(mapping[well], {}))
                      for well in labels}
    return Screen(
        features=screen.features, scores=screen.scores, wells=screen.wells,
        truth=screen.truth, fractions=fractions,
        true_fractions=true_fractions, guides=screen.guides,
        meta={**screen.meta, "permuted": True, "permutation_seed": seed})


def order_sensitivity(run: Callable[[Sequence[Tuple[str, float]]], Sequence[str]],
                      ranking: Sequence[Tuple[str, float]], *,
                      repeats: int = 3,
                      seed: int = 0) -> Dict[str, float]:
    """Measure sensitivity of a sequential method to ranking order.

    :param run: takes a ranking and returns one guide name per cell.
    :param ranking: the honest order.
    :returns: ``{"changed": share of cells that differ, "repeats": n}``.

    Each repeat randomly swaps adjacent ranking entries, preserving the broad
    confidence order while perturbing ties and near-ties. The returned mean
    and worst changed-cell shares quantify order dependence.
    """
    rng = np.random.default_rng(int(seed))
    base = list(run(list(ranking)))
    if not base:
        return {"changed": 0.0, "repeats": 0}
    differences: List[float] = []
    for _ in range(int(repeats)):
        order = list(ranking)
        for index in range(len(order) - 1):
            if rng.random() < 0.5:
                order[index], order[index + 1] = order[index + 1], order[index]
        other = list(run(order))
        if len(other) != len(base):
            continue
        differences.append(
            sum(1 for a, b in zip(base, other) if a != b) / len(base))
    return {"changed": float(np.mean(differences)) if differences else 0.0,
            "worst": float(np.max(differences)) if differences else 0.0,
            "repeats": len(differences)}


# ---------------------------------------------------------------------------
# 4. every strategy, on the same screens
# ---------------------------------------------------------------------------

def benchmark(strategies: Mapping[str, Callable[[Screen], Sequence[str]]],
              scenarios: Optional[Mapping[str, Screen]] = None, *,
              null_seed: int = 7) -> Dict[str, Dict[str, object]]:
    """Evaluate annotation strategies on simulated and permuted screens.

    :param strategies: ``{name: screen -> one guide per cell}``.
    :param scenarios: ``{name: Screen}``. Defaults to
        :func:`default_scenarios`.
    :returns: ``{scenario: {strategy: {"real": Verdict, "null": Verdict}}}``.

    Each scenario is paired with its own guide-to-well permutation. The
    reported gain is the difference between real and null precision. The
    sequencing-only majority and chance baselines are included automatically.
    """
    scenes = dict(scenarios if scenarios is not None else default_scenarios())
    # THE BASELINES ARE NOT OPTIONAL. They are what separates "the method
    # works" from "the fractions work", and a caller who forgot them would
    # read the second as the first.
    everything = {**BASELINES, **dict(strategies)}
    out: Dict[str, Dict[str, object]] = {}
    for scene_name, screen in scenes.items():
        null = permuted(screen, seed=null_seed)
        here: Dict[str, object] = {}
        for name, strategy in everything.items():
            try:
                real = score_annotation(screen.truth, strategy(screen),
                                        guides=screen.guides)
            except Exception as exc:                          # noqa: BLE001
                here[name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            try:
                chance = score_annotation(null.truth, strategy(null),
                                          guides=null.guides)
            except Exception as exc:                          # noqa: BLE001
                chance = None
                here[name] = {"real": real,
                              "null_error": f"{type(exc).__name__}: {exc}"}
                continue
            here[name] = {
                "real": real, "null": chance,
                # What the data was worth, which is the number to read.
                "gain": float(real.precision - chance.precision),
            }
        out[scene_name] = here
    return out


def baseline_majority(screen: Screen) -> List[str]:
    """Assign every cell to the largest-fraction guide in its well.

    This sequencing-only baseline measures how much apparent performance is
    available from the guide fractions without cell-level features.
    """
    wells = np.asarray(screen.wells)
    out = ["Non_annotated"] * len(screen)
    for well in sorted(set(wells.tolist())):
        here = screen.fractions.get(well) or {}
        if not here:
            continue
        biggest = max(here, key=lambda g: float(here[g]))
        for index in np.flatnonzero(wells == well):
            out[int(index)] = str(biggest)
    return out


def baseline_chance(screen: Screen, *, seed: int = 0) -> List[str]:
    """Sample each cell's guide from its well's sequencing fractions.

    Together with :func:`baseline_majority`, this baseline brackets the
    performance available from sequencing counts without cell measurements.
    """
    rng = np.random.default_rng(int(seed))
    wells = np.asarray(screen.wells)
    out = ["Non_annotated"] * len(screen)
    for well in sorted(set(wells.tolist())):
        here = screen.fractions.get(well) or {}
        if not here:
            continue
        names = list(here)
        weights = np.array([max(float(here[g]), 0.0) for g in names])
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        rows = np.flatnonzero(wells == well)
        drawn = rng.choice(len(names), size=rows.size, p=weights)
        for index, pick in zip(rows, drawn):
            out[int(index)] = names[int(pick)]
    return out


#: Included in every benchmark automatically. A comparison without them is
#: a comparison that cannot say where the performance came from.
BASELINES: Dict[str, Callable[[Screen], Sequence[str]]] = {
    "baseline:majority": baseline_majority,
    "baseline:chance": baseline_chance,
}


def default_scenarios(seed: int = 0) -> Dict[str, Screen]:
    """Return simulated screens that isolate major annotation confounds.

    Separate scenarios cover no feature signal, incomplete penetrance,
    fraction inflation, classifier error, crowded wells, and their combined
    realistic case.
    """
    return {
        # Nothing wrong: the ceiling. A method that fails here is broken.
        "clean": synthesise(seed=seed),
        # No signal at all: the floor. Everything must be at chance, and a
        # method that beats chance here is reading its own anchors.
        "no_effect": synthesise(effect=0.0, seed=seed + 1),
        # Half the cells do not show the phenotype.
        "penetrance_0.5": synthesise(penetrance=0.5, seed=seed + 2),
        # The 207 mechanism: threshold, then renormalise, then inflate.
        "inflated_fractions": synthesise(fraction_threshold=0.10,
                                         fraction_bias=1.8, seed=seed + 3),
        # The maintainer's stated classifier.
        "classifier_0.94": synthesise(classifier_accuracy=0.94, seed=seed + 4),
        # Crowded wells: more guides sharing, so more chances to confuse.
        "crowded": synthesise(guides_per_well=8, guides=16, seed=seed + 5),
        # Everything at once, which is the real screen.
        "realistic": synthesise(penetrance=0.6, fraction_threshold=0.10,
                                fraction_bias=1.8, classifier_accuracy=0.94,
                                guides_per_well=6, seed=seed + 6),
    }


# ---------------------------------------------------------------------------
# 5. mixed-ratio control wells -- ground truth on REAL data
# ---------------------------------------------------------------------------
#
# The maintainer's proposal, 2026-08-21: "can the hold out be the mixed ratio
# wells, where we dont know the identity of each cell but we do know how many
# cells are PC and how many are NC from the sequencing. these were not use for
# training."
#
# IT IS BETTER THAN THE SIMULATION AND BETTER THAN INSTRUCTION 214's SINGLE
# POSITIVE CONTROL, for a reason worth stating precisely.
#
# 214 records that a single positive control cannot separate PENETRANCE from
# FRACTION BIAS: the slope of imaging-fraction on sequencing-fraction is their
# product. A RATIO SERIES separates them, because the two enter at different
# places. A well that is a proportion `pi` of PC cells has a feature
# distribution that is exactly the mixture
#
#     F_w  =  pi * F_PC  +  (1 - pi) * F_NC
#
# and `F_PC` is estimated from the extreme wells INCLUDING its non-penetrant
# cells -- a PC cell showing no phenotype is still a PC cell and is still in
# `F_PC`. So the mixture fit recovers the true CELLULAR proportion with
# penetrance already absorbed, and comparing that to what sequencing reported
# isolates the fraction bias on its own.
#
# WHAT IT DOES NOT SHOW, said here because it is the easy thing to forget:
# PC-versus-NC is a two-class problem with the largest phenotype difference in
# the screen. A method can be perfect on it and still fail at six guides in
# one well, which is the actual task. This validates the calibration and the
# discrimination; the simulation above remains the only check of the
# multi-guide assignment.

def mixture_proportion(features: np.ndarray,
                       positive: np.ndarray,
                       negative: np.ndarray) -> float:
    """Estimate a well's positive-control share from cell features.

    :param features: ``(n_cells, n_features)`` for one mixed well.
    :param positive: the pure-PC reference cells.
    :param negative: the pure-NC reference cells.
    :returns: the estimated proportion, clipped to ``[0, 1]``.

    Projects the well's mean onto the line between the two reference means:

        mu_w = pi * mu_PC + (1 - pi) * mu_NC

    solved by least squares. The method requires no per-cell labels in the
    mixed well and clips the estimate to ``[0, 1]``. It uses feature means
    rather than fitting a high-dimensional mixture density.
    """
    here = np.asarray(features, dtype=float)
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if here.size == 0 or pos.size == 0 or neg.size == 0:
        return float("nan")
    centre_pos = pos.mean(axis=0)
    centre_neg = neg.mean(axis=0)
    direction = centre_pos - centre_neg
    span = float(direction @ direction)
    if span <= 0:
        # The two controls are indistinguishable: there is no line to
        # project onto, and any number would be invented.
        return float("nan")
    return float(np.clip(((here.mean(axis=0) - centre_neg) @ direction) / span,
                         0.0, 1.0))


def mixed_ratio_calibration(features: np.ndarray,
                            wells: Sequence[str],
                            reported: Mapping[str, float], *,
                            pure_pc_wells: Optional[Sequence[str]] = None,
                            pure_nc_wells: Optional[Sequence[str]] = None,
                            pure_low: float = 0.05,
                            pure_high: float = 0.95) -> Dict[str, object]:
    """Compare imaging-derived and sequencing-reported control mixtures.

    :param features: ``(n_cells, n_features)`` over the control wells.
    :param wells: one well label per cell.
    :param reported: ``{well: PC fraction sequencing reported}``.
    :param pure_low: at or below this, a well is taken as pure NC.
    :param pure_high: at or above this, a well is taken as pure PC.
    :returns: slope, intercept, the per-well estimates, and what was used.

    A slope near one indicates agreement between cellular and sequencing
    fractions. Values below one indicate that sequencing reports a larger
    positive-control fraction than imaging estimates.

    The median pairwise slope limits the influence of one-sided contamination:
    a hit sharing a control well can add phenotype-positive cells, whereas a
    least-squares slope would be pulled toward that contaminated well.
    """
    values = np.asarray(features, dtype=float)
    labels = np.asarray([str(w) for w in wells])
    # WHICH WELLS ARE PURE IS A FACT ABOUT THE PLATE, not about the numbers
    # under test. Picking them by the REPORTED fraction is circular -- that
    # fraction is precisely the biased quantity this is measuring, so a bias
    # large enough to matter moves a pure well below the cut-off and the fit
    # refuses to run on exactly the screens that need it. Caught that way:
    # a 0.55 bias made every 100%-PC well report 0.55 and no pure well was
    # found.
    circular = pure_pc_wells is None or pure_nc_wells is None
    if pure_pc_wells is not None:
        pure_pc = [str(w) for w in pure_pc_wells]
    else:
        pure_pc = [w for w, f in reported.items()
                   if float(f) >= float(pure_high)]
    if pure_nc_wells is not None:
        pure_nc = [str(w) for w in pure_nc_wells]
    else:
        pure_nc = [w for w, f in reported.items()
                   if float(f) <= float(pure_low)]
    if not pure_pc or not pure_nc:
        return {"error": "no pure wells at either end; the references that "
                         "define the mixture line cannot be estimated",
                "pure_pc": len(pure_pc), "pure_nc": len(pure_nc)}

    reference_pc = values[np.isin(labels, pure_pc)]
    reference_nc = values[np.isin(labels, pure_nc)]

    xs: List[float] = []
    ys: List[float] = []
    per_well: Dict[str, Tuple[float, float]] = {}
    for well, said in sorted(reported.items()):
        rows = values[labels == str(well)]
        if rows.shape[0] < 3:
            continue
        seen = mixture_proportion(rows, reference_pc, reference_nc)
        if not np.isfinite(seen):
            continue
        per_well[str(well)] = (float(said), float(seen))
        xs.append(float(said))
        ys.append(float(seen))

    if len(xs) < 3:
        return {"error": f"only {len(xs)} usable well(s); a slope needs more",
                "per_well": per_well}

    slope, intercept = _theil_sen(np.asarray(xs), np.asarray(ys))
    residuals = np.asarray(ys) - (slope * np.asarray(xs) + intercept)
    out: Dict[str, object] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "median_absolute_residual": float(np.median(np.abs(residuals))),
        "wells": len(xs),
        "pure_pc_wells": len(pure_pc),
        "pure_nc_wells": len(pure_nc),
        "per_well": per_well,
        "reading": _read_slope(float(slope)),
        "reference_wells_from_design": not circular,
    }
    if circular:
        out["warning"] = (
            "the pure wells were identified by their REPORTED fraction, "
            "which is the quantity under test -- name them from the plate "
            "design with pure_pc_wells / pure_nc_wells")
    return out


def _theil_sen(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Median-of-pairwise-slopes fit -- robust to one-sided contamination."""
    slopes: List[float] = []
    for i in range(x.size):
        for j in range(i + 1, x.size):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return float("nan"), float("nan")
    slope = float(np.median(slopes))
    return slope, float(np.median(y - slope * x))


def _read_slope(slope: float) -> str:
    """What the slope says, in the terms the question was asked in."""
    if not np.isfinite(slope):
        return "no slope could be fitted"
    if slope > 1.15:
        return ("imaging finds MORE control cells than sequencing reported: "
                "the reported fractions understate the cellular ones")
    if slope < 0.85:
        return ("imaging finds FEWER control cells than sequencing reported: "
                "the reported fractions overstate the cellular ones, which "
                "is the direction filter-renormalisation produces")
    return "sequencing's fractions match the cellular fractions"


def count_agreement(called: Sequence[str],
                    wells: Sequence[str],
                    reported: Mapping[str, float],
                    guide: str) -> Dict[str, object]:
    """Compare per-well guide calls with sequencing fractions.

    Returns reported and called shares for each usable well together with
    median and worst absolute errors. Count agreement evaluates aggregate
    calibration; it does not establish that individual cells received the
    correct guide.
    """
    said = [str(c) for c in called]
    labels = np.asarray([str(w) for w in wells])
    rows: Dict[str, Tuple[float, float]] = {}
    errors: List[float] = []
    for well, expected in sorted(reported.items()):
        here = np.flatnonzero(labels == str(well))
        if here.size == 0:
            continue
        mine = sum(1 for i in here if said[i] == str(guide))
        share = mine / here.size
        rows[str(well)] = (float(expected), float(share))
        errors.append(abs(share - float(expected)))
    return {
        "guide": str(guide),
        "per_well": rows,
        "median_absolute_error": float(np.median(errors)) if errors else
        float("nan"),
        "worst_absolute_error": float(np.max(errors)) if errors else
        float("nan"),
        "wells": len(rows),
    }
