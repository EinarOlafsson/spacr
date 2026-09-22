"""The CPU detectors: a histogram's worth of thresholds, and propagation.

Make Masks' plain-threshold mode was Otsu and only Otsu. Otsu asks one
question of a histogram -- where is the cut that separates it into two
groups with the least variance inside each -- and it is a good question
for a field with two clear populations and a poor one for everything else.
A field that is mostly background, or whose objects are a thin bright
tail, has been thresholded better by Li's minimum cross entropy or by the
triangle method since before spaCR existed, and both are one call away in
scikit-image.

So the threshold algorithms here are NAMES OF A LEVEL AND NOTHING ELSE.
Each is one line in :data:`spacr.qt.mask_engine.GLOBAL_THRESHOLDS` or
:data:`spacr.qt.mask_engine.LOCAL_THRESHOLDS`, and everything around the
level -- the smoothing, the threshold correction, the bright or dark side,
filling holes, the watershed split, the border rule, the minimum area --
is the code Otsu already went through. Choosing Li instead of Otsu changes
where one number comes from and nothing else, which is exactly what it
should change.

MAXIMA + PROPAGATE IS THE ONE THAT IS NOT A THRESHOLD. It is
CellProfiler's Propagate: blur, find the bright centres, grow an object
out of each one until a stop rule is met. It exists because a threshold
cannot separate two objects that touch and share a border above the level,
however good the level is -- and two touching vacuoles with bright centres
are most of what a curator is correcting by hand.

WHAT IS NOT HERE, AND WHY. Local MEAN and local GAUSSIAN thresholding are
not offered again: they are the Adaptive threshold mode, which runs the
organelle engine's ``adaptive`` branch with a block size and an offset.
Local OTSU is not offered again either: it is the "Local threshold (uneven
illumination)" switch, which every global algorithm on this list can be
combined with. Niblack uses ``T = m - k*s``; Sauvola uses
``T = m*(1 + k*(s/R - 1))``, where ``m`` and ``s`` are the local mean and
standard deviation. The engine supplies float32 values without rescaling
their range, so scikit-image's default Sauvola ``R`` is 1. These formulas
are different, and changing the intensity scale can change Sauvola's result.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

#: The global threshold algorithms, as ``mode -> the caption the Mode box
#: shows``, in the order it offers them. ``otsu`` is not here: it is the
#: screen's own first row and predates this module.
THRESHOLD_LABELS: Dict[str, str] = {
    "li": "Li (minimum cross entropy)",
    "yen": "Yen",
    "triangle": "Triangle",
    "isodata": "IsoData (Ridler-Calvard)",
    "mean": "Mean",
    "minimum": "Minimum",
    "multiotsu": "Multi-Otsu",
    "sauvola": "Sauvola (local)",
    "niblack": "Niblack (local)",
}

#: The mode that grows objects out of local maxima.
PROPAGATE = "propagate"

#: Every mode this module adds, as ``mode -> caption``.
MODE_LABELS: Dict[str, str] = {
    **THRESHOLD_LABELS,
    PROPAGATE: "Maxima + propagate",
}

#: Multi-Otsu is a global algorithm in the box but is asked for by a CLASS
#: COUNT in the engine, since that is the parameter it really has. The
#: screen forces the count to at least three while this mode is chosen.
MULTIOTSU = "multiotsu"

#: What each mode suits, in the sentence pattern the organelle methods use
#: (:func:`spacr.organelle_types.method_guidance`). English; a caller
#: showing one passes it through ``tr``.
GUIDANCE: Dict[str, str] = {
    "otsu": "Suits a field with two clear populations -- objects and "
            "background, with a valley between them in the histogram. It "
            "is the right first try and the wrong one for a field that is "
            "mostly background, where Li or Triangle cut better.",
    "cellpose": "Suits cells and nuclei of the kind Cellpose was trained "
                "on, and is the only method here that knows what an "
                "object looks like rather than only how bright it is. It "
                "is slow without a GPU.",
    "li": "Suits a field whose objects are a small, bright minority -- Li "
          "minimises the cross entropy between the field and its "
          "thresholded self rather than assuming two equal populations, so "
          "it holds up where Otsu drifts into the background.",
    "yen": "Suits a field with a long bright tail and little else; Yen "
           "maximises an entropy criterion and tends to cut higher than "
           "Otsu, keeping only the confident pixels.",
    "triangle": "Suits a field with ONE dominant peak and a shoulder -- "
                "mostly background with faint objects on it. The level is "
                "geometric: the point on the histogram furthest from the "
                "line between the peak and the far end.",
    "isodata": "Suits much what Otsu suits and is the classic "
               "Ridler-Calvard iteration: the level settles where it is "
               "the mean of the two means either side of it. A reasonable "
               "second opinion when Otsu looks wrong.",
    "mean": "Suits a field split near half and half. The level is the "
            "mean intensity, which is as crude as it sounds and is here "
            "as a baseline to judge the others against.",
    "minimum": "Suits a histogram with two clear humps and a real valley "
               "between them; the level is the bottom of the valley. It "
               "refuses a histogram it cannot smooth into two humps, and "
               "says so rather than guessing.",
    "multiotsu": "Suits a field with THREE or more populations -- "
                 "background, a dim halo and bright objects -- where one "
                 "level between all of them is a compromise that fits "
                 "none. Set the class count and which class is the "
                 "foreground.",
    "sauvola": "Suits uneven illumination when local contrast separates "
               "objects: the mean is "
               "weighted by local contrast. Check the image intensity "
               "scale: this detector uses R=1 without rescaling its float "
               "input. A constant positive window can become foreground "
               "with bright objects and positive k; blank areas are not "
               "automatically rejected.",
    "niblack": "Suits uneven illumination when local intensity statistics "
               "separate objects, using the "
               "window's mean minus k times its standard deviation. "
               "Increasing k lowers the threshold and keeps more bright "
               "pixels, including background; negative k raises it. "
               "For dark objects the direction of pixel inclusion reverses.",
    PROPAGATE: "Suits touching round objects with a bright centre -- "
               "nuclei, vacuoles, colonies. Each centre becomes one "
               "object, and two that touch come apart at the ridge "
               "between them rather than at a level.",
}


class CpuParams(NamedTuple):
    """Everything the CPU modes read, as one hashable value.

    One tuple for the same reason :class:`spacr.qt.organelle_modes.MethodParams`
    is one: it rides on the magnifier's request key, and a
    mode that falls back to another must still find its own settings in it.

    :param local_k: dimensionless local contrast weight; default 0.2 for
        both modes. Niblack uses ``T = m - k*s``: increasing k lowers the
        threshold, admitting more bright pixels and fewer dark pixels.
        Sauvola uses ``T = m*(1 + k*(s/R - 1))``. The engine passes floats
        without range rescaling, giving scikit-image's default ``R = 1``.
        Its response to k depends on the local mean m and deviation s;
        positive k does not guarantee rejection of a flat background.
    :param propagate_sigma: the Gaussian blur before the maxima are found,
        in pixels. SEPARATE FROM THE IMAGE ENHANCEMENT CHAIN'S DENOISE,
        which has already run by the time a detector sees the image: this
        one exists because the blur that makes one object have one centre
        is usually far stronger than the blur anyone wants the object's
        EDGE measured through, and the propagation measures the edge on the
        same blurred image. Leave the chain's denoise off unless the field
        is genuinely noisy, or the two blurs compound.
    :param propagate_min_distance: the smallest gap between two seeds, in
        pixels; about one object radius.
    :param propagate_seed_level: how bright a maximum must be to be a seed.
    :param propagate_seed_percentile: read the seed level as a percentile
        of the blurred image rather than as an absolute intensity.
    :param propagate_exclude_border: drop seeds near the edge.
    :param propagate_stop: a key of
        :data:`spacr.qt.mask_engine.PROPAGATE_STOPS`.
    :param propagate_stop_value: the fraction, intensity or percentile the
        stop rule reads.
    :param propagate_stop_algorithm: which global threshold provides the
        floor under the ``threshold`` stop rule.
    """

    local_k: float = 0.2
    propagate_sigma: float = 2.0
    propagate_min_distance: int = 10
    propagate_seed_level: float = 90.0
    propagate_seed_percentile: bool = True
    propagate_exclude_border: bool = False
    propagate_stop: str = "seed_fraction"
    propagate_stop_value: float = 0.4
    propagate_stop_algorithm: str = "otsu"


#: The defaults, as a value to compare a request against.
DEFAULT_PARAMS = CpuParams()

#: ``mode -> the fields of :class:`CpuParams` it reads``, in the order a
#: form should show them. Read by the panel and by :func:`provenance`, so a
#: control that is on screen, a value that is recorded and a number the
#: engine is given are one list.
PARAMETERS_FOR: Dict[str, Tuple[str, ...]] = {
    "sauvola": ("local_k",),
    "niblack": ("local_k",),
    PROPAGATE: ("propagate_sigma", "propagate_min_distance",
                "propagate_seed_level", "propagate_seed_percentile",
                "propagate_exclude_border", "propagate_stop",
                "propagate_stop_value", "propagate_stop_algorithm"),
}


def modes() -> Tuple[str, ...]:
    """Every mode this module adds, in the box's order."""
    return tuple(MODE_LABELS)


def threshold_modes() -> Tuple[str, ...]:
    """The modes that are a threshold algorithm, Multi-Otsu included."""
    return tuple(THRESHOLD_LABELS)


def guidance(mode: str) -> str:
    """What ``mode`` suits, as one sentence a picker can show.

    :param mode: the detector mode key, such as ``otsu`` or ``li``.
    """
    return GUIDANCE.get(str(mode), "")


def engine_algorithm(mode: str) -> str:
    """The name :mod:`spacr.qt.mask_engine` knows ``mode`` by.

    Multi-Otsu is asked for by a class count rather than by name, so it
    maps back to ``otsu``; the engine's ``classes`` parameter is what makes
    it multi-level. Everything else is its own name.

    :param mode: the detector mode key to map to an engine algorithm.
    """
    return "otsu" if str(mode) == MULTIOTSU else str(mode)


def provenance(mode: str, params: CpuParams) -> Dict[str, object]:
    """The parameters this mode actually read, for a mask's ledger entry.

    :param mode: the selected detector mode key.
    :param params: the CPU-mode settings carried by the detection request.
    :returns: ``{field: value}``, JSON-safe; empty for a mode that reads
        none of them -- a global threshold reads only the Otsu category's
        own settings, which the entry already carries.
    """
    return {field: getattr(params, field)
            for field in PARAMETERS_FOR.get(str(mode), ())}


def propagate(image: np.ndarray, params: CpuParams, *, min_area: int = 0,
              fill_holes: bool = True):
    """Grow objects out of the local maxima of ``image``.

    A thin wrapper on :func:`spacr.qt.mask_engine.maxima_propagate_instances`,
    so the screen has one place to turn its
    controls into that call's keywords.

    :param image: the 2-D field or region, already through the chain.
    :param params: the settings as the panel holds them.
    :param min_area: the smallest object to keep, in pixels.
    :param fill_holes: close the holes inside each grown object.
    :returns: a :class:`spacr.qt.mask_engine.PropagateResult`.
    """
    from . import mask_engine as engine

    return engine.maxima_propagate_instances(
        image,
        sigma=float(params.propagate_sigma),
        min_distance=int(params.propagate_min_distance),
        seed_level=float(params.propagate_seed_level),
        seed_level_is_percentile=bool(params.propagate_seed_percentile),
        exclude_border=bool(params.propagate_exclude_border),
        stop=str(params.propagate_stop),
        stop_value=float(params.propagate_stop_value),
        stop_algorithm=str(params.propagate_stop_algorithm),
        min_area=int(min_area), fill_holes=bool(fill_holes))
