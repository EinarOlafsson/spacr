"""What a crop holds on disk, and what the model sees at load time.

Instruction 91:

    "i would like the user to be able to choose to keep the origional dtpe or
     scale the images to either 1-255 (uint8) or between 0 and 1. what do you
     think, and what is the current code doing, i tried to keep uint16
     throughout whenever possible to not loose any information. was this a
     good call?"

IT WAS A GOOD CALL, AND THE CODE ALREADY DOES IT. `.npy` crops keep whatever
the merged stack held, PNG crops narrow once through
:func:`spacr.crops.narrow_to_uint8` -- the high byte of a uint16, a linear
rescale rather than a clip -- and measurements are taken from the
full-precision array. So the structure the request asks for is the structure:
original precision wherever it can be, one declared narrowing at the only
boundary that requires one.

THE CHOICE IS NOT REALLY THE DTYPE, and this module exists to say so in code
rather than only in a comment. ``transforms.ToTensor()`` divides by 255 and
hands the model a float in [0, 1] whatever the file held, so "scale to 0-1"
is already what happens at the point it matters. The dtype on disk decides
file size and what other tools can open the crop -- a storage decision.

THE SETTING THAT MOVES A NUMBER IS THE ONE AFTER IT. spaCR normalises with

    mean = std = (0.5, 0.5, 0.5)

which maps [0, 1] onto [-1, 1]. Every ImageNet-pretrained torchvision model
was fitted on

    mean = (0.485, 0.456, 0.406)   std = (0.229, 0.224, 0.225)

so a finetune starts by handing pretrained weights inputs distributed
differently from the ones they learned. A long finetune adapts; a short one,
or a frozen backbone, pays for it. That was a literal in two places and is
now a choice.

THE DEFAULT IS UNCHANGED, deliberately. ``symmetric`` is what spaCR has
always done, and switching it silently would move every existing model's
scores with nothing in the artifact to say why. Somebody should measure both
on a real dataset and then change the default on the evidence.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.normalization")

__all__ = [
    "CROP_DTYPES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NORMALIZATIONS",
    "apply_crop_dtype",
    "normalization_stats",
    "describe_normalization",
]

#: What a crop file may hold. ``original`` is what the pipeline produced --
#: uint16 for a 16-bit camera -- and is the default because it is the only
#: one that loses nothing.
CROP_DTYPES: Tuple[str, ...] = ("original", "uint8", "uint16")

#: The statistics every ImageNet-pretrained torchvision model was fitted on.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

#: How the loader normalises after ``ToTensor()`` has already produced [0, 1].
#:
#: ``symmetric``  what spaCR has always done: [0, 1] -> [-1, 1].
#: ``imagenet``   what a pretrained model expects.
#: ``none``       leave it in [0, 1]. For training from scratch, where there
#:                are no pretrained statistics to match and centring is the
#:                optimiser's problem rather than the data's.
NORMALIZATIONS: Tuple[str, ...] = ("symmetric", "imagenet", "none")


def normalization_stats(mode: Any) -> Optional[Tuple[Tuple[float, ...],
                                                     Tuple[float, ...]]]:
    """``(mean, std)`` for ``mode``, or None when nothing should be applied.

    :param mode: one of :data:`NORMALIZATIONS`. Anything unrecognised falls
        back to ``symmetric`` with a log line -- a typo must not silently
        train a model on statistics nobody chose, but it must not stop a run
        either, and ``symmetric`` is what the run would have used before this
        setting existed.
    """
    name = str(mode or "symmetric").strip().lower()
    if name not in NORMALIZATIONS:
        LOG.info("normalize_input %r is not one of %s; using symmetric, "
                 "which is what spaCR did before this setting existed",
                 mode, list(NORMALIZATIONS))
        name = "symmetric"
    if name == "none":
        return None
    if name == "imagenet":
        return (IMAGENET_MEAN, IMAGENET_STD)
    return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def describe_normalization(mode: Any) -> str:
    """One line for the log, so a model card records what it was trained on."""
    stats = normalization_stats(mode)
    if stats is None:
        return "inputs left in [0, 1]; no mean/std normalisation"
    mean, std = stats
    if tuple(mean) == IMAGENET_MEAN:
        return (f"inputs normalised with the ImageNet statistics "
                f"mean={mean}, std={std} -- what a pretrained backbone "
                f"expects")
    return (f"inputs normalised with mean={mean}, std={std}, mapping [0, 1] "
            f"to [-1, 1]")


def apply_crop_dtype(array: np.ndarray, dtype: Any = "original") -> np.ndarray:
    """Return ``array`` in the dtype a crop file should hold.

    :param array: the crop as the pipeline produced it.
    :param dtype: one of :data:`CROP_DTYPES`.
    :returns: the array, unchanged for ``original``.

    NARROWING GOES THROUGH THE ONE RULE. ``uint8`` uses
    :func:`spacr.crops.narrow_to_uint8`, which is documented there as "the one
    and only narrowing rule" -- the high byte of a uint16, a linear rescale
    rather than a clip. A second rescale written here would be a second answer
    to the question of what a 16-bit value means as an 8-bit one, and the two
    would disagree on the crop the user compares.

    WIDENING TO ``uint16`` IS A CAST AND NOT A STRETCH. An 8-bit crop asked
    for as uint16 keeps its numbers; multiplying by 257 to "use the range"
    would change every measured intensity for no information gained.
    """
    name = str(dtype or "original").strip().lower()
    if name not in CROP_DTYPES:
        LOG.info("crop_dtype %r is not one of %s; keeping the original",
                 dtype, list(CROP_DTYPES))
        return array
    if name == "original":
        return array
    array = np.asarray(array)
    if name == "uint8":
        if array.dtype == np.uint8:
            return array
        from .crops import narrow_to_uint8
        return narrow_to_uint8(array)
    if array.dtype == np.uint16:
        return array
    if np.issubdtype(array.dtype, np.floating):
        # A float crop has no declared range, so the only honest widening is
        # through the same narrowing rule the PNG path uses and back up --
        # anything else invents a scale factor.
        from .crops import narrow_to_uint8
        return narrow_to_uint8(array).astype(np.uint16)
    return array.astype(np.uint16)
