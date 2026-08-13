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
from itertools import islice
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.normalization")

__all__ = [
    "CROP_DTYPES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NORMALIZATIONS",
    "CLIP_MEAN",
    "CLIP_STD",
    "INCEPTION_MEAN",
    "INCEPTION_STD",
    "apply_crop_dtype",
    "dataset_statistics",
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

#: OpenAI CLIP's, for a CLIP or OpenCLIP backbone. Close to ImageNet's and
#: NOT the same; a CLIP model fed ImageNet statistics is being handed inputs
#: half a standard deviation off on the blue channel.
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

#: Inception / TF-slim's, and what spaCR's historic 0.5/0.5 actually is.
#: Named so a user recognises it rather than having to recognise the numbers.
INCEPTION_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
INCEPTION_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

#: How the loader normalises after ``ToTensor()`` has already produced [0, 1].
#:
#: ``symmetric``  what spaCR has always done: [0, 1] -> [-1, 1]. The same
#:                thing Inception and TF-slim call their preprocessing.
#: ``imagenet``   what a torchvision pretrained backbone expects.
#: ``clip``       what a CLIP / OpenCLIP backbone expects. Close to
#:                ImageNet's and not the same.
#: ``dataset``    the mean and standard deviation of THIS dataset, per
#:                channel. See :func:`dataset_statistics` -- for fluorescence
#:                this is the one with an argument behind it.
#: ``custom``     numbers the user supplies, for a backbone whose
#:                preprocessing is none of the above.
#: ``none``       leave it in [0, 1]. For training from scratch, where there
#:                are no pretrained statistics to match and centring is the
#:                optimiser's problem rather than the data's.
NORMALIZATIONS: Tuple[str, ...] = (
    "symmetric", "imagenet", "clip", "dataset", "custom", "none")


def normalization_stats(mode: Any, *,
                        mean: Optional[Sequence[float]] = None,
                        std: Optional[Sequence[float]] = None,
                        channels: int = 3
                        ) -> Optional[Tuple[Tuple[float, ...],
                                            Tuple[float, ...]]]:
    """``(mean, std)`` for ``mode``, or None when nothing should be applied.

    :param mode: one of :data:`NORMALIZATIONS`. Anything unrecognised falls
        back to ``symmetric`` with a log line -- a typo must not silently
        train a model on statistics nobody chose, but it must not stop a run
        either, and ``symmetric`` is what the run would have used before this
        setting existed.
    :param mean: for ``custom`` and ``dataset``, the per-channel means. One
        value is broadcast to every channel, which is what a single-stain
        dataset wants.
    :param std: as ``mean``. A zero is replaced by 1.0 rather than dividing
        by it -- a channel with no variance is a constant channel, and
        dividing it by its own zero spread produces inf and then a loss of
        nan, several minutes into training, with nothing saying why.
    :param channels: how many planes the model will see. Only used to
        broadcast a single supplied value.
    """
    name = str(mode or "symmetric").strip().lower()
    if name not in NORMALIZATIONS:
        LOG.info("input_statistics %r is not one of %s; using symmetric, "
                 "which is what spaCR did before this setting existed",
                 mode, list(NORMALIZATIONS))
        name = "symmetric"
    if name == "none":
        return None
    if name == "imagenet":
        return (IMAGENET_MEAN, IMAGENET_STD)
    if name == "clip":
        return (CLIP_MEAN, CLIP_STD)
    if name in ("custom", "dataset"):
        if mean is None or std is None:
            # Named, and NOT silently fallen back to: a run that asked for
            # its own statistics and got ImageNet's would be a run whose
            # model card says one thing and whose weights learned another.
            raise ValueError(
                f"input_statistics={name!r} needs both mean and std. "
                f"For 'dataset', compute them with "
                f"spacr.normalization.dataset_statistics; for 'custom', "
                f"supply the numbers your backbone was trained with.")
        return (_broadcast(mean, channels), _clean_std(std, channels))
    return (INCEPTION_MEAN, INCEPTION_STD)


def _broadcast(values: Sequence[float], channels: int) -> Tuple[float, ...]:
    out = [float(v) for v in values]
    if len(out) == 1:
        return tuple(out * max(1, int(channels)))
    return tuple(out)


#: Below this, a channel is constant. NOT ``== 0``: the sum-of-squares
#: identity in :func:`dataset_statistics` leaves a constant channel at about
#: 7e-09 rather than at zero through floating-point cancellation, and
#: dividing by 7e-09 multiplies that channel by 1.3e8 -- the same disaster as
#: dividing by zero, arrived at by a route an equality check does not catch.
#: On [0, 1] data nothing real has a spread this small.
CONSTANT_CHANNEL_STD = 1e-6


def _clean_std(values: Sequence[float], channels: int) -> Tuple[float, ...]:
    """Per-channel spreads, with a constant channel normalised by one.

    A constant channel has no spread, and dividing by it produces inf or an
    enormous number, then a nan loss several minutes into training with
    nothing saying why. A spread of 1 leaves that channel alone, which is the
    only sensible thing to do with a channel that carries no variation.
    """
    out = []
    for value in _broadcast(values, channels):
        number = abs(float(value))
        if not np.isfinite(number) or number < CONSTANT_CHANNEL_STD:
            LOG.info("a channel has no spread (%.3g); normalising it by 1.0 "
                     "rather than multiplying it by %.3g", number,
                     1.0 / number if number else float("inf"))
            number = 1.0
        out.append(number)
    return tuple(out)


def dataset_statistics(loader: Any, *, max_batches: Optional[int] = None
                       ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Per-channel mean and standard deviation of the data itself.

    WHY THIS IS THE ONE WITH AN ARGUMENT BEHIND IT FOR MICROSCOPY. ImageNet's
    statistics describe photographs: three broadly correlated channels, most
    of the frame occupied by something. A fluorescence crop is mostly black
    with one bright compartment, and its channels are unrelated stains whose
    exposures were set independently. Nothing about 0.485/0.456/0.406
    describes that, and normalising by it centres the data somewhere that has
    no meaning for it.

    Computed in one streaming pass with the sum-of-squares identity, so a
    dataset that does not fit in memory still yields exact statistics rather
    than statistics of whatever fitted.

    :param loader: anything iterable yielding ``(images, ...)`` batches, or
        bare image tensors, shaped ``(N, C, H, W)`` on [0, 1].
    :param max_batches: stop after this many. The mean of a plate converges
        in far fewer batches than the plate has, and a full pass over a
        million crops to compute two numbers per channel is a cost with no
        matching gain. ``None`` reads everything.
    :returns: ``(mean, std)``, per channel.
    :raises ValueError: the loader yielded nothing, so the answer would be
        the statistics of an empty set rather than of this dataset.
    """
    total = None
    total_sq = None
    count = 0
    # islice rather than a counter and a break: a `for` pulls the next item
    # BEFORE the body can stop, so a counter reads one batch more than was
    # asked for -- which matters for a loader with side effects, one that
    # reads from disk or advances a shuffle.
    source = loader if max_batches is None else islice(loader, max_batches)
    for batch in source:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        values = np.asarray(getattr(images, "numpy", lambda: images)(),
                            dtype=np.float64)
        if values.ndim != 4:
            continue
        # (N, C, H, W) -> per channel over every pixel of every image.
        flat = values.transpose(1, 0, 2, 3).reshape(values.shape[1], -1)
        if total is None:
            total = flat.sum(axis=1)
            total_sq = (flat ** 2).sum(axis=1)
        else:
            total += flat.sum(axis=1)
            total_sq += (flat ** 2).sum(axis=1)
        count += flat.shape[1]
    if not count or total is None:
        raise ValueError(
            "the loader yielded no images, so these would be the statistics "
            "of an empty set rather than of this dataset")
    mean = total / count
    # max(0, ...): the identity can go a hair negative on a constant channel
    # through floating-point cancellation, and sqrt of that is nan.
    variance = np.maximum(total_sq / count - mean ** 2, 0.0)
    return (tuple(float(v) for v in mean),
            _clean_std(tuple(float(v) for v in np.sqrt(variance)),
                       len(mean)))


def describe_normalization(mode: Any, **kwargs) -> str:
    """One line for the log, so a model card records what it was trained on."""
    try:
        stats = normalization_stats(mode, **kwargs)
    except ValueError as exc:
        return str(exc)
    if stats is None:
        return "inputs left in [0, 1]; no mean/std normalisation"
    mean, std = stats
    known = {IMAGENET_MEAN: "ImageNet", CLIP_MEAN: "CLIP",
             INCEPTION_MEAN: "Inception/TF-slim"}
    name = known.get(tuple(mean))
    if name == "Inception/TF-slim":
        return (f"inputs normalised with mean={mean}, std={std} "
                f"({name}), mapping [0, 1] to [-1, 1]")
    if name:
        return (f"inputs normalised with the {name} statistics "
                f"mean={mean}, std={std} -- what that pretrained backbone "
                f"expects")
    return (f"inputs normalised with mean={mean}, std={std}, measured from "
            f"this dataset or supplied by hand")


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
