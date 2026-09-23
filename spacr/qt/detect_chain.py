"""Optional image enhancement before detection and label cleanup afterward.

In Make Masks, Apply enables the configured chain for detection and display;
Compare previews it without enabling it. Processing uses copies: files on disk
and the original values reported by the hover readout remain unchanged.

:data:`CHAIN_ORDER` fixes the sequence::

    percentile stretch -> background -> denoise -> contrast -> sharpen
        -> detect -> morphology -> split

The optional percentile stretch belongs to Make Masks, outside :class:`Chain`.
Its levels come from the whole field before a magnifier region is cropped, so
moving the box does not redefine the percentile levels.

Background subtraction precedes denoising. Denoising precedes contrast to
avoid amplifying noise; contrast precedes sharpening to avoid stretching its
halos. Morphology precedes splitting because it changes which pixels are
connected. :func:`prepare` runs background, denoise, contrast and sharpen;
:func:`finish` runs morphology and split after the selected detector.

Disabled steps return their input unchanged. :func:`heavy_steps` identifies
enabled steps that warrant a progress warning; non-local means can take
minutes on a 2,000 px field. Whole-image runs retain progress and Cancel
controls. Selecting a background method does not enable denoising.

The Mask pipeline has separate preprocessing. :func:`spacr.object._preprocess_batch`
applies rolling ball then CLAHE to organelle batches using
``organelle_rolling_ball`` and ``organelle_clahe``. The Mask settings
``remove_background``, ``background`` and ``signal_to_noise`` instead apply
per-channel intensity floors. This Make Masks chain configures neither route.

Enhancement changes the objects proposed for curation. Models trained on
enhanced images require matching preprocessing at inference; this module does
not configure the training or inference pipeline.
"""
from __future__ import annotations

import logging
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

#: The stages, in the order they run. The first is Make Masks' own
#: "detect on the normalized image" switch and is applied by the screen
#: before :func:`prepare` is called; ``detect`` is the Mode box's detector.
#: The rest are this module's.
CHAIN_ORDER: Tuple[str, ...] = (
    "percentile stretch", "background", "denoise", "contrast", "sharpen",
    "detect", "morphology", "split")

#: ``background`` values: what is subtracted, and nothing when ``none``.
BACKGROUND_METHODS: Tuple[str, ...] = ("none", "rolling_ball", "tophat")

#: ``denoise`` values.
DENOISE_METHODS: Tuple[str, ...] = (
    "none", "gaussian", "median", "bilateral", "nlm")

#: ``morphology`` values, applied to what the detector labelled.
MORPHOLOGY_OPS: Tuple[str, ...] = ("none", "open", "close", "open_close")

#: Above this EFFECTIVE radius -- the radius times
#: :attr:`Chain.background_scale`, which is the radius the estimate really
#: runs at -- a background subtraction is slow enough on a whole field to
#: be worth warning about. 30 is where the measurements in
#: :func:`background_surface` cross about four seconds on a 1,994 px
#: field; the default radius of 50 at the default scale of 0.5 is an
#: effective 25 and takes about a second, so it does not warn.
HEAVY_BACKGROUND_RADIUS = 30

#: The steps slow enough that a screen offering them should say so, as
#: ``(what to say, a predicate on the chain)``. Read by
#: :func:`heavy_steps`.
#:
#: CPU timings on one 1,994 px toxo vacuole field, for the chain alone:
#: gamma 0.01 s, CLAHE 0.19 s, median denoise 1.30 s, a rolling ball
#: at radius 50 and the default scale 1.07 s, at scale 1.0 14.49 s; a
#: top-hat at radius 50 and the default scale 2.30 s, at scale 1.0 35.89 s.
#: The one number on the card that can still turn a second into half a
#: minute is therefore the background's EFFECTIVE radius, which is why the
#: warning reads the radius and the scale together rather than the method.
#: On a magnifier box every one of these is a small fraction of the above,
#: since the box is a few hundred pixels across and the field is four
#: megapixels.
_HEAVY = (
    ("non-local means", lambda chain: chain.denoise == "nlm"),
    ("bilateral denoising", lambda chain: chain.denoise == "bilateral"),
    ("a background estimated at over {radius} px".format(
        radius=HEAVY_BACKGROUND_RADIUS),
     lambda chain: (chain.background != "none"
                    and int(chain.background_radius)
                    * min(max(float(chain.background_scale), 0.05), 1.0)
                    > HEAVY_BACKGROUND_RADIUS)),
)


class Chain(NamedTuple):
    """One pre- and post-detection chain, as a value a request can carry.

    A plain tuple of scalars, so it is hashable and goes into the
    magnifier's request key: the same box under two different chains is two
    different questions and must not be answered from one cached result.

    Every field's default is the step switched off, so
    :data:`NO_CHAIN` is "detect the image as it is" and an old session that
    knows nothing about this chain keeps behaving as it did.

    :param background: one of :data:`BACKGROUND_METHODS`.
    :param background_radius: the ball's or the top-hat disk's radius, in
        pixels. Set it comfortably larger than the largest object: a radius
        under the object size eats the objects along with the background.
    :param background_scale: the fraction of full size the background is
        ESTIMATED at, 0.1 to 1.0. See :func:`background_surface` for what
        that buys and what it costs; 1.0 is scikit-image's own answer,
        exactly.
    :param denoise: one of :data:`DENOISE_METHODS`.
    :param denoise_strength: the Gaussian's sigma in pixels, the median's
        and the bilateral's disk radius, or the non-local means' cut-off in
        multiples of the estimated noise.
    :param gamma: the exponent the intensities are raised to on 0..1. Below
        1 lifts the dim end (faint objects become visible), above 1 pushes
        it down. 1.0 is off.
    :param clahe: contrast-limited adaptive histogram equalisation.
    :param clahe_tile: the side of one CLAHE tile, in pixels.
    :param clahe_clip: CLAHE's clip limit, 0..1. Higher is more contrast
        and more amplified noise.
    :param equalize: global histogram equalisation.
    :param sharpen: an unsharp mask.
    :param sharpen_radius: the blur radius the mask is built from, in
        pixels: about the scale of the edges to sharpen.
    :param sharpen_amount: how much of the mask is added back.
    :param morphology: one of :data:`MORPHOLOGY_OPS`, applied to what was
        detected. ``open`` separates objects joined by a thin bridge,
        ``close`` joins objects broken into pieces, ``open_close`` does
        both in that order.
    :param morphology_radius: the disk radius, in pixels.
    :param split: a distance-transform watershed on what was detected,
        cutting an object with two centres in two. It is the Otsu mode's
        "Split objects that touch", offered to every other method.
    """

    background: str = "none"
    background_radius: int = 50
    background_scale: float = 0.5
    denoise: str = "none"
    denoise_strength: float = 1.0
    gamma: float = 1.0
    clahe: bool = False
    clahe_tile: int = 64
    clahe_clip: float = 0.01
    equalize: bool = False
    sharpen: bool = False
    sharpen_radius: float = 1.0
    sharpen_amount: float = 1.0
    morphology: str = "none"
    morphology_radius: int = 1
    split: bool = False


#: The chain that does nothing: what Make Masks detected with before this
#: module, and what every request carries until a box is ticked.
NO_CHAIN = Chain()


def pre_active(chain: Chain) -> bool:
    """Whether :func:`prepare` would change the image at all.

    Asked before a copy is made: a chain with nothing switched on must hand
    the detector the very array it would have had, not a float copy of it.

    :param chain: the configured pre- and post-detection steps to inspect.
    """
    return bool(
        chain.background != "none"
        or chain.denoise != "none"
        or abs(float(chain.gamma) - 1.0) > 1e-9
        or chain.clahe or chain.equalize or chain.sharpen)


def post_active(chain: Chain) -> bool:
    """Whether :func:`finish` would change what the detector labelled.

    :param chain: the configured pre- and post-detection steps to inspect.
    """
    return bool(chain.morphology != "none" or chain.split)


def is_active(chain: Chain) -> bool:
    """Whether any step of the chain is switched on.

    :param chain: the configured pre- and post-detection steps to inspect.
    """
    return pre_active(chain) or post_active(chain)


def heavy_steps(chain: Chain) -> Tuple[str, ...]:
    """The switched-on steps slow enough to warn about, in words.

    :param chain: the configured pre- and post-detection steps to inspect.
    :returns: the names, in :data:`CHAIN_ORDER`; empty when nothing
        switched on is heavy.
    """
    return tuple(words for words, is_heavy in _HEAVY if is_heavy(chain))


def _span(image: np.ndarray) -> Tuple[float, float]:
    """``(low, high-low)`` of ``image``, with a span that is never zero.

    The unit interval every scikit-image contrast routine wants is entered
    and left through this pair, so a field comes back out on the
    intensities it went in on rather than on 0..1 -- a hysteresis threshold
    given as an absolute level, and an intensity filter, both read those
    numbers.
    """
    low = float(np.min(image)) if image.size else 0.0
    high = float(np.max(image)) if image.size else 1.0
    return low, (high - low) or 1.0


def _unit(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """``image`` on 0..1, with the ``(low, span)`` that puts it back."""
    low, span = _span(image)
    return np.clip((image - low) / span, 0.0, 1.0), low, span


#: The smallest side, in pixels, a downscaled copy is allowed to have. A
#: background estimated on a thumbnail is a thumbnail's worth of detail,
#: and below this the upsampled surface stops following the illumination
#: it is meant to remove.
MIN_BACKGROUND_SIDE = 64


def background_surface(image: np.ndarray, chain: Chain) -> np.ndarray:
    """The slowly varying background under ``image``, at ``image``'s size.

    Estimate the background on a smaller copy at
    :attr:`Chain.background_scale`, then resize the surface to the input
    shape. The radius shrinks with the image to preserve its relative
    extent. This approximates slowly varying illumination while reducing
    both the pixel count and neighbourhood size.

    At scale 1.0, call ``rolling_ball`` or ``white_tophat`` at full
    resolution and return its exact result. Smaller images also use full
    resolution when downsampling would cross :data:`MIN_BACKGROUND_SIDE`.
    Full-resolution estimation can be slow: on a 1,994 px uint16 field
    converted to float32, CPU rolling-ball timings were 0.3 s at radius 5,
    0.9 s at 10, 4.1 s at 25 and 14.5 s at 50. At the default radius 50
    and scale 0.5, the same field took about a second. Timings depend on
    the hardware and image; callers should run this outside the GUI thread.

    :param image: the field or region, as float.
    :param chain: the chain, for its background method, radius and scale.
    :returns: the background, the same shape as ``image``; zeros when the
        chain subtracts no background.
    """
    if chain.background == "none":
        return np.zeros_like(image)
    radius = max(1, int(chain.background_radius))
    scale = min(max(float(chain.background_scale), 0.05), 1.0)
    height, width = (int(v) for v in image.shape[:2])
    if (scale >= 1.0 or min(height, width) * scale < MIN_BACKGROUND_SIDE
            or max(1, int(round(radius * scale))) >= radius):
        return _background_estimate(image, chain.background, radius)

    from skimage.transform import resize

    small = resize(image, (max(MIN_BACKGROUND_SIDE, int(round(height * scale))),
                           max(MIN_BACKGROUND_SIDE, int(round(width * scale)))),
                   order=1, mode="reflect", anti_aliasing=True,
                   preserve_range=True).astype(np.float32)
    surface = _background_estimate(
        small, chain.background, max(1, int(round(radius * scale))))
    return resize(surface, (height, width), order=1, mode="edge",
                  anti_aliasing=False,
                  preserve_range=True).astype(np.float32)


def _background_estimate(image: np.ndarray, method: str,
                         radius: int) -> np.ndarray:
    """scikit-image's own background surface for ``method`` at ``radius``.

    The rolling ball's surface is what ``rolling_ball`` returns; the
    top-hat's is the morphological opening, since ``white_tophat`` IS
    ``image - opening`` and the caller subtracts.
    """
    if method == "rolling_ball":
        from skimage.restoration import rolling_ball

        return np.asarray(rolling_ball(image, radius=int(radius)),
                          dtype=np.float32)
    from skimage.morphology import disk, opening

    return np.asarray(opening(image, disk(int(radius))), dtype=np.float32)


def _background(image: np.ndarray, chain: Chain) -> np.ndarray:
    """Subtract the slowly varying background ``chain`` names.

    At a scale of 1.0 the exact scikit-image call is made and its result
    used unchanged, so ``tophat`` is ``white_tophat`` to the bit; below it
    the surface comes from :func:`background_surface` and is subtracted
    here. Either way the result is clipped at zero: a background
    subtraction that goes negative is saying the background was over-
    estimated there, and a negative intensity has no meaning downstream.
    """
    if chain.background == "none":
        return image
    radius = max(1, int(chain.background_radius))
    if float(chain.background_scale) >= 1.0:
        if chain.background == "rolling_ball":
            from skimage.restoration import rolling_ball

            return np.clip(image - rolling_ball(image, radius=radius),
                           0.0, None)
        from skimage.morphology import disk, white_tophat

        return white_tophat(image, disk(radius))
    return np.clip(image - background_surface(image, chain), 0.0, None)


def _noise_sigma(unit: np.ndarray) -> float:
    """How noisy ``unit`` is, without PyWavelets.

    Pool second differences along rows and columns, then scale their median
    absolute deviation for independent Gaussian noise. This uses numpy
    without the optional PyWavelets dependency required by wavelet-based
    noise estimators.

    :param unit: the image, scaled to 0-1.
    :returns: the noise's standard deviation, never negative.
    """
    rows = np.diff(unit, n=2, axis=0) if unit.shape[0] > 2 else np.zeros(1)
    columns = np.diff(unit, n=2, axis=1) if unit.shape[1] > 2 else np.zeros(1)
    both = np.concatenate([np.ravel(rows), np.ravel(columns)])
    if not both.size:
        return 0.0
    mad = float(np.median(np.abs(both - np.median(both))))
    return max(mad / (0.6745 * np.sqrt(6.0)), 0.0)


def _denoise(image: np.ndarray, chain: Chain) -> np.ndarray:
    """Smooth the noise ``chain`` names away, keeping the intensities.

    ``denoise="none"`` returns the input unchanged. Other modes run only
    the selected denoiser; enabling background correction or contrast
    adjustment does not implicitly enable non-local means.
    """
    if chain.denoise == "none":
        return image
    strength = float(chain.denoise_strength)
    if chain.denoise == "gaussian":
        from skimage.filters import gaussian

        return gaussian(image, sigma=max(strength, 1e-3),
                        preserve_range=True).astype(np.float32)
    if chain.denoise == "median":
        from skimage.filters import median
        from skimage.morphology import disk

        return median(image, disk(max(1, int(round(strength))))
                      ).astype(np.float32)
    unit, low, span = _unit(image)
    if chain.denoise == "bilateral":
        from skimage.restoration import denoise_bilateral

        out = denoise_bilateral(unit, sigma_color=max(strength * 0.05, 1e-3),
                                sigma_spatial=max(strength, 1e-3))
    else:
        from skimage.restoration import denoise_nl_means

        sigma = _noise_sigma(unit)
        out = denoise_nl_means(unit, h=max(strength, 0.1) * max(sigma, 1e-4),
                               sigma=sigma, fast_mode=True, patch_size=5,
                               patch_distance=6, channel_axis=None)
    return (np.asarray(out, dtype=np.float32) * span + low).astype(np.float32)


def _contrast(image: np.ndarray, chain: Chain) -> np.ndarray:
    """Gamma, then CLAHE, then histogram equalisation, on 0..1 and back.

    All three are curves through the same histogram, so they run in one
    trip to the unit interval: converting back between them would only
    re-measure a span that a monotone curve cannot have changed the ends
    of.

    NOTHING SWITCHED ON IS A NO-OP AND RETURNS THE VERY ARRAY. The trip to
    the unit interval and back is a divide and a multiply in float32, so
    an "off" contrast stage that made the round trip anyway moved the
    bottom bits of every pixel -- which is why a chain of nothing but a
    background subtraction did not agree with scikit-image's own answer to
    the last decimal.
    """
    if (abs(float(chain.gamma) - 1.0) <= 1e-9 and not chain.clahe
            and not chain.equalize):
        return image
    unit, low, span = _unit(image)
    if abs(float(chain.gamma) - 1.0) > 1e-9:
        from skimage.exposure import adjust_gamma

        unit = adjust_gamma(unit, gamma=max(float(chain.gamma), 1e-3))
    if chain.clahe:
        from skimage.exposure import equalize_adapthist

        tile = max(8, int(chain.clahe_tile))
        unit = equalize_adapthist(
            unit, kernel_size=(min(tile, unit.shape[0]),
                               min(tile, unit.shape[1])),
            clip_limit=float(chain.clahe_clip))
    if chain.equalize:
        from skimage.exposure import equalize_hist

        unit = equalize_hist(unit)
    return (np.asarray(unit, dtype=np.float32) * span + low).astype(np.float32)


def _sharpen(image: np.ndarray, chain: Chain) -> np.ndarray:
    """An unsharp mask, on 0..1 and back."""
    from skimage.filters import unsharp_mask

    unit, low, span = _unit(image)
    out = unsharp_mask(unit, radius=max(float(chain.sharpen_radius), 1e-3),
                       amount=float(chain.sharpen_amount))
    return (np.asarray(out, dtype=np.float32) * span + low).astype(np.float32)


def prepare(image: np.ndarray, chain: Chain) -> np.ndarray:
    """The image a detector is to read: the first four stages of the chain.

    Background, then denoise, then contrast, then sharpen --
    :data:`CHAIN_ORDER`, whose docstring says why that order and not
    another. The percentile stretch is the screen's, applied to the whole
    field before ``image`` was cut from it.

    A STEP THAT CANNOT RUN IS SKIPPED, NOT RAISED. This is reached from a
    mouse-move (the readout under the cursor) and from the magnifier, so
    an exception here is one per mouse event: a missing optional package
    filled the console with tracebacks once (PyWavelets, 2026-09-22) and
    made the screen unusable. The detector then reads the image as far as
    the chain got, and the log says which step was dropped.

    :param image: the field, or the magnifier's box region cut from it.
        Never modified.
    :param chain: what to do to it.
    :returns: ``image`` itself when nothing is switched on -- so a detector
        reading an untouched field reads the very array and not a float
        copy -- and otherwise a new float32 array of the same shape.
    """
    if not pre_active(chain):
        return image
    out = np.asarray(image, dtype=np.float32)
    for name, step in (("background", _background), ("denoise", _denoise),
                       ("contrast", _contrast),
                       ("sharpen", _sharpen if chain.sharpen else None)):
        if step is None:
            continue
        try:
            out = step(out, chain)
        except Exception:                                    # noqa: BLE001
            LOG.warning("the %s step of the detection chain could not run; "
                        "the image is used as it is so far", name,
                        exc_info=True)
    return np.asarray(out, dtype=np.float32)


def finish(labels: np.ndarray, chain: Chain,
           intensity: Optional[np.ndarray] = None) -> np.ndarray:
    """What the detector labelled, opened or closed and then split.

    Morphology and the split are both questions about WHICH PIXELS ARE ONE
    OBJECT, so both are asked of the detection as a whole: the labels are
    read as a foreground, reshaped, and labelled again. The ids therefore
    change, which costs nothing here -- Make Masks renumbers every object it
    pastes into a mask (:func:`spacr.qt.mask_engine._paste_region_objects`).

    :param labels: the detector's label image.
    :param chain: what to do to it.
    :param intensity: the image the detector read, used as the watershed's
        landscape where it is given.
    :returns: ``labels`` itself when neither step is switched on.
    """
    if not post_active(chain):
        return labels
    from skimage.measure import label as sk_label
    from skimage.morphology import closing, disk, opening

    binary = np.asarray(labels) > 0
    if not binary.any():
        return labels
    if chain.morphology != "none":
        footprint = disk(max(1, int(chain.morphology_radius)))
        if chain.morphology in ("open", "open_close"):
            binary = opening(binary, footprint)
        if chain.morphology in ("close", "open_close"):
            binary = closing(binary, footprint)
        if not binary.any():
            return np.zeros_like(np.asarray(labels))
    if chain.split:
        from ..object import _watershed_split

        landscape = (np.asarray(intensity, dtype=np.float32)
                     if intensity is not None
                     else binary.astype(np.float32))
        return np.asarray(_watershed_split(binary, landscape))
    return np.asarray(sk_label(binary))


def provenance(chain: Chain, *, percentile_stretch: bool = False) -> Dict:
    """The chain as it goes into a mask's ledger entry.

    ONLY THE STEPS THAT RAN, plus the order they ran in, so an entry says
    what was done rather than listing fourteen defaults around it; a chain
    with nothing on records nothing but the switch that was off. The order
    is written out because it is what makes the entry enough to reproduce
    the mask: the same steps in another order are another mask.

    :param chain: the chain the detection used.
    :param percentile_stretch: whether the screen's "detect on the
        normalized image" was on -- the chain's first stage, which is the
        screen's and not this module's.
    :returns: a JSON-safe dict, empty of steps when none ran.
    """
    steps: Dict[str, object] = {}
    if percentile_stretch:
        steps["percentile_stretch"] = True
    if chain.background != "none":
        steps["background"] = str(chain.background)
        steps["background_radius"] = int(chain.background_radius)
        steps["background_scale"] = float(chain.background_scale)
    if chain.denoise != "none":
        steps["denoise"] = str(chain.denoise)
        steps["denoise_strength"] = float(chain.denoise_strength)
    if abs(float(chain.gamma) - 1.0) > 1e-9:
        steps["gamma"] = float(chain.gamma)
    if chain.clahe:
        steps["clahe"] = True
        steps["clahe_tile"] = int(chain.clahe_tile)
        steps["clahe_clip"] = float(chain.clahe_clip)
    if chain.equalize:
        steps["histogram_equalisation"] = True
    if chain.sharpen:
        steps["unsharp_mask"] = True
        steps["unsharp_radius"] = float(chain.sharpen_radius)
        steps["unsharp_amount"] = float(chain.sharpen_amount)
    if chain.morphology != "none":
        steps["morphology"] = str(chain.morphology)
        steps["morphology_radius"] = int(chain.morphology_radius)
    if chain.split:
        steps["split"] = True
    if not steps:
        return {"enhancement": "none"}
    steps["order"] = list(CHAIN_ORDER)
    return {"enhancement": steps}


#: The words each value of a chain's choice fields is named by, for
#: :func:`step_names`. A caption a person reads should say "top-hat
#: background", not the field name and the value it holds.
_STEP_WORDS: Dict[str, Dict[str, str]] = {
    "background": {"rolling_ball": "rolling-ball background",
                   "tophat": "top-hat background"},
    "denoise": {"gaussian": "Gaussian denoise", "median": "median denoise",
                "bilateral": "bilateral denoise",
                "nlm": "non-local means denoise"},
    "morphology": {"open": "morphological opening",
                   "close": "morphological closing",
                   "open_close": "morphological opening then closing"},
}


def step_names(chain: Chain, *,
               percentile_stretch: bool = False) -> Tuple[str, ...]:
    """The switched-on steps, in order, in the words a caption uses.

    English, and each name is one row a caller can translate on its own --
    the alternative, one row per combination of fifteen fields, is a
    catalog nobody can fill.

    :param chain: the chain to describe.
    :param percentile_stretch: whether the screen's stretch was on.
    :returns: the names, in :data:`CHAIN_ORDER`; empty when nothing ran.
    """
    names = []
    if percentile_stretch:
        names.append("percentile stretch")
    for field in ("background", "denoise"):
        value = getattr(chain, field)
        if value != "none":
            names.append(_STEP_WORDS[field].get(value, value))
    if abs(float(chain.gamma) - 1.0) > 1e-9:
        names.append(f"gamma {float(chain.gamma):.2f}")
    if chain.clahe:
        names.append("CLAHE")
    if chain.equalize:
        names.append("histogram equalisation")
    if chain.sharpen:
        names.append("unsharp mask")
    if chain.morphology != "none":
        names.append(_STEP_WORDS["morphology"].get(chain.morphology,
                                                   chain.morphology))
    if chain.split:
        names.append("split touching objects")
    return tuple(names)


def describe(chain: Chain, *, percentile_stretch: bool = False) -> str:
    """The switched-on steps in one line, for a status line or a tooltip.

    :param chain: the configured pre- and post-detection steps to inspect.
    :param percentile_stretch: whether to include the screen's initial
        percentile stretch before the steps in the chain.
    :returns: the steps separated by arrows, or an empty string when the
        chain does nothing.
    """
    return " → ".join(step_names(chain,
                                 percentile_stretch=percentile_stretch))
