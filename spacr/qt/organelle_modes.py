"""Organelle detection's methods, offered to Make Masks as magnifier modes.

spaCR already implements eight ways of finding an object -- ``otsu``,
``adaptive``, ``log``, ``dog``, ``ridge``, ``hysteresis``, ``cellpose`` and
``unet`` -- in :func:`spacr.object.generate_organelle_masks_sam`. Make Masks
offered two of them. A curator correcting a mask of tubules had Otsu and
Cellpose; the ridge filter that would have found the tubules was three
screens away, in a batch pipeline, and could not be tried on the field in
front of them.

THIS MODULE IS A BRIDGE AND NOT A SECOND IMPLEMENTATION. Every classical
method runs through :func:`spacr.object._segment_single_image`, the one the
organelle pipeline's workers call, and ``unet`` through
:func:`spacr.object._segment_unet`. So "adaptive" means in Make Masks
exactly what it means in a mask run, and a curator who tunes a block size
on one field is tuning the setting the pipeline will read. The whole of
this module's own work is naming the parameters, turning them into the
``organelle_*`` keys that engine reads, and saying what each method suits.

WHICH MORPHOLOGY A MODE RUNS UNDER. The organelle engine dispatches on
``organelle_morphology`` FIRST and the method second, because the same word
means different code for different shapes. Make Masks has no morphology
box -- a curator picks a detector, not a cell-biology category -- so each
mode names the morphology whose branch implements that method in the form
a curator of whole objects wants, and :data:`MODE_MORPHOLOGY` is that
choice written down:

* ``adaptive`` runs under ``irregular``: the branch that smooths, closes,
  opens, fills holes and watershed-splits, which is what a solid object
  wants. The ``spots`` branch's adaptive is a top-hat filter first and
  erases anything wider than its disk.
* ``log`` and ``dog`` run under ``spots``, the only branch that has them.
* ``ridge``, ``hysteresis`` and ``unet`` run under ``network``, likewise.

:data:`spacr.organelle_types.LEGAL_METHODS` remains the statement of which
method is legal for which shape, and :func:`guidance` reads the sentence
each mode shows straight out of it rather than restating it here.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

#: ``mode -> the organelle morphology whose branch runs it``. The module
#: docstring says why each is what it is.
MODE_MORPHOLOGY: Dict[str, str] = {
    "adaptive": "irregular",
    "log": "spots",
    "dog": "spots",
    "ridge": "network",
    "hysteresis": "network",
    "unet": "network",
}

#: ``mode -> the caption the Mode box shows``, in the order it offers them.
MODE_LABELS: Dict[str, str] = {
    "adaptive": "Adaptive threshold",
    "log": "LoG blobs",
    "dog": "DoG blobs",
    "ridge": "Ridge filter",
    "hysteresis": "Hysteresis",
    "unet": "U-Net",
}

#: The modes that are slow enough to say so before they run, and what to
#: say. ``unet`` loads a torch checkpoint and runs a network.
HEAVY_MODES: Dict[str, str] = {
    "unet": "loads a torch model and runs a network",
}


class MethodParams(NamedTuple):
    """Every parameter the organelle methods read, as one hashable value.

    One tuple rather than one per method, because it goes into the
    magnifier's request key: a request carries every setting a detector
    could read, so a mode that falls back to another still finds its own
    settings in it. The defaults are
    :func:`spacr.settings._set_organelle_defaults`', so a method means the
    same thing here as it does in a mask run until somebody moves a box.

    :param adaptive_block: the local threshold's window, in pixels, forced
        odd by the engine. Read by ``adaptive`` and by ``ridge`` when its
        threshold is adaptive.
    :param adaptive_offset: subtracted from the local mean before the
        comparison; raise it to take fewer background pixels.
    :param morph_radius: the cleanup disk, in pixels. ``adaptive`` also
        pre-smooths with half of it; the network branches close with half.
    :param fill_holes: holes up to this area, in square pixels, are filled.
        ``adaptive`` only.
    :param watershed_spots: whether ``log`` and ``dog`` grow a watershed
        from each blob centre rather than stamping a disk.
    :param log_min_sigma: smallest Gaussian scale LoG searches, in pixels;
        a blob's radius is about sigma times root two.
    :param log_max_sigma: the largest.
    :param log_num_sigma: how many scales between the two.
    :param log_threshold: the blob-response cut-off, read by ``log`` AND by
        ``dog``, which has no threshold of its own.
    :param dog_sigma_low: DoG's smallest scale, in pixels.
    :param dog_sigma_high: DoG's largest.
    :param ridge_filter: ``frangi``, ``sato`` or ``meijering``.
    :param ridge_sigmas: the filament half-widths to look for, in pixels.
    :param ridge_threshold: ``otsu`` or ``adaptive``, how the ridge
        response is cut.
    :param skeletonize: reduce a network to a one-pixel skeleton and label
        that, so area measures length rather than thickness.
    :param hysteresis_low: the weak level; under 1.0 it is read as a
        fraction and becomes that percentile of the smoothed image.
    :param hysteresis_high: the seeding level, read the same way.
    :param unet_model_path: the ``.pt``/``.pth`` file to load.
    :param unet_threshold: the probability the sigmoid output is cut at.
    """

    adaptive_block: int = 51
    adaptive_offset: float = 5.0
    morph_radius: int = 3
    fill_holes: int = 64
    watershed_spots: bool = True
    log_min_sigma: float = 1.0
    log_max_sigma: float = 10.0
    log_num_sigma: int = 10
    log_threshold: float = 0.01
    dog_sigma_low: float = 1.0
    dog_sigma_high: float = 3.0
    ridge_filter: str = "frangi"
    ridge_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)
    ridge_threshold: str = "otsu"
    skeletonize: bool = False
    hysteresis_low: float = 0.2
    hysteresis_high: float = 0.6
    unet_model_path: str = ""
    unet_threshold: float = 0.5


#: The defaults, as a value to compare a request against.
DEFAULT_PARAMS = MethodParams()

#: ``mode -> the fields of :class:`MethodParams` it actually reads``, in
#: the order a form should show them. A screen shows a mode's own
#: parameters and no others by asking here, so a control that is being
#: ignored is never on screen looking as though it is not.
PARAMETERS_FOR: Dict[str, Tuple[str, ...]] = {
    "adaptive": ("adaptive_block", "adaptive_offset", "morph_radius",
                 "fill_holes"),
    "log": ("log_min_sigma", "log_max_sigma", "log_num_sigma",
            "log_threshold", "watershed_spots"),
    "dog": ("dog_sigma_low", "dog_sigma_high", "log_threshold",
            "watershed_spots"),
    "ridge": ("ridge_filter", "ridge_sigmas", "ridge_threshold",
              "adaptive_block", "adaptive_offset", "morph_radius",
              "skeletonize"),
    "hysteresis": ("hysteresis_low", "hysteresis_high", "morph_radius",
                   "skeletonize"),
    "unet": ("unet_model_path", "unet_threshold", "skeletonize"),
}


def modes() -> Tuple[str, ...]:
    """Every mode this module adds to Make Masks, in the box's order."""
    return tuple(MODE_LABELS)


def guidance(mode: str) -> str:
    """What this method suits, from :mod:`spacr.organelle_types`.

    The sentence is built from
    :data:`spacr.organelle_types.LEGAL_METHODS`, which is where spaCR
    already records which method belongs to which shape, so a method that
    gains or loses a shape there gains or loses it here too rather than
    drifting into a second opinion.

    :param mode: one of :func:`modes`, or any organelle method name.
    :returns: an English sentence, not yet translated: a caller showing it
        passes it through ``tr``.
    """
    from ..organelle_types import method_guidance

    return method_guidance(mode)


def organelle_settings(mode: str, params: MethodParams,
                       min_area: int = 0) -> Dict[str, object]:
    """``params`` as the ``organelle_*`` keys the engine reads.

    :param mode: one of :func:`modes`.
    :param params: the parameters as the screen holds them.
    :param min_area: the smallest object to keep, in pixels -- Make Masks'
        own Min area, so one number governs the magnifier, the detect
        buttons and the Remove-small button.
    :returns: a dict for :func:`spacr.object._segment_single_image`.
    :raises KeyError: for a mode this module does not add.
    """
    morphology = MODE_MORPHOLOGY[mode]
    return {
        "organelle_morphology": morphology,
        "organelle_method": mode,
        "organelle_min_area": int(min_area),
        "organelle_adaptive_block_size": int(params.adaptive_block),
        "organelle_adaptive_offset": float(params.adaptive_offset),
        "organelle_morph_radius": int(params.morph_radius),
        "organelle_fill_holes": int(params.fill_holes),
        "organelle_tophat_radius": 5,
        "organelle_watershed_spots": bool(params.watershed_spots),
        "organelle_log_min_sigma": float(params.log_min_sigma),
        "organelle_log_max_sigma": float(params.log_max_sigma),
        "organelle_log_num_sigma": int(params.log_num_sigma),
        "organelle_log_threshold": float(params.log_threshold),
        "organelle_dog_sigma_low": float(params.dog_sigma_low),
        "organelle_dog_sigma_high": float(params.dog_sigma_high),
        "organelle_ridge_filter": str(params.ridge_filter),
        "organelle_ridge_sigmas": [float(s) for s in params.ridge_sigmas],
        "organelle_network_threshold": str(params.ridge_threshold),
        "organelle_skeletonize": bool(params.skeletonize),
        "organelle_hysteresis_low": float(params.hysteresis_low),
        "organelle_hysteresis_high": float(params.hysteresis_high),
        "organelle_unet_model_path": str(params.unet_model_path) or None,
        "organelle_unet_threshold": float(params.unet_threshold),
    }


def provenance(mode: str, params: MethodParams) -> Dict[str, object]:
    """The parameters this mode actually read, for a mask's ledger entry.

    A mode's own parameters and no others, so an entry says what the
    detection was told rather than carrying nineteen numbers eighteen of
    which no branch looked at.

    :param mode: one of :func:`modes`, or any other magnifier mode, which
        reads nothing here and records nothing.
    :param params: the parameters the detection ran with.
    :returns: ``{field: value}``, JSON-safe; empty for a mode that is not
        one of this module's.
    """
    out: Dict[str, object] = {}
    for field in PARAMETERS_FOR.get(mode, ()):
        value = getattr(params, field)
        out[field] = list(value) if isinstance(value, tuple) else value
    return out


def segment(image: np.ndarray, mode: str, params: MethodParams,
            min_area: int = 0, model=None) -> np.ndarray:
    """Detect objects in ``image`` the way organelle detection would.

    :param image: a 2-D field or region. Read, never modified.
    :param mode: one of :func:`modes`.
    :param params: the parameters the method reads.
    :param min_area: the smallest object to keep, in pixels.
    :param model: a loaded U-Net, for ``unet``; loaded from
        :attr:`MethodParams.unet_model_path` when not given.
    :returns: a 2-D label image the shape of ``image``.
    :raises KeyError: for a mode this module does not add.
    :raises ValueError: from the engine, for a parameter it refuses -- an
        unknown ridge filter, a U-Net path that is not a file.
    """
    from ..object import _load_unet_model, _segment_single_image, _segment_unet

    settings = organelle_settings(mode, params, min_area)
    field = np.asarray(image, dtype=np.float32)
    if mode == "unet":
        net = model if model is not None else _load_unet_model(settings)
        masks = _segment_unet(field[None, ...], net, settings)
        return np.asarray(masks[0])
    return np.asarray(_segment_single_image(field, settings))
