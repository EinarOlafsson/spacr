"""Well detection and physical scale for the plaque assay.

A plaque assay is counted from images that arrive in two shapes, and they need
different handling:

* **one plaque field per image** -- segment it and count, which is what
  :func:`spacr.submodules.analyze_plaques` has always done;
* **several wells in one image** -- a whole plate, or a strip. Segmenting that
  directly counts every plaque in every well into one number and loses which
  well each came from, which is the entire experiment.

This module supplies the front half for the second case: find the wells, then
hand each one to the segmenter separately.

WHY THE WELL IS MEASURED AND NOT JUST CROPPED. Plaque *area* in pixels is a
property of the microscope, not of the biology. The same plaque imaged at two
magnifications gives two areas, and a study that pools them is comparing
optics. A well, by contrast, is a manufactured object of known physical size --
a 6-well plate well is 34.8 mm whatever images it. So the well's diameter in
pixels is a ruler that is present in the image itself, and dividing by it turns
every area into a physical one that can be pooled across microscopes,
objectives and days.

That is why :func:`detect_wells` returns a diameter rather than only a box, and
why :func:`scale_from_well` is the piece the analysis actually consumes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "Well",
    "PlaqueScale",
    "WELL_DIAMETERS_MM",
    "detect_wells",
    "crop_well",
    "scale_from_well",
    "segment_plaque_image",
    "plaque_flow_outputs",
]

#: Interior diameter, in millimetres, of a well in each standard plate format.
#:
#: These are the flat-bottom culture-plate values every major supplier builds
#: to; they are a property of the plate, not of a vendor, which is what makes
#: them usable as a ruler. A format not listed here is not guessed at --
#: :func:`scale_from_well` takes an explicit diameter instead, because being
#: wrong about the ruler silently rescales every measurement in the study.
WELL_DIAMETERS_MM: Dict[str, float] = {
    "6-well": 34.8,
    "12-well": 22.1,
    "24-well": 15.6,
    "48-well": 11.1,
    "96-well": 6.4,
}

#: Detections below this confidence are dropped. The shipped detector reports
#: precision and recall of 0.987 at its own default, so this is deliberately
#: not tuned tight: a missed well loses a whole condition, while a spurious one
#: is visible immediately as a well with no plaques and an odd diameter.
DEFAULT_CONFIDENCE = 0.25


@dataclass(frozen=True)
class Well:
    """One detected well.

    :param x0: left edge in pixels.
    :param y0: top edge in pixels.
    :param x1: right edge in pixels.
    :param y1: bottom edge in pixels.
    :param confidence: the detector's score for this box.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return int(self.y1 - self.y0)

    @property
    def diameter_px(self) -> float:
        """The well's diameter in pixels, as the mean of the box sides.

        A well is round, so a correct box is square and the two sides agree.
        The MEAN rather than either side alone is what makes a slightly loose
        box degrade gently instead of biasing one way -- and the disagreement
        itself is reported by :attr:`axis_ratio`, so a box that is not square
        is visible rather than silently averaged into a plausible number.
        """
        return (self.width + self.height) / 2.0

    @property
    def axis_ratio(self) -> float:
        """Shorter box side over longer, so 1.0 is square.

        THE HONESTY CHECK ON THE RULER. A well is circular; a box much wider
        than it is tall means the detector clipped it at an image edge, or
        merged two wells, or found something that is not a well. Any of those
        makes :attr:`diameter_px` wrong, and since that diameter rescales
        every area in the well, a wrong one is worse than a missing one.
        """
        long_side = max(self.width, self.height)
        if long_side <= 0:
            return 0.0
        return min(self.width, self.height) / long_side

    def as_dict(self) -> Dict[str, Any]:
        """The box plus its derived measures, for a results table.

        :returns: The stored coordinates and confidence plus derived
            ``diameter_px`` and ``axis_ratio`` values.
        """
        out = asdict(self)
        out.update(diameter_px=self.diameter_px, axis_ratio=self.axis_ratio)
        return out


@dataclass(frozen=True)
class PlaqueScale:
    """Pixels-to-millimetres for one well, and what it was derived from.

    :param px_per_mm: pixels per millimetre.
    :param well_diameter_px: the measured diameter the scale came from.
    :param well_diameter_mm: the physical diameter it was compared against.
    :param source: how ``well_diameter_mm`` was decided -- a plate format name,
        or ``"explicit"``.
    """

    px_per_mm: float
    well_diameter_px: float
    well_diameter_mm: float
    source: str

    def area_mm2(self, area_px: float) -> float:
        """Convert a pixel area to mm^2.

        :param area_px: an area in pixels.
        :returns: the same area in square millimetres.
        """
        return float(area_px) / (self.px_per_mm ** 2)


def _load_detector(weights: str):
    """Load the YOLO well detector, or say what to install.

    :param weights: Checkpoint path passed unchanged to
        :class:`ultralytics.YOLO`.
    :returns: The constructed YOLO detector.
    :raises ImportError: when the optional ``ultralytics`` dependency is
        unavailable.

    Kept separate so the import failure has ONE address and one message.
    ``ultralytics`` is an optional spaCR dependency: most users never need
    well detection, so requiring its detection framework and model download
    for every installation would be the wrong trade.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Well detection needs the 'ultralytics' package, which spaCR does "
            "not install by default. Install it with:\n"
            "  pip install \"spacr[plaque]\"\n"
            "Or run the plaque analysis without well detection, by giving it "
            "images that each hold a single plaque field."
        ) from exc
    return YOLO(weights)


def _to_detector_channel_order(image: np.ndarray) -> np.ndarray:
    """One image in the channel order ultralytics reads an array in.

    :param image: the caller's image. A three-channel array is taken to be
        RGB, which is what :func:`cellpose.io.imread` -- spaCR's house reader
        -- returns and what every other spaCR entry point passes around.
    :returns: the same pixels with red and blue exchanged when the input is an
        ``H x W x 3`` array; anything else unchanged, since only a
        three-channel colour array has a channel order to get wrong.

    ULTRALYTICS READS AN ARRAY AS BGR AND DOES NOT CONVERT. Given a file path
    it decodes with OpenCV, which is BGR, and that is how every image these
    detectors were trained on reached them; given an array it assumes the
    caller already did the same. Handing it RGB therefore asks the detector a
    question about an image nobody has. See
    ``docs/notes/spacr/plaque.md`` for the measurement that settled this.
    """
    if not isinstance(image, np.ndarray):
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    return np.ascontiguousarray(image[:, :, ::-1])


def _host_array(value: Any) -> np.ndarray:
    """A detector output as a numpy array, wherever it was computed.

    Ultralytics returns its boxes as torch tensors on the device it ran on.
    ``np.asarray`` of a CUDA tensor raises "can't convert cuda:0 device type
    tensor to numpy", which is how Figure mode fails on a GPU while every
    CPU test passes. A tensor is copied to the host
    first.

    :param value: a tensor, an array or a sequence.
    :returns: the values as a host numpy array.
    """
    to_host = getattr(value, "cpu", None)
    if callable(to_host):
        value = to_host()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(value)


def detect_wells(image: np.ndarray, weights: str, *,
                 confidence: float = DEFAULT_CONFIDENCE,
                 imgsz: int = 640,
                 min_axis_ratio: float = 0.7) -> List[Well]:
    """Find the wells in one image.

    :param image: the field, as an ``H x W x 3`` array in **RGB** channel
        order -- what :func:`cellpose.io.imread` returns for a colour image.
        It is converted to BGR here, because that is the order ultralytics
        reads an array in and therefore the order these detectors were
        trained in. A greyscale or otherwise non-three-channel array is
        passed through untouched, and so is a file path, which ultralytics
        decodes itself.
    :param weights: path to the YOLO checkpoint.
    :param confidence: drop detections scoring below this.
    :param imgsz: inference size; 640 is what the shipped detector trained at.
    :param min_axis_ratio: reject boxes less square than this. See
        :attr:`Well.axis_ratio` -- a non-square box makes the diameter, and
        therefore every area in that well, wrong.
    :returns: the wells, ordered top-to-bottom then left-to-right, which is
        reading order and therefore the order a plate map is written in.
    :raises ImportError: when ``ultralytics`` is not installed.
    """
    model = _load_detector(weights)
    results = model.predict(source=_to_detector_channel_order(image),
                            conf=float(confidence),
                            imgsz=int(imgsz), verbose=False)
    wells: List[Well] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x0, y0, x1, y1 = (float(v) for v in _host_array(box.xyxy).ravel()[:4])
            score = (float(_host_array(box.conf).ravel()[0])
                     if box.conf is not None else 1.0)
            well = Well(int(round(x0)), int(round(y0)),
                        int(round(x1)), int(round(y1)), score)
            if well.width <= 0 or well.height <= 0:
                continue
            if well.axis_ratio < float(min_axis_ratio):
                LOG.warning(
                    "well at (%d, %d) rejected: axis ratio %.2f is below %.2f, "
                    "so its diameter cannot be trusted as a scale",
                    well.x0, well.y0, well.axis_ratio, min_axis_ratio)
                continue
            wells.append(well)
    wells.sort(key=lambda w: (w.y0, w.x0))
    return wells


def crop_well(image: np.ndarray, well: Well, *, pad: int = 0) -> np.ndarray:
    """The image inside one well.

    :param image: the full field.
    :param well: the box to cut out.
    :param pad: extra pixels around the box, clipped to the image.
    :returns: the clipped image region selected by the padded box.
    """
    height, width = image.shape[:2]
    x0 = max(0, well.x0 - pad)
    y0 = max(0, well.y0 - pad)
    x1 = min(width, well.x1 + pad)
    y1 = min(height, well.y1 + pad)
    return image[y0:y1, x0:x1]


def scale_from_well(well: Well, *,
                    plate_format: Optional[str] = None,
                    well_diameter_mm: Optional[float] = None
                    ) -> Optional[PlaqueScale]:
    """Pixels-per-millimetre from a detected well, or ``None``.

    :param well: the detected well to measure.
    :param plate_format: a key of :data:`WELL_DIAMETERS_MM`.
    :param well_diameter_mm: the physical diameter, overriding ``plate_format``.
    :returns: the scale, or ``None`` when neither argument says how big the
        well physically is.
    :raises KeyError: if ``plate_format`` is not a known format.

    RETURNS ``None`` RATHER THAN ASSUMING. Without a physical diameter there is
    no scale, and inventing one -- a default plate format, say -- would convert
    every area into confident millimetres that are wrong by whatever the real
    plate was. The caller keeps pixels and says so.
    """
    if well_diameter_mm is None and plate_format:
        if plate_format not in WELL_DIAMETERS_MM:
            raise KeyError(
                f"unknown plate format {plate_format!r}; known formats are "
                f"{sorted(WELL_DIAMETERS_MM)}, or pass well_diameter_mm")
        well_diameter_mm = WELL_DIAMETERS_MM[plate_format]
        source = plate_format
    else:
        source = "explicit"
    if not well_diameter_mm or well_diameter_mm <= 0:
        return None
    diameter_px = well.diameter_px
    if diameter_px <= 0:
        return None
    return PlaqueScale(px_per_mm=diameter_px / float(well_diameter_mm),
                       well_diameter_px=diameter_px,
                       well_diameter_mm=float(well_diameter_mm),
                       source=source)


def _number(settings: Dict[str, Any], key: str,
            default: Optional[float]) -> Optional[float]:
    """A numeric setting, or ``default`` when it is empty or not a number.

    :param settings: the plaque settings.
    :param key: the setting.
    :param default: what an empty or unreadable value means.
    :returns: the number.
    """
    value = settings.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plaque_flow_outputs(output: Any) -> Dict[str, Optional[np.ndarray]]:
    """The flow picture and cell probability out of a Cellpose result.

    Cellpose's ``eval`` returns ``(masks, flows, styles)``, and ``flows`` is
    a list whose first entry is the flow field already drawn as an RGB image
    (direction as hue, strength as brightness, the picture the Cellpose GUI
    shows) and whose third is the cell-probability map, in logits. Either
    can be a torch tensor on the device the model ran on, so both go through
    :func:`_host_array`. A result that has no flows -- a stub, or a model
    that returned only masks -- gives ``None`` for both.

    :param output: what ``model.eval`` returned.
    :returns: ``{'flow_rgb': H x W x 3 uint8 or None,
        'cellprob': H x W float32 or None}``.
    """
    found: Dict[str, Optional[np.ndarray]] = {"flow_rgb": None,
                                              "cellprob": None}
    if not isinstance(output, (list, tuple)) or len(output) < 2:
        return found
    flows = output[1]
    if not isinstance(flows, (list, tuple)):
        flows = [flows]
    try:
        if len(flows) > 0 and flows[0] is not None:
            rgb = np.squeeze(_host_array(flows[0]))
            if rgb.ndim == 3 and rgb.shape[-1] >= 3:
                found["flow_rgb"] = np.ascontiguousarray(
                    np.clip(rgb[..., :3], 0, 255).astype(np.uint8))
        if len(flows) > 2 and flows[2] is not None:
            prob = np.squeeze(_host_array(flows[2]))
            if prob.ndim == 2:
                found["cellprob"] = prob.astype(np.float32)
    except Exception:
        LOG.debug("the Cellpose flows could not be read", exc_info=True)
    return found


def segment_plaque_image(model: Any, image: np.ndarray,
                         settings: Dict[str, Any], *,
                         return_flows: bool = False) -> Any:
    """Segment one plaque image the way both Plaque mode's preview and run do.

    The image goes to Cellpose as it is, RGB or grey, and Cellpose normalises
    it. It is NOT sent through the run's historical loader
    (``_load_normalized_images_and_labels`` with ``background=200``): on an
    8-bit crop that loader saturated every pixel to 1.0 -- measured on
    ``malnio__2.tif``, min = max = 1.0 in every channel -- so the run found
    no plaques while the preview, reading the image as it is, found 67. One
    function for both is what keeps them from disagreeing again.

    :param model: a Cellpose model.
    :param image: ``H x W`` or ``H x W x 3``.
    :param settings: ``diameter``, ``flow_threshold`` and ``CP_prob``.
    :param return_flows: also hand back what the live preview's Flows and
        Cell probability tabs show, from the same call.
    :returns: the label image; with ``return_flows``, ``(labels, flows)``
        where ``flows`` is :func:`plaque_flow_outputs`.
    """
    from .spacr_cellpose import cellpose_channel_axis

    diameter = _number(settings, "diameter", None)
    output = model.eval(image, channel_axis=cellpose_channel_axis(image),
                        diameter=diameter if diameter else None,
                        flow_threshold=_number(settings, "flow_threshold", 0.4),
                        cellprob_threshold=_number(settings, "CP_prob", 0.0))
    labels = _host_array(output[0])
    if return_flows:
        return labels, plaque_flow_outputs(output)
    return labels


_COLONY_TOO_MANY = 300
"""Colony counts above this are flagged "too many to count" (TNTC).

Thirty to three hundred is the countable window of standard plate-count
methods: above it neighbouring colonies merge and compete, so the count
underestimates what was plated."""

_COLONY_TOO_FEW = 30
"""Colony counts below this are flagged "too few to count" (TFTC): a handful
of colonies carries a Poisson error too large for the CFU/mL it implies."""

_COLONY_WORKING_PX = 1600
"""Longest side, in pixels, colony segmentation works at.

A phone photo of a plate is 3000 to 6000 pixels across, far more than a
colony needs, and segmenting it whole costs gigabytes; areas are scaled back
to the original pixels afterwards."""


def _colony_gray(image: np.ndarray) -> np.ndarray:
    """One grey plane of a plate photo, as float32.

    :param image: ``H x W`` or ``H x W x C``; a colour image is taken to be
        RGB, as :func:`cellpose.io.imread` returns it.
    :returns: the luminance (Rec. 601 weights) of a colour image, the plane
        itself for a grey one, and the first plane of any other stack.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        weights = np.array([0.299, 0.587, 0.114], np.float32)
        return array[..., :3].astype(np.float32) @ weights
    if array.ndim == 3:
        return array[..., 0].astype(np.float32)
    raise ValueError(f"a plate photo must be 2-D or 3-D, not {array.shape}")


def _to_working_size(image: np.ndarray,
                     longest: int = _COLONY_WORKING_PX
                     ) -> Tuple[np.ndarray, float]:
    """The image shrunk so its longest side is at most ``longest`` pixels.

    :param image: the photo.
    :param longest: the longest side allowed.
    :returns: ``(image, factor)``, where ``factor`` is working pixels per
        original pixel, 1.0 when the image was small enough already.
    """
    import cv2

    height, width = image.shape[:2]
    factor = min(1.0, float(longest) / max(height, width, 1))
    if factor >= 1.0:
        return image, 1.0
    size = (max(1, int(round(width * factor))),
            max(1, int(round(height * factor))))
    source = image if image.dtype != np.bool_ else image.astype(np.uint8)
    return cv2.resize(source, size, interpolation=cv2.INTER_AREA), factor


def _find_dish(image: np.ndarray) -> Tuple[Well, str]:
    """The dish or well in a plate photo, when there is no detector for it.

    Tried in order: a Hough circle transform, the largest round disc left
    by an automatic threshold, and the image frame itself.

    :param image: the photo, grey or RGB.
    :returns: ``(well, method)``: the dish's bounding box, which may reach
        past the image edge when the photo clips the dish, and ``'hough'``,
        ``'largest disc'`` or ``'frame'``.

    OF THE STRONGEST CIRCLES THE LARGEST IS TAKEN. The rim of a dish is two
    circles, the wall and the agar edge, and glare or a lid adds more inside;
    the circle a colony can sit anywhere inside is the outermost one, and
    ring artefacts just inside it are removed by the segmentation.
    """
    import cv2
    from skimage import filters, measure

    gray = _colony_gray(image)
    height, width = gray.shape
    factor = 800.0 / max(height, width, 1)
    small = cv2.resize(gray, (max(1, int(width * factor)),
                              max(1, int(height * factor))),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2)
    shortest = min(small.shape)
    as_bytes = cv2.normalize(small, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    circles = cv2.HoughCircles(as_bytes, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=shortest, param1=60, param2=30,
                               minRadius=int(0.3 * shortest),
                               maxRadius=int(0.62 * shortest))
    if circles is not None and len(circles[0]):
        x, y, radius = max(circles[0][:5], key=lambda c: c[2])
        x, y, radius = x / factor, y / factor, radius / factor
        return Well(int(round(x - radius)), int(round(y - radius)),
                    int(round(x + radius)), int(round(y + radius))), "hough"
    best = None
    for mask in (small > filters.threshold_otsu(small),
                 small <= filters.threshold_otsu(small)):
        for region in measure.regionprops(measure.label(mask)):
            if region.area < 0.1 * small.size:
                continue
            roundness = 4 * np.pi * region.area / max(region.perimeter, 1) ** 2
            if roundness < 0.6:
                continue
            if best is None or region.area > best.area:
                best = region
    if best is not None:
        y, x = best.centroid
        radius = best.equivalent_diameter / 2.0
        x, y, radius = x / factor, y / factor, radius / factor
        return Well(int(round(x - radius)), int(round(y - radius)),
                    int(round(x + radius)), int(round(y + radius))), \
            "largest disc"
    return Well(0, 0, int(width), int(height)), "frame"


def _colony_background(gray: np.ndarray, radius_px: float, polarity: str,
                       reach: float = 0.05) -> np.ndarray:
    """The agar under the colonies, as a smooth image.

    A morphological opening (a closing for dark colonies) with a disc a
    ``reach`` fraction of the dish diameter wide removes every colony
    narrower than it and keeps the slow shading of the agar, lighting and
    lid. It is computed on a copy 400 pixels across the dish and scaled back.

    :param gray: the grey plane, with the area outside the dish already
        filled in from the nearest agar.
    :param radius_px: the dish radius in the pixels of ``gray``.
    :param polarity: ``'bright'`` or ``'dark'`` colonies.
    :param reach: disc radius as a fraction of 400 working pixels.
    :returns: the background, the shape of ``gray``.
    """
    import cv2

    height, width = gray.shape
    factor = 400.0 / max(2.0 * radius_px, 1.0)
    small = cv2.resize(gray, (max(1, int(width * factor)),
                              max(1, int(height * factor))),
                       interpolation=cv2.INTER_AREA)
    disc = max(2, int(round(reach * 400)))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * disc + 1, 2 * disc + 1))
    operation = cv2.MORPH_OPEN if polarity == "bright" else cv2.MORPH_CLOSE
    background = cv2.morphologyEx(small, operation, element)
    background = cv2.GaussianBlur(background, (0, 0), disc / 2.0)
    return cv2.resize(background, (width, height),
                      interpolation=cv2.INTER_LINEAR)


def _typical_colony_radius(mask: np.ndarray) -> float:
    """The radius of a typical single colony in a foreground mask.

    :param mask: the colony foreground.
    :returns: the equivalent radius of the median convex object (solidity
        above 0.9, which clumps rarely reach), or 3 pixels when there is none.
    """
    from skimage import measure

    areas = [region.area for region in measure.regionprops(measure.label(mask))
             if region.solidity > 0.9]
    return float(np.sqrt(np.median(areas) / np.pi)) if areas else 3.0


def _split_colonies(mask: np.ndarray, signal: np.ndarray,
                    depth: float = 0.08, blend: float = 0.5) -> np.ndarray:
    """Touching colonies cut apart by a distance-transform watershed.

    :param mask: the colony foreground.
    :param signal: the background-subtracted colony signal, positive for
        colonies.
    :param depth: how deep the waist between two colonies must be, as a
        fraction of the clump's own peak, for them to be cut apart.
    :param blend: the weight of the distance transform in the landscape the
        watershed floods; the rest is the smoothed colony signal.
    :returns: an integer label image, one label per colony.

    THE LANDSCAPE IS SHAPE AND BRIGHTNESS TOGETHER. The distance transform
    alone puts a peak at the centre of every round lobe, which cuts two
    colonies whose outlines still show a waist; a colony is also a dome,
    brightest at its centre, which separates neighbours whose outlines have
    merged into a straight-sided chain. Each term is scaled to its clump's
    own maximum, so the depth means the same for a small pair and a large
    one: a fixed depth in pixels either leaves small touching pairs whole or
    cuts large single colonies at every notch in their edge.
    """
    import cv2
    from scipy import ndimage as ndi
    from skimage import measure, morphology, segmentation

    clumps = measure.label(mask)
    if clumps.max() == 0:
        return clumps.astype(np.int32)
    radius = _typical_colony_radius(mask)
    distance = ndi.distance_transform_edt(mask).astype(np.float32)
    distance = cv2.GaussianBlur(distance, (0, 0), 1.0)
    dome = cv2.GaussianBlur(np.clip(signal, 0, None).astype(np.float32),
                            (0, 0), max(1.0, 0.3 * radius))
    index = np.arange(clumps.max() + 1)
    peak_distance = np.asarray(ndi.maximum(distance, clumps, index=index),
                               np.float32)
    peak_dome = np.asarray(ndi.maximum(dome, clumps, index=index), np.float32)
    peak_distance[0] = peak_dome[0] = 1.0
    landscape = (blend * distance / np.maximum(peak_distance[clumps], 1e-3)
                 + (1.0 - blend) * dome / np.maximum(peak_dome[clumps], 1e-3))
    landscape = np.where(mask, landscape, 0).astype(np.float32)
    peaks = morphology.h_maxima(landscape, float(depth)).astype(bool) & mask
    labels = segmentation.watershed(-landscape, measure.label(peaks), mask=mask)
    unclaimed = mask & (labels == 0)
    if unclaimed.any():
        extra = measure.label(unclaimed)
        labels[unclaimed] = extra[unclaimed] + labels.max()
    return labels.astype(np.int32)


def _colony_candidates(signal: np.ndarray, dish: np.ndarray, radius: float,
                       *, threshold: float, min_area: int
                       ) -> Tuple[np.ndarray, float, float]:
    """Colony foreground from a background-subtracted signal.

    :param signal: colony minus agar, positive for colonies.
    :param dish: the pixels inside the dish.
    :param radius: the dish radius in pixels.
    :param threshold: the cut, in multiples of the agar's noise.
    :param min_area: smallest colony kept, in pixels.
    :returns: ``(mask, cut, noise)``: the foreground, the cut and the
        agar noise it was a multiple of.

    THE NOISE IS READ FROM THE DIMMER 60 % OF THE DISH, which is agar on any
    plate that is still countable, so a crowded plate does not raise its own
    threshold. Holes are filled only up to a large colony's size: filling
    every hole would fill the whole dish whenever the rim glare closes a
    ring.
    """
    from skimage import morphology

    values = signal[dish]
    if values.size == 0:
        return np.zeros_like(dish), 0.0, 1.0
    lower = values[values <= np.percentile(values, 60)]
    noise = max(1.0, 1.4826 * float(np.median(np.abs(lower - np.median(lower)))))
    cut = float(np.median(values)) + float(threshold) * noise
    mask = (signal > cut) & dish
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.remove_small_holes(mask, max(16, int((0.06 * radius) ** 2)))
    mask = morphology.remove_small_objects(mask, max(1, int(min_area)))
    return mask, cut, noise


def _colony_likeness(excess: np.ndarray, mask: np.ndarray,
                     radius: float) -> float:
    """How colony-like one polarity's objects are, for choosing polarity.

    :param excess: signal above the cut.
    :param mask: that polarity's foreground.
    :param radius: the dish radius in pixels.
    :returns: the summed excess signal of the compact objects only.

    ONLY COMPACT OBJECTS COUNT. Read with the wrong polarity, a plate still
    gives foreground -- the agar between dense colonies, halos, stains --
    but as large ragged regions. Summing all foreground let those win on
    area alone; colonies are round and small against the dish.
    """
    from skimage import measure

    largest = np.pi * (0.1 * radius) ** 2
    score = 0.0
    for region in measure.regionprops(measure.label(mask)):
        if region.area > largest or region.solidity < 0.85:
            continue
        rows, cols = region.coords[:, 0], region.coords[:, 1]
        score += float(np.sum(np.clip(excess[rows, cols], 0, None)))
    return score


def _drop_rim_arcs(mask: np.ndarray, centre, radius: float,
                   rim: float) -> np.ndarray:
    """The foreground without the arcs the dish wall leaves along its rim.

    :param mask: the colony foreground.
    :param centre: the dish centre, ``(x, y)``.
    :param radius: the dish radius.
    :param rim: width of the outer ring, as a fraction of the radius.
    :returns: ``mask`` without the clumps that lie mostly in the outer ring
        and are more than 3.5 times as long as they are wide.

    DONE BEFORE SPLITTING, because the watershed cuts a thin arc into a
    chain of short pieces, each of them round enough to pass for a colony.
    """
    from skimage import measure

    out = mask.copy()
    clumps = measure.label(mask)
    cx, cy = centre
    for region in measure.regionprops(clumps):
        rows, cols = region.coords[:, 0], region.coords[:, 1]
        outer = np.mean(np.hypot(cols - cx, rows - cy) > (1.0 - rim) * radius)
        long = region.major_axis_length > 3.5 * max(region.minor_axis_length, 1.0)
        if outer > 0.5 and long:
            out[rows, cols] = False
    return out


def _keep_colonies(labels: np.ndarray, signal: np.ndarray, *, centre,
                   radius: float, cut: float, noise: float, rim: float,
                   min_solidity: float, min_contrast: float,
                   edge: Optional[float] = None, min_area: int = 0
                   ) -> np.ndarray:
    """The labels that look like colonies; the rest set to 0.

    :param labels: the split colony labels.
    :param signal: the colony signal the labels were cut from.
    :param centre: the dish centre, ``(x, y)``.
    :param radius: the dish radius.
    :param cut: the foreground threshold on ``signal``.
    :param noise: the agar noise.
    :param rim: width of the outer ring of the dish, as a fraction of the
        radius, where a label must also be round (solidity 0.85,
        eccentricity below 0.9): the wall's glare and the agar meniscus
        leave arcs there. A label there that reaches the edge of the counted
        disc must also be at least a quarter of the median label's area and
        twice ``min_area``: the wall leaves specks along that edge.
    :param min_solidity: area over convex-hull area every label must reach.
        A colony, or each colony cut out of a clump, is convex; the edges of
        glare, lid reflections and scratches are ragged.
    :param min_contrast: how far, in agar-noise units, a label's brighter
        pixels (its 90th percentile) must rise above the cut. Colonies stand
        well clear of it; texture, bubbles and the fringes of glare barely
        cross it.
    :param edge: the radius of the counted disc; ``radius`` when ``None``.
    :param min_area: the smallest colony kept, in pixels.
    :returns: the filtered label image.
    """
    from skimage import measure

    edge = float(radius if edge is None else edge)
    keep = np.zeros(int(labels.max()) + 1, bool)
    cx, cy = centre
    regions = measure.regionprops(labels)
    typical = float(np.median([r.area for r in regions])) if regions else 0.0
    for region in regions:
        y, x = region.centroid
        ok = region.solidity >= min_solidity
        if ok and np.hypot(x - cx, y - cy) > (1.0 - rim) * radius:
            ok = region.solidity >= 0.85 and region.eccentricity < 0.9
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            reach = float(np.max(np.hypot(cols - cx, rows - cy)))
            if ok and reach >= edge - 1.5:
                ok = region.area >= max(0.25 * typical, 2 * min_area)
        if ok and min_contrast:
            values = signal[region.coords[:, 0], region.coords[:, 1]]
            ok = (np.percentile(values, 90) - cut) / noise >= min_contrast
        keep[region.label] = ok
    return np.where(keep[labels], labels, 0).astype(np.int32)


def _segment_colonies(image: np.ndarray, *, centre=None, radius=None,
                      polarity: str = "auto", threshold: float = 4.0,
                      split: float = 0.08, min_area_px: Optional[float] = None,
                      margin: float = 0.01, rim: float = 0.10,
                      min_solidity: float = 0.8, min_contrast: float = 3.0
                      ) -> Dict[str, Any]:
    """Colonies on one dish or well, as a label image.

    The photo is shrunk to :data:`_COLONY_WORKING_PX`, the agar background
    is removed (:func:`_colony_background`), the colonies are thresholded
    against the agar noise (:func:`_colony_candidates`), touching ones are
    cut apart (:func:`_split_colonies`) and debris is dropped
    (:func:`_keep_colonies`).

    :param image: the dish, grey or RGB. When ``centre`` and ``radius`` are
        not given the dish is found with :func:`_find_dish`.
    :param centre: the dish centre, ``(x, y)`` in the image's pixels.
    :param radius: the dish radius in the image's pixels.
    :param polarity: ``'bright'`` colonies on darker agar, ``'dark'`` on
        lighter agar, or ``'auto'`` to try both and keep the one whose
        compact objects stand further above the agar noise.
    :param threshold: the foreground cut in multiples of the agar noise.
    :param split: the watershed depth, see :func:`_split_colonies`.
    :param min_area_px: smallest colony in original pixels; ``None`` uses
        0.4 % of the dish diameter squared.
    :param margin: fraction of the radius trimmed off the dish edge.
    :param rim: see :func:`_keep_colonies`.
    :param min_solidity: see :func:`_keep_colonies`; 0 keeps every shape.
    :param min_contrast: see :func:`_keep_colonies`; 0 keeps every label.
    :returns: ``{'labels', 'factor', 'signal', 'cut', 'noise', 'centre',
        'radius', 'polarity', 'method'}``, where ``labels`` and the
        background-subtracted ``signal`` are at the working size, ``cut`` is
        the foreground threshold on it and ``noise`` the agar noise, ``factor``
        is working pixels per original pixel and ``centre``/``radius`` are in
        working pixels.
    :raises ValueError: for a polarity that is not one of the three.
    """
    import cv2
    from scipy import ndimage as ndi

    if polarity not in ("auto", "bright", "dark"):
        raise ValueError(
            f"colony polarity must be 'auto', 'bright' or 'dark', not {polarity!r}")
    work, factor = _to_working_size(np.asarray(image))
    gray = _colony_gray(work)
    method = "given"
    if centre is None or radius is None:
        well, method = _find_dish(work)
        centre = ((well.x0 + well.x1) / 2.0, (well.y0 + well.y1) / 2.0)
        radius = well.diameter_px / 2.0
    else:
        centre = (float(centre[0]) * factor, float(centre[1]) * factor)
        radius = float(radius) * factor
    height, width = gray.shape
    rows, cols = np.ogrid[:height, :width]
    dish = np.hypot(cols - centre[0], rows - centre[1]) <= (1.0 - margin) * radius
    empty = dict(labels=np.zeros(gray.shape, np.int32), factor=factor,
                 signal=np.zeros(gray.shape, np.float32), cut=0.0, noise=1.0,
                 centre=centre, radius=radius, polarity=polarity,
                 method=method)
    if not dish.any():
        return empty
    if min_area_px is None:
        min_area = max(6, int(round((2 * radius * 0.004) ** 2)))
    else:
        min_area = max(1, int(round(float(min_area_px) * factor ** 2)))
    nearest = ndi.distance_transform_edt(~dish, return_distances=False,
                                         return_indices=True)
    filled = gray[tuple(nearest)]
    best = None
    for side in (("bright", "dark") if polarity == "auto" else (polarity,)):
        background = _colony_background(filled, radius, side)
        signal = filled - background if side == "bright" else background - filled
        signal = cv2.GaussianBlur(signal, (0, 0), 0.8)
        mask, cut, noise = _colony_candidates(signal, dish, radius,
                                              threshold=threshold,
                                              min_area=min_area)
        score = _colony_likeness(signal - cut, mask, radius)
        if best is None or score > best[0]:
            best = (score, mask, side, signal, cut, noise)
    _score, mask, side, signal, cut, noise = best
    mask = _drop_rim_arcs(mask, centre, radius, rim)
    labels = _split_colonies(mask, signal, split)
    labels = _keep_colonies(labels, signal, centre=centre, radius=radius,
                            cut=cut, noise=noise, rim=rim,
                            min_solidity=min_solidity,
                            min_contrast=min_contrast,
                            edge=(1.0 - margin) * radius, min_area=min_area)
    return dict(empty, labels=labels, signal=signal, cut=cut, noise=noise,
                polarity=side)


_COLONY_DETECTOR_CONFIDENCE = 0.25
"""Score a colony detector's box must reach to be counted.

Chosen on the validation plates of the detector's training split, never on
its test plates, as the cut with the smallest mean absolute log ratio of
count to hand count. A cut chosen by the median error alone sat on a flat
optimum and lost most colonies of the species the detector scores lowest."""

_COLONY_DETECTOR_PX = 1280
"""Inference size of the colony detector, the size it was trained at."""

_COLONY_DETECTOR_IOU = 0.5
"""Overlap above which the detector keeps only the higher-scoring of two
boxes. Colonies in a chain overlap, but less than this; at the ultralytics
default of 0.7 one colony is often counted twice."""


def _detect_colonies(image: np.ndarray, weights: str, *, centre=None,
                     radius=None,
                     confidence: float = _COLONY_DETECTOR_CONFIDENCE,
                     imgsz: int = _COLONY_DETECTOR_PX,
                     iou: float = _COLONY_DETECTOR_IOU) -> Dict[str, Any]:
    """Colonies on one dish, found by a YOLO colony detector.

    The photo is shrunk to :data:`_COLONY_WORKING_PX`, as for
    :func:`_segment_colonies`, and every box scoring at least
    ``confidence`` whose centre lies inside the dish is one colony.

    :param image: the dish, grey or RGB.
    :param weights: path to the detector checkpoint.
    :param centre: the dish centre, ``(x, y)`` in the image's pixels; with
        ``radius``, found by :func:`_find_dish` when not given.
    :param radius: the dish radius in the image's pixels.
    :param confidence: the lowest box score counted.
    :param imgsz: the detector's inference size.
    :param iou: the overlap above which two boxes are one colony.
    :returns: the keys :func:`_segment_colonies` returns, with ``labels``
        holding one filled ellipse per box (a higher-scoring box drawn over
        a lower-scoring one), ``polarity`` set to ``'detector'``, and
        ``boxes``, an ``N x 5`` array of ``x0, y0, x1, y1, score`` in working
        pixels, highest score first.
    :raises ImportError: when ``ultralytics`` is not installed.

    COUNT AND SIZES COME FROM THE BOXES, NOT THE LABELS. Colonies in a chain
    overlap, so their ellipses cover each other in the label image; the
    boxes keep every colony whole.
    """
    import cv2

    work, factor = _to_working_size(np.asarray(image))
    method = "given"
    if centre is None or radius is None:
        well, method = _find_dish(work)
        centre = ((well.x0 + well.x1) / 2.0, (well.y0 + well.y1) / 2.0)
        radius = well.diameter_px / 2.0
    else:
        centre = (float(centre[0]) * factor, float(centre[1]) * factor)
        radius = float(radius) * factor
    source = np.asarray(work)
    if source.ndim == 3 and source.shape[2] == 1:
        source = source[..., 0]
    if source.ndim == 2:
        source = np.stack([source] * 3, axis=-1)
    source = source[..., :3]
    if source.dtype != np.uint8:
        top = float(source.max()) or 1.0
        source = np.clip(source.astype(np.float32) / top * 255.0, 0, 255
                         ).astype(np.uint8)
    model = _load_detector(weights)
    results = model.predict(source=_to_detector_channel_order(source),
                            conf=float(confidence), imgsz=int(imgsz),
                            iou=float(iou), max_det=5000, verbose=False)
    boxes = []
    for result in results:
        found = getattr(result, "boxes", None)
        if found is None or not len(found):
            continue
        corners = _host_array(found.xyxy).reshape(-1, 4)
        scores = _host_array(found.conf).reshape(-1)
        for (x0, y0, x1, y1), score in zip(corners, scores):
            if np.hypot((x0 + x1) / 2.0 - centre[0],
                        (y0 + y1) / 2.0 - centre[1]) > radius:
                continue
            boxes.append((float(x0), float(y0), float(x1), float(y1),
                          float(score)))
    boxes = np.array(sorted(boxes, key=lambda b: -b[4]), np.float32
                     ).reshape(-1, 5)
    labels = np.zeros(work.shape[:2], np.int32)
    for index in range(len(boxes) - 1, -1, -1):
        x0, y0, x1, y1, _score = boxes[index]
        axes = (max(1, int(round((x1 - x0) / 2.0))),
                max(1, int(round((y1 - y0) / 2.0))))
        middle = (int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0)))
        cv2.ellipse(labels, middle, axes, 0, 0, 360, int(index + 1), -1)
    return dict(labels=labels, factor=factor,
                signal=np.zeros(work.shape[:2], np.float32), cut=0.0,
                noise=1.0, centre=centre, radius=radius, polarity="detector",
                method=method, boxes=boxes)


def _box_colonies(boxes: np.ndarray, factor: float,
                  px_per_mm: Optional[float], *, offset=(0, 0)
                  ) -> List[Dict[str, Any]]:
    """One row per detected colony, the columns of :func:`_measure_colonies`.

    Each box is read as the ellipse inscribed in it: its area is
    pi/4 x width x height, its equivalent diameter the square root of
    width x height, and its eccentricity that of the ellipse. Solidity is
    1, an ellipse being convex.

    :param boxes: ``N x 5`` boxes from :func:`_detect_colonies`, in working
        pixels.
    :param factor: working pixels per original pixel.
    :param px_per_mm: original pixels per millimetre, or ``None``.
    :param offset: ``(x, y)`` of the crop's corner in the original photo.
    :returns: dicts with the keys :func:`_measure_colonies` writes.
    """
    rows = []
    for index, (x0, y0, x1, y1, _score) in enumerate(np.asarray(boxes),
                                                     start=1):
        width = max(float(x1 - x0), 1e-6) / factor
        height = max(float(y1 - y0), 1e-6) / factor
        area = float(np.pi / 4.0 * width * height)
        diameter = float(np.sqrt(width * height))
        major, minor = max(width, height), min(width, height)
        rows.append(dict(
            colony_id=index, area_px=area, diameter_px=diameter,
            area_mm2=area / px_per_mm ** 2 if px_per_mm else None,
            diameter_mm=diameter / px_per_mm if px_per_mm else None,
            centroid_x=float(x0 + x1) / 2.0 / factor + offset[0],
            centroid_y=float(y0 + y1) / 2.0 / factor + offset[1],
            eccentricity=float(np.sqrt(1.0 - (minor / major) ** 2)),
            solidity=1.0))
    return rows


def _dilution_factor(dilution: Any) -> Optional[float]:
    """A dilution as the factor the count is multiplied by.

    :param dilution: the factor, 10000 for a 10^-4 dilution; a fraction below
        1, such as 1e-4, is read as the dilution itself and inverted.
    :returns: the factor, or ``None`` when it is missing, not a number or
        not positive.
    """
    try:
        factor = float(dilution)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(factor) or factor <= 0:
        return None
    factor = 1.0 / factor if factor < 1.0 else factor
    return factor if np.isfinite(factor) else None


def _cfu_per_ml(count: Any, dilution: Any, plated_volume_ul: Any
                ) -> Optional[float]:
    """Colony-forming units per millilitre of the undiluted sample.

    CFU/mL = colonies x dilution factor / plated volume in mL.

    :param count: colonies counted on the plate.
    :param dilution: the dilution, as :func:`_dilution_factor` reads it.
    :param plated_volume_ul: the volume spread on the plate, in microlitres.
    :returns: the CFU/mL, or ``None`` when the count, the dilution or the
        volume is missing or not positive.
    """
    factor = _dilution_factor(dilution)
    try:
        count = float(count)
        volume_ml = float(plated_volume_ul) / 1000.0
    except (TypeError, ValueError, OverflowError):
        return None
    if factor is None or not np.isfinite(count) or count < 0 or not np.isfinite(volume_ml) \
            or volume_ml <= 0:
        return None
    titre = count * factor / volume_ml
    return titre if np.isfinite(titre) else None


def _colony_count_flag(count: int, too_many: Any = _COLONY_TOO_MANY,
                       too_few: Any = _COLONY_TOO_FEW) -> str:
    """Whether a plate count lies in the countable window.

    :param count: colonies on the plate.
    :param too_many: counts above this are ``'too many to count'``; empty
        turns the upper check off.
    :param too_few: counts below this are ``'too few to count'``; empty
        turns the lower check off.
    :returns: ``'too many to count'``, ``'too few to count'`` or
        ``'countable'``.
    """
    if too_many not in (None, "") and count > float(too_many):
        return "too many to count"
    if too_few not in (None, "") and count < float(too_few):
        return "too few to count"
    return "countable"


def _load_colony_dilutions(value):
    """Resolve an optional UTF-8 CSV with ``file,dilution`` columns.

    Numeric inputs and dictionaries retain their existing behavior. A CSV
    is read and validated completely before returning its filename/stem
    mapping, so a bad later row cannot yield a partially usable map.
    """
    import csv
    import io
    import os
    from pathlib import Path

    if not isinstance(value, (str, os.PathLike)):
        return value
    if isinstance(value, str):
        if not value.strip():
            return value
        try:
            float(value)
        except ValueError:
            pass
        else:
            return value  # Numeric strings were accepted before CSV support.
    path = Path(value).expanduser()
    limit = 8 * 1024 * 1024
    try:
        with path.open('rb') as handle:
            content = handle.read(limit + 1)
        if len(content) > limit:
            raise ValueError('CSV exceeds the 8 MiB limit')
        reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline=''),
                                strict=True)
        headers = reader.fieldnames
        if not headers:
            raise ValueError('CSV needs file,dilution headers and at least one plate')
        headers = [name.strip() for name in headers]
        if any(not name for name in headers) or len(headers) != len(set(headers)):
            raise ValueError('CSV headers must be nonempty and unique')
        if not {'file', 'dilution'} <= set(headers):
            raise ValueError('CSV needs file,dilution headers')
        reader.fieldnames = headers
        mapping = {}
        for row in reader:
            line = reader.line_num
            if None in row or any(cell is None for cell in row.values()):
                raise ValueError(f'CSV row {line} has a different number of fields than the header')
            name = row['file'].strip()
            if not name or name in ('.', '..') or any(c in name for c in ('/', '\\', '\x00')):
                raise ValueError(f'CSV row {line}: file must be a filename or stem, not a path')
            if name in mapping:
                raise ValueError(f'CSV row {line}: duplicate file identifier {name!r}')
            raw = row['dilution'].strip()
            if _dilution_factor(raw) is None:
                raise ValueError(f'CSV row {line}: dilution must be positive, finite and representable')
            mapping[name] = float(raw)
        if not mapping:
            raise ValueError('CSV contains no plate dilutions')
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ValueError(f'colony_dilution CSV {str(path)!r}: {exc}') from exc
    return mapping


def _dilution_for(name: str, dilution: Any) -> Any:
    """The dilution factor that applies to one plate.

    :param name: the plate's file name.
    :param dilution: one factor for every plate, or a dict from file name or
        file stem to factor.
    :returns: the factor, or ``None`` when a dict does not name this plate.
    """
    dilution = _load_colony_dilutions(dilution)
    if isinstance(dilution, dict):
        stem = name.rsplit(".", 1)[0]
        for key in (name, stem):
            if key in dilution:
                return dilution[key]
        return None
    return dilution


def _measure_colonies(labels: np.ndarray, factor: float,
                      px_per_mm: Optional[float], *, offset=(0, 0)
                      ) -> List[Dict[str, Any]]:
    """One row per colony, in original pixels and, with a scale, mm.

    :param labels: the colony label image at the working size.
    :param factor: working pixels per original pixel.
    :param px_per_mm: original pixels per millimetre, or ``None``.
    :param offset: ``(x, y)`` of the label image's corner in the original
        photo, so centroids are photo coordinates.
    :returns: dicts with ``colony_id``, ``area_px``, ``diameter_px``,
        ``area_mm2``, ``diameter_mm``, ``centroid_x``, ``centroid_y``,
        ``eccentricity`` and ``solidity``.
    """
    from skimage import measure

    rows = []
    for region in measure.regionprops(labels):
        area = float(region.area) / factor ** 2
        diameter = float(region.equivalent_diameter) / factor
        y, x = region.centroid
        rows.append(dict(
            colony_id=int(region.label), area_px=area, diameter_px=diameter,
            area_mm2=area / px_per_mm ** 2 if px_per_mm else None,
            diameter_mm=diameter / px_per_mm if px_per_mm else None,
            centroid_x=x / factor + offset[0],
            centroid_y=y / factor + offset[1],
            eccentricity=float(region.eccentricity),
            solidity=float(region.solidity)))
    return rows


def _count_colony_plate(image: np.ndarray, *, name: str = "",
                        well: Optional[Well] = None,
                        scale: Optional[PlaqueScale] = None,
                        settings: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """Count and measure the colonies on one dish or well.

    :param image: the whole photo, grey or RGB.
    :param name: the photo's file name, which a per-plate
        ``colony_dilution`` dict is looked up by.
    :param well: the dish or well to count inside, from
        :func:`detect_wells`; ``None`` finds it with :func:`_find_dish`.
    :param scale: the ruler to use; ``None`` derives one from the well and
        ``plate_format`` / ``well_diameter_mm`` in ``settings`` when they say
        how large it is, and keeps pixels otherwise.
    :param settings: the plaque settings; the ``colony_*`` keys, and
        ``plate_format`` / ``well_diameter_mm`` / ``plaque_pixels_per_um``
        for the scale. A ``colony_detector`` checkpoint path finds the
        colonies with :func:`_detect_colonies` instead of
        :func:`_segment_colonies`; the resolved path is expected here, not a
        model-zoo key.
    :returns: ``{'summary', 'colonies', 'labels', 'image', 'well',
        'method', 'polarity', 'centre', 'radius', 'factor', 'offset'}``: a
        summary row, one row per colony, the label image, the working-size
        crop it was drawn on, the dish circle in that crop's pixels, working
        pixels per original pixel, and the crop's corner in the photo.
    """
    settings = dict(settings or {})
    dilution = _dilution_for(name, settings.get("colony_dilution", 1))
    method = "detector"
    if well is None:
        well, method = _find_dish(image)
    height, width = image.shape[:2]
    crop = crop_well(image, well)
    x_off, y_off = max(0, well.x0), max(0, well.y0)
    centre = ((well.x0 + well.x1) / 2.0 - x_off,
              (well.y0 + well.y1) / 2.0 - y_off)
    if scale is None:
        manual = _number(settings, "plaque_pixels_per_um", None)
        if manual and manual > 0:
            scale = PlaqueScale(px_per_mm=manual * 1000.0,
                                well_diameter_px=well.diameter_px,
                                well_diameter_mm=well.diameter_px / (manual * 1000.0),
                                source="manual settings")
        else:
            plate_format = settings.get("plate_format") or None
            if plate_format in ("None", ""):
                plate_format = None
            scale = scale_from_well(
                well, plate_format=plate_format,
                well_diameter_mm=_number(settings, "well_diameter_mm", None))
    polarity = str(settings.get("colony_polarity") or "auto")
    detector = settings.get("colony_detector") or None
    if detector:
        found = _detect_colonies(crop, str(detector), centre=centre,
                                 radius=well.diameter_px / 2.0)
    else:
        found = _segment_colonies(
            crop, centre=centre, radius=well.diameter_px / 2.0,
            polarity=polarity,
            threshold=_number(settings, "colony_threshold", 4.0) or 4.0,
            min_area_px=_number(settings, "colony_min_area_px", None))
    px_per_mm = scale.px_per_mm if scale else None
    if detector:
        colonies = _box_colonies(found["boxes"], found["factor"], px_per_mm,
                                 offset=(x_off, y_off))
    else:
        colonies = _measure_colonies(found["labels"], found["factor"],
                                     px_per_mm, offset=(x_off, y_off))
    count = len(colonies)
    areas = np.array([row["area_px"] for row in colonies], float)
    diameters = np.array([row["diameter_px"] for row in colonies], float)
    dish_area = np.pi * (well.diameter_px / 2.0) ** 2
    flag = _colony_count_flag(count, settings.get("colony_too_many", _COLONY_TOO_MANY),
                              settings.get("colony_too_few", _COLONY_TOO_FEW))
    volume = _number(settings, "colony_plated_volume_ul", 100.0)
    summary = dict(
        colony_count=count, count_flag=flag,
        cfu_per_ml=_cfu_per_ml(count, dilution, volume),
        dilution=_dilution_factor(dilution), plated_volume_ul=volume,
        dish_method=method, dish_x0=well.x0, dish_y0=well.y0,
        dish_x1=well.x1, dish_y1=well.y1,
        dish_diameter_px=well.diameter_px, dish_clipped=bool(
            well.x0 < 0 or well.y0 < 0 or well.x1 > width or well.y1 > height),
        px_per_mm=px_per_mm, scale_source=scale.source if scale else "unknown",
        polarity=found["polarity"],
        mean_area_px=float(areas.mean()) if count else None,
        median_area_px=float(np.median(areas)) if count else None,
        median_diameter_px=float(np.median(diameters)) if count else None,
        mean_area_mm2=float(areas.mean()) / px_per_mm ** 2 if count and px_per_mm else None,
        median_diameter_mm=float(np.median(diameters)) / px_per_mm if count and px_per_mm else None,
        covered_fraction=float(areas.sum() / dish_area) if dish_area > 0 else None)
    return dict(summary=summary, colonies=colonies, labels=found["labels"],
                image=_to_working_size(np.asarray(crop))[0], well=well,
                method=method, polarity=found["polarity"],
                centre=found["centre"], radius=found["radius"],
                factor=found["factor"], offset=(x_off, y_off))


def _colony_overlay_figure(result: Dict[str, Any], title: str = ""):
    """The dish with each counted colony outlined, as a matplotlib figure.

    :param result: what :func:`_count_colony_plate` returned.
    :param title: shown above the image, with the count and its flag.
    :returns: the figure.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from .figures.style import _figure_axes

    image = np.asarray(result["image"])
    shown = image.astype(np.float32)
    if shown.ndim == 2:
        shown = np.stack([shown] * 3, axis=-1)
    shown = shown[..., :3]
    top = float(shown.max()) or 1.0
    shown = np.clip(shown / top, 0, 1)
    edges = find_boundaries(result["labels"], mode="outer")
    shown[edges] = (0.1, 1.0, 0.2)
    with _figure_axes(figsize=(6, 6)) as (figure, axis):
        axis.imshow(shown)
        cx, cy = result["centre"]
        axis.add_patch(plt.Circle((cx, cy), result["radius"], fill=False,
                                  color=(1.0, 0.85, 0.0), linewidth=1))
        summary = result["summary"]
        axis.set_title(f"{title}  {summary['colony_count']} colonies "
                       f"({summary['count_flag']})".strip(), fontsize=9)
        axis.set_axis_off()
    return figure


def _colony_size_figure(colonies: Sequence[Dict[str, Any]]):
    """The distribution of colony diameters across every counted plate.

    :param colonies: per-colony rows from :func:`_measure_colonies`, with
        ``diameter_mm`` when a scale was known and ``diameter_px`` always.
    :returns: the figure; millimetres when every colony has a scale.
    """
    from .figures.style import _figure_axes

    physical = bool(colonies) and all(
        row.get("diameter_mm") is not None for row in colonies)
    key = "diameter_mm" if physical else "diameter_px"
    values = np.array([row[key] for row in colonies], float)
    with _figure_axes(figsize=(5, 3.5)) as (figure, axis):
        if values.size:
            axis.hist(values, bins=min(50, max(5, int(np.sqrt(values.size)))),
                      color="#4c78a8")
            axis.axvline(float(np.median(values)), color="#e45756", linewidth=1)
        axis.set_xlabel("colony diameter (mm)" if physical else "colony diameter (px)")
        axis.set_ylabel("colonies")
        axis.set_title(f"{values.size} colonies", fontsize=9)
        figure.tight_layout()
    return figure
