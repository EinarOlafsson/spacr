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
            x0, y0, x1, y1 = (float(v) for v in np.asarray(box.xyxy).ravel()[:4])
            score = float(np.asarray(box.conf).ravel()[0]) if box.conf is not None else 1.0
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


def segment_plaque_image(model: Any, image: np.ndarray,
                         settings: Dict[str, Any]) -> np.ndarray:
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
    :returns: the label image.
    """
    from .spacr_cellpose import cellpose_channel_axis

    diameter = _number(settings, "diameter", None)
    output = model.eval(image, channel_axis=cellpose_channel_axis(image),
                        diameter=diameter if diameter else None,
                        flow_threshold=_number(settings, "flow_threshold", 0.4),
                        cellprob_threshold=_number(settings, "CP_prob", 0.0))
    return np.asarray(output[0])
