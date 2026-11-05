"""Plaque v6's training-time well normalization, kept as the test reference.

Vendored from the ``execution/normalize.py`` of the v6 training run
(plaque-v6-20261024, archived on the NAS) so that
``tests/test_plaque_well_resize.py`` can compare spaCR's well resize with the
transform the model was trained on, pixel for pixel, on every machine.

Only the transforms are kept. Image reading, loading the YOLO detector from
disk and the convenience generator were removed; ``detect_wells`` takes the
detector as an argument. Every remaining function body is unchanged, so do not
"fix" this file: it is the record of what training did.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

SIDE = 1000
DETECTOR = dict(
    confidence=0.35, padding_fraction=0.025, imgsz=640,
    min_box_px=12, aspect_range=(0.55, 1.8), merge_overlap=0.4)


def detector_input(image):
    """8-bit BGR view used only for well detection (matches the 2026-10-23 geometry run)."""
    if image.dtype != np.uint8:
        low, high = np.percentile(image, [1, 99.8])
        image = (np.clip((image.astype(np.float64) - low) / max(1, high - low), 0, 1) * 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image[..., ::-1])  # RGB -> BGR (cv2.imread order)


# ---------------------------------------------------------------- detection
def merge_boxes(records, shape):
    """Merge duplicate overlapping hypotheses; keep near-square single-well images whole."""
    height, width = shape
    boxes = [dict(box=list(r['box']), confidence=r['confidence']) for r in records]
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(boxes):
            x0, y0, x1, y1 = a['box']
            area = (x1 - x0) * (y1 - y0)
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                u0, v0, u1, v1 = b['box']
                other = (u1 - u0) * (v1 - v0)
                inter = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
                if inter / max(1, min(area, other)) >= DETECTOR['merge_overlap']:
                    scores = [s for s in (a['confidence'], b['confidence']) if s is not None]
                    boxes[i] = dict(box=[min(x0, u0), min(y0, v0), max(x1, u1), max(y1, v1)],
                                    confidence=max(scores) if scores else None)
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    if len(boxes) == 1 and .7 <= width / max(1, height) <= 1.43:
        x0, y0, x1, y1 = boxes[0]['box']
        area = (x1 - x0) * (y1 - y0) / (height * width)
        centered = abs((x0 + x1) / 2 - width / 2) < .2 * width and abs((y0 + y1) / 2 - height / 2) < .2 * height
        if area >= .4 and centered:
            boxes[0]['box'] = [0, 0, width, height]
            boxes[0]['whole_frame_rule'] = True
    return sorted(boxes, key=lambda b: (b['box'][1], b['box'][0]))


def detect_wells(image, model):
    """Return {'boxes': [{'box': [x0,y0,x1,y1], 'confidence': float|None}], 'fallback_to_frame': bool}."""
    bgr = detector_input(image)
    h, w = bgr.shape[:2]
    prediction = model.predict(bgr, imgsz=DETECTOR['imgsz'], device='cpu',
                               conf=DETECTOR['confidence'], verbose=False)[0]
    boxes = []
    for xyxy, score in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
        x0, y0, x1, y1 = map(float, xyxy)
        bw, bh = x1 - x0, y1 - y0
        lo, hi = DETECTOR['aspect_range']
        if min(bw, bh) < DETECTOR['min_box_px'] or not lo < bw / max(1, bh) < hi:
            continue
        pad = DETECTOR['padding_fraction'] * max(bw, bh)
        boxes.append(dict(box=[max(0, int(math.floor(x0 - pad))), max(0, int(math.floor(y0 - pad))),
                               min(w, int(math.ceil(x1 + pad))), min(h, int(math.ceil(y1 + pad)))],
                          confidence=float(score)))
    fallback = not boxes
    if fallback:
        boxes = [dict(box=[0, 0, w, h], confidence=None)]
    return dict(boxes=merge_boxes(boxes, (h, w)), fallback_to_frame=fallback, shape=[h, w])


# ---------------------------------------------------------------- transforms
def parse_rule(rule):
    """Scale rule string -> dict. Supported (comma-combinable):
      'side:N'   longest side of the well crop becomes N px
      'factor:F' fixed isotropic enlargement factor F (1 = native pixels)
      'max:N'    cap: longest side never exceeds N px (applied after side/factor)
      'min:N'    floor: longest side at least N px (applied after side/factor)
      'maxfactor:F' cap on enlargement: scale = min(scale, F) (applied last; shrinking unaffected)
      'native'   same as factor:1
    Canonical v6 rule (maintainer, 2026-10-24): 'side:1500,maxfactor:6'.
    """
    if isinstance(rule, (int, float)):
        return {'side': float(rule)}
    out = {}
    for part in str(rule).split(','):
        part = part.strip()
        if part == 'native':
            out['factor'] = 1.0
            continue
        key, value = part.split(':')
        if key not in ('side', 'factor', 'max', 'min', 'maxfactor'):
            raise ValueError(f'Unknown scale rule {part!r}')
        out[key] = float(value)
    if ('side' in out) == ('factor' in out):
        raise ValueError(f'Rule {rule!r} needs exactly one of side:/factor:/native')
    return out


def scale_for(width, height, rule):
    r = parse_rule(rule)
    longest = max(width, height)
    ratio = r['side'] / longest if 'side' in r else r['factor']
    if 'max' in r:
        ratio = min(ratio, r['max'] / longest)
    if 'min' in r:
        ratio = max(ratio, r['min'] / longest)
    if 'maxfactor' in r:
        ratio = min(ratio, r['maxfactor'])
    return ratio


def forward(image, box, side=SIDE, labels=None):
    """Crop `box` and resize isotropically by the scale rule `side`.

    `side` may be a number (longest side in px, the original behaviour) or a rule
    string understood by parse_rule ('side:1000', 'factor:4', 'factor:4,max:2048', 'native').

    No letterbox is applied (Cellpose accepts any rectangle). Area interpolation
    when shrinking, bilinear when enlarging, nearest-neighbour for labels.
    Returns (crop, label_crop_or_None, meta). Original coordinate (y, x) maps to
    crop coordinate ((y - y0) * scale_y, (x - x0) * scale_x).
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f'Empty box {box}')
    ratio = scale_for(width, height, side)
    nw, nh = max(1, round(width * ratio)), max(1, round(height * ratio))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    if abs(nw / width - 1) < 1e-12 and abs(nh / height - 1) < 1e-12:
        interpolation = None  # native pixels: exact copy, no resampling
    region = np.ascontiguousarray(image[y0:y1, x0:x1])
    if region.dtype not in (np.uint8, np.uint16, np.float32):
        region = region.astype(np.float32)
    crop = region.copy() if interpolation is None else cv2.resize(region, (nw, nh), interpolation=interpolation)
    if image.ndim == 3 and crop.ndim == 2:
        crop = crop[..., None]
    out_labels = None
    if labels is not None:
        # OpenCV cannot resize uint32; float64 represents every uint32 id exactly.
        out_labels = labels[y0:y1, x0:x1].astype(np.uint32) if interpolation is None else cv2.resize(
            labels[y0:y1, x0:x1].astype(np.float64), (nw, nh),
            interpolation=cv2.INTER_NEAREST_EXACT).astype(np.uint32)
    meta = dict(box=[x0, y0, x1, y1], rule=str(side), width=nw, height=nh,
                scale_x=nw / width, scale_y=nh / height, ratio=ratio,
                interpolation='none' if interpolation is None else 'area' if ratio < 1 else 'linear')
    return crop, out_labels, meta


def inverse(mask, meta):
    """Map a crop-space label image back to the box in original pixels (nearest)."""
    x0, y0, x1, y1 = meta['box']
    if mask.shape[:2] != (meta['height'], meta['width']):
        raise ValueError('Mask does not match the forward transform')
    return cv2.resize(mask.astype(np.float64), (x1 - x0, y1 - y0),
                      interpolation=cv2.INTER_NEAREST_EXACT).astype(np.uint32)


def paste(full, mask, meta):
    """Add the inverse-mapped instances of one well into `full` (original pixels).

    Existing labels win where crops overlap; new ids are offset to stay unique.
    """
    region = inverse(mask, meta)
    x0, y0, x1, y1 = meta['box']
    ids = np.unique(region)
    ids = ids[ids > 0]
    if not len(ids):
        return full
    lut = np.zeros(int(region.max()) + 1, np.uint32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.uint32) + np.uint32(full.max())
    target = full[y0:y1, x0:x1]
    new = lut[region]
    free = (target == 0) & (new > 0)
    target[free] = new[free]
    return full


def normalize_intensity(crop):
    """Channels-first float32 (3 x h x w); per-channel 1st/99th percentile scaling.

    Gray input occupies channel 0 with two zero channels; RGB keeps its three
    channels. Mirrors cellpose.transforms.normalize99 per channel.
    """
    return apply_intensity(crop, intensity_params(crop))


def _channels_first(crop):
    x = np.asarray(crop, dtype=np.float32)
    if x.ndim == 2:
        return np.stack([x, np.zeros_like(x), np.zeros_like(x)])
    x = np.moveaxis(x, -1, 0)
    if x.shape[0] < 3:
        x = np.concatenate([x, np.zeros((3 - x.shape[0], *x.shape[1:]), np.float32)])
    return x[:3].copy()


def intensity_params(crop):
    """Per-channel (low, high) = 1st/99th percentiles of the 3-channel float32 view."""
    x = _channels_first(crop)
    return [tuple(np.percentile(x[c], [1, 99])) for c in range(3)]


def apply_intensity(crop, params):
    """Scale with precomputed params; identical to normalize_intensity(crop) for its own params."""
    x = _channels_first(crop)
    for c in range(3):
        lo, hi = params[c]
        if hi - lo > 1e-3:
            x[c] = (x[c] - lo) / (hi - lo)
        else:
            x[c] = 0
    return x
