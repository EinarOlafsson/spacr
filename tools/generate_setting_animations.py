"""Generate deterministic setting animations and review sheets for spaCR.

The shipped GIFs and manifest are written below ``spacr/resources``. Local
contact sheets, storyboards and the searchable review gallery are written to
the ignored ``build`` directory. The visual grammar deliberately uses a black
canvas, a rounded field boundary and restrained biological illustrations.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import textwrap
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "spacr" / "resources" / "setting_animations"
REVIEW_ROOT = REPOSITORY / "build" / "setting_animation_review"
TEMPLATE_ROOT = REPOSITORY / "tools" / "setting_animation_templates"
ASSETS = ROOT / "gifs"
SHEETS = REVIEW_ROOT / "contact_sheets"
STORYBOARDS = REVIEW_ROOT / "storyboards"
DOCS_PAGE = REPOSITORY / "docs" / "source" / "setting_animations.rst"
# Scene geometry retains the original 3:2 logical coordinate system. It is
# vertically centred in a square output so the reviewed motion paths remain
# stable while every animation has the requested 1:1 footprint.
W, H, CANVAS, SCALE = 360, 240, 360, 4
Y_OFFSET = (CANVAS - H) / 2.0
FRAMES = 28
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (78, 145, 255)
TEAL = (43, 205, 190)
MAGENTA = (226, 139, 199)
GRAY = (105, 112, 122)
LINE_SCALE = 1.35
FILL_ALPHA = 0.16

OBJECT_COLORS = {
    "cell": WHITE,
    "nucleus": BLUE,
    "pathogen": TEAL,
    "organelle": MAGENTA,
}

SVG_TEMPLATES = {
    "cell": "cell.svg",
    "nucleus": "nucleus.svg",
    "pathogen": "pathogen.svg",
    "organelle": "golgi.svg",
}

# Existing scene sizes were tuned before the supplied pathogen and Golgi were
# redrawn on their correct tall/compact view boxes. These factors preserve the
# relative dimensions in the user's combined ``cell_all.svg`` reference.
SVG_DRAW_SCALES = {
    "cell": 1.0,
    "nucleus": 1.0,
    "pathogen": 1.85,
    "organelle": 1.50,
}


@dataclass(frozen=True)
class Spec:
    """One shipped GIF and the exact setting keys that link to it."""

    slug: str
    title: str
    category: str
    scene: str
    settings: Tuple[str, ...]
    params: Dict[str, Any] = field(default_factory=dict)


def _mix(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(round(channel * amount)) for channel in color)


def _smooth(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _cycle(index: int, count: int = FRAMES) -> float:
    """Hold before/after and return smoothly to the start for a clean loop."""
    phase = index / max(1, count - 1)
    if phase < 0.16:
        return 0.0
    if phase < 0.42:
        return _smooth((phase - 0.16) / 0.26)
    if phase < 0.68:
        return 1.0
    return 1.0 - _smooth((phase - 0.68) / 0.32)


def _blob_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    phase: float = 0.0,
    roughness: float = 0.055,
    count: int = 72,
) -> List[Tuple[float, float]]:
    points = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        modulation = (
            1.0
            + roughness * math.sin(3.0 * angle + phase)
            + roughness * 0.55 * math.sin(5.0 * angle - phase * 0.7)
        )
        points.append((
            cx + rx * modulation * math.cos(angle),
            cy + ry * modulation * math.sin(angle),
        ))
    return points


@lru_cache(maxsize=None)
def _template_text(kind: str) -> str:
    """Return one trusted, repository-owned SVG template as text."""
    try:
        filename = SVG_TEMPLATES[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown SVG object template: {kind!r}") from exc
    path = TEMPLATE_ROOT / filename
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    forbidden = ("<image", "<script", "<use", "xlink:href", "<!entity")
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(
            f"Unsafe external/dynamic SVG content in {path}: {', '.join(found)}"
        )
    return text


@lru_cache(maxsize=None)
def _template_view_box(kind: str) -> Tuple[float, float]:
    """Return the intrinsic width and height of one SVG template."""
    match = re.search(r'viewBox="([^"]+)"', _template_text(kind))
    if match is None:
        raise ValueError(f"{SVG_TEMPLATES[kind]} has no SVG viewBox")
    values = [float(value) for value in match.group(1).replace(",", " ").split()]
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"{SVG_TEMPLATES[kind]} has an invalid SVG viewBox")
    return values[2], values[3]


def _svg_hex(colour: Tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in colour)


def _styled_template(kind: str, source_stroke_width: float) -> bytes:
    """Apply spaCR colours/opacity while preserving supplied SVG paths."""
    colour = _svg_hex(OBJECT_COLORS[kind])
    overrides = {
        "cell": (
            f".cls-1{{fill:{colour};fill-opacity:{FILL_ALPHA:.3f};"
            f"stroke:{colour};}}"
        ),
        "nucleus": (
            f".cls-1{{fill:none;stroke:{colour};}}"
            f".cls-2{{fill:{colour};fill-opacity:.18;}}"
            f".cls-3{{fill:{colour};fill-opacity:.68;}}"
        ),
        "pathogen": (
            f".cls-1{{fill:{colour};fill-opacity:.68;stroke:none;}}"
            f".cls-2{{fill:none;stroke:{colour};}}"
        ),
        "organelle": (
            f".cls-1{{fill:{colour};fill-opacity:.46;stroke:{colour};}}"
        ),
    }[kind]
    text = _template_text(kind)
    if "</style>" not in text:
        raise ValueError(f"{SVG_TEMPLATES[kind]} has no style element")
    text = text.replace("</style>", overrides + "</style>", 1)
    stroke_pattern = r"stroke-width:\s*(?:\d+(?:\.\d*)?|\.\d+)px"
    text, replacements = re.subn(
        stroke_pattern,
        f"stroke-width: {source_stroke_width:.6f}px",
        text,
    )
    if replacements == 0:
        raise ValueError(f"{SVG_TEMPLATES[kind]} has no scalable stroke width")
    return text.encode("utf-8")


@lru_cache(maxsize=384)
def _render_svg_sprite(
    kind: str,
    pixel_width: int,
    pixel_height: int,
    logical_stroke_milli: int,
) -> Image.Image:
    """Render an exact SVG template into a cached transparent RGBA sprite."""
    from PySide6.QtCore import QByteArray, QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    pixel_width = max(2, int(pixel_width))
    pixel_height = max(2, int(pixel_height))
    view_width, view_height = _template_view_box(kind)
    pixels_per_source_unit = min(
        pixel_width / view_width,
        pixel_height / view_height,
    )
    logical_stroke = logical_stroke_milli / 1000.0
    device_stroke = logical_stroke * LINE_SCALE * SCALE
    source_stroke = device_stroke / max(1.0e-9, pixels_per_source_unit)
    renderer = QSvgRenderer(QByteArray(_styled_template(kind, source_stroke)))
    if not renderer.isValid():
        raise ValueError(f"Could not render SVG template {SVG_TEMPLATES[kind]}")

    image = QImage(
        pixel_width,
        pixel_height,
        QImage.Format_ARGB32_Premultiplied,
    )
    image.fill(0)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        renderer.render(
            painter,
            QRectF(0.0, 0.0, float(pixel_width), float(pixel_height)),
        )
    finally:
        painter.end()

    rgba = image.convertToFormat(QImage.Format_RGBA8888)
    data = bytes(rgba.constBits())
    expected = pixel_width * pixel_height * 4
    if len(data) < expected:
        raise ValueError(f"Short Qt SVG render for {SVG_TEMPLATES[kind]}")
    return Image.frombytes(
        "RGBA",
        (pixel_width, pixel_height),
        data[:expected],
        "raw",
        "RGBA",
    )


def _template_hashes() -> Dict[str, str]:
    """Return SHA-256 hashes for every checked-in source SVG."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(TEMPLATE_ROOT.glob("*.svg"))
    }


def _validate_templates() -> Dict[str, str]:
    """Validate all supplied SVGs before replacing generated assets."""
    expected = {
        "all.svg",
        "cell.svg",
        "cell_all.svg",
        "golgi.svg",
        "nucleus.svg",
        "pathogen.svg",
    }
    present = {path.name for path in TEMPLATE_ROOT.glob("*.svg")}
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise ValueError(
            "Setting-animation SVG templates do not match the reviewed set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for kind in SVG_TEMPLATES:
        view_width, view_height = _template_view_box(kind)
        if view_width >= view_height:
            pixel_width = 256
            pixel_height = max(2, int(round(256 * view_height / view_width)))
        else:
            pixel_height = 256
            pixel_width = max(2, int(round(256 * view_width / view_height)))
        sprite = _render_svg_sprite(kind, pixel_width, pixel_height, 500)
        if sprite.getbbox() is None:
            raise ValueError(f"SVG template {SVG_TEMPLATES[kind]} rendered empty")
    return _template_hashes()


class Painter:
    """Supersampled renderer for square scientific diagrams."""

    def __init__(self):
        self.image = Image.new("RGB", (CANVAS * SCALE, CANVAS * SCALE), BLACK)
        self.draw = ImageDraw.Draw(self.image)

    @staticmethod
    def point(point: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(round(point[0] * SCALE)),
            int(round((point[1] + Y_OFFSET) * SCALE)),
        )

    @staticmethod
    def width(value: float) -> int:
        return max(1, int(round(value * LINE_SCALE * SCALE)))

    @staticmethod
    def box(
        box: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = box
        return tuple(int(round(value * SCALE)) for value in (
            left, top + Y_OFFSET, right, bottom + Y_OFFSET,
        ))

    def line(
        self,
        points: Sequence[Tuple[float, float]],
        color: Tuple[int, int, int] = WHITE,
        width: float = 0.5,
        closed: bool = False,
        fill_closed: bool = True,
    ) -> None:
        pts = list(points)
        if not pts or max(color) <= 2:
            return
        if closed and fill_closed and len(pts) >= 3:
            self.draw.polygon(
                [self.point(point) for point in pts],
                fill=_mix(color, FILL_ALPHA),
            )
        if closed and pts:
            pts.append(pts[0])
        self.draw.line(
            [self.point(point) for point in pts],
            fill=color,
            width=self.width(width),
            joint="curve",
        )

    def polygon(
        self,
        points: Sequence[Tuple[float, float]],
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int] | None = None,
        width: float = 0.5,
    ) -> None:
        """Draw a filled polygon with an optional anti-aliased outline."""
        pts = list(points)
        if len(pts) < 3:
            return
        self.draw.polygon([self.point(point) for point in pts], fill=fill)
        if outline is not None:
            self.line(pts, outline, width, closed=True, fill_closed=False)

    def ellipse(
        self,
        box: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        width: float = 0.5,
    ) -> None:
        self.draw.ellipse(
            self.box(box),
            outline=color,
            width=self.width(width),
        )

    def rectangle(
        self,
        box: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        width: float = 0.5,
        radius: float = 0.0,
    ) -> None:
        values = self.box(box)
        if radius > 0:
            self.draw.rounded_rectangle(
                values,
                radius=int(round(radius * SCALE)),
                outline=color,
                width=self.width(width),
            )
        else:
            self.draw.rectangle(values, outline=color, width=self.width(width))

    def canvas_rectangle(
        self,
        box: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        width: float = 0.5,
        radius: float = 0.0,
    ) -> None:
        """Draw a frame in square-output coordinates without scene offset."""
        values = tuple(int(round(value * SCALE)) for value in box)
        if radius > 0:
            self.draw.rounded_rectangle(
                values,
                radius=int(round(radius * SCALE)),
                outline=color,
                width=self.width(width),
            )
        else:
            self.draw.rectangle(values, outline=color, width=self.width(width))

    def dot(
        self,
        center: Tuple[float, float],
        radius: float,
        color: Tuple[int, int, int],
    ) -> None:
        cx, cy = center
        self.draw.ellipse(
            self.box((cx - radius, cy - radius, cx + radius, cy + radius)),
            fill=color,
        )

    def dashed(
        self,
        points: Sequence[Tuple[float, float]],
        color: Tuple[int, int, int],
        width: float = 0.45,
        dash: float = 4.0,
        gap: float = 3.0,
    ) -> None:
        for start, end in zip(points, points[1:]):
            x1, y1 = start
            x2, y2 = end
            length = math.hypot(x2 - x1, y2 - y1)
            if length <= 0:
                continue
            ux, uy = (x2 - x1) / length, (y2 - y1) / length
            cursor = 0.0
            while cursor < length:
                stop = min(length, cursor + dash)
                self.line(
                    [(x1 + ux * cursor, y1 + uy * cursor),
                     (x1 + ux * stop, y1 + uy * stop)],
                    color,
                    width,
                )
                cursor += dash + gap

    def svg_object(
        self,
        kind: str,
        center: Tuple[float, float],
        size: Tuple[float, float],
        amount: float = 1.0,
        angle: float = 0.0,
        mirror_vertical: bool = False,
        width: float = 0.5,
        tint: Tuple[int, int, int] | None = None,
    ) -> None:
        """Composite one supplied SVG object template onto the scene.

        SVG paths are rendered directly by Qt at the supersampled target size;
        Pillow is used only for affine transforms and final compositing. This
        avoids the faceted outlines produced by sampled polygon stand-ins.
        """
        amount = max(0.0, min(1.0, amount))
        if amount <= 0.01:
            return
        rx, ry = size
        draw_scale = SVG_DRAW_SCALES[kind]
        max_width = max(2, int(round(2.0 * rx * draw_scale * SCALE)))
        max_height = max(2, int(round(2.0 * ry * draw_scale * SCALE)))
        view_width, view_height = _template_view_box(kind)
        aspect = view_width / view_height
        if max_width / max_height > aspect:
            pixel_height = max_height
            pixel_width = max(2, int(round(pixel_height * aspect)))
        else:
            pixel_width = max_width
            pixel_height = max(2, int(round(pixel_width / aspect)))

        sprite = _render_svg_sprite(
            kind,
            pixel_width,
            pixel_height,
            int(round(width * 1000.0)),
        ).copy()
        if tint is not None:
            alpha = sprite.getchannel("A")
            sprite = Image.new("RGBA", sprite.size, (*tint, 0))
            sprite.putalpha(alpha)
        if mirror_vertical:
            sprite = ImageOps.flip(sprite)
        if abs(angle) > 1.0e-6:
            sprite = sprite.rotate(
                math.degrees(angle),
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
        if amount < 0.999:
            alpha = sprite.getchannel("A")
            lookup = [int(round(value * amount)) for value in range(256)]
            sprite.putalpha(alpha.point(lookup))

        cx, cy = self.point(center)
        destination = (
            int(round(cx - sprite.width / 2.0)),
            int(round(cy - sprite.height / 2.0)),
        )
        self.image.paste(sprite, destination, sprite)

    def finish(self) -> Image.Image:
        return self.image.resize((CANVAS, CANVAS), Image.Resampling.LANCZOS)


def _well(painter: Painter, color: Tuple[int, int, int] = WHITE) -> None:
    painter.canvas_rectangle(
        (12, 12, CANVAS - 12, CANVAS - 12), color, 0.75, 20,
    )


def _transform_points(
    points: Sequence[Tuple[float, float]],
    center: Tuple[float, float],
    angle: float = 0.0,
) -> List[Tuple[float, float]]:
    """Rotate local points by ``angle`` and translate them to ``center``."""
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = center
    return [
        (cx + x * cosine - y * sine, cy + x * sine + y * cosine)
        for x, y in points
    ]


def _dendritic_points(
    center: Tuple[float, float],
    size: Tuple[float, float],
    phase: float,
) -> List[Tuple[float, float]]:
    """Return a motile immune-cell soma with directional dendritic processes."""
    rx, ry = size
    spike_directions = (0.05, 2.15, 4.35)
    spike_strengths = (0.55, 0.40, 0.34)

    def circular_distance(left: float, right: float) -> float:
        return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)

    local = []
    for index in range(56):
        angle = 2.0 * math.pi * index / 56.0
        radius = 0.70 + 0.055 * math.sin(4.0 * angle + phase)
        for direction, strength in zip(spike_directions, spike_strengths):
            distance = circular_distance(angle, direction)
            radius += strength * math.exp(-0.5 * (distance / 0.20) ** 2)
        local.append((rx * radius * math.cos(angle),
                      ry * radius * math.sin(angle)))
    return _transform_points(local, center, 0.12 * math.sin(phase))


def _draw_motile_cell(
    painter: Painter,
    center: Tuple[float, float],
    size: Tuple[float, float],
    amount: float = 1.0,
    phase: float = 0.0,
    width: float = 0.5,
) -> None:
    """Draw a translucent migrating dendritic-cell silhouette."""
    if amount <= 0.01:
        return
    colour = _mix(WHITE, amount)
    painter.polygon(
        _dendritic_points(center, size, phase),
        _mix(WHITE, FILL_ALPHA * amount),
        colour,
        width,
    )


def _drawn_half_size(
    kind: str,
    size: Tuple[float, float],
) -> Tuple[float, float]:
    """Return the logical half width and height an object actually occupies.

    ``size`` is the requested half size; ``Painter.svg_object`` keeps the
    template's own aspect ratio inside it, so the drawn extent is usually
    smaller in one axis. Scenes that annotate an object -- the caliper in
    :func:`_diameter_scene` -- need the drawn extent, not the request.
    """
    half_width, half_height = (
        value * SVG_DRAW_SCALES[kind] for value in size
    )
    view_width, view_height = _template_view_box(kind)
    aspect = view_width / view_height
    if half_width / half_height > aspect:
        half_width = half_height * aspect
    else:
        half_height = half_width / aspect
    return half_width, half_height


def _object_outline(
    painter: Painter,
    kind: str,
    center: Tuple[float, float],
    size: Tuple[float, float],
    amount: float = 1.0,
    phase: float = 0.0,
    width: float = 0.5,
) -> None:
    amount = max(0.0, min(1.0, amount))
    if amount <= 0.01:
        return
    if kind not in SVG_TEMPLATES:
        raise ValueError(f"Unknown biological object kind: {kind!r}")
    if kind == "cell":
        angle = 0.075 * math.sin(phase * 0.91)
        mirror_vertical = math.sin(phase * 1.31) < -0.12
    elif kind == "nucleus":
        angle = 0.14 * math.sin(phase + 0.35)
        mirror_vertical = False
    elif kind == "pathogen":
        angle = 0.075 * math.sin(phase + 0.4)
        mirror_vertical = False
    else:
        angle = -0.08 + 0.025 * math.sin(phase)
        mirror_vertical = False
    painter.svg_object(
        kind,
        center,
        size,
        amount=amount,
        angle=angle,
        mirror_vertical=mirror_vertical,
        width=width,
    )


def _filter_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    variant = spec.params["variant"]
    sizes = [(15, 12), (22, 17), (30, 23), (40, 30)]
    centers = [(72, 72), (155, 65), (258, 75), (180, 166)]
    if kind == "cell":
        sizes = [(25, 20), (35, 27), (45, 34), (56, 42)]
        centers = [(73, 73), (165, 64), (278, 76), (180, 166)]
    # These settings are THRESHOLDS: everything past the threshold is
    # discarded, not one representative object. Fading a single object made
    # the `minimum` and `dim` variants measure a fifth of a percent -- the
    # object they fade is the smallest one -- and taught the viewer that the
    # setting drops one thing. A threshold sweeping through the objects in
    # order is both the larger change and the true illustration.
    by_intensity = variant in ("dim", "bright")
    # Intensity variants need visibly different brightnesses for an order to
    # exist at all; area variants read their order off the sizes.
    base = [0.4, 0.6, 0.8, 1.0] if by_intensity else [1.0] * len(sizes)
    if by_intensity:
        rank_by = lambda index: base[index]
    else:
        rank_by = lambda index: sizes[index][0] * sizes[index][1]
    order = sorted(
        range(len(sizes)), key=rank_by,
        reverse=variant in ("maximum", "bright"),
    )
    for rank, index in enumerate(order):
        # One object always survives; an empty frame reads as a broken render
        # rather than as a filter.
        gone = min(1.0, max(0.0, action * (len(order) - 1) - rank))
        _object_outline(
            painter, kind, centers[index], sizes[index],
            base[index] * (1.0 - gone),
            phase=index * 0.8, width=0.5,
        )


def _assert_border_layout(
    objects: Sequence[Tuple[Tuple[float, float], Tuple[float, float], bool]],
    close: Tuple[float, float, float, float],
) -> None:
    """Refuse to draw a border scene that contradicts what it illustrates.

    Two independent faults produced the reported "removes one edge object and
    one that is not", and neither is visible in a pixel diff that trusts the
    ``touches`` flag as ground truth:

    * an object flagged as touching the border sat wholly inside the well, so
      the animation removed an interior object;
    * a KEPT object lay outside the close camera, so it left the frame during
      the zoom and read as removed.

    Both are geometry, so both can be checked here rather than watched for.
    """
    edge_low, edge_high = 12.0, float(CANVAS) - 12.0
    left, top, right, bottom = close
    for index, (center, size, touches) in enumerate(objects):
        x0, x1 = center[0] - size[0] / 2.0, center[0] + size[0] / 2.0
        y0, y1 = center[1] - size[1] / 2.0, center[1] + size[1] / 2.0
        straddles = (
            x0 < edge_low < x1 or x0 < edge_high < x1
            or y0 < edge_low < y1 or y0 < edge_high < y1
        )
        if touches and not straddles:
            raise ValueError(
                f"border object {index} is flagged as touching the well edge "
                f"but lies at x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}], which "
                f"does not cross {edge_low:.0f} or {edge_high:.0f}; removing it "
                "would illustrate the opposite of the setting"
            )
        if not touches and straddles:
            raise ValueError(
                f"object {index} is kept but crosses the well edge, so the "
                "animation shows a border object surviving"
            )
        if not (left <= x0 and x1 <= right and top <= y0 and y1 <= bottom):
            raise ValueError(
                f"object {index} at x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}] "
                f"is not contained in the close camera {close}; it would leave "
                "the frame during the zoom and read as removed"
            )


def _border_scene(painter: Painter, spec: Spec, index: int) -> None:
    kind = spec.params["kind"]
    phase = index / max(1, FRAMES - 1)
    if phase < 0.15:
        zoom, remove = 0.0, 0.0
    elif phase < 0.38:
        zoom, remove = _smooth((phase - 0.15) / 0.23), 0.0
    elif phase < 0.50:
        zoom, remove = 1.0, 0.0
    elif phase < 0.68:
        zoom, remove = 1.0, _smooth((phase - 0.50) / 0.18)
    elif phase < 0.80:
        zoom, remove = 1.0, 1.0
    else:
        reverse = _smooth((phase - 0.80) / 0.20)
        zoom, remove = 1.0 - reverse, 1.0 - reverse

    # Camera coordinates are square-output coordinates. Biological scenes use
    # a centred 240 px band, hence object y positions gain Y_OFFSET before the
    # camera transform and lose it again before Painter applies its offset.
    full = (0.0, 0.0, float(CANVAS), float(CANVAS))
    # The close camera must contain every object, including the ones that are
    # KEPT. An object that leaves the frame during the zoom is indistinguishable
    # from one that is removed, which is half of what made this scene wrong.
    close = (228.0, 70.5, 378.0, 220.5)
    camera = tuple(a + (b - a) * zoom for a, b in zip(full, close))
    left, top, right, bottom = camera
    sx, sy = CANVAS / (right - left), CANVAS / (bottom - top)

    def transform(point: Tuple[float, float]) -> Tuple[float, float]:
        return ((point[0] - left) * sx, (point[1] - top) * sy)

    well_top_left = transform((12, 12))
    well_bottom_right = transform((CANVAS - 12, CANVAS - 12))
    painter.rectangle(
        (
            well_top_left[0], well_top_left[1] - Y_OFFSET,
            well_bottom_right[0], well_bottom_right[1] - Y_OFFSET,
        ),
        WHITE,
        0.75,
        20 * min(sx, sy),
    )
    # The two removed objects STRADDLE the well edge; the two kept objects are
    # clear of it. Both facts are asserted below against the drawn rectangle,
    # because the previous layout removed an object that sat wholly inside the
    # well -- the reported "removes one edge object and one that is not".
    objects = [
        ((348, 40 + Y_OFFSET), (30, 23), True),
        ((350, 130 + Y_OFFSET), (27, 22), True),
        ((255, 45 + Y_OFFSET), (24, 19), False),
        ((262, 135 + Y_OFFSET), (18, 15), False),
    ]
    if kind != "cell":
        objects = [
            ((348, 40 + Y_OFFSET), (16, 13), True),
            ((350, 130 + Y_OFFSET), (15, 12), True),
            ((255, 45 + Y_OFFSET), (14, 11), False),
            ((262, 135 + Y_OFFSET), (11, 9), False),
        ]
    _assert_border_layout(objects, close)
    for item_index, (center, size, touches) in enumerate(objects):
        transformed = transform(center)
        _object_outline(
            painter,
            kind,
            (transformed[0], transformed[1] - Y_OFFSET),
            (size[0] * sx, size[1] * sy),
            (1.0 - remove) if touches else 1.0,
            item_index * 0.7,
        )


def _merge_edge_pathogen_cells(painter: Painter, action: float) -> None:
    _well(painter)
    _object_outline(
        painter, "cell", (180, 120), (126, 74), action, 0.6,
    )
    _object_outline(
        painter, "cell", (128, 122), (78, 62), 1.0 - action, 0.2,
    )
    _object_outline(
        painter, "cell", (230, 119), (80, 64), 1.0 - action, 1.0,
    )
    _object_outline(painter, "nucleus", (105, 117), (18, 15), 1.0, 0.2)
    _object_outline(painter, "nucleus", (252, 114), (18, 15), 1.0, 0.8)
    _object_outline(painter, "pathogen", (180, 121), (27, 15), 1.0, 0.5)


def _adjust_cells(painter: Painter, action: float) -> None:
    _well(painter)
    _object_outline(
        painter, "cell", (172, 123), (112, 67), action, 0.7,
    )
    _object_outline(
        painter, "cell", (145, 122), (86, 64), 1.0 - action, 0.4,
    )
    _object_outline(
        painter, "cell", (235, 128), (38, 28), 1.0 - action, 1.1,
    )
    _object_outline(painter, "nucleus", (128, 116), (20, 16), 1.0, 0.4)
    _object_outline(painter, "pathogen", (211, 123), (24, 13), 1.0, 0.8)


def _generic_merge(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    scale = 1.45 if kind == "cell" else 1.0
    _object_outline(
        painter, kind, (176, 120), (67 * scale, 42 * scale), action, 0.5,
    )
    _object_outline(
        painter, kind, (145, 120), (34 * scale, 38 * scale),
        1.0 - action, 0.2,
    )
    _object_outline(
        painter, kind, (207, 120), (34 * scale, 38 * scale),
        1.0 - action, 0.9,
    )
    if spec.params.get("intensity"):
        pulse = 0.4 + 0.6 * math.sin(action * math.pi)
        painter.line(
            [(176, 85), (176, 155)],
            _mix(OBJECT_COLORS[kind], pulse),
            0.7,
        )


def _split_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    scale = 1.45 if kind == "cell" else 1.0
    _object_outline(
        painter, kind, (150, 120), (35 * scale, 39 * scale), action, 0.2,
    )
    _object_outline(
        painter, kind, (210, 120), (35 * scale, 39 * scale), action, 0.9,
    )
    _object_outline(
        painter, kind, (180, 120), (67 * scale, 42 * scale),
        1.0 - action, 0.5,
    )


def _probability_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    centers = [(90, 82), (177, 76), (263, 87), (142, 165), (240, 160)]
    for idx, center in enumerate(centers):
        confidence = (0.28, 0.48, 0.68, 0.86, 1.0)[idx]
        cutoff = 0.25 + action * 0.55
        amount = confidence if confidence >= cutoff else confidence * (1.0 - action)
        radius = (22, 17) if kind == "cell" else (14, 11)
        _object_outline(painter, kind, center, radius, amount, idx * 0.7)


def _flow_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    color = OBJECT_COLORS[kind]
    size = (57, 34) if kind == "cell" else (42, 34)
    _object_outline(painter, kind, (112, 120), size, 1.0, 0.3)
    _object_outline(painter, kind, (248, 120), size, 1.0 - action, 0.8)
    ragged = _blob_points(248, 120, size[0], size[1], 0.8, 0.19)
    painter.line(
        ragged,
        _mix(color, 0.70 * (1.0 - action)),
        0.45,
        True,
        fill_closed=False,
    )


def _diameter_scene(painter: Painter, spec: Spec, action: float) -> None:
    # The setting sizes the OBJECT, so the object is the subject of the
    # animation and the caliper is only the annotation on it. Drawing the
    # outline at a fixed radius and sweeping the caliper alone changed 0.01%
    # of the frame for pathogen_diameter -- the viewer saw nothing move.
    _well(painter)
    kind = spec.params["kind"]
    color = OBJECT_COLORS[kind]
    center = (180, 120)
    radius = 64 if kind == "cell" else 40
    scale = 0.55 + 0.45 * action
    size = (radius * scale, radius * scale * 0.8)
    _object_outline(painter, kind, center, size, 1.0, 0.4)
    extent = _drawn_half_size(kind, size)[0]
    painter.line(
        ((center[0] - extent, center[1]),
         (center[0] + extent, center[1])),
        _mix(color, 0.78),
        0.5,
    )
    for x in (center[0] - extent, center[0] + extent):
        painter.line(
            ((x, center[1] - 5), (x, center[1] + 5)),
            _mix(color, 0.78),
            0.5,
        )


def _background_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    for offset, amount in ((0, 0.36), (12, 0.25), (24, 0.16)):
        painter.line(
            _blob_points(180, 120, 115 - offset, 78 - offset / 2, offset, 0.025),
            _mix(GRAY, amount * (1.0 - action)), 0.5, True,
            fill_closed=False,
        )
    for idx, center in enumerate(((105, 105), (185, 88), (243, 145))):
        _object_outline(
            painter, kind, center,
            (30, 23) if kind == "cell" else (14, 11),
            0.55 + 0.45 * action, idx,
        )


def _signal_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params["kind"]
    for idx, (center, base) in enumerate(zip(
        ((90, 118), (180, 118), (270, 118)), (0.25, 0.55, 1.0))):
        amount = base + (1.0 - base) * action
        _object_outline(
            painter, kind, center,
            (30, 23) if kind == "cell" else (16, 12),
            amount, idx * 0.6,
        )


def _fill_holes(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    kind = spec.params.get("kind", "cell")
    color = OBJECT_COLORS[kind]
    size = (92, 69) if kind == "cell" else (55, 42)
    hole = 23 if kind == "cell" else 14
    _object_outline(painter, kind, (180, 120), size, 1.0, 0.3)
    painter.ellipse(
        (180 - hole, 120 - hole, 180 + hole, 120 + hole),
        _mix(color, 1.0 - action), 0.5,
    )


def _organelle_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    mode = spec.params["mode"]
    if mode == "watershed":
        _object_outline(
            painter, "organelle", (180, 120), (62, 30), 1.0 - action, 0.4,
        )
        _object_outline(
            painter, "organelle", (150, 120), (32, 25), action, 0.2,
        )
        _object_outline(
            painter, "organelle", (210, 120), (32, 25), action, 0.9,
        )
    elif mode == "skeleton":
        _object_outline(
            painter, "organelle", (180, 120), (94, 38), 1.0 - action, 0.5,
        )
        skeleton = [
            (88, 124), (126, 108), (164, 116), (205, 101), (272, 119),
        ]
        painter.line(skeleton, _mix(MAGENTA, action), 0.55)
    elif mode == "rolling_ball":
        for y, strength in ((76, 0.25), (118, 0.35), (166, 0.22)):
            wave = [(x, y + 10 * math.sin(x / 45.0)) for x in range(30, 331, 8)]
            painter.line(wave, _mix(GRAY, strength * (1.0 - action)), 0.5)
        for idx, center in enumerate(((105, 120), (180, 93), (248, 145))):
            _object_outline(painter, "organelle", center, (13, 10), 1.0, idx)
    elif mode == "clahe":
        for idx, center in enumerate(((82, 85), (145, 145), (216, 82), (282, 151))):
            base = 0.25 + idx * 0.18
            _object_outline(
                painter, "organelle", center, (14, 10),
                base + (1.0 - base) * action, idx,
            )
    elif mode == "within_cells":
        _object_outline(painter, "cell", (175, 122), (100, 72), 1.0, 0.4)
        inside = ((125, 104), (181, 84), (225, 139), (158, 159))
        outside = ((51, 69), (303, 78), (307, 180))
        for idx, center in enumerate(inside):
            _object_outline(painter, "organelle", center, (10, 8), 1.0, idx)
        for idx, center in enumerate(outside):
            _object_outline(
                painter, "organelle", center, (10, 8), 1.0 - action, idx,
            )
    elif mode == "threshold":
        for idx, (center, base) in enumerate(zip(
            ((83, 87), (148, 145), (218, 86), (282, 150)),
            (0.25, 0.45, 0.72, 1.0),
        )):
            amount = base if base >= (0.2 + 0.55 * action) else base * (1.0 - action)
            _object_outline(painter, "organelle", center, (13, 10), amount, idx)


def _outline_thickness(painter: Painter, action: float) -> None:
    _well(painter)
    _object_outline(
        painter, "cell", (180, 120), (92, 68), 1.0, 0.4,
        width=0.35 + 0.65 * action,
    )
    _object_outline(
        painter, "nucleus", (160, 112), (20, 16), 1.0, 0.7,
        width=0.35 + 0.65 * action,
    )
    _object_outline(
        painter, "pathogen", (212, 124), (15, 9), 1.0, 0.2,
        width=0.35 + 0.65 * action,
    )


def _normalization(painter: Painter, action: float) -> None:
    _well(painter)
    objects = [
        ("cell", (92, 120), (48, 38), 0.32),
        ("nucleus", (180, 120), (25, 20), 0.58),
        ("pathogen", (264, 120), (20, 12), 1.0),
    ]
    for idx, (kind, center, size, base) in enumerate(objects):
        _object_outline(
            painter, kind, center, size,
            base + (1.0 - base) * action, idx,
        )


def _crop_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    mode = spec.params["mode"]
    _object_outline(painter, "cell", (180, 120), (72, 58), 1.0, 0.4)
    _object_outline(painter, "nucleus", (162, 110), (20, 16), 1.0, 0.7)
    _object_outline(painter, "pathogen", (211, 128), (15, 9), 1.0, 0.2)
    if mode == "bounding_box":
        _object_outline(painter, "cell", (265, 135), (43, 35), action, 1.2)
        painter.rectangle((95, 50, 278, 192), _mix(WHITE, 0.5 + 0.5 * action), 0.5, 5)
    elif mode == "dilate":
        margin = 2 + 18 * action
        painter.line(
            _blob_points(180, 120, 72 + margin, 58 + margin, 0.4, 0.07),
            _mix(WHITE, 0.65), 0.45, True,
        )
    elif mode == "png_size":
        margin = 8 + 38 * action
        painter.rectangle(
            (108 - margin, 62 - margin / 2, 252 + margin, 178 + margin / 2),
            _mix(WHITE, 0.72), 0.5, 5,
        )
    elif mode == "crop_mode":
        _object_outline(painter, "organelle", (195, 92), (9, 7), 1.0, 0.5)
        targets = ((180, 120, 92, 76), (162, 110, 46, 38),
                   (211, 128, 38, 26), (195, 92, 28, 24))
        cycle = (action * 3.0) % 3.0
        index = int(round(cycle))
        cx, cy, width, height = targets[index]
        painter.rectangle(
            (cx - width / 2, cy - height / 2,
             cx + width / 2, cy + height / 2),
            _mix(WHITE, 0.72), 0.5, 4,
        )


def _cytoplasm(painter: Painter, action: float) -> None:
    _well(painter)
    _object_outline(painter, "cell", (180, 120), (92, 69), 1.0, 0.3)
    _object_outline(painter, "nucleus", (154, 111), (22, 18), 1.0, 0.4)
    _object_outline(painter, "pathogen", (215, 128), (16, 10), 1.0, 0.8)
    _object_outline(painter, "organelle", (193, 88), (10, 8), 1.0, 0.2)
    inset = 5 + 3 * action
    painter.line(
        _blob_points(180, 120, 92 - inset, 69 - inset, 0.3, 0.07),
        _mix(WHITE, 0.35 + 0.55 * action), 0.45, True,
    )


def _radial(painter: Painter, action: float) -> None:
    _well(painter)
    _object_outline(painter, "cell", (180, 120), (105, 76), 1.0, 0.4)
    _object_outline(painter, "pathogen", (180, 120), (17, 11), 1.0, 0.2)
    for ring in range(1, 5):
        radius = 18 + ring * (10 + 4 * action)
        painter.ellipse(
            (180 - radius, 120 - radius * 0.65,
             180 + radius, 120 + radius * 0.65),
            _mix(TEAL, 0.25 + ring * 0.1), 0.4,
        )


def _uninfected(painter: Painter, action: float) -> None:
    _well(painter)
    _object_outline(painter, "cell", (112, 120), (62, 52), 1.0 - action, 0.3)
    _object_outline(painter, "nucleus", (105, 115), (18, 15), 1.0 - action, 0.4)
    _object_outline(painter, "cell", (247, 120), (62, 52), 1.0, 0.9)
    _object_outline(painter, "nucleus", (238, 113), (18, 15), 1.0, 0.7)
    _object_outline(painter, "pathogen", (266, 128), (14, 8), 1.0, 0.2)


def _track_path(
    painter: Painter,
    positions: Sequence[Tuple[float, float]],
    amount: float = 1.0,
    dashed: bool = False,
) -> None:
    color = _mix(WHITE, amount)
    if dashed:
        painter.dashed(positions, color, 0.45)
    else:
        painter.line(positions, color, 0.45)
    for idx, position in enumerate(positions):
        _draw_motile_cell(
            painter,
            position,
            (16, 13),
            amount * (0.32 + 0.68 * (idx + 1) / len(positions)),
            idx,
        )


def _tracking_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    mode = spec.params["mode"]
    if mode == "transient":
        _track_path(painter, ((60, 82), (110, 87), (160, 90), (210, 94), (270, 98)))
        _track_path(
            painter, ((90, 168), (138, 155), (181, 161)), 1.0 - action,
        )
    elif mode == "displacement":
        start = (120, 120)
        near = (190, 110)
        far = (278, 151)
        radius = 45 + 85 * action
        painter.ellipse(
            (start[0] - radius, start[1] - radius,
             start[0] + radius, start[1] + radius),
            _mix(WHITE, 0.35), 0.4,
        )
        _draw_motile_cell(painter, start, (22, 18), 0.55, 0.2)
        _draw_motile_cell(painter, near, (22, 18), 1.0, 0.7)
        _draw_motile_cell(painter, far, (22, 18), 1.0, 1.1)
        target = far if action > 0.72 else near
        painter.dashed((start, target), _mix(WHITE, 0.7), 0.45)
    elif mode == "memory":
        positions = ((65, 126), (112, 112), (160, 105), (210, 111), (270, 127))
        _track_path(painter, positions, 1.0, dashed=bool(action > 0.45))
        # Missing middle detection; memory bridges the gap.
        _draw_motile_cell(
            painter, positions[2], (16, 13), 1.0 - action, 2,
        )
    elif mode in ("link", "stitch"):
        first = (145, 120)
        second = (190 + 38 * action, 120)
        draw_cell = _draw_motile_cell if mode == "link" else _object_outline
        if mode == "link":
            draw_cell(painter, first, (42, 32), 0.6, 0.2)
            draw_cell(painter, second, (42, 32), 1.0, 0.8)
        else:
            draw_cell(painter, "cell", first, (42, 32), 0.6, 0.2)
            draw_cell(painter, "cell", second, (42, 32), 1.0, 0.8)
        overlap = max(0.0, 1.0 - action * 1.15)
        painter.dashed((first, second), _mix(WHITE, 0.25 + 0.65 * overlap), 0.45)
    elif mode == "projection":
        for idx, offset in enumerate((-24, -12, 0, 12, 24)):
            _object_outline(
                painter, "cell", (180 + offset * (1.0 - action),
                                  120 - offset * 0.35 * (1.0 - action)),
                (46, 34), 0.28 + idx * 0.12, idx,
            )
    elif mode == "project_tracking":
        separation = 28 * (1.0 - action)
        _draw_motile_cell(
            painter, (180 - separation, 120), (38, 29), 1.0, 0.2,
        )
        _draw_motile_cell(
            painter, (180 + separation, 120), (38, 29),
            1.0 - 0.65 * action, 0.8,
        )
    elif mode == "straightness":
        _track_path(
            painter, ((55, 76), (105, 77), (155, 78), (205, 79), (275, 80)),
            1.0 - action,
        )
        _track_path(
            painter, ((60, 170), (108, 145), (155, 178), (213, 140), (277, 163)),
            1.0,
        )
    elif mode == "zscore":
        normal = [(62, 120), (110, 115), (158, 112), (206, 110), (264, 108)]
        outlier = list(normal)
        outlier[2] = (158, 46 + 66 * action)
        _track_path(painter, outlier)
    elif mode == "division":
        _track_path(painter, ((70, 120), (125, 120), (180, 120)))
        spread = 38 * action
        for endpoint in ((245, 120 - spread), (245, 120 + spread)):
            painter.line(((180, 120), endpoint), _mix(WHITE, 0.8), 0.45)
            _draw_motile_cell(painter, endpoint, (18, 14), action, 0.4)
    elif mode == "contour_sigma":
        _draw_motile_cell(
            painter, (180, 120), (78, 58), 1.0, 0.3 + 2.0 * action,
        )
        rough = _blob_points(180, 120, 80, 60, 0.3, 0.18)
        painter.line(
            rough, _mix(WHITE, 0.72 * (1.0 - action)), 0.45,
            True, fill_closed=False,
        )


CLUSTERS = (
    ((75, 78), (100, 65), (119, 91), (89, 107), (132, 54)),
    ((225, 78), (256, 66), (280, 91), (240, 108), (292, 55)),
    ((140, 165), (175, 148), (204, 174), (158, 190), (218, 147)),
)
CLUSTER_COLORS = (BLUE, TEAL, MAGENTA)


def _mini_cell(painter: Painter, center: Tuple[float, float], size: float,
               color: Tuple[int, int, int], amount: float = 1.0) -> None:
    painter.svg_object(
        "cell",
        center,
        (size * 1.2, size * 0.8),
        amount=amount,
        angle=0.035 * math.sin(size),
        width=0.45,
        tint=color,
    )


def _umap_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    mode = spec.params["mode"]
    if mode == "neighbors":
        points = [point for cluster in CLUSTERS for point in cluster]
        for point in points:
            distances = sorted(
                (math.hypot(point[0] - other[0], point[1] - other[1]), other)
                for other in points if other != point)
            count = 1 + int(round(action * 3))
            for _distance, other in distances[:count]:
                painter.line((point, other), _mix(WHITE, 0.12 + 0.08 * count), 0.35)
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.5, color)
    elif mode == "min_dist":
        centers = ((100, 84), (260, 82), (176, 166))
        for cluster, color, center in zip(CLUSTERS, CLUSTER_COLORS, centers):
            original_center = (
                sum(point[0] for point in cluster) / len(cluster),
                sum(point[1] for point in cluster) / len(cluster),
            )
            spread = 0.45 + 0.85 * action
            for point in cluster:
                moved = (
                    center[0] + (point[0] - original_center[0]) * spread,
                    center[1] + (point[1] - original_center[1]) * spread,
                )
                painter.dot(moved, 1.6, color)
    elif mode == "images":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.7, _mix(color, 1.0 - action))
                _mini_cell(painter, point, 4.5, color, action)
    elif mode == "canvas":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster[::2]:
                painter.rectangle(
                    (point[0] - 7, point[1] - 7,
                     point[0] + 7, point[1] + 7),
                    _mix(GRAY, 0.65 * (1.0 - action)), 0.45, 1,
                )
                _mini_cell(painter, point, 4.5, color)
    elif mode == "outlines":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.5, color)
        for box, color in zip(
            ((57, 42, 145, 122), (205, 42, 310, 122), (120, 134, 235, 204)),
            CLUSTER_COLORS,
        ):
            painter.ellipse(box, _mix(color, action), 0.5)
    elif mode == "points":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.7, _mix(color, action))
        for center, color in zip(((100, 82), (260, 82), (177, 168)), CLUSTER_COLORS):
            painter.ellipse((center[0] - 43, center[1] - 34,
                             center[0] + 43, center[1] + 34), color, 0.5)
    elif mode == "smooth":
        polygon = ((56, 86), (75, 46), (132, 49), (150, 92), (117, 124), (66, 116))
        painter.line(polygon, _mix(BLUE, 1.0 - action), 0.5, True)
        painter.ellipse((55, 43, 151, 126), _mix(BLUE, action), 0.5)
        for point in CLUSTERS[0]:
            painter.dot(point, 1.5, BLUE)
    elif mode == "noise":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.5, color)
        noise = ((33, 45), (330, 48), (320, 190), (45, 196), (180, 35), (300, 135))
        for point in noise:
            painter.dot(point, 1.6, _mix(WHITE, 1.0 - action))
    elif mode == "by_cluster":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.2, _mix(color, 0.45))
            choice = cluster[0]
            _mini_cell(painter, choice, 5.0 + 2.0 * action, color)
    elif mode == "dot_size":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 0.9 + 3.0 * action, color)
    elif mode == "image_zoom":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster[::2]:
                _mini_cell(painter, point, 2.5 + 6.0 * action, color)
    elif mode == "density":
        for cluster, color in zip(CLUSTERS, CLUSTER_COLORS):
            for point in cluster:
                painter.dot(point, 1.5, color)
        radius = 12 + 28 * action
        for center in ((100, 82), (260, 82), (177, 168)):
            painter.ellipse(
                (center[0] - radius, center[1] - radius,
                 center[0] + radius, center[1] + radius),
                _mix(WHITE, 0.35), 0.4,
            )


def _alignment_scene(painter: Painter, spec: Spec, action: float) -> None:
    _well(painter)
    mode = spec.params["mode"]
    seam = 180
    painter.line(((seam, 13), (seam, H - 13)), _mix(GRAY, 0.75 * (1.0 - action)), 0.5)
    offset = 22 * (1.0 - action)
    _object_outline(
        painter, "cell", (165, 120), (58, 38),
        0.55 + 0.45 * action, 0.3,
    )
    _object_outline(
        painter, "cell", (165 + offset, 120), (58, 38),
        0.55 * (1.0 - action), 0.3,
    )
    if mode == "overlap":
        width = 10 + 38 * action
        painter.rectangle((seam - width, 26, seam + width, H - 26), _mix(TEAL, 0.35), 0.4)


def render_frame(spec: Spec, index: int) -> Image.Image:
    painter = Painter()
    action = _cycle(index)
    if spec.scene == "filter":
        _filter_scene(painter, spec, action)
    elif spec.scene == "border":
        _border_scene(painter, spec, index)
    elif spec.scene == "merge_edge_pathogen_cells":
        _merge_edge_pathogen_cells(painter, action)
    elif spec.scene == "adjust_cells":
        _adjust_cells(painter, action)
    elif spec.scene == "merge":
        _generic_merge(painter, spec, action)
    elif spec.scene == "split":
        _split_scene(painter, spec, action)
    elif spec.scene == "probability":
        _probability_scene(painter, spec, action)
    elif spec.scene == "flow":
        _flow_scene(painter, spec, action)
    elif spec.scene == "diameter":
        _diameter_scene(painter, spec, action)
    elif spec.scene == "background":
        _background_scene(painter, spec, action)
    elif spec.scene == "signal":
        _signal_scene(painter, spec, action)
    elif spec.scene == "fill_holes":
        _fill_holes(painter, spec, action)
    elif spec.scene == "organelle":
        _organelle_scene(painter, spec, action)
    elif spec.scene == "outline_thickness":
        _outline_thickness(painter, action)
    elif spec.scene == "normalization":
        _normalization(painter, action)
    elif spec.scene == "crop":
        _crop_scene(painter, spec, action)
    elif spec.scene == "cytoplasm":
        _cytoplasm(painter, action)
    elif spec.scene == "radial":
        _radial(painter, action)
    elif spec.scene == "uninfected":
        _uninfected(painter, action)
    elif spec.scene == "tracking":
        _tracking_scene(painter, spec, action)
    elif spec.scene == "umap":
        _umap_scene(painter, spec, action)
    elif spec.scene == "alignment":
        _alignment_scene(painter, spec, action)
    else:
        raise ValueError(f"Unknown scene {spec.scene!r}")
    return painter.finish()


def _specs() -> List[Spec]:
    specs: List[Spec] = []
    category = "Mask filtering"
    aliases = {
        "cell": {
            "border": ("cell_remove_border_objects", "remove_border_cells"),
            "minimum": ("cell_min_area", "cell_min_size"),
            "maximum": ("cell_max_area",),
            "dim": ("cell_min_intensity_percentile",),
            "bright": ("cell_max_intensity_percentile",),
        },
        "nucleus": {
            "border": ("nucleus_remove_border_objects", "remove_border_nuclei"),
            "minimum": ("nucleus_min_area", "nucleus_min_size"),
            "maximum": ("nucleus_max_area",),
            "dim": ("nucleus_min_intensity_percentile",),
            "bright": ("nucleus_max_intensity_percentile",),
        },
        "pathogen": {
            "border": ("pathogen_remove_border_objects", "remove_border_pathogens"),
            "minimum": ("pathogen_min_area", "pathogen_min_size"),
            "maximum": ("pathogen_max_area",),
            "dim": ("pathogen_min_intensity_percentile",),
            "bright": ("pathogen_max_intensity_percentile",),
        },
        "organelle": {
            "border": (
                "organelle_remove_border_objects", "organelle_remove_border",
                "remove_border_organelles",
            ),
            "minimum": ("organelle_min_area", "organelle_min_size"),
            "maximum": ("organelle_max_area", "organelle_max_size"),
            "dim": ("organelle_min_intensity_percentile",),
            "bright": ("organelle_max_intensity_percentile",),
        },
    }
    names = {
        "border": "Remove border objects",
        "minimum": "Minimum object area",
        "maximum": "Maximum object area",
        "dim": "Minimum intensity percentile",
        "bright": "Maximum intensity percentile",
    }
    for kind, variants in aliases.items():
        for variant, keys in variants.items():
            scene = "border" if variant == "border" else "filter"
            specs.append(Spec(
                keys[0], f"{kind.capitalize()} — {names[variant]}", category,
                scene, keys, {"kind": kind, "variant": variant},
            ))

    specs.extend([
        Spec(
            "merge_edge_pathogen_cells", "Merge edge-pathogen cells",
            "Mask repair", "merge_edge_pathogen_cells",
            ("merge_edge_pathogen_cells",),
        ),
        Spec(
            "adjust_cells", "Adjust fragmented cells",
            "Mask repair", "adjust_cells", ("adjust_cells",),
        ),
    ])
    for kind in OBJECT_COLORS:
        specs.append(Spec(
            f"{kind}_perimeter_fraction", f"{kind.capitalize()} perimeter merge",
            "Mask repair", "merge", (f"{kind}_perimeter_fraction",),
            {"kind": kind},
        ))
        specs.append(Spec(
            f"{kind}_intensity_merge", f"{kind.capitalize()} intensity merge",
            "Mask repair", "merge",
            (f"{kind}_intensity_merge", f"{kind}_intensity_threshold_method",
             f"{kind}_intensity_percentile"),
            {"kind": kind, "intensity": True},
        ))
        specs.append(Spec(
            f"{kind}_intensity_split", f"{kind.capitalize()} watershed split",
            "Mask repair", "split",
            (f"{kind}_intensity_split", f"{kind}_area_multiplier",
             f"{kind}_min_distance", f"{kind}_min_object_area"),
            {"kind": kind},
        ))

    for kind in ("cell", "nucleus", "pathogen"):
        specs.extend([
            Spec(
                f"{kind}_CP_prob", f"{kind.capitalize()} probability threshold",
                "Segmentation", "probability",
                (f"{kind}_CP_prob",), {"kind": kind},
            ),
            Spec(
                f"{kind}_FT", f"{kind.capitalize()} flow threshold",
                "Segmentation", "flow", (f"{kind}_FT",), {"kind": kind},
            ),
            Spec(
                f"{kind}_diameter", f"{kind.capitalize()} diameter",
                "Segmentation", "diameter", (f"{kind}_diameter",),
                {"kind": kind},
            ),
            Spec(
                f"remove_background_{kind}",
                f"{kind.capitalize()} background subtraction",
                "Image preprocessing", "background",
                (f"remove_background_{kind}", f"{kind}_background"),
                {"kind": kind},
            ),
            Spec(
                f"{kind}_Signal_to_noise", f"{kind.capitalize()} signal-to-noise",
                "Image preprocessing", "signal",
                (f"{kind}_Signal_to_noise",), {"kind": kind},
            ),
        ])
    specs.extend([
        Spec(
            "organelle_diameter", "Organelle diameter", "Segmentation",
            "diameter", ("organelle_diameter",), {"kind": "organelle"},
        ),
        Spec(
            "organelle_CP_prob", "Organelle probability threshold",
            "Segmentation", "probability", ("organelle_CP_prob",),
            {"kind": "organelle"},
        ),
        Spec(
            "organelle_FT", "Organelle flow threshold", "Segmentation",
            "flow", ("organelle_FT",), {"kind": "organelle"},
        ),
        Spec(
            "fill_in", "Fill holes in masks", "Mask repair", "fill_holes",
            ("fill_in",), {"kind": "cell"},
        ),
        Spec(
            "organelle_fill_holes", "Fill small organelle holes",
            "Organelle preprocessing", "fill_holes",
            ("organelle_fill_holes",), {"kind": "organelle"},
        ),
        Spec(
            "organelle_watershed_spots", "Split touching organelle spots",
            "Organelle preprocessing", "organelle",
            ("organelle_watershed_spots",), {"mode": "watershed"},
        ),
        Spec(
            "organelle_skeletonize", "Skeletonize organelle networks",
            "Organelle preprocessing", "organelle",
            ("organelle_skeletonize",), {"mode": "skeleton"},
        ),
        Spec(
            "organelle_rolling_ball", "Rolling-ball background correction",
            "Organelle preprocessing", "organelle",
            ("organelle_rolling_ball", "organelle_rolling_ball_radius"),
            {"mode": "rolling_ball"},
        ),
        Spec(
            "organelle_clahe", "Organelle local contrast (CLAHE)",
            "Organelle preprocessing", "organelle",
            ("organelle_clahe", "organelle_clahe_clip_limit"),
            {"mode": "clahe"},
        ),
        Spec(
            "organelle_mask_within_cells", "Mask organelles within cells",
            "Organelle preprocessing", "organelle",
            ("organelle_mask_within_cells",), {"mode": "within_cells"},
        ),
        Spec(
            "organelle_log_threshold", "Organelle segmentation threshold",
            "Organelle preprocessing", "organelle",
            ("organelle_log_threshold", "organelle_unet_threshold"),
            {"mode": "threshold"},
        ),
        Spec(
            "outline_thickness", "Mask outline thickness", "Plot appearance",
            "outline_thickness", ("outline_thickness",),
        ),
        Spec(
            "normalization_percentiles", "Image normalization percentiles",
            "Image preprocessing", "normalization",
            ("normalization_percentiles", "normalize", "normalize_plots"),
        ),
    ])

    specs.extend([
        Spec(
            "use_bounding_box", "Keep bounding-box context", "Crop output",
            "crop", ("use_bounding_box",), {"mode": "bounding_box"},
        ),
        Spec(
            "dialate_pngs", "Dilate crop masks", "Crop output", "crop",
            ("dialate_pngs", "dialate_png_ratios"), {"mode": "dilate"},
        ),
        Spec(
            "crop_mode", "Choose crop target", "Crop output", "crop",
            ("crop_mode",), {"mode": "crop_mode"},
        ),
        Spec(
            "png_size", "Crop canvas size", "Crop output", "crop",
            ("png_size",), {"mode": "png_size"},
        ),
        Spec(
            "cytoplasm", "Derive cytoplasm compartment", "Measurement",
            "cytoplasm", ("cytoplasm",),
        ),
        Spec(
            "radial_dist", "Radial-distance shells", "Measurement", "radial",
            ("radial_dist", "distance_gaussian_sigma"),
        ),
        Spec(
            "uninfected", "Keep or remove uninfected cells", "Measurement",
            "uninfected", ("uninfected",),
        ),
    ])

    tracking = [
        ("timelapse_remove_transient", "Remove transient tracks", "transient",
         ("timelapse_remove_transient", "timelapse_frame_limits")),
        ("timelapse_displacement", "Maximum linking displacement", "displacement",
         ("timelapse_displacement", "ultrack_max_distance",
          "t_max_displacement_px", "t_max_displacement_um")),
        ("timelapse_memory", "Tracking memory", "memory", ("timelapse_memory",)),
        ("t_link_threshold", "Timepoint overlap threshold", "link",
         ("t_link_threshold",)),
        ("stitch_threshold", "Z-plane stitch threshold", "stitch",
         ("stitch_threshold",)),
        ("z_projection", "Z projection", "projection",
         ("z_projection", "pick_slice")),
        ("t_project_for_tracking", "Project volumes for tracking", "project_tracking",
         ("t_project_for_tracking",)),
        ("straightness_filter", "Remove overly straight tracks", "straightness",
         ("straightness_filter", "straightness_threshold")),
        ("zscore_thresh", "Smooth per-track outliers", "zscore", ("zscore_thresh",)),
        ("ultrack_division_weight", "Cell-division linking", "division",
         ("ultrack_division_weight",)),
        ("ultrack_contour_sigma", "Contour smoothing", "contour_sigma",
         ("ultrack_contour_sigma",)),
    ]
    for slug, title, mode, keys in tracking:
        specs.append(Spec(
            slug, title, "Tracking & volumetric", "tracking", keys, {"mode": mode},
        ))

    umap = [
        ("n_neighbors", "UMAP neighborhood size", "neighbors", ("n_neighbors",)),
        ("min_dist", "UMAP minimum distance", "min_dist", ("min_dist",)),
        ("plot_images", "Show object images", "images", ("plot_images",)),
        ("remove_image_canvas", "Remove image canvas", "canvas",
         ("remove_image_canvas",)),
        ("plot_outlines", "Show cluster outlines", "outlines", ("plot_outlines",)),
        ("plot_points", "Show embedding points", "points", ("plot_points",)),
        ("smooth_lines", "Smooth cluster outlines", "smooth", ("smooth_lines",)),
        ("remove_cluster_noise", "Remove cluster noise", "noise",
         ("remove_cluster_noise",)),
        ("plot_by_cluster", "Sample images by cluster", "by_cluster",
         ("plot_by_cluster",)),
        ("dot_size", "Embedding point size", "dot_size", ("dot_size",)),
        ("img_zoom", "Embedding image zoom", "image_zoom", ("img_zoom",)),
        ("eps", "Density clustering radius", "density",
         ("eps", "min_samples", "clustering")),
    ]
    for slug, title, mode, keys in umap:
        specs.append(Spec(
            slug, title, "Image UMAP", "umap", keys, {"mode": mode},
        ))

    specs.extend([
        Spec(
            "overlap", "Tile overlap", "Alignment & stitching", "alignment",
            ("overlap",), {"mode": "overlap"},
        ),
        Spec(
            "blend", "Tile seam blending", "Alignment & stitching", "alignment",
            ("blend",), {"mode": "blend"},
        ),
    ])
    return specs


def _write_gif(spec: Spec) -> Path:
    frames = [render_frame(spec, index) for index in range(FRAMES)]
    path = ASSETS / f"{spec.slug}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=85,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return path


def _validate(path: Path) -> Dict[str, Any]:
    with Image.open(path) as image:
        hashes = []
        for frame in ImageSequence.Iterator(image):
            rgb = frame.convert("RGB")
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        if image.size != (CANVAS, CANVAS):
            raise AssertionError(f"{path.name}: wrong size {image.size}")
        # Pillow coalesces identical before/after hold frames into longer GIF
        # durations, so the encoded frame count can be lower than FRAMES even
        # though the timing and transition are unchanged.
        if len(hashes) < 4:
            raise AssertionError(f"{path.name}: only {len(hashes)} frames")
        if len(set(hashes)) < 4:
            raise AssertionError(f"{path.name}: animation is effectively static")
        first = image.seek(0) or image.convert("RGB")
        if first.getpixel((0, 0)) != BLACK:
            raise AssertionError(f"{path.name}: corner is not black")
    return {
        "frames": len(hashes),
        "unique_frames": len(set(hashes)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _contact_sheets(specs: Sequence[Spec]) -> None:
    groups: Dict[str, List[Spec]] = {}
    for spec in specs:
        groups.setdefault(spec.category, []).append(spec)
    title_font = _font(15, True)
    key_font = _font(11)
    for category, items in groups.items():
        columns = 3
        tile_w, tile_h = 380, 430
        rows = math.ceil(len(items) / columns)
        sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (18, 18, 20))
        draw = ImageDraw.Draw(sheet)
        for index, spec in enumerate(items):
            x = (index % columns) * tile_w
            y = (index // columns) * tile_h
            preview = render_frame(spec, FRAMES // 2).resize((324, 324))
            sheet.paste(preview, (x + 28, y + 58))
            title_lines = textwrap.wrap(
                spec.title,
                width=36,
                max_lines=2,
                placeholder="...",
            )
            draw.multiline_text(
                (x + 28, y + 8),
                "\n".join(title_lines),
                fill=WHITE,
                font=title_font,
                spacing=2,
            )
            key_text = ", ".join(spec.settings)
            if len(key_text) > 56:
                key_text = key_text[:53] + "..."
            draw.text((x + 28, y + 390), key_text, fill=(170, 175, 185), font=key_font)
        filename = category.lower().replace("&", "and").replace(" ", "_")
        (SHEETS / f"{filename}.png").parent.mkdir(parents=True, exist_ok=True)
        sheet.save(SHEETS / f"{filename}.png", optimize=True)


def _storyboards(specs: Sequence[Spec]) -> None:
    """Write before/transition/after panels for the two requested examples."""
    selected = {
        "merge_edge_pathogen_cells",
        "cell_remove_border_objects",
    }
    label_font = _font(14, True)
    labels = ("BEFORE", "TRANSITION", "AFTER")
    for spec in specs:
        if spec.slug not in selected:
            continue
        frame_indices = (
            (0, 16, 20)
            if spec.slug == "cell_remove_border_objects"
            else (0, 8, 14)
        )
        board = Image.new(
            "RGB", (CANVAS * 3, CANVAS + 36), (18, 18, 20),
        )
        draw = ImageDraw.Draw(board)
        for column, (frame_index, label) in enumerate(zip(frame_indices, labels)):
            x = column * CANVAS
            board.paste(render_frame(spec, frame_index), (x, 36))
            box = draw.textbbox((0, 0), label, font=label_font)
            label_width = box[2] - box[0]
            draw.text(
                (x + (CANVAS - label_width) / 2, 9),
                label,
                fill=WHITE,
                font=label_font,
            )
        board.save(STORYBOARDS / f"{spec.slug}.png", optimize=True)


def _write_gallery(
    specs: Sequence[Spec],
    manifest: Sequence[Dict[str, Any]],
    template_hashes: Dict[str, str],
) -> None:
    categories: Dict[str, List[Spec]] = {}
    for spec in specs:
        categories.setdefault(spec.category, []).append(spec)
    sections = []
    for category, items in categories.items():
        cards = []
        for spec in items:
            keys = " · ".join(html.escape(key) for key in spec.settings)
            search = html.escape(" ".join((spec.title, category, *spec.settings)).lower())
            cards.append(
                f'<article class="card" data-search="{search}">'
                f'<img src="gifs/{html.escape(spec.slug)}.gif" '
                f'alt="{html.escape(spec.title)} animation">'
                f'<h3>{html.escape(spec.title)}</h3>'
                f'<p>{keys}</p></article>'
            )
        sections.append(
            f'<section><h2>{html.escape(category)} <span>{len(items)}</span></h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>spaCR setting animation review</title>
<style>
:root{{--bg:#101012;--card:#17171a;--line:#34343a;--fg:#f5f5f7;--muted:#a8a8b2;--purple:#9b009b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:4;padding:20px 28px;background:#101012ee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}}
h1{{margin:0 0 6px;font-size:24px}} header p{{margin:0;color:var(--muted)}}
input{{margin-top:14px;width:min(520px,100%);padding:10px 13px;border:1px solid var(--line);border-radius:9px;background:#202025;color:var(--fg)}}
main{{padding:4px 28px 40px}} section{{padding-top:22px}} h2{{font-size:18px}} h2 span{{color:var(--muted);font-weight:400}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}}
.card{{overflow:hidden;border:1px solid var(--line);border-radius:14px;background:var(--card)}}
.card img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#000}}
.card h3{{margin:13px 15px 3px;font-size:15px}} .card p{{margin:0 15px 15px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
.hidden{{display:none}} .legend{{color:#e28bc7}}
</style></head><body>
<header><h1>spaCR setting animation review</h1>
<p>{len(specs)} shipped GIFs · exact setting-key lookup · <span class="legend">purple dots open these animations inside spaCR</span></p>
<input id="search" placeholder="Filter by title or exact setting key" autofocus></header>
<main>{''.join(sections)}</main>
<script>const q=document.querySelector('#search');q.addEventListener('input',()=>{{const s=q.value.toLowerCase().trim();document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hidden',s&&!c.dataset.search.includes(s)));}});</script>
</body></html>"""
    (REVIEW_ROOT / "index.html").write_text(document, encoding="utf-8")
    _write_manifest(manifest, template_hashes)


def _write_manifest(
    manifest: Sequence[Dict[str, Any]],
    template_hashes: Dict[str, str],
) -> None:
    """Write the shipped manifest describing every packaged GIF."""
    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "template_sha256": template_hashes,
                "animations": list(manifest),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_docs_gallery(specs: Sequence[Spec]) -> None:
    """Generate the Sphinx gallery and stable anchors used by API links."""
    groups: Dict[str, List[Spec]] = {}
    for spec in specs:
        groups.setdefault(spec.category, []).append(spec)

    lines = [
        "Setting animation gallery",
        "=========================",
        "",
        "spaCR includes short, deterministic GIFs for settings whose effect is",
        "easier to understand visually. In the desktop interface a purple dot",
        "above the teal API dot opens the corresponding animation immediately",
        "above the setting. The midpoint between both dots remains aligned with",
        "the setting label.",
        "",
        "The diagrams use a shared biological grammar: white fibroblast or",
        "motile immune-cell outlines, blue nuclei with unequal nucleoli, teal",
        "Toxoplasma tachyzoites inside an outline-only parasitophorous vacuole,",
        "and soft-magenta Golgi cisternae. Filled regions are translucent and",
        "the rounded field perimeter remains white on black.",
        "",
        "The cell, nucleus and nucleoli, two-parasite vacuole, and Golgi use",
        "the reviewed, artist-authored SVG paths checked into ``tools/``.",
        "Qt renders those exact Bezier paths at high resolution before each",
        "animation frame is composited, keeping the outlines smooth at small",
        "sizes. Source-template SHA-256 hashes are recorded in the manifest.",
        "",
        "Animations are resolved by exact setting key through",
        ":mod:`spacr.setting_animations`; the assets and manifest are generated",
        "reproducibly by ``tools/generate_setting_animations.py``.",
        "",
    ]
    for category, items in groups.items():
        lines.extend((category, "-" * len(category), ""))
        for spec in items:
            anchor = "setting-animation-" + spec.slug.replace("_", "-")
            lines.extend((
                f".. _{anchor}:",
                "",
                spec.title,
                "~" * len(spec.title),
                "",
                f".. image:: ../../spacr/resources/setting_animations/gifs/{spec.slug}.gif",
                f"   :alt: {spec.title} setting animation",
                "   :width: 300px",
                "",
                "**Settings:** " + ", ".join(
                    f"``{key}``" for key in spec.settings
                ),
                "",
            ))
    DOCS_PAGE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _regenerate_subset(specs: Sequence[Spec], slugs: Sequence[str]) -> int:
    """Rebuild only the named GIFs and merge them into the shipped manifest.

    A full run re-encodes all 94 GIFs, and a different Pillow build writes
    byte-different files for pixel-identical frames. Rebuilding one scene
    keeps every other packaged GIF, and its recorded hash, exactly as
    shipped: manifest entries for animations that were not regenerated are
    copied verbatim, never rewritten from the current specs.
    """
    template_hashes = _validate_templates()
    by_slug = {spec.slug: spec for spec in specs}
    wanted: List[str] = []
    for slug in slugs:
        if slug not in by_slug:
            raise SystemExit(f"Unknown animation slug: {slug}")
        if slug not in wanted:
            wanted.append(slug)
    try:
        shipped = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not read the shipped manifest: {error}")
    if shipped.get("template_sha256") not in (None, template_hashes):
        raise SystemExit(
            "The SVG templates changed since the shipped manifest was written; "
            "run without --only so every GIF is rebuilt from them."
        )
    entries = {
        entry["slug"]: entry
        for entry in shipped.get("animations", [])
        if isinstance(entry, dict) and "slug" in entry
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    for slug in wanted:
        spec = by_slug[slug]
        path = _write_gif(spec)
        entry = asdict(spec)
        entry["file"] = str(path.relative_to(ROOT))
        entry["validation"] = _validate(path)
        entries[slug] = entry
    missing = [spec.slug for spec in specs if spec.slug not in entries]
    if missing:
        raise SystemExit(
            "The shipped manifest has no entry for "
            + ", ".join(missing)
            + "; run without --only to generate every GIF."
        )
    _write_manifest([entries[spec.slug] for spec in specs], template_hashes)
    print(f"Regenerated {len(wanted)} of {len(specs)} GIFs: {', '.join(wanted)}")
    print("Contact sheets, storyboards and the docs gallery need a full run.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate all review GIFs, validate them, and build the local gallery."""
    parser = argparse.ArgumentParser(
        description="Generate the shipped spaCR setting animations.",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SLUG",
        help=(
            "Regenerate just this animation and keep every other packaged GIF "
            "untouched. Repeatable."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.only:
        return _regenerate_subset(_specs(), arguments.only)
    template_hashes = _validate_templates()
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    if SHEETS.exists():
        shutil.rmtree(SHEETS)
    if STORYBOARDS.exists():
        shutil.rmtree(STORYBOARDS)
    ASSETS.mkdir(parents=True)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True)
    STORYBOARDS.mkdir(parents=True)
    specs = _specs()
    manifest = []
    for spec in specs:
        path = _write_gif(spec)
        check = _validate(path)
        entry = asdict(spec)
        entry["file"] = str(path.relative_to(ROOT))
        entry["validation"] = check
        manifest.append(entry)
    _contact_sheets(specs)
    _storyboards(specs)
    _write_gallery(specs, manifest, template_hashes)
    _write_docs_gallery(specs)
    total_bytes = sum(item["validation"]["bytes"] for item in manifest)
    print(f"Generated and validated {len(specs)} GIFs ({total_bytes / 1024:.1f} KiB)")
    print(REVIEW_ROOT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
