"""A napari-style layer model — images, labels, points and shapes in one world.

Every viewer spaCR has ever shipped draws one thing. The live preview draws an
image with outlines burnt into it; the timelapse preview draws a frame with
tracks burnt into it; Make Masks draws an image with a brush burnt into it.
"Burnt in" is the problem: the mask is not a *thing* you can hide, fade, recolour
or put underneath something else — it is pixels that were already RGB by the time
the widget saw them.

This module is the model underneath a viewer where those are separate objects.
It is deliberately **pure numpy** with no Qt anywhere, for the same reasons
:mod:`spacr.selection` is:

* it can be tested without a display, and the compositing rules — which layer
  wins, what opacity means, where a point lands — are exactly the rules that
  need testing;
* the same stack can be rendered headless into a figure or a report;
* and five later features (ROI shapes honoured by Measure, a counting points
  layer, a label brush, orthogonal views, a comparison grid) are written
  against *this*, not against a widget, so none of them needs a running
  ``QApplication`` to have a unit test.

:mod:`spacr.qt.layer_viewer` is the Qt view over it.

The world, and why it is not optional
------------------------------------

Layers do not agree on pixels. A labels mask may be at full resolution while a
downsampled preview is not; a points layer of centroids is in continuous
coordinates and has no grid at all; and confocal z-stacks in this codebase are
routinely anisotropic — 0.65 µm in x and y, 2 µm in z is an ordinary spaCR
stack.

So a layer never says "row 40, column 12". It says "this data axis is *z*, one
step along it is 2 µm, and element 0 sits at 0 µm" — a :class:`Spacing` — and
everything downstream happens in world units. A :class:`Canvas` is a window
onto that world (an origin, a step and a size, over two named world axes), and
rendering is: for every canvas pixel, ask each layer what is at that world
coordinate.

Treating spacing as decoration is a *silent* error, not a cosmetic one: an
overlay drawn a slice out of register still looks like a plausible image, and
the number that comes out of it is wrong with no warning. Hence the guards
here are loud — a zero scale raises, and a stack refuses to mix µm layers with
px layers rather than quietly drawing them on top of each other.

Order
-----

``stack[0]`` is the BOTTOM layer and is drawn first, like a stack of acetates
and like napari's own layer list. Moving a layer up moves it towards the front.

Blending
--------

Every layer contributes ``coverage`` (where it has something to say) and
``opacity`` (how loudly). All five modes combine them as ``alpha = coverage *
opacity``, except :data:`Blending.OPAQUE`, which first hardens coverage to
all-or-nothing — that is what makes it a curtain rather than a veil, and it is
the only place the two are treated differently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping,
                    Optional, Sequence, Tuple, Union)

import numpy as np

# The object identity every table in measurements.db already agrees on. A
# labels layer that invented its own key scheme would be a fifth island; the
# whole point of `linked_selection` is that there are no more of those.
from .selection import OBJECT_KEY_COLUMNS, object_keys

__all__ = [
    "LayerError",
    "Blending",
    "Spacing",
    "Canvas",
    "Colormap",
    "colormap",
    "COLORMAPS",
    "to_rgba",
    "label_color",
    "label_colors",
    "FieldKey",
    "Layer",
    "ImageLayer",
    "LabelsLayer",
    "PointsLayer",
    "Shape",
    "ShapesLayer",
    "LayerEvent",
    "LayerStack",
    "DEFAULT_PERCENTILES",
]


class LayerError(ValueError):
    """A layer, spacing or stack that cannot mean what it was asked to mean.

    Raised rather than repaired. Every case this covers — a zero voxel size, a
    stack mixing µm with px, a points array of the wrong width — produces a
    picture that still *looks* right while being out of register, which is the
    failure mode that reaches a figure.
    """


#: The percentile stretch an image layer uses when nobody sets contrast limits.
#: The same pair :func:`spacr.qt.widgets.live_preview._to_uint8` has always
#: used, so a field looks the same in this viewer as in the live preview.
DEFAULT_PERCENTILES: Tuple[float, float] = (2.0, 98.0)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

#: Colour names understood by :func:`to_rgba`, beyond ``#rrggbb`` hex. Small on
#: purpose: these are the ones a microscopy channel is actually given.
_NAMED_COLORS: Dict[str, Tuple[float, float, float]] = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.55, 0.0),
    "violet": (0.6, 0.3, 1.0),
}


def to_rgba(colour: Any, *, alpha: Optional[float] = None
            ) -> Tuple[float, float, float, float]:
    """Coerce anything that names a colour into ``(r, g, b, a)`` in 0–1.

    Accepts a name from :data:`_NAMED_COLORS`, ``#rgb`` / ``#rrggbb`` /
    ``#rrggbbaa``, a 3- or 4-tuple in 0–1, or a 3- or 4-tuple in 0–255 (any
    component above 1 switches the whole tuple to the 0–255 reading, which is
    the only unambiguous rule — ``(1, 1, 1)`` is white either way).

    :param alpha: overrides whatever the colour carried.
    :raises LayerError: on anything else.
    """
    if isinstance(colour, str):
        text = colour.strip().lower()
        if text in _NAMED_COLORS:
            rgba = _NAMED_COLORS[text] + (1.0,)
        elif text.startswith("#"):
            digits = text[1:]
            if len(digits) == 3:
                digits = "".join(c * 2 for c in digits)
            if len(digits) not in (6, 8):
                raise LayerError(
                    f"{colour!r} is not a #rgb/#rrggbb/#rrggbbaa hex colour")
            try:
                parts = [int(digits[i:i + 2], 16) / 255.0
                         for i in range(0, len(digits), 2)]
            except ValueError:
                raise LayerError(f"{colour!r} is not a hex colour") from None
            rgba = tuple(parts) if len(parts) == 4 else tuple(parts) + (1.0,)
        else:
            raise LayerError(
                f"unknown colour {colour!r}; use a hex string or one of "
                f"{sorted(_NAMED_COLORS)}")
    else:
        try:
            parts = [float(v) for v in colour]
        except (TypeError, ValueError):
            raise LayerError(
                f"cannot read a colour out of {colour!r}") from None
        if len(parts) not in (3, 4):
            raise LayerError(
                f"a colour needs 3 or 4 components, got {len(parts)}")
        if any(v > 1.0 for v in parts):
            parts = [v / 255.0 for v in parts]
        rgba = tuple(parts) if len(parts) == 4 else tuple(parts) + (1.0,)
    if alpha is not None:
        rgba = rgba[:3] + (float(alpha),)
    return tuple(float(min(1.0, max(0.0, v))) for v in rgba)  # type: ignore


class Colormap:
    """A named ramp from a value in 0–1 to a colour.

    Linear interpolation between ``colors`` at ``stops``. Two stops covers
    every microscopy channel LUT there is (black → the channel's colour), and
    more stops covers the perceptual maps without pulling matplotlib into a
    module that has to stay importable in a worker.
    """

    __slots__ = ("name", "_colors", "_stops")

    def __init__(self, name: str, colors: Sequence[Any],
                 stops: Optional[Sequence[float]] = None):
        cols = np.asarray([to_rgba(c)[:3] for c in colors], dtype=np.float32)
        if len(cols) < 2:
            raise LayerError(
                f"colormap {name!r} needs at least two colours, got {len(cols)}")
        if stops is None:
            positions = np.linspace(0.0, 1.0, len(cols), dtype=np.float64)
        else:
            positions = np.asarray(stops, dtype=np.float64)
            if positions.shape != (len(cols),):
                raise LayerError(
                    f"colormap {name!r} has {len(cols)} colours but "
                    f"{positions.size} stops")
            if np.any(np.diff(positions) <= 0):
                raise LayerError(
                    f"colormap {name!r} stops must increase: {list(positions)}")
        self.name = str(name)
        self._colors = cols
        self._stops = positions

    @property
    def colors(self) -> np.ndarray:
        """The ramp's colours, ``(K, 3)`` float32 — a copy."""
        return self._colors.copy()

    @property
    def stops(self) -> np.ndarray:
        """Where each colour sits in 0–1 — a copy."""
        return self._stops.copy()

    def map(self, values: Any) -> np.ndarray:
        """Map ``values`` (any shape, 0–1, clipped) to ``(..., 3)`` float32."""
        t = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        out = np.empty(t.shape + (3,), dtype=np.float32)
        for c in range(3):
            out[..., c] = np.interp(t, self._stops, self._colors[:, c])
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Colormap({self.name!r}, {len(self._colors)} stops)"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Colormap) and other.name == self.name
                and np.array_equal(other._colors, self._colors)
                and np.array_equal(other._stops, self._stops))

    def __hash__(self) -> int:
        return hash((self.name, self._colors.tobytes()))


def _ramp(name: str, colour: Any) -> Colormap:
    return Colormap(name, ["black", colour])


#: The built-in colormaps, by name. The single-colour ramps are what a
#: fluorescence channel wants; ``gray`` is the default for a lone channel.
COLORMAPS: Dict[str, Colormap] = {
    "gray": _ramp("gray", "white"),
    "grey": _ramp("grey", "white"),
    "red": _ramp("red", "red"),
    "green": _ramp("green", "green"),
    "blue": _ramp("blue", "blue"),
    "cyan": _ramp("cyan", "cyan"),
    "magenta": _ramp("magenta", "magenta"),
    "yellow": _ramp("yellow", "yellow"),
    "orange": _ramp("orange", "orange"),
    "violet": _ramp("violet", "violet"),
    "fire": Colormap("fire", ["black", "red", "orange", "yellow", "white"]),
    "ice": Colormap("ice", ["black", "blue", "cyan", "white"]),
}

#: The colours channels 0, 1, 2, … get when nobody says otherwise. Green first
#: because a one-channel spaCR field is nearly always the cell stain, and the
#: order after that keeps neighbouring channels distinguishable to a
#: red/green-blind reader (blue and magenta before red).
DEFAULT_CHANNEL_COLORMAPS: Tuple[str, ...] = (
    "green", "magenta", "cyan", "yellow", "blue", "red", "orange", "violet")


def colormap(spec: Any) -> Colormap:
    """Resolve ``spec`` to a :class:`Colormap`.

    Takes a :class:`Colormap` (returned unchanged), a name in
    :data:`COLORMAPS`, a colour (any :func:`to_rgba` form) which becomes a
    black→colour ramp, or — as a last resort — a matplotlib colormap name.

    Matplotlib is tried last and lazily. This module must stay importable
    where matplotlib is not installed and cheap to import in a worker, so it
    is never a hard dependency; but if a user types ``"viridis"`` and
    matplotlib is there, refusing would be pedantry.

    :raises LayerError: if ``spec`` names nothing.
    """
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in COLORMAPS:
            return COLORMAPS[key]
        if key in _NAMED_COLORS or key.startswith("#"):
            return _ramp(key, key)
        try:  # pragma: no cover - depends on the installed matplotlib
            from matplotlib import colormaps as _mpl_colormaps
            mpl = _mpl_colormaps[key]
            samples = [tuple(mpl(v)[:3]) for v in np.linspace(0.0, 1.0, 16)]
            return Colormap(key, samples)
        except Exception:
            raise LayerError(
                f"unknown colormap {spec!r}; use one of {sorted(COLORMAPS)}, "
                f"a colour, or an installed matplotlib colormap name"
            ) from None
    return _ramp(str(spec), spec)


#: Golden-ratio step for the label hue sequence.
_GOLDEN = 0.6180339887498949


def label_color(label: int, *, seed: int = 0) -> Tuple[float, float, float]:
    """A stable, vivid colour for integer ``label``.

    Golden-ratio hue spacing with a deterministic saturation/value jitter, so
    adjacent labels are far apart in hue and the same object keeps its colour
    across zooms, re-renders and sessions. Label 0 is background and is black
    (callers draw it transparent).

    The construction is deliberately the same one
    :func:`spacr.qt.widgets.live_preview._random_outline_palette` uses for
    outlines, so a mask has the same object colours in both viewers. It is
    reimplemented rather than imported because that module imports PySide6 and
    this one must not — see the module docstring.
    """
    n = int(label)
    if n == 0:
        return (0.0, 0.0, 0.0)
    h = ((n + int(seed)) * _GOLDEN) % 1.0
    s = 0.65 + 0.35 * (((n * 7 + int(seed)) % 5) / 4.0)
    v = 0.75 + 0.25 * (((n * 13 + int(seed)) % 4) / 3.0)
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def label_colors(labels: Any, *, seed: int = 0) -> np.ndarray:
    """:func:`label_color` over an array, returning ``(..., 3)`` float32.

    Built by looking up the unique labels once rather than per pixel: a
    2048×2048 mask holds a few hundred objects and tens of millions of pixels.
    """
    arr = np.asarray(labels)
    out = np.zeros(arr.shape + (3,), dtype=np.float32)
    if arr.size == 0:
        return out
    unique = np.unique(arr)
    lut = np.array([label_color(int(u), seed=seed) for u in unique],
                   dtype=np.float32)
    index = np.searchsorted(unique, arr)
    return lut[index]


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

class Blending:
    """How a layer combines with what is already on the canvas.

    Five modes, all of which honour ``opacity`` the same way — ``alpha =
    coverage * opacity`` — except :data:`OPAQUE`, which hardens coverage to
    0-or-1 first so that a soft-edged layer becomes a curtain. Opacity still
    scales an opaque layer; "opaque" describes what it does to the layers
    below it where it *has* something, not whether it can be faded.
    """

    #: Standard source-over. The default, and what a mask overlay wants.
    TRANSLUCENT = "translucent"
    #: Sum the colours. What two fluorescence channels want — the union of a
    #: green and a magenta channel should read white, not "whichever is on top".
    ADDITIVE = "additive"
    #: Hide what is underneath wherever this layer has any coverage.
    OPAQUE = "opaque"
    #: Multiply — a shading/attenuation layer.
    MULTIPLY = "multiply"
    #: Keep the darker of the two. For inverted (brightfield) LUTs, where
    #: "more signal" means darker and adding would wash the image out.
    MINIMUM = "minimum"

    #: Every mode, in the order a combo box should list them.
    ALL: Tuple[str, ...] = (TRANSLUCENT, ADDITIVE, OPAQUE, MULTIPLY, MINIMUM)

    @staticmethod
    def check(mode: str) -> str:
        """Normalise and validate a blending name.

        :raises LayerError: on an unknown mode — a typo that silently fell
            back to translucent would be a compositing bug nobody could see.
        """
        text = str(mode).strip().lower()
        if text not in Blending.ALL:
            raise LayerError(
                f"unknown blending {mode!r}; use one of {list(Blending.ALL)}")
        return text

    @staticmethod
    def apply(dst: np.ndarray, src: np.ndarray, coverage: np.ndarray,
              opacity: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Composite ``src`` over ``dst``; returns ``(rgb, alpha)``.

        :param dst: the canvas so far, ``(H, W, 3)`` float in 0–1.
        :param src: this layer's colour, ``(H, W, 3)`` float in 0–1.
        :param coverage: where this layer has something, ``(H, W)`` in 0–1.
        :param opacity: the layer's opacity, 0–1.
        :param mode: one of :data:`Blending.ALL`.
        :returns: the new canvas and the alpha that was actually used, so a
            caller can accumulate the composite's own coverage.
        """
        mode = Blending.check(mode)
        cov = np.clip(np.asarray(coverage, dtype=np.float32), 0.0, 1.0)
        if mode == Blending.OPAQUE:
            alpha = (cov > 0.0).astype(np.float32) * float(opacity)
        else:
            alpha = cov * float(opacity)
        a = alpha[..., None]
        if mode == Blending.ADDITIVE:
            out = np.clip(dst + src * a, 0.0, 1.0)
        elif mode == Blending.MULTIPLY:
            out = dst * (1.0 - a) + dst * src * a
        elif mode == Blending.MINIMUM:
            out = dst * (1.0 - a) + np.minimum(dst, src) * a
        else:  # translucent and opaque share the source-over arithmetic
            out = dst * (1.0 - a) + src * a
        return out.astype(np.float32), alpha


# ---------------------------------------------------------------------------
# Spacing — the one thing that must not be guessed
# ---------------------------------------------------------------------------

def _default_axes(ndim: int) -> Tuple[str, ...]:
    """Axis names for ``ndim`` dimensions: ``(…, "z", "y", "x")``."""
    tail = ("z", "y", "x")
    if ndim <= 3:
        return tail[3 - ndim:]
    return tuple(f"d{i}" for i in range(ndim - 3)) + tail


@dataclass(frozen=True)
class Spacing:
    """Where a layer's array elements sit in the world.

    ``world = translate + scale * index``, per axis, with axes named so that
    two layers can agree on what "z" means without agreeing on array shape.

    :param scale: world size of one element along each axis — the voxel size.
        Non-zero and finite, per axis: a zero would collapse the axis and
        every world query on it would answer with element 0, which draws a
        plausible picture of the wrong slice.
    :param translate: world coordinate of element 0 along each axis. Defaults
        to zeros. This is the crop offset — a tile cut out of a mosaic keeps
        its place in the mosaic by carrying it here.
    :param axes: axis names, outermost first. Defaults to ``("z", "y", "x")``
        truncated to ``len(scale)``.
    :param units: what one world unit is. Compared by name when layers are
        stacked, because "0.65" means nothing until you know it is µm.

    Anisotropy is the normal case, not the exception::

        Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
    """

    scale: Tuple[float, ...]
    translate: Tuple[float, ...] = ()
    axes: Tuple[str, ...] = ()
    units: str = "px"

    def __post_init__(self) -> None:
        try:
            scale = tuple(float(v) for v in self.scale)
        except (TypeError, ValueError):
            raise LayerError(
                f"scale must be a sequence of numbers, got {self.scale!r}"
            ) from None
        if not scale:
            raise LayerError("a spacing needs at least one axis")
        for i, v in enumerate(scale):
            if v == 0.0 or not math.isfinite(v):
                raise LayerError(
                    f"voxel size along axis {i} is {v!r}. A zero or non-finite "
                    f"scale collapses the axis: every world coordinate on it "
                    f"would resolve to element 0, and the overlay would be "
                    f"drawn a slice out of register with no visible symptom.")
        object.__setattr__(self, "scale", scale)

        translate = tuple(float(v) for v in self.translate) or (0.0,) * len(scale)
        if len(translate) != len(scale):
            raise LayerError(
                f"translate has {len(translate)} axes but scale has "
                f"{len(scale)}")
        if not all(math.isfinite(v) for v in translate):
            raise LayerError(f"translate must be finite, got {translate}")
        object.__setattr__(self, "translate", translate)

        axes = tuple(str(a) for a in self.axes) or _default_axes(len(scale))
        if len(axes) != len(scale):
            raise LayerError(
                f"axes {axes} does not match the {len(scale)} scale entries")
        if len(set(axes)) != len(axes):
            raise LayerError(f"axis names must be unique, got {axes}")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "units", str(self.units))

    # -- constructors ---------------------------------------------------
    @classmethod
    def isotropic(cls, ndim: int = 2, step: float = 1.0,
                  units: str = "px") -> "Spacing":
        """Equal spacing on every axis — the pixel-grid default."""
        return cls(scale=(float(step),) * int(ndim), units=units)

    @classmethod
    def from_map(cls, sizes: Mapping[str, float], *,
                 origin: Optional[Mapping[str, float]] = None,
                 units: str = "px") -> "Spacing":
        """Build from ``{"z": 2.0, "y": 0.65, "x": 0.65}``.

        Order is the mapping's own, which for a dict literal is the order it
        was written — the same order the array's axes are in. Naming the axes
        at the call site is the point: ``(2.0, 0.65, 0.65)`` on its own has
        been read backwards before.
        """
        names = tuple(str(k) for k in sizes)
        scale = tuple(float(v) for v in sizes.values())
        origin = origin or {}
        translate = tuple(float(origin.get(n, 0.0)) for n in names)
        return cls(scale=scale, translate=translate, axes=names, units=units)

    # -- queries --------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.scale)

    def axis_index(self, axis: str) -> int:
        """Position of ``axis`` in this spacing.

        :raises LayerError: if this spacing has no such axis.
        """
        try:
            return self.axes.index(str(axis))
        except ValueError:
            raise LayerError(
                f"no axis {axis!r} in {self.axes}") from None

    def has_axis(self, axis: str) -> bool:
        return str(axis) in self.axes

    def to_world(self, index: Sequence[float]) -> Tuple[float, ...]:
        """Data index → world coordinate, per axis."""
        idx = tuple(float(v) for v in index)
        if len(idx) != self.ndim:
            raise LayerError(
                f"index {idx} has {len(idx)} axes, spacing has {self.ndim}")
        return tuple(t + s * i
                     for t, s, i in zip(self.translate, self.scale, idx))

    def to_data(self, world: Sequence[float]) -> Tuple[float, ...]:
        """World coordinate → (fractional) data index, per axis."""
        pos = tuple(float(v) for v in world)
        if len(pos) != self.ndim:
            raise LayerError(
                f"world point {pos} has {len(pos)} axes, spacing has "
                f"{self.ndim}")
        return tuple((w - t) / s
                     for w, t, s in zip(pos, self.translate, self.scale))

    def world_map(self, index: Sequence[float]) -> Dict[str, float]:
        """:meth:`to_world` keyed by axis name."""
        return dict(zip(self.axes, self.to_world(index)))

    def data_from_map(self, world: Mapping[str, float]) -> Tuple[float, ...]:
        """Fractional data index from a ``{axis: world}`` mapping.

        Axes the mapping does not mention are taken to be 0 in world units,
        which is what a 2-D click on a 3-D layer means once the viewer has
        supplied the slice it is showing.
        """
        return self.to_data([float(world.get(a, 0.0)) for a in self.axes])

    def extent(self, shape: Sequence[int]) -> Dict[str, Tuple[float, float]]:
        """World bounding box of an array of ``shape``, keyed by axis.

        Measured to the OUTER EDGE of the end elements (half a voxel beyond
        the first and last centres), because that is the region the data
        actually covers — a canvas fitted to element centres clips half a
        voxel off every side, which at 2 µm z-steps is a visible slab.
        """
        if len(shape) != self.ndim:
            raise LayerError(
                f"shape {tuple(shape)} has {len(shape)} axes, spacing has "
                f"{self.ndim}")
        out: Dict[str, Tuple[float, float]] = {}
        for axis, n, s, t in zip(self.axes, shape, self.scale, self.translate):
            a = t - 0.5 * s
            b = t + s * (float(n) - 0.5)
            out[axis] = (min(a, b), max(a, b))
        return out

    def rescaled(self, **sizes: float) -> "Spacing":
        """A copy with some axes' voxel sizes replaced: ``sp.rescaled(z=1.5)``."""
        scale = list(self.scale)
        for axis, value in sizes.items():
            scale[self.axis_index(axis)] = float(value)
        return replace(self, scale=tuple(scale))

    def translated(self, **offsets: float) -> "Spacing":
        """A copy with some axes' origins replaced."""
        translate = list(self.translate)
        for axis, value in offsets.items():
            translate[self.axis_index(axis)] = float(value)
        return replace(self, translate=tuple(translate))

    def describe(self) -> str:
        """One line for a status bar: ``z 2, y 0.65, x 0.65 um``."""
        body = ", ".join(f"{a} {s:g}" for a, s in zip(self.axes, self.scale))
        return f"{body} {self.units}"


# ---------------------------------------------------------------------------
# Canvas — a window onto the world
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Canvas:
    """The world window a render fills: an origin, a step and a size.

    Rows run along ``axes[0]`` and columns along ``axes[1]``; every other
    world axis is pinned by :attr:`depth`. Naming the plane rather than
    assuming ``(y, x)`` is what makes an orthogonal view a different
    :class:`Canvas` over the same stack rather than a different renderer::

        Canvas.covering(stack, height=512, axes=("z", "x"), depth={"y": 40.0})

    :param origin: world coordinate of the centre of pixel ``(0, 0)``.
    :param step: world units per canvas pixel, along rows and columns. This is
        the zoom: halve it to zoom in.
    :param shape: ``(height, width)`` in canvas pixels.
    :param axes: which world axes rows and columns run along.
    :param depth: world coordinate for the axes not in the plane — the slice
        being shown. Axes absent from it are read as 0.
    """

    origin: Tuple[float, float]
    step: Tuple[float, float]
    shape: Tuple[int, int]
    axes: Tuple[str, str] = ("y", "x")
    depth: Mapping[str, float] = field(default_factory=dict)
    units: str = "px"

    def __post_init__(self) -> None:
        origin = tuple(float(v) for v in self.origin)
        step = tuple(float(v) for v in self.step)
        shape = tuple(int(v) for v in self.shape)
        axes = tuple(str(a) for a in self.axes)
        if len(origin) != 2 or len(step) != 2 or len(shape) != 2 or len(axes) != 2:
            raise LayerError(
                "a canvas is two-dimensional: origin, step, shape and axes "
                "all take exactly two entries")
        if axes[0] == axes[1]:
            raise LayerError(f"canvas rows and columns share axis {axes[0]!r}")
        if any(v <= 0 for v in shape):
            raise LayerError(f"canvas shape must be positive, got {shape}")
        for v in step:
            if v == 0.0 or not math.isfinite(v):
                raise LayerError(
                    f"canvas step must be non-zero and finite, got {step}")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "depth", MappingProxyType(
            {str(k): float(v) for k, v in dict(self.depth).items()}))
        object.__setattr__(self, "units", str(self.units))

    # -- constructors ---------------------------------------------------
    @classmethod
    def for_grid(cls, spacing: "Spacing", shape: Sequence[int], *,
                 axes: Tuple[str, str] = ("y", "x"),
                 depth: Optional[Mapping[str, float]] = None) -> "Canvas":
        """The canvas that samples ``spacing``'s own grid, one pixel per element.

        The identity render: a layer drawn onto ``Canvas.for_grid(layer.spacing,
        layer.shape)`` comes back at its native resolution and alignment, which
        is what a shapes-to-mask conversion needs (see
        :meth:`ShapesLayer.mask`).
        """
        r = spacing.axis_index(axes[0])
        c = spacing.axis_index(axes[1])
        return cls(origin=(spacing.translate[r], spacing.translate[c]),
                   step=(spacing.scale[r], spacing.scale[c]),
                   shape=(int(shape[r]), int(shape[c])),
                   axes=axes, depth=depth or {}, units=spacing.units)

    @classmethod
    def covering(cls, source: Any, *, height: Optional[int] = None,
                 width: Optional[int] = None,
                 axes: Tuple[str, str] = ("y", "x"),
                 depth: Optional[Mapping[str, float]] = None,
                 margin: float = 0.0) -> "Canvas":
        """A canvas showing all of ``source`` — a stack, a layer, or an extent.

        Exactly one of ``height`` / ``width`` fixes the resolution; the other
        follows from the world aspect ratio, so an anisotropic stack viewed in
        ``("z", "x")`` is not squashed. Giving both is allowed and stretches.

        :param source: a :class:`LayerStack`, a :class:`Layer`, or a
            ``{axis: (low, high)}`` extent mapping.
        :param margin: fraction of the extent to add on every side.
        """
        if isinstance(source, LayerStack):
            extent = source.world_extent()
            units = source.units
        elif isinstance(source, Layer):
            extent = source.world_extent()
            units = source.spacing.units
        else:
            extent = {str(k): (float(v[0]), float(v[1]))
                      for k, v in dict(source).items()}
            units = "px"
        missing = [a for a in axes if a not in extent]
        if missing:
            raise LayerError(
                f"cannot fit a canvas to axes {axes}: nothing in the stack "
                f"spans {missing} (it spans {sorted(extent)})")
        spans = []
        origins = []
        for a in axes:
            lo, hi = extent[a]
            pad = (hi - lo) * float(margin)
            lo, hi = lo - pad, hi + pad
            spans.append(max(hi - lo, _EPS))
            origins.append(lo)
        if height is None and width is None:
            height = 512
        if height is None:
            height = max(1, int(round(int(width) * spans[0] / spans[1])))
        if width is None:
            width = max(1, int(round(int(height) * spans[1] / spans[0])))
        height, width = max(1, int(height)), max(1, int(width))
        step = (spans[0] / height, spans[1] / width)
        # Pixel centres, not corners: half a step in from the extent edge.
        origin = (origins[0] + 0.5 * step[0], origins[1] + 0.5 * step[1])
        return cls(origin=origin, step=step, shape=(height, width), axes=axes,
                   depth=depth or {}, units=units)

    # -- queries --------------------------------------------------------
    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def row_world(self) -> np.ndarray:
        """World coordinate of every row centre, ``(height,)``."""
        return self.origin[0] + self.step[0] * np.arange(
            self.shape[0], dtype=np.float64)

    def column_world(self) -> np.ndarray:
        """World coordinate of every column centre, ``(width,)``."""
        return self.origin[1] + self.step[1] * np.arange(
            self.shape[1], dtype=np.float64)

    def world_at(self, row: float, column: float) -> Dict[str, float]:
        """The world point under canvas pixel ``(row, column)``.

        Includes the pinned :attr:`depth` axes, so the result is a complete
        world position a layer can be asked about.
        """
        out = dict(self.depth)
        out[self.axes[0]] = self.origin[0] + self.step[0] * float(row)
        out[self.axes[1]] = self.origin[1] + self.step[1] * float(column)
        return out

    def pixel_at(self, world: Mapping[str, float]) -> Tuple[float, float]:
        """The (fractional) canvas pixel a world point falls on.

        The inverse of :meth:`world_at`, and the reason a points layer and a
        labels layer at the same world coordinate land on the same pixel: they
        are both put through this, not through their own array indices.
        """
        return ((float(world[self.axes[0]]) - self.origin[0]) / self.step[0],
                (float(world[self.axes[1]]) - self.origin[1]) / self.step[1])

    def zoomed(self, factor: float,
               centre: Optional[Tuple[float, float]] = None) -> "Canvas":
        """A canvas ``factor``× closer, about ``centre`` (canvas pixels).

        ``factor > 1`` zooms in. The default centre is the middle of the view,
        which is what a keyboard zoom means; a wheel zoom passes the cursor.
        """
        factor = float(factor)
        if factor <= 0 or not math.isfinite(factor):
            raise LayerError(f"zoom factor must be positive, got {factor}")
        if centre is None:
            centre = ((self.shape[0] - 1) / 2.0, (self.shape[1] - 1) / 2.0)
        anchor = self.world_at(*centre)
        step = (self.step[0] / factor, self.step[1] / factor)
        origin = (anchor[self.axes[0]] - step[0] * float(centre[0]),
                  anchor[self.axes[1]] - step[1] * float(centre[1]))
        return replace(self, origin=origin, step=step)

    def panned(self, d_row: float, d_column: float) -> "Canvas":
        """A canvas moved by ``(d_row, d_column)`` canvas pixels."""
        return replace(self, origin=(
            self.origin[0] + self.step[0] * float(d_row),
            self.origin[1] + self.step[1] * float(d_column)))

    def resized(self, height: int, width: int) -> "Canvas":
        """The same world window at a different pixel size.

        The world *span* is held, not the step, so a widget resize shows the
        same field of view rather than more of the sample.
        """
        height, width = max(1, int(height)), max(1, int(width))
        span = (self.step[0] * self.shape[0], self.step[1] * self.shape[1])
        return replace(self, shape=(height, width),
                       step=(span[0] / height, span[1] / width))

    def at_depth(self, **coords: float) -> "Canvas":
        """The same window at a different slice: ``canvas.at_depth(z=12.0)``."""
        merged = dict(self.depth)
        merged.update({k: float(v) for k, v in coords.items()})
        return replace(self, depth=merged)


# ---------------------------------------------------------------------------
# Object identity for a labels layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldKey:
    """Which field a labels layer segments, in the schema's own key columns.

    A labels layer on its own knows that object 17 is object 17. Only the
    field it came from turns that into ``plate1_A_1_1_17`` — the key the UMAP,
    the plate view and the annotation grid all already use. Keys are built by
    handing a frame to :func:`spacr.selection.object_keys` rather than joining
    strings here, so this cannot drift from the identity the rest of the
    codebase agrees on.

    :param values: the field key columns and their values — everything in
        :data:`~spacr.selection.OBJECT_KEY_COLUMNS` except the object label
        (and the timepoint too, when ``timelapse``).
    :param timelapse: key each frame of an object separately, which requires a
        ``timeid`` in ``values``.
    """

    values: Mapping[str, Any]
    timelapse: bool = False

    def __post_init__(self) -> None:
        values = {str(k): v for k, v in dict(self.values).items()}
        needed = list(type(self).columns(timelapse=bool(self.timelapse)))
        missing = [c for c in needed if c not in values]
        if missing:
            raise LayerError(
                f"a field key needs {needed}; missing {missing}. Without them "
                f"a clicked object cannot be named in the same terms as the "
                f"measurement table, and the selection would reach no other "
                f"view.")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "timelapse", bool(self.timelapse))

    @classmethod
    def columns(cls, *, timelapse: bool = False) -> Tuple[str, ...]:
        """The key columns a field key of this flavour needs, label excluded."""
        label_column = OBJECT_KEY_COLUMNS[-1]
        if timelapse:
            from . import schema
            everything = schema.TIMEPOINT_KEY_COLUMNS + (label_column,)
        else:
            everything = OBJECT_KEY_COLUMNS
        return tuple(c for c in everything if c != label_column)

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *,
                 timelapse: bool = False) -> "FieldKey":
        """Take the key columns out of a measurement row (or any mapping).

        Anything else on the row is dropped — a field key is an identity, and
        carrying a measurement along with it would make two keys for the same
        field compare unequal.
        """
        wanted = cls.columns(timelapse=timelapse)
        return cls(values={c: row[c] for c in wanted if c in row},
                   timelapse=timelapse)

    def frame(self, labels: Iterable[int]):
        """A one-column-per-key-column frame for ``labels``, in their order."""
        import pandas as pd
        ids = [int(v) for v in labels]
        data = {c: [self.values[c]] * len(ids) for c in self.values}
        data[OBJECT_KEY_COLUMNS[-1]] = ids
        return pd.DataFrame(data)

    def object_keys(self, labels: Iterable[int]):
        """:class:`pandas.Index` of object keys for ``labels``, in order."""
        return object_keys(self.frame(labels), timelapse=self.timelapse)

    def object_key(self, label: int) -> str:
        """The one object key for ``label``."""
        return str(self.object_keys([int(label)])[0])


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerEvent:
    """Something changed. What a view listens for instead of polling.

    :param kind: ``"inserted"``, ``"removed"``, ``"moved"``, ``"renamed"``,
        ``"changed"`` (a display property), ``"data"`` (the array itself) or
        ``"selected"``.
    :param layer: the layer it happened to, or ``None`` for stack-wide events.
    :param index: where it is (or was) in the stack, or ``-1``.
    :param detail: the property name for ``"changed"``, else free text.
    """

    kind: str
    layer: Optional["Layer"] = None
    index: int = -1
    detail: str = ""

    #: The kinds that mean "the picture is different" — a view that only
    #: repaints can subscribe to these and ignore the rest.
    REPAINT: ClassVar[Tuple[str, ...]] = ("inserted", "removed", "moved",
                                          "changed", "data")


class Layer:
    """One thing in the stack: a name, a place in the world, and how to draw it.

    Subclasses implement :meth:`_draw`; everything shared — visibility,
    opacity, blending, spacing, notification — lives here.

    Display properties are plain attributes with setters that notify the
    stack, so a view repaints because the model changed rather than because
    the widget that changed it remembered to ask.
    """

    #: What kind of layer this is, for a view choosing an icon or an editor.
    kind: str = "layer"

    def __init__(self, *, name: str, spacing: Optional[Spacing] = None,
                 visible: bool = True, opacity: float = 1.0,
                 blending: str = Blending.TRANSLUCENT,
                 metadata: Optional[Mapping[str, Any]] = None):
        self._name = self._check_name(name)
        self._visible = bool(visible)
        self._opacity = self._check_opacity(opacity)
        self._blending = Blending.check(blending)
        self._spacing = spacing if spacing is not None else Spacing.isotropic(
            self.ndim)
        if self._spacing.ndim != self.ndim:
            raise LayerError(
                f"layer {self._name!r} has {self.ndim} spatial axes but its "
                f"spacing has {self._spacing.ndim} ({self._spacing.axes})")
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._stack: Optional["LayerStack"] = None

    # -- identity -------------------------------------------------------
    @staticmethod
    def _check_name(name: str) -> str:
        text = str(name).strip()
        if not text:
            raise LayerError("a layer needs a non-blank name")
        return text

    @staticmethod
    def _check_opacity(value: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise LayerError(f"opacity must be a number, got {value!r}") from None
        if not math.isfinite(v):
            raise LayerError(f"opacity must be finite, got {value!r}")
        # Clamped rather than refused: a slider that overshoots by a float
        # rounding error should not raise in a paint handler.
        return min(1.0, max(0.0, v))

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        text = self._check_name(value)
        if text == self._name:
            return
        if self._stack is not None:
            # Go through the stack so uniqueness is still enforced.
            self._stack.rename(self, text)
            return
        self._name = text

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        value = bool(value)
        if value != self._visible:
            self._visible = value
            self._notify("visible")

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        value = self._check_opacity(value)
        if value != self._opacity:
            self._opacity = value
            self._notify("opacity")

    @property
    def blending(self) -> str:
        return self._blending

    @blending.setter
    def blending(self, value: str) -> None:
        value = Blending.check(value)
        if value != self._blending:
            self._blending = value
            self._notify("blending")

    @property
    def spacing(self) -> Spacing:
        return self._spacing

    @spacing.setter
    def spacing(self, value: Spacing) -> None:
        if not isinstance(value, Spacing):
            raise LayerError(f"spacing must be a Spacing, got {value!r}")
        if value.ndim != self.ndim:
            raise LayerError(
                f"layer {self._name!r} has {self.ndim} spatial axes; the new "
                f"spacing has {value.ndim}")
        if self._stack is not None and self._stack.units != value.units:
            raise LayerError(
                f"layer {self._name!r} is in a stack measured in "
                f"{self._stack.units!r}; its spacing cannot become "
                f"{value.units!r} while it is there")
        self._spacing = value
        self._notify("spacing")

    @property
    def stack(self) -> Optional["LayerStack"]:
        """The stack this layer is in, or ``None``."""
        return self._stack

    # -- geometry -------------------------------------------------------
    @property
    def ndim(self) -> int:
        """How many spatial axes this layer has."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]:
        """The layer's spatial shape in elements, ``()`` when it has no grid."""
        return ()

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._spacing.axes

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The world box this layer occupies, keyed by axis."""
        raise NotImplementedError

    def to_world(self, index: Sequence[float]) -> Dict[str, float]:
        """Data index → ``{axis: world}``."""
        return self._spacing.world_map(index)

    def to_data(self, world: Mapping[str, float]) -> Tuple[float, ...]:
        """``{axis: world}`` → fractional data index."""
        return self._spacing.data_from_map(world)

    # -- rendering ------------------------------------------------------
    def render(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Draw onto ``canvas``; returns ``(rgb, coverage)``.

        ``rgb`` is ``(H, W, 3)`` float32 in 0–1 and ``coverage`` is ``(H, W)``
        float32 in 0–1 saying where this layer has anything to say. Opacity and
        blending are NOT applied here — the stack applies them, so a layer's
        own render is the same whatever it is composited with.
        """
        rgb, coverage = self._draw(canvas)
        return (np.asarray(rgb, dtype=np.float32),
                np.asarray(coverage, dtype=np.float32))

    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _blank(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        h, w = canvas.shape
        return (np.zeros((h, w, 3), dtype=np.float32),
                np.zeros((h, w), dtype=np.float32))

    def _sample_index(self, canvas: Canvas
                      ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Nearest-neighbour data index for every canvas pixel.

        Returns one index array per spatial axis, broadcast to the canvas
        shape, plus a bool mask of the pixels that actually fall inside the
        data. Axes the canvas does not span are pinned at
        :attr:`Canvas.depth`, so a 2-D layer under a 3-D canvas is drawn on
        every slice (it occupies all of them) while a 3-D layer is not.
        """
        h, w = canvas.shape
        sp = self._spacing
        shape = self.shape
        index: List[np.ndarray] = []
        valid = np.ones((h, w), dtype=bool)
        for axis_at, (axis, size) in enumerate(zip(sp.axes, shape)):
            if axis == canvas.axes[0]:
                world = canvas.row_world()[:, None]
            elif axis == canvas.axes[1]:
                world = canvas.column_world()[None, :]
            else:
                world = np.float64(canvas.depth.get(axis, 0.0))
            frac = (world - sp.translate[axis_at]) / sp.scale[axis_at]
            idx = np.rint(np.broadcast_to(frac, (h, w))).astype(np.int64)
            valid &= (idx >= 0) & (idx < int(size))
            index.append(np.clip(idx, 0, max(int(size) - 1, 0)))
        return tuple(index), valid

    # -- plumbing -------------------------------------------------------
    def _notify(self, detail: str, kind: str = "changed") -> None:
        if self._stack is not None:
            self._stack._emit(LayerEvent(kind=kind, layer=self,
                                         index=self._stack.index(self),
                                         detail=detail))

    def describe(self) -> str:
        """One line for a layer list row."""
        vis = "" if self._visible else " · hidden"
        return (f"{self._name} ({self.kind}) · {self._blending} · "
                f"{self._opacity:.0%}{vis}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self._name!r} {self.shape}>"


class ImageLayer(Layer):
    """Intensity data — one channel or many, each with its own LUT.

    Channels are composited *additively within the layer*, which is what makes
    a two-channel field read as one picture: a green nucleus channel and a
    magenta pathogen channel overlap as white rather than as whichever channel
    happened to be second. Across layers the user picks the blending.

    :param data: the array. The spatial axes are every axis except
        ``channel_axis``, in order.
    :param channel_axis: which axis holds channels, if any. Negative indices
        work. A ``(H, W, 3)`` RGB stack is ``channel_axis=-1``.
    :param colormaps: one per channel; defaults walk
        :data:`DEFAULT_CHANNEL_COLORMAPS` (a single-channel image gets grey).
    :param contrast_limits: ``(low, high)`` per channel in data units.
        ``None`` means the :data:`DEFAULT_PERCENTILES` stretch, computed once
        and remembered so that panning does not change the brightness — a
        per-view stretch is how two crops of the same field end up looking
        like different exposures.
    :param channel_visible: per-channel visibility, for turning one stain off
        without splitting the layer.
    """

    kind = "image"

    def __init__(self, data: Any, *, name: str = "image",
                 channel_axis: Optional[int] = None,
                 colormaps: Optional[Sequence[Any]] = None,
                 contrast_limits: Optional[Any] = None,
                 channel_names: Optional[Sequence[str]] = None,
                 channel_visible: Optional[Sequence[bool]] = None,
                 **kwargs: Any):
        array = np.asarray(data)
        if array.ndim == 0:
            raise LayerError("an image layer needs at least a 1-D array")
        if channel_axis is not None:
            axis = int(channel_axis) % array.ndim
            spatial = tuple(s for i, s in enumerate(array.shape) if i != axis)
            self._planes = [np.take(array, c, axis=axis)
                            for c in range(array.shape[axis])]
        else:
            axis = None
            spatial = tuple(array.shape)
            self._planes = [array]
        if len(spatial) not in (2, 3):
            raise LayerError(
                f"an image layer needs 2 or 3 spatial axes, got {spatial}. "
                f"Pass channel_axis if one of those axes is channels.")
        self._data = array
        self._channel_axis = axis
        self._spatial = spatial
        n = len(self._planes)
        super().__init__(name=name, **kwargs)
        self._colormaps = self._resolve_colormaps(colormaps, n)
        self._limits: List[Optional[Tuple[float, float]]] = \
            self._resolve_limits(contrast_limits, n)
        self._channel_names = tuple(
            str(c) for c in (channel_names or
                             [f"ch{i}" for i in range(n)]))
        if len(self._channel_names) != n:
            raise LayerError(
                f"{n} channels but {len(self._channel_names)} channel names")
        self._channel_visible = list(
            bool(v) for v in (channel_visible if channel_visible is not None
                              else [True] * n))
        if len(self._channel_visible) != n:
            raise LayerError(
                f"{n} channels but {len(self._channel_visible)} visibilities")

    # -- construction helpers -------------------------------------------
    @staticmethod
    def _resolve_colormaps(spec: Optional[Sequence[Any]],
                           n: int) -> List[Colormap]:
        if spec is None:
            if n == 1:
                return [colormap("gray")]
            return [colormap(DEFAULT_CHANNEL_COLORMAPS[
                i % len(DEFAULT_CHANNEL_COLORMAPS)]) for i in range(n)]
        if isinstance(spec, (str, Colormap)):
            return [colormap(spec) for _ in range(n)]
        maps = [colormap(s) for s in spec]
        if len(maps) != n:
            raise LayerError(f"{n} channels but {len(maps)} colormaps")
        return maps

    @staticmethod
    def _resolve_limits(spec: Any, n: int) -> List[Optional[Tuple[float, float]]]:
        if spec is None:
            return [None] * n
        pairs = list(spec)
        if len(pairs) == 2 and all(np.isscalar(v) for v in pairs):
            pairs = [tuple(pairs)] * n
        if len(pairs) != n:
            raise LayerError(f"{n} channels but {len(pairs)} contrast limits")
        out: List[Optional[Tuple[float, float]]] = []
        for p in pairs:
            if p is None:
                out.append(None)
                continue
            lo, hi = (float(p[0]), float(p[1]))
            if hi <= lo:
                raise LayerError(
                    f"contrast limits must increase, got ({lo}, {hi})")
            out.append((lo, hi))
        return out

    # -- data -----------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def ndim(self) -> int:
        return len(self._spatial)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._spatial

    @property
    def n_channels(self) -> int:
        return len(self._planes)

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return self._channel_names

    def channel_data(self, channel: int) -> np.ndarray:
        """The spatial array for one channel (a view, not a copy)."""
        return self._planes[int(channel)]

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        return self._spacing.extent(self._spatial)

    # -- display --------------------------------------------------------
    @property
    def colormaps(self) -> Tuple[Colormap, ...]:
        return tuple(self._colormaps)

    def set_colormap(self, value: Any, channel: int = 0) -> None:
        """Give one channel a new LUT."""
        cm = colormap(value)
        if self._colormaps[int(channel)] != cm:
            self._colormaps[int(channel)] = cm
            self._notify("colormap")

    @property
    def colormap(self) -> Colormap:
        """Channel 0's LUT — the one a single-channel layer has."""
        return self._colormaps[0]

    @colormap.setter
    def colormap(self, value: Any) -> None:
        self.set_colormap(value, 0)

    def contrast_limits(self, channel: int = 0) -> Tuple[float, float]:
        """The limits in use for ``channel``, computing the default if needed."""
        channel = int(channel)
        if self._limits[channel] is None:
            plane = self._planes[channel]
            if plane.size:
                lo, hi = np.percentile(np.asarray(plane, dtype=np.float64),
                                       DEFAULT_PERCENTILES)
            else:
                lo, hi = 0.0, 1.0
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                lo = float(np.min(plane)) if plane.size else 0.0
                hi = float(np.max(plane)) if plane.size else 1.0
                if hi <= lo:
                    hi = lo + 1.0
            self._limits[channel] = (float(lo), float(hi))
        return self._limits[channel]  # type: ignore[return-value]

    def set_contrast_limits(self, low: float, high: float,
                            channel: int = 0) -> None:
        """Set one channel's limits.

        :raises LayerError: if they do not increase — an inverted pair renders
            a black image, which reads as "no signal" rather than as a bad
            setting.
        """
        lo, hi = float(low), float(high)
        if hi <= lo:
            raise LayerError(
                f"contrast limits must increase, got ({lo}, {hi})")
        if self._limits[int(channel)] != (lo, hi):
            self._limits[int(channel)] = (lo, hi)
            self._notify("contrast_limits")

    def auto_contrast(self, percentiles: Tuple[float, float] = DEFAULT_PERCENTILES
                      ) -> None:
        """Recompute every channel's limits from the data."""
        for c, plane in enumerate(self._planes):
            if plane.size:
                lo, hi = np.percentile(np.asarray(plane, dtype=np.float64),
                                       percentiles)
                if hi > lo:
                    self._limits[c] = (float(lo), float(hi))
                    continue
            self._limits[c] = None
        self._notify("contrast_limits")

    def channel_is_visible(self, channel: int) -> bool:
        return self._channel_visible[int(channel)]

    def set_channel_visible(self, channel: int, visible: bool) -> None:
        channel = int(channel)
        if self._channel_visible[channel] != bool(visible):
            self._channel_visible[channel] = bool(visible)
            self._notify("channel_visible")

    # -- rendering ------------------------------------------------------
    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        index, valid = self._sample_index(canvas)
        h, w = canvas.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for c, plane in enumerate(self._planes):
            if not self._channel_visible[c]:
                continue
            lo, hi = self.contrast_limits(c)
            values = np.asarray(plane[index], dtype=np.float64)
            t = np.clip((values - lo) / max(hi - lo, _EPS), 0.0, 1.0)
            rgb += self._colormaps[c].map(t)
        np.clip(rgb, 0.0, 1.0, out=rgb)
        rgb[~valid] = 0.0
        return rgb, valid.astype(np.float32)


class LabelsLayer(Layer):
    """An integer segmentation mask. 0 is background and draws as nothing.

    Colours come from :func:`label_color`, so an object keeps its colour
    between sessions and between this viewer and the live preview.

    :param field: the :class:`FieldKey` this mask segments. Optional, but
        without it a click can only say "label 17" — with it, the click can
        publish ``plate1_A_1_1_17`` to every other view.
    """

    kind = "labels"

    def __init__(self, data: Any, *, name: str = "labels",
                 field: Optional[FieldKey] = None, seed: int = 0,
                 **kwargs: Any):
        array = np.asarray(data)
        if array.ndim not in (2, 3):
            raise LayerError(
                f"a labels layer needs a 2-D or 3-D array, got {array.shape}")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(np.mod(array, 1), 0)):
                raise LayerError(
                    f"a labels layer needs integer labels, got dtype "
                    f"{array.dtype} with fractional values")
            array = array.astype(np.int64)
        self._data = array
        self._seed = int(seed)
        self._field = field
        self._selected_label = 0
        super().__init__(name=name, **kwargs)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        array = np.asarray(value)
        if array.shape != self._data.shape:
            raise LayerError(
                f"labels layer {self.name!r} is {self._data.shape}; the new "
                f"array is {array.shape}. Replace the layer rather than its "
                f"data if the grid really changed — the spacing describes the "
                f"old grid.")
        self._data = array
        self._notify("data", kind="data")

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def field(self) -> Optional[FieldKey]:
        return self._field

    @field.setter
    def field(self, value: Optional[FieldKey]) -> None:
        if value is not None and not isinstance(value, FieldKey):
            raise LayerError(f"field must be a FieldKey, got {value!r}")
        self._field = value
        self._notify("field")

    @property
    def selected_label(self) -> int:
        """The label the user last picked. 0 means none."""
        return self._selected_label

    @selected_label.setter
    def selected_label(self, value: int) -> None:
        value = int(value)
        if value != self._selected_label:
            self._selected_label = value
            self._notify("selected_label")

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        return self._spacing.extent(self._data.shape)

    def labels(self) -> np.ndarray:
        """Every non-zero label present, sorted."""
        unique = np.unique(self._data)
        return unique[unique != 0]

    # -- picking --------------------------------------------------------
    def label_at_world(self, world: Mapping[str, float]) -> int:
        """The label under a world point, or 0 for background / outside."""
        frac = self._spacing.data_from_map(world)
        idx = []
        for value, size in zip(frac, self._data.shape):
            i = int(round(value))
            if i < 0 or i >= int(size):
                return 0
            idx.append(i)
        return int(self._data[tuple(idx)])

    def object_key_at_world(self, world: Mapping[str, float]) -> Optional[str]:
        """The measurement-table key of the object under a world point.

        ``None`` for background, or when this layer was not told which field
        it segments. This is the whole reason a labels layer carries a
        :class:`FieldKey`: the string this returns is the same string the
        UMAP, the plate view and the annotation grid use, so publishing it
        through :mod:`spacr.qt.linked_selection` highlights the same cell
        everywhere.
        """
        label = self.label_at_world(world)
        if label == 0 or self._field is None:
            return None
        return self._field.object_key(label)

    def object_keys(self, labels: Optional[Iterable[int]] = None):
        """Keys for ``labels`` (default: every label present), in order."""
        if self._field is None:
            raise LayerError(
                f"labels layer {self.name!r} has no field key, so its objects "
                f"cannot be named in measurement-table terms. Give it a "
                f"FieldKey when the field it came from is known.")
        ids = self.labels() if labels is None else list(labels)
        return self._field.object_keys(ids)

    # -- editing (the seam the brush item builds on) ---------------------
    def paint(self, world: Mapping[str, float], label: int, *,
              radius: float = 0.0) -> int:
        """Set every element within ``radius`` world units of a point.

        The brush is a ball in WORLD space, so on an anisotropic stack it
        covers fewer z-slices than y-rows — which is what the user means by a
        5 µm brush. A brush measured in array elements would be a 5-slice
        cylinder, i.e. 10 µm deep and 3.25 µm wide on an ordinary spaCR stack.

        :returns: how many elements changed.
        """
        sp = self._spacing
        centre = sp.data_from_map(world)
        radius = float(radius)
        slices = []
        offsets = []
        for axis_at, size in enumerate(self._data.shape):
            reach = radius / abs(sp.scale[axis_at])
            lo = max(0, int(math.floor(centre[axis_at] - reach)))
            hi = min(int(size), int(math.ceil(centre[axis_at] + reach)) + 1)
            if lo >= hi:
                return 0
            slices.append(slice(lo, hi))
            offsets.append((np.arange(lo, hi, dtype=np.float64)
                            - centre[axis_at]) * abs(sp.scale[axis_at]))
        dist2 = np.zeros([len(o) for o in offsets], dtype=np.float64)
        for axis_at, off in enumerate(offsets):
            shaped = off.reshape([-1 if i == axis_at else 1
                                  for i in range(len(offsets))])
            dist2 = dist2 + shaped ** 2
        inside = dist2 <= max(radius, _EPS) ** 2
        region = self._data[tuple(slices)]
        changed = int(np.count_nonzero(inside & (region != int(label))))
        if changed:
            region[inside] = int(label)
            self._notify("paint", kind="data")
        return changed

    # -- rendering ------------------------------------------------------
    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        index, valid = self._sample_index(canvas)
        values = self._data[index]
        present = valid & (values != 0)
        rgb = label_colors(np.where(present, values, 0), seed=self._seed)
        rgb[~present] = 0.0
        return rgb, present.astype(np.float32)


class PointsLayer(Layer):
    """Points in the world — centroids, counted objects, clicked markers.

    Coordinates are stored in DATA units with the layer's own spacing, exactly
    like an image, so a points layer built from a centroid table in pixel
    coordinates lines up with the mask those centroids came from without the
    caller converting anything. Use :meth:`add_world` /
    :attr:`world` when you have world coordinates already.

    :param data: ``(N, ndim)`` array of data coordinates, in the spacing's
        axis order.
    :param size: point DIAMETER in world units — scalar or one per point.
        World, not pixels: a 5 µm marker is 5 µm at every zoom, and on an
        anisotropic stack it is a sphere rather than an ellipsoid.
    :param properties: ``{name: (N,) array}`` rider columns — what the
        counting item hangs its categories off.
    """

    kind = "points"

    def __init__(self, data: Any = None, *, name: str = "points",
                 ndim: int = 2, size: Any = 10.0,
                 face_color: Any = "yellow", border_color: Any = "black",
                 border_width: float = 0.0,
                 properties: Optional[Mapping[str, Any]] = None,
                 **kwargs: Any):
        array = self._as_points(data, ndim)
        self._ndim = array.shape[1]
        self._data = array
        super().__init__(name=name, **kwargs)
        self._size = self._as_sizes(size, len(array))
        # Kept apart from `_size` so that the layer's declared size survives an
        # empty layer: `_size` is empty until the first point exists, and a
        # counting layer starts empty by definition.
        self._default_size = (float(size) if np.isscalar(size)
                              else (float(self._size[0]) if len(self._size)
                                    else 10.0))
        self._face = to_rgba(face_color)
        self._border = to_rgba(border_color)
        self._border_width = max(0.0, float(border_width))
        self.properties: Dict[str, np.ndarray] = {
            str(k): np.asarray(v) for k, v in dict(properties or {}).items()}
        for key, column in self.properties.items():
            if len(column) != len(array):
                raise LayerError(
                    f"property {key!r} has {len(column)} values for "
                    f"{len(array)} points")

    @staticmethod
    def _as_points(data: Any, ndim: int) -> np.ndarray:
        if data is None:
            return np.zeros((0, int(ndim)), dtype=np.float64)
        array = np.asarray(data, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise LayerError(
                f"points must be an (N, ndim) array, got shape {array.shape}")
        if array.size == 0:
            return np.zeros((0, int(ndim)), dtype=np.float64)
        return array

    @staticmethod
    def _as_sizes(size: Any, n: int) -> np.ndarray:
        if np.isscalar(size):
            return np.full(n, float(size), dtype=np.float64)
        array = np.asarray(size, dtype=np.float64).reshape(-1)
        if len(array) != n:
            raise LayerError(f"{n} points but {len(array)} sizes")
        return array

    # -- data -----------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """The points in data coordinates, ``(N, ndim)``."""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        array = self._as_points(value, self._ndim)
        if array.shape[1] != self._ndim:
            raise LayerError(
                f"points layer {self.name!r} is {self._ndim}-D; the new array "
                f"is {array.shape[1]}-D")
        stale = [k for k, v in self.properties.items() if len(v) != len(array)]
        if stale:
            raise LayerError(
                f"replacing the points of {self.name!r} would leave "
                f"{stale} holding {len(self.properties[stale[0]])} values for "
                f"{len(array)} points. Clear the properties first, or replace "
                f"the layer — a property column silently out of step with its "
                f"points mislabels every one of them.")
        self._data = array
        self._size = self._as_sizes(self._default_size, len(array))
        self._notify("data", kind="data")

    @property
    def world(self) -> np.ndarray:
        """The points in world coordinates, ``(N, ndim)``."""
        scale = np.asarray(self._spacing.scale, dtype=np.float64)
        offset = np.asarray(self._spacing.translate, dtype=np.float64)
        return self._data * scale + offset

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self) -> np.ndarray:
        """Per-point diameters in world units."""
        return self._size

    def set_size(self, value: Any) -> None:
        """Set every point's diameter, or one per point."""
        self._size = self._as_sizes(value, len(self._data))
        if np.isscalar(value):
            self._default_size = float(value)
        self._notify("size")

    @property
    def face_color(self) -> Tuple[float, float, float, float]:
        return self._face

    @face_color.setter
    def face_color(self, value: Any) -> None:
        self._face = to_rgba(value)
        self._notify("face_color")

    @property
    def border_color(self) -> Tuple[float, float, float, float]:
        return self._border

    @border_color.setter
    def border_color(self, value: Any) -> None:
        self._border = to_rgba(value)
        self._notify("border_color")

    @property
    def border_width(self) -> float:
        """Border thickness in world units. 0 draws no border."""
        return self._border_width

    @border_width.setter
    def border_width(self, value: float) -> None:
        self._border_width = max(0.0, float(value))
        self._notify("border_width")

    def add(self, point: Sequence[float], *, size: Optional[float] = None,
            **properties: Any) -> int:
        """Append one point in DATA coordinates; returns its index."""
        row = np.asarray(point, dtype=np.float64).reshape(-1)
        if row.size != self._ndim:
            raise LayerError(
                f"a point in this layer has {self._ndim} coordinates, got "
                f"{row.size}")
        self._data = np.vstack([self._data, row[None, :]])
        self._size = np.append(
            self._size,
            self._default_size if size is None else float(size))
        for key in set(self.properties) | set(properties):
            column = self.properties.get(key)
            value = properties.get(key)
            if column is None:
                column = np.array([None] * (len(self._data) - 1), dtype=object)
            self.properties[key] = np.append(column.astype(object), value)
        self._notify("add", kind="data")
        return len(self._data) - 1

    def add_world(self, world: Mapping[str, float], **kwargs: Any) -> int:
        """Append one point given as ``{axis: world}``."""
        return self.add(self._spacing.data_from_map(world), **kwargs)

    def remove(self, index: int) -> None:
        """Drop one point."""
        i = int(index)
        if i < 0 or i >= len(self._data):
            raise LayerError(
                f"no point {index} in a layer of {len(self._data)}")
        self._data = np.delete(self._data, i, axis=0)
        self._size = np.delete(self._size, i)
        self.properties = {k: np.delete(v, i)
                           for k, v in self.properties.items()}
        self._notify("remove", kind="data")

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        world = self.world
        if len(world) == 0:
            return {a: (0.0, 0.0) for a in self._spacing.axes}
        half = self._size[:, None] / 2.0
        return {a: (float(np.min(world[:, i] - half[:, 0])),
                    float(np.max(world[:, i] + half[:, 0])))
                for i, a in enumerate(self._spacing.axes)}

    def nearest(self, world: Mapping[str, float]) -> Optional[int]:
        """Index of the point whose disc contains ``world``, nearest first."""
        if len(self._data) == 0:
            return None
        target = np.array([float(world.get(a, 0.0))
                           for a in self._spacing.axes], dtype=np.float64)
        d = np.linalg.norm(self.world - target[None, :], axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= self._size[i] / 2.0 else None

    # -- rendering ------------------------------------------------------
    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        rgb, coverage = self._blank(canvas)
        if len(self._data) == 0:
            return rgb, coverage
        axes = self._spacing.axes
        try:
            row_axis = axes.index(canvas.axes[0])
            col_axis = axes.index(canvas.axes[1])
        except ValueError:
            # This layer does not live in the plane being drawn.
            return rgb, coverage
        off_plane = [(i, canvas.depth.get(a, 0.0))
                     for i, a in enumerate(axes) if a not in canvas.axes]
        world = self.world
        h, w = canvas.shape
        rows = np.arange(h, dtype=np.float64)
        cols = np.arange(w, dtype=np.float64)
        face = np.asarray(self._face[:3], dtype=np.float32)
        border = np.asarray(self._border[:3], dtype=np.float32)
        for p in range(len(world)):
            radius = float(self._size[p]) / 2.0
            if radius <= 0:
                continue
            gap2 = sum((world[p, i] - depth) ** 2 for i, depth in off_plane)
            if gap2 >= radius ** 2:
                continue
            effective = math.sqrt(radius ** 2 - gap2)
            centre_row = (world[p, row_axis] - canvas.origin[0]) / canvas.step[0]
            centre_col = (world[p, col_axis] - canvas.origin[1]) / canvas.step[1]
            span_row = effective / abs(canvas.step[0])
            span_col = effective / abs(canvas.step[1])
            r0 = max(0, int(math.floor(centre_row - span_row)))
            r1 = min(h, int(math.ceil(centre_row + span_row)) + 1)
            c0 = max(0, int(math.floor(centre_col - span_col)))
            c1 = min(w, int(math.ceil(centre_col + span_col)) + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            dr = (rows[r0:r1, None] - centre_row) / max(span_row, _EPS)
            dc = (cols[None, c0:c1] - centre_col) / max(span_col, _EPS)
            norm = dr ** 2 + dc ** 2
            inside = norm <= 1.0
            if not inside.any():
                continue
            tile = rgb[r0:r1, c0:c1]
            tile[inside] = face
            if self._border_width > 0:
                inner = max(0.0, effective - self._border_width) / max(
                    effective, _EPS)
                ring = inside & (norm >= inner ** 2)
                tile[ring] = border
            rgb[r0:r1, c0:c1] = tile
            cov = coverage[r0:r1, c0:c1]
            cov[inside] = np.maximum(cov[inside], float(self._face[3]))
            if self._border_width > 0:
                inner = max(0.0, effective - self._border_width) / max(
                    effective, _EPS)
                ring = inside & (norm >= inner ** 2)
                cov[ring] = float(self._border[3])
            coverage[r0:r1, c0:c1] = cov
        return rgb, coverage


@dataclass(eq=False)
class Shape:
    """One drawn region: a polygon, rectangle, ellipse, line or path.

    Compared by identity (``eq=False``): a generated ``__eq__`` would compare
    the vertex arrays elementwise and raise "truth value of an array is
    ambiguous" from anything as ordinary as ``shape in layer.shapes``.

    :param kind: ``"polygon"``, ``"rectangle"``, ``"ellipse"``, ``"line"`` or
        ``"path"``. The first three enclose an area and can be turned into a
        mask; the last two are open and only have an outline.
    :param data: ``(M, ndim)`` vertices in DATA coordinates, in the layer's
        axis order. A rectangle and an ellipse are stored as their four
        bounding corners.
    :param name: what the ROI is called, so a mask can be attributed.
    """

    kind: str
    data: np.ndarray
    face_color: Any = (1.0, 1.0, 1.0, 0.25)
    edge_color: Any = "yellow"
    edge_width: float = 1.0
    name: str = ""

    #: The kinds that enclose an area. ``ClassVar``, so the dataclass does not
    #: turn it into a constructor argument.
    CLOSED: ClassVar[Tuple[str, ...]] = ("polygon", "rectangle", "ellipse")
    #: Every kind.
    KINDS: ClassVar[Tuple[str, ...]] = ("polygon", "rectangle", "ellipse",
                                        "line", "path")

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip().lower()
        if self.kind not in self.KINDS:
            raise LayerError(
                f"unknown shape kind {self.kind!r}; use one of "
                f"{list(self.KINDS)}")
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 2:
            raise LayerError(
                f"a {self.kind} needs an (M, ndim) array of at least two "
                f"vertices, got shape {data.shape}")
        if self.kind in ("rectangle", "ellipse"):
            if data.shape[0] == 2:
                data = self._corners(data)
            elif data.shape[0] != 4:
                raise LayerError(
                    f"a {self.kind} is given as two opposite corners or four "
                    f"corners, got {data.shape[0]}")
        if self.kind == "polygon" and data.shape[0] < 3:
            raise LayerError("a polygon needs at least three vertices")
        self.data = data
        self.face_color = to_rgba(self.face_color)
        self.edge_color = to_rgba(self.edge_color)
        self.edge_width = max(0.0, float(self.edge_width))
        self.name = str(self.name)

    @staticmethod
    def _corners(two: np.ndarray) -> np.ndarray:
        """Expand two opposite corners into four, in order."""
        lo = np.minimum(two[0], two[1])
        hi = np.maximum(two[0], two[1])
        out = np.tile(lo, (4, 1))
        out[1, -1] = hi[-1]
        out[2] = hi
        out[3, -2] = hi[-2]
        return out

    @property
    def ndim(self) -> int:
        return int(self.data.shape[1])

    @property
    def is_closed(self) -> bool:
        return self.kind in self.CLOSED


class ShapesLayer(Layer):
    """Drawn regions of interest — the layer Measure will later read.

    Shapes are geometry, not pixels: they are stored as vertices in data
    coordinates and rasterised on demand, so the same ROI can be turned into a
    mask for a full-resolution mask layer and for a downsampled preview and
    mean the same region in both. :meth:`mask` is that conversion.
    """

    kind = "shapes"

    def __init__(self, shapes: Optional[Iterable[Shape]] = None, *,
                 name: str = "shapes", ndim: int = 2, **kwargs: Any):
        self._shapes: List[Shape] = list(shapes or [])
        dims = {s.ndim for s in self._shapes}
        if len(dims) > 1:
            raise LayerError(
                f"every shape in a layer needs the same number of axes, got "
                f"{sorted(dims)}")
        self._ndim = dims.pop() if dims else int(ndim)
        super().__init__(name=name, **kwargs)

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shapes(self) -> Tuple[Shape, ...]:
        return tuple(self._shapes)

    def __len__(self) -> int:
        return len(self._shapes)

    def add(self, shape: Shape) -> int:
        """Append a shape; returns its index."""
        if not isinstance(shape, Shape):
            raise LayerError(f"expected a Shape, got {shape!r}")
        if shape.ndim != self._ndim:
            raise LayerError(
                f"this layer holds {self._ndim}-D shapes; that one is "
                f"{shape.ndim}-D")
        self._shapes.append(shape)
        self._notify("add", kind="data")
        return len(self._shapes) - 1

    def add_polygon(self, vertices: Any, **kwargs: Any) -> int:
        return self.add(Shape("polygon", vertices, **kwargs))

    def add_rectangle(self, corner_a: Any, corner_b: Any, **kwargs: Any) -> int:
        return self.add(Shape("rectangle", np.asarray([corner_a, corner_b]),
                              **kwargs))

    def add_ellipse(self, corner_a: Any, corner_b: Any, **kwargs: Any) -> int:
        return self.add(Shape("ellipse", np.asarray([corner_a, corner_b]),
                              **kwargs))

    def add_path(self, vertices: Any, **kwargs: Any) -> int:
        return self.add(Shape("path", vertices, **kwargs))

    def remove(self, index: int) -> Shape:
        """Drop a shape and return it."""
        try:
            shape = self._shapes.pop(int(index))
        except IndexError:
            raise LayerError(
                f"no shape {index} in a layer of {len(self._shapes)}") from None
        self._notify("remove", kind="data")
        return shape

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        axes = self._spacing.axes
        if not self._shapes:
            return {a: (0.0, 0.0) for a in axes}
        scale = np.asarray(self._spacing.scale, dtype=np.float64)
        offset = np.asarray(self._spacing.translate, dtype=np.float64)
        stacked = np.vstack([s.data * scale + offset for s in self._shapes])
        return {a: (float(np.min(stacked[:, i])), float(np.max(stacked[:, i])))
                for i, a in enumerate(axes)}

    # -- rasterisation ---------------------------------------------------
    def _plane_vertices(self, shape: Shape, canvas: Canvas
                        ) -> Optional[np.ndarray]:
        """Shape vertices as ``(M, 2)`` world coords on the canvas plane.

        ``None`` when the shape does not lie in the plane being drawn, which
        for a 3-D shapes layer is most of the slices.
        """
        axes = self._spacing.axes
        try:
            row_axis = axes.index(canvas.axes[0])
            col_axis = axes.index(canvas.axes[1])
        except ValueError:
            return None
        scale = np.asarray(self._spacing.scale, dtype=np.float64)
        offset = np.asarray(self._spacing.translate, dtype=np.float64)
        world = shape.data * scale + offset
        for i, axis in enumerate(axes):
            if axis in canvas.axes:
                continue
            depth = canvas.depth.get(axis, 0.0)
            tolerance = max(abs(scale[i]) / 2.0, _EPS)
            if abs(float(np.mean(world[:, i])) - depth) > tolerance:
                return None
        return np.column_stack([world[:, row_axis], world[:, col_axis]])

    @staticmethod
    def _inside_polygon(rows: np.ndarray, cols: np.ndarray,
                        vertices: np.ndarray) -> np.ndarray:
        """Even-odd point-in-polygon over a canvas grid."""
        vr, vc = vertices[:, 0], vertices[:, 1]
        inside = np.zeros((rows.size, cols.size), dtype=bool)
        r = rows[:, None]
        c = cols[None, :]
        n = len(vertices)
        j = n - 1
        for i in range(n):
            dr = vr[j] - vr[i]
            crosses = (vr[i] > r) != (vr[j] > r)
            # `dr == 0` only where `crosses` is False, so the guarded divide
            # never contributes; guarding it keeps numpy from warning.
            edge_c = vc[i] + (r - vr[i]) * (vc[j] - vc[i]) / (dr if dr else 1.0)
            inside ^= crosses & (c < edge_c)
            j = i
        return inside

    @staticmethod
    def _inside_ellipse(rows: np.ndarray, cols: np.ndarray,
                        vertices: np.ndarray) -> np.ndarray:
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        centre = (lo + hi) / 2.0
        radii = np.maximum((hi - lo) / 2.0, _EPS)
        dr = (rows[:, None] - centre[0]) / radii[0]
        dc = (cols[None, :] - centre[1]) / radii[1]
        return dr ** 2 + dc ** 2 <= 1.0

    @staticmethod
    def _on_edge(rows: np.ndarray, cols: np.ndarray, vertices: np.ndarray,
                 width: float, closed: bool) -> np.ndarray:
        """Pixels within ``width``/2 world units of the outline."""
        half = max(width, _EPS) / 2.0
        out = np.zeros((rows.size, cols.size), dtype=bool)
        r = rows[:, None]
        c = cols[None, :]
        n = len(vertices)
        pairs = range(n) if closed else range(n - 1)
        for i in pairs:
            a = vertices[i]
            b = vertices[(i + 1) % n]
            dr, dc = b[0] - a[0], b[1] - a[1]
            length2 = dr * dr + dc * dc
            if length2 <= _EPS:
                dist2 = (r - a[0]) ** 2 + (c - a[1]) ** 2
            else:
                t = ((r - a[0]) * dr + (c - a[1]) * dc) / length2
                t = np.clip(t, 0.0, 1.0)
                dist2 = (r - (a[0] + t * dr)) ** 2 + (c - (a[1] + t * dc)) ** 2
            out |= dist2 <= half * half
        return out

    def mask(self, canvas: Canvas,
             indices: Optional[Iterable[int]] = None) -> np.ndarray:
        """Rasterise the enclosed shapes onto ``canvas``; ``(H, W)`` bool.

        The conversion Measure needs: hand it
        ``Canvas.for_grid(mask_layer.spacing, mask_layer.shape)`` and the ROI
        comes back on that layer's own grid, whatever grid it was drawn on.

        Boundaries are HALF-OPEN: a pixel centre exactly on the low edge of a
        rectangle is inside it and one exactly on the high edge is not. That is
        the even-odd rule's own convention, and it is the one that matters —
        two ROIs sharing an edge partition the pixels between them instead of
        both claiming the seam, so an object on the boundary is counted once.

        Open shapes (lines, paths) enclose nothing and contribute nothing.
        """
        rows = canvas.row_world()
        cols = canvas.column_world()
        out = np.zeros(canvas.shape, dtype=bool)
        chosen = (self._shapes if indices is None
                  else [self._shapes[int(i)] for i in indices])
        for shape in chosen:
            if not shape.is_closed:
                continue
            vertices = self._plane_vertices(shape, canvas)
            if vertices is None:
                continue
            if shape.kind == "ellipse":
                out |= self._inside_ellipse(rows, cols, vertices)
            else:
                out |= self._inside_polygon(rows, cols, vertices)
        return out

    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        rgb, coverage = self._blank(canvas)
        rows = canvas.row_world()
        cols = canvas.column_world()
        for shape in self._shapes:
            vertices = self._plane_vertices(shape, canvas)
            if vertices is None:
                continue
            if shape.is_closed:
                if shape.kind == "ellipse":
                    inside = self._inside_ellipse(rows, cols, vertices)
                else:
                    inside = self._inside_polygon(rows, cols, vertices)
                if shape.face_color[3] > 0:
                    rgb[inside] = np.asarray(shape.face_color[:3],
                                             dtype=np.float32)
                    coverage[inside] = np.maximum(coverage[inside],
                                                  shape.face_color[3])
            if shape.edge_width > 0 and shape.edge_color[3] > 0:
                edge = self._on_edge(rows, cols, vertices, shape.edge_width,
                                     shape.is_closed)
                rgb[edge] = np.asarray(shape.edge_color[:3], dtype=np.float32)
                coverage[edge] = np.maximum(coverage[edge],
                                            shape.edge_color[3])
        return rgb, coverage


# ---------------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------------

LayerLike = Union[Layer, str, int]
Listener = Callable[[LayerEvent], None]


class LayerStack:
    """An ordered list of layers sharing one world.

    ``stack[0]`` is the bottom and is drawn first. Everything a viewer does to
    the list — add, remove, reorder, rename, select — happens here and is
    announced through :meth:`subscribe`, so the widget is a view over this
    rather than the other way round.

    The stack owns two invariants the layers cannot enforce alone:

    * **Names are unique.** A duplicate name means "hide the mask" hides the
      wrong one. Colliding names are suffixed rather than refused, because a
      user adding a second mask should get one, not an error dialog.
    * **Units agree.** A layer measured in µm and a layer measured in pixels
      cannot be composited: the numbers would line up and the picture would
      not. Mixing them raises rather than drawing something plausible.
    """

    def __init__(self, layers: Optional[Iterable[Layer]] = None, *,
                 units: Optional[str] = None):
        self._layers: List[Layer] = []
        self._units = None if units is None else str(units)
        self._selected: Optional[Layer] = None
        self._listeners: List[Listener] = []
        for layer in layers or ():
            self.append(layer)

    # -- sequence -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, Layer):
            return any(l is item for l in self._layers)
        return any(l.name == str(item) for l in self._layers)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return self._layers[key]
        if isinstance(key, str):
            for layer in self._layers:
                if layer.name == key:
                    return layer
            raise KeyError(
                f"no layer named {key!r}; have {list(self.names)}")
        return self._layers[int(key)]

    @property
    def names(self) -> Tuple[str, ...]:
        """Every layer's name, bottom first."""
        return tuple(l.name for l in self._layers)

    def index(self, layer: LayerLike) -> int:
        """Where ``layer`` sits, by object, name or index.

        :raises LayerError: if it is not in this stack. ``-1`` would be a
            valid index from the top and would silently affect the wrong layer.
        """
        if isinstance(layer, Layer):
            for i, other in enumerate(self._layers):
                if other is layer:
                    return i
            raise LayerError(f"layer {layer.name!r} is not in this stack")
        if isinstance(layer, str):
            for i, other in enumerate(self._layers):
                if other.name == layer:
                    return i
            raise LayerError(
                f"no layer named {layer!r}; have {list(self.names)}")
        i = int(layer)
        if i < 0:
            i += len(self._layers)
        if not 0 <= i < len(self._layers):
            raise LayerError(
                f"no layer at index {layer} in a stack of {len(self._layers)}")
        return i

    def get(self, layer: LayerLike) -> Layer:
        """The layer named by an object, a name or an index."""
        return self._layers[self.index(layer)]

    # -- units ----------------------------------------------------------
    @property
    def units(self) -> str:
        """The world unit every layer in this stack is measured in."""
        return self._units or "px"

    def _check_units(self, layer: Layer) -> None:
        units = layer.spacing.units
        if self._units is None:
            self._units = units
            return
        if units != self._units:
            raise LayerError(
                f"layer {layer.name!r} is measured in {units!r} but this stack "
                f"is in {self._units!r}. Compositing them would put a "
                f"{units}-sized object where a {self._units}-sized one "
                f"belongs — convert the spacing rather than mixing the units.")

    # -- mutation -------------------------------------------------------
    def _unique_name(self, name: str, exclude: Optional[Layer] = None) -> str:
        taken = {l.name for l in self._layers if l is not exclude}
        if name not in taken:
            return name
        i = 1
        while f"{name} [{i}]" in taken:
            i += 1
        return f"{name} [{i}]"

    def append(self, layer: Layer) -> Layer:
        """Put ``layer`` on top."""
        return self.insert(len(self._layers), layer)

    def insert(self, index: int, layer: Layer) -> Layer:
        """Put ``layer`` at ``index`` (0 is the bottom)."""
        if not isinstance(layer, Layer):
            raise LayerError(f"expected a Layer, got {layer!r}")
        if any(l is layer for l in self._layers):
            raise LayerError(
                f"layer {layer.name!r} is already in this stack")
        self._check_units(layer)
        layer._name = self._unique_name(layer.name)
        layer._stack = self
        i = max(0, min(int(index), len(self._layers)))
        self._layers.insert(i, layer)
        if self._selected is None:
            self._selected = layer
        self._emit(LayerEvent("inserted", layer, i))
        return layer

    def remove(self, layer: LayerLike) -> Layer:
        """Take a layer out and return it."""
        i = self.index(layer)
        removed = self._layers.pop(i)
        removed._stack = None
        if self._selected is removed:
            self._selected = (self._layers[min(i, len(self._layers) - 1)]
                              if self._layers else None)
        self._emit(LayerEvent("removed", removed, i))
        return removed

    def clear(self) -> None:
        """Remove every layer, top first."""
        while self._layers:
            self.remove(len(self._layers) - 1)

    def move(self, source: LayerLike, destination: int) -> int:
        """Move a layer to ``destination``; returns where it ended up.

        The z-order control. ``destination`` is clamped, so "move to the top"
        can be spelled with any large number.
        """
        i = self.index(source)
        layer = self._layers.pop(i)
        j = max(0, min(int(destination), len(self._layers)))
        self._layers.insert(j, layer)
        if i != j:
            self._emit(LayerEvent("moved", layer, j, detail=str(i)))
        return j

    def raise_layer(self, layer: LayerLike) -> int:
        """One step towards the front."""
        i = self.index(layer)
        return self.move(i, min(i + 1, len(self._layers) - 1))

    def lower_layer(self, layer: LayerLike) -> int:
        """One step towards the back."""
        i = self.index(layer)
        return self.move(i, max(i - 1, 0))

    def to_top(self, layer: LayerLike) -> int:
        return self.move(layer, len(self._layers) - 1)

    def to_bottom(self, layer: LayerLike) -> int:
        return self.move(layer, 0)

    def rename(self, layer: LayerLike, name: str) -> str:
        """Rename a layer, uniquifying if needed; returns the name it got."""
        target = self.get(layer)
        wanted = Layer._check_name(name)
        final = self._unique_name(wanted, exclude=target)
        if final == target.name:
            return final
        target._name = final
        self._emit(LayerEvent("renamed", target, self.index(target),
                              detail=final))
        return final

    # -- convenience constructors ---------------------------------------
    def add_image(self, data: Any, **kwargs: Any) -> ImageLayer:
        return self.append(ImageLayer(data, **kwargs))  # type: ignore[return-value]

    def add_labels(self, data: Any, **kwargs: Any) -> LabelsLayer:
        return self.append(LabelsLayer(data, **kwargs))  # type: ignore[return-value]

    def add_points(self, data: Any = None, **kwargs: Any) -> PointsLayer:
        return self.append(PointsLayer(data, **kwargs))  # type: ignore[return-value]

    def add_shapes(self, shapes: Optional[Iterable[Shape]] = None,
                   **kwargs: Any) -> ShapesLayer:
        return self.append(ShapesLayer(shapes, **kwargs))  # type: ignore[return-value]

    # -- selection ------------------------------------------------------
    @property
    def selected(self) -> Optional[Layer]:
        """The layer the layer-list has highlighted — what edits apply to."""
        return self._selected

    @property
    def selected_index(self) -> int:
        """Where the selected layer sits, or ``-1`` when nothing is selected."""
        return -1 if self._selected is None else self.index(self._selected)

    def select(self, layer: Optional[LayerLike]) -> Optional[Layer]:
        """Select a layer (or ``None``); returns it."""
        target = None if layer is None else self.get(layer)
        if target is not self._selected:
            self._selected = target
            self._emit(LayerEvent("selected", target,
                                  -1 if target is None else self.index(target)))
        return target

    # -- world ----------------------------------------------------------
    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The union of every layer's world box, keyed by axis.

        Empty layers (a points layer with no points yet) do not drag the
        extent to the origin — an empty layer occupies nothing, and letting it
        vote would zoom the view out to include a point nobody has placed.
        """
        out: Dict[str, Tuple[float, float]] = {}
        for layer in self._layers:
            if isinstance(layer, PointsLayer) and len(layer.data) == 0:
                continue
            if isinstance(layer, ShapesLayer) and len(layer) == 0:
                continue
            for axis, (lo, hi) in layer.world_extent().items():
                if axis in out:
                    out[axis] = (min(out[axis][0], lo), max(out[axis][1], hi))
                else:
                    out[axis] = (lo, hi)
        return out

    def canvas(self, **kwargs: Any) -> Canvas:
        """A :class:`Canvas` showing all of this stack — see
        :meth:`Canvas.covering`."""
        return Canvas.covering(self, **kwargs)

    # -- rendering ------------------------------------------------------
    def render(self, canvas: Canvas) -> np.ndarray:
        """Composite every visible layer; ``(H, W, 3)`` float32 in 0–1."""
        return self.render_rgba(canvas)[..., :3]

    def render_rgba(self, canvas: Canvas) -> np.ndarray:
        """:meth:`render` plus the composite's own coverage, ``(H, W, 4)``.

        The alpha channel is accumulated source-over whatever the layers'
        blending modes were: it records where the stack drew *anything*, which
        is what a caller compositing this onto a page background needs.
        """
        h, w = canvas.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        alpha = np.zeros((h, w), dtype=np.float32)
        for layer in self._layers:
            if not layer.visible or layer.opacity <= 0.0:
                continue
            src, coverage = layer.render(canvas)
            rgb, used = Blending.apply(rgb, src, coverage, layer.opacity,
                                       layer.blending)
            alpha = alpha + used * (1.0 - alpha)
        return np.concatenate([rgb, alpha[..., None]], axis=-1)

    def render_uint8(self, canvas: Canvas) -> np.ndarray:
        """:meth:`render` as ``(H, W, 3)`` uint8 — what a QImage wants."""
        return np.clip(self.render(canvas) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # -- picking --------------------------------------------------------
    def pick(self, canvas: Canvas, row: float, column: float
             ) -> Tuple[Optional[Layer], Dict[str, float], Any]:
        """What is under a canvas pixel: ``(layer, world, value)``.

        Walks from the TOP down and stops at the first visible layer with
        something there, which is what a click means — the thing you can see.
        ``value`` is the label for a labels layer, the point index for a points
        layer, the shape index for a shapes layer, and ``None`` otherwise.
        """
        world = canvas.world_at(row, column)
        for layer in reversed(self._layers):
            if not layer.visible or layer.opacity <= 0.0:
                continue
            if isinstance(layer, LabelsLayer):
                label = layer.label_at_world(world)
                if label:
                    return layer, world, label
            elif isinstance(layer, PointsLayer):
                found = layer.nearest(world)
                if found is not None:
                    return layer, world, found
            elif isinstance(layer, ShapesLayer):
                for i in range(len(layer) - 1, -1, -1):
                    if layer.mask(Canvas(origin=(world[canvas.axes[0]],
                                                 world[canvas.axes[1]]),
                                         step=canvas.step, shape=(1, 1),
                                         axes=canvas.axes, depth=canvas.depth,
                                         units=canvas.units), [i])[0, 0]:
                        return layer, world, i
        return None, world, None

    # -- events ---------------------------------------------------------
    def subscribe(self, listener: Listener) -> Listener:
        """Be told when anything changes. Returns ``listener``, for unsubscribing.

        Listeners are held by strong reference and are NOT weak: a view that
        subscribes must :meth:`unsubscribe` when it closes, exactly as
        :class:`spacr.qt.linked_selection.LinkedView` must unlink.
        """
        if not callable(listener):
            raise LayerError(f"a listener must be callable, got {listener!r}")
        if listener not in self._listeners:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: Listener) -> bool:
        """Stop being told. ``True`` if this call is what removed it."""
        try:
            self._listeners.remove(listener)
        except ValueError:
            return False
        return True

    def _emit(self, event: LayerEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def describe(self) -> str:
        """One line per layer, top first — what the layer list shows."""
        if not self._layers:
            return "no layers"
        return "\n".join(l.describe() for l in reversed(self._layers))
