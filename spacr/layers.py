"""A napari-style layer model — images, labels, points and shapes in one world.

A viewer that draws overlays into the image pixels — outlines in the live
preview, tracks in the timelapse preview, brush strokes in Make Masks — returns
one RGB array, in which an overlay cannot be hidden, faded, recoloured or moved
below another object.

This module is the model for a viewer in which each of these is a separate layer.
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
-------------------------------------

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

Ignoring spacing produces no visible error: an overlay drawn a slice out of
register still looks like a plausible image, and any number measured from it is
wrong without a warning. For this reason a zero scale raises, and a stack raises
:exc:`LayerError` when µm layers and px layers are mixed instead of drawing them
on top of each other.

Order
-----

``stack[0]`` is the BOTTOM layer and is drawn first, like a stack of acetates
and like napari's own layer list. Moving a layer up moves it towards the front.

Blending
--------

Every layer contributes ``coverage`` (where it draws, 0–1) and ``opacity`` (a
0–1 weight for the whole layer). All five modes combine them as ``alpha =
coverage * opacity``, except :data:`Blending.OPAQUE`, which first sets every
non-zero coverage to 1 so that soft edges cover fully; this is the only mode
that treats the two differently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, cast,
                    Optional, Sequence, Tuple, Union)

import numpy as np


__all__ = [
    "LayerError",
    "Blending",
    "Spacing",
    "Canvas",
    "OrthoViews",
    "CanvasLink",
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
    "VOLUME_RENDERINGS",
    "LABEL_RENDERINGS",
    "VolumeRays",
    "volume_steps",
    "downsample_volume",
    "render_volume",
    "SurfaceFinish",
    "SURFACE_FINISHES",
    "VOLUME_FINISHES",
    "surface_finish",
    "SurfaceLight",
    "surface_normals",
    "shade_surface",
    "VolumeLens",
    "CUBE_EDGES",
    "RenderScale",
    "render_scale",
    "GLVolumeRenderer",
    "WGPU_BACKENDS",
    "GPU_RENDER_OFF_ENV",
    "gpu_render_refusal",
    "wgpu_adapters",
    "wgpu_adapter",
    "WGPUVolumeRenderer",
]


class LayerError(ValueError):
    """Raised for an invalid layer, spacing, canvas, stack or colour argument.

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

    :param colour: named, hexadecimal or three-/four-component colour value
        to normalise; numeric components may use either the 0–1 or 0–255
        scale.
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

    :param name: Display and lookup name stored with the colour ramp; it is
        coerced to a string.
    :param colors: Ordered sequence of at least two colour specifications
        accepted by :func:`to_rgba`; the ramp interpolates them in order.
    :param stops: where each colour sits, in 0..1. Defaults to spacing them
        evenly. One stop per colour, ascending -- an out-of-order stop would
        make the ramp non-monotonic and is refused rather than sorted, since
        sorting would silently give a different ramp than the caller wrote.
    :raises LayerError: with fewer than two colours, or when ``stops`` does
        not match ``colors`` one for one.
    """

    __slots__ = ("name", "_colors", "_stops")

    def __init__(self, name: str, colors: Sequence[Any],
                 stops: Optional[Sequence[float]] = None):
        """Build a colormap from at least two colours.

        Fewer than two is refused: one colour is not a map, and interpolating
        between a colour and nothing has no answer.
        """
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
        """Map ``values`` (any shape, 0–1, clipped) to ``(..., 3)`` float32.

        :param values: Numeric scalar or array-like values to interpolate;
            values outside 0–1 are clipped to the nearest end of the ramp.
        """
        t = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        out = np.empty(t.shape + (3,), dtype=np.float32)
        for c in range(3):
            out[..., c] = np.interp(t, self._stops, self._colors[:, c])
        return out

    def __repr__(self) -> str:
        """The map's name and how many stops it has."""
        return f"Colormap({self.name!r}, {len(self._colors)} stops)"

    def __eq__(self, other: Any) -> bool:
        """Equal when the name AND the colours and stops match.

        The name alone is not enough -- two maps can be renamed to the same thing
        -- and the arrays alone are not either, because the name is what a saved
        setting refers to.
        """
        return (isinstance(other, Colormap) and other.name == self.name
                and np.array_equal(other._colors, self._colors)
                and np.array_equal(other._stops, self._stops))

    def __hash__(self) -> int:
        """Hash the name and the colours, so equal maps hash alike."""
        return hash((self.name, self._colors.tobytes()))


def _ramp(name: str, colour: Any) -> Colormap:
    """Build a black-to-colour ramp.

    :param name: the colormap's name.
    :param colour: the colour the ramp ends on.
    :returns: the colormap -- black at zero, so an unlabelled pixel is
        background rather than a dark shade of the channel.
    """
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

    :param spec: Existing :class:`Colormap`, built-in or matplotlib colormap
        name, or a colour specification from which to build a black-to-colour
        ramp.
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
        try:
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

    :param label: Integer-like object label. Label 0 is reserved for the black
        background colour.
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

    :param labels: Array-like integer labels; the output preserves their shape
        and appends one RGB axis.
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



class Blending:
    """How a layer combines with what is already on the canvas.

    Five modes, all of which honour ``opacity`` the same way — ``alpha =
    coverage * opacity`` — except :data:`OPAQUE`, which first sets every
    non-zero coverage to 1, so a soft-edged layer fully covers the layers
    below it wherever it has any coverage. Opacity still
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

        :param mode: Blending-mode name. Leading and trailing whitespace and
            letter case are normalised before validation against
            :attr:`Blending.ALL`.
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
        else:
            out = dst * (1.0 - a) + src * a
        return out.astype(np.float32), alpha



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
        """Coerce and validate the voxel size, then freeze it.

        A zero or non-finite scale raises :exc:`LayerError` rather than being
        clamped. Such a scale collapses an axis -- every world coordinate on it
        resolves to element 0 -- and the overlay is then drawn a slice out of
        register with no visible sign of the error.
        """
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

        :param sizes: Mapping from axis name to non-zero finite voxel size;
            mapping iteration order becomes the array-axis order.
        """
        names = tuple(str(k) for k in sizes)
        scale = tuple(float(v) for v in sizes.values())
        origin = origin or {}
        translate = tuple(float(origin.get(n, 0.0)) for n in names)
        return cls(scale=scale, translate=translate, axes=names, units=units)

    @property
    def ndim(self) -> int:
        """How many world axes this spacing describes.

        :returns: the axis count.
        """
        return len(self.scale)

    def axis_index(self, axis: str) -> int:
        """Position of ``axis`` in this spacing.

        :param axis: Axis name to locate in :attr:`axes`.
        :raises LayerError: if this spacing has no such axis.
        """
        try:
            return self.axes.index(str(axis))
        except ValueError:
            raise LayerError(
                f"no axis {axis!r} in {self.axes}") from None

    def has_axis(self, axis: str) -> bool:
        """Whether this spacing contains ``axis``.

        :param axis: Axis name to test after coercion to a string.
        """
        return str(axis) in self.axes

    def to_world(self, index: Sequence[float]) -> Tuple[float, ...]:
        """Data index → world coordinate, per axis.

        :param index: Fractional data coordinate in :attr:`axes` order, with
            one entry per spacing axis.
        """
        idx = tuple(float(v) for v in index)
        if len(idx) != self.ndim:
            raise LayerError(
                f"index {idx} has {len(idx)} axes, spacing has {self.ndim}")
        return tuple(t + s * i
                     for t, s, i in zip(self.translate, self.scale, idx))

    def to_data(self, world: Sequence[float]) -> Tuple[float, ...]:
        """World coordinate → (fractional) data index, per axis.

        :param world: World coordinate in :attr:`axes` order, with one entry
            per spacing axis.
        """
        pos = tuple(float(v) for v in world)
        if len(pos) != self.ndim:
            raise LayerError(
                f"world point {pos} has {len(pos)} axes, spacing has "
                f"{self.ndim}")
        return tuple((w - t) / s
                     for w, t, s in zip(pos, self.translate, self.scale))

    def world_map(self, index: Sequence[float]) -> Dict[str, float]:
        """:meth:`to_world` keyed by axis name.

        :param index: Fractional data coordinate in :attr:`axes` order.
        """
        return dict(zip(self.axes, self.to_world(index)))

    def data_from_map(self, world: Mapping[str, float]) -> Tuple[float, ...]:
        """Fractional data index from a ``{axis: world}`` mapping.

        Axes the mapping does not mention are taken to be 0 in world units,
        which is what a 2-D click on a 3-D layer means once the viewer has
        supplied the slice it is showing.

        :param world: Mapping from axis names to world coordinates. Unnamed
            spacing axes are evaluated at world coordinate zero.
        """
        return self.to_data([float(world.get(a, 0.0)) for a in self.axes])

    def extent(self, shape: Sequence[int]) -> Dict[str, Tuple[float, float]]:
        """World bounding box of an array of ``shape``, keyed by axis.

        Measured to the OUTER EDGE of the end elements (half a voxel beyond
        the first and last centres), because that is the region the data
        actually covers — a canvas fitted to element centres clips half a
        voxel off every side, which at 2 µm z-steps is a visible slab.

        :param shape: Array shape in :attr:`axes` order, with one size per
            spacing axis.
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



def _extent_of(source: Any) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """``({axis: (low, high)}, units)`` for a stack, a layer or an extent map."""
    if isinstance(source, LayerStack):
        return source.world_extent(), source.units
    if isinstance(source, Layer):
        return source.world_extent(), source.spacing.units
    return ({str(k): (float(v[0]), float(v[1]))
             for k, v in dict(source).items()}, "px")


def _voxel_steps(source: Any) -> Dict[str, float]:
    """World size of one element along each axis, over everything with a grid.

    The finest voxel wins where two layers disagree, because that is the
    smallest step a slice slider could usefully take: a slider that stepped by
    the coarse layer's z would skip slices of the fine one.
    """
    steps: Dict[str, float] = {}
    layers = (list(source) if isinstance(source, LayerStack)
              else [source] if isinstance(source, Layer) else [])
    for layer in layers:
        if not layer.shape:
            continue
        for axis, size in zip(layer.spacing.axes, layer.spacing.scale):
            size = abs(float(size))
            if axis not in steps or size < steps[axis]:
                steps[axis] = size
    return steps


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
        """Coerce and validate the canvas geometry, then freeze it.

        Two dimensions exactly, two distinct axes, a positive shape and a non-zero
        finite step. Each is refused rather than repaired: a canvas whose rows and
        columns name the same axis has no meaning to fall back to.
        """
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

    @classmethod
    def for_grid(cls, spacing: "Spacing", shape: Sequence[int], *,
                 axes: Tuple[str, str] = ("y", "x"),
                 depth: Optional[Mapping[str, float]] = None) -> "Canvas":
        """The canvas that samples ``spacing``'s own grid, one pixel per element.

        The identity render: a layer drawn onto ``Canvas.for_grid(layer.spacing,
        layer.shape)`` comes back at its native resolution and alignment, which
        is what a shapes-to-mask conversion needs (see
        :meth:`ShapesLayer.mask`).

        :param spacing: the grid to sample. Origin, step and :attr:`units` are
            all taken from it, so the canvas reports the layer's world units
            instead of the ``"px"`` a bare :class:`Canvas` defaults to. Pixel
            ``(0, 0)`` is centred exactly on element 0 — no half-step inset,
            unlike :meth:`covering`.
        :param shape: the layer's whole array shape, in the spacing's axis
            order. It is indexed by each plane axis's position in
            ``spacing.axes``, not by 0 and 1, so a three-axis spacing wants
            three entries even though only two are read; a two-entry shape
            against a 3-D spacing raises :exc:`IndexError`. Entries for axes
            outside the plane are ignored, non-integers are truncated, and a
            zero or negative in-plane size raises :exc:`LayerError`.
        :param axes: which two spacing axes rows and columns run along. Any
            pair the spacing names is allowed, in any order — ``("x", "y")``
            transposes the result. An axis the spacing does not have, or the
            same axis twice, raises :exc:`LayerError`.
        :param depth: world coordinate for the axes outside the plane.
            When omitted, each out-of-plane axis is pinned at the spacing's
            own origin, so a translated volume starts on its first plane.
            Passing a mapping selects those world coordinates explicitly;
            entries naming an in-plane axis are kept but never consulted.
        """
        r = spacing.axis_index(axes[0])
        c = spacing.axis_index(axes[1])
        if depth is None:
            depth = {
                axis: spacing.translate[index]
                for index, axis in enumerate(spacing.axes)
                if axis not in axes
            }
        return cls(origin=(spacing.translate[r], spacing.translate[c]),
                   step=(spacing.scale[r], spacing.scale[c]),
                   shape=(int(shape[r]), int(shape[c])),
                   axes=axes, depth=depth, units=spacing.units)

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
        extent, units = _extent_of(source)
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
        origin = (origins[0] + 0.5 * step[0], origins[1] + 0.5 * step[1])
        return cls(origin=origin, step=step, shape=(height, width), axes=axes,
                   depth=depth or {}, units=units)

    @property
    def height(self) -> int:
        """The window's height in pixels: rows run along ``axes[0]``.

        :returns: the row count.
        """
        return self.shape[0]

    @property
    def width(self) -> int:
        """The window's width in pixels: columns run along ``axes[1]``.

        :returns: the column count.
        """
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

        :param row: Fractional canvas-row coordinate.
        :param column: Fractional canvas-column coordinate.
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

        :param world: Mapping containing world coordinates for both axes of
            this canvas plane.
        """
        return ((float(world[self.axes[0]]) - self.origin[0]) / self.step[0],
                (float(world[self.axes[1]]) - self.origin[1]) / self.step[1])

    def zoomed(self, factor: float,
               centre: Optional[Tuple[float, float]] = None) -> "Canvas":
        """A canvas ``factor``\\ × closer, about ``centre`` (canvas pixels).

        ``factor > 1`` zooms in. The default centre is the middle of the view,
        which is what a keyboard zoom means; a wheel zoom passes the cursor.

        :param factor: Positive finite magnification factor; values above one
            zoom in and values below one zoom out.
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
        """A canvas moved by ``(d_row, d_column)`` canvas pixels.

        :param d_row: Signed displacement in canvas-row pixels.
        :param d_column: Signed displacement in canvas-column pixels.
        """
        return replace(self, origin=(
            self.origin[0] + self.step[0] * float(d_row),
            self.origin[1] + self.step[1] * float(d_column)))

    def resized(self, height: int, width: int) -> "Canvas":
        """The same world window at a different pixel size.

        The world *span* is held, not the step, so a widget resize shows the
        same field of view rather than more of the sample.

        :param height: New canvas height in pixels, coerced to at least one.
        :param width: New canvas width in pixels, coerced to at least one.
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



@dataclass(frozen=True)
class OrthoViews:
    """The three planes of one volume, laid out so that they line up.

    A z-stack seen only from above hides the thing a z-stack is acquired for.
    The three canvases here are the top view and the two side views through a
    crosshair, arranged the way ImageJ's orthogonal views are:

    * :attr:`xy` — the top view, rows ``y`` and columns ``x``, at depth ``z``;
    * :attr:`zx` — BELOW it, rows ``z`` and columns ``x``, so it shares its
      columns with the top view;
    * :attr:`yz` — BESIDE it, rows ``y`` and columns ``z``, so it shares its
      rows with the top view.

    **Every panel uses one world-units-per-pixel scale.** That is the whole
    difficulty, and it is why this is a class rather than three
    :meth:`Canvas.covering` calls. Confocal stacks in this codebase are
    routinely 0.65 µm in xy and 2 µm in z; a side view drawn one pixel per
    slice is three times too thin, and it does not look wrong — it looks like
    a slightly flat cell. Nuclear sphericity, colocalisation depth and every
    3-D shape feature read off such a picture are wrong by a factor nobody
    sees. Here the side views are :math:`z_{span}/step` pixels tall, so a
    10-slice 2 µm stack is 31 px of a 0.65 µm/px view, not 10.

    Build one with :meth:`covering` and move the crosshair with :meth:`at`.

    :param xy: Top-view canvas whose rows and columns span the second and
        third volume axes.
    :param zx: Lower side-view canvas whose rows span depth and whose columns
        align with :attr:`xy`.
    :param yz: Side-view canvas whose rows align with :attr:`xy` and whose
        columns span depth.
    :param point: Mapping from world-axis names to the shared crosshair
        coordinates.
    """

    xy: Canvas
    zx: Canvas
    yz: Canvas
    point: Mapping[str, float]
    steps: Mapping[str, float] = field(default_factory=dict)
    extent: Mapping[str, Tuple[float, float]] = field(default_factory=dict)

    #: The world axes, depth first: ``(z, y, x)``.
    DEFAULT_AXES: ClassVar[Tuple[str, str, str]] = ("z", "y", "x")

    def __post_init__(self) -> None:
        """Coerce and validate the axis names, then freeze them."""
        object.__setattr__(self, "point", MappingProxyType(
            {str(k): float(v) for k, v in dict(self.point).items()}))
        object.__setattr__(self, "steps", MappingProxyType(
            {str(k): float(v) for k, v in dict(self.steps).items()}))
        object.__setattr__(self, "extent", MappingProxyType(
            {str(k): (float(v[0]), float(v[1]))
             for k, v in dict(self.extent).items()}))

    @classmethod
    def covering(cls, source: Any, *, width: int = 512,
                 point: Optional[Mapping[str, float]] = None,
                 axes: Tuple[str, str, str] = DEFAULT_AXES,
                 margin: float = 0.0) -> "OrthoViews":
        """Three canvases showing all of ``source``, crossing at ``point``.

        :param source: a :class:`LayerStack`, a :class:`Layer` or a
            ``{axis: (low, high)}`` extent mapping.
        :param width: the top view's width in pixels. Every other panel size
            follows from it and from the world extents, so that the layout
            lines up and nothing is squashed.
        :param point: where the crosshair starts, ``{axis: world}``. Defaults
            to the middle of the volume, which is the slice a user opening a
            stack wants to see first — slice 0 of a confocal stack is usually
            empty.
        :param axes: the three world axes, depth first.
        :param margin: fraction of the extent to add on every side.
        :raises LayerError: if the source does not span all three axes; a 2-D
            source raises rather than being drawn as a single plane.
        """
        extent, units = _extent_of(source)
        depth_axis, row_axis, column_axis = (str(a) for a in axes)
        missing = [a for a in (depth_axis, row_axis, column_axis)
                   if a not in extent]
        if missing:
            raise LayerError(
                f"cannot build orthogonal views on axes {tuple(axes)}: nothing "
                f"in the source spans {missing} (it spans {sorted(extent)}). "
                f"A 2-D field has no second plane to show.")
        spans: Dict[str, float] = {}
        lows: Dict[str, float] = {}
        for axis in (depth_axis, row_axis, column_axis):
            lo, hi = extent[axis]
            pad = (hi - lo) * float(margin)
            lo, hi = lo - pad, hi + pad
            lows[axis] = lo
            spans[axis] = max(hi - lo, _EPS)
        width = max(1, int(width))
        scale = spans[column_axis] / width
        rows = max(1, int(round(spans[row_axis] / scale)))
        depth_pixels = max(1, int(round(spans[depth_axis] / scale)))

        centre = {axis: lows[axis] + 0.5 * spans[axis]
                  for axis in (depth_axis, row_axis, column_axis)}
        centre.update({str(k): float(v) for k, v in dict(point or {}).items()})

        def _canvas(pair, shape):
            """Return a shared-scale ``Canvas`` for ``pair`` and ``shape``."""
            origin = tuple(lows[a] + 0.5 * scale for a in pair)
            pinned = {a: v for a, v in centre.items() if a not in pair}
            return Canvas(origin=origin, step=(scale, scale), shape=shape,
                          axes=pair, depth=pinned, units=units)

        steps = _voxel_steps(source)
        return cls(xy=_canvas((row_axis, column_axis), (rows, width)),
                   zx=_canvas((depth_axis, column_axis),
                              (depth_pixels, width)),
                   yz=_canvas((row_axis, depth_axis), (rows, depth_pixels)),
                   point=centre, steps=steps,
                   extent={a: extent[a] for a in
                           (depth_axis, row_axis, column_axis)})

    @property
    def axes(self) -> Tuple[str, str, str]:
        """The three world axes, depth first."""
        return (self.zx.axes[0], self.xy.axes[0], self.xy.axes[1])

    @property
    def scale(self) -> float:
        """World units per pixel — the same in every panel, by construction."""
        return self.xy.step[1]

    def canvases(self) -> Dict[str, Canvas]:
        """``{'xy': …, 'zx': …, 'yz': …}`` — what a three-panel view draws."""
        return {"xy": self.xy, "zx": self.zx, "yz": self.yz}

    def slider(self, axis: str) -> Tuple[float, float, float]:
        """``(low, high, step)`` in WORLD units for a slice slider.

        The step is the source's own voxel size along that axis, so dragging
        the slider moves one slice at a time rather than an arbitrary
        fraction of the volume — and a µm-calibrated stack's slider is
        labelled in µm, which is the number a user can check against the
        acquisition settings.

        :param axis: World axis for which to return bounds and voxel step.
        """
        axis = str(axis)
        if axis not in self.extent:
            raise LayerError(
                f"no axis {axis!r} in these views; they span "
                f"{sorted(self.extent)}")
        low, high = self.extent[axis]
        return low, high, self.steps.get(axis, self.scale)

    def n_slices(self, axis: str) -> int:
        """How many slices the slider on ``axis`` has.

        :param axis: World axis whose extent and voxel step define the count.
        """
        low, high, step = self.slider(axis)
        return max(1, int(round((high - low) / max(step, _EPS))))

    def at(self, **coords: float) -> "OrthoViews":
        """The same views with the crosshair moved: ``views.at(z=12.0)``.

        Each panel is re-pinned at the axes it does not span, so moving z
        changes what the top view shows and nothing else — which is what makes
        a z slider one call rather than three.
        """
        point = dict(self.point)
        point.update({str(k): float(v) for k, v in coords.items()})
        moved = {}
        for name, canvas in self.canvases().items():
            pinned = {a: v for a, v in point.items() if a not in canvas.axes}
            moved[name] = replace(canvas, depth=pinned)
        return replace(self, point=point, **moved)

    def clamped(self, **coords: float) -> "OrthoViews":
        """:meth:`at`, with each coordinate held inside the volume.

        What a slider wants: dragging past the last slice shows the last
        slice, not an empty canvas that looks like a failed load.
        """
        held = {}
        for axis, value in coords.items():
            low, high = self.extent.get(str(axis), (float(value),
                                                    float(value)))
            held[axis] = min(max(float(value), low), high)
        return self.at(**held)

    def at_pixel(self, panel: str, row: float, column: float) -> "OrthoViews":
        """Move the crosshair to the world point under a pixel of one panel.

        Clicking the side view to move the top view's slice, which is the
        interaction an orthogonal view is for.

        :param panel: Panel key: ``"xy"``, ``"zx"`` or ``"yz"``.
        :param row: Fractional row coordinate within the selected panel.
        :param column: Fractional column coordinate within the selected panel.
        """
        canvas = self.canvases().get(str(panel))
        if canvas is None:
            raise LayerError(
                f"no panel {panel!r}; the panels are 'xy', 'zx' and 'yz'")
        world = canvas.world_at(row, column)
        return self.clamped(**{a: world[a] for a in canvas.axes})

    def zoomed(self, factor: float) -> "OrthoViews":
        """Every panel ``factor`` times closer, about the crosshair.

        All three together, and about the crosshair rather than about each
        panel's own middle, so the panels still line up afterwards.

        :param factor: Positive finite magnification factor applied to every
            panel; values above one zoom in.
        """
        factor = float(factor)
        if factor <= 0 or not math.isfinite(factor):
            raise LayerError(f"zoom factor must be positive, got {factor}")
        moved = {}
        for name, canvas in self.canvases().items():
            centre = canvas.pixel_at(self.point)
            moved[name] = canvas.zoomed(factor, centre=centre)
        return replace(self, **moved)

    def resized(self, width: int) -> "OrthoViews":
        """The same crosshair at a different pixel width, panels still aligned.

        :param width: New top-panel width in pixels; derived panel dimensions
            remain aligned to the same world scale.
        """
        return replace(
            type(self).covering(self.extent, width=width, point=self.point,
                                axes=self.axes),
            steps=self.steps)

    def render(self, stack: "LayerStack") -> Dict[str, np.ndarray]:
        """``{panel: (H, W, 3)}`` — the same stack drawn on all three planes.

        :param stack: Layer stack to render through each orthogonal canvas.
        """
        return {name: stack.render(canvas)
                for name, canvas in self.canvases().items()}

    def describe(self) -> str:
        """One line for a status bar: where the crosshair is."""
        body = " · ".join(f"{a} {self.point[a]:.6g}" for a in self.axes
                          if a in self.point)
        return f"{body} ({self.xy.units})"



class CanvasLink:
    """Several canvases held on the same world window.

    The model under a comparison grid: four panels showing four channels of
    one field, or the same field at four timepoints, or the same well under
    four conditions. Panning one pans all of them and zooming one zooms all of
    them, so a difference between two panels reflects a difference in the
    *data* rather than a different field of view.

    What is shared is the world window — the origin and the step — and NOT the
    pixel shape, because the panels are different widgets with different sizes.
    Sharing the shape would make the grid's own layout part of the state; this
    way a panel that is 10 px narrower shows 10 px less of the same view at the
    same magnification, which is what a grid of unequal cells should do.

    :param canvases: the panels to start with, as ``{key: canvas}``. Added
        one at a time through :meth:`add`, so each is adopted onto the shared
        world window as it arrives rather than the first one winning.

    A panel can opt out with :meth:`unlock`, for the ordinary case of wanting
    to look closely at one of them without losing the others' place.
    """

    def __init__(self, canvases: Optional[Mapping[str, Canvas]] = None):
        """Link a set of named canvases so they can move together."""
        self._canvases: Dict[str, Canvas] = {}
        self._locked: Dict[str, bool] = {}
        self._listeners: List[Callable[[str], None]] = []
        for key, canvas in dict(canvases or {}).items():
            self.add(key, canvas)

    def __len__(self) -> int:
        """How many panels this link holds."""
        return len(self._canvases)

    def __contains__(self, key: Any) -> bool:
        """Whether a panel key is in this link."""
        return str(key) in self._canvases

    def __getitem__(self, key: str) -> Canvas:
        """One panel's canvas, by key."""
        try:
            return self._canvases[str(key)]
        except KeyError:
            raise LayerError(
                f"no panel {key!r}; the panels are {list(self._canvases)}"
            ) from None

    @property
    def keys(self) -> Tuple[str, ...]:
        """Every panel's key, in the order they were added."""
        return tuple(self._canvases)

    def canvases(self) -> Dict[str, Canvas]:
        """``{key: canvas}`` — a copy, so a caller cannot mutate the link."""
        return dict(self._canvases)

    def add(self, key: str, canvas: Canvas, *, locked: bool = True) -> Canvas:
        """Put a panel in the link, adopting the shared window if there is one.

        A panel added to a link that has already been panned starts where the
        others are, not where it was built — otherwise adding a fifth channel
        to a grid the user has zoomed into shows the whole field in one cell.

        :param key: Unique panel identifier, coerced to a string.
        :param canvas: Canvas that defines this panel's plane, size and world
            window.
        """
        if not isinstance(canvas, Canvas):
            raise LayerError(f"a linked panel needs a Canvas, got {canvas!r}")
        key = str(key)
        if key in self._canvases:
            raise LayerError(f"panel {key!r} is already in this link")
        self._locked[key] = bool(locked)
        leader = self._leader()
        if locked and leader is not None:
            canvas = self._aligned(canvas, leader)
        self._canvases[key] = canvas
        self._emit(key)
        return canvas

    def remove(self, key: str) -> Canvas:
        """Take a panel out and return its canvas.

        :param key: Identifier of an existing panel.
        """
        canvas = self[key]
        del self._canvases[str(key)]
        self._locked.pop(str(key), None)
        self._emit(str(key))
        return canvas

    def _leader(self) -> Optional[Canvas]:
        """The first FOLLOWING canvas, or ``None`` if none follow.

        The leader is whichever linked panel comes first, not a panel anyone
        nominated: an unlocked panel has opted out of the link, so it cannot be
        what the others follow.
        """
        for key, canvas in self._canvases.items():
            if self._locked.get(key, True):
                return canvas
        return None

    @staticmethod
    def _aligned(canvas: Canvas, leader: Canvas) -> Canvas:
        """``canvas`` on ``leader``'s world window, keeping its own pixel size."""
        return replace(canvas, origin=leader.origin, step=leader.step)

    def is_locked(self, key: str) -> bool:
        """Whether a panel follows the others.

        :param key: Identifier of an existing panel.
        """
        self[key]
        return self._locked.get(str(key), True)

    def lock(self, key: str) -> None:
        """Make a panel follow the others again, moving it there now.

        :param key: Identifier of the panel to align and lock.
        """
        canvas = self[key]
        self._locked[str(key)] = True
        leader = cast(Canvas, self._leader())
        self._canvases[str(key)] = self._aligned(canvas, leader)
        self._emit(str(key))

    def unlock(self, key: str) -> None:
        """Let a panel be moved on its own.

        :param key: Identifier of the panel to detach from linked movement.
        """
        self[key]
        self._locked[str(key)] = False
        self._emit(str(key))

    def set(self, key: str, canvas: Canvas) -> None:
        """Replace one panel's canvas, and bring the locked ones with it.

        :param key: Identifier of the existing panel to replace.
        :param canvas: Replacement canvas whose world window locked peers
            adopt.
        """
        if not isinstance(canvas, Canvas):
            raise LayerError(f"a linked panel needs a Canvas, got {canvas!r}")
        key = str(key)
        self[key]
        self._canvases[key] = canvas
        if self._locked.get(key, True):
            for other, existing in list(self._canvases.items()):
                if other != key and self._locked.get(other, True):
                    self._canvases[other] = self._aligned(existing, canvas)
        self._emit(key)

    def resize(self, key: str, height: int, width: int) -> None:
        """Set the pixel shape of one panel only, after its widget resized.

        The step is held, not the span — the opposite of
        :meth:`Canvas.resized`. In a linked grid the shared quantity is the
        magnification: a cell that is 10 px narrower than its neighbour shows
        10 px less of the sample at the same scale, so panels of different
        sizes stay at the same magnification and remain comparable.

        :param key: Identifier of the existing panel whose widget resized.
        :param height: New panel height in pixels, coerced to at least one.
        :param width: New panel width in pixels, coerced to at least one.
        """
        height, width = max(1, int(height)), max(1, int(width))
        self._canvases[str(key)] = replace(self[key], shape=(height, width))
        self._emit(str(key))

    def zoom(self, factor: float, *, key: Optional[str] = None,
             centre: Optional[Tuple[float, float]] = None) -> None:
        """Zoom ``key`` (or every panel) by ``factor``, about ``centre``.

        The world point under ``centre`` in the panel being driven is what the
        other panels are zoomed about too — not their own middles, or a wheel
        over one cell would slide the rest of the grid sideways.

        :param factor: Positive finite magnification factor passed to each
            affected canvas.
        """
        driver = self._require_key(key)
        canvas = self[driver]
        if centre is None:
            centre = ((canvas.shape[0] - 1) / 2.0, (canvas.shape[1] - 1) / 2.0)
        anchor = canvas.world_at(*centre)
        self._apply(driver, lambda c: c.zoomed(
            factor, centre=c.pixel_at(anchor)))

    def pan(self, d_row: float, d_column: float, *,
            key: Optional[str] = None) -> None:
        """Pan by a number of the DRIVING panel's pixels.

        Converted to world before it is handed to the others, so a grid whose
        panels are at different zooms (one unlocked, then relocked) still moves
        by the same distance across the sample rather than the same number of
        pixels.

        :param d_row: Signed row displacement in pixels of the driving panel.
        :param d_column: Signed column displacement in pixels of the driving
            panel.
        """
        driver = self._require_key(key)
        canvas = self[driver]
        shift = (canvas.step[0] * float(d_row), canvas.step[1] * float(d_column))
        self._apply(driver, lambda c: replace(c, origin=(
            c.origin[0] + shift[0], c.origin[1] + shift[1])))

    def at_depth(self, *, key: Optional[str] = None, **coords: float) -> None:
        """Move every locked panel to another slice: ``link.at_depth(z=12.0)``."""
        driver = self._require_key(key)
        self._apply(driver, lambda c: c.at_depth(**coords))

    def reset(self, source: Any, **kwargs: Any) -> None:
        """Fit every locked panel to ``source`` again, keeping its pixel size.

        :param source: Layer stack, layer or extent mapping accepted by
            :meth:`Canvas.covering`.
        """
        for key, canvas in list(self._canvases.items()):
            if not self._locked.get(key, True):
                continue
            fitted = Canvas.covering(source, height=canvas.shape[0],
                                     width=canvas.shape[1], axes=canvas.axes,
                                     depth=dict(canvas.depth), **kwargs)
            self._canvases[key] = fitted
        self._emit("")

    def _require_key(self, key: Optional[str]) -> str:
        """Resolve ``key``, or the first panel when it is ``None``.

        Raises rather than returning ``None`` for an empty link -- a move with no
        panel to move is a caller error, not a no-op.
        """
        if key is not None:
            self[key]
            return str(key)
        if not self._canvases:
            raise LayerError("this link has no panels to move")
        return next(iter(self._canvases))

    def _apply(self, driver: str, move: Callable[[Canvas], Canvas]) -> None:
        """Apply ``move`` to ``driver`` and to every other locked panel."""
        follow = self._locked.get(driver, True)
        for key, canvas in list(self._canvases.items()):
            if key == driver or (follow and self._locked.get(key, True)):
                self._canvases[key] = move(canvas)
        self._emit(driver)

    def subscribe(self, listener: Callable[[str], None]
                  ) -> Callable[[str], None]:
        """Register a listener called with the key of each panel that changes.

        Held by strong reference, exactly like :meth:`LayerStack.subscribe`: a
        view that subscribes must :meth:`unsubscribe` when it closes.

        :param listener: Callable receiving the identifier of the panel that
            changed; it is retained by strong reference.
        """
        if not callable(listener):
            raise LayerError(f"a listener must be callable, got {listener!r}")
        if listener not in self._listeners:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: Callable[[str], None]) -> bool:
        """Remove a listener; return ``True`` if this call removed it.

        :param listener: Previously subscribed callback to remove.
        """
        try:
            self._listeners.remove(listener)
        except ValueError:
            return False
        return True

    def _emit(self, key: str) -> None:
        """Tell every listener that one panel moved.

        Iterates a COPY, so a listener that adds or removes another during the
        call does not change the list being walked.
        """
        for listener in list(self._listeners):
            listener(key)

    def describe(self) -> str:
        """One line per panel — what it is showing and whether it follows."""
        if not self._canvases:
            return "no panels"
        lines = []
        for key, canvas in self._canvases.items():
            state = "linked" if self._locked.get(key, True) else "free"
            lines.append(f"{key}: {canvas.shape[1]}×{canvas.shape[0]} at "
                         f"{canvas.step[1]:.4g} {canvas.units}/px · {state}")
        return "\n".join(lines)



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
    :param object_type: which mask this layer segments (``'cell'``,
        ``'nucleus'``, …). Without it the keys are untyped, so clicking the
        nucleus labelled 1 publishes the same string as clicking the pathogen
        labelled 1 in the same field, and the two views the click was supposed
        to link land on whichever of the two their table happened to hold
        first. Use ``None`` for a mask loaded from a file that does not record
        its object type.
    """

    values: Mapping[str, Any]
    timelapse: bool = False
    object_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Coerce the field's parts to strings and freeze them."""
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
        object.__setattr__(
            self, "object_type",
            str(self.object_type).strip().lower()
            if self.object_type is not None else None)

    @classmethod
    def columns(cls, *, timelapse: bool = False) -> Tuple[str, ...]:
        """The key columns a field key of this flavour needs, label excluded."""
        from .selection import OBJECT_KEY_COLUMNS
        label_column = OBJECT_KEY_COLUMNS[-1]
        if timelapse:
            from . import schema
            everything = schema.TIMEPOINT_KEY_COLUMNS + (label_column,)
        else:
            everything = OBJECT_KEY_COLUMNS
        return tuple(c for c in everything if c != label_column)

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *,
                 timelapse: bool = False,
                 object_type: Optional[str] = None) -> "FieldKey":
        """Take the key columns out of a measurement row (or any mapping).

        Anything else on the row is dropped — a field key is an identity, and
        carrying a measurement along with it would make two keys for the same
        field compare unequal.

        :param row: Measurement-row mapping from which the required identity
            columns are extracted.
        """
        wanted = cls.columns(timelapse=timelapse)
        if object_type is None:
            from . import schema as _schema
            candidate = row.get(_schema.OBJECT_TYPE_KEY)
            object_type = candidate if _schema.is_object_type(candidate) \
                else None
        return cls(values={c: row[c] for c in wanted if c in row},
                   timelapse=timelapse, object_type=object_type)

    def frame(self, labels: Iterable[int]):
        """A one-column-per-key-column frame for ``labels``, in their order.

        :param labels: Iterable of integer-like object labels to attach to this
            field identity.
        """
        import pandas as pd
        ids = [int(v) for v in labels]
        data = {c: [self.values[c]] * len(ids) for c in self.values}
        from .selection import OBJECT_KEY_COLUMNS
        data[OBJECT_KEY_COLUMNS[-1]] = ids
        return pd.DataFrame(data)

    def object_keys(self, labels: Iterable[int]):
        """:class:`pandas.Index` of object keys for ``labels``, in order.

        :param labels: Iterable of integer-like object labels to encode.
        """
        from .selection import object_keys
        return object_keys(self.frame(labels), timelapse=self.timelapse,
                           object_type=self.object_type)

    def object_key(self, label: int) -> str:
        """The one object key for ``label``.

        :param label: Integer-like object label to encode for this field.
        """
        return str(self.object_keys([int(label)])[0])



@dataclass(frozen=True)
class LayerEvent:
    """A change notification that a :class:`LayerStack` sends to its listeners.

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
    """Base class for stack layers: a name, a :class:`Spacing` and display properties.

    Subclasses implement :meth:`_draw`; everything shared — visibility,
    opacity, blending, spacing, notification — lives here.

    Display properties are plain attributes with setters that notify the
    stack, so a view repaints because the model changed rather than because
    the widget that changed it remembered to ask.

    :param name: Non-blank layer identifier; names are uniquified when the
        layer enters a :class:`LayerStack`.
    :param spacing: physical size of one pixel along each spatial axis.
        Defaults to isotropic. It is what makes a measurement in microns mean
        the same thing on two microscopes, so it must have exactly
        :attr:`ndim` axes.
    :param visible: whether the layer draws at all. Default ``True``.
    :param opacity: 0.0 transparent to 1.0 opaque. Default ``1.0``.
    :param blending: how the layer combines with what is under it, from
        :class:`Blending`. Default translucent.
    :param metadata: arbitrary values carried with the layer. Copied, so the
        caller's mapping is not held; spaCR itself reads nothing from it.
    :raises LayerError: when the name is blank, the opacity is not a finite
        number, the blending is unknown, or ``spacing`` has the wrong number
        of axes for this layer.
    """

    #: What kind of layer this is, for a view choosing an icon or an editor.
    kind: str = "layer"

    def __init__(self, *, name: str, spacing: Optional[Spacing] = None,
                 visible: bool = True, opacity: float = 1.0,
                 blending: str = Blending.TRANSLUCENT,
                 metadata: Optional[Mapping[str, Any]] = None):
        """Set the layer's name, opacity and visibility, validated."""
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

    @staticmethod
    def _check_name(name: str) -> str:
        """A non-blank name, stripped. Blank is refused.

        A layer is addressed by name, so an empty one is a layer nothing can
        refer to.
        """
        text = str(name).strip()
        if not text:
            raise LayerError("a layer needs a non-blank name")
        return text

    @staticmethod
    def _check_opacity(value: float) -> float:
        """A finite number in range. Anything else is refused.

        Not clamped: a caller passing 5.0 has misunderstood the scale, and
        silently making it 1.0 hides that.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise LayerError(f"opacity must be a number, got {value!r}") from None
        if not math.isfinite(v):
            raise LayerError(f"opacity must be finite, got {value!r}")
        return min(1.0, max(0.0, v))

    @property
    def name(self) -> str:
        """This layer's name, unique within its stack.

        :returns: the name.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Rename this layer, through the stack when it is in one.

        The stack is what enforces uniqueness, so renaming behind its back
        would let two layers share a name and make the second unreachable by
        the only handle anything else has on it.

        :param value: the new name.
        """
        text = self._check_name(value)
        if text == self._name:
            return
        if self._stack is not None:
            self._stack.rename(self, text)
            return
        self._name = text

    @property
    def visible(self) -> bool:
        """Whether this layer is drawn.

        :returns: True when visible.
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Show or hide this layer, notifying only on a real change.

        :param value: True to draw it.
        """
        value = bool(value)
        if value != self._visible:
            self._visible = value
            self._notify("visible")

    @property
    def opacity(self) -> float:
        """How opaque this layer is drawn, from 0 to 1.

        :returns: the opacity.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set the opacity, notifying only on a real change.

        An opacity slider emits continuously while it is dragged; notifying
        only when the value changes avoids a redraw per emitted value.

        :param value: 0 to 1. Values outside the range are clamped; a
            non-numeric or non-finite value raises :exc:`LayerError`.
        """
        value = self._check_opacity(value)
        if value != self._opacity:
            self._opacity = value
            self._notify("opacity")

    @property
    def blending(self) -> str:
        """How this layer's pixels combine with what is under it.

        :returns: the blending mode's name.
        """
        return self._blending

    @blending.setter
    def blending(self, value: str) -> None:
        """Set the blending mode, notifying only on a real change.

        :param value: the mode's name; validated before it is stored.
        """
        value = Blending.check(value)
        if value != self._blending:
            self._blending = value
            self._notify("blending")

    @property
    def spacing(self) -> Spacing:
        """Where this layer's elements sit in the world.

        :returns: the :class:`Spacing`.
        """
        return self._spacing

    @spacing.setter
    def spacing(self, value: Spacing) -> None:
        """Move or rescale this layer in the world.

        :param value: the new :class:`Spacing`.
        """
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
        """The world axes this layer's data is indexed by, outermost first.

        Axes are named rather than positional, so two layers with different
        dimensionality agree on what "z" means; this lets a 2D mask be drawn
        over a 3D stack.

        :returns: the axis names.
        """
        return self._spacing.axes

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The world box this layer occupies, keyed by axis."""
        raise NotImplementedError

    def to_world(self, index: Sequence[float]) -> Dict[str, float]:
        """Data index → ``{axis: world}``.

        :param index: Fractional data coordinate in this layer's spacing-axis
            order.
        """
        return self._spacing.world_map(index)

    def to_data(self, world: Mapping[str, float]) -> Tuple[float, ...]:
        """``{axis: world}`` → fractional data index.

        :param world: Mapping from world-axis names to coordinates; omitted
            spacing axes are evaluated at world coordinate zero.
        """
        return self._spacing.data_from_map(world)

    def render(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Draw onto ``canvas``; returns ``(rgb, coverage)``.

        ``rgb`` is ``(H, W, 3)`` float32 in 0–1 and ``coverage`` is ``(H, W)``
        float32 in 0–1 marking where this layer draws anything. Opacity and
        blending are NOT applied here — the stack applies them, so a layer's
        own render is the same whatever it is composited with.

        :param canvas: World window and output grid on which to draw the layer.
        """
        rgb, coverage = self._draw(canvas)
        return (np.asarray(rgb, dtype=np.float32),
                np.asarray(coverage, dtype=np.float32))

    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Render this layer onto ``canvas``. Subclasses implement it.

        :returns: ``(rgb, alpha)`` at the canvas's shape.
        """
        raise NotImplementedError

    def _blank(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Fully transparent RGB and alpha at the canvas's shape.

        What a layer with nothing to draw returns, so a caller never has to
        branch on "no output".
        """
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

    def _notify(self, detail: str, kind: str = "changed") -> None:
        """Tell the stack this layer changed, if it is in one.

        Silent when the layer is not in a stack: a layer edited before it is
        added is not an error, it just has nobody to tell.
        """
        if self._stack is not None:
            self._stack._emit(LayerEvent(kind=kind, layer=self,
                                         index=self._stack.index(self),
                                         detail=detail))

    def describe(self) -> str:
        """One line for a layer list row."""
        vis = "" if self._visible else " · hidden"
        return (f"{self._name} ({self.kind}) · {self._blending} · "
                f"{self._opacity:.0%}{vis}")

    def __repr__(self) -> str:
        """The layer's class, name and visibility."""
        return f"<{type(self).__name__} {self._name!r} {self.shape}>"


class ImageLayer(Layer):
    """Intensity data — one channel or many, each with its own LUT.

    :param name: layer name. Default ``"image"``.
    :param channel_names: a label per channel, for a channel list or a
        legend. Defaults to positional names -- and naming them is what lets
        a reader tell which channel is the nucleus without counting.

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
        """Build an image layer.

        :param data: the pixel array, channels included.
        :param kwargs: name, spacing, colormaps and the rest of the layer contract.
        """
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

    @staticmethod
    def _resolve_colormaps(spec: Optional[Sequence[Any]],
                           n: int) -> List[Colormap]:
        """One colormap per channel, whatever the caller supplied.

        A SINGLE CHANNEL GETS GREY, several get the distinguishable defaults:
        an image with three channels drawn in one colormap is three pictures
        of the same colour, which is what a naive default produces.

        :param spec: colormaps, names, or None for the defaults.
        :param n: how many channels there are.
        :returns: one colormap per channel.
        """
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
        """One contrast range per channel, or None to auto-scale it.

        :param spec: the limits, or None.
        :param n: how many channels there are.
        :returns: one range per channel.
        """
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

    @property
    def data(self) -> np.ndarray:
        """The pixel array, channels included.

        :returns: the array as given; not a copy.
        """
        return self._data

    @property
    def ndim(self) -> int:
        """How many SPATIAL axes this image has, excluding channels.

        A channel is not a dimension you can navigate, so counting it here
        would put a slider on the viewer for something that is not a place.

        :returns: the spatial axis count.
        """
        return len(self._spatial)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The spatial shape, excluding channels.

        :returns: the spatial extent per axis.
        """
        return self._spatial

    @property
    def n_channels(self) -> int:
        """How many channels this image carries.

        :returns: the channel count.
        """
        return len(self._planes)

    @property
    def channel_names(self) -> Tuple[str, ...]:
        """What each channel is called, in channel order.

        :returns: one name per channel.
        """
        return self._channel_names

    def channel_data(self, channel: int) -> np.ndarray:
        """The spatial array for one channel (a view, not a copy).

        :param channel: Zero-based channel index accepted by normal sequence
            indexing.
        """
        return self._planes[int(channel)]

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The world box this layer occupies, per named axis.

        What lets the viewer frame a stack whose layers have different
        shapes and spacings without any of them knowing about the others.

        :returns: ``{axis: (low, high)}``.
        """
        return self._spacing.extent(self._spatial)

    @property
    def colormaps(self) -> Tuple[Colormap, ...]:
        """One colormap per channel, in channel order.

        A tuple copy, so a caller cannot reorder this layer's colormaps by
        mutating what it was handed.

        :returns: the colormaps.
        """
        return tuple(self._colormaps)

    def set_colormap(self, value: Any, channel: int = 0) -> None:
        """Give one channel a new LUT.

        :param value: Colormap or colour specification accepted by
            :func:`colormap`.
        """
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
        """Set the FIRST channel's colormap.

        The convenience spelling for the single-channel case, which is most
        of them; :meth:`set_colormap` names the channel when there is more
        than one.

        :param value: a colormap, or a name to look one up by.
        """
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

        :param low: Lower intensity bound in the channel's data units.
        :param high: Upper intensity bound in the channel's data units; it
            must be greater than ``low``.
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
        """Whether one channel participates in rendering.

        :param channel: Zero-based channel index accepted by normal sequence
            indexing.
        """
        return self._channel_visible[int(channel)]

    def set_channel_visible(self, channel: int, visible: bool) -> None:
        """Set whether one channel participates in rendering.

        :param channel: Zero-based channel index accepted by normal sequence
            indexing.
        :param visible: Truth value for the channel's new visibility state.
        """
        channel = int(channel)
        if self._channel_visible[channel] != bool(visible):
            self._channel_visible[channel] = bool(visible)
            self._notify("channel_visible")

    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Paint this layer into the canvas.

        :param canvas: the world window being drawn.
        :returns: the rendered RGBA array.
        """
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

    :param name: layer name. Default ``"labels"``.
    :param seed: seeds the label-to-colour mapping. The same seed gives the
        same object the same colour across sessions and across figures, which
        is what makes two screenshots of one field comparable.

    Colours come from :func:`label_color`, so an object keeps its colour
    between sessions and between this viewer and the live preview.

    :param data: Two- or three-dimensional array-like segmentation labels;
        values must be integers or exactly integral numbers.
    :param field: the :class:`FieldKey` this mask segments. Optional, but
        without it a click can only say "label 17" — with it, the click can
        publish ``plate1_A_1_1_17`` to every other view.
    """

    kind = "labels"

    def __init__(self, data: Any, *, name: str = "labels",
                 field: Optional[FieldKey] = None, seed: int = 0,
                 **kwargs: Any):
        """Build a labels layer.

        :param data: the label array; 0 is background.
        :param kwargs: name, spacing, field and the rest of the layer contract.
        """
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
        """The label array: one integer per pixel, 0 for background.

        :returns: the array as given; not a copy.
        """
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """Replace the labels, keeping the same shape.

        The new array must have the current shape. A labels layer is
        registered against an image, so accepting a differently shaped array
        would break that correspondence and mislabel every pixel without an
        error.

        :param value: the new label array.
        :raises LayerError: when the shape differs, or when non-integer values
            have a fractional part.
        """
        array = np.asarray(value)
        if array.shape != self._data.shape:
            raise LayerError(
                f"labels layer {self.name!r} is {self._data.shape}; the new "
                f"array is {array.shape}. Replace the layer rather than its "
                f"data if the grid really changed — the spacing describes the "
                f"old grid.")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(np.mod(array, 1), 0)):
                raise LayerError(
                    f"a labels layer needs integer labels, got dtype "
                    f"{array.dtype} with fractional values")
            array = array.astype(np.int64)
        self._data = array
        self._notify("data", kind="data")

    @property
    def ndim(self) -> int:
        """How many axes the label array has.

        :returns: the axis count.
        """
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """The label array's shape.

        :returns: the extent per axis.
        """
        return tuple(self._data.shape)

    @property
    def field(self) -> Optional[FieldKey]:
        """Which imaging field these labels belong to, if any.

        :returns: the :class:`FieldKey`, or None when unattached.
        """
        return self._field

    @field.setter
    def field(self, value: Optional[FieldKey]) -> None:
        """Attach these labels to an imaging field.

        Typed rather than free text, because the field key is what joins a
        mask back to its measurements and a near-miss string would join to
        nothing without saying so.

        :param value: the field, or None to detach.
        :raises LayerError: when ``value`` is not a FieldKey.
        """
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
        """Highlight one label, notifying only on a real change.

        :param value: the label's integer id; 0 selects nothing.
        """
        value = int(value)
        if value != self._selected_label:
            self._selected_label = value
            self._notify("selected_label")

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The world box this layer occupies, per named axis.

        What lets the viewer frame a stack whose layers have different
        shapes and spacings without any of them knowing about the others.

        :returns: ``{axis: (low, high)}``.
        """
        return self._spacing.extent(self._data.shape)

    def labels(self) -> np.ndarray:
        """Every non-zero label present, sorted."""
        unique = np.unique(self._data)
        return unique[unique != 0]

    def label_at_world(self, world: Mapping[str, float]) -> int:
        """The label under a world point, or 0 for background / outside.

        :param world: Mapping from axis names to a point in world coordinates.
        """
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

        :param world: Mapping from axis names to the world point to identify.
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

    def brush_index(self, world: Mapping[str, float], *,
                    radius: float = 0.0) -> Tuple[np.ndarray, ...]:
        """The elements a world-space ball brush covers, as index arrays.

        The brush is a ball in WORLD space, so on an anisotropic stack it
        covers fewer z-slices than y-rows — which is what the user means by a
        5 µm brush. A brush measured in array elements would be a 5-slice
        cylinder, i.e. 10 µm deep and 3.25 µm wide on an ordinary spaCR stack.

        Split out of :meth:`paint` so a caller that has to *record* what it
        changed — :class:`spacr.curation.MaskCuration`, which keeps an undo
        history and an audit trail — asks the layer for the geometry instead
        of re-deriving the anisotropy rule and getting it subtly different.
        There is one ball, and it is defined here.

        :param world: Mapping from axis names to the brush centre in world
            coordinates; omitted spacing axes are evaluated at zero.
        :returns: one integer array per axis, ready to index :attr:`data`
            with. Empty arrays when the brush falls entirely off the grid,
            which indexes to nothing rather than raising.
        """
        sp = self._spacing
        centre = sp.data_from_map(world)
        radius = float(radius)
        empty = tuple(np.zeros(0, dtype=np.intp)
                      for _ in range(self._data.ndim))
        slices = []
        offsets = []
        for axis_at, size in enumerate(self._data.shape):
            reach = radius / abs(sp.scale[axis_at])
            lo = max(0, int(math.floor(centre[axis_at] - reach)))
            hi = min(int(size), int(math.ceil(centre[axis_at] + reach)) + 1)
            if lo >= hi:
                return empty
            slices.append(slice(lo, hi))
            offsets.append((np.arange(lo, hi, dtype=np.float64)
                            - centre[axis_at]) * abs(sp.scale[axis_at]))
        dist2 = np.zeros([len(o) for o in offsets], dtype=np.float64)
        for axis_at, off in enumerate(offsets):
            shaped = off.reshape([-1 if i == axis_at else 1
                                  for i in range(len(offsets))])
            dist2 = dist2 + shaped ** 2
        inside = dist2 <= max(radius, _EPS) ** 2
        local = np.nonzero(inside)
        return tuple(np.asarray(part, dtype=np.intp) + s.start
                     for part, s in zip(local, slices))

    def set_labels_at(self, index: Sequence[np.ndarray], label: int) -> int:
        """Set the elements ``index`` names to ``label``; how many changed.

        The write half of :meth:`brush_index`, and the ONE place a labels
        layer's data is edited element-wise — so exactly one call notifies.
        A caller doing its own arithmetic on :attr:`data` would mutate the
        array without telling the canvas, and the picture would come right
        only at the next unrelated repaint.

        :param index: One integer index array per data axis, suitable for
            NumPy advanced indexing.
        :param label: Integer-like label to write at the selected elements.
        """
        index = tuple(index)
        if not index or not len(index[0]):
            return 0
        previous = self._data[index]
        changed = int(np.count_nonzero(previous != int(label)))
        if changed:
            self._data[index] = int(label)
            self._notify("paint", kind="data")
        return changed

    def paint(self, world: Mapping[str, float], label: int, *,
              radius: float = 0.0) -> int:
        """Set every element within ``radius`` world units of a point.

        :meth:`brush_index` owns the ball; this is that plus the write.

        :param world: Mapping from axis names to the brush centre in world
            coordinates.
        :param label: Integer-like label to paint.
        :returns: how many elements changed.
        """
        return self.set_labels_at(
            self.brush_index(world, radius=radius), label)

    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Paint this layer into the canvas.

        :param canvas: the world window being drawn.
        :returns: the rendered RGBA array.
        """
        index, valid = self._sample_index(canvas)
        values = self._data[index]
        present = valid & (values != 0)
        rgb = label_colors(np.where(present, values, 0), seed=self._seed)
        rgb[~present] = 0.0
        return rgb, present.astype(np.float32)


class PointsLayer(Layer):
    """Points in the world — centroids, counted objects, clicked markers.

    :param name: layer name. Default ``"points"``.
    :param ndim: how many axes a point has. Used only when ``data`` is empty;
        otherwise the data decides, so an explicit ``ndim`` cannot contradict
        the points that were passed.
    :param face_color: fill colour, as any :func:`to_rgba` specification.
    :param border_color: outline colour.
    :param border_width: outline width in world units. ``0`` draws no
        outline, which is the default.

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
    :param properties: ``{name: (N,) array}`` per-point columns, such as a
        category for each counted point. Each must have one value per point.
    """

    kind = "points"

    def __init__(self, data: Any = None, *, name: str = "points",
                 ndim: int = 2, size: Any = 10.0,
                 face_color: Any = "yellow", border_color: Any = "black",
                 border_width: float = 0.0,
                 properties: Optional[Mapping[str, Any]] = None,
                 **kwargs: Any):
        """Build a points layer.

        :param data: an ``(n, ndim)`` array of data coordinates, or None.
        :param kwargs: name, spacing, colours and the rest of the layer contract.
        """
        array = self._as_points(data, ndim)
        self._ndim = array.shape[1]
        self._data = array
        super().__init__(name=name, **kwargs)
        self._size = self._as_sizes(size, len(array))
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
        """Coerce whatever was given into an ``(n, ndim)`` float array.

        NONE BECOMES AN EMPTY ARRAY OF THE RIGHT WIDTH rather than a zero-
        column one, so an empty layer still has a dimensionality and can be
        appended to without deciding it later.

        :param data: the points, or None.
        :param ndim: how many world axes each point has.
        :returns: the coordinates.
        """
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
        """One size per point, broadcasting a scalar.

        :param size: one size for all, or one per point.
        :param n: how many points there are.
        :returns: the sizes.
        """
        if np.isscalar(size):
            return np.full(n, float(size), dtype=np.float64)
        array = np.asarray(size, dtype=np.float64).reshape(-1)
        if len(array) != n:
            raise LayerError(f"{n} points but {len(array)} sizes")
        return array

    @property
    def data(self) -> np.ndarray:
        """The points in data coordinates, ``(N, ndim)``."""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """Replace the points, keeping the same dimensionality.

        The number of coordinates per point must stay the same. A points
        layer sits in a world whose axes are already named, so accepting a
        differently shaped array would put every point at a coordinate on a
        different axis.

        :param value: an ``(n, ndim)`` array of data coordinates.
        :raises LayerError: when the dimensionality differs, or when a column
            in :attr:`properties` would no longer have one value per point.
        """
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
        """How many world axes each point is placed on.

        :returns: the axis count.
        """
        return self._ndim

    @property
    def size(self) -> np.ndarray:
        """Per-point diameters in world units."""
        return self._size

    def set_size(self, value: Any) -> None:
        """Set every point's diameter, or one per point.

        :param value: Scalar diameter applied to every point, or an array-like
            sequence with one world-unit diameter per point.
        """
        self._size = self._as_sizes(value, len(self._data))
        if np.isscalar(value):
            self._default_size = float(value)
        self._notify("size")

    @property
    def face_color(self) -> Tuple[float, float, float, float]:
        """The fill colour every point is drawn with.

        :returns: RGBA, each component 0 to 1.
        """
        return self._face

    @face_color.setter
    def face_color(self, value: Any) -> None:
        """Set the fill colour, accepting any spelling `to_rgba` understands.

        :param value: a colour name, hex string, or RGB(A) sequence.
        """
        self._face = to_rgba(value)
        self._notify("face_color")

    @property
    def border_color(self) -> Tuple[float, float, float, float]:
        """The outline colour every point is drawn with.

        :returns: RGBA, each component 0 to 1.
        """
        return self._border

    @border_color.setter
    def border_color(self, value: Any) -> None:
        """Set the outline colour, accepting any spelling `to_rgba` understands.

        :param value: a colour name, hex string, or RGB(A) sequence.
        """
        self._border = to_rgba(value)
        self._notify("border_color")

    @property
    def border_width(self) -> float:
        """Border thickness in world units. 0 draws no border."""
        return self._border_width

    @border_width.setter
    def border_width(self, value: float) -> None:
        """Set the outline width, clamped at zero.

        A negative width is clamped to 0 (no outline) rather than raising.

        :param value: the width; negatives become 0.
        """
        self._border_width = max(0.0, float(value))
        self._notify("border_width")

    def add(self, point: Sequence[float], *, size: Optional[float] = None,
            **properties: Any) -> int:
        """Append one point in DATA coordinates; returns its index.

        :param point: Data-coordinate sequence with one value per layer axis.
        """
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
        """Append one point given as ``{axis: world}``.

        :param world: Mapping from axis names to world coordinates; omitted
            layer axes are evaluated at world coordinate zero.
        """
        return self.add(self._spacing.data_from_map(world), **kwargs)

    def remove(self, index: int) -> None:
        """Drop one point.

        :param index: Zero-based index of the point to remove; negative
            indices are rejected.
        """
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
        """The world box these points occupy, INCLUDING their drawn size.

        Half a point's size is added on each side, because a point centred
        on the edge of the data is still drawn past it -- and a viewer that
        framed the centres would clip every point on the boundary.

        An empty layer reports a zero box rather than an infinite one.

        :returns: ``{axis: (low, high)}``.
        """
        world = self.world
        if len(world) == 0:
            return {a: (0.0, 0.0) for a in self._spacing.axes}
        half = self._size[:, None] / 2.0
        return {a: (float(np.min(world[:, i] - half[:, 0])),
                    float(np.max(world[:, i] + half[:, 0])))
                for i, a in enumerate(self._spacing.axes)}

    def nearest(self, world: Mapping[str, float]) -> Optional[int]:
        """Index of the point whose disc contains ``world``, nearest first.

        :param world: Mapping from axis names to the query point in world
            coordinates; omitted layer axes are evaluated at zero.
        """
        if len(self._data) == 0:
            return None
        target = np.array([float(world.get(a, 0.0))
                           for a in self._spacing.axes], dtype=np.float64)
        d = np.linalg.norm(self.world - target[None, :], axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= self._size[i] / 2.0 else None

    def _draw(self, canvas: Canvas) -> Tuple[np.ndarray, np.ndarray]:
        """Paint this layer into the canvas.

        :param canvas: the world window being drawn.
        :returns: the rendered RGBA array.
        """
        rgb, coverage = self._blank(canvas)
        if len(self._data) == 0:
            return rgb, coverage
        axes = self._spacing.axes
        try:
            row_axis = axes.index(canvas.axes[0])
            col_axis = axes.index(canvas.axes[1])
        except ValueError:
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
        """Normalise the kind and refuse one this layer cannot draw.

        An unknown kind raises :exc:`LayerError` at construction rather than at
        paint time, because a shape with an unknown kind draws nothing and the
        fault would otherwise appear to be in the renderer.
        """
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
        """How many world axes this shape's vertices are placed on.

        :returns: the axis count.
        """
        return int(self.data.shape[1])

    @property
    def is_closed(self) -> bool:
        """Whether the outline joins back to its first vertex.

        A closed shape can be filled and can be asked what is inside it; an
        open one is a path and can only be drawn.

        :returns: True when closed.
        """
        return self.kind in self.CLOSED


class ShapesLayer(Layer):
    """Drawn regions of interest; :mod:`spacr.roi` uses them to limit Measure to a region.

    :param shapes: the :class:`Shape` objects to start with. Every one must
        have the same number of axes -- a layer holding a 2-D and a 3-D shape
        could not be rasterised into one mask.
    :param name: layer name. Default ``"shapes"``.
    :param ndim: axes per shape, used only when ``shapes`` is empty; the
        shapes decide otherwise.
    :raises LayerError: when the shapes disagree about how many axes they have.

    Shapes are geometry, not pixels: they are stored as vertices in data
    coordinates and rasterised on demand, so the same ROI can be turned into a
    mask for a full-resolution mask layer and for a downsampled preview and
    mean the same region in both. :meth:`mask` is that conversion.
    """

    kind = "shapes"

    def __init__(self, shapes: Optional[Iterable[Shape]] = None, *,
                 name: str = "shapes", ndim: int = 2, **kwargs: Any):
        """Build a shapes layer.

        :param shapes: the shapes to start with, or None for an empty layer.
        :param kwargs: name, spacing and the rest of the layer contract.
        """
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
        """How many world axes this layer's shapes are placed on.

        :returns: the axis count.
        """
        return self._ndim

    @property
    def shapes(self) -> Tuple[Shape, ...]:
        """The shapes this layer holds.

        A tuple copy, so a caller cannot add or reorder the layer's shapes
        by mutating what it was handed.

        :returns: the shapes, in draw order.
        """
        return tuple(self._shapes)

    def __len__(self) -> int:
        """How many shapes this layer holds.

        :returns: the shape count.
        """
        return len(self._shapes)

    def add(self, shape: Shape) -> int:
        """Append a shape; returns its index.

        :param shape: :class:`Shape` with the same dimensionality as this
            layer.
        """
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
        """Append a closed polygon.

        :param vertices: Array-like vertices in data coordinates, shaped
            ``(M, ndim)`` with at least three rows.
        """
        return self.add(Shape("polygon", vertices, **kwargs))

    def add_rectangle(self, corner_a: Any, corner_b: Any, **kwargs: Any) -> int:
        """Append an axis-aligned rectangle from opposite corners.

        :param corner_a: First corner in data coordinates, one value per axis.
        :param corner_b: Opposite corner in data coordinates, one value per
            axis.
        """
        return self.add(Shape("rectangle", np.asarray([corner_a, corner_b]),
                              **kwargs))

    def add_ellipse(self, corner_a: Any, corner_b: Any, **kwargs: Any) -> int:
        """Append an axis-aligned ellipse inside an opposite-corner box.

        :param corner_a: First bounding-box corner in data coordinates, one
            value per axis.
        :param corner_b: Opposite bounding-box corner in data coordinates, one
            value per axis.
        """
        return self.add(Shape("ellipse", np.asarray([corner_a, corner_b]),
                              **kwargs))

    def add_path(self, vertices: Any, **kwargs: Any) -> int:
        """Append an open path.

        :param vertices: Array-like vertices in data coordinates, shaped
            ``(M, ndim)`` with at least two rows.
        """
        return self.add(Shape("path", vertices, **kwargs))

    def remove(self, index: int) -> Shape:
        """Drop a shape and return it.

        :param index: Integer index of the shape to remove; normal Python
            negative indexing is accepted.
        """
        try:
            shape = self._shapes.pop(int(index))
        except IndexError:
            raise LayerError(
                f"no shape {index} in a layer of {len(self._shapes)}") from None
        self._notify("remove", kind="data")
        return shape

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The world box every shape's vertices fall within.

        An empty layer reports a zero box rather than an infinite one, so a
        viewer framing the stack is not dragged to infinity by a layer that
        has nothing in it yet.

        :returns: ``{axis: (low, high)}``.
        """
        axes = self._spacing.axes
        if not self._shapes:
            return {a: (0.0, 0.0) for a in axes}
        scale = np.asarray(self._spacing.scale, dtype=np.float64)
        offset = np.asarray(self._spacing.translate, dtype=np.float64)
        stacked = np.vstack([s.data * scale + offset for s in self._shapes])
        return {a: (float(np.min(stacked[:, i])), float(np.max(stacked[:, i])))
                for i, a in enumerate(axes)}

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
            edge_c = vc[i] + (r - vr[i]) * (vc[j] - vc[i]) / (dr if dr else 1.0)
            inside ^= crosses & (c < edge_c)
            j = i
        return inside

    @staticmethod
    def _inside_ellipse(rows: np.ndarray, cols: np.ndarray,
                        vertices: np.ndarray) -> np.ndarray:
        """Which of the given pixels fall inside an ellipse.

        The ellipse is given by its bounding vertices, so a rotated one is
        handled by the same test as an axis-aligned one.

        :param rows: pixel row indices.
        :param cols: pixel column indices.
        :param vertices: the ellipse's bounding box.
        :returns: a boolean mask over the pixels.
        """
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

        :param canvas: World plane and pixel grid on which to rasterise the
            closed shapes.
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
        """Paint this layer into the canvas.

        :param canvas: the world window being drawn.
        :returns: the rendered RGBA array.
        """
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



LayerLike = Union[Layer, str, int]
Listener = Callable[[LayerEvent], None]


class LayerStack:
    """An ordered list of layers sharing one world.

    :param layers: the layers to start with, bottom first. Added through
        :meth:`add`, so each is renamed to be unique and adopted as it
        arrives.
    :param units: the name of the world unit -- ``"um"``, ``"px"`` -- shown
        on scale bars and axis labels. It LABELS the spacing rather than
        converting it: it is what the numbers in :class:`Spacing` already
        mean, so setting it does not rescale anything.

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
        """Build a stack, optionally seeded with layers.

        :param layers: the layers to start with.
        """
        self._layers: List[Layer] = []
        self._units = None if units is None else str(units)
        self._selected: Optional[Layer] = None
        self._listeners: List[Listener] = []
        for layer in layers or ():
            self.append(layer)

    def __len__(self) -> int:
        """How many layers the stack holds.

        :returns: the layer count.
        """
        return len(self._layers)

    def __iter__(self):
        """Iterate the layers, bottom to top.

        :returns: an iterator over the layers.
        """
        return iter(self._layers)

    def __contains__(self, item: Any) -> bool:
        """Whether a layer or a layer name is in the stack.

        :param item: a layer, or its name.
        :returns: True when present.
        """
        if isinstance(item, Layer):
            return any(l is item for l in self._layers)
        return any(l.name == str(item) for l in self._layers)

    def __getitem__(self, key: Any) -> Any:
        """One layer, by name or by position.

        :param key: the layer's name, or its index.
        :returns: the layer.
        """
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

        :param layer: Layer object, unique layer name or integer stack index
            to resolve.
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
        """The layer named by an object, a name or an index.

        :param layer: Layer object, unique layer name or integer stack index
            to resolve.
        """
        return self._layers[self.index(layer)]

    @property
    def units(self) -> str:
        """The world unit every layer in this stack is measured in."""
        return self._units or "px"

    def _check_units(self, layer: Layer) -> None:
        """Refuse a layer whose world axes disagree with the stack's.

        TWO LAYERS IN DIFFERENT UNITS CANNOT BE OVERLAID, and the failure is
        silent if it is not caught here: they simply draw in the wrong places
        relative to each other.

        :param layer: the layer being added.
        """
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

    def _unique_name(self, name: str, exclude: Optional[Layer] = None) -> str:
        """A name not already in the stack, suffixing if needed.

        Names are the handle everything else has on a layer, so a duplicate
        would make the second one unreachable.

        :param name: the wanted name.
        :returns: a free name.
        """
        taken = {l.name for l in self._layers if l is not exclude}
        if name not in taken:
            return name
        i = 1
        while f"{name} [{i}]" in taken:
            i += 1
        return f"{name} [{i}]"

    def append(self, layer: Layer) -> Layer:
        """Put ``layer`` on top.

        :param layer: Layer instance to attach to this stack.
        """
        return self.insert(len(self._layers), layer)

    def insert(self, index: int, layer: Layer) -> Layer:
        """Put ``layer`` at ``index`` (0 is the bottom).

        :param index: Requested insertion position, clamped between zero and
            the current stack length.
        :param layer: Layer instance to attach to this stack.
        """
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
        """Take a layer out and return it.

        :param layer: Layer object, unique layer name or integer stack index
            to remove.
        """
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

        :param source: Layer object, unique layer name or integer stack index
            to move.
        :param destination: Requested destination index, clamped to the stack
            after the source has been removed.
        """
        i = self.index(source)
        layer = self._layers.pop(i)
        j = max(0, min(int(destination), len(self._layers)))
        self._layers.insert(j, layer)
        if i != j:
            self._emit(LayerEvent("moved", layer, j, detail=str(i)))
        return j

    def raise_layer(self, layer: LayerLike) -> int:
        """Move a layer one position towards the front; return its new index.

        :param layer: Layer object, unique layer name or integer stack index
            to raise.
        """
        i = self.index(layer)
        return self.move(i, min(i + 1, len(self._layers) - 1))

    def lower_layer(self, layer: LayerLike) -> int:
        """Move a layer one position towards the back; return its new index.

        :param layer: Layer object, unique layer name or integer stack index
            to lower.
        """
        i = self.index(layer)
        return self.move(i, max(i - 1, 0))

    def to_top(self, layer: LayerLike) -> int:
        """Move one layer to the front of the stack.

        :param layer: Layer object, unique layer name or integer stack index
            to move.
        """
        return self.move(layer, len(self._layers) - 1)

    def to_bottom(self, layer: LayerLike) -> int:
        """Move one layer to the back of the stack.

        :param layer: Layer object, unique layer name or integer stack index
            to move.
        """
        return self.move(layer, 0)

    def rename(self, layer: LayerLike, name: str) -> str:
        """Rename a layer, uniquifying if needed; returns the name it got.

        :param layer: Layer object, unique layer name or integer stack index
            to rename.
        :param name: Requested non-blank name; a numeric suffix is added if it
            collides with another layer.
        """
        target = self.get(layer)
        wanted = Layer._check_name(name)
        final = self._unique_name(wanted, exclude=target)
        if final == target.name:
            return final
        target._name = final
        self._emit(LayerEvent("renamed", target, self.index(target),
                              detail=final))
        return final

    def add_image(self, data: Any, **kwargs: Any) -> ImageLayer:
        """Construct and append an image layer.

        :param data: Array-like intensity data passed to :class:`ImageLayer`.
        """
        return self.append(ImageLayer(data, **kwargs))  # type: ignore[return-value]

    def add_labels(self, data: Any, **kwargs: Any) -> LabelsLayer:
        """Construct and append a labels layer.

        :param data: Array-like segmentation labels passed to
            :class:`LabelsLayer`.
        """
        return self.append(LabelsLayer(data, **kwargs))  # type: ignore[return-value]

    def add_points(self, data: Any = None, **kwargs: Any) -> PointsLayer:
        """Build a points layer and append it to this stack.

        :param data: an ``(n, ndim)`` array of data coordinates, or None.
        :param kwargs: passed to :class:`PointsLayer`.
        :returns: the layer that was added.
        """
        return self.append(PointsLayer(data, **kwargs))  # type: ignore[return-value]

    def add_shapes(self, shapes: Optional[Iterable[Shape]] = None,
                   **kwargs: Any) -> ShapesLayer:
        """Build a shapes layer and append it to this stack.

        :param shapes: the shapes to start with, or None for an empty layer.
        :param kwargs: passed to :class:`ShapesLayer`.
        :returns: the layer that was added.
        """
        return self.append(ShapesLayer(shapes, **kwargs))  # type: ignore[return-value]

    @property
    def selected(self) -> Optional[Layer]:
        """The layer the layer-list has highlighted — what edits apply to."""
        return self._selected

    @property
    def selected_index(self) -> int:
        """Where the selected layer sits, or ``-1`` when nothing is selected."""
        return -1 if self._selected is None else self.index(self._selected)

    def select(self, layer: Optional[LayerLike]) -> Optional[Layer]:
        """Select a layer (or ``None``); returns it.

        :param layer: Layer object, unique layer name, integer stack index or
            ``None`` to clear selection.
        """
        target = None if layer is None else self.get(layer)
        if target is not self._selected:
            self._selected = target
            self._emit(LayerEvent("selected", target,
                                  -1 if target is None else self.index(target)))
        return target

    def world_extent(self) -> Dict[str, Tuple[float, float]]:
        """The union of every layer's world box, keyed by axis.

        Empty points and shapes layers are skipped, so a layer with nothing in
        it yet does not extend the box to the origin and zoom the view out.
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

    def render(self, canvas: Canvas) -> np.ndarray:
        """Composite every visible layer; ``(H, W, 3)`` float32 in 0–1.

        :param canvas: World window and output grid on which to composite the
            stack.
        """
        return self.render_rgba(canvas)[..., :3]

    def render_rgba(self, canvas: Canvas) -> np.ndarray:
        """:meth:`render` plus the composite's own coverage, ``(H, W, 4)``.

        The alpha channel is accumulated source-over whatever the layers'
        blending modes were: it records where the stack drew *anything*, which
        is what a caller compositing this onto a page background needs.

        :param canvas: World window and output grid on which to composite the
            stack.
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
        """:meth:`render` as ``(H, W, 3)`` uint8 — what a QImage wants.

        :param canvas: World window and output grid on which to composite the
            stack.
        """
        return np.clip(self.render(canvas) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def pick(self, canvas: Canvas, row: float, column: float
             ) -> Tuple[Optional[Layer], Dict[str, float], Any]:
        """What is under a canvas pixel: ``(layer, world, value)``.

        Walks from the TOP down and stops at the first visible layer with
        something there, which is what a click means — the thing you can see.
        ``value`` is the label for a labels layer, the point index for a points
        layer, the shape index for a shapes layer, and ``None`` otherwise.

        :param canvas: Canvas whose displayed stack is being queried.
        :param row: Fractional canvas-row coordinate of the query pixel.
        :param column: Fractional canvas-column coordinate of the query pixel.
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

    def subscribe(self, listener: Listener) -> Listener:
        """Register a listener called with every :class:`LayerEvent`; return it.

        Listeners are held by strong reference and are NOT weak: a view that
        subscribes must :meth:`unsubscribe` when it closes, exactly as
        :class:`spacr.qt.linked_selection.LinkedView` must unlink.

        :param listener: Callable receiving each :class:`LayerEvent`; it is
            retained by strong reference.
        """
        if not callable(listener):
            raise LayerError(f"a listener must be callable, got {listener!r}")
        if listener not in self._listeners:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: Listener) -> bool:
        """Remove a listener; return ``True`` if this call removed it.

        :param listener: Previously subscribed event callback to remove.
        """
        try:
            self._listeners.remove(listener)
        except ValueError:
            return False
        return True

    def _emit(self, event: LayerEvent) -> None:
        """Tell every subscriber the stack has changed.

        :param event: which change it was.
        """
        for listener in list(self._listeners):
            listener(event)

    def describe(self) -> str:
        """One line per layer, top first — what the layer list shows."""
        if not self._layers:
            return "no layers"
        return "\n".join(l.describe() for l in reversed(self._layers))


#: How an image layer is drawn in a 3-D view, as napari names them: the
#: brightest sample along each ray (``mip``), the brightest sample dimmed by
#: what lies in front of it (``attenuated_mip``), a glowing fog whose density
#: is the intensity (``translucent``), the mean (``average``) and the dimmest
#: sample (``minip``).
VOLUME_RENDERINGS: Tuple[str, ...] = (
    "mip", "attenuated_mip", "translucent", "average", "minip")

#: How a labels layer is drawn in a 3-D view: each object's nearest surface,
#: shaded (``surface``), or every object as coloured glass (``translucent``).
LABEL_RENDERINGS: Tuple[str, ...] = ("surface", "translucent")

#: How fast intensity in front of a sample dims it in ``attenuated_mip``,
#: per voxel of full-scale intensity crossed.
VOLUME_ATTENUATION = 0.05

#: Opacity per voxel of full-scale intensity in ``translucent`` rendering.
VOLUME_DENSITY = 0.08

#: Opacity per voxel inside an object in ``translucent`` labels rendering.
LABEL_DENSITY = 0.2


def volume_steps(shape: Sequence[int], max_voxels: Optional[int]
                 ) -> Tuple[int, ...]:
    """The integer stride along each axis that fits a volume in a budget.

    The longest strided axis is halved in resolution first, so a thin stack
    (40 sections of 1,000 x 1,000) keeps its depth while its width is
    reduced.

    :param shape: the volume's shape.
    :param max_voxels: the most voxels the strided volume may hold, or
        ``None`` for no limit.
    :returns: one stride per axis, each at least 1.
    """
    shape = tuple(max(1, int(length)) for length in shape)
    steps = [1] * len(shape)
    if max_voxels is None:
        return tuple(steps)
    budget = max(1, int(max_voxels))

    def strided(axis: int) -> int:
        """Length of one axis at its current stride."""
        return -(-shape[axis] // steps[axis])

    while math.prod(strided(axis) for axis in range(len(shape))) > budget:
        longest = max(range(len(shape)), key=strided)
        if strided(longest) <= 1:
            break
        steps[longest] += 1
    return tuple(steps)


def downsample_volume(array: Any, max_voxels: Optional[int]
                      ) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """A volume small enough to turn interactively, and how it was reduced.

    Every ``step``-th voxel is kept along each axis (see
    :func:`volume_steps`), so voxel ``j`` of the result is voxel
    ``j * step`` of the input. Labels keep their exact IDs and an image its
    exact values; nothing is averaged or interpolated.

    :param array: the volume.
    :param max_voxels: the most voxels the result may hold, or ``None`` to
        keep every voxel.
    :returns: ``(volume, steps)``; the input itself when nothing is removed.
    """
    data = np.asarray(array)
    steps = volume_steps(data.shape, max_voxels)
    if all(step == 1 for step in steps):
        return data, steps
    return np.ascontiguousarray(data[tuple(slice(None, None, step)
                                           for step in steps)]), steps


@dataclass(frozen=True)
class VolumeRays:
    """Parallel rays through a volume, one per output pixel.

    Positions are continuous array indices of the volume being drawn (voxel
    ``i`` is centred on index ``i``). The sample ``k`` of the ray through
    output row ``r`` and column ``c`` lies at
    ``origin + c * across + r * down + k * depth``; sample 0 is nearest the
    viewer.

    :param origin: the first sample of the top-left ray.
    :param across: the step from one output column to the next.
    :param down: the step from one output row to the next.
    :param depth: the step between samples, away from the viewer.
    :param shape: ``(rows, columns, samples)``.
    """

    origin: Tuple[float, ...]
    across: Tuple[float, ...]
    down: Tuple[float, ...]
    depth: Tuple[float, ...]
    shape: Tuple[int, int, int]

    @property
    def sample_length(self) -> float:
        """The distance between two samples of a ray, in voxels.

        :returns: the length of :attr:`depth`.
        """
        return float(np.linalg.norm(np.asarray(self.depth, dtype=np.float64)))

    def positions(self, first: int, last: int) -> np.ndarray:
        """Every sample of the rays through output rows ``first`` to ``last``.

        :param first: the first row.
        :param last: one past the last row.
        :returns: an ``(axes, rays, samples)`` array of indices.
        """
        rows, columns, samples = self.shape
        r = np.arange(first, last, dtype=np.float32)
        c = np.arange(columns, dtype=np.float32)
        k = np.arange(samples, dtype=np.float32)
        out = np.empty((len(self.origin), last - first, columns, samples),
                       dtype=np.float32)
        for axis in range(len(self.origin)):
            out[axis] = (self.origin[axis]
                         + r[:, None, None] * self.down[axis]
                         + c[None, :, None] * self.across[axis]
                         + k[None, None, :] * self.depth[axis])
        return out.reshape(len(self.origin), (last - first) * columns, samples)

    def ray(self, row: float, column: float) -> "VolumeRays":
        """The single ray through one output pixel, as its own 1 x 1 grid.

        Rendering it draws (and picks) exactly what that pixel of the full
        grid shows, so a click can be answered without drawing the frame.

        :param row: the output row (may be fractional).
        :param column: the output column (may be fractional).
        :returns: a :class:`VolumeRays` of shape ``(1, 1, samples)``.
        """
        origin = tuple(float(o + row * d + column * a)
                       for o, d, a in zip(self.origin, self.down, self.across))
        return VolumeRays(origin=origin, across=self.across, down=self.down,
                          depth=self.depth, shape=(1, 1, self.shape[2]))


def _layer_rendering(layer: "Layer") -> str:
    """The 3-D rendering a layer asks for in its metadata, checked."""
    allowed = LABEL_RENDERINGS if isinstance(layer, LabelsLayer) else VOLUME_RENDERINGS
    value = str(layer.metadata.get("rendering", allowed[0]))
    if value not in allowed:
        raise LayerError(f"layer {layer.name!r} cannot be rendered as {value!r}; "
                         f"choose one of {', '.join(allowed)}")
    return value


def _reduce_image(values: np.ndarray, inside: np.ndarray, rendering: str,
                  length: float) -> np.ndarray:
    """Collapse the normalised samples of each ray into one value."""
    if rendering == "mip":
        return values.max(axis=1)
    if rendering == "minip":
        return np.where(inside.any(axis=1),
                        np.where(inside, values, 1.0).min(axis=1), 0.0)
    if rendering == "average":
        return values.sum(axis=1) / np.maximum(inside.sum(axis=1), 1)
    if rendering == "attenuated_mip":
        crossed = np.cumsum(values, axis=1) - values
        return (values * np.exp(-VOLUME_ATTENUATION * length * crossed)).max(axis=1)
    alpha = np.clip(values * VOLUME_DENSITY * length, 0.0, 1.0)
    clear = np.cumprod(1.0 - alpha, axis=1)
    before = np.concatenate([np.ones_like(clear[:, :1]), clear[:, :-1]], axis=1)
    return np.minimum((before * alpha).sum(axis=1) * 2.0, 1.0)


@dataclass(frozen=True)
class SurfaceFinish:
    """How a mask surface reflects light in a 3-D view (Blinn-Phong).

    A surface point of base colour ``c`` with unit normal ``n`` is drawn as
    ``c * (ambient + diffuse * max(n.l, 0)) + specular * max(n.h, 0) **
    shininess * mix(white, c, tint)``, where ``l`` points at the light and
    ``h`` halfway between the light and the viewer. The GPU and the CPU
    renderers use the same numbers.

    :param name: the finish's key in :data:`SURFACE_FINISHES`.
    :param ambient: light every point gets, facing the light or not.
    :param diffuse: weight of the light falling on the surface.
    :param specular: weight of the highlight; 0 has none.
    :param shininess: how small and sharp the highlight is.
    :param tint: 0 for a white highlight (plastic), 1 for a highlight in the
        object's own colour (metal).
    """

    name: str
    ambient: float
    diffuse: float
    specular: float
    shininess: float
    tint: float


#: The finishes a mask surface can have in a 3-D view, dullest first.
SURFACE_FINISHES: Mapping[str, SurfaceFinish] = MappingProxyType({
    "matte": SurfaceFinish("matte", 0.35, 0.65, 0.0, 1.0, 0.0),
    "satin": SurfaceFinish("satin", 0.30, 0.60, 0.25, 12.0, 0.0),
    "glossy": SurfaceFinish("glossy", 0.25, 0.60, 0.70, 60.0, 0.0),
    "metallic": SurfaceFinish("metallic", 0.15, 0.45, 0.95, 28.0, 0.85),
})

#: The finish names, in the order a list offers them.
VOLUME_FINISHES: Tuple[str, ...] = tuple(SURFACE_FINISHES)


def surface_finish(name: Any) -> SurfaceFinish:
    """The lighting numbers of one finish.

    :param name: a key of :data:`SURFACE_FINISHES`; ``None`` is ``matte``.
    :returns: the :class:`SurfaceFinish`.
    :raises LayerError: for a name that is not a finish.
    """
    key = "matte" if name is None else str(name)
    if key not in SURFACE_FINISHES:
        raise LayerError(f"no surface finish is called {key!r}; "
                         f"choose one of {', '.join(VOLUME_FINISHES)}")
    return SURFACE_FINISHES[key]


@dataclass(frozen=True)
class SurfaceLight:
    """Where the light and the viewer are, for shading surfaces along rays.

    Every vector is a unit vector in physical units, one component per array
    axis of the volume (its storage order). The light is a head light a
    little above and left of the viewer, so it turns with the camera.

    :param view: towards the viewer.
    :param light: towards the light.
    :param half: halfway between :attr:`light` and :attr:`view`.
    :param size: the physical length of one voxel along each array axis.
    """

    view: Tuple[float, ...]
    light: Tuple[float, ...]
    half: Tuple[float, ...]
    size: Tuple[float, ...]

    @classmethod
    def for_rays(cls, rays: "VolumeRays",
                 voxel_size: Optional[Sequence[float]] = None) -> "SurfaceLight":
        """The light for one camera.

        :param rays: the camera's rays, in array indices.
        :param voxel_size: physical length of one voxel along each array
            axis; ``None`` for cubic voxels of length 1.
        :returns: the :class:`SurfaceLight`.
        """
        axes = len(rays.origin)
        size = np.ones(axes) if voxel_size is None else np.asarray(voxel_size, dtype=np.float64)
        if size.shape != (axes,) or np.any(size <= 0):
            raise LayerError(f"voxel_size needs {axes} positive lengths, got {voxel_size!r}")

        def unit(vector: np.ndarray) -> np.ndarray:
            """``vector`` scaled to length 1 (a zero vector is left alone)."""
            length = float(np.linalg.norm(vector))
            return vector / length if length > _EPS else vector

        view = unit(-np.asarray(rays.depth, dtype=np.float64) * size)
        right = unit(np.asarray(rays.across, dtype=np.float64) * size)
        up = unit(-np.asarray(rays.down, dtype=np.float64) * size)
        light = unit(view + 0.45 * up - 0.30 * right)
        half = unit(light + view)
        return cls(tuple(view), tuple(light), tuple(half), tuple(size))


def surface_normals(occupancy: np.ndarray, points: np.ndarray, light: SurfaceLight
                    ) -> np.ndarray:
    """Outward unit normals of a mask's surface at points on it.

    Central differences, one voxel each way, of the trilinearly sampled
    occupancy (1 inside an object, 0 outside), scaled to physical units and
    turned to face the viewer. Where the occupancy is flat the normal is the
    direction to the viewer.

    :param occupancy: ``float32`` occupancy volume.
    :param points: ``(axes, n)`` array indices.
    :param light: the camera's :class:`SurfaceLight`.
    :returns: ``(n, axes)`` normals.
    """
    from scipy import ndimage

    points = np.asarray(points, dtype=np.float32)
    axes, count = points.shape
    gradient = np.empty((count, axes), dtype=np.float64)
    for axis in range(axes):
        ahead, behind = points.copy(), points.copy()
        ahead[axis] += 1.0
        behind[axis] -= 1.0
        gradient[:, axis] = (
            ndimage.map_coordinates(occupancy, ahead, order=1, mode="nearest")
            - ndimage.map_coordinates(occupancy, behind, order=1, mode="nearest")
        ) / (2.0 * light.size[axis])
    length = np.linalg.norm(gradient, axis=1)
    view = np.asarray(light.view, dtype=np.float64)
    normal = np.where(length[:, None] > 1e-6, -gradient / np.maximum(length, _EPS)[:, None],
                      view[None, :])
    facing = normal @ view
    return np.where(facing[:, None] < 0, -normal, normal)


def shade_surface(colour: np.ndarray, normal: np.ndarray, light: SurfaceLight,
                  finish: SurfaceFinish) -> np.ndarray:
    """Light surface points of known colour and normal by one finish.

    :param colour: ``(..., 3)`` base colours, 0 to 1.
    :param normal: ``(..., axes)`` unit normals facing the viewer.
    :param light: the camera's :class:`SurfaceLight`.
    :param finish: the :class:`SurfaceFinish`.
    :returns: ``(..., 3)`` lit colour (not clipped).
    """
    colour = np.asarray(colour, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    diffuse = np.clip(normal @ np.asarray(light.light), 0.0, None)
    shine = np.zeros_like(diffuse)
    if finish.specular > 0:
        shine = np.clip(normal @ np.asarray(light.half), 0.0, None) ** finish.shininess
    highlight = finish.tint * colour + (1.0 - finish.tint)
    return (colour * (finish.ambient + finish.diffuse * diffuse)[..., None]
            + finish.specular * shine[..., None] * highlight)


@dataclass(frozen=True)
class VolumeLens:
    """A magnifying cube in a volume, for a 3-D view's magnifier.

    Inside the cube the volume is drawn ``zoom`` times larger about the
    cube's centre; outside it, everything is dimmed to ``dim`` of its
    brightness. Positions are array indices, as for :class:`VolumeRays`.

    :param centre: the cube's centre.
    :param half: half the cube's side along each array axis.
    :param zoom: how many times larger the inside is drawn (at least 1).
    :param dim: brightness kept outside the cube, 0 to 1.
    """

    centre: Tuple[float, ...]
    half: Tuple[float, ...]
    zoom: float = 2.0
    dim: float = 0.35

    def __post_init__(self) -> None:
        """Refuse a cube with no size or a zoom that shrinks."""
        if len(self.centre) != len(self.half) or any(h <= 0 for h in self.half):
            raise LayerError(f"a lens needs a positive half-side per axis, got {self.half!r}")
        if not self.zoom >= 1.0 or not 0.0 <= self.dim <= 1.0:
            raise LayerError(f"a lens needs zoom >= 1 and dim in 0..1, got {self.zoom!r}, {self.dim!r}")

    def contains(self, points: Any) -> np.ndarray:
        """Whether points lie inside the cube.

        :param points: ``(axes, ...)`` array indices.
        :returns: a boolean array of the points' shape without the axes.
        """
        points = np.asarray(points)
        centre = np.asarray(self.centre, dtype=np.float32).reshape((-1,) + (1,) * (points.ndim - 1))
        half = np.asarray(self.half, dtype=np.float32).reshape(centre.shape)
        return np.all(np.abs(points - centre) <= half, axis=0)

    def warp(self, points: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Where each point samples the volume, and how bright it is drawn.

        :param points: ``(axes, ...)`` array indices.
        :returns: ``(where, weight)``: inside the cube a point samples
            ``centre + (point - centre) / zoom`` at weight 1; outside it
            samples itself at weight :attr:`dim`.
        """
        points = np.asarray(points, dtype=np.float32)
        inside = self.contains(points)
        centre = np.asarray(self.centre, dtype=np.float32).reshape((-1,) + (1,) * (points.ndim - 1))
        where = np.where(inside, centre + (points - centre) / np.float32(self.zoom), points)
        return where.astype(np.float32), np.where(inside, 1.0, self.dim).astype(np.float32)

    def corners(self) -> np.ndarray:
        """The cube's eight corners.

        :returns: ``(8, axes)`` array indices.
        """
        centre, half = np.asarray(self.centre, float), np.asarray(self.half, float)
        signs = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        return centre + signs[:, :len(centre)] * half

    def covers(self, rays: "VolumeRays") -> np.ndarray:
        """Which rays pass through the cube between their first and last sample.

        :param rays: the camera's rays.
        :returns: ``(rows, columns)`` booleans; ``False`` pixels are dimmed
            all the way along.
        """
        rows, columns, samples = rays.shape
        r, c = np.mgrid[0:rows, 0:columns].astype(np.float64)
        low = np.zeros((rows, columns))
        high = np.full((rows, columns), float(max(samples - 1, 0)))
        for axis in range(len(self.centre)):
            start = (rays.origin[axis] + c * rays.across[axis] + r * rays.down[axis]
                     - self.centre[axis])
            step = float(rays.depth[axis])
            if abs(step) < _EPS:
                outside = np.abs(start) > self.half[axis]
                high = np.where(outside, -1.0, high)
                continue
            a = (-self.half[axis] - start) / step
            b = (self.half[axis] - start) / step
            low = np.maximum(low, np.minimum(a, b))
            high = np.minimum(high, np.maximum(a, b))
        return low <= high


#: The edges of a cube, as pairs of :meth:`VolumeLens.corners` indices.
CUBE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
    (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7))


@dataclass(frozen=True)
class RenderScale:
    """How finely one 3-D frame is drawn.

    :param percent: the share of full resolution, 1 to 100, along each axis.
    :param max_voxels: the most voxels the frame is drawn from (see
        :func:`volume_steps`), or ``None`` for every voxel.
    :param pixel: display pixels per rendered pixel along each side.
    :param sample: voxels between two samples along a line of sight.
    """

    percent: int
    max_voxels: Optional[int]
    pixel: float
    sample: float


#: The coarsest rendered pixel, in display pixels, at any scale.
RENDER_SCALE_MAX_PIXEL = 8.0

#: The fewest voxels a frame is drawn from, so 1 % still shows a shape.
RENDER_SCALE_MIN_VOXELS = 512

#: Most voxels the CPU draws from while the view is being dragged.
RENDER_DRAG_VOXELS = 262_144

#: Most voxels a GPU texture holds per channel (512 MB of ``float32``).
GPU_MAX_VOXELS = 1 << 27


def render_scale(percent: Any, voxels: int, *, dragging: bool = False,
                 gpu: bool = False, gpu_max_voxels: Optional[int] = GPU_MAX_VOXELS
                 ) -> RenderScale:
    """Turn the Render scale slider into what one frame is drawn from.

    ``percent`` is the share of full resolution along each axis: at 50 %
    every second voxel along X, Y and Z (an eighth of the voxels) is drawn,
    into a picture of half the display's width and height. While the view
    is dragged the frame is drawn coarser still and then again at the
    slider's scale once it stops: the CPU draws from at most
    :data:`RENDER_DRAG_VOXELS` voxels into pixels of at least 4 display
    pixels; the GPU keeps its voxels (so nothing is uploaded again) and
    draws pixels of at least 2.

    :param percent: the slider, 1 to 100 (clipped).
    :param voxels: how many voxels the full volume holds.
    :param dragging: whether the view is moving.
    :param gpu: whether the GPU draws the frame.
    :param gpu_max_voxels: the most voxels one GPU texture may hold.
    :returns: the :class:`RenderScale`.
    """
    percent = int(round(min(100.0, max(1.0, float(percent)))))
    fraction = percent / 100.0
    voxels = max(1, int(voxels))
    budget = None if percent >= 100 else min(voxels, max(RENDER_SCALE_MIN_VOXELS,
                                                         int(voxels * fraction ** 3)))
    if gpu and gpu_max_voxels and (budget or voxels) > gpu_max_voxels:
        budget = int(gpu_max_voxels)
    pixel = min(RENDER_SCALE_MAX_PIXEL, 1.0 / fraction)
    sample = 1.0
    if dragging and gpu:
        pixel = min(RENDER_SCALE_MAX_PIXEL, max(pixel, 2.0))
    elif dragging:
        budget = min(budget or voxels, RENDER_DRAG_VOXELS)
        pixel = min(RENDER_SCALE_MAX_PIXEL, max(pixel, 4.0))
        sample = 1.5
    return RenderScale(percent, budget, float(pixel), sample)


def render_volume(layers: Iterable["Layer"], rays: VolumeRays, *,
                  highlight: int = 0,
                  cancelled: Optional[Callable[[], bool]] = None,
                  chunk: int = 300_000,
                  voxel_size: Optional[Sequence[float]] = None,
                  lens: Optional[VolumeLens] = None
                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Draw 3-D image and labels layers along parallel rays, as napari does.

    Image layers are added together, each through its colormap, between its
    contrast limits, times its opacity; each is drawn by the
    :data:`VOLUME_RENDERINGS` mode in ``layer.metadata["rendering"]``
    (``mip`` when absent). Labels layers are then laid over them in stack
    order, each object in its :func:`label_color`, by the
    :data:`LABEL_RENDERINGS` mode (``surface`` when absent) and the layer's
    opacity. A surface is lit by the :data:`SURFACE_FINISHES` entry in
    ``layer.metadata["finish"]`` (``matte`` when absent), with normals from
    the mask itself (:func:`surface_normals`). Hidden layers draw nothing.
    Every layer must have the volume's shape; image layers draw their first
    channel. :class:`GLVolumeRenderer` draws the same picture on the GPU.

    The rays are walked a few rows at a time, so a caller on a worker thread
    can stop a render that is no longer wanted.

    :param layers: the layers, bottom first.
    :param rays: where to sample, in the layers' array indices.
    :param highlight: an object drawn lighter, as a chosen object is.
    :param cancelled: asked between chunks; returning ``True`` abandons the
        render.
    :param chunk: about how many samples one step holds.
    :param voxel_size: physical length of one voxel along each array axis,
        for lighting; ``None`` for cubic voxels.
    :param lens: a magnifier cube (:class:`VolumeLens`), or ``None``.
    :returns: ``(rgb, picked)``: ``rows x columns x 3`` float32 colour
        between 0 and 1, and the object nearest the viewer under each pixel
        (0 for none), or ``None`` when cancelled.
    :raises LayerError: for a rendering or finish no layer of that kind has,
        or a layer whose shape differs from the others'.
    """
    from scipy import ndimage

    rows, columns, samples = (int(value) for value in rays.shape)
    drawn = [layer for layer in layers if layer.visible
             and isinstance(layer, (ImageLayer, LabelsLayer))]
    shapes = {layer.shape if isinstance(layer, LabelsLayer) else layer.channel_data(0).shape
              for layer in drawn}
    if len(shapes) > 1:
        raise LayerError(f"3-D layers must share one shape, got {sorted(shapes)}")
    modes = [_layer_rendering(layer) for layer in drawn]
    finishes = [surface_finish(layer.metadata.get("finish")) if isinstance(layer, LabelsLayer)
                else None for layer in drawn]
    length = max(rays.sample_length, 1e-6)
    axes = len(rays.origin)
    images = [np.zeros((rows, columns), dtype=np.float32) for _ in drawn]
    nearest = np.zeros((rows, columns), dtype=np.int64)
    first_hit = np.full((rows, columns), np.nan, dtype=np.float32)
    glass = {index: np.zeros((rows, columns, 4), dtype=np.float32)
             for index, layer in enumerate(drawn)
             if isinstance(layer, LabelsLayer) and modes[index] == "translucent"}
    surfaces = {index: SimpleNamespace(
        ids=np.zeros((rows, columns), dtype=np.int64),
        depth=np.full((rows, columns), np.nan, dtype=np.float32),
        where=np.zeros((axes, rows, columns), dtype=np.float32),
        weight=np.ones((rows, columns), dtype=np.float32))
        for index, layer in enumerate(drawn)
        if isinstance(layer, LabelsLayer) and modes[index] == "surface"}
    shape = next(iter(shapes)) if shapes else ()
    step = max(1, int(chunk) // max(1, columns * samples))
    for first in range(0, rows, step):
        if cancelled is not None and cancelled():
            return None
        last = min(rows, first + step)
        where = rays.positions(first, last)
        weight = None
        if lens is not None:
            where, weight = lens.warp(where)
        flat = where.reshape(axes, -1)
        inside = np.ones(flat.shape[1], dtype=bool)
        for axis, size in enumerate(shape):
            inside &= (flat[axis] >= -0.5) & (flat[axis] <= size - 0.5)
        inside = inside.reshape(where.shape[1:])
        for index, layer in enumerate(drawn):
            if isinstance(layer, ImageLayer):
                low, high = layer.contrast_limits(0)
                values = ndimage.map_coordinates(
                    np.asarray(layer.channel_data(0), dtype=np.float32), flat,
                    order=1, mode="nearest").reshape(inside.shape)
                values = np.clip((values - low) / max(high - low, _EPS), 0.0, 1.0) * inside
                if weight is not None:
                    values = values * weight
                images[index][first:last] = _reduce_image(
                    values, inside, modes[index], length).reshape(last - first, columns)
                continue
            labels = ndimage.map_coordinates(
                np.asarray(layer.data), flat, order=0, mode="nearest"
            ).reshape(inside.shape).astype(np.int64) * inside
            hit = labels != 0
            seen = hit.any(axis=1)
            depth = hit.argmax(axis=1)
            ray_index = np.arange(labels.shape[0])
            ids = labels[ray_index, depth] * seen
            block = slice(first, last)
            older = nearest[block].reshape(-1)
            old_depth = first_hit[block].reshape(-1)
            closer = seen & ~(old_depth <= depth)
            nearest[block] = np.where(closer, ids, older).reshape(last - first, columns)
            first_hit[block] = np.where(closer, depth, old_depth).reshape(last - first, columns)
            if index in surfaces:
                surface = surfaces[index]
                surface.ids[block] = ids.reshape(last - first, columns)
                surface.depth[block] = np.where(seen, depth, np.nan).reshape(last - first, columns)
                surface.where[:, block] = where[:, ray_index, depth].reshape(axes, last - first, columns)
                if weight is not None:
                    surface.weight[block] = weight[ray_index, depth].reshape(last - first, columns)
                continue
            colours = label_colors(labels).astype(np.float32)
            if weight is not None:
                colours = colours * weight[..., None]
            alpha = np.where(hit, min(1.0, LABEL_DENSITY * length), 0.0).astype(np.float32)
            clear = np.cumprod(1.0 - alpha, axis=1)
            before = np.concatenate([np.ones_like(clear[:, :1]), clear[:, :-1]], axis=1)
            share = before * alpha
            rgb = (share[..., None] * colours).sum(axis=1)
            glass[index][block] = np.concatenate(
                [rgb, share.sum(axis=1)[:, None]], axis=1).reshape(last - first, columns, 4)
    out = np.zeros((rows, columns, 3), dtype=np.float32)
    for index, layer in enumerate(drawn):
        if isinstance(layer, ImageLayer):
            out += layer.colormap.map(images[index]) * float(layer.opacity)
    np.clip(out, 0.0, 1.0, out=out)
    light = SurfaceLight.for_rays(rays, voxel_size) if surfaces else None
    for index, layer in enumerate(drawn):
        if index in glass:
            rgba = glass[index]
            alpha = rgba[..., 3:] * float(layer.opacity)
            colour = rgba[..., :3] / np.maximum(rgba[..., 3:], _EPS)
            if highlight:
                colour = np.where((nearest == highlight)[..., None],
                                  0.5 * colour + 0.5, colour)
            out = out * (1.0 - alpha) + colour * alpha
        elif index in surfaces:
            surface = surfaces[index]
            seen = surface.ids != 0
            if not seen.any():
                continue
            occupancy = (np.asarray(layer.data) != 0).astype(np.float32)
            normals = surface_normals(occupancy, surface.where[:, seen], light)
            ids = surface.ids[seen]
            lit = shade_surface(label_colors(ids), normals, light, finishes[index])
            cue = 1.0 - 0.35 * surface.depth[seen] / max(samples, 1)
            lit = np.clip(lit * cue[:, None], 0.0, 1.0)
            if highlight:
                lit = np.where((ids == highlight)[:, None], 0.5 * lit + 0.5, lit)
            lit = lit * surface.weight[seen][:, None]
            colour = np.zeros((rows, columns, 3), dtype=np.float32)
            colour[seen] = lit
            alpha = (seen * float(layer.opacity))[..., None]
            out = out * (1.0 - alpha) + colour * alpha
    return np.clip(out, 0.0, 1.0).astype(np.float32), nearest


_GL_VERTEX = """
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    v_uv = (a_position + 1.0) / 2.0;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_GL_RAYS = """
uniform vec3 u_origin;
uniform vec3 u_across;
uniform vec3 u_down;
uniform vec3 u_depth;
uniform vec3 u_shape;
uniform float u_rows;
uniform int u_samples;
uniform vec3 u_lens_centre;
uniform vec3 u_lens_half;
uniform float u_lens_zoom;
uniform float u_lens_dim;
const int MAX_SAMPLES = 2048;

vec3 ray_start() {
    float column = gl_FragCoord.x - 0.5;
    float row = u_rows - gl_FragCoord.y - 0.5;
    return u_origin + column * u_across + row * u_down;
}

vec3 warp(vec3 p, out float weight) {
    weight = 1.0;
    if (u_lens_zoom <= 0.0) return p;
    vec3 offset = p - u_lens_centre;
    if (all(lessThanEqual(abs(offset), u_lens_half))) return u_lens_centre + offset / u_lens_zoom;
    weight = u_lens_dim;
    return p;
}

bool inside(vec3 p) {
    return all(greaterThanEqual(p, vec3(-0.5))) && all(lessThanEqual(p, u_shape - 0.5));
}

vec3 texcoord(vec3 p) {
    return vec3((p.z + 0.5) / u_shape.z, (p.y + 0.5) / u_shape.y, (p.x + 0.5) / u_shape.x);
}
"""

_GL_IMAGE = """
uniform sampler3D u_volume;
uniform sampler2D u_lut;
uniform int u_mode;
uniform float u_low;
uniform float u_high;
uniform float u_opacity;
uniform float u_length;
uniform float u_attenuation;
uniform float u_density;

void main() {
    vec3 start = ray_start();
    float best = 0.0;
    float least = 1.0;
    float seen = 0.0;
    float total = 0.0;
    float count = 0.0;
    float crossed = 0.0;
    float clear = 1.0;
    float glow = 0.0;
    float span = max(u_high - u_low, 1e-12);
    for (int k = 0; k < MAX_SAMPLES; k++) {
        if (k >= u_samples) break;
        float weight;
        vec3 p = warp(start + float(k) * u_depth, weight);
        float v = 0.0;
        bool within = inside(p);
        if (within) v = clamp((texture3D(u_volume, texcoord(p)).r - u_low) / span, 0.0, 1.0) * weight;
        if (u_mode == 0) {
            best = max(best, v);
        } else if (u_mode == 1) {
            best = max(best, v * exp(-u_attenuation * u_length * crossed));
            crossed += v;
        } else if (u_mode == 2) {
            float a = clamp(v * u_density * u_length, 0.0, 1.0);
            glow += clear * a;
            clear *= 1.0 - a;
        } else if (u_mode == 3) {
            total += v;
            if (within) count += 1.0;
        } else if (within) {
            least = min(least, v);
            seen = 1.0;
        }
    }
    float value = best;
    if (u_mode == 2) value = min(glow * 2.0, 1.0);
    else if (u_mode == 3) value = total / max(count, 1.0);
    else if (u_mode == 4) value = seen > 0.5 ? least : 0.0;
    vec3 colour = texture2D(u_lut, vec2((value * 255.0 + 0.5) / 256.0, 0.5)).rgb;
    gl_FragColor = vec4(colour * u_opacity, 1.0);
}
"""

_GL_LABELS = """
uniform sampler3D u_ids;
uniform sampler3D u_occupancy;
uniform sampler2D u_colours;
uniform vec2 u_colours_size;
uniform int u_mode;
uniform float u_highlight;
uniform float u_opacity;
uniform float u_alpha;
uniform vec3 u_size;
uniform vec3 u_view;
uniform vec3 u_light;
uniform vec3 u_half;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform float u_tint;

float object_at(vec3 p) {
    return floor(texture3D(u_ids, texcoord(p)).r + 0.5);
}

vec3 colour_of(float id) {
    float x = mod(id, u_colours_size.x);
    float y = floor(id / u_colours_size.x);
    return texture2D(u_colours, vec2((x + 0.5) / u_colours_size.x,
                                     (y + 0.5) / u_colours_size.y)).rgb;
}

float occupancy(vec3 p) {
    return texture3D(u_occupancy, texcoord(p)).r;
}

void main() {
    vec3 start = ray_start();
    if (u_mode == 1) {
        vec3 rgb = vec3(0.0);
        float share = 0.0;
        float clear = 1.0;
        float first = 0.0;
        for (int k = 0; k < MAX_SAMPLES; k++) {
            if (k >= u_samples) break;
            float weight;
            vec3 p = warp(start + float(k) * u_depth, weight);
            if (!inside(p)) continue;
            float id = object_at(p);
            if (id < 0.5) continue;
            if (first < 0.5) first = id;
            rgb += clear * u_alpha * colour_of(id) * weight;
            share += clear * u_alpha;
            clear *= 1.0 - u_alpha;
        }
        if (share <= 0.0) discard;
        vec3 colour = rgb / share;
        if (u_highlight > 0.5 && abs(first - u_highlight) < 0.5) colour = 0.5 * colour + 0.5;
        gl_FragColor = vec4(colour, share * u_opacity);
        return;
    }
    float id = 0.0;
    float depth = 0.0;
    float weight = 1.0;
    vec3 hit = vec3(0.0);
    for (int k = 0; k < MAX_SAMPLES; k++) {
        if (k >= u_samples) break;
        float w;
        vec3 p = warp(start + float(k) * u_depth, w);
        if (!inside(p)) continue;
        float here = object_at(p);
        if (here > 0.5) {
            id = here;
            depth = float(k);
            weight = w;
            hit = p;
            break;
        }
    }
    if (id < 0.5) discard;
    if (u_mode == 2) {
        float red = mod(id, 256.0);
        float green = mod(floor(id / 256.0), 256.0);
        float blue = floor(id / 65536.0);
        gl_FragColor = vec4(red / 255.0, green / 255.0, blue / 255.0, 1.0);
        gl_FragDepth = (depth + 0.5) / float(MAX_SAMPLES);
        return;
    }
    vec3 g = vec3(occupancy(hit + vec3(1.0, 0.0, 0.0)) - occupancy(hit - vec3(1.0, 0.0, 0.0)),
                  occupancy(hit + vec3(0.0, 1.0, 0.0)) - occupancy(hit - vec3(0.0, 1.0, 0.0)),
                  occupancy(hit + vec3(0.0, 0.0, 1.0)) - occupancy(hit - vec3(0.0, 0.0, 1.0)))
             / (2.0 * u_size);
    vec3 n = length(g) > 1e-6 ? -normalize(g) : u_view;
    if (dot(n, u_view) < 0.0) n = -n;
    vec3 base = colour_of(id);
    float diffuse = max(dot(n, u_light), 0.0);
    float shine = u_specular > 0.0 ? pow(max(dot(n, u_half), 0.0), u_shininess) : 0.0;
    vec3 colour = base * (u_ambient + u_diffuse * diffuse)
                  + u_specular * shine * mix(vec3(1.0), base, u_tint);
    colour = clamp(colour * (1.0 - 0.35 * depth / float(u_samples)), 0.0, 1.0);
    if (u_highlight > 0.5 && abs(id - u_highlight) < 0.5) colour = 0.5 * colour + 0.5;
    gl_FragColor = vec4(colour * weight, u_opacity);
}
"""

_GL_COPY = """
uniform sampler2D u_image;
varying vec2 v_uv;
void main() {
    gl_FragColor = vec4(texture2D(u_image, v_uv).rgb, 1.0);
}
"""

#: The image renderings by the number the GPU shader knows them by.
_GL_IMAGE_MODES = {"mip": 0, "attenuated_mip": 1, "translucent": 2, "average": 3, "minip": 4}

#: Width of the texture of object colours the GPU looks an object ID up in.
_GL_COLOUR_WIDTH = 4096


def _array_owner(array: Any) -> Any:
    """The array that owns the memory a view looks at (itself if none)."""
    owner = array
    while isinstance(getattr(owner, "base", None), np.ndarray):
        owner = owner.base
    return owner


def _array_key(array: Any) -> Tuple[Any, ...]:
    """What identifies a volume's voxels: its memory, shape, strides and type.

    Two views built over the same cached array, frame after frame, have
    one key, so their texture is uploaded once.
    """
    data = np.asarray(array)
    return (id(_array_owner(data)), data.__array_interface__["data"][0], data.shape,
            data.strides, data.dtype.str)


class GLVolumeRenderer:
    """:func:`render_volume`, ray-marched on the GPU through OpenGL (VisPy gloo).

    The same rays (:class:`VolumeRays`), the same layer model, renderings,
    finishes and lens, so a frame drawn here matches the CPU's to within
    8-bit rounding and picks the same objects. Each fragment walks one ray;
    image layers add up, labels layers are laid over them, and a last pass
    writes the object nearest the viewer under each pixel for picking.

    The caller owns the OpenGL context and makes it current around every
    call; this class creates nothing until the first :meth:`render`. A
    volume is uploaded once and kept, keyed by its memory, for as long as
    every frame draws it; a frame frees the textures it did not use. A mask
    edited in place must therefore be handed over as a new array.
    """

    def __init__(self) -> None:
        """Hold no GPU object until the first frame."""
        self._canvas = None
        self._programs: Dict[str, Any] = {}
        self._textures: Dict[Tuple[int, str], Tuple[Any, Any]] = {}
        self._targets: Dict[Tuple[int, int, str], Any] = {}
        self._luts: Dict[Any, Any] = {}
        self._used: set = set()

    def _gloo(self):
        """Make this renderer's GLIR queue the current one; return ``gloo``.

        Only when it is not current already: making it current again in the
        middle of a frame would rebind the default framebuffer at the next
        draw, and the rest of the frame would be drawn off the picture.
        """
        from vispy import gloo
        from vispy.gloo.context import FakeCanvas, get_current_canvas, set_current_canvas

        if self._canvas is None:
            self._canvas = FakeCanvas()
        elif get_current_canvas() is not self._canvas:
            set_current_canvas(self._canvas)
        return gloo

    def _release(self) -> None:
        """Run the queued commands and stop being VisPy's current canvas."""
        from vispy.gloo.context import forget_canvas

        self._canvas.flush()
        forget_canvas(self._canvas)

    def limits(self) -> Dict[str, Any]:
        """The context's renderer name and its largest 3-D texture side.

        :returns: ``{"renderer": str, "version": str, "max_3d": int}``.
        """
        from vispy.gloo import gl

        self._gloo()
        try:
            return {"renderer": str(gl.glGetParameter(gl.GL_RENDERER)),
                    "version": str(gl.glGetParameter(gl.GL_VERSION)),
                    "max_3d": int(gl.glGetParameter(0x8073))}
        finally:
            self._release()

    def _program(self, name: str):
        """One compiled program, built the first time it is asked for."""
        gloo = self._gloo()
        program = self._programs.get(name)
        if program is None:
            fragment = {"image": _GL_RAYS + _GL_IMAGE, "labels": _GL_RAYS + _GL_LABELS,
                        "copy": _GL_COPY}[name]
            program = gloo.Program(_GL_VERTEX, fragment)
            program["a_position"] = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32)
            self._programs[name] = program
        return program

    def _texture(self, array: np.ndarray, kind: str):
        """The GPU copy of one volume (``image``, ``ids`` or ``occupancy``)."""
        gloo = self._gloo()
        key = (_array_key(array), kind)
        held = self._textures.get(key)
        if held is not None and held[0] is _array_owner(array):
            self._used.add(key)
            return held[1]
        if kind == "occupancy":
            data = (np.asarray(array) != 0).astype(np.float32)
            texture = gloo.Texture3D(data, interpolation="linear", wrapping="clamp_to_edge",
                                     internalformat="r32f")
        else:
            data = np.ascontiguousarray(np.asarray(array), dtype=np.float32)
            texture = gloo.Texture3D(data, interpolation="nearest" if kind == "ids" else "linear",
                                     wrapping="clamp_to_edge", internalformat="r32f")
        self._textures[key] = (_array_owner(array), texture)
        self._used.add(key)
        return texture

    def _forget_unused(self) -> None:
        """Free the textures the frame just drawn did not use."""
        for key in [key for key in self._textures if key not in self._used]:
            self._textures.pop(key)[1].delete()
        self._used = set()

    def _target(self, rows: int, columns: int, kind: str):
        """A framebuffer of one size: ``frame`` (sampled later), ``pick`` or ``out``."""
        gloo = self._gloo()
        key = (rows, columns, kind)
        target = self._targets.get(key)
        if target is None:
            for old in [old for old in self._targets if old[2] == kind]:
                self._targets.pop(old).delete()
            if kind == "frame":
                colour = gloo.Texture2D(shape=(rows, columns, 4), interpolation="linear",
                                        wrapping="clamp_to_edge")
            else:
                colour = gloo.RenderBuffer((rows, columns, 4))
            depth = gloo.RenderBuffer((rows, columns), format="depth") if kind == "pick" else None
            target = gloo.FrameBuffer(color=colour, depth=depth)
            self._targets[key] = target
        return target

    @staticmethod
    def _ray_uniforms(program, rays: VolumeRays, shape: Sequence[int],
                      lens: Optional[VolumeLens]) -> None:
        """Hand one camera, volume shape and lens to a program."""
        program["u_origin"] = tuple(float(v) for v in rays.origin)
        program["u_across"] = tuple(float(v) for v in rays.across)
        program["u_down"] = tuple(float(v) for v in rays.down)
        program["u_depth"] = tuple(float(v) for v in rays.depth)
        program["u_shape"] = tuple(float(v) for v in shape)
        program["u_rows"] = float(rays.shape[0])
        program["u_samples"] = int(rays.shape[2])
        if lens is None:
            program["u_lens_centre"] = (0.0, 0.0, 0.0)
            program["u_lens_half"] = (1.0, 1.0, 1.0)
            program["u_lens_zoom"] = 0.0
            program["u_lens_dim"] = 1.0
        else:
            program["u_lens_centre"] = tuple(float(v) for v in lens.centre)
            program["u_lens_half"] = tuple(float(v) for v in lens.half)
            program["u_lens_zoom"] = float(lens.zoom)
            program["u_lens_dim"] = float(lens.dim)

    def _lut(self, colormap: "Colormap"):
        """A 256-entry texture of one colormap, kept for the next frame."""
        gloo = self._gloo()
        texture = self._luts.get(colormap)
        if texture is None:
            lut = np.ones((1, 256, 4), dtype=np.float32)
            lut[0, :, :3] = colormap.map(np.linspace(0.0, 1.0, 256))
            texture = gloo.Texture2D(lut, interpolation="linear", wrapping="clamp_to_edge",
                                     internalformat="rgba32f")
            self._luts[colormap] = texture
        return texture

    def _colours(self, labels: np.ndarray):
        """A texture of every object's colour, indexed by object ID."""
        gloo = self._gloo()
        key = (_array_key(labels), "colours")
        held = self._textures.get(key)
        if held is not None and held[0] is _array_owner(labels):
            self._used.add(key)
            return held[1]
        present = np.unique(np.asarray(labels))
        top = int(present.max()) if present.size else 0
        if top >= 1 << 24:
            raise LayerError("object IDs of 2**24 and above cannot be drawn on the GPU")
        height = top // _GL_COLOUR_WIDTH + 1
        table = np.zeros((height * _GL_COLOUR_WIDTH, 4), dtype=np.float32)
        table[present, :3] = label_colors(present)
        table[:, 3] = 1.0
        texture = gloo.Texture2D(table.reshape(height, _GL_COLOUR_WIDTH, 4),
                                 interpolation="nearest", internalformat="rgba32f")
        self._textures[key] = (_array_owner(labels), texture)
        self._used.add(key)
        return texture

    def render(self, layers: Iterable["Layer"], rays: VolumeRays, *,
               highlight: int = 0, voxel_size: Optional[Sequence[float]] = None,
               lens: Optional[VolumeLens] = None,
               size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Draw one frame; the same picture as :func:`render_volume`.

        :param layers: the layers, bottom first.
        :param rays: where to sample, in the layers' array indices.
        :param highlight: an object drawn lighter.
        :param voxel_size: physical voxel length along each array axis.
        :param lens: a magnifier cube, or ``None``.
        :param size: ``(rows, columns)`` of the picture returned, scaled up
            from the rays' grid on the GPU; ``None`` keeps the rays' grid.
        :returns: ``(rgb, picked)``: ``uint8`` colour of ``size`` and the
            object nearest the viewer under each ray (``int64``, the rays'
            grid).
        :raises LayerError: as :func:`render_volume`.
        """
        rows, columns, _samples = (int(value) for value in rays.shape)
        drawn = [layer for layer in layers if layer.visible
                 and isinstance(layer, (ImageLayer, LabelsLayer))]
        shapes = {layer.shape if isinstance(layer, LabelsLayer) else layer.channel_data(0).shape
                  for layer in drawn}
        if len(shapes) > 1:
            raise LayerError(f"3-D layers must share one shape, got {sorted(shapes)}")
        if any(len(shape) != 3 for shape in shapes):
            raise LayerError("the GPU draws 3-D layers only")
        modes = [_layer_rendering(layer) for layer in drawn]
        finishes = [surface_finish(layer.metadata.get("finish")) if isinstance(layer, LabelsLayer)
                    else None for layer in drawn]
        light = SurfaceLight.for_rays(rays, voxel_size)
        length = max(rays.sample_length, 1e-6)
        gloo = self._gloo()
        try:
            frame = self._target(rows, columns, "frame")
            with frame:
                gloo.set_viewport(0, 0, columns, rows)
                gloo.set_state(depth_test=False, cull_face=False, blend=True,
                               blend_func=("one", "one"))
                gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                for index, layer in enumerate(drawn):
                    if not isinstance(layer, ImageLayer):
                        continue
                    program = self._program("image")
                    data = layer.channel_data(0)
                    self._ray_uniforms(program, rays, data.shape, lens)
                    low, high = layer.contrast_limits(0)
                    program["u_volume"] = self._texture(data, "image")
                    program["u_lut"] = self._lut(layer.colormap)
                    program["u_mode"] = _GL_IMAGE_MODES[modes[index]]
                    program["u_low"], program["u_high"] = float(low), float(high)
                    program["u_opacity"] = float(layer.opacity)
                    program["u_length"] = float(length)
                    program["u_attenuation"] = float(VOLUME_ATTENUATION)
                    program["u_density"] = float(VOLUME_DENSITY)
                    program.draw("triangle_strip")
                gloo.set_state(blend_func=("src_alpha", "one_minus_src_alpha"))
                for index, layer in enumerate(drawn):
                    if isinstance(layer, LabelsLayer):
                        program = self._labels_program(layer, rays, lens, light,
                                                       finishes[index], highlight)
                        program["u_mode"] = 0 if modes[index] == "surface" else 1
                        program["u_opacity"] = float(layer.opacity)
                        program["u_alpha"] = float(min(1.0, LABEL_DENSITY * length))
                        program.draw("triangle_strip")
            pick = self._target(rows, columns, "pick")
            with pick:
                gloo.set_viewport(0, 0, columns, rows)
                gloo.set_state(blend=False, depth_test=True, depth_func="less")
                gloo.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
                for index, layer in enumerate(drawn):
                    if isinstance(layer, LabelsLayer):
                        program = self._labels_program(layer, rays, lens, light,
                                                       finishes[index], highlight)
                        program["u_mode"] = 2
                        program.draw("triangle_strip")
                codes = pick.read("color", alpha=False).astype(np.int64)
                gloo.set_state(depth_test=False)
            picked = codes[..., 0] + 256 * codes[..., 1] + 65536 * codes[..., 2]
            if size is None or tuple(size) == (rows, columns):
                with frame:
                    rgb = frame.read("color", alpha=False)
            else:
                height, width = (max(1, int(v)) for v in size)
                out = self._target(height, width, "out")
                program = self._program("copy")
                program["u_image"] = frame.color_buffer
                with out:
                    gloo.set_viewport(0, 0, width, height)
                    gloo.set_state(blend=False, depth_test=False)
                    gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
                    program.draw("triangle_strip")
                    rgb = out.read("color", alpha=False)
            self._forget_unused()
        finally:
            self._release()
        return np.ascontiguousarray(rgb), picked

    def _labels_program(self, layer: "LabelsLayer", rays: VolumeRays,
                        lens: Optional[VolumeLens], light: SurfaceLight,
                        finish: SurfaceFinish, highlight: int):
        """The labels program set up for one layer, camera and finish."""
        program = self._program("labels")
        data = np.asarray(layer.data)
        self._ray_uniforms(program, rays, data.shape, lens)
        colours = self._colours(layer.data)
        program["u_ids"] = self._texture(layer.data, "ids")
        program["u_occupancy"] = self._texture(layer.data, "occupancy")
        program["u_colours"] = colours
        program["u_colours_size"] = (float(_GL_COLOUR_WIDTH), float(colours.shape[0]))
        program["u_highlight"] = float(highlight)
        program["u_size"] = tuple(float(v) for v in light.size)
        program["u_view"] = tuple(float(v) for v in light.view)
        program["u_light"] = tuple(float(v) for v in light.light)
        program["u_half"] = tuple(float(v) for v in light.half)
        program["u_ambient"] = float(finish.ambient)
        program["u_diffuse"] = float(finish.diffuse)
        program["u_specular"] = float(finish.specular)
        program["u_shininess"] = float(finish.shininess)
        program["u_tint"] = float(finish.tint)
        program["u_opacity"] = float(layer.opacity)
        program["u_alpha"] = float(min(1.0, LABEL_DENSITY * max(rays.sample_length, 1e-6)))
        return program

    def close(self) -> None:
        """Free every texture, framebuffer and program (context current)."""
        if self._canvas is None:
            return
        self._gloo()
        try:
            for _array, texture in self._textures.values():
                texture.delete()
            for target in self._targets.values():
                target.delete()
            for other in (*self._luts.values(), *self._programs.values()):
                other.delete()
        finally:
            self._release()
        self._textures, self._targets, self._programs, self._luts = {}, {}, {}, {}


#: The wgpu backends the True 3D view draws with: the view's key, and the
#: ``backend_type`` wgpu reports for an adapter of it. wgpu's own OpenGL
#: adapters are left out; OpenGL is :class:`GLVolumeRenderer`'s.
WGPU_BACKENDS: Mapping[str, str] = MappingProxyType({
    "vulkan": "vulkan", "metal": "metal", "d3d12": "d3d12"})

#: How an adapter of each wgpu ``adapter_type`` ranks when one is chosen;
#: ``CPU`` (lavapipe, WARP) is a software rasteriser.
_WGPU_ADAPTER_RANK = {"discretegpu": 0, "integratedgpu": 1, "virtualgpu": 2, "cpu": 4}

#: Set to anything but ``0`` or empty, no 3-D view draws on the GPU. An
#: empty ``CUDA_VISIBLE_DEVICES`` does the same, so a computer (or a test
#: run) that hides its GPU from spaCR keeps it hidden from the 3-D view too.
GPU_RENDER_OFF_ENV = "SPACR_NO_GPU_RENDER"

#: The adapters :func:`wgpu_adapters` found, once it has looked.
_WGPU_FOUND: Optional[Tuple[Dict[str, str], ...]] = None

_WGSL_COMMON = """
struct Params { v: array<vec4<f32>, 17> };
@group(0) @binding(0) var<uniform> P: Params;

struct Warped { p: vec3<f32>, weight: f32 };

fn ray_start(column: u32, row: u32) -> vec3<f32> {
    return P.v[0].xyz + f32(column) * P.v[1].xyz + f32(row) * P.v[2].xyz;
}

fn warp(p: vec3<f32>) -> Warped {
    if (P.v[8].x <= 0.0) { return Warped(p, 1.0); }
    let offset = p - P.v[6].xyz;
    if (all(abs(offset) <= P.v[7].xyz)) { return Warped(P.v[6].xyz + offset / P.v[8].x, 1.0); }
    return Warped(p, P.v[8].y);
}

fn inside(p: vec3<f32>) -> bool {
    return all(p >= vec3<f32>(-0.5)) && all(p <= P.v[4].xyz - vec3<f32>(0.5));
}

fn last_voxel() -> vec3<i32> {
    return vec3<i32>(P.v[4].xyz) - vec3<i32>(1);
}

fn corner_weight(t: vec3<f32>, corner: u32) -> f32 {
    let o = vec3<u32>(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
    let w = select(vec3<f32>(1.0) - t, t, o == vec3<u32>(1u));
    return w.x * w.y * w.z;
}

fn corner_offset(corner: u32) -> vec3<i32> {
    return vec3<i32>(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u));
}
"""

_WGSL_CLEAR = """
@group(0) @binding(1) var<storage, read_write> frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pick: array<vec2<u32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = u32(P.v[5].x);
    let columns = u32(P.v[5].y);
    if (gid.x >= columns || gid.y >= rows) { return; }
    let index = gid.y * columns + gid.x;
    frame[index] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    pick[index] = vec2<u32>(0u, 0xffffffffu);
}
"""

_WGSL_IMAGE = """
@group(0) @binding(1) var<storage, read_write> frame: array<vec4<f32>>;
@group(0) @binding(2) var volume: texture_3d<u32>;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>>;

fn voxel(i: vec3<i32>) -> f32 {
    let c = clamp(i, vec3<i32>(0), last_voxel());
    return bitcast<f32>(textureLoad(volume, vec3<i32>(c.z, c.y, c.x), 0).r);
}

fn trilinear(p: vec3<f32>) -> f32 {
    let f = floor(p);
    let t = p - f;
    let b = vec3<i32>(f);
    var total = 0.0;
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        total = total + corner_weight(t, corner) * voxel(b + corner_offset(corner));
    }
    return total;
}

fn colour(value: f32) -> vec3<f32> {
    let t = clamp(value, 0.0, 1.0) * 255.0;
    let i = min(u32(floor(t)), 254u);
    return mix(lut[i].rgb, lut[i + 1u].rgb, t - f32(i));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = u32(P.v[5].x);
    let columns = u32(P.v[5].y);
    if (gid.x >= columns || gid.y >= rows) { return; }
    let samples = u32(P.v[5].z);
    let mode = u32(P.v[5].w);
    let low = P.v[9].x;
    let span = max(P.v[9].y - P.v[9].x, 1e-12);
    let opacity = P.v[9].z;
    let step_length = P.v[9].w;
    let attenuation = P.v[10].x;
    let density = P.v[10].y;
    let start = ray_start(gid.x, gid.y);
    var best = 0.0;
    var least = 1.0;
    var seen = false;
    var total = 0.0;
    var count = 0.0;
    var crossed = 0.0;
    var transparency = 1.0;
    var glow = 0.0;
    for (var k = 0u; k < samples; k = k + 1u) {
        let w = warp(start + f32(k) * P.v[3].xyz);
        let within = inside(w.p);
        var v = 0.0;
        if (within) { v = clamp((trilinear(w.p) - low) / span, 0.0, 1.0) * w.weight; }
        if (mode == 0u) {
            best = max(best, v);
        } else if (mode == 1u) {
            best = max(best, v * exp(-attenuation * step_length * crossed));
            crossed = crossed + v;
        } else if (mode == 2u) {
            let a = clamp(v * density * step_length, 0.0, 1.0);
            glow = glow + transparency * a;
            transparency = transparency * (1.0 - a);
        } else if (mode == 3u) {
            total = total + v;
            if (within) { count = count + 1.0; }
        } else if (within) {
            least = min(least, v);
            seen = true;
        }
    }
    var value = best;
    if (mode == 2u) { value = min(glow * 2.0, 1.0); }
    else if (mode == 3u) { value = total / max(count, 1.0); }
    else if (mode == 4u) { value = select(0.0, least, seen); }
    let index = gid.y * columns + gid.x;
    frame[index] = vec4<f32>(frame[index].rgb + colour(value) * opacity, 1.0);
}
"""

_WGSL_LABELS = """
@group(0) @binding(1) var<storage, read_write> frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pick: array<vec2<u32>>;
@group(0) @binding(3) var ids: texture_3d<u32>;
@group(0) @binding(4) var<storage, read> colours: array<vec4<f32>>;

fn id_at(p: vec3<f32>) -> u32 {
    let c = clamp(vec3<i32>(floor(p + vec3<f32>(0.5))), vec3<i32>(0), last_voxel());
    return textureLoad(ids, vec3<i32>(c.z, c.y, c.x), 0).r;
}

fn occupied(i: vec3<i32>) -> f32 {
    let c = clamp(i, vec3<i32>(0), last_voxel());
    return select(0.0, 1.0, textureLoad(ids, vec3<i32>(c.z, c.y, c.x), 0).r != 0u);
}

fn occupancy(p: vec3<f32>) -> f32 {
    let f = floor(p);
    let t = p - f;
    let b = vec3<i32>(f);
    var total = 0.0;
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        total = total + corner_weight(t, corner) * occupied(b + corner_offset(corner));
    }
    return total;
}

fn colour_of(id: u32) -> vec3<f32> {
    return colours[min(id, arrayLength(&colours) - 1u)].rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = u32(P.v[5].x);
    let columns = u32(P.v[5].y);
    if (gid.x >= columns || gid.y >= rows) { return; }
    let samples = u32(P.v[5].z);
    let mode = u32(P.v[5].w);
    let opacity = P.v[9].z;
    let alpha = P.v[10].z;
    let highlight = u32(P.v[10].w + 0.5);
    let index = gid.y * columns + gid.x;
    let start = ray_start(gid.x, gid.y);
    var base = clamp(frame[index].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var first = 0u;
    var first_k = 0u;
    var first_weight = 1.0;
    var hit = vec3<f32>(0.0);
    var rgb = vec3<f32>(0.0);
    var share = 0.0;
    var transparency = 1.0;
    for (var k = 0u; k < samples; k = k + 1u) {
        let w = warp(start + f32(k) * P.v[3].xyz);
        if (!inside(w.p)) { continue; }
        let id = id_at(w.p);
        if (id == 0u) { continue; }
        if (first == 0u) {
            first = id;
            first_k = k;
            first_weight = w.weight;
            hit = w.p;
            if (mode == 0u) { break; }
        }
        rgb = rgb + transparency * alpha * colour_of(id) * w.weight;
        share = share + transparency * alpha;
        transparency = transparency * (1.0 - alpha);
    }
    if (first != 0u && first_k < pick[index].y) {
        pick[index] = vec2<u32>(first, first_k);
    }
    if (mode == 0u && first != 0u) {
        let size = P.v[11].xyz;
        let g = vec3<f32>(
            occupancy(hit + vec3<f32>(1.0, 0.0, 0.0)) - occupancy(hit - vec3<f32>(1.0, 0.0, 0.0)),
            occupancy(hit + vec3<f32>(0.0, 1.0, 0.0)) - occupancy(hit - vec3<f32>(0.0, 1.0, 0.0)),
            occupancy(hit + vec3<f32>(0.0, 0.0, 1.0)) - occupancy(hit - vec3<f32>(0.0, 0.0, 1.0)))
            / (2.0 * size);
        let view = P.v[12].xyz;
        var n = view;
        if (length(g) > 1e-6) { n = -normalize(g); }
        if (dot(n, view) < 0.0) { n = -n; }
        let colour = colour_of(first);
        let diffuse = max(dot(n, P.v[13].xyz), 0.0);
        var shine = 0.0;
        if (P.v[15].z > 0.0) { shine = pow(max(dot(n, P.v[14].xyz), 0.0), P.v[15].w); }
        var lit = colour * (P.v[15].x + P.v[15].y * diffuse)
                  + P.v[15].z * shine * mix(vec3<f32>(1.0), colour, P.v[16].x);
        lit = clamp(lit * (1.0 - 0.35 * f32(first_k) / max(f32(samples), 1.0)),
                    vec3<f32>(0.0), vec3<f32>(1.0));
        if (highlight > 0u && first == highlight) { lit = 0.5 * lit + vec3<f32>(0.5); }
        lit = lit * first_weight;
        base = base * (1.0 - opacity) + lit * opacity;
    } else if (mode == 1u && share > 0.0) {
        var glass = rgb / share;
        if (highlight > 0u && first == highlight) { glass = 0.5 * glass + vec3<f32>(0.5); }
        let a = share * opacity;
        base = base * (1.0 - a) + glass * a;
    }
    frame[index] = vec4<f32>(base, 1.0);
}
"""

_WGSL_PACK = """
@group(0) @binding(1) var<storage, read> frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> picture: array<u32>;

fn texel(x: i32, y: i32, columns: i32) -> vec3<f32> {
    return clamp(frame[u32(y * columns + x)].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = i32(P.v[5].x);
    let columns = i32(P.v[5].y);
    let height = u32(P.v[16].y);
    let width = u32(P.v[16].z);
    if (gid.x >= width || gid.y >= height) { return; }
    let sx = (f32(gid.x) + 0.5) * f32(columns) / f32(width) - 0.5;
    let sy = (f32(gid.y) + 0.5) * f32(rows) / f32(height) - 0.5;
    let fx = floor(sx);
    let fy = floor(sy);
    let tx = sx - fx;
    let ty = sy - fy;
    let x0 = clamp(i32(fx), 0, columns - 1);
    let x1 = clamp(i32(fx) + 1, 0, columns - 1);
    let y0 = clamp(i32(fy), 0, rows - 1);
    let y1 = clamp(i32(fy) + 1, 0, rows - 1);
    let top = mix(texel(x0, y0, columns), texel(x1, y0, columns), tx);
    let bottom = mix(texel(x0, y1, columns), texel(x1, y1, columns), tx);
    picture[gid.y * width + gid.x] = pack4x8unorm(vec4<f32>(mix(top, bottom, ty), 1.0));
}
"""

#: The compute kernels of :class:`WGPUVolumeRenderer`, by name.
_WGSL_KERNELS = {"clear": _WGSL_CLEAR, "image": _WGSL_IMAGE, "labels": _WGSL_LABELS,
                 "pack": _WGSL_PACK}


def gpu_render_refusal() -> str:
    """Why a 3-D view must not draw on the GPU here, or ``""``.

    :returns: ``"SPACR_NO_GPU_RENDER is set"`` (:data:`GPU_RENDER_OFF_ENV`,
        any value but ``0`` or empty), ``"CUDA_VISIBLE_DEVICES is empty"``
        (set to nothing), or ``""`` when the GPU may be used.
    """
    import os

    flag = os.environ.get(GPU_RENDER_OFF_ENV, "").strip()
    if flag and flag != "0":
        return f"{GPU_RENDER_OFF_ENV} is set"
    if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"].strip():
        return "CUDA_VISIBLE_DEVICES is empty"
    return ""


def _wgpu_enumerate() -> List[Any]:
    """Every adapter wgpu finds; none when wgpu is missing or cannot look."""
    from importlib.util import find_spec

    if find_spec("wgpu") is None:
        return []
    try:
        import wgpu

        found = getattr(wgpu.gpu, "enumerate_adapters_sync", None) or wgpu.gpu.enumerate_adapters
        return list(found())
    except Exception:
        return []


def _wgpu_info(adapter: Any) -> Dict[str, str]:
    """An adapter's backend key (``vulkan``, ``metal``, ``d3d12`` or ``""``) and names."""
    info = dict(getattr(adapter, "info", None) or {})
    kind = str(info.get("backend_type", "")).lower().replace(" ", "")
    backend = next((key for key, name in WGPU_BACKENDS.items() if kind == name), "")
    return {"backend": backend, "name": str(info.get("device") or info.get("description") or ""),
            "adapter_type": str(info.get("adapter_type", "")),
            "description": str(info.get("description", ""))}


def wgpu_adapters(refresh: bool = False) -> Tuple[Dict[str, str], ...]:
    """The graphics adapters wgpu can draw the True 3D view with.

    wgpu is imported the first time this is asked, never before, so a
    computer without it pays nothing; it is then remembered.

    :param refresh: look again rather than use what was found before.
    :returns: one ``{"backend", "name", "adapter_type", "description"}`` per
        Vulkan, Metal or Direct3D 12 adapter; empty without wgpu.
    """
    global _WGPU_FOUND
    if _WGPU_FOUND is None or refresh:
        infos = (_wgpu_info(adapter) for adapter in _wgpu_enumerate())
        _WGPU_FOUND = tuple(info for info in infos if info["backend"])
    return _WGPU_FOUND


def wgpu_adapter(backend: str, allow_software: bool = True) -> Any:
    """A fresh wgpu adapter of one backend, the best kind first.

    A discrete GPU is preferred to an integrated one, and a software
    rasteriser (``adapter_type`` ``CPU``) comes last, or not at all.

    :param backend: a key of :data:`WGPU_BACKENDS`.
    :param allow_software: whether a software rasteriser is accepted.
    :returns: the adapter, ready for one device.
    :raises RuntimeError: with a one-line reason when there is none.
    """
    if backend not in WGPU_BACKENDS:
        raise RuntimeError(f"{backend!r} is not a wgpu backend")
    ranked = []
    for adapter in _wgpu_enumerate():
        info = _wgpu_info(adapter)
        if info["backend"] != backend:
            continue
        rank = _WGPU_ADAPTER_RANK.get(info["adapter_type"].lower(), 3)
        if rank == _WGPU_ADAPTER_RANK["cpu"] and not allow_software:
            continue
        ranked.append((rank, len(ranked), adapter))
    if not ranked:
        raise RuntimeError(f"wgpu has no {'' if allow_software else 'hardware '}"
                           f"{backend} adapter on this computer")
    return min(ranked)[2]


class WGPUVolumeRenderer:
    """:func:`render_volume`, ray-marched on the GPU through wgpu (WGSL).

    The third renderer of the same frame: Vulkan, Metal or Direct3D 12 by the
    adapter it is given. The same rays (:class:`VolumeRays`), renderings,
    finishes and lens as :func:`render_volume` and
    :class:`GLVolumeRenderer`; one compute invocation walks one ray. Volumes
    are read with ``textureLoad`` and interpolated in the shader exactly as
    :func:`scipy.ndimage.map_coordinates` does (``order=1``, edges
    repeated), image layers add up, labels layers are laid over them in
    stack order, and each labels layer writes the object nearest the viewer
    under each ray for picking. The frame is drawn into a storage buffer,
    scaled to the picture's size and read back, so it lands under the same
    matplotlib axes as the CPU's.

    A volume is uploaded once and kept, keyed by its memory, for as long as
    every frame draws it, as :class:`GLVolumeRenderer` does.

    :param adapter: a wgpu adapter (:func:`wgpu_adapter`).
    """

    def __init__(self, adapter: Any) -> None:
        """Ask the adapter for a device with its own largest limits."""
        self._adapter = adapter
        info = _wgpu_info(adapter)
        self.backend = info["backend"]
        offered = dict(getattr(adapter, "limits", None) or {})
        wanted = {key: value for key, value in offered.items()
                  if key.replace("_", "-") in ("max-texture-dimension-3d",
                                               "max-storage-buffer-binding-size",
                                               "max-buffer-size")}
        ask = getattr(adapter, "request_device_sync", None) or adapter.request_device
        self._device = ask(required_limits=wanted)
        self._info = info
        self._offered = offered
        self._pipelines: Dict[str, Any] = {}
        self._textures: Dict[Tuple[Any, ...], Tuple[Any, Any]] = {}
        self._buffers: Dict[str, Tuple[int, Any]] = {}
        self._luts: Dict[Any, Any] = {}
        self._used: set = set()

    def _limit(self, name: str, default: int) -> int:
        """One of the device's limits, by its hyphenated WebGPU name."""
        for key in (name, name.replace("-", "_")):
            if key in self._offered:
                return int(self._offered[key])
        return int(default)

    def limits(self) -> Dict[str, Any]:
        """The adapter's name, its backend and its largest 3-D texture side.

        :returns: ``{"renderer": str, "version": str, "max_3d": int,
            "adapter_type": str}``.
        """
        return {"renderer": self._info["name"], "version": self._info["description"],
                "max_3d": self._limit("max-texture-dimension-3d", 2048),
                "adapter_type": self._info["adapter_type"]}

    def _pipeline(self, name: str):
        """One compute pipeline, compiled the first time it is asked for."""
        pipeline = self._pipelines.get(name)
        if pipeline is None:
            module = self._device.create_shader_module(code=_WGSL_COMMON + _WGSL_KERNELS[name])
            pipeline = self._device.create_compute_pipeline(
                layout="auto", compute={"module": module, "entry_point": "main"})
            self._pipelines[name] = pipeline
        return pipeline

    def _volume(self, array: Any, kind: str):
        """The GPU copy of one volume: ``image`` (float bits) or ``ids``."""
        import wgpu

        key = (_array_key(array), kind)
        held = self._textures.get(key)
        if held is not None and held[0] is _array_owner(array):
            self._used.add(key)
            return held[1]
        data = np.asarray(array)
        if kind == "ids":
            top = int(data.max()) if data.size else 0
            if int(data.min() if data.size else 0) < 0 or top >= 1 << 32:
                raise LayerError("object IDs must lie between 0 and 2**32 - 1 to be drawn on the GPU")
            texels = np.ascontiguousarray(data, dtype=np.uint32)
        else:
            texels = np.ascontiguousarray(data, dtype=np.float32).view(np.uint32)
        depth, height, width = texels.shape
        side = self._limit("max-texture-dimension-3d", 2048)
        if max(texels.shape) > side:
            raise LayerError(f"a volume side of {max(texels.shape)} exceeds this GPU's {side}")
        texture = self._device.create_texture(
            size=(width, height, depth), dimension="3d", format=wgpu.TextureFormat.r32uint,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)}, texels,
            {"offset": 0, "bytes_per_row": width * 4, "rows_per_image": height},
            (width, height, depth))
        view = texture.create_view()
        self._textures[key] = (_array_owner(array), (texture, view))
        self._used.add(key)
        return texture, view

    def _colours(self, labels: Any):
        """A storage buffer of every object's colour, indexed by object ID."""
        import wgpu

        key = (_array_key(labels), "colours")
        held = self._textures.get(key)
        if held is not None and held[0] is _array_owner(labels):
            self._used.add(key)
            return held[1]
        present = np.unique(np.asarray(labels))
        top = int(present.max()) if present.size else 0
        if (top + 1) * 16 > self._limit("max-storage-buffer-binding-size", 1 << 27):
            raise LayerError(f"object IDs up to {top} are too many for this GPU's colour table")
        table = np.zeros((top + 1, 4), dtype=np.float32)
        table[present, :3] = label_colors(present)
        table[:, 3] = 1.0
        buffer = self._device.create_buffer_with_data(data=table, usage=wgpu.BufferUsage.STORAGE)
        self._textures[key] = (_array_owner(labels), (buffer, None))
        self._used.add(key)
        return buffer, None

    def _lut(self, colormap: "Colormap"):
        """A 256-entry storage buffer of one colormap, kept for the next frame."""
        import wgpu

        buffer = self._luts.get(colormap)
        if buffer is None:
            lut = np.ones((256, 4), dtype=np.float32)
            lut[:, :3] = colormap.map(np.linspace(0.0, 1.0, 256))
            buffer = self._device.create_buffer_with_data(data=lut, usage=wgpu.BufferUsage.STORAGE)
            self._luts[colormap] = buffer
        return buffer

    def _buffer(self, name: str, size: int, readable: bool):
        """A storage buffer of at least ``size`` bytes, reused while it is big enough."""
        import wgpu

        held = self._buffers.get(name)
        if held is not None and held[0] == size:
            return held[1]
        if held is not None:
            held[1].destroy()
        usage = wgpu.BufferUsage.STORAGE | (wgpu.BufferUsage.COPY_SRC if readable else 0)
        buffer = self._device.create_buffer(size=size, usage=usage)
        self._buffers[name] = (size, buffer)
        return buffer

    def _forget_unused(self) -> None:
        """Free the volumes the frame just drawn did not use."""
        for key in [key for key in self._textures if key not in self._used]:
            resource = self._textures.pop(key)[1][0]
            resource.destroy()
        self._used = set()

    @staticmethod
    def _params(rays: VolumeRays, shape: Sequence[int], lens: Optional[VolumeLens],
                mode: int = 0) -> np.ndarray:
        """The uniform block every kernel reads, for one camera and volume."""
        params = np.zeros((17, 4), dtype=np.float32)
        params[0, :3], params[1, :3] = rays.origin, rays.across
        params[2, :3], params[3, :3] = rays.down, rays.depth
        params[4, :3] = tuple(float(v) for v in shape) if len(shape) == 3 else (1.0, 1.0, 1.0)
        params[5] = (rays.shape[0], rays.shape[1], rays.shape[2], mode)
        if lens is not None:
            params[6, :3], params[7, :3] = lens.centre, lens.half
            params[8, :2] = (lens.zoom, lens.dim)
        return params

    def _dispatch(self, compute, name: str, params: np.ndarray, resources: Sequence[Any],
                  groups: Tuple[int, int]) -> None:
        """Run one kernel over a grid of 8 x 8 workgroups."""
        import wgpu

        pipeline = self._pipeline(name)
        uniform = self._device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM)
        entries = [{"binding": 0, "resource": {"buffer": uniform, "offset": 0,
                                               "size": uniform.size}}]
        for binding, resource in enumerate(resources, start=1):
            if isinstance(resource, wgpu.GPUBuffer):
                entries.append({"binding": binding, "resource": {
                    "buffer": resource, "offset": 0, "size": resource.size}})
            else:
                entries.append({"binding": binding, "resource": resource})
        group = self._device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
                                               entries=entries)
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(int(groups[0]), int(groups[1]), 1)

    def render(self, layers: Iterable["Layer"], rays: VolumeRays, *,
               highlight: int = 0, voxel_size: Optional[Sequence[float]] = None,
               lens: Optional[VolumeLens] = None,
               size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Draw one frame; the same picture as :func:`render_volume`.

        :param layers: the layers, bottom first.
        :param rays: where to sample, in the layers' array indices.
        :param highlight: an object drawn lighter.
        :param voxel_size: physical voxel length along each array axis.
        :param lens: a magnifier cube, or ``None``.
        :param size: ``(rows, columns)`` of the picture returned, scaled up
            from the rays' grid on the GPU; ``None`` keeps the rays' grid.
        :returns: ``(rgb, picked)``: ``uint8`` colour of ``size`` and the
            object nearest the viewer under each ray (``int64``, the rays'
            grid).
        :raises LayerError: as :func:`render_volume`.
        """
        rows, columns, _samples = (int(value) for value in rays.shape)
        drawn = [layer for layer in layers if layer.visible
                 and isinstance(layer, (ImageLayer, LabelsLayer))]
        shapes = {layer.shape if isinstance(layer, LabelsLayer) else layer.channel_data(0).shape
                  for layer in drawn}
        if len(shapes) > 1:
            raise LayerError(f"3-D layers must share one shape, got {sorted(shapes)}")
        if any(len(shape) != 3 for shape in shapes):
            raise LayerError("the GPU draws 3-D layers only")
        modes = [_layer_rendering(layer) for layer in drawn]
        finishes = [surface_finish(layer.metadata.get("finish")) if isinstance(layer, LabelsLayer)
                    else None for layer in drawn]
        light = SurfaceLight.for_rays(rays, voxel_size)
        length = max(rays.sample_length, 1e-6)
        height, width = (rows, columns) if size is None else (max(1, int(v)) for v in size)
        shape = next(iter(shapes)) if shapes else (1, 1, 1)
        frame = self._buffer("frame", rows * columns * 16, False)
        pick = self._buffer("pick", rows * columns * 8, True)
        picture = self._buffer("picture", height * width * 4, True)
        grid = (-(-columns // 8), -(-rows // 8))
        encoder = self._device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        self._dispatch(compute, "clear", self._params(rays, shape, lens), (frame, pick), grid)
        for index, layer in enumerate(drawn):
            if not isinstance(layer, ImageLayer):
                continue
            data = layer.channel_data(0)
            low, high = layer.contrast_limits(0)
            params = self._params(rays, data.shape, lens, _GL_IMAGE_MODES[modes[index]])
            params[9] = (low, high, layer.opacity, length)
            params[10, :2] = (VOLUME_ATTENUATION, VOLUME_DENSITY)
            self._dispatch(compute, "image", params,
                           (frame, self._volume(data, "image")[1], self._lut(layer.colormap)),
                           grid)
        for index, layer in enumerate(drawn):
            if not isinstance(layer, LabelsLayer):
                continue
            finish = finishes[index]
            params = self._params(rays, layer.shape, lens, 0 if modes[index] == "surface" else 1)
            params[9, 2] = layer.opacity
            params[10, 2:] = (min(1.0, LABEL_DENSITY * length), float(highlight))
            params[11, :3], params[12, :3] = light.size, light.view
            params[13, :3], params[14, :3] = light.light, light.half
            params[15] = (finish.ambient, finish.diffuse, finish.specular, finish.shininess)
            params[16, 0] = finish.tint
            self._dispatch(compute, "labels", params,
                           (frame, pick, self._volume(layer.data, "ids")[1],
                            self._colours(layer.data)[0]), grid)
        params = self._params(rays, shape, lens)
        params[16, 1:3] = (height, width)
        self._dispatch(compute, "pack", params, (frame, picture),
                       (-(-width // 8), -(-height // 8)))
        compute.end()
        self._device.queue.submit([encoder.finish()])
        rgb = np.frombuffer(self._device.queue.read_buffer(picture), dtype=np.uint8)
        codes = np.frombuffer(self._device.queue.read_buffer(pick), dtype=np.uint32)
        self._forget_unused()
        rgb = rgb[:height * width * 4].reshape(height, width, 4)[..., :3]
        picked = codes[:rows * columns * 2].reshape(rows, columns, 2)[..., 0].astype(np.int64)
        return np.ascontiguousarray(rgb), picked

    def close(self) -> None:
        """Free every volume, buffer and the device."""
        for _owner, (resource, _view) in self._textures.values():
            resource.destroy()
        for _size, buffer in self._buffers.values():
            buffer.destroy()
        for buffer in self._luts.values():
            buffer.destroy()
        self._textures, self._buffers, self._luts, self._pipelines = {}, {}, {}, {}
        destroy = getattr(self._device, "destroy", None)
        if destroy is not None:
            destroy()
