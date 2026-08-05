"""A drawn region of interest, honoured by Measure.

Draw a polygon over a field and measure only the objects inside it. The
drawing half is :mod:`spacr.layers` — a :class:`~spacr.layers.ShapesLayer`
holds the vertices as geometry, in world coordinates, so the same ROI means the
same region on a downsampled preview and on the full-resolution mask. This
module is the other half: turning that geometry into the keep/drop decision
:func:`spacr.measure_hooks.apply_region_filter_hooks` asks for, and — the part
that is easy to get silently wrong — getting it into the worker processes that
do the measuring.

Nothing here edits :mod:`spacr.measure`. The extension point already exists
(:func:`spacr.measure_hooks.register_region_filter_hook`), it is applied after
the size filters and before ``_exclude_objects``, and a dropped label is zeroed
out of its mask *before* a single ``regionprops`` call, so keeping 5 of 500
objects costs 5 objects' worth of work.

Why an ROI is stored in world coordinates
-----------------------------------------

A polygon drawn on a 512-pixel preview of a 2048-pixel field is not the same
set of array indices as the region it names. Storing ``(row, column)`` would
make the ROI mean four different things at four zoom levels, and the picture
would look right in every one of them. So a saved ROI is a list of world
points plus the unit they are measured in, and placing it on a mask goes
through :class:`~spacr.layers.Spacing` and
:meth:`~spacr.layers.Canvas.for_grid` exactly like every other render in this
codebase. Mixing a µm ROI with a pixel-spaced measurement raises rather than
drawing a plausible region in the wrong place.

Two decision rules
------------------

``mode='centroid'`` (the default) keeps an object when its centroid falls
inside the ROI. It is the rule that partitions cleanly: an object is inside
exactly one of two ROIs that share an edge, so an object on a boundary is
counted once, and every object type is judged the same way without the cell
and its nucleus ever disagreeing.

``mode='overlap'`` keeps an object when at least :attr:`RoiSet.min_overlap` of
its pixels are inside. Use it when the objects are large compared with the ROI
and "the middle of the cell" is not the question being asked.

Reaching the workers
--------------------

:func:`spacr.measure.measure_crop` measures fields in a process pool. Under
``spawn`` (Windows, macOS, ``SPACR_START_METHOD=spawn``, and Python 3.14 on
Linux) a worker is a fresh interpreter with an empty hook registry: a filter
registered with :func:`~spacr.measure_hooks.register_region_filter_hook` in the
parent applies to **nothing at all**, the run completes, and every object in
the field is measured while the user believes only the ROI was. That is a
silent scientific error, so this module refuses to rely on inheritance:
:func:`enable_roi_filter` writes the ROI to disk and names
``spacr.roi:install`` in :data:`~spacr.measure_hooks.HOOKS_ENV_VAR`, and each
worker installs the filter for itself from the environment it inherits.
:func:`worker_delivery_status` answers "will this actually reach the workers?"
before the run rather than after it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as _field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import ConfigurationError
from .measure_hooks import (HOOKS_ENV_VAR, OBJECT_TYPES,
                            region_filter_hooks,
                            register_region_filter_hook,
                            unregister_region_filter_hook)

__all__ = [
    'RoiError',
    'ROI_ENV_VAR',
    'ON_MISSING_ENV_VAR',
    'INSTALLER_ENTRY',
    'HOOK_NAME',
    'HOOK_PRIORITY',
    'ANY_FIELD',
    'MODES',
    'ON_MISSING',
    'RegionOfInterest',
    'RoiSet',
    'RoiRegionFilter',
    'install',
    'enable_roi_filter',
    'disable_roi_filter',
    'worker_delivery_status',
]

#: Environment variable holding the path of the saved :class:`RoiSet`. Read by
#: :func:`install` in every process, including a cold ``spawn`` worker.
ROI_ENV_VAR = 'SPACR_ROI'

#: Environment variable holding what to do with a field the ROI set says
#: nothing about: ``'error'``, ``'all'`` or ``'none'``. See :class:`RoiSet`.
ON_MISSING_ENV_VAR = 'SPACR_ROI_ON_MISSING'

#: What :func:`enable_roi_filter` adds to :data:`~spacr.measure_hooks.HOOKS_ENV_VAR`.
INSTALLER_ENTRY = 'spacr.roi:install'

#: The registry key the filter is registered under. Fixed, so re-enabling
#: replaces the filter rather than intersecting two of them.
HOOK_NAME = 'spacr.roi.region_filter'

#: Region filters intersect, so priority only affects reporting order.
HOOK_PRIORITY = 0

#: The field name meaning "every field that has no ROI of its own".
ANY_FIELD = '*'

#: How an object is judged against the ROI.
MODES: Tuple[str, ...] = ('centroid', 'overlap')

#: What happens to a field the ROI set does not cover.
ON_MISSING: Tuple[str, ...] = ('error', 'all', 'none')


class RoiError(ConfigurationError):
    """An ROI that cannot be placed on the mask it was handed.

    A :class:`spacr.errors.ConfigurationError`, like every other measurement
    hook failure: a mis-specified ROI is wrong for every field on the plate,
    not bad luck on one of them.
    """


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RegionOfInterest:
    """One drawn region, in world coordinates.

    Compared by identity (``eq=False``), for the reason
    :class:`spacr.layers.Shape` is: a generated ``__eq__`` would compare the
    vertex arrays elementwise and raise "truth value of an array is ambiguous"
    from anything as ordinary as ``roi in roi_set.fields['*']``.

    :param kind: ``'polygon'``, ``'rectangle'`` or ``'ellipse'`` — the closed
        kinds of :class:`spacr.layers.Shape`. An open shape (a line, a path)
        encloses nothing and is not an ROI.
    :param vertices: ``(M, 2)`` world coordinates, in the order named by
        :attr:`RoiSet.axes` (``(y, x)`` unless something says otherwise). A
        rectangle or an ellipse may be given as two opposite corners.
    :param name: what the user called it, carried through to the diagnostic so
        "dropped by ROI 'well edge'" is possible.
    """

    kind: str
    vertices: Any
    name: str = ''

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in ('polygon', 'rectangle', 'ellipse'):
            raise RoiError(
                f"an ROI is a closed shape: 'polygon', 'rectangle' or "
                f"'ellipse', not {self.kind!r}. A line or a path encloses no "
                f"area, so there is no inside for Measure to keep.")
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise RoiError(
                f"an ROI needs an (M, 2) array of world points, got shape "
                f"{vertices.shape}")
        if vertices.shape[0] < 2:
            raise RoiError(
                f"a {kind} needs at least two points, got "
                f"{vertices.shape[0]}")
        if kind == 'polygon' and vertices.shape[0] < 3:
            raise RoiError('a polygon needs at least three points')
        if not np.all(np.isfinite(vertices)):
            raise RoiError('ROI vertices must all be finite')
        object.__setattr__(self, 'kind', kind)
        object.__setattr__(self, 'vertices', vertices)
        object.__setattr__(self, 'name', str(self.name))

    def to_shape(self, spacing) -> Any:
        """This ROI as a :class:`spacr.layers.Shape` on ``spacing``'s grid.

        The world→data conversion is ``spacing``'s own, so a shape built here
        rasterises onto that grid through the ordinary
        :meth:`spacr.layers.ShapesLayer.mask` path.

        :param spacing: a two-axis :class:`spacr.layers.Spacing` whose axes are
            this ROI's axes, in order.
        """
        from .layers import Shape
        scale = np.asarray(spacing.scale, dtype=np.float64)
        offset = np.asarray(spacing.translate, dtype=np.float64)
        return Shape(self.kind, (self.vertices - offset) / scale,
                     name=self.name)

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict, the form :meth:`RoiSet.save` writes."""
        return {'kind': self.kind, 'name': self.name,
                'vertices': [[float(v) for v in row] for row in self.vertices]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'RegionOfInterest':
        """Rebuild one from :meth:`as_dict`."""
        try:
            return cls(kind=payload['kind'], vertices=payload['vertices'],
                       name=payload.get('name', ''))
        except KeyError as exc:
            raise RoiError(
                f"an ROI entry needs {exc} — the file was not written by "
                f"RoiSet.save()") from exc


# ---------------------------------------------------------------------------
# The set, and how it is judged
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoiSet:
    """Which regions apply to which fields, and how an object is judged.

    :param fields: ``{field name: (RegionOfInterest, ...)}``. The field name is
        the ``.npy`` stem the pipeline uses, e.g. ``plate1_A01_F001``;
        :data:`ANY_FIELD` (``'*'``) is the fallback for every field with no
        entry of its own. Several ROIs on one field are a UNION — drawing a
        second polygon adds to what is measured.
    :param axes: which world axes the vertex columns are, **outermost first**,
        matching the mask's own array axis order — ``('y', 'x')`` unless the
        ROI was drawn on some other plane. The order is not cosmetic: a
        transposed pair puts the region somewhere plausible and wrong, so the
        standard pair is only accepted the standard way round.
    :param units: what one world unit is, compared by name against the
        measurement's own spacing. Mixing µm with px raises rather than drawing
        a plausible region in the wrong place.
    :param mode: ``'centroid'`` or ``'overlap'``; see the module docstring.
    :param min_overlap: for ``'overlap'``, the fraction of an object's pixels
        that must be inside for it to be kept.
    :param invert: keep the objects OUTSIDE the ROI instead. "Exclude this
        debris" is as common a request as "measure this colony".
    :param object_types: which of :data:`spacr.measure_hooks.OBJECT_TYPES` the
        ROI applies to. The default is all of them, which is what makes the
        cell and its nucleus agree.
    :param on_missing: what a field with no ROI means — ``'error'`` (the
        default: refuse, because measuring everything when the user drew a
        region is the silent answer), ``'all'`` (measure the whole field) or
        ``'none'`` (measure nothing in it).
    """

    fields: Mapping[str, Tuple[RegionOfInterest, ...]] = _field(
        default_factory=dict)
    axes: Tuple[str, str] = ('y', 'x')
    units: str = 'px'
    mode: str = 'centroid'
    min_overlap: float = 0.5
    invert: bool = False
    object_types: Tuple[str, ...] = OBJECT_TYPES
    on_missing: str = 'error'

    def __post_init__(self) -> None:
        fields: Dict[str, Tuple[RegionOfInterest, ...]] = {}
        for name, rois in dict(self.fields).items():
            entries = tuple(rois)
            for roi in entries:
                if not isinstance(roi, RegionOfInterest):
                    raise RoiError(
                        f"field {name!r} holds {type(roi).__name__}, not a "
                        f"RegionOfInterest")
            fields[str(name)] = entries
        object.__setattr__(self, 'fields', fields)

        axes = tuple(str(a) for a in self.axes)
        if len(axes) != 2 or axes[0] == axes[1]:
            raise RoiError(
                f"an ROI lies in a plane: axes takes exactly two different "
                f"axis names, got {axes}")
        if axes == ('x', 'y'):
            # Refused rather than transposed for the caller: a mask is (Y, X)
            # and the vertices are stored row-first, so this pair means the ROI
            # would be rasterised on its side. It still draws a region, which
            # is exactly why it has to raise.
            raise RoiError(
                "axes ('x', 'y') is the vertex order reversed: an ROI's "
                "vertices are stored in the mask's own axis order, outermost "
                "first, so a (Y, X) mask wants ('y', 'x'). Swap the vertex "
                "columns rather than the axis names.")
        object.__setattr__(self, 'axes', axes)
        object.__setattr__(self, 'units', str(self.units))

        mode = str(self.mode).strip().lower()
        if mode not in MODES:
            raise RoiError(f"unknown ROI mode {self.mode!r}; use one of "
                           f"{list(MODES)}")
        object.__setattr__(self, 'mode', mode)

        overlap = float(self.min_overlap)
        if not 0.0 < overlap <= 1.0:
            raise RoiError(
                f"min_overlap is a fraction of an object's pixels and must be "
                f"in (0, 1], got {self.min_overlap!r}")
        object.__setattr__(self, 'min_overlap', overlap)
        object.__setattr__(self, 'invert', bool(self.invert))

        types = tuple(str(t) for t in self.object_types)
        unknown = [t for t in types if t not in OBJECT_TYPES]
        if unknown:
            raise RoiError(
                f"unknown object type(s) {unknown}; Measure offers "
                f"{list(OBJECT_TYPES)}. A name that matches nothing would "
                f"leave the ROI applying to no object at all.")
        object.__setattr__(self, 'object_types', types)

        on_missing = str(self.on_missing).strip().lower()
        if on_missing not in ON_MISSING:
            raise RoiError(
                f"unknown on_missing {self.on_missing!r}; use one of "
                f"{list(ON_MISSING)}")
        object.__setattr__(self, 'on_missing', on_missing)

    # -- queries ---------------------------------------------------------
    def __len__(self) -> int:
        return sum(len(v) for v in self.fields.values())

    def covers(self, file_name: str) -> bool:
        """Whether this set has anything to say about ``file_name``."""
        return self.rois_for(file_name) is not None

    def rois_for(self, file_name: str
                 ) -> Optional[Tuple[RegionOfInterest, ...]]:
        """The ROIs that apply to a field, or ``None`` when none do.

        The field's own entry wins over :data:`ANY_FIELD`; a field entry that
        is present but empty means "this field has an ROI and it encloses
        nothing", which is not the same as having no entry.
        """
        stem = _field_stem(file_name)
        if stem in self.fields:
            return self.fields[stem]
        if ANY_FIELD in self.fields:
            return self.fields[ANY_FIELD]
        return None

    def describe(self) -> str:
        """One line for a status bar or a run log."""
        named = sorted(k for k in self.fields if k != ANY_FIELD)
        where = (f"{len(named)} field(s)" if named else 'every field')
        if ANY_FIELD in self.fields and named:
            where += ' plus a default for the rest'
        inside = 'outside' if self.invert else 'inside'
        rule = (f"{self.mode} rule"
                if self.mode == 'centroid'
                else f"{self.mode} rule at {self.min_overlap:.0%}")
        return (f"{len(self)} ROI(s) over {where}, measuring {inside} them "
                f"({rule}, {self.units}, {', '.join(self.object_types)})")

    # -- construction ----------------------------------------------------
    @classmethod
    def from_shapes_layer(cls, layer, *, fields: Any = ANY_FIELD,
                          **kwargs: Any) -> 'RoiSet':
        """Take the closed shapes off a :class:`spacr.layers.ShapesLayer`.

        The layer's vertices are in data coordinates on its own grid; they are
        converted to world here through the layer's
        :class:`~spacr.layers.Spacing`, which is what lets an ROI drawn on a
        preview be applied to a full-resolution mask.

        :param layer: the shapes layer the user drew on.
        :param fields: a field name, an iterable of them, or
            :data:`ANY_FIELD`. Every closed shape is attached to each.
        :raises RoiError: if the layer holds no closed shape, or its spacing
            has no axis for the plane the shapes were drawn in.
        """
        spacing = layer.spacing
        axes = tuple(kwargs.pop('axes', None) or _plane_axes(spacing))
        try:
            columns = [spacing.axis_index(a) for a in axes]
        except Exception as exc:
            raise RoiError(
                f"the shapes layer is on axes {spacing.axes}, which do not "
                f"include {axes}") from exc
        scale = np.asarray(spacing.scale, dtype=np.float64)
        offset = np.asarray(spacing.translate, dtype=np.float64)
        rois = []
        for shape in layer.shapes:
            if not shape.is_closed:
                continue
            world = np.asarray(shape.data, dtype=np.float64) * scale + offset
            rois.append(RegionOfInterest(
                kind=shape.kind, vertices=world[:, columns],
                name=shape.name or ''))
        if not rois:
            raise RoiError(
                f"layer {layer.name!r} holds no closed shape, so there is no "
                f"region to measure inside. Draw a polygon, a rectangle or an "
                f"ellipse — a line or a path encloses no area.")
        names = ([fields] if isinstance(fields, str)
                 else [str(f) for f in fields])
        return cls(fields={_field_stem(n): tuple(rois) for n in names},
                   axes=axes, units=spacing.units, **kwargs)

    # -- persistence -----------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict — what :meth:`save` writes and :meth:`load` reads."""
        return {
            'spacr_roi_version': 1,
            'axes': list(self.axes),
            'units': self.units,
            'mode': self.mode,
            'min_overlap': self.min_overlap,
            'invert': self.invert,
            'object_types': list(self.object_types),
            'on_missing': self.on_missing,
            'fields': {name: [roi.as_dict() for roi in rois]
                       for name, rois in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'RoiSet':
        """Rebuild a set from :meth:`as_dict`."""
        data = dict(payload)
        try:
            fields = {str(name): tuple(RegionOfInterest.from_dict(entry)
                                       for entry in entries)
                      for name, entries in dict(data.get('fields') or {}).items()}
        except (TypeError, AttributeError) as exc:
            raise RoiError(
                f"the 'fields' entry must map a field name to a list of ROIs, "
                f"got {data.get('fields')!r}") from exc
        return cls(fields=fields,
                   axes=tuple(data.get('axes') or ('y', 'x')),
                   units=data.get('units', 'px'),
                   mode=data.get('mode', 'centroid'),
                   min_overlap=data.get('min_overlap', 0.5),
                   invert=data.get('invert', False),
                   object_types=tuple(data.get('object_types') or OBJECT_TYPES),
                   on_missing=data.get('on_missing', 'error'))

    def save(self, path: str) -> str:
        """Write this set to ``path`` as JSON; returns the absolute path.

        JSON rather than ``.npz`` on purpose: an ROI is a few dozen numbers and
        a human being should be able to read the file that decided which cells
        were measured. Creates the parent folder.

        :raises RoiError: if the file cannot be written.
        """
        target = os.path.abspath(str(path))
        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as handle:
                json.dump(self.as_dict(), handle, indent=2)
        except OSError as exc:
            raise RoiError(
                f"the ROI could not be written to {target!r}: {exc}. A spawn "
                f"worker can only reach it through the file system, so an "
                f"unsaved ROI would apply to nothing.") from exc
        return target

    @classmethod
    def load(cls, path: str) -> 'RoiSet':
        """Read a set back from :meth:`save`.

        :raises RoiError: if the file is missing or is not an ROI file. Loudly,
            because a worker that cannot load the ROI must not go on to measure
            the whole field.
        """
        target = os.path.abspath(str(path))
        if not os.path.isfile(target):
            raise RoiError(
                f"the ROI file {target!r} does not exist. Set {ROI_ENV_VAR} to "
                f"a file written by RoiSet.save() — or call "
                f"spacr.roi.enable_roi_filter(roi_set), which does both.")
        try:
            with open(target, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            raise RoiError(
                f"the ROI file {target!r} could not be read: {exc}") from exc
        if not isinstance(payload, dict):
            raise RoiError(
                f"the ROI file {target!r} holds a "
                f"{type(payload).__name__}, not an ROI written by "
                f"RoiSet.save()")
        return cls.from_dict(payload)


def _field_stem(file_name: Any) -> str:
    """The field key a name is filed under: basename, no ``.npy`` suffix."""
    text = str(file_name)
    if text == ANY_FIELD:
        return ANY_FIELD
    stem = os.path.basename(text)
    if stem.lower().endswith('.npy'):
        stem = stem[:-4]
    return stem


def _plane_axes(spacing) -> Tuple[str, str]:
    """The two in-plane axes of a spacing: its last two, ``(y, x)`` normally."""
    axes = tuple(spacing.axes)
    if len(axes) < 2:
        raise RoiError(
            f"an ROI lies in a plane, but the layer's spacing has only "
            f"{len(axes)} axis ({axes})")
    return axes[-2], axes[-1]


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------

class RoiRegionFilter:
    """The callable :func:`spacr.measure_hooks.register_region_filter_hook` runs.

    Holds one :class:`RoiSet` and the raster of the field it last saw. The
    cache matters: ``_measure_crop_core`` consults the filter once per object
    type, five times per field, and every one of those calls wants the same
    rasterised polygon on the same grid.

    :param roi_set: the regions and the rule.
    :param on_missing: overrides :attr:`RoiSet.on_missing` when given.
    """

    def __init__(self, roi_set: RoiSet, *, on_missing: Optional[str] = None):
        if not isinstance(roi_set, RoiSet):
            raise RoiError(
                f"an ROI filter needs a RoiSet, got {type(roi_set).__name__}")
        self.roi_set = roi_set
        if on_missing is not None:
            on_missing = str(on_missing).strip().lower()
            if on_missing not in ON_MISSING:
                raise RoiError(
                    f"unknown on_missing {on_missing!r}; use one of "
                    f"{list(ON_MISSING)}")
        self.on_missing = on_missing or roi_set.on_missing
        #: ``{'kept': n, 'dropped': n, 'fields': n}`` since this filter was made.
        self.stats: Dict[str, int] = {'kept': 0, 'dropped': 0, 'fields': 0}
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache: Optional[np.ndarray] = None

    # -- the hook --------------------------------------------------------
    def __call__(self, context) -> np.ndarray:
        """Decide which of ``context.labels`` are inside the ROI.

        :param context: a :class:`spacr.measure_hooks.RegionContext`.
        :returns: a boolean array of ``len(context.labels)``.
        :raises RoiError: if the field is not covered and ``on_missing`` is
            ``'error'``, or if the ROI cannot be placed on this mask.
        """
        labels = context.labels
        if context.object_type not in self.roi_set.object_types:
            return np.ones(labels.shape, dtype=bool)
        rois = self.roi_set.rois_for(context.file_name)
        if rois is None:
            return self._missing(context, labels)
        if labels.size == 0:
            return np.ones((0,), dtype=bool)
        inside = self._raster(context, rois)
        if self.roi_set.mode == 'centroid':
            keep = self._by_centroid(context, inside)
        else:
            keep = self._by_overlap(context, inside)
        self._record(keep)
        return keep

    def _missing(self, context, labels: np.ndarray) -> np.ndarray:
        if self.on_missing == 'error':
            raise RoiError(
                f"no ROI covers field {context.file_name!r}. The ROI set names "
                f"{sorted(self.roi_set.fields)!r}; measuring the whole field "
                f"because a region was not drawn on it is the silent answer, "
                f"so it is refused. Add an ROI for this field, add a "
                f"{ANY_FIELD!r} default, or choose on_missing='all' (measure "
                f"it whole) or 'none' (skip it).")
        keep = np.full(labels.shape, self.on_missing == 'all', dtype=bool)
        self._record(keep)
        return keep

    def _record(self, keep: np.ndarray) -> None:
        self.stats['kept'] += int(np.count_nonzero(keep))
        self.stats['dropped'] += int(keep.size - np.count_nonzero(keep))

    # -- placing the ROI on the mask -------------------------------------
    def _raster(self, context, rois: Sequence[RegionOfInterest]) -> np.ndarray:
        """The ROI rasterised on the mask's own in-plane grid, ``(ny, nx)``.

        Cached on the field, the grid shape and the voxel spacing, so the five
        object types of one field rasterise once.
        """
        plane = tuple(int(v) for v in context.mask.shape[-2:])
        key = (context.file_name, plane, context.spacing)
        if self._cache_key == key and self._cache is not None:
            return self._cache
        from .layers import Canvas, ShapesLayer, Spacing

        spacing = self._plane_spacing(context)
        layer = ShapesLayer([roi.to_shape(spacing) for roi in rois],
                            name='roi', spacing=spacing)
        inside = layer.mask(Canvas.for_grid(spacing, plane, axes=spacing.axes))
        if self.roi_set.invert:
            inside = ~inside
        self._cache_key = key
        self._cache = inside
        self.stats['fields'] += 1
        return inside

    def _plane_spacing(self, context):
        """The :class:`spacr.layers.Spacing` of the mask's in-plane grid.

        2-D measurements are never scaled by spaCR (see
        :func:`spacr.measure.resolve_measurement_spacing`), so the grid is one
        world unit per pixel and the ROI must be in pixels. In 3-D the voxel
        size is real, and whether it is µm or xy-pixel units is decided by
        whether the run set ``voxel_size_xy_um`` — the same rule that function
        uses to stamp the measurement.
        """
        from .layers import Spacing

        axes = self.roi_set.axes
        if context.spacing is None:
            scale = (1.0, 1.0)
            units = 'px'
        else:
            scale = tuple(float(v) for v in context.spacing[-2:])
            units = ('um' if context.settings.get('voxel_size_xy_um')
                     else 'px')
        if self.roi_set.units != units:
            raise RoiError(
                f"the ROI is measured in {self.roi_set.units!r} but field "
                f"{context.file_name!r} is measured in {units!r}. Placing one "
                f"on the other would put the region somewhere plausible and "
                f"wrong: convert the ROI, or draw it on a layer with the same "
                f"spacing as the measurement.")
        return Spacing(scale=scale, axes=axes, units=units)

    # -- the two rules ---------------------------------------------------
    @staticmethod
    def _by_centroid(context, inside: np.ndarray) -> np.ndarray:
        """Keep an object when the pixel under its centroid is inside."""
        centroids = context.centroids
        rows = np.rint(centroids[:, -2]).astype(np.int64)
        columns = np.rint(centroids[:, -1]).astype(np.int64)
        rows = np.clip(rows, 0, inside.shape[0] - 1)
        columns = np.clip(columns, 0, inside.shape[1] - 1)
        return np.asarray(inside[rows, columns], dtype=bool)

    def _by_overlap(self, context, inside: np.ndarray) -> np.ndarray:
        """Keep an object when enough of its pixels are inside."""
        mask = np.asarray(context.mask)
        labels = context.labels
        width = int(labels.max()) + 1
        flat = mask.reshape(-1).astype(np.int64, copy=False)
        # A 2-D ROI applies to every z of a 3-D mask: the polygon was drawn
        # looking down the stack, so it names a column through it.
        covered = np.broadcast_to(inside, mask.shape).reshape(-1)
        total = np.bincount(flat, minlength=width)
        hit = np.bincount(flat, weights=covered.astype(np.float64),
                          minlength=width)
        fraction = hit[labels] / np.maximum(total[labels], 1)
        return fraction >= self.roi_set.min_overlap

    def report(self) -> str:
        """One line summarising what this filter has done so far."""
        return (f"ROI filter: {self.stats['kept']} object(s) kept, "
                f"{self.stats['dropped']} dropped across "
                f"{self.stats['fields']} field(s)")


# ---------------------------------------------------------------------------
# Enabling it — including in worker processes
# ---------------------------------------------------------------------------

def _env_entries(value: str) -> list:
    """Split a ``SPACR_MEASURE_HOOKS`` value into its non-empty entries."""
    return [item.strip() for item in str(value or '').split(',') if item.strip()]


def install() -> str:
    """Install the ROI filter in **this** process from the environment.

    The zero-argument installer :data:`~spacr.measure_hooks.HOOKS_ENV_VAR`
    names, and the only route that survives a ``spawn`` worker: the worker is a
    fresh interpreter, so it imports this module and calls this function for
    itself, reading the ROI from :data:`ROI_ENV_VAR`.

    :returns: the registry key the filter was registered under.
    :raises RoiError: if the environment does not name a readable ROI file.
        Refusing loudly is the point — a worker that cannot load the ROI must
        not go on to measure every object in the field.
    """
    path = os.environ.get(ROI_ENV_VAR, '').strip()
    if not path:
        raise RoiError(
            f"{INSTALLER_ENTRY} is in {HOOKS_ENV_VAR} but {ROI_ENV_VAR} is not "
            f"set, so there is no ROI to install. Call "
            f"spacr.roi.enable_roi_filter(roi_set), which sets both.")
    roi_set = RoiSet.load(path)
    on_missing = os.environ.get(ON_MISSING_ENV_VAR, '').strip() or None
    return register_region_filter_hook(
        RoiRegionFilter(roi_set, on_missing=on_missing),
        name=HOOK_NAME, priority=HOOK_PRIORITY)


def enable_roi_filter(roi_set: Any, *, path: Optional[str] = None,
                      on_missing: Optional[str] = None,
                      verbose: bool = True) -> str:
    """Measure only inside these regions, here and in every worker process.

    Three things happen, and the third is the one that matters:

    1. the ROI is written to disk if it is not already a path — a ``spawn``
       worker can only reach it through the file system;
    2. :data:`ROI_ENV_VAR` (and :data:`ON_MISSING_ENV_VAR`) are set;
    3. ``spacr.roi:install`` is APPENDED to
       :data:`~spacr.measure_hooks.HOOKS_ENV_VAR` — appended, not assigned,
       because the illumination correction may already be in there — and the
       registry is then consulted so this process installs the filter through
       that same environment route.

    :param roi_set: a :class:`RoiSet`, a :class:`spacr.layers.ShapesLayer`, or
        the path of a saved set.
    :param path: where to write ``roi_set`` when it is not already a path.
        Defaults to ``./roi/measure_roi.json``.
    :param on_missing: override the set's own rule for uncovered fields.
    :param verbose: print what was enabled and whether the workers will see it.
    :returns: the registry key the filter is registered under.
    :raises RoiError: when the ROI cannot be saved or loaded.
    """
    if isinstance(roi_set, (str, os.PathLike)):
        roi_path = os.path.abspath(str(roi_set))
        RoiSet.load(roi_path)  # fail here, not in a worker
    else:
        if not isinstance(roi_set, RoiSet):
            roi_set = RoiSet.from_shapes_layer(roi_set)
        if on_missing is not None:
            roi_set = replace(roi_set, on_missing=on_missing)
        roi_path = roi_set.save(path or os.path.join(os.getcwd(), 'roi',
                                                     'measure_roi.json'))

    os.environ[ROI_ENV_VAR] = roi_path
    if on_missing is not None:
        os.environ[ON_MISSING_ENV_VAR] = str(on_missing)
    entries = _env_entries(os.environ.get(HOOKS_ENV_VAR, ''))
    if INSTALLER_ENTRY not in entries:
        entries.append(INSTALLER_ENTRY)
    os.environ[HOOKS_ENV_VAR] = ','.join(entries)

    # Consulting the registry runs the environment installers, which is how
    # this process ends up with a hook tagged 'env' — the same tag a worker
    # gets, and the one measure_crop's start-method warning knows not to shout
    # about. The variable is only read once per process, so if it has already
    # been read this does nothing and we install directly instead.
    if HOOK_NAME not in [entry.name for entry in region_filter_hooks()]:
        install()
    if verbose:
        ok, message = worker_delivery_status()
        print(f"ROI filter ENABLED from {roi_path}")
        print(('  ' if ok else '  WARNING: ') + message)
    return HOOK_NAME


def disable_roi_filter() -> bool:
    """Measure whole fields again, here and for any worker started afterwards.

    Unregisters the filter and removes this module's entry from
    ``SPACR_MEASURE_HOOKS`` — leaving any other extension's entries alone —
    plus the two ROI variables.

    :returns: True if a filter was registered and has been removed.
    """
    removed = unregister_region_filter_hook(HOOK_NAME)
    entries = [item for item in _env_entries(os.environ.get(HOOKS_ENV_VAR, ''))
               if item != INSTALLER_ENTRY]
    if entries:
        os.environ[HOOKS_ENV_VAR] = ','.join(entries)
    else:
        os.environ.pop(HOOKS_ENV_VAR, None)
    os.environ.pop(ROI_ENV_VAR, None)
    os.environ.pop(ON_MISSING_ENV_VAR, None)
    return removed


def worker_delivery_status(start_method: Optional[str] = None
                           ) -> Tuple[bool, str]:
    """Whether the ROI will actually reach ``measure_crop``'s workers.

    The failure this answers is silent by construction: a filter registered
    only in the parent process is a no-op in every ``spawn`` worker, the run
    completes, and every object in the field is measured while the user
    believes only the ROI was.

    :param start_method: the pool start method to judge against. Defaults to
        whatever ``SPACR_START_METHOD`` selects, falling back to the platform
        default — i.e. what :func:`spacr.measure.measure_crop` will use.
    :returns: ``(ok, message)``. ``ok`` is False whenever a field could be
        measured whole without anything saying so.
    """
    if start_method is None:
        import multiprocessing as mp
        start_method = (os.environ.get('SPACR_START_METHOD', '').strip().lower()
                        or mp.get_start_method())
    registered = {entry.name for entry in region_filter_hooks()}
    if HOOK_NAME not in registered:
        return False, ('the ROI filter is not registered in this process; '
                       'measure_crop will measure every object in the field.')
    in_env = INSTALLER_ENTRY in _env_entries(os.environ.get(HOOKS_ENV_VAR, ''))
    roi_path = os.environ.get(ROI_ENV_VAR, '').strip()
    if in_env and roi_path and os.path.isfile(roi_path):
        return True, (f"workers install it themselves from {HOOKS_ENV_VAR}="
                      f"'{INSTALLER_ENTRY}' and {ROI_ENV_VAR}='{roi_path}', so "
                      f"a '{start_method}' pool is covered.")
    if start_method == 'fork':
        return True, (f"a 'fork' pool inherits this process's registry, so the "
                      f"ROI reaches the workers — but {HOOKS_ENV_VAR} does not "
                      f"name {INSTALLER_ENTRY}, so it would NOT survive "
                      f"SPACR_START_METHOD=spawn.")
    return False, (f"the ROI is registered in this process only and the pool "
                   f"starts workers with '{start_method}', which does not "
                   f"inherit it: every object in every field would be "
                   f"measured. Call enable_roi_filter(), which sets "
                   f"{HOOKS_ENV_VAR} and {ROI_ENV_VAR}.")
