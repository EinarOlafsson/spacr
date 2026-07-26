"""Foreign-data importer — somebody else's images, masks and measurements
turned into a working spaCR project.

The problem
-----------

A collaborator sends a folder of TIFFs, a folder of label images, and a
``results.csv`` out of CellProfiler / Fiji / QuPath / their own script.
None of it is shaped like a spaCR experiment, and none of their column
names mean what spaCR's mean. What this module produces is a real
project root::

    dst/
      images/                       Yokogawa-named TIFFs + conversion_map.csv
      stack/<plate>_<well>_<field>.npy
      masks/cell_mask_stack/<plate>_<well>_<field>.npy
      masks/nucleus_mask_stack/…
      merged/<plate>_<well>_<field>.npy
      measurements/measurements.db
      crops/<object>/…
      column_map.csv                the mapping that was actually applied

…which Mask, Measure, Annotate, Classify, the Plate Viewer and the
Database Browser all read without knowing it was imported.

What is actually hard
---------------------

Moving files is the easy half, and :mod:`spacr.convert` already does it:
:func:`spacr.convert.scan` / :func:`~spacr.convert.plan` /
:func:`~spacr.convert.convert` handle the image formats, the Yokogawa
naming, the well assignment and the ``conversion_map.csv`` that maps
every converted name back to the file it came from. This module reuses
all of it rather than growing a second copy.

The hard half is mapping an **arbitrary external schema** onto spaCR's,
and the four ways that goes silently wrong:

1. **A guessed mapping applied without review.** Their ``Area`` in µm²
   written into spaCR's ``cell_area`` in px² is a number that is wrong by
   a factor of a few hundred and looks completely plausible. So
   :func:`infer_column_map` only ever *proposes*; the proposal is a file
   the user edits (:func:`save_column_map` / :func:`load_column_map`);
   and :func:`run_import` applies what was agreed, nothing else.
2. **Units.** spaCR's intensities are raw uncalibrated counts, and its
   geometry is px²/px for a 2-D run — but a 3-D run measures volumes,
   in µm³ when it was given voxel_size_z_um/voxel_size_xy_um, under the
   same column names; the row's ``measurement_units`` says which, and
   :mod:`spacr.feature_dict` resolves it per table. A foreign table in
   µm needs a scale factor unless the target rows are µm too, and every
   mapping therefore carries ``unit_in`` / ``unit_out``. When a
   conversion is declared but the pixel size is unknown, the value is
   **not** multiplied by 1.0 and pretended to be pixels: the column is
   redirected to the ``foreign_`` prefix, recorded with its own unit and
   ``calibrated = 0``, and named in the plan and the summary.
3. **Columns that could not be mapped.** They are listed by name in the
   plan, in :meth:`ImportResult.summary`, and in the ``foreign_columns``
   table — and they are still *imported*, under the ``foreign_`` prefix.
   Dropping a column the user cared about, quietly, is the failure mode
   this exists to prevent.
4. **Name collisions.** A foreign column called ``cell_area`` that means
   something else would corrupt every downstream analysis with no error
   at all. Targets are checked against :func:`spacr.feature_dict.parse_column`
   and against spaCR's reserved key columns; a collision is a
   :class:`Conflict` that either refuses the import (``on_conflict='refuse'``,
   the default) or renames the column (``on_conflict='rename'``) — never
   an overwrite.

Object identity is the join
---------------------------

Their measurement rows have to line up with the objects in their masks.
The key is ``(field, object label)``: an ``image_key`` column that says
which image a row belongs to, and a ``label_key`` column holding the
integer label of the object in that image's mask. Both are stated in the
plan, and both are *verified* against the label images before anything is
written — :class:`JoinReport` counts the rows that resolve to no field,
the rows whose label is in no mask, and the mask objects that no row
measures. An import where 40% of the rows match nothing is broken, and it
says so with the number rather than quietly inner-joining it away.

Typical use::

    from spacr import foreign as fg

    plan = fg.plan_import('/data/theirs/images',
                          {'cell': '/data/theirs/cell_masks'},
                          '/data/theirs/results.csv',
                          um_per_px=0.65)
    print(fg.format_plan(plan))          # nothing has been written
    fg.save_column_map(plan, '/data/column_map.csv')
    # …the user edits that file…
    plan = fg.plan_import(..., column_maps=fg.load_column_map('/data/column_map.csv'))
    result = fg.run_import(plan, '/data/imported')
    print(result.summary())
"""
from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping as TMapping,
                    Optional, Sequence, Set, Tuple, Union)

import numpy as np
import pandas as pd

from . import convert as cv
from . import crops as cropping
from . import feature_dict as fdict
from .errors import ConfigurationError, RunLedger

__all__ = [
    'ColumnMap',
    'ResolvedColumn',
    'Conflict',
    'MaskMapping',
    'PairingReport',
    'JoinReport',
    'ImportPlan',
    'ImportResult',
    'infer_column_map',
    'load_column_map',
    'save_column_map',
    'read_measurements',
    'plan_import',
    'run_import',
    'format_plan',
    'import_project',
    'default_settings',
    'is_spacr_name',
    'FOREIGN_PREFIX',
    'COLUMN_MAP_COLUMNS',
    'COLUMN_MAP_FILENAME',
    'FOREIGN_COLUMNS_TABLE',
    'IMPORT_TABLE',
    'RESERVED_COLUMNS',
    'TRANSFORMS',
    'IMAGES_DIRNAME',
    'MASK_SUFFIXES',
    'ON_CONFLICT',
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Prefix every column that is *theirs* rather than spaCR's carries. One
#: glance at a column name answers "did this number come out of spaCR or
#: out of their table", which is the question every reader of an imported
#: database eventually asks.
FOREIGN_PREFIX = 'foreign_'

#: Columns of the editable column-map file, in order.
COLUMN_MAP_COLUMNS: Tuple[str, ...] = (
    'source', 'target', 'transform', 'unit_in', 'unit_out', 'note')

#: Name of the map file written into the destination.
COLUMN_MAP_FILENAME = 'column_map.csv'

#: Table recording, per column, where it came from and what was done to it.
FOREIGN_COLUMNS_TABLE = 'foreign_columns'

#: Table recording the import run itself.
IMPORT_TABLE = 'foreign_import'

#: Sub-folder of the destination holding the converted Yokogawa TIFFs.
IMAGES_DIRNAME = 'images'

#: Columns a foreign mapping may never target: they are the join keys the
#: whole database is addressed by, and a foreign value in one of them
#: does not corrupt a measurement, it corrupts the index.
RESERVED_COLUMNS: Tuple[str, ...] = (
    'object_label', 'plateID', 'rowID', 'columnID', 'fieldID',
    'prc', 'prcf', 'prcfo', 'file_name', 'path_name', 'timeID', 'cell_id')

#: Accepted ``ColumnMap.transform`` names, plus the literal ``*k`` / ``/k``.
TRANSFORMS: Tuple[str, ...] = ('identity', 'length', 'area', 'volume')

#: How many powers of the pixel size each transform applies.
_POWERS: Dict[str, int] = {'identity': 0, 'length': 1, 'area': 2, 'volume': 3}

#: What to do about a target that collides with a spaCR name.
ON_CONFLICT: Tuple[str, ...] = ('refuse', 'rename')

#: Tokens stripped off a mask filename stem before it is matched to an
#: image field. ``fov01_cell_mask.tif`` has to find ``fov01_C1.tif``.
MASK_SUFFIXES: Tuple[str, ...] = (
    'cp_masks', 'cp_mask', 'masks', 'mask', 'segmentation', 'segmented',
    'seg', 'labels', 'label', 'labelmap', 'labelled', 'labeled', 'objects',
    'outlines', 'cell', 'cells', 'nucleus', 'nuclei', 'nuc', 'pathogen',
    'pathogens', 'parasite', 'parasites', 'organelle', 'organelles')

#: Column names that usually identify which image a measurement row is from.
_IMAGE_KEY_HINTS: Tuple[str, ...] = (
    'imagenumber', 'image_number', 'image_id', 'imageid', 'image',
    'filename', 'file_name', 'file', 'image_filename', 'imagefilename',
    'url', 'field', 'fieldname', 'fov', 'site', 'frame', 'well_field',
    'metadata_filename', 'metadata_field', 'metadata_fov', 'metadata_site')

#: Column names that usually hold the object's integer label in the mask.
_LABEL_KEY_HINTS: Tuple[str, ...] = (
    'objectnumber', 'object_number', 'object_label', 'objectlabel',
    'object_id', 'objectid', 'label', 'labels', 'mask_label', 'cell_label',
    'roi', 'roi_id', 'object', 'id')

#: Unit spellings normalised to one token each. Anything not here is kept
#: verbatim so a unit we do not know is still *recorded*, never blanked.
_UNIT_ALIASES: Dict[str, str] = {
    '': '', 'none': '', 'na': '', 'n/a': '', 'unknown': '', '-': '',
    'um': 'um', 'micron': 'um', 'microns': 'um', 'micrometer': 'um',
    'micrometre': 'um', 'micrometers': 'um', 'micrometres': 'um',
    'um^2': 'um^2', 'um2': 'um^2', 'sq_um': 'um^2', 'square_micron': 'um^2',
    'square_microns': 'um^2',
    'um^3': 'um^3', 'um3': 'um^3',
    'px': 'px', 'pixel': 'px', 'pixels': 'px', 'pix': 'px',
    'px^2': 'px^2', 'px2': 'px^2', 'pixel^2': 'px^2', 'pixels^2': 'px^2',
    'sq_px': 'px^2', 'pixel_count': 'px^2',
    'px^3': 'px^3', 'px3': 'px^3', 'voxel': 'px^3', 'voxels': 'px^3',
    'au': 'au', 'a.u.': 'au', 'arbitrary': 'au', 'arbitrary_units': 'au',
    'counts': 'counts', 'count': 'counts', 'adu': 'counts',
}

#: Non-ASCII characters that appear in real column headers and have an
#: obvious ASCII reading. Applied before sanitising, so ``Area (µm²)``
#: becomes ``area_um2`` and not ``area``.
_TRANSLITERATE: Tuple[Tuple[str, str], ...] = (
    ('²', '2'), ('³', '3'), ('µ', 'u'), ('μ', 'u'), ('Å', 'angstrom'),
    ('°', 'deg'), ('%', 'pct'),
)

#: ``*2.5`` / ``/0.65`` / ``x2`` — an explicit numeric factor.
_LITERAL_RE = re.compile(
    r'^\s*([*/xX×])\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$')

#: Unit written into a column header, e.g. ``Area (um^2)`` or ``length_um``.
_TRAILING_UNIT_RE = re.compile(
    r'(?ix)(?:[\[(]\s*(?P<b>[^\])]{1,12}?)\s*[\])]\s*$)'
    r'|(?:[_\-\s](?P<s>um\^?2|um2|um\^?3|um3|um|px\^?2|px2|px|nm|counts|au)\s*$)')

#: Name fragments that say what physical family a column belongs to.
_AREA_HINTS = ('area', 'footprint')
_LENGTH_HINTS = ('perimeter', 'diameter', 'radius', 'length', 'width',
                 'height', 'axis', 'distance', 'thickness', 'feret')
_VOLUME_HINTS = ('volume',)

#: spaCR's own units, as the default ``unit_out`` for each transform.
_SPACR_UNIT: Dict[str, str] = {
    'identity': '', 'length': 'px', 'area': 'px^2', 'volume': 'px^3'}

#: Extensions :func:`read_measurements` knows how to open.
_TABLE_READERS: Dict[str, str] = {
    '.csv': 'csv', '.txt': 'csv', '.tsv': 'tsv', '.tab': 'tsv',
    '.xlsx': 'excel', '.xls': 'excel', '.xlsm': 'excel',
    '.db': 'sqlite', '.sqlite': 'sqlite', '.sqlite3': 'sqlite',
    '.parquet': 'parquet',
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _transliterate(text: str) -> str:
    """Replace the handful of non-ASCII characters real headers contain."""
    out = str(text)
    for source, replacement in _TRANSLITERATE:
        out = out.replace(source, replacement)
    return out


def _sanitise_column(name: str) -> str:
    """Reduce an arbitrary header to a safe lower-case SQL identifier.

    ``Area (µm²)`` -> ``area_um2``. Deliberately lossy and deliberately
    reversible only through the map file, which records the original
    ``source`` next to every target.
    """
    text = _transliterate(name)
    text = re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_').lower()
    if not text:
        text = 'column'
    if text[0].isdigit():
        text = f'c{text}'
    return text


def _norm_unit(unit: Any) -> str:
    """Normalise a unit spelling. Unknown units are kept, never blanked."""
    text = _transliterate('' if unit is None else str(unit)).strip().lower()
    text = text.replace(' ', '').replace('**', '^')
    if text in _UNIT_ALIASES:
        return _UNIT_ALIASES[text]
    return text


def _unit_family(unit: str) -> Tuple[Optional[str], int]:
    """Return ``(family, power)`` for a normalised unit.

    ``('metric', 2)`` for ``um^2``, ``('pixel', 1)`` for ``px``,
    ``(None, 0)`` for an undeclared unit and ``('other', 0)`` for one this
    module has no opinion about (``counts``, ``au``, …).
    """
    if not unit:
        return None, 0
    match = re.fullmatch(r'(um|px)(?:\^(\d))?', unit)
    if match is None:
        return 'other', 0
    family = 'metric' if match.group(1) == 'um' else 'pixel'
    return family, int(match.group(2) or 1)


def is_spacr_name(name: str) -> bool:
    """True when ``name`` is a column spaCR itself writes.

    Delegated to :func:`spacr.feature_dict.parse_column` rather than a
    second parser: that module already implements the whole grammar
    (``<object>_channel_<i>_<stat>``, the radial-distribution and
    organelle-summary forms, the merge suffixes) and knows every metadata
    column. ``family == 'unknown'`` is exactly "spaCR would not write
    this".
    """
    try:
        return fdict.parse_column(str(name)).family != 'unknown'
    except Exception:
        return False


def _pretty_unit(unit: str) -> str:
    """A unit for humans; ``''`` becomes an explicit 'not declared'."""
    return unit if unit else 'not declared'


def _stem_of(plate: str, well: str, field: Any) -> str:
    """The ``plate1_A01_3`` stem every per-field artifact is named by.

    Matches :func:`spacr.utils._map_wells`, which splits on ``_`` and
    reads plate / well / field from positions 0, 1 and 2 — so a merged
    array named this way produces the same ``prcf`` as a native run.
    """
    return f'{plate}_{well}_{int(field)}'


def _unique(name: str, taken: Set[str]) -> str:
    """Return ``name``, or ``name_2`` / ``name_3`` … if it is already used."""
    if name not in taken:
        return name
    index = 2
    while f'{name}_{index}' in taken:
        index += 1
    return f'{name}_{index}'


# ---------------------------------------------------------------------------
# The column mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnMap:
    """One foreign column and what it becomes. The unit of review.

    A :class:`ColumnMap` is a *proposal* until a human has looked at it.
    :func:`infer_column_map` writes them, :func:`save_column_map` puts
    them in a CSV the user edits, and :func:`run_import` applies exactly
    what comes back from :func:`load_column_map` — there is no path by
    which an inferred mapping reaches the database unreviewed.

    :ivar source: the column name in their table, verbatim.
    :ivar target: the column name in ``measurements.db``. Empty means
        "not mapped", which is reported rather than dropped.
    :ivar transform: ``'identity'`` (copy), ``'length'`` / ``'area'`` /
        ``'volume'`` (apply the pixel size to the 1st / 2nd / 3rd power),
        or an explicit literal factor such as ``'*0.65'`` or ``'/1000'``.
    :ivar unit_in: the unit their values are in, e.g. ``'um^2'``. Required
        for any non-identity, non-literal transform: converting without
        knowing what you are converting *from* is guessing.
    :ivar unit_out: the unit the stored values are in. Defaults to
        spaCR's own (``px`` / ``px^2`` / ``px^3``) for a scaling transform.
    :ivar note: free text carried into the plan, the map file and the
        ``foreign_columns`` table.
    """

    source: str
    target: str = ''
    transform: str = 'identity'
    unit_in: str = ''
    unit_out: str = ''
    note: str = ''

    # -- parsing -----------------------------------------------------------

    @property
    def literal_factor(self) -> Optional[float]:
        """The explicit numeric factor of a ``'*k'`` / ``'/k'`` transform."""
        match = _LITERAL_RE.match(str(self.transform or ''))
        if match is None:
            return None
        value = float(match.group(2))
        if match.group(1) == '/':
            if value == 0:
                return None
            return 1.0 / value
        return value

    @property
    def power(self) -> int:
        """How many powers of the pixel size this transform applies."""
        return _POWERS.get(str(self.transform or 'identity').strip().lower(), 0)

    @property
    def is_literal(self) -> bool:
        """True when the transform carries its own number."""
        return _LITERAL_RE.match(str(self.transform or '')) is not None

    @property
    def normalised_unit_in(self) -> str:
        """:attr:`unit_in`, normalised."""
        return _norm_unit(self.unit_in)

    @property
    def normalised_unit_out(self) -> str:
        """:attr:`unit_out`, normalised, defaulted to spaCR's own unit."""
        declared = _norm_unit(self.unit_out)
        if declared:
            return declared
        return _SPACR_UNIT.get(
            str(self.transform or 'identity').strip().lower(), '')

    @property
    def declares_conversion(self) -> bool:
        """True when the two units are different scales of one quantity.

        The check that catches a half-filled row: units saying ``um^2 ->
        px^2`` with ``transform='identity'`` is not a copy, it is a
        conversion somebody forgot to declare.
        """
        family_in, power_in = _unit_family(self.normalised_unit_in)
        family_out, power_out = _unit_family(self.normalised_unit_out)
        if family_in is None or family_out is None:
            return False
        if family_in == 'other' or family_out == 'other':
            return family_in != family_out or self.normalised_unit_in != self.normalised_unit_out
        return family_in != family_out or power_in != power_out

    # -- resolution --------------------------------------------------------

    def resolve(self, um_per_px: Optional[float] = None) -> Tuple[Optional[float], str]:
        """Return ``(factor, reason)`` for this mapping.

        ``factor`` is None when the conversion cannot be performed — and
        that is the whole point of returning it separately from a number.
        ``reason`` is empty on success and names the missing piece
        otherwise, in words that go straight into the plan.

        :param um_per_px: micrometres per pixel. None means unknown, and
            an unknown pixel size never silently becomes 1.0.
        """
        literal = self.literal_factor
        if literal is not None:
            return literal, ''
        name = str(self.transform or 'identity').strip().lower()
        if name not in TRANSFORMS:
            # Includes '/0', whose literal_factor is None because dividing
            # by zero is not a unit conversion.
            return None, (f'unknown transform {self.transform!r}; expected '
                          f'one of {", ".join(TRANSFORMS)} or a literal '
                          f'factor like "*0.65"')
        power = self.power
        if power == 0:
            if self.declares_conversion:
                return None, (
                    f'units say {_pretty_unit(self.normalised_unit_in)} -> '
                    f'{_pretty_unit(self.normalised_unit_out)} but the '
                    f'transform is "identity", which would copy the values '
                    f'unchanged into a column labelled as converted')
            return 1.0, ''

        family_in, power_in = _unit_family(self.normalised_unit_in)
        if family_in is None:
            return None, (f'transform {name!r} needs unit_in, and none was '
                          f'declared — the direction of the conversion is not '
                          f'guessable')
        if family_in == 'other':
            return None, (f'unit_in {self.normalised_unit_in!r} is not a '
                          f'length/area/volume unit, so a {name} conversion '
                          f'has no meaning')
        family_out, power_out = _unit_family(self.normalised_unit_out)
        if family_out is None or family_out == 'other':
            return None, (f'transform {name!r} needs unit_out to be a pixel or '
                          f'micrometre unit; got {_pretty_unit(self.normalised_unit_out)}')
        if family_in == family_out:
            return None, (f'unit_in and unit_out are both '
                          f'{family_in} units ({self.normalised_unit_in} -> '
                          f'{self.normalised_unit_out}), so transform '
                          f'{name!r} has nothing to convert')
        if power_in != power or power_out != power:
            return None, (f'transform {name!r} is a power-{power} conversion '
                          f'but the units are {self.normalised_unit_in} -> '
                          f'{self.normalised_unit_out}')
        if um_per_px is None:
            return None, ('no pixel size was given (um_per_px), so the '
                          'micrometre values cannot be expressed in pixels')
        try:
            scale = float(um_per_px)
        except (TypeError, ValueError):
            return None, f'um_per_px={um_per_px!r} is not a number'
        if not np.isfinite(scale) or scale <= 0:
            return None, f'um_per_px={um_per_px!r} must be a positive number'
        if family_in == 'metric':
            # px = um / (um per px)
            return float(scale ** -power), ''
        return float(scale ** power), ''

    # -- serialisation -----------------------------------------------------

    def to_row(self) -> Dict[str, str]:
        """This mapping as one row of the column-map file."""
        return {
            'source': str(self.source),
            'target': str(self.target or ''),
            'transform': str(self.transform or 'identity'),
            'unit_in': str(self.unit_in or ''),
            'unit_out': str(self.unit_out or ''),
            'note': str(self.note or ''),
        }

    @classmethod
    def from_row(cls, row: TMapping[str, Any]) -> 'ColumnMap':
        """Build a mapping from one row of the column-map file."""
        def _get(key: str) -> str:
            value = row.get(key, '')
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ''
            return str(value).strip()

        source = _get('source')
        if not source:
            raise ConfigurationError(
                'A column-map row has no "source" column name; every row must '
                'name the foreign column it describes.')
        return cls(source=source, target=_get('target'),
                   transform=_get('transform') or 'identity',
                   unit_in=_get('unit_in'), unit_out=_get('unit_out'),
                   note=_get('note'))


@dataclass(frozen=True)
class ResolvedColumn:
    """A :class:`ColumnMap` after conflicts and units have been settled.

    This — not :class:`ColumnMap` — is what :func:`run_import` executes,
    and what the ``foreign_columns`` table records. Keeping the two apart
    is what makes "the mapping you saved is the mapping that ran" a
    property that can be checked: the mapping is the input, the resolution
    is the derivation, and the derivation is deterministic.

    :ivar mapping: the reviewed :class:`ColumnMap` this came from.
    :ivar target: the column name actually written, after any rename.
    :ivar factor: multiplier applied to every value, or None when the
        values are written unchanged because the conversion could not be
        performed.
    :ivar calibrated: False when the stored values are *not* in
        ``unit_out`` — the flag that stops a µm² column being read as px².
    :ivar unit: the unit the stored values really are in.
    :ivar status: ``'mapped'``, ``'renamed'``, ``'uncalibrated'`` or
        ``'unmapped'``.
    :ivar reason: why, in words, for anything but ``'mapped'``.
    """

    mapping: ColumnMap
    target: str
    factor: Optional[float]
    calibrated: bool
    unit: str
    status: str
    reason: str = ''

    @property
    def source(self) -> str:
        """The foreign column name."""
        return self.mapping.source

    def apply(self, values: 'pd.Series') -> 'pd.Series':
        """Return ``values`` with this resolution's factor applied.

        A non-numeric column is passed through untouched however the
        factor reads: multiplying a string column of treatment names by
        0.65 is not a unit conversion, it is a crash.
        """
        if self.factor is None or self.factor == 1.0:
            return values
        numeric = pd.to_numeric(values, errors='coerce')
        if numeric.notna().sum() == 0:
            return values
        return numeric * float(self.factor)

    def to_record(self, table: str) -> Dict[str, Any]:
        """One row of the ``foreign_columns`` provenance table."""
        return {
            'table': table,
            'column': self.target,
            'origin': 'foreign',
            'source_column': self.source,
            'transform': str(self.mapping.transform or 'identity'),
            'factor': None if self.factor is None else float(self.factor),
            'unit_in': self.mapping.normalised_unit_in,
            'unit_declared': self.mapping.normalised_unit_out,
            'unit': self.unit,
            'calibrated': int(bool(self.calibrated)),
            'status': self.status,
            'reason': self.reason,
            'note': str(self.mapping.note or ''),
        }


@dataclass(frozen=True)
class Conflict:
    """A foreign column whose target would collide with something of spaCR's.

    :ivar kind: ``'reserved'`` (a join key), ``'spacr_name'`` (a column
        spaCR itself writes), ``'duplicate_target'`` (two sources, one
        target) or ``'shadows_spacr'`` (an unmapped column whose own name
        is a spaCR name — carried under the foreign prefix, so not
        blocking, but the user needs to know).
    :ivar blocking: True when the import refuses until it is resolved.
    """

    kind: str
    source: str
    target: str
    detail: str
    blocking: bool = True

    def __str__(self) -> str:
        return f'[{self.kind}] {self.source!r} -> {self.target!r}: {self.detail}'


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _column_unit_hint(name: str) -> str:
    """Read a unit out of a column header, or ``''``.

    ``Area (µm²)`` -> ``um^2``; ``perimeter_um`` -> ``um``. Only a hint:
    it goes into a *proposal* the user reviews, never straight into the
    database.
    """
    text = _transliterate(str(name)).strip()
    match = _TRAILING_UNIT_RE.search(text)
    if match is None:
        return ''
    token = match.group('b') or match.group('s') or ''
    unit = _norm_unit(token)
    family, _power = _unit_family(unit)
    return unit if family in ('metric', 'pixel') else ''


def _column_family_hint(name: str) -> str:
    """Guess whether a column is a length, an area or a volume, by name."""
    text = _sanitise_column(name)
    if any(hint in text for hint in _VOLUME_HINTS):
        return 'volume'
    if any(hint in text for hint in _AREA_HINTS):
        return 'area'
    if any(hint in text for hint in _LENGTH_HINTS):
        return 'length'
    return 'identity'


def _pick_key(columns: Sequence[str], hints: Sequence[str]) -> Optional[str]:
    """Return the first column whose sanitised name matches a hint."""
    lookup = {_sanitise_column(c): c for c in columns}
    for hint in hints:
        if hint in lookup:
            return lookup[hint]
    for hint in hints:
        for sanitised, original in lookup.items():
            if sanitised.endswith('_' + hint) or sanitised.startswith(hint + '_'):
                return original
    return None


def infer_column_map(df: 'pd.DataFrame',
                     image_key: Optional[str] = None,
                     label_key: Optional[str] = None,
                     prefix: str = FOREIGN_PREFIX,
                     skip: Optional[Iterable[str]] = None) -> List[ColumnMap]:
    """Propose a mapping for every column of a foreign measurement table.

    **A proposal, never an application.** Nothing in this module writes a
    database from the output of this function without it having passed
    through :func:`save_column_map` / :func:`load_column_map` or having
    been handed back explicitly — because the one mistake this module
    exists to prevent is an inferred ``Area`` -> ``cell_area`` that nobody
    read.

    The proposal is deliberately conservative:

    * **Nothing is proposed onto a spaCR feature name.** Their ``Area`` is
      not spaCR's ``cell_area``: different segmentation, different
      definition, different unit. Every feature column is proposed as
      ``foreign_<name>``. A user who genuinely wants their column in
      spaCR's slot edits the target and passes
      ``allow_spacr_targets=True`` to :func:`plan_import`, which is a
      decision with a name on it.
    * **A unit read out of the header becomes a declared conversion.**
      ``Area (µm²)`` is proposed as ``transform='area'``,
      ``unit_in='um^2'``, ``unit_out='px^2'`` — which then *needs* a pixel
      size, and says so if there is none.
    * **The join keys are left out**: they are consumed by the join, not
      imported as measurements, and :class:`JoinReport` states them.

    :param df: their measurement table.
    :param image_key: the column identifying the image; inferred from the
        headers when None.
    :param label_key: the column holding the object label; likewise.
    :param prefix: prefix for the proposed targets.
    :param skip: extra columns to leave out of the proposal.
    :returns: one :class:`ColumnMap` per remaining column, in table order.
    """
    columns = [str(c) for c in df.columns]
    if image_key is None:
        image_key = _pick_key(columns, _IMAGE_KEY_HINTS)
    if label_key is None:
        label_key = _pick_key(columns, _LABEL_KEY_HINTS)
    excluded = {str(c) for c in (skip or ())}
    for key in (image_key, label_key):
        if key:
            excluded.add(str(key))

    proposals: List[ColumnMap] = []
    taken: Set[str] = set()
    for column in columns:
        if column in excluded:
            continue
        target = _unique(f'{prefix}{_sanitise_column(column)}', taken)
        taken.add(target)

        unit = _column_unit_hint(column)
        family, _power = _unit_family(unit)
        if family == 'metric':
            transform = _column_family_hint(column)
            if transform == 'identity':
                transform = {1: 'length', 2: 'area', 3: 'volume'}[_power or 1]
            note = (f'unit {unit} read from the column name; spaCR measures in '
                    f'{_SPACR_UNIT[transform]}, so this needs a pixel size')
            proposals.append(ColumnMap(
                source=column, target=target, transform=transform,
                unit_in=unit, unit_out=_SPACR_UNIT[transform], note=note))
            continue
        if family == 'pixel':
            proposals.append(ColumnMap(
                source=column, target=target, transform='identity',
                unit_in=unit, unit_out=unit,
                note=f'unit {unit} read from the column name; already in '
                     f'spaCR\'s units, copied unchanged'))
            continue

        hint = _column_family_hint(column)
        if hint != 'identity':
            note = (f'name suggests a {hint}, but no unit is declared — set '
                    f'unit_in (and transform={hint}) if these are not already '
                    f'in {_SPACR_UNIT[hint]}')
        else:
            note = 'copied unchanged; declare unit_in/unit_out if it needs scaling'
        proposals.append(ColumnMap(source=column, target=target,
                                   transform='identity', note=note))
    return proposals


# ---------------------------------------------------------------------------
# The column-map file
# ---------------------------------------------------------------------------

def save_column_map(plan_or_maps: Union['ImportPlan', Sequence[ColumnMap]],
                    path: str) -> Path:
    """Write the reviewable column-map CSV.

    Six columns — :data:`COLUMN_MAP_COLUMNS` — and a few ``#`` comment
    lines above them explaining what the file is for. It opens in a
    spreadsheet, and the round trip through :func:`load_column_map` is
    exact.

    :param plan_or_maps: an :class:`ImportPlan` or a list of
        :class:`ColumnMap`.
    :param path: destination CSV.
    :returns: the written path.
    """
    maps = (list(plan_or_maps.column_maps)
            if isinstance(plan_or_maps, ImportPlan) else list(plan_or_maps))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', newline='', encoding='utf-8') as handle:
        handle.write(
            '# spaCR foreign column map. Edit "target", "transform",\n'
            '# "unit_in" and "unit_out"; leave "target" empty to leave a\n'
            '# column unmapped (it is still imported, under the\n'
            f'# "{FOREIGN_PREFIX}" prefix, and reported by name).\n'
            '# transform: identity | length | area | volume | *<k> | /<k>\n')
        writer = csv.DictWriter(handle, fieldnames=list(COLUMN_MAP_COLUMNS))
        writer.writeheader()
        for mapping in maps:
            writer.writerow(mapping.to_row())
    return target


def load_column_map(path: str) -> List[ColumnMap]:
    """Read a column-map CSV back.

    :param path: a CSV written by :func:`save_column_map`, possibly edited.
    :returns: the mappings, in file order.
    :raises ConfigurationError: the file is missing, is not a column map,
        or names the same source column twice — which would make "what
        was applied" ambiguous.
    """
    target = Path(path)
    if not target.is_file():
        raise ConfigurationError(f'Column map does not exist: {path}')
    with open(target, newline='', encoding='utf-8') as handle:
        lines = [line for line in handle
                 if not line.lstrip().startswith('#')]
    if not lines:
        raise ConfigurationError(
            f'{path} has no rows — a column map needs at least a header.')
    reader = csv.DictReader(lines)
    header = [h.strip() for h in (reader.fieldnames or [])]
    missing = [c for c in ('source', 'target') if c not in header]
    if missing:
        raise ConfigurationError(
            f'{path} is not a spaCR column map — missing column(s): '
            f'{", ".join(missing)}. Expected: {", ".join(COLUMN_MAP_COLUMNS)}')
    maps: List[ColumnMap] = []
    seen: Set[str] = set()
    for row in reader:
        clean = {k.strip(): v for k, v in row.items() if k is not None}
        if not any(str(v or '').strip() for v in clean.values()):
            continue
        mapping = ColumnMap.from_row(clean)
        if mapping.source in seen:
            raise ConfigurationError(
                f'{path} maps the source column {mapping.source!r} twice; '
                f'which one applies would be undefined. Delete one.')
        seen.add(mapping.source)
        maps.append(mapping)
    return maps


# ---------------------------------------------------------------------------
# Reading their measurement table
# ---------------------------------------------------------------------------

def _sqlite_tables(path: str) -> List[str]:
    """Every user table in a SQLite file, in schema order."""
    connection = sqlite3.connect(str(path), timeout=30)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall()
    finally:
        connection.close()
    return [str(r[0]) for r in rows]


def read_measurements(source: Union[str, 'pd.DataFrame'],
                      table: Optional[str] = None) -> 'pd.DataFrame':
    """Read a foreign measurement table into a DataFrame.

    CSV / TSV / Excel / Parquet / SQLite, or a DataFrame straight through.

    :param source: path, or an already-loaded DataFrame.
    :param table: table name, for a SQLite source. The only table is used
        when there is exactly one; otherwise the name is required, because
        picking one at random is how you import the wrong 40 000 rows.
    :returns: the table.
    :raises ConfigurationError: unreadable, unknown extension, or an
        ambiguous SQLite source.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = str(source)
    if not os.path.isfile(path):
        raise ConfigurationError(f'Measurement table does not exist: {path}')
    ext = os.path.splitext(path)[1].lower()
    kind = _TABLE_READERS.get(ext)
    if kind is None:
        raise ConfigurationError(
            f'{path}: no reader for {ext!r} measurement tables. Supported: '
            f'{", ".join(sorted(_TABLE_READERS))}')
    try:
        if kind == 'csv':
            return pd.read_csv(path)
        if kind == 'tsv':
            return pd.read_csv(path, sep='\t')
        if kind == 'excel':
            return pd.read_excel(path)
        if kind == 'parquet':
            return pd.read_parquet(path)
        tables = _sqlite_tables(path)
        if table is None:
            if len(tables) == 1:
                table = tables[0]
            else:
                raise ConfigurationError(
                    f'{path} holds {len(tables)} tables '
                    f'({", ".join(tables) or "none"}); name the one to import '
                    f'with measurement_table=.')
        if table not in tables:
            raise ConfigurationError(
                f'{path} has no table {table!r}; it has: '
                f'{", ".join(tables) or "none"}')
        connection = sqlite3.connect(path, timeout=30)
        try:
            return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
        finally:
            connection.close()
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f'{path} could not be read as a measurement table: {exc}') from exc


# ---------------------------------------------------------------------------
# Images <-> masks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskMapping:
    """One foreign label image and the field it belongs to.

    :ivar source: the mask file on disk.
    :ivar object_type: ``'cell'`` / ``'nucleus'`` / ``'pathogen'`` /
        ``'organelle'``.
    :ivar stem: the ``plate1_A01_3`` stem it will be written under.
    :ivar match: ``'exact'`` when the mask's field key equalled the
        image's, ``'normalised'`` when it matched only after stripping a
        mask suffix. Recorded because a normalised match is a guess, and
        the plan shows it.
    """

    source: str
    object_type: str
    stem: str
    plate: str
    well: str
    field: int
    source_field: str
    match: str = 'exact'
    labels: Tuple[int, ...] = ()


@dataclass
class PairingReport:
    """Which image fields have which masks, and everything that did not pair.

    The rule is field-for-field: a mask with no image and an image with no
    mask are both reported *per file*. A converter that quietly kept the
    intersection would produce a smaller, perfectly consistent, wrong
    experiment.

    :ivar fields: ``{stem: {object_type: MaskMapping}}`` for the fields
        that have a full set of masks.
    :ivar images_without_masks: ``(image path, object_type)`` per source
        image whose field has no mask of that type.
    :ivar masks_without_images: ``(mask path, object_type)`` per mask that
        matched no image field.
    :ivar unreadable_masks: ``(mask path, reason)``.
    :ivar excluded: stems dropped because they lacked a required mask.
    """

    fields: Dict[str, Dict[str, MaskMapping]] = dc_field(default_factory=dict)
    images_without_masks: List[Tuple[str, str]] = dc_field(default_factory=list)
    masks_without_images: List[Tuple[str, str]] = dc_field(default_factory=list)
    unreadable_masks: List[Tuple[str, str]] = dc_field(default_factory=list)
    excluded: List[str] = dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every image has every mask and every mask an image."""
        return not (self.images_without_masks or self.masks_without_images
                    or self.unreadable_masks)


def _normalise_mask_field(field_key: str, object_type: str,
                          suffixes: Sequence[str]) -> str:
    """Strip mask tokens off a mask's field key so it can meet an image's.

    ``fov01_cell_mask`` -> ``fov01``. Every stripped token is one of
    :data:`MASK_SUFFIXES` (plus the object type's own name), never an
    arbitrary trailing word — dropping ``fov01_treated`` down to ``fov01``
    would merge two different fields.
    """
    tokens = tuple(suffixes) + (object_type,)
    text = str(field_key)
    changed = True
    while changed:
        changed = False
        for token in sorted(tokens, key=len, reverse=True):
            pattern = re.compile(r'(?i)(?:^|[_\-. ])' + re.escape(token) + r'$')
            match = pattern.search(text)
            if match is not None and match.start() > 0:
                text = text[:match.start()]
                changed = True
                break
    return text.strip('_-. ') or str(field_key)


def _source_for(path: str) -> 'cv.SourceImage':
    """Describe one file as a :class:`spacr.convert.SourceImage`.

    Lets a mask be read by path alone, without re-walking the folder it
    came from — :func:`run_import` writes hundreds of masks and a rescan
    per mask is a quadratic directory walk.
    """
    ext = cv._split_ext(os.path.basename(path))[1]
    described = cv._describe(path, ext)
    meta = {k: v for k, v in described.items() if k != 'n_series'}
    meta['ext'] = ext
    return cv.SourceImage(path=path, plate='', well='', field='',
                          z=int(described['n_z']), t=int(described['n_t']),
                          n_channels=int(described['n_c']), meta=meta)


def _read_mask(source: Union[str, 'cv.SourceImage']) -> np.ndarray:
    """Read one label image as a 2-D integer array.

    Goes through :func:`spacr.convert._read_source`, so every format the
    converter can open — TIFF, PNG, ND2, CZI, LIF — is a mask format too,
    with no second reader table to keep in step.
    """
    if not isinstance(source, cv.SourceImage):
        source = _source_for(str(source))
    raw = np.asarray(cv._read_source(source))
    array = np.squeeze(raw)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ConfigurationError(
            f'{source.path}: a mask must be a 2-D label image, got shape '
            f'{raw.shape}')
    return array


def _mask_labels(array: np.ndarray) -> Tuple[int, ...]:
    """Every non-zero label in a mask, sorted."""
    values = np.unique(np.asarray(array))
    return tuple(int(v) for v in values if int(v) > 0)


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

@dataclass
class JoinReport:
    """How their measurement rows line up with the objects in their masks.

    The join key is ``(field, object label)``. Both halves are verified
    against the label images at plan time, and every failure is *counted*:
    an import where 40% of the rows match no mask object is broken, and
    the number is the only thing that makes that visible before the
    database exists.

    :ivar image_key: their column naming the image a row belongs to, or
        ``''`` when the whole table is one field.
    :ivar label_key: their column holding the object's integer label.
    :ivar rows_total: rows in their table.
    :ivar rows_matched: rows whose ``(field, label)`` exists in a mask.
    :ivar unresolved_fields: ``(value, count)`` for image-key values that
        matched no converted image.
    :ivar rows_no_object: ``(stem, count)`` for rows whose label is absent
        from that field's mask.
    :ivar objects_unmeasured: ``(stem, count)`` for mask objects that no
        row measures.
    :ivar ambiguous_keys: image-key spellings that matched more than one
        field and were therefore not used.
    """

    image_key: str = ''
    label_key: str = ''
    object_type: str = 'cell'
    rows_total: int = 0
    rows_matched: int = 0
    unresolved_fields: List[Tuple[str, int]] = dc_field(default_factory=list)
    rows_no_object: List[Tuple[str, int]] = dc_field(default_factory=list)
    objects_unmeasured: List[Tuple[str, int]] = dc_field(default_factory=list)
    ambiguous_keys: List[str] = dc_field(default_factory=list)
    examples: List[str] = dc_field(default_factory=list)

    @property
    def rows_unmatched(self) -> int:
        """Rows that will carry no object — the honest headline number."""
        return max(int(self.rows_total) - int(self.rows_matched), 0)

    @property
    def n_unresolved(self) -> int:
        """Rows whose image key resolved to no field at all."""
        return sum(count for _value, count in self.unresolved_fields)

    @property
    def n_no_object(self) -> int:
        """Rows whose label is in no mask."""
        return sum(count for _stem, count in self.rows_no_object)

    @property
    def n_objects_unmeasured(self) -> int:
        """Mask objects with no measurement row."""
        return sum(count for _stem, count in self.objects_unmeasured)

    @property
    def match_rate(self) -> float:
        """Fraction of their rows that found an object. 1.0 when empty."""
        if not self.rows_total:
            return 1.0
        return float(self.rows_matched) / float(self.rows_total)

    @property
    def key_description(self) -> str:
        """The join, in one sentence, for the plan and the summary."""
        image = self.image_key or '(single field — no image column)'
        label = self.label_key or '(row order)'
        return (f'{self.object_type}: their "{image}" identifies the field, '
                f'their "{label}" is the object label in the '
                f'{self.object_type} mask')

    def summary(self) -> str:
        """A multi-line rendering of every count above."""
        lines = [f'Join key — {self.key_description}',
                 f'  {self.rows_matched}/{self.rows_total} measurement row(s) '
                 f'matched a mask object ({self.match_rate:.1%}).']
        if self.unresolved_fields:
            lines.append(f'  {self.n_unresolved} row(s) name an image that is '
                         f'not in the import:')
            for value, count in self.unresolved_fields[:10]:
                lines.append(f'      {value!r} x{count}')
            if len(self.unresolved_fields) > 10:
                lines.append(f'      … and {len(self.unresolved_fields) - 10} '
                             f'more distinct value(s)')
        if self.rows_no_object:
            lines.append(f'  {self.n_no_object} row(s) carry a label that is '
                         f'in no mask:')
            for stem, count in self.rows_no_object[:10]:
                lines.append(f'      {stem}: {count} row(s)')
            if len(self.rows_no_object) > 10:
                lines.append(f'      … and {len(self.rows_no_object) - 10} '
                             f'more field(s)')
        if self.objects_unmeasured:
            lines.append(f'  {self.n_objects_unmeasured} mask object(s) have '
                         f'no measurement row:')
            for stem, count in self.objects_unmeasured[:10]:
                lines.append(f'      {stem}: {count} object(s)')
            if len(self.objects_unmeasured) > 10:
                lines.append(f'      … and {len(self.objects_unmeasured) - 10} '
                             f'more field(s)')
        if self.ambiguous_keys:
            lines.append(f'  {len(self.ambiguous_keys)} image-key spelling(s) '
                         f'matched more than one field and were not used: '
                         f'{", ".join(self.ambiguous_keys[:10])}')
        for example in self.examples[:5]:
            lines.append(f'      e.g. {example}')
        return '\n'.join(lines)


def _field_aliases(row: TMapping[str, Any]) -> List[str]:
    """Every spelling of "which image" one conversion-map row answers to.

    Their table might identify an image by relative path, by basename, by
    stem, by the stem with the channel token removed, or by their own
    field name. All of them are indexed; a spelling that would point at
    two different fields is dropped and reported rather than picking one.
    """
    aliases: List[str] = []
    relpath = str(row.get('source_relpath') or '')
    source = str(row.get('source') or '')
    for value in (relpath, source, os.path.basename(source),
                  os.path.basename(relpath)):
        if value:
            aliases.append(value)
            aliases.append(value.replace('\\', '/'))
    for value in (os.path.basename(source), os.path.basename(relpath)):
        if not value:
            continue
        stem, _ext = cv._split_ext(value)
        aliases.append(stem)
        field_key, _c, _z, _t = cv._strip_tokens(stem)
        aliases.append(field_key)
    for key in ('source_field', 'target'):
        value = str(row.get(key) or '')
        if value:
            aliases.append(value)
            aliases.append(cv._split_ext(value)[0])
    plate, well = str(row.get('plate') or ''), str(row.get('well') or '')
    field = row.get('field')
    if plate and well and field is not None:
        aliases.append(_stem_of(plate, well, field))
    return [a for a in aliases if a]


def _build_field_index(map_rows: Sequence[TMapping[str, Any]]
                       ) -> Tuple[Dict[str, str], List[str]]:
    """Return ``({alias: stem}, ambiguous aliases)`` for the whole import."""
    index: Dict[str, Set[str]] = {}
    for row in map_rows:
        stem = _stem_of(str(row.get('plate') or ''), str(row.get('well') or ''),
                        row.get('field') or 0)
        for alias in _field_aliases(row):
            index.setdefault(str(alias), set()).add(stem)
            index.setdefault(str(alias).lower(), set()).add(stem)
    resolved = {alias: next(iter(stems)) for alias, stems in index.items()
                if len(stems) == 1}
    ambiguous = sorted(alias for alias, stems in index.items() if len(stems) > 1)
    return resolved, ambiguous


def _resolve_field(value: Any, index: TMapping[str, str]) -> Optional[str]:
    """Map one image-key value onto a field stem, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidates = [text, text.replace('\\', '/'), text.lower()]
    base = os.path.basename(text.replace('\\', '/'))
    if base:
        candidates.extend([base, base.lower()])
        stem, _ext = cv._split_ext(base)
        candidates.extend([stem, stem.lower()])
        field_key, _c, _z, _t = cv._strip_tokens(stem)
        candidates.extend([field_key, field_key.lower()])
    for candidate in candidates:
        stem = index.get(candidate)
        if stem is not None:
            return stem
    return None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class ImportPlan:
    """Everything the import would do, before any of it is done.

    Nothing here has touched the destination. :func:`format_plan` renders
    it, :attr:`ok` says whether :func:`run_import` will accept it, and the
    three lists that matter — :attr:`unmapped`, :attr:`conflicts` and
    :attr:`warnings` — are the ones a user has to read before agreeing.

    :ivar images: the :class:`spacr.convert.ConversionPlan` for their
        image files. Built by :func:`spacr.convert.plan`; this module adds
        no second naming scheme.
    :ivar masks: :class:`PairingReport` — which mask belongs to which
        field, and every file on either side that did not pair.
    :ivar measurements: their table, as read.
    :ivar column_maps: the reviewed mapping that will be applied.
    :ivar unmapped: source columns with no mapping, **by name**. They are
        still imported, under :data:`FOREIGN_PREFIX`.
    :ivar conflicts: :class:`Conflict` entries; a blocking one makes
        :attr:`ok` False.
    :ivar warnings: non-blocking things the user must see — an
        uncalibrated column, a low join match rate, a lossy z handling.
    :ivar resolved: the derived, executable form of ``column_maps``.
    :ivar join: :class:`JoinReport`.
    :ivar base_warnings: the warnings that do *not* come from the column
        mapping (unpaired masks, the join, z handling). Kept apart so
        :meth:`with_column_maps` can rebuild the mapping's own warnings
        without losing them or duplicating them.
    :ivar base_errors: likewise for blocking problems.
    """

    images: 'cv.ConversionPlan'
    masks: PairingReport
    measurements: 'pd.DataFrame'
    column_maps: List[ColumnMap] = dc_field(default_factory=list)
    unmapped: List[str] = dc_field(default_factory=list)
    conflicts: List[Conflict] = dc_field(default_factory=list)
    warnings: List[str] = dc_field(default_factory=list)
    resolved: List[ResolvedColumn] = dc_field(default_factory=list)
    join: JoinReport = dc_field(default_factory=JoinReport)
    errors: List[str] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)
    object_types: List[str] = dc_field(default_factory=list)
    n_channels: int = 0
    mask_dims: Dict[str, int] = dc_field(default_factory=dict)
    um_per_px: Optional[float] = None
    prefix: str = FOREIGN_PREFIX
    on_conflict: str = 'refuse'
    allow_spacr_targets: bool = False
    sources: Dict[str, str] = dc_field(default_factory=dict)
    base_warnings: List[str] = dc_field(default_factory=list)
    base_errors: List[str] = dc_field(default_factory=list)
    proposed: bool = True

    def with_column_maps(self, column_maps: Sequence[ColumnMap], *,
                         um_per_px: Any = '<keep>',
                         on_conflict: Optional[str] = None,
                         allow_spacr_targets: Optional[bool] = None
                         ) -> 'ImportPlan':
        """Return this plan with a different column mapping applied.

        Pure CPU — no folder is rescanned and no file reopened — which is
        what lets a GUI re-run the conflict and unit checks on every
        keystroke in the mapping table. The join, the pairing and the
        image plan are carried over untouched, because none of them
        depends on how the columns are named.

        :param column_maps: the mapping to apply instead.
        :param um_per_px: a new pixel size; omitted keeps the plan's.
        :param on_conflict: ``'refuse'`` / ``'rename'``; omitted keeps.
        :param allow_spacr_targets: omitted keeps.
        :returns: a new :class:`ImportPlan`.
        """
        scale = self.um_per_px if um_per_px == '<keep>' else um_per_px
        conflict_mode = on_conflict or self.on_conflict
        allow = (self.allow_spacr_targets if allow_spacr_targets is None
                 else bool(allow_spacr_targets))

        columns = [str(c) for c in self.measurements.columns]
        known = set(columns)
        consumed = {k for k in (self.join.image_key, self.join.label_key) if k}
        maps = list(column_maps)
        named = {m.source for m in maps}
        errors = list(self.base_errors)
        for source in sorted(named - known):
            errors.append(
                f'The column map names {source!r}, which is not a column of '
                f'the measurement table. Columns present: '
                f'{", ".join(columns[:20])}{"…" if len(columns) > 20 else ""}')
        maps = [m for m in maps if m.source in known]
        no_entry = [c for c in columns if c not in named and c not in consumed]

        resolved, conflicts, resolve_warnings = _resolve_columns(
            maps, no_entry, scale, self.prefix, conflict_mode, allow)
        # "Unmapped" covers both ways a column ends up undecided: no row in
        # the map file at all, and a row whose target was left blank. They
        # are the same thing to the user, so they are one list.
        status = {r.source: r.status for r in resolved}
        unmapped = [c for c in columns if status.get(c) == 'unmapped']

        warnings = list(self.base_warnings)
        if unmapped:
            warnings.append(
                f'{len(unmapped)} column(s) have no mapping and are imported '
                f'verbatim under the "{self.prefix}" prefix, with no unit '
                f'recorded: {", ".join(unmapped)}')
        warnings.extend(resolve_warnings)

        return ImportPlan(
            images=self.images, masks=self.masks,
            measurements=self.measurements, column_maps=maps,
            unmapped=unmapped, conflicts=conflicts, warnings=warnings,
            resolved=resolved, join=self.join, errors=errors,
            notes=list(self.notes), object_types=list(self.object_types),
            n_channels=self.n_channels, mask_dims=dict(self.mask_dims),
            um_per_px=scale, prefix=self.prefix, on_conflict=conflict_mode,
            allow_spacr_targets=allow, sources=dict(self.sources),
            base_warnings=list(self.base_warnings),
            base_errors=list(self.base_errors), proposed=False)

    @property
    def blocking_conflicts(self) -> List[Conflict]:
        """Conflicts that stop the import."""
        return [c for c in self.conflicts if c.blocking]

    @property
    def ok(self) -> bool:
        """True when :func:`run_import` will accept this plan."""
        return (self.images.ok and not self.errors
                and not self.blocking_conflicts)

    @property
    def stems(self) -> List[str]:
        """Field stems that will be imported, sorted."""
        return sorted(self.masks.fields)

    @property
    def uncalibrated(self) -> List[ResolvedColumn]:
        """Columns whose values are not in the unit they were meant to be."""
        return [r for r in self.resolved if not r.calibrated]

    def targets(self) -> List[str]:
        """Every column name the import will write, in order."""
        return [r.target for r in self.resolved]

    def target_for(self, source: str) -> str:
        """The column name ``source`` will actually be written under."""
        for resolution in self.resolved:
            if resolution.source == str(source):
                return resolution.target
        return ''


def _resolve_columns(column_maps: Sequence[ColumnMap],
                     unmapped: Sequence[str],
                     um_per_px: Optional[float],
                     prefix: str,
                     on_conflict: str,
                     allow_spacr_targets: bool,
                     ) -> Tuple[List[ResolvedColumn], List[Conflict], List[str]]:
    """Turn reviewed mappings into executable ones, settling every collision.

    Returns ``(resolved, conflicts, warnings)``. Deterministic: the same
    mappings and the same pixel size always produce the same resolution,
    which is what makes "run_import applies exactly what was saved"
    checkable.
    """
    resolved: List[ResolvedColumn] = []
    conflicts: List[Conflict] = []
    warnings: List[str] = []
    taken: Set[str] = set()

    def _foreign(source: str) -> str:
        return _unique(f'{prefix}{_sanitise_column(source)}', taken)

    def _note_shadow(source: str, target: str) -> None:
        """Record that their column is *named* like one of spaCR's.

        Not blocking — the value lands under the foreign prefix, so
        nothing is overwritten — but a user reading ``cell_area`` in the
        source table and ``cell_area`` in a spaCR database has to be told
        they are two different numbers.
        """
        if not is_spacr_name(source) or target == source:
            return
        conflicts.append(Conflict(
            'shadows_spacr', source, target,
            f'their column is literally called {source!r}, which is also a '
            f'spaCR {fdict.parse_column(source).family} column. It is imported '
            f'as {target!r}, so nothing is overwritten — but the two are not '
            f'the same measurement',
            blocking=False))

    for mapping in column_maps:
        source = mapping.source
        target = (mapping.target or '').strip()
        status = 'mapped'
        reason = ''

        if not target:
            target = _foreign(source)
            status = 'unmapped'
            reason = ('no target was given in the column map; imported under '
                      'the foreign prefix rather than dropped')

        # -- reserved key columns: never, under any setting ----------------
        if target in RESERVED_COLUMNS:
            detail = (f'{target!r} is one of spaCR\'s key columns; a foreign '
                      f'value there does not corrupt a measurement, it '
                      f'corrupts the index every table is joined on')
            renamed = _foreign(source)
            conflicts.append(Conflict('reserved', source, target, detail,
                                      blocking=(on_conflict != 'rename')))
            warnings.append(f'{source!r} was renamed {target!r} -> '
                            f'{renamed!r}: {detail}.')
            target, status, reason = renamed, 'renamed', detail

        # -- a name spaCR itself writes ------------------------------------
        elif is_spacr_name(target) and not allow_spacr_targets:
            entry = fdict.parse_column(target)
            detail = (f'{target!r} is a spaCR {entry.family} column '
                      f'(unit: {entry.unit or "n/a"}); writing their values '
                      f'there would silently replace spaCR\'s own meaning of '
                      f'that name. Pass allow_spacr_targets=True if that is '
                      f'genuinely what you want')
            renamed = _foreign(source)
            conflicts.append(Conflict('spacr_name', source, target, detail,
                                      blocking=(on_conflict != 'rename')))
            if on_conflict == 'rename':
                warnings.append(f'{source!r} was renamed {target!r} -> '
                                f'{renamed!r}: {detail}.')
            target, status, reason = renamed, 'renamed', detail

        # -- two sources, one target ---------------------------------------
        if target in taken:
            renamed = _foreign(source)
            detail = (f'another column already maps to {target!r}; two '
                      f'different measurements cannot share one column')
            conflicts.append(Conflict('duplicate_target', source, target,
                                      detail,
                                      blocking=(on_conflict != 'rename')))
            if on_conflict == 'rename':
                warnings.append(f'{source!r} was renamed {target!r} -> '
                                f'{renamed!r}: {detail}.')
            target, status, reason = renamed, 'renamed', detail

        taken.add(target)

        # -- units ----------------------------------------------------------
        factor, problem = mapping.resolve(um_per_px)
        unit = mapping.normalised_unit_out
        calibrated = True
        if problem:
            calibrated = False
            factor = None
            unit = mapping.normalised_unit_in
            spacr_target = target
            if not target.startswith(prefix):
                target = _unique(f'{prefix}{_sanitise_column(source)}', taken)
                taken.add(target)
            reason = (f'{problem}; the values are stored UNCONVERTED, in '
                      f'{_pretty_unit(unit)}, under {target!r}')
            warnings.append(
                f'{source!r} is UNCALIBRATED: {problem}. Its values are '
                f'stored unchanged, in {_pretty_unit(unit)}, as {target!r} '
                f'(not {spacr_target!r}), and foreign_columns records '
                f'calibrated = 0.')
            status = 'uncalibrated'

        _note_shadow(source, target)
        resolved.append(ResolvedColumn(
            mapping=mapping, target=target, factor=factor,
            calibrated=calibrated, unit=unit, status=status, reason=reason))

    # -- columns nobody mapped at all --------------------------------------
    for source in unmapped:
        target = _foreign(source)
        taken.add(target)
        reason = ('this column has no entry in the column map; it is imported '
                  'verbatim under the foreign prefix so it is not lost, but '
                  'nothing is known about its unit')
        _note_shadow(source, target)
        resolved.append(ResolvedColumn(
            mapping=ColumnMap(source=source, target='', transform='identity',
                              note='not present in the column map'),
            target=target, factor=1.0, calibrated=False, unit='',
            status='unmapped', reason=reason))

    return resolved, conflicts, warnings


def _mask_folders(masks: Union[str, TMapping[str, str], Sequence[Any], None]
                  ) -> Dict[str, str]:
    """Normalise the ``masks`` argument to ``{object_type: folder}``."""
    if masks is None:
        return {}
    if isinstance(masks, (str, os.PathLike)):
        return {'cell': str(masks)}
    if isinstance(masks, TMapping):
        pairs = list(masks.items())
    else:
        pairs = [tuple(item) for item in masks]
    folders: Dict[str, str] = {}
    for object_type, folder in pairs:
        name = str(object_type)
        if name not in cropping.MASK_PLANE_ORDER:
            raise ConfigurationError(
                f'Unknown mask object type {name!r}; spaCR\'s merged arrays '
                f'hold planes for {", ".join(cropping.MASK_PLANE_ORDER)} in '
                f'that order (see spacr.crops.MASK_PLANE_ORDER).')
        folders[name] = str(folder)
    return {name: folders[name] for name in cropping.MASK_PLANE_ORDER
            if name in folders}


def plan_import(images: str,
                masks: Union[str, TMapping[str, str], Sequence[Any], None],
                measurements: Union[str, 'pd.DataFrame'],
                *,
                layout: str = 'auto',
                z_handling: str = cv.Z_MAX,
                plate_naming: str = 'index',
                mask_layout: Optional[str] = None,
                mask_suffixes: Optional[Sequence[str]] = None,
                measurement_table: Optional[str] = None,
                measurement_object: Optional[str] = None,
                image_key: Optional[str] = None,
                label_key: Optional[str] = None,
                column_maps: Optional[Sequence[ColumnMap]] = None,
                um_per_px: Optional[float] = None,
                on_conflict: str = 'refuse',
                allow_spacr_targets: bool = False,
                prefix: str = FOREIGN_PREFIX,
                verify_labels: bool = True) -> ImportPlan:
    """Work out the whole import and write nothing.

    Five things happen here, in order, and every one of them can only
    produce a report:

    1. **Images** are scanned and planned by :mod:`spacr.convert` — the
       Yokogawa naming, the well assignment and the ``conversion_map.csv``
       come from there, unchanged.
    2. **Masks** are scanned the same way and paired to image fields. A
       mask with no image and an image with no mask are both listed by
       path in :attr:`ImportPlan.masks`.
    3. **Their table** is read and, when ``column_maps`` is not given,
       :func:`infer_column_map` proposes one. A proposal is not an
       application: it is what you save, edit and hand back.
    4. **Columns are resolved** — collisions with spaCR names settled, unit
       conversions computed or refused.
    5. **The join is verified** against the label images: every count in
       :class:`JoinReport` is real, not assumed.

    :param images: folder of their image files.
    :param masks: folder (taken as ``cell``), or ``{object_type: folder}``.
    :param measurements: their table, or a path to it.
    :param layout: source layout for :func:`spacr.convert.scan`.
    :param z_handling: :data:`spacr.convert.Z_MAX` by default, because a
        merged array holds one plane per channel; keeping every z would
        produce fields spaCR cannot merge, and the plan says so.
    :param mask_layout: layout for the mask folders; defaults to ``layout``.
    :param mask_suffixes: tokens stripped from mask filenames before
        matching (:data:`MASK_SUFFIXES` by default).
    :param measurement_object: which object type their table measures;
        defaults to the first mask type given.
    :param image_key: their column naming the image. Inferred when None.
    :param label_key: their column holding the object label. Inferred when
        None.
    :param column_maps: the **reviewed** mapping. When None, an inferred
        proposal is used and the plan says loudly that it is a proposal.
    :param um_per_px: micrometres per pixel. None means unknown, and any
        column needing it is reported uncalibrated rather than scaled by 1.
    :param on_conflict: ``'refuse'`` (default) or ``'rename'``.
    :param allow_spacr_targets: opt in to writing foreign values into
        spaCR's own column names.
    :param verify_labels: read the masks to check the join. On by default.
    :returns: an :class:`ImportPlan`.
    :raises ConfigurationError: for an unreadable input or an unknown
        option — a setup mistake, not a per-item failure.
    """
    if on_conflict not in ON_CONFLICT:
        raise ConfigurationError(
            f'Unknown on_conflict {on_conflict!r}; expected one of '
            f'{", ".join(ON_CONFLICT)}')

    # -- 1. images -----------------------------------------------------------
    image_sources = cv.scan(images, layout=layout)
    image_plan = cv.plan(image_sources, z_handling=z_handling,
                         plate_naming=plate_naming)

    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []

    # A merged array is (H, W, C): one plane per channel, no z axis. Keeping
    # every plane would produce N files per channel with nothing to merge
    # them into, so it is refused here rather than half-way through.
    if z_handling == cv.Z_KEEP and any(m.z > 1 for m in image_plan.mappings):
        errors.append(
            "z_handling='keep' was requested but some fields hold more than "
            "one z plane. A spaCR merged array has one plane per channel and "
            "no z axis, so there would be nothing to merge them into. Use "
            "z_handling='max' or 'first'.")

    stem_of_mapping: Dict[Tuple[str, str, str], str] = {}
    field_sources: Dict[Tuple[str, str, str], List[str]] = {}
    channels_by_stem: Dict[str, Dict[int, 'cv.Mapping']] = {}
    for mapping in image_plan.mappings:
        key = (mapping.source_plate, mapping.source_well, mapping.source_field)
        stem = _stem_of(mapping.plate, mapping.well, mapping.field)
        stem_of_mapping[key] = stem
        field_sources.setdefault(key, [])
        if mapping.source not in field_sources[key]:
            field_sources[key].append(mapping.source)
        channels_by_stem.setdefault(stem, {})
        existing = channels_by_stem[stem].get(mapping.channel)
        if existing is not None and (existing.z, existing.t) != (mapping.z, mapping.t):
            errors.append(
                f'{stem}: channel {mapping.channel} has more than one plane '
                f'(z={existing.z},t={existing.t} and z={mapping.z},t={mapping.t}). '
                f'A merged array holds exactly one plane per channel.')
        channels_by_stem[stem][mapping.channel] = mapping

    n_channels = len({m.channel for m in image_plan.mappings})
    ragged = sorted(stem for stem, chans in channels_by_stem.items()
                    if len(chans) != n_channels)
    for stem in ragged[:20]:
        errors.append(
            f'{stem} has {len(channels_by_stem[stem])} channel(s) but the '
            f'experiment has {n_channels}; a merged array needs the same '
            f'planes in every field.')
    if len(ragged) > 20:
        errors.append(f'… and {len(ragged) - 20} more field(s) with a '
                      f'different channel count.')

    # -- 2. masks ------------------------------------------------------------
    folders = _mask_folders(masks)
    if not folders:
        errors.append('No mask folder was given. A spaCR project without '
                      'object masks has nothing to measure and nothing to '
                      'crop; pass masks={"cell": "/path/to/cell_masks"}.')
    suffixes = tuple(mask_suffixes) if mask_suffixes else MASK_SUFFIXES
    pairing = PairingReport()
    per_type: Dict[str, Dict[str, MaskMapping]] = {}

    for object_type, folder in folders.items():
        mask_sources = cv.scan(folder, layout=mask_layout or layout)
        matched: Dict[str, MaskMapping] = {}
        for source in mask_sources:
            if not source.readable:
                pairing.unreadable_masks.append((source.path, source.error))
                continue
            keys = [(source.plate, source.well, source.field, 'exact')]
            normalised = _normalise_mask_field(source.field, object_type, suffixes)
            if normalised != source.field:
                keys.append((source.plate, source.well, normalised, 'normalised'))
            # A mask tree that has no plate/well folders of its own still
            # has to reach the images' single plate and well.
            hit = None
            for plate_key, well_key, field_key, how in keys:
                candidate = (plate_key, well_key, field_key)
                if candidate in stem_of_mapping:
                    hit = (candidate, how)
                    break
                loose = [k for k in stem_of_mapping if k[2] == field_key]
                if len(loose) == 1:
                    hit = (loose[0], how)
                    break
            if hit is None:
                pairing.masks_without_images.append((source.path, object_type))
                continue
            key, how = hit
            stem = stem_of_mapping[key]
            if stem in matched:
                pairing.masks_without_images.append((source.path, object_type))
                errors.append(
                    f'{source.path} and {matched[stem].source} both claim to '
                    f'be the {object_type} mask of {stem}.')
                continue
            plate, well, field = stem.split('_')
            labels: Tuple[int, ...] = ()
            if verify_labels:
                try:
                    labels = _mask_labels(_read_mask(source))
                except ConfigurationError:
                    raise
                except Exception as exc:
                    pairing.unreadable_masks.append(
                        (source.path, f'{exc.__class__.__name__}: {exc}'))
                    continue
            matched[stem] = MaskMapping(
                source=source.path, object_type=object_type, stem=stem,
                plate=plate, well=well, field=int(field),
                source_field=source.field, match=how, labels=labels)
        per_type[object_type] = matched

    all_stems = sorted(set(stem_of_mapping.values()))
    for stem in all_stems:
        present = {t: per_type[t][stem] for t in folders if stem in per_type[t]}
        if len(present) == len(folders) and folders:
            pairing.fields[stem] = present
            continue
        pairing.excluded.append(stem)
        for object_type in folders:
            if object_type in present:
                continue
            for key, sources in field_sources.items():
                if stem_of_mapping.get(key) != stem:
                    continue
                for path in sources:
                    pairing.images_without_masks.append((path, object_type))

    if pairing.images_without_masks:
        warnings.append(
            f'{len(pairing.images_without_masks)} image file(s) have no '
            f'matching mask; the {len(set(pairing.excluded))} field(s) they '
            f'belong to are NOT imported (a merged array needs the same mask '
            f'planes in every field). Every one is listed by path in the plan.')
    if pairing.masks_without_images:
        warnings.append(
            f'{len(pairing.masks_without_images)} mask file(s) match no image '
            f'and are NOT imported. Every one is listed by path in the plan.')
    if pairing.unreadable_masks:
        warnings.append(f'{len(pairing.unreadable_masks)} mask file(s) could '
                        f'not be read.')
    if folders and not pairing.fields:
        errors.append('No field has a complete set of masks, so there is '
                      'nothing to import.')

    mask_dims = {name: n_channels + index
                 for index, name in enumerate(folders)}

    # -- 3. their table ------------------------------------------------------
    # reset_index so row positions are 0..n-1: every count below is
    # positional, and a table read back from SQL with a non-unique index
    # would otherwise silently multiply rows on the .loc select.
    frame = read_measurements(measurements,
                              table=measurement_table).reset_index(drop=True)
    columns = [str(c) for c in frame.columns]
    if image_key is None:
        image_key = _pick_key(columns, _IMAGE_KEY_HINTS)
    if label_key is None:
        label_key = _pick_key(columns, _LABEL_KEY_HINTS)
    if label_key is None:
        errors.append(
            'No column in the measurement table looks like an object label, '
            'so their rows cannot be tied to objects in their masks. Pass '
            'label_key=<column>.')
    if image_key is None and len(pairing.fields) > 1:
        errors.append(
            f'No column identifies which image each measurement row came '
            f'from, and the import covers {len(pairing.fields)} fields. Pass '
            f'image_key=<column>.')

    object_type = measurement_object or (next(iter(folders)) if folders else 'cell')
    if folders and object_type not in folders:
        errors.append(
            f'measurement_object={object_type!r} has no mask folder; the '
            f'table would be joined against objects that do not exist. Mask '
            f'folders given: {", ".join(folders) or "none"}.')

    if um_per_px is None:
        notes.append(
            'No pixel size (um_per_px) was given. A 2-D spaCR run measures '
            'areas in px^2, lengths in px and intensities in raw uncalibrated '
            'counts (see spacr.feature_dict), so any column declared in '
            'micrometres is imported unconverted and flagged calibrated = 0. '
            'A 3-D run stamped measurement_units = "um" is already in '
            'micrometres — check the target table before converting.')

    # -- 4. the join ---------------------------------------------------------
    join = JoinReport(image_key=str(image_key or ''),
                      label_key=str(label_key or ''),
                      object_type=object_type,
                      rows_total=int(len(frame)))
    if label_key and (image_key or len(pairing.fields) <= 1):
        index, ambiguous = _build_field_index(
            [m.to_row(status='planned') for m in image_plan.mappings])
        join.ambiguous_keys = ambiguous
        only_stem = next(iter(pairing.fields), '') if len(pairing.fields) == 1 else ''
        labels_by_stem = {
            stem: set(per_type.get(object_type, {})[stem].labels)
            for stem in pairing.fields
            if stem in per_type.get(object_type, {})}

        unresolved: Dict[str, int] = {}
        no_object: Dict[str, int] = {}
        seen: Dict[str, Set[int]] = {stem: set() for stem in labels_by_stem}
        matched_rows = 0
        for _position, row in frame.iterrows():
            stem = (_resolve_field(row.get(image_key), index)
                    if image_key else only_stem)
            if not stem or stem not in labels_by_stem:
                value = str(row.get(image_key)) if image_key else '(no image key)'
                unresolved[value] = unresolved.get(value, 0) + 1
                continue
            try:
                label = int(row.get(label_key))
            except (TypeError, ValueError):
                no_object[stem] = no_object.get(stem, 0) + 1
                if len(join.examples) < 5:
                    join.examples.append(
                        f'{stem}: label {row.get(label_key)!r} is not an integer')
                continue
            if verify_labels and label not in labels_by_stem[stem]:
                no_object[stem] = no_object.get(stem, 0) + 1
                if len(join.examples) < 5:
                    join.examples.append(
                        f'{stem}: no object with label {label} in the '
                        f'{object_type} mask')
                continue
            seen[stem].add(label)
            matched_rows += 1
        join.rows_matched = matched_rows
        join.unresolved_fields = sorted(unresolved.items(),
                                        key=lambda kv: (-kv[1], kv[0]))
        join.rows_no_object = sorted(no_object.items(),
                                     key=lambda kv: (-kv[1], kv[0]))
        if verify_labels:
            join.objects_unmeasured = sorted(
                ((stem, len(labels - seen[stem]))
                 for stem, labels in labels_by_stem.items()
                 if labels - seen[stem]),
                key=lambda kv: (-kv[1], kv[0]))

        if join.rows_unmatched:
            warnings.append(
                f'{join.rows_unmatched} of {join.rows_total} measurement '
                f'row(s) ({1 - join.match_rate:.1%}) match no object in the '
                f'masks: {join.n_unresolved} name an image that is not in the '
                f'import and {join.n_no_object} carry a label that is in no '
                f'mask. They are imported with the metadata that could be '
                f'resolved, never silently dropped.')
        if join.n_objects_unmeasured:
            warnings.append(
                f'{join.n_objects_unmeasured} mask object(s) have no row in '
                f'the measurement table; they exist in the merged arrays and '
                f'in the crops, with no foreign measurements attached.')

    # -- 5. the columns ------------------------------------------------------
    # Built last, and through with_column_maps(), so that the mapping a GUI
    # re-resolves on every edit goes down exactly the same code path as the
    # one plan_import produces. One resolver, one set of conflicts.
    proposed = column_maps is None
    if proposed:
        reviewed = infer_column_map(frame, image_key=image_key,
                                    label_key=label_key, prefix=prefix)
        notes.append(
            f'The column mapping is an INFERRED PROPOSAL for '
            f'{len(reviewed)} column(s). Save it with save_column_map(), read '
            f'it, edit it, and pass it back as column_maps= — nothing here '
            f'has checked that their columns mean what their names suggest.')
    else:
        reviewed = list(column_maps)

    base = ImportPlan(
        images=image_plan, masks=pairing, measurements=frame,
        join=join, notes=notes,
        object_types=list(folders), n_channels=n_channels,
        mask_dims=mask_dims, um_per_px=um_per_px, prefix=prefix,
        on_conflict=on_conflict, allow_spacr_targets=allow_spacr_targets,
        base_warnings=warnings, base_errors=errors,
        sources={'images': os.path.abspath(str(images)),
                 'measurements': (measurements if isinstance(measurements, str)
                                  else '<DataFrame>'),
                 **{f'masks:{k}': os.path.abspath(v)
                    for k, v in folders.items()}})
    plan = base.with_column_maps(reviewed)
    plan.proposed = proposed
    plan.notes.insert(0, (
        f'{len(pairing.fields)} field(s), {n_channels} channel(s), '
        f'{len(folders)} mask class(es) ({", ".join(folders) or "none"}), '
        f'{len(frame)} measurement row(s), {len(frame.columns)} column(s).'))
    return plan


# ---------------------------------------------------------------------------
# Rendering the plan
# ---------------------------------------------------------------------------

def format_plan(plan: ImportPlan) -> str:
    """Render an :class:`ImportPlan` as the block a user reads before agreeing.

    Ordered by what can hurt them: blocking problems, then conflicts, then
    the columns that could not be mapped, then the join, then the plain
    counts.
    """
    lines: List[str] = ['spaCR foreign import — nothing has been written.']
    lines.extend(plan.notes)

    if plan.errors or not plan.images.ok:
        lines.append('')
        lines.append('BLOCKING PROBLEMS')
        for error in plan.images.errors:
            lines.append(f'  ERROR: {error}')
        for error in plan.errors:
            lines.append(f'  ERROR: {error}')

    if plan.conflicts:
        lines.append('')
        lines.append(f'COLUMN CONFLICTS ({len(plan.blocking_conflicts)} '
                     f'blocking of {len(plan.conflicts)})')
        for conflict in plan.conflicts:
            mark = 'BLOCKING' if conflict.blocking else 'note'
            lines.append(f'  [{mark}] {conflict}')

    if plan.unmapped:
        lines.append('')
        lines.append(f'COLUMNS WITH NO MAPPING ({len(plan.unmapped)}) — '
                     f'imported under "{plan.prefix}", not dropped:')
        for source in plan.unmapped:
            lines.append(f'  {source}  ->  {plan.target_for(source)}')

    uncalibrated = plan.uncalibrated
    if uncalibrated:
        lines.append('')
        lines.append(f'UNCALIBRATED COLUMNS ({len(uncalibrated)}) — stored in '
                     f'their own units, calibrated = 0:')
        for resolution in uncalibrated:
            lines.append(f'  {resolution.source}  ->  {resolution.target}  '
                         f'[{_pretty_unit(resolution.unit)}]  {resolution.reason}')

    lines.append('')
    lines.append(plan.join.summary())

    pairing = plan.masks
    if pairing.images_without_masks or pairing.masks_without_images:
        lines.append('')
        lines.append('IMAGES AND MASKS THAT DID NOT PAIR')
        for path, object_type in pairing.images_without_masks[:20]:
            lines.append(f'  image with no {object_type} mask: {path}')
        if len(pairing.images_without_masks) > 20:
            lines.append(f'  … and {len(pairing.images_without_masks) - 20} '
                         f'more image file(s)')
        for path, object_type in pairing.masks_without_images[:20]:
            lines.append(f'  {object_type} mask with no image: {path}')
        if len(pairing.masks_without_images) > 20:
            lines.append(f'  … and {len(pairing.masks_without_images) - 20} '
                         f'more mask file(s)')
    for path, reason in pairing.unreadable_masks[:20]:
        lines.append(f'  unreadable mask: {path}: {reason}')

    mapped = [r for r in plan.resolved if r.status == 'mapped']
    if mapped:
        lines.append('')
        lines.append(f'COLUMNS THAT MAP CLEANLY ({len(mapped)}):')
        for resolution in mapped[:40]:
            scale = ('' if resolution.factor in (None, 1.0)
                     else f'  x{resolution.factor:g}')
            lines.append(f'  {resolution.source}  ->  {resolution.target}  '
                         f'[{_pretty_unit(resolution.unit)}]{scale}')
        if len(mapped) > 40:
            lines.append(f'  … and {len(mapped) - 40} more')

    if plan.warnings:
        lines.append('')
        for warning in plan.warnings:
            lines.append(f'WARNING: {warning}')

    lines.append('')
    lines.append(f'Plan is {"READY" if plan.ok else "NOT READY"} to run.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass
class ImportResult:
    """What :func:`run_import` actually did.

    :ivar conversion: the :class:`spacr.convert.ConversionResult` for
        their images — the provenance back to the original filenames.
    :ivar db_path: the ``measurements.db`` that was written.
    :ivar merged: merged ``.npy`` paths, one per imported field.
    :ivar rows: rows written into each foreign object table.
    :ivar crops: PNG paths cut from the merged arrays, if any.
    :ivar measured: True when spaCR's own measurements were re-extracted.
    """

    plan: ImportPlan
    dst: str
    conversion: Optional['cv.ConversionResult'] = None
    db_path: str = ''
    column_map_path: str = ''
    stacks: List[str] = dc_field(default_factory=list)
    mask_files: List[str] = dc_field(default_factory=list)
    merged: List[str] = dc_field(default_factory=list)
    rows: Dict[str, int] = dc_field(default_factory=dict)
    crops: List[str] = dc_field(default_factory=list)
    measured: bool = False
    ledger: Optional[RunLedger] = None
    warnings: List[str] = dc_field(default_factory=list)

    @property
    def n_fields(self) -> int:
        """Fields that produced a merged array."""
        return len(self.merged)

    @property
    def n_rows(self) -> int:
        """Measurement rows written, across every object table."""
        return sum(self.rows.values())

    @property
    def is_complete(self) -> bool:
        """True when nothing was skipped and nothing failed."""
        if self.conversion is not None and not self.conversion.is_complete:
            return False
        return self.ledger is None or self.ledger.is_complete

    def summary(self) -> str:
        """The end-of-run block: what landed, and everything that did not."""
        plan = self.plan
        lines = [f'Imported {self.n_fields} field(s) into {self.dst}.']
        if self.conversion is not None:
            lines.append(f'  {self.conversion.n_written} image file(s) '
                         f'converted; map: {self.conversion.map_path}')
        lines.append(f'  {len(self.mask_files)} mask file(s) written, '
                     f'{len(self.merged)} merged array(s).')
        for table, count in sorted(self.rows.items()):
            lines.append(f'  {count} row(s) -> {table}')
        if self.crops:
            lines.append(f'  {len(self.crops)} crop(s) cut.')
        lines.append(f'  measurements.db: {self.db_path}')
        lines.append(f'  applied column map: {self.column_map_path}')

        theirs = [r.target for r in plan.resolved]
        lines.append('')
        lines.append(f'{len(theirs)} column(s) in the object tables are '
                     f'THEIRS (every one prefixed "{plan.prefix}" unless you '
                     f'targeted a spaCR name explicitly); the '
                     f'plateID/rowID/columnID/fieldID/prcf/object_label keys '
                     f'are spaCR\'s. The {FOREIGN_COLUMNS_TABLE} table says '
                     f'which is which, per column.')
        if self.measured:
            lines.append("spaCR's own measurements were re-extracted into the "
                         "standard object tables; their columns stay in the "
                         "foreign_* tables and join on (prcf, object_label).")
        else:
            lines.append("spaCR's own measurements were NOT re-extracted — "
                         "every feature column in this database is theirs.")

        if plan.unmapped:
            lines.append('')
            lines.append(f'{len(plan.unmapped)} column(s) had no mapping and '
                         f'were imported verbatim: {", ".join(plan.unmapped)}')
        uncalibrated = plan.uncalibrated
        if uncalibrated:
            lines.append(f'{len(uncalibrated)} column(s) are UNCALIBRATED '
                         f'(calibrated = 0): '
                         f'{", ".join(r.target for r in uncalibrated)}')
        if plan.join.rows_unmatched:
            lines.append(f'{plan.join.rows_unmatched} of {plan.join.rows_total} '
                         f'measurement row(s) matched no mask object.')
        for warning in self.warnings:
            lines.append(f'WARNING: {warning}')
        if not self.is_complete:
            lines.append('THIS IMPORT IS INCOMPLETE — see the run stamp next '
                         'to the database.')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Running the import
# ---------------------------------------------------------------------------

def _imread(path: str) -> np.ndarray:
    """Read one converted TIFF back. Split out so tests can make it fail."""
    import tifffile

    return np.asarray(tifffile.imread(path))


def _save_npy(path: str, array: np.ndarray) -> str:
    """Write ``array`` atomically as ``.npy``.

    Delegates to :func:`spacr.io._save_array_atomic` so imported stacks
    and masks get the same crash-safety as a native run: an interrupted
    import leaves either nothing or a complete array, never a truncated
    file the next run mistakes for done.
    """
    from .io import _save_array_atomic

    return _save_array_atomic(path, np.asarray(array))


def _build_merged(dst: str, plan: ImportPlan) -> List[str]:
    """Build ``merged/`` from ``stack/`` and ``masks/`` — with spaCR's own code.

    :func:`spacr.io._load_and_concatenate_arrays` is what a native run
    uses, so calling it is the only way to be sure an imported merged
    array has exactly the layout every downstream reader assumes:
    intensity planes first, then one label plane per class in the order
    cell, nucleus, pathogen, organelle
    (:data:`spacr.crops.MASK_PLANE_ORDER`).
    """
    from .io import _load_and_concatenate_arrays

    _load_and_concatenate_arrays(
        dst, None,
        cell_chann_dim=None, nucleus_chann_dim=None,
        pathogen_chann_dim=None, organelle_chann_dim=None, resume=False)
    folder = os.path.join(dst, 'merged')
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, name)
                  for name in os.listdir(folder) if name.endswith('.npy'))


def _foreign_frame(plan: ImportPlan, stems: Sequence[str],
                   merged_dir: str) -> 'pd.DataFrame':
    """Assemble the rows that go into the foreign object table.

    spaCR's metadata columns first, in the order
    :func:`spacr.utils._merge_and_save_to_database` writes them, then one
    column per :class:`ResolvedColumn` — so the table looks like a spaCR
    object table to everything that reads one.
    """
    frame = plan.measurements
    join = plan.join
    index, _ambiguous = _build_field_index(
        [m.to_row(status='planned') for m in plan.images.mappings])
    known = set(stems)
    only_stem = next(iter(known), '') if len(known) == 1 else ''

    resolved_stems: List[str] = []
    keep: List[int] = []
    for position, row in frame.iterrows():
        stem = (_resolve_field(row.get(join.image_key), index)
                if join.image_key else only_stem)
        if not stem or stem not in known:
            continue
        resolved_stems.append(stem)
        keep.append(position)
    subset = frame.loc[keep]

    out = pd.DataFrame(index=range(len(subset)))
    labels: List[Any] = []
    for value in (subset[join.label_key] if join.label_key in subset.columns
                  else [0] * len(subset)):
        try:
            labels.append(int(value))
        except (TypeError, ValueError):
            labels.append(0)
    out['object_label'] = labels

    plates, rows_, columns_, fields, prcs, prcfs, files, paths = (
        [], [], [], [], [], [], [], [])
    for stem in resolved_stems:
        plate, well, field = stem.split('_')
        row_id, column_id = cv._well_ids(well)
        prc = f'{plate}_{row_id}_{column_id}'
        plates.append(plate)
        rows_.append(row_id)
        columns_.append(column_id)
        fields.append(f'f{int(field)}')
        prcs.append(prc)
        prcfs.append(f'{prc}_f{int(field)}')
        files.append(stem)
        paths.append(os.path.join(merged_dir, f'{stem}.npy'))
    out['plateID'] = plates
    out['rowID'] = rows_
    out['columnID'] = columns_
    out['fieldID'] = fields
    out['prc'] = prcs
    out['prcf'] = prcfs
    out['file_name'] = files
    out['path_name'] = paths

    for resolution in plan.resolved:
        if resolution.source not in subset.columns:
            continue
        values = subset[resolution.source].reset_index(drop=True)
        out[resolution.target] = resolution.apply(values).values
    return out


def _write_provenance(db_path: str, plan: ImportPlan, dst: str,
                      tables: Sequence[str]) -> None:
    """Write ``foreign_columns`` and ``foreign_import``.

    ``foreign_columns`` is the answer to "is this column theirs or
    spaCR's", per column, per table — including the unmapped ones and the
    uncalibrated ones, with the reason in words.
    """
    records: List[Dict[str, Any]] = []
    for table in tables:
        for name in ('object_label', 'plateID', 'rowID', 'columnID',
                     'fieldID', 'prc', 'prcf', 'file_name', 'path_name'):
            records.append({
                'table': table, 'column': name, 'origin': 'spacr',
                'source_column': '', 'transform': '', 'factor': None,
                'unit_in': '', 'unit_declared': '', 'unit': '',
                'calibrated': 1, 'status': 'metadata',
                'reason': 'written by the importer from the conversion map',
                'note': '',
            })
        for resolution in plan.resolved:
            records.append(resolution.to_record(table))

    run = {
        'images': plan.sources.get('images', ''),
        'measurements': plan.sources.get('measurements', ''),
        'masks_json': json.dumps(
            {k.split(':', 1)[1]: v for k, v in plan.sources.items()
             if k.startswith('masks:')}, sort_keys=True),
        'dst': os.path.abspath(dst),
        'object_types': ','.join(plan.object_types),
        'n_channels': int(plan.n_channels),
        'mask_dims_json': json.dumps(plan.mask_dims, sort_keys=True),
        'um_per_px': None if plan.um_per_px is None else float(plan.um_per_px),
        'prefix': plan.prefix,
        'image_key': plan.join.image_key,
        'label_key': plan.join.label_key,
        'join': plan.join.key_description,
        'rows_total': int(plan.join.rows_total),
        'rows_matched': int(plan.join.rows_matched),
        'rows_unmatched': int(plan.join.rows_unmatched),
        'objects_unmeasured': int(plan.join.n_objects_unmeasured),
        'n_unmapped': len(plan.unmapped),
        'unmapped': ', '.join(plan.unmapped),
        'n_uncalibrated': len(plan.uncalibrated),
        'uncalibrated': ', '.join(r.target for r in plan.uncalibrated),
        'n_conflicts': len(plan.conflicts),
        'conflicts': ' | '.join(str(c) for c in plan.conflicts),
    }

    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        pd.DataFrame(records).to_sql(FOREIGN_COLUMNS_TABLE, connection,
                                     if_exists='replace', index=False)
        pd.DataFrame([run]).to_sql(IMPORT_TABLE, connection,
                                   if_exists='replace', index=False)
        connection.commit()
    finally:
        connection.close()


def _write_view(db_path: str, object_type: str) -> None:
    """Create ``<object>_with_foreign`` when both halves exist.

    The view is the answer to "how do I see their number next to spaCR's":
    a join on ``(prcf, object_label)``, which is the same key
    :func:`spacr.utils._merge_and_save_to_database` writes.
    """
    foreign = f'{FOREIGN_PREFIX}{object_type}'
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if object_type not in names or foreign not in names:
            return
        connection.execute(f'DROP VIEW IF EXISTS {object_type}_with_foreign')
        connection.execute(
            f'CREATE VIEW {object_type}_with_foreign AS '
            f'SELECT s.*, f.* FROM {object_type} AS s '
            f'JOIN {foreign} AS f '
            f'  ON s.prcf = f.prcf AND s.object_label = f.object_label')
        connection.commit()
    finally:
        connection.close()


def _cut_crops(dst: str, plan: ImportPlan, frame: 'pd.DataFrame',
               object_type: str, channels: Optional[Sequence[int]] = None,
               size: Tuple[int, int] = (224, 224),
               limit: Optional[int] = None) -> List[str]:
    """Cut one crop per imported object, through :mod:`spacr.crops`.

    Reuses :func:`spacr.crops.extract_crops` (one memory-map and one label
    index per field, however many objects come out of it) and
    :func:`spacr.crops.png_view` (the exact BGR/16-bit narrowing a spaCR
    PNG goes through), so an imported crop is the same array a native run
    would have written.
    """
    from PIL import Image

    if channels is None:
        channels = tuple(range(min(3, max(plan.n_channels, 1))))
    channels = tuple(int(c) for c in channels)
    out_dir = os.path.join(dst, 'crops', object_type)
    os.makedirs(out_dir, exist_ok=True)

    written: List[str] = []
    for stem, group in frame.groupby('file_name', sort=True):
        merged_path = os.path.join(dst, 'merged', f'{stem}.npy')
        if not os.path.isfile(merged_path):
            continue
        rows = list(group.itertuples(index=False))
        if limit is not None:
            rows = rows[:max(int(limit) - len(written), 0)]
        specs = [cropping.CropSpec(merged_path=merged_path,
                                   object_type=object_type,
                                   label=int(row.object_label),
                                   channels=channels, size=size,
                                   mask_dims=plan.mask_dims)
                 for row in rows]
        cut = cropping.extract_crops(merged_path, specs,
                                     mask_dims=plan.mask_dims, on_error='none')
        for row, crop in zip(rows, cut):
            if crop is None:
                continue
            path = os.path.join(out_dir, f'{stem}_obj{int(row.object_label)}.png')
            Image.fromarray(cropping.png_view(crop)).save(path)
            written.append(path)
        if limit is not None and len(written) >= int(limit):
            break
    return written


def run_import(plan: ImportPlan, dst: str, *,
               overwrite: bool = False,
               measure: bool = False,
               measure_settings: Optional[TMapping[str, Any]] = None,
               crops: bool = False,
               crop_channels: Optional[Sequence[int]] = None,
               crop_size: Tuple[int, int] = (224, 224),
               crop_limit: Optional[int] = None,
               progress: Optional[Callable[[int, int, str], None]] = None,
               ledger: Optional[RunLedger] = None) -> ImportResult:
    """Execute a reviewed plan: build the project at ``dst``.

    In order: convert their images (:func:`spacr.convert.convert`), write
    the intensity stacks, write the mask stacks, build the merged arrays
    with spaCR's own :func:`spacr.io._load_and_concatenate_arrays`, load
    the conversion map into the database, write their measurements next to
    spaCR's metadata, record the provenance, and — optionally, and
    separately — re-extract spaCR's own measurements and cut crops.

    Re-running is a no-op rather than a duplication: existing TIFFs are
    left alone by the converter, and every table this writes is *replaced*,
    never appended to.

    :param plan: a plan from :func:`plan_import` whose ``ok`` is True.
    :param dst: destination project root. Created if missing.
    :param overwrite: rewrite converted images that already exist.
    :param measure: also run :func:`spacr.measure.measure_crop` over the
        imported project. Off by default, and *separate*: when it is on,
        the standard object tables are spaCR's and theirs stay in the
        ``foreign_*`` tables, joined by a ``<object>_with_foreign`` view.
    :param measure_settings: extra settings for ``measure_crop``.
    :param crops: cut one PNG per imported object.
    :param progress: ``progress(done, total, message)``.
    :param ledger: reuse an existing :class:`spacr.errors.RunLedger`.
    :returns: an :class:`ImportResult`.
    :raises ConfigurationError: the plan is not ``ok``.
    """
    if not plan.ok:
        problems = list(plan.images.errors) + list(plan.errors)
        problems += [str(c) for c in plan.blocking_conflicts]
        raise ConfigurationError(
            'This import cannot run — fix these first:\n  '
            + '\n  '.join(problems))

    dst = os.path.abspath(str(dst))
    os.makedirs(dst, exist_ok=True)
    run = ledger if ledger is not None else RunLedger('foreign_import')
    result = ImportResult(plan=plan, dst=dst, ledger=run)
    steps = ['converting images', 'writing stacks', 'writing masks',
             'merging arrays', 'writing measurements']
    total = len(steps) + int(bool(measure)) + int(bool(crops))

    def _step(index: int, message: str) -> None:
        if progress is not None:
            progress(index, total, message)

    # -- 1. their images -> Yokogawa TIFFs, via spacr.convert ---------------
    _step(1, 'converting images')
    images_dir = os.path.join(dst, IMAGES_DIRNAME)
    result.conversion = cv.convert(plan.images, images_dir,
                                   overwrite=overwrite, ledger=run)

    written = {m.target for m in result.conversion.written}
    written |= {m.target for m in result.conversion.existing}

    # -- 2. intensity stacks -------------------------------------------------
    _step(2, 'writing stacks')
    stack_dir = os.path.join(dst, 'stack')
    os.makedirs(stack_dir, exist_ok=True)
    by_stem: Dict[str, Dict[int, 'cv.Mapping']] = {}
    for mapping in plan.images.mappings:
        stem = _stem_of(mapping.plate, mapping.well, mapping.field)
        if stem not in plan.masks.fields:
            continue
        by_stem.setdefault(stem, {})[mapping.channel] = mapping

    usable: List[str] = []
    for stem in sorted(by_stem):
        channels = by_stem[stem]
        missing = [m.target for m in channels.values() if m.target not in written]
        if missing:
            result.warnings.append(
                f'{stem}: {len(missing)} channel image(s) were not converted '
                f'({", ".join(missing[:3])}), so the field has no stack.')
            continue
        with run.item(stem, stage='stack'):
            planes = [_imread(os.path.join(images_dir, channels[c].target))
                      for c in sorted(channels)]
            stack = np.stack([np.asarray(p) for p in planes], axis=-1)
            result.stacks.append(_save_npy(
                os.path.join(stack_dir, f'{stem}.npy'), stack))
            usable.append(stem)

    # -- 3. their masks ------------------------------------------------------
    _step(3, 'writing masks')
    for object_type in plan.object_types:
        os.makedirs(os.path.join(dst, 'masks', f'{object_type}_mask_stack'),
                    exist_ok=True)
    # Iterating the field's own masks rather than the declared classes: a
    # stem only reaches ``masks.fields`` when it has every class, so there
    # is no "missing" case here to guess at.
    for stem in usable:
        for object_type, mask in plan.masks.fields[stem].items():
            folder = os.path.join(dst, 'masks', f'{object_type}_mask_stack')
            with run.item(f'{stem}:{object_type}', stage='mask'):
                array = _read_mask(mask.source).astype(np.uint16, copy=False)
                result.mask_files.append(_save_npy(
                    os.path.join(folder, f'{stem}.npy'), array))

    # -- 4. merged arrays, with spaCR's own merger --------------------------
    _step(4, 'merging arrays')
    result.merged = _build_merged(dst, plan)
    merged_stems = {os.path.splitext(os.path.basename(p))[0]
                    for p in result.merged}
    for stem in usable:
        if stem not in merged_stems:
            result.warnings.append(f'{stem}: no merged array was produced.')

    # -- 5. the database -----------------------------------------------------
    _step(5, 'writing measurements')
    db_dir = os.path.join(dst, 'measurements')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'measurements.db')
    result.db_path = db_path

    cv.populate_db_from_map(db_path, result.conversion.map_path)

    object_type = plan.join.object_type
    frame = _foreign_frame(plan, sorted(merged_stems),
                           os.path.join(dst, 'merged'))
    table = f'{FOREIGN_PREFIX}{object_type}'
    connection = sqlite3.connect(db_path, timeout=30)
    try:
        # replace, never append: a second run of the same import must not
        # leave two generations of the same rows behind.
        frame.to_sql(table, connection, if_exists='replace', index=False)
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{table}_prcf_obj '
            f'ON {table} (prcf, object_label)')
        if not measure:
            frame.to_sql(object_type, connection, if_exists='replace',
                         index=False)
            connection.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{object_type}_prcf_obj '
                f'ON {object_type} (prcf, object_label)')
        connection.commit()
    finally:
        connection.close()
    result.rows[table] = int(len(frame))
    if not measure:
        result.rows[object_type] = int(len(frame))

    _write_provenance(db_path, plan, dst, sorted(set(result.rows)))
    result.column_map_path = str(save_column_map(
        plan.column_maps, os.path.join(dst, COLUMN_MAP_FILENAME)))

    # -- 6. spaCR's own measurements, optional and separate -----------------
    if measure:
        _step(6, 're-extracting spaCR measurements')
        with run.item(dst, stage='measure'):
            from .measure import measure_crop

            settings = dict(measure_settings or {})
            settings.setdefault('src', dst)
            settings.setdefault('experiment', 'foreign_import')
            settings.setdefault('channels', list(range(plan.n_channels)))
            for name in cropping.MASK_PLANE_ORDER:
                key = f'{name}_mask_dim'
                if name in plan.mask_dims:
                    settings.setdefault(key, plan.mask_dims[name])
                else:
                    settings.setdefault(key, None)
            settings.setdefault('save_measurements', True)
            measure_crop(settings)
            result.measured = True
        _write_view(db_path, object_type)

    # -- 7. crops, optional --------------------------------------------------
    if crops:
        _step(total, 'cutting crops')
        with run.item(dst, stage='crops'):
            result.crops = _cut_crops(dst, plan, frame, object_type,
                                      channels=crop_channels, size=crop_size,
                                      limit=crop_limit)

    run.finalize(artifact=db_path)
    return result


# ---------------------------------------------------------------------------
# The settings-dict entry point
# ---------------------------------------------------------------------------

def default_settings(settings: Optional[TMapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Return the settings :func:`import_project` understands, with defaults.

    Shaped like every other ``spacr`` settings factory — pass a partial
    dict, get it back filled in — so a CLI or a GUI can build a panel from
    it without special-casing this module.
    """
    resolved: Dict[str, Any] = {
        'images': None,
        'masks': None,
        'measurements': None,
        'dst': None,
        'layout': 'auto',
        'z_handling': cv.Z_MAX,
        'plate_naming': 'index',
        'measurement_table': None,
        'measurement_object': None,
        'image_key': None,
        'label_key': None,
        'column_map': None,
        'um_per_px': None,
        'on_conflict': 'refuse',
        'allow_spacr_targets': False,
        'measure': False,
        'crops': False,
        'overwrite': False,
        'preview_only': False,
    }
    resolved.update(dict(settings or {}))
    return resolved


def import_project(settings: Optional[TMapping[str, Any]] = None,
                   **overrides: Any) -> ImportResult:
    """Plan and run one foreign import from a settings dict.

    Always prints the plan before writing anything, so even a headless run
    leaves the mapping, the unmapped columns and the join counts in the log
    where a surprised user can find them.

    ``column_map`` is the path to a reviewed
    :func:`save_column_map` file. Leaving it None runs with the *inferred*
    proposal, which the printed plan says in as many words — use
    ``preview_only`` first, save the map, read it, then run.

    :returns: the :class:`ImportResult`; for ``preview_only`` an empty one
        carrying the plan.
    :raises ConfigurationError: a missing input, or a plan with blocking
        problems.
    """
    resolved = default_settings(settings)
    resolved.update(overrides)

    for key in ('images', 'masks', 'measurements'):
        if not resolved.get(key):
            raise ConfigurationError(
                f"import_project needs '{key}'. It takes a folder of their "
                f"images, their mask folder(s) and their measurement table.")
    images = os.path.abspath(str(resolved['images']))
    dst = resolved.get('dst') or (os.path.normpath(images) + '_spacr')
    dst = os.path.abspath(str(dst))

    column_maps = None
    if resolved.get('column_map'):
        column_maps = load_column_map(str(resolved['column_map']))

    plan = plan_import(
        images, resolved['masks'], resolved['measurements'],
        layout=str(resolved.get('layout') or 'auto'),
        z_handling=str(resolved.get('z_handling') or cv.Z_MAX),
        plate_naming=str(resolved.get('plate_naming') or 'index'),
        measurement_table=resolved.get('measurement_table'),
        measurement_object=resolved.get('measurement_object'),
        image_key=resolved.get('image_key'),
        label_key=resolved.get('label_key'),
        column_maps=column_maps,
        um_per_px=resolved.get('um_per_px'),
        on_conflict=str(resolved.get('on_conflict') or 'refuse'),
        allow_spacr_targets=bool(resolved.get('allow_spacr_targets')))

    print(format_plan(plan))
    if not plan.ok:
        raise ConfigurationError(
            'Import refused — nothing was written:\n  '
            + '\n  '.join(list(plan.images.errors) + list(plan.errors)
                          + [str(c) for c in plan.blocking_conflicts]))
    if resolved.get('preview_only'):
        print(f'preview_only is set — nothing was written. The project would '
              f'go to {dst}.')
        return ImportResult(plan=plan, dst=dst)

    result = run_import(plan, dst,
                        overwrite=bool(resolved.get('overwrite')),
                        measure=bool(resolved.get('measure')),
                        crops=bool(resolved.get('crops')))
    print(result.summary())
    return result
