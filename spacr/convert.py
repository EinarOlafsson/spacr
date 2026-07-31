"""Format converter / importer — vendor microscopy files into Yokogawa TIFFs.

This is the standalone version of "get my images into a shape spaCR can
read". It is deliberately **decoupled from the mask pipeline**: nothing
here imports torch, cellpose or :mod:`spacr.core`, so it can be run,
tested and previewed on a laptop with no GPU and no segmentation
settings in sight.

What it does
------------

A folder tree like::

    run1/
      wt/
        fov01_C1.tif  fov01_C2.tif
        fov02_C1.tif  fov02_C2.tif
        …

becomes one flat folder of Yokogawa-named TIFFs::

    plate1_A01_T0001F001L01A01Z01C01.tif
    plate1_A01_T0001F001L01A01Z01C02.tif
    plate1_A01_T0001F002L01A01Z01C01.tif
    …

``run1`` is the plate, ``wt`` is the well, each field-set gets its own
field id, each channel its own channel id. That name is exactly what
``spacr.utils._get_regex('cellvoyager', 'tif')`` parses, so the output
folder drops straight into Mask/Measure with
``metadata_type='cellvoyager'``.

Why a second converter
----------------------

:func:`spacr.io.convert_to_yokogawa` already converts ND2/CZI/LIF/TIFF,
and it is still the right tool when you want an in-place rename of a
flat folder. It cannot do the three things this module exists for:

* **Preview.** It writes as it walks. There is no point at which you can
  look at "``run1/wt/fov01_C1.tif`` → ``plate1_A01_…C01.tif``" and say
  no. :func:`plan` produces that table and writes nothing.
* **A map you can read back.** It writes a ``rename_log.csv`` whose
  columns differ per input format (the LIF and TIFF branches record two
  columns; the ND2 branch records seven; the CZI branch records nine),
  and nothing in spaCR ever reads it. :func:`write_map` emits one fixed
  schema and :func:`read_map` / :func:`populate_db_from_map` read it
  back, so after a run the original filename for any measurement is one
  SQL join away.
* **Z that is not silently flattened.** ``convert_to_yokogawa``
  max-projects unconditionally in three of its four branches (ND2, LIF,
  and both the 3-D and 4-D TIFF cases), and says so nowhere in its
  output. Here the default is :data:`Z_KEEP` — every plane is written
  with its own ``Z##`` — and choosing to project is an explicit
  ``z_handling='max'``, announced in the plan and recorded per row in
  the map file.

Design rules
------------

* **No new required dependencies.** ``nd2reader`` / ``czifile`` /
  ``readlif`` are probed with :func:`importlib.util.find_spec`; when one
  is missing the affected files are reported as skipped with the package
  name and the ``pip install`` line, never as an ImportError traceback.
* **Never overwrite.** Two sources landing on one target name is a
  *plan-time error* naming both, not a last-writer-wins at convert time.
  An output that already exists on disk is skipped and counted, so a
  re-run is a no-op rather than a silent rewrite.
* **Atomic writes.** Every TIFF is written to a temp file in the
  destination and :func:`os.replace`\\ d into place, so an interrupted
  run leaves no half-written file that the next run mistakes for done.
* **Ledgered.** A conversion that skipped 12 unreadable files says so in
  one grouped block at the end and stamps
  ``conversion_map.run_status.json`` next to its output — see
  :mod:`spacr.errors`.

Typical use::

    from spacr import convert as cv

    sources = cv.scan('/data/run1')
    plan = cv.plan(sources, z_handling=cv.Z_KEEP)
    print(plan.to_frame())          # the preview — nothing written yet
    if plan.ok:
        result = cv.convert(plan, '/data/run1_yokogawa')
        print(result.summary())
        cv.populate_db_from_map('/data/run1_yokogawa/measurements.db',
                                result.map_path)
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping as TMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schema
from .checkpoint import CheckpointStore, fingerprint
from .cancellation import checkpoint as cancellation_checkpoint
from .errors import ConfigurationError, RunLedger
from .tiff_io import write_tiff

__all__ = [
    'SourceImage',
    'Mapping',
    'ConversionPlan',
    'ConversionResult',
    'scan',
    'plan',
    'convert',
    'convert_folder',
    'default_settings',
    'write_map',
    'read_map',
    'populate_db_from_map',
    'reader_requirement',
    'reader_available',
    'missing_reader_message',
    'target_name',
    'assign_wells',
    'normalise_well',
    'well_sequence',
    'plate_format_for_names',
    'natural_key',
    'off_plate_reason',
    'DEFAULT_PLATE_FORMAT',
    'WELL_SEQUENCES',
    'IMAGE_EXTENSIONS',
    'MAP_FILENAME',
    'CHECKPOINT_FILENAME',
    'MAP_COLUMNS',
    'CONVERSION_TABLE',
    'LAYOUTS',
    'Z_KEEP',
    'Z_MAX',
    'Z_FIRST',
    'Z_HANDLING',
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Extensions the scanner will pick up, in the order they are documented.
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    '.tif', '.tiff', '.ome.tif', '.ome.tiff',
    '.png', '.jpg', '.jpeg', '.bmp',
    '.nd2', '.czi', '.lif',
)

#: Name of the map file written into the destination folder.
MAP_FILENAME = 'conversion_map.csv'

#: Atomic field-level checkpoint written beside converted images.
CHECKPOINT_FILENAME = '.spacr_conversion.checkpoint.json'

#: Table :func:`populate_db_from_map` writes into ``measurements.db``.
CONVERSION_TABLE = 'conversion_map'

#: Layout names understood by :func:`scan`.
LAYOUTS: Tuple[str, ...] = ('auto', 'flat', 'well', 'plate_well')

#: Every z plane is written as its own TIFF with its own ``Z##``. Default.
Z_KEEP = 'keep'
#: Z planes are max-projected into one TIFF. Lossy — announced in the plan.
Z_MAX = 'max'
#: Only the first z plane is written. Lossy — announced in the plan.
Z_FIRST = 'first'

#: Accepted ``z_handling`` values.
Z_HANDLING: Tuple[str, ...] = (Z_KEEP, Z_MAX, Z_FIRST)

#: The plate :func:`assign_wells` uses when nothing in the source asks for
#: a bigger one. Not a limit — see :func:`plate_format_for_names`, which
#: grows to 1536 when a real well or the well *count* needs it. It is the
#: default only so that a source of eight unnamed folders keeps producing
#: ``A01…A08`` rather than being re-laid-out onto a 6-well plate.
DEFAULT_PLATE_FORMAT = 384


def well_sequence(n_wells: int = DEFAULT_PLATE_FORMAT) -> Tuple[str, ...]:
    """Return every well id of an ``n_wells`` plate, row-major.

    Built from :data:`spacr.schema.PLATE_FORMATS` and rendered by
    :func:`spacr.schema.well_id`, so a 1536-well plate's rows past ``Z``
    come out ``AA``…``AF`` and its columns run to 48. This module used to
    carry its own ``'ABCDEFGHIJKLMNOP'`` and ``range(1, 25)``, which is
    exactly why it could not name a well on a plate bigger than 384.

    :param n_wells: a key of :data:`spacr.schema.PLATE_FORMATS`.
    :returns: the well ids, ``A01`` first.
    :raises ConfigurationError: for a non-standard plate format.
    """
    if n_wells not in schema.PLATE_FORMATS:
        raise ConfigurationError(
            f'{n_wells!r} is not a standard plate format; known formats are '
            f'{sorted(schema.PLATE_FORMATS)}.')
    n_rows, n_columns = schema.PLATE_FORMATS[n_wells]
    return tuple(schema.well_id(row, column)
                 for row in range(1, n_rows + 1)
                 for column in range(1, n_columns + 1))


#: ``n_wells -> every well id of that plate``, row-major — the order names
#: are handed out in by :func:`assign_wells`.
WELL_SEQUENCES: Dict[int, Tuple[str, ...]] = {
    n_wells: well_sequence(n_wells) for n_wells in sorted(schema.PLATE_FORMATS)}

#: The 384-well plate, kept under its old name for callers that imported it.
WELL_SEQUENCE: Tuple[str, ...] = WELL_SEQUENCES[DEFAULT_PLATE_FORMAT]
WELL_ROWS = 'ABCDEFGHIJKLMNOP'
WELL_COLS = tuple(range(1, 25))

#: Well used when the layout has no well folder at all (a flat drop).
DEFAULT_WELL = 'A01'

#: Optional readers: extension -> (module name, pip install command).
_READERS: Dict[str, Tuple[str, str]] = {
    '.nd2': ('nd2reader', 'pip install nd2reader'),
    '.czi': ('czifile', 'pip install czifile'),
    '.lif': ('readlif', 'pip install readlif'),
}

#: Columns of the map file, in order. Fixed — :func:`read_map` validates
#: against this and :func:`populate_db_from_map` writes exactly these
#: plus the spaCR join keys.
MAP_COLUMNS: Tuple[str, ...] = (
    'target', 'target_path', 'source', 'source_relpath',
    'plate', 'well', 'field', 'channel', 'z', 't',
    'source_plate', 'source_well', 'source_field', 'source_channel',
    'source_z', 'source_t',
    'plateID', 'rowID', 'columnID', 'fieldID', 'prc', 'prcf',
    'z_handling', 'n_z_planes', 'n_timepoints', 'status', 'meta_json',
)

#: Required columns for a file to be accepted as a spaCR conversion map.
_REQUIRED_MAP_COLUMNS: Tuple[str, ...] = (
    'target', 'source', 'plate', 'well', 'field', 'channel', 'z', 't')

#: Prefix for the temp files atomic writes go through. Anything left
#: behind with this prefix is a crashed run, never a usable image.
_TMP_PREFIX = '.spacr_convert_'

# Filename token patterns. Each requires a separator in front so that a
# stem like ``BC1`` is not read as channel 1.
_CHANNEL_TOKEN = re.compile(r'(?i)(?<=[_\-. ])(?:ch|channel|c|w)[_\-]?(\d{1,3})(?=$|[_\-. ])')
_Z_TOKEN = re.compile(r'(?i)(?<=[_\-. ])(?:z|zs|slice)[_\-]?(\d{1,4})(?=$|[_\-. ])')
_T_TOKEN = re.compile(r'(?i)(?<=[_\-. ])(?:t|time|tp)[_\-]?(\d{1,5})(?=$|[_\-. ])')

#: A name that already looks like Yokogawa output. Matched only to warn.
#: One or two row letters and two-or-more column digits, because a 1536
#: plate's wells are ``AA01`` and ``A48`` as well as ``A01``.
_YOKO_NAME = re.compile(
    r'(?i)^.+_[A-Z]{1,2}\d{2,}_T\d{4}F\d{3}L\d{2}(A\d{2})?(Z\d{2})?C\d{2}$')


# ---------------------------------------------------------------------------
# Optional readers
# ---------------------------------------------------------------------------

def _module_available(name: str) -> bool:
    """True when ``name`` can be imported, without importing it.

    Wraps :func:`importlib.util.find_spec` so a probe never costs the
    import and never raises: ``find_spec`` itself throws for a namespace
    package whose parent is missing, and a broken third-party
    ``__init__`` can raise anything at all.

    An already-imported module short-circuits the probe. ``find_spec``
    raises ``ValueError`` for a module in ``sys.modules`` whose
    ``__spec__`` is None, which is true of anything injected by hand —
    and reporting "not installed" for a module that is right there would
    be a lie.

    :param name: top-level module name, e.g. ``'nd2reader'``.
    """
    if sys.modules.get(name) is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def reader_requirement(ext: str) -> Optional[Tuple[str, str]]:
    """Return ``(module, pip command)`` needed to read ``ext``, or None.

    :param ext: file extension including the dot, case-insensitive.
    :returns: None for formats served by always-present dependencies
        (TIFF, PNG, JPEG, BMP).
    """
    return _READERS.get(str(ext).lower())


def reader_available(ext: str) -> bool:
    """True when ``ext`` can actually be read on this machine.

    :param ext: file extension including the dot.
    """
    requirement = reader_requirement(ext)
    if requirement is None:
        return True
    return _module_available(requirement[0])


def missing_reader_message(ext: str) -> str:
    """Return the user-facing sentence for an unreadable format.

    Names the package and the exact install command. This is what the
    plan and the ledger carry instead of an ImportError traceback.

    :param ext: file extension including the dot.
    """
    requirement = reader_requirement(ext)
    if requirement is None:
        return f'{ext} files are not a supported input format'
    module, command = requirement
    return (f'{ext} files need the optional "{module}" package, which is '
            f'not installed. Install it with: {command}')


def _import_reader(ext: str):
    """Import and return the reader module for ``ext``.

    :raises ConfigurationError: with :func:`missing_reader_message` when
        the package is absent — the message, never a traceback.
    """
    requirement = reader_requirement(ext)
    if requirement is None:
        raise ConfigurationError(missing_reader_message(ext))
    if not _module_available(requirement[0]):
        raise ConfigurationError(missing_reader_message(ext))
    try:
        return importlib.import_module(requirement[0])
    except Exception as exc:
        raise ConfigurationError(
            f'{missing_reader_message(ext)} (import failed: {exc})') from exc


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _natural_key(text: Any) -> Tuple[Tuple[int, str], ...]:
    """Sort key that orders ``C2`` before ``C10``.

    Digit runs compare numerically, everything else case-insensitively.
    Used everywhere an id is handed out from a sorted list, so the same
    tree always produces the same plate / well / field / channel numbers.
    """
    parts = [p for p in re.split(r'(\d+)', str(text)) if p != '']
    if not parts:
        return ((10 ** 15, ''),)
    return tuple((int(p), '') if p.isdigit() else (10 ** 15, p.lower())
                 for p in parts)


#: Public name for :func:`_natural_key`. :mod:`spacr.io` sorts its own
#: synthetic well assignment with it, so the two converters hand out ids in
#: the same order instead of each having an opinion.
natural_key = _natural_key


def _sanitise(token: str) -> str:
    """Return ``token`` reduced to characters that are safe in a filename.

    Underscores are stripped too: the ``cellvoyager`` regex splits the
    plate from the well on ``_``, so a plate literally called
    ``my_run`` would move the split point and misparse the well.
    """
    cleaned = re.sub(r'[^A-Za-z0-9]+', '-', str(token)).strip('-')
    return cleaned or 'plate'


def _split_ext(name: str) -> Tuple[str, str]:
    """Split ``name`` into (stem, extension), keeping ``.ome.tif`` whole."""
    lower = name.lower()
    for double in ('.ome.tif', '.ome.tiff'):
        if lower.endswith(double):
            return name[:-len(double)], double
    stem, ext = os.path.splitext(name)
    return stem, ext.lower()


def _strip_tokens(stem: str) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    """Pull the channel / z / t tokens out of a filename stem.

    ``fov02_Z03_C2`` -> ``('fov02', 'C2', 3, None)``. The remaining stem
    is the *field key*: every file that reduces to the same stem is one
    field-set, which is what makes ten ``fov*_C1/_C2`` pairs come out as
    ten fields with two channels rather than twenty fields.

    :returns: ``(field_key, channel_key, z_index, t_index)``. Any of the
        last three is None when no such token is present.
    """
    remaining = stem
    t_index: Optional[int] = None
    z_index: Optional[int] = None
    channel: Optional[str] = None

    match = _T_TOKEN.search(remaining)
    if match is not None:
        t_index = int(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]
    match = _Z_TOKEN.search(remaining)
    if match is not None:
        z_index = int(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]
    match = _CHANNEL_TOKEN.search(remaining)
    if match is not None:
        channel = f'C{int(match.group(1))}'
        remaining = remaining[:match.start()] + remaining[match.end():]

    remaining = re.sub(r'[_\-. ]{2,}', '_', remaining).strip('_- .')
    return (remaining or stem), channel, z_index, t_index


def normalise_well(name: str, *, n_wells: Optional[int] = None) -> Optional[str]:
    """Return ``name`` as a canonical well id (``A01``, ``AA48``), or None.

    ``a1``, ``A-1`` and ``A01`` all normalise to ``A01``; ``aa1`` gives
    ``AA01``, which is a real well of a 1536-plate. Anything that is not a
    well address at all (``wt``, ``KO_clone3``, ``fov01``) returns None and
    gets a synthetic id from :func:`assign_wells` instead.

    This used to be a private ``^([A-Pa-p])[ _\\-]?(\\d{1,2})$`` with an
    extra ``1 <= column <= 24`` check, so ``Q01`` (row 17) and ``A25``
    (column 25) — both perfectly good wells of the 1536-plate the heatmap
    already draws — came back None and were *renamed* to the next free
    synthetic address. :func:`spacr.schema.parse_well` is now the parser,
    and :func:`spacr.schema.plate_format_for` decides whether the position
    it produced is a well that exists.

    :param name: the source well-folder name.
    :param n_wells: restrict to one plate format, e.g. ``384``. The
        default accepts any address that exists on *some* standard plate,
        which is to say up to 1536.
    :returns: the canonical well id, or None when ``name`` is not one.
    """
    try:
        row, column = schema.parse_well(str(name).strip(), strict=True)
    except schema.WellParseError:
        return None
    plate_format = schema.plate_format_for(row, column)
    if plate_format is None:
        # Reads as a position (``ZZ99`` -> r702/c99) but no standard plate
        # has it. See :func:`off_plate_reason`, which is what the plan says.
        return None
    if n_wells is not None and plate_format > n_wells:
        return None
    return schema.well_id(row, column)


def off_plate_reason(name: str) -> Optional[str]:
    """Say why ``name`` looks like a well but is not one, or return None.

    The dangerous middle case. ``wt`` is obviously not a well and nobody is
    surprised when it gets a synthetic address; ``ZZ99`` and ``A0`` *parse*
    into a row and a column and then turn out to sit on no plate that
    exists, so handing them a synthetic address silently is how a typo
    becomes a well name nobody can trace.

    :param name: the source well-folder name.
    :returns: a sentence naming the name and the position it read as, or
        None when the name either is a real well or is not well-shaped.
    """
    try:
        row, column = schema.parse_well(str(name).strip(), strict=True)
    except schema.WellParseError:
        return None
    if schema.plate_format_for(row, column) is not None:
        return None
    return (f'{name!r} reads as row {row} column {column}, which exists on no '
            f'standard plate (known formats: '
            f'{", ".join(str(n) for n in sorted(schema.PLATE_FORMATS))} '
            f'wells)')


def plate_format_for_names(n_names: int, wells: Sequence[str],
                           minimum: int = DEFAULT_PLATE_FORMAT) -> Optional[int]:
    """Return the plate format that has to be used for one plate's wells.

    The smallest standard format that is at least ``minimum``, holds every
    address in ``wells``, and has room for ``n_names`` distinct sources.
    That second clause is the 1536 fix: one folder named ``AA01`` means the
    plate *is* a 1536, whether or not there are 1536 of them.

    :param n_names: how many distinct source names must be given a well.
    :param wells: the canonical addresses already claimed by name.
    :param minimum: never return a format smaller than this.
    :returns: the well count of the format to use, or None when the names
        fit no standard plate.
    """
    needed = int(minimum)
    for well in wells:
        row, column = schema.parse_well(well)
        # normalise_well is what produced these, so plate_format_for cannot
        # come back None here.
        needed = max(needed, schema.plate_format_for(row, column))
    for n_wells in sorted(schema.PLATE_FORMATS):
        if n_wells >= needed and n_wells >= n_names:
            return n_wells
    return None


def assign_wells(names: Sequence[str], *,
                 n_wells: Optional[int] = None) -> Dict[str, str]:
    """Map arbitrary well-folder names onto plate well ids.

    The rule, in full:

    1. A name that already *is* a well address keeps it —
       :func:`normalise_well` handles ``a1`` / ``A-1`` / ``A01`` / ``aa1``.
    2. The plate format is chosen by :func:`plate_format_for_names`: 384
       unless a claimed address or the sheer number of names needs the
       1536, in which case rows run to ``AF`` and columns to 48.
    3. Every remaining name is sorted with :func:`_natural_key` and handed
       the next free id of that plate, skipping any id claimed in step 1.

    All three are deterministic — the same folder names always produce the
    same wells — and step 3 is only *reversible* because the map file
    records ``source_well`` next to ``well``. Which is the point: after
    conversion ``plate1_A01`` means nothing without the map, so
    :func:`plan` also reports every synthetic assignment by name.

    :param names: the well-folder names found for one plate.
    :param n_wells: force a plate format instead of choosing one.
    :returns: ``{original name: well id}``.
    :raises ConfigurationError: when the names fit no standard plate — the
        real limit is 1536, not 384.
    """
    unique = sorted({str(n) for n in names}, key=_natural_key)
    assigned: Dict[str, str] = {}
    claimed = set()
    for name in unique:
        canonical = normalise_well(name, n_wells=n_wells)
        if canonical is not None:
            assigned[name] = canonical
            claimed.add(canonical)

    if n_wells is None:
        plate_format = plate_format_for_names(len(unique), sorted(claimed))
        limit = max(schema.PLATE_FORMATS)
    else:
        # well_sequence validates the format, so a non-standard n_wells is a
        # ConfigurationError naming the known formats, not a KeyError.
        limit = len(well_sequence(n_wells))
        plate_format = n_wells if len(unique) <= limit else None
    if plate_format is None:
        n_rows, n_columns = schema.PLATE_FORMATS[limit]
        raise ConfigurationError(
            f'{len(unique)} distinct wells were found, but the largest plate '
            f'available here has {limit} ({n_rows}x{n_columns}); split the '
            f'source into more plate folders, or pass well_map= explicitly.')

    free = (well for well in WELL_SEQUENCES[plate_format]
            if well not in claimed)
    for name in unique:
        if name in assigned:
            continue
        # plate_format_for_names sized the plate to len(unique), and every
        # claimed id is one of the names being counted, so the sequence
        # cannot run dry before the names do.
        assigned[name] = next(free)
    return assigned


def target_name(plate: str, well: str, field: int, channel: int,
                z: int = 1, t: int = 1, action: int = 1) -> str:
    """Build one Yokogawa filename.

    ``plate1_A01_T0001F001L01A01Z01C01.tif`` — the exact shape
    ``spacr.utils._get_regex('cellvoyager', 'tif')`` parses, so a folder
    of these can be handed to Mask/Measure with
    ``metadata_type='cellvoyager'`` and nothing else.

    :param plate: plate token (no underscores — see :func:`_sanitise`).
    :param well: canonical well id, e.g. ``A01``.
    :param field: 1-based field id.
    :param channel: 1-based channel id.
    :param z: 1-based z-slice id.
    :param t: 1-based timepoint id.
    :param action: the ``A##`` action id; spaCR ignores it, so it is 1.
    """
    return (f'{plate}_{well}_T{int(t):04d}F{int(field):03d}L01'
            f'A{int(action):02d}Z{int(z):02d}C{int(channel):02d}.tif')


def _well_ids(well: str) -> Tuple[str, str]:
    """Return ``(rowID, columnID)`` in spaCR's ``r1`` / ``c1`` form.

    :func:`spacr.schema.parse_well` is the definition; this wrapper only adds
    the passthrough for a well with no column at all, which the map file needs
    because a conversion must produce a row for every source file even when
    the source folder is named something spaCR cannot read as a well.

    :param well: well identifier.
    :returns: ``(rowID, columnID)``.
    """
    try:
        return schema.parse_well(well)
    except schema.WellParseError:
        text = str(well)
        return text, text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceImage:
    """One readable unit of input: a file, or one series inside a file.

    A vendor file holding six scenes produces six ``SourceImage``\\ s, so
    "each field a unique field id" holds whether the fields arrived as
    separate files or as series inside one.

    :ivar path: absolute path of the source file.
    :ivar plate: the *source* plate key — the folder name, before any
        renaming. ``''`` when the layout has no plate folder.
    :ivar well: the *source* well key — the folder name, before any
        renaming.
    :ivar field: the *source* field key: the filename stem with its
        channel/z/t tokens removed, plus ``#s<n>`` for a series.
    :ivar channel: the *source* channel key parsed from the filename
        (``'C2'``), or None when the channels live inside the file.
    :ivar z: number of z planes this source contributes.
    :ivar t: number of timepoints this source contributes.
    :ivar meta: everything else — ``ext``, ``shape``, ``dtype``,
        ``axes``, ``series``, ``z_index``/``t_index`` (the plane index
        parsed from the filename, when the stack is spread across
        files), ``axes_assumed`` and, for a source that cannot be read,
        ``error``.
    :ivar n_channels: number of channels inside the file itself.
    """

    path: str
    plate: str
    well: str
    field: str
    channel: Optional[str] = None
    z: int = 1
    t: int = 1
    meta: TMapping[str, Any] = dc_field(default_factory=dict)
    n_channels: int = 1

    @property
    def readable(self) -> bool:
        """False when :func:`scan` could not open this source."""
        return not self.meta.get('error')

    @property
    def error(self) -> str:
        """Why this source is unreadable, or ``''``."""
        return str(self.meta.get('error') or '')

    @property
    def ext(self) -> str:
        """Lower-case extension, e.g. ``'.nd2'``."""
        return str(self.meta.get('ext') or _split_ext(os.path.basename(self.path))[1])


@dataclass(frozen=True)
class Mapping:
    """One source plane and the one output TIFF it becomes.

    Every field needed to walk the arrow backwards is here, which is why
    the map file can be a straight dump of these.

    :ivar source: absolute source path.
    :ivar target: output filename (basename only).
    :ivar plate: assigned plate token.
    :ivar well: assigned well id.
    :ivar field: assigned 1-based field id.
    :ivar channel: assigned 1-based channel id.
    :ivar z: assigned 1-based z id.
    :ivar t: assigned 1-based timepoint id.
    :ivar source_z: the z index this came from — the filename token or
        the plane index — or ``'max(1..N)'`` when it was projected.
    :ivar plane: ``(t, z, c)`` index into the source's 5-D array;
        ``z == -1`` means "every plane, projected".
    """

    source: str
    target: str
    plate: str
    well: str
    field: int
    channel: int
    z: int = 1
    t: int = 1
    source_plate: str = ''
    source_well: str = ''
    source_field: str = ''
    source_channel: str = ''
    source_z: str = ''
    source_t: str = ''
    z_handling: str = Z_KEEP
    n_z_planes: int = 1
    n_timepoints: int = 1
    plane: Tuple[int, int, int] = (0, 0, 0)
    meta: TMapping[str, Any] = dc_field(default_factory=dict)

    def to_row(self, dst: str = '', status: str = 'converted') -> Dict[str, Any]:
        """Render this mapping as one map-file row.

        :param dst: destination folder, used to fill ``target_path``.
        :param status: ``'converted'``, ``'existing'`` or ``'failed'``.
        """
        row_id, column_id = _well_ids(self.well)
        field_key = schema.field_id(self.field)
        # Joined here rather than through schema.compose_prc: the plate token
        # is a sanitised source folder name and must be allowed to be anything
        # a conversion can produce, including a name schema would refuse.
        prc = schema.KEY_SEPARATOR.join([str(self.plate), row_id, column_id])
        return {
            'target': self.target,
            'target_path': os.path.join(dst, self.target) if dst else self.target,
            'source': self.source,
            'source_relpath': str(self.meta.get('source_relpath') or
                                  os.path.basename(self.source)),
            'plate': self.plate,
            'well': self.well,
            'field': int(self.field),
            'channel': int(self.channel),
            'z': int(self.z),
            't': int(self.t),
            'source_plate': self.source_plate,
            'source_well': self.source_well,
            'source_field': self.source_field,
            'source_channel': self.source_channel,
            'source_z': self.source_z,
            'source_t': self.source_t,
            schema.PLATE_KEY: self.plate,
            schema.ROW_KEY: row_id,
            schema.COLUMN_KEY: column_id,
            schema.FIELD_KEY: field_key,
            schema.PRC_KEY: prc,
            schema.PRCF_KEY: schema.KEY_SEPARATOR.join([prc, field_key]),
            'z_handling': self.z_handling,
            'n_z_planes': int(self.n_z_planes),
            'n_timepoints': int(self.n_timepoints),
            'status': status,
            'meta_json': json.dumps(dict(self.meta), default=str, sort_keys=True),
        }


@dataclass
class ConversionPlan:
    """The preview: what would be written, and what would go wrong.

    Nothing in here has touched the disk. ``plan.to_frame()`` is the
    table a user reads before deciding; ``plan.ok`` is what a caller
    checks before handing it to :func:`convert`.

    :ivar mappings: one entry per output TIFF.
    :ivar errors: blocking problems. A non-empty list means
        :func:`convert` refuses — the only member today is a target-name
        collision, which is exactly the case where writing anyway would
        destroy data.
    :ivar warnings: non-blocking but load-bearing: z projection, files
        that cannot be read, an already-converted-looking source.
    :ivar notes: neutral facts about the plan (counts, assumptions).
    :ivar unreadable: sources :func:`scan` could not open, carried
        through so the preview shows them and the ledger counts them.
    :ivar well_map: ``{(plate key, well key): well id}`` — the record of
        how folder names became wells.
    :ivar plate_map: ``{plate key: plate token}``.
    """

    mappings: List[Mapping] = dc_field(default_factory=list)
    errors: List[str] = dc_field(default_factory=list)
    warnings: List[str] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)
    sources: List[SourceImage] = dc_field(default_factory=list)
    unreadable: List[SourceImage] = dc_field(default_factory=list)
    well_map: Dict[Tuple[str, str], str] = dc_field(default_factory=dict)
    plate_map: Dict[str, str] = dc_field(default_factory=dict)
    channel_map: Dict[Tuple[str, str], int] = dc_field(default_factory=dict)
    z_handling: str = Z_KEEP

    def __len__(self) -> int:
        return len(self.mappings)

    @property
    def ok(self) -> bool:
        """True when the plan can be converted (no blocking errors)."""
        return not self.errors

    @property
    def n_sources(self) -> int:
        """Distinct source files referenced by the plan."""
        return len({m.source for m in self.mappings})

    def to_frame(self) -> pd.DataFrame:
        """Return the preview table.

        One row per output, plus one row per unreadable source with an
        empty ``target`` and the reason in ``status`` — a preview that
        quietly omitted the files it could not read would be exactly the
        wrong shape of honest.
        """
        rows: List[Dict[str, Any]] = []
        for mapping in self.mappings:
            rows.append({
                'source': mapping.source,
                'target': mapping.target,
                'plate': mapping.plate,
                'well': mapping.well,
                'field': mapping.field,
                'channel': mapping.channel,
                'z': mapping.z,
                't': mapping.t,
                'source_plate': mapping.source_plate,
                'source_well': mapping.source_well,
                'source_field': mapping.source_field,
                'source_channel': mapping.source_channel,
                'z_handling': mapping.z_handling,
                'n_z_planes': mapping.n_z_planes,
                'status': 'planned',
            })
        for source in self.unreadable:
            rows.append({
                'source': source.path,
                'target': '',
                'plate': '',
                'well': '',
                'field': 0,
                'channel': 0,
                'z': 0,
                't': 0,
                'source_plate': source.plate,
                'source_well': source.well,
                'source_field': source.field,
                'source_channel': source.channel or '',
                'z_handling': self.z_handling,
                'n_z_planes': 0,
                'status': f'SKIP — {source.error}',
            })
        columns = ['source', 'target', 'plate', 'well', 'field', 'channel',
                   'z', 't', 'source_plate', 'source_well', 'source_field',
                   'source_channel', 'z_handling', 'n_z_planes', 'status']
        return pd.DataFrame(rows, columns=columns)

    def summary(self) -> str:
        """A short human-readable rendering of the plan."""
        lines = [f'{len(self.mappings)} file(s) would be written from '
                 f'{self.n_sources} source(s).']
        lines.extend(self.notes)
        for warning in self.warnings:
            lines.append(f'WARNING: {warning}')
        for error in self.errors:
            lines.append(f'ERROR: {error}')
        return '\n'.join(lines)


@dataclass
class ConversionResult:
    """What :func:`convert` actually did.

    :ivar written: mappings whose TIFF was created by this run.
    :ivar existing: mappings whose target was already on disk and was
        therefore left alone — a re-run is a no-op, never a rewrite.
    :ivar failed: mappings belonging to sources that raised.
    :ivar skipped: sources skipped without being attempted (an absent
        optional reader, an unreadable file), as ``(path, reason)``.
    :ivar ledger: the :class:`spacr.errors.RunLedger` for the run.
    :ivar map_path: where the map file went.
    :ivar checkpoint_path: atomic field-level resume document.
    :ivar resumed_fields: fields accepted from a compatible checkpoint.
    """

    plan: ConversionPlan
    dst: str
    written: List[Mapping] = dc_field(default_factory=list)
    existing: List[Mapping] = dc_field(default_factory=list)
    failed: List[Mapping] = dc_field(default_factory=list)
    skipped: List[Tuple[str, str]] = dc_field(default_factory=list)
    ledger: Optional[RunLedger] = None
    map_path: str = ''
    checkpoint_path: str = ''
    resumed_fields: List[str] = dc_field(default_factory=list)

    @property
    def n_written(self) -> int:
        """How many TIFFs this run created."""
        return len(self.written)

    @property
    def n_skipped(self) -> int:
        """How many sources were skipped, for any reason."""
        return len(self.skipped)

    @property
    def is_complete(self) -> bool:
        """True when nothing was skipped and nothing failed."""
        if self.skipped or self.failed:
            return False
        return self.ledger is None or self.ledger.is_complete

    def rows(self) -> List[Dict[str, Any]]:
        """Every map-file row for this result, in write order."""
        rows = [m.to_row(self.dst, 'converted') for m in self.written]
        rows += [m.to_row(self.dst, 'existing') for m in self.existing]
        rows += [m.to_row(self.dst, 'failed') for m in self.failed]
        return rows

    def summary(self) -> str:
        """The end-of-run sentence, including *why* things were skipped."""
        lines = [f'Converted {self.n_written} file(s) into {self.dst}.']
        if self.existing:
            lines.append(f'{len(self.existing)} target(s) already existed and '
                         f'were left untouched.')
        if self.failed:
            lines.append(f'{len(self.failed)} planned file(s) were not written '
                         f'because their source failed.')
        if self.skipped:
            lines.append(f'Skipped {len(self.skipped)} source(s):')
            for path, reason in self.skipped:
                lines.append(f'  {os.path.basename(path)}: {reason}')
        if self.map_path:
            lines.append(f'Map file: {self.map_path}')
        if self.resumed_fields:
            lines.append(
                f'Resumed {len(self.resumed_fields)} completed field(s) from '
                f'{self.checkpoint_path}.')
        elif self.checkpoint_path:
            lines.append(f'Checkpoint: {self.checkpoint_path}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _iter_files(src: str, extensions: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    """Yield ``(abs path, relative path parts)`` for every image under ``src``."""
    found: List[Tuple[str, Tuple[str, ...]]] = []
    exts = tuple(e.lower() for e in extensions)
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for name in sorted(filenames):
            if name.startswith('.'):
                continue
            lower = name.lower()
            if not any(lower.endswith(e) for e in exts):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, src)
            found.append((path, tuple(rel.split(os.sep))))
    return found


def _detect_layout(entries: Sequence[Tuple[str, Tuple[str, ...]]]) -> str:
    """Pick a layout from how deep the image files sit under ``src``."""
    if not entries:
        return 'flat'
    depth = max(len(parts) - 1 for _path, parts in entries)
    if depth <= 0:
        return 'flat'
    if depth == 1:
        return 'well'
    return 'plate_well'


def _keys_for(parts: Tuple[str, ...], layout: str, src_name: str) -> Tuple[str, str, str]:
    """Return ``(plate key, well key, extra field prefix)`` for one file."""
    dirs = list(parts[:-1])
    if layout == 'flat':
        return src_name, DEFAULT_WELL, ''
    if layout == 'well':
        plate = src_name
        well = dirs[0] if dirs else DEFAULT_WELL
        extra = dirs[1:]
    else:  # plate_well
        plate = dirs[0] if dirs else src_name
        well = dirs[1] if len(dirs) > 1 else DEFAULT_WELL
        extra = dirs[2:]
    return plate, well, '/'.join(extra)


def _axes_dims(shape: Sequence[int], axes: str) -> Tuple[int, int, int, str]:
    """Resolve a shape + tifffile axes string into ``(n_t, n_z, n_c, note)``.

    tifffile only reports ``T``/``Z``/``C`` when the file actually
    recorded them; a plain ``imwrite`` of a 3-D array comes back as
    ``QYX`` (unknown) or ``SYX`` (samples). Rather than guess silently
    the way :func:`spacr.io.convert_to_yokogawa` does, the guess is made
    explicit and returned as a note that ends up in the plan.

    :returns: ``(n_t, n_z, n_c, note)``; ``note`` is ``''`` when the file
        said what its axes were.
    """
    axes = (axes or '').upper()
    sizes = dict(zip(axes, shape))
    n_t = int(sizes.get('T', 0) or 0)
    n_z = int(sizes.get('Z', 0) or 0)
    n_c = int(sizes.get('C', 0) or 0)
    note = ''
    if not n_c and 'S' in sizes:
        # 'S' is tifffile's "samples per pixel" — RGB-style interleaving.
        # Treating those samples as separate spaCR channels is a decision,
        # not a fact recorded in the file, so it gets said out loud.
        n_c = int(sizes['S'])
        note = (f"the file's {n_c} interleaved samples were read as channels; "
                f'if that is an RGB rendering rather than {n_c} stains, '
                f'export the channels separately')

    unknown = [(i, letter) for i, letter in enumerate(axes)
               if letter in ('Q', 'I', '?')]
    if unknown:
        # Left-to-right: T, then Z. A single unknown axis of 4 or fewer
        # planes reads as channels — the same heuristic io.py uses, but
        # said out loud instead of applied in silence.
        if len(unknown) == 1 and not n_c and shape[unknown[0][0]] <= 4:
            n_c = int(shape[unknown[0][0]])
            note = (f'axes not recorded in the file; the leading axis of '
                    f'{n_c} was read as channels')
        else:
            labels = []
            for order, (index, _letter) in enumerate(unknown):
                size = int(shape[index])
                if order == 0 and not n_t and len(unknown) > 1:
                    n_t, label = size, 'T'
                elif not n_z:
                    n_z, label = size, 'Z'
                elif not n_t:
                    n_t, label = size, 'T'
                elif not n_c:
                    n_c, label = size, 'C'
                else:
                    label = 'ignored'
                labels.append(f'{label}={size}')
            note = ('axes not recorded in the file; assumed '
                    + ', '.join(labels))
    return max(n_t, 1), max(n_z, 1), max(n_c, 1), note


def _describe_tiff(path: str) -> Dict[str, Any]:
    """Read a TIFF's dimensions from its header — no pixel data."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        shape = tuple(int(s) for s in series.shape)
        axes = str(series.axes or '')
        dtype = str(series.dtype)
        n_series = len(handle.series)
    n_t, n_z, n_c, note = _axes_dims(shape, axes)
    return {'shape': shape, 'axes': axes, 'dtype': dtype, 'n_t': n_t,
            'n_z': n_z, 'n_c': n_c, 'n_series': n_series,
            'axes_assumed': note, 'reader': 'tifffile'}


def _describe_plain(path: str) -> Dict[str, Any]:
    """Read a PNG/JPEG/BMP's dimensions via Pillow — no pixel decode."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    n_c = {'L': 1, 'I': 1, 'I;16': 1, 'F': 1, 'LA': 2,
           'RGB': 3, 'RGBA': 4}.get(mode, 1)
    note = ''
    if n_c > 1:
        note = (f"the {mode} image's {n_c} interleaved samples were read as "
                f'channels; if that is an RGB rendering rather than {n_c} '
                f'stains, export the channels separately')
    return {'shape': (height, width) if n_c == 1 else (height, width, n_c),
            'axes': 'YX' if n_c == 1 else 'YXS', 'dtype': mode,
            'n_t': 1, 'n_z': 1, 'n_c': n_c, 'n_series': 1,
            'axes_assumed': note, 'reader': 'PIL'}


def _describe_nd2(path: str) -> Dict[str, Any]:
    """Read an ND2's dimensions via ``nd2reader``."""
    module = _import_reader('.nd2')
    with module.ND2Reader(path) as handle:
        sizes = dict(getattr(handle, 'sizes', {}) or {})
    return {'shape': (int(sizes.get('y', 0)), int(sizes.get('x', 0))),
            'axes': 'ND2', 'dtype': '',
            'n_t': max(int(sizes.get('t', 1) or 1), 1),
            'n_z': max(int(sizes.get('z', 1) or 1), 1),
            'n_c': max(int(sizes.get('c', 1) or 1), 1),
            'n_series': max(int(sizes.get('v', 1) or 1), 1),
            'axes_assumed': '', 'reader': 'nd2reader'}


def _describe_czi(path: str) -> Dict[str, Any]:
    """Read a CZI's dimensions via ``czifile`` (header only)."""
    module = _import_reader('.czi')
    with module.CziFile(path) as handle:
        shape = tuple(int(s) for s in handle.shape)
        axes = str(handle.axes or '').upper()
        dtype = str(getattr(handle, 'dtype', ''))
    sizes = dict(zip(axes, shape))
    return {'shape': shape, 'axes': axes, 'dtype': dtype,
            'n_t': max(int(sizes.get('T', 1) or 1), 1),
            'n_z': max(int(sizes.get('Z', 1) or 1), 1),
            'n_c': max(int(sizes.get('C', 1) or 1), 1),
            'n_series': max(int(sizes.get('S', 1) or 1), 1),
            'axes_assumed': '', 'reader': 'czifile'}


def _describe_lif(path: str) -> Dict[str, Any]:
    """Read a LIF's dimensions via ``readlif``."""
    module = _import_reader('.lif')
    reader = module.Reader(path)
    images = list(reader.getIterImage())
    if not images:
        raise ConfigurationError(f'{path} contains no images')
    first = images[0]
    dims = getattr(first, 'dims', None)
    return {'shape': (int(getattr(dims, 'y', 0) or 0),
                      int(getattr(dims, 'x', 0) or 0)),
            'axes': 'LIF', 'dtype': '',
            'n_t': max(int(getattr(dims, 't', 1) or 1), 1),
            'n_z': max(int(getattr(dims, 'z', 1) or 1), 1),
            'n_c': max(int(getattr(first, 'channels', 1) or 1), 1),
            'n_series': len(images),
            'axes_assumed': '', 'reader': 'readlif'}


def _describe(path: str, ext: str) -> Dict[str, Any]:
    """Dispatch to the right describer for ``ext``."""
    if ext in ('.tif', '.tiff', '.ome.tif', '.ome.tiff'):
        return _describe_tiff(path)
    if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        return _describe_plain(path)
    if ext == '.nd2':
        return _describe_nd2(path)
    if ext == '.czi':
        return _describe_czi(path)
    if ext == '.lif':
        return _describe_lif(path)
    raise ConfigurationError(f'{ext} is not a supported input format')


def scan(src: str, layout: str = 'auto',
         extensions: Optional[Sequence[str]] = None) -> List[SourceImage]:
    """Walk ``src`` and describe every image it holds. Writes nothing.

    This is the read-only half of the converter: it opens headers, not
    pixels, and produces the :class:`SourceImage` list that :func:`plan`
    turns into a preview.

    Layouts:

    ``'plate_well'``
        ``src/<plate>/<well>/…`` — the user's ``run1/wt/`` case.
    ``'well'``
        ``src/<well>/…``, with ``src``'s own name as the plate.
    ``'flat'``
        images directly in ``src``; one plate, one well (``A01``), one
        field per file.
    ``'auto'`` (default)
        chooses from how deep the images actually sit.

    A file whose reader is not installed, or that will not open, comes
    back as a ``SourceImage`` with ``meta['error']`` set rather than
    raising — the plan shows it, the ledger counts it and the summary
    names it.

    :param src: folder to scan.
    :param layout: one of :data:`LAYOUTS`.
    :param extensions: override the scanned extensions.
    :returns: one :class:`SourceImage` per file (or per series in a file).
    :raises ConfigurationError: when ``src`` is not a directory or
        ``layout`` is not recognised.
    """
    if not src or not os.path.isdir(src):
        raise ConfigurationError(f'Source folder does not exist: {src!r}')
    if layout not in LAYOUTS:
        raise ConfigurationError(
            f'Unknown layout {layout!r}; expected one of {", ".join(LAYOUTS)}')

    exts = tuple(extensions) if extensions else IMAGE_EXTENSIONS
    entries = _iter_files(src, exts)
    resolved = _detect_layout(entries) if layout == 'auto' else layout
    src_name = os.path.basename(os.path.normpath(src)) or 'plate'

    sources: List[SourceImage] = []
    for path, parts in entries:
        plate_key, well_key, extra = _keys_for(parts, resolved, src_name)
        stem, ext = _split_ext(parts[-1])
        field_key, channel_key, z_index, t_index = _strip_tokens(stem)
        if extra:
            field_key = f'{extra}/{field_key}'
        rel = os.path.relpath(path, src)
        base_meta: Dict[str, Any] = {
            'ext': ext, 'layout': resolved, 'source_relpath': rel,
            'stem': stem, 'z_index': z_index, 't_index': t_index,
            'looks_converted': bool(_YOKO_NAME.match(stem)),
        }

        if not reader_available(ext):
            sources.append(SourceImage(
                path=path, plate=plate_key, well=well_key, field=field_key,
                channel=channel_key, z=0, t=0,
                meta=dict(base_meta, error=missing_reader_message(ext))))
            continue

        try:
            described = _describe(path, ext)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            sources.append(SourceImage(
                path=path, plate=plate_key, well=well_key, field=field_key,
                channel=channel_key, z=0, t=0,
                meta=dict(base_meta, error=f'{exc.__class__.__name__}: {message}')))
            continue

        n_series = max(int(described.get('n_series', 1) or 1), 1)
        for series in range(n_series):
            series_field = field_key if n_series == 1 else f'{field_key}#s{series + 1}'
            sources.append(SourceImage(
                path=path,
                plate=plate_key,
                well=well_key,
                field=series_field,
                channel=channel_key,
                z=int(described['n_z']),
                t=int(described['n_t']),
                n_channels=int(described['n_c']),
                meta=dict(base_meta, series=series, **{
                    k: v for k, v in described.items() if k != 'n_series'})))
    return sources


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _channel_keys(source: SourceImage) -> List[str]:
    """Return the channel keys this source contributes.

    A filename channel token wins; otherwise the channels inside the
    file become ``C1…Cn``, so a folder mixing ``fov_C1.tif`` with a
    2-channel ``fov.tif`` still lines the two up on the same ids.
    """
    if source.channel:
        return [source.channel]
    return [f'C{i + 1}' for i in range(max(int(source.n_channels), 1))]


def plan(sources: Sequence[SourceImage], z_handling: str = Z_KEEP,
         plate_naming: str = 'index',
         well_map: Optional[TMapping[Any, str]] = None,
         plate_map: Optional[TMapping[str, str]] = None) -> ConversionPlan:
    """Turn scanned sources into the preview table. Writes nothing.

    Ids are handed out deterministically, always from a
    :func:`_natural_key` sort so that the same tree yields the same
    numbers on every machine:

    * **plate** — ``plate1, plate2, …`` in sorted order of the source
      plate folder (``plate_naming='index'``, the default, and what
      produces the ``plate1_A01_…`` the converter is specified against),
      or the sanitised folder name with ``plate_naming='name'``.
    * **well** — see :func:`assign_wells`.
    * **field** — 1..N per well, over the distinct field keys.
    * **channel** — 1..N per plate, over the distinct channel keys, so
      ``C01`` means the same stain in every well of a plate.

    ``z_handling`` is explicit on purpose. :data:`Z_KEEP` (default)
    writes every plane; :data:`Z_MAX` max-projects and :data:`Z_FIRST`
    keeps only plane 1 — both are announced in :attr:`ConversionPlan.warnings`
    and recorded per row in the map file, because a converter that
    silently flattens a z-stack loses data nobody notices for months.

    :param sources: output of :func:`scan`.
    :param z_handling: one of :data:`Z_HANDLING`.
    :param plate_naming: ``'index'`` or ``'name'``.
    :param well_map: explicit ``{well key: well id}`` or
        ``{(plate key, well key): well id}`` overrides.
    :param plate_map: explicit ``{plate key: plate token}`` overrides.
    :returns: a :class:`ConversionPlan`.
    :raises ConfigurationError: for an unknown ``z_handling`` or
        ``plate_naming``.
    """
    if z_handling not in Z_HANDLING:
        raise ConfigurationError(
            f'Unknown z_handling {z_handling!r}; expected one of '
            f'{", ".join(Z_HANDLING)}')
    if plate_naming not in ('index', 'name'):
        raise ConfigurationError(
            f'Unknown plate_naming {plate_naming!r}; expected "index" or "name"')

    result = ConversionPlan(z_handling=z_handling)
    readable = [s for s in sources if s.readable]
    result.sources = list(readable)
    result.unreadable = [s for s in sources if not s.readable]

    for source in result.unreadable:
        result.warnings.append(f'{source.path}: {source.error}')

    if not readable:
        result.notes.append('No readable images were found.')
        return result

    # -- plate names ------------------------------------------------------
    plate_keys = sorted({s.plate for s in readable}, key=_natural_key)
    overrides = dict(plate_map or {})
    for index, key in enumerate(plate_keys, start=1):
        if key in overrides:
            result.plate_map[key] = _sanitise(overrides[key])
        elif plate_naming == 'index':
            result.plate_map[key] = f'plate{index}'
        else:
            result.plate_map[key] = _sanitise(key)

    # -- wells, per plate -------------------------------------------------
    explicit_wells = dict(well_map or {})
    for plate_key in plate_keys:
        well_keys = {s.well for s in readable if s.plate == plate_key}
        pending = []
        for well_key in sorted(well_keys, key=_natural_key):
            forced = explicit_wells.get((plate_key, well_key),
                                        explicit_wells.get(well_key))
            if forced:
                result.well_map[(plate_key, well_key)] = str(forced).upper()
            else:
                pending.append(well_key)
        claimed = {result.well_map[(plate_key, k)]
                   for k in well_keys if (plate_key, k) in result.well_map}
        if pending:
            assigned = assign_wells(pending + sorted(claimed))
            for well_key in pending:
                result.well_map[(plate_key, well_key)] = assigned[well_key]

            # A synthetic address is never handed out silently. A name that
            # is a well keeps it (including Q01 and A25, which a 1536 plate
            # has and a 384 does not); everything else is listed here by
            # name, and a name that *looks* like a well but sits on no plate
            # at all is a warning of its own — that is the case where a typo
            # turns into a well id nobody can trace back.
            synthetic = [(key, assigned[key])
                         for key in sorted(pending, key=_natural_key)
                         if normalise_well(key) is None]
            for name, well in synthetic:
                reason = off_plate_reason(name)
                if reason is not None:
                    result.warnings.append(
                        f'{reason}. It was given the synthetic address '
                        f'{well}; the map file records the original name in '
                        f'source_well.')
            if synthetic:
                listed = ', '.join(f'{name!r} -> {well}'
                                   for name, well in synthetic)
                result.notes.append(
                    f'{len(synthetic)} source name(s) under {plate_key!r} are '
                    f'not well addresses and were given a synthetic one: '
                    f'{listed}. Only the map file\'s source_well column can '
                    f'take that back.')

    # -- channels, per plate ----------------------------------------------
    for plate_key in plate_keys:
        keys = set()
        for source in readable:
            if source.plate == plate_key:
                keys.update(_channel_keys(source))
        for index, key in enumerate(sorted(keys, key=_natural_key), start=1):
            result.channel_map[(plate_key, key)] = index

    # -- fields, per (plate, well) ----------------------------------------
    field_map: Dict[Tuple[str, str, str], int] = {}
    for plate_key in plate_keys:
        well_keys = sorted({s.well for s in readable if s.plate == plate_key},
                           key=_natural_key)
        for well_key in well_keys:
            fields = sorted({s.field for s in readable
                             if s.plate == plate_key and s.well == well_key},
                            key=_natural_key)
            for index, field_key in enumerate(fields, start=1):
                field_map[(plate_key, well_key, field_key)] = index

    # -- mappings ----------------------------------------------------------
    for source in sorted(readable, key=lambda s: (_natural_key(s.plate),
                                                  _natural_key(s.well),
                                                  _natural_key(s.field),
                                                  _natural_key(s.channel or ''),
                                                  s.path)):
        plate = result.plate_map[source.plate]
        well = result.well_map[(source.plate, source.well)]
        field = field_map[(source.plate, source.well, source.field)]
        z_index = source.meta.get('z_index')
        t_index = source.meta.get('t_index')
        series = int(source.meta.get('series', 0) or 0)

        # A filename z/t token means the stack is spread over files: the
        # token is the output index. A file that ALSO holds planes
        # internally is ambiguous, and guessing which one wins is how
        # planes silently overwrite each other, so it is an error.
        if z_index is not None and source.z > 1:
            result.errors.append(
                f'{source.path}: the filename carries a Z{z_index} token but '
                f'the file also holds {source.z} z planes — remove the token '
                f'or split the file, otherwise the two numbering schemes '
                f'collide.')
            continue
        if t_index is not None and source.t > 1:
            result.errors.append(
                f'{source.path}: the filename carries a T{t_index} token but '
                f'the file also holds {source.t} timepoints.')
            continue
        if source.channel and source.n_channels > 1:
            result.errors.append(
                f'{source.path}: the filename says channel {source.channel} '
                f'but the file itself holds {source.n_channels} channels — '
                f'one of the two would be silently dropped. Remove the '
                f'channel token from the filename, or split the file.')
            continue

        channel_keys = _channel_keys(source)
        internal_channels = source.channel is None

        t_values: List[Tuple[int, int, str]] = []
        if t_index is not None:
            t_values.append((max(int(t_index), 1) if int(t_index) > 0 else 1,
                             0, str(t_index)))
        else:
            for index in range(source.t):
                t_values.append((index + 1, index, str(index + 1)))

        if z_handling == Z_KEEP:
            if z_index is not None:
                z_values = [(max(int(z_index), 1) if int(z_index) > 0 else 1,
                             0, str(z_index))]
            else:
                z_values = [(i + 1, i, str(i + 1)) for i in range(source.z)]
        elif z_handling == Z_FIRST:
            z_values = [(1, 0, '1')]
        else:  # Z_MAX
            z_values = [(1, -1, f'max(1..{source.z})')]

        for t_out, t_in, t_src in t_values:
            for z_out, z_in, z_src in z_values:
                for channel_offset, channel_key in enumerate(channel_keys):
                    channel = result.channel_map[(source.plate, channel_key)]
                    c_in = channel_offset if internal_channels else 0
                    result.mappings.append(Mapping(
                        source=source.path,
                        target=target_name(plate, well, field, channel,
                                           z=z_out, t=t_out),
                        plate=plate, well=well, field=field, channel=channel,
                        z=z_out, t=t_out,
                        source_plate=source.plate,
                        source_well=source.well,
                        source_field=source.field,
                        source_channel=channel_key,
                        source_z=z_src, source_t=t_src,
                        z_handling=z_handling,
                        n_z_planes=int(source.z),
                        n_timepoints=int(source.t),
                        plane=(t_in, z_in, c_in),
                        meta={'series': series,
                              'ext': source.ext,
                              'reader': source.meta.get('reader', ''),
                              'source_relpath': source.meta.get(
                                  'source_relpath',
                                  os.path.basename(source.path)),
                              'axes': source.meta.get('axes', ''),
                              'axes_assumed': source.meta.get('axes_assumed', '')}))

    # -- collisions --------------------------------------------------------
    by_target: Dict[str, List[Mapping]] = {}
    for mapping in result.mappings:
        by_target.setdefault(mapping.target, []).append(mapping)
    for target, group in sorted(by_target.items()):
        distinct = sorted({(m.source, m.plane) for m in group})
        if len(distinct) > 1:
            listed = '\n    '.join(f'{src} (plane t={p[0]} z={p[1]} c={p[2]})'
                                   for src, p in distinct)
            result.errors.append(
                f'{len(distinct)} sources would be written to the same file '
                f'{target}:\n    {listed}\n  Nothing was written. Split them '
                f'into different well folders, or use plate_naming/well_map '
                f'to separate them.')

    # -- warnings and notes ------------------------------------------------
    stacked = [s for s in readable if s.z > 1]
    if stacked:
        planes = sum(s.z for s in stacked)
        if z_handling == Z_MAX:
            result.warnings.append(
                f'{len(stacked)} source(s) hold z planes ({planes} in total). '
                f"z_handling='max' max-projects them: the individual planes "
                f'will NOT be written. Every row of the map file records '
                f"z_handling='max' and how many planes went into it.")
        elif z_handling == Z_FIRST:
            result.warnings.append(
                f'{len(stacked)} source(s) hold z planes ({planes} in total). '
                f"z_handling='first' keeps plane 1 only; the remaining "
                f'{planes - len(stacked)} plane(s) are discarded.')
        else:
            result.notes.append(
                f'{len(stacked)} source(s) hold z planes ({planes} in total); '
                f"z_handling='keep' writes each plane as its own file with "
                f'its own Z index.')

    timed = [s for s in readable if s.t > 1]
    if timed:
        result.notes.append(
            f'{len(timed)} source(s) hold multiple timepoints; each becomes '
            f'its own file with its own T index.')

    assumed = sorted({s.meta.get('axes_assumed') for s in readable
                      if s.meta.get('axes_assumed')})
    for note in assumed:
        result.warnings.append(
            f'{note}. Check the preview before converting.')

    converted_looking = [s for s in readable if s.meta.get('looks_converted')]
    if converted_looking:
        result.warnings.append(
            f'{len(converted_looking)} source file(s) are already named like '
            f'Yokogawa output. Converting them renumbers the wells and '
            f'fields — the map file records the original names.')

    result.notes.insert(0, (
        f'{len(result.plate_map)} plate(s), '
        f'{len(result.well_map)} well(s), '
        f'{len(result.channel_map)} channel id(s).'))
    return result


# ---------------------------------------------------------------------------
# Reading pixels
# ---------------------------------------------------------------------------

def _to_5d(array: np.ndarray, axes: str, n_t: int, n_z: int, n_c: int) -> np.ndarray:
    """Reshape ``array`` into ``(T, Z, C, Y, X)``.

    When the file recorded its axes (``'ZCYX'``, ``'TCZYX'``, …) they are
    used verbatim. When it did not — a plain ``tifffile.imwrite`` comes
    back as ``'QYX'`` or ``'QSYX'`` — the counts :func:`_axes_dims`
    resolved are used instead, so the reshape agrees with the plan the
    user reviewed rather than being re-guessed here.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, None, None, :, :]
    known = (axes or '').upper()
    if len(known) == array.ndim and 'Y' in known and 'X' in known and \
            not ({'Q', 'I', '?'} & set(known)):
        # Collapse any axis this module does not model (mosaic, block,
        # view) to its first element.
        for index in range(array.ndim - 1, -1, -1):
            if known[index] not in 'TZCSYX':
                array = np.take(array, 0, axis=index)
                known = known[:index] + known[index + 1:]
        if 'S' in known:
            index = known.index('S')
            if 'C' in known:
                # Both present: 'S' is interleaved samples of a channel.
                array = np.take(array, 0, axis=index)
                known = known[:index] + known[index + 1:]
            else:
                known = known[:index] + 'C' + known[index + 1:]
        for letter in 'CZT':
            if letter not in known:
                array = array[None]
                known = letter + known
        return np.transpose(array, [known.index(letter) for letter in 'TZCYX'])

    # Unknown axes: fall back to the counts the describer resolved.
    total = int(np.prod(array.shape[:-2])) if array.ndim > 2 else 1
    n_t = max(int(n_t), 1)
    n_z = max(int(n_z), 1)
    n_c = max(int(n_c), 1)
    if n_t * n_z * n_c != total:
        n_t, n_z, n_c = 1, 1, total
    return array.reshape((n_t, n_z, n_c) + array.shape[-2:])


def _read_tiff(source: SourceImage) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(source.path) as handle:
        array = handle.series[0].asarray()
        axes = str(handle.series[0].axes or '')
    return _to_5d(array, axes, source.t, source.z, source.n_channels)


def _read_plain(source: SourceImage) -> np.ndarray:
    from PIL import Image

    with Image.open(source.path) as image:
        array = np.array(image)
    if array.ndim == 3:
        array = np.moveaxis(array, -1, 0)
        return array[None, None, ...]
    return array[None, None, None, ...]


def _read_nd2(source: SourceImage) -> np.ndarray:
    module = _import_reader('.nd2')
    series = int(source.meta.get('series', 0) or 0)
    planes = []
    with module.ND2Reader(source.path) as handle:
        for t in range(source.t):
            for z in range(source.z):
                for c in range(source.n_channels):
                    planes.append(np.asarray(
                        handle.get_frame_2D(t=t, z=z, c=c, v=series)))
    stacked = np.stack(planes)
    return stacked.reshape((source.t, source.z, source.n_channels)
                           + stacked.shape[-2:])


def _read_czi(source: SourceImage) -> np.ndarray:
    module = _import_reader('.czi')
    series = int(source.meta.get('series', 0) or 0)
    with module.CziFile(source.path) as handle:
        array = np.asarray(handle.asarray())
        axes = str(handle.axes or '').upper()
    if 'S' in axes:
        index = axes.index('S')
        array = np.take(array, series, axis=index)
        axes = axes[:index] + axes[index + 1:]
    return _to_5d(array, axes, source.t, source.z, source.n_channels)


def _read_lif(source: SourceImage) -> np.ndarray:
    module = _import_reader('.lif')
    series = int(source.meta.get('series', 0) or 0)
    images = list(module.Reader(source.path).getIterImage())
    image = images[series]
    planes = []
    for t in range(source.t):
        for z in range(source.z):
            for c in range(source.n_channels):
                planes.append(np.asarray(image.getFrame(z=z, t=t, c=c)))
    stacked = np.stack(planes)
    return stacked.reshape((source.t, source.z, source.n_channels)
                           + stacked.shape[-2:])


def _read_source(source: SourceImage) -> np.ndarray:
    """Return ``source``'s pixels as a ``(T, Z, C, Y, X)`` array."""
    ext = source.ext
    if ext in ('.tif', '.tiff', '.ome.tif', '.ome.tiff'):
        return _read_tiff(source)
    if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        return _read_plain(source)
    if ext == '.nd2':
        return _read_nd2(source)
    if ext == '.czi':
        return _read_czi(source)
    if ext == '.lif':
        return _read_lif(source)
    raise ConfigurationError(f'{ext} is not a supported input format')


def _extract(array: np.ndarray, plane: Tuple[int, int, int]) -> np.ndarray:
    """Pull one output plane out of a ``(T, Z, C, Y, X)`` array.

    ``plane[1] == -1`` means "max over every z plane" — the only lossy
    path, and it is only ever reached because the caller asked for
    ``z_handling='max'``.
    """
    t_in, z_in, c_in = plane
    t_in = min(max(int(t_in), 0), array.shape[0] - 1)
    c_in = min(max(int(c_in), 0), array.shape[2] - 1)
    if z_in < 0:
        return np.max(array[t_in, :, c_in], axis=0)
    z_in = min(int(z_in), array.shape[1] - 1)
    return array[t_in, z_in, c_in]


def _imwrite(path: str, array: np.ndarray) -> None:
    """Write one TIFF. Split out so tests can make a single write fail."""
    write_tiff(path, np.asarray(array))


def _atomic_write(path: str, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` via a temp file + :func:`os.replace`.

    An interrupted run therefore leaves either nothing or a complete
    file at ``path`` — never a truncated TIFF that the next run's
    "already exists, skip" check would treat as done.
    """
    folder = os.path.dirname(path) or '.'
    handle, temp = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix='.tif', dir=folder)
    os.close(handle)
    try:
        _imwrite(temp, array)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Converting
# ---------------------------------------------------------------------------

def _conversion_field(mapping: Mapping) -> str:
    """Return a stable plate/well/field checkpoint id for one mapping."""
    return f'{mapping.plate}/{mapping.well}/f{int(mapping.field):04d}'


def _source_identity(path: str) -> Dict[str, Any]:
    """Return the cheap source identity used to guard conversion resume."""
    try:
        stat = os.stat(path)
    except OSError as exc:
        return {'path': os.path.abspath(path), 'error': str(exc)}
    return {
        'path': os.path.abspath(path),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _conversion_signature(conversion_plan: ConversionPlan, dst: str) -> str:
    """Digest source identities and every planned source-to-target mapping."""
    sources = sorted({mapping.source for mapping in conversion_plan.mappings})
    mappings = [{
        'source': os.path.abspath(mapping.source),
        'target': mapping.target,
        'plate': mapping.plate,
        'well': mapping.well,
        'field': int(mapping.field),
        'channel': int(mapping.channel),
        'z': int(mapping.z),
        't': int(mapping.t),
        'plane': mapping.plane,
        'z_handling': mapping.z_handling,
    } for mapping in conversion_plan.mappings]
    return fingerprint({
        'destination': os.path.abspath(dst),
        'z_handling': conversion_plan.z_handling,
        'sources': [_source_identity(path) for path in sources],
        'mappings': mappings,
    })


def _valid_converted_tiff(path: str) -> bool:
    """Return True when ``path`` is a readable, non-empty TIFF.

    Only TIFF metadata/pages are opened, so validating a resumed plate does
    not load its pixels into memory.
    """
    try:
        if os.path.getsize(path) < 8:
            return False
        import tifffile
        with tifffile.TiffFile(path) as handle:
            if not handle.pages:
                return False
            shape = tuple(int(value) for value in handle.series[0].shape)
            return bool(shape) and all(value > 0 for value in shape)
    except Exception:
        # This is a validity predicate used to decide whether a field may be
        # resumed. tifffile has changed the public base class of
        # ``TiffFileError`` across releases, so enumerating its exception
        # hierarchy let truncated files escape in some supported
        # environments. Any reader failure means the artifact is not valid
        # enough to trust and must be rebuilt.
        return False


def convert(conversion_plan: ConversionPlan, dst: str, overwrite: bool = False,
            map_name: str = MAP_FILENAME,
            progress: Optional[Callable[[int, int, str], None]] = None,
            ledger: Optional[RunLedger] = None,
            resume: bool = False,
            checkpoint_path: Optional[str] = None) -> ConversionResult:
    """Execute a plan: write the TIFFs, the map file and the run stamp.

    Each source is opened once and all of its planes written from that
    one read. A source that raises is recorded on the ledger and the
    batch carries on, so one corrupt file out of 400 costs one file.

    :param conversion_plan: the plan from :func:`plan`, already reviewed.
    :param dst: destination folder, created if missing. Must not be the
        source folder: converting in place is what makes
        :func:`spacr.io.convert_to_yokogawa`'s output impossible to
        re-scan, since its own outputs land next to its inputs.
    :param overwrite: when False (default) a target that already exists
        is left alone and counted in :attr:`ConversionResult.existing`.
    :param map_name: filename for the map, written inside ``dst``.
    :param progress: optional ``progress(done, total, message)``, called
        once per source.
    :param ledger: reuse an existing ledger instead of making one.
    :param resume: reuse fields recorded by a compatible checkpoint. Every
        target in a recorded field is revalidated as a readable TIFF before
        that field is skipped; missing or corrupt targets are repaired.
    :param checkpoint_path: checkpoint JSON path. Defaults to
        ``dst/.spacr_conversion.checkpoint.json``. A checkpoint is written
        after every complete field even when ``resume`` is False, so a later
        invocation can opt in after a crash.
    :returns: a :class:`ConversionResult`.
    :raises ConfigurationError: when the plan has blocking errors (a
        target-name collision), or ``dst`` is the source folder.
    """
    if not conversion_plan.ok:
        raise ConfigurationError(
            'This plan cannot be converted — fix these first:\n'
            + '\n'.join(conversion_plan.errors))

    dst = os.path.abspath(dst)
    sources = {m.source for m in conversion_plan.mappings}
    for source in sources:
        if os.path.abspath(os.path.dirname(source)) == dst:
            raise ConfigurationError(
                f'The destination {dst} is the folder the images are being '
                f'read from. Convert into a new folder so the originals stay '
                f'untouched and the output can be re-scanned.')
    os.makedirs(dst, exist_ok=True)

    run = ledger if ledger is not None else RunLedger('convert_to_yokogawa_plan')
    checkpoint_target = (
        os.path.abspath(str(checkpoint_path)) if checkpoint_path
        else os.path.join(dst, CHECKPOINT_FILENAME)
    )
    checkpoint = CheckpointStore(
        checkpoint_target,
        workflow='format_conversion',
        signature=_conversion_signature(conversion_plan, dst),
        boundary='field',
        resume=bool(resume),
    )
    result = ConversionResult(
        plan=conversion_plan, dst=dst, ledger=run,
        checkpoint_path=str(checkpoint.path))

    for source in conversion_plan.unreadable:
        run.record_failure(source.path, stage='scan', exc=source.error)
        result.skipped.append((source.path, source.error))

    by_field: Dict[str, List[Mapping]] = {}
    for mapping in conversion_plan.mappings:
        by_field.setdefault(_conversion_field(mapping), []).append(mapping)

    # A JSON claim never outranks the artifact. Re-open TIFF headers before
    # accepting a field, and re-queue it when one target is absent or corrupt.
    completed_fields = set()
    if checkpoint.resumed and not overwrite:
        for field_id in checkpoint.completed:
            mappings = by_field.get(field_id, [])
            if mappings and all(
                    _valid_converted_tiff(os.path.join(dst, item.target))
                    for item in mappings):
                completed_fields.add(field_id)
        result.resumed_fields = sorted(completed_fields)

    # One read per (file, series): a six-scene CZI is opened once, not six
    # times, and its scenes still land in six different fields.
    by_source: Dict[Tuple[str, int], List[Mapping]] = {}
    for mapping in conversion_plan.mappings:
        key = (mapping.source, int(mapping.meta.get('series', 0) or 0))
        by_source.setdefault(key, []).append(mapping)
    lookup = {(s.path, int(s.meta.get('series', 0) or 0)): s
              for s in conversion_plan.sources}
    multi_series = {path for path, _ in by_source} & {
        path for path, series in by_source if series > 0}

    ordered = sorted(by_source)
    total = len(ordered)
    for index, key in enumerate(ordered, start=1):
        cancellation_checkpoint()
        path, series = key
        group = by_source[key]
        pending_group = [
            mapping for mapping in group
            if _conversion_field(mapping) not in completed_fields
        ]
        source = lookup.get(key)
        item = f'{path} (series {series + 1})' if path in multi_series else path
        if progress is not None:
            progress(index, total, os.path.basename(path))
        if not pending_group:
            result.existing.extend(group)
            continue
        before = run.n_failed
        with run.item(item, stage='convert'):
            if source is None:
                raise ConfigurationError(
                    f'{path} is in the plan but was not scanned')
            array = _read_source(source)
            for mapping in pending_group:
                target = os.path.join(dst, mapping.target)
                if os.path.exists(target) and not overwrite:
                    if not resume or _valid_converted_tiff(target):
                        result.existing.append(mapping)
                        continue
                    print(
                        f'Checkpoint repair: {target} exists but is not a '
                        'readable TIFF; rewriting it atomically.')
                _atomic_write(target, _extract(array, mapping.plane))
                result.written.append(mapping)
        if run.n_failed > before:
            written = {m.target for m in result.written}
            existing = {m.target for m in result.existing}
            for mapping in group:
                if mapping.target not in written and mapping.target not in existing:
                    result.failed.append(mapping)
            result.skipped.append((item, run.failures[-1].message))

        # Mark only whole fields. A field spanning several source files is not
        # accepted until every planned channel/z/t target validates.
        for field_id in {_conversion_field(mapping) for mapping in group}:
            mappings = by_field[field_id]
            if all(_valid_converted_tiff(os.path.join(dst, mapping.target))
                   for mapping in mappings):
                checkpoint.mark(field_id, {
                    'targets': [mapping.target for mapping in mappings],
                    'n_targets': len(mappings),
                })
                if resume and not overwrite:
                    completed_fields.add(field_id)

    result.map_path = str(write_map(result, os.path.join(dst, map_name)))
    # Stamp the map itself: a conversion_map.csv that lists 380 of 384
    # wells looks exactly like a 380-well experiment until the sidecar
    # says otherwise.
    run.finalize(artifact=result.map_path)
    checkpoint.finish(meta={
        'map_path': result.map_path,
        'n_fields': len(checkpoint.completed),
        'n_targets': len(conversion_plan.mappings),
    })
    return result


# ---------------------------------------------------------------------------
# The map file
# ---------------------------------------------------------------------------

def write_map(result: ConversionResult, path: str) -> Path:
    """Write the map file for ``result``.

    The map is the whole point of the exercise: once the images are
    called ``plate1_A01_T0001F001L01A01Z01C01.tif``, the only thing that
    can say which microscope file that came from is this table.

    One row per output TIFF, columns :data:`MAP_COLUMNS`:

    * ``target`` / ``target_path`` — the converted file;
    * ``source`` / ``source_relpath`` — the original file;
    * ``plate`` / ``well`` / ``field`` / ``channel`` / ``z`` / ``t`` —
      the assigned ids;
    * ``source_plate`` / ``source_well`` / ``source_field`` /
      ``source_channel`` / ``source_z`` / ``source_t`` — what those ids
      were before, which is what makes the renaming reversible;
    * ``plateID`` / ``rowID`` / ``columnID`` / ``fieldID`` / ``prc`` /
      ``prcf`` — the spaCR join keys, in the exact form
      :func:`spacr.utils._map_wells` produces;
    * ``z_handling`` / ``n_z_planes`` / ``n_timepoints`` — how the third
      and fourth dimensions were treated;
    * ``status`` — ``converted``, ``existing`` or ``failed``;
    * ``meta_json`` — reader, extension, series and axes.

    :param result: a finished :class:`ConversionResult`.
    :param path: destination CSV.
    :returns: the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(result.rows(), columns=list(MAP_COLUMNS))
    frame.to_csv(target, index=False)
    return target


def read_map(path: str) -> pd.DataFrame:
    """Read a map file back.

    :param path: a CSV written by :func:`write_map`.
    :returns: the map as a DataFrame.
    :raises ConfigurationError: when the file is missing or is not a
        spaCR conversion map — a wrong path here would otherwise
        populate a database with somebody else's columns.
    """
    target = Path(path)
    if not target.is_file():
        raise ConfigurationError(f'Map file does not exist: {path}')
    try:
        frame = pd.read_csv(target)
    except Exception as exc:
        raise ConfigurationError(
            f'{path} could not be read as a conversion map: {exc}') from exc
    missing = [c for c in _REQUIRED_MAP_COLUMNS if c not in frame.columns]
    if missing:
        raise ConfigurationError(
            f'{path} is not a spaCR conversion map — missing column(s): '
            f'{", ".join(missing)}')
    return frame


def populate_db_from_map(db_path: str, map_path: str,
                         table: str = CONVERSION_TABLE) -> int:
    """Load a map file into ``measurements.db`` so the run can be joined back.

    This is the read-back that closes the loop. After spaCR has measured
    the converted images, every row of every measurement table carries
    ``plateID`` / ``rowID`` / ``columnID`` / ``fieldID`` — but nothing
    that says the field came from ``run1/wt/fov07_C2.tif``. This writes
    the map into the same database as ``conversion_map``, keyed exactly
    the way the measurement tables are, so::

        SELECT c.*, m.source, m.source_well, m.source_field
        FROM cell AS c
        JOIN conversion_map AS m
          ON  c.plateID  = m.plateID
          AND c.rowID    = m.rowID
          AND c.columnID = m.columnID
          AND c.fieldID  = m.fieldID

    joins the original metadata onto the measurements. ``prc`` (well
    level, ``plate_r1_c1``) and ``prcf`` (field level,
    ``plate_r1_c1_f1``) are there for the same join in one column.

    The table is replaced, not appended: re-running a conversion and
    re-populating must not leave two generations of rows behind.

    :param db_path: SQLite database to write into; created if missing.
    :param map_path: the CSV from :func:`write_map`.
    :param table: table name, ``conversion_map`` by default.
    :returns: number of rows written.
    :raises ConfigurationError: when the map is missing or malformed.
    """
    frame = read_map(map_path)
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        frame.to_sql(table, connection, if_exists='replace', index=False)
        if 'prcf' in frame.columns:
            connection.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{table}_prcf '
                f'ON {table} (prcf)')
        if 'target' in frame.columns:
            connection.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{table}_target '
                f'ON {table} (target)')
        connection.commit()
    finally:
        connection.close()
    return int(len(frame))


# ---------------------------------------------------------------------------
# The settings-dict entry point
# ---------------------------------------------------------------------------

def default_settings(settings: Optional[TMapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the settings :func:`convert_folder` understands, with defaults.

    Shaped like every other ``spacr.settings`` factory — pass a partial
    dict, get it back filled in — so the CLI and the GUIs can build a
    panel from it without special-casing this module.

    :param settings: partial settings; keys given here win.
    """
    resolved: Dict[str, Any] = {
        'src': None,
        'dst': None,
        'layout': 'auto',
        'z_handling': Z_KEEP,
        'plate_naming': 'index',
        'overwrite': False,
        'map_name': MAP_FILENAME,
        'db_path': None,
        'preview_only': False,
        'preview_rows': 20,
        'resume': False,
        'checkpoint_path': None,
    }
    resolved.update(dict(settings or {}))
    return resolved


def convert_folder(settings: Optional[TMapping[str, Any]] = None,
                   **overrides: Any) -> ConversionResult:
    """Scan, preview, convert and map one folder in a single call.

    The entry point the CLI and the Qt bridge want: one function, one
    settings dict, the same three phases underneath. It always prints
    the plan before writing anything, so even a headless
    ``spacr-run format_convert`` leaves the source → target table in the
    log where a surprised user can find it.

    Settings (see :func:`default_settings`):

    ``src``
        folder to convert. Required.
    ``dst``
        destination; defaults to ``<src>_yokogawa``. Never ``src``.
    ``layout`` / ``z_handling`` / ``plate_naming``
        passed to :func:`scan` and :func:`plan`.
    ``overwrite``
        False by default — existing targets are left alone.
    ``db_path``
        when set, :func:`populate_db_from_map` loads the map into it.
    ``preview_only``
        print the plan and stop. The returned result has written
        nothing and has no ``map_path``.
    ``resume``
        accept complete fields from a compatible checkpoint after validating
        every output TIFF.
    ``checkpoint_path``
        optional checkpoint JSON path; defaults inside ``dst``.

    :returns: the :class:`ConversionResult`; for ``preview_only`` an
        empty one carrying the plan.
    :raises ConfigurationError: no ``src``, or a plan with blocking
        errors — a name collision must stop the run, not be printed and
        walked past.
    """
    resolved = default_settings(settings)
    resolved.update(overrides)

    src = resolved.get('src')
    if not src:
        raise ConfigurationError(
            "convert_folder needs a 'src' folder of images to convert.")
    src = os.path.abspath(str(src))
    dst = resolved.get('dst') or (os.path.normpath(src) + '_yokogawa')
    dst = os.path.abspath(str(dst))

    sources = scan(src, layout=str(resolved.get('layout') or 'auto'))
    conversion_plan = plan(sources,
                           z_handling=str(resolved.get('z_handling') or Z_KEEP),
                           plate_naming=str(resolved.get('plate_naming') or 'index'))

    print(conversion_plan.summary())
    rows = int(resolved.get('preview_rows') or 0)
    if rows > 0 and len(conversion_plan.mappings):
        frame = conversion_plan.to_frame()
        print(frame.head(rows).to_string(index=False))
        if len(frame) > rows:
            print(f'… and {len(frame) - rows} more row(s).')

    if not conversion_plan.ok:
        raise ConfigurationError(
            'Conversion refused — nothing was written:\n'
            + '\n'.join(conversion_plan.errors))

    if resolved.get('preview_only'):
        print(f'preview_only is set — nothing was written. Target folder '
              f'would be {dst}.')
        return ConversionResult(plan=conversion_plan, dst=dst)

    result = convert(conversion_plan, dst,
                     overwrite=bool(resolved.get('overwrite')),
                     map_name=str(resolved.get('map_name') or MAP_FILENAME),
                     resume=bool(resolved.get('resume', False)),
                     checkpoint_path=resolved.get('checkpoint_path'))
    print(result.summary())

    db_path = resolved.get('db_path')
    if db_path:
        written = populate_db_from_map(str(db_path), result.map_path)
        print(f'Wrote {written} conversion_map row(s) into {db_path}.')
    return result
