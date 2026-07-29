"""The database contract: what a spaCR key is called, and what it means.

Why this module exists
----------------------
Every table spaCR writes is keyed on the same four strings — ``plateID``,
``rowID``, ``columnID``, ``fieldID`` — plus ``timeID`` for a timelapse, and
they are joined on the composed forms ``prc``, ``prcf`` and ``prcfo``. Those
keys were derived by hand in at least five places, and **the copies disagreed
about malformed wells**. One measured example, both halves written by the real
writers into one database (see ``tests/test_schema.py``)::

    field 'plate1_AA01_1'          # AA is a real 1536-plate row

    utils._merge_and_save_to_database -> cell     : ('error','error','error','error')
    utils.filepaths_to_database       -> png_list : ('plate1','r1','c0','f1')

    png_list.merge(cell, on=[plateID,rowID,columnID,fieldID]) -> 0 rows

Two rows describing the same objects, given two different identities by two
functions in the same file, and the join between them silently returns
nothing. That class of bug is not fixable one call site at a time — it is
fixable only by there being one definition. This module is that definition.

It is deliberately **declarative and additive**. Nothing here rewrites the
existing call sites; :func:`legacy_map_wells` and :func:`legacy_well_ids`
reproduce today's behaviour bit for bit so a migration can be done one call
site at a time with a test pinning exactly what changed.

The ``f0`` problem
------------------
``utils._safe_int_convert`` returns ``0`` when a token will not parse. That is
the single most destructive line in the metadata path, because it is *silent*
and it *collides*: three ImageXpress sites ``s1``/``s2``/``s3`` all become
``f0`` and therefore one ``prcf``, and object 1 of each becomes the same
``prcfo``. Three fields go in, one comes out, and nothing anywhere says so.

This module never invents a number it did not read. :func:`parse_int_token`
returns ``None``, not ``0``. Above it, key construction is graded:

* **Parseable** — ``'3'``, ``'003'``, ``'s3'``, ``'T0003'``, ``'F003'`` all
  mean field 3. A vendor prefix is a spelling, not a different field.
* **Present but not a number** — ``'xy'`` becomes ``'fxy'``. Not a number, so
  it cannot be mistaken for one and shows up immediately in QC; still
  distinct per token, so three bad fields stay three fields; still a valid
  join key, so the run continues and every table agrees on it.
* **Absent** — ``''`` or ``None`` raises :class:`KeyParseError`. An empty
  field id is not an identity, it is the absence of one, and every row so
  keyed would merge with every other.

``strict=True`` promotes the middle tier to an exception, which is what a
preflight or QC pass wants. The default is the long-run path: a ten-hour
``measure_crop`` must not die on field 8000 because one file was named
badly, but it must not lie about it either. Both requirements are met by
making the failure *representable in the data* instead of choosing between
silence and death.

Dependencies
------------
The standard library. Nothing else — not even pandas at module scope, let
alone torch, cellpose or Qt, and no ``spacr`` import. Everything wants this
module, including GUI paths and CLI preflights that must not pay for a torch
import, so ``tests/test_schema.py`` asserts it imports clean in a subprocess
(the same guard ``spacr.crops`` carries).

``spacr.resume`` is the reason the last dependency went too. It runs at the
top of ``measure_crop``, in a process that may never load a model, and
``tests/test_resume.py`` asserts that importing it pulls in **no numpy and no
pandas** — so a module-scope ``import pandas`` here would have made the one
call site this module was written for the one call site that could not use
it. The two frame helpers at the bottom import pandas themselves; everything
above them is strings and integers and needs nothing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple

__all__ = [
    # errors
    'SchemaError', 'WellParseError', 'KeyParseError',
    'ObjectTableSchemaError',
    # key names
    'PLATE_KEY', 'ROW_KEY', 'COLUMN_KEY', 'FIELD_KEY', 'TIME_KEY',
    'CHANNEL_KEY', 'SLICE_KEY', 'OBJECT_LABEL_KEY',
    'FIELD_KEY_COLUMNS', 'TIMEPOINT_KEY_COLUMNS', 'WELL_KEY_COLUMNS',
    'PRC_KEY', 'PRCF_KEY', 'PRCFO_KEY',
    'KEY_PREFIXES', 'OBJECT_PREFIX', 'KEY_SEPARATOR',
    'LEGACY_COLUMN_NAMES', 'TIME_COLUMN_ALIASES', 'canonical_column_name',
    # scalars
    'parse_int_token', 'row_index_from_letters', 'letters_from_row_index',
    'row_id', 'column_id', 'field_id', 'time_id', 'object_id',
    'row_index', 'column_index', 'field_index', 'time_index', 'object_index',
    'strip_prefix',
    # wells
    'parse_well', 'well_id', 'is_positional_well', 'is_positional_pair',
    # identities
    'FieldID', 'ObjectID',
    'compose_prc', 'compose_prcf', 'compose_prcfo',
    'parse_prcf', 'parse_prcfo',
    'parse_field_stem', 'parse_object_stem',
    # plate formats
    'PLATE_FORMATS', 'plate_format_for', 'is_within_plate_format',
    # tables
    'PARENT_OBJECT_TABLES', 'CHILD_OBJECT_TABLES', 'OBJECT_TABLES',
    'ORGANELLE_SUMMARY_TABLES', 'CROP_TABLES', 'MEASUREMENT_TABLES',
    'BOOKKEEPING_TABLES', 'OWNED_TABLES', 'table_key_columns',
    'CANONICAL_OBJECT_TABLES', 'OBJECT_TABLE_REQUIRED_COLUMNS',
    'OBJECT_TABLE_OPTIONAL_COLUMNS', 'ObjectTableSchema',
    'OBJECT_TABLE_SCHEMAS', 'object_table_schema',
    # pandas
    'add_identity_columns', 'canonicalise_columns',
    'validate_object_table_frame',
    # legacy
    'legacy_well_ids', 'legacy_map_wells', 'legacy_safe_int_convert',
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Base for every failure to build or read a spaCR key.

    A subclass of :class:`ValueError` so that call sites which already guard
    a parse with ``except ValueError`` keep working.
    """


class WellParseError(SchemaError):
    """A well identifier could not be turned into a row and a column."""


class KeyParseError(SchemaError):
    """A key token was absent, or was rejected under ``strict=True``."""


class ObjectTableSchemaError(SchemaError):
    """An object-table frame violates its declared column or row contract."""


# ---------------------------------------------------------------------------
# The canonical key names
# ---------------------------------------------------------------------------

#: The plate. Free-form text — it is the plate folder's name — but it may not
#: contain :data:`KEY_SEPARATOR`, because ``prcf`` is separator-joined.
PLATE_KEY = 'plateID'
#: The plate row, 1-based, rendered ``'r<N>'``. Row ``A`` is ``'r1'``.
ROW_KEY = 'rowID'
#: The plate column, 1-based, rendered ``'c<N>'``. Column ``01`` is ``'c1'``.
COLUMN_KEY = 'columnID'
#: The imaging site within the well, rendered ``'f<N>'``.
FIELD_KEY = 'fieldID'
#: The timepoint, rendered ``'t<N>'``. Absent outside a timelapse.
TIME_KEY = 'timeID'
#: The acquisition channel, rendered ``'<N>'`` by the filename regexes.
CHANNEL_KEY = 'chanID'
#: The z slice.
SLICE_KEY = 'sliceID'
#: The integer label of an object inside its field's mask.
OBJECT_LABEL_KEY = 'object_label'

#: ``plate_row_column`` — the well. Identifies a well across every field.
PRC_KEY = 'prc'
#: ``plate_row_column_field`` (``..._time`` in a timelapse) — the field.
PRCF_KEY = 'prcf'
#: ``prcf_o<label>`` — one object. The row key of every merged measurement.
PRCFO_KEY = 'prcfo'

#: The four columns that identify a field. Every measurement table carries
#: all four; :mod:`spacr.resume` keys its resume-delete on exactly these.
FIELD_KEY_COLUMNS: Tuple[str, ...] = (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY)

#: :data:`FIELD_KEY_COLUMNS` plus the timepoint — the identity of one frame.
TIMEPOINT_KEY_COLUMNS: Tuple[str, ...] = FIELD_KEY_COLUMNS + (TIME_KEY,)

#: The three columns that identify a well, ignoring which field it came from.
WELL_KEY_COLUMNS: Tuple[str, ...] = (PLATE_KEY, ROW_KEY, COLUMN_KEY)

#: The single-letter prefix each numeric key is rendered with.
KEY_PREFIXES: Dict[str, str] = {
    ROW_KEY: 'r',
    COLUMN_KEY: 'c',
    FIELD_KEY: 'f',
    TIME_KEY: 't',
}

#: Objects are prefixed too, but ``object_label`` is stored bare and only
#: gains its ``o`` when composed into ``prcfo``. See :func:`compose_prcfo`.
OBJECT_PREFIX = 'o'

#: What ``prc`` / ``prcf`` / ``prcfo`` are joined on. A plate name containing
#: this character cannot be round-tripped; :func:`compose_prc` refuses it.
KEY_SEPARATOR = '_'


#: Every legacy spelling spaCR has written, and the canonical name it means.
#:
#: This is the superset of ``utils.DB_COLUMN_RENAMES`` (applied to databases
#: by ``utils.rename_columns_in_db`` on first read) and
#: ``utils.correct_metadata_column_names`` (applied to CSVs). They were two
#: lists that had drifted apart, so a database carrying ``row_name`` was only
#: half repaired. Going forward both should read this one.
LEGACY_COLUMN_NAMES: Dict[str, str] = {
    'row':          ROW_KEY,
    'row_name':     ROW_KEY,
    'rowid':        ROW_KEY,
    'row_id':       ROW_KEY,
    'column':       COLUMN_KEY,
    'col':          COLUMN_KEY,
    'column_name':  COLUMN_KEY,
    'column_id':    COLUMN_KEY,
    'col_name':     COLUMN_KEY,
    'plate':        PLATE_KEY,
    'plate_name':   PLATE_KEY,
    'plate_id':     PLATE_KEY,
    'field':        FIELD_KEY,
    'field_name':   FIELD_KEY,
    'field_id':     FIELD_KEY,
    'time':         TIME_KEY,
    'time_id':      TIME_KEY,
    'timepoint':    TIME_KEY,
    'channel':      CHANNEL_KEY,
    'channel_name': CHANNEL_KEY,
    'chan_id':      CHANNEL_KEY,
    'slice_id':     SLICE_KEY,
}

#: Both spellings of the time column that exist in databases on disk.
#: ``png_list`` was written with ``time_id`` while every object table got
#: ``timeID``; readers must accept either until the migration has run.
TIME_COLUMN_ALIASES: Tuple[str, ...] = (TIME_KEY, 'time_id')


def canonical_column_name(name: Any) -> str:
    """Return the canonical spelling of a metadata column name.

    Case-insensitive, so ``'RowID'``, ``'rowid'`` and ``'row_name'`` all
    resolve to ``'rowID'``. A name with no known alias is returned unchanged.

    :param name: column name as it appears in a table or CSV.
    :returns: the canonical name, or ``name`` unchanged.

    Example:
        .. code-block:: python

            >>> canonical_column_name('column_name')
            'columnID'
            >>> canonical_column_name('cell_area')
            'cell_area'
    """
    text = str(name)
    lowered = text.lower()
    for canonical in (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY, TIME_KEY,
                      CHANNEL_KEY, SLICE_KEY):
        if lowered == canonical.lower():
            return canonical
    return LEGACY_COLUMN_NAMES.get(lowered, text)


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

#: An integer wearing a vendor prefix: ``s1``, ``T0001``, ``F003``, ``Z01``.
#: One or two leading ASCII letters, then digits, then nothing.
_PREFIXED_INT = re.compile(r'^([A-Za-z]{1,2})(\d+)$')

#: A bare integer, optionally signed and zero padded.
_BARE_INT = re.compile(r'^[+-]?\d+$')

#: A well: the row letters then the column digits. Separators are tolerated
#: because plate readers emit ``'A-01'`` and ``'A 1'``.
#:
#: One or two letters, matching ``plate_qc._WELL_RE``. The largest standard
#: plate has 32 rows (``AF``) and two letters reach ``ZZ`` = 702, so a third
#: buys nothing real and would start swallowing tokens that are not wells at
#: all. The two modules must agree on the *shape* of a well, not only on the
#: letter arithmetic, or "is this a well?" gets two answers again.
_WELL = re.compile(r'^([A-Za-z]{1,2})[\s_\-]*(\d{1,4})$')

#: Row letters with no column at all, e.g. a folder named ``'A'``.
_ROW_ONLY = re.compile(r'^([A-Za-z]{1,2})$')


def parse_int_token(token: Any, *, allow_prefix: bool = True) -> Optional[int]:
    """Return the integer ``token`` denotes, or ``None`` — **never** ``0``.

    This is the replacement for ``utils._safe_int_convert``, and the whole
    point of it is the return type. ``_safe_int_convert`` answers "what
    number is this?" with ``0`` when the honest answer is "there isn't one",
    and ``0`` is a perfectly good field id, so the lie is unrecoverable
    downstream. ``None`` is not a field id, so every caller is forced to
    decide what to do — and the callers here do decide, see :func:`field_id`.

    Vendor prefixes are understood, because they are a spelling of a number
    rather than a different number: an ImageXpress site ``s3``, a CellVoyager
    field ``F003`` and a bare ``3`` are the same field, and a pipeline that
    gave them three different ids would be just as wrong as one that gave
    them all ``f0``.

    :param token: anything — a string, an int, a float, ``None``.
    :param allow_prefix: strip one or two leading ASCII letters when what
        follows is all digits. Default ``True``.
    :returns: the integer, or ``None`` when the token holds no integer.

    Example:
        .. code-block:: python

            >>> parse_int_token('003'), parse_int_token('s3')
            (3, 3)
            >>> parse_int_token('T0001'), parse_int_token('x')
            (1, None)
            >>> parse_int_token('') is None, parse_int_token(None) is None
            (True, True)
    """
    if token is None:
        return None
    if isinstance(token, bool):
        # bool is an int subclass; a True field id is a caller bug, not a 1.
        return None
    if isinstance(token, int):
        return int(token)
    if isinstance(token, float):
        # NaN and inf hold no integer. int(nan) raises, int(2.7) truncates
        # silently, so neither is acceptable without a check.
        if token != token or token in (float('inf'), float('-inf')):
            return None
        if float(token).is_integer():
            return int(token)
        return None
    if isinstance(token, (bytes, bytearray)):
        try:
            text = bytes(token).decode('ascii')
        except UnicodeDecodeError:
            return None
    else:
        text = str(token)

    text = text.strip()
    if not text:
        return None
    if _BARE_INT.match(text):
        return int(text)
    if allow_prefix:
        match = _PREFIXED_INT.match(text)
        if match:
            return int(match.group(2))
    return None


def row_index_from_letters(letters: Any) -> Optional[int]:
    """``'A'`` → 1, ``'Z'`` → 26, ``'AA'`` → 27, ``'AF'`` → 32.

    Bijective base 26. Multi-letter rows are not an edge case: a 1536-well
    plate has 32 rows and runs ``A``…``Z``, ``AA``…``AF``. Both
    ``utils._map_wells`` (which raises, becoming ``'error'``) and
    ``utils._map_wells_png`` (which yields ``'c0'``) get these wrong, in two
    different ways. This matches ``plate_qc._alpha_to_index`` exactly, so the
    QC module and the database agree.

    :param letters: one or more ASCII letters, any case.
    :returns: the 1-based row index, or ``None`` when ``letters`` is not
        purely alphabetic or is empty.
    """
    if not isinstance(letters, str):
        # str(None) is 'None', which is four perfectly good row letters and
        # would come back as row 256573. Anything that is not already text
        # is not a row label.
        return None
    text = letters.strip().upper()
    if not text:
        return None
    total = 0
    for char in text:
        if not ('A' <= char <= 'Z'):
            return None
        total = total * 26 + (ord(char) - 64)
    return total or None


def letters_from_row_index(index: int) -> str:
    """Inverse of :func:`row_index_from_letters`. ``27`` → ``'AA'``.

    :param index: 1-based row index.
    :returns: the row letters.
    :raises KeyParseError: when ``index`` is not a positive integer.
    """
    value = parse_int_token(index, allow_prefix=False)
    if value is None or value < 1:
        raise KeyParseError(
            f'row index {index!r} is not a positive integer; there is no '
            f'row letter for it.')
    out = ''
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        out = chr(65 + remainder) + out
    return out


def _sanitise_token(token: Any) -> str:
    """Make an unparseable token safe to embed in a separator-joined key.

    Whitespace is stripped and the separator is replaced, because a token
    containing ``'_'`` would silently add a component to ``prcf`` and make
    the key unsplittable.
    """
    text = str(token).strip()
    return text.replace(KEY_SEPARATOR, '-')


def _prefixed_id(kind: str, token: Any, *, strict: bool) -> str:
    """Build one prefixed key. The graded-failure policy lives here."""
    prefix = KEY_PREFIXES[kind]
    value = parse_int_token(token)
    if value is not None:
        return f'{prefix}{value}'

    text = '' if token is None else str(token).strip()
    if not text:
        # Tier 3: nothing to key on. This is the one case that must raise —
        # an empty id is not an identity, and every row carrying it would
        # merge with every other row that also failed to parse.
        raise KeyParseError(
            f'cannot build a {kind} from {token!r}: it is empty. An empty '
            f'{kind} is not an identity, and every row keyed on it would '
            f'silently merge with every other unparseable row.')
    if strict:
        raise KeyParseError(
            f'cannot build a {kind} from {token!r}: it holds no integer. '
            f'Accepted spellings are a bare integer ("3", "003") or an '
            f'integer behind a one- or two-letter vendor prefix ("s3", '
            f'"F003", "T0003").')
    # Tier 2: keep the token. Distinct per input, visibly not a number, and
    # still a usable join key — the run continues without inventing a 0.
    return f'{prefix}{_sanitise_token(text)}'


def row_id(row: Any, *, strict: bool = False) -> str:
    """Return the canonical ``'r<N>'`` row id.

    Accepts an index (``1``, ``'1'``), an already-prefixed id (``'r1'``) —
    which round-trips rather than becoming ``'rr1'`` — or row letters
    (``'A'``, ``'AA'``).

    :param row: row index, ``'r<N>'``, or row letters.
    :param strict: raise instead of preserving an unparseable token.
    :returns: ``'r<N>'``.
    :raises KeyParseError: on an empty token, or any bad token when ``strict``.
    """
    if isinstance(row, str):
        letters = row.strip()
        if _ROW_ONLY.match(letters) and not _PREFIXED_INT.match(letters):
            index = row_index_from_letters(letters)
            if index is not None:
                return f'r{index}'
    return _prefixed_id(ROW_KEY, row, strict=strict)


def column_id(column: Any, *, strict: bool = False) -> str:
    """Return the canonical ``'c<N>'`` column id.

    :param column: column index, or an already-prefixed ``'c<N>'``.
    :param strict: raise instead of preserving an unparseable token.
    :returns: ``'c<N>'``.
    """
    return _prefixed_id(COLUMN_KEY, column, strict=strict)


def field_id(field: Any, *, strict: bool = False) -> str:
    """Return the canonical ``'f<N>'`` field id.

    ``'3'``, ``'003'``, ``'s3'``, ``'F003'`` and ``3`` all give ``'f3'``.
    A token holding no integer is preserved (``'xy'`` → ``'fxy'``) rather
    than becoming ``'f0'``; see the module docstring for why.

    :param field: field token.
    :param strict: raise instead of preserving an unparseable token.
    :returns: ``'f<N>'``, or ``'f<token>'`` for an unparseable token.
    """
    return _prefixed_id(FIELD_KEY, field, strict=strict)


def time_id(time: Any, *, strict: bool = False) -> str:
    """Return the canonical ``'t<N>'`` timepoint id.

    ``'T0003'`` → ``'t3'``. Under the old ``_safe_int_convert`` every
    ``T####`` token became ``t0``, which collapsed a whole timelapse onto
    one frame.

    :param time: timepoint token.
    :param strict: raise instead of preserving an unparseable token.
    :returns: ``'t<N>'``.
    """
    return _prefixed_id(TIME_KEY, time, strict=strict)


def object_id(label: Any, *, strict: bool = False) -> str:
    """Return the canonical ``'o<N>'`` object id used in ``prcfo``.

    :param label: object label, bare or already ``'o<N>'``.
    :param strict: raise instead of preserving an unparseable token.
    :returns: ``'o<N>'``.
    :raises KeyParseError: on an empty token, or any bad token when ``strict``.
    """
    value = parse_int_token(label)
    if value is not None:
        return f'{OBJECT_PREFIX}{value}'
    text = '' if label is None else str(label).strip()
    if not text:
        raise KeyParseError(
            f'cannot build an object id from {label!r}: it is empty.')
    if strict:
        raise KeyParseError(
            f'cannot build an object id from {label!r}: it holds no integer.')
    return f'{OBJECT_PREFIX}{_sanitise_token(text)}'


def strip_prefix(value: Any, prefix: str) -> str:
    """Remove one leading ``prefix`` from ``value`` if it is there.

    :param value: the id, e.g. ``'r12'``.
    :param prefix: the single-letter prefix, e.g. ``'r'``.
    :returns: the remainder, e.g. ``'12'``.
    """
    text = str(value).strip()
    if prefix and text[:len(prefix)].lower() == prefix.lower():
        return text[len(prefix):]
    return text


def _index_of(value: Any, prefix: str) -> Optional[int]:
    if value is None:
        return None
    return parse_int_token(strip_prefix(value, prefix), allow_prefix=False)


def row_index(value: Any) -> Optional[int]:
    """``'r3'`` → ``3``; ``'C'`` → ``3``; an unparseable id → ``None``."""
    if isinstance(value, str):
        text = value.strip()
        if _ROW_ONLY.match(text) and not _PREFIXED_INT.match(text):
            return row_index_from_letters(text)
    return _index_of(value, KEY_PREFIXES[ROW_KEY])


def column_index(value: Any) -> Optional[int]:
    """``'c12'`` → ``12``; an unparseable id → ``None``."""
    return _index_of(value, KEY_PREFIXES[COLUMN_KEY])


def field_index(value: Any) -> Optional[int]:
    """``'f2'`` → ``2``; ``'fxy'`` → ``None``."""
    return _index_of(value, KEY_PREFIXES[FIELD_KEY])


def time_index(value: Any) -> Optional[int]:
    """``'t7'`` → ``7``; an unparseable id → ``None``."""
    return _index_of(value, KEY_PREFIXES[TIME_KEY])


def object_index(value: Any) -> Optional[int]:
    """``'o41'`` → ``41``; an unparseable id → ``None``."""
    return _index_of(value, OBJECT_PREFIX)


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

def is_positional_well(well: Any) -> bool:
    """True when ``well`` is a bare number rather than ``<letters><digits>``.

    Some acquisitions name wells ``'12'``. There is no way to know whether
    that means row 1 column 2 or the twelfth well, so :func:`parse_well`
    passes it through into both slots unchanged — which is what all five
    existing implementations do, and there is data on disk keyed that way.
    This predicate lets a caller detect the case instead of discovering it
    from a ``rowID`` that does not start with ``r``.

    :param well: well identifier.
    :returns: True when the well holds no row letters.
    """
    text = str(well).strip()
    return bool(text) and _WELL.match(text) is None and _ROW_ONLY.match(text) is None


def is_positional_pair(row: Any, column: Any) -> bool:
    """True when ``(rowID, columnID)`` came from the positional passthrough.

    :func:`parse_well` puts an unrecognisable well into *both* slots
    verbatim, so an unprefixed pair of equal values is that passthrough and
    not a real row and column. Without this check ``('12', '12')`` looks
    like row 12 / column 12 and :func:`well_id` happily renders it ``'L12'``
    — a well name for a well that was never identified.

    Only *strings* can be a passthrough. A bare ``int`` is unambiguously an
    index — ``well_id(1, 1)`` is a caller asking for well A01, not a well
    that failed to parse — so an integer pair is never flagged, however
    equal. (This is not hypothetical: it is the bug the round-trip test in
    ``tests/test_schema.py`` caught in the first version of this function,
    where ``well_id(1, 1)`` raised.)

    :param row: the ``rowID`` as stored.
    :param column: the ``columnID`` as stored.
    :returns: whether the pair is a passthrough rather than a position.
    """
    if not isinstance(row, str) or not isinstance(column, str):
        return False
    row_text, column_text = row.strip(), column.strip()
    if not row_text or row_text != column_text:
        return False
    return (row_text[:1].lower() != KEY_PREFIXES[ROW_KEY]
            and column_text[:1].lower() != KEY_PREFIXES[COLUMN_KEY])


def parse_well(well: Any, *, strict: bool = False) -> Tuple[str, str]:
    """Return ``(rowID, columnID)`` for a well identifier.

    ``'A01'``, ``'a1'``, ``'A-01'`` and ``' A01 '`` all give
    ``('r1', 'c1')``. ``'AA01'`` — a real 1536-plate well — gives
    ``('r27', 'c1')``, where ``utils._map_wells`` raises into ``'error'``
    and ``utils._map_wells_png`` returns ``('r1', 'c0')``.

    A well with letters but no digits (``'A'``) has no column. Under
    ``_map_wells_png`` it became ``'c0'``, i.e. indistinguishable from a
    genuine column 0; here it raises, because a well with no column is not
    a well.

    A bare number is passed through into both slots — see
    :func:`is_positional_well`.

    :param well: well identifier of any of the above shapes.
    :param strict: also reject the bare-number passthrough.
    :returns: ``(rowID, columnID)``.
    :raises WellParseError: when the well is empty, has no column, or is a
        bare number and ``strict`` is set.

    Example:
        .. code-block:: python

            >>> parse_well('A01'), parse_well('aa1')
            (('r1', 'c1'), ('r27', 'c1'))
    """
    text = '' if well is None else str(well).strip()
    if not text:
        raise WellParseError(
            'cannot parse a well from an empty value: it identifies no row '
            'and no column, and every row keyed on it would merge.')

    match = _WELL.match(text)
    if match:
        # _WELL's first group is [A-Za-z]{1,3}, so row_index_from_letters
        # cannot fail here; no defensive branch, because an unreachable one
        # could never be tested and would only ever be wrong.
        return (f'r{row_index_from_letters(match.group(1))}',
                f'c{int(match.group(2))}')

    if _ROW_ONLY.match(text):
        raise WellParseError(
            f'well {well!r} has row letters but no column. '
            f'_map_wells_png turned this into column "c0", which is '
            f'indistinguishable from a real column 0.')

    if strict:
        raise WellParseError(
            f'well {well!r} is not <letters><digits>. Under strict parsing a '
            f'bare well number is refused, because whether "12" means row 1 '
            f'column 2 or the twelfth well is not knowable.')
    # Legacy passthrough: every existing implementation does this, and
    # databases on disk carry rowID == columnID == the raw well.
    return text, text


def well_id(row: Any, column: Any) -> str:
    """Return the canonical well name: ``('r3', 'c7')`` → ``'C07'``.

    The inverse of :func:`parse_well` for wells that have one. Matches
    ``plate_qc.well_id``.

    :param row: row index or ``'r<N>'`` or row letters.
    :param column: column index or ``'c<N>'``.
    :returns: the well name, zero padded to two digits.
    :raises KeyParseError: when either index is unusable, or the pair is a
        positional passthrough (see :func:`is_positional_pair`).
    """
    if is_positional_pair(row, column):
        raise KeyParseError(
            f'({row!r}, {column!r}) is a positional-well passthrough, not a '
            f'row and a column; there is no well name for it.')
    r_index = row_index(row)
    c_index = column_index(column)
    if r_index is None or r_index < 1:
        raise KeyParseError(f'cannot build a well name from row {row!r}.')
    if c_index is None or c_index < 1:
        raise KeyParseError(f'cannot build a well name from column {column!r}.')
    return f'{letters_from_row_index(r_index)}{c_index:02d}'


# ---------------------------------------------------------------------------
# Plate formats
# ---------------------------------------------------------------------------

#: ``n_wells -> (n_rows, n_columns)`` for the standard SBS plate formats.
PLATE_FORMATS: Dict[int, Tuple[int, int]] = {
    6:    (2, 3),
    12:   (3, 4),
    24:   (4, 6),
    48:   (6, 8),
    96:   (8, 12),
    384:  (16, 24),
    1536: (32, 48),
}


def plate_format_for(row: Any, column: Any) -> Optional[int]:
    """Return the smallest standard plate format containing ``(row, column)``.

    A column past 24 is not an error — a 1536-well plate has 48 of them —
    so nothing in this module rejects one. This is how a caller that *does*
    care checks.

    :param row: row index or ``'r<N>'``.
    :param column: column index or ``'c<N>'``.
    :returns: the well count of the smallest format that contains the
        position, or ``None`` when it fits no standard plate.
    """
    r_index = row_index(row)
    c_index = column_index(column)
    if r_index is None or c_index is None or r_index < 1 or c_index < 1:
        return None
    for n_wells in sorted(PLATE_FORMATS):
        n_rows, n_columns = PLATE_FORMATS[n_wells]
        if r_index <= n_rows and c_index <= n_columns:
            return n_wells
    return None


def is_within_plate_format(row: Any, column: Any, n_wells: int) -> bool:
    """True when ``(row, column)`` lies inside an ``n_wells`` plate.

    :param row: row index or ``'r<N>'``.
    :param column: column index or ``'c<N>'``.
    :param n_wells: a key of :data:`PLATE_FORMATS`.
    :returns: whether the position exists on that plate.
    :raises KeyParseError: when ``n_wells`` is not a standard format.
    """
    if n_wells not in PLATE_FORMATS:
        raise KeyParseError(
            f'{n_wells!r} is not a standard plate format; known formats are '
            f'{sorted(PLATE_FORMATS)}.')
    n_rows, n_columns = PLATE_FORMATS[n_wells]
    r_index = row_index(row)
    c_index = column_index(column)
    if r_index is None or c_index is None:
        return False
    return 1 <= r_index <= n_rows and 1 <= c_index <= n_columns


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def _check_plate(plate: Any) -> str:
    text = '' if plate is None else str(plate).strip()
    if not text:
        raise KeyParseError(
            'cannot build a key from an empty plate id.')
    if KEY_SEPARATOR in text:
        raise KeyParseError(
            f'plate id {plate!r} contains {KEY_SEPARATOR!r}, which is the '
            f'key separator. "{text}_r1_c1_f1" could not be split back into '
            f'its parts, so the plate must not contain one.')
    return text


def compose_prc(plate: Any, row: Any, column: Any) -> str:
    """Return the ``prc`` well key: ``'plate1_r1_c1'``.

    :param plate: plate id.
    :param row: row index or ``'r<N>'``.
    :param column: column index or ``'c<N>'``.
    :returns: the composed key.
    """
    return KEY_SEPARATOR.join(
        [_check_plate(plate), row_id(row), column_id(column)])


def compose_prcf(plate: Any, row: Any, column: Any, field: Any,
                 time: Any = None) -> str:
    """Return the ``prcf`` field key.

    ``'plate1_r1_c1_f2'``, or ``'plate1_r1_c1_f2_t3'`` when ``time`` is
    given. The timepoint goes **after** the field — that is the order
    ``_map_wells(timelapse=True)`` writes and every table on disk carries.

    :param plate: plate id.
    :param row: row index or ``'r<N>'``.
    :param column: column index or ``'c<N>'``.
    :param field: field token.
    :param time: timepoint token, or ``None`` outside a timelapse.
    :returns: the composed key.
    """
    parts = [_check_plate(plate), row_id(row), column_id(column),
             field_id(field)]
    if time is not None and str(time).strip() != '':
        parts.append(time_id(time))
    return KEY_SEPARATOR.join(parts)


def compose_prcfo(plate: Any, row: Any, column: Any, field: Any,
                  obj: Any, time: Any = None) -> str:
    """Return the ``prcfo`` object key: ``'plate1_r1_c1_f2_o7'``.

    With a timepoint the object still goes last:
    ``'plate1_r1_c1_f2_t3_o7'``. That matches both
    ``utils._map_wells_png(timelapse=True)`` and the ``prcf + '_' + 'o' +
    object_label`` composition in ``io._read_and_join_tables``.

    :param plate: plate id.
    :param row: row index or ``'r<N>'``.
    :param column: column index or ``'c<N>'``.
    :param field: field token.
    :param obj: object label, bare or ``'o<N>'``.
    :param time: timepoint token, or ``None``.
    :returns: the composed key.
    """
    return KEY_SEPARATOR.join(
        [compose_prcf(plate, row, column, field, time), object_id(obj)])


@dataclass(frozen=True)
class FieldID:
    """One imaging field's identity — the key of every measurement row.

    :ivar plateID: plate id.
    :ivar rowID: ``'r<N>'``.
    :ivar columnID: ``'c<N>'``.
    :ivar fieldID: ``'f<N>'``.
    :ivar timeID: ``'t<N>'``, or ``None`` outside a timelapse.
    """

    plateID: str
    rowID: str
    columnID: str
    fieldID: str
    timeID: Optional[str] = None

    @property
    def prc(self) -> str:
        """The ``prc`` well key."""
        return KEY_SEPARATOR.join([self.plateID, self.rowID, self.columnID])

    @property
    def prcf(self) -> str:
        """The ``prcf`` field key, with the timepoint when there is one."""
        parts = [self.plateID, self.rowID, self.columnID, self.fieldID]
        if self.timeID:
            parts.append(self.timeID)
        return KEY_SEPARATOR.join(parts)

    @property
    def positional(self) -> bool:
        """True when this identity came from the positional passthrough."""
        return is_positional_pair(self.rowID, self.columnID)

    @property
    def well(self) -> Optional[str]:
        """The well name (``'A01'``), or ``None`` for a positional well."""
        try:
            return well_id(self.rowID, self.columnID)
        except (KeyParseError, WellParseError):
            return None

    def with_object(self, obj: Any) -> 'ObjectID':
        """Return the :class:`ObjectID` for one object in this field."""
        return ObjectID(plateID=self.plateID, rowID=self.rowID,
                        columnID=self.columnID, fieldID=self.fieldID,
                        timeID=self.timeID, objectID=object_id(obj))

    def to_dict(self, *, include_prcf: bool = False) -> Dict[str, str]:
        """Return the identity as the dict the tables carry.

        :param include_prcf: also emit ``prc`` and ``prcf``.
        :returns: ``{plateID, rowID, columnID, fieldID[, timeID][, prc, prcf]}``.
        """
        out = {PLATE_KEY: self.plateID, ROW_KEY: self.rowID,
               COLUMN_KEY: self.columnID, FIELD_KEY: self.fieldID}
        if self.timeID is not None:
            out[TIME_KEY] = self.timeID
        if include_prcf:
            out[PRC_KEY] = self.prc
            out[PRCF_KEY] = self.prcf
        return out

    @classmethod
    def build(cls, plate: Any, well: Any = None, field: Any = None,
              time: Any = None, *, row: Any = None, column: Any = None,
              strict: bool = False) -> 'FieldID':
        """Construct from a well string, or from a row and column.

        :param plate: plate id.
        :param well: well identifier, e.g. ``'A01'``. Mutually exclusive
            with ``row``/``column``.
        :param field: field token.
        :param time: timepoint token, or ``None``.
        :param row: row index or ``'r<N>'``, when there is no well string.
        :param column: column index or ``'c<N>'``.
        :param strict: reject unparseable field/time tokens and odd wells.
        :returns: the :class:`FieldID`.
        :raises KeyParseError: when neither a well nor a row/column pair is
            given.
        """
        if well is not None:
            row_key, column_key = parse_well(well, strict=strict)
        elif row is not None and column is not None:
            row_key, column_key = row_id(row, strict=strict), \
                column_id(column, strict=strict)
        else:
            raise KeyParseError(
                'FieldID.build needs either a well or both a row and a '
                'column.')
        return cls(plateID=_check_plate(plate), rowID=row_key,
                   columnID=column_key,
                   fieldID=field_id(field, strict=strict),
                   timeID=None if time is None or str(time).strip() == ''
                   else time_id(time, strict=strict))


@dataclass(frozen=True)
class ObjectID:
    """One segmented object's identity — the ``prcfo`` a merged row is keyed on.

    :ivar objectID: ``'o<N>'``.
    """

    plateID: str
    rowID: str
    columnID: str
    fieldID: str
    objectID: str
    timeID: Optional[str] = None

    @property
    def field(self) -> FieldID:
        """The field this object sits in."""
        return FieldID(plateID=self.plateID, rowID=self.rowID,
                       columnID=self.columnID, fieldID=self.fieldID,
                       timeID=self.timeID)

    @property
    def prcf(self) -> str:
        """The ``prcf`` of the containing field."""
        return self.field.prcf

    @property
    def prcfo(self) -> str:
        """The ``prcfo`` object key."""
        return KEY_SEPARATOR.join([self.prcf, self.objectID])

    def to_dict(self, *, include_prcf: bool = False) -> Dict[str, str]:
        """Return the identity as a dict, including ``prcfo``."""
        out = self.field.to_dict(include_prcf=include_prcf)
        out[PRCFO_KEY] = self.prcfo
        return out


def parse_prcf(text: Any) -> FieldID:
    """Parse a ``prcf`` string back into a :class:`FieldID`.

    Parsed **right to left**, which is what makes it correct: the components
    are optional in the middle (``timeID`` may or may not be there), and
    ``ml.py`` splits ``prcfo`` left to right into a fixed five columns, so a
    timelapse key with six parts silently misaligns every column.

    :param text: e.g. ``'plate1_r1_c1_f2'`` or ``'plate1_r1_c1_f2_t3'``.
    :returns: the :class:`FieldID`.
    :raises KeyParseError: when the string is not a ``prcf``.
    """
    parts = str(text).strip().split(KEY_SEPARATOR)
    if len(parts) < 4:
        raise KeyParseError(
            f'{text!r} is not a prcf: expected at least '
            f'plate_row_column_field, got {len(parts)} part(s).')
    time_key = None
    if parts[-1][:1].lower() == 't' and time_index(parts[-1]) is not None:
        time_key = parts.pop()
    field_key = parts.pop()
    column_key = parts.pop()
    row_key = parts.pop()
    if not parts:
        raise KeyParseError(f'{text!r} is not a prcf: it has no plate.')
    if field_key[:1].lower() != 'f':
        raise KeyParseError(
            f'{text!r} is not a prcf: {field_key!r} is not a field id.')
    return FieldID(plateID=KEY_SEPARATOR.join(parts), rowID=row_key,
                   columnID=column_key, fieldID=field_key, timeID=time_key)


def parse_prcfo(text: Any) -> ObjectID:
    """Parse a ``prcfo`` string back into an :class:`ObjectID`.

    :param text: e.g. ``'plate1_r1_c1_f2_o7'`` or ``'plate1_r1_c1_f2_t3_o7'``.
    :returns: the :class:`ObjectID`.
    :raises KeyParseError: when the string is not a ``prcfo``.
    """
    parts = str(text).strip().split(KEY_SEPARATOR)
    if len(parts) < 5:
        raise KeyParseError(
            f'{text!r} is not a prcfo: expected at least '
            f'plate_row_column_field_object, got {len(parts)} part(s).')
    object_key = parts.pop()
    if object_key[:1].lower() != OBJECT_PREFIX:
        raise KeyParseError(
            f'{text!r} is not a prcfo: {object_key!r} is not an object id.')
    field = parse_prcf(KEY_SEPARATOR.join(parts))
    return field.with_object(object_key)


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------

def parse_field_stem(name: Any, *, timelapse: bool = False,
                     strict: bool = False) -> FieldID:
    """Parse a merged-stack file name into a :class:`FieldID`.

    The canonical replacement for ``utils._map_wells``. The name is
    ``<plate>_<well>_<field>`` (``_<time>`` when ``timelapse``), with or
    without a directory and an extension.

    Differences from ``_map_wells``, every one of them a case ``_map_wells``
    gets wrong rather than a change of contract:

    * ``'AA01'`` gives ``r27``; ``_map_wells`` raises and returns the
      five-tuple ``('error',) * 5`` — losing the *plate* as well as the well.
    * a lowercase well parses; ``_map_wells`` raises on it.
    * a whitespace-padded well parses; ``_map_wells`` treats it as a
      positional well and puts the padded text in both slots.
    * a non-numeric field is preserved (``'s3'`` → ``f3``, ``'xy'`` →
      ``'fxy'``); ``_map_wells`` returns ``f0`` for both.
    * too few parts raises instead of returning ``'error'`` strings that
      then get written into the database as if they were an identity.

    :param name: file name, path, or stem.
    :param timelapse: expect and parse a trailing timepoint.
    :param strict: reject unparseable field/time tokens and odd wells.
    :returns: the :class:`FieldID`.
    :raises KeyParseError: when the name has too few components.
    :raises WellParseError: when the well cannot be parsed.

    Example:
        .. code-block:: python

            >>> parse_field_stem('plate1_A01_3').prcf
            'plate1_r1_c1_f3'
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    parts = stem.split(KEY_SEPARATOR)
    needed = 4 if timelapse else 3
    if len(parts) < needed:
        raise KeyParseError(
            f'cannot identify a field from {stem!r}: expected at least '
            f'{"plate_well_field_time" if timelapse else "plate_well_field"} '
            f'({needed} parts), got {len(parts)}. _map_wells returned the '
            f'string "error" in every slot here, and those strings were then '
            f'written into the database as an identity.')
    return FieldID.build(parts[0], well=parts[1], field=parts[2],
                         time=parts[3] if timelapse else None, strict=strict)


def parse_object_stem(name: Any, *, timelapse: bool = False,
                      strict: bool = False) -> ObjectID:
    """Parse a crop-PNG file name into an :class:`ObjectID`.

    The canonical replacement for ``utils._map_wells_png``. The name is
    ``<plate>_<well>_<field>[_<time>]_<object>``, with the object label
    always last.

    Differences from ``_map_wells_png``:

    * ``'AA01'`` gives ``('r27', 'c1')``; ``_map_wells_png`` gives
      ``('r1', 'c0')`` — silently dropping the second row letter *and*
      inventing column 0.
    * a well with no column (``'A'``) raises; ``_map_wells_png`` gives
      ``'c0'``.
    * a lowercase well parses; ``_map_wells_png`` raises into
      ``('error',) * 6``.
    * a non-numeric field or object is preserved rather than collapsed to
      ``f0`` / ``o0``.
    * ``_map_wells_png`` reads the object from ``parts[-1]`` and the field
      from ``parts[2]``, which are the same token in a three-part name —
      ``'plate1_A01_5.png'`` becomes field 5 *and* object 5. Here a name
      that short raises.

    :param name: file name, path, or stem.
    :param timelapse: expect a timepoint between field and object.
    :param strict: reject unparseable tokens and odd wells.
    :returns: the :class:`ObjectID`.
    :raises KeyParseError: when the name has too few components.
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    parts = stem.split(KEY_SEPARATOR)
    needed = 5 if timelapse else 4
    if len(parts) < needed:
        raise KeyParseError(
            f'cannot identify an object from {stem!r}: expected at least '
            f'{"plate_well_field_time_object" if timelapse else "plate_well_field_object"} '
            f'({needed} parts), got {len(parts)}.')
    field = FieldID.build(parts[0], well=parts[1], field=parts[2],
                          time=parts[3] if timelapse else None, strict=strict)
    return field.with_object(parts[-1])


# ---------------------------------------------------------------------------
# The tables spaCR owns
# ---------------------------------------------------------------------------

#: Object tables whose rows are top-level objects with no parent link.
PARENT_OBJECT_TABLES: Tuple[str, ...] = ('cell', 'cytoplasm')

#: Object tables whose rows carry a ``cell_id`` link to their parent cell.
CHILD_OBJECT_TABLES: Tuple[str, ...] = ('nucleus', 'pathogen', 'organelle')

#: Every per-object measurement table.
OBJECT_TABLES: Tuple[str, ...] = PARENT_OBJECT_TABLES + CHILD_OBJECT_TABLES

#: Per-parent rollups of the organelles inside them. One row per parent.
ORGANELLE_SUMMARY_TABLES: Tuple[str, ...] = (
    'cell_organelle_summary',
    'nucleus_organelle_summary',
    'pathogen_organelle_summary',
    'cytoplasm_organelle_summary',
)

#: Tables recording crop PNGs on disk. Keyed on ``prcfo``, not on a label.
CROP_TABLES: Tuple[str, ...] = ('png_list',)

#: Everything written into ``measurements/measurements.db`` that carries a
#: field identity.
MEASUREMENT_TABLES: Tuple[str, ...] = (
    OBJECT_TABLES + ORGANELLE_SUMMARY_TABLES + CROP_TABLES)

#: Tables spaCR owns that are *not* keyed on a field: the settings snapshot
#: and the per-file object tallies, both keyed on ``file_name``.
BOOKKEEPING_TABLES: Tuple[str, ...] = ('settings', 'object_counts')

#: Every table spaCR creates in a measurements database. A table not in here
#: was put there by someone else and must be left alone by any migration.
OWNED_TABLES: Tuple[str, ...] = MEASUREMENT_TABLES + BOOKKEEPING_TABLES


#: The four analysis compartments covered by the canonical object-table
#: contract. ``organelle`` remains an owned child table but is intentionally
#: outside this first contract: its per-object table is optional and its
#: stable public output is the per-parent summary family above.
CANONICAL_OBJECT_TABLES: Tuple[str, ...] = (
    'cell', 'cytoplasm', 'nucleus', 'pathogen')

#: Columns every canonical object table carries, in writer order. Time is
#: conditional and parent links are table-specific, so both live in the
#: optional/conditional part of :class:`ObjectTableSchema`.
OBJECT_TABLE_REQUIRED_COLUMNS: Tuple[str, ...] = (
    OBJECT_LABEL_KEY,
    PLATE_KEY,
    ROW_KEY,
    COLUMN_KEY,
    FIELD_KEY,
    PRCF_KEY,
    'file_name',
    'path_name',
)

#: Stable non-feature columns that may be absent in legacy or non-timelapse
#: tables. Measurement-stamp columns are added by the ``optional_columns``
#: property from :mod:`spacr.measurement_schema`, keeping their one canonical
#: definition and preserving this module's dependency-free import boundary.
OBJECT_TABLE_OPTIONAL_COLUMNS: Tuple[str, ...] = (
    TIME_KEY,
    'cell_id',
    'label_list_morphology',
    'label_list_intensity',
)


@dataclass(frozen=True)
class ObjectTableSchema:
    """Declarative contract for one object measurement table.

    Feature columns are open-ended because channel counts and enabled
    measurements vary per run, but they are not untyped: a feature written by
    a table starts with ``<object_type>_`` and is numeric. Unknown annotation
    or provenance columns remain permitted so older databases and user-added
    labels are not destroyed by validation.

    :param table: SQLite table name.
    :param object_type: required feature-column prefix.
    :param parent_column: optional link to a parent cell.
    """

    table: str
    object_type: str
    parent_column: Optional[str] = None

    @property
    def required_columns(self) -> Tuple[str, ...]:
        """Columns every row set of this table must expose."""
        return OBJECT_TABLE_REQUIRED_COLUMNS

    @property
    def identifier_columns(self) -> Tuple[str, ...]:
        """Prefixed link/label columns emitted by morphology measurement."""
        # measure._morphological_measurements prefixes the child mapping before
        # _check_integrity runs. These look like ordinary feature names but
        # are identifiers: feature_dict._LINK_COLUMNS documents the same
        # distinction. They may use object dtype after DataFrame.explode()
        # even though every non-null value denotes an integer label.
        return (
            f'{self.object_type}_{self.object_type}',
            f'{self.object_type}_cell_id',
        )

    @property
    def optional_columns(self) -> Tuple[str, ...]:
        """Stable optional metadata, including the shared provenance stamp."""
        from .measurement_schema import MEASUREMENT_STAMP_COLUMNS

        columns = list(OBJECT_TABLE_OPTIONAL_COLUMNS)
        if self.parent_column is None:
            columns.remove('cell_id')
        return (
            tuple(columns)
            + self.identifier_columns
            + tuple(MEASUREMENT_STAMP_COLUMNS)
        )

    def row_key_columns(self, *, timelapse: bool = False) -> Tuple[str, ...]:
        """Return the columns that must be unique within one write batch."""
        base = TIMEPOINT_KEY_COLUMNS if timelapse else FIELD_KEY_COLUMNS
        return base + (OBJECT_LABEL_KEY,)

    def feature_column(self, name: Any) -> bool:
        """Return whether ``name`` belongs to this table's feature namespace."""
        return str(name).startswith(f'{self.object_type}_')

    def validate(self, frame, *, timelapse: Optional[bool] = None):
        """Validate and return a canonical-column copy of ``frame``."""
        return validate_object_table_frame(
            frame, self.table, timelapse=timelapse)


OBJECT_TABLE_SCHEMAS = MappingProxyType({
    'cell': ObjectTableSchema('cell', 'cell'),
    'cytoplasm': ObjectTableSchema('cytoplasm', 'cytoplasm'),
    'nucleus': ObjectTableSchema('nucleus', 'nucleus', 'cell_id'),
    'pathogen': ObjectTableSchema('pathogen', 'pathogen', 'cell_id'),
})


def object_table_schema(table: str) -> ObjectTableSchema:
    """Return the canonical schema for ``table``.

    :raises ObjectTableSchemaError: when no canonical contract exists.
    """
    try:
        return OBJECT_TABLE_SCHEMAS[str(table)]
    except KeyError as exc:
        raise ObjectTableSchemaError(
            f'{table!r} has no canonical object-table schema; expected one of '
            f'{list(CANONICAL_OBJECT_TABLES)}.') from exc


def table_key_columns(table: str, *, timelapse: bool = False) -> Tuple[str, ...]:
    """Return the columns that identify a row of ``table``.

    :param table: table name.
    :param timelapse: include ``timeID``.
    :returns: the key columns, most significant first.
    :raises KeyParseError: when ``table`` is not one spaCR owns.

    Example:
        .. code-block:: python

            >>> table_key_columns('cell')
            ('plateID', 'rowID', 'columnID', 'fieldID', 'object_label')
    """
    if table not in OWNED_TABLES:
        raise KeyParseError(
            f'{table!r} is not a table spaCR owns; known tables are '
            f'{sorted(OWNED_TABLES)}.')
    if table in BOOKKEEPING_TABLES:
        return ('file_name',) if table == 'object_counts' else ()
    base = TIMEPOINT_KEY_COLUMNS if timelapse else FIELD_KEY_COLUMNS
    if table in CROP_TABLES:
        return base + (PRCFO_KEY,)
    return base + (OBJECT_LABEL_KEY,)


# ---------------------------------------------------------------------------
# pandas
# ---------------------------------------------------------------------------

def canonicalise_columns(df):
    """Return ``df`` with every legacy metadata column renamed canonically.

    A rename is **skipped when the canonical name is already present**, which
    is the same rule ``utils.rename_columns_in_db`` follows: a frame carrying
    both spellings keeps both untouched rather than having one silently
    overwrite the other. Dropping data to tidy a name is never the right
    trade — a human can decide which column is authoritative, and until then
    both stay reachable.

    :param df: :class:`pandas.DataFrame` whose columns may use legacy names.
    :returns: a new frame with canonical column names.
    """
    have = set(df.columns)
    mapping = {}
    for name in df.columns:
        canonical = canonical_column_name(name)
        if canonical != name and canonical not in have:
            mapping[name] = canonical
            have.add(canonical)
    return df.rename(columns=mapping) if mapping else df.copy()


def validate_object_table_frame(
        frame, table: str, *, timelapse: Optional[bool] = None):
    """Validate an object-table frame against its canonical contract.

    Validation is deliberately strict at the writer boundary and
    compatibility-preserving in shape:

    * required identity/provenance columns must exist and be non-null;
    * labels (and present parent links) must be positive integers;
    * ``prcf`` must exactly match the component key columns;
    * one write batch may contain at most one row per object key;
    * measurement stamps are all present or all absent;
    * features from another compartment are rejected, and this table's own
      feature namespace must be numeric.

    Extra columns are allowed because annotation columns are user-defined and
    historical databases contain extensions. Legacy metadata spellings are
    canonicalised on the returned copy before validation.

    pandas is imported only when this function is called. Importing
    :mod:`spacr.schema` itself remains standard-library-only for CLI,
    multiprocessing, and resume preflight paths.

    :param frame: pandas DataFrame to validate.
    :param table: one of :data:`CANONICAL_OBJECT_TABLES`.
    :param timelapse: require/forbid ``timeID``; ``None`` infers it.
    :returns: canonical-column DataFrame copy.
    :raises ObjectTableSchemaError: on any contract violation.
    """
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    if not isinstance(frame, pd.DataFrame):
        raise ObjectTableSchemaError(
            f'{table} must be validated from a pandas DataFrame, got '
            f'{type(frame).__name__}.')

    contract = object_table_schema(table)
    out = canonicalise_columns(frame)

    duplicated_columns = out.columns[out.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ObjectTableSchemaError(
            f'{table} has duplicated column names: {duplicated_columns}.')

    missing = [
        column for column in contract.required_columns
        if column not in out.columns
    ]
    if missing:
        raise ObjectTableSchemaError(
            f'{table} is missing required canonical column(s) {missing}; '
            f'got {list(out.columns)}.')

    has_time = TIME_KEY in out.columns
    if timelapse is True and not has_time:
        raise ObjectTableSchemaError(
            f'{table} is a timelapse table but has no {TIME_KEY!r} column.')
    if timelapse is False and has_time:
        raise ObjectTableSchemaError(
            f'{table} is non-timelapse but unexpectedly carries '
            f'{TIME_KEY!r}.')
    is_timelapse = has_time if timelapse is None else timelapse

    from .measurement_schema import MEASUREMENT_STAMP_COLUMNS

    stamp_columns = [
        column for column in MEASUREMENT_STAMP_COLUMNS
        if column in out.columns
    ]
    if stamp_columns and len(stamp_columns) != len(MEASUREMENT_STAMP_COLUMNS):
        absent = [
            column for column in MEASUREMENT_STAMP_COLUMNS
            if column not in out.columns
        ]
        raise ObjectTableSchemaError(
            f'{table} has a partial measurement provenance stamp: present '
            f'{stamp_columns}, missing {absent}. Write all stamp columns or '
            f'none for a legacy 2-D table.')

    text_columns = list(FIELD_KEY_COLUMNS) + [
        PRCF_KEY, 'file_name', 'path_name']
    if is_timelapse:
        text_columns.append(TIME_KEY)
    for column in text_columns:
        invalid = out[column].isna() | out[column].map(
            lambda value: not isinstance(value, str) or not value.strip())
        if invalid.any():
            examples = out.index[invalid].tolist()[:3]
            raise ObjectTableSchemaError(
                f'{table}.{column} must contain non-empty strings; invalid '
                f'row indexes: {examples}.')

    def _validate_positive_integer(column: str, *, nullable: bool = False):
        values = out[column]
        check = values.dropna() if nullable else values
        if not nullable and values.isna().any():
            raise ObjectTableSchemaError(
                f'{table}.{column} must contain positive integer labels and '
                f'may not contain NULL.')
        numeric = pd.to_numeric(check, errors='coerce')
        invalid = (
            numeric.isna()
            | numeric.mod(1).ne(0)
            | numeric.le(0)
        )
        if invalid.any():
            examples = check.loc[invalid].head(3).tolist()
            raise ObjectTableSchemaError(
                f'{table}.{column} must contain positive integer labels'
                f'{" or NULL" if nullable else ""}; invalid values: '
                f'{examples}.')

    _validate_positive_integer(OBJECT_LABEL_KEY)
    if contract.parent_column and contract.parent_column in out.columns:
        _validate_positive_integer(contract.parent_column, nullable=True)
    for column in contract.identifier_columns:
        if column in out.columns:
            _validate_positive_integer(column, nullable=True)
    if stamp_columns:
        _validate_positive_integer('measurement_ndim')
        _validate_positive_integer('n_z')
        invalid_ndim = ~out['measurement_ndim'].isin((2, 3))
        if invalid_ndim.any():
            raise ObjectTableSchemaError(
                f"{table}.measurement_ndim must be 2 or 3; invalid values: "
                f"{out.loc[invalid_ndim, 'measurement_ndim'].head(3).tolist()}.")
        valid_units = {'px', 'px_xy', 'um'}
        invalid_units = ~out['measurement_units'].isin(valid_units)
        if invalid_units.any():
            raise ObjectTableSchemaError(
                f"{table}.measurement_units must be one of "
                f"{sorted(valid_units)}; invalid values: "
                f"{out.loc[invalid_units, 'measurement_units'].head(3).tolist()}.")
        for column in ('voxel_size_z_um', 'voxel_size_xy_um'):
            present = out[column].dropna()
            numeric = pd.to_numeric(present, errors='coerce')
            invalid = numeric.isna() | numeric.le(0)
            if invalid.any():
                raise ObjectTableSchemaError(
                    f'{table}.{column} must contain positive numeric values '
                    f'or NULL; invalid values: '
                    f'{present.loc[invalid].head(3).tolist()}.')

        flat = out['measurement_ndim'].eq(2)
        invalid_flat = flat & (
            out['n_z'].ne(1) | out['measurement_units'].ne('px'))
        if invalid_flat.any():
            examples = out.loc[
                invalid_flat,
                ['measurement_ndim', 'measurement_units', 'n_z'],
            ].head(3).to_dict(orient='records')
            raise ObjectTableSchemaError(
                f'{table} 2-D rows must use measurement_units="px" and '
                f'n_z=1; invalid rows: {examples}.')

    expected_prcf = (
        out[PLATE_KEY]
        + KEY_SEPARATOR + out[ROW_KEY]
        + KEY_SEPARATOR + out[COLUMN_KEY]
        + KEY_SEPARATOR + out[FIELD_KEY]
    )
    if is_timelapse:
        expected_prcf = expected_prcf + KEY_SEPARATOR + out[TIME_KEY]
    mismatch = out[PRCF_KEY].ne(expected_prcf)
    if mismatch.any():
        examples = [
            {
                'index': index,
                'stored': out.at[index, PRCF_KEY],
                'expected': expected_prcf.at[index],
            }
            for index in out.index[mismatch][:3]
        ]
        raise ObjectTableSchemaError(
            f'{table}.{PRCF_KEY} disagrees with its component identity '
            f'columns; examples: {examples}.')

    row_keys = contract.row_key_columns(timelapse=is_timelapse)
    duplicated = out.duplicated(subset=list(row_keys), keep=False)
    if duplicated.any():
        examples = (
            out.loc[duplicated, list(row_keys)]
            .drop_duplicates()
            .head(3)
            .to_dict(orient='records')
        )
        raise ObjectTableSchemaError(
            f'{table} violates its one-row-per-object key {list(row_keys)}; '
            f'duplicated keys: {examples}.')

    stable_columns = (
        set(contract.required_columns)
        | set(contract.optional_columns)
    )
    foreign_prefixes = {
        name: f'{schema.object_type}_'
        for name, schema in OBJECT_TABLE_SCHEMAS.items()
        if name != table
    }
    for column in out.columns:
        if column in stable_columns:
            continue
        foreign = [
            name for name, prefix in foreign_prefixes.items()
            if str(column).startswith(prefix)
        ]
        if foreign:
            raise ObjectTableSchemaError(
                f'{table} contains {foreign[0]} feature {column!r}; features '
                f'must remain in their owning object table.')
        if contract.feature_column(column):
            series = out[column]
            if series.notna().any() and not is_numeric_dtype(series):
                raise ObjectTableSchemaError(
                    f'{table} feature {column!r} must be numeric, got '
                    f'{series.dtype}.')

    return out


def add_identity_columns(df, source: str = 'file_name', *,
                         timelapse: bool = False, objects: bool = False,
                         strict: bool = False,
                         include_prcf: bool = True):
    """Parse a name column into the canonical key columns.

    The vectorised form of :func:`parse_field_stem` /
    :func:`parse_object_stem`, for the writers that today do
    ``df[[...]] = df[col].apply(lambda x: pd.Series(_map_wells(x)))`` — a
    line that positionally unpacks a tuple whose length changes with
    ``timelapse``, so a mismatched flag misaligns every column.

    :param df: :class:`pandas.DataFrame` with a column of file names.
    :param source: name of that column. Default ``'file_name'``.
    :param timelapse: names carry a timepoint.
    :param objects: names are crop PNGs, so also emit ``prcfo``.
    :param strict: reject unparseable tokens.
    :param include_prcf: also emit ``prc`` and ``prcf``.
    :returns: a new frame with the key columns added.
    :raises KeyParseError: when ``source`` is not a column of ``df``.
    """
    import pandas as pd

    if source not in df.columns:
        raise KeyParseError(
            f'{source!r} is not a column of the frame; got '
            f'{list(df.columns)}.')
    parse = parse_object_stem if objects else parse_field_stem
    records = [parse(name, timelapse=timelapse, strict=strict)
               .to_dict(include_prcf=include_prcf)
               for name in df[source]]
    keys = pd.DataFrame(records, index=df.index)
    out = df.copy()
    for column in keys.columns:
        out[column] = keys[column]
    return out


# ---------------------------------------------------------------------------
# Bug-compatible copies of what is on disk today
# ---------------------------------------------------------------------------
#
# These exist so a migration can be done one call site at a time with a test
# pinning exactly what changed, and so that a reader of an old database can
# reproduce the key it was written with. They are not for new code.

def legacy_safe_int_convert(value: Any, default: Any = 0) -> Any:
    """``utils._safe_int_convert`` exactly, including the ``0`` default.

    Kept only for the migration tests. New code calls
    :func:`parse_int_token`, which returns ``None``.

    :param value: token to convert.
    :param default: what to return on ``ValueError``. Note that ``TypeError``
        — which is what ``None`` raises — is *not* caught, here or in the
        original.
    :returns: the int, or ``default``.
    """
    try:
        return int(value)
    except ValueError:
        return default


def legacy_well_ids(well: Any) -> Tuple[str, str]:
    """``utils._map_wells``' well handling exactly, raising where it raises.

    :param well: well identifier.
    :returns: ``(rowID, columnID)``.
    :raises ValueError: on the wells ``_map_wells`` swallows into ``'error'``.
    :raises IndexError: on an empty well, as ``_map_wells`` does.
    """
    import string
    text = str(well)
    if text[0].isalpha():
        return ('r' + str(string.ascii_uppercase.index(text[0]) + 1),
                'c' + str(int(text[1:])))
    return text, text


def legacy_map_wells(file_name: Any, timelapse: bool = False) -> Tuple[str, ...]:
    """``utils._map_wells`` reproduced bit for bit, ``'error'`` tuple and all.

    Used by ``tests/test_schema.py`` to assert that the canonical parser
    agrees with the legacy one on **every well that works today**, so the
    migration is provably a strict repair rather than a change of contract.

    :param file_name: stack file name.
    :param timelapse: parse a trailing timepoint.
    :returns: the same tuple ``_map_wells`` returns.
    """
    timeid = None
    try:
        parts = str(file_name).split('_')
        plate = parts[0]
        well = parts[1]
        field = 'f' + str(legacy_safe_int_convert(parts[2]))
        if timelapse:
            timeid = 't' + str(legacy_safe_int_convert(parts[3]))
        row, column = legacy_well_ids(well)
        if timelapse:
            prcf = '_'.join([plate, row, column, field, timeid])
        else:
            prcf = '_'.join([plate, row, column, field])
    except Exception:
        plate = row = column = field = timeid = prcf = 'error'
    if timelapse:
        return plate, row, column, field, timeid, prcf
    return plate, row, column, field, prcf
