"""Resume / checkpointing for spaCR's two long batch stages.

A plate is a thousand fields and several hours. When a run dies at field
900 — a full disk, an OOM-killed worker, a laptop lid — spaCR currently
throws away all 900 and starts again. This module lets the next
invocation pick up where the last one stopped.

The skipping is the easy half. The half that decides whether resuming is
*safe* is everything else in here:

**A file that exists is not a file that finished.**
``spacr.io._load_and_concatenate_arrays`` writes ``merged/<field>.npy``
with a bare ``np.save`` straight onto the destination path, so a crash
mid-write leaves a short, structurally valid-looking ``.npy`` on disk. A
resume that skips on ``os.path.exists`` would accept that truncated
field, measure whatever bytes happened to land, and put the result in the
same table as the good fields. So :func:`completed_fields_in_merged`
*validates*: it parses the ``.npy`` header and checks that the file is
actually as long as its own shape claims. Anything short is reported as
rejected and re-queued, never skipped. (io.py now also writes merged
arrays atomically, so files produced from here on cannot be truncated —
but the ones already on disk from previous runs can be.)

**Measure appends; therefore a resume must delete first.**
``_merge_and_save_to_database`` and ``filepaths_to_database`` both use
``to_sql(..., if_exists='append')``. Re-measuring a field that already
wrote some rows does not overwrite them, it *adds* to them, and every
per-well aggregate downstream is then computed over inflated counts with
nothing anywhere to indicate it. :func:`clear_field_rows` is the
delete-before-insert that makes re-running a field idempotent, and it is
the reason this module exists at all. It runs in one transaction across
every table the field touched, so a failure part-way leaves the database
exactly as it was.

**measurements.db is not measure's private file.** ``convert`` writes
``conversion_map`` into it, ``align`` writes ``align_coordinates``,
``foreign`` writes ``foreign_*``, ``timelapse`` writes its track table —
all keyed on the same four columns as the measurements, deliberately, so
that they join. Discovering "tables to clear" structurally therefore
found all of them, and a resume deleted every pending field's row from
each: the only record of which vendor file became ``plate1_A01_1``, of
where each tile was stitched, of somebody else's imported measurements.
None of it can be recomputed from the database. The tables a resume may
delete from are now an explicit allow-list,
:data:`MEASURE_OWNED_TABLES`, checked both when the list is discovered
and again per table immediately before the DELETE is prepared.

A name allow-list answers "could measure have written this table?", and
there is one case where the honest answer to "did it?" is still no:
``foreign.run_import`` copies the imported rows into the canonical
``cell`` / ``nucleus`` / ``pathogen`` table when nothing of anyone
else's is there, so that a purely-imported project is readable by every
spaCR tool. Those rows sit under spaCR's own column names in a table on
the allow-list, and nothing *in the table* tells them apart from
measure's. A resume therefore clears them along with the pending field.
That is a known bug, recorded — and pinned — by the ``xfail(strict=True)``
tests in ``tests/test_resume_owned_tables.py``.

**A guard against it was attempted here and backed out; do not put it
back in this shape.** It read ``foreign_import.canonical_table_written``
and refused any delete from a table that record claimed. Reproduced
failures, every one of them measured on a database built by the real
importer:

1. the claim is *table*-scoped. Once an import has filled ``cell``,
   a delete of a field the import never covered — rows ``measure_crop``
   itself wrote — is refused too, and the project can never be resumed;
2. its own first remedy destroys the rows. "Re-run the import with
   ``measure=True``" replaces the ``foreign_import`` record with
   ``canonical_table_written = 0`` while leaving every imported row in
   ``cell``, so the next resume deletes them with no warning at all —
   the guard fails open, in the destructive direction, by being obeyed;
3. it is unreachable on the flow it exists for. In a purely-imported
   project ``cell`` is the only field table and holds every field, so
   :func:`completed_fields_in_db` reports the whole plate done and no
   delete is ever planned;
4. its second remedy is a dead end. Dropping the canonical copy does not
   drop the record, and nothing else ever un-claims a table, so the
   refusal outlives the rows it was about;
5. it fails open on exactly the databases most likely to hit the bug:
   the first importer wrote the canonical copy but no
   ``canonical_table`` marker, so those read as "not claimed".

Two things bound the damage and are worth knowing before anyone tries
again. The imported rows in the canonical table are a byte-identical
copy of ``foreign_<object>`` — same columns, same rows — and
``foreign_<object>`` is *not* on the allow-list, so a resume cannot
touch it; deleting from the canonical copy loses a duplicate, not a
measurement. And ``foreign_columns`` records, per table, every column
the importer wrote, which is the ownership signal
``foreign._importer_owns`` already uses and the one any future attempt
should build on — a marker row added here for the purpose was strictly
worse.

The real defect is upstream of resume: ``measure_crop`` appends into a
canonical table an import already filled, whether or not a resume is
involved, and no resume-time guard can undo that. It belongs where the
append happens.

A database already damaged by an earlier resume cannot be repaired on
read — the rows are gone, and nothing in the file records what they
were. It can be *rewritten* from the sources outside the database, and
all three writers use ``if_exists='replace'``, so re-running one is a
full repair rather than a second generation of rows:
``convert.populate_db_from_map(db, '<converted>/conversion_map.csv')``
restores the map from the CSV that sits beside the converted images,
``align.save_coordinates`` restores the stitch coordinates, and
``foreign.run_import`` restores the imported measurements.

**A field is identified by its well coordinates, never by a name prefix.**
Deleting with ``LIKE 'plate1_A01_f1%'`` also matches ``f10`` … ``f19`` —
nineteen innocent fields destroyed to clean up one. Every statement in
here matches on equality over ``plateID`` / ``rowID`` / ``columnID`` /
``fieldID`` (plus ``timeID`` for timelapse), and refuses to touch a table
that does not carry all four.

**A resume across different settings is not a resume.** Concatenating
fields measured with one channel/diameter/crop configuration onto fields
measured with another produces a dataset that is half one thing and half
another, and nothing downstream can tell.
:func:`check_settings_compatible` compares the current settings against
the ones recorded by the previous run and raises :class:`ResumeRefused`
naming exactly what differs. Environment drift (a numpy version bump)
does not block.

Resume is **opt-in**: everything here is inert unless
``settings['resume']`` is true, and a run without it behaves exactly as
it did before.

The module is deliberately **stdlib-only** — ``os``/``sqlite3``/``ast``
and nothing else at import time. It is consulted at the very top of
``measure_crop``, long before any model is loaded, and must never be the
thing that drags torch or cellpose into a process.

Typical use, from ``measure_crop``::

    from .resume import plan_measure_resume
    plan = plan_measure_resume(settings)      # None when resume is off
    files = [f for f in os.listdir(src) if f.endswith('.npy')]
    if plan is not None:
        files = plan.filter_files(files)
"""
from __future__ import annotations

import ast
import os
import sqlite3
from dataclasses import dataclass, field as _dc_field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple)

from . import schema
from .database_concurrency import connect as connect_database, transaction
from .errors import ConfigurationError, read_run_status

__all__ = [
    'ResumeState',
    'ResumeRefused',
    'SettingsComparison',
    'FIELD_KEY_COLUMNS',
    'TIME_KEY_COLUMNS',
    'MEASURE_OWNED_TABLES',
    'NON_FIELD_TABLES',
    'resume_enabled',
    'field_identity',
    'identity_to_prcf',
    'read_npy_header',
    'validate_merged_field',
    'completed_fields_in_merged',
    'discover_field_tables',
    'completed_fields_in_db',
    'clear_field_rows',
    'plan_resume',
    'format_resume',
    'read_recorded_settings',
    'compare_settings',
    'check_settings_compatible',
    'run_already_complete',
    'expected_min_planes',
    'plan_measure_resume',
]

#: Columns that together identify one field of view. Every delete this
#: module issues matches on **all** of these by equality — never a LIKE
#: on a name prefix, which would make ``f1`` match ``f10``…``f19``.
#:
#: Re-exported from :data:`spacr.schema.FIELD_KEY_COLUMNS` rather than
#: spelled out again: a resume delete keyed on a different four columns than
#: the writer used is a delete that misses, and the only way to guarantee it
#: cannot happen is for there to be one tuple.
FIELD_KEY_COLUMNS = schema.FIELD_KEY_COLUMNS

#: Timepoint column, added to the key for timelapse runs. ``timeID`` is
#: canonical and is what both ``_merge_and_save_to_database`` and
#: ``filepaths_to_database`` write. ``time_id`` is the spelling ``png_list``
#: carried before the two were unified; ``utils.rename_columns_in_db``
#: migrates it in place on first read, but a database not read since then
#: still has it, so both are honoured here.
TIME_KEY_COLUMNS = schema.TIME_COLUMN_ALIASES

#: The tables a measure run writes, and therefore the **only** tables a
#: measure resume may delete from.
#:
#: This is an allow-list on purpose. The set is closed and small:
#: ``utils._merge_and_save_to_database`` raises ``ValueError`` for any
#: ``table_type`` outside ``_PARENT_OBJECT_TABLES`` /
#: ``_CHILD_OBJECT_TABLES`` / ``_ORGANELLE_SUMMARY_TABLES``, and
#: ``utils.filepaths_to_database`` writes exactly one more, ``png_list``.
#: (``tests/test_resume_owned_tables.py`` asserts this literal still
#: equals those four groups; the names are spelled out here rather than
#: imported because this module must not drag ``utils`` — and therefore
#: torch — into a process. See the module docstring.)
#:
#: It is **not** aliased to :data:`spacr.schema.MEASUREMENT_TABLES`, which
#: happens to hold the same ten names today. They answer different questions:
#: schema's list is "what does spaCR write?", this one is "what may a resume
#: **delete**?". Coupling them would mean a table added to schema silently
#: authorises a delete from it, which is the deny-list failure mode wearing
#: a different hat. ``FIELD_KEY_COLUMNS`` above *is* aliased, because there
#: the opposite is true: a delete keyed on different columns than the writer
#: used is a delete that misses.
#:
#: A deny-list was tried first and is what the bug was. ``measurements.db``
#: is shared: ``convert.populate_db_from_map`` writes ``conversion_map``,
#: ``align.save_coordinates`` writes ``align_coordinates``,
#: ``foreign.run_import`` writes ``foreign_*``, and
#: ``timelapse._save_measurements_and_well_summary`` writes a
#: user-named track table. All of them are keyed on
#: :data:`FIELD_KEY_COLUMNS` *deliberately*, so that they join to the
#: measurements — which made every one of them look like per-field measure
#: output to a rule that asked "does it have the key columns, and is it
#: not on the deny-list?". A resume then deleted the pending fields'
#: rows from all of them: the only record of which vendor file became
#: ``plate1_A01_1``, where each tile was stitched, and somebody else's
#: imported measurements. None of it is recomputable from the database.
#: A deny-list goes stale the moment a module is added — which is exactly
#: how those three came to be deleted. An allow-list goes stale the other
#: way: a new measure table would be left uncleared and its rows
#: duplicated, which the consistency test named above turns into a failing
#: test rather than silent data loss.
MEASURE_OWNED_TABLES = frozenset({
    'cell', 'cytoplasm',                       # _PARENT_OBJECT_TABLES
    'nucleus', 'pathogen', 'organelle',        # _CHILD_OBJECT_TABLES
    'cell_organelle_summary', 'nucleus_organelle_summary',
    'pathogen_organelle_summary', 'cytoplasm_organelle_summary',
    'png_list',                                # filepaths_to_database
})

#: Tables that live in ``measurements.db`` but are **not** per-field
#: measure output, and so must never be cleared by a measure resume.
#: ``object_counts`` / ``pivoted_counts`` belong to the mask stage;
#: ``settings`` and ``run_status`` are run metadata. None of them carries
#: the four key columns and none of them is in
#: :data:`MEASURE_OWNED_TABLES`, so they are excluded twice over — this is
#: kept as the historical deny-list, not as the mechanism.
NON_FIELD_TABLES = frozenset({
    'object_counts', 'pivoted_counts', 'settings', 'run_status',
    'sqlite_sequence',
})

#: Settings whose value cannot change a measured number: output verbosity,
#: worker counts, plot cosmetics, the resume flag itself. A change in one
#: of these is reported as drift and does **not** block a resume.
COSMETIC_SETTINGS = frozenset({
    'src', 'resume', 'n_jobs', 'plot', 'verbose', 'progress', 'update_gui',
    'test_mode', 'test_images', 'test_nr', 'random_test', 'test_size',
    'examples_to_plot', 'figuresize', 'cmap', 'save_figures', 'show',
    'dry_run', 'strict_errors', 'timestamp', 'from_scratch',
})

#: Keys of ``spacr.run_journal._env_snapshot``. A package version bump is
#: worth reporting but must not stop a resume — the task explicitly says
#: env drift alone does not block.
ENV_SETTINGS = frozenset({
    'spacr', 'spacr_git', 'python', 'platform', 'torch', 'torchvision',
    'cellpose', 'pyside6', 'numpy', 'scipy', 'pandas', 'scikit_image',
    'scikit_learn',
})

_NPY_MAGIC = b'\x93NUMPY'

_TRUTHY = frozenset({'1', 'true', 'yes', 'on', 'y', 't'})

# Reason codes recorded against a field in ResumeState.reasons.
REASON_DONE = 'done'
REASON_MISSING = 'missing'
REASON_TRUNCATED = 'truncated'
REASON_EMPTY = 'empty'
REASON_UNREADABLE = 'unreadable'
REASON_TOO_FEW_PLANES = 'too-few-planes'
REASON_PARTIAL_DB = 'partial-db-rows'
REASON_NOT_MEASURED = 'not-measured'

#: Reasons that mean "a file was present but rejected", as opposed to
#: "there was simply nothing there yet". These are the ones worth
#: shouting about: they are the fields a naive exists() check would have
#: silently accepted.
REJECTION_REASONS = frozenset({
    REASON_TRUNCATED, REASON_EMPTY, REASON_UNREADABLE,
    REASON_TOO_FEW_PLANES, REASON_PARTIAL_DB,
})


class ResumeRefused(ConfigurationError):
    """The requested resume would produce a dataset that is not one dataset.

    Raised when the fields already on disk were produced under settings
    that differ materially from the current ones. This is a
    :class:`~spacr.errors.ConfigurationError` on purpose: it is a setup
    mistake, not a per-field failure, and continuing past it would append
    apples to oranges in a single table with nothing to mark the seam.
    """


# ---------------------------------------------------------------------------
# Opt-in switch
# ---------------------------------------------------------------------------

def resume_enabled(settings: Any) -> bool:
    """True when the caller asked for a resume.

    Resume is off unless explicitly requested, so a settings dict that
    has never heard of the key behaves exactly as it always did.

    :param settings: a spaCR settings dict (or anything else, which reads
        as "off").
    """
    if not isinstance(settings, Mapping):
        return False
    value = settings.get('resume', False)
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


# ---------------------------------------------------------------------------
# Field identity
# ---------------------------------------------------------------------------

def field_identity(field: Any, timelapse: bool = False) -> Dict[str, str]:
    """Parse a merged-stack name into the well coordinates the database stores.

    :func:`spacr.schema.parse_field_stem` is the parse; this function adds the
    mapping passthrough and the "refuse rather than guess" errors a resume
    needs. It used to be a hand-rolled copy of ``spacr.utils._map_wells``,
    because importing ``spacr.utils`` would pull torch and cellpose into every
    process that merely wants to know whether it can skip a field —
    :mod:`spacr.schema` is stdlib-only precisely so that reason no longer
    forces a copy, and ``tests/test_resume.py`` still asserts this module
    imports nothing heavy.

    The copy had drifted, in the direction that matters most here: it gave
    ``'plate1_A_3'`` the identity ``('r1', 'c0')`` while ``_map_wells`` wrote
    ``'error'`` into the database for the same name, so a resume computed a
    delete key that matched **no** rows and the field was measured twice.
    ``'AA01'`` was the same story. Both now raise, which
    :func:`plan_measure_resume` reports rather than silently mis-deleting.

    :param field: ``'plate1_A01_3'``, ``'plate1_A01_3.npy'``, or an
        already-parsed identity mapping (returned filtered, so callers can
        pass either).
    :param timelapse: also parse the 4th component as ``timeID``.
    :returns: dict with ``plateID`` / ``rowID`` / ``columnID`` /
        ``fieldID`` (and ``timeID`` when ``timelapse``). Values match what
        ``_merge_and_save_to_database`` wrote, e.g.
        ``{'plateID': 'plate1', 'rowID': 'r1', 'columnID': 'c1',
        'fieldID': 'f3'}``.
    :raises ValueError: when the name has too few underscore-separated
        parts to identify a field, or when the well cannot be read.
        Guessing here would produce a delete key that matches the wrong
        rows. (:class:`spacr.schema.SchemaError` *is* a ``ValueError``, so
        callers guarding on ``ValueError`` keep working.)

    Example:
        .. code-block:: python

            >>> field_identity('plate1_A01_3')['fieldID']
            'f3'
    """
    if isinstance(field, Mapping):
        keys = FIELD_KEY_COLUMNS + (TIME_KEY_COLUMNS if timelapse else ())
        out = {k: str(field[k]) for k in keys if k in field}
        missing = [k for k in FIELD_KEY_COLUMNS if k not in out]
        if missing:
            raise ValueError(
                f'field identity {dict(field)!r} is missing {missing}; '
                f'a delete keyed on a partial identity would match rows '
                f'belonging to other fields.')
        return out

    stem = os.path.splitext(os.path.basename(str(field)))[0]
    parts = stem.split(schema.KEY_SEPARATOR)
    if len(parts) < 3:
        raise ValueError(
            f'cannot identify a field from {stem!r}: expected at least '
            f'plate_well_field (e.g. "plate1_A01_3"), got {len(parts)} '
            f'part(s). Refusing to guess — a wrong key deletes the wrong '
            f'rows.')
    if timelapse and len(parts) < 4:
        raise ValueError(
            f'timelapse resume needs a time component in {stem!r} '
            f'(plate_well_field_time); got {len(parts)} parts.')

    return schema.parse_field_stem(stem, timelapse=timelapse).to_dict()


def identity_to_prcf(identity: Mapping[str, str]) -> str:
    """Render an identity dict as the ``prcf`` string the tables also carry.

    Used only for display and for the "no candidate list" mode of
    :func:`completed_fields_in_db`; never as a delete key.
    """
    parts = [identity[key] for key in FIELD_KEY_COLUMNS]
    for key in TIME_KEY_COLUMNS:
        if identity.get(key):
            parts.append(str(identity[key]))
            break
    return schema.KEY_SEPARATOR.join(str(p) for p in parts)


def _identity_tuple(identity: Mapping[str, str],
                    columns: Sequence[str]) -> Tuple[str, ...]:
    """Project an identity onto ``columns`` as a comparable tuple of strings."""
    return tuple('' if identity.get(c) is None else str(identity[c])
                 for c in columns)


# ---------------------------------------------------------------------------
# .npy validation — the difference between "the file is there" and "it finished"
# ---------------------------------------------------------------------------

def _descr_itemsize(descr: Any) -> Optional[int]:
    """Bytes per element for a simple numpy dtype string, else None.

    Only the plain forms a merged stack ever uses are understood
    (``'<u2'``, ``'|u1'``, ``'<f4'`` …). A structured or object dtype
    returns None, and the caller treats "cannot verify" as "re-run it" —
    re-running a field is always safe, skipping a bad one is not.
    """
    if not isinstance(descr, str) or not descr:
        return None
    body = descr[1:] if descr[0] in '<>|=' else descr
    if not body:
        return None
    kind, rest = body[0], body[1:]
    if rest.isdigit():
        return int(rest)
    if kind in ('b', '?') and not rest:
        return 1
    return None


def read_npy_header(path: str) -> Dict[str, Any]:
    """Parse a ``.npy`` header without loading (or allocating) the array.

    Reading the header alone is what makes validating a thousand
    100-megabyte fields cheap enough to do on every resume — and it is
    strictly *better* at catching truncation than ``np.load``, which has
    to allocate the full array before it discovers the file is short.

    :param path: path to a ``.npy`` file.
    :returns: dict with ``shape`` (tuple), ``descr``, ``fortran_order``,
        ``header_end`` (byte offset of the data), ``itemsize``
        (None when the dtype is not a simple numeric one),
        ``expected_bytes`` (None when itemsize is unknown) and
        ``actual_bytes``.
    :raises ValueError: when the file is empty, or is not a ``.npy`` at
        all, or its header is unparseable.

    Example:
        .. code-block:: python

            >>> read_npy_header('merged/plate1_A01_3.npy')['shape']
            (1080, 1080, 7)
    """
    actual = os.path.getsize(path)
    if actual == 0:
        raise ValueError(f'{path} is zero bytes — the write never started')
    with open(path, 'rb') as handle:
        magic = handle.read(6)
        if magic != _NPY_MAGIC:
            raise ValueError(f'{path} is not a .npy file (bad magic {magic!r})')
        version = handle.read(2)
        if len(version) != 2:
            raise ValueError(f'{path} ends inside its magic string')
        major = version[0]
        len_size = 2 if major == 1 else 4
        raw_len = handle.read(len_size)
        if len(raw_len) != len_size:
            raise ValueError(f'{path} ends inside its header length')
        header_len = int.from_bytes(raw_len, 'little')
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(
                f'{path} ends inside its header — it claims a {header_len} '
                f'byte header but only {len(raw_header)} are present')
        header_end = 6 + 2 + len_size + header_len
        encoding = 'utf-8' if major >= 3 else 'latin1'
        try:
            meta = ast.literal_eval(raw_header.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f'{path} has an unparseable .npy header: {exc}')
    if not isinstance(meta, dict) or 'shape' not in meta:
        raise ValueError(f'{path} has a .npy header that is not a shape dict')

    shape = tuple(int(dim) for dim in meta.get('shape', ()))
    itemsize = _descr_itemsize(meta.get('descr'))
    expected = None
    if itemsize is not None:
        count = 1
        for dim in shape:
            count *= dim
        expected = header_end + count * itemsize
    return {
        'shape': shape,
        'descr': meta.get('descr'),
        'fortran_order': bool(meta.get('fortran_order', False)),
        'header_end': header_end,
        'itemsize': itemsize,
        'expected_bytes': expected,
        'actual_bytes': actual,
    }


def validate_merged_field(path: str,
                          min_planes: Optional[int] = None) -> Tuple[bool, str]:
    """Decide whether a ``merged/*.npy`` is a finished field or a crash scar.

    :param path: path to the candidate ``.npy``.
    :param min_planes: when given, the array's last axis must be at least
        this large. Derive it from the settings with
        :func:`expected_min_planes` — a stack with too few planes cannot
        be measured with the requested ``cell_mask_dim`` / ``channels``
        regardless of whether its bytes are all there.
    :returns: ``(ok, reason)``. ``reason`` is :data:`REASON_DONE` when
        ok, otherwise one of :data:`REASON_MISSING`, :data:`REASON_EMPTY`,
        :data:`REASON_TRUNCATED`, :data:`REASON_UNREADABLE`,
        :data:`REASON_TOO_FEW_PLANES`.

    Example:
        .. code-block:: python

            >>> validate_merged_field('merged/plate1_A01_3.npy', min_planes=7)
            (True, 'done')
    """
    if not os.path.isfile(path):
        return False, REASON_MISSING
    try:
        if os.path.getsize(path) == 0:
            return False, REASON_EMPTY
        header = read_npy_header(path)
    except OSError:
        return False, REASON_UNREADABLE
    except ValueError:
        # Zero bytes is caught above; anything else here is a header that
        # does not parse — a partial write, or not an array at all.
        return False, REASON_UNREADABLE

    expected = header['expected_bytes']
    if expected is None:
        # Unverifiable dtype. Cannot prove the file is complete, so treat
        # it as pending: re-measuring is safe, skipping garbage is not.
        return False, REASON_UNREADABLE
    if header['actual_bytes'] < expected:
        return False, REASON_TRUNCATED
    if min_planes is not None:
        planes = header['shape'][-1] if header['shape'] else 0
        if planes < min_planes:
            return False, REASON_TOO_FEW_PLANES
    return True, REASON_DONE


def expected_min_planes(settings: Any) -> Optional[int]:
    """Smallest last-axis size a merged stack must have for these settings.

    Every plane index the measure stage will subscript — ``channels``,
    ``png_dims`` and the four ``*_mask_dim`` entries — has to exist in the
    array, so the stack needs at least ``max(index) + 1`` planes.

    :param settings: a measure settings dict.
    :returns: the minimum plane count, or None when the settings name no
        plane at all.
    """
    if not isinstance(settings, Mapping):
        return None
    indices: List[int] = []
    for key in ('channels', 'png_dims'):
        value = settings.get(key)
        if isinstance(value, (list, tuple)):
            indices.extend(int(v) for v in value
                           if isinstance(v, (int, float)) and not isinstance(v, bool))
    for key in ('cell_mask_dim', 'nucleus_mask_dim', 'pathogen_mask_dim',
                'organelle_mask_dim', 'cytoplasm_mask_dim'):
        value = settings.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            indices.append(int(value))
    if not indices:
        return None
    return max(indices) + 1


def completed_fields_in_merged(src: str,
                               min_planes: Optional[int] = None,
                               reasons: Optional[Dict[str, str]] = None,
                               fields: Optional[Iterable[str]] = None,
                               ) -> Set[str]:
    """Field stems in ``src`` whose ``.npy`` is *verified* complete.

    This is the function that stops a resume from turning into corrupt
    output. It never answers on the strength of ``os.path.exists``: each
    candidate's header is parsed and its length checked against its own
    declared shape, because ``np.save`` writes merged arrays in place and
    a run killed mid-write leaves a short file behind.

    :param src: the ``merged/`` folder (or any folder of ``.npy`` fields).
    :param min_planes: minimum last-axis size, see
        :func:`expected_min_planes`. When None and at least three files
        validate, the *modal* plane count across the folder is used
        instead — every field in a merged folder is written with the same
        plane count, so an odd one out is a bad field.
    :param reasons: optional dict, populated in place with
        ``{field: reason}`` for every candidate that was **rejected**.
        This is how the caller can report "3 fields rejected as
        truncated" rather than silently doing more work.
    :param fields: restrict the scan to these stems. Defaults to every
        ``.npy`` in ``src``.
    :returns: set of field stems (basename without ``.npy``) that are
        safe to skip.

    Example:
        .. code-block:: python

            rejected = {}
            done = completed_fields_in_merged('/data/p1/merged',
                                              min_planes=7, reasons=rejected)
            print(f'{len(done)} done, {len(rejected)} rejected: {rejected}')
    """
    if reasons is None:
        reasons = {}
    if not os.path.isdir(src):
        return set()

    if fields is None:
        names = sorted(f for f in os.listdir(src) if f.endswith('.npy'))
        stems = [os.path.splitext(n)[0] for n in names]
    else:
        stems = [os.path.splitext(os.path.basename(str(f)))[0] for f in fields]

    # Pass 1: structural validity and plane counts.
    verdicts: Dict[str, Tuple[bool, str]] = {}
    planes: Dict[str, int] = {}
    for stem in stems:
        path = os.path.join(src, stem + '.npy')
        ok, reason = validate_merged_field(path, min_planes=min_planes)
        verdicts[stem] = (ok, reason)
        if ok:
            try:
                shape = read_npy_header(path)['shape']
            except (OSError, ValueError):
                shape = ()
            planes[stem] = shape[-1] if shape else 0

    # Pass 2: modal plane count, only when the caller gave no explicit
    # floor and there is enough of a population to have a mode. A merged
    # folder is written by one loop with one channel layout, so a field
    # with fewer planes than its neighbours did not finish.
    if min_planes is None and len(planes) >= 3:
        counts: Dict[int, int] = {}
        for value in planes.values():
            counts[value] = counts.get(value, 0) + 1
        modal = max(sorted(counts), key=lambda v: counts[v])
        for stem, value in planes.items():
            if value < modal:
                verdicts[stem] = (False, REASON_TOO_FEW_PLANES)

    done = set()
    for stem, (ok, reason) in verdicts.items():
        if ok:
            done.add(stem)
        else:
            reasons[stem] = reason
    return done


# ---------------------------------------------------------------------------
# Database side
# ---------------------------------------------------------------------------

def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Column names of ``table``, or ``[]`` when it does not exist."""
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return []
    return [row[1] for row in rows]


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    """Every real table in the database, in name order."""
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [row[0] for row in rows]


def discover_field_tables(db_path: str,
                          include_non_field: bool = False,
                          owned_only: bool = True) -> List[str]:
    """The measure tables in ``db_path`` that store rows *per field of view*.

    Enumerated from the database rather than hard-coded, because a delete
    that misses one table leaves orphan rows that join incorrectly
    forever — and the set genuinely varies with the settings: no
    ``organelle`` table without an organelle channel, no
    ``*_organelle_summary`` tables without ``summarize_organelles_by``,
    no ``png_list`` without ``save_png``.

    A table qualifies when it is in :data:`MEASURE_OWNED_TABLES` **and**
    carries all of :data:`FIELD_KEY_COLUMNS`. Both halves are needed. The
    column test alone is not a test of ownership: ``conversion_map``,
    ``align_coordinates`` and ``foreign_*`` all carry the same four
    columns, precisely so that they join to the measurements, and they
    passed it — which is how a measure resume came to delete three other
    modules' provenance tables. The name test alone would delete from a
    table whose schema is not what this module thinks it is.

    :param db_path: path to ``measurements.db``.
    :param include_non_field: skip the :data:`NON_FIELD_TABLES` filter.
        For inspection only — never pass this to :func:`clear_field_rows`.
    :param owned_only: keep only :data:`MEASURE_OWNED_TABLES`. Pass False
        to see every per-field table in the database *whoever wrote it* —
        for inspection only, and :func:`clear_field_rows` refuses the
        extras by name if that list is handed to it.
    :returns: sorted table names.
    """
    if not os.path.isfile(db_path):
        return []
    conn = connect_database(db_path, readonly=True, timeout=30)
    try:
        out = []
        for table in _list_tables(conn):
            if not include_non_field and table in NON_FIELD_TABLES:
                continue
            if owned_only and table not in MEASURE_OWNED_TABLES:
                continue
            columns = set(_table_columns(conn, table))
            if all(key in columns for key in FIELD_KEY_COLUMNS):
                out.append(table)
        return out
    finally:
        conn.close()


def _key_columns_for(conn: sqlite3.Connection, table: str,
                     identity: Mapping[str, str]) -> List[str]:
    """Key columns usable on ``table``, or raise if the four are not all there.

    Refusing is the point: matching on a subset of the key would delete
    every field in the plate, or every field in the row.
    """
    columns = set(_table_columns(conn, table))
    if not columns:
        raise ValueError(f'table {table!r} does not exist in this database')
    missing = [key for key in FIELD_KEY_COLUMNS if key not in columns]
    if missing:
        raise ValueError(
            f'table {table!r} has no {missing} column(s); a delete keyed on '
            f'only {sorted(columns & set(FIELD_KEY_COLUMNS))} would match '
            f'rows from other fields. Refusing.')
    keys = list(FIELD_KEY_COLUMNS)
    for time_col in TIME_KEY_COLUMNS:
        if time_col in columns and identity.get('timeID') is not None:
            keys.append(time_col)
            break
    return keys


def _bind_values(identity: Mapping[str, str],
                 keys: Sequence[str]) -> List[str]:
    """Values for ``keys``, mapping either time-column spelling onto timeID."""
    values = []
    for key in keys:
        if key in TIME_KEY_COLUMNS:
            values.append(str(identity.get('timeID')
                              if identity.get('timeID') is not None
                              else identity.get(key)))
        else:
            values.append(str(identity[key]))
    return values


def completed_fields_in_db(db_path: str,
                           tables: Optional[Sequence[str]] = None,
                           fields: Optional[Iterable[str]] = None,
                           timelapse: bool = False,
                           require_all: bool = True,
                           partial: Optional[Dict[str, str]] = None,
                           ) -> Set[str]:
    """Fields that are already measured in ``db_path``.

    "Already measured" deliberately means *present in every table this
    run writes*, not "present somewhere". ``_measure_crop_core`` inserts
    into ``cell``, then ``nucleus``, then ``pathogen``, then
    ``cytoplasm``, then ``png_list``, one call after another with a
    commit each — so a process killed between two of those calls leaves a
    field that has cell rows and no nucleus rows. Counting that field as
    done would permanently lose its nuclei; counting it as pending
    without clearing it first would duplicate its cells. It is reported
    through ``partial`` so the caller can do the third, correct thing:
    clear it and re-run it.

    :param db_path: path to ``measurements.db``.
    :param tables: tables to consult. Defaults to
        :func:`discover_field_tables`.
    :param fields: candidate field stems (e.g. from ``merged/``). When
        given, the return value is the subset of these that are complete
        — which is what a caller filtering a file list wants. When None,
        ``prcf`` strings assembled from the rows themselves are returned.
    :param timelapse: include the timepoint in the field key.
    :param require_all: when False, a field counts as done if it appears
        in *any* table. Only useful for inspection.
    :param partial: optional dict, populated with ``{field:
        'partial-db-rows'}`` for fields present in some tables but not
        all. These must be cleared before being re-run.
    :returns: set of completed field identifiers.
    """
    if partial is None:
        partial = {}
    if not os.path.isfile(db_path):
        return set()
    if tables is None:
        tables = discover_field_tables(db_path)
    tables = [t for t in tables if t not in NON_FIELD_TABLES]
    if not tables:
        return set()

    key_columns = list(FIELD_KEY_COLUMNS)
    conn = connect_database(db_path, readonly=True, timeout=30)
    try:
        present: Dict[str, Set[Tuple[str, ...]]] = {}
        for table in tables:
            columns = set(_table_columns(conn, table))
            if not all(key in columns for key in FIELD_KEY_COLUMNS):
                continue
            select = list(key_columns)
            time_col = None
            if timelapse:
                for candidate in TIME_KEY_COLUMNS:
                    if candidate in columns:
                        time_col = candidate
                        select.append(candidate)
                        break
            quoted = ', '.join(f'"{c}"' for c in select)
            try:
                rows = conn.execute(
                    f'SELECT DISTINCT {quoted} FROM "{table}"').fetchall()
            except sqlite3.Error:
                continue
            seen = set()
            for row in rows:
                values = ['' if v is None else str(v) for v in row]
                if timelapse and time_col is None:
                    values.append('')
                seen.add(tuple(values))
            present[table] = seen
        if not present:
            return set()

        if fields is not None:
            done = set()
            for raw in fields:
                stem = os.path.splitext(os.path.basename(str(raw)))[0]
                try:
                    identity = field_identity(stem, timelapse=timelapse)
                except ValueError:
                    continue
                key = _identity_tuple(
                    identity,
                    key_columns + (['timeID'] if timelapse else []))
                hits = [t for t, seen in present.items() if key in seen]
                if not hits:
                    continue
                if len(hits) == len(present) or not require_all:
                    done.add(stem)
                else:
                    partial[stem] = REASON_PARTIAL_DB
            return done

        # No candidate list: answer in terms of the rows themselves.
        all_keys: Set[Tuple[str, ...]] = set()
        for seen in present.values():
            all_keys |= seen
        done = set()
        for key in all_keys:
            hits = sum(1 for seen in present.values() if key in seen)
            identity = dict(zip(key_columns + (['timeID'] if timelapse else []),
                                key))
            label = identity_to_prcf(identity)
            if hits == len(present) or not require_all:
                done.add(label)
            else:
                partial[label] = REASON_PARTIAL_DB
        return done
    finally:
        conn.close()


def clear_field_rows(db_path: str,
                     tables: Optional[Sequence[str]],
                     field: Any,
                     timelapse: bool = False) -> int:
    """Delete every row belonging to one field, from every table, atomically.

    **This is the delete-before-insert.** Measure appends
    (``to_sql(if_exists='append')``), so re-running a field that already
    wrote rows adds a second copy of every object rather than replacing
    the first. Downstream, ``count_cell`` doubles, per-well means are
    computed over the doubled population, and nothing in the artifact
    says so. Calling this immediately before re-measuring a field is what
    makes a resume idempotent.

    Three safety properties, all tested:

    * **All or nothing.** Every table is deleted from inside one
      ``BEGIN IMMEDIATE`` … ``COMMIT``. If any statement fails — a
      missing table, a trigger, a lock — the transaction is rolled back
      and the database is left exactly as it was. A half-cleared field
      would be worse than an uncleared one.
    * **Keyed on the field, not on its name.** The WHERE clause is
      equality over ``plateID``/``rowID``/``columnID``/``fieldID``, so
      clearing ``f1`` cannot touch ``f10``–``f19`` the way a
      ``LIKE 'plate1_A01_f1%'`` would. A table missing any of the four
      raises rather than running a broader delete.
    * **Only measure's own tables.** Every table is checked against
      :data:`MEASURE_OWNED_TABLES` before anything is deleted, and one
      that is not on it aborts the whole call. ``measurements.db`` is
      shared — ``conversion_map``, ``align_coordinates`` and ``foreign_*``
      live there and carry the same four key columns so that they join —
      and clearing a field out of those destroys the only record of how
      the project's files were named and registered.

    Owned *by name* is not the same as owned *in fact*, and this does not
    check the difference. In a project built by ``foreign.run_import``
    the rows in ``cell`` are the import's — a byte-identical copy of
    ``foreign_cell``, made so that a purely-imported project is readable
    by every spaCR tool — and clearing a pending field takes them with
    it. The duplicate in ``foreign_cell`` is untouched (it is not on the
    allow-list), so nothing is lost that this database does not still
    hold, but ``cell`` is left half theirs and half spaCR's. A guard
    against that was tried here and backed out for being worse than the
    bug; the module docstring records what it did and the five ways it
    failed, and ``tests/test_resume_owned_tables.py`` pins the behaviour
    with ``xfail(strict=True)``.

    :param db_path: path to ``measurements.db``.
    :param tables: tables to clear. ``None`` uses
        :func:`discover_field_tables`, which is the safe default —
        naming tables by hand is how one gets missed.
    :param field: field stem (``'plate1_A01_3'``), filename, or an
        identity mapping.
    :param timelapse: include the timepoint in the key, so one frame can
        be cleared without touching the rest of the movie.
    :returns: total number of rows deleted.
    :raises ValueError: when the field cannot be identified, a named
        table lacks the key columns, or a named table is not one measure
        writes.
    :raises sqlite3.Error: propagated after rollback.

    Example:
        .. code-block:: python

            n = clear_field_rows(db, None, 'plate1_A01_3')
            print(f'cleared {n} stale rows before re-measuring')
    """
    if not os.path.isfile(db_path):
        return 0
    identity = field_identity(field, timelapse=timelapse)
    if tables is None:
        tables = discover_field_tables(db_path)
    tables = [t for t in tables if t not in NON_FIELD_TABLES]
    if not tables:
        return 0

    conn = connect_database(db_path, timeout=30)
    try:
        deleted = 0
        with transaction(conn):
            # Pre-flight and deletes share one write transaction. No other
            # writer can replace a checked table between validation and use.
            plans = []
            for table in tables:
                keys = _key_columns_for(conn, table, identity)
                if table not in MEASURE_OWNED_TABLES:
                    raise ValueError(
                        f'table {table!r} carries the field key columns but is '
                        f'not written by the measure stage — measure writes only '
                        f'{sorted(MEASURE_OWNED_TABLES)}. Other modules key their '
                        f'tables the same way so that they join: convert writes '
                        f'conversion_map, align writes align_coordinates, foreign '
                        f'writes foreign_*, timelapse writes its track table. '
                        f'Deleting from {table!r} would destroy the only record '
                        f'of how this project was registered, and none of it can '
                        f'be recomputed from the database. Refusing.')
                where = ' AND '.join(f'"{k}" = ?' for k in keys)
                plans.append((f'DELETE FROM "{table}" WHERE {where}',
                              _bind_values(identity, keys)))
            for sql, values in plans:
                cursor = conn.execute(sql, values)
                deleted += cursor.rowcount if cursor.rowcount > 0 else 0
        return deleted
    finally:
        conn.close()


def run_already_complete(db_path: str, name: Optional[str] = None) -> bool:
    """True when ``db_path`` carries a :class:`~spacr.errors.RunLedger` stamp saying so.

    A run that finished cleanly needs no resume at all, and this is the
    cheapest way to know: ``measure_crop`` ends with
    ``ledger.finalize(artifact=db_path)``, which appends a ``run_status``
    row recording how many fields were attempted and how many failed.

    Note this is stricter than :func:`spacr.errors.run_is_complete`,
    which reads an *unstamped* artifact as complete (stamping is newer
    than most data on disk). For a resume, "nobody ever said" must mean
    "go and check the files", not "nothing to do".

    :param db_path: path to ``measurements.db``.
    :param name: only consider stamps from this ledger, e.g.
        ``'measure_crop'``. None considers the most recent stamp of any
        stage.
    :returns: True only when a matching stamp exists and its latest entry
        recorded zero failures over at least one attempted item.
    """
    try:
        records = read_run_status(db_path)
    except Exception:
        return False
    if name is not None:
        records = [r for r in records if str(r.get('name', '')) == name]
    if not records:
        return False
    last = records[-1]
    return (int(last.get('n_failed', 0) or 0) == 0
            and int(last.get('n_attempted', 0) or 0) > 0)


# ---------------------------------------------------------------------------
# Settings compatibility — a resume across different settings is not a resume
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SettingsComparison:
    """What differs between the recorded settings and the current ones.

    Bucketed by *consequence*, not by key, following the same reasoning
    as :func:`spacr.run_journal.diff_runs`: a flat dict diff of two runs
    is mostly schema noise, and the one change that matters drowns in it.

    :ivar changed: material differences — these block the resume.
    :ivar drift: cosmetic differences (worker counts, plotting) that
        cannot alter a measured number.
    :ivar env: package/platform differences. Reported, never blocking.
    :ivar only_in_recorded: keys the old run had and this one does not.
    :ivar only_in_current: keys this run has and the old one did not,
        with an inert (null/empty) value.
    :ivar same: count of keys that agree.
    """

    changed: Tuple[Dict[str, Any], ...] = ()
    drift: Tuple[Dict[str, Any], ...] = ()
    env: Tuple[Dict[str, Any], ...] = ()
    only_in_recorded: Tuple[str, ...] = ()
    only_in_current: Tuple[str, ...] = ()
    same: int = 0

    @property
    def blocks_resume(self) -> bool:
        """True when at least one material setting differs."""
        return bool(self.changed)

    def describe(self) -> str:
        """One-line-per-change rendering naming exactly what differs."""
        lines = []
        for entry in self.changed:
            lines.append(f"  {entry['key']}: {entry['recorded']!r} -> "
                         f"{entry['current']!r}")
        return '\n'.join(lines)


def _normalize(value: Any) -> Any:
    """Canonicalise a settings value, reusing the journal's own rules.

    ``spacr.run_journal`` already solved this: settings reach disk as a
    live dict, as JSON with ``default=str``, or as an all-strings CSV, so
    ``[0, 1, 2]`` and ``"[0, 1, 2]"`` and ``"None"``/``None`` are the same
    setting recorded twice. Comparing by repr invents differences.
    Imported lazily so this module's import stays stdlib-only.
    """
    try:
        from .run_journal import _normalize_value
    except Exception:
        return value
    try:
        return _normalize_value(value)
    except Exception:
        return repr(value)


def _values_equal(a: Any, b: Any) -> bool:
    """Structural equality of two settings values; never raises."""
    try:
        from .run_journal import values_equal
    except Exception:
        return repr(a) == repr(b)
    return values_equal(a, b)


def _is_inert(value: Any) -> bool:
    """True when a value means "unset" — None, '', 'none', 'null'."""
    normalized = _normalize(value)
    return normalized is None or normalized == ()


def read_recorded_settings(source: str) -> Dict[str, Any]:
    """Read the settings a previous run wrote, from wherever it wrote them.

    Three shapes are understood, which between them cover both stages:

    * a ``measurements.db`` — read from the ``settings`` table
      (``setting_key`` / ``setting_value``) that
      ``spacr.io._save_settings_to_db`` writes at the top of every
      ``measure_crop``. Note that call happens *before* field
      enumeration and uses ``if_exists='replace'``, so a resume must read
      this table before the current run overwrites it.
    * a ``Key,Value`` CSV — ``<src>/settings/gen_mask_settings.csv`` or
      ``measure_crop_settings.csv``, as written by
      ``spacr.utils.save_settings``.
    * a run-journal folder containing ``settings.json``.

    :param source: path to any of the above.
    :returns: the recorded settings dict, or ``{}`` when nothing is
        recorded (an artifact that predates settings capture — which must
        read as "no information", not as "everything matches").
    """
    if not source:
        return {}
    path = str(source)
    if os.path.isdir(path):
        try:
            from .run_journal import load_run_settings
            from pathlib import Path
            return dict(load_run_settings(Path(path)) or {})
        except Exception:
            return {}
    if not os.path.isfile(path):
        return {}
    if path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
        conn = connect_database(path, readonly=True, timeout=30)
        try:
            rows = conn.execute(
                'SELECT setting_key, setting_value FROM settings').fetchall()
        except sqlite3.Error:
            return {}
        finally:
            conn.close()
        return {str(k): v for k, v in rows}
    if path.lower().endswith('.csv'):
        import csv
        out: Dict[str, Any] = {}
        try:
            with open(path, newline='') as handle:
                for row in csv.reader(handle):
                    if row and row[0] and row[0] not in ('Key', 'setting_key'):
                        out[row[0]] = row[1] if len(row) > 1 else ''
        except OSError:
            return {}
        return out
    if path.lower().endswith('.json'):
        import json
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return dict(data) if isinstance(data, dict) else {}
    return {}


def compare_settings(recorded: Mapping[str, Any],
                     current: Mapping[str, Any],
                     cosmetic: Iterable[str] = COSMETIC_SETTINGS,
                     ) -> SettingsComparison:
    """Bucket the difference between two settings dicts by consequence.

    A key is **material** unless it is explicitly known to be cosmetic.
    That direction is deliberate: an unrecognised new setting that
    changed will block the resume, which is the outcome one wants to be
    wrong in. The alternative — allow-listing the settings that matter —
    silently permits every knob nobody thought of.

    A key present on only one side is schema drift and does not block,
    *unless* it appears only in the current settings with a real
    (non-null) value: enabling ``organelle_channel`` on the resume run
    genuinely changes what is measured even though the old run never
    recorded the key.

    :param recorded: settings read back from the previous run.
    :param current: the settings this run is about to use.
    :param cosmetic: keys treated as inconsequential.
    :returns: a :class:`SettingsComparison`.
    """
    cosmetic = frozenset(cosmetic)
    recorded = dict(recorded or {})
    current = dict(current or {})

    changed: List[Dict[str, Any]] = []
    drift: List[Dict[str, Any]] = []
    env: List[Dict[str, Any]] = []
    same = 0

    for key in sorted(set(recorded) & set(current)):
        if _values_equal(recorded[key], current[key]):
            same += 1
            continue
        entry = {'key': key, 'recorded': recorded[key], 'current': current[key]}
        if key in ENV_SETTINGS:
            env.append(entry)
        elif key in cosmetic:
            drift.append(entry)
        else:
            changed.append(entry)

    only_recorded = sorted(set(recorded) - set(current))
    only_current = sorted(set(current) - set(recorded))
    inert_only_current = []
    for key in only_current:
        if key in cosmetic or key in ENV_SETTINGS or _is_inert(current[key]):
            inert_only_current.append(key)
        else:
            changed.append({'key': key, 'recorded': None,
                            'current': current[key]})
    changed.sort(key=lambda e: str(e['key']))

    return SettingsComparison(
        changed=tuple(changed),
        drift=tuple(drift),
        env=tuple(env),
        only_in_recorded=tuple(only_recorded),
        only_in_current=tuple(inert_only_current),
        same=same,
    )


def check_settings_compatible(recorded: Mapping[str, Any],
                              current: Mapping[str, Any],
                              source: str = '',
                              cosmetic: Iterable[str] = COSMETIC_SETTINGS,
                              ) -> SettingsComparison:
    """Raise :class:`ResumeRefused` when resuming would mix two datasets.

    The fields already on disk were produced under ``recorded``. If
    ``current`` differs in anything that affects the numbers, appending
    new fields to them yields a table that is half one experiment and
    half another, with no column marking the boundary — the single worst
    outcome this whole module exists to prevent. Better to refuse and
    make the user say what they meant.

    :param recorded: settings read back from the previous run.
    :param current: the settings this run would use.
    :param source: where ``recorded`` came from, quoted in the error.
    :param cosmetic: keys treated as inconsequential.
    :returns: the :class:`SettingsComparison` when compatible.
    :raises ResumeRefused: naming every material difference.
    """
    comparison = compare_settings(recorded, current, cosmetic=cosmetic)
    if not comparison.blocks_resume:
        return comparison
    where = f' recorded in {source}' if source else ''
    raise ResumeRefused(
        f'Cannot resume: {len(comparison.changed)} setting(s) differ from the '
        f'run that produced the fields already on disk{where}. Resuming would '
        f'append fields measured one way onto fields measured another, in the '
        f'same tables, with nothing to tell them apart.\n'
        f'{comparison.describe()}\n'
        f'Either restore those settings, or start a fresh run into a clean '
        f'output folder.')


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class ResumeState:
    """What a resume decided to do, before it does any of it.

    :ivar total: number of candidate fields considered.
    :ivar done: fields verified complete — these are skipped.
    :ivar pending: fields that will be run, in input order.
    :ivar skipped: the fields actually skipped (``done`` restricted to the
        candidates, so the two counts always add up to ``total``).
    :ivar reasons: ``{field: reason}`` for every pending field —
        ``'truncated'``, ``'partial-db-rows'``, ``'not-measured'``, … This
        is the audit trail: a field that a naive ``exists()`` check would
        have skipped shows up here with the reason it was not.
    :ivar enabled: False when resume was not requested, in which case
        ``pending`` is every field and nothing is skipped.
    :ivar cleared_rows: stale rows deleted by the delete-before-insert.
    :ivar src: the merged folder inspected, for the report.
    :ivar db_path: the database inspected, for the report.
    """

    total: int = 0
    done: Tuple[str, ...] = ()
    pending: Tuple[str, ...] = ()
    skipped: Tuple[str, ...] = ()
    reasons: Dict[str, str] = _dc_field(default_factory=dict)
    enabled: bool = True
    cleared_rows: int = 0
    src: str = ''
    db_path: str = ''

    @property
    def n_done(self) -> int:
        """How many fields were verified complete."""
        return len(self.done)

    @property
    def n_pending(self) -> int:
        """How many fields will be run."""
        return len(self.pending)

    @property
    def n_skipped(self) -> int:
        """How many fields are being skipped."""
        return len(self.skipped)

    @property
    def rejected(self) -> Tuple[str, ...]:
        """Fields that were present but rejected — the important number.

        A truncated ``.npy``, or a field with rows in some tables and not
        others. These would have been silently skipped by an
        ``os.path.exists`` resume, and measuring them would have produced
        garbage that looked exactly like data.
        """
        return tuple(f for f in self.pending
                     if self.reasons.get(f) in REJECTION_REASONS)

    @property
    def n_rejected(self) -> int:
        """How many present-but-unusable fields were re-queued."""
        return len(self.rejected)

    def rejection_counts(self) -> Dict[str, int]:
        """``{reason: count}`` over :attr:`rejected`, for the report."""
        counts: Dict[str, int] = {}
        for name in self.rejected:
            reason = self.reasons.get(name, REASON_NOT_MEASURED)
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def filter_files(self, files: Iterable[str]) -> List[str]:
        """Restrict a list of ``.npy`` filenames to the pending fields.

        Order and extensions are preserved, so the caller's list is the
        same list minus the fields already done. With
        :attr:`enabled` False the list is returned unchanged — that is
        what keeps resume opt-in.

        :param files: filenames as ``os.listdir`` produced them.
        :returns: the subset still to process.
        """
        files = list(files)
        if not self.enabled:
            return files
        pending = set(self.pending)
        return [f for f in files
                if os.path.splitext(os.path.basename(f))[0] in pending]


def plan_resume(all_fields: Iterable[str],
                done: Iterable[str],
                reasons: Optional[Mapping[str, str]] = None,
                enabled: bool = True,
                src: str = '',
                db_path: str = '',
                cleared_rows: int = 0) -> ResumeState:
    """Work out what to run, what to skip, and record why.

    Nothing is done here — that is the point. The plan is computed and
    reported first, so a resume that is about to skip 900 fields says so
    before it skips them, and a resume that rejected three truncated
    files says that too.

    :param all_fields: every candidate field stem (or filename), in the
        order they should be processed.
    :param done: field stems verified complete and safe to skip.
    :param reasons: known ``{field: reason}`` entries, e.g. from
        :func:`completed_fields_in_merged`'s rejection dict.
    :param enabled: when False, every field is pending and none is
        skipped — the default-behaviour path.
    :param src: merged folder, recorded for the report.
    :param db_path: database, recorded for the report.
    :param cleared_rows: rows already deleted by the caller's
        delete-before-insert, recorded for the report.
    :returns: a :class:`ResumeState`.
    """
    stems = [os.path.splitext(os.path.basename(str(f)))[0]
             for f in all_fields]
    # Preserve order, drop duplicates.
    ordered: List[str] = []
    seen: Set[str] = set()
    for stem in stems:
        if stem not in seen:
            seen.add(stem)
            ordered.append(stem)

    done_set = {os.path.splitext(os.path.basename(str(f)))[0] for f in done}
    known = dict(reasons or {})

    if not enabled:
        return ResumeState(
            total=len(ordered), done=(), pending=tuple(ordered),
            skipped=(), reasons={}, enabled=False, cleared_rows=0,
            src=src, db_path=db_path)

    pending: List[str] = []
    skipped: List[str] = []
    out_reasons: Dict[str, str] = {}
    for stem in ordered:
        if stem in done_set:
            skipped.append(stem)
        else:
            pending.append(stem)
            out_reasons[stem] = known.get(stem, REASON_NOT_MEASURED)

    return ResumeState(
        total=len(ordered),
        done=tuple(sorted(done_set & seen)),
        pending=tuple(pending),
        skipped=tuple(skipped),
        reasons=out_reasons,
        enabled=True,
        cleared_rows=cleared_rows,
        src=src,
        db_path=db_path,
    )


_RULE = '=' * 78


def format_resume(state: ResumeState, max_examples: int = 4) -> str:
    """Render the plan as the block printed before any work starts.

    Reports the four numbers that decide whether the resume is doing what
    the user thinks: how many fields exist, how many are being skipped,
    how many will run, and — the one that matters — how many were found
    on disk and **rejected** anyway, with the reason.

    :param state: the plan from :func:`plan_resume`.
    :param max_examples: field names shown per rejection reason.
    :returns: a multi-line string ready to print.
    """
    lines = [_RULE]
    if not state.enabled:
        lines.append(' spaCR resume: OFF — processing all fields')
        lines.append(_RULE)
        return '\n'.join(lines)

    lines.append(' spaCR RESUME')
    lines.append(_RULE)
    if state.src:
        lines.append(f' source    : {state.src}')
    if state.db_path:
        lines.append(f' database  : {state.db_path}')
    lines.append(f' total     : {state.total} field(s)')
    lines.append(f' done      : {state.n_skipped} skipped (already complete)')
    lines.append(f' to run    : {state.n_pending}')

    counts = state.rejection_counts()
    if counts:
        lines.append(f' rejected  : {state.n_rejected} field(s) were already '
                     f'present but NOT usable, and will be redone:')
        by_reason: Dict[str, List[str]] = {}
        for name in state.rejected:
            by_reason.setdefault(state.reasons.get(name, REASON_NOT_MEASURED),
                                 []).append(name)
        for reason in sorted(by_reason):
            names = by_reason[reason]
            shown = ', '.join(names[:max_examples])
            more = (f', … +{len(names) - max_examples} more'
                    if len(names) > max_examples else '')
            lines.append(f'   {reason:<16} x{len(names)}  {shown}{more}')
    if state.cleared_rows:
        lines.append(f' cleared   : {state.cleared_rows} stale row(s) deleted '
                     f'before re-insert (delete-before-insert)')
    if state.n_pending == 0:
        lines.append('')
        lines.append(' Nothing to do — every field is already measured.')
    lines.append(_RULE)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# The measure_crop entry point
# ---------------------------------------------------------------------------

def measurements_db_path(settings: Mapping[str, Any]) -> str:
    """Where ``measure_crop`` writes, derived exactly as it derives it.

    ``settings['src']`` points at ``<experiment>/merged`` by the time the
    resume runs, and the database is ``<experiment>/measurements/
    measurements.db`` — the same ``os.path.dirname`` hop
    ``_measure_crop_core`` and ``_save_settings_to_db`` make.
    """
    src = str(settings.get('src', ''))
    return os.path.join(os.path.dirname(src), 'measurements', 'measurements.db')


def plan_measure_resume(settings: Any,
                        verbose: bool = True) -> Optional[ResumeState]:
    """Plan (and make safe) a resumed ``measure_crop``. The one call measure makes.

    Returns ``None`` immediately unless ``settings['resume']`` is set, so
    a default run does exactly what it always did.

    When resume *is* requested, in order:

    1. Read the previous run's settings out of the ``settings`` table in
       ``measurements.db`` and refuse loudly if anything material
       differs. **This must happen before**
       ``spacr.io._save_settings_to_db`` overwrites that table with the
       current settings — hence the placement of the call in
       ``measure_crop``.
    2. Enumerate the fields in ``merged/`` and *validate* each one:
       a truncated ``.npy`` from a crash mid-``np.save`` is pending, not
       done, and is counted as rejected.
    3. Ask the database which fields already have rows in **every** table
       the run writes. Fields with rows in only some tables are partial —
       a process killed between two inserts — and are re-queued.
    4. Delete every row belonging to a pending field before it is re-run.
       This is what stops the resume from doubling rows.
    5. Print the plan.

    :param settings: the measure settings dict.
    :param verbose: print the plan block.
    :returns: a :class:`ResumeState` whose :meth:`ResumeState.filter_files`
        turns the caller's file list into the pending subset, or ``None``
        when resume is off.
    :raises ResumeRefused: when the recorded settings differ materially.
    """
    if not resume_enabled(settings):
        return None

    src = str(settings.get('src', ''))
    db_path = measurements_db_path(settings)
    timelapse = bool(settings.get('timelapse', False))
    min_planes = expected_min_planes(settings)

    all_files = []
    if os.path.isdir(src):
        all_files = sorted(f for f in os.listdir(src) if f.endswith('.npy'))
    all_fields = [os.path.splitext(f)[0] for f in all_files]

    # 1. Settings guard. A recorded-settings table that does not exist
    #    yet means nothing has been measured, so there is nothing to be
    #    incompatible with.
    recorded = read_recorded_settings(db_path)
    if recorded:
        check_settings_compatible(recorded, dict(settings), source=db_path)

    # 2. Which fields are physically usable.
    rejected: Dict[str, str] = {}
    usable = completed_fields_in_merged(src, min_planes=min_planes,
                                        reasons=rejected, fields=all_fields)

    # 3. Which fields the database already has, in full.
    partial: Dict[str, str] = {}
    measured = completed_fields_in_db(db_path, tables=None, fields=usable,
                                      timelapse=timelapse, require_all=True,
                                      partial=partial)

    reasons: Dict[str, str] = {}
    reasons.update(rejected)      # truncated / empty / too-few-planes
    reasons.update(partial)       # rows in some tables but not all

    state = plan_resume(all_fields, measured, reasons=reasons, enabled=True,
                        src=src, db_path=db_path)

    # 4. Delete-before-insert. Every field about to be re-measured has any
    #    rows it already left behind removed first, in one transaction per
    #    field. Without this the resume silently doubles objects.
    #
    #    Only the fields that actually have rows are cleared. On a typical
    #    resume that is one field — the one that was mid-flight when the
    #    run died — and issuing a DELETE for the other ninety-nine would
    #    mean ninety-nine full scans of a million-row table to delete
    #    nothing. `require_all=False` is the "has rows anywhere" query.
    cleared = 0
    if os.path.isfile(db_path):
        tables = discover_field_tables(db_path)
        if tables and state.pending:
            dirty = completed_fields_in_db(db_path, tables=tables,
                                           fields=state.pending,
                                           timelapse=timelapse,
                                           require_all=False)
            for name in state.pending:
                if name not in dirty:
                    continue
                try:
                    cleared += clear_field_rows(db_path, tables, name,
                                                timelapse=timelapse)
                except (ValueError, sqlite3.Error) as exc:
                    raise ResumeRefused(
                        f'Cannot resume: failed to clear existing rows for '
                        f'field {name!r} in {db_path} ({exc}). Re-measuring '
                        f'it now would append a second copy of every object '
                        f'and inflate every downstream per-well count.'
                    ) from exc
    state.cleared_rows = cleared

    if verbose:
        print(format_resume(state))
    return state
