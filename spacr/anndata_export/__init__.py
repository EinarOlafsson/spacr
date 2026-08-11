"""Export a spaCR ``measurements.db`` as AnnData (``.h5ad``).

spaCR already produces the exact shape AnnData was designed for: N objects
x M features, plus per-object metadata and an embedding. Writing that out as
``.h5ad`` puts scanpy, scvi-tools and squidpy within reach of a spaCR user
for the cost of one function call, instead of a bespoke join script per lab.

Nothing here invents an identity, a filter or a feature/metadata boundary.
All four already exist in spaCR and are reused verbatim:

``obs_names``
    :func:`spacr.selection.object_keys` over
    :data:`spacr.selection.OBJECT_KEY_COLUMNS` -- the schema's own row key
    (``plateID``, ``rowID``, ``columnID``, ``fieldID``, ``object_label``),
    joined on :data:`spacr.schema.KEY_SEPARATOR`. A key from a UMAP lasso
    names the same object in an exported ``.h5ad``.
``X`` / ``obs`` split
    :func:`spacr.schema.is_provenance_column`, which is the boundary
    :mod:`spacr.feature_dict` and every model path already use, plus
    :mod:`spacr.agreement`'s knowledge of which columns a *model* wrote and
    which a *human* did.
``var``
    :func:`spacr.feature_dict.parse_column`, resolved against the row's own
    ``measurement_units`` so a 3-D run's ``cell_area`` is documented as the
    volume it is rather than the area it is named after.
Filtering
    :class:`spacr.selection.DataFilter` and :class:`spacr.selection.Selection`
    -- the same objects the GUI's linked views publish, so "export what I am
    looking at" is one call and not a second filter language.

Typical use::

    from spacr.anndata_export import export_anndata
    from spacr.selection import DataFilter, RangeFilter

    result = export_anndata(
        "/data/exp1/measurements/measurements.db",
        "/data/exp1/results/exp1.h5ad",
        data_filter=DataFilter().add(RangeFilter("cell_area", low=200)),
    )
    print(result.describe())

    import scanpy as sc
    adata = sc.read_h5ad("/data/exp1/results/exp1.h5ad")

What happens to NaN
-------------------
**Nothing, by default, and that is a decision rather than an omission.**

AnnData stores NaN happily. Most of what people reach for next does not:
``sc.pp.scale``, ``sc.pp.pca``, ``sc.pp.neighbors`` and essentially all of
scvi-tools either propagate NaN across the whole matrix or raise. So a
silent default matters, and there are only two honest candidates: keep the
NaN and say so loudly, or impute and say so loudly. Quietly filling with
zero is not one of them -- in spaCR a NaN is usually *meaningful*: a
``pathogen_*`` column is NaN for a cell with no pathogen in it, a Zernike
column is NaN when mahotas was not installed, and a correlation column is
NaN when one channel was flat. Zero-filling the first turns "no pathogen"
into "a pathogen of zero size" and puts it into the same distribution as a
measured one.

:data:`NAN_KEEP` (the default) therefore writes the NaN through, and pays
for it with *visibility*:

* ``var['n_missing']`` / ``var['frac_missing']`` -- per feature;
* ``obs['n_missing_features']`` -- per object;
* ``uns['spacr']['nan']`` -- totals, the policy that was applied, the shape
  those totals were counted over (``n_objects_counted`` x
  ``n_features_counted``, i.e. post-filter and pre-policy), and the ten
  worst columns by name;
* a printed warning naming those columns and the scanpy calls that will
  fail on them.

Every count in that record -- and :attr:`ExportResult.n_missing` with it --
is taken on the matrix **as the policy received it**, which for a dropping
policy is bigger than the one written. ``ExportResult.frac_missing`` divides
by that same matrix (:attr:`ExportResult.counted_shape`) rather than by the
written shape, so it stays a fraction.

The alternatives are explicit, and every one of them records what it did:

``NAN_DROP_FEATURES``
    drop any feature column containing a NaN. Cheap and safe; on a real
    database it often removes every ``pathogen_*`` column, which is the
    honest consequence of asking for a complete matrix.
``NAN_DROP_OBJECTS``
    drop any object row containing a NaN. Usually removes almost
    everything, for the same reason; offered because on a
    single-object-type export it is often exactly right.
``NAN_ZERO`` / ``NAN_MEAN``
    impute. Both write a ``layers['missing']`` boolean mask of the same
    shape by default, because an imputed matrix that cannot be told from a
    measured one is a trap, and 1 byte per cell against 4 is a fair price.

**Infinities are treated as missing under every policy.** spaCR produces
+/-inf from ratio features (a denominator of zero), and an inf survives
``dropna`` while destroying any scaling, PCA or distance computed from it.
They are converted to NaN before the policy runs and counted separately in
``uns['spacr']['nan']['n_infinite']`` so the substitution is never silent.

Where the cell -> nucleus -> pathogen links go, and why not ``obsp``
--------------------------------------------------------------------
They go in ``obs`` and ``uns``. **Never ``obsp``.**

``obsp`` is an ``n_obs x n_obs`` matrix: a relation among *the observations
of this AnnData*. The parent links are not that, in either export shape:

* In the joined, cell-anchored export (the default) the nuclei and pathogens
  are not observations at all -- the join in
  :func:`spacr.io._read_and_join_tables` has already averaged them onto
  their parent cell. There is no row for an ``obsp`` entry to point at.
* In a per-table export of ``nucleus``, the *cells* are not observations,
  for the mirror-image reason.
* An ``obsp`` link would only be well defined for one AnnData whose ``obs``
  is the disjoint union of cells, nuclei and pathogens -- and that matrix is
  useless downstream, because ``X`` would be block-structured by
  construction (a nucleus row has no ``cell_area``) and every tool would see
  it as ~60% missing data.

What is recorded instead is exact and survives everything ``obsp`` does not
(subsetting, concatenation, an ``h5ad`` round trip through a tool that
knows nothing about spaCR):

* joined export -- ``obs['count_nucleus']`` / ``obs['count_pathogen']``
  (how many children were averaged into this row) and
  ``var['is_aggregated']`` (which columns are such an average). Without the
  second one it is genuinely easy to report a per-nucleus statistic that is
  in fact a per-cell mean of nuclei.
* per-table export of a child -- ``obs['cell_id']``, the schema's own
  ``parent_column`` from :data:`spacr.schema.OBJECT_TABLE_SCHEMAS`. A plain
  foreign key, which is what ``adata.obs.groupby('cell_id')`` already
  expects.
* both -- ``uns['spacr']['relationships']``, stating the parent table, the
  key it is joined on, whether the child features were aggregated, and (for
  :func:`export_anndata_set`) the sibling ``.h5ad`` file holding the parent.

The labels the table join drops
-------------------------------
:func:`spacr.io._read_and_join_tables` takes six named columns off
``png_list`` -- the object id, ``png_path`` and the four field keys -- and
drops the rest. The rest is every annotation column the Annotate app added
and every score the classifier wrote, which for this export are among the
most valuable columns in the database: without them ``obs`` has no label to
train on, group by, or colour a UMAP with. They are re-attached by object
key (:func:`_attach_png_labels`) *before* filtering, so a filter can name
one -- "export the cells I annotated as infected" -- and they are separated
into ``uns['spacr']['annotation_columns']`` and
``uns['spacr']['prediction_columns']`` using :mod:`spacr.agreement`'s own
rule, which exists because a model's class column is indistinguishable *by
shape* from an annotation pass.

They land in ``obs``, never in ``X``: a label is not a measurement, and
scaling, PCA and clustering it alongside the features is how a classifier's
own output ends up as a "feature" that predicts it.

Provenance
----------
``uns['spacr']`` carries the spaCR version, the settings hash
(:func:`spacr.artifacts.settings_hash`), the run id, the absolute source
database path, the tables read, the filter that was applied and the counts
before and after it. The written file is also registered with
:mod:`spacr.artifacts` under kind :data:`ANNDATA_KIND`, with the project's
``measurements-db`` artifact as its input -- so ``spacr.artifacts.is_stale``
reports the export as stale the moment Measure is re-run, which is the
whole reason the registry exists.

The optional dependency
-----------------------
``anndata`` is an optional extra: ``pip install "spacr[anndata]"``. It is
imported inside the functions that need it, never at module scope, so
``import spacr.anndata_export`` works without it and the failure is one
actionable sentence (:exc:`AnnDataExtraMissing`) rather than a traceback
six frames deep.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass, replace as _dataclass_replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .. import schema
from ..selection import (OBJECT_KEY_COLUMNS, OBJECT_TYPE_COLUMN, DataFilter,
                         FilterError, Selection, object_keys,
                         untyped_object_key, with_object_type)

__all__ = [
    "ANNDATA_EXTRA",
    "ANNDATA_KIND",
    "ANNDATA_MISSING_MESSAGE",
    "APP_KEY",
    "AnnDataExtraMissing",
    "CONDITION_FALLBACK",
    "DEFAULT_CONDITION_MAP",
    "DEFAULT_TABLES",
    "DuplicateObjectKeys",
    "ExportResult",
    "NAN_DROP_FEATURES",
    "NAN_DROP_OBJECTS",
    "NAN_KEEP",
    "NAN_MEAN",
    "NAN_POLICIES",
    "NAN_ZERO",
    "anndata_export_settings",
    "build_anndata",
    "default_out_path",
    "export_anndata",
    "export_anndata_set",
    "feature_columns",
    "register_anndata_app",
    "register_anndata_settings",
    "require_anndata",
    "resolve_db_path",
    "run_anndata_export",
]


# ---------------------------------------------------------------------------
# The optional dependency
# ---------------------------------------------------------------------------

#: The ``setup.py`` extra that provides :mod:`anndata`.
ANNDATA_EXTRA = "anndata"

#: What the user is told when the extra is not installed. One sentence of
#: diagnosis and one command, following :data:`spacr.qt._QT_MISSING_MESSAGE`.
ANNDATA_MISSING_MESSAGE = """\
Exporting to AnnData (.h5ad) needs the optional `anndata` extra, which is
not installed in this environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[anndata]"

scanpy is a separate install and is NOT required to write the file:

    python -m pip install scanpy\
"""


class AnnDataExtraMissing(ImportError):
    """:mod:`anndata` is not installed.

    An :class:`ImportError` subclass so a caller that already guards the
    export with ``except ImportError`` keeps working, and so the message --
    not a traceback through ``anndata``'s own import machinery -- is what
    reaches the user.
    """


class DuplicateObjectKeys(ValueError):
    """Two rows claim the same object key.

    AnnData requires unique ``obs_names``, and spaCR's writers append: two
    rows *can* share all five key columns when a field was measured twice
    (see ``tests/test_db_contract.py``, which measures exactly that). This
    is raised rather than repaired, because deduplicating -- keeping the
    first, keeping the last, or averaging -- changes the numbers and is the
    caller's decision, not the exporter's.
    """


def require_anndata():
    """Import and return :mod:`anndata`, or raise a message worth reading.

    :returns: the imported :mod:`anndata` module.
    :raises AnnDataExtraMissing: when the extra is not installed.
    """
    try:
        import anndata
    except ImportError as exc:                        # pragma: no cover - env
        module = (getattr(exc, "name", None) or "anndata").split(".", 1)[0]
        raise AnnDataExtraMissing(
            ANNDATA_MISSING_MESSAGE.format(module=module)) from exc
    return anndata


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The artifact kind an exported ``.h5ad`` registers under.
#: :data:`spacr.ports.ALL_KINDS` is documented as the built-in vocabulary
#: rather than a closed set, so a new kind is a declaration, not a violation.
ANNDATA_KIND = "anndata"

#: Settings / app key, shared by :func:`register_anndata_settings` and
#: :func:`register_anndata_app` so the panel and the sidebar row agree.
APP_KEY = "anndata_export"

#: Tables read for the default, cell-anchored export. Exactly the list
#: :func:`spacr.core.generate_image_umap` reads, so an export and an
#: embedding describe the same population.
DEFAULT_TABLES: Tuple[str, ...] = (
    "cell", "cytoplasm", "nucleus", "pathogen", "png_list")

#: Keep NaN in ``X`` and report it. The default; see the module docstring.
NAN_KEEP = "keep"
#: Drop every feature column containing a NaN.
NAN_DROP_FEATURES = "drop_features"
#: Drop every object row containing a NaN.
NAN_DROP_OBJECTS = "drop_objects"
#: Replace NaN with ``0.0``, recording a ``layers['missing']`` mask.
NAN_ZERO = "zero"
#: Replace NaN with the feature's mean, recording a ``layers['missing']`` mask.
NAN_MEAN = "mean"

#: Every accepted ``nan_policy``.
NAN_POLICIES: Tuple[str, ...] = (
    NAN_KEEP, NAN_DROP_FEATURES, NAN_DROP_OBJECTS, NAN_ZERO, NAN_MEAN)

#: The column -> condition mapping :func:`spacr.utils.map_condition` applies
#: by default. Spelled out here rather than imported: ``spacr.utils`` pulls
#: torch and cellpose, which is a 10-second import for a four-entry dict, and
#: this module is deliberately importable on a machine that cannot segment.
#: ``tests/test_anndata_export.py`` pins the two definitions together.
DEFAULT_CONDITION_MAP: Dict[str, str] = {
    "c1": "neg", "c2": "pos", "c3": "mix"}

#: What a column not named in the condition map maps to -- the same fallback
#: :func:`spacr.utils.map_condition` uses.
CONDITION_FALLBACK = "screen"

#: Columns that are numeric and are nonetheless not measurements: the
#: cluster label :func:`spacr.core.generate_image_umap` writes back, and the
#: aggregation counts :func:`spacr.io._read_and_join_tables` adds. These land
#: in ``obs``. ``count_*`` is already caught by
#: :func:`spacr.schema.is_provenance_column`; ``cluster`` is not.
_NON_FEATURE_NUMERIC = frozenset({"cluster"})

#: Prefixes of columns this module adds to ``obs`` itself, reserved so a
#: database column of the same name cannot silently overwrite one.
_RESERVED_OBS = ("n_missing_features",)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportResult:
    """What one export produced, and what it left out.

    :param path: the ``.h5ad`` written, or ``""`` for an in-memory build.
    :param n_obs: objects in the exported matrix.
    :param n_vars: feature columns in ``X``.
    :param n_obs_before_filter: objects read from the database.
    :param obs_columns: the ``obs`` column names.
    :param obsm_keys: the ``obsm`` keys written.
    :param nan_policy: the policy that was applied.
    :param n_missing: NaN cells in ``X`` *before* the policy ran.
    :param n_infinite: non-finite cells converted to NaN before that count.
    :param dropped_features: feature columns removed by the policy.
    :param dropped_objects: objects removed by the policy.
    :param n_obs_counted: rows of the matrix ``n_missing`` was counted in --
        i.e. after ``data_filter``/``selection``/``row_limit`` and *before*
        the ``nan_policy`` ran. ``0`` on a record that did not record it, in
        which case :attr:`counted_shape` reconstructs it from the drops.
    :param n_vars_counted: columns of that same pre-policy matrix.
    :param artifact_id: the :mod:`spacr.artifacts` id, or ``""`` when the
        file was not registered.
    :param warnings: everything the export decided the user must know.

    Three object counts live here and they are three different stages, which
    is the whole reason they are named apart: :attr:`n_obs_before_filter` is
    what the database held, :attr:`counted_shape` is what survived the filter
    and was handed to the NaN policy, and :attr:`n_obs` is what was written.
    :meth:`describe` attributes each loss to the stage that caused it rather
    than charging all of them to the filter.
    """

    path: str
    n_obs: int
    n_vars: int
    n_obs_before_filter: int
    obs_columns: Tuple[str, ...] = ()
    obsm_keys: Tuple[str, ...] = ()
    nan_policy: str = NAN_KEEP
    n_missing: int = 0
    n_infinite: int = 0
    dropped_features: Tuple[str, ...] = ()
    dropped_objects: int = 0
    n_obs_counted: int = 0
    n_vars_counted: int = 0
    artifact_id: str = ""
    warnings: Tuple[str, ...] = ()

    @property
    def counted_shape(self) -> Tuple[int, int]:
        """``(rows, columns)`` of the matrix :attr:`n_missing` was counted in.

        That matrix is the post-filter, **pre**-``nan_policy`` one: the NaN
        were counted before anything was dropped or imputed. A record built
        by this module states it outright; one built by hand (or by an older
        spaCR) leaves the two fields at ``0``, and the shape is reconstructed
        from the drops instead -- ``drop_objects`` is the only policy that
        removes rows and ``drop_features`` the only one that removes columns,
        so adding them back recovers what the policy was given.
        """
        rows = self.n_obs_counted or (self.n_obs + self.dropped_objects)
        columns = (self.n_vars_counted
                   or (self.n_vars + len(self.dropped_features)))
        return int(rows), int(columns)

    @property
    def frac_missing(self) -> float:
        """:attr:`n_missing` over the cells of :attr:`counted_shape`.

        Numerator and denominator describe **the same matrix** -- the one the
        NaN were counted in, before the policy dropped or imputed anything --
        so this is a fraction and can never exceed 1.0. Dividing the
        pre-policy count by the post-policy ``n_obs * n_vars`` is what made
        ``drop_objects`` report 114.3% missing.

        :returns: the fraction, or ``0.0`` for an empty matrix.
        """
        rows, columns = self.counted_shape
        cells = rows * columns
        return (self.n_missing / cells) if cells else 0.0

    def describe(self) -> str:
        """One human paragraph: shape, filtering, missingness, provenance.

        Each count is charged to the stage that caused it: the filter line
        counts only what the filter removed, and the ``dropped ...`` lines
        only what the NaN policy removed. Charging the policy's row drops to
        the filter as well made one loss of four objects read as eight.
        """
        counted_obs, counted_vars = self.counted_shape
        lines = [
            f"{self.n_obs} objects x {self.n_vars} features"
            + (f" -> {self.path}" if self.path else " (in memory)")
        ]
        removed_by_filter = self.n_obs_before_filter - counted_obs
        if removed_by_filter > 0:
            lines.append(
                f"  filtered from {self.n_obs_before_filter} objects "
                f"({removed_by_filter} removed)")
        if self.n_missing:
            over = ("" if (counted_obs, counted_vars) == (self.n_obs,
                                                          self.n_vars)
                    else f" of the {counted_obs} x {counted_vars} matrix "
                         f"the policy was given")
            lines.append(
                f"  {self.n_missing} missing values "
                f"({self.frac_missing:.1%}{over}), "
                f"policy {self.nan_policy!r}")
        if self.n_infinite:
            lines.append(
                f"  {self.n_infinite} non-finite values treated as missing")
        if self.dropped_features:
            shown = ", ".join(self.dropped_features[:5])
            more = (f" +{len(self.dropped_features) - 5}"
                    if len(self.dropped_features) > 5 else "")
            lines.append(
                f"  dropped features (nan_policy {self.nan_policy!r}): "
                f"{shown}{more}")
        if self.dropped_objects:
            lines.append(
                f"  dropped objects (nan_policy {self.nan_policy!r}): "
                f"{self.dropped_objects}")
        if self.obsm_keys:
            lines.append(f"  obsm: {', '.join(self.obsm_keys)}")
        if self.artifact_id:
            lines.append(f"  artifact {self.artifact_id}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _available_tables(db_path: Union[str, os.PathLike]) -> Tuple[str, ...]:
    """Return the tables present in ``db_path``, in ``sqlite_master`` order.

    :param db_path: a ``measurements.db``.
    :returns: table names; empty when the file does not exist.
    """
    if not os.path.isfile(os.fspath(db_path)):
        return ()
    connection = sqlite3.connect(os.fspath(db_path))
    try:
        return tuple(
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"))
    finally:
        connection.close()


def _read_frame(db_path: str, tables: Sequence[str],
                single_table: Optional[str]) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Read the object frame this export will describe.

    :param db_path: a ``measurements.db``.
    :param tables: tables to join, for the default cell-anchored export.
    :param single_table: read exactly this one object table instead, one row
        per object of that type.
    :returns: ``(frame, tables_actually_read)``.
    :raises ValueError: when nothing usable is in the database.
    """
    present = set(_available_tables(db_path))
    if not present:
        raise ValueError(
            f"{db_path} holds no tables; it is not a spaCR measurements "
            f"database. Point at <project>/measurements/measurements.db.")

    if single_table is not None:
        if single_table not in present:
            raise ValueError(
                f"table {single_table!r} is not in {db_path}. Present: "
                f"{sorted(present & set(schema.OWNED_TABLES))}.")
        connection = sqlite3.connect(db_path)
        try:
            frame = pd.read_sql(f'SELECT * FROM "{single_table}"', connection)
        finally:
            connection.close()
        # The frame does not know which table it is; this function does, and
        # `obs_names` is built from it. Without the stamp a per-table export
        # of `nucleus` and one of `pathogen` index the same object two ways
        # when the labels overlap -- and they always overlap, because each
        # mask is labelled from 1 independently.
        return with_object_type(frame, single_table), (single_table,)

    wanted = [t for t in tables if t in present]
    if not wanted:
        raise ValueError(
            f"none of {list(tables)} is in {db_path}. Present: "
            f"{sorted(present & set(schema.OWNED_TABLES))}.")
    if "cell" not in wanted:
        raise ValueError(
            f"the joined export is anchored on the 'cell' table, which "
            f"{db_path} does not have. Pass single_table= to export one of "
            f"{sorted(t for t in wanted if t in schema.OBJECT_TABLES)} "
            f"on its own.")

    # Imported here, not at module scope: spacr.io pulls torch and cellpose,
    # and this module must stay importable on a machine that cannot segment.
    from ..io import _read_and_join_tables

    frame = _read_and_join_tables(db_path, list(wanted))
    if frame is None:
        raise ValueError(
            f"could not join {wanted} from {db_path}; the join returned "
            f"nothing. Run `spacr doctor --db {db_path}` for the diagnosis.")
    # One row per CELL -- the join is anchored there and the children arrive
    # as columns, not rows -- so that is the type of every observation.
    return with_object_type(frame, "cell"), tuple(wanted)


def _attach_png_labels(frame: pd.DataFrame, db_path: str, anchor: str,
                       *, timelapse: bool) -> Tuple[pd.DataFrame, List[str]]:
    """Bring the annotation and prediction columns back out of ``png_list``.

    :func:`spacr.io._read_and_join_tables` takes six named columns off
    ``png_list`` -- the object id, ``png_path`` and the four field keys --
    and drops everything else. Everything else is precisely the annotation
    columns the Annotate app added and the score columns the classifier
    wrote, which for an AnnData export are among the most valuable columns
    in the database: without them ``adata.obs`` has no label to train on,
    group by, or colour a UMAP with, and a user would have to re-join
    ``png_list`` by hand outside spaCR.

    The columns are recognised, not guessed: everything ``png_list`` gets
    from :func:`spacr.utils.filepaths_to_database` is listed in
    :data:`spacr.agreement._METADATA_COLUMNS`, so what is left over is what
    a human or a model put there.

    The join is on the anchor object's own crop id (``cell_id`` for a cell
    export, ``nucleus_id`` for a nucleus one), migrated to an integer label
    with :func:`spacr.utils.object_label_from_png_id` -- the one
    implementation that survives the ``'omulti'`` / ``'onone'`` / ``'error'``
    / NULL values the real writers produce.

    :returns: ``(frame, attached)`` -- the frame and the column names added.
        A ``png_list`` that cannot be attached unambiguously adds nothing
        and is reported, rather than attaching a plausible wrong label.
    """
    from ..agreement import _METADATA_COLUMNS
    from ..utils import PNG_OBJECT_ID_COLUMNS, object_label_from_png_id

    id_column = PNG_OBJECT_ID_COLUMNS.get(anchor)
    if not id_column:
        return frame, []
    connection = sqlite3.connect(db_path)
    try:
        png = pd.read_sql('SELECT * FROM "png_list"', connection)
    except Exception:
        return frame, []
    finally:
        connection.close()

    if id_column not in png.columns:
        return frame, []
    extra = [c for c in png.columns
             if c not in _METADATA_COLUMNS and c not in frame.columns
             and c != id_column]
    if not extra:
        return frame, []

    key_columns = list(OBJECT_KEY_COLUMNS)
    if timelapse:
        key_columns = list(schema.TIMEPOINT_KEY_COLUMNS) + [
            schema.OBJECT_LABEL_KEY]
    if not all(c in png.columns for c in key_columns[:-1]):
        return frame, []

    labels = object_label_from_png_id(png[id_column])
    png = png.loc[labels.notna()].copy()
    if png.empty:
        return frame, []
    png[schema.OBJECT_LABEL_KEY] = labels.loc[png.index].astype("int64")

    try:
        # `id_column` is the anchor's own id column, so these crops are
        # anchor-typed. Both sides of the reindex below have to be keyed the
        # same way or every attached label lands as NaN -- which is silent,
        # and costs exactly the annotation columns this function exists for.
        keys = object_keys(png, timelapse=timelapse, object_type=anchor)
    except FilterError:
        # A timelapse database whose png_list still spells the timepoint
        # `time_id` (see spacr.schema.TIME_COLUMN_ALIASES) cannot be keyed
        # against a `timeID` object table without guessing which frame a
        # crop belongs to. Attaching nothing loses the labels; attaching the
        # wrong frame's label loses the experiment.
        return frame, []
    subset = png[extra].copy()
    subset.index = pd.Index(keys)
    duplicated = subset.index.duplicated(keep=False)
    if duplicated.any():
        conflicting = subset.loc[duplicated]
        agreed = conflicting.groupby(level=0).nunique(dropna=False).le(1).all()
        if not bool(agreed.all()):
            return frame, []
        subset = subset[~subset.index.duplicated(keep="first")]

    frame = frame.copy()
    frame_keys = object_keys(frame, timelapse=timelapse)
    aligned = subset.reindex(pd.Index(frame_keys))
    for column in extra:
        frame[column] = aligned[column].to_numpy()
    return frame, list(extra)


def _measurement_units(frame: pd.DataFrame) -> Optional[str]:
    """The single ``measurement_units`` value of ``frame``, or ``None``.

    ``None`` when the column is absent (a legacy database) or carries more
    than one value (a database measured twice under different calibration).
    A mixed frame deliberately gets ``None`` rather than a majority vote:
    :func:`spacr.feature_dict.parse_column` then states the condition instead
    of asserting a unit that is wrong for some of the rows.
    """
    if "measurement_units" not in frame.columns:
        return None
    values = frame["measurement_units"].dropna().unique()
    if len(values) != 1:
        return None
    return str(values[0])


# ---------------------------------------------------------------------------
# The X / obs boundary
# ---------------------------------------------------------------------------

def _label_columns(frame: pd.DataFrame,
                   db_path: Optional[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split the label-ish columns into ``(annotation, prediction)``.

    Both are numeric and neither is a measurement, so both would otherwise
    land in ``X`` and be scaled, PCA'd and clustered as if they were
    features. :mod:`spacr.agreement` already knows the difference -- it has
    to, because offering a model's column as a third annotator gave a real
    database kappa = -0.004 -- so its answer is reused rather than a second
    list being written here.

    :param frame: the frame being exported.
    :param db_path: the database, when the human-annotation guess (which
        needs the column's distinct-value count) can be made against it.
    :returns: ``(annotation_columns, prediction_columns)``, both in frame
        order and disjoint.
    """
    from ..agreement import PNG_TABLE, _is_model_column, annotation_columns

    columns = list(frame.columns)
    predictions = tuple(c for c in columns if _is_model_column(c, columns))

    annotations: Tuple[str, ...] = ()
    if db_path:
        try:
            guessed = annotation_columns(db_path, table=PNG_TABLE)
        except Exception:
            # A database with no png_list, or an unreadable one. The export
            # is still correct without the annotation hint; losing the whole
            # export over a missing optional table would not be.
            guessed = []
        annotations = tuple(c for c in columns
                            if c in set(guessed) and c not in predictions)
    return annotations, predictions


def feature_columns(frame: pd.DataFrame,
                    *, exclude: Sequence[str] = (),
                    db_path: Optional[str] = None) -> List[str]:
    """The columns of ``frame`` that belong in ``X``, in frame order.

    A column is a feature when it is numeric, is not provenance or identity
    by :func:`spacr.schema.is_provenance_column`, is not a human annotation
    or a model output, is not the cluster label, and was not excluded by the
    caller.

    :param frame: an object frame read from a measurements database.
    :param exclude: extra column names to keep out of ``X``.
    :param db_path: the source database, used to recognise human annotation
        columns; omit it and only the model columns are recognised.
    :returns: feature column names.
    """
    annotations, predictions = _label_columns(frame, db_path)
    blocked = (set(exclude) | set(annotations) | set(predictions)
               | set(_NON_FEATURE_NUMERIC))
    keep: List[str] = []
    for column in frame.columns:
        if column in blocked:
            continue
        if schema.is_provenance_column(column):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        keep.append(str(column))
    return keep


def _source_table(column: str, entry, tables: Sequence[str]) -> str:
    """Which measurement table ``column`` came out of.

    Read off the object type :func:`spacr.feature_dict.parse_column` found,
    falling back to the join suffix (``..._nucleus``) and then to the
    anchor table. ``""`` when it cannot be attributed, which is honest --
    the alternative is a confident wrong answer in ``var``.
    """
    for obj in (entry.object_type_2, entry.object_type):
        if obj and obj in tables:
            return str(obj)
    for table in tables:
        if column.endswith(f"_{table}"):
            return table
    if column.startswith("count_"):
        return column[len("count_"):]
    return tables[0] if tables else ""


def _build_var(features: Sequence[str], frame: pd.DataFrame,
               tables: Sequence[str], anchor: str,
               units: Optional[str]) -> pd.DataFrame:
    """Build the per-feature ``var`` frame.

    Every field comes from :func:`spacr.feature_dict.parse_column`, resolved
    under ``units`` so geometric columns carry a concrete unit rather than a
    conditional one. Nothing is guessed: an unrecognised column arrives with
    ``family='unknown'`` and an empty description, which is the dictionary's
    own answer and is more use than a fabricated one.
    """
    from ..feature_dict import describe_columns

    entries = describe_columns(features, units)
    aggregated_tables = {t for t in tables
                         if t in schema.CHILD_OBJECT_TABLES} if anchor == "cell" else set()

    rows = []
    for column, entry in zip(features, entries):
        source = _source_table(column, entry, tables)
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        rows.append({
            "object_type": entry.object_type or "",
            "object_type_2": entry.object_type_2 or "",
            "channel": (-1 if entry.channel is None else int(entry.channel)),
            "channel_2": (-1 if entry.channel_2 is None else int(entry.channel_2)),
            "channel_scope": entry.channel_scope,
            "family": entry.family,
            "feature_key": entry.key or "",
            "description": entry.description or "",
            "unit": entry.unit or "",
            "measurement_units": entry.measurement_units or (units or ""),
            "computed_by": entry.computed_by,
            "module": entry.module,
            "notes": entry.notes or "",
            "written_when": entry.written_when or "",
            "concepts": ";".join(entry.concepts),
            "source_table": source,
            "is_aggregated": bool(source in aggregated_tables),
            "n_missing": int((~finite).sum()),
            "frac_missing": (float((~finite).sum()) / len(frame)) if len(frame) else 0.0,
            "n_infinite": int(np.isinf(
                values.to_numpy(dtype=float, na_value=np.nan)).sum()),
        })
    var = pd.DataFrame(rows, index=pd.Index(list(features), name=None))
    for categorical in ("object_type", "channel_scope", "family",
                        "source_table", "measurement_units"):
        var[categorical] = var[categorical].astype("category")
    return var


def _redundant_identity_columns(columns: Sequence[str],
                                tables: Sequence[str]) -> List[str]:
    """Join-suffixed copies of identity columns the anchor already carries.

    :func:`spacr.io._read_and_join_tables` suffixes every colliding column,
    so a four-table join hands back ``plateID`` and then ``plateID_nucleus``,
    ``plateID_pathogen`` and ``plateID_cytoplasm`` -- four spellings of one
    plate. Two of them are worse than redundant: ``object_label_nucleus``
    and ``cell_id_pathogen`` come through the child aggregation, so their
    values are the *mean of the child labels*, a number with no referent at
    all.

    Only provenance columns are removed, and only when the unsuffixed
    column is present to stand for them. Feature columns keep their
    suffixes, because ``nucleus_area_nucleus`` is a real measurement.

    :param columns: the candidate ``obs`` columns.
    :param tables: the tables that were joined.
    :returns: the columns to drop, in input order.
    """
    present = set(columns)
    drop: List[str] = []
    for column in columns:
        for table in tables:
            suffix = f"_{table}"
            if not column.endswith(suffix):
                continue
            base = column[:-len(suffix)]
            if (base in present and base != column
                    and schema.is_provenance_column(base)):
                drop.append(column)
            break
    return drop


def _build_obs(frame: pd.DataFrame, features: Sequence[str],
               annotations: Sequence[str], predictions: Sequence[str],
               *, timelapse: bool,
               condition_map: Optional[Mapping[str, str]],
               condition_column: str,
               drop_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Build ``obs``: everything that is not a feature, plus the key columns.

    The index is :func:`spacr.selection.object_keys`, which is spaCR's own
    object identity -- not a new one invented for AnnData.
    """
    obs = frame.drop(columns=[c for c in features if c in frame.columns])
    obs = obs.drop(columns=[c for c in drop_columns if c in obs.columns])
    obs = obs.copy()

    keys = object_keys(frame, timelapse=timelapse)
    duplicated = pd.Index(keys).duplicated()
    if duplicated.any():
        offenders = sorted(set(pd.Index(keys)[duplicated]))
        raise DuplicateObjectKeys(
            f"{int(duplicated.sum())} of {len(keys)} rows repeat an object "
            f"key, and AnnData requires unique obs_names. First few: "
            f"{offenders[:5]}. This means the database holds more than one "
            f"row for the same object -- usually a field measured twice. "
            f"Deduplicate it deliberately (spacr.resume can delete a field's "
            f"rows before a re-measure); this export will not guess which "
            f"row is the right one."
            + ("" if timelapse else
               " If this is a timelapse database, pass timelapse=True so each "
               "frame keys separately."))
    obs.index = pd.Index(keys, name="object_key")

    if condition_map is not None and condition_column in frame.columns:
        mapping = dict(condition_map)
        obs["condition"] = [
            mapping.get(str(value), CONDITION_FALLBACK)
            for value in frame[condition_column]]

    # Low-cardinality text is stored categorical: it is what scanpy's
    # groupby/plotting expects, and on a million-object export it is the
    # difference between a 40 MB obs and a 4 MB one.
    categorical = list(OBJECT_KEY_COLUMNS[:-1]) + [
        schema.PRC_KEY, schema.PRCF_KEY, "condition", "cluster",
        "measurement_units", *annotations, *predictions]
    if timelapse:
        categorical.append(schema.TIME_KEY)
    for column in categorical:
        if column in obs.columns and obs[column].nunique(dropna=False) <= 2048:
            obs[column] = obs[column].astype("category")
    return obs


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def _apply_nan_policy(matrix: np.ndarray, features: List[str],
                      policy: str) -> Tuple[np.ndarray, List[str], np.ndarray,
                                            Optional[np.ndarray], Dict[str, Any]]:
    """Apply ``policy`` to ``matrix``, returning what it did.

    :param matrix: the float feature matrix, non-finite values already
        converted to NaN by the caller.
    :param features: column names of ``matrix``.
    :param policy: one of :data:`NAN_POLICIES`.
    :returns: ``(matrix, features, keep_rows, missing_mask, report)`` where
        ``keep_rows`` is a boolean mask over the original rows and
        ``missing_mask`` is the ``layers['missing']`` array or ``None``.
    :raises ValueError: on an unknown policy.
    """
    if policy not in NAN_POLICIES:
        raise ValueError(
            f"nan_policy={policy!r} is not one of {list(NAN_POLICIES)}. See "
            f"the spacr.anndata_export module docstring for what each does "
            f"and why 'keep' is the default.")

    missing = np.isnan(matrix)
    report: Dict[str, Any] = {
        "policy": policy,
        "n_missing": int(missing.sum()),
        # The shape `n_missing` was counted over. Every count in this report
        # is measured on the matrix as the policy received it, so the shape
        # of that matrix has to be recorded with them: divide `n_missing` by
        # the *written* shape instead and a dropping policy reports more than
        # 100% missing.
        "n_objects_counted": int(matrix.shape[0]),
        "n_features_counted": int(matrix.shape[1]),
        "n_features_with_missing": int((missing.any(axis=0)).sum()),
        "n_objects_with_missing": int((missing.any(axis=1)).sum()),
        "imputed": False,
        "dropped_features": [],
        "dropped_objects": 0,
    }
    keep_rows = np.ones(matrix.shape[0], dtype=bool)
    mask: Optional[np.ndarray] = None

    if policy == NAN_KEEP or not missing.any():
        report["worst_features"] = _worst_features(missing, features)
        return matrix, list(features), keep_rows, mask, report

    if policy == NAN_DROP_FEATURES:
        keep = ~missing.any(axis=0)
        report["dropped_features"] = [f for f, k in zip(features, keep) if not k]
        report["worst_features"] = _worst_features(missing, features)
        return (matrix[:, keep], [f for f, k in zip(features, keep) if k],
                keep_rows, mask, report)

    if policy == NAN_DROP_OBJECTS:
        keep_rows = ~missing.any(axis=1)
        report["dropped_objects"] = int((~keep_rows).sum())
        report["worst_features"] = _worst_features(missing, features)
        return matrix[keep_rows], list(features), keep_rows, mask, report

    # The two imputing policies. The mask is what keeps an imputed matrix
    # distinguishable from a measured one.
    mask = missing
    report["imputed"] = True
    if policy == NAN_ZERO:
        matrix = np.where(missing, 0.0, matrix)
    else:                                              # NAN_MEAN
        with warnings.catch_warnings():
            # An all-NaN column has no mean; numpy says so and returns NaN,
            # which is then filled with 0.0 below. The warning is expected
            # here and would be noise on every export of a sparse feature.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(matrix, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        matrix = np.where(missing, means[None, :], matrix)
    report["worst_features"] = _worst_features(missing, features)
    return matrix, list(features), keep_rows, mask, report


def _worst_features(missing: np.ndarray, features: Sequence[str],
                    limit: int = 10) -> List[List[Any]]:
    """The ``limit`` features with the most missing values, worst first.

    Returned as ``[[name, count], ...]`` rather than a dict because ``uns``
    is written to HDF5 and a list of pairs round-trips through every
    ``h5ad`` reader, while a dict of arbitrary column names does not.
    """
    if not len(features) or not missing.any():
        return []
    counts = missing.sum(axis=0)
    order = np.argsort(-counts)[:limit]
    return [[str(features[i]), int(counts[i])] for i in order
            if counts[i] > 0]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _obsm_name(name: str) -> str:
    """Normalise an embedding name to the scanpy ``X_*`` convention."""
    text = str(name)
    return text if text.startswith("X_") else f"X_{text}"


def _align_embedding(values: Any, keys: pd.Index,
                     *, timelapse: bool, name: str) -> np.ndarray:
    """Coerce one embedding to an ``(n_obs, k)`` array aligned to ``keys``.

    A :class:`pandas.DataFrame` carrying the object key columns is aligned
    **by key**; anything else is taken positionally against the *unfiltered*
    frame and then narrowed. Aligning a keyed frame positionally is the bug
    this exists to prevent: the whole point of the filtered export is that
    the exported rows are a subset, and a positional take would then attach
    the wrong point to every object after the first gap.

    :raises ValueError: on a length or key mismatch.
    """
    if isinstance(values, pd.DataFrame) and all(
            c in values.columns for c in OBJECT_KEY_COLUMNS):
        coordinate_columns = [c for c in values.columns
                              if c not in set(OBJECT_KEY_COLUMNS)
                              and pd.api.types.is_numeric_dtype(values[c])
                              and c != schema.OBJECT_LABEL_KEY]
        if not coordinate_columns:
            raise ValueError(
                f"embedding {name!r} carries the object key columns but no "
                f"numeric coordinate columns to go with them.")
        indexed = values.set_index(
            object_keys(values, timelapse=timelapse))[coordinate_columns]
        # An embedding computed before object types existed, or by a caller
        # that did not state one, keys its rows untyped. It still names the
        # same objects, so it is resolved by dropping the type rather than
        # reported as a population mismatch.
        wanted = [k if k in indexed.index else untyped_object_key(k)
                  for k in keys]
        missing = [k for k in wanted if k not in indexed.index]
        if missing:
            raise ValueError(
                f"embedding {name!r} has no coordinates for "
                f"{len(missing)} of the {len(keys)} exported objects "
                f"(first: {missing[:3]}). It was computed on a different "
                f"population; recompute it on the filtered frame or pass "
                f"the same filter to both.")
        return np.asarray(indexed.loc[wanted].to_numpy(), dtype=np.float32)

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] == len(keys):
        return array
    raise ValueError(
        f"embedding {name!r} has {array.shape[0]} rows but the export has "
        f"{len(keys)} objects. Pass a DataFrame carrying "
        f"{list(OBJECT_KEY_COLUMNS)} to have it aligned by object key "
        f"instead of by position.")


def _compute_umap(matrix: np.ndarray, features: Sequence[str],
                  settings: Mapping[str, Any]) -> np.ndarray:
    """Compute a 2-D UMAP through the same path the UMAP app uses.

    :func:`spacr.utils.reduction_and_clustering` is the function
    :func:`spacr.core.generate_image_umap` calls, with the seed it uses, so
    an exported ``X_umap`` and the app's plot are the same embedding rather
    than two that merely look alike.

    NaN is imputed to the feature mean *for the reducer only* -- UMAP cannot
    consume NaN and would otherwise return an all-NaN embedding -- and ``X``
    itself is untouched. The substitution is recorded in
    ``uns['spacr']['umap']``.
    """
    from ..utils import reduction_and_clustering

    values = np.array(matrix, dtype=float, copy=True)
    finite = np.isfinite(values)
    if not finite.all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(np.where(finite, values, np.nan), axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        values = np.where(finite, values, means[None, :])
    embedding, _labels, _reducer = reduction_and_clustering(
        values,
        n_neighbors=settings.get("n_neighbors", 15),
        min_dist=settings.get("min_dist", 0.1),
        metric=settings.get("metric", "euclidean"),
        eps=settings.get("eps", 0.5),
        min_samples=settings.get("min_samples", 5),
        clustering=settings.get("clustering", "dbscan"),
        reduction_method=settings.get("reduction_method", "umap"),
        verbose=False,
        n_jobs=settings.get("n_jobs", 1))
    return np.asarray(embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# uns
# ---------------------------------------------------------------------------

def _h5ad_safe(value: Any) -> Any:
    """Coerce ``value`` into something ``h5ad`` can store.

    ``None`` becomes ``""``, tuples become lists, everything unrecognised
    becomes its ``repr``. AnnData's HDF5 writer raises on a ``None`` or an
    arbitrary object buried three dicts deep, and losing a finished export
    to a provenance field would be absurd.
    """
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _h5ad_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_h5ad_safe(v) for v in value]
    return repr(value)


def _filter_record(data_filter: Optional[DataFilter],
                   selection: Optional[Selection]) -> Dict[str, Any]:
    """Describe the filtering applied, in prose and in structure."""
    record: Dict[str, Any] = {
        "description": data_filter.describe() if data_filter else "no filter",
        "clauses": [],
        "selection_source": "",
        "selection_size": 0,
    }
    for clause in (data_filter.clauses if data_filter else []):
        entry = {"column": clause.column, "type": type(clause).__name__}
        if hasattr(clause, "low"):
            entry["low"] = "" if clause.low is None else float(clause.low)
            entry["high"] = "" if clause.high is None else float(clause.high)
        else:
            entry["values"] = [str(v) for v in clause.values]
        record["clauses"].append(entry)
    if selection is not None and selection.is_active:
        record["selection_source"] = selection.source
        record["selection_size"] = len(selection)
    return record


def _relationships(tables: Sequence[str], anchor: str,
                   joined: bool) -> Dict[str, Any]:
    """Describe the cell -> nucleus / pathogen links this export carries.

    See the module docstring for why this is ``uns`` and ``obs`` rather than
    ``obsp``.
    """
    record: Dict[str, Any] = {
        "storage": "obs and uns, never obsp",
        "why_not_obsp": (
            "obsp is an n_obs x n_obs relation among THIS AnnData's "
            "observations. In a cell-anchored export the nuclei and "
            "pathogens have been aggregated onto their parent and are not "
            "observations at all, so there is no row to point at; in a "
            "child-anchored export the cells are not observations, for the "
            "mirror-image reason. The link is a foreign key, and it is "
            "stored as one."),
        "anchor": anchor,
        "children": {},
    }
    if joined:
        for table in tables:
            if table in schema.CHILD_OBJECT_TABLES and table != anchor:
                record["children"][table] = {
                    "aggregated": True,
                    "aggregation": "mean over the children of each parent",
                    "count_column": f"count_{table}",
                    "column_suffix": f"_{table}",
                    "see": "var['is_aggregated']",
                }
    else:
        contract = schema.OBJECT_TABLE_SCHEMAS.get(anchor)
        parent = getattr(contract, "parent_column", None) if contract else None
        if parent:
            record["parent"] = {
                "table": "cell",
                "obs_column": parent,
                "aggregated": False,
                "note": ("a plain foreign key: adata.obs.groupby('cell_id') "
                         "gives this export's objects per parent cell."),
            }
    return record


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def build_anndata(db_path: Union[str, os.PathLike],
                  *,
                  tables: Sequence[str] = DEFAULT_TABLES,
                  single_table: Optional[str] = None,
                  data_filter: Optional[DataFilter] = None,
                  selection: Optional[Selection] = None,
                  row_limit: Optional[int] = None,
                  timelapse: bool = False,
                  nan_policy: str = NAN_KEEP,
                  missing_layer: Optional[bool] = None,
                  dtype: str = "float32",
                  exclude: Sequence[str] = (),
                  embeddings: Optional[Mapping[str, Any]] = None,
                  compute_umap: bool = False,
                  umap_settings: Optional[Mapping[str, Any]] = None,
                  condition_map: Optional[Mapping[str, str]] = None,
                  condition_column: str = schema.COLUMN_KEY,
                  attach_labels: bool = True,
                  drop_redundant_identity: bool = True,
                  settings: Optional[Mapping[str, Any]] = None,
                  run_id: str = "",
                  verbose: bool = True):
    """Build an :class:`anndata.AnnData` from a spaCR measurements database.

    The mapping is described in full in the module docstring; in short,
    ``X`` is the numeric measurements, ``obs`` is everything else about the
    object, ``var`` is :mod:`spacr.feature_dict`'s description of each
    feature, ``obsm`` holds embeddings and ``uns`` holds provenance.

    :param db_path: a ``measurements.db``.
    :param tables: tables to join for the default cell-anchored export.
    :param single_table: export exactly this object table instead, one row
        per object of that type -- which is the only way to get a
        nucleus-level or pathogen-level matrix, since the join averages
        children onto their parent.
    :param data_filter: a :class:`spacr.selection.DataFilter`. Declarative
        and re-appliable; recorded in ``uns``.
    :param selection: a :class:`spacr.selection.Selection` -- the keys a
        view pointed at. Applied after ``data_filter``.
    :param row_limit: hard cap on exported objects, applied last. A blunt
        instrument on purpose, for "give me something I can open" without
        inventing a filter that means something it does not.
    :param timelapse: key each frame of an object separately.
    :param nan_policy: one of :data:`NAN_POLICIES`; see the module
        docstring. The default keeps NaN and reports it.
    :param missing_layer: write ``layers['missing']``. Defaults to True for
        the imputing policies and False otherwise.
    :param dtype: ``X`` dtype. ``float32`` by default -- the scanpy
        convention, half the memory, and far more precision than any
        microscope measurement carries.
    :param exclude: feature columns to keep out of ``X``.
    :param embeddings: ``{name: array or DataFrame}``. Names are normalised
        to the scanpy ``X_*`` convention, so ``'umap'`` becomes ``X_umap``.
        A DataFrame carrying :data:`spacr.selection.OBJECT_KEY_COLUMNS` is
        aligned **by key**, which is what makes an embedding computed on the
        whole plate usable with a filtered export.
    :param compute_umap: compute ``X_umap`` here, through the same
        :func:`spacr.utils.reduction_and_clustering` call
        :func:`spacr.core.generate_image_umap` makes. Off by default: it
        imports the segmentation stack and costs minutes on a large table.
    :param umap_settings: overrides for that computation.
    :param condition_map: ``{column value: label}`` written to
        ``obs['condition']``; :data:`DEFAULT_CONDITION_MAP` is the mapping
        :func:`spacr.utils.map_condition` applies.
    :param condition_column: which column ``condition_map`` reads.
    :param attach_labels: bring the annotation and prediction columns back
        out of ``png_list``, which the table join drops. On by default --
        see :func:`_attach_png_labels`.
    :param drop_redundant_identity: drop the join's suffixed copies of
        identity columns (``plateID_nucleus``, ``object_label_pathogen``)
        from ``obs``. On by default; see
        :func:`_redundant_identity_columns` for why two of them are worse
        than merely duplicated.
    :param settings: the run settings, hashed into ``uns`` provenance.
    :param run_id: the run this export belongs to.
    :param verbose: print the summary and the missing-data warning.
    :returns: ``(adata, result)`` -- the AnnData and an :class:`ExportResult`.
    :raises AnnDataExtraMissing: when ``anndata`` is not installed.
    :raises DuplicateObjectKeys: when two rows claim one object key.
    :raises ValueError: on an unknown ``nan_policy``, an unusable database,
        or an embedding that cannot be aligned.
    """
    anndata = require_anndata()
    db_path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))

    frame, read_tables = _read_frame(db_path, tables, single_table)
    n_before = len(frame)
    joined = single_table is None
    anchor = "cell" if joined else str(single_table)
    notes: List[str] = []

    # Before filtering, so a filter may name an annotation column: "export
    # the cells I called infected" is one of the two things this feature is
    # for, and it cannot work if the label arrives after the mask.
    if attach_labels and "png_list" in _available_tables(db_path):
        frame, attached = _attach_png_labels(
            frame, db_path, anchor, timelapse=timelapse)
        if attached:
            notes.append(
                f"attached {len(attached)} label column(s) from png_list "
                f"that the table join drops: {', '.join(attached[:6])}"
                + (" ..." if len(attached) > 6 else ""))

    if data_filter is not None and not data_filter.is_empty:
        frame = frame.loc[data_filter.mask(frame)]
    if selection is not None and selection.is_active:
        frame = frame.loc[selection.mask_for(frame, timelapse=timelapse)]
    if row_limit is not None and len(frame) > int(row_limit):
        notes.append(
            f"row_limit={int(row_limit)} truncated the export from "
            f"{len(frame)} objects; it is a cap, not a filter, so the "
            f"exported objects are simply the first {int(row_limit)} in "
            f"table order.")
        frame = frame.iloc[:int(row_limit)]
    frame = frame.reset_index(drop=True)

    units = _measurement_units(frame)
    annotations, predictions = _label_columns(frame, db_path)
    features = feature_columns(frame, exclude=exclude, db_path=db_path)
    if not features:
        raise ValueError(
            f"no feature columns found in {db_path}: every numeric column "
            f"was identity, provenance, an annotation or a model output. "
            f"This is usually a database whose object tables were never "
            f"written -- check `spacr doctor --db {db_path}`.")

    redundant: List[str] = []
    if joined and drop_redundant_identity:
        non_features = [c for c in frame.columns if c not in set(features)]
        redundant = _redundant_identity_columns(non_features, read_tables)
        if redundant:
            notes.append(
                f"{len(redundant)} join-suffixed copies of identity columns "
                f"were dropped from obs (e.g. "
                f"{', '.join(redundant[:3])}); the anchor table's own "
                f"columns carry the same values, and the ones that came "
                f"through the child aggregation carried the mean of a label "
                f"rather than a label. Pass "
                f"drop_redundant_identity=False to keep them.")

    obs = _build_obs(frame, features, annotations, predictions,
                     timelapse=timelapse, condition_map=condition_map,
                     condition_column=condition_column,
                     drop_columns=redundant)
    var = _build_var(features, frame, read_tables, anchor, units)

    matrix = frame[features].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    infinite = np.isinf(matrix)
    n_infinite = int(infinite.sum())
    if n_infinite:
        matrix = np.where(infinite, np.nan, matrix)
        notes.append(
            f"{n_infinite} non-finite (+/-inf) values were converted to NaN "
            f"before the nan_policy ran: an inf survives dropna() and then "
            f"destroys any scaling, PCA or distance computed from it.")

    matrix, features, keep_rows, missing_mask, nan_report = _apply_nan_policy(
        matrix, features, nan_policy)
    if not keep_rows.all():
        obs = obs.loc[keep_rows]
        frame = frame.loc[keep_rows].reset_index(drop=True)
    if nan_report["dropped_features"]:
        var = var.loc[features]
    # `var` was built from the frame, i.e. before the policy ran. Keep those
    # counts under `n_missing_raw` -- for an imputing policy they are the
    # only remaining record that a value was invented -- and let `n_missing`
    # describe the matrix that was actually written, which is what a reader
    # inspecting `var` is asking about.
    var = var.copy()
    var["n_missing_raw"] = var["n_missing"].to_numpy()
    var["frac_missing_raw"] = var["frac_missing"].to_numpy()
    final_missing = np.isnan(matrix).sum(axis=0)
    var["n_missing"] = final_missing.astype(np.int64)
    var["frac_missing"] = (final_missing / matrix.shape[0]
                           if matrix.shape[0] else
                           np.zeros(matrix.shape[1]))

    n_missing_before = int(nan_report["n_missing"])
    per_object_missing = (
        np.isnan(matrix).sum(axis=1) if nan_policy in (NAN_KEEP,
                                                       NAN_DROP_FEATURES,
                                                       NAN_DROP_OBJECTS)
        else (missing_mask.sum(axis=1) if missing_mask is not None
              else np.zeros(matrix.shape[0], dtype=int)))
    obs = obs.copy()
    obs["n_missing_features"] = np.asarray(per_object_missing, dtype=np.int32)

    matrix = np.ascontiguousarray(matrix, dtype=np.dtype(dtype))

    adata = anndata.AnnData(X=matrix, obs=obs, var=var)

    if missing_layer is None:
        missing_layer = bool(nan_report["imputed"])
    if missing_layer and missing_mask is not None:
        adata.layers["missing"] = missing_mask
    elif missing_layer:
        adata.layers["missing"] = np.isnan(np.asarray(matrix, dtype=float))

    # ---- obsm -----------------------------------------------------------
    keys = pd.Index(adata.obs_names)
    obsm_notes: Dict[str, Any] = {}
    for name, values in dict(embeddings or {}).items():
        adata.obsm[_obsm_name(name)] = _align_embedding(
            values, keys, timelapse=timelapse, name=str(name))
    if compute_umap and "X_umap" not in adata.obsm:
        if len(adata) < 3:
            notes.append(
                f"compute_umap was asked for but the export has "
                f"{len(adata)} objects; UMAP needs at least 3. No X_umap "
                f"was written.")
        else:
            adata.obsm["X_umap"] = _compute_umap(
                np.asarray(adata.X, dtype=float), features,
                dict(umap_settings or {}))
            obsm_notes["X_umap"] = {
                "computed_by": "spacr.utils.reduction_and_clustering",
                "nan_handling": ("feature-mean imputed for the reducer only; "
                                 "X itself is untouched"),
            }

    # ---- uns ------------------------------------------------------------
    from ..artifacts import material_settings, settings_hash
    from ..version import get_version

    provenance: Dict[str, Any] = {
        "spacr_version": get_version(),
        "settings_hash": settings_hash(settings),
        "run_id": str(run_id or _run_id_from_db(db_path)),
        "source_database": db_path,
        "source_tables": list(read_tables),
        "anchor_object": anchor,
        "joined": bool(joined),
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "object_key_columns": list(OBJECT_KEY_COLUMNS),
        "timelapse": bool(timelapse),
        "measurement_units": units or "",
        "n_objects": int(adata.n_obs),
        "n_objects_before_filter": int(n_before),
        "n_features": int(adata.n_vars),
        "filter": _filter_record(data_filter, selection),
        "nan": nan_report,
        "annotation_columns": list(annotations),
        "prediction_columns": list(predictions),
        "relationships": _relationships(read_tables, anchor, joined),
        "notes": list(notes),
        # NOT an artifact id. The id is a hash *of this file's bytes* (see
        # spacr.artifacts._artifact_id), so writing it inside the file would
        # change the bytes it was computed from. What is stored instead is
        # everything needed to look the record up, which is what somebody
        # holding an orphaned .h5ad actually needs.
        "artifact": {
            "module": APP_KEY,
            "kind": ANNDATA_KIND,
            "role": "h5ad",
            "note": ("the artifact id is derived from this file's content "
                     "fingerprint and so cannot live inside it; find the "
                     "record with spacr.artifacts.by_kind('anndata', "
                     "project=<project root>)"),
        },
    }
    if obsm_notes:
        provenance["umap"] = obsm_notes.get("X_umap", {})

    from ..feature_dict import coverage as feature_coverage

    explained = feature_coverage(features, units)
    provenance["feature_dictionary"] = {
        "total": int(explained.total),
        "explained": int(explained.explained),
        "unknown": list(explained.unknown[:50]),
    }

    adata.uns["spacr"] = _h5ad_safe(provenance)
    adata.uns["spacr_settings"] = _h5ad_safe(material_settings(settings))

    result = ExportResult(
        path="",
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        n_obs_before_filter=int(n_before),
        obs_columns=tuple(str(c) for c in adata.obs.columns),
        obsm_keys=tuple(str(k) for k in adata.obsm.keys()),
        nan_policy=nan_policy,
        n_missing=n_missing_before,
        n_infinite=n_infinite,
        dropped_features=tuple(nan_report["dropped_features"]),
        dropped_objects=int(nan_report["dropped_objects"]),
        n_obs_counted=int(nan_report["n_objects_counted"]),
        n_vars_counted=int(nan_report["n_features_counted"]),
        warnings=tuple(notes),
    )
    if verbose:
        print(result.describe())
        _warn_about_missing(nan_report, nan_policy)
        for note in notes:
            print(f"  note: {note}")
    return adata, result


def _warn_about_missing(report: Mapping[str, Any], policy: str) -> None:
    """Print the missing-data warning, naming the columns and the risk."""
    if policy != NAN_KEEP or not report.get("n_missing"):
        return
    worst = report.get("worst_features") or []
    names = ", ".join(f"{name} ({count})" for name, count in worst[:5])
    print(
        f"  {report['n_missing']} missing values kept in X across "
        f"{report['n_features_with_missing']} features. AnnData stores "
        f"them; sc.pp.scale, sc.pp.pca and sc.pp.neighbors do not -- they "
        f"will propagate NaN across the whole matrix. Worst: {names}. "
        f"Pass nan_policy='drop_features' or 'mean' if you need a complete "
        f"matrix; see var['n_missing'] for the full picture.")


def _run_id_from_db(db_path: str) -> str:
    """The most recent run id recorded in ``db_path``, or ``""``.

    Read from ``settings_history`` through :func:`spacr.io.read_settings_history`
    rather than from :mod:`spacr.runctx`, so an export run days after the
    measurement still attributes the data to the run that produced it.
    """
    try:
        from ..io import read_settings_history

        history = read_settings_history(db_path)
    except Exception:
        return ""
    for entry in reversed(history or []):
        run = str(entry.get("run_id") or "")
        if run:
            return run
    return ""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def export_anndata(db_path: Union[str, os.PathLike],
                   out_path: Union[str, os.PathLike],
                   *,
                   compression: Optional[str] = "gzip",
                   register: bool = True,
                   project: Union[str, os.PathLike, None] = None,
                   settings: Optional[Mapping[str, Any]] = None,
                   **kwargs: Any) -> ExportResult:
    """Write a spaCR measurements database to ``out_path`` as ``.h5ad``.

    Everything :func:`build_anndata` accepts is accepted here and passed
    through; this adds the write and the artifact registration.

    :param db_path: a ``measurements.db``.
    :param out_path: the ``.h5ad`` to write. Parent directories are created.
    :param compression: ``h5py`` compression for ``X`` and the layers;
        ``None`` for an uncompressed file. ``gzip`` typically halves a
        feature matrix and costs a few seconds.
    :param register: record the file with :mod:`spacr.artifacts`, with the
        project's ``measurements-db`` as its input so a re-run of Measure
        marks this export stale.
    :param project: the project root the artifact belongs to. Inferred from
        ``db_path`` (``<project>/measurements/measurements.db``) when
        omitted.
    :param settings: the run settings, hashed into the artifact and ``uns``.
    :param kwargs: passed to :func:`build_anndata`.
    :returns: the :class:`ExportResult`, with ``path`` and ``artifact_id``
        filled in.
    :raises AnnDataExtraMissing: when ``anndata`` is not installed.
    """
    out_path = os.path.abspath(os.path.expanduser(os.fspath(out_path)))
    adata, result = build_anndata(db_path, settings=settings, **kwargs)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    adata.write_h5ad(out_path, compression=compression)

    artifact_id = ""
    if register:
        artifact_id = _register(out_path, db_path, project, settings,
                                result, adata)

    # `replace` rather than a field-by-field copy: the copy silently dropped
    # whichever field was added to ExportResult last, and a count that
    # arrives as its default is indistinguishable from a real zero.
    return _dataclass_replace(result, path=out_path, artifact_id=artifact_id)


def _project_root(db_path: str,
                  project: Union[str, os.PathLike, None]) -> str:
    """The project root for the artifact registry.

    ``<project>/measurements/measurements.db`` is the layout every spaCR
    writer uses, so the root is two directories up.
    """
    if project is not None:
        return os.path.abspath(os.path.expanduser(os.fspath(project)))
    measurements = os.path.dirname(os.path.abspath(db_path))
    if os.path.basename(measurements) == "measurements":
        return os.path.dirname(measurements)
    return measurements


def _register(out_path: str, db_path: str,
              project: Union[str, os.PathLike, None],
              settings: Optional[Mapping[str, Any]],
              result: ExportResult, adata) -> str:
    """Register the written file, returning its artifact id or ``""``.

    Never raises: a registry that cannot be opened (a read-only project, a
    network filesystem that refuses the lock) must not lose a finished
    export. The failure is warned about and the file stands on its own,
    because ``uns['spacr']`` already carries the same provenance.
    """
    try:
        from .. import artifacts, ports

        root = _project_root(db_path, project)
        inputs: List[str] = []
        try:
            upstream = artifacts.latest(ports.MEASUREMENTS_DB, project=root)
            if upstream is not None:
                inputs.append(upstream.artifact_id)
        except Exception:
            pass
        record = artifacts.register(
            project=root,
            module=APP_KEY,
            kind=ANNDATA_KIND,
            role="h5ad",
            path=out_path,
            settings=settings,
            inputs=inputs,
            run_id=str(adata.uns["spacr"].get("run_id", "")),
            extra={
                "n_obs": int(result.n_obs),
                "n_vars": int(result.n_vars),
                "n_obs_before_filter": int(result.n_obs_before_filter),
                "nan_policy": result.nan_policy,
                "n_missing": int(result.n_missing),
                "source_database": db_path,
                "obsm": list(result.obsm_keys),
            })
        return record.artifact_id
    except Exception as exc:                       # pragma: no cover - env
        warnings.warn(
            f"the AnnData export at {out_path} was written but could not be "
            f"registered with spacr.artifacts ({exc}). Its provenance is "
            f"still in uns['spacr'].", RuntimeWarning, stacklevel=2)
        return ""


def export_anndata_set(db_path: Union[str, os.PathLike],
                       out_dir: Union[str, os.PathLike],
                       *,
                       object_tables: Sequence[str] = ("cell", "nucleus",
                                                       "pathogen",
                                                       "cytoplasm"),
                       prefix: str = "",
                       **kwargs: Any) -> Dict[str, ExportResult]:
    """Write one ``.h5ad`` per object table, cross-referenced by ``uns``.

    The honest shape for a multi-compartment experiment: a nucleus is not a
    cell and averaging it onto one loses the distribution. Each file holds
    one object type at its own granularity, and each child file records its
    parent in ``uns['spacr']['relationships']`` -- table, key column and the
    sibling file -- so the set can be reassembled (or handed to ``MuData``)
    without guessing.

    :param db_path: a ``measurements.db``.
    :param out_dir: folder to write into; created if missing.
    :param object_tables: which object tables to export. Tables absent from
        the database are skipped, not an error.
    :param prefix: prepended to each file name.
    :param kwargs: passed to :func:`export_anndata`.
    :returns: ``{table: ExportResult}`` for the tables actually written.
    """
    db_path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
    out_dir = os.path.abspath(os.path.expanduser(os.fspath(out_dir)))
    os.makedirs(out_dir, exist_ok=True)

    present = set(_available_tables(db_path))
    results: Dict[str, ExportResult] = {}
    files = {table: os.path.join(out_dir, f"{prefix}{table}.h5ad")
             for table in object_tables if table in present}

    for table, path in files.items():
        contract = schema.OBJECT_TABLE_SCHEMAS.get(table)
        parent_column = getattr(contract, "parent_column", None)
        result = export_anndata(db_path, path, single_table=table, **kwargs)
        results[table] = result
        if parent_column and "cell" in files:
            _stamp_parent_file(path, files["cell"], parent_column)
    return results


def _stamp_parent_file(child_path: str, parent_path: str,
                       parent_column: str) -> None:
    """Record the sibling file holding this child's parents.

    Done as a small re-open rather than in :func:`build_anndata` because
    only the caller writing the whole set knows where the parent landed.
    """
    try:
        anndata = require_anndata()
        adata = anndata.read_h5ad(child_path)
        relationships = dict(adata.uns["spacr"].get("relationships", {}))
        parent = dict(relationships.get("parent", {}))
        parent["file"] = os.path.basename(parent_path)
        parent["obs_column"] = parent_column
        relationships["parent"] = parent
        provenance = dict(adata.uns["spacr"])
        provenance["relationships"] = relationships
        adata.uns["spacr"] = provenance
        adata.write_h5ad(child_path)
    except Exception as exc:                       # pragma: no cover - env
        warnings.warn(
            f"could not record the parent file in {child_path}: {exc}",
            RuntimeWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# Registration seams
# ---------------------------------------------------------------------------

def anndata_export_settings(settings: Optional[Mapping[str, Any]] = None
                            ) -> Dict[str, Any]:
    """Return this module's settings, filling in anything absent.

    :param settings: an existing settings dict to complete.
    :returns: a new dict; the caller's is not mutated.
    """
    resolved = dict(settings or {})
    resolved.setdefault("src", "")
    resolved.setdefault("anndata_out", "")
    resolved.setdefault("anndata_tables", list(DEFAULT_TABLES))
    resolved.setdefault("anndata_single_table", "")
    resolved.setdefault("anndata_nan_policy", NAN_KEEP)
    resolved.setdefault("anndata_dtype", "float32")
    resolved.setdefault("anndata_row_limit", 0)
    resolved.setdefault("anndata_compute_umap", False)
    resolved.setdefault("anndata_compression", "gzip")
    resolved.setdefault("anndata_register_artifact", True)
    return resolved


def resolve_db_path(src: Union[str, os.PathLike]) -> str:
    """The ``measurements.db`` a settings ``src`` means.

    ``src`` is a project root everywhere else in spaCR, and this module's
    argument is a database, so one of the two has to give. A path that ends
    in a database file is taken as one, and anything else is read as a
    project root laid out the way every spaCR writer leaves it.

    :param src: project root, or the database itself.
    :returns: an absolute path, which is NOT checked for existence -- an
        absent file is :func:`spacr.validate.validate_settings`'s report to
        make, with the path in it, rather than an exception from here.
    """
    src = os.path.abspath(os.path.expanduser(os.fspath(src)))
    if os.path.splitext(src)[1].lower() in (".db", ".sqlite", ".sqlite3"):
        return src
    return os.path.join(src, "measurements", "measurements.db")


def default_out_path(src: Union[str, os.PathLike],
                     single_table: str = "") -> str:
    """Where the export lands when ``anndata_out`` is empty.

    ``<project>/results/<project name>.h5ad`` -- beside the other things a
    finished run produced, named after the project, because a folder of
    ``export.h5ad`` files is a folder nobody can tell apart.

    :param src: project root, or the database.
    :param single_table: the one object table being exported, if any; it
        joins the file name, since a nucleus-level export and a cell-level
        one of the same project are different files.
    :returns: an absolute ``.h5ad`` path.
    """
    root = _project_root(resolve_db_path(src), None)
    name = os.path.basename(root.rstrip(os.sep)) or "spacr"
    if single_table:
        name = f"{name}_{single_table}"
    return os.path.join(root, "results", f"{name}.h5ad")


def run_anndata_export(settings: Optional[Mapping[str, Any]] = None
                       ) -> ExportResult:
    """Run the export from a settings dict. The headless entry point.

    The ``fn(settings)`` shape ``spacr-run``, the Qt Run button and
    :mod:`spacr.validate` all dispatch to, wrapped around
    :func:`export_anndata` -- which keeps its own explicit keyword
    signature, because a function whose arguments are a dict is a function
    nobody can call from a notebook.

    Every key it reads is one :func:`anndata_export_settings` declares and
    :func:`register_anndata_settings` gave a type and a tooltip, so the form
    the GUI draws and the keys honoured here are the same list.

    :param settings: the run settings. ``src`` is the project root (or the
        database); everything else falls back to
        :func:`anndata_export_settings`.
    :returns: the :class:`ExportResult`, whose ``describe()`` is what the
        console prints.
    :raises ValueError: when ``src`` is empty -- there is nothing to export
        and no path to name in the message otherwise.
    :raises AnnDataExtraMissing: when ``anndata`` is not installed.
    """
    resolved = anndata_export_settings(settings)
    src = str(resolved.get("src") or "").strip()
    if not src:
        raise ValueError(
            "anndata_export needs src: the spaCR project whose "
            "measurements/measurements.db is exported.")

    db_path = resolve_db_path(src)
    single_table = str(resolved.get("anndata_single_table") or "").strip()
    out_path = str(resolved.get("anndata_out") or "").strip()
    if not out_path:
        out_path = default_out_path(src, single_table)

    tables = resolved.get("anndata_tables") or list(DEFAULT_TABLES)
    if isinstance(tables, str):
        # A settings.csv round trip spells a list as one comma-separated
        # cell; taking it apart here means `--set anndata_tables=cell,nucleus`
        # works and does not silently export a table called "cell,nucleus".
        tables = [part.strip() for part in tables.split(",") if part.strip()]

    row_limit = int(resolved.get("anndata_row_limit") or 0)
    compression = str(resolved.get("anndata_compression") or "").strip()

    return export_anndata(
        db_path, out_path,
        compression=compression or None,
        register=bool(resolved.get("anndata_register_artifact", True)),
        settings=resolved,
        tables=tuple(tables),
        single_table=single_table or None,
        row_limit=row_limit or None,
        nan_policy=str(resolved.get("anndata_nan_policy") or NAN_KEEP),
        dtype=str(resolved.get("anndata_dtype") or "float32"),
        compute_umap=bool(resolved.get("anndata_compute_umap", False)),
    )


_TYPES = {
    "anndata_out": str,
    "anndata_tables": list,
    "anndata_single_table": str,
    "anndata_nan_policy": str,
    "anndata_dtype": str,
    "anndata_row_limit": int,
    "anndata_compute_umap": bool,
    "anndata_compression": str,
    "anndata_register_artifact": bool,
}

_TOOLTIPS = {
    "anndata_out": (
        "(str) - Path of the .h5ad file to write. An existing file at this "
        "path is overwritten, and the export is registered against it in "
        "artifacts.db, so writing twice to one path replaces the earlier "
        "record rather than adding a second. Default "
        "<src>/results/<project>.h5ad."),
    "anndata_tables": (
        "(list) - Object tables joined into the cell-anchored export. "
        "Dropping a table drops its features from X: without 'pathogen' "
        "there are no pathogen columns and no count_pathogen, and without "
        "'png_list' the crop paths, annotations and model scores are all "
        "absent from obs. Default ['cell', 'cytoplasm', 'nucleus', "
        "'pathogen', 'png_list']."),
    "anndata_single_table": (
        "(str) - Export this one object table instead of the join, one row per object of that type. The only way to get a nucleus-level or pathogen-level matrix: the join averages children onto their parent cell. Empty means the joined export. Default ''."),
    "anndata_nan_policy": (
        "(str) - What happens to missing values in X: 'keep' (default; "
        "AnnData stores them, scanpy's scale/pca/neighbors do not), "
        "'drop_features', 'drop_objects', 'zero' or 'mean'. The two "
        "imputing policies also write a layers['missing'] mask."),
    "anndata_dtype": (
        "(str) - dtype of X. Default 'float32' - the scanpy convention and "
        "half the memory of float64."),
    "anndata_row_limit": (
        '(int) - Hard cap on exported objects, applied after filtering. 0 means no cap. A cap, not a filter: the objects kept are simply the first N in table order. Default 0.'),
    "anndata_compute_umap": (
        "(bool) - Compute obsm['X_umap'] during the export, through the "
        "same reducer the UMAP app uses. Off by default: it costs minutes "
        "on a large table."),
    "anndata_compression": (
        "(str) - HDF5 compression for X and the layers. 'gzip' (default) "
        "typically halves the file; '' writes it uncompressed."),
    "anndata_register_artifact": (
        "(bool) - Record the written file with spacr.artifacts, so a re-run "
        "of Measure marks the export stale. Default True."),
}

_DESCRIPTION = (
    "Export the measurement tables as AnnData (.h5ad) - N objects x M "
    "features with per-object metadata, feature definitions, embeddings and "
    "provenance - so scanpy, scvi-tools and squidpy can read a spaCR run "
    "directly."
)


def register_anndata_settings(replace: bool = False) -> bool:
    """Register this module's settings through the defaults seam.

    Uses :func:`spacr.settings.register_defaults` rather than appending to
    ``spacr/settings.py``, so this module owns its own knobs and adding one
    is not a merge conflict in a file nobody owns.

    :param replace: re-register over an existing registration.
    :returns: True if it registered, False if it already was.
    """
    from ..settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY) and not replace:
        return False
    register_defaults(
        APP_KEY, anndata_export_settings, replace=replace,
        expected_types=_TYPES, tooltips=_TOOLTIPS,
        categories={
            "General": ["anndata_out", "anndata_single_table",
                        "anndata_nan_policy"],
            "Advanced": ["anndata_tables", "anndata_dtype",
                         "anndata_row_limit", "anndata_compute_umap",
                         "anndata_compression",
                         "anndata_register_artifact"],
        },
        description=_DESCRIPTION)
    return True


#: "AnnData Export" in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English -- sv, de, es, zh_CN,
#: pt, hi, ko, is, fr. "AnnData" is a file format and a library name, so it
#: is not translated in any of them; the verb around it is.
APP_TRANSLATIONS = (
    "AnnData-export",
    "AnnData-Export",
    "Exportar a AnnData",
    "导出 AnnData",
    "Exportar para AnnData",
    "AnnData निर्यात",
    "AnnData 내보내기",
    "AnnData-útflutningur",
    "Export AnnData",
)


def register_anndata_app(replace: bool = False) -> bool:
    """Register the Qt app row through :func:`spacr.qt.app.register_app`.

    Called at import, but **only when the Qt app module is already
    imported** -- that is, when there is a GUI in this process to register
    with. Importing ``spacr.qt.app`` from here would drag PySide6 (and, on
    some platforms, a display connection) into every headless export, which
    is the opposite of what an optional GUI is for.

    That guard is also why ``spacr.qt.app`` names this function in its own
    ``_SELF_REGISTERING_APPS`` table and calls it from the bottom of its
    import: called only from here, the row existed or not depending on
    whether something else had already imported the Qt registry, which is
    an app inventory decided by import order.

    **No screen of its own, and none is wanted.** The app registers no
    ``factory``, so it gets the generic settings-driven ``AppScreen`` -- and
    every knob this module has is already a typed, tooltipped key in
    :func:`register_anndata_settings`, so that generic form IS the export
    dialog. ``defaults_module`` is what makes it appear: it tells
    ``settings_model`` to import this module before asking whether the key
    has defaults. The Run button runs :func:`run_anndata_export`, which is
    also what ``spacr-run anndata_export`` runs -- an export is a batch step
    you want on the cluster after Measure, not an interactive tool.

    :param replace: re-register over an existing row.
    :returns: True if it registered, False otherwise.
    """
    module = sys.modules.get("spacr.qt.app")
    if module is None:
        return False
    try:
        if replace:
            module.unregister_app(APP_KEY)
        elif any(row[0] == APP_KEY for row in module.APPS):
            return False
        module.register_app(
            APP_KEY, "AnnData Export",
            "Write the measurements as .h5ad for scanpy and scvi-tools",
            module.SECTION_EXPLORE, stage=module.STAGE_ALPHA,
            title="AnnData Export", intro=_DESCRIPTION,
            api_module="anndata_export",
            entry="spacr.anndata_export:run_anndata_export",
            defaults_module="spacr.anndata_export",
            translations=APP_TRANSLATIONS)
        return True
    except Exception:                              # pragma: no cover - env
        # A registry that has changed shape, or a section that no longer
        # exists. A missing sidebar row is cosmetic; an exception here would
        # break `import spacr.anndata_export` for every headless caller.
        return False


register_anndata_settings()
register_anndata_app()
