"""Typed input/output ports: what each spaCR module consumes and produces.

Every pipeline module in spaCR reads a folder laid out by the module before
it and writes a folder the module after it expects.  Until now that contract
lived only in the code that happened to open the path, so "can Measure run
here?" could only be answered by starting Measure and watching it fail twenty
minutes in.

This module makes the contract data.  Each module declares:

* what it **consumes** — a :class:`Port` per input, carrying a *kind*
  (``"merged-arrays"``, ``"measurements-db"``, ``"crops"``, …), the path it
  lives at relative to the project root, and a shape/table contract;
* what it **produces** — the same, for its outputs.

:func:`check_ready` then answers *before a run starts* whether a module can
run, and when it cannot, says why and what to do about it.  The answer is a
list of :class:`spacr.validate.Problem` — the same type the settings
pre-flight returns — so a caller can print both through
:func:`spacr.validate.format_report`.

Path conventions are not invented here.  They are the ones already in the
code: ``<root>/merged/*.npy`` and ``<root>/measurements/measurements.db``
(:func:`spacr.core.preprocess_generate_masks`), the "``src`` may already be
the merged folder" hop (``spacr.crops._looks_like_experiment_root``,
``spacr.validate._resolve_merged_dir``), the ``src`` / ``orig`` /
``consolidated`` search order for raw images
(``spacr.validate._scan_raw_images``), ``<root>/data/**/*_png``
(:mod:`spacr.measure`), ``<root>/model/...`` (:mod:`spacr.deep_spacr`),
``<root>/results/...`` (:mod:`spacr.ml`).  Module keys are
:data:`spacr.validate.APP_FUNCTIONS` keys, so ``check_ready("measure", s)``
and ``validate_settings(s, "measure")`` speak the same language.

Public API
----------
``Port`` / ``ShapeContract`` / ``ModulePorts``
    The declarations.
``PORTS``, ``module_ports``, ``register_module_ports``, ``known_modules``
    The registry and its extension seam.
``project_root``, ``resolve_port``, ``declared_inputs``, ``declared_outputs``
    Path resolution.
``check_ready``, ``format_readiness``, ``describe_ports``
    "Can module X run, and if not, why not?"
``producers_of``, ``consumers_of``, ``next_modules``, ``upstream_modules``
    The module graph, for auto-chaining and "continue to next step".

The module imports only the standard library plus two dependency-light spaCR
modules (:mod:`spacr.validate`, :mod:`spacr.resume`).  No numpy, torch,
pandas or Cellpose: a readiness check has to cost less than the run it is
protecting.
"""
from __future__ import annotations

import glob as _glob
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .resume import read_npy_header
from .validate import (ALT_SRC_KEYS, APP_ALIASES, DB_APPS, ERROR,
                       IMAGE_EXTENSIONS, WARNING, Problem)

__all__ = [
    "ALL_KINDS",
    "BARCODE_MAP",
    "CHANNEL_STACKS",
    "CROPS",
    "EMBEDDING",
    "MASKS",
    "MEASUREMENTS_DB",
    "MERGED_ARRAYS",
    "MODEL_WEIGHTS",
    "OBJECT_COUNTS",
    "PORTS",
    "PREDICTIONS",
    "RAW_IMAGES",
    "REGRESSION_RESULTS",
    "ROOT_KEYS",
    "SEQUENCING_READS",
    "SETTINGS_CSV",
    "ModulePorts",
    "Port",
    "Readiness",
    "ResolvedPort",
    "ShapeContract",
    "UnknownModule",
    "check_ready",
    "consumers_of",
    "declared_inputs",
    "declared_outputs",
    "describe_ports",
    "format_readiness",
    "known_modules",
    "module_ports",
    "next_modules",
    "producers_of",
    "project_root",
    "register_module_ports",
    "resolve_port",
    "upstream_modules",
]


# ---------------------------------------------------------------------------
# The kind vocabulary
# ---------------------------------------------------------------------------

#: Raw microscope files as they come off the instrument.
RAW_IMAGES = "raw-images"
#: ``<root>/stack/*.npy`` — one array per field, image channels concatenated.
CHANNEL_STACKS = "channel-stacks"
#: ``<root>/merged/*.npy`` — image channels and label masks in one array. The
#: unit Measure iterates over.
MERGED_ARRAYS = "merged-arrays"
#: ``<root>/masks/`` — per-object label arrays. Removed by
#: ``spacr.utils.cleanup_pipeline_folders`` unless ``keep_intermediate`` is
#: set, which is why every port for it is optional.
MASKS = "masks"
#: ``<root>/measurements/measurements.db`` — the per-object measurement
#: tables every analysis module reads. Only Measure produces these.
MEASUREMENTS_DB = "measurements-db"
#: The object-count and run-status tables the mask pipeline writes into the
#: same database file. A distinct kind on purpose: mask creating the file
#: does not make the *measurements* in it current, and treating the two as
#: one artifact let a re-run of Mask pass itself off as a fresh Measure.
OBJECT_COUNTS = "object-counts"
#: ``<root>/data/**/*_png`` — single-object PNG crops.
CROPS = "crops"
#: A trained classifier, ``*.pth``.
MODEL_WEIGHTS = "model-weights"
#: Per-object model scores, written back into the measurements database.
PREDICTIONS = "predictions"
#: A UMAP/tSNE embedding and the figures drawn from it.
EMBEDDING = "embedding"
#: ``*.fastq.gz`` reads from a pooled screen.
SEQUENCING_READS = "sequencing-reads"
#: Row / column / gRNA barcode assignments per read.
BARCODE_MAP = "barcode-map"
#: Regression coefficients, hits and their figures.
REGRESSION_RESULTS = "regression-results"
#: A recorded settings CSV sitting beside the data it produced.
SETTINGS_CSV = "settings-csv"

#: Every kind spaCR declares. Third-party ports may use others; this is the
#: built-in vocabulary, not a closed set.
ALL_KINDS: Tuple[str, ...] = (
    RAW_IMAGES, CHANNEL_STACKS, MERGED_ARRAYS, MASKS, MEASUREMENTS_DB,
    OBJECT_COUNTS, CROPS, MODEL_WEIGHTS, PREDICTIONS, EMBEDDING,
    SEQUENCING_READS, BARCODE_MAP, REGRESSION_RESULTS, SETTINGS_CSV,
)

#: Where the mask pipeline looks for raw images, in order — the same three
#: folders ``spacr.validate._scan_raw_images`` searches.
RAW_IMAGE_GLOB = "*|orig/*|consolidated/*"

#: Settings key naming the project root when it is not ``src``.
#: :func:`spacr.foreign.import_project` reads someone else's project and
#: *writes* the spaCR one to ``dst``, so ``dst`` is the project it produces.
#: Input-only overrides come from :data:`spacr.validate.ALT_SRC_KEYS`.
ROOT_KEYS: Dict[str, str] = {"foreign": "dst"}


class UnknownModule(KeyError):
    """No ports are declared for the requested module key."""


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeContract:
    """What an array at a port has to look like.

    Checked from the ``.npy`` header alone (see
    :func:`spacr.resume.read_npy_header`), so validating a thousand
    100-megabyte fields stays cheap — and it catches truncation, which
    ``np.load`` only discovers after allocating the whole array.

    :param ndim: required number of axes, or None for any.
    :param min_planes: smallest acceptable length of the last axis. A merged
        array needs at least one image plane and one mask plane.
    :param dtype: required numpy dtype string such as ``"uint16"``, or ``""``
        for any.
    """

    ndim: Optional[int] = None
    min_planes: Optional[int] = None
    dtype: str = ""

    def describe(self) -> str:
        """Return a one-line human description of the contract."""
        parts: List[str] = []
        if self.ndim is not None:
            parts.append(f"{self.ndim}-D")
        if self.min_planes is not None:
            parts.append(f"at least {self.min_planes} planes")
        if self.dtype:
            parts.append(f"dtype {self.dtype}")
        return ", ".join(parts) if parts else "any array"


@dataclass(frozen=True)
class Port:
    """One thing a module consumes or produces.

    :param kind: the vocabulary term, e.g. :data:`MERGED_ARRAYS`. Ports match
        across modules by kind, which is what makes auto-chaining possible.
    :param role: short name, unique within a module, e.g. ``"merged"``. It
        labels the problem messages and the registered artifact.
    :param path: location relative to the project root. ``""`` is the root.
    :param pattern: glob applied inside ``path``. Empty means ``path`` names a
        single file or folder that must simply exist. ``**`` is honoured, and
        ``|`` separates alternatives, all of which are searched.
    :param required: False for an output that may legitimately be absent
        (``masks/`` after cleanup) or an input a module can do without.
    :param min_count: how many matches ``pattern`` must yield.
    :param extensions: when set, matches are kept only if their lowercased
        name ends with one of these.
    :param shape: array contract, for ``.npy`` ports.
    :param tables: SQLite tables that must exist and hold at least one row.
        Only meaningful for :data:`MEASUREMENTS_DB` ports.
    :param description: one line, shown by :func:`describe_ports`.
    """

    kind: str
    role: str
    path: str = ""
    pattern: str = ""
    required: bool = True
    min_count: int = 1
    extensions: Tuple[str, ...] = ()
    shape: Optional[ShapeContract] = None
    tables: Tuple[str, ...] = ()
    description: str = ""

    def relative(self) -> str:
        """Return the port's location relative to the project root, for display."""
        parts = [p for p in (self.path, self.pattern) if p]
        return os.path.join(*parts) if parts else "."


@dataclass(frozen=True)
class ModulePorts:
    """Everything one module reads and writes.

    :param key: the module key, as in :data:`spacr.validate.APP_FUNCTIONS`.
    :param consumes: inputs.
    :param produces: outputs.
    :param summary: one line describing the module, for :func:`describe_ports`.
    """

    key: str
    consumes: Tuple[Port, ...] = ()
    produces: Tuple[Port, ...] = ()
    summary: str = ""

    def port(self, role: str) -> Port:
        """Return the consumed or produced port with ``role``.

        :param role: the port's role name.
        :raises KeyError: when this module declares no such role.
        """
        for candidate in self.consumes + self.produces:
            if candidate.role == role:
                return candidate
        raise KeyError(f"{self.key} declares no port with role {role!r}")


# ---------------------------------------------------------------------------
# The built-in declarations
# ---------------------------------------------------------------------------

_MASK_PORTS = ModulePorts(
    key="mask",
    summary="raw microscope files to merged arrays carrying label masks",
    consumes=(
        Port(RAW_IMAGES, "images", "", RAW_IMAGE_GLOB,
             extensions=tuple(IMAGE_EXTENSIONS),
             description="raw files in src, orig/ or consolidated/"),
    ),
    produces=(
        Port(MERGED_ARRAYS, "merged", "merged", "*.npy",
             shape=ShapeContract(ndim=3, min_planes=2),
             description="image channels and label masks, one array per field"),
        Port(OBJECT_COUNTS, "counts", "measurements/measurements.db",
             description="object counts and the run-status stamp"),
        Port(MASKS, "masks", "masks", "*", required=False,
             description="per-object label arrays; removed by cleanup"),
        Port(SETTINGS_CSV, "settings", "settings/gen_mask_settings.csv",
             required=False, description="the settings this run used"),
    ),
)

_MEASURE_PORTS = ModulePorts(
    key="measure",
    summary="merged arrays to per-object measurements and crops",
    consumes=(
        Port(MERGED_ARRAYS, "merged", "merged", "*.npy",
             shape=ShapeContract(ndim=3, min_planes=2),
             description="one array per field, written by mask"),
    ),
    produces=(
        Port(MEASUREMENTS_DB, "db", "measurements/measurements.db",
             description="one table per object type"),
        Port(CROPS, "crops", "data", "**/*_png", required=False,
             description="single-object PNGs, when save_png is on"),
    ),
)

_CLASSIFY_PORTS = ModulePorts(
    key="classify",
    summary="crops to a trained classifier and per-object scores",
    consumes=(
        Port(MEASUREMENTS_DB, "db", "measurements/measurements.db",
             tables=("png_list",),
             description="png_list points at the crops to train on"),
        Port(CROPS, "crops", "data", "**/*_png", required=False,
             description="the crops themselves, when not read from a tar"),
    ),
    produces=(
        Port(MODEL_WEIGHTS, "model", "model", "**/*.pth",
             description="the trained classifier"),
        Port(PREDICTIONS, "scores", "measurements/measurements.db",
             required=False,
             description="per-object scores merged back into the database"),
    ),
)

_UMAP_PORTS = ModulePorts(
    key="umap",
    summary="per-object features to a 2-D embedding",
    consumes=(
        Port(MEASUREMENTS_DB, "db", "measurements/measurements.db",
             tables=("png_list",),
             description="features, and the crop paths drawn on the embedding"),
    ),
    produces=(
        Port(EMBEDDING, "embedding", "results", "*", required=False,
             description="embedding table, scatter and grid figures"),
        Port(SETTINGS_CSV, "settings", "settings/embedding_settings.csv",
             required=False, description="the settings this run used"),
    ),
)

_REGRESSION_PORTS = ModulePorts(
    key="regression",
    summary="measurements plus a barcode map to gene-level hits",
    consumes=(
        Port(MEASUREMENTS_DB, "db", "measurements/measurements.db",
             description="the scores being regressed"),
    ),
    produces=(
        Port(REGRESSION_RESULTS, "results", "results", "**/results*.csv",
             description="coefficients, and the significant subset"),
    ),
)

_ML_PORTS = ModulePorts(
    key="ml_analyze",
    summary="measurements to classical-ML per-object scores",
    consumes=(
        Port(MEASUREMENTS_DB, "db", "measurements/measurements.db",
             description="the feature table"),
    ),
    produces=(
        Port(PREDICTIONS, "scores", "measurements/measurements.db",
             description="per-object scores merged back into the database"),
    ),
)

_BARCODE_PORTS = ModulePorts(
    key="map_barcodes",
    summary="FASTQ reads to row / column / gRNA barcode assignments",
    consumes=(
        Port(SEQUENCING_READS, "reads", "", "*.fastq.gz",
             description="paired or single-end reads"),
    ),
    produces=(
        Port(BARCODE_MAP, "reads_h5", "", "*/annotated_reads.h5",
             description="one row per read, with its three barcodes"),
        Port(BARCODE_MAP, "combinations", "", "*/unique_combinations.csv",
             required=False, description="collapsed well x gRNA counts"),
    ),
)

#: Declared ports, keyed by module. Mutated only through
#: :func:`register_module_ports`.
PORTS: Dict[str, ModulePorts] = {}


def register_module_ports(ports: ModulePorts, *,
                          overwrite: bool = False) -> ModulePorts:
    """Add or replace one module's port declaration.

    The seam a plugin — or a module written after this one — uses to join the
    graph, so :data:`PORTS` never has to be edited by hand.

    :param ports: the declaration. Its key is lowercased before storage.
    :param overwrite: allow replacing an existing declaration. Off by default,
        so two plugins claiming one key is an error rather than a silent
        last-one-wins.
    :returns: the stored declaration, for chaining.
    :raises ValueError: when the key is empty, when it is already declared and
        ``overwrite`` is False, or when two of its ports share a role.
    """
    key = str(ports.key).strip().lower()
    if not key:
        raise ValueError("a module port declaration needs a non-empty key")
    if key in PORTS and not overwrite:
        raise ValueError(
            f"ports for {key!r} are already declared; pass overwrite=True to "
            f"replace them")
    roles = [p.role for p in ports.consumes + ports.produces]
    duplicates = sorted({r for r in roles if roles.count(r) > 1})
    if duplicates:
        raise ValueError(
            f"{key}: port roles must be unique, repeated: "
            f"{', '.join(duplicates)}")
    stored = ports if ports.key == key else ModulePorts(
        key=key, consumes=ports.consumes, produces=ports.produces,
        summary=ports.summary)
    PORTS[key] = stored
    return stored


for _declaration in (_MASK_PORTS, _MEASURE_PORTS, _CLASSIFY_PORTS, _UMAP_PORTS,
                     _REGRESSION_PORTS, _ML_PORTS, _BARCODE_PORTS):
    register_module_ports(_declaration)

# The timelapse module *is* the mask pipeline with tracking on —
# spacr.core.preprocess_generate_masks_timelapse calls
# preprocess_generate_masks — so it has the mask pipeline's ports.
register_module_ports(ModulePorts(
    key="timelapse",
    summary="mask generation with objects linked across frames",
    consumes=_MASK_PORTS.consumes, produces=_MASK_PORTS.produces))

# Every app spacr.validate already knows opens
# <src>/measurements/measurements.db gets a declaration derived from that
# fact rather than from invention: enough to answer "is there a database to
# read?", and no claim about outputs nobody has verified.
for _db_app in sorted(DB_APPS):
    if _db_app not in PORTS:
        register_module_ports(ModulePorts(
            key=_db_app,
            summary="reads the measurements database",
            consumes=(Port(MEASUREMENTS_DB, "db",
                           "measurements/measurements.db",
                           description="the measurements this analysis reads"),)))


def known_modules() -> Tuple[str, ...]:
    """Return every module key with declared ports, sorted."""
    return tuple(sorted(PORTS))


def module_ports(module: str) -> ModulePorts:
    """Return the port declaration for ``module``.

    Accepts every alias :data:`spacr.validate.APP_ALIASES` accepts, so
    ``"measure_crop"`` and ``"measure"`` resolve to the same declaration.

    :param module: module key or alias.
    :returns: the :class:`ModulePorts` declared for it.
    :raises UnknownModule: when nothing is declared for it.
    """
    key = str(module).strip().lower()
    key = APP_ALIASES.get(key, key)
    if key not in PORTS:
        raise UnknownModule(
            f"no ports declared for {module!r}; known: "
            f"{', '.join(known_modules())}")
    return PORTS[key]


# ---------------------------------------------------------------------------
# The module graph
# ---------------------------------------------------------------------------

def producers_of(kind: str) -> Tuple[str, ...]:
    """Return the modules that produce ``kind``, sorted.

    :param kind: a vocabulary term such as :data:`MERGED_ARRAYS`.
    """
    return tuple(sorted(
        key for key, spec in PORTS.items()
        if any(p.kind == kind for p in spec.produces)))


def consumers_of(kind: str, *, required_only: bool = False) -> Tuple[str, ...]:
    """Return the modules that consume ``kind``, sorted.

    :param kind: a vocabulary term such as :data:`MEASUREMENTS_DB`.
    :param required_only: only modules that *need* it, not those that can
        optionally use it.
    """
    return tuple(sorted(
        key for key, spec in PORTS.items()
        if any(p.kind == kind and (p.required or not required_only)
               for p in spec.consumes)))


def next_modules(module: str) -> Tuple[str, ...]:
    """Return the modules that can run on what ``module`` produces.

    The answer to "this run finished — what would you like to do next?".

    :param module: module key or alias.
    :raises UnknownModule: when nothing is declared for it.
    """
    spec = module_ports(module)
    produced = {p.kind for p in spec.produces}
    return tuple(sorted(
        key for key, candidate in PORTS.items()
        if key != spec.key
        and any(p.kind in produced and p.required for p in candidate.consumes)))


def upstream_modules(module: str) -> Tuple[str, ...]:
    """Return the modules that produce what ``module`` requires.

    The answer to "Measure needs merged arrays — who makes those?".

    :param module: module key or alias.
    :raises UnknownModule: when nothing is declared for it.
    """
    spec = module_ports(module)
    needed = {p.kind for p in spec.consumes if p.required}
    return tuple(sorted(
        key for key, candidate in PORTS.items()
        if key != spec.key and any(p.kind in needed for p in candidate.produces)))


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def project_root(settings_or_src: Union[str, Mapping[str, Any], None],
                 module: str = "") -> str:
    """Return the absolute project root a run works in.

    Reuses the conventions already in the code rather than inventing one:

    * a list of sources means several plates, and the first names the first
      project (:func:`spacr.core.preprocess_generate_masks` loops over
      ``settings['src']``);
    * a ``src`` that already ends in ``merged`` names the *merged folder*,
      not the project — the same hop
      ``spacr.crops._looks_like_experiment_root`` and
      ``spacr.validate._resolve_merged_dir`` make;
    * modules whose folder is not ``src`` are looked up in :data:`ROOT_KEYS`
      and :data:`spacr.validate.ALT_SRC_KEYS`.

    :param settings_or_src: a settings dict, a path, or None.
    :param module: module key or alias, used only to pick the settings key.
    :returns: an absolute path, or ``""`` when no source is set.
    """
    if settings_or_src is None:
        return ""
    if isinstance(settings_or_src, Mapping):
        key = str(module).strip().lower()
        key = APP_ALIASES.get(key, key)
        source_key = ROOT_KEYS.get(key) or ALT_SRC_KEYS.get(key, "src")
        value: Any = settings_or_src.get(source_key)
    else:
        value = settings_or_src
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if not isinstance(value, str) or not value.strip():
        return ""
    absolute = os.path.abspath(os.path.expanduser(value.strip()))
    if os.path.basename(os.path.normpath(absolute)).endswith("merged"):
        return os.path.dirname(os.path.normpath(absolute))
    return absolute


@dataclass(frozen=True)
class ResolvedPort:
    """A :class:`Port` bound to a project root and looked up on disk.

    :param port: the declaration.
    :param root: the project root it was resolved against.
    :param target: absolute path of the file or folder the port names.
    :param paths: absolute paths the port's pattern matched, sorted. Empty
        for a pattern-less port.
    :param exists: whether the port is present at all.
    :param count: number of matches; 1 for a pattern-less port that exists.
    """

    port: Port
    root: str
    target: str
    paths: Tuple[str, ...]
    exists: bool
    count: int

    @property
    def kind(self) -> str:
        """The port's kind, for convenience."""
        return self.port.kind

    @property
    def role(self) -> str:
        """The port's role, for convenience."""
        return self.port.role

    @property
    def location(self) -> str:
        """The single path this port stands for: the file, or the folder."""
        return self.target


def resolve_port(port: Port, root: str) -> ResolvedPort:
    """Bind ``port`` to ``root`` and look it up on disk.

    :param port: the declaration.
    :param root: absolute project root.
    :returns: a :class:`ResolvedPort`. A missing path is not an exception —
        absence is an answer, and :func:`check_ready` is what judges it.
    """
    target = os.path.join(root, port.path) if port.path else root
    if not port.pattern:
        exists = os.path.exists(target)
        return ResolvedPort(port=port, root=root, target=target, paths=(),
                            exists=exists, count=1 if exists else 0)
    found: set = set()
    for alternative in port.pattern.split("|"):
        found.update(_glob.glob(os.path.join(target, alternative),
                                recursive=True))
    if port.extensions:
        found = {p for p in found if p.lower().endswith(port.extensions)}
    matches = tuple(sorted(found))
    return ResolvedPort(port=port, root=root, target=target, paths=matches,
                        exists=bool(matches), count=len(matches))


def declared_outputs(module: str,
                     settings: Union[str, Mapping[str, Any], None] = None,
                     *, root: str = "") -> Tuple[ResolvedPort, ...]:
    """Return this module's produced ports, resolved against the project.

    What a finished run should have written — the list
    :func:`spacr.artifacts.register_run_outputs` walks.

    :param module: module key or alias.
    :param settings: settings dict or source path, used to derive the root.
    :param root: explicit project root, overriding ``settings``.
    :raises UnknownModule: when nothing is declared for ``module``.
    """
    spec = module_ports(module)
    resolved_root = root or project_root(settings, spec.key)
    return tuple(resolve_port(port, resolved_root) for port in spec.produces)


def declared_inputs(module: str,
                    settings: Union[str, Mapping[str, Any], None] = None,
                    *, root: str = "") -> Tuple[ResolvedPort, ...]:
    """Return this module's consumed ports, resolved against the project.

    :param module: module key or alias.
    :param settings: settings dict or source path, used to derive the root.
    :param root: explicit project root, overriding ``settings``.
    :raises UnknownModule: when nothing is declared for ``module``.
    """
    spec = module_ports(module)
    resolved_root = root or project_root(settings, spec.key)
    return tuple(resolve_port(port, resolved_root) for port in spec.consumes)


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Readiness:
    """Whether a module can run, and why not when it cannot.

    :param module: the canonical module key that was checked.
    :param root: the project root it was checked against.
    :param ok: True when nothing blocking was found. ``bool(readiness)`` is
        the same answer.
    :param problems: :class:`spacr.validate.Problem` instances, errors and
        warnings mixed, with the port role in ``Problem.setting``.
    :param satisfied: roles of the inputs that were found and accepted.
    :param inputs: artifact ids backing the satisfied inputs, when the check
        was given a registry.
    """

    module: str
    root: str
    ok: bool
    problems: Tuple[Problem, ...] = ()
    satisfied: Tuple[str, ...] = ()
    inputs: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """True when the module can run."""
        return self.ok

    @property
    def errors(self) -> Tuple[Problem, ...]:
        """The blocking problems."""
        return tuple(p for p in self.problems if p.is_error)

    @property
    def warnings(self) -> Tuple[Problem, ...]:
        """The non-blocking problems."""
        return tuple(p for p in self.problems if not p.is_error)

    @property
    def reason(self) -> str:
        """One human-readable line saying why the answer is what it is."""
        if self.ok:
            found = (", ".join(self.satisfied) if self.satisfied
                     else "nothing required")
            return (f"{self.module} can run in "
                    f"{self.root or '(no project)'}: {found}")
        first = self.errors[0]
        more = len(self.errors) - 1
        tail = f" (+{more} more)" if more else ""
        return f"{self.module} cannot run: {first.message}{tail}"

    def __str__(self) -> str:
        """The full report; see :func:`format_readiness`."""
        return format_readiness(self)


def _shape_problems(port: Port, paths: Sequence[str],
                    sample: int) -> List[Problem]:
    """Check the first ``sample`` arrays at a port against its shape contract."""
    problems: List[Problem] = []
    contract = port.shape
    for path in list(paths)[:sample]:
        try:
            header = read_npy_header(path)
        except (ValueError, OSError) as exc:
            problems.append(Problem(
                ERROR, port.role,
                f"{os.path.basename(path)} is not a readable .npy: {exc}",
                "Delete the damaged file and re-run the module that wrote it; "
                "a crash mid-write leaves exactly this."))
            continue
        shape = header["shape"]
        expected = header["expected_bytes"]
        if expected is not None and header["actual_bytes"] < expected:
            problems.append(Problem(
                ERROR, port.role,
                f"{os.path.basename(path)} is truncated: "
                f"{header['actual_bytes']} of {expected} bytes",
                "Delete it and re-run the module that wrote it."))
        elif contract.ndim is not None and len(shape) != contract.ndim:
            problems.append(Problem(
                ERROR, port.role,
                f"{os.path.basename(path)} has shape {shape}, expected "
                f"{contract.ndim} axes",
                f"This port needs {contract.describe()}. Check that "
                f"{port.relative()} holds what the previous step writes."))
        elif (contract.min_planes is not None and shape
                and shape[-1] < contract.min_planes):
            problems.append(Problem(
                ERROR, port.role,
                f"{os.path.basename(path)} has {shape[-1]} plane(s), at least "
                f"{contract.min_planes} are needed",
                "Re-run mask generation with the object channels you intend "
                "to measure: a merged array needs its image planes and at "
                "least one mask plane."))
        elif contract.dtype and _dtype_of(header) != contract.dtype:
            problems.append(Problem(
                ERROR, port.role,
                f"{os.path.basename(path)} is {_dtype_of(header)}, expected "
                f"{contract.dtype}",
                "Re-run the producing step; converting in place would change "
                "the numbers."))
    return problems


def _dtype_of(header: Mapping[str, Any]) -> str:
    """Return a ``.npy`` header's dtype without its byte-order prefix."""
    return str(header.get("descr", "")).lstrip("<>|=")


def _table_problems(port: Port, path: str) -> List[Problem]:
    """Check ``port.tables`` exist and hold rows in the database at ``path``."""
    from .database_concurrency import connect

    problems: List[Problem] = []
    try:
        connection = connect(path, readonly=True, timeout=5.0)
    except sqlite3.Error as exc:
        return [Problem(
            ERROR, port.role, f"{path} cannot be opened: {exc}",
            "Check the file is readable — on shared storage it is usually "
            "owned by whoever ran the previous step.")]
    try:
        present = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
    except sqlite3.DatabaseError as exc:
        connection.close()
        return [Problem(
            ERROR, port.role, f"{path} is not a SQLite database: {exc}",
            "Something else is sitting at that path. Move it aside and "
            "re-run the step that writes the database.")]
    try:
        for table in port.tables:
            if table not in present:
                problems.append(Problem(
                    ERROR, port.role,
                    f"{os.path.basename(path)} has no '{table}' table",
                    f"Run the step that writes '{table}' first. For png_list "
                    f"that is Measure with save_png on."))
            elif not connection.execute(
                    f'SELECT EXISTS(SELECT 1 FROM "{table}")').fetchone()[0]:
                problems.append(Problem(
                    ERROR, port.role,
                    f"'{table}' exists but is empty in "
                    f"{os.path.basename(path)}",
                    "The producing run created the table and wrote nothing to "
                    "it; re-run it and read its failure report."))
    finally:
        connection.close()
    return problems


def _where(resolved: ResolvedPort) -> str:
    """Render the places a port was looked for, readably.

    A pattern may carry several alternatives; naming all of them is what
    makes "no raw images" actionable, because the user learns that ``orig/``
    and ``consolidated/`` were searched too.
    """
    port = resolved.port
    if not port.pattern:
        return resolved.target
    return " or ".join(os.path.join(resolved.target, alternative)
                       for alternative in port.pattern.split("|"))


def _port_problems(resolved: ResolvedPort, *, sample: int) -> List[Problem]:
    """Every blocking problem with one resolved input port."""
    port = resolved.port
    producers = producers_of(port.kind)
    if not resolved.exists:
        fix = (f"Run {' or '.join(producers)} on this project first."
               if producers else
               f"Put the {port.kind} there, or point the source setting at "
               f"the folder that already has it.")
        return [Problem(ERROR, port.role,
                        f"no {port.kind} at {_where(resolved)}", fix)]
    if port.pattern and resolved.count < port.min_count:
        return [Problem(
            ERROR, port.role,
            f"only {resolved.count} {port.kind} at {_where(resolved)}, "
            f"{port.min_count} required",
            f"Re-run "
            f"{' or '.join(producers) or 'the producing step'} — it stopped "
            f"early or wrote somewhere else.")]
    problems: List[Problem] = []
    if port.shape is not None and resolved.paths:
        problems.extend(_shape_problems(port, resolved.paths, sample))
    if port.tables and os.path.isfile(resolved.target):
        problems.extend(_table_problems(port, resolved.target))
    return problems


def _registry_notes(registry: Any, resolved: ResolvedPort,
                    problems: List[Problem]) -> List[str]:
    """Attach provenance for one satisfied port; append staleness warnings."""
    artifact = registry.latest(resolved.kind, path=resolved.location)
    if artifact is None:
        return []
    staleness = registry.is_stale(artifact.artifact_id)
    if staleness.stale:
        problems.append(Problem(
            WARNING, resolved.role,
            f"the {resolved.kind} at {resolved.location} is stale: "
            f"{'; '.join(staleness.reasons)}",
            f"Re-run {artifact.module} before using it, or accept that this "
            f"result will not match its inputs."))
    return [artifact.artifact_id]


def check_ready(module: str,
                settings: Union[str, Mapping[str, Any], None] = None,
                *,
                root: str = "",
                registry: Any = None,
                sample: int = 3) -> Readiness:
    """Answer "can ``module`` run here?" before anything is loaded.

    Every declared input port is resolved against the project root and looked
    up on disk: present, plentiful enough, the right array shape, and — for a
    measurements database — carrying the tables the module reads. A missing
    *optional* port produces a warning instead of an error.

    When a :class:`spacr.artifacts.Registry` is supplied, each satisfied input
    is also matched against the registry, so the ids of the artifacts being
    consumed come back in :attr:`Readiness.inputs` and an input the registry
    reports as stale is added as a warning.

    :param module: module key or alias, as in
        :data:`spacr.validate.APP_FUNCTIONS`.
    :param settings: the settings dict about to be run, or a source path.
    :param root: explicit project root, overriding ``settings``.
    :param registry: optional :class:`spacr.artifacts.Registry`, for
        provenance and staleness.
    :param sample: how many arrays per port to check the shape of. Reading a
        ``.npy`` header is cheap but not free, and a bad run is bad from its
        first field.
    :returns: a :class:`Readiness`; ``bool(result)`` is the yes/no answer and
        :attr:`Readiness.reason` is the sentence to show a user.
    :raises UnknownModule: when nothing is declared for ``module``.
    """
    spec = module_ports(module)
    resolved_root = root or project_root(settings, spec.key)
    problems: List[Problem] = []
    satisfied: List[str] = []
    inputs: List[str] = []

    if not resolved_root:
        source_key = ROOT_KEYS.get(spec.key) or ALT_SRC_KEYS.get(spec.key, "src")
        return Readiness(
            module=spec.key, root="", ok=False,
            problems=(Problem(
                ERROR, "src", f"{spec.key} has no project folder to work in",
                f"Set '{source_key}' to the plate folder."),))
    if not os.path.isdir(resolved_root):
        return Readiness(
            module=spec.key, root=resolved_root, ok=False,
            problems=(Problem(
                ERROR, "src", f"{resolved_root} does not exist",
                "Check the path — a typo here costs the whole run."),))

    for port in spec.consumes:
        resolved = resolve_port(port, resolved_root)
        found = _port_problems(resolved, sample=sample)
        if found:
            problems.extend(
                found if port.required
                else [Problem(WARNING, p.setting, p.message, p.fix)
                      for p in found])
            continue
        satisfied.append(port.role)
        if registry is not None:
            inputs.extend(_registry_notes(registry, resolved, problems))

    return Readiness(module=spec.key, root=resolved_root,
                     ok=not any(p.is_error for p in problems),
                     problems=tuple(problems), satisfied=tuple(satisfied),
                     inputs=tuple(inputs))


def format_readiness(readiness: Readiness) -> str:
    """Render a :class:`Readiness` as a block of text for a user.

    :param readiness: the result of :func:`check_ready`.
    :returns: a multi-line report; errors first, each with its fix line.
    """
    verdict = "READY" if readiness.ok else "NOT READY"
    lines = [f"{verdict}: {readiness.module} in "
             f"{readiness.root or '(no project)'}"]
    if readiness.satisfied:
        lines.append(f"  inputs found: {', '.join(readiness.satisfied)}")
    for problem in readiness.errors:
        lines.append(f"  error   [{problem.setting}] {problem.message}")
        lines.append(f"          fix: {problem.fix}")
    for problem in readiness.warnings:
        lines.append(f"  warning [{problem.setting}] {problem.message}")
        lines.append(f"          fix: {problem.fix}")
    if readiness.inputs:
        lines.append(f"  artifacts: {', '.join(readiness.inputs)}")
    return "\n".join(lines)


def describe_ports(module: str) -> str:
    """Render one module's declared contract as text.

    :param module: module key or alias.
    :returns: a multi-line description of what it consumes and produces.
    :raises UnknownModule: when nothing is declared for ``module``.
    """
    spec = module_ports(module)
    lines = [f"{spec.key} — {spec.summary}" if spec.summary else spec.key]
    for label, ports in (("consumes", spec.consumes),
                         ("produces", spec.produces)):
        lines.append(f"  {label}:")
        if not ports:
            lines.append("    (nothing declared)")
            continue
        for port in ports:
            flag = "" if port.required else "  [optional]"
            lines.append(
                f"    {port.role}: {port.kind} at {port.relative()}{flag}")
            if port.description:
                lines.append(f"      {port.description}")
            if port.shape is not None:
                lines.append(f"      shape: {port.shape.describe()}")
            if port.tables:
                lines.append(f"      tables: {', '.join(port.tables)}")
    return "\n".join(lines)
