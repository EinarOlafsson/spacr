"""The work behind the OPS Live preview (alpha).

The OPS Live preview has one tab per pipeline step: align and stitch, spots
and genotyping, objects and phenotype, merge and aggregation, and clustering.
Each tab runs only its own step, on a small sample, with the settings on the
form. It calls the same spaCR functions a run calls, and it writes only into
its own workspace under ``~/.cache/spacr/ops_live``, never into the run's
output folder. With no sequencing folder set, it uses the OPS end-to-end test
data, which holds both sequencing and phenotype images of the same sites.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from .ops_settings import ops_setting

import numpy as np

_STEPS: Tuple[str, ...] = ("align", "spots", "objects", "merge", "cluster",
                           "voyager3")

_STEP_NEEDS: Dict[str, Tuple[str, ...]] = {
    "align": (),
    "spots": ("align",),
    "objects": ("align",),
    "merge": ("align",),
    "cluster": ("merge",),
    "voyager3": ("cluster",),
}

_EVERY_STEP = _STEPS

_SHARED_CATEGORY_WORDS: Tuple[str, ...] = ("input", "performance")

_STEM_STEPS: Tuple[Tuple[str, str], ...] = (
    ("ops_voyager3_", "voyager3"),
    ("ops_registration_", "align"),
    ("ops_channel_registration", "align"),
    ("ops_spot_min_peak", "align"),
    ("ops_spot_overrule_share", "align"),
    ("ops_raster_", "align"),
    ("ops_stitch_", "align"),
    ("ops_placement_", "align"),
    ("ops_ph_placement_", "align"),
    ("ops_align_", "align"),
    ("ops_sites", "align"),
    ("ops_compose_", "align"),
    ("cellpose_", "spots"),
    ("ops_sbs_nucleus_model", "spots"),
    ("ops_objects_", "spots"),
    ("ops_window_", "spots"),
    ("ops_merge", "merge"),
    ("ops_aggregate", "merge"),
    ("ops_design", "merge"),
    ("ops_control_prefix", "merge"),
    ("ops_bootstrap", "merge"),
    ("ops_pool_", "merge"),
    ("ops_nt_rows", "merge"),
    ("ops_feature_qc", "merge"),
    ("ops_genotype_table", "merge"),
    ("ops_test_features", "merge"),
    ("ops_pvalue", "merge"),
    ("ops_cluster", "cluster"),
    ("ops_profile", "cluster"),
    ("ops_tvn", "cluster"),
    ("ops_consensus", "cluster"),
    ("ops_modules", "cluster"),
    ("ops_scoring", "cluster"),
    ("ops_bench", "cluster"),
    ("ops_etest", "cluster"),
    ("ops_eval", "cluster"),
    ("ops_ph_", "objects"),
    ("ops_flatfield", "objects"),
)

_KEY_STEPS: Dict[str, Tuple[str, ...]] = {
    "ops_cellpose_model": ("spots", "objects"),
}
"""Settings more than one tab reads: the stock Cellpose-SAM weights segment
the sequencing nuclei (spots) and the phenotype objects (item 975)."""

_CATEGORY_STEPS: Tuple[Tuple[str, str], ...] = (
    ("cluster", "cluster"),
    ("profil", "cluster"),
    ("embedding", "cluster"),
    ("module", "cluster"),
    ("scor", "cluster"),
    ("merg", "merge"),
    ("aggregat", "merge"),
    ("segment", "objects"),
    ("object", "objects"),
    ("measure", "objects"),
    ("flat", "objects"),
    ("crop", "objects"),
    ("phenotyp", "objects"),
    ("decod", "spots"),
    ("decoy", "spots"),
    ("prefix", "spots"),
    ("pixel fallback", "spots"),
    ("call quality", "spots"),
    ("genotyp", "spots"),
    ("spot", "spots"),
    ("barcode", "spots"),
    ("read", "spots"),
    ("sequenc", "spots"),
    ("align", "align"),
    ("stitch", "align"),
    ("placement", "align"),
    ("registration", "align"),
)

_PATH_KEYS: Tuple[str, ...] = ("genotype_source", "phenotype_source",
                               "ops_library", "ops_flatfield_maps",
                               "ops_ph_placement_seed", "ops_design")

_LIVE_STEPS: Dict[str, Tuple[str, ...]] = {
    "ops_live_well": _STEPS,
    "ops_live_region": ("align",),
    "ops_live_block": ("align",),
    "ops_live_field": ("align",),
    "ops_live_sbs_crop": ("spots", "objects"),
    "ops_live_ph_fields": ("objects",),
    "ops_live_merge_fields": ("merge",),
    "ops_live_genes": ("cluster",),
    "ops_live_voyager3_clusters": ("voyager3",),
}
"""The preview-only settings (the region and the Sample of each tab) and
the steps each one changes. They are never written into the OPS run."""

_MAX_REGIONS = 24

_TEST_FOLDER = "ops_testdata_funk2022_A1"

_SBS_CROP = 400

_PH_CROP = 512

_MOSAIC_STEP = 8

_OBJECT_LAYERS: List[Callable[..., Dict[str, np.ndarray]]] = []
"""Extra object masks Objects & phenotype draws, for item 970.

Each entry is called as ``layer(settings, stack, channels, nuclei, cells)``
with the phenotype crop and returns ``{name: label mask}``; every mask is
outlined on the crop beside the nuclei and the cells. Empty by default."""


class _Cancelled(Exception):
    """The pass was cancelled between two of its steps."""


class _NotReady(ValueError):
    """A tab's input is not there yet; the tab says what to run first.

    :param reason: what is missing: ``"run"`` (no finished run in
        ``dst_root``), ``"rows"`` (the run lacks a table), ``"fields"`` (no
        phenotype field is placed), ``"phenotype"`` (no phenotype folder),
        ``"genes"`` (no gene table) or ``"few_genes"`` (too few genes to
        cluster).
    :param detail: the values the tab's sentence names, such as a path.
    """

    def __init__(self, reason: str, **detail: Any):
        """Keep the reason and its details beside the English message."""
        self.reason = str(reason)
        self.detail = dict(detail)
        super().__init__(f"{self.reason}: " + ", ".join(
            f"{key}={value}" for key, value in sorted(self.detail.items())))


def _steps_of(key: str, category: str) -> Tuple[str, ...]:
    """The tabs a setting belongs to, read from its registry category.

    :param key: the setting.
    :param category: its category title in the settings registry.
    :returns: one step, or every step for an input or performance setting
        and for a setting no rule places.
    """
    if key in _KEY_STEPS:
        return _KEY_STEPS[key]
    name = str(category or "").lower()
    if any(word in name for word in _SHARED_CATEGORY_WORDS):
        return _EVERY_STEP
    for stem, step in _STEM_STEPS:
        if key.startswith(stem):
            return (step,)
    for word, step in _CATEGORY_STEPS:
        if word in name:
            return (step,)
    return _EVERY_STEP


def _registry_categories(app_key: str = "ops") -> Dict[str, str]:
    """Each setting of a module and the category the registry files it in.

    :param app_key: the module.
    :returns: ``{key: category title}``; empty when the registry cannot be
        read.
    """
    try:
        from .settings import categories as registered

        from .ops_settings import OPS_DEFAULTS
        try:
            from .qt.screens.settings_model import categories_for_app
            found = categories_for_app(app_key, registered)
        except Exception:
            found = dict(registered)
    except Exception:
        return {}
    keys = set(OPS_DEFAULTS)
    out: Dict[str, str] = {}
    for title, members in found.items():
        for key in members:
            if key in keys and key not in out:
                out[key] = str(title)
    return out


def _step_map(app_key: str = "ops") -> Dict[str, Tuple[str, ...]]:
    """Every OPS setting and the tabs it refreshes.

    :param app_key: the module whose registry categories are read.
    :returns: ``{key: steps}``, every key of the OPS defaults included.
    """
    from .ops_settings import OPS_DEFAULTS

    categories = _registry_categories(app_key)
    out = {key: _steps_of(key, categories.get(key, ""))
           for key in OPS_DEFAULTS}
    out.update(_LIVE_STEPS)
    return out


def _closure(step: str) -> Tuple[str, ...]:
    """The step and every step it reads from, upstream first."""
    seen: List[str] = []

    def visit(name: str) -> None:
        """Add ``name`` after what it needs."""
        for need in _STEP_NEEDS[name]:
            visit(need)
        if name not in seen:
            seen.append(name)
    visit(step)
    return tuple(seen)


def _downstream(step: str) -> Tuple[str, ...]:
    """Every step that reads, directly or not, from ``step``."""
    return tuple(name for name in _STEPS
                 if name != step and step in _closure(name))


def _signature(settings: Mapping[str, Any], step: str,
               steps: Mapping[str, Tuple[str, ...]]) -> str:
    """A digest of the settings one step and its inputs depend on.

    :param settings: the effective settings.
    :param step: the step.
    :param steps: from :func:`_step_map`.
    :returns: a hex digest that changes when any of those settings does.
    """
    wanted = set(_closure(step))
    picked = {key: settings.get(key) for key, owners in sorted(steps.items())
              if wanted & set(owners)}
    picked["_live_root"] = settings.get("_live_root")
    text = json.dumps(picked, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


_REGIONS_TEST_FOLDER = "ops_testdata_funk2022_A1_regions"

_PROVENANCE_FILES: Tuple[str, ...] = ("PROVENANCE_rev2_addendum.json",
                                      "PROVENANCE_regions.json")


def _is_test_root(folder: Path) -> bool:
    """Whether ``folder`` is an unpacked tier of the OPS end-to-end test data."""
    return (folder / "settings" / "ops_testdata_settings.json").is_file() and any(
        (folder / "reference" / name).is_file() for name in _PROVENANCE_FILES)


def _test_data_root() -> Optional[Path]:
    """The unpacked OPS end-to-end test data, or None.

    The regions tier first (three regions to choose between), then the full
    tier, then the core tier.
    """
    from .example_archives import (OPS_E2E_FULL_FOLDER, OPS_E2E_REGIONS_FOLDER,
                                   example_set_folder)

    core = example_set_folder("ops_e2e")
    for root in (core.parent / OPS_E2E_REGIONS_FOLDER / _REGIONS_TEST_FOLDER,
                 core.parent / OPS_E2E_FULL_FOLDER / _TEST_FOLDER,
                 core / _TEST_FOLDER):
        if _is_test_root(root):
            return root
    return None


def _test_root_of(source: Any) -> Optional[Path]:
    """The OPS test data folder ``source`` lies in, or None.

    A ``genotype_source`` pointing into the downloaded test data is the test
    data, whatever else the form says, so the tabs that read a finished run
    use the stored step outputs the test data carries.

    :param source: a folder, usually the form's ``genotype_source``.
    """
    try:
        folder = Path(str(source)).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for candidate in (folder, *folder.parents):
        if _is_test_root(candidate):
            return candidate
    return None


def _test_settings(root: Path) -> Dict[str, Any]:
    """The test data's shipped settings, with its paths made absolute.

    :param root: the test data folder.
    :returns: the settings, plus the placement seed and the guide design the
        shipped file leaves for item 940 and for the aggregate phase.
    """
    with open(root / "settings" / "ops_testdata_settings.json",
              encoding="utf-8") as handle:
        settings = dict(json.load(handle))
    settings["ops_ph_placement_seed"] = (
        "reference/inputs/ops_ph_placement_seed.json")
    if not settings.get("ops_design"):
        settings["ops_design"] = "library/pool10_design.csv"
    for key in _PATH_KEYS:
        value = settings.get(key)
        if value and not os.path.isabs(str(value)):
            settings[key] = str(root / str(value))
    return settings


def _same(first: Any, second: Any) -> bool:
    """Whether a form value says the same as a default.

    None, an empty string and an empty list all mean "not set"; anything
    else compares by its text, so ``2`` and ``"2"`` are the same.
    """
    def empty(value: Any) -> bool:
        """Whether ``value`` is a way of leaving a setting unset."""
        return value is None or (isinstance(value, (str, list, tuple))
                                 and not str(value).strip("[]() '\""))
    if empty(first) or empty(second):
        return empty(first) and empty(second)
    return first == second or str(first) == str(second)


def _effective_settings(form: Mapping[str, Any]
                        ) -> Tuple[Dict[str, Any], Optional[Path]]:
    """The settings the preview runs with, and the test data root if used.

    :param form: the settings on the OPS form.
    :returns: ``(settings, test root)``. With a sequencing folder on the form
        the form is used as it is; the test root is the test data folder the
        sequencing folder lies in, if any. Without one the test data's shipped
        settings are the base and every form value that differs from the OPS
        default is laid over them, so an edit on the form is what the preview
        tests.
    """
    from .ops_settings import OPS_DEFAULTS

    from .settings import _fold_renamed_settings

    form = _fold_renamed_settings(dict(form or {}))
    source = form.get("genotype_source")
    if source and os.path.isdir(str(source)):
        return form, _test_root_of(source)
    root = _test_data_root()
    if root is None:
        return form, None
    settings = _test_settings(root)
    for key, value in form.items():
        if key in _PATH_KEYS and not value:
            continue
        if key in OPS_DEFAULTS and not _same(value, OPS_DEFAULTS[key]):
            settings[key] = value
        elif key not in OPS_DEFAULTS and key not in settings:
            settings[key] = value
    return _fold_renamed_settings(settings), root


def _workspace(settings: Mapping[str, Any]) -> Path:
    """The preview's own folder for these inputs.

    Each region of a well gets its own folder, so one region's run tables
    and masks never stand in for another's.

    :param settings: read for the two image folders, the plate and the
        region (``ops_live_region``, ``ops_live_block``).
    :returns: ``~/.cache/spacr/ops_live/<digest>``, created.
    """
    text = "|".join(str(settings.get(key) or "") for key in
                    ("genotype_source", "phenotype_source", "plate"))
    region = _live_int(settings, "ops_live_region", 0)
    side = _live_int(settings, "ops_live_block", 2)
    if region != 0 or side != 2:
        text += f"|region={region}|side={side}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    folder = Path.home() / ".cache" / "spacr" / "ops_live" / digest
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _preview_settings(settings: Mapping[str, Any], folder: Path
                      ) -> Dict[str, Any]:
    """The settings a preview phase runs with.

    :param settings: the effective settings.
    :param folder: the preview workspace.
    :returns: a copy writing into ``folder`` with one worker; every other
        setting is the form's.
    """
    out = dict(settings)
    out["dst_root"] = str(folder)
    out["n_workers"] = 1
    out["ops_gpu_admission"] = False
    out["ops_archive_root"] = None
    out["ops_stress_report"] = False
    return out


def _check(cancelled: Optional[Callable[[], bool]]) -> None:
    """Stop the pass here when it was cancelled.

    :param cancelled: answers whether the pass was cancelled.
    :raises _Cancelled: when it was.
    """
    if cancelled is not None and cancelled():
        raise _Cancelled()


def _tiles(cache: dict, root: str, scheme: str):
    """The engine's tile index of ``root``, read once per folder and scheme.

    :param cache: the panel's cache, which keeps the index.
    :param root: the acquisition folder.
    :param scheme: ``"cycled"`` or ``"uncycled"``.
    :returns: :func:`spacr.ops_engine._index_tiles`' answer.
    """
    from .ops_engine import _index_tiles

    held = cache.setdefault("_index", {})
    if (root, scheme) not in held:
        held[(root, scheme)] = (_index_tiles(root, scheme)
                                if os.path.isdir(root) else {})
    return held[(root, scheme)]


def _well_of(settings: Mapping[str, Any], cache: dict) -> str:
    """The well the preview shows: ``ops_live_well``, else the first one."""
    root = str(settings.get("genotype_source") or "")
    index = _tiles(cache, root, "cycled")
    if not index:
        raise ValueError("genotype_source holds no sequencing tiles")
    asked = str(settings.get("ops_live_well") or "").upper()
    return asked if asked in index else sorted(index)[0]


def _sample_blocks(index_well: Mapping[int, Mapping[int, Any]],
                   reference: int, side: int = 2,
                   limit: int = _MAX_REGIONS) -> List[List[int]]:
    """The regions of a well: square blocks of sequencing tiles, middle first.

    Blocks do not share a tile. A well with no more tiles than one block
    holds is one region of all its tiles.

    :param index_well: ``cycle -> site -> files`` of the well.
    :param reference: the cycle holding the nuclear stain.
    :param side: tiles along each side of a block.
    :param limit: the most regions returned.
    :returns: each region as its sorted sites, every one in the reference cycle.
    """
    from .ops_layout import round_well_layout

    side = max(1, int(side))
    sites = sorted(int(s) for s in index_well[reference])
    if len(sites) <= side * side:
        return [sites]
    layout = round_well_layout(max(sites) + 1)
    where = {site: layout.position(site) for site in sites}
    centre = np.mean(np.array(list(where.values()), float), axis=0)
    by_place = {place: site for site, place in where.items()}
    ranked = sorted(sites, key=lambda s: float(np.hypot(*(np.array(where[s]) - centre))))
    used: set = set()
    blocks: List[List[int]] = []
    for site in ranked:
        column, row = where[site]
        block = [by_place.get((column + dc, row + dr))
                 for dc in range(side) for dr in range(side)]
        if all(b is not None and b not in used for b in block):
            blocks.append(sorted(int(b) for b in block))
            used.update(block)
            if len(blocks) >= limit:
                break
    return blocks or [ranked[:side * side]]


def _sample_block(index_well: Mapping[int, Mapping[int, Any]],
                  reference: int) -> List[int]:
    """A 2 x 2 block of sequencing tiles near the middle of the well.

    :param index_well: ``cycle -> site -> files`` of the well.
    :param reference: the cycle holding the nuclear stain.
    :returns: the sites, every one in the reference cycle.
    """
    return _sample_blocks(index_well, reference)[0]


def _live_int(settings: Mapping[str, Any], key: str, fallback: int) -> int:
    """A preview-only whole-number setting, or ``fallback`` when unset or unreadable."""
    try:
        value = settings.get(key)
        return int(value) if value not in (None, "") else int(fallback)
    except (TypeError, ValueError):
        return int(fallback)


@dataclass
class _Context:
    """What every tab of one preview pass shares.

    :ivar settings: the effective settings, before the workspace overrides.
    :ivar test_root: the test data folder when it is the input, else None.
    :ivar folder: the preview workspace.
    :ivar well: the well shown.
    :ivar steps: ``{setting: steps}`` from the registry.
    :ivar cache: ``{step: (signature, result)}``, shared across passes.
    """

    settings: Dict[str, Any]
    test_root: Optional[Path]
    folder: Path
    well: str
    steps: Dict[str, Tuple[str, ...]]
    cache: Dict[str, Tuple[str, Any]] = field(default_factory=dict)
    planes: Dict[Tuple[str, Optional[int]], np.ndarray] = field(default_factory=dict)


def _context(form: Mapping[str, Any], cache: Optional[dict] = None,
             steps: Optional[Dict[str, Tuple[str, ...]]] = None) -> _Context:
    """Resolve the inputs of a preview pass.

    :param form: the settings on the form.
    :param cache: the cache of earlier passes, reused.
    :param steps: the setting-to-step map, read from the registry when None.
    :returns: the context.
    :raises ValueError: when there is no sequencing input and no test data.
    """
    settings, root = _effective_settings(form)
    if not settings.get("genotype_source"):
        raise ValueError("no genotype_source and no OPS test data")
    folder = _workspace(settings)
    settings["_live_root"] = str(root or settings.get("genotype_source"))
    cache = cache if cache is not None else {}
    return _Context(settings=settings, test_root=root, folder=folder,
                    well=_well_of(settings, cache), steps=steps or _step_map(),
                    cache=cache)


def _read(ctx: _Context, source) -> Optional[np.ndarray]:
    """One plane through the engine's reader, kept for later passes."""
    from .ops_engine import _read_plane

    key = (str(source[0]), source[1]) if source else ("", None)
    if key not in ctx.planes:
        if len(ctx.planes) > 96:
            ctx.planes.clear()
        ctx.planes[key] = _read_plane(source, [])
    return ctx.planes[key]


def _result(ctx: _Context, step: str,
            cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """A step's result, from the cache when its settings have not changed.

    :param ctx: the pass context.
    :param step: the step.
    :param cancelled: answers whether the pass was cancelled.
    :returns: the step's result.
    """
    signature = _signature(ctx.settings, step, ctx.steps)
    held = ctx.cache.get(step)
    if held is not None and held[0] == signature:
        return held[1]
    for need in _STEP_NEEDS[step]:
        _result(ctx, need, cancelled)
    _check(cancelled)
    started = time.perf_counter()
    out = _RUNNERS[step](ctx, cancelled)
    out["seconds"] = round(time.perf_counter() - started, 2)
    ctx.cache[step] = (signature, out)
    return out


def _model(ctx: _Context, key: str = "ops_sbs_nucleus_model"):
    """The Cellpose model a setting names, built once and kept.

    :param ctx: the pass context; the model is kept in its cache.
    :param key: the setting naming the model.
    :returns: :func:`spacr.ops_engine._cellpose_model`'s model.
    """
    from .ops_engine import _cellpose_model

    gpu = bool(ops_setting(ctx.settings, "ops_gpu"))
    name = (str(ctx.settings.get(key) or ""), gpu,
            ctx.settings.get("mask_inference_precision"),
            str(ops_setting(ctx.settings, "ops_cellpose_model")))
    held = ctx.cache.setdefault("_models", {})
    if name not in held:
        if len(held) > 3:
            held.clear()
        held[name] = _cellpose_model(ctx.settings, gpu, key)
    return held[name]


def _sbs_index(ctx: _Context) -> Tuple[Dict[int, Dict[int, Any]], int]:
    """The well's sequencing tiles and its nuclear cycle."""
    from .ops_engine import _NUCLEAR, _plane_sources

    index = _tiles(ctx.cache, str(ctx.settings["genotype_source"]),
                   "cycled")[ctx.well]
    nuclear = [cycle for cycle in sorted(index)
               if any(_NUCLEAR in _plane_sources(files)
                      for files in index[cycle].values())]
    return index, nuclear[0]


def _ph_index(ctx: _Context) -> Dict[int, Dict[str, str]]:
    """The well's phenotype fields, ``site -> channel -> path``."""
    root = ctx.settings.get("phenotype_source")
    if not root or not os.path.isdir(str(root)):
        return {}
    index = (_tiles(ctx.cache, str(root), "uncycled")
             or _tiles(ctx.cache, str(root), "cycled"))
    per_cycle = index.get(ctx.well) or {}
    return per_cycle.get(min(per_cycle), {}) if per_cycle else {}


def _rows(db: str, table: str, plate: str, well: str):
    """One well's rows of an OPS table, or None."""
    from .ops_engine import _well_rows
    from .ops_store import _exists

    if not _exists(db):
        return None
    return _well_rows(db, table, plate, well)


def _downsample(plane: np.ndarray, step: int) -> np.ndarray:
    """``plane`` block-averaged by ``step`` with the engine's own reduction."""
    from .ops_engine import _ph_downscale

    return _ph_downscale(np.asarray(plane, np.float32), step)


def _run_align(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Stitch the sample block, place the phenotype fields, align one crop.

    Calls :func:`spacr.ops_engine.run_ops` with the ``stitch`` phase (and
    ``phenotype`` when a phenotype folder is set) into the preview
    workspace, then :func:`spacr.ops_cycles.align_field` on one crop with
    the decode's registration options.

    :returns: the tile placements, the mosaic, the field placements, the
        crop and its cycle shifts. The tile placements are each tile's
        origin under ``ops_tile_origin`` (:func:`spacr.ops_compose.tile_origins`),
        so a crop of a tile's own pixels and a well-frame point agree as
        they do in the decode.
    """
    from .ops_compose import tile_origin_rule, tile_origins
    from .ops_cycles import align_field
    from .ops_engine import (_base_channels, _NUCLEAR, _plane_sources,
                             _registration_choice, _setting_number, run_ops)

    from .ops_engine import _ops_sites

    index, reference = _sbs_index(ctx)
    asked = _ops_sites(ctx.settings)
    regions = _sample_blocks(index, reference,
                             side=_live_int(ctx.settings, "ops_live_block", 2))
    region = min(max(0, _live_int(ctx.settings, "ops_live_region", 0)),
                 len(regions) - 1)
    block = (sorted(asked)[:16] if asked else regions[region])
    settings = _preview_settings(ctx.settings, ctx.folder)
    whole = set(block) == set(int(s) for s in index[reference])
    settings["ops_sites"] = None if whole else ",".join(str(s) for s in block)
    phases = ["stitch"]
    notes: List[str] = []
    if settings.get("phenotype_source"):
        phases.append("phenotype")
    try:
        report = run_ops(settings, wells=[ctx.well], phases=phases)
    except ValueError as error:
        if "phenotype" not in phases:
            raise
        notes.append(f"phenotype placement: {error}")
        report = run_ops(settings, wells=[ctx.well], phases=["stitch"])
    _check(cancelled)
    db = str(report["db"])
    plate = str(report.get("plate") or settings.get("plate") or "")
    geometry = _rows(db, "ops_geometry", plate, ctx.well)
    placed = _rows(db, "ops_phenotype", plate, ctx.well)
    if geometry is None or geometry.empty:
        raise ValueError("the stitch placed no tile of the sample block")
    geometry = geometry[geometry["cycle"] == reference]
    tiles = tile_origins({int(r.site): (float(r.y), float(r.x))
                          for r in geometry.itertuples()},
                         tile_origin_rule(ops_setting(ctx.settings,
                                                      "ops_tile_origin")))
    shape = (int(geometry["tile_height"].iloc[0]),
             int(geometry["tile_width"].iloc[0]))
    top = min(y for y, _x in tiles.values())
    left = min(x for _y, x in tiles.values())
    bottom = max(y for y, _x in tiles.values()) + shape[0]
    right = max(x for _y, x in tiles.values()) + shape[1]
    step = _MOSAIC_STEP
    mosaic = np.zeros((int((bottom - top) // step) + 1,
                       int((right - left) // step) + 1), np.float32)
    for site, (y, x) in tiles.items():
        dapi = _read(ctx, _plane_sources(index[reference][site]).get(_NUCLEAR))
        _check(cancelled)
        if dapi is None:
            continue
        small = _downsample(dapi, step)
        oy, ox = int((y - top) // step), int((x - left) // step)
        h = min(small.shape[0], mosaic.shape[0] - oy)
        w = min(small.shape[1], mosaic.shape[1] - ox)
        mosaic[oy:oy + h, ox:ox + w] = np.maximum(mosaic[oy:oy + h, ox:ox + w],
                                                  small[:h, :w])
    fields: List[Dict[str, Any]] = []
    if placed is not None and not placed.empty:
        ph_files = _ph_index(ctx)
        for row in placed.itertuples():
            site = int(row.site)
            centre = (float(row.centre_y), float(row.centre_x))
            if not (top - shape[0] < centre[0] < bottom + shape[0]
                    and left - shape[1] < centre[1] < right + shape[1]):
                continue
            fields.append({"site": site, "centre": centre,
                           "source": str(getattr(row, "transform_source", "") or ""),
                           "scale": float(row.scale),
                           "rotation": float(row.rotation_deg),
                           "shape": (int(row.field_height), int(row.field_width)),
                           "files": ph_files.get(site)})
    sample_field = _sample_field(ctx, fields)
    crop_site, crop = _sbs_crop(tiles, shape, sample_field,
                                ctx.test_root is not None)
    tile_sources = {site: _plane_sources(index[reference][site]).get(_NUCLEAR)
                    for site in tiles}
    channels = _base_channels(ctx.settings)
    planes: Dict[int, Optional[list]] = {}
    raw: Dict[int, Optional[list]] = {}
    y0, x0, size = crop
    for cycle in sorted(index):
        sources = _plane_sources(index[cycle].get(crop_site, {}))
        got = [_read(ctx, sources.get(name)) for name in channels]
        if any(plane is None for plane in got):
            planes[cycle] = None
            continue
        planes[cycle] = [np.ascontiguousarray(p[y0:y0 + size, x0:x0 + size])
                         for p in got]
        raw[cycle] = planes[cycle]
    _check(cancelled)
    registration = {
        "refine_px": int(_setting_number(
            ctx.settings.get("ops_registration_refine_px"),
            "ops_registration_refine_px", 3)),
        "spot_min_peak": float(_setting_number(
            ctx.settings.get("ops_spot_min_peak"), "ops_spot_min_peak", 20.0)),
        "spot_overrule_share": float(_setting_number(
            ctx.settings.get("ops_spot_overrule_share"),
            "ops_spot_overrule_share", 0.5)),
        "precision": _registration_choice(
            ctx.settings.get("ops_registration_precision"),
            "ops_registration_precision"),
        "backend": _registration_choice(
            ctx.settings.get("ops_registration_backend"),
            "ops_registration_backend"),
    }
    from .ops_engine import _CHANNEL_REGISTRATION

    aligned = align_field(
        planes, reference=reference, gpu=bool(ops_setting(ctx.settings, "ops_gpu")),
        channel_mode=str(ctx.settings.get("ops_channel_registration")
                         or _CHANNEL_REGISTRATION).strip().lower(),
        **registration)
    last = aligned.kept[-1]
    before = (np.mean(raw[reference], axis=0), np.mean(raw[last], axis=0))
    after = (aligned.stack[0].mean(axis=0), aligned.stack[-1].mean(axis=0))
    seam = _seam(ctx, index, reference, tiles, shape)
    _check(cancelled)
    sbs_dapi = _read(ctx, _plane_sources(index[reference][crop_site]).get(_NUCLEAR))
    placement = (_placement(ctx, sample_field, tiles, crop_site, crop,
                            sbs_dapi[y0:y0 + size, x0:x0 + size])
                 if sbs_dapi is not None else None)
    placed_after = (_tile_affine_placement(ctx, plate, sample_field, tiles, crop_site,
                                           crop, sbs_dapi[y0:y0 + size, x0:x0 + size])
                    if sbs_dapi is not None else None)
    return {
        "db": db, "plate": plate, "reference": reference, "block": block,
        "tiles": tiles, "tile_shape": shape, "origin": (top, left),
        "mosaic": mosaic, "step": step, "fields": fields,
        "sample_field": sample_field, "crop_site": crop_site, "crop": crop,
        "cycle_shifts": {int(k): tuple(v) for k, v in aligned.cycle_shifts.items()},
        "kept": list(aligned.kept), "refused": list(aligned.refused),
        "missing": list(aligned.missing), "last_cycle": int(last),
        "before": before, "after": after, "notes": notes,
        "seam": seam, "placement": placement, "placed_after": placed_after,
        "phases": phases, "sample_block": settings["ops_sites"],
        "regions": [list(r) for r in regions], "region": region,
        "tile_sources": tile_sources,
    }


def _sample_field(ctx: _Context, fields: Sequence[Mapping[str, Any]]
                  ) -> Optional[Dict[str, Any]]:
    """The placed phenotype field the preview samples.

    ``ops_live_field`` when it names a placed field; else the test data's
    field 1264, on the block's four-tile junction; else the field nearest
    the middle of the block.
    """
    usable = [f for f in fields if f.get("files")]
    if not usable:
        return None
    asked = ctx.settings.get("ops_live_field")
    for wanted in (asked, 1264 if ctx.test_root else None):
        for one in usable:
            if wanted is not None and int(one["site"]) == int(wanted):
                return dict(one)
    middle = np.mean([f["centre"] for f in usable], axis=0)
    return dict(min(usable, key=lambda f: float(np.hypot(
        *(np.asarray(f["centre"]) - middle)))))


def _ph_crop_window(field_info: Mapping[str, Any], test: bool
                    ) -> Tuple[int, int, int]:
    """Where the phenotype crop sits in its field: ``(row, column, size)``.

    The test data's core tier holds pixels only inside a region of interest,
    rows 952 to 2152 and columns 0 to 1200 of field 1264, so its crop is
    centred there; any other field is cropped at its middle.
    """
    height, width = field_info["shape"]
    size = min(_PH_CROP, height, width)
    if test and int(field_info["site"]) == 1264:
        centre = (1552, 600)
    else:
        centre = (height // 2, width // 2)
    row = int(np.clip(centre[0] - size // 2, 0, height - size))
    column = int(np.clip(centre[1] - size // 2, 0, width - size))
    return row, column, size


def _ph_to_well(field_info: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    """Field ``(row, column)`` points in the well frame, as merge maps them."""
    from .ops_merge import field_to_well, field_transform

    transform = field_transform(field_info["scale"], field_info["rotation"],
                                field_info["centre"], field_info["shape"])
    return field_to_well(np.asarray(points, float).reshape(-1, 2), transform)


def _sbs_crop(tiles: Mapping[int, Tuple[float, float]], shape: Tuple[int, int],
              field_info: Optional[Mapping[str, Any]], test: bool = False,
              size: int = _SBS_CROP) -> Tuple[int, Tuple[int, int, int]]:
    """The sequencing tile and crop under the sampled phenotype crop.

    :param tiles: ``site -> (y, x)`` placements of the reference cycle.
    :param shape: the tile shape.
    :param field_info: the sampled phenotype field, or None.
    :param test: the input is the test data (see :func:`_ph_crop_window`).
    :param size: the side of the crop in sequencing pixels.

    :returns: ``(site, (row, column, size))`` in the tile's own pixels; the
        middle of the most central tile when no field is placed.
    """
    size = int(min(max(16, int(size)), shape[0], shape[1]))
    if field_info is not None:
        row, column, ph_size = _ph_crop_window(field_info, test)
        target = _ph_to_well(field_info, [[row + ph_size / 2,
                                           column + ph_size / 2]])[0]
    else:
        centre = np.mean([[y + shape[0] / 2, x + shape[1] / 2]
                          for y, x in tiles.values()], axis=0)
        target = centre

    def inside(site: int) -> float:
        """How far ``target`` lies inside the tile, at its nearest edge."""
        y, x = tiles[site]
        return min(target[0] - y, y + shape[0] - target[0],
                   target[1] - x, x + shape[1] - target[1])
    site = max(tiles, key=inside)
    y, x = tiles[site]
    row = int(np.clip(target[0] - y - size / 2, 0, shape[0] - size))
    column = int(np.clip(target[1] - x - size / 2, 0, shape[1] - size))
    return int(site), (row, column, size)


def _genotyped_crop(ctx: _Context, align: Mapping[str, Any]
                    ) -> Tuple[int, Tuple[int, int, int]]:
    """The sequencing crop Spots & genotyping reads: the tile and ``(row, column, size)``.

    Centred where the Align & stitch crop is, at the side the Spots Sample
    asks for (``ops_live_sbs_crop``).
    """
    size = _live_int(ctx.settings, "ops_live_sbs_crop", _SBS_CROP)
    if size == int(align["crop"][2]):
        return int(align["crop_site"]), tuple(align["crop"])
    return _sbs_crop(align["tiles"], align["tile_shape"], align["sample_field"],
                     ctx.test_root is not None, size=size)


def _well_to_field(field_info: Mapping[str, Any], points) -> np.ndarray:
    """Well-frame ``(y, x)`` points in a phenotype field, the inverse of :func:`_ph_to_well`."""
    from .ops_merge import field_transform

    scale, rotation, translation = field_transform(
        field_info["scale"], field_info["rotation"], field_info["centre"],
        field_info["shape"])
    well = np.asarray(points, float).reshape(-1, 2) - np.asarray(translation, float)
    return (np.asarray(rotation, float).T @ well.T).T / float(scale)


def _region_windows(align: Mapping[str, Any], site: int,
                    crop: Tuple[int, int, int], limit: int = 9
                    ) -> List[Tuple[Dict[str, Any], Tuple[int, int, int, int]]]:
    """Every placed phenotype field over a sequencing crop, and the part of it the crop covers.

    :param align: the Align & stitch result.
    :param site: the crop tile.
    :param crop: ``(row, column, size)`` in the tile.
    :param limit: the most fields returned, largest part first.
    :returns: ``(field, (row, column, height, width))`` in field pixels.
    """
    ty, tx = align["tiles"][site]
    row, column, size = crop
    corners = np.array([[ty + row, tx + column], [ty + row, tx + column + size],
                        [ty + row + size, tx + column + size],
                        [ty + row + size, tx + column]], float)
    out = []
    for info in align["fields"]:
        if not info.get("files"):
            continue
        height, width = info["shape"]
        inside = _well_to_field(info, corners)
        top = int(np.clip(np.floor(inside[:, 0].min()), 0, height))
        bottom = int(np.clip(np.ceil(inside[:, 0].max()), 0, height))
        left = int(np.clip(np.floor(inside[:, 1].min()), 0, width))
        right = int(np.clip(np.ceil(inside[:, 1].max()), 0, width))
        if bottom - top >= 32 and right - left >= 32:
            out.append((dict(info), (top, left, bottom - top, right - left)))
    out.sort(key=lambda pair: -pair[1][2] * pair[1][3])
    return out[:max(1, int(limit))]


_PLANES: Dict[str, np.ndarray] = {}


def _mosaic_window(result: Mapping[str, Any], window: Tuple[float, float, float, float],
                   step: int) -> np.ndarray:
    """The stitched nuclear stain over part of the block, at a chosen resolution.

    Read from the tiles themselves, so a zoomed view stays sharp where the
    whole-block mosaic, drawn every ``result['step']`` pixels, is coarse.

    :param result: the Align & stitch result; its ``tile_sources``,
        ``tiles`` and ``tile_shape`` are read.
    :param window: ``(top, left, bottom, right)`` in the well frame.
    :param step: well pixels per output pixel, 1 or more.
    :returns: the window as float32, ``ceil(height / step)`` by
        ``ceil(width / step)``.
    """
    from .ops_engine import _read_plane

    step = max(1, int(step))
    top, left, bottom, right = (float(v) for v in window)
    out = np.zeros((max(1, int(np.ceil((bottom - top) / step))),
                    max(1, int(np.ceil((right - left) / step)))), np.float32)
    height, width = result["tile_shape"]
    for site, (y, x) in result["tiles"].items():
        y0, x0 = max(top, y), max(left, x)
        y1, x1 = min(bottom, y + height), min(right, x + width)
        if y1 <= y0 or x1 <= x0:
            continue
        source = result.get("tile_sources", {}).get(site)
        if not source:
            continue
        key = f"{source[0]}|{source[1]}"
        plane = _PLANES.get(key)
        if plane is None:
            plane = _read_plane(source, [])
            if plane is None:
                continue
            if len(_PLANES) >= 8:
                _PLANES.pop(next(iter(_PLANES)))
            _PLANES[key] = plane
        part = plane[int(y0 - y):int(y1 - y), int(x0 - x):int(x1 - x)]
        small = _downsample(part, step) if step > 1 else np.asarray(part, np.float32)
        oy, ox = int((y0 - top) // step), int((x0 - left) // step)
        h = min(small.shape[0], out.shape[0] - oy)
        w = min(small.shape[1], out.shape[1] - ox)
        if h > 0 and w > 0:
            out[oy:oy + h, ox:ox + w] = np.maximum(out[oy:oy + h, ox:ox + w],
                                                   small[:h, :w])
    return out


def _write_crop(folder: Path, planes: Mapping[Tuple[int, str], np.ndarray]
                ) -> Dict[Tuple[int, str], Tuple[str, None]]:
    """Write each cropped plane as its own TIFF for the engine's reader.

    :returns: ``(cycle, channel) -> (path, None)``, the source form
        :func:`spacr.ops_engine._read_plane` reads.
    """
    from .tiff_io import write_tiff

    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for (cycle, channel), plane in planes.items():
        path = folder / f"c{cycle}_{channel}.tif"
        write_tiff(str(path), np.asarray(plane, np.float32))
        out[(cycle, channel)] = (str(path), None)
    return out


def _decode_options(settings: Mapping[str, Any], library: Sequence[str], *,
                    well: Optional[str] = None) -> Dict[str, Any]:
    """The per-field decode options, read from the settings as ``_decode`` reads them.

    Mirrors the option block of :func:`spacr.ops_engine._decode` for one
    field in manual threshold mode, with the engine's own readers. The auto
    threshold mode tunes over a whole well, so a crop is decoded at the
    entered thresholds.

    :param settings: effective settings before preview workspace overrides.
    :param library: production guide sequences.
    :param well: production well when known; never methods then read its
        saved scale from dst_root without calibrating the preview crop.
    :returns: the task keys every field shares, ``rule`` and the decoys.
    """
    from .ops_decode import decode_rule, resolve_settings, _reported_fdr_scale
    from .ops_engine import (_CHANNEL_REGISTRATION, _CHANNEL_REGISTRATIONS,
                             _CROSSTALK, _CROSSTALK_METHODS, _FOOTPRINT,
                             _SINGLE_READ_QUALITY, _THRESHOLD_READS,
                             _cell_rule, _registration_choice,
                             _setting_number, _spot_detector)
    from .ops_sbs import _read_mode_setting, decoy_library

    settings = resolve_settings(settings)
    rule_a = decode_rule(settings)
    threshold = float(_setting_number(settings.get("ops_read_threshold"),
                                      "ops_read_threshold", _THRESHOLD_READS))
    footprint = float(_setting_number(settings.get("ops_footprint"),
                                      "ops_footprint", _FOOTPRINT))
    crosstalk = str(settings.get("ops_crosstalk") or _CROSSTALK).strip().lower()
    if crosstalk not in _CROSSTALK_METHODS:
        raise ValueError(f"ops_crosstalk must be one of "
                         f"{list(_CROSSTALK_METHODS)}; got {crosstalk!r}")
    channel_mode = str(settings.get("ops_channel_registration")
                       or _CHANNEL_REGISTRATION).strip().lower()
    if channel_mode not in _CHANNEL_REGISTRATIONS:
        raise ValueError(f"ops_channel_registration must be one of "
                         f"{list(_CHANNEL_REGISTRATIONS)}; got {channel_mode!r}")
    registration = {
        "refine_px": int(_setting_number(settings.get("ops_registration_refine_px"),
                                         "ops_registration_refine_px", 3)),
        "spot_min_peak": float(_setting_number(settings.get("ops_spot_min_peak"),
                                               "ops_spot_min_peak", 20.0)),
        "spot_overrule_share": float(_setting_number(
            settings.get("ops_spot_overrule_share"), "ops_spot_overrule_share",
            0.5)),
        "precision": _registration_choice(
            settings.get("ops_registration_precision"), "ops_registration_precision"),
        "backend": _registration_choice(
            settings.get("ops_registration_backend"), "ops_registration_backend"),
    }
    rule, mismatches, min_reads, min_fraction = _cell_rule(settings, library)
    read_mode = _read_mode_setting(settings)
    if read_mode == "auto":
        read_mode = "log"
    decoys: List[str] = []
    if rule_a["active"] and rule == "mapped":
        rule_a.update({
            "threshold": threshold, "footprint": footprint,
            "max_distance": mismatches, "min_fraction": min_fraction,
            "single_read_quality": float(_setting_number(
                settings.get("ops_single_read_quality"),
                "ops_single_read_quality", _SINGLE_READ_QUALITY)),
            "features": rule_a["features"],
            "store_cache": True,
        })
        if well is not None and settings.get("dst_root"):
            rule_a["fdr_scale"] = _reported_fdr_scale(
                os.path.join(str(settings["dst_root"]), "measurements.db"), well,
                rule_a["fdr_scale"], str(ops_setting(settings, "ops_decode_fdr_calibration")))
        if rule_a["decoy_k"] > 0:
            decoys = decoy_library(library, k=rule_a["decoy_k"])
        if read_mode != "log":
            rule_a["read_mode"] = read_mode
    else:
        rule_a = None
    return {
        "task": {"threshold": threshold, "footprint": footprint,
                 "store_reads": True, "spot_detector": _spot_detector(settings),
                 "crosstalk": crosstalk, "channel_mode": channel_mode,
                 "registration": registration, "rule_a": rule_a,
                 **({"read_mode": read_mode} if read_mode != "log" else {}),
                 "diagnostics_folder": None, "dapi": None},
        "rule": rule, "min_reads": min_reads, "min_fraction": min_fraction,
        "mismatches": mismatches, "decoys": decoys,
    }


def _run_spots(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Segment the crop's nuclei, then detect, call and vote its reads.

    The nuclei come from the objects step's model
    (:func:`spacr.ops_engine._cellpose_model` on the crop's cycle-1 DAPI,
    kept until a nucleus setting changes); the reads from
    :func:`spacr.ops_engine._decode_field` on the crop, with the task
    :func:`_decode_options` builds.

    :returns: the crop's DAPI, nuclei, spots, read calls and per-cell calls.
    """
    from scipy import ndimage

    from . import ops_engine as engine
    from .ops_sbs import (assign_reads_to_objects, call_reads,
                          correct_to_library)

    align = _result(ctx, "align", cancelled)
    index, reference = _sbs_index(ctx)
    site, crop = _genotyped_crop(ctx, align)
    y0, x0, size = crop
    sources = engine._plane_sources(index[reference][site])
    dapi = _read(ctx, sources.get(engine._NUCLEAR))
    if dapi is None:
        raise ValueError(f"tile {site} has no readable DAPI plane")
    dapi = np.ascontiguousarray(dapi[y0:y0 + size, x0:x0 + size])
    nuclei_key = json.dumps([site, crop, ctx.settings.get("ops_sbs_nucleus_model"),
                             ops_setting(ctx.settings, "ops_cellpose_model"),
                             ctx.settings.get("cellpose_diameter"),
                             bool(ops_setting(ctx.settings, "ops_gpu")),
                             ctx.settings.get("mask_eval_batch_size"),
                             ctx.settings.get("ops_sbs_tile_batch"),
                             ctx.settings.get("mask_inference_precision")], default=str)
    held = ctx.cache.get("_sbs_nuclei")
    if held is not None and held[0] == nuclei_key:
        nuclei = held[1]
    else:
        model = _model(ctx, "ops_sbs_nucleus_model")
        diameter = ctx.settings.get("cellpose_diameter")
        extra = {"diameter": float(diameter)} if diameter else {}
        from ._mask_array_batches import _array_batches

        def load_dapi(_record):
            """Retain the current OPS crop and its public normalisation."""
            return dapi, None

        with _array_batches([site], load_dapi, ctx.settings,
                            gpu=bool(ops_setting(ctx.settings, "ops_gpu"))) as resident:
            for _site, image, _metadata, field_count in resident:
                nuclei = np.asarray(engine._ops_mask_eval(
                    model, image, ctx.settings, site="ops_stitched",
                    field_count=field_count, **extra)[0], np.int32)
        ctx.cache["_sbs_nuclei"] = (nuclei_key, nuclei)
    _check(cancelled)
    labels = np.unique(nuclei)
    labels = labels[labels > 0]
    centroids = (np.array(ndimage.center_of_mass(nuclei > 0, nuclei, labels))
                 if len(labels) else np.zeros((0, 2)))
    areas = (np.asarray(ndimage.sum(np.ones_like(nuclei), nuclei, labels), float)
             if len(labels) else np.zeros(0))
    library = engine._load_library(ctx.settings.get("ops_library") or None)
    options = _decode_options(ctx.settings, library, well=ctx.well)
    channels = engine._base_channels(ctx.settings)
    cycles = sorted(index)
    crop_planes: Dict[Tuple[int, str], np.ndarray] = {}
    for cycle in cycles:
        cyc_sources = engine._plane_sources(index[cycle].get(site, {}))
        for name in channels:
            plane = _read(ctx, cyc_sources.get(name))
            if plane is not None:
                crop_planes[(cycle, name)] = plane[y0:y0 + size, x0:x0 + size]
    _check(cancelled)
    written = _write_crop(ctx.folder / "spots_crop", crop_planes)
    cache_folder = ctx.folder / "spots_cache"
    task = {
        "site": site, "cycles": cycles, "reference": reference,
        "planes": {cycle: [written.get((cycle, name)) for name in channels]
                   for cycle in cycles},
        "centroids": centroids, "areas": areas,
        "ids": labels.astype(np.int64), "owned": np.ones(len(labels), bool),
        "cache_folder": str(cache_folder), **options["task"],
        "gpu": bool(ops_setting(ctx.settings, "ops_gpu")),
    }
    previous = (engine._LIBRARY, engine._DECOYS)
    try:
        engine._init_decode_worker(frozenset(library), frozenset(options["decoys"]))
        result = engine._decode_field(task)
    finally:
        engine._LIBRARY, engine._DECOYS = previous
    _check(cancelled)
    if result.get("skipped"):
        raise ValueError(f"the decode skipped the crop: {result['skipped']}")
    rows: List[Dict[str, Any]] = []
    if result.get("rule_a") is not None:
        evidence_path = cache_folder / f"site{int(site):04d}.npz"
        with np.load(evidence_path) as saved:
            peaks = saved["peaks"].astype(float)
            strength = saved["strength"].astype(float)
            values = saved["values"].astype(np.float32)
        keep = strength >= float(options["task"]["threshold"])
        peaks, strength, values = peaks[keep], strength[keep], values[keep]
        for object_id, call in sorted(result["rule_a"].get("calls", {}).items()):
            rows.append({"object_id": int(object_id),
                         "guide": str(call.get("guide") or ""),
                         "barcode": str(call.get("barcode") or ""),
                         "reads": int(call.get("n_reads") or 0),
                         "agreeing": int(call.get("n_agreeing") or 0),
                         "fraction": float(call.get("fraction") or 0.0),
                         "quality": float(call.get("quality") or 0.0),
                         "decoy": bool(call.get("decoy")),
                         "pixels": str(call.get("source")) == "pixel_fallback"})
    else:
        peaks = np.asarray(result.get("peaks", np.zeros((0, 2))), float)
        values = np.asarray(result.get("intensities",
                                       np.zeros((0, len(cycles), 4))), np.float32)
        strength = np.full(len(peaks), np.nan)
        assigned = assign_reads_to_objects(
            np.asarray(result["ids"]), result["calls"], quality=result["quality"],
            min_reads=options["min_reads"], min_fraction=options["min_fraction"])
        distinct = sorted({row["barcode"] for row in assigned.values()})
        mapped = dict(zip(distinct, correct_to_library(
            distinct, library, max_distance=options["mismatches"]))) if library else {}
        for object_id, row in sorted(assigned.items()):
            rows.append({"object_id": int(object_id),
                         "guide": str(mapped.get(row["barcode"]) or ""),
                         "barcode": str(row["barcode"]), "reads": int(row["reads"]),
                         "agreeing": int(row["agreeing"]),
                         "fraction": float(row["fraction"]),
                         "quality": float(row["quality"] or 0.0), "decoy": False,
                         "pixels": False})
    from .ops_engine import _BASES, _crosstalk_arguments

    letters, quality = (call_reads(values, bases=_BASES, gpu=False,
                                   **_crosstalk_arguments(task))
                        if len(values) else ([], np.zeros(0)))
    return {
        "site": site, "crop": crop, "dapi": dapi, "nuclei": nuclei,
        "peaks": peaks, "strength": strength, "reads": list(letters),
        "read_quality": np.asarray(quality, float), "cells": rows,
        "n_nuclei": int(len(labels)), "n_cycles": int(result.get("n_cycles") or 0),
        "rule": "rule (a)" if result.get("rule_a") is not None else options["rule"],
        "threshold": float(options["task"]["threshold"]),
        "library": len(library),
    }


def _run_objects(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Segment and measure the genotyped region on every phenotype field over it.

    The region is the sequencing crop Spots & genotyping reads
    (:func:`_genotyped_crop`); each placed phenotype field over it is cut to
    the part the crop covers (:func:`_region_windows`), up to the Objects
    Sample (``ops_live_ph_fields``), and segmented and measured as the
    ``ph_segment`` and ``ph_measure`` phases do. Without a placed field the
    first phenotype field is cropped at its middle.

    :returns: one part per field, each with its crop, nuclei, cells, extra
        object layers and cell table; the first part's keys are repeated at
        the top level, and ``table`` holds every part's cells with a
        ``site`` column.
    """
    import pandas as pd

    from . import ops_engine as engine
    from .ops_sample import measure_phenotype_field

    align = _result(ctx, "align", cancelled)
    channels = engine._ph_channels(ctx.settings)
    site, crop = _genotyped_crop(ctx, align)
    windows = _region_windows(align, site, crop,
                              _live_int(ctx.settings, "ops_live_ph_fields", 9))
    if not windows:
        field_info = align["sample_field"]
        if field_info is None:
            files = _ph_index(ctx)
            if not files:
                raise _NotReady("phenotype")
            first = sorted(files)[0]
            field_info = {"site": first, "files": files[first], "shape": None}
        windows = [(dict(field_info), None)]
    parts: List[Dict[str, Any]] = []
    for field_info, window in windows:
        stack, key = _ph_stack(ctx, field_info, channels)
        _check(cancelled)
        info = dict(field_info)
        info["shape"] = stack.shape[:2]
        if window is None:
            row, column, size = _ph_crop_window(info, bool(ctx.test_root))
            window = (row, column, size, size)
        row, column, height, width = window
        part_crop = np.ascontiguousarray(stack[row:row + height, column:column + width])
        nuclei, cells = _ph_masks(ctx, field_info, key, info["shape"], window,
                                  part_crop, channels, cancelled)
        layers: Dict[str, np.ndarray] = {}
        for layer in _OBJECT_LAYERS:
            layers.update(layer(ctx.settings, part_crop, channels, nuclei, cells) or {})
        table = measure_phenotype_field(
            nuclei, cells, part_crop, channels,
            texture=bool(ctx.settings.get("ops_ph_texture", True)),
            funk_features=bool(ctx.settings.get("ops_ph_funk_features", False)),
            zernike=bool(ctx.settings.get("ops_ph_zernike", False)),
            gpu=bool(ctx.settings.get("ops_ph_measure_gpu", False)),
            profile_features=engine._ph_profile_features(ctx.settings),
            granularity_subsample=float(
                engine._ph_granularity_subsample(ctx.settings) or 1.0))
        _check(cancelled)
        if "site" not in table.columns:
            table.insert(0, "site", int(field_info["site"]))
        else:
            table["site"] = int(field_info["site"])
        parts.append({"site": int(field_info["site"]),
                      "window": (row, column, height, width),
                      "stack": part_crop, "nuclei": nuclei, "cells": cells,
                      "layers": layers, "table": table})
    first = parts[0]
    joined = pd.concat([p["table"] for p in parts], ignore_index=True)
    return {"site": first["site"], "window": first["window"], "stack": first["stack"],
            "channels": list(channels), "nuclei": first["nuclei"],
            "cells": first["cells"], "layers": first["layers"], "table": joined,
            "parts": parts, "sbs_site": int(site), "sbs_crop": tuple(crop),
            "cell_channel": str(ctx.settings.get("ops_ph_cell_channel") or "AF750")}


def _ph_masks(ctx: _Context, field_info: Mapping[str, Any], key: tuple,
              shape: Tuple[int, int], window: Tuple[int, int, int, int],
              crop: np.ndarray, channels: Sequence[str], cancelled=None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """The nuclei and cells of one phenotype crop, kept until a segmentation setting changes."""
    from . import ops_engine as engine

    masks_key = json.dumps([key, shape, window] + [
        ctx.settings.get(name) for name in (
            "ops_ph_nucleus_model", "ops_ph_cell_model", "ops_ph_nucleus_diameter",
            "ops_ph_cell_diameter", "ops_ph_downsample", "mask_eval_batch_size",
            "ops_ph_tile_batch", "mask_inference_precision", "batch_size",
            "ops_ph_cell_channel", "ops_gpu")], default=str)
    held = ctx.cache.setdefault("_ph_masks", {})
    if not isinstance(held, dict):
        held = ctx.cache["_ph_masks"] = {}
    if masks_key in held:
        return held[masks_key]
    nucleus_model = _model(ctx, "ops_ph_nucleus_model")
    same = (str(ctx.settings.get("ops_ph_cell_model") or "cpsam")
            == str(ctx.settings.get("ops_ph_nucleus_model") or "cpsam"))
    models = (nucleus_model, nucleus_model if same else
              _model(ctx, "ops_ph_cell_model"))
    _check(cancelled)
    from ._mask_array_batches import _array_batches

    def load_crop(_record):
        """Keep the same prepared phenotype crop for its two mask roles."""
        return crop, None

    with _array_batches([field_info['site']], load_crop, ctx.settings,
                        gpu=bool(ops_setting(ctx.settings, "ops_gpu"))) as resident:
        for _site, image, _metadata, field_count in resident:
            nuclei, cells, _paired = engine._ph_segment_field(
                image, channels, models, ctx.settings, field_count=field_count)
    if len(held) >= 9:
        held.pop(next(iter(held)))
    held[masks_key] = (nuclei, cells)
    return nuclei, cells


def _ph_stack(ctx: _Context, field_info: Mapping[str, Any],
              channels: Sequence[str]) -> Tuple[np.ndarray, tuple]:
    """One phenotype field's channel stack, read once and kept.

    :param ctx: the pass context; the stack is kept in its cache.
    :param field_info: the field, with its ``site`` and ``files``.
    :param channels: the phenotype channels, in stack order.
    :returns: ``(stack, key)``; the key names the field and its channels.
    :raises ValueError: when the field cannot be read.
    """
    from . import ops_engine as engine

    key = ("_ph_stack", int(field_info["site"]), tuple(channels))
    held = ctx.cache.setdefault("_ph_stacks", {})
    if key in held:
        return held[key], key
    unreadable: list = []
    stack = engine._ph_read_field(field_info["files"], list(channels), unreadable,
                                  None)
    if stack is None:
        raise ValueError(f"phenotype field {field_info['site']} could not be "
                         f"read: {unreadable[:1]}")
    if len(held) >= 4:
        held.pop(next(iter(held)))
    held[key] = stack
    return stack, key


def _neighbour_pair(tiles: Mapping[int, Tuple[float, float]],
                    shape: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
    """Two tiles that share a seam, and which axis the seam runs along.

    :param tiles: ``site -> (y, x)`` placements of the reference cycle.
    :param shape: the tile shape.
    :returns: ``(first, second, axis)`` with the second tile right of
        (``axis`` 1) or below (``axis`` 0) the first, the closest such pair;
        None for a single tile.
    """
    best = None
    for axis in (1, 0):
        other = 1 - axis
        for a, (ya, xa) in tiles.items():
            for b, (yb, xb) in tiles.items():
                gap = (yb, xb)[axis] - (ya, xa)[axis]
                if a == b or gap <= 0 or gap >= shape[axis]:
                    continue
                if abs((yb, xb)[other] - (ya, xa)[other]) > shape[other] / 2:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, int(a), int(b), axis)
        if best is not None:
            return best[1:]
    return None


def _shifted(plane: np.ndarray, window: Tuple[int, int, int, int],
             offset: Tuple[float, float]) -> np.ndarray:
    """The part of a tile under a window of its neighbour, at an offset.

    :param plane: the second tile's plane.
    :param window: ``(top, bottom, left, right)`` in the first tile's pixels.
    :param offset: the second tile's ``(dy, dx)`` from the first.
    :returns: the window's pixels of ``plane``, zero where it has none.
    """
    top, bottom, left, right = window
    out = np.zeros((bottom - top, right - left), np.float32)
    dy, dx = int(round(offset[0])), int(round(offset[1]))
    rows = np.arange(top, bottom) - dy
    cols = np.arange(left, right) - dx
    keep_r = (rows >= 0) & (rows < plane.shape[0])
    keep_c = (cols >= 0) & (cols < plane.shape[1])
    if keep_r.any() and keep_c.any():
        out[np.ix_(keep_r, keep_c)] = plane[np.ix_(rows[keep_r], cols[keep_c])]
    return out


def _seam(ctx: _Context, index, reference: int,
          tiles: Mapping[int, Tuple[float, float]], shape: Tuple[int, int]
          ) -> Optional[Dict[str, Any]]:
    """Two neighbouring tiles' overlap on the raster grid and once stitched.

    The raster grid puts the second tile ``tile size - ops_raster_overlap``
    pixels from the first, straight along the seam; the stitch moves it to
    where its nuclei meet the first tile's. Both are drawn over the same
    window of the first tile.

    :returns: the two sites, the axis, both offsets and the window's two
        planes at each offset; None when no two tiles share a seam.
    """
    from .ops_engine import _NUCLEAR, _plane_sources, _setting_number

    pair = _neighbour_pair(tiles, shape)
    if pair is None:
        return None
    first, second, axis = pair
    planes = [_read(ctx, _plane_sources(index[reference][site]).get(_NUCLEAR))
              for site in (first, second)]
    if any(plane is None for plane in planes):
        return None
    overlap = int(_setting_number(ctx.settings.get("ops_raster_overlap"),
                                  "ops_raster_overlap", 213))
    nominal = [0.0, 0.0]
    nominal[axis] = float(shape[axis] - overlap)
    stitched = (tiles[second][0] - tiles[first][0],
                tiles[second][1] - tiles[first][1])
    margin = 48
    half = min(shape[1 - axis] // 2, 240)
    middle = shape[1 - axis] // 2
    start = int(max(0, min(nominal[axis], stitched[axis]) - margin))
    if axis == 1:
        window = (middle - half, middle + half, start, shape[1])
    else:
        window = (start, shape[0], middle - half, middle + half)
    top, bottom, left, right = window
    own = np.asarray(planes[0], np.float32)[top:bottom, left:right]
    return {"sites": (first, second), "axis": axis, "overlap": overlap,
            "nominal": tuple(nominal), "stitched": tuple(stitched),
            "window": window,
            "before": (own, _shifted(np.asarray(planes[1], np.float32), window,
                                     nominal)),
            "after": (own, _shifted(np.asarray(planes[1], np.float32), window,
                                    stitched))}


def _placement(ctx: _Context, field_info: Optional[Mapping[str, Any]],
               tiles: Mapping[int, Tuple[float, float]], crop_site: int,
               crop: Tuple[int, int, int], sbs: np.ndarray
               ) -> Optional[Dict[str, Any]]:
    """The sampled phenotype field's DNA resampled onto the sequencing crop.

    Each pixel of the sequencing crop is mapped into the field by the
    inverse of the field's placement (:func:`spacr.ops_merge.field_transform`)
    and read there, so where the placement is right the two images' nuclei
    coincide.

    :param sbs: the crop of the reference cycle's DAPI plane.
    :returns: the field's site, scale, rotation and centre and the two
        planes; None when no field is placed or it cannot be read.
    """
    from scipy import ndimage

    from .ops_merge import field_transform

    dna = _field_dna(ctx, field_info)
    if dna is None:
        return None
    scale, rotation, translation = field_transform(
        field_info["scale"], field_info["rotation"], field_info["centre"],
        dna.shape[:2])
    y0, x0, size = crop
    ty, tx = tiles[crop_site]
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    well = np.stack([rows.ravel() + ty + y0, cols.ravel() + tx + x0], axis=1)
    inside = (np.asarray(rotation, float).T @ (well - translation).T).T / scale
    sampled = ndimage.map_coordinates(np.asarray(dna, np.float32),
                                      [inside[:, 0], inside[:, 1]], order=1,
                                      cval=0.0).reshape(size, size)
    return {"site": int(field_info["site"]), "scale": float(field_info["scale"]),
            "rotation": float(field_info["rotation"]),
            "centre": tuple(float(v) for v in field_info["centre"]),
            "source": str(field_info.get("source") or ""),
            "sbs": np.asarray(sbs, np.float32), "phenotype": sampled}


def _field_dna(ctx: _Context, field_info: Optional[Mapping[str, Any]]
               ) -> Optional[np.ndarray]:
    """The sampled phenotype field's DNA plane, or None when it cannot be read."""
    from . import ops_engine as engine

    if not field_info or not field_info.get("files"):
        return None
    channels = engine._ph_channels(ctx.settings)
    try:
        stack, _key = _ph_stack(ctx, field_info, channels)
    except ValueError:
        return None
    return stack[..., channels.index("DAPI")] if "DAPI" in channels else stack[..., 0]


def _merged_placement_row(ctx: _Context, plate: str, site: int):
    """The finished merge's tile-affine row of one phenotype field, or None.

    For the test data the stored ``reference/merge/tile_affine`` table; for a
    user's data the ``ops_phenotype`` table of the run in ``dst_root``.
    """
    import pandas as pd

    if ctx.test_root is not None:
        path = ctx.test_root / "reference" / "merge" / "tile_affine" / "ops_phenotype.parquet"
        table = pd.read_parquet(path) if path.is_file() else None
    else:
        run = str(ctx.settings.get("dst_root") or "")
        table = _rows(os.path.join(run, "measurements.db"), "ops_phenotype", plate,
                      ctx.well) if run else None
    if table is None or table.empty or "align_a11" not in table.columns:
        return None
    rows = table[(table["site"].astype(int) == int(site))
                 & (table["align_model"].astype(str) == "tile_affine")]
    if rows.empty or not np.isfinite(float(rows["align_a11"].iloc[0])):
        return None
    return rows.iloc[0]


def _tile_affine_placement(ctx: _Context, plate: str,
                           field_info: Optional[Mapping[str, Any]],
                           tiles: Mapping[int, Tuple[float, float]], crop_site: int,
                           crop: Tuple[int, int, int], sbs: np.ndarray
                           ) -> Optional[Dict[str, Any]]:
    """The sampled field's DNA resampled onto the sequencing crop at its tile-affine placement.

    The finished merge's affine (:func:`_merged_placement_row`) is in that run's
    well frame. Both runs place the field before merge from the same seed or
    raster, so the difference of the two stored centres moves the affine into
    the preview's frame. The crop takes the affine of its own sequencing tile
    group when the merge fitted one, else the field affine
    (:func:`spacr.ops_merge.tile_affine_to_well`).

    :param sbs: the crop of the reference cycle's DAPI plane.
    :returns: the field's site, the affine used (``'tile'`` or ``'field'``),
        how far the tile-affine placement moves the field at the crop centre
        from the placement before, ``(dy, dx)`` in well pixels, and the two
        planes; None without a finished tile-affine merge of the field.
    """
    import json

    from scipy import ndimage

    from .ops_merge import field_to_well, field_transform

    if not field_info:
        return None
    row = _merged_placement_row(ctx, plate, int(field_info["site"]))
    dna = _field_dna(ctx, field_info) if row is not None else None
    if dna is None:
        return None
    offset = (np.asarray(field_info["centre"], float)
              - np.array([float(row["centre_y"]), float(row["centre_x"])]))
    matrix = np.array([[row["align_a11"], row["align_a12"]],
                       [row["align_a21"], row["align_a22"]]], float)
    translation = np.array([row["align_ty"], row["align_tx"]], float)
    used = "field"
    for group in json.loads(row["align_tile_transforms"] or "[]"):
        if int(group[0]) == int(crop_site):
            matrix = np.array(group[2:6], float).reshape(2, 2)
            translation = np.array(group[6:8], float)
            used = "tile"
            break
    y0, x0, size = crop
    ty, tx = tiles[crop_site]
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    well = np.stack([rows.ravel() + ty + y0, cols.ravel() + tx + x0], axis=1)
    inside = np.linalg.solve(matrix, (well - offset - translation).T).T
    sampled = ndimage.map_coordinates(np.asarray(dna, np.float32),
                                      [inside[:, 0], inside[:, 1]], order=1,
                                      cval=0.0).reshape(size, size)
    middle = np.array([[size / 2.0 + ty + y0, size / 2.0 + tx + x0]])
    before = field_transform(field_info["scale"], field_info["rotation"],
                             field_info["centre"], dna.shape[:2])
    seen = np.linalg.solve(matrix, (middle - offset - translation).T).T
    shift = middle - field_to_well(seen, before)
    return {"site": int(field_info["site"]), "affine": used,
            "shift": (float(shift[0, 0]), float(shift[0, 1])),
            "sbs": np.asarray(sbs, np.float32), "phenotype": sampled}


def _aggregation_inputs(ctx: _Context, db: str, plate: str,
                        block: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Put the aggregation set's nuclei, calls and cells in the workspace.

    For the test data these are its stored step outputs (``reference/``);
    for a user's data, the same tables of the user's finished run
    (``dst_root``), read and copied, never written.

    :param block: the sequencing tiles of the region shown; the regions
        tier keeps one reference set per block under ``reference/regions``.
    :returns: what was used, for the tab to say.
    :raises _NotReady: when there is no finished run to read.
    """
    import shutil

    import pandas as pd

    from .ops_engine import _ph_cells_path, _replace_well_rows

    target = _ph_cells_path(str(ctx.folder), ctx.well)
    if ctx.test_root is not None:
        ref = ctx.test_root / "reference"
        regions = ref / "regions"
        if regions.is_dir():
            name = "block_" + "-".join(str(int(t)) for t in sorted(block or ()))
            blocks = sorted(p for p in regions.iterdir() if p.is_dir())
            ref = regions / name if (regions / name).is_dir() else blocks[0]
        for table, path in (("ops_objects", "objects/ops_objects.parquet"),
                            ("ops_barcodes", "genotype/ops_barcodes.parquet")):
            _replace_well_rows(db, table, pd.read_parquet(ref / path), plate,
                               ctx.well)
        shutil.copy(ref / "phenotype" / f"ops_phenotype_cells_{ctx.well}.parquet",
                    target)
        masks = ctx.folder / ctx.well / "phenotype_masks"
        masks.mkdir(parents=True, exist_ok=True)
        for path in sorted((ref / "phenotype" / "phenotype_masks").iterdir()):
            if (masks / path.name).exists():
                continue
            if path.is_dir():
                shutil.copytree(path, masks / path.name)
            elif path.suffix == ".npz" or path.name.endswith(".json"):
                shutil.copy(path, masks / path.name)
        return {"kind": "test", "path": str(ref), "masks": True}
    run = str(ctx.settings.get("dst_root") or "")
    run_db = os.path.join(run, "measurements.db")
    cells = _ph_cells_path(run, ctx.well)
    from .ops_store import _exists

    if not (_exists(run_db) and os.path.isfile(cells)):
        raise _NotReady("run", path=run or "dst_root",
                        missing=", ".join(name for name, there in (
                            ("measurements", _exists(run_db)),
                            (os.path.basename(cells), os.path.isfile(cells)))
                            if not there))
    for table in ("ops_objects", "ops_barcodes"):
        rows = _rows(run_db, table, plate, ctx.well)
        if rows is None or rows.empty:
            raise _NotReady("rows", path=run, table=table)
        _replace_well_rows(db, table, rows, plate, ctx.well)
    shutil.copy(cells, target)
    maskpass = os.path.join(run, f"ops_maskpass_{ctx.well}.parquet")
    if os.path.isfile(maskpass):
        shutil.copy(maskpass, os.path.join(str(ctx.folder),
                                           f"ops_maskpass_{ctx.well}.parquet"))
    return {"kind": "run", "path": run, "masks": False}


def _run_merge(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Match phenotype cells to called nuclei, then reduce them to genes.

    Calls :func:`spacr.ops_engine.run_ops` with the ``merge`` phase and then
    the ``aggregate`` phase in the preview workspace, on the aggregation set
    (:func:`_aggregation_inputs`).

    :returns: the merge report, matched pairs near the sample, the share
        matched per field and the gene table.
    """
    import pandas as pd

    from .ops_engine import run_ops

    align = _result(ctx, "align", cancelled)
    if not align["fields"]:
        raise _NotReady("fields")
    db, plate = align["db"], align["plate"]
    used = _aggregation_inputs(ctx, db, plate, align.get("block"))
    settings = _preview_settings(ctx.settings, ctx.folder)
    settings["ops_sites"] = align["sample_block"]
    phases = ["ph_masks", "merge"] if used.get("masks") else ["merge"]
    report = run_ops(settings, wells=[ctx.well], phases=phases)
    _check(cancelled)
    merge_report = report["wells"][ctx.well].get("merge", {})
    aggregate = run_ops(settings, wells=[ctx.well], phases=["aggregate"])
    _check(cancelled)
    outputs = (aggregate.get("aggregate") or {}).get("outputs") or {}
    genes = (pd.read_parquet(outputs["genes_parquet"])
             if outputs.get("genes_parquet") else pd.DataFrame())
    merged = _rows(db, "ops_merge", plate, ctx.well)
    objects = _rows(db, "ops_objects", plate, ctx.well)
    cells = pd.read_parquet(os.path.join(str(ctx.folder),
                                         f"ops_phenotype_cells_{ctx.well}.parquet"),
                            columns=["site", "label", "centroid_row", "centroid_col"])
    by_site = {f["site"]: f for f in align["fields"]}
    per_field = []
    points = []
    for site, group in cells.groupby("site"):
        info = by_site.get(int(site))
        if info is None:
            continue
        well_xy = _ph_to_well(info, group[["centroid_row", "centroid_col"]].to_numpy(float))
        points.append(pd.DataFrame({"site": int(site), "label": group["label"].to_numpy(),
                                    "y": well_xy[:, 0], "x": well_xy[:, 1]}))
        matched = (0 if merged is None or merged.empty
                   else int((merged["site"] == site).sum()))
        per_field.append({"site": int(site), "cells": int(len(group)),
                          "matched": int(matched)})
    placed = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    shown = _live_int(ctx.settings, "ops_live_merge_fields", 0)
    if shown > 0:
        per_field = sorted(per_field, key=lambda f: -f["cells"])[:shown]
    profiles = _profiles_root() if ctx.test_root is not None else None
    if profiles is not None:
        genes, used = _profile_genes(ctx.settings, profiles), {
            "kind": "profiles", "path": str(profiles), "base": used}
    return {"report": merge_report, "used": used, "genes": genes,
            "merged": merged, "objects": objects, "cells": placed,
            "per_field": per_field, "align": align,
            "control_prefix": str(ctx.settings.get("ops_control_prefix") or "nontargeting"),
            "aggregate": {k: v for k, v in (aggregate.get("aggregate") or {}).items()
                          if not isinstance(v, (dict, list))}}


def _profiles_root() -> Optional[Path]:
    """The unpacked OPS profiles archive (item 977), or None.

    It unpacks beside the full tier of the OPS end-to-end test data and holds
    the cells of 98 CORUM-complex genes and the non-targeting guides of wells
    A1-A3, enough cells per gene for the aggregation and clustering tabs.
    """
    from .example_archives import OPS_E2E_FULL_FOLDER, example_set_folder

    root = (example_set_folder("ops_e2e").parent / OPS_E2E_FULL_FOLDER
            / "ops_testdata_funk2022_A1_profiles")
    return root if (root / "profiles_cells.parquet").is_file() else None


def _profile_genes(settings: Mapping[str, Any], root: Path):
    """The profiles archive's cells reduced to genes, as the aggregate phase reduces them.

    Every feature becomes a robust z against the same well's non-targeting
    cells, a guide is the median of its cells and a gene the median of its
    guides (or of its cells, with ``ops_aggregate_level`` 'cells'), with the
    functions :func:`spacr.ops_engine._ph_aggregate` uses.

    :param settings: read for ``ops_control_prefix`` and ``ops_aggregate_level``.
    :param root: from :func:`_profiles_root`.
    :returns: one row per gene with ``n_cells``.
    """
    import pandas as pd

    from .gene_measurement_compare import (aggregate_genes, aggregate_guides,
                                           robust_z_to_controls)
    from .ops_engine import _ph_feature_columns

    cells = pd.read_parquet(root / "profiles_cells.parquet")
    features = [c for c in _ph_feature_columns(cells.columns)
                if pd.api.types.is_numeric_dtype(cells[c])]
    prefix = str(settings.get("ops_control_prefix") or "nontargeting").lower()
    control = cells["gene"].astype(str).str.lower().str.startswith(prefix)
    scored = pd.concat([cells[["well", "sgRNA", "gene"]].reset_index(drop=True),
                        robust_z_to_controls(cells, features, control=control
                                             ).reset_index(drop=True)], axis=1)
    level = str(settings.get("ops_aggregate_level") or "guides").strip().lower()
    guides = aggregate_guides(scored, features, guide="sgRNA", gene="gene")
    return aggregate_genes(guides, features, level=level, cells=scored)


def _string_score(genes: Sequence[str], labels, prefix: str) -> Optional[Dict[str, float]]:
    """STRING precision and recall of the pairs that share a cluster, or None.

    Read from the profiles archive's ``references/string_pair_benchmark.tsv``
    with :func:`spacr.ops_clusters.string_cluster_pairs`; KEGG and CORUM are
    not shipped, so STRING alone scores the preview.
    """
    root = _profiles_root()
    table = root / "references" / "string_pair_benchmark.tsv" if root else None
    if labels is None or table is None or not table.is_file():
        return None
    from .ops_clusters import read_benchmark, string_cluster_pairs, string_positives

    return string_cluster_pairs(string_positives(read_benchmark(str(table))),
                                list(genes), np.asarray(labels), control_prefix=prefix)


def _first(value: Any, cast, fallback):
    """The first entry of a list setting, or the value itself, cast."""
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return cast(value) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback


def _run_cluster(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Embed the aggregated genes and cluster them as the sweep does.

    :func:`spacr.ops_clusters.embed_profiles`, then
    :func:`~spacr.ops_clusters.gene_graph` and
    :func:`~spacr.ops_clusters.leiden_partitions` at the first graph,
    resolution and seed the settings name. Without the ``clustering`` extra
    the embedding is shown and the clusters are not.

    :returns: the genes, their embedding, the clusters and what ran.
    """
    from .ops_clusters import embed_profiles

    merged = _result(ctx, "merge", cancelled)
    genes = merged["genes"]
    if genes is None or genes.empty:
        raise _NotReady("genes")
    floor = int(_first(ctx.settings.get("ops_profile_min_cells"), int, 0))
    kept = genes
    if "n_cells" in genes.columns:
        kept = genes[genes["n_cells"] >= floor]
    prefix = str(ctx.settings.get("ops_control_prefix") or "nontargeting").lower()
    names = kept["gene"].astype(str)
    kept = kept[~names.str.lower().str.startswith(prefix)].reset_index(drop=True)
    most = _live_int(ctx.settings, "ops_live_genes", 0)
    if most > 0 and len(kept) > most and "n_cells" in kept.columns:
        kept = kept.sort_values("n_cells", ascending=False, kind="stable"
                                ).head(most).reset_index(drop=True)
    numeric = [c for c in kept.columns
               if c not in ("gene", "n_cells", "n_guides", "sgRNA")
               and np.issubdtype(kept[c].dtype, np.number)
               and not c.startswith(("p_", "q_", "n_"))]
    if len(kept) < 5 or not numeric:
        raise _NotReady("few_genes", n=len(kept), floor=floor)
    seed = int(_first(ctx.settings.get("ops_cluster_seeds"), int,
                      _first(ctx.settings.get("ops_cluster_seed"), int, 42)))
    components, n_components, n_features = embed_profiles(kept, numeric, seed=seed)
    _check(cancelled)
    kind = str(_first(ctx.settings.get("ops_cluster_graph"), str, "phate")).lower()
    k = int(_first(ctx.settings.get("ops_cluster_knn"), int, 12))
    resolution = float(_first(ctx.settings.get("ops_cluster_resolutions"), float, 1.0))
    labels = None
    note = ""
    try:
        from .ops_clusters import gene_graph, leiden_partitions

        graph = gene_graph(components, kind, k=min(k, len(kept) - 1), seed=seed)
        labels = leiden_partitions(graph, [resolution], [seed])[(seed, resolution)]
    except ImportError:
        note = "install the clustering extra (pip install spacr[clustering]) to see the clusters"
    return {"genes": kept["gene"].astype(str).tolist(), "components": components,
            "n_cells": (kept["n_cells"].to_numpy(float).tolist()
                        if "n_cells" in kept.columns else None),
            "string": _string_score(kept["gene"].astype(str).tolist(), labels, prefix),
            "n_components": n_components, "n_features": n_features,
            "labels": labels, "kind": kind, "k": k, "resolution": resolution,
            "seed": seed, "floor": floor, "note": note,
            "total_genes": int(len(genes))}


def _run_voyager3(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Name the role of the preview clusters with Voyager3, as the ``voyager3`` phase does.

    The Clustering tab clusters are written as a cluster table in the
    preview workspace and :func:`spacr.voyager3.run_voyager3` reads it with
    the form's ``ops_voyager3_*`` settings, writing into the workspace. The
    largest clusters are annotated, as many as the Voyager3 Sample
    (``ops_live_voyager3_clusters``, 0 for all) asks.

    :returns: the cluster and candidate tables Voyager3 wrote, the method
        and provider, and the progress lines.
    :raises _NotReady: when the Clustering tab made no clusters.
    """
    import pandas as pd

    from .tabular import read_table
    from .voyager3 import run_voyager3

    clustered = _result(ctx, "cluster", cancelled)
    labels = clustered.get("labels")
    if labels is None:
        raise _NotReady("clusters")
    folder = ctx.folder / "voyager3"
    folder.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"gene": list(clustered["genes"]),
                          "cluster": [int(c) for c in labels]})
    source = folder / "live_clusters.csv"
    table.to_csv(source, index=False)
    sizes = table.groupby("cluster")["gene"].nunique().sort_values(ascending=False, kind="stable")
    sizes = sizes[sizes >= 2]
    if sizes.empty:
        raise _NotReady("clusters")
    most = _live_int(ctx.settings, "ops_live_voyager3_clusters", 3)
    chosen = [str(c) for c in (sizes.index[:most] if most > 0 else sizes.index)]
    settings = _preview_settings(ctx.settings, ctx.folder)
    settings.update({"ops_voyager3_clusters": str(source),
                     "ops_voyager3_cluster_column": "cluster",
                     "ops_voyager3_cluster_ids": ",".join(chosen)})
    lines: List[str] = []

    def say(text: str) -> None:
        """Keep the progress line and stop here when the pass was cancelled."""
        lines.append(str(text))
        _check(cancelled)

    run = run_voyager3(settings, report=say, out_dir=str(folder / "out"))
    outputs = run["outputs"]
    clusters = read_table(outputs["clusters"], canonicalise=False, report=None)
    candidates = read_table(outputs["candidates"], canonicalise=False, report=None)
    with open(run["provenance"], encoding="utf-8") as handle:
        provenance = json.load(handle)
    return {"clusters": clusters, "candidates": candidates,
            "method": str(provenance.get("method") or "ai"),
            "provider": str(provenance.get("provider_label") or provenance.get("provider") or ""),
            "model": str(provenance.get("model") or ""), "lines": lines,
            "annotated": len(chosen), "total_clusters": int(len(sizes))}


_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "align": _run_align,
    "spots": _run_spots,
    "objects": _run_objects,
    "merge": _run_merge,
    "cluster": _run_cluster,
    "voyager3": _run_voyager3,
}


def _run_step(form: Mapping[str, Any], step: str, cache: dict,
              cancelled: Optional[Callable[[], bool]] = None,
              steps: Optional[Dict[str, Tuple[str, ...]]] = None
              ) -> Dict[str, Any]:
    """Run one tab's step with the form's settings, reusing the cache.

    :param form: the settings on the OPS form.
    :param step: one of the steps in ``_STEPS``.
    :param cache: kept by the panel between passes.
    :param cancelled: answers whether the pass was cancelled.
    :param steps: the setting-to-step map, when the caller holds it.
    :returns: the step's result, with ``test_data`` saying whether the
        test data was the input.
    """
    ctx = _context(form, cache, steps)
    out = dict(_result(ctx, step, cancelled))
    out["test_data"] = ctx.test_root is not None
    out["well"] = ctx.well
    return out
