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

import numpy as np

_STEPS: Tuple[str, ...] = ("align", "spots", "objects", "merge", "cluster")

_STEP_NEEDS: Dict[str, Tuple[str, ...]] = {
    "align": (),
    "spots": ("align",),
    "objects": ("align",),
    "merge": ("align",),
    "cluster": ("merge",),
}

_EVERY_STEP = _STEPS

_SHARED_CATEGORY_WORDS: Tuple[str, ...] = ("input", "performance")

_STEM_STEPS: Tuple[Tuple[str, str], ...] = (
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


def _steps_of(key: str, category: str) -> Tuple[str, ...]:
    """The tabs a setting belongs to, read from its registry category.

    :param key: the setting.
    :param category: its category title in the settings registry.
    :returns: one step, or every step for an input or performance setting
        and for a setting no rule places.
    """
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
    return {key: _steps_of(key, categories.get(key, ""))
            for key in OPS_DEFAULTS}


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


def _test_data_root() -> Optional[Path]:
    """The unpacked OPS end-to-end test data, full tier first, or None."""
    from .example_archives import OPS_E2E_FULL_FOLDER, example_set_folder

    core = example_set_folder("ops_e2e")
    for folder in (core.parent / OPS_E2E_FULL_FOLDER, core):
        root = folder / _TEST_FOLDER
        if (root / "settings" / "ops_testdata_settings.json").is_file() and (
                root / "reference" / "PROVENANCE_rev2_addendum.json").is_file():
            return root
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
        the form is used as it is. Without one the test data's shipped
        settings are the base and every form value that differs from the OPS
        default is laid over them, so an edit on the form is what the preview
        tests.
    """
    from .ops_settings import OPS_DEFAULTS

    from .settings import _fold_renamed_settings

    form = _fold_renamed_settings(dict(form or {}))
    source = form.get("genotype_source")
    if source and os.path.isdir(str(source)):
        return form, None
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

    :param settings: read for the two image folders and the plate.
    :returns: ``~/.cache/spacr/ops_live/<digest>``, created.
    """
    text = "|".join(str(settings.get(key) or "") for key in
                    ("genotype_source", "phenotype_source", "plate"))
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


def _sample_block(index_well: Mapping[int, Mapping[int, Any]],
                  reference: int) -> List[int]:
    """A 2 x 2 block of sequencing tiles near the middle of the well.

    :param index_well: ``cycle -> site -> files`` of the well.
    :param reference: the cycle holding the nuclear stain.
    :returns: the sites, every one in the reference cycle.
    """
    from .ops_layout import round_well_layout

    sites = sorted(index_well[reference])
    if len(sites) <= 4:
        return sites
    layout = round_well_layout(max(sites) + 1)
    where = {site: layout.position(site) for site in sites}
    centre = np.mean(np.array(list(where.values()), float), axis=0)
    by_place = {place: site for site, place in where.items()}
    ranked = sorted(sites, key=lambda s: float(np.hypot(*(np.array(where[s]) - centre))))
    for site in ranked:
        column, row = where[site]
        block = [by_place.get((column + dc, row + dr))
                 for dc in (0, 1) for dr in (0, 1)]
        if all(b is not None for b in block):
            return sorted(int(b) for b in block)
    return ranked[:4]


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

    gpu = bool(ctx.settings.get("ops_gpu", False))
    name = (str(ctx.settings.get(key) or ""), gpu,
            ctx.settings.get("mask_inference_precision"))
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

    if not os.path.isfile(db):
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
        crop and its cycle shifts.
    """
    from .ops_cycles import align_field
    from .ops_engine import (_base_channels, _NUCLEAR, _plane_sources,
                             _registration_choice, _setting_number, run_ops)

    from .ops_engine import _ops_sites

    index, reference = _sbs_index(ctx)
    asked = _ops_sites(ctx.settings)
    block = (sorted(asked)[:16] if asked else _sample_block(index, reference))
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
    tiles = {int(r.site): (float(r.y), float(r.x)) for r in geometry.itertuples()}
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
                           "scale": float(row.scale),
                           "rotation": float(row.rotation_deg),
                           "shape": (int(row.field_height), int(row.field_width)),
                           "files": ph_files.get(site)})
    sample_field = _sample_field(ctx, fields)
    crop_site, crop = _sbs_crop(tiles, shape, sample_field,
                                ctx.test_root is not None)
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
        planes, reference=reference, gpu=bool(ctx.settings.get("ops_gpu", False)),
        channel_mode=str(ctx.settings.get("ops_channel_registration")
                         or _CHANNEL_REGISTRATION).strip().lower(),
        **registration)
    last = aligned.kept[-1]
    before = (np.mean(raw[reference], axis=0), np.mean(raw[last], axis=0))
    after = (aligned.stack[0].mean(axis=0), aligned.stack[-1].mean(axis=0))
    return {
        "db": db, "plate": plate, "reference": reference, "block": block,
        "tiles": tiles, "tile_shape": shape, "origin": (top, left),
        "mosaic": mosaic, "step": step, "fields": fields,
        "sample_field": sample_field, "crop_site": crop_site, "crop": crop,
        "cycle_shifts": {int(k): tuple(v) for k, v in aligned.cycle_shifts.items()},
        "kept": list(aligned.kept), "refused": list(aligned.refused),
        "missing": list(aligned.missing), "last_cycle": int(last),
        "before": before, "after": after, "notes": notes,
        "phases": phases, "sample_block": settings["ops_sites"],
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
              field_info: Optional[Mapping[str, Any]], test: bool = False
              ) -> Tuple[int, Tuple[int, int, int]]:
    """The sequencing tile and crop under the sampled phenotype crop.

    :param tiles: ``site -> (y, x)`` placements of the reference cycle.
    :param shape: the tile shape.
    :param field_info: the sampled phenotype field, or None.
    :param test: the input is the test data (see :func:`_ph_crop_window`).

    :returns: ``(site, (row, column, size))`` in the tile's own pixels; the
        middle of the most central tile when no field is placed.
    """
    size = min(_SBS_CROP, shape[0], shape[1])
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


def _decode_options(settings: Mapping[str, Any], library: Sequence[str]
                    ) -> Dict[str, Any]:
    """The per-field decode options, read from the settings as ``_decode`` reads them.

    Mirrors the option block of :func:`spacr.ops_engine._decode` for one
    field in manual threshold mode, with the engine's own readers. The auto
    threshold mode tunes over a whole well, so a crop is decoded at the
    entered thresholds.

    :returns: the task keys every field shares, ``rule`` and the decoys.
    """
    from .ops_decode import decode_rule, resolve_settings
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
    site = align["crop_site"]
    y0, x0, size = align["crop"]
    sources = engine._plane_sources(index[reference][site])
    dapi = _read(ctx, sources.get(engine._NUCLEAR))
    if dapi is None:
        raise ValueError(f"tile {site} has no readable DAPI plane")
    dapi = np.ascontiguousarray(dapi[y0:y0 + size, x0:x0 + size])
    nuclei_key = json.dumps([site, align["crop"], ctx.settings.get("ops_sbs_nucleus_model"),
                             ctx.settings.get("cellpose_diameter"),
                             ctx.settings.get("ops_gpu"),
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
                            gpu=bool(ctx.settings.get("ops_gpu"))) as resident:
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
    options = _decode_options(ctx.settings, library)
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
        "gpu": bool(ctx.settings.get("ops_gpu", False)),
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
        "site": site, "crop": align["crop"], "dapi": dapi, "nuclei": nuclei,
        "peaks": peaks, "strength": strength, "reads": list(letters),
        "read_quality": np.asarray(quality, float), "cells": rows,
        "n_nuclei": int(len(labels)), "n_cycles": int(result.get("n_cycles") or 0),
        "rule": "rule (a)" if result.get("rule_a") is not None else options["rule"],
        "threshold": float(options["task"]["threshold"]),
        "library": len(library),
    }


def _run_objects(ctx: _Context, cancelled=None) -> Dict[str, Any]:
    """Segment and measure one phenotype crop as the ``ph_segment`` and ``ph_measure`` phases do.

    :returns: the crop, its nuclei and cells, any extra object layers and
        the measured cell table.
    """
    from . import ops_engine as engine
    from .ops_sample import measure_phenotype_field

    align = _result(ctx, "align", cancelled)
    field_info = align["sample_field"]
    if field_info is None:
        files = _ph_index(ctx)
        if not files:
            raise ValueError("no phenotype_source, so there is no field to segment")
        site = sorted(files)[0]
        field_info = {"site": site, "files": files[site], "shape": None}
    channels = engine._ph_channels(ctx.settings)
    secondary = None
    unreadable: list = []
    key = ("_ph_stack", int(field_info["site"]), tuple(channels))
    held = ctx.cache.get("_ph_stack")
    if held is not None and held[0] == key:
        stack = held[1]
    else:
        stack = engine._ph_read_field(field_info["files"], channels, unreadable,
                                      secondary)
        if stack is None:
            raise ValueError(f"phenotype field {field_info['site']} could not be "
                             f"read: {unreadable[:1]}")
        ctx.cache["_ph_stack"] = (key, stack)
    _check(cancelled)
    info = dict(field_info)
    info["shape"] = stack.shape[:2]
    row, column, size = _ph_crop_window(info, bool(ctx.test_root))
    crop = np.ascontiguousarray(stack[row:row + size, column:column + size])
    masks_key = json.dumps([key, info["shape"], (row, column, size)] + [
        ctx.settings.get(name) for name in (
            "ops_ph_nucleus_model", "ops_ph_cell_model", "ops_ph_nucleus_diameter",
            "ops_ph_cell_diameter", "ops_ph_downsample", "mask_eval_batch_size",
            "ops_ph_tile_batch", "mask_inference_precision", "batch_size",
            "ops_ph_cell_channel", "ops_gpu")], default=str)
    held = ctx.cache.get("_ph_masks")
    if held is not None and held[0] == masks_key:
        nuclei, cells = held[1]
    else:
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
                            gpu=bool(ctx.settings.get("ops_gpu"))) as resident:
            for _site, image, _metadata, field_count in resident:
                nuclei, cells, _paired = engine._ph_segment_field(
                    image, channels, models, ctx.settings, field_count=field_count)
        ctx.cache["_ph_masks"] = (masks_key, (nuclei, cells))
    _check(cancelled)
    layers: Dict[str, np.ndarray] = {}
    for layer in _OBJECT_LAYERS:
        layers.update(layer(ctx.settings, crop, channels, nuclei, cells) or {})
    table = measure_phenotype_field(
        nuclei, cells, crop, channels,
        texture=bool(ctx.settings.get("ops_ph_texture", True)),
        funk_features=bool(ctx.settings.get("ops_ph_funk_features", False)),
        zernike=bool(ctx.settings.get("ops_ph_zernike", False)),
        gpu=bool(ctx.settings.get("ops_ph_measure_gpu", False)),
        profile_features=engine._ph_profile_features(ctx.settings),
        granularity_subsample=float(
            engine._ph_granularity_subsample(ctx.settings) or 1.0))
    return {"site": int(field_info["site"]), "window": (row, column, size),
            "stack": crop, "channels": list(channels), "nuclei": nuclei,
            "cells": cells, "layers": layers, "table": table,
            "cell_channel": str(ctx.settings.get("ops_ph_cell_channel") or "AF750")}


def _aggregation_inputs(ctx: _Context, db: str, plate: str
                        ) -> Dict[str, Any]:
    """Put the aggregation set's nuclei, calls and cells in the workspace.

    For the test data these are its stored step outputs (``reference/``);
    for a user's data, the same tables of the user's finished run
    (``dst_root``), read and copied, never written.

    :returns: what was used, for the tab to say.
    :raises ValueError: when there is no finished run to read.
    """
    import shutil

    import pandas as pd

    from .ops_engine import _ph_cells_path, _replace_well_rows

    target = _ph_cells_path(str(ctx.folder), ctx.well)
    if ctx.test_root is not None:
        ref = ctx.test_root / "reference"
        for table, path in (("ops_objects", "objects/ops_objects.parquet"),
                            ("ops_barcodes", "genotype/ops_barcodes.parquet")):
            _replace_well_rows(db, table, pd.read_parquet(ref / path), plate,
                               ctx.well)
        shutil.copy(ref / "phenotype" / f"ops_phenotype_cells_{ctx.well}.parquet",
                    target)
        masks = ctx.folder / ctx.well / "phenotype_masks"
        masks.mkdir(parents=True, exist_ok=True)
        for path in sorted((ref / "phenotype" / "phenotype_masks").glob("site-*.npz")):
            if not (masks / path.name).is_file():
                shutil.copy(path, masks / path.name)
        return {"kind": "test", "path": str(ref), "masks": True}
    run = str(ctx.settings.get("dst_root") or "")
    run_db = os.path.join(run, "measurements.db")
    cells = _ph_cells_path(run, ctx.well)
    if not (os.path.isfile(run_db) and os.path.isfile(cells)):
        raise ValueError("merge and aggregation need a finished run in dst_root "
                         "(measurements.db and the phenotype cell table)")
    for table in ("ops_objects", "ops_barcodes"):
        rows = _rows(run_db, table, plate, ctx.well)
        if rows is None or rows.empty:
            raise ValueError(f"the run in dst_root has no {table} rows")
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
        raise ValueError("no phenotype field is placed, so there is nothing to merge")
    db, plate = align["db"], align["plate"]
    used = _aggregation_inputs(ctx, db, plate)
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
    profiles = _profiles_root() if ctx.test_root is not None else None
    if profiles is not None:
        genes, used = _profile_genes(ctx.settings, profiles), {
            "kind": "profiles", "path": str(profiles), "base": used}
    return {"report": merge_report, "used": used, "genes": genes,
            "merged": merged, "objects": objects, "cells": placed,
            "per_field": per_field, "align": align,
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
        raise ValueError("the aggregate phase wrote no gene table")
    floor = int(_first(ctx.settings.get("ops_profile_min_cells"), int, 0))
    kept = genes
    if "n_cells" in genes.columns:
        kept = genes[genes["n_cells"] >= floor]
    prefix = str(ctx.settings.get("ops_control_prefix") or "nontargeting").lower()
    names = kept["gene"].astype(str)
    kept = kept[~names.str.lower().str.startswith(prefix)].reset_index(drop=True)
    numeric = [c for c in kept.columns
               if c not in ("gene", "n_cells", "n_guides", "sgRNA")
               and np.issubdtype(kept[c].dtype, np.number)
               and not c.startswith(("p_", "q_", "n_"))]
    if len(kept) < 5 or not numeric:
        raise ValueError(f"{len(kept)} genes have at least {floor} cells "
                         "(ops_profile_min_cells); clustering needs 5 or more")
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
            "string": _string_score(kept["gene"].astype(str).tolist(), labels, prefix),
            "n_components": n_components, "n_features": n_features,
            "labels": labels, "kind": kind, "k": k, "resolution": resolution,
            "seed": seed, "floor": floor, "note": note,
            "total_genes": int(len(genes))}


_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "align": _run_align,
    "spots": _run_spots,
    "objects": _run_objects,
    "merge": _run_merge,
    "cluster": _run_cluster,
}


def _run_step(form: Mapping[str, Any], step: str, cache: dict,
              cancelled: Optional[Callable[[], bool]] = None,
              steps: Optional[Dict[str, Tuple[str, ...]]] = None
              ) -> Dict[str, Any]:
    """Run one tab's step with the form's settings, reusing the cache.

    :param form: the settings on the OPS form.
    :param step: one of the five steps.
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
