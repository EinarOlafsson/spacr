"""Measure PERISCOPE painting images through sequencing-frame masks (item 998, alpha).

PERISCOPE (cpg0021, CP186A/B) publishes its corrected painting crops
(``CorrDNA`` and the other painting channels) in the painting frame, but its
own Cells and Nuclei masks are in the sequencing (SBS) frame. Its 04Align
module records the shift between the two frames per site
(``Align_Yshift_DAPI_Painting`` and ``Align_Xshift_DAPI_Painting`` in the
site's ``Image.csv``; 3-77 px on CP186A Well 1). Painting features measured
through those masks without that shift read the wrong pixels.

The ``cpg_measure`` phase reads each site's painting planes, moves every plane
into the mask frame by the site's shift (``cpg_painting_frame_shift``),
measures it with :func:`spacr.ops_sample.measure_phenotype_field`, flags the
cells that touch the zero-filled band the shift leaves at the field border
(``painting_shift_border``) and records every applied shift in
``painting_frame_shift_<plate>_<well>.csv``.

``cpg_painting_frame_shift``:

* ``'periscope_logged'`` (default) takes the shift PERISCOPE recorded
  (:func:`spacr.cpg_cropped.painting_to_sbs_shift`, ``cpg_align_table``).
* ``'register'`` measures it by phase correlation of ``CorrDNA`` on
  ``Cycle01_DAPI``.
* ``'none'`` moves nothing; it is valid only for masks drawn on the painting
  images (``ops_prefix_masks_frame 'painting'``).

The masks' frame is read from ``ops_prefix_masks_frame`` (item 919), never
guessed from the pixels. Sequencing-frame masks with no shift are refused, a
site without a recorded shift is refused (never given a zero shift), and every
site's shift is found before anything is written.

Typical use::

    from spacr.cpg_measure import run_cpg_measure

    run_cpg_measure({"ops_layout": "cpg_cropped", "cpg_root": ".../CP186A_Well1",
                     "dst_root": "/tmp/run", "ops_prefix_masks": ".../{site}_cells.tiff",
                     "ops_prefix_nuclei_masks": ".../{site}_nuclei.tiff",
                     "cpg_painting_frame_shift": "periscope_logged"})
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .qt.i18n import tr

#: How ``cpg_painting_frame_shift`` moves the painting images into the frame
#: of the masks; the first is the default.
PAINTING_FRAME_SHIFTS = ("periscope_logged", "register", "none")

#: The columns of ``painting_frame_shift_<plate>_<well>.csv``, in order.
SHIFT_TABLE_COLUMNS = ("plate", "well", "site", "mode", "dy", "dx", "source",
                       "border_y_px", "border_x_px", "cells", "border_cells")

#: The settings the phase reads besides the cropped-site reader's, stamped
#: into ``cpg_measure_settings.json``.
MEASURE_SETTING_KEYS = ("cpg_painting_frame_shift", "cpg_align_table",
                        "ops_prefix_masks", "ops_prefix_nuclei_masks",
                        "ops_prefix_masks_frame", "ops_ph_texture", "ops_ph_funk_features",
                        "ops_ph_zernike", "ops_ph_profile_features",
                        "ops_ph_profile_optional_families", "ops_ph_granularity_subsample")

_NO_RECORD = ("No recorded Align shift for site {site} of {plate} {well}; the painting "
              "images cannot be matched to the masks. Set cpg_align_table or use Register.")
_GUARD = ("These masks are in the sequencing frame but the painting images are not "
          "moved. Set cpg_painting_frame_shift to \"periscope_logged\" or \"register\".")
_PAINTING_MASKS = ("These masks are in the painting frame, so the painting images must "
                   "not be moved. Set cpg_painting_frame_shift to \"none\".")
_UNKNOWN_MODE = "cpg_painting_frame_shift must be one of {choices}; got {value}."
_UNKNOWN_FRAME = "ops_prefix_masks_frame must be one of {choices}; got {value}."
_NO_MASKS = ("No cell or nucleus labels for site {site} of {plate} {well}. Set "
             "ops_prefix_masks and ops_prefix_nuclei_masks.")
_SHAPES = ("The masks of site {site} of {plate} {well} are {masks}, but its painting "
           "images are {images}.")
_NO_DESTINATION = "The cpg_measure phase needs dst_root."


def painting_frame_mode(settings: Mapping[str, Any]) -> str:
    """The ``cpg_painting_frame_shift`` a measurement may use, after the frame guard.

    :param settings: read for ``cpg_painting_frame_shift`` and
        ``ops_prefix_masks_frame`` (``'sbs'`` when empty, as item 919 reads it).
    :returns: ``'periscope_logged'``, ``'register'`` or ``'none'``.
    :raises ValueError: for an unknown mode or frame; for sequencing-frame
        masks while the shift is ``'none'`` or unset; for painting-frame masks
        while the shift is anything but ``'none'``.
    """
    from .ops_prefix import MASK_FRAMES

    raw = settings.get("cpg_painting_frame_shift")
    mode = "" if raw is None else str(raw).strip().lower()
    if mode and mode not in PAINTING_FRAME_SHIFTS:
        raise ValueError(tr(_UNKNOWN_MODE, choices=list(PAINTING_FRAME_SHIFTS), value=raw))
    raw_frame = settings.get("ops_prefix_masks_frame")
    frame = str(raw_frame or MASK_FRAMES[0]).strip().lower()
    if frame not in MASK_FRAMES:
        raise ValueError(tr(_UNKNOWN_FRAME, choices=list(MASK_FRAMES), value=raw_frame))
    if frame == "sbs" and mode in ("", "none"):
        raise ValueError(tr(_GUARD))
    if frame == "painting" and mode != "none":
        raise ValueError(tr(_PAINTING_MASKS))
    return mode


def site_painting_shift(index: pd.DataFrame, plate: str, well: str, site: int, *,
                        mode: str, cpg_root: str,
                        template: Optional[str] = None) -> Dict[str, Any]:
    """The whole-pixel shift that moves one site's painting planes into the mask frame.

    :param index: from :func:`spacr.cpg_cropped.cpg_cropped_index`.
    :param plate: the plate.
    :param well: ``Well1`` or ``W1``.
    :param site: the site.
    :param mode: a :data:`PAINTING_FRAME_SHIFTS` value.
    :param cpg_root: the reader's root, to find PERISCOPE's tables.
    :param template: ``cpg_align_table``.
    :returns: ``dy``, ``dx`` (as :func:`spacr.cpg_cropped.shift_image` takes
        them) and ``source``: the ``Image.csv`` path, ``register`` with the
        peak ratio, or ``none``.
    :raises ValueError: under ``'periscope_logged'`` for a site with no Align
        record or non-finite values, and under ``'register'`` for a
        registration that fails the item-919 acceptance, naming the site.
    """
    from .cpg_cropped import painting_to_sbs_shift

    if mode == "none":
        return {"dy": 0, "dx": 0, "source": "none"}
    try:
        found = painting_to_sbs_shift(index, plate, well, site, cpg_root=cpg_root,
                                      source=mode, template=template)
    except (ValueError, OverflowError) as exc:
        if mode == "register":
            raise
        raise ValueError(tr(_NO_RECORD, site=int(site), plate=plate, well=well)) from exc
    if not (np.isfinite(found["dy"]) and np.isfinite(found["dx"])):
        raise ValueError(tr(_NO_RECORD, site=int(site), plate=plate, well=well))
    source = (str(found["record"]) if mode == "periscope_logged"
              else f"register (peak ratio {float(found['peak_ratio']):.4g})")
    return {"dy": int(found["dy"]), "dx": int(found["dx"]), "source": source}


def shift_painting(stack: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move every plane of a ``(Y, X, C)`` painting stack by whole pixels.

    :param stack: the painting planes, channels last.
    :param dy: rows to move down (negative moves up).
    :param dx: columns to move right (negative moves left).
    :returns: a new stack of the same shape and dtype; the uncovered band is
        zero and nothing wraps (:func:`spacr.cpg_cropped.shift_image`).
    """
    from .cpg_cropped import shift_image

    stack = np.asarray(stack)
    return np.stack([shift_image(stack[..., k], dy, dx) for k in range(stack.shape[-1])],
                    axis=-1)


def shift_border_labels(nuclei: np.ndarray, cells: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """The labels whose nucleus or cell touches the zero band of a shift.

    The band is the ``|dy|`` rows and ``|dx|`` columns a
    :func:`shift_painting` move leaves empty, grown by one pixel so an
    object that only borders it counts too.

    :param nuclei: the nucleus labels in the mask frame.
    :param cells: the cell labels, the same grid.
    :param dy: the applied row shift.
    :param dx: the applied column shift.
    :returns: the sorted labels, as int64.
    """
    height, width = np.asarray(cells).shape[:2]
    band = np.zeros((height, width), bool)
    dy, dx = int(dy), int(dx)
    if dy > 0:
        band[:min(dy + 1, height)] = True
    elif dy < 0:
        band[max(height + dy - 1, 0):] = True
    if dx > 0:
        band[:, :min(dx + 1, width)] = True
    elif dx < 0:
        band[:, max(width + dx - 1, 0):] = True
    found = np.union1d(np.asarray(nuclei)[band], np.asarray(cells)[band])
    return found[found > 0].astype(np.int64)


def painting_column_names(channels: Sequence[str]) -> List[str]:
    """The short channel names the feature columns carry.

    :param channels: the painting channels, for example ``CorrDNA``.
    :returns: each name lower-cased without a leading ``Corr``
        (``CorrDNA`` gives ``dna``).
    """
    out = []
    for name in channels:
        text = str(name)
        if text.lower().startswith("corr") and len(text) > 4:
            text = text[4:]
        out.append(text.lower())
    return out


def measure_painting_site(painting: np.ndarray, nuclei: np.ndarray, cells: np.ndarray,
                          channel_names: Sequence[str], *, dy: int, dx: int,
                          texture: bool = True, funk_features: bool = False,
                          zernike: bool = False, profile_features=None,
                          granularity_subsample: Optional[float] = None) -> pd.DataFrame:
    """One site's painting features through its masks, after the frame shift.

    :param painting: ``(Y, X, C)`` painting planes as stored.
    :param nuclei: nucleus labels in the mask frame; nucleus ``k`` lies in
        cell ``k``.
    :param cells: cell labels in the mask frame.
    :param channel_names: one short name per channel.
    :param dy: the row shift (:func:`site_painting_shift`); 0 with ``dx`` 0
        measures the planes unmoved.
    :param dx: the column shift.
    :param texture: include the texture block.
    :param funk_features: add Funk et al. 2022's features.
    :param zernike: add the Zernike moments.
    :param profile_features: the profiling families
        (:func:`spacr.measure.measure_profile_features`), or None.
    :param granularity_subsample: the granularity sieve's subsample, or None.
    :returns: :func:`spacr.ops_sample.measure_phenotype_field`'s table plus
        ``painting_shift_border`` (the cell or its nucleus touches the band
        filled with zeros by the shift). Every feature of a cell away from
        the band is the feature of the aligned image, except ``frac_high90``
        and ``frac_low10``: they are thresholded on the whole field's
        percentiles, and the band's zeros are part of the field.
    """
    from .ops_sample import measure_phenotype_field

    moved = painting if (int(dy), int(dx)) == (0, 0) else shift_painting(painting, dy, dx)
    table = measure_phenotype_field(
        nuclei, cells, moved, channel_names, texture=bool(texture),
        funk_features=bool(funk_features), zernike=bool(zernike),
        profile_features=profile_features,
        granularity_subsample=float(granularity_subsample or 1.0))
    border = shift_border_labels(nuclei, cells, dy, dx)
    table["painting_shift_border"] = table["label"].isin(border).to_numpy(bool)
    return table


def _site_masks(settings: Mapping[str, Any], *, plate: str, well: str, well_id: str,
                site: int, destination: str):
    """One site's ``(nuclei, cells)`` labels from the item-919 mask settings."""
    from .cpg_cropped import merged_path
    from .ops_prefix import _site_nuclei, site_labels

    merged = merged_path(destination, plate, well_id, site)
    cells = site_labels(settings.get("ops_prefix_masks") or None, plate=plate, well=well,
                        well_id=well_id, site=site, merged=merged)
    nuclei = _site_nuclei(settings.get("ops_prefix_nuclei_masks") or None, plate=plate,
                          well=well, well_id=well_id, site=site, merged=merged)
    if cells is None or nuclei is None:
        raise ValueError(tr(_NO_MASKS, site=int(site), plate=plate, well=well))
    return np.asarray(nuclei, np.int32), np.asarray(cells, np.int32)


def measure_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings a ``cpg_measure`` run reads, as stamped into its settings file.

    :param settings: the OPS settings.
    :returns: ``ops_layout``, the pinned cropped-site reader settings
        (:func:`spacr.cpg_cropped.cpg_settings`), ``dst_root`` and every
        :data:`MEASURE_SETTING_KEYS` value, with ``cpg_painting_frame_shift``
        as :func:`painting_frame_mode` resolves it and ``ops_ph_texture``
        (default True), ``ops_ph_funk_features`` and ``ops_ph_zernike``
        (default False) as booleans. Passing this back to
        :func:`run_cpg_measure` repeats the run.
    """
    from .cpg_cropped import cpg_settings

    out: Dict[str, Any] = {"ops_layout": "cpg_cropped", **cpg_settings(settings),
                           "dst_root": settings.get("dst_root")}
    for key in MEASURE_SETTING_KEYS:
        out[key] = settings.get(key)
    for key, default in (("ops_ph_texture", True), ("ops_ph_funk_features", False),
                         ("ops_ph_zernike", False)):
        out[key] = bool(default if settings.get(key) is None else settings.get(key))
    out["cpg_painting_frame_shift"] = painting_frame_mode(settings)
    return out


def run_cpg_measure(settings: Mapping[str, Any],
                    report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The ``cpg_measure`` phase: painting features through the site masks (item 998).

    Every site's shift is found first, so a refused site stops the run before
    any table is written. Writes, under ``<dst_root>/results/cpg_measure/``:
    ``cells_<plate>_<well>.parquet`` (one row per cell: ``plate``, ``well``,
    ``site``, the features of :func:`measure_painting_site` and
    ``painting_shift_border``), ``painting_frame_shift_<plate>_<well>.csv``
    (:data:`SHIFT_TABLE_COLUMNS`) and ``cpg_measure_settings.json``
    (:func:`measure_settings`).

    :param settings: the OPS settings of a cropped-site run: ``dst_root``,
        the ``cpg_*`` reader settings, ``cpg_painting_frame_shift``,
        ``cpg_align_table``, ``ops_prefix_masks`` and ``ops_prefix_nuclei_masks`` (path templates,
        or the merged site files' ``cells`` and ``nuclei``),
        ``ops_prefix_masks_frame`` and the ``ops_ph_*`` measurement options.
    :param report: called with progress lines.
    :returns: ``tables`` and ``shifts`` (the paths per well), ``settings``
        (the stamped file), ``mode``, ``cells``, ``border_cells`` and
        ``seconds``.
    :raises ValueError: without ``dst_root``; from :func:`painting_frame_mode`
        and :func:`site_painting_shift`; for a site without masks or with
        masks of another grid than its painting planes.
    """
    from .cpg_cropped import cpg_site_index, load_cpg_site
    from .ops_engine import _ph_granularity_subsample, _ph_profile_features
    from .tabular import write_table

    say = report or print
    started = time.perf_counter()
    destination = settings.get("dst_root")
    if not destination:
        raise ValueError(tr(_NO_DESTINATION))
    destination = str(destination)
    stamped = measure_settings(settings)
    mode = stamped["cpg_painting_frame_shift"]
    built = cpg_site_index(settings, md5=False)
    index, pinned = built["index"], built["pinned"]
    channels = list(pinned["cpg_phenotype_channels"])
    names = painting_column_names(channels)
    options = {"texture": stamped["ops_ph_texture"],
               "funk_features": stamped["ops_ph_funk_features"],
               "zernike": stamped["ops_ph_zernike"],
               "profile_features": _ph_profile_features(settings),
               "granularity_subsample": _ph_granularity_subsample(settings)}
    wells = []
    for (plate, well, well_id), rows in index.groupby(["plate", "well", "well_id"], sort=True):
        sites = sorted(int(s) for s in rows["site"].unique())
        shifts = {site: site_painting_shift(index, plate, well, site, mode=mode,
                                            cpg_root=pinned["cpg_root"],
                                            template=settings.get("cpg_align_table") or None)
                  for site in sites}
        wells.append((plate, well, well_id, sites, shifts))
    folder = os.path.join(destination, "results", "cpg_measure")
    os.makedirs(folder, exist_ok=True)
    result: Dict[str, Any] = {"tables": {}, "shifts": {}, "mode": mode, "cells": 0,
                              "border_cells": 0}
    for plate, well, well_id, sites, shifts in wells:
        tables, records = [], []
        for site in sites:
            painting = load_cpg_site(index, plate, well, site, phenotype_channels=channels,
                                     parts=("phenotype",))["phenotype"]
            nuclei, cells = _site_masks(settings, plate=plate, well=well, well_id=well_id,
                                        site=site, destination=destination)
            if cells.shape != painting.shape[:2] or nuclei.shape != cells.shape:
                raise ValueError(tr(_SHAPES, site=site, plate=plate, well=well,
                                     masks=f"{nuclei.shape} / {cells.shape}",
                                     images=tuple(painting.shape[:2])))
            found = shifts[site]
            table = measure_painting_site(painting, nuclei, cells, names, dy=found["dy"],
                                          dx=found["dx"], **options)
            for column, value in (("site", site), ("well", well_id), ("plate", plate)):
                table.insert(0, column, value)
            border = int(table["painting_shift_border"].sum()) if len(table) else 0
            records.append({"plate": plate, "well": well_id, "site": site, "mode": mode,
                            "dy": found["dy"], "dx": found["dx"], "source": found["source"],
                            "border_y_px": abs(found["dy"]), "border_x_px": abs(found["dx"]),
                            "cells": int(len(table)), "border_cells": border})
            tables.append(table)
            say(f"cpg_measure {plate} {well_id} site {site}: {len(table)} cells, shift "
                f"({found['dy']}, {found['dx']}), {border} at the shift border")
        stem = f"{plate}_{well_id}"
        result["tables"][stem] = write_table(pd.concat(tables, ignore_index=True),
                                             os.path.join(folder, f"cells_{stem}.parquet"),
                                             canonicalise=False)
        result["shifts"][stem] = write_table(
            pd.DataFrame(records, columns=list(SHIFT_TABLE_COLUMNS)),
            os.path.join(folder, f"painting_frame_shift_{stem}.csv"), canonicalise=False)
        result["cells"] += sum(r["cells"] for r in records)
        result["border_cells"] += sum(r["border_cells"] for r in records)
    from . import __version__

    result["settings"] = os.path.join(folder, "cpg_measure_settings.json")
    with open(result["settings"] + ".part", "w", encoding="utf-8") as handle:
        json.dump({**stamped, "spacr_version": __version__}, handle, indent=1, default=str)
    os.replace(result["settings"] + ".part", result["settings"])
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result
