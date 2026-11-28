"""Run the full OPS phases on the Cell Painting Gallery cropped-site layout (item 1058, alpha).

The cropped sites of PERISCOPE (cpg0021) are already one field each, so the
standard OPS phases need no stitching and no registration here. This module
maps the layout onto the inputs those phases already read and lets
:func:`spacr.ops_engine.run_ops` run them unchanged:

* the sequencing planes of each site become a ``cycle -> site -> channel ->
  path`` index (``DAPI`` for ``Cycle01_DAPI``, one token per base);
* the painting planes become the phenotype index of the well
  (``DAPI``, ``PHALLOIDIN``, ``MITO``, ``ER``, ``WGA``);
* the placement phases write the same ``ops_geometry`` and ``ops_phenotype``
  rows a stitch and a phenotype placement would write: every site is a tile of
  its own, laid out in a grid with a gap wider than a segmentation window, and
  the phenotype field of a site is its own sequencing tile (scale 1, no
  rotation) moved by PERISCOPE's logged painting-to-sequencing shift.

Everything after the placement (objects, ph_flatfield, ph_segment,
ph_measure, aggregate and the rest) is the engine's own code, so each well's
``ops_report.json`` carries the same provenance block, with the md5 of the
Cellpose weights of every role that ran. The spacer-prefix decode
(``prefix_decode``) reads the cell labels ``ph_segment`` stored, in the
painting frame.
"""
from __future__ import annotations

import contextvars
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_PREPARED: contextvars.ContextVar = contextvars.ContextVar("cpg_full_prepared", default=None)

GRID_GAP_PX = 256

_PAINTING_TOKENS = {"CORRDNA": "DAPI", "CORRPHALLOIDIN": "PHALLOIDIN", "CORRMITO": "MITO",
                    "CORRER": "ER", "CORRWGA": "WGA"}


def painting_token(channel: str) -> str:
    """The channel name the phenotype phases use for a painting folder.

    :param channel: a folder such as ``CorrDNA``.
    :returns: ``DAPI`` for the nuclear stain, else the stain upper-cased.
    """
    name = str(channel).strip()
    known = _PAINTING_TOKENS.get(name.upper())
    if known:
        return known
    return (name[4:] if name.lower().startswith("corr") else name).upper()


def current() -> Optional["Prepared"]:
    """The layout of the cropped-site run in progress, or None."""
    return _PREPARED.get()


class Prepared:
    """One cropped-site run mapped onto the tile inputs of the OPS phases.

    :param settings: the OPS settings with ``ops_layout`` ``'cpg_cropped'``.
    """

    def __init__(self, settings: Mapping[str, Any]):
        from .cpg_cropped import cpg_site_index

        checked = cpg_site_index(settings)
        self.settings = dict(settings)
        self.pinned = checked["pinned"]
        self.frame: pd.DataFrame = checked["index"]
        self.root = self.pinned["cpg_root"]
        self.plate = str(settings.get("plate") or self.frame["plate"].iloc[0])
        self.channels = [painting_token(c) for c in self.pinned["cpg_phenotype_channels"]]
        self.index: Dict[str, Dict[int, Dict[int, Dict[str, str]]]] = {}
        self.phenotype: Dict[str, Dict[int, Dict[int, Dict[str, str]]]] = {}
        self.folders: Dict[str, str] = {}
        self.sites: Dict[str, List[int]] = {}
        self._placed: Dict[tuple, Dict[str, Any]] = {}
        painting = dict(zip(self.pinned["cpg_phenotype_channels"], self.channels))
        for (well_id, well), rows in self.frame.groupby(["well_id", "well"], sort=True):
            cycles: Dict[int, Dict[int, Dict[str, str]]] = {}
            fields: Dict[int, Dict[str, str]] = {}
            for row in rows.itertuples():
                if row.modality == "phenotype":
                    fields.setdefault(int(row.site), {})[painting.get(row.channel, row.channel)] = row.path
                else:
                    token = "DAPI" if row.base == "DAPI" else row.base
                    cycles.setdefault(int(row.cycle), {}).setdefault(int(row.site), {})[token] = row.path
            self.index[well_id] = cycles
            self.phenotype[well_id] = {1: fields}
            self.folders[well_id] = well
            self.sites[well_id] = sorted(fields)

    @property
    def wells(self) -> List[str]:
        """The engine names of the wells, as ``W1``."""
        return sorted(self.index)

    def tile_shape(self, well: str) -> tuple:
        """The ``(height, width)`` of the well's tiles, from a TIFF header.

        :param well: the engine well name.
        :returns: the tile height and width.
        """
        rows = self.frame[(self.frame["well_id"] == well) & (self.frame["modality"] == "phenotype")]
        height, width = (int(v) for v in str(rows["shape"].iloc[0]).split("x")[:2])
        return height, width

    def shift(self, well: str, site: int) -> Dict[str, Any]:
        """The painting-to-sequencing shift of one site (never a silent zero).

        :param well: the engine well name.
        :param site: the site.
        :returns: the record of :func:`spacr.cpg_cropped.painting_to_sbs_shift`.
        """
        from .cpg_cropped import painting_to_sbs_shift

        return painting_to_sbs_shift(
            self.frame, self.plate, well, site, cpg_root=self.root,
            source=str(self.settings.get("cpg_align_source") or "periscope_logged"),
            template=self.settings.get("cpg_align_table"))

    def place_stitch(self, db: str, well: str) -> Dict[str, Any]:
        """Write the ``ops_geometry`` rows: one tile per site on a gapped grid.

        :param db: the measurements database.
        :param well: the engine well name.
        :returns: the stitch report.
        """
        from .ops_engine import _replace_well_rows

        if ("stitch", well) in self._placed:
            return self._placed[("stitch", well)]
        height, width = self.tile_shape(well)
        sites = self.sites[well]
        columns = int(np.ceil(np.sqrt(len(sites))))
        rows = []
        for number, site in enumerate(sites):
            row, column = divmod(number, columns)
            rows.append({"plate": self.plate, "well": well, "cycle": 1, "site": site,
                         "y": float(row * (height + GRID_GAP_PX)),
                         "x": float(column * (width + GRID_GAP_PX)),
                         "origin_y": 0.0, "origin_x": 0.0,
                         "tile_height": height, "tile_width": width})
        _replace_well_rows(db, "ops_geometry", pd.DataFrame(rows), self.plate, well)
        self._placed[("stitch", well)] = {"placement": "cpg_cropped", "sites": len(sites), "placed": len(sites),
                "grid_columns": columns, "gap_px": GRID_GAP_PX, "edges_proposed": None,
                "edges_accepted": None}
        return self._placed[("stitch", well)]

    def place_phenotype(self, db: str, well: str) -> Dict[str, Any]:
        """Write the ``ops_phenotype`` rows: each field on its own tile, moved by its logged shift.

        :param db: the measurements database.
        :param well: the engine well name.
        :returns: the phenotype report.
        """
        from .ops_engine import _placements, _ph_transform_columns, _replace_well_rows

        if ("phenotype", well) in self._placed:
            return self._placed[("phenotype", well)]
        placements, shape = _placements(db, self.plate, well)
        rows = []
        shifts = {}
        for site in self.sites[well]:
            found = self.shift(well, site)
            shifts[site] = (int(found["dy"]), int(found["dx"]))
            y, x = placements[site]
            rows.append({"plate": self.plate, "well": well, "site": site,
                         "centre_y": y + shape[0] / 2.0 + shifts[site][0],
                         "centre_x": x + shape[1] / 2.0 + shifts[site][1],
                         "sbs_site": site, "is_anchor": 0, "inliers": 0,
                         "alignment_residual_px": float("nan"),
                         "raster_residual_px": float("nan")})
        frame = _ph_transform_columns(pd.DataFrame(rows), 1.0, 0.0, shape)
        frame["transform_source"] = "cpg_cropped"
        _replace_well_rows(db, "ops_phenotype", frame, self.plate, well)
        self._placed[("phenotype", well)] = {
            "placement": "cpg_cropped", "fields": len(rows), "fields_mapped": len(rows),
            "painting_shift_px": {str(k): list(v) for k, v in shifts.items()},
            "ops_phenotype_rows": len(rows)}
        return self._placed[("phenotype", well)]

    def prefix_decode(self, settings: Mapping[str, Any], well: str, destination: str,
                      report=None) -> Dict[str, Any]:
        """Decode one well from the cell labels ``ph_segment`` stored (painting frame).

        :param settings: the OPS settings of the run.
        :param well: the engine well name.
        :param destination: ``dst_root``.
        :param report: progress callback.
        :returns: the report of :func:`spacr.ops_prefix.run_prefix_decode`.
        """
        from .ops_prefix import run_prefix_decode

        chosen = dict(settings)
        chosen.update({
            "cpg_wells": [self.folders[well]],
            "cpg_sites": list(self.sites[well]),
            "ops_prefix_masks": os.path.join(destination, well, "phenotype_masks",
                                             "site-{site}.npz"),
            "ops_prefix_masks_frame": "painting"})
        return run_prefix_decode(chosen, report=report)


def run_cpg_full(settings: Mapping[str, Any], phases: Sequence[str],
                 report=None) -> Dict[str, Any]:
    """Run the OPS phases on a cropped-site layout (item 1058, alpha).

    :param settings: the OPS settings with ``ops_layout`` ``'cpg_cropped'``.
    :param phases: the phases asked for: any of the engine's phases, and the
        cropped-site phases ``cpg_index``, ``cpg_merged`` and
        ``prefix_decode``.
    :param report: progress callback.
    :returns: the result of :func:`spacr.ops_engine.run_ops`.
    """
    from .ops_engine import run_ops

    prepared = Prepared(settings)
    if not settings.get("dst_root"):
        raise ValueError("the cropped-site phases need dst_root")
    chosen = dict(settings)
    chosen.update({
        "genotype_source": prepared.root, "phenotype_source": prepared.root,
        "plate": prepared.plate,
        "ops_base_channels": "G,T,A,C",
        "ops_ph_channels": ",".join(prepared.channels),
        "ops_ph_cell_channel": "PHALLOIDIN",
        "ops_ph_image_format": "npz"})
    wanted = list(phases)
    engine = ["stitch", "phenotype"] + [p for p in wanted if p not in ("stitch", "phenotype")]
    db = os.path.join(str(settings["dst_root"]), "measurements.db")
    os.makedirs(str(settings["dst_root"]), exist_ok=True)
    for well in prepared.wells:
        prepared.place_stitch(db, well)
        prepared.place_phenotype(db, well)
    token = _PREPARED.set(prepared)
    try:
        return run_ops(chosen, phases=engine)
    finally:
        _PREPARED.reset(token)
