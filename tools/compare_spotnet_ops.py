"""Compare SpotNet with spaCR's native spot detector on real OPS fields.

Item 475. Each named field of a well is decoded twice by the shipped
``spacr.ops_engine._decode_field`` -- once with ``spot_detector='native'``,
once with ``'spotnet'`` -- against the objects and placements an earlier
plate run stored in its measurements.db, so everything but the detector is
the same code on the same pixels. Reported per detector: spots found, reads
owned by a nucleus, objects given a barcode, the share of those barcodes in
the library, and seconds; and between them the share of each detector's
owned reads with the other's within 2 px.

SpotNet needs its environment (Model Zoo) and a DeepCell token in
DEEPCELL_ACCESS_TOKEN or ~/.spacr/deepcell_token; without them only the
native half runs and the report says why. The token is never printed.

    python tools/compare_spotnet_ops.py \\
        --tiles /mnt/wd4tb/spacr_testdata/ops/raw/ops/sequencing \\
        --db /mnt/wd4tb/spacr_testdata/ops_plate_run/measurements.db \\
        --plate 20200202_6W-LaC024A --well A1 --sites 331 332 \\
        --library /mnt/wd4tb/spacr_testdata/ops_plate_run/library/pool10_prefixes.csv \\
        --out features/data/475_spotnet_vs_native.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacr import ops_engine  # noqa: E402
from spacr._segmentation_backends import _spotnet_readiness  # noqa: E402
from spacr.ops_sbs import assign_reads_to_objects  # noqa: E402


def _tasks(tiles, db, plate, well, sites, reference, gpu=False):
    """The decode tasks for ``sites``, built as ``_decode`` builds them."""
    index = ops_engine._index_tiles(tiles, "cycled")
    cycle_files = index[well.upper()]
    placements, shape = ops_engine._placements(db, plate, well)
    objects = ops_engine._well_rows(db, "ops_objects", plate, well)
    oy = objects["centroid_y"].to_numpy(float)
    ox = objects["centroid_x"].to_numpy(float)
    areas = objects["area"].to_numpy(float)
    ids = objects["object_id"].to_numpy(np.int64)
    from scipy.spatial import cKDTree

    order = sorted(placements)
    centres = np.array([[placements[s][0] + shape[0] / 2,
                         placements[s][1] + shape[1] / 2] for s in order])
    _, nearest = cKDTree(centres).query(np.column_stack([oy, ox]))
    owner_site = np.asarray(order)[nearest]
    cycles = sorted(cycle_files)
    channels = ops_engine._base_channels({})
    out = []
    for site in sites:
        top, left = placements[site]
        near = np.flatnonzero((oy > top - 30) & (oy < top + shape[0] + 30)
                              & (ox > left - 30) & (ox < left + shape[1] + 30))
        planes = {}
        for cycle in cycles:
            sources = ops_engine._plane_sources(cycle_files[cycle].get(site, {}))
            planes[cycle] = [sources.get(name) for name in channels]
        out.append({
            "site": site, "planes": planes, "cycles": cycles,
            "reference": reference,
            "centroids": np.column_stack([oy[near] - top, ox[near] - left]),
            "areas": areas[near], "ids": ids[near],
            "owned": owner_site[near] == site,
            "threshold": ops_engine._THRESHOLD_READS,
            "footprint": ops_engine._FOOTPRINT, "store_reads": True,
            "gpu": bool(gpu),
        })
    return out


def _matched(a, b, radius=2.0):
    """The share of ``a``'s points with a ``b`` point within ``radius``."""
    if not len(a):
        return None
    if not len(b):
        return 0.0
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(b).query(a, distance_upper_bound=radius)
    return round(float(np.isfinite(distance).mean()), 4)


def _run(tasks, detector, library):
    """Decode every task with ``detector``; totals and owned positions."""
    ops_engine._init_decode_worker(frozenset(library))
    started = time.perf_counter()
    results = [ops_engine._decode_field({**t, "spot_detector": detector})
               for t in tasks]
    seconds = time.perf_counter() - started
    decoded = [r for r in results if "skipped" not in r]
    ids = np.concatenate([r["ids"] for r in decoded]) if decoded else []
    calls = [c for r in decoded for c in r["calls"]]
    quality = (np.concatenate([r["quality"] for r in decoded])
               if decoded else np.zeros(0, np.float32))
    assigned = assign_reads_to_objects(np.asarray(ids), calls, quality=quality)
    exact = sum(row["barcode"] in library for row in assigned.values())
    positions = {r["site"]: np.asarray(r["peaks"], float) for r in decoded}
    return {
        "fields_decoded": len(decoded),
        "skipped": {str(r["site"]): r["skipped"] for r in results
                    if "skipped" in r},
        "spots": int(sum(r["spots"] for r in decoded)),
        "reads_owned": len(calls),
        "spots_library_exact": int(sum(r["exact"] or 0 for r in decoded)),
        "objects_with_a_read": int(len(np.unique(ids))) if len(ids) else 0,
        "objects_assigned": len(assigned),
        "objects_assigned_library_exact": int(exact),
        "assigned_library_exact_rate": (round(exact / len(assigned), 4)
                                        if assigned else None),
        "seconds": round(seconds, 1),
        "field_seconds": [r.get("seconds") for r in decoded],
    }, positions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--plate", required=True)
    parser.add_argument("--well", required=True)
    parser.add_argument("--sites", type=int, nargs="+", required=True)
    parser.add_argument("--reference", type=int, default=1)
    parser.add_argument("--library", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--gpu", action="store_true",
                        help="let the native alignment and peaks use the card")
    args = parser.parse_args(argv)

    library = ops_engine._load_library(args.library) if args.library else []
    tasks = _tasks(args.tiles, args.db, args.plate, args.well.upper(),
                   args.sites, args.reference, gpu=args.gpu)
    report = {"item": 475, "gpu": bool(args.gpu), "plate": args.plate,
              "well": args.well.upper(),
              "sites": args.sites, "library_size": len(library),
              "match_radius_px": 2.0, "note": (
                  "Owned reads only: the positions compared are the reads "
                  "each detector's field gave to a nucleus it owns.")}
    native, native_at = _run(tasks, "native", library)
    report["native"] = native
    ready, reason = _spotnet_readiness()
    if not ready:
        report["spotnet"] = {"not_run": reason}
    else:
        from spacr._segmentation_backends import _detect_spots

        started = time.perf_counter()
        _detect_spots(np.zeros((64, 64), np.float32))
        report["spotnet_startup_seconds"] = round(
            time.perf_counter() - started, 1)
        spotnet, spotnet_at = _run(tasks, "spotnet", library)
        report["spotnet"] = spotnet
        both = [s for s in native_at if s in spotnet_at]
        a = np.concatenate([native_at[s] for s in both]) if both else []
        b = np.concatenate([spotnet_at[s] for s in both]) if both else []
        report["native_reads_matched_by_spotnet"] = _matched(a, b)
        report["spotnet_reads_matched_by_native"] = _matched(b, a)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
