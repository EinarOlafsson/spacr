"""Compare SpotNet and Spotiflow with spaCR's native spot detector on real
OPS fields.

Items 475 and 554. Each named field of a well is decoded twice by the shipped
``spacr.ops_engine._decode_field`` -- once with ``spot_detector='native'``,
once with each of ``--detectors`` (``spotnet``, ``spotiflow``) -- against the objects and placements an earlier
plate run stored in its measurements.db, so everything but the detector is
the same code on the same pixels. Reported per detector: spots found, reads
owned by a nucleus, objects given a barcode, the share of those barcodes in
the library, and seconds; and between them the share of each detector's
owned reads with native's within 2 px, and native's within 2 px of it.

SpotNet needs its environment (Model Zoo) and a DeepCell token in
DEEPCELL_ACCESS_TOKEN or ~/.spacr/deepcell_token; without them only the
native half runs and the report says why. The token is never printed.
Spotiflow needs only its environment (Model Zoo).

Add ``--strict`` with a new ``--out`` path for an acceptance receipt. All
requested detectors must be installed, have cached model artifacts, and
return spots from every requested field. Input planes, database, library,
worker code and cached models are hashed before and after the comparison.
Missing or changed inputs and incomplete results cannot produce a passing
receipt. This establishes reproducibility and completeness, not biological
accuracy. Without ``--strict``, missing detectors remain exploratory skips.

    python tools/compare_spotnet_ops.py \\
        --tiles /mnt/wd4tb/spacr_testdata/ops/raw/ops/sequencing \\
        --db /mnt/wd4tb/spacr_testdata/ops_plate_run/measurements.db \\
        --plate 20200202_6W-LaC024A --well A1 --sites 331 332 \\
        --library /mnt/wd4tb/spacr_testdata/ops_plate_run/library/pool10_prefixes.csv \\
        --detectors spotnet spotiflow \\
        --out features/data/554_spotiflow_vs_spotnet_vs_native.json
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacr import ops_engine  # noqa: E402
from spacr._segmentation_backends import (  # noqa: E402
    _detect_spots, _spotiflow_readiness, _spotiflow_spots, _spotnet_readiness)
from spacr.ops_sbs import assign_reads_to_objects  # noqa: E402
from spacr import _segmentation_backends as SB  # noqa: E402


def _echo_device_lines(label, line):
    """Print the worker's device lines (TensorFlow's ``Created device ...
    GPU:0``, or a CUDA library it could not open) into this run's log."""
    if any(key in line for key in ("GPU:0", "device:GPU", "libcud",
                                   "Could not load dynamic library")):
        print(f"[{label}] {line.rstrip()}", flush=True)


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



def _hash_inputs(paths):
    """Hash selected regular files in bounded chunks and reject changing inputs."""
    records = {}
    total = 0
    for path in sorted({str(Path(p).resolve(strict=True)) for p in paths}):
        before = os.stat(path)
        if not Path(path).is_file():
            raise ValueError(f"Not a regular input file: {path}")
        total += before.st_size
        if total > 16 * 1024**3:
            raise ValueError("Comparison provenance exceeds the 16 GiB input limit")
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        after = os.stat(path)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"Input changed while hashing: {path}")
        records[path] = {'bytes': after.st_size, 'sha256': digest.hexdigest()}
    return records


def _model_files(detectors):
    """Bind the installed worker markers and bounded model caches, never tokens."""
    paths, environments = [], {}
    for detector in detectors:
        state = SB._backend_state(detector)
        env = Path(state.env).resolve(strict=True)
        cache = (env / 'models' if detector == 'spotiflow' else
                 Path(SB._spotnet_home(str(env))) / '.deepcell' / 'models')
        artifacts = []
        for directory, folders, files in os.walk(cache, followlinks=False):
            folders[:] = sorted(f for f in folders if not f.startswith('.'))
            artifacts.extend(Path(directory) / f for f in sorted(files)
                             if not f.startswith('.'))
            if len(artifacts) > 2048:
                raise ValueError(f"Too many cached model files for {detector}")
        if not artifacts:
            raise ValueError(f"No cached model artifacts for {detector}: {cache}")
        paths.extend([env / 'spacr-backend.json', *artifacts])
        environments[detector] = {'environment': str(env), 'cache': str(cache),
                                  'model': 'general' if detector == 'spotiflow'
                                  else 'SpotDetection-8'}
    return paths, environments


def _strict_provenance(args, tasks):
    """Capture the exact source files and plane selectors before decoding."""
    if (not tasks or len(tasks) != len(set(args.sites))
            or {task['site'] for task in tasks} != set(args.sites)):
        raise ValueError("Every distinct requested field must have a decode task")
    inputs = [args.db]
    if args.library:
        inputs.append(args.library)
    planes = []
    for task in tasks:
        for cycle, sources in task['planes'].items():
            for channel, source in enumerate(sources):
                if not source:
                    raise ValueError(f"Missing plane: site {task['site']}, cycle {cycle}, channel {channel}")
                path, plane = source
                inputs.append(path)
                planes.append({'site': int(task['site']), 'cycle': int(cycle),
                               'channel': channel, 'path': str(Path(path).resolve()),
                               'plane': plane})
    model_paths, environments = _model_files(args.detectors)
    code = [__file__, ops_engine.__file__, SB.__file__]
    import spacr.ops_sbs as sbs
    code.append(sbs.__file__)
    files = _hash_inputs([*inputs, *model_paths, *code])
    destination = Path(args.out).expanduser().absolute() if args.out else None
    if destination is None:
        raise ValueError('--strict requires --out for its reproducible receipt')
    if os.path.lexists(destination):
        raise FileExistsError(f"Strict receipt already exists: {destination}")
    if str(destination.resolve()) in files:
        raise ValueError('Receipt path aliases an input')
    return {'files': files, 'planes': planes, 'backends': environments,
            'settings': {'threshold_reads': ops_engine._THRESHOLD_READS,
                         'footprint': ops_engine._FOOTPRINT,
                         'spotnet_threshold': 0.95, 'spotiflow_threshold': None,
                         'reference_cycle': args.reference, 'native_gpu': bool(args.gpu)}}


def _strict_failures(report, sites, detectors):
    """List incomplete comparisons without calling library agreement accuracy."""
    failures = []
    for detector in ['native', *detectors]:
        result = report.get(detector, {})
        if result.get('not_run'):
            failures.append(f"{detector}: {result['not_run']}")
        elif (result.get('skipped') or result.get('fields_decoded') != len(sites)
              or set(result.get('sites', [])) != set(sites)):
            failures.append(f"{detector}: not every requested field decoded")
        elif result.get('spots', 0) <= 0:
            failures.append(f"{detector}: no spots detected")
    return failures


def _write_strict_receipt(path, text):
    """Publish a new complete receipt atomically without replacing any file."""
    destination = Path(path).expanduser().absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                prefix='.spacr-comparison-', dir=destination.parent,
                delete=False) as handle:
            temporary = handle.name
            handle.write(text + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link is atomic and refuses existing destinations.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            os.unlink(temporary)

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
    parser.add_argument("--strict", action="store_true",
                        help="require complete detectors/fields and write a new source-hashed receipt")
    parser.add_argument("--gpu", action="store_true",
                        help="let the native alignment and peaks use the card")
    parser.add_argument("--detectors", nargs="+", default=["spotnet"],
                        choices=["spotnet", "spotiflow"])
    args = parser.parse_args(argv)
    SB._listen_to_workers(_echo_device_lines)

    checks = {"spotnet": (_spotnet_readiness, _detect_spots, 0.95),
              "spotiflow": (_spotiflow_readiness, _spotiflow_spots, None)}
    if args.strict:
        if not args.out or os.path.lexists(args.out):
            parser.error('--strict requires a new --out receipt path')
        if len(set(args.sites)) != len(args.sites):
            parser.error('--strict requires distinct sites')
        for detector in args.detectors:
            ready, reason = checks[detector][0]()
            if not ready:
                parser.error(f'{detector}: {reason}')
    input_head = (_hash_inputs([args.db, *([args.library] if args.library else [])])
                  if args.strict else None)
    if args.strict and Path(str(Path(args.db).resolve()) + '-wal').exists():
        raise ValueError('Checkpoint the input database before a strict comparison')
    library = ops_engine._load_library(args.library) if args.library else []
    tasks = _tasks(args.tiles, args.db, args.plate, args.well.upper(),
                   args.sites, args.reference, gpu=args.gpu)
    provenance = _strict_provenance(args, tasks) if args.strict else None
    if args.strict and any(provenance['files'][p] != value
                           for p, value in input_head.items()):
        raise ValueError('Database or library changed while preparing comparison')
    report = {"item": 475 if args.detectors == ["spotnet"] else 554, "gpu": bool(args.gpu), "plate": args.plate,
              "well": args.well.upper(),
              "sites": args.sites, "library_size": len(library),
              "match_radius_px": 2.0, "note": (
                  "Owned reads only: the positions compared are the reads "
                  "each detector's field gave to a nucleus it owns.")}
    native, native_at = _run(tasks, "native", library)
    report["native"] = native
    if args.strict:
        native["sites"] = sorted(native_at)
    for detector in args.detectors:
        readiness, detect, threshold = checks[detector]
        ready, reason = readiness()
        if not ready:
            report[detector] = {"not_run": reason}
            continue
        started = time.perf_counter()
        detect(np.zeros((64, 64), np.float32), threshold=threshold)
        report[f"{detector}_startup_seconds"] = round(
            time.perf_counter() - started, 1)
        worker = SB._WORKERS.get(detector)
        hello = getattr(worker, "hello", None) or {}
        report[f"{detector}_device"] = hello.get("device", "")
        print(f"{detector} worker device: {report[f'{detector}_device']!r}",
              flush=True)
        found, found_at = _run(tasks, detector, library)
        report[detector] = found
        if args.strict:
            found["sites"] = sorted(found_at)
        both = [s for s in native_at if s in found_at]
        a = np.concatenate([native_at[s] for s in both]) if both else []
        b = np.concatenate([found_at[s] for s in both]) if both else []
        report[f"native_reads_matched_by_{detector}"] = _matched(a, b)
        report[f"{detector}_reads_matched_by_native"] = _matched(b, a)
    failures = []
    if args.strict:
        failures = _strict_failures(report, args.sites, args.detectors)
        if _hash_inputs(provenance['files']) != provenance['files']:
            failures.append('Inputs or cached model artifacts changed during comparison')
        model_paths, _ = _model_files(args.detectors)
        if any(str(Path(path).resolve()) not in provenance['files'] for path in model_paths):
            failures.append('New cached model artifacts appeared during comparison')
        if Path(str(Path(args.db).resolve()) + '-wal').exists():
            failures.append('Database gained a write-ahead log during comparison')
        report['provenance'] = provenance
        report['acceptance'] = {'complete': not failures, 'failures': failures,
                                'scope': 'Detector comparison, not biological accuracy'}
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        if args.strict:
            _write_strict_receipt(args.out, text)
        else:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
