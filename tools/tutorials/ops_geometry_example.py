#!/usr/bin/env python3
"""Four actual OPS tiles: registration and composition, not a full screen.

This bounded API example reads cycle-1 DAPI from an existing acquisition.
It never segments, decodes, changes a source, or calls the legacy GUI engine.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SITES = (0, 1, 10, 11)
SHAPE = (5, 1480, 1480)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def overlap_correlation(first, second, shift):
    """Measure paired pixels in the overlap, independently of the FFT peak."""
    dy, dx = map(lambda value: int(round(value)), shift)
    height, width = first.shape
    top, left = max(0, dy), max(0, dx)
    bottom, right = min(height, dy + height), min(width, dx + width)
    if bottom - top < 32 or right - left < 32:
        raise ValueError('An overlap needs at least 32 pixels on each axis')
    a = first[top:bottom, left:right].ravel().astype(float)
    b = second[top-dy:bottom-dy, left-dx:right-dx].ravel().astype(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.std(), b.std()) == 0:
        raise ValueError('Need finite, nonconstant actual overlap pixels')
    return float(np.corrcoef(a, b)[0, 1])


def require_geometry(well, planes):
    """Require a closed four-edge patch and independently visible alignment."""
    if set(well.placements) != set(SITES) or well.proposed != 4 or well.accepted != 4:
        raise ValueError('The four-site patch did not place all four real adjacencies')
    if max(abs(value - 2747) for value in well.canvas) > 40:
        raise ValueError('Patch canvas disagrees with two 1480-pixel tiles at 1267-pixel pitch')
    if len(well.residuals) != 4 or max(well.residuals) > 2:
        raise ValueError('The four-edge loop does not close within two pixels')
    checks = []
    for (a, b), edge in sorted(well.edges.items()):
        correct = overlap_correlation(planes[a], planes[b], edge.shift)
        # Both wrong offsets use actual pixels, not a manufactured image.
        controls = [overlap_correlation(planes[a], planes[b],
                    (edge.dy + dy, edge.dx + dx)) for dy, dx in ((17, 0), (0, 17))]
        if not correct > max(controls) + .1:
            raise ValueError(f'Pair {a}/{b}: measured shift lacks independent pixel support')
        checks.append({'sites': [a, b], 'shift': list(edge.shift),
                       'aligned_pearson': correct, 'offset_17px_controls': controls})
    return checks


def run(source, destination):
    """Read exactly four original files and create a NEW, bounded output folder."""
    import tifffile
    from spacr import __version__
    from spacr.ops_layout import round_well_layout
    from spacr.ops_stitch import stitch_well
    from spacr.ops_compose import Window, compose_window

    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new destination; existing runs are preserved')
    if destination.is_relative_to(source):
        raise ValueError('Never write inside the source acquisition')
    paths = {site: source / f'10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif'
             for site in SITES}
    sources, planes = [], {}
    for site, path in paths.items():
        before = digest(path)
        with tifffile.TiffFile(path) as handle:
            if handle.series[0].shape != SHAPE:
                raise ValueError('This example requires the documented five-plane acquisition')
            pixels = handle.asarray()
        if pixels.dtype != np.uint16 or not np.isfinite(pixels).all():
            raise ValueError('Expected unchanged uint16 acquisition pixels')
        planes[site] = pixels[0].copy()
        sources.append({'site': site, 'name': path.name, 'sha256': before,
                        'shape': list(pixels.shape), 'plane': 0, 'dtype': str(pixels.dtype)})
    print('Actual cycle-1 DAPI: sites 0, 1, 10, 11; only four of 333 tiles.', flush=True)
    layout = round_well_layout(333)  # Preserve original site coordinates, not a new four-tile well.
    start = time.monotonic()
    well = stitch_well(planes, layout, overlap=213, tolerance=4, skew=24,
                       gpu=False, sites=SITES)
    checks = require_geometry(well, planes)
    print(f'Patch: {well.placed}/4 placed, {well.accepted}/{well.proposed} edges; '
          f'canvas {well.canvas}; maximum loop residual {max(well.residuals):.3f}px', flush=True)
    ymin = min(y for y, _ in well.placements.values())
    xmin = min(x for _, x in well.placements.values())
    placements = {site: (y-ymin, x-xmin) for site, (y, x) in well.placements.items()}
    image, coverage = compose_window(Window(0, 0, *well.canvas), placements,
                                     planes.__getitem__, tile_shape=(1480, 1480))
    if not np.isfinite(image).all() or coverage.max() != 4 or (coverage > 0).sum() == 0:
        raise ValueError('The composed patch lacks expected real overlap coverage')
    for row in sources:
        if digest(paths[row['site']]) != row['sha256']:
            raise ValueError('An original acquisition file changed')
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / 'composed_dapi.npy', image, allow_pickle=False)
    np.save(destination / 'coverage.npy', coverage, allow_pickle=False)
    np.testing.assert_array_equal(np.load(destination / 'composed_dapi.npy'), image)
    np.testing.assert_array_equal(np.load(destination / 'coverage.npy'), coverage)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    limits = np.percentile(image[coverage > 0], (1, 99))
    axes[0].imshow(image, cmap='gray', vmin=limits[0], vmax=limits[1])
    axes[0].set_title('Four real cycle-1 DAPI tiles — geometry only')
    mesh = axes[1].imshow(coverage, vmin=0, vmax=4, cmap='viridis')
    axes[1].set_title('Coverage: zero means no acquisition pixel, not background')
    figure.colorbar(mesh, ax=axes[1], ticks=range(5), shrink=.75)
    for ax in axes:
        ax.set_axis_off()
    figure.savefig(destination / 'geometry_and_coverage.png', dpi=160)
    plt.close(figure)
    report = {'accepted': True, 'scope': 'Four-tile geometry/composition API example only',
              'app_version': __version__, 'source_folder': str(source), 'sources': sources,
              'helper_sha256': digest(__file__), 'source_unchanged': True,
              'sites': list(SITES), 'full_well_sites': 333, 'canvas': list(well.canvas),
              'placed': well.placed, 'accepted_edges': well.accepted,
              'residuals': well.residuals.tolist(), 'independent_overlap_checks': checks,
              'placements': {str(k): list(v) for k, v in placements.items()},
              'backends': sorted({edge.backend for edge in well.edges.values()}),
              'coverage_counts': {str(k): int((coverage == k).sum()) for k in range(5)},
              'elapsed_seconds': time.monotonic() - start,
              'full_pipeline_completed': False, 'gui_pipeline_completed': False,
              'segmentation_or_decoding_performed': False, 'published': False,
              'artifacts': {p.name: digest(p) for p in sorted(destination.iterdir())}}
    (destination / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Saved exact arrays, actual figure, source hashes and independently checked overlaps.', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
