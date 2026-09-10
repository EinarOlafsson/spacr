"""Independent pixel-derived checks for the bounded synthetic Motility lesson.

This checker imports neither the application nor its tracking/metric helpers.
It deliberately requires complete, consecutive synthetic tracks without
teleports; it does not pretend to validate smoothing or biological identity.
"""
import math
from statistics import mean

import numpy as np


def pixel_reference(masks):
    masks = np.asarray(masks)
    if (masks.ndim != 3 or masks.shape[0] < 2 or
            not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0)):
        raise ValueError('Expected a nonnegative integer time-series mask')
    identities = set(np.unique(masks[0])) - {0}
    if not identities or any(set(np.unique(frame)) - {0} != identities for frame in masks):
        raise ValueError('The bounded reference requires the same complete tracks in every frame')
    points, tracks = {}, {}
    for identity in sorted(identities):
        coordinates = []
        for frame, labels in enumerate(masks):
            y, x = np.nonzero(labels == identity)
            xy = (float(x.mean()), float(y.mean()))
            coordinates.append(xy)
            points[(frame, int(identity))] = dict(x=xy[0], y=xy[1], area=int(x.size))
        steps = [math.dist(a, b) for a, b in zip(coordinates, coordinates[1:])]
        length = sum(steps)
        net = math.dist(coordinates[0], coordinates[-1])
        tracks[int(identity)] = dict(n_frames=len(coordinates), v_px_per_frame=mean(steps),
            path_length=length, net_displacement=net,
            straightness=net / length if length > 0 else float('nan'), max_step=max(steps))
    return dict(points=points, tracks=tracks, mask_pixels=int(masks.size))


def equal_number(got, expected, label):
    try:
        got = float(got)
        expected = float(expected)
    except (TypeError, ValueError) as error:
        raise ValueError('Non-numeric ' + label) from error
    if math.isnan(expected):
        if not math.isnan(got):
            raise ValueError('Undefined value differs: ' + label)
    elif not math.isfinite(got) or not math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError('Numeric value differs: ' + label)


def verify_snapshot(masks, snapshot):
    ref = pixel_reference(masks)
    knobs = snapshot['knobs']
    if (knobs['pathogen_plane'] != -1 or knobs['tracked_plane'] != 2 or
            knobs['n_channels'] != 2 or knobs['propagate'] is not False):
        raise ValueError('Wrong synthetic input planes or propagation state')
    if any(t['max_step'] > knobs['max_displacement'] for t in ref['tracks'].values()):
        raise ValueError('This independent reference does not cover teleport correction')
    found = set()
    for row in snapshot['points']:
        key = (row['frame'], row['cellID'])
        if key not in ref['points'] or key in found:
            raise ValueError('Duplicate or unknown pixel observation')
        found.add(key)
        if row['infected'] is not False:
            raise ValueError('A no-pathogen example was labelled infected')
        for name in ('x', 'y', 'area'):
            equal_number(row[name], ref['points'][key][name], 'pixel observation ' + name)
    if found != set(ref['points']):
        raise ValueError('The point table omits pixel observations')
    ppu, interval = knobs['pixels_per_um'], knobs['seconds_per_frame']
    if min(ppu, interval) < 0 or not all(math.isfinite(v) for v in (ppu, interval)):
        raise ValueError('Invalid calibration values')
    calibrated = ppu > 0 and interval > 0
    factor = 60 / (ppu * interval) if calibrated else 1
    unit = 'µm/min' if calibrated else 'px/frame'
    wanted = {key: t for key, t in ref['tracks'].items()
              if not knobs['straightness_filter'] or t['straightness'] < knobs['straightness']}
    found = set()
    for row in snapshot['tracks']:
        identity = row['cellID']
        if identity not in wanted or identity in found:
            raise ValueError('Duplicate, unknown or incorrectly filtered track')
        found.add(identity)
        track = wanted[identity]
        for name in ('n_frames', 'v_px_per_frame', 'straightness', 'path_length', 'net_displacement'):
            equal_number(row[name], track[name], 'track ' + name)
        equal_number(row['velocity'], track['v_px_per_frame'] * factor, 'converted velocity')
        if (row['velocity_unit'] != unit or row['infected'] is not False or
                row['too_short'] != (track['n_frames'] < knobs['min_length'])):
            raise ValueError('Track units, infection or length flag differs')
    if found != set(wanted):
        raise ValueError('Filtered track inventory differs')
    used = [t for t in wanted.values() if t['n_frames'] >= knobs['min_length']]
    velocity = mean(t['v_px_per_frame'] * factor for t in used) if used else float('nan')
    expected = dict(n_tracks=len(wanted), n_used=len(used), n_short=len(wanted) - len(used),
        min_length=knobs['min_length'], mean_velocity=velocity, mean_velocity_infected=float('nan'),
        mean_velocity_uninfected=velocity, mean_straightness=mean(t['straightness'] for t in used) if used else float('nan'),
        n_infected=0, n_uninfected=len(used),
        n_high_straightness=sum(t['straightness'] >= knobs['straightness'] for t in used),
        straightness_threshold=knobs['straightness'], glitches_fixed=0, tracks_dropped=0)
    for name, number in expected.items():
        equal_number(snapshot['summary'][name], number, 'summary ' + name)
    if snapshot['summary']['unit'] != unit or snapshot['summary']['calibrated'] != calibrated:
        raise ValueError('Summary calibration state differs')
    if snapshot['plot_visible'] is not True or snapshot['plot_is_null'] is not False:
        raise ValueError('The native preview plot is missing')
    return dict(passed=True, pixel_observations=len(ref['points']),
                mask_pixels=ref['mask_pixels'], retained_tracks=len(wanted), used_tracks=len(used),
                velocity_unit=unit, hypothetical_calibration=calibrated,
                checked_summary_fields=len(expected), biological_validation=False)


def verify_batch(arrays, rows, wells, *, pixels_per_um, seconds_per_frame):
    """Check saved centroids, raw channel means and the one-well speed summary.

    Deliberately not all morphology columns: the real assay smooths some of
    those before writing. The two named channel means are not smoothed.
    """
    arrays = np.asarray(arrays)
    if arrays.ndim != 4 or arrays.shape[-1] != 4:
        raise ValueError('Expected the four-plane synthetic batch')
    ref = pixel_reference(arrays[..., 2])
    found = set()
    for row in rows:
        key = (row['frame'], row['cellID'])
        if key not in ref['points'] or key in found:
            raise ValueError('Duplicate or unknown saved object')
        found.add(key)
        if (row['plateID'], row['wellID'], str(row['fieldID'])) != ('plate1', 'A01', '1'):
            raise ValueError('Wrong saved plate, well or field identity')
        if row['infected'] != 0:
            raise ValueError('A saved no-pathogen track was labelled infected')
        point = ref['points'][key]
        equal_number(row['cell_centroid-1'], point['x'], 'saved centroid x')
        equal_number(row['cell_centroid-0'], point['y'], 'saved centroid y')
        frame, identity = key
        selected = arrays[frame, ..., 2] == identity
        for channel in (0, 1):
            expected = float(arrays[frame, ..., channel][selected].mean())
            equal_number(row[f'cell_mean_intensity_ch{channel}'], expected, 'saved channel mean')
    if found != set(ref['points']):
        raise ValueError('The saved table omits objects')
    if len(wells) != 1 or (wells[0]['plateID'], wells[0]['wellID']) != ('plate1', 'A01'):
        raise ValueError('Expected one saved synthetic well')
    if (not all(math.isfinite(v) and v > 0 for v in (pixels_per_um, seconds_per_frame))):
        raise ValueError('Expected explicit positive hypothetical calibration')
    velocity = mean(t['v_px_per_frame'] for t in ref['tracks'].values()) * 60 / (pixels_per_um * seconds_per_frame)
    expected = dict(n_tracks=len(ref['tracks']), n_infected_tracks=0,
                    n_uninfected_tracks=len(ref['tracks']), mean_velocity_all=velocity,
                    mean_velocity_uninfected=velocity)
    for name, value in expected.items():
        equal_number(wells[0][name], value, 'saved well ' + name)
    if wells[0]['mean_velocity_infected'] is not None or wells[0]['velocity_unit'] != 'µm/min':
        raise ValueError('Saved null infection group or units differ')
    return dict(passed=True, objects=len(rows), centroid_coordinates=2 * len(rows),
                raw_channel_means=2 * len(rows), well_summary_fields=7,
                hypothetical_mean_velocity=velocity, biological_validation=False)
