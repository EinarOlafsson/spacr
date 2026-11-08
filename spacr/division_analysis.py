"""What tracked cells do before they divide (alpha).

A tracked label stack says, for every frame, which pixels belong to which
cell, and a lineage table says which cell came from which. Together they hold
the whole history of every division in the movie. This module turns them into
one question with a control: does a measurement change in the frames before a
division more than it changes in cells that are *not* about to divide?

The steps, each a function so a caller can stop or swap any of them:

1. :func:`measure_label_stack` -- per cell, per frame: area, major and minor
   axis length, aspect ratio, eccentricity, circularity, solidity and mean
   intensity (normalised to the frame median, so a drifting lamp or bleaching
   does not read as biology). :func:`add_motion_features` adds centroid speed,
   displacement from the track start and turning angle.
2. :func:`division_events` -- one row per division. Two lineage conventions
   are read: *fission* (CTC ``man_track``: the parent ends, its daughters start
   on the next frame) and *budding* (the mother continues; a bud is a daughter
   whose start lies inside the mother's lifetime). For both, the alignment
   frame ``t_div`` is the last frame before a daughter is seen.
3. :func:`control_slots` and :func:`match_controls` -- for every division, a
   control from the same movie: another track, observed over the same window
   length, overlapping the division in time, with a similar size at the start
   of the window, and **not** dividing in the window nor within ``horizon``
   frames after its pseudo-division time. :func:`shuffled_alignment` aligns the
   dividing tracks themselves on a random time away from their division; it
   should show nothing, and when it does the effect is a property of dividing
   tracks rather than of the approach to division.
4. :func:`align_units` and :func:`unit_changes` -- trajectories on
   ``rel_t = t - t_div`` and, per unit, the change from a baseline window at
   the start of the window to a late window that ends at ``rel_t = 0``.
5. :func:`effect_size` -- Hedges' g (dividing vs control change) with a
   bootstrap 95% confidence interval and its large-sample variance.
6. :func:`random_effects_meta` -- DerSimonian-Laird random-effects pooling
   with tau^2, Q and I^2.
7. :func:`cell_cycles` and :func:`size_control_fits` -- size at birth vs size
   at division and added size, the slopes that separate sizer (0 / -1), adder
   (1 / 0) and timer (2 / 1) control.

Nothing here reads images from disk, runs a model or needs a GPU: the caller
passes arrays and tables. Units are frames and pixels unless the caller
converts; the module never guesses a frame interval.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

#: Per-frame shape and intensity measurements written by measure_label_stack.
SHAPE_MEASURES = ('area', 'major_axis', 'minor_axis', 'aspect_ratio',
                  'eccentricity', 'circularity', 'solidity', 'mean_intensity')
#: Motion measurements written by add_motion_features.
MOTION_MEASURES = ('speed', 'displacement', 'turning')
#: Measurements compared before division (displacement grows with track age
#: by construction, so it is reported but not compared).
COMPARED_MEASURES = SHAPE_MEASURES + ('speed', 'turning')
#: Strictly positive measurements whose change is a log ratio; every other
#: measurement changes as a difference.
LOG_MEASURES = frozenset(('area', 'major_axis', 'minor_axis', 'aspect_ratio',
                          'mean_intensity'))
LINEAGE_COLUMNS = ('label', 't_start', 't_end', 'parent')


def measure_label_stack(masks: Iterable[np.ndarray],
                        images: Optional[Iterable[np.ndarray]] = None,
                        times: Optional[Sequence[int]] = None,
                        min_area: int = 1) -> pd.DataFrame:
    """Measure every labelled object in every frame of a tracked stack.

    :param masks: 2D integer label planes; the label value is the track id.
    :param images: optional matching intensity planes (2D, or channel-first
        3D of which channel 0 is measured). Mean intensity is divided by the
        frame median so slow illumination drift and bleaching cancel.
    :param times: frame time of each plane; defaults to ``0, 1, 2, ...``.
    :param min_area: objects smaller than this many pixels are dropped.
    :returns: one row per (``t``, ``track_id``) with :data:`SHAPE_MEASURES`
        and the centroid (``centroid_y``, ``centroid_x``) in pixels.
    :raises ValueError: when ``images`` or ``times`` does not match the
        number of masks, a mask is not 2D, or an image's shape differs from
        its mask's.
    """
    from skimage.measure import regionprops

    masks = list(masks)
    images = None if images is None else list(images)
    if images is not None and len(images) != len(masks):
        raise ValueError('images and masks must have the same length')
    times = list(range(len(masks))) if times is None else list(times)
    if len(times) != len(masks):
        raise ValueError('times and masks must have the same length')
    rows = []
    for i, mask in enumerate(masks):
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f'mask {i} is not 2D: shape {mask.shape}')
        image = None
        norm = 1.0
        if images is not None:
            image = np.asarray(images[i], dtype=np.float64)
            if image.ndim == 3:
                image = image[0]
            if image.shape != mask.shape:
                raise ValueError(f'frame {i}: image {image.shape} != mask {mask.shape}')
            med = float(np.median(image))
            norm = med if med > 0 else 1.0
        for r in regionprops(mask.astype(np.int32), intensity_image=image):
            area = float(r.area)
            if area < min_area:
                continue
            major = float(r.axis_major_length)
            minor = float(r.axis_minor_length)
            perim = float(r.perimeter)
            rows.append((
                int(times[i]), int(r.label), area, major, minor,
                major / minor if minor > 0 else np.nan,
                float(r.eccentricity),
                4.0 * math.pi * area / perim ** 2 if perim > 0 else np.nan,
                float(r.solidity),
                float(r.intensity_mean) / norm if image is not None else np.nan,
                float(r.centroid[0]), float(r.centroid[1])))
    cols = ('t', 'track_id') + SHAPE_MEASURES + ('centroid_y', 'centroid_x')
    return pd.DataFrame(rows, columns=cols)


def add_motion_features(measures: pd.DataFrame, min_step: float = 0.5) -> pd.DataFrame:
    """Add speed, displacement and turning angle to a per-frame table.

    ``speed`` is the centroid step to the previous observation divided by the
    frames between them (pixels per frame). ``displacement`` is the distance
    from the track's first centroid. ``turning`` is the unsigned angle in
    radians between consecutive steps, left empty when either step is shorter
    than ``min_step`` pixels (a direction that small is pixel noise) or when a
    frame is missing between them.

    :param measures: per-frame table with ``track_id``, ``t``,
        ``centroid_y`` and ``centroid_x``, as from :func:`measure_label_stack`.
    :param min_step: the shortest step, in pixels, that has a direction.
    :returns: a copy sorted by track and frame with ``speed``,
        ``displacement`` and ``turning`` added.
    """
    df = measures.sort_values(['track_id', 't']).reset_index(drop=True)
    g = df.groupby('track_id', sort=False)
    dt = g['t'].diff()
    dy = g['centroid_y'].diff()
    dx = g['centroid_x'].diff()
    step = np.hypot(dx, dy)
    df['speed'] = step / dt
    df['displacement'] = np.hypot(df['centroid_y'] - g['centroid_y'].transform('first'),
                                  df['centroid_x'] - g['centroid_x'].transform('first'))
    pdx = dx.groupby(df['track_id']).shift()
    pdy = dy.groupby(df['track_id']).shift()
    pdt = dt.groupby(df['track_id']).shift()
    pstep = np.hypot(pdx, pdy)
    cosang = (dx * pdx + dy * pdy) / (step * pstep)
    ok = (step >= min_step) & (pstep >= min_step) & (dt == 1) & (pdt == 1)
    df['turning'] = np.where(ok, np.arccos(np.clip(cosang, -1.0, 1.0)), np.nan)
    return df


def read_lineage_table(path) -> pd.DataFrame:
    """Read a lineage CSV (``label, t_start, t_end, parent``; CTC semantics).

    :param path: CSV file with at least the :data:`LINEAGE_COLUMNS`.
    :returns: those four columns as integers.
    :raises ValueError: when a lineage column is missing.
    """
    from .tabular import read_table
    df = read_table(path, canonicalise=False, report=None)
    missing = [c for c in LINEAGE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: lineage table lacks columns {missing}')
    return df[list(LINEAGE_COLUMNS)].astype(np.int64)


def division_events(lineage: pd.DataFrame, mode: str = 'auto',
                    max_gap: int = 1) -> pd.DataFrame:
    """One row per division: parent, alignment frame, kind and daughters.

    ``mode='fission'`` reads only daughters that start within ``max_gap``
    frames after the parent's last frame; ``'budding'`` reads only daughters
    that start while the mother is alive (each bud is one event); ``'auto'``
    reads both. ``t_div`` is the last frame before the daughter(s) appear.

    :param lineage: lineage table with ``label``, ``t_start``, ``t_end`` and
        ``parent`` (0 for no parent).
    :param mode: ``'auto'``, ``'fission'`` or ``'budding'``.
    :param max_gap: the most frames a fission daughter may start after its
        parent's last frame.
    :returns: one row per division with ``parent``, ``t_div``, ``kind``,
        ``n_daughters`` and ``daughters`` (a tuple of labels), sorted by
        ``t_div``.
    :raises ValueError: for an unknown ``mode``.
    """
    if mode not in ('auto', 'fission', 'budding'):
        raise ValueError(f'unknown mode {mode!r}')
    lin = lineage.set_index('label')
    rows = []
    kids_all = lineage[lineage['parent'] > 0]
    for parent, kids in kids_all.groupby('parent'):
        if parent not in lin.index:
            continue
        p0 = int(lin.at[parent, 't_start'])
        p1 = int(lin.at[parent, 't_end'])
        if mode in ('auto', 'fission'):
            ended = kids[(kids['t_start'] > p1) & (kids['t_start'] <= p1 + max_gap)]
            if len(ended):
                rows.append((int(parent), p1, 'fission', len(ended),
                             tuple(sorted(int(x) for x in ended['label']))))
        if mode in ('auto', 'budding'):
            buds = kids[(kids['t_start'] > p0) & (kids['t_start'] <= p1)]
            for _, b in buds.iterrows():
                rows.append((int(parent), int(b['t_start']) - 1, 'budding', 1,
                             (int(b['label']),)))
    ev = pd.DataFrame(rows, columns=['parent', 't_div', 'kind', 'n_daughters', 'daughters'])
    return ev.sort_values(['t_div', 'parent']).reset_index(drop=True)


def median_cycle_frames(lineage: pd.DataFrame, events: pd.DataFrame) -> float:
    """Median observed cycle length in frames (NaN when none is observed).

    Fission: tracks born at an observed division that themselves divide.
    Budding: frames between consecutive buds of the same mother.

    :param lineage: lineage table with ``label``, ``t_start`` and ``t_end``.
    :param events: division events from :func:`division_events`.
    :returns: the median cycle length in frames, or NaN.
    """
    durations = []
    fis = events[events['kind'] == 'fission']
    if len(fis):
        born = {d for ds in fis['daughters'] for d in ds}
        dividing = set(fis['parent'])
        lin = lineage.set_index('label')
        for lab in born & dividing:
            if lab in lin.index:
                durations.append(int(lin.at[lab, 't_end']) - int(lin.at[lab, 't_start']) + 1)
    bud = events[events['kind'] == 'budding']
    for _, g in bud.groupby('parent'):
        t = np.sort(g['t_div'].to_numpy())
        durations.extend(np.diff(t)[np.diff(t) > 0].tolist())
    return float(np.median(durations)) if durations else float('nan')


def _track_arrays(g: pd.DataFrame, column: str):
    """Lay one track's rows out on a dense frame axis.

    :param g: the rows of one track, with a frame column ``t``.
    :param column: the measurement to lay out.
    :returns: ``(t_min, t_max, present, values)``, where ``present`` marks the
        frames the track was seen in and ``values`` holds ``column`` on those
        frames and NaN on the others, both indexed from ``t_min``.
    """
    ts = g['t'].to_numpy(dtype=np.int64)
    t_min, t_max = int(ts.min()), int(ts.max())
    pres = np.zeros(t_max - t_min + 1, dtype=bool)
    pres[ts - t_min] = True
    val = np.full(t_max - t_min + 1, np.nan)
    val[ts - t_min] = g[column].to_numpy(dtype=np.float64)
    return t_min, t_max, pres, val


def control_slots(measures: pd.DataFrame, events: pd.DataFrame, window: int,
                  horizon: int, min_coverage: float = 0.8,
                  size_column: str = 'area') -> pd.DataFrame:
    """Every (track, pseudo-division time) that can serve as a control.

    A slot ``(track_id, tp)`` qualifies when the track is seen at ``tp - window``
    (the baseline size) and at ``tp + horizon`` (its fate over the horizon is
    known), is seen on at least ``min_coverage`` of the frames
    ``tp - window .. tp``, and has no division event in
    ``tp - window .. tp + horizon``. ``base_size`` is the size at
    ``tp - window``.

    :param measures: per-frame table with ``track_id``, ``t`` and
        ``size_column``.
    :param events: division events with ``parent`` and ``t_div``.
    :param window: frames before the pseudo-division time that must be
        observed.
    :param horizon: frames after it over which the track must not divide.
    :param min_coverage: the smallest share of the window the track must be
        seen on.
    :param size_column: the measurement used as the baseline size.
    :returns: one row per slot with ``track_id``, ``tp`` and ``base_size``,
        sorted by ``tp``.
    """
    ev_times = {p: np.asarray(g['t_div'].to_numpy()) for p, g in events.groupby('parent')}
    out = []
    for tid, g in measures.groupby('track_id', sort=False):
        t_min, t_max, pres, val = _track_arrays(g, size_column)
        lo, hi = t_min + window, t_max - horizon
        if hi < lo:
            continue
        tp = np.arange(lo, hi + 1)
        i = tp - t_min
        cs = np.concatenate([[0], np.cumsum(pres)])
        cov = (cs[i + 1] - cs[i - window]) / (window + 1)
        ok = pres[i - window] & pres[i + horizon] & (cov >= min_coverage)
        et = ev_times.get(tid)
        if et is not None and len(et):
            et = np.sort(et)
            a = np.searchsorted(et, tp - window, side='left')
            b = np.searchsorted(et, tp + horizon, side='right')
            ok &= (b - a) == 0
        if ok.any():
            out.append(pd.DataFrame({'track_id': tid, 'tp': tp[ok],
                                     'base_size': val[i[ok] - window]}))
    if not out:
        return pd.DataFrame(columns=['track_id', 'tp', 'base_size'])
    return pd.concat(out, ignore_index=True).sort_values('tp').reset_index(drop=True)


def match_controls(events: pd.DataFrame, slots: pd.DataFrame, window: int,
                   size_tol: float = 0.25, rng=None) -> pd.DataFrame:
    """Pick one matched control slot per division event.

    ``events`` needs ``event_id``, ``parent``, ``t_div``, ``base_size`` and
    ``daughters``. A slot is eligible when its pseudo time lies within
    ``window`` frames of the division, its track is neither the parent nor a
    daughter, and its base size is within a factor ``1 +/- size_tol`` of the
    parent's (on a log scale). One eligible slot is drawn at random; events
    with none are returned with ``matched=False``.

    :param events: the division events to match.
    :param slots: control slots from :func:`control_slots`.
    :param window: the most frames a slot's pseudo time may lie from the
        division.
    :param size_tol: the relative base-size difference a control may have.
    :param rng: seed or generator for the random draw.
    :returns: one row per event with ``event_id``, ``track_id``, ``t0``,
        ``size_ratio`` and ``matched``.
    """
    rng = np.random.default_rng(rng)
    tp = slots['tp'].to_numpy()
    tracks = slots['track_id'].to_numpy()
    logsize = np.log(np.maximum(slots['base_size'].to_numpy(dtype=float), 1e-12))
    tol = math.log1p(size_tol)
    rows = []
    for e in events.itertuples(index=False):
        a = np.searchsorted(tp, e.t_div - window, side='left')
        b = np.searchsorted(tp, e.t_div + window, side='right')
        if b <= a or not np.isfinite(e.base_size) or e.base_size <= 0:
            rows.append((e.event_id, -1, -1, np.nan, False))
            continue
        excl = {int(e.parent), *map(int, e.daughters)}
        cand = np.arange(a, b)
        cand = cand[~np.isin(tracks[cand], list(excl))]
        cand = cand[np.abs(logsize[cand] - math.log(e.base_size)) <= tol]
        if not len(cand):
            rows.append((e.event_id, -1, -1, np.nan, False))
            continue
        j = int(rng.choice(cand))
        rows.append((e.event_id, int(tracks[j]), int(tp[j]),
                     float(slots['base_size'].iat[j] / e.base_size), True))
    return pd.DataFrame(rows, columns=['event_id', 'track_id', 't0', 'size_ratio', 'matched'])


def shuffled_alignment(events: pd.DataFrame, slots: pd.DataFrame, rng=None) -> pd.DataFrame:
    """Align each dividing parent on a random eligible time of its own track.

    The time comes from the same slot table as the controls, so it is at least
    ``horizon`` frames before any division of that track. An effect that
    survives this shuffle belongs to dividing tracks, not to the approach to
    division.

    :param events: division events with ``event_id`` and ``parent``.
    :param slots: control slots from :func:`control_slots`.
    :param rng: seed or generator for the random draw.
    :returns: one row per event with ``event_id``, ``track_id``, ``t0`` and
        ``matched``.
    """
    rng = np.random.default_rng(rng)
    by_track = {t: g['tp'].to_numpy() for t, g in slots.groupby('track_id')}
    rows = []
    for e in events.itertuples(index=False):
        tps = by_track.get(int(e.parent))
        if tps is None or not len(tps):
            rows.append((e.event_id, -1, -1, False))
        else:
            rows.append((e.event_id, int(e.parent), int(rng.choice(tps)), True))
    return pd.DataFrame(rows, columns=['event_id', 'track_id', 't0', 'matched'])


def align_units(measures: pd.DataFrame, units: pd.DataFrame, window: int,
                after: int = 0, columns: Sequence[str] = COMPARED_MEASURES) -> pd.DataFrame:
    """Measurements of each unit's track on ``rel_t = t - t0``.

    ``units`` needs ``unit_id``, ``track_id`` and ``t0``; with an optional
    ``daughters`` column (tuples of track ids) the daughters are added for
    ``rel_t = 1 .. after`` with ``role='daughter'``. The unit's own track is
    read for ``rel_t = -window .. after`` with ``role='self'`` (a fission
    parent simply has no rows after 0; a budding mother continues).

    :param measures: per-frame table with ``track_id``, ``t`` and the
        measurement columns.
    :param units: the dividing or control units to align.
    :param window: frames read before ``t0``.
    :param after: frames read after ``t0``.
    :param columns: the measurements to carry; absent ones are skipped.
    :returns: one row per unit, track and frame with ``unit_id``,
        ``track_id``, ``t0``, ``rel_t``, ``t``, ``role`` and the measurements.
    """
    cols = [c for c in columns if c in measures.columns]
    m = measures[['track_id', 't'] + cols]
    rel = np.arange(-window, after + 1)
    base = units[['unit_id', 'track_id', 't0']]
    own = base.loc[base.index.repeat(len(rel))].copy()
    own['rel_t'] = np.tile(rel, len(base))
    own['t'] = own['t0'] + own['rel_t']
    own['role'] = 'self'
    parts = [own]
    if after > 0 and 'daughters' in units.columns:
        d = units[['unit_id', 't0', 'daughters']].explode('daughters').dropna()
        if len(d):
            d = d.rename(columns={'daughters': 'track_id'})
            d['track_id'] = d['track_id'].astype(np.int64)
            rel_d = np.arange(1, after + 1)
            d = d.loc[d.index.repeat(len(rel_d))].copy()
            d['rel_t'] = np.tile(rel_d, len(d) // len(rel_d))
            d['t'] = d['t0'] + d['rel_t']
            d['role'] = 'daughter'
            parts.append(d)
    grid = pd.concat(parts, ignore_index=True)
    out = grid.merge(m, on=['track_id', 't'], how='inner')
    return out.sort_values(['unit_id', 'role', 'track_id', 'rel_t']).reset_index(drop=True)


def unit_changes(aligned: pd.DataFrame, window: int, k: int,
                 columns: Sequence[str] = COMPARED_MEASURES) -> pd.DataFrame:
    """Per unit: change from the baseline window to the late window.

    Baseline = mean over ``rel_t = -window .. -window + k - 1``; late = mean
    over ``rel_t = -k + 1 .. 0`` (the unit's own track only). The change is
    ``log(late / baseline)`` for :data:`LOG_MEASURES` and ``late - baseline``
    otherwise. Needs ``2 * k <= window + 1`` so the windows do not overlap.

    :param aligned: aligned measurements from :func:`align_units`.
    :param window: the alignment window used for ``aligned``.
    :param k: frames in each of the baseline and late windows.
    :param columns: the measurements to compare; absent ones are skipped.
    :returns: one row per unit with ``unit_id`` and one change per
        measurement.
    :raises ValueError: when the two windows would overlap.
    """
    if 2 * k > window + 1:
        raise ValueError('baseline and late windows overlap: need 2*k <= window+1')
    own = aligned[aligned['role'] == 'self']
    cols = [c for c in columns if c in own.columns]
    base = own[own['rel_t'] <= -window + k - 1].groupby('unit_id')[cols].mean()
    late = own[(own['rel_t'] >= -k + 1) & (own['rel_t'] <= 0)].groupby('unit_id')[cols].mean()
    base, late = base.align(late, join='inner')
    out = pd.DataFrame(index=base.index)
    for c in cols:
        if c in LOG_MEASURES:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.log(late[c] / base[c])
            out[c] = r.replace([np.inf, -np.inf], np.nan)
        else:
            out[c] = late[c] - base[c]
    return out.reset_index()


def _hedges_g(x, y):
    """Hedges' g of ``x`` against ``y`` with the small-sample correction.

    :param x: the first group's values.
    :param y: the second group's values.
    :returns: the corrected standardised mean difference, or NaN when either
        group has fewer than two values or the pooled deviation is zero.
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    s = math.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return np.nan
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return j * (np.mean(x) - np.mean(y)) / s


def effect_size(treated, control, n_boot: int = 2000, rng=None) -> dict:
    """Hedges' g of ``treated`` vs ``control`` with a bootstrap 95% CI.

    Non-finite values are dropped first.

    :param treated: values of the treated group (dividing units).
    :param control: values of the control group.
    :param n_boot: bootstrap resamples; 0 skips the interval.
    :param rng: seed or generator for the resampling.
    :returns: ``g``, ``var_g`` (large-sample variance used for
        meta-analysis), ``ci_low`` and ``ci_high`` (percentile bootstrap, each
        group resampled independently), ``mean_diff``, ``n_treated`` and
        ``n_control``.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(treated, float)
    y = np.asarray(control, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    g = _hedges_g(x, y)
    res = {'g': g, 'var_g': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
           'mean_diff': (x.mean() - y.mean()) if n1 and n2 else np.nan,
           'n_treated': n1, 'n_control': n2}
    if not np.isfinite(g):
        return res
    res['var_g'] = (n1 + n2) / (n1 * n2) + g ** 2 / (2.0 * (n1 + n2))
    if n_boot:
        bx = rng.integers(0, n1, size=(n_boot, n1))
        by = rng.integers(0, n2, size=(n_boot, n2))
        xs, ys = x[bx], y[by]
        sp = np.sqrt(((n1 - 1) * xs.var(axis=1, ddof=1) + (n2 - 1) * ys.var(axis=1, ddof=1))
                     / (n1 + n2 - 2))
        j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            gs = j * (xs.mean(axis=1) - ys.mean(axis=1)) / sp
        gs = gs[np.isfinite(gs)]
        if len(gs):
            res['ci_low'], res['ci_high'] = np.percentile(gs, [2.5, 97.5])
    return res


def random_effects_meta(effects, variances) -> dict:
    """DerSimonian-Laird random-effects pooling.

    Studies with a non-finite effect or non-positive variance are skipped.

    :param effects: one effect size per study.
    :param variances: the matching variances.
    :returns: the pooled effect ``mu`` with ``se``, a normal 95% CI
        (``ci_low``, ``ci_high``), ``z``, two-sided ``p``, between-study
        variance ``tau2``, Cochran's ``Q``, ``I2`` (percent) and ``k``
        studies.
    """
    from scipy import stats

    y = np.asarray(effects, float)
    v = np.asarray(variances, float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    nan = float('nan')
    if k == 0:
        return {'mu': nan, 'se': nan, 'ci_low': nan, 'ci_high': nan, 'z': nan,
                'p': nan, 'tau2': nan, 'Q': nan, 'I2': nan, 'k': 0}
    w = 1.0 / v
    mu_f = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu_f) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se = float(math.sqrt(1.0 / np.sum(ws)))
    z = mu / se
    i2 = max(0.0, (q - (k - 1)) / q) * 100.0 if k > 1 and q > 0 else 0.0
    return {'mu': mu, 'se': se, 'ci_low': mu - 1.959964 * se, 'ci_high': mu + 1.959964 * se,
            'z': z, 'p': float(2 * stats.norm.sf(abs(z))), 'tau2': tau2, 'Q': q,
            'I2': i2, 'k': k}


def cell_cycles(measures: pd.DataFrame, lineage: pd.DataFrame, events: pd.DataFrame,
                size_columns: Sequence[str] = ('area', 'major_axis')) -> pd.DataFrame:
    """Complete fission cycles: born at an observed two-daughter division and seen to divide into two.

    Birth size is the size on the first frame, division size on the last.

    :param measures: per-frame table with ``track_id``, ``t`` and the size
        columns.
    :param lineage: lineage table with ``label``, ``t_start`` and ``t_end``.
    :param events: division events from :func:`division_events`.
    :param size_columns: the measurements read at birth and division.
    :returns: one row per cycle with ``track_id``, ``t_birth``,
        ``t_division``, ``duration`` and, per size column, ``<col>_birth``,
        ``<col>_division`` and ``<col>_added``.
    """
    fis = events[(events['kind'] == 'fission') & (events['n_daughters'] == 2)]
    born = {d for ds in fis['daughters'] for d in ds}
    dividing = set(fis['parent'])
    lin = lineage.set_index('label')
    idx = measures.set_index(['track_id', 't'])
    rows = []
    for lab in sorted(born & dividing):
        if lab not in lin.index:
            continue
        t0, t1 = int(lin.at[lab, 't_start']), int(lin.at[lab, 't_end'])
        if (lab, t0) not in idx.index or (lab, t1) not in idx.index:
            continue
        r = {'track_id': lab, 't_birth': t0, 't_division': t1, 'duration': t1 - t0 + 1}
        for c in size_columns:
            b, d = float(idx.at[(lab, t0), c]), float(idx.at[(lab, t1), c])
            r[f'{c}_birth'], r[f'{c}_division'], r[f'{c}_added'] = b, d, d - b
        rows.append(r)
    return pd.DataFrame(rows)


def size_control_fits(birth, division, n_boot: int = 2000, rng=None) -> dict:
    """Slopes of division size and added size on birth size, with bootstrap CIs.

    Reference slopes (division ~ birth / added ~ birth): sizer 0 / -1,
    adder 1 / 0, timer (exponential growth, fixed duration) 2 / 1.

    :param birth: size at birth, one value per cycle.
    :param division: size at division, matching ``birth``.
    :param n_boot: bootstrap resamples; 0 skips the intervals.
    :param rng: seed or generator for the resampling.
    :returns: ``n``, ``slope_division`` and ``slope_added`` with their
        ``_ci_low`` and ``_ci_high`` bounds, and ``pearson_r_added``; NaN
        slopes with fewer than three cycles or constant birth size.
    """
    rng = np.random.default_rng(rng)
    b = np.asarray(birth, float)
    d = np.asarray(division, float)
    ok = np.isfinite(b) & np.isfinite(d)
    b, d = b[ok], d[ok]
    n = len(b)
    out = {'n': n}
    if n < 3 or np.var(b) == 0:
        for key in ('slope_division', 'slope_added'):
            out[key] = out[key + '_ci_low'] = out[key + '_ci_high'] = np.nan
        out['pearson_r_added'] = np.nan
        return out

    def slope(x, y):
        """Least-squares slope of ``y`` on ``x`` along the last axis."""
        x = x - x.mean(axis=-1, keepdims=True)
        y = y - y.mean(axis=-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x * y).sum(axis=-1) / (x * x).sum(axis=-1)

    s_div = float(slope(b, d))
    out['slope_division'] = s_div
    out['slope_added'] = s_div - 1.0
    out['pearson_r_added'] = float(np.corrcoef(b, d - b)[0, 1])
    if n_boot:
        i = rng.integers(0, n, size=(n_boot, n))
        s = slope(b[i], d[i])
        s = s[np.isfinite(s)]
        lo, hi = np.percentile(s, [2.5, 97.5])
        out['slope_division_ci_low'], out['slope_division_ci_high'] = lo, hi
        out['slope_added_ci_low'], out['slope_added_ci_high'] = lo - 1.0, hi - 1.0
    return out


_RESULT_TABLES = ('events', 'effects', 'trajectories', 'cycles', 'size_control')
_SERIES = ('dividing', 'control', 'shuffled', 'shuffled_control')
_CONTRASTS = (('dividing_vs_control', 'dividing', 'control'),
              ('shuffled_vs_its_control', 'shuffled', 'shuffled_control'))


def lineage_from_tracks(tracks: pd.DataFrame, max_distance: float = 30.0) -> pd.DataFrame:
    """Lineage table of one field's spaCR tracks table.

    Each track gives one row: its first and last frame and the track it was
    born from. A ``parent_track_id`` column written by the tracker is used as
    it is; without one, divisions are inferred from nearby track starts as
    the lineage trees infer them. A daughter the tracker kept under its
    mother's id is not a new track, so that mother continues and its division
    reads as a bud.

    :param tracks: a spaCR tracks table with ``frame``, ``track_id``, ``x``
        and ``y`` for one field, as written to
        ``<tracker>_tracks_<object>_<field>.csv``.
    :param max_distance: largest mother-to-daughter distance in pixels for an
        inferred division.
    :returns: the :data:`LINEAGE_COLUMNS` as integers, ``parent`` 0 for a
        track with no known mother.
    """
    from ._lineage_trees import _lineage_segments

    segments = _lineage_segments(tracks, max_distance=float(max_distance))
    if segments.empty:
        return pd.DataFrame({c: pd.Series(dtype=np.int64) for c in LINEAGE_COLUMNS})
    track_of = dict(zip(segments['segment_id'], segments['track_id']))
    first = segments.sort_values('start_frame').groupby('track_id', sort=True).first()
    spans = segments.groupby('track_id').agg(t_start=('start_frame', 'min'),
                                             t_end=('end_frame', 'max'))
    parent = first['parent_segment_id'].map(lambda s: track_of.get(int(s), 0) if s else 0)
    parent = parent.where(parent != parent.index.to_series(), 0)
    out = spans.assign(parent=parent).reset_index().rename(columns={'track_id': 'label'})
    return out[list(LINEAGE_COLUMNS)].astype(np.int64)


def _event_quality(events, measures, window):
    """Why each division can or cannot be aligned.

    A division is kept when its parent is seen at ``t_div``, its daughters
    on the next frame, a fission has two daughters, and the parent is seen
    on at least 80% of the ``window + 1`` frames up to ``t_div``, the first
    of them included (its size there is the control matching size).

    :returns: ``events`` with ``base_size``, ``coverage`` and ``status``
        (``ok`` or the reason it was dropped).
    """
    present = set(zip(measures['track_id'].to_numpy(), measures['t'].to_numpy()))
    frames = {tid: set(g) for tid, g in measures.groupby('track_id')['t']}
    area = measures.set_index(['track_id', 't'])['area']
    base, coverage, status = [], [], []
    for e in events.itertuples(index=False):
        seen = frames.get(e.parent, set())
        coverage.append(sum((e.t_div - j) in seen for j in range(window + 1)) / (window + 1))
        base.append(float(area.get((e.parent, e.t_div - window), np.nan)))
        if e.kind == 'fission' and e.n_daughters != 2:
            status.append('not_two_daughters')
        elif (e.parent, e.t_div) not in present:
            status.append('parent_unseen_at_division')
        elif not all((d, e.t_div + 1) in present for d in e.daughters):
            status.append('daughter_unseen_next_frame')
        elif not np.isfinite(base[-1]) or coverage[-1] < 0.8:
            status.append('short_prehistory')
        else:
            status.append('ok')
    return events.assign(base_size=base, coverage=coverage, status=status)


def _baseline_normalised(aligned, window, k, columns):
    """Each unit's values relative to the mean of its own first ``k`` frames."""
    own = aligned[aligned['role'] == 'self']
    base = own[own['rel_t'] <= -window + k - 1].groupby('unit_id')[list(columns)].mean()
    joined = aligned.join(base, on='unit_id', rsuffix='__base')
    out = joined[['unit_id', 'role', 'track_id', 'rel_t']].copy()
    for c in columns:
        if c in LOG_MEASURES:
            with np.errstate(divide='ignore', invalid='ignore'):
                value = np.log(joined[c] / joined[c + '__base'])
            out[c] = value.replace([np.inf, -np.inf], np.nan)
        else:
            out[c] = joined[c] - joined[c + '__base']
    return out


def _trajectory_summary(normalised, series, columns):
    """Mean, standard error and normal 95% band per role, frame and measure."""
    rows = []
    if not len(normalised):
        return rows
    grouped = normalised.groupby(['role', 'rel_t'])[list(columns)]
    mean, sd, n = grouped.mean(), grouped.std(), grouped.count()
    for c in columns:
        table = pd.DataFrame({'mean': mean[c], 'sd': sd[c], 'n': n[c]}).reset_index()
        table['sem'] = table['sd'] / np.sqrt(table['n'])
        table['ci_low'] = table['mean'] - 1.96 * table['sem']
        table['ci_high'] = table['mean'] + 1.96 * table['sem']
        table['measure'] = c
        table['series'] = series
        table['scale'] = ('log_ratio_to_baseline' if c in LOG_MEASURES
                          else 'difference_from_baseline')
        rows.append(table)
    return rows


def analyse_divisions(measures: pd.DataFrame, lineage: pd.DataFrame, mode: str = 'auto',
                      window: Optional[int] = None, late_frames: Optional[int] = None,
                      size_tol: float = 0.25, n_boot: int = 2000, rng=0,
                      columns: Sequence[str] = COMPARED_MEASURES) -> dict:
    """Align one movie's divisions and compare them with matched controls.

    Runs the steps of this module in order. Divisions are read with
    :func:`division_events`; ``mode='auto'`` keeps whichever kind, fission
    or budding, the movie has more of. Each division is checked (parent seen
    at ``t_div`` and on 80% of the window, daughters seen on the next frame,
    two daughters for a fission). The kept divisions get size-matched
    controls (:func:`match_controls`) and a shuffled alignment
    (:func:`shuffled_alignment`) with its own size-matched controls, so the
    null runs the same matching at a random time. The late-vs-baseline
    change of every unit (:func:`unit_changes`) is compared as Hedges' g
    (:func:`effect_size`), dividing against control and shuffled against
    its control. Complete fission cycles give the size-control fits.

    :param measures: per-frame table from :func:`measure_label_stack`,
        usually with :func:`add_motion_features`; it needs ``area``.
    :param lineage: lineage table with the :data:`LINEAGE_COLUMNS`.
    :param mode: ``'auto'``, ``'fission'`` or ``'budding'``.
    :param window: frames before division in the alignment window; None
        takes 0.3 times the median cycle (:func:`median_cycle_frames`),
        clipped to 4..30 frames, or 6 frames when no cycle is observed.
    :param late_frames: frames in each of the baseline and late windows, and
        the horizon a control must stay undivided after its pseudo-division
        time; None takes a quarter of ``window`` (at least 2). A value too
        large for ``window`` is lowered so the two windows do not overlap.
    :param size_tol: the relative size difference a control may have at the
        start of the window.
    :param n_boot: bootstrap resamples for every confidence interval.
    :param rng: seed or generator for the controls, the shuffle and the
        bootstraps.
    :param columns: the measurements compared; absent ones are skipped.
    :returns: dict with the tables ``events`` (every division with its
        ``status``), ``effects`` (one row per contrast and measure),
        ``trajectories`` (baseline-normalised mean and 95% band per series,
        role, ``rel_t`` and measure), ``cycles`` and ``size_control``, and
        ``summary`` (the window, counts and drop reasons).
    :raises ValueError: for an unknown ``mode``, a ``window`` below 2, or
        ``measures`` without ``track_id``, ``t`` or ``area``.
    """
    if mode not in ('auto', 'fission', 'budding'):
        raise ValueError(f'unknown mode {mode!r}')
    missing = [c for c in ('track_id', 't', 'area') if c not in measures.columns]
    if missing:
        raise ValueError(f'measures lack columns {missing}')
    rng = np.random.default_rng(rng)
    columns = [c for c in columns if c in measures.columns]
    every = division_events(lineage, mode='auto')
    if mode == 'auto':
        counts = every['kind'].value_counts()
        mode = 'budding' if counts.get('budding', 0) > counts.get('fission', 0) else 'fission'
    events = division_events(lineage, mode=mode)
    cycle = median_cycle_frames(lineage, events)
    if window is None:
        window = int(np.clip(round(0.3 * cycle), 4, 30)) if np.isfinite(cycle) else 6
    window = int(window)
    if window < 2:
        raise ValueError('window must be at least 2 frames')
    k = int(late_frames) if late_frames else max(2, int(round(window / 4)))
    k = max(1, min(k, (window + 1) // 2))
    events = _event_quality(events, measures, window)
    events['event_id'] = np.arange(len(events))
    kept = events[events['status'] == 'ok'].reset_index(drop=True)
    slots = control_slots(measures, every, window, k)
    controls = match_controls(kept, slots, window, size_tol=size_tol, rng=rng)
    shuffled = shuffled_alignment(kept, slots, rng=rng)
    moved = shuffled[shuffled['matched']][['event_id', 't0']].merge(
        kept[['event_id', 'parent', 'daughters']], on='event_id').rename(columns={'t0': 't_div'})
    area = measures.set_index(['track_id', 't'])['area']
    moved['base_size'] = [float(area.get((p, t - window), np.nan))
                          for p, t in zip(moved['parent'], moved['t_div'])]
    shuffled_controls = (match_controls(moved, slots, window, size_tol=size_tol, rng=rng)
                         if len(moved) else controls.iloc[0:0])
    units = {
        'dividing': kept.assign(unit_id=kept['event_id'], track_id=kept['parent'],
                                t0=kept['t_div']),
        'control': controls[controls['matched'].astype(bool)].assign(
            unit_id=lambda d: d['event_id']),
        'shuffled': shuffled[shuffled['matched'].astype(bool)].assign(
            unit_id=lambda d: d['event_id']),
        'shuffled_control': shuffled_controls[shuffled_controls['matched'].astype(bool)].assign(
            unit_id=lambda d: d['event_id']),
    }
    changes, trajectories = {}, []
    for series in _SERIES:
        unit = units[series]
        if not len(unit):
            changes[series] = pd.DataFrame(columns=['unit_id'] + columns)
            continue
        aligned = align_units(measures, unit, window,
                              after=window if series == 'dividing' else 0, columns=columns)
        changes[series] = unit_changes(aligned, window, k, columns=columns)
        trajectories += _trajectory_summary(_baseline_normalised(aligned, window, k, columns),
                                            series, columns)
    effects = []
    for contrast, treated, control in _CONTRASTS:
        a, b = changes[treated], changes[control]
        for c in columns:
            result = effect_size(a[c], b[c], n_boot=n_boot, rng=rng)
            effects.append({'contrast': contrast, 'measure': c,
                            'change_scale': 'log_ratio' if c in LOG_MEASURES else 'difference',
                            'mean_change_treated': _finite_mean(a[c]),
                            'mean_change_control': _finite_mean(b[c]),
                            **result})
    sizes = [c for c in ('area', 'major_axis') if c in measures.columns]
    cycles = (cell_cycles(measures, lineage, events, size_columns=sizes)
              if mode == 'fission' else pd.DataFrame())
    size_control = [{'size_measure': c,
                     **size_control_fits(cycles[f'{c}_birth'], cycles[f'{c}_division'],
                                         n_boot=n_boot, rng=rng)}
                    for c in sizes if len(cycles)]
    reasons = events['status'].value_counts().to_dict()
    summary = {'mode': mode, 'window': window, 'late_frames': k, 'horizon': k,
               'median_cycle_frames': cycle, 'size_tol': float(size_tol),
               'n_tracks': int(measures['track_id'].nunique()),
               'n_events': int(len(events)), 'n_events_kept': int(len(kept)),
               'n_controls_matched': int(len(units['control'])),
               'n_shuffled_matched': int(len(units['shuffled'])),
               'n_cycles': int(len(cycles)),
               'drop_reasons': {str(key): int(value) for key, value in reasons.items()
                                if key != 'ok'}}
    if trajectories:
        trajectories = pd.concat(trajectories, ignore_index=True)
    else:
        trajectories = pd.DataFrame(columns=['role', 'rel_t', 'mean', 'sd', 'n', 'sem',
                                             'ci_low', 'ci_high', 'measure', 'series', 'scale'])
    return {'events': events, 'effects': pd.DataFrame(effects),
            'trajectories': trajectories, 'cycles': cycles,
            'size_control': pd.DataFrame(size_control), 'summary': summary}


def _finite_mean(values):
    """Mean of the finite values, NaN when there are none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float('nan')


_SERIES_STYLE = {
    'dividing': ('highlight', '-', 'dividing'),
    'control': ('control_negative', '-', 'matched control'),
    'shuffled': ('control_positive', '--', 'shuffled alignment'),
}


def _trajectory_figure(trajectories, measures):
    """Aligned trajectories: one panel per measure, a 95% band per series."""
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES

    n = max(1, len(measures))
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig = Figure(figsize=(2.4 * ncols + 0.6, 2.0 * nrows + 0.6), dpi=100)
    axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    _register_figure_data(fig, trajectories, x='rel_t', y='mean', hue='series', kind='line',
                          title='Division-aligned trajectories')
    for ax, measure in zip(axes, measures):
        part = trajectories[trajectories['measure'] == measure]
        for series, (role, style, label) in _SERIES_STYLE.items():
            for who in ('self', 'daughter'):
                line = part[(part['series'] == series) & (part['role'] == who)].sort_values('rel_t')
                if not len(line):
                    continue
                colour = ROLES[role]
                ax.fill_between(line['rel_t'], line['ci_low'], line['ci_high'],
                                color=colour, alpha=0.2, linewidth=0)
                ax.plot(line['rel_t'], line['mean'], style, color=colour, lw=1.2,
                        alpha=0.6 if who == 'daughter' else 1.0,
                        label=label if who == 'self' else None)
        ax.axvline(0, color=ROLES['reference'], lw=0.6)
        ax.axhline(0, color=ROLES['reference'], lw=0.6, ls=':')
        scale = part['scale'].iloc[0] if len(part) else ''
        ax.set_title(measure)
        ax.set_xlabel('frames from division')
        ax.set_ylabel('log ratio to baseline' if scale.startswith('log') else 'change from baseline')
    for ax in axes[len(measures):]:
        ax.set_visible(False)
    if not len(trajectories):
        axes[0].text(0.5, 0.5, 'no aligned divisions', ha='center', va='center',
                     transform=axes[0].transAxes)
    else:
        axes[0].legend(frameon=False, fontsize='small')
    fig.tight_layout()
    return fig


def _forest_figure(effects, measures):
    """Hedges' g with its 95% interval per measure, dividing and shuffled."""
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES

    fig = Figure(figsize=(5.0, 0.38 * max(1, len(measures)) + 1.2), dpi=100)
    ax = fig.subplots()
    _register_figure_data(fig, effects, x='g', y='measure', hue='contrast', kind='point',
                          title="Hedges' g per measure")
    marks = (('dividing_vs_control', 'highlight', 'o', 'dividing vs control', -0.15),
             ('shuffled_vs_its_control', 'control_positive', 's',
              'shuffled vs its control', 0.15))
    for contrast, role, marker, label, offset in marks if len(effects) else ():
        part = effects[effects['contrast'] == contrast].set_index('measure')
        ys, gs, lows, highs = [], [], [], []
        for y, measure in enumerate(measures):
            if measure in part.index and np.isfinite(part.at[measure, 'g']):
                ys.append(y + offset)
                gs.append(part.at[measure, 'g'])
                lows.append(part.at[measure, 'ci_low'])
                highs.append(part.at[measure, 'ci_high'])
        if not ys:
            continue
        gs, lows, highs = np.asarray(gs), np.asarray(lows, float), np.asarray(highs, float)
        err = np.vstack([np.where(np.isfinite(lows), gs - lows, 0.0),
                         np.where(np.isfinite(highs), highs - gs, 0.0)])
        ax.errorbar(gs, ys, xerr=err, fmt=marker, color=ROLES[role], ms=4, lw=1.0,
                    capsize=2, label=label)
    ax.axvline(0, color=ROLES['reference'], lw=0.6)
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(measures)
    ax.invert_yaxis()
    ax.set_xlabel("Hedges' g of the late-vs-baseline change (95% CI)")
    if len(effects):
        ax.legend(frameon=False, fontsize='small')
    else:
        ax.text(0.5, 0.5, 'no effect sizes', ha='center', va='center', transform=ax.transAxes)
    fig.tight_layout()
    return fig


def _size_control_figure(cycles, size_control, measure='area'):
    """Size at division against size at birth, with sizer, adder and timer lines."""
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import ROLES

    fig = Figure(figsize=(4.2, 3.6), dpi=100)
    ax = fig.subplots()
    birth_col, division_col = f'{measure}_birth', f'{measure}_division'
    _register_figure_data(fig, cycles, x=birth_col, y=division_col, kind='scatter',
                          title='Size at division against size at birth')
    if not len(cycles) or birth_col not in cycles:
        ax.text(0.5, 0.5, 'no complete cell cycles', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    birth = cycles[birth_col].to_numpy(float)
    division = cycles[division_col].to_numpy(float)
    ax.scatter(birth, division, s=10, color=ROLES['highlight'], alpha=0.6, linewidths=0)
    grid = np.linspace(np.nanmin(birth), np.nanmax(birth), 20)
    mean_b, mean_d = np.nanmean(birth), np.nanmean(division)
    for slope, style, label in ((0.0, ':', 'sizer'), (1.0, '--', 'adder'), (2.0, '-.', 'timer')):
        ax.plot(grid, mean_d + slope * (grid - mean_b), style, color=ROLES['reference'],
                lw=0.8, label=label)
    fits = size_control[size_control['size_measure'] == measure] if len(size_control) else size_control
    if len(fits) and np.isfinite(fits['slope_division'].iloc[0]):
        row = fits.iloc[0]
        ax.plot(grid, mean_d + row['slope_division'] * (grid - mean_b), color=ROLES['highlight'],
                lw=1.4, label=f"fit, slope {row['slope_division']:.2f} "
                              f"[{row.get('slope_division_ci_low', np.nan):.2f}, "
                              f"{row.get('slope_division_ci_high', np.nan):.2f}]")
    ax.set_xlabel(f'{measure} at birth')
    ax.set_ylabel(f'{measure} at division')
    ax.legend(frameon=False, fontsize='small')
    fig.tight_layout()
    return fig


def division_figures(results: dict, measures: Optional[Sequence[str]] = None) -> dict:
    """Draw the three figures of one division analysis.

    :param results: the dict from :func:`analyse_divisions` or
        :func:`read_division_results`.
    :param measures: the measurements to draw, in order; None draws every
        measurement that has effect sizes.
    :returns: ``{'trajectories': figure, 'forest': figure, 'size_control':
        figure}``, :class:`matplotlib.figure.Figure` objects: the aligned
        baseline-normalised trajectories of dividing cells (with their
        daughters after division), matched controls and the shuffled
        alignment with 95% bands; Hedges' g per measurement with its
        bootstrap interval; and area at division against area at birth with
        the sizer, adder and timer slopes.
    """
    effects = results.get('effects', pd.DataFrame())
    trajectories = results.get('trajectories', pd.DataFrame())
    if measures is None:
        measures = list(dict.fromkeys(effects['measure'])) if len(effects) else []
    measures = list(measures)
    return {'trajectories': _trajectory_figure(trajectories, measures),
            'forest': _forest_figure(effects, measures),
            'size_control': _size_control_figure(results.get('cycles', pd.DataFrame()),
                                                 results.get('size_control', pd.DataFrame()))}


def write_division_results(results: dict, out_dir, plot: bool = True) -> dict:
    """Write one division analysis as CSV tables, a JSON summary and figures.

    The tables are ``events.csv`` (daughters as ``;``-separated track ids),
    ``effects.csv``, ``trajectories.csv``, ``cycles.csv`` and
    ``size_control.csv``, written through :func:`spacr.tabular.write_table`;
    the summary is ``summary.json``. With ``plot``, the figures of
    :func:`division_figures` are saved through :func:`spacr.plot.save_figure`
    as ``division_trajectories``, ``division_forest`` and
    ``division_size_control`` in the figure format preference.

    :param results: the dict from :func:`analyse_divisions`.
    :param out_dir: the folder to write into; created when missing.
    :param plot: save the figures too.
    :returns: ``{name: path}`` of everything written.
    """
    import json
    import os

    from .tabular import write_table

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name in _RESULT_TABLES:
        table = results.get(name, pd.DataFrame())
        if name == 'events' and 'daughters' in table:
            table = table.assign(daughters=[';'.join(str(int(d)) for d in ds)
                                            for ds in table['daughters']])
        paths[name] = write_table(table, os.path.join(out_dir, f'{name}.csv'))
    summary = os.path.join(out_dir, 'summary.json')
    with open(summary, 'w', encoding='utf-8') as handle:
        json.dump(results.get('summary', {}), handle, indent=2, default=float)
    paths['summary'] = summary
    if plot:
        from .plot import save_figure

        for name, fig in division_figures(results).items():
            paths[f'figure_{name}'] = save_figure(
                fig, os.path.join(out_dir, f'division_{name}.pdf'), close=True)
    return paths


def read_division_results(folder) -> dict:
    """Read back a folder written by :func:`write_division_results`.

    :param folder: the results folder.
    :returns: the same dict :func:`analyse_divisions` returns, with
        ``events`` daughters as tuples again; a missing or empty table is an
        empty frame and a missing summary an empty dict.
    :raises FileNotFoundError: when ``folder`` holds neither ``summary.json``
        nor ``effects.csv``.
    """
    import json
    import os

    from .tabular import read_table

    if not any(os.path.isfile(os.path.join(folder, f)) for f in ('summary.json', 'effects.csv')):
        raise FileNotFoundError(f'no division analysis results in {folder}')
    out = {}
    for name in _RESULT_TABLES:
        path = os.path.join(folder, f'{name}.csv')
        try:
            out[name] = read_table(path, canonicalise=False, report=None)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            out[name] = pd.DataFrame()
    events = out['events']
    if 'daughters' in events:
        out['events'] = events.assign(daughters=[
            tuple(int(d) for d in str(ds).split(';') if d.strip()) if pd.notna(ds) else ()
            for ds in events['daughters']])
    summary = os.path.join(folder, 'summary.json')
    out['summary'] = {}
    if os.path.isfile(summary):
        with open(summary, encoding='utf-8') as handle:
            out['summary'] = json.load(handle)
    return out


def _division_settings(settings):
    """The division-analysis keyword arguments from the run settings."""
    mode = settings.get('timelapse_division_mode') or 'auto'
    window = settings.get('timelapse_division_window')
    tolerance = settings.get('timelapse_division_size_tolerance')
    if window in ('', 0):
        window = None
    if tolerance is None or tolerance == '':
        tolerance = 0.25
    if isinstance(tolerance, bool) or not np.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError('timelapse_division_size_tolerance must be a positive number')
    return {'mode': str(mode), 'window': None if window is None else int(window),
            'size_tol': float(tolerance)}


def _run_division_step(src, name, object_type, tracker, settings, *,
                       label_stack=None, images=None):
    """Run the division analysis on the tracks one field just produced.

    Reads ``<dirname(src)>/tracks/<tracker>_tracks_<object>_<name>.csv``,
    turns it into a lineage table (:func:`lineage_from_tracks`, with
    ``timelapse_lineage_max_distance``), measures ``label_stack`` (labels
    are track ids) with the first channel of ``images`` as intensity, and
    writes :func:`analyse_divisions` with ``timelapse_division_mode``,
    ``timelapse_division_window`` and ``timelapse_division_size_tolerance``
    to ``<tracks folder>/division/<tracks file stem>/`` with ``lineage.csv``
    beside it. A failure is reported and does not stop the run.

    :returns: the results dict, or None.
    """
    import os

    from .tabular import read_table, write_table

    prefix = 'trackpy' if tracker == 'iou' else tracker
    tracks_path = os.path.join(os.path.dirname(src), 'tracks',
                               f'{prefix}_tracks_{object_type}_{name}.csv')
    if not os.path.isfile(tracks_path):
        print(f'Division analysis skipped: no tracks table at {tracks_path}')
        return None
    if label_stack is None:
        print('Division analysis skipped: no tracked label stack')
        return None
    try:
        options = _division_settings(settings)
        tracks = read_table(tracks_path, report=None)
        distance = float(settings.get('timelapse_lineage_max_distance') or 30.0)
        lineage = lineage_from_tracks(tracks, max_distance=distance)
        planes = [np.asarray(plane) for plane in label_stack]
        intensity = None
        if images is not None:
            stack = np.asarray(images)
            stack = stack[..., 0] if stack.ndim == 4 else stack
            if stack.shape[:1] == (len(planes),) and stack.shape[1:] == planes[0].shape:
                intensity = list(stack)
        measures = add_motion_features(measure_label_stack(planes, intensity))
        results = analyse_divisions(measures, lineage, **options)
        stem = os.path.splitext(os.path.basename(tracks_path))[0]
        out_dir = os.path.join(os.path.dirname(tracks_path), 'division', stem)
        paths = write_division_results(
            results, out_dir, plot=bool(settings.get('save', True) or settings.get('plot', False)))
        paths['lineage'] = write_table(lineage, os.path.join(out_dir, 'lineage.csv'))
        results['paths'] = paths
    except Exception as exc:
        print(f'Division analysis could not run for {tracks_path}: {exc}')
        return None
    summary = results['summary']
    print(f"Division analysis ({object_type}, {name}): {summary['n_events_kept']} of "
          f"{summary['n_events']} {summary['mode']} divisions aligned, "
          f"{summary['n_controls_matched']} matched controls, window {summary['window']} frames; "
          f'written to {out_dir}')
    return results
