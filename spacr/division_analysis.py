"""What tracked cells do before they divide (alpha, feature item 771).

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


# --------------------------------------------------------------------------- #
# 1. measurements
# --------------------------------------------------------------------------- #
def measure_label_stack(masks: Iterable[np.ndarray],
                        images: Optional[Iterable[np.ndarray]] = None,
                        times: Optional[Sequence[int]] = None,
                        min_area: int = 1) -> pd.DataFrame:
    """Measure every labelled object in every frame of a tracked stack.

    Parameters
    ----------
    masks
        2D integer label planes; the label value is the track id.
    images
        Optional matching intensity planes (2D, or channel-first 3D of which
        channel 0 is measured). Mean intensity is divided by the frame median
        so slow illumination drift and bleaching cancel.
    times
        Frame time of each plane; defaults to ``0, 1, 2, ...``.
    min_area
        Objects smaller than this many pixels are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per (t, track_id) with :data:`SHAPE_MEASURES` and the
        centroid (``centroid_y``, ``centroid_x``) in pixels.
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


# --------------------------------------------------------------------------- #
# 2. lineage and division events
# --------------------------------------------------------------------------- #
def read_lineage_table(path) -> pd.DataFrame:
    """Read a lineage CSV (``label, t_start, t_end, parent``; CTC semantics)."""
    df = pd.read_csv(path)
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


# --------------------------------------------------------------------------- #
# 3. controls
# --------------------------------------------------------------------------- #
def _track_arrays(g: pd.DataFrame, column: str):
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


# --------------------------------------------------------------------------- #
# 4. alignment and per-unit change
# --------------------------------------------------------------------------- #
def align_units(measures: pd.DataFrame, units: pd.DataFrame, window: int,
                after: int = 0, columns: Sequence[str] = COMPARED_MEASURES) -> pd.DataFrame:
    """Measurements of each unit's track on ``rel_t = t - t0``.

    ``units`` needs ``unit_id``, ``track_id`` and ``t0``; with an optional
    ``daughters`` column (tuples of track ids) the daughters are added for
    ``rel_t = 1 .. after`` with ``role='daughter'``. The unit's own track is
    read for ``rel_t = -window .. after`` with ``role='self'`` (a fission
    parent simply has no rows after 0; a budding mother continues).
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


# --------------------------------------------------------------------------- #
# 5. effect sizes and 6. meta-analysis
# --------------------------------------------------------------------------- #
def _hedges_g(x, y):
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

    Returns ``g``, ``var_g`` (large-sample variance used for meta-analysis),
    ``ci_low``/``ci_high`` (percentile bootstrap, each group resampled
    independently), ``mean_diff`` and the group sizes. Non-finite values are
    dropped first.
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

    Returns the pooled effect ``mu`` with ``se``, a normal 95% CI, ``z``,
    two-sided ``p``, between-study variance ``tau2``, Cochran's ``Q`` and
    ``I2`` (percent), and ``k`` studies. Studies with a non-finite effect or
    non-positive variance are skipped.
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


# --------------------------------------------------------------------------- #
# 7. cell-size control
# --------------------------------------------------------------------------- #
def cell_cycles(measures: pd.DataFrame, lineage: pd.DataFrame, events: pd.DataFrame,
                size_columns: Sequence[str] = ('area', 'major_axis')) -> pd.DataFrame:
    """Complete fission cycles: born at an observed two-daughter division and
    seen to divide into two.

    Birth size is the size on the first frame, division size on the last.
    Returns ``track_id, t_birth, t_division, duration`` and, per size column,
    ``<col>_birth``, ``<col>_division`` and ``<col>_added``.
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
