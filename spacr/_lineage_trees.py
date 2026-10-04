"""Lineage trees built from tracked objects.

Division events are read from explicit parent columns or inferred from
nearby track starts, then drawn as tree figures, written as Newick text
and summarised as per-lineage statistics, optionally calibrated to hours.
"""
import os

import numpy as np
import pandas as pd


def _save_lineage_figure(figure, path, **kwargs):
    """Load the common figure exporter only when an export is requested."""
    from .plot import save_figure

    return save_figure(figure, path, **kwargs)


_LINEAGE_SEGMENT_STATS = ('generation_time', 'generation_time_hours', 'generation', 'start_frame',
                          'n_frames')


def _lineage_explicit_parents(df, spans):
    """Read the tracker's own division links from a ``parent_track_id`` column.

    Trackers that link divisions natively (Trackastra) write the mother's
    track id on every row of a daughter track; 0, blank or an id that is not a
    track in the table means no parent.

    :param df: one field's tracks table.
    :param spans: first and last frame per track, indexed by track id.
    :returns: ``{daughter_track_id: mother_track_id}``.
    """
    if 'parent_track_id' not in df.columns:
        return {}
    links = {}
    firsts = df.dropna(subset=['parent_track_id']).groupby('track_id')['parent_track_id'].first()
    for track, parent in firsts.items():
        parent = int(parent)
        if parent > 0 and parent != int(track) and parent in spans.index:
            links[int(track)] = parent
    return links


def _lineage_inferred_parents(df, spans, max_distance, skip, strict=False, *,
                              persist_frames=1, border_margin=0.0,
                              field_shape=None):
    """Infer division links from where new tracks start.

    A track that starts after the field's first frame is a daughter of the
    object nearest its first position in the frame before, within
    ``max_distance`` pixels. The event is kept as a division when that mother
    track goes on (the tracker kept one daughter under the mother's id) or
    when at least two tracks start beside the mother as it ends (both
    daughters got new ids). One new track beside an ending one is a broken
    track rather than a division and is left as its own tree.

    :param df: one field's tracks table with ``frame``, ``track_id``, ``x``
        and ``y``.
    :param spans: first and last frame per track, indexed by track id.
    :param max_distance: largest mother-to-daughter distance in pixels.
    :param skip: track ids whose parent is already known.
    :param strict: for a tracker that reports its own divisions, where a
        mother never keeps her id through one. Only two or more tracks that
        start together beside a mother whose track ends in the frame before
        them form a division; a lone new track, or one beside a mother that
        goes on, is a track entering or breaking and stays a root.
    :param persist_frames: every daughter, and a mother that goes on after
        the division, must be seen in at least this many frames from the
        division (fewer only when the movie ends sooner), and the mother
        must be seen in each of this many frames before it (fewer only when
        her track starts later). A flickering mask or a track that breaks
        just before the event therefore gives no division. A new track that
        ends sooner is not a daughter unless it ends at the edge of the field
        (it left the field).
    :param border_margin: pixels from the edge of the field. A mother whose
        track ends this close to the edge has left the field; no division is
        inferred from her.
    :param field_shape: the image ``(height, width)`` in pixels; without it
        no track counts as being at the edge.
    :returns: ``{daughter_track_id: mother_track_id}``.
    """
    first_frame = spans['start'].min()
    last_frame = spans['end'].max()
    persist = max(int(persist_frames), 1)
    frames_of = df.groupby('track_id')['frame'].agg(lambda f: set(f.tolist()))
    last_xy = df.sort_values('frame').groupby('track_id')[['x', 'y']].last()
    if field_shape is not None and border_margin > 0:
        height, width = (float(v) - 1 for v in field_shape[-2:])
        edge = pd.concat([last_xy['x'], width - last_xy['x'],
                          last_xy['y'], height - last_xy['y']], axis=1).min(axis=1)
        at_edge = edge < border_margin
    else:
        at_edge = pd.Series(False, index=last_xy.index)
    starts = df.sort_values('frame').groupby('track_id').first()
    candidates = {}
    for track, row in starts.iterrows():
        track = int(track)
        if track in skip or row['frame'] <= first_frame:
            continue
        before = df[(df['frame'] == row['frame'] - 1) & (df['track_id'] != track)]
        if before.empty:
            continue
        dist = np.hypot(before['x'].to_numpy() - row['x'],
                        before['y'].to_numpy() - row['y'])
        nearest = int(np.argmin(dist))
        if dist[nearest] <= max_distance:
            candidates.setdefault((int(before['track_id'].iloc[nearest]),
                                   int(row['frame'])), []).append(track)
    links = {}
    for (mother, frame), daughters in candidates.items():
        continues = spans.loc[mother, 'end'] >= frame
        after = min(persist, last_frame - frame + 1)
        before = range(max(frame - persist, spans.loc[mother, 'start']), frame)
        steady = [d for d in daughters if len(frames_of[d]) >= after or at_edge[d]]
        settled = not continues and frame - persist >= spans.loc[mother, 'start']
        if not (settled and steady and len(daughters) >= 2):
            daughters = steady
        if not daughters:
            continue
        if continues and sum(f >= frame for f in frames_of[mother]) < after:
            continue
        if not set(before) <= frames_of[mother]:
            continue
        if not continues and at_edge[mother]:
            continue
        if strict:
            if len(daughters) >= 2 and spans.loc[mother, 'end'] == frame - 1:
                links.update({d: mother for d in daughters})
        elif continues or len(daughters) >= 2:
            links.update({d: mother for d in daughters})
    return links


def _lineage_hours_per_frame(df, frame_interval_s):
    """Hours per frame from a ``time_s`` column or a seconds-per-frame value.

    :returns: a positive float, or None when the table carries no usable
        timing and no interval is given.
    """
    if 'time_s' in df.columns and not df['time_s'].map(
            lambda value: isinstance(value, (bool, np.bool_))).any():
        times = pd.to_numeric(df['time_s'], errors='coerce')
        frames = df['frame'][times.notna()]
        span = frames.max() - frames.min() if len(frames) else 0
        if span > 0:
            elapsed = times.max() - times.min()
            if np.isfinite(elapsed) and elapsed > 0:
                return float(elapsed) / float(span) / 3600.0
    try:
        interval = float(frame_interval_s)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(frame_interval_s, (bool, np.bool_)) or not np.isfinite(interval) or interval <= 0:
        return None
    return interval / 3600.0


def _lineage_drop_short_cycles(parents, explicit, spans, min_cycle):
    """Remove inferred divisions that would make a cell cycle implausibly short.

    Divisions are visited in time order. The cycle of a dividing cell starts
    at her own birth by a division, or at an earlier division her track went
    on through. When that start is fewer than ``min_cycle`` frames before
    the new division, one of the two is an artefact. A division whose mother
    track ends there outweighs an earlier inferred one that the track went
    on through, or that gave more than two daughters (only this cell is
    taken out of it); otherwise the later inferred division is dropped.
    Tracker-reported divisions are always kept.

    :param parents: ``{daughter_track_id: mother_track_id}``.
    :param explicit: daughter track ids whose link the tracker reported.
    :param spans: first and last frame per track, indexed by track id.
    :param min_cycle: shortest plausible cycle in frames.
    :returns: the kept ``{daughter_track_id: mother_track_id}``.
    """
    events = {}
    for daughter, mother in parents.items():
        events.setdefault((mother, int(spans.loc[daughter, 'start'])), []).append(daughter)
    kept = {}
    for mother, birth in sorted(events, key=lambda e: e[1]):
        daughters = events[(mother, birth)]
        starts = [(int(spans.loc[mother, 'start']), event)
                  for event, kids in kept.items() if mother in kids]
        starts += [(event[1], event) for event in kept
                   if event[0] == mother and event[1] < birth
                   and spans.loc[mother, 'end'] >= event[1]]
        if starts:
            start, earlier = max(starts)
            if birth - start < min_cycle:
                later_known = any(d in explicit for d in daughters)
                earlier_known = any(d in explicit for d in kept[earlier])
                earlier_goes_on = spans.loc[earlier[0], 'end'] >= earlier[1]
                later_ends = spans.loc[mother, 'end'] < birth
                if earlier_known or not (later_ends and (
                        earlier_goes_on or len(kept[earlier]) > 2)):
                    if not later_known:
                        continue
                else:
                    kids = [d for d in kept[earlier] if d != mother]
                    if earlier[0] == mother or not kids or (
                            len(kids) < 2 and not earlier_goes_on):
                        del kept[earlier]
                    else:
                        kept[earlier] = kids
        kept[(mother, birth)] = daughters
    return {d: m for (m, _), kids in kept.items() for d in kids}


def _lineage_segments(tracks, max_distance=30.0, *, min_division_h=None,
                      frame_interval_s=None, persist_frames=3, border_margin=10.0,
                      field_shape=None):
    """Cut one field's tracks into cell-cycle segments linked by divisions.

    A segment is one cell from its birth, or its first frame, to its last
    frame before it divides, or its last frame. Division links come from a
    ``parent_track_id`` column where the tracker wrote one and are otherwise
    inferred (see :func:`_lineage_inferred_parents`). Tracks a native
    tracker marked in ``parent_track_id_source`` are never inferred; when a
    ``parent_track_id`` column is present without that mark, inference is
    strict: two daughters beside a mother that ends just before them. When the tracker kept
    one daughter under the mother's id, that track is cut at the division
    and its remainder becomes a daughter segment.

    The generation time of a segment is the number of frames from its birth
    to its division (the first frame of its daughters minus its own first
    frame). Only segments born by a division that divide again have one; a
    tree's root and its leaves are incomplete cycles and get NaN.

    :param tracks: tracks table with ``frame``, ``track_id``, ``x`` and ``y``.
    :param max_distance: largest mother-to-daughter distance in pixels for an
        inferred division.
    :param min_division_h: shortest plausible time between divisions in
        hours. With a frame time (``time_s`` or ``frame_interval_s``), no
        division is inferred in a movie shorter than half of it, and of two
        divisions closer together in one cell's history the inferred one is
        dropped (see :func:`_lineage_drop_short_cycles`). None, or no
        frame time, leaves inference unlimited by time.
    :param frame_interval_s: seconds per frame, used when the table has no
        ``time_s`` column.
    :param persist_frames: frames an inferred daughter must persist, and
        the mother must be seen without a gap before dividing.
    :param border_margin: pixels from the field edge within which a track
        that ends has left the field rather than divided.
    :param field_shape: the image ``(height, width)`` the positions lie in;
        None turns the field-edge rule off.
    :returns: one row per segment with ``segment_id``, ``track_id``,
        ``parent_segment_id`` (0 for a root), ``lineage_id`` (the root's
        segment id), ``generation``, ``start_frame``, ``end_frame``,
        ``n_frames``, ``n_daughters``, ``generation_time`` and
        ``division_source`` (tracker, inferred or blank).
    """
    df = tracks.dropna(subset=['track_id', 'frame', 'x', 'y']).copy()
    columns = ['segment_id', 'track_id', 'parent_segment_id', 'lineage_id',
               'generation', 'start_frame', 'end_frame', 'n_frames',
               'n_daughters', 'generation_time', 'division_source']
    if df.empty:
        return pd.DataFrame(columns=columns)
    df['track_id'] = df['track_id'].astype(int)
    df['frame'] = df['frame'].astype(int)
    spans = df.groupby('track_id')['frame'].agg(start='min', end='max')

    explicit = _lineage_explicit_parents(df, spans)
    known = set(explicit)
    if 'parent_track_id_source' in df:
        # A native root (including an orphan after filtering) must not be
        # silently attached to whichever unrelated track is closest.
        native = df['parent_track_id_source'].fillna('').isin(
            ['ultrack', 'btrack', 'trackastra', 'sam2'])
        known.update(df.loc[native, 'track_id'].astype(int))
    hours = _lineage_hours_per_frame(df, frame_interval_s)
    min_cycle = None
    if min_division_h and hours:
        min_cycle = float(min_division_h) / hours
        if (spans['end'].max() - spans['start'].min()) < min_cycle / 2:
            known.update(spans.index.astype(int))
    inferred = _lineage_inferred_parents(df, spans, float(max_distance),
                                         known,
                                         strict='parent_track_id' in df.columns,
                                         persist_frames=persist_frames,
                                         border_margin=float(border_margin or 0.0),
                                         field_shape=field_shape)
    parents = {**inferred, **explicit}
    if min_cycle is not None:
        parents = _lineage_drop_short_cycles(parents, set(explicit), spans, min_cycle)

    segments = {}
    by_track = {}
    for sid, (track, span) in enumerate(spans.iterrows(), start=1):
        segments[sid] = {'segment_id': sid, 'track_id': int(track),
                         'parent_segment_id': 0,
                         'start_frame': int(span['start']),
                         'end_frame': int(span['end']),
                         'division_source': ''}
        by_track[int(track)] = [sid]

    def covering(track, frame):
        """The segment of ``track`` alive at ``frame``, else its last one before."""
        chosen = by_track[track][0]
        for sid in by_track[track]:
            if segments[sid]['start_frame'] <= frame:
                chosen = sid
        return chosen

    for daughter in sorted(parents, key=lambda d: spans.loc[d, 'start']):
        mother = parents[daughter]
        birth = int(spans.loc[daughter, 'start'])
        if spans.loc[mother, 'start'] >= birth:
            continue
        msid = covering(mother, birth - 1)
        mseg = segments[msid]
        if mseg['end_frame'] >= birth:
            nsid = max(segments) + 1
            segments[nsid] = {'segment_id': nsid, 'track_id': mother,
                              'parent_segment_id': msid, 'start_frame': birth,
                              'end_frame': mseg['end_frame'],
                              'division_source': ''}
            mseg['end_frame'] = birth - 1
            by_track[mother] = sorted(by_track[mother] + [nsid],
                                      key=lambda s: segments[s]['start_frame'])
        dsid = by_track[daughter][0]
        segments[dsid]['parent_segment_id'] = msid
        mseg['division_source'] = ('tracker' if daughter in explicit
                                   else 'inferred')

    seg = pd.DataFrame(list(segments.values()))
    seg['n_frames'] = seg['end_frame'] - seg['start_frame'] + 1
    parent_of = dict(zip(seg['segment_id'], seg['parent_segment_id']))
    children = seg[seg['parent_segment_id'] > 0].groupby('parent_segment_id')
    seg['n_daughters'] = seg['segment_id'].map(children.size()).fillna(0).astype(int)
    division_at = children['start_frame'].min()

    def root_and_depth(sid):
        """The root segment of ``sid`` and how many divisions separate them."""
        depth = 0
        while parent_of.get(sid, 0):
            sid = parent_of[sid]
            depth += 1
        return sid, depth

    rooted = [root_and_depth(s) for s in seg['segment_id']]
    seg['lineage_id'] = [r for r, _ in rooted]
    seg['generation'] = [d for _, d in rooted]
    complete = (seg['parent_segment_id'] > 0) & (seg['n_daughters'] > 0)
    seg['generation_time'] = np.where(
        complete, seg['segment_id'].map(division_at) - seg['start_frame'],
        np.nan)
    return seg[columns].sort_values(['lineage_id', 'start_frame', 'segment_id']).reset_index(drop=True)


def _lineage_colour_values(segments, tracks, color_by, measurements=None):
    """The value each segment is coloured by.

    A segment statistic (``generation_time``, ``generation``,
    ``start_frame``, ``n_frames``) is used as it is. Any other name is a
    measurement column, averaged over the segment's frames, looked up first
    in ``measurements`` (keyed by ``frame`` and ``track_id``) and then in the
    tracks table.

    :returns: a float Series aligned with ``segments``, NaN where unknown.
    :raises KeyError: when ``color_by`` is found nowhere.
    """
    if color_by in _LINEAGE_SEGMENT_STATS:
        return segments[color_by].astype(float)
    source = None
    for frame in (measurements, tracks):
        if frame is not None and color_by in frame.columns:
            source = frame[['frame', 'track_id', color_by]].dropna()
            break
    if source is None:
        raise KeyError(
            f"color_by {color_by!r} is neither a segment statistic "
            f"({', '.join(_LINEAGE_SEGMENT_STATS)}) nor a column of the "
            f"tracks or measurements")
    values = []
    for _, row in segments.iterrows():
        rows = source[(source['track_id'] == row['track_id'])
                      & source['frame'].between(row['start_frame'], row['end_frame'])]
        values.append(float(pd.to_numeric(rows[color_by], errors='coerce').mean())
                      if len(rows) else np.nan)
    return pd.Series(values, index=segments.index, dtype=float)


def _sibling_correlation(segments):
    """Pearson correlation of sister generation times.

    Sisters are complete-cycle segments with the same mother; every ordered
    pair counts once each way, so the correlation is symmetric.

    :returns: ``(number_of_sister_pairs, r)``; r is NaN below three pairs.
    """
    done = segments.dropna(subset=['generation_time'])
    a, b = [], []
    for _, sisters in done.groupby('parent_segment_id'):
        times = sisters['generation_time'].to_numpy(dtype=float)
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                a += [times[i], times[j]]
                b += [times[j], times[i]]
    pairs = len(a) // 2
    if pairs < 3 or np.std(a) == 0:
        return pairs, np.nan
    return pairs, float(np.corrcoef(a, b)[0, 1])


def _lineage_statistics(segments):
    """Per-lineage statistics, with a last row over every lineage.

    :returns: one row per lineage with ``lineage_id``, ``root_track_id``,
        ``n_cells``, ``n_divisions``, ``max_generation``,
        ``n_complete_cycles``, the mean, median and standard deviation of
        ``generation_time`` in frames, ``sibling_pairs`` and
        ``sibling_correlation``; the ``all`` row pools every lineage.
    """
    rows = []
    groups = [(lid, g) for lid, g in segments.groupby('lineage_id')]
    groups.append(('all', segments))
    for lid, g in groups:
        times = g['generation_time'].dropna()
        pairs, r = _sibling_correlation(g)
        roots = g[g['parent_segment_id'] == 0]
        rows.append({
            'lineage_id': lid,
            'root_track_id': (int(roots['track_id'].iloc[0])
                              if lid != 'all' and len(roots) else np.nan),
            'n_cells': len(g),
            'n_divisions': int((g['n_daughters'] > 0).sum()),
            'max_generation': int(g['generation'].max()) if len(g) else 0,
            'n_complete_cycles': len(times),
            'generation_time_mean': float(times.mean()) if len(times) else np.nan,
            'generation_time_median': float(times.median()) if len(times) else np.nan,
            'generation_time_sd': float(times.std()) if len(times) > 1 else np.nan,
            'sibling_pairs': pairs,
            'sibling_correlation': r,
        })
    return pd.DataFrame(rows)


def _lineage_newick(segments):
    """Write each lineage as a Newick tree, one tree per line.

    Nodes are named ``t<track_id>_f<start_frame>`` and branch lengths are the
    segment's length in frames.

    :returns: the Newick text, each tree ending in ``;``.
    """
    kids = {sid: g['segment_id'].tolist()
            for sid, g in segments.groupby('parent_segment_id')}
    by_id = segments.set_index('segment_id')

    def node(sid):
        """The Newick text of ``sid`` and everything below it."""
        row = by_id.loc[sid]
        label = f"t{int(row['track_id'])}_f{int(row['start_frame'])}:{int(row['n_frames'])}"
        below = kids.get(sid, [])
        if not below:
            return label
        return '(' + ','.join(node(k) for k in below) + ')' + label

    roots = segments.loc[segments['parent_segment_id'] == 0, 'segment_id']
    return ''.join(node(int(r)) + ';\n' for r in roots)


def _lineage_tree_figure(segments, values, color_by, title='', max_lineages=40):
    """Draw lineages as trees on a frame axis, coloured by ``values``.

    Each segment is a horizontal line from its first frame to where it
    divides or ends; a vertical line joins sisters at the division. Lineages
    with divisions are drawn first, at most ``max_lineages`` of them.

    :returns: a :class:`matplotlib.figure.Figure`.
    """
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    from .figures.style import ROLES

    seg = segments.assign(_value=values.to_numpy())
    kids = {sid: g.sort_values('segment_id')['segment_id'].tolist()
            for sid, g in seg.groupby('parent_segment_id')}
    by_id = seg.set_index('segment_id')
    divided = seg.groupby('lineage_id')['n_daughters'].sum().sort_values(ascending=False)
    lineages = divided.index.tolist()[:max_lineages]

    finite = seg['_value'][np.isfinite(seg['_value'])]
    norm = Normalize(finite.min(), finite.max()) if len(finite) else Normalize(0, 1)
    if len(finite) and norm.vmin == norm.vmax:
        norm = Normalize(norm.vmin - 0.5, norm.vmax + 0.5)
    cmap = colormaps['viridis']

    leaves = int(sum((seg['lineage_id'].isin(lineages)) & (seg['n_daughters'] == 0)))
    fig = Figure(figsize=(7.0, max(2.5, 0.22 * leaves + 1.2)), dpi=100)
    ax = fig.subplots()
    cursor = [0.0]

    def place(sid):
        """Draw ``sid`` and its descendants; return the y it was drawn at."""
        row = by_id.loc[sid]
        below = kids.get(sid, [])
        if below:
            ys = [place(k) for k in below]
            y = float(np.mean(ys))
            x_div = min(by_id.loc[k, 'start_frame'] for k in below)
            ax.plot([x_div, x_div], [min(ys), max(ys)], lw=0.8,
                    color=ROLES['reference'])
            x_end = x_div
        else:
            y = cursor[0]
            cursor[0] += 1.0
            x_end = row['end_frame'] + 1
        value = row['_value']
        colour = cmap(norm(value)) if np.isfinite(value) else ROLES['data']
        ax.plot([row['start_frame'], x_end], [y, y], lw=2.0, color=colour,
                solid_capstyle='butt')
        return y

    for lid in lineages:
        place(int(lid))
        cursor[0] += 0.8
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel('frame')
    ax.set_title(title or 'lineage trees')
    if not lineages:
        ax.text(0.5, 0.5, 'no tracks', ha='center', va='center',
                transform=ax.transAxes)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, label=color_by, fraction=0.04)
    fig.tight_layout()
    return fig



def _lineage_calibrate_time(segments, tracks, frame_interval_s=None):
    """Add hours without changing frame-based lineage quantities.

    A complete time_s column maps observed frame numbers to seconds; all
    observations of a frame must agree and time must strictly increase.
    An optional positive interval must agree with these timestamps (an
    arbitrary time origin is allowed). Otherwise the interval supplies the
    calibration. No calibration leaves hours missing, as do incomplete
    cycles. Neither playback rate nor motility defaults are consulted.
    """
    interval = None
    if frame_interval_s is not None:
        if isinstance(frame_interval_s, (bool, np.bool_)):
            raise ValueError('lineage frame_interval_s must be finite and positive')
        try:
            interval = float(frame_interval_s)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('lineage frame_interval_s must be finite and positive') from exc
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('lineage frame_interval_s must be finite and positive')
    source = 'missing'
    by_frame = None
    if 'time_s' in tracks.columns and len(tracks):
        data = tracks[['frame', 'time_s']].copy()
        for column in data:
            if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError(f'lineage {column} must contain finite numbers, not booleans')
            data[column] = pd.to_numeric(data[column], errors='coerce')
            if not np.isfinite(data[column]).all():
                raise ValueError(f'lineage {column} must contain finite numbers')
        if (data['frame'] % 1 != 0).any():
            raise ValueError('lineage frame numbers must be integers')
        grouped = data.groupby('frame')['time_s']
        if (grouped.nunique() != 1).any():
            raise ValueError('lineage time_s disagrees between observations of the same frame')
        by_frame = grouped.first().sort_index()
        times = by_frame.to_numpy(dtype=np.longdouble)
        frames = by_frame.index.to_numpy(dtype=np.longdouble)
        elapsed = np.diff(times)
        if (elapsed <= 0).any():
            raise ValueError('lineage time_s must strictly increase with frame')
        if interval is not None and not np.allclose(
                elapsed, np.diff(frames) * np.longdouble(interval), rtol=1e-9, atol=0):
            raise ValueError('lineage time_s conflicts with frame_interval_s')
        source = 'time_s'
    elif interval is not None:
        source = 'frame_interval_s'
    result = segments.copy()
    hours = np.full(len(result), np.nan, dtype=float)
    divisions = result[result['parent_segment_id'] > 0].groupby(
        'parent_segment_id')['start_frame'].min()
    for position, row in enumerate(result.itertuples()):
        if not np.isfinite(row.generation_time) or source == 'missing':
            continue
        if by_frame is not None:
            birth = by_frame.loc[row.start_frame]
            division = by_frame.loc[divisions.loc[row.segment_id]]
            value = (np.longdouble(division) - np.longdouble(birth)) / 3600
        else:
            value = np.longdouble(row.generation_time) * np.longdouble(interval) / 3600
        with np.errstate(over='ignore', under='ignore'):
            hours[position] = float(value)
        if not np.isfinite(hours[position]) or hours[position] <= 0:
            raise ValueError('lineage generation time in hours is not representable')
    result['generation_time_hours'] = hours
    calibration = {'time_calibration_source': source,
                   'calibration_frame_interval_s': interval if interval is not None else np.nan,
                   'generation_time_unit': 'frames', 'calibrated_time_unit': 'hours'}
    for key, value in calibration.items():
        result[key] = value
    return result, calibration


def _lineage_hour_statistics(segments, statistics, calibration):
    """Append calibrated complete-cycle summaries with the same lineage grouping."""
    result = statistics.copy()
    for stat in ('mean', 'median', 'sd'):
        result[f'generation_time_hours_{stat}'] = np.nan
    for index, row in result.iterrows():
        group = segments if row['lineage_id'] == 'all' else segments[
            segments['lineage_id'] == row['lineage_id']]
        values = group['generation_time_hours'].dropna()
        if len(values):
            scale = float(values.max())
            scaled = values.to_numpy(dtype=float) / scale
            result.loc[index, 'generation_time_hours_mean'] = float(scaled.mean() * scale)
            result.loc[index, 'generation_time_hours_median'] = float(np.median(scaled) * scale)
            if len(values) > 1:
                result.loc[index, 'generation_time_hours_sd'] = float(scaled.std(ddof=1) * scale)
    for key, value in calibration.items():
        result[key] = value
    return result


def _lineage_trees_from_tracks(tracks_path, out_dir=None, *,
                               color_by='generation_time', max_distance=30.0,
                               measurements=None, plot=True, frame_interval_s=None, tracks_snapshot=None,
                               min_division_h=None, field_shape=None):
    """Build lineage trees from one tracks table and write them beside it.

    Reads a tracks CSV written by any spaCR tracker (one field per file),
    cuts it into cell-cycle segments linked by divisions
    (:func:`_lineage_segments`) and writes, under ``out_dir``, the
    ``<stem>_segments.csv`` table with each segment's colour value,
    ``<stem>_lineage_stats.csv`` with per-lineage generation times and
    sibling correlation, ``<stem>.nwk`` with one Newick tree per lineage and,
    with ``plot``, the ``<stem>_lineage`` figure.

    :param tracks_path: a ``*_tracks_<object>_<field>.csv`` file.
    :param out_dir: output folder; ``<tracks folder>/lineage`` when None.
    :param color_by: segment statistic or measurement column to colour by.
    :param max_distance: largest mother-to-daughter distance in pixels for an
        inferred division.
    :param measurements: optional table keyed by ``frame`` and ``track_id``
        to take ``color_by`` from.
    :param plot: save the tree figure (its axis stays in frames).
    :param frame_interval_s: optional positive seconds per frame; validates
        against time_s when present, otherwise calibrates generation times.
        Missing calibration leaves the added hours columns NaN.
    :param tracks_snapshot: optional validated tracks snapshot from the measured-feature reader.
    :param min_division_h: shortest plausible hours between divisions for
        inferred divisions (see :func:`_lineage_segments`); None for no limit.
    :param field_shape: optional image ``(height, width)`` for the field-edge
        rule of inferred divisions.
    :returns: dict with the ``segments`` and ``statistics`` frames and the
        ``paths`` written.
    """
    from .tabular import read_table, write_table

    tracks = (read_table(tracks_path, report=None) if tracks_snapshot is None
              else tracks_snapshot.copy())
    stem = os.path.splitext(os.path.basename(tracks_path))[0]
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(tracks_path)), 'lineage')
    segments = _lineage_segments(tracks, max_distance=max_distance,
                                  min_division_h=min_division_h,
                                  frame_interval_s=frame_interval_s,
                                  field_shape=field_shape)
    segments, calibration = _lineage_calibrate_time(segments, tracks, frame_interval_s)
    try:
        values = _lineage_colour_values(segments, tracks, color_by, measurements)
    except KeyError as exc:
        print(f"{exc.args[0]}; colouring by generation_time instead.")
        color_by = 'generation_time'
        values = segments['generation_time'].astype(float)
    segments[f'color_{color_by}'] = values.to_numpy()
    stats = _lineage_hour_statistics(segments, _lineage_statistics(segments), calibration)
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        'segments': write_table(segments, os.path.join(out_dir, f'{stem}_segments.csv')),
        'statistics': write_table(stats, os.path.join(out_dir, f'{stem}_lineage_stats.csv')),
    }
    newick = os.path.join(out_dir, f'{stem}.nwk')
    with open(newick, 'w', encoding='utf-8') as handle:
        handle.write(_lineage_newick(segments))
    paths['newick'] = newick
    if plot:
        fig = _lineage_tree_figure(segments, values, color_by, title=stem)
        paths['figure'] = _save_lineage_figure(
            fig, os.path.join(out_dir, f'{stem}_lineage.pdf'), close=True)
    return {'segments': segments, 'statistics': stats, 'paths': paths,
            'calibration': calibration}


def _lineage_min_division_h(settings):
    """The ``timelapse_lineage_min_division_h`` setting in hours, or None.

    A missing key gives the 6-hour default; blank, zero or a negative value
    turns the limit off; anything that is not a finite number is refused.
    """
    value = settings.get('timelapse_lineage_min_division_h', 6.0)
    if value is None or value == '':
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('timelapse_lineage_min_division_h must be a number of hours')
    try:
        hours = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('timelapse_lineage_min_division_h must be a number of hours') from exc
    if not np.isfinite(hours):
        raise ValueError('timelapse_lineage_min_division_h must be finite')
    return hours if hours > 0 else None


def _run_lineage_step(src, name, object_type, mode, settings, *,
                      frame_sources=None, label_stack=None):
    """Draw lineage trees from the tracks one field just produced.

    Looks for ``<dirname(src)>/tracks/<tracker>_tracks_<object>_<name>.csv``
    and passes it to :func:`_lineage_trees_from_tracks` with the
    ``timelapse_lineage_color_by``, ``timelapse_lineage_max_distance`` and
    ``timelapse_lineage_min_division_h`` settings. A failure is reported and does not stop the run.

    :returns: the result of :func:`_lineage_trees_from_tracks`, or None.
    """
    prefix = 'trackpy' if mode == 'iou' else mode
    tracks_path = os.path.join(os.path.dirname(src), 'tracks',
                               f'{prefix}_tracks_{object_type}_{name}.csv')
    if not os.path.isfile(tracks_path):
        print(f"Lineage trees skipped: no tracks table at {tracks_path}")
        return None
    try:
        source_record = None
        if frame_sources is not None:
            from ._lineage_measurements import _SUFFIX, _atomic_json, _prepare_lineage_sources
            source_record = _prepare_lineage_sources(
                tracks_path, object_type, frame_sources, label_stack, settings.get('frame_interval_s'))
        result = _lineage_trees_from_tracks(
            tracks_path,
            color_by=settings.get('timelapse_lineage_color_by') or 'generation_time',
            max_distance=float(settings.get('timelapse_lineage_max_distance') or 30.0),
            frame_interval_s=settings.get('frame_interval_s'),
            min_division_h=_lineage_min_division_h(settings),
            field_shape=None if label_stack is None else np.shape(label_stack)[-2:],
            plot=bool(settings.get('save', True) or settings.get('plot', False)))
        if source_record is not None:
            _atomic_json(tracks_path + _SUFFIX, source_record)
    except Exception as exc:
        print(f"Lineage trees could not be built for {tracks_path}: {exc}")
        return None
    overall = result['statistics'].iloc[-1]
    print(f"Lineage trees ({object_type}, {name}): "
          f"{int(overall['n_divisions'])} divisions, "
          f"{int(overall['n_complete_cycles'])} complete cycles; "
          f"written to {os.path.dirname(result['paths']['segments'])}")
    return result
