"""Division-aligned analysis on small synthetic lineages (feature item 771)."""
import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da


def _disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _rect(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_measurements_read_area_axes_and_normalised_intensity():
    mask = np.zeros((40, 60), np.int32)
    mask[_rect(mask.shape, 10, 14, 5, 25)] = 3      # 4 x 20 rod
    mask[_disc(mask.shape, 25, 45, 6)] = 7          # disc
    img = np.ones_like(mask, float) * 10.0
    img[mask == 7] = 40.0
    df = da.measure_label_stack([mask], [img], times=[5])
    rod = df[df.track_id == 3].iloc[0]
    disc = df[df.track_id == 7].iloc[0]
    assert rod.t == 5 and rod.area == 80
    assert rod.aspect_ratio > 4 and disc.aspect_ratio < 1.1
    assert disc.circularity > rod.circularity
    assert rod.solidity == pytest.approx(1.0) and 0.85 < disc.solidity <= 1.0
    # frame median is the background (10), so the disc reads 4x and the rod 1x
    assert disc.mean_intensity == pytest.approx(4.0)
    assert rod.mean_intensity == pytest.approx(1.0)


def test_measurement_rejects_mismatched_planes():
    with pytest.raises(ValueError):
        da.measure_label_stack([np.zeros((4, 4), int)], [np.zeros((5, 5))])


def test_motion_speed_displacement_and_turning():
    df = pd.DataFrame({'track_id': 1, 't': [0, 1, 2, 3],
                       'centroid_y': [0.0, 0.0, 3.0, 3.0],
                       'centroid_x': [0.0, 3.0, 3.0, 3.0]})
    out = da.add_motion_features(df)
    assert np.isnan(out.speed.iat[0])
    assert out.speed.tolist()[1:] == [3.0, 3.0, 0.0]
    assert out.displacement.iat[2] == pytest.approx(np.hypot(3, 3))
    assert out.turning.iat[2] == pytest.approx(np.pi / 2)
    assert np.isnan(out.turning.iat[3])  # zero-length step has no direction


def test_fission_and_budding_events_are_read_from_one_convention_each():
    fission = pd.DataFrame({'label': [1, 2, 3], 't_start': [0, 10, 10],
                            't_end': [9, 20, 20], 'parent': [0, 1, 1]})
    ev = da.division_events(fission, mode='fission')
    assert ev.to_dict('records') == [{'parent': 1, 't_div': 9, 'kind': 'fission',
                                      'n_daughters': 2, 'daughters': (2, 3)}]
    budding = pd.DataFrame({'label': [1, 2, 3], 't_start': [0, 8, 20],
                            't_end': [40, 40, 40], 'parent': [0, 1, 1]})
    ev = da.division_events(budding, mode='budding')
    assert ev.t_div.tolist() == [7, 19] and set(ev.kind) == {'budding'}
    assert da.median_cycle_frames(budding, ev) == 12
    assert len(da.division_events(budding, mode='fission')) == 0


def _synthetic_movie(n_lineages=30, cycle=12, n_gen=3, seed=0):
    """Rods that elongate linearly and divide in two; the parent rounds up
    (minor axis grows) over the last three frames before division."""
    rng = np.random.default_rng(seed)
    rows, lin, nxt = [], [], 1
    for _ in range(n_lineages):
        queue = [(nxt, int(rng.integers(0, 4)), 0, 100.0 * rng.uniform(0.9, 1.1))]
        nxt += 1
        while queue:
            lab, t0, par, birth = queue.pop()
            gen = 0 if par == 0 else 1
            t1 = t0 + cycle - 1
            lin.append((lab, t0, t1, par))
            for t in range(t0, t1 + 1):
                age = t - t0
                rel = t - t1
                rows.append({'t': t, 'track_id': lab,
                             'area': birth + birth * age / cycle,
                             'minor_axis': 5.0 + (2.0 if rel > -3 else 0.0) + rng.normal(0, .1),
                             'centroid_y': float(lab), 'centroid_x': float(t)})
            if t0 + cycle < n_gen * cycle and gen < 2:
                for _ in range(2):
                    queue.append((nxt, t1 + 1, lab, (birth * 2) / 2))
                    nxt += 1
    meas = pd.DataFrame(rows)
    lineage = pd.DataFrame(lin, columns=list(da.LINEAGE_COLUMNS))
    return meas, lineage


def test_pipeline_finds_the_planted_pre_division_rounding_and_controls_do_not():
    meas, lineage = _synthetic_movie()
    ev = da.division_events(lineage, mode='fission')
    ev = ev[ev.n_daughters == 2].reset_index(drop=True)
    assert len(ev) > 20
    assert da.median_cycle_frames(lineage, ev) == 12
    window, k, horizon = 6, 2, 2
    idx = meas.set_index(['track_id', 't'])['area']
    ev['event_id'] = np.arange(len(ev))
    ev['base_size'] = [idx.get((p, t - window), np.nan) for p, t in zip(ev.parent, ev.t_div)]
    ev = ev[np.isfinite(ev.base_size)].reset_index(drop=True)
    slots = da.control_slots(meas, ev, window, horizon)
    assert len(slots)
    # no slot sits within the horizon of its own track's division
    tdiv = ev.set_index('parent').t_div
    near = slots[slots.track_id.isin(tdiv.index)]
    assert ((near.track_id.map(tdiv) - near.tp) > horizon).all()
    ctrl = da.match_controls(ev, slots, window, size_tol=0.3, rng=1)
    assert ctrl.matched.mean() > 0.8
    assert (ctrl[ctrl.matched].size_ratio.sub(1).abs() <= 0.3 + 1e-9).all()
    shuf = da.shuffled_alignment(ev, slots, rng=2)

    units_div = ev.assign(unit_id=ev.event_id, track_id=ev.parent, t0=ev.t_div)
    units_ctrl = ctrl[ctrl.matched].assign(unit_id=lambda d: d.event_id)
    units_shuf = shuf[shuf.matched].assign(unit_id=lambda d: d.event_id)
    cols = ('area', 'minor_axis')
    ch = {name: da.unit_changes(da.align_units(meas, u, window, after=3, columns=cols),
                                window, k, columns=cols)
          for name, u in (('div', units_div), ('ctrl', units_ctrl), ('shuf', units_shuf))}
    planted = da.effect_size(ch['div'].minor_axis, ch['ctrl'].minor_axis, n_boot=300, rng=0)
    assert planted['g'] > 3 and planted['ci_low'] > 1
    null = da.effect_size(ch['shuf'].minor_axis, ch['ctrl'].minor_axis, n_boot=300, rng=0)
    assert abs(null['g']) < 0.8
    assert null['ci_low'] < 0 < null['ci_high'] or abs(null['g']) < 0.5

    aligned = da.align_units(meas, units_div, window, after=3, columns=cols)
    assert set(aligned.role) == {'self', 'daughter'}
    assert aligned[aligned.role == 'self'].rel_t.max() == 0
    assert aligned[aligned.role == 'daughter'].rel_t.min() == 1

    cyc = da.cell_cycles(meas.assign(major_axis=meas.area / 5), lineage,
                         ev.assign(n_daughters=2))
    assert len(cyc) and (cyc.duration == 12).all()


def test_unit_changes_refuses_overlapping_windows():
    with pytest.raises(ValueError):
        da.unit_changes(pd.DataFrame(columns=['unit_id', 'role', 'rel_t', 'area']), 3, 3)


def test_random_effects_meta_matches_a_hand_computed_case():
    homo = da.random_effects_meta([0.5, 0.5, 0.5], [0.01, 0.02, 0.04])
    assert homo['mu'] == pytest.approx(0.5) and homo['tau2'] == 0 and homo['I2'] == 0
    het = da.random_effects_meta([0.0, 1.0, 2.0], [0.01, 0.01, 0.01])
    # Q = 200, C = 300 - 300/3 = 200 -> tau2 = (200 - 2) / 200
    assert het['Q'] == pytest.approx(200.0)
    assert het['tau2'] == pytest.approx(0.99)
    assert het['I2'] == pytest.approx(99.0)
    assert het['mu'] == pytest.approx(1.0)
    assert da.random_effects_meta([np.nan], [1.0])['k'] == 0


def test_size_control_slopes_recover_adder_sizer_and_timer():
    rng = np.random.default_rng(0)
    b = rng.uniform(50, 150, 400)
    adder = da.size_control_fits(b, b + 100 + rng.normal(0, 2, 400), n_boot=200, rng=0)
    sizer = da.size_control_fits(b, 300 + rng.normal(0, 2, 400), n_boot=200, rng=0)
    timer = da.size_control_fits(b, 2 * b + rng.normal(0, 2, 400), n_boot=200, rng=0)
    assert adder['slope_division'] == pytest.approx(1, abs=0.05)
    assert adder['slope_added_ci_low'] < 0 < adder['slope_added_ci_high']
    assert sizer['slope_division'] == pytest.approx(0, abs=0.05)
    assert timer['slope_division'] == pytest.approx(2, abs=0.05)
    assert np.isnan(da.size_control_fits([1, 1], [2, 2])['slope_division'])


def test_effect_size_handles_too_few_values():
    r = da.effect_size([1.0], [2.0, 3.0])
    assert np.isnan(r['g']) and r['n_treated'] == 1
