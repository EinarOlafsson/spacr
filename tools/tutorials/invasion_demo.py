"""Independent counting of the preserved synthetic two-colour teaching data.

No application functions calculate the reference. This validates plumbing and
arithmetic, not actual stain quality, segmented images or treatment efficacy.
"""
from collections import defaultdict
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile

from replication_demo import digest, verify_files, verify_bars
from stage_lesson import write


def quantile(values, probability):
    """Linear interpolation between sorted observations, without NumPy."""
    values = sorted(values)
    if not values or not 0 <= probability <= 1 or not all(map(math.isfinite, values)):
        raise ValueError('A quantile needs finite observations and a valid probability')
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def expected(rows):
    """Apply only the explicit, fixed tutorial configuration to its input rows."""
    labels = [r['object_label'] for r in rows]
    if len(labels) != len(set(labels)):
        raise ValueError('Synthetic object identities must be unique')
    for r in rows:
        if (not 200 <= r['pathogen_area'] <= 800 or r['cell_id'] <= 0 or
                not r['pathogen_channel_0_mean_intensity'] >= 100 or
                not math.isfinite(r['pathogen_channel_1_mean_intensity'])):
            raise ValueError('The demonstrated filter and finite-signal premise changed')
    controls = [r for r in rows if r['columnID'] == 'c1']
    if len(controls) < 10:
        raise ValueError('The baseline must supply at least ten objects')
    cut = quantile([r['pathogen_channel_1_mean_intensity'] for r in controls], .99)
    if cut <= 0:
        raise ValueError('The preserved synthetic baseline must have a positive threshold')
    calls, fields, wells, conditions = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        if r['columnID'] == 'c1':
            continue
        condition = {'c2': 'HeLa_vehicle', 'c3': 'HeLa_inhibitor'}[r['columnID']]
        well = '_'.join(r[k] for k in ('plateID', 'rowID', 'columnID'))
        value = r['pathogen_channel_1_mean_intensity']
        outside = value > cut
        record = dict(prcf=r['prcf'], prc=well, condition=condition,
            outside_intensity_raw=value, outside_background=0, outside_intensity=value,
            threshold=cut, threshold_low=cut * .75, threshold_high=cut * 1.25,
            threshold_source='control', is_outside=int(outside),
            invasion_class='attached' if outside else 'invaded',
            is_outside_low_threshold=int(value > cut * .75),
            is_outside_high_threshold=int(value > cut * 1.25), no_host_cell=False)
        calls[str(r['object_label'])] = record
        fields[r['prcf']].append(record)
        wells[well].append(record)
        conditions[condition].append(record)

    def counts(observations):
        n = len(observations)
        inside = sum(r['invasion_class'] == 'invaded' for r in observations)
        return dict(n_attached=n-inside, n_invaded=inside, n_total=n,
                    invasion_efficiency=inside/n)

    field_rows = {key: dict(counts(obs), n_objects=len(obs),
        condition=obs[0]['condition'], threshold=cut, threshold_source='control',
        reference_threshold=cut, control_threshold=cut, threshold_low=cut*.75,
        threshold_high=cut*1.25, threshold_relative_difference=0,
        qc_flag_threshold_disagrees=False, qc_flag_no_threshold=False)
        for key, obs in fields.items()}
    well_rows = {}
    for key, obs in wells.items():
        low = sum(not r['is_outside_low_threshold'] for r in obs)/len(obs)
        high = sum(not r['is_outside_high_threshold'] for r in obs)/len(obs)
        row = dict(counts(obs), condition=obs[0]['condition'], n_objects=len(obs),
            n_unclassified=0, n_no_host_cell=0, n_fields=len({r['prcf'] for r in obs}),
            invasion_efficiency_low_threshold=low, invasion_efficiency_high_threshold=high,
            threshold_median=cut, threshold_source='control', reference_threshold_median=cut,
            threshold_relative_difference=0, qc_flag_threshold_disagrees=False,
            qc_flag_low_total=len(obs) < 50)
        row['invasion_efficiency_inflation'] = high-row['invasion_efficiency']
        row['qc_flag_threshold_inflates'] = row['invasion_efficiency_inflation'] > .05
        well_rows[key] = row
    summary = {}
    for condition, obs in conditions.items():
        eff = [r['invasion_efficiency'] for r in well_rows.values() if r['condition'] == condition]
        row = counts(obs)
        row.update(invasion_efficiency_pooled=row['invasion_efficiency'],
            invasion_efficiency=statistics.mean(eff),
            invasion_efficiency_median=statistics.median(eff), n_wells=len(eff),
            n_wells_scored=len(eff), invasion_efficiency_sd=statistics.stdev(eff) if len(eff)>1 else 0,
            invasion_efficiency_sem=statistics.stdev(eff)/math.sqrt(len(eff)) if len(eff)>1 else 0)
        summary[condition] = row
    return dict(control_objects=len(controls), threshold=cut,
        tables={'parasite_calls.csv': ('object_label', calls),
                'field_thresholds.csv': ('prcf', field_rows),
                'well_invasion.csv': ('prc', well_rows),
                'condition_summary.csv': ('condition', summary)})


def check_preserved(manifest):
    for key in ('source', 'database'):
        if digest(manifest[key]) != manifest[key + '_sha256']:
            raise ValueError('An original or private input database changed')


def prepare(stage):
    stage = Path(stage)
    source = stage.parent / 'synthetic/invasion/measurements/measurements.db'
    source_hash = digest(source)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute('SELECT * FROM pathogen ORDER BY object_label')]
    reference = expected(rows)
    if (len(rows) != 1800 or reference['control_objects'] != 360 or
            len(reference['tables']['field_thresholds.csv'][1]) != 24):
        raise ValueError('The preserved synthetic example changed in size')
    runs = stage/'invasion_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='SYNTHETIC-two-colour-', dir=runs))
    database = root/'measurements/measurements.db'
    database.parent.mkdir()
    shutil.copy2(source, database)
    manifest = dict(source=str(source), source_sha256=source_hash,
        database=str(database), database_sha256=source_hash,
        expected=reference, synthetic=True, images_available=False,
        biological_effect_claim=False, rows=len(rows))
    check_preserved(manifest)
    write(root/'manifest.json', manifest)
    return manifest


def verify_output(directory, manifest):
    return verify_files(directory, manifest['expected']['tables'])


def verify_figure(groups, series, table):
    """Reuse the independently tested bar-geometry guard with invasion fractions."""
    reference = {key: dict(frac_attached=r['n_attached']/r['n_total'],
                          frac_invaded=r['n_invaded']/r['n_total']) for key, r in table.items()}
    return verify_bars(groups, series, reference)


def verify_histograms(panels, calls, threshold):
    """Count the raw per-well intensities inside actual plotted rectangles."""
    wells = defaultdict(list)
    for record in calls.values():
        wells[record['prc']].append(record['outside_intensity'])
    identities = [p['well'] for p in panels]
    if len(identities) != len(set(identities)) or set(identities) != set(wells):
        raise ValueError('Histogram well identities differ')
    checked = 0
    for panel in panels:
        values = wells[panel['well']]
        bars = panel['bars']
        if not bars or not math.isclose(bars[0]['left'], min(values), abs_tol=1e-9) or not math.isclose(
                bars[-1]['left']+bars[-1]['width'], max(values), abs_tol=1e-9):
            raise ValueError('Histogram range does not cover the actual observations')
        cuts = panel['thresholds']
        if (len(cuts) != 1 or len(cuts[0]) != 2 or
                not all(math.isclose(v, threshold, rel_tol=1e-12, abs_tol=1e-12) for v in cuts[0]) or
                panel['denominator'] != len(values)):
            raise ValueError('Histogram threshold or denominator differs')
        for i, bar in enumerate(bars):
            low, high = bar['left'], bar['left']+bar['width']
            if bar['width'] <= 0 or (i and not math.isclose(low,
                    bars[i-1]['left']+bars[i-1]['width'], abs_tol=1e-9)):
                raise ValueError('Histogram bin geometry is invalid')
            # Matplotlib reconstructs rectangle left edges from centres. Two
            # actual first bins moved up one ULP, putting the observed minimum
            # just outside its artist geometry. Bound that roundtrip allowance
            # to four ULPs at the outer endpoints, never across interior bins.
            count = sum((low <= v or (i == 0 and 0 <= low-v <= 4*math.ulp(low))) and
                (v < high or (i == len(bars)-1 and v <= high+4*math.ulp(high))) for v in values)
            if bar['height'] != count or bar['bottom'] != 0:
                raise ValueError('Histogram height or base differs from independent counts')
            checked += 1
    return dict(panels=len(panels), bins_checked=checked,
                threshold_lines_checked=len(panels), observations=sum(map(len, wells.values())))
