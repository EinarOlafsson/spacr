"""Independent arithmetic for the tutorial's finite, nondegenerate single feature.

This deliberately does not implement MCD, zero-spread fallback or inference.
The real example's four wells cannot satisfy the five-well comparison minimum.
"""
from collections import defaultdict
import math
from statistics import median, quantiles, NormalDist


def expected_scores(rows, feature, *, method='mad', transform='none', threshold=3.5):
    if method not in {'mad', 'iqr'} or transform not in {'none', 'log10'}:
        raise ValueError('Only MAD/IQR and explicit none/log10 are checked')
    values = [float(row[feature]) for row in rows]
    if len(values) < 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('This scoped oracle needs at least four finite values')
    if transform == 'log10':
        if min(values) <= 0:
            raise ValueError('Real non-positive measurements cannot be log transformed')
        values = [math.log10(v) for v in values]
    if method == 'mad':
        centre = median(values)
        spread = median(abs(v-centre) for v in values) / NormalDist().inv_cdf(.75)
        if spread <= 0:
            raise ValueError('Zero-spread fallback is outside this oracle')
        scores = [abs(v-centre)/spread for v in values]
        fences = [centre-threshold*spread, centre+threshold*spread]
    else:
        q1, _, q3 = quantiles(values, n=4, method='inclusive')
        centre, spread = (q1+q3)/2, q3-q1
        if spread <= 0:
            raise ValueError('Zero-spread fallback is outside this oracle')
        scores = [max(q1-v, v-q3)/spread for v in values]
        fences = [q1-threshold*spread, q3+threshold*spread]
    return dict(scores=scores, flags=[s>threshold for s in scores],
                centre=centre, spread=spread, fences=fences)


def verify_scan(result, rows, feature, *, method='mad', transform='none', threshold=3.5):
    expected = expected_scores(rows, feature, method=method,
                               transform=transform, threshold=threshold)
    if (result.n_rows_in != len(rows) or result.n_scored != len(rows)
            or tuple(result.features) != (feature,) or result.method != method
            or result.transform != transform or result.threshold != threshold
            or len(result.scores) != len(rows)):
        raise ValueError('Scan population, feature, method, transform or threshold differs')
    actual = [*result.scores, result.centres[feature], result.scales[feature],
              *result.fences[feature]]
    wanted = [*expected['scores'], expected['centre'], expected['spread'], *expected['fences']]
    errors = [abs(float(a)-b) for a,b in zip(actual, wanted)]
    if len(actual) != len(wanted) or any(not math.isfinite(e) or e>1e-10 for e in errors):
        raise ValueError('Scan differs from independent median/quartile arithmetic')
    if [bool(v) for v in result.flags] != expected['flags']:
        raise ValueError('Flagged input positions differ from independent threshold comparison')
    return dict(rows=len(rows), feature=feature, method=method, transform=transform,
                threshold=threshold, flagged=sum(expected['flags']),
                centre=expected['centre'], spread=expected['spread'],
                fences=expected['fences'], max_numeric_error=max(errors))


def verify_unscored_wells(result, rows, feature):
    groups = defaultdict(list)
    for index,row in enumerate(rows):
        groups[tuple(row[k] for k in ('plateID','rowID','columnID'))].append(index)
    if not 1 <= len(groups) < 5 or tuple(result.well_keys) != ('plateID','rowID','columnID'):
        raise ValueError('This well check is scoped to fewer than five actual wells')
    actual = result.well_frame().to_dict('records')
    if (len(actual) != len(groups) or result.n_wells_scored != 0
            or {tuple(r[k] for k in result.well_keys) for r in actual} != set(groups)):
        raise ValueError('Unscored well identity or population differs')
    evidence=[]
    for row in actual:
        key=tuple(row[k] for k in result.well_keys)
        indices=groups[key]
        values=[float(rows[i][feature]) for i in indices]
        if result.transform=='log10':values=[math.log10(v) for v in values]
        flagged=sum(bool(result.flags[i]) for i in indices)
        if (row['n_objects']!=len(indices) or row['n_scored_objects']!=len(indices)
                or row['n_flagged_objects']!=flagged or row['well_scored']
                or row['well_outlier'] or not math.isnan(row['well_outlier_score'])
                or not row['well_outlier_reason'].startswith('not scored:')
                or abs(row['flagged_share']-flagged/len(indices))>1e-12
                or abs(row[feature+'_median']-median(values))>1e-10):
            raise ValueError('Well medians, object flags or not-scored status differ')
        evidence.append(dict(key=key, n=len(indices), flagged_objects=flagged,
                             median=median(values), well_scored=False,
                             reason=row['well_outlier_reason']))
    return evidence
